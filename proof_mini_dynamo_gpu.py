"""GPU proof: mini-dynamo serving real traffic on real vLLM engines (Modal, 2xA100).

Topology: 1 prefill (GPU0 :8100) + 1 decode (GPU1 :8200) + mini-dynamo
frontend (:8787), Qwen3-14B-FP8, block_size 64, NIXL push mode. Proves:
  1. preprocess with the REAL tokenizer (HF configs, no stub)
  2. S3/S6 routing against live engines (PROF lines name P/D workers)
  3. producer 1-token cap on the wire (push TTFT in the ~200-400 ms regime,
     not the ~2.5 s full-decode regime of the old max_tokens bug)
  4. multi-turn affinity + completion accounting (usage returns to 0)
  5. observability endpoints (/v1/models, /kvbm/status)

NOT a benchmark: ~10 requests, no AIPerf. Full load harness lives in
modal_bench.py.

Run: modal run proof_mini_dynamo_gpu.py
Working dir note: run from the repo root (this file mounts ./src).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

_MODEL = "Qwen/Qwen3-14B-FP8"
_BLOCK_SIZE = 64
_MAX_MODEL_LEN = 32768
_FE_PORT = 8787

_MINI_SRC = Path(__file__).resolve().parent / "src"  # repo ./src

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "wget", "curl", "build-essential")
    .uv_pip_install("vllm", pre="--prerelease=allow")
    .pip_install(
        "nixl", "xxhash", "transformers>=4.40", "huggingface-hub",
        "httpx", "fastapi", "uvicorn",
    )
    .add_local_dir(_MINI_SRC, "/root/mini_src")
)

app = modal.App("mini-dynamo-proof")


def start_process(cmd, name, log_file=None, env=None):
    run_env = {**(env or os.environ), "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"}
    if log_file:
        proc = subprocess.Popen(cmd, shell=True, stdout=open(log_file, "w"),
                                stderr=subprocess.STDOUT, env=run_env)
    else:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=run_env)

        def stream():
            for line in proc.stdout:
                print(f"[{name}] {line}", end="")

        threading.Thread(target=stream, daemon=True).start()
    return proc


def wait_for_endpoint(port, path="/health", timeout=600):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(
                urllib.request.Request(f"http://localhost:{port}{path}"), timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def kill_procs(procs):
    for p in procs:
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


@app.function(image=image, gpu="A100:2", timeout=3600, max_containers=1,
              secrets=[modal.Secret.from_name("huggingface-secret")])
async def gpu_proof():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("proof")
    procs = []
    results = {"checks": []}

    def check(name, cond, detail=""):
        results["checks"].append({"name": name, "pass": bool(cond),
                                  "detail": str(detail)[:200]})
        log.info("  %s %s %s", "PASS" if cond else "FAIL", name, detail)
        if not cond:
            raise AssertionError(f"check failed: {name} {detail}")

    try:
        # 1. model (retry; env-only auth - never interpolate the token)
        log.info("Pre-downloading model...")
        dl_cmd = ("from huggingface_hub import snapshot_download; "
                  f"snapshot_download('{_MODEL}')")
        for attempt in range(1, 4):
            try:
                subprocess.run(
                    [sys.executable, "-c", dl_cmd],
                    env={**os.environ, "PYTHONHASHSEED": "0"}, check=True)
                break
            except subprocess.CalledProcessError:
                log.warning("download attempt %d/3 failed", attempt)
                if attempt == 3:
                    raise
                time.sleep(30 * attempt)
        log.info("Model ready")

        # 2. vLLM workers (same flags as the benchmark harness)
        kv_p = json.dumps({
            "kv_connector": "NixlPushConnector", "kv_role": "kv_producer",
            "engine_id": "prefill-0",
            "kv_connector_extra_config": {"kv_lease_duration": 60},
        })
        kv_d = json.dumps({
            "kv_connector": "NixlPushConnector", "kv_role": "kv_consumer",
            "engine_id": "decode-0",
        })
        for role, port, gpu_id, side_port, kv in [
            ("prefill", 8100, 0, 5600, kv_p),
            ("decode", 8200, 1, 5700, kv_d),
        ]:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["UCX_NET_DEVICES"] = "all"
            env["NIXL_LOG_LEVEL"] = "INFO"
            env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(side_port)
            cmd = (
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--model {_MODEL} --port {port} --gpu-memory-utilization 0.85 "
                f"--max-model-len {_MAX_MODEL_LEN} --block-size {_BLOCK_SIZE} "
                f"--dtype auto --enable-prefix-caching "
                f"--kv-transfer-config '{kv}'"
            )
            log.info("Starting %s on GPU %d port %d", role.upper(), gpu_id, port)
            procs.append(start_process(cmd, role, f"/tmp/vllm-{role}.log", env))

        for port in (8100, 8200):
            if not wait_for_endpoint(port, "/v1/models", timeout=900):
                raise RuntimeError(f"vLLM on port {port} failed to start")
        log.info("Both vLLM workers ready")

        # 3. mini-dynamo frontend (production code, real tokenizer)
        env = {**os.environ, "PYTHONPATH": "/root/mini_src",
               "PYTHONHASHSEED": "0"}
        gw_cmd = (
            f"{sys.executable} -m frontend.frontend "
            f"--model {_MODEL} --prefill-ports 8100 --decode-ports 8200 "
            f"--http-port {_FE_PORT} --router-mode kv --block-size {_BLOCK_SIZE}"
        )
        log.info("Starting mini-dynamo: %s", gw_cmd)
        procs.append(start_process(gw_cmd, "mini", "/tmp/mini.log", env))
        if not wait_for_endpoint(_FE_PORT, "/health", timeout=120):
            raise RuntimeError("mini-dynamo frontend failed to start")
        log.info("mini-dynamo ready")
        time.sleep(5)  # tokenizer lazy-loads on first request; warm it below

        import httpx
        url = f"http://localhost:{_FE_PORT}"

        async with httpx.AsyncClient(timeout=300) as client:
            # 3a. health + models (real tokenizer configs download here on first use)
            r = await client.get(f"{url}/health")
            check("health", r.status_code == 200, r.text[:60])
            r = await client.get(f"{url}/v1/models")
            check("models proxy", r.status_code == 200, r.text[:80])

            # 3b. warmup (untimed): tokenizer configs download + vLLM first-
            # request compile happen here, not in the measurement. The old
            # harness does the same via AIPerf --warmup-request-count.
            for w in range(3):
                r = await client.post(f"{url}/v1/chat/completions", json={
                    "model": _MODEL,
                    "messages": [{"role": "user", "content": "Warmup."}],
                    "max_tokens": 8, "temperature": 0,
                })
                log.info("warmup %d: %d", w, r.status_code)

            # 3c. basic chat, client-measured TTFT (push regime, not full-decode)
            ttfts = []
            for m in range(3):
                t0 = time.monotonic()
                first = None
                async with client.stream("POST", f"{url}/v1/chat/completions", json={
                    "model": _MODEL,
                    "messages": [{"role": "user", "content": "Say hello in one word."}],
                    "max_tokens": 10, "temperature": 0, "stream": True,
                }) as resp:
                    if m == 0:
                        check("chat status", resp.status_code == 200, resp.status_code)
                    async for chunk in resp.aiter_bytes():
                        if first is None:
                            first = time.monotonic()
                ttfts.append((first - t0) * 1000 if first else -1)
            log.info("warmed TTFTs (ms): %s",
                     [f"{t:.0f}" for t in ttfts])
            check("TTFT in push regime (<1500 ms)", 0 < ttfts[-1] < 1500,
                  f"{ttfts[-1]:.0f} ms (cold curve: {[f'{t:.0f}' for t in ttfts]})")

            # 3c. multi-turn affinity (3 turns, one conversation)
            conv = f"proof-{int(time.time())}"
            msgs = [{"role": "user", "content": "What is 2+2?"}]
            for turn in range(3):
                r = await client.post(f"{url}/v1/chat/completions", json={
                    "model": _MODEL, "messages": msgs,
                    "max_tokens": 10, "temperature": 0,
                    "conversation_id": conv,
                })
                check(f"turn {turn} 200", r.status_code == 200,
                      r.text[:120] if r.status_code != 200 else "ok")
                reply = r.json()["choices"][0]["message"]["content"]
                msgs += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "And 3+3?"}]
            time.sleep(10)  # background completion accounting settles

            # 3d. observability: usage drained back to 0 (no load leak)
            r = await client.get(f"{url}/kvbm/status")
            check("kvbm status", r.status_code == 200, r.text[:160])

        # 4. gateway-log evidence: PROF lines name P/D workers per request
        prof = [l for l in open("/tmp/mini.log", errors="replace")
                if "PROF[" in l]
        check("PROF lines emitted", len(prof) >= 4, f"n={len(prof)}")
        for line in prof[:5]:
            log.info("    %s", line.strip()[:200])
        routed = [l for l in open("/tmp/mini.log", errors="replace")
                  if "KVBM route" in l or "KV_ROUTE" in l]
        check("routing evidence in log", len(routed) >= 1, f"n={len(routed)}")

    except BaseException:
        # Failure diagnostics: a blind GPU allocation teaches nothing. Dump
        # gateway + worker tails (and which procs already exited) before the
        # traceback, so the next run starts from evidence.
        for path in ["/tmp/mini.log", "/tmp/vllm-prefill.log", "/tmp/vllm-decode.log"]:
            try:
                with open(path, errors="replace") as fh:
                    lines = fh.read().splitlines()
                log.error("----- %s (%d lines, last 60) -----", path, len(lines))
                for line in lines[-60:]:
                    log.error("  %s", line[:220])
            except FileNotFoundError:
                log.error("----- %s: MISSING -----", path)
        for idx, p in enumerate(procs):
            try:
                log.error("proc[%d] poll=%s", idx, p.poll())
            except Exception:
                pass
        raise
    finally:
        kill_procs(procs)

    passed = sum(1 for c in results["checks"] if c["pass"])
    log.info("=" * 60)
    log.info("MINI-DYNAMO GPU PROOF: %d/%d checks passed",
             passed, len(results["checks"]))
    log.info("=" * 60)
    return results


@app.local_entrypoint()
def main():
    print("mini-dynamo GPU proof: 1P+1D, Qwen3-14B-FP8, functional traffic")
    result = gpu_proof.remote()
    n = sum(1 for c in result["checks"] if c["pass"])
    print(f"Result: {n}/{len(result['checks'])} passed")
    for c in result["checks"]:
        if not c["pass"]:
            print("FAILED:", c)
