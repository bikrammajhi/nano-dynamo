"""Load comparison: nano-dynamo 2P+2D under AIPerf (multi_turn + mixed_workload).

Same topology, model, engine flags, and AIPerf scenarios as the original
nano-dynamo harness (2P+2D, Qwen3-14B-FP8, NIXL push) so the numbers compare
apples-to-apples; the ONLY variable is the routing layer (nano-dynamo vs old
gateway vs README's NVIDIA Dynamo).

Reference numbers being compared against:
  old gateway multi_turn : TTFT 253-264 ms, 371-384 tok/s, lat ~2083 ms
  old gateway mixed      : TTFT 326 ms, 1121 tok/s, lat 3085 ms (README 08-01)
  NVIDIA Dynamo multi    : TTFT 195 ms, 405 tok/s, lat 1992 ms (README 08-01)
  NVIDIA Dynamo mixed    : TTFT 247 ms, 1155 tok/s, lat 2984 ms (README 08-01)

Run: modal run bench_nano_dynamo.py [--scenario multi_turn|mixed_workload|all|stress]
"""
from __future__ import annotations

import asyncio
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
INPUT_TOKENS = 256
OUTPUT_TOKENS = 128
_FE_PORT = 8787

_NANO_SRC = Path(__file__).resolve().parent / "src"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "wget", "curl", "build-essential")
    .uv_pip_install("vllm", pre="--prerelease=allow")
    .pip_install(
        "nixl", "xxhash", "transformers>=4.40", "huggingface-hub",
        "httpx", "fastapi", "uvicorn", "aiperf",
    )
    .add_local_dir(_NANO_SRC, "/root/nano_src")
)

app = modal.App("nano-dynamo-bench")

SCENARIOS = {
    "multi_turn": {
        "description": "Multi-turn conversations with KV cache reuse",
        "cmd": lambda url, art: [
            "aiperf", "profile", "--model", _MODEL, "--url", url,
            "--endpoint-type", "chat", "--streaming",
            "--conversation-num", "30",
            "--conversation-turn-mean", "5",
            "--conversation-turn-stddev", "1",
            "--conversation-turn-delay-mean", "1000",
            "--synthetic-input-tokens-mean", str(INPUT_TOKENS),
            "--output-tokens-mean", str(OUTPUT_TOKENS),
            "--concurrency", "10",
            "--warmup-request-count", "10",
            "--artifact-dir", art, "--tokenizer", _MODEL,
            "--extra-inputs", "ignore_eos:true",
        ],
    },
    "mixed_workload": {
        "description": "Mixed ISL/OSL distribution (chatbot simulation)",
        "cmd": lambda url, art: [
            "aiperf", "profile", "--model", _MODEL, "--url", url,
            "--endpoint-type", "chat", "--streaming",
            "--sequence-distribution", "128|20,64|10:40;512|50,256|30:35;1024|80,256|40:25",
            "--concurrency", "20",
            "--request-count", "200",
            "--warmup-request-count", "20",
            "--artifact-dir", art, "--tokenizer", _MODEL,
            "--extra-inputs", "ignore_eos:true",
        ],
    },
    "stress": {
        # Crossover probe (audit §4): long prefills (compute-dominated) at
        # high concurrency (decode-batch-dominated). If disaggregation wins
        # anywhere at 14B, it wins here. ISL 4k fits max_model_len 32768.
        "description": "Long-context high-concurrency stress (crossover probe)",
        "cmd": lambda url, art: [
            "aiperf", "profile", "--model", _MODEL, "--url", url,
            "--endpoint-type", "chat", "--streaming",
            "--synthetic-input-tokens-mean", "4096",
            "--output-tokens-mean", "256",
            "--concurrency", "30",
            "--request-count", "60",
            "--warmup-request-count", "5",
            "--artifact-dir", art, "--tokenizer", _MODEL,
            "--extra-inputs", "ignore_eos:true",
        ],
    },
}


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


def wait_for_endpoint(port, path="/health", timeout=300):
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


@app.function(image=image, gpu="A100:4", timeout=7200, max_containers=1,
              secrets=[modal.Secret.from_name("huggingface-secret")])
async def run_bench(scenario: str = "all"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("bench")
    num_prefill, num_decode = 2, 2
    prefill_ports = [8100 + i for i in range(num_prefill)]
    decode_ports = [8200 + i for i in range(num_decode)]
    all_results = {}
    procs = []

    try:
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

        for i, port in enumerate(prefill_ports):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i)
            env["UCX_NET_DEVICES"] = "all"
            env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(5600 + i)
            kv_config = json.dumps({
                "kv_connector": "NixlPushConnector",
                "kv_role": "kv_producer",
                "engine_id": f"prefill-{i}",
                "kv_connector_extra_config": {"kv_lease_duration": 60},
            })
            cmd = (
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--model {_MODEL} --port {port} --gpu-memory-utilization 0.85 "
                f"--max-model-len {_MAX_MODEL_LEN} --block-size {_BLOCK_SIZE} "
                f"--dtype auto --enable-prefix-caching "
                f"--kv-transfer-config '{kv_config}'"
            )
            log.info("Starting PREFILL %d on GPU %d port %d", i, i, port)
            procs.append(start_process(cmd, f"prefill-{i}",
                                       f"/tmp/vllm-prefill-{i}.log", env))

        for i, port in enumerate(decode_ports):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(num_prefill + i)
            env["UCX_NET_DEVICES"] = "all"
            env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(5700 + i)
            kv_config = json.dumps({
                "kv_connector": "NixlPushConnector",
                "kv_role": "kv_consumer",
                "engine_id": f"decode-{i}",
            })
            cmd = (
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--model {_MODEL} --port {port} --gpu-memory-utilization 0.85 "
                f"--max-model-len {_MAX_MODEL_LEN} --block-size {_BLOCK_SIZE} "
                f"--dtype auto --enable-prefix-caching "
                f"--kv-transfer-config '{kv_config}'"
            )
            log.info("Starting DECODE %d on GPU %d port %d", i, num_prefill + i, port)
            procs.append(start_process(cmd, f"decode-{i}",
                                       f"/tmp/vllm-decode-{i}.log", env))

        for port in prefill_ports + decode_ports:
            if not wait_for_endpoint(port, "/v1/models", timeout=900):
                raise RuntimeError(f"vLLM on port {port} failed")
        log.info("All 4 vLLM workers ready")

        env = {**os.environ, "PYTHONPATH": "/root/nano_src",
               "PYTHONHASHSEED": "0"}
        gw_cmd = (
            f"{sys.executable} -m frontend.frontend "
            f"--model {_MODEL} "
            f"--prefill-ports {' '.join(map(str, prefill_ports))} "
            f"--decode-ports {' '.join(map(str, decode_ports))} "
            f"--http-port {_FE_PORT} --router-mode kv "
            f"--block-size {_BLOCK_SIZE}"
        )
        log.info("Starting nano-dynamo: %s", gw_cmd)
        procs.append(start_process(gw_cmd, "nano", "/tmp/nano.log", env))
        if not wait_for_endpoint(_FE_PORT, "/health", timeout=180):
            raise RuntimeError("nano-dynamo frontend failed")
        log.info("nano-dynamo ready")

        url = f"http://localhost:{_FE_PORT}"
        to_run = SCENARIOS if scenario == "all" else {scenario: SCENARIOS[scenario]}
        for name, cfg in to_run.items():
            artifact_dir = f"/tmp/aiperf_{name}"
            cmd = cfg["cmd"](url, artifact_dir)
            log.info("=" * 60)
            log.info("SCENARIO: %s", name.upper())
            log.info("=" * 60)
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                log.error("AIPerf %s failed (%d): %s", name, proc.returncode,
                          stderr.decode()[-800:])
                all_results[name] = {}
                continue
            rf = Path(artifact_dir) / "profile_export_aiperf.json"
            result = json.loads(rf.read_text()) if rf.exists() else {}
            all_results[name] = result
            ttft = result.get("time_to_first_token", {}).get("avg", 0)
            tp = result.get("output_token_throughput", {}).get("avg", 0)
            lat = result.get("request_latency", {}).get("avg", 0)
            pcts = {k: round(v, 1) for k, v in result.get("time_to_first_token", {}).items()
                    if k in ("min", "p50", "p90", "p95", "p99", "max")}
            log.info("  TTFT=%.0fms tok/s=%.0f lat=%.0fms", ttft, tp, lat)
            log.info("  TTFT pct: %s", pcts)

        # gateway PROF summary (per-request routing evidence)
        try:
            prof = [l for l in open("/tmp/nano.log", errors="replace") if "PROF[" in l]
            log.info("gateway PROF n=%d", len(prof))
            for line in prof[:3]:
                log.info("    %s", line.strip()[:200])
        except FileNotFoundError:
            log.warning("no /tmp/nano.log")

    except BaseException:
        for path in ["/tmp/nano.log"] + \
                    [f"/tmp/vllm-prefill-{i}.log" for i in range(num_prefill)] + \
                    [f"/tmp/vllm-decode-{i}.log" for i in range(num_decode)]:
            try:
                with open(path, errors="replace") as fh:
                    lines = fh.read().splitlines()
                log.error("----- %s (%d lines, last 40) -----", path, len(lines))
                for line in lines[-40:]:
                    log.error("  %s", line[:220])
            except FileNotFoundError:
                log.error("----- %s: MISSING -----", path)
        raise
    finally:
        kill_procs(procs)

    log.info("=" * 70)
    log.info("NANO-DYNAMO LOAD BENCH: 2P+2D %s", scenario)
    for name, r in all_results.items():
        log.info("  %-16s TTFT=%7.0fms tok/s=%7.0f lat=%7.0fms",
                 name, r.get("time_to_first_token", {}).get("avg", 0),
                 r.get("output_token_throughput", {}).get("avg", 0),
                 r.get("request_latency", {}).get("avg", 0))
    log.info("=" * 70)
    return all_results


@app.local_entrypoint()
def main(scenario: str = "all"):
    print(f"nano-dynamo load bench | 2P+2D | {scenario}")
    results = run_bench.remote(scenario)
    print(f"Done: {list(results)}")
