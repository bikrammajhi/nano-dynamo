"""Prefill bottleneck profiling: exact phase breakdown of the ~1670ms TTFT.

Runs the nano-dynamo stack (1P + 1D by default, Qwen3-14B-FP8) with:
  - gateway PROF lines (tokenize / prefill / decode dispatch timestamps)
  - vLLM --enable-log-requests (per-request TTFT + E2E inside each vLLM process)
  - NIXL_TRACE (transfer-level timing inside the NixlPushConnector)

Then drives controlled requests and prints the breakdown per phase.
"""
from __future__ import annotations
import asyncio, json, logging, os, subprocess, sys, threading, time
import httpx
from pathlib import Path
import modal

_MODEL = "Qwen/Qwen3-14B-FP8"
_BLOCK_SIZE = 16
_MAX_MODEL_LEN = 8192
_REPO_ROOT = Path("/home/bikram_/workspace/nano-dynamo")
_ISL = 256
_OSL = 32
_N_REQ = 10

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "wget", "curl", "build-essential")
    .pip_install(
        "vllm",
        "nixl", "xxhash", "transformers>=4.40", "huggingface-hub",
        "httpx", "fastapi", "uvicorn", "sse-starlette",
    )
    .add_local_dir(_REPO_ROOT / "src", "/root/src")
)

app = modal.App("nano-dynamo-profile")

def start_process(cmd, name, log_file=None, env=None):
    run_env = {**(env or os.environ), "PYTHONHASHSEED": "0"}
    if log_file:
        proc = subprocess.Popen(cmd, shell=True, stdout=open(log_file, "w"), stderr=subprocess.STDOUT, env=run_env)
    else:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=run_env)
        def stream():
            for line in proc.stdout:
                print(f"[{name}] {line.decode(errors='replace')}", end="")
        threading.Thread(target=stream, daemon=True).start()
    return proc

def wait_for_endpoint(port, path="/health", timeout=600):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}{path}"), timeout=5)
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
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill()

@app.function(image=image, gpu="A100:2", timeout=3600, max_containers=1,
              secrets=[modal.Secret.from_name("huggingface-secret")])
async def profile(n_req: int = _N_REQ):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("profile")
    gateway_port = 8787
    kv_connector = "NixlPushConnector"
    n_prefill, n_decode = 1, 1
    prefill_ports = [8100 + i for i in range(n_prefill)]
    decode_ports = [8200 + i for i in range(n_decode)]
    total_gpus = n_prefill + n_decode
    log.info("Config: %d GPUs (%dP + %dD), model=%s", total_gpus, n_prefill, n_decode, _MODEL)

    hf_token = os.environ.get("HF_TOKEN")
    dl_cmd = (f"from huggingface_hub import snapshot_download; "
              f"snapshot_download('{_MODEL}'" +
              (f", token='{hf_token}')" if hf_token else ")"))
    log.info("Pre-downloading model...")
    subprocess.run([sys.executable, "-c", dl_cmd],
                   env={**os.environ, "PYTHONHASHSEED": "0"}, check=True)
    log.info("Model ready")

    vllm_env = {
        "VLLM_LOG_LEVEL": "INFO",
        "NIXL_LOG_LEVEL": "TRACE",
        "UCX_NET_DEVICES": "all",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
    }

    procs = []
    try:
        for i, (port, gpu_id) in enumerate(zip(prefill_ports, range(n_prefill))):
            env = {**os.environ, **vllm_env, "CUDA_VISIBLE_DEVICES": str(gpu_id),
                   "VLLM_NIXL_SIDE_CHANNEL_PORT": str(5600 + i)}
            kv_config = json.dumps({
                "kv_connector": kv_connector,
                "kv_role": "kv_producer",
                "engine_id": f"prefill-{i}",
                "kv_connector_extra_config": {"kv_lease_duration": 60},
            })
            cmd = (
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--model {_MODEL} --port {port} --gpu-memory-utilization 0.80 "
                f"--max-model-len {_MAX_MODEL_LEN} --block-size {_BLOCK_SIZE} "
                f"--dtype auto --enforce-eager --enable-prefix-caching --enable-log-requests "
                f"--kv-transfer-config '{kv_config}'"
            )
            log.info("Starting PREFILL_%d GPU %d port %d", i, gpu_id, port)
            procs.append(start_process(cmd, f"prefill-{i}", f"/tmp/vllm-prefill-{i}.log", env))

        for i, (port, gpu_id) in enumerate(zip(decode_ports, range(n_prefill, total_gpus))):
            env = {**os.environ, **vllm_env, "CUDA_VISIBLE_DEVICES": str(gpu_id),
                   "VLLM_NIXL_SIDE_CHANNEL_PORT": str(5700 + i)}
            kv_config = json.dumps({
                "kv_connector": kv_connector,
                "kv_role": "kv_consumer",
                "engine_id": f"decode-{i}",
            })
            cmd = (
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--model {_MODEL} --port {port} --gpu-memory-utilization 0.80 "
                f"--max-model-len {_MAX_MODEL_LEN} --block-size {_BLOCK_SIZE} "
                f"--dtype auto --enforce-eager --enable-prefix-caching --enable-log-requests "
                f"--kv-transfer-config '{kv_config}'"
            )
            log.info("Starting DECODE_%d GPU %d port %d", i, gpu_id, port)
            procs.append(start_process(cmd, f"decode-{i}", f"/tmp/vllm-decode-{i}.log", env))

        for port in prefill_ports + decode_ports:
            role = "prefill" if port in prefill_ports else "decode"
            if not wait_for_endpoint(port, "/v1/models", timeout=600):
                raise RuntimeError(f"{role} vLLM on port {port} failed")
        log.info("All %d vLLM workers ready", total_gpus)

        gw_cmd = (
            f"{sys.executable} -m src.gateway "
            f"--model {_MODEL} "
            f"--prefill-ports {' '.join(str(p) for p in prefill_ports)} "
            f"--decode-ports {' '.join(str(p) for p in decode_ports)} "
            f"--port {gateway_port} "
            f"--prefill-kv-host 127.0.0.1 "
            f"--prefill-side-channel-ports {' '.join(str(5600 + i) for i in range(n_prefill))}"
        )
        gateway_proc = start_process(gw_cmd, "proxy", "/tmp/proxy.log")
        procs.append(gateway_proc)
        if not wait_for_endpoint(gateway_port, "/health", timeout=30):
            raise RuntimeError("Gateway failed")
        log.info("Gateway ready")

        url = f"http://localhost:{gateway_port}"
        prompt = " ".join(f"word{ (i*7)%97 }" for i in range(_ISL))

        # ── Phase 1: warmup (3 requests, steady state) ────────────────
        log.info("WARMUP: 3 requests")
        async with httpx.AsyncClient(timeout=120) as client:
            for i in range(3):
                await client.post(f"{url}/v1/chat/completions", json={
                    "model": _MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16, "temperature": 0,
                })

        # ── Phase 2: sequential timed requests ─────────────────────────
        log.info("MEASURE: %d sequential requests (isl=%d osl=%d)", n_req, _ISL, _OSL)
        rows = []
        async with httpx.AsyncClient(timeout=120) as client:
            for i in range(n_req):
                t0 = time.monotonic()
                async with client.stream("POST", f"{url}/v1/chat/completions", json={
                    "model": _MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": _OSL, "temperature": 0, "stream": True,
                }) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        log.info("REQ[%d] HTTP %d body=%.200s", i, resp.status_code, body.decode()[:200])
                    first_chunk_t = None
                    nbytes = 0
                    async for chunk in resp.aiter_bytes():
                        if first_chunk_t is None:
                            first_chunk_t = time.monotonic()
                        nbytes += len(chunk)
                t1 = time.monotonic()
                ttft = (first_chunk_t - t0) * 1000 if first_chunk_t else -1
                rows.append((ttft, (t1 - t0) * 1000, nbytes))
                log.info("REQ[%d] ttft=%.0fms total=%.0fms bytes=%d", i, ttft, (t1 - t0) * 1000, nbytes)

        # ── Phase 3: concurrent burst (queueing behavior) ──────────────
        log.info("BURST: 8 concurrent requests")
        async def burst_req(client, i):
            t0 = time.monotonic()
            async with client.stream("POST", f"{url}/v1/chat/completions", json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16, "temperature": 0, "stream": True,
            }) as resp:
                async for chunk in resp.aiter_bytes():
                    pass
            return (time.monotonic() - t0) * 1000
        async with httpx.AsyncClient(timeout=300) as client:
            burst = await asyncio.gather(*[burst_req(client, i) for i in range(8)])
        log.info("BURST done: total_latencies=%s", [f"{b:.0f}" for b in burst])

        # ── Phase 4: log analysis ─────────────────────────────────────
        avg_ttft = sum(r[0] for r in rows) / len(rows)
        log.info("=" * 70)
        log.info("RESULTS (%d req, isl=%d):", n_req, _ISL)
        log.info("  gateway TTFT avg=%.0fms  (min=%.0f max=%.0f)",
                 avg_ttft, min(r[0] for r in rows), max(r[0] for r in rows))
        log.info("  e2e        avg=%.0fms", sum(r[1] for r in rows) / len(rows))
        log.info("")

        # gateway PROF lines
        pf_parts = {}
        try:
            with open("/tmp/proxy.log") as f:
                for line in f:
                    if "PROF[" in line and "mode=" in line:
                        parts = {}
                        for tok in line.split("PROF[")[1].split("]")[1].split():
                            if "=" in tok:
                                k, v = tok.split("=", 1)
                                parts[k] = float(v) if v.replace(".", "", 1).isdigit() else v
                        pf_parts[parts.get("mode", "?")] = parts
        except FileNotFoundError:
            log.warning("no proxy log")

        # vLLM --enable-log-requests lines: Finished request ... TTFT ... E2E ...
        def vllm_req_times(logf):
            out = {"ttft": [], "e2e": []}
            try:
                with open(logf) as f:
                    for line in f:
                        if "Finished request" in line and "TTFT:" in line:
                            ttft = None; e2e = None
                            for tok in line.split():
                                if tok.startswith("TTFT:"):
                                    ttft = float(tok.split(":")[1].rstrip("ms,"))
                                if tok.startswith("E2E:"):
                                    e2e = float(tok.split(":")[1].rstrip("ms,"))
                            if ttft is not None: out["ttft"].append(ttft)
                            if e2e is not None: out["e2e"].append(e2e)
            except FileNotFoundError:
                pass
            return out

        for role, logf in [("prefill", "/tmp/vllm-prefill-0.log"),
                           ("decode", "/tmp/vllm-decode-0.log")]:
            t = vllm_req_times(logf)
            if t["ttft"]:
                log.info("  vLLM[%s]  TTFT avg=%.0fms  E2E avg=%.0fms  (n=%d)",
                         role, sum(t["ttft"]) / len(t["ttft"]),
                         sum(t["e2e"]) / len(t["e2e"]), len(t["ttft"]))
            else:
                log.info("  vLLM[%s]  no finished-request lines found", role)

        # NIXL trace: transfer durations
        try:
            xfer_ms = []
            with open("/tmp/vllm-prefill-0.log") as f:
                for line in f:
                    if "transfer" in line.lower() and ("done" in line.lower() or "us" in line.lower() or "ms" in line.lower()):
                        xfer_ms.append(line.strip()[:160])
            log.info("  NIXL trace lines (prefill-0): %d", len(xfer_ms))
            for l in xfer_ms[:5]:
                log.info("    %s", l)
        except FileNotFoundError:
            pass

        # Error surfacing: gateway tail + vLLM error lines (always)
        for lf in ["/tmp/proxy.log"]:
            try:
                with open(lf) as f:
                    lines = f.readlines()
                    log.info("  proxy.log tail (last %d of %d lines):", min(12, len(lines)), len(lines))
                    for l in lines[-12:]:
                        log.info("    %s", l.rstrip()[:180])
            except FileNotFoundError:
                pass
        for role, lf in [("prefill", "/tmp/vllm-prefill-0.log"), ("decode", "/tmp/vllm-decode-0.log")]:
            try:
                with open(lf) as f:
                    errs = [l.strip()[:200] for l in f if any(k in l for k in ("ERROR", "Exception", "Traceback", " failed", " error:"))]
                log.info("  vLLM[%s] error lines: %d", role, len(errs))
                for l in errs[-5:]:
                    log.info("    %s", l)
            except FileNotFoundError:
                pass

        log.info("=" * 70)
        return {"avg_gateway_ttft_ms": avg_ttft,
                "rows": rows}

    except Exception as e:
        log.error("PROFILE FAILED: %s", e)
        import traceback; traceback.print_exc()
        for lf in ["/tmp/proxy.log"] + [f"/tmp/vllm-prefill-{i}.log" for i in range(2)] + [f"/tmp/vllm-decode-{i}.log" for i in range(2)]:
            try:
                with open(lf) as f:
                    log.error("--- %s (last 30) ---", lf)
                    for l in f.readlines()[-30:]:
                        log.error("  %s", l.rstrip())
            except Exception:
                pass
        return {"status": "failed", "error": str(e)}
    finally:
        kill_procs(procs)

@app.local_entrypoint()
def main(n_req: int = _N_REQ):
    print(f"Profiling nano-dynamo (push mode, {n_req} requests)...")
    result = profile.remote(n_req)
    print(f"Result: {result}")
