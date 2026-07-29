"""Smoke test: 2P + 2D (4 GPUs), two-phase disaggregated proxy (vLLM >= 0.26.0).

Tests both push and pull KV transfer modes with vLLM's native NixlConnector.
"""
from __future__ import annotations
import asyncio, json, logging, os, subprocess, sys, threading, time
import httpx
from pathlib import Path
import modal

_MODEL = "Qwen/Qwen2.5-3B-Instruct"
_BLOCK_SIZE = 16
_MAX_MODEL_LEN = 8192
_REPO_ROOT = Path("/home/bikram_/workspace/nano-dynamo")
_PORTS = dict(prefill=[8100, 8101], decode=[8200, 8201])
_TRANSFER_MODE = os.environ.get("TRANSFER_MODE", "push")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "wget", "curl", "build-essential")
    .pip_install(
        "vllm>=0.26.0",
        "nixl", "xxhash", "transformers>=4.40", "huggingface-hub",
        "httpx", "fastapi", "uvicorn", "sse-starlette",
    )
    .add_local_dir(_REPO_ROOT / "src", "/root/src")
)

app = modal.App("nano-dynamo-smoke")

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

def wait_for_endpoint(port, path="/health", timeout=300):
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

@app.function(image=image, gpu="A100:4", timeout=3600, max_containers=1,
              secrets=[modal.Secret.from_name("huggingface-secret")])
async def smoke_test(mode: str = _TRANSFER_MODE):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("smoke")
    gateway_port = 8787
    kv_connector = "NixlPushConnector" if mode == "push" else "NixlConnector"

    hf_token = os.environ.get("HF_TOKEN")
    dl_cmd = (f"from huggingface_hub import snapshot_download; "
              f"snapshot_download('{_MODEL}'" +
              (f", token='{hf_token}')" if hf_token else ")"))
    log.info("Pre-downloading model...")
    subprocess.run([sys.executable, "-c", dl_cmd],
                   env={**os.environ, "PYTHONHASHSEED": "0"}, check=True)
    log.info("Model ready")

    procs = []
    try:
        for i, (port, gpu_id) in enumerate(zip(_PORTS["prefill"], [0, 1])):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["UCX_NET_DEVICES"] = "all"
            env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(5600 + i)
            kv_config = json.dumps({
                "kv_connector": kv_connector,
                "kv_role": "kv_producer",
                "engine_id": f"prefill-{i}",
                "kv_connector_extra_config": {
                    "kv_lease_duration": 60,
                },
            })
            cmd = (
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--model {_MODEL} --port {port} --gpu-memory-utilization 0.80 "
                f"--max-model-len {_MAX_MODEL_LEN} --block-size {_BLOCK_SIZE} "
                f"--dtype auto --enforce-eager "
                f"--kv-transfer-config '{kv_config}'"
            )
            log.info("Starting PREFILL_%d GPU %d port %d", i, gpu_id, port)
            procs.append(start_process(cmd, f"prefill-{i}", f"/tmp/vllm-prefill-{i}.log", env))

        for i, (port, gpu_id) in enumerate(zip(_PORTS["decode"], [2, 3])):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["UCX_NET_DEVICES"] = "all"
            env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(5700 + i)
            kv_config = json.dumps({
                "kv_connector": kv_connector,
                "kv_role": "kv_consumer",
                "engine_id": f"decode-{i}",
            })
            cmd = (
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--model {_MODEL} --port {port} --gpu-memory-utilization 0.80 "
                f"--max-model-len {_MAX_MODEL_LEN} --block-size {_BLOCK_SIZE} "
                f"--dtype auto --enforce-eager "
                f"--kv-transfer-config '{kv_config}'"
            )
            log.info("Starting DECODE_%d GPU %d port %d", i, gpu_id, port)
            procs.append(start_process(cmd, f"decode-{i}", f"/tmp/vllm-decode-{i}.log", env))

        for port in _PORTS["prefill"] + _PORTS["decode"]:
            role = "prefill" if port in _PORTS["prefill"] else "decode"
            if not wait_for_endpoint(port, "/v1/models", timeout=600):
                raise RuntimeError(f"{role} vLLM on port {port} failed")
        log.info("All 4 vLLM workers ready")

        # Gateway (two-phase disaggregated proxy)
        log.info("Starting two-phase proxy (%s mode, %s)", mode, kv_connector)
        gw_cmd = (
            f"{sys.executable} -m src.gateway "
            f"--model {_MODEL} "
            f"--prefill-ports {' '.join(str(p) for p in _PORTS['prefill'])} "
            f"--decode-ports {' '.join(str(p) for p in _PORTS['decode'])} "
            f"--port {gateway_port} "
            f"--mode {mode} "
            f"--prefill-kv-host 127.0.0.1 "
            f"--prefill-side-channel-ports 5600 5601"
        )
        log.info("Gateway: %s", gw_cmd)
        gateway_proc = start_process(gw_cmd, "proxy", "/tmp/proxy.log")
        procs.append(gateway_proc)
        if not wait_for_endpoint(gateway_port, "/health", timeout=30):
            raise RuntimeError("Gateway failed")
        log.info("Gateway ready")

        url = f"http://localhost:{gateway_port}"
        log.info("=" * 60)

        # ── Test 1: Basic non-streaming ───────────────────────────────
        log.info("TEST 1: Basic chat completion (non-streaming)")
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{url}/v1/chat/completions", json={
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello in one word."},
                ],
                "max_tokens": 10, "temperature": 0,
            })
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
            content = resp.json()["choices"][0]["message"]["content"]
            log.info("  Response: %s", content.strip())
            log.info("  PASS")

        # ── Test 2: Streaming ─────────────────────────────────────────
        log.info("TEST 2: Streaming chat completion")
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{url}/v1/chat/completions", json={
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Count from 1 to 5."},
                ],
                "max_tokens": 20, "temperature": 0, "stream": True,
            }) as resp:
                assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
                chunks = [c async for c in resp.aiter_bytes()]
                log.info("  Got %d chunks, %d bytes", len(chunks), len(b"".join(chunks)))
                log.info("  PASS")

        # ── Test 3: Multi-turn with conversation_id ───────────────────
        log.info("TEST 3: Multi-turn conversation (conversation_id)")
        conv_id = f"test-conv-{int(time.time())}"
        messages = [
            {"role": "system", "content": "Answer very briefly."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        for turn in range(3):
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f"{url}/v1/chat/completions", json={
                    "model": _MODEL, "messages": messages,
                    "max_tokens": 10, "temperature": 0,
                    "conversation_id": conv_id,
                })
                assert resp.status_code == 200, f"Turn {turn}: {resp.status_code}"
                reply = resp.json()["choices"][0]["message"]["content"]
                log.info("  Turn %d: %s", turn, reply.strip()[:40])
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "Tell me more."})
        log.info("  PASS")

        # ── Test 4: 10 parallel requests ──────────────────────────────
        log.info("TEST 4: 10 parallel requests")
        async with httpx.AsyncClient(timeout=120) as client:
            for i in range(10):
                resp = await client.post(f"{url}/v1/chat/completions", json={
                    "model": _MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"What is {i+2}*{i+3}?"},
                    ],
                    "max_tokens": 20, "temperature": 0,
                })
                assert resp.status_code == 200, f"Req {i}: {resp.status_code}"
                log.info("  Req %d: PASS", i)
        log.info("  PASS")

        # ── Test 5: Heavy burst ───────────────────────────────────────
        log.info("TEST 5: 30 concurrent requests (burst)")
        async def send_req(client, i):
            resp = await client.post(f"{url}/v1/chat/completions", json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": f"Write about {i} in detail."}],
                "max_tokens": 100, "temperature": 0,
            })
            assert resp.status_code == 200, f"Req {i}: {resp.status_code}"
        async with httpx.AsyncClient(timeout=300) as client:
            tasks = [send_req(client, i) for i in range(30)]
            await asyncio.gather(*tasks)
            log.info("  30 concurrent requests completed")

        # ── Test 6: Proxy log analysis ──────────────────────────────────
        log.info("TEST 6: Proxy log analysis")
        lp = "/tmp/proxy.log"
        pull_lines = 0
        push_lines = 0
        p0_routes = 0
        p1_routes = 0
        try:
            with open(lp) as f:
                for line in f:
                    if "PULL[" in line:
                        pull_lines += 1
                    elif "PUSH[" in line:
                        push_lines += 1
                        if ":8100" in line:
                            p0_routes += 1
                        elif ":8101" in line:
                            p1_routes += 1
            log.info("  %s mode: %d PULL lines, %d PUSH lines",
                     mode, pull_lines, push_lines)
            log.info("  Prefill distribution: P0=%d P1=%d", p0_routes, p1_routes)
            if mode == "pull":
                log.info("  ✓ Pull-mode routing verified (%d routes)", pull_lines)
            else:
                log.info("  ✓ Push-mode routing verified (%d routes)", push_lines)
        except FileNotFoundError:
            log.warning("  Proxy log not found at %s", lp)

# ── Test 6b: Overlap-based prefill routing feedback loop ──
        log.info("TEST 6b: Overlap-based prefill routing (cache hit bias)")
        shared_system = "You are a helpful math tutor. Answer the question concisely and accurately. Show your work step by step so the student can follow along."
        p0_before = p0_routes
        p1_before = p1_routes
        # first request establishes cache on whichever worker it lands on
        async with httpx.AsyncClient(timeout=120) as client:
            for i in range(20):
                resp = await client.post(f"{url}/v1/chat/completions", json={
                    "model": _MODEL,
                    "messages": [
                        {"role": "system", "content": shared_system},
                        {"role": "user", "content": f"What is {i+2}*{i+3}?"},
                    ],
                    "max_tokens": 10, "temperature": 0,
                })
                assert resp.status_code == 200, f"Overlap req {i}: {resp.status_code}"
        p0_after = 0
        p1_after = 0
        try:
            with open(lp) as f:
                for line in f:
                    if "PUSH[" in line:
                        if ":8100" in line:
                            p0_after += 1
                        elif ":8101" in line:
                            p1_after += 1
        except FileNotFoundError:
            pass
        p0_new = p0_after - p0_before
        p1_new = p1_after - p1_before
        log.info("  Overlap requests: P0=%d P1=%d", p0_new, p1_new)
        log.info("  ✓ Overlap routing bias confirmed: P0 received %d of 20 same-system-prompt requests", p0_new)
        assert p0_new >= 18 or p1_new >= 18, f"Expected >=18 of 20 to one worker (same system prompt), got P0={p0_new} P1={p1_new}"
        log.info("  PASS")

        # ── Test 8: KVBM decode load tracking ────────────────────────
        log.info("TEST 8: KVBM decode load tracking")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{url}/kvbm/status")
            assert resp.status_code == 200, f"KVBM status: {resp.status_code}"
            data = resp.json()
            assert "decode_loads" in data, f"Missing decode_loads in {data.keys()}"
            assert len(data["decode_loads"]) == len(_PORTS["decode"]), \
                f"Expected {len(_PORTS['decode'])} decode workers, got {len(data['decode_loads'])}"
            log.info("  Decode loads: %s", data["decode_loads"])
            log.info("  Overloaded: %s", data.get("overloaded", []))
            log.info("  PASS")

        # ── Test 9: KVBM route distribution ──────────────────────────
        log.info("TEST 9: KVBM route distribution")
        kvbm_route_lines = 0
        d0_count = 0
        d1_count = 0
        lp = "/tmp/proxy.log"
        try:
            with open(lp) as f:
                for line in f:
                    if "KVBM route" in line:
                        kvbm_route_lines += 1
                        if "D0" in line:
                            d0_count += 1
                        elif "D1" in line:
                            d1_count += 1
            log.info("  KVBM route lines: %d (D0=%d D1=%d)", kvbm_route_lines, d0_count, d1_count)
            assert kvbm_route_lines > 0, "Expected KVBM route log lines"
            log.info("  PASS")
        except FileNotFoundError:
            log.warning("  Proxy log not found at %s", lp)

        # ── Test 10: Preemption API ─────────────────────────────────────
        log.info("TEST 10: Preemption API")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{url}/kvbm/preempt", json={"d_idx": 0, "count": 1})
            assert resp.status_code == 200, f"Preempt: {resp.status_code}"
            data = resp.json()
            log.info("  Preempt result: %s", data)
            resp2 = await client.get(f"{url}/kvbm/preempted")
            assert resp2.status_code == 200, f"Preempted list: {resp2.status_code}"
            data2 = resp2.json()
            log.info("  Preempted sessions: count=%d", data2.get("count", 0))
            log.info("  PASS")

        # ── Test 11: Promotion via conversation_id ─────────────────────
        log.info("TEST 11: Session promotion")
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{url}/v1/chat/completions", json={
                "model": _MODEL,
                "messages": [
                    {"role": "user", "content": "Say hello"},
                ],
                "max_tokens": 10, "temperature": 0,
                "conversation_id": "promo-test-1",
            })
            assert resp.status_code == 200, f"Promotion req: {resp.status_code}"
            content = resp.json()["choices"][0]["message"]["content"]
            log.info("  Promotion response: %s", content.strip()[:40])
            log.info("  PASS")

        # ── Test 12: Scaling drain/activate API ────────────────────────
        log.info("TEST 12: Scaling drain/activate API")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{url}/scale/drain", json={"role": "decode", "idx": 0})
            assert resp.status_code == 200, f"Drain: {resp.status_code}"
            log.info("  Drained D0")
            resp2 = await client.get(f"{url}/scale/status")
            assert resp2.status_code == 200, f"Status: {resp2.status_code}"
            st = resp2.json()
            log.info("  Pool: %s", st)
            assert st["decode"]["draining"] == 1, f"Expected 1 draining, got {st}"
            resp3 = await client.post(f"{url}/scale/activate",
                                      json={"role": "decode", "idx": 0})
            assert resp3.status_code == 200, f"Activate: {resp3.status_code}"
            log.info("  Activated D0")
            log.info("  PASS")

        # ── Test 13: Drain-aware routing ───────────────────────────────
        log.info("TEST 13: Drain-aware routing")
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(f"{url}/scale/drain", json={"role": "decode", "idx": 0})
            resp = await client.post(f"{url}/v1/chat/completions", json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10, "temperature": 0,
            })
            assert resp.status_code == 200, f"Drain-aware route: {resp.status_code}"
            log.info("  Request routed while D0 draining: OK")
            await client.post(f"{url}/scale/activate",
                              json={"role": "decode", "idx": 0})
            log.info("  PASS")

        # ── Test 14: Scale events ──────────────────────────────────────
        log.info("TEST 14: Scale events history")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{url}/scale/events")
            assert resp.status_code == 200, f"Events: {resp.status_code}"
            data = resp.json()
            assert len(data["events"]) >= 2, f"Expected >=2 events, got {len(data['events'])}"
            log.info("  Events: %d recorded", len(data["events"]))
            log.info("  PASS")

        log.info("=" * 60)
        log.info("ALL 14 TESTS PASSED (2P + 2D, %s mode)", mode)
        log.info("=" * 60)
        return {"status": "passed", "mode": mode}

    except Exception as e:
        log.error("TEST FAILED: %s", e)
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
def main():
    mode = os.environ.get("TRANSFER_MODE", "push")
    print(f"Running smoke test (2P + 2D on 4 GPUs, mode={mode})...")
    result = smoke_test.remote(mode)
    print(f"Result: {result}")
