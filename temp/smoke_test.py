"""Quick smoke test: 1 prefill + 1 decode, 5 conversations, verify routing works."""
from __future__ import annotations
import asyncio, json, logging, os, subprocess, sys, threading, time
from pathlib import Path
import modal

_MODEL = "Qwen/Qwen2.5-3B-Instruct"
_BLOCK_SIZE = 64
_MAX_MODEL_LEN = 8192
_REPO_ROOT = Path("/home/bikram_/workspace/nano-dynamo")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "curl", "build-essential")
    .pip_install(
        "vllm>=0.7.2",
        "nixl", "xxhash", "transformers>=4.40", "huggingface-hub",
        "httpx", "fastapi", "uvicorn", "sse-starlette",
    )
    .add_local_dir(_REPO_ROOT / "src", "/root/src")
)

app = modal.App("nano-smoke-test")

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

@app.function(image=image, gpu="A10G:2", timeout=3600, max_containers=1,
              secrets=[modal.Secret.from_name("huggingface-secret")])
async def smoke_test():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("smoke")

    prefill_port = 8100
    decode_port = 8200
    gateway_port = 8787

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
        # Prefill vLLM
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["UCX_NET_DEVICES"] = "all"
        env["NCCL_CUMEM_ENABLE"] = "1"
        env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
        env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "5600"
        kv_config = json.dumps({"kv_connector": "NixlConnector", "kv_role": "kv_producer", "kv_rank": 0})
        cmd = (
            f"{sys.executable} -m vllm.entrypoints.openai.api_server "
            f"--model {_MODEL} --port {prefill_port} --gpu-memory-utilization 0.85 "
            f"--max-model-len {_MAX_MODEL_LEN} --block-size {_BLOCK_SIZE} "
            f"--dtype auto --enforce-eager "
            f"--kv-transfer-config '{kv_config}'"
        )
        log.info("Starting PREFILL vLLM on GPU 0, port %d", prefill_port)
        procs.append(start_process(cmd, "prefill", "/tmp/vllm-prefill.log", env))
        if not wait_for_endpoint(prefill_port, "/v1/models", timeout=300):
            raise RuntimeError("Prefill vLLM failed")
        log.info("Prefill ready")

        # Decode vLLM
        env2 = os.environ.copy()
        env2["CUDA_VISIBLE_DEVICES"] = "1"
        env2["UCX_NET_DEVICES"] = "all"
        env2["NCCL_CUMEM_ENABLE"] = "1"
        env2["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
        env2["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "5700"
        kv_config2 = json.dumps({"kv_connector": "NixlConnector", "kv_role": "kv_consumer", "kv_rank": 0})
        cmd2 = (
            f"{sys.executable} -m vllm.entrypoints.openai.api_server "
            f"--model {_MODEL} --port {decode_port} --gpu-memory-utilization 0.85 "
            f"--max-model-len {_MAX_MODEL_LEN} --block-size {_BLOCK_SIZE} "
            f"--dtype auto --enforce-eager "
            f"--kv-transfer-config '{kv_config2}'"
        )
        log.info("Starting DECODE vLLM on GPU 1, port %d", decode_port)
        procs.append(start_process(cmd2, "decode", "/tmp/vllm-decode.log", env2))
        if not wait_for_endpoint(decode_port, "/v1/models", timeout=300):
            raise RuntimeError("Decode vLLM failed")
        log.info("Decode ready")

        # Gateway
        log.info("Starting KV Router gateway")
        gateway_proc = start_process(
            f"{sys.executable} -m src.gateway --prefill-ports {prefill_port} "
            f"--decode-ports {decode_port} --port {gateway_port}",
            "kv-router", "/tmp/kv-router.log"
        )
        procs.append(gateway_proc)
        if not wait_for_endpoint(gateway_port, "/health", timeout=30):
            raise RuntimeError("Gateway failed")
        log.info("Gateway ready")

        url = f"http://localhost:{gateway_port}"

        # Test 1: Basic chat (non-streaming)
        log.info("=" * 60)
        log.info("TEST 1: Basic chat completion (non-streaming)")
        import httpx
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

        # Test 2: Streaming
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

        # Test 3: Multi-turn with cache reuse
        log.info("TEST 3: Multi-turn conversation (5 turns)")
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer very briefly."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        for turn in range(5):
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f"{url}/v1/chat/completions", json={
                    "model": _MODEL, "messages": messages,
                    "max_tokens": 10, "temperature": 0,
                })
                assert resp.status_code == 200, f"Turn {turn}: {resp.status_code}"
                reply = resp.json()["choices"][0]["message"]["content"]
                log.info("  Turn %d: user=%s → %s", turn,
                         messages[-1]["content"][:30], reply.strip()[:30])
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user",
                                 "content": "Tell me more in one sentence."})
        log.info("  PASS")

        # Test 4: Parallel requests
        log.info("TEST 4: Multiple parallel requests")
        async with httpx.AsyncClient(timeout=120) as client:
            for i in range(5):
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

        log.info("=" * 60)
        log.info("ALL SMOKE TESTS PASSED")
        log.info("=" * 60)
        return {"status": "passed"}

    except Exception as e:
        log.error("TEST FAILED: %s", e)
        import traceback; traceback.print_exc()
        for lf in ["/tmp/kv-router.log", "/tmp/vllm-prefill.log", "/tmp/vllm-decode.log"]:
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
    print("Running smoke test (1P + 1D)...")
    result = smoke_test.remote()
    print(f"Result: {result}")
