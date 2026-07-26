from __future__ import annotations
import asyncio, json, logging, os, subprocess, sys, threading, time
from pathlib import Path
from typing import List
import modal

_MODEL = "Qwen/Qwen2.5-3B-Instruct"
_BLOCK_SIZE = 64
_MAX_MODEL_LEN = 8192
INPUT_TOKENS = 256
OUTPUT_TOKENS = 128

_REPO_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "curl", "build-essential")
    .pip_install(
        "vllm>=0.7.2",
        "nixl", "xxhash", "transformers>=4.40", "huggingface-hub",
        "httpx", "fastapi", "uvicorn", "sse-starlette", "aiperf",
    )
    .add_local_dir(_REPO_ROOT / "src", "/root/src")
)

app = modal.App("nano-benchmark")


def start_process(cmd, name, log_file=None, env=None):
    run_env = {**(env or os.environ), "PYTHONHASHSEED": "0"}
    if log_file:
        proc = subprocess.Popen(cmd, shell=True, stdout=open(log_file, "w"), stderr=subprocess.STDOUT, env=run_env)
    else:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=run_env)
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
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

def start_vllm_server(gpu_id, port, block_size, kv_role, kv_rank, side_channel_port):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["UCX_NET_DEVICES"] = "all"
    env["NCCL_CUMEM_ENABLE"] = "1"
    env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
    env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(side_channel_port)

    kv_config = json.dumps({"kv_connector": "NixlConnector", "kv_role": kv_role, "kv_rank": kv_rank})

    cmd = (
        f"{sys.executable} -m vllm.entrypoints.openai.api_server "
        f"--model {_MODEL} --port {port} --gpu-memory-utilization 0.85 "
        f"--max-model-len {_MAX_MODEL_LEN} --block-size {block_size} "
        f"--dtype auto --enforce-eager "
        f"--kv-transfer-config '{kv_config}'"
    )

    log_name = f"vllm-{kv_role}-g{gpu_id}"
    return start_process(cmd, log_name, f"/tmp/{log_name}.log", env)


async def run_aiperf(name, cmd, artifact_dir):
    log = logging.getLogger("benchmark")
    log.info("Running AIPerf [%s]", name)
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        log.error("AIPerf failed for %s (code %d)", name, proc.returncode)
        return {}
    rf = Path(artifact_dir) / "profile_export_aiperf.json"
    if rf.exists():
        with open(rf) as f:
            return json.load(f)
    return {}


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
}


@app.function(image=image, gpu="A10G:4", timeout=7200, max_containers=1, secrets=[modal.Secret.from_name("huggingface-secret")])
async def run_benchmark(scenario: str = "all", num_prefill: int = 2, num_decode: int = 2):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("benchmark")

    num_prefill = max(1, min(4, num_prefill))
    num_decode = max(1, min(4, num_decode))
    total_gpus = num_prefill + num_decode
    prefill_ports = [8100 + i for i in range(num_prefill)]
    decode_ports = [8200 + i for i in range(num_decode)]
    gateway_port = 8787

    log.info("Config: %d GPUs (%d prefill + %d decode)", total_gpus, num_prefill, num_decode)

    hf_token = os.environ.get("HF_TOKEN")
    dl_cmd = f"from huggingface_hub import snapshot_download; snapshot_download('{_MODEL}'" + (f", token='{hf_token}')" if hf_token else ")")
    log.info("Pre-downloading model...")
    subprocess.run([sys.executable, "-c", dl_cmd], env={**os.environ, "PYTHONHASHSEED": "0"}, check=True)
    log.info("Model ready")

    procs = []
    try:
        for i, port in enumerate(prefill_ports):
            gpu_id = i
            side_channel_port = 5600 + i
            log.info("Starting PREFILL vLLM %d on GPU %d, port %d", i + 1, gpu_id, port)
            proc = start_vllm_server(gpu_id=gpu_id, port=port, block_size=_BLOCK_SIZE, kv_role="kv_producer", kv_rank=i, side_channel_port=side_channel_port)
            procs.append(proc)
            if not wait_for_endpoint(port, "/v1/models", timeout=300):
                raise RuntimeError(f"Prefill vLLM {i+1} failed on port {port}")
            log.info("Prefill server %d ready", i + 1)

        for i, port in enumerate(decode_ports):
            gpu_id = num_prefill + i
            side_channel_port = 5700 + i
            log.info("Starting DECODE vLLM %d on GPU %d, port %d", i + 1, gpu_id, port)
            proc = start_vllm_server(gpu_id=gpu_id, port=port, block_size=_BLOCK_SIZE, kv_role="kv_consumer", kv_rank=i, side_channel_port=side_channel_port)
            procs.append(proc)
            if not wait_for_endpoint(port, "/v1/models", timeout=300):
                raise RuntimeError(f"Decode vLLM {i+1} failed on port {port}")
            log.info("Decode server %d ready", i + 1)

        prefill_ports_str = " ".join(map(str, prefill_ports))
        decode_ports_str = " ".join(map(str, decode_ports))
        log.info("Starting KV Router gateway on port %d", gateway_port)
        gateway_proc = start_process(
            f"{sys.executable} -m src.gateway --prefill-ports {prefill_ports_str} --decode-ports {decode_ports_str} --port {gateway_port}",
            "kv-router", "/tmp/kv-router.log"
        )
        procs.append(gateway_proc)
        if not wait_for_endpoint(gateway_port, "/health", timeout=30):
            raise RuntimeError(f"KV Router gateway failed on port {gateway_port}")
        log.info("Gateway ready")

        url = f"http://localhost:{gateway_port}"
        to_run = SCENARIOS if scenario == "all" else {scenario: SCENARIOS[scenario]}
        all_results = {}

        for name, cfg in to_run.items():
            artifact_dir = f"/tmp/aiperf_{name}"
            cmd = cfg["cmd"](url, artifact_dir)
            log.info("=" * 60)
            log.info("SCENARIO: %s", name.upper())
            log.info("=" * 60)
            result = await run_aiperf(name, cmd, artifact_dir)
            all_results[name] = result

            if result:
                ttft = result.get("time_to_first_token", {}).get("avg", 0)
                tp = result.get("output_token_throughput", {}).get("avg", 0)
                lat = result.get("request_latency", {}).get("avg", 0)
                log.info("  TTFT=%.0fms  tok/s=%.0f  lat=%.0fms", ttft, tp, lat)

    finally:
        kill_procs(procs)

    log.info("=" * 60)
    log.info("RESULTS: %dP+%dD", num_prefill, num_decode)
    log.info("=" * 60)
    for name in all_results:
        r = all_results[name]
        ttft = r.get("time_to_first_token", {}).get("avg", 0)
        tp = r.get("output_token_throughput", {}).get("avg", 0)
        lat = r.get("request_latency", {}).get("avg", 0)
        log.info("  %-18s  TTFT=%6.0fms  tok/s=%6.0f  lat=%6.0fms", name, ttft, tp, lat)
    log.info("=" * 60)

    return all_results


@app.local_entrypoint()
def main(scenario: str = "all", num_prefill: int = 2, num_decode: int = 2):
    print(f"nano-dynamo benchmark | {num_prefill}P + {num_decode}D | {scenario}")
    run_benchmark.remote(scenario, num_prefill, num_decode)
