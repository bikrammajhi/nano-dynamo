from __future__ import annotations
import asyncio, json, logging, os, subprocess, sys, threading, time
from pathlib import Path
import modal

_MODEL = "Qwen/Qwen2.5-3B-Instruct"
_BLOCK_SIZE = 64
_MAX_MODEL_LEN = 8192
INPUT_TOKENS = 256
OUTPUT_TOKENS = 128

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "curl", "build-essential")
    .uv_pip_install("vllm>=0.7.2", pre="--prerelease=allow")
    .uv_pip_install("ai-dynamo[vllm]", pre="--prerelease=allow")
    .pip_install("nixl", "xxhash", "transformers>=4.40", "huggingface-hub", "httpx", "fastapi", "uvicorn", "sse-starlette", "aiperf")
)

app = modal.App("dynamo-benchmark")

def start_process(cmd, name, log_file=None):
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    if log_file:
        proc = subprocess.Popen(cmd, shell=True, stdout=open(log_file, "w"), stderr=subprocess.STDOUT, env=env)
    else:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        def stream():
            for line in proc.stdout:
                print(f"[{name}] {line}", end="")
        threading.Thread(target=stream, daemon=True).start()
    return proc

def wait_for_models(port, timeout=300):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/models"), timeout=5)
            if resp.status == 200:
                body = json.loads(resp.read().decode())
                if body.get("data"):
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False

def wait_for_health(port, timeout=300):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/health"), timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

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

def kill_procs(procs):
    for p in procs:
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

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
    frontend_port = 8000

    log.info("Config: %d GPUs (%dP + %dD)", total_gpus, num_prefill, num_decode)

    hf_token = os.environ.get("HF_TOKEN")
    dl_cmd = f"from huggingface_hub import snapshot_download; snapshot_download('{_MODEL}'" + (f", token='{hf_token}')" if hf_token else ")")
    log.info("Pre-downloading model...")
    subprocess.run([sys.executable, "-c", dl_cmd], env={**os.environ, "PYTHONHASHSEED": "0"}, check=True)
    log.info("Model ready")

    results = {}
    to_run = SCENARIOS if scenario == "all" else {scenario: SCENARIOS[scenario]}

    for name, cfg in to_run.items():
        log.info("=" * 60)
        log.info("SCENARIO: %s", name.upper())
        log.info("=" * 60)

        d_procs = []
        try:
            d_procs.append(start_process(
                f"python3 -m dynamo.frontend --http-port {frontend_port} --discovery-backend file --router-mode kv",
                "dynamo-fe", f"/tmp/dynamo_{name}_frontend.log"
            ))
            time.sleep(5)
            if not wait_for_health(frontend_port, timeout=60):
                raise RuntimeError("Dynamo frontend failed")
            log.info("Dynamo frontend ready")

            widx = 0
            for i in range(num_prefill):
                sp, scp, kep = 8081 + widx, 20090 + widx, 20080 + widx
                widx += 1
                d_procs.append(start_process(
                    f"DYN_SYSTEM_PORT={sp} VLLM_NIXL_SIDE_CHANNEL_PORT={scp} CUDA_VISIBLE_DEVICES={i} "
                    f"python3 -m dynamo.vllm --model {_MODEL} --block-size {_BLOCK_SIZE} --disaggregation-mode prefill "
                    f"--kv-transfer-config '{{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}}' --discovery-backend file "
                    f"--kv-events-config '{{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:{kep}\",\"enable_kv_cache_events\":true}}'",
                    f"d-pf-{i}", f"/tmp/dynamo_{name}_pf_{i}.log"
                ))
            for i in range(num_decode):
                gpu_id = num_prefill + i
                sp, scp = 8081 + widx, 20090 + widx
                widx += 1
                d_procs.append(start_process(
                    f"DYN_SYSTEM_PORT={sp} VLLM_NIXL_SIDE_CHANNEL_PORT={scp} CUDA_VISIBLE_DEVICES={gpu_id} "
                    f"python3 -m dynamo.vllm --model {_MODEL} --block-size {_BLOCK_SIZE} --disaggregation-mode decode "
                    f"--kv-transfer-config '{{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}}' --discovery-backend file",
                    f"d-dc-{i}", f"/tmp/dynamo_{name}_dc_{i}.log"
                ))

            log.info("Waiting for Dynamo workers...")
            if not wait_for_models(frontend_port, timeout=300):
                raise RuntimeError("Dynamo workers failed to register")
            log.info("Dynamo ready")

            artifact_dir = f"/tmp/aiperf_dynamo_{name}"
            cmd = cfg["cmd"](f"http://localhost:{frontend_port}", artifact_dir)
            result = await run_aiperf(name, cmd, artifact_dir)
            results[name] = result

            if result:
                ttft = result.get("time_to_first_token", {}).get("avg", 0)
                tp = result.get("output_token_throughput", {}).get("avg", 0)
                lat = result.get("request_latency", {}).get("avg", 0)
                log.info("  TTFT=%.0fms  tok/s=%.0f  lat=%.0fms", ttft, tp, lat)
        except Exception as e:
            log.error("Scenario %s failed: %s", name, e)
            results[name] = {}
        finally:
            kill_procs(d_procs)
            if d_procs:
                log.info("Waiting 30s for GPU memory to release...")
                time.sleep(30)

    log.info("=" * 60)
    log.info("RESULTS: %dP+%dD", num_prefill, num_decode)
    log.info("=" * 60)
    for name in results:
        r = results[name]
        ttft = r.get("time_to_first_token", {}).get("avg", 0)
        tp = r.get("output_token_throughput", {}).get("avg", 0)
        lat = r.get("request_latency", {}).get("avg", 0)
        log.info("  %-18s  TTFT=%6.0fms  tok/s=%6.0f  lat=%6.0fms", name, ttft, tp, lat)
    log.info("=" * 60)

    return results


@app.local_entrypoint()
def main(scenario: str = "all", num_prefill: int = 2, num_decode: int = 2):
    print(f"nvidia-dynamo benchmark | {num_prefill}P + {num_decode}D | {scenario}")
    run_benchmark.remote(scenario, num_prefill, num_decode)
