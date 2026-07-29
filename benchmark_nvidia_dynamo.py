from __future__ import annotations
import asyncio, json, logging, os, subprocess, sys, threading, time
from pathlib import Path
import modal

_MODEL = "Qwen/Qwen2.5-3B-Instruct"
_BLOCK_SIZE = 64
_MAX_MODEL_LEN = 32768
_NUM_PREFILL = 2
_NUM_DECODE = 2
_TOTAL_GPUS = _NUM_PREFILL + _NUM_DECODE
FRONTEND_PORT = 8000

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
    import urllib.request, urllib.error
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
    import urllib.request, urllib.error
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
    log.info("Running AIPerf [%s]: %s", name, " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        log.error("AIPerf failed for %s (code %d)", name, proc.returncode)
        if stderr.decode().strip():
            log.error("STDERR: %s", stderr.decode()[-1500:])
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

# ── Benchmark scenarios (same as nano-dynamo) ────────────────────────────

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
            "--synthetic-input-tokens-mean", "256",
            "--output-tokens-mean", "128",
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


@app.function(image=image, gpu=f"A100:{_TOTAL_GPUS}", timeout=7200, max_containers=1, secrets=[modal.Secret.from_name("huggingface-secret")])
async def run_benchmark(scenario: str = "all"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("benchmark")
    hf_token = os.environ.get("HF_TOKEN")
    dl_cmd = f"from huggingface_hub import snapshot_download; snapshot_download('{_MODEL}'" + (f", token='{hf_token}')" if hf_token else ")")
    log.info("Pre-downloading model...")
    subprocess.run([sys.executable, "-c", dl_cmd], env={**os.environ, "PYTHONHASHSEED": "0"}, check=True)
    log.info("Model ready")

    results = {}

    to_run = SCENARIOS if scenario == "all" else {scenario: SCENARIOS[scenario]}

    for name, cfg in to_run.items():
        log.info("=" * 70)
        log.info("SCENARIO: %s — %s", name.upper(), cfg["description"])
        log.info("=" * 70)

        d_procs, d_logs = [], []
        try:
            fe_log = f"/tmp/dynamo_{name}_frontend.log"
            d_logs.append(fe_log)
            d_procs.append(start_process(
                f"python3 -m dynamo.frontend --http-port {FRONTEND_PORT} --discovery-backend file --router-mode kv",
                "dynamo-fe", fe_log
            ))
            time.sleep(5)
            if not wait_for_health(FRONTEND_PORT, timeout=60):
                raise RuntimeError("Dynamo frontend failed")
            log.info("Dynamo frontend ready")

            widx = 0
            for i in range(_NUM_PREFILL):
                sp, scp, kep = 8081 + widx, 20090 + widx, 20080 + widx
                widx += 1
                lg = f"/tmp/dynamo_{name}_pf_{i}.log"
                d_logs.append(lg)
                d_procs.append(start_process(
                    f"DYN_SYSTEM_PORT={sp} VLLM_NIXL_SIDE_CHANNEL_PORT={scp} CUDA_VISIBLE_DEVICES={i} "
                    f"python3 -m dynamo.vllm --model {_MODEL} --block-size {_BLOCK_SIZE} --disaggregation-mode prefill "
                    f"--kv-transfer-config '{{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}}' --discovery-backend file "
                    f"--kv-events-config '{{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:{kep}\",\"enable_kv_cache_events\":true}}'",
                    f"d-pf-{i}", lg
                ))
            for i in range(_NUM_DECODE):
                gpu_id = _NUM_PREFILL + i
                sp, scp = 8081 + widx, 20090 + widx
                widx += 1
                lg = f"/tmp/dynamo_{name}_dc_{i}.log"
                d_logs.append(lg)
                d_procs.append(start_process(
                    f"DYN_SYSTEM_PORT={sp} VLLM_NIXL_SIDE_CHANNEL_PORT={scp} CUDA_VISIBLE_DEVICES={gpu_id} "
                    f"python3 -m dynamo.vllm --model {_MODEL} --block-size {_BLOCK_SIZE} --disaggregation-mode decode "
                    f"--kv-transfer-config '{{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}}' --discovery-backend file",
                    f"d-dc-{i}", lg
                ))

            log.info("Waiting for Dynamo workers...")
            if not wait_for_models(FRONTEND_PORT, timeout=300):
                raise RuntimeError("Dynamo workers failed to register")
            log.info("Dynamo ready — running AIPerf")

            artifact_dir = f"/tmp/aiperf_dynamo_{name}"
            cmd = cfg["cmd"](f"http://localhost:{FRONTEND_PORT}", artifact_dir)
            result = await run_aiperf(name, cmd, artifact_dir)
            results[name] = result

            if result:
                ttft = result.get("time_to_first_token", {}).get("avg", 0)
                tp = result.get("output_token_throughput", {}).get("avg", 0)
                lat = result.get("request_latency", {}).get("avg", 0)
                goodput = result.get("goodput", {}).get("avg", 0)
                log.info("  TTFT=%.0fms  tok/s=%.0f  lat=%.0fms  goodput=%.1f", ttft, tp, lat, goodput)
        except Exception as e:
            log.error("Scenario %s failed: %s", name, e)
            results[name] = {}
        finally:
            kill_procs(d_procs)
            if d_procs:
                log.info("Waiting 30s for GPU memory to release...")
                time.sleep(30)

    log.info("=" * 70)
    log.info("BENCHMARK SUMMARY")
    log.info("=" * 70)
    log.info("  Config: %d GPUs (%dP + %dD), %s", _TOTAL_GPUS, _NUM_PREFILL, _NUM_DECODE, _MODEL)
    log.info("")

    header = f"{'Scenario':<20} {'TTFT(ms)':>10} {'tok/s':>10} {'Lat(ms)':>10} {'Goodput':>10}"
    log.info(header)
    log.info("-" * 65)
    for name in results:
        r = results[name]
        ttft = r.get("time_to_first_token", {}).get("avg", 0)
        tp = r.get("output_token_throughput", {}).get("avg", 0)
        lat = r.get("request_latency", {}).get("avg", 0)
        gp = r.get("goodput", {}).get("avg", 0)
        log.info("  %-18s %10.0f %10.0f %10.0f %10.1f", name, ttft, tp, lat, gp)

    log.info("=" * 70)

    output = Path("/tmp/dynamo_benchmark_results.json")
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", output)
    return results


@app.local_entrypoint()
def main(scenario: str = "all"):
    print("NVIDIA Dynamo benchmark suite (disaggregated prefill/decode)")
    print(f"Model: {_MODEL} | GPUs: {_TOTAL_GPUS}xA100 ({_NUM_PREFILL}P + {_NUM_DECODE}D)")
    print(f"Scenarios: {', '.join(SCENARIOS.keys()) if scenario == 'all' else scenario}")
    run_benchmark.remote(scenario)
