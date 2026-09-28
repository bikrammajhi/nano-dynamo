"""Aggregated control (audit experiment #1): one vLLM engine, TP=4, prefix caching.

Same 4×A100, same model (Qwen3-14B-FP8), same AIPerf scenarios as the
disaggregated runs. Answers: does disaggregation earn its complexity here,
or does one aggregated engine match/beat it? No gateway, no transfer, no
router - the floor every disaggregated number must clear.

Run: modal run bench_aggregated_control.py [--scenario multi_turn|mixed_workload|all|stress]
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
_PORT = 8000

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "wget", "curl", "build-essential")
    .uv_pip_install("vllm", pre="--prerelease=allow")
    .pip_install("huggingface-hub", "aiperf")
)

app = modal.App("nano-dynamo-aggregated")

SCENARIOS = {
    "multi_turn": {
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
        # Same crossover probe as modal_bench.py: ISL 4k, OSL 256, conc 30.
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
    proc = subprocess.Popen(cmd, shell=True, stdout=open(log_file, "w"),
                            stderr=subprocess.STDOUT, env=run_env)
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


@app.function(image=image, gpu="A100:4", timeout=7200, max_containers=1,
              secrets=[modal.Secret.from_name("huggingface-secret")])
async def run_control(scenario: str = "all"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("control")
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

        # One engine, all 4 GPUs (TP=4), prefix caching on, NO disagg flags.
        cmd = (
            f"{sys.executable} -m vllm.entrypoints.openai.api_server "
            f"--model {_MODEL} --port {_PORT} --tensor-parallel-size 4 "
            f"--gpu-memory-utilization 0.85 "
            f"--max-model-len {_MAX_MODEL_LEN} --block-size {_BLOCK_SIZE} "
            f"--dtype auto --enable-prefix-caching"
        )
        log.info("Starting aggregated vLLM (TP=4): %s", cmd)
        procs.append(start_process(cmd, "agg", "/tmp/vllm-agg.log", os.environ.copy()))
        if not wait_for_endpoint(_PORT, "/v1/models", timeout=900):
            raise RuntimeError("aggregated vLLM failed to start")
        log.info("Aggregated engine ready")

        url = f"http://localhost:{_PORT}"
        to_run = SCENARIOS if scenario == "all" else {scenario: SCENARIOS[scenario]}
        for name, cfg in to_run.items():
            artifact_dir = f"/tmp/aiperf_{name}"
            proc = await asyncio.create_subprocess_exec(
                *cfg["cmd"](url, artifact_dir),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                log.error("AIPerf %s failed: %s", name, stderr.decode()[-500:])
                all_results[name] = {}
                continue
            rf = Path(artifact_dir) / "profile_export_aiperf.json"
            result = json.loads(rf.read_text()) if rf.exists() else {}
            all_results[name] = result
            log.info("  %-16s TTFT=%7.0fms tok/s=%7.0f lat=%7.0fms", name,
                     result.get("time_to_first_token", {}).get("avg", 0),
                     result.get("output_token_throughput", {}).get("avg", 0),
                     result.get("request_latency", {}).get("avg", 0))
    except BaseException:
        try:
            with open("/tmp/vllm-agg.log", errors="replace") as fh:
                lines = fh.read().splitlines()
            log.error("----- agg log (%d lines, last 40) -----", len(lines))
            for line in lines[-40:]:
                log.error("  %s", line[:220])
        except FileNotFoundError:
            log.error("no agg log")
        raise
    finally:
        for p in procs:
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()

    log.info("=" * 70)
    log.info("AGGREGATED CONTROL (TP=4): %s", scenario)
    for name, r in all_results.items():
        log.info("  %-16s TTFT=%7.0fms tok/s=%7.0f lat=%7.0fms",
                 name, r.get("time_to_first_token", {}).get("avg", 0),
                 r.get("output_token_throughput", {}).get("avg", 0),
                 r.get("request_latency", {}).get("avg", 0))
    return all_results


@app.local_entrypoint()
def main(scenario: str = "all"):
    print(f"aggregated control | TP=4 | {scenario}")
    print(run_control.remote(scenario))
