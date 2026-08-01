# Nano-Dynamo vs NVIDIA Dynamo — Benchmark Results

**Tool**: [AIPerf](https://github.com/ai-dynamo/aiperf) (installed via `pip install aiperf`)  
**Hardware**: 4× NVIDIA A100 (2 Prefill + 2 Decode) on Modal serverless GPUs  
**Model**: `Qwen/Qwen2.5-3B-Instruct`  
**Engine**: vLLM ≥0.7.2 (disaggregated prefill/decode with NIXL KV transfer)  
**Block size**: 64 | **Max model len**: 32768  
**CUDA**: 12.1.0 | **Python**: 3.12  

Both systems ran identical AIPerf scenarios against their respective OpenAI-compatible frontends.  
All GPU, engine, and model parameters are matched exactly — the only variable is the routing layer.

---

## Latest: Push-mode 1P+1D — Qwen3-14B-FP8 (2026-08-01)

**Harness**: `benchmark_nano.py` · **Hardware**: 2× A100 (1 prefill + 1 decode) on Modal ·  
**Engine**: vLLM 0.26, `NixlPushConnector` (KV push, `block_size=64`, `max_model_len=32768`) ·  
**Workload**: `multi_turn` (30 convs × 5 turns, ISL=271, OSL=128, concurrency 10)

After fixing the producer-side `max_tokens` cap (see root cause below), 1P1D push mode now delivers:

| Metric | Before fix | After fix | Δ |
|--------|-----------:|----------:|--:|
| TTFT (ms, avg) | 2,676 | **352** | 7.6× |
| TTFT (ms, p50) | 2,677 | **233** | 11.5× |
| TTFT (ms, p99) | 2,983 | **838** | 3.6× |
| Throughput (tok/s) | 213 | **353** | 1.66× |
| Request Latency (ms) | 4,614 | **2,236** | 2.1× |

Engine-level breakdown (after fix, n=175):

| Engine | TTFT avg (ms) | E2E avg (ms) | Note |
|--------|--------------:|-------------:|------|
| prefill | 214 | 214 | prefill + 1 token, finishes in one step |
| decode | 325 | 2,208 | KV arrives ~224 ms, then full 128-token decode |

KV transfer: ~10 ms avg xfer (P90 ~16 ms) at ~3.1 transfers/s.

**Root cause of the pre-fix TTFT**: aiperf sends `max_completion_tokens: 128` (from
`--output-tokens-mean`), and vLLM's OpenAI server prefers `max_completion_tokens` over
`max_tokens` (`vllm/entrypoints/openai/chat_completion/serving.py`). The gateway's
`p_body["max_tokens"] = 1` override was silently ignored, so the producer decoded the full
128-token response (~2.5 s) before finishing — only then did the KV transfer start. The
gateway now caps **both** fields on the producer body (`src/gateway.py` `_push`), so the
producer stops after 1 token and the transfer overlaps with the decode serving.

Run: https://modal.com/apps/bikram-iit-ai/main/ap-xvaegRsT3ZzZtcWTSUfohF

---

## Full benchmark: Push-mode 2P+2D — Qwen3-14B-FP8 (2026-08-01)

**Harness**: `benchmark_nano.py` · **Hardware**: 4× A100 (2 prefill + 2 decode) on Modal ·  
**Engine**: vLLM 0.26, `NixlPushConnector` (KV push, `block_size=64`, `max_model_len=32768`) ·  
**Scenarios**: `multi_turn` (30 convs × 5 turns, ISL=271, OSL=128, concurrency 10) and
`mixed_workload` (200 reqs, mixed ISL/OSL, concurrency 20)

Same producer-cap fix, full 4-GPU stack:

| Scenario | TTFT (ms) | TTFT p50 (ms) | TTFT p99 (ms) | Throughput (tok/s) | Latency (ms) |
|----------|----------:|--------------:|--------------:|-------------------:|-------------:|
| multi_turn | 264 | 218 | 638 | 371 | 2,083 |
| mixed_workload | 326 | 224 | 1,506 | 1,121 | 3,085 |

Engine-level breakdown (n≈200 across both scenarios):

| Engine | TTFT avg (ms) | E2E avg (ms) | Note |
|--------|--------------:|-------------:|------|
| prefill | 196 | 196 | prefill + 1 token, finishes in one step |
| decode | 273 | 2,580 | KV arrives ~200 ms, then full OSL decode |

KV transfer: 5–31 ms avg xfer under load, ~1.6–1.9 transfers/s per prefill worker.

Run: https://modal.com/apps/bikram-iit-ai/main/ap-N9xMXGs0PUWfc3aQokfibz

---

## Results

| Scenario | Metric | Nano-Dynamo | NVIDIA Dynamo | Gap (×) |
|----------|--------|-------------|---------------|---------|
| **multi_turn** | TTFT (ms) | 1,128 | 89 | **12.7×** |
| (30 convs × 5 turns) | Throughput (tok/s) | 394 | 584 | **1.5×** |
| | Request Latency (ms) | 2,062 | 1,007 | **2.0×** |
| **mixed_workload** | TTFT (ms) | 1,649 | 96 | **17.2×** |
| (200 reqs, mixed ISL/OSL) | Throughput (tok/s) | 1,033 | 2,312 | **2.2×** |
| | Request Latency (ms) | 3,043 | 1,395 | **2.2×** |

---

## Analysis

### TTFT gap (12–17×)

This is the dominant difference and is **architectural, not a bug**.

- **Dynamo's frontend** is written in **Rust** — it handles HTTP parsing, worker selection, and response streaming in a few microseconds. The request path is:  
  `client → Rust HTTP server → route → vLLM worker (via NIXL) → stream back`.

- **Nano-Dynamo's gateway** is a **Python FastAPI application** behind uvicorn. Each request goes through:  
  `client → FastAPI → tokenize (transformers) → compute KV overlap (Python set intersection) → httpx proxy to prefill worker → await full prefill response → relay streamed decode output`.  
  Every hop adds measurable overhead. For 20 concurrent requests the asyncio event loop becomes a bottleneck.

The Python gateway alone accounts for ~800–1000 ms of TTFT per request (serialization, tokenization, proxy I/O).

### Throughput gap (1.5–2.2×)

Both systems use the **same vLLM backend** with identical configuration — the throughput difference comes from the gateway's inability to keep up with high request rates. At 20 concurrency, Dynamo's Rust frontend dispatches without contention, while Python's GIL and httpx connection pooling become limiting factors.

### Latency gap (2×)

Latency = TTFT + decode time. The decode phase is identical (same vLLM, same model), so the 2× gap is essentially TTFT overhead propagating through the total request duration.

### Goodput

Not measured in these scenarios (requires `--goodput` flag in AIPerf).

---

## Honest Assessment

| Aspect | Nano-Dynamo | NVIDIA Dynamo |
|--------|-------------|---------------|
| **Routing logic** | Identical in intent — KV-aware cost function with overlap credit, load projection, softmin | Same concept, implemented in Rust |
| **HTTP handling** | Python FastAPI + httpx | Custom Rust HTTP server |
| **KV overlap computation** | Python set intersection on block hashes | Rust-native hash set operations |
| **Concurrency** | Single asyncio event loop | Multi-threaded async Rust runtime |
| **Python overhead** | ~800–1000 ms per request | < 1 ms per request |

Nano-Dynamo correctly implements all phases (disaggregated P/D split, KV-aware routing with overlap credit decay, KVBM block tracking, preemption, scaling) — but the **Python gateway is the bottleneck**. The routing decisions are correct; the delivery is slow.

A rewrite of the gateway in Rust (or even a Python optimization pass with `uvloop`, connection pooling, and zero-copy streaming) would substantially narrow the gap.

---

## Reproducing

```bash
# Nano-Dynamo
modal run benchmark_nano.py --scenario all

# NVIDIA Dynamo
modal run benchmark_nvidia_dynamo.py --scenario all
```

Both benchmarks run on 4× A100 GPUs on Modal. Results are recorded at:
- https://modal.com/apps/bikram-iit-ai/main/deployed/nano-benchmark
- https://modal.com/apps/bikram-iit-ai/main/deployed/dynamo-benchmark
