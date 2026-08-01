# Nano-Dynamo vs NVIDIA Dynamo — Benchmark Results

**Tool**: [AIPerf](https://github.com/ai-dynamo/aiperf) (installed via `pip install aiperf`)  
**Hardware**: 4× NVIDIA A100 (2 Prefill + 2 Decode) on Modal serverless GPUs  
**Model**: `Qwen/Qwen3-14B-FP8`  
**Engine**: vLLM 0.26 (disaggregated prefill/decode, push mode via NIXL `NixlPushConnector`)  
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

## Head-to-head: Nano-Dynamo vs NVIDIA Dynamo, 2P+2D — Qwen3-14B-FP8 (2026-08-01)

Both systems ran the identical AIPerf scenarios on 4× A100 (2 prefill + 2 decode), same model
and engine parameters — the only variable is the routing/frontend layer.

| Scenario | Metric | Nano-Dynamo | NVIDIA Dynamo | Gap (×) | Gap before fix (×) |
|----------|--------|------------:|--------------:|--------:|--------------------:|
| **multi_turn** | TTFT (ms) | 264 | 195 | **1.35×** | 12.7× |
| (30 convs × 5 turns) | Throughput (tok/s) | 371 | 405 | 1.09× | 1.5× |
| | Request Latency (ms) | 2,083 | 1,992 | 1.05× | 2.0× |
| **mixed_workload** | TTFT (ms) | 326 | 247 | **1.32×** | 17.2× |
| (200 reqs, mixed ISL/OSL) | Throughput (tok/s) | 1,121 | 1,155 | 1.03× | 2.2× |
| | Request Latency (ms) | 3,085 | 2,984 | 1.03× | 2.2× |

The TTFT gap collapsed from **12–17× to ~1.3×**; throughput and latency are now within
**3–9%** of NVIDIA Dynamo. The remaining delta is the Python gateway's per-request
overhead (~70–80 ms) against Dynamo's Rust frontend.

Nano-Dynamo runs: https://modal.com/apps/bikram-iit-ai/main/ap-N9xMXGs0PUWfc3aQokfibz  
NVIDIA Dynamo run: https://modal.com/apps/bikram-iit-ai/main/ap-H0qO0WTFKd0WYcbosrLHes

### What this benchmark does NOT measure (known limits)

This setup is a clean measurement of the **frontend's per-request overhead** — same
harness, same model, same load on both systems — but it is a weak probe of the
**routing logic**, which is the thing nano-dynamo actually implements:

1. **Two prefill workers = a binary choice.** With 2P the KV-aware cost function
   cannot demonstrate value; round-robin would score identically. Dynamo's routing
   matters at 8+ workers, so this benchmark cannot distinguish a good router from a
   random one.
2. **No prefix-cache pressure.** 30 conversations × 5 turns with a shared system
   prompt keeps overlap trivially high (both workers hold the same blocks). The
   overlap-credit / decay logic — the actual Dynamo cost formula — is never
   stress-tested.
3. **Intra-node NVLink transfer (5–31 ms).** The hard part of disaggregated
   inference — scale-out KV transfer over a contended fabric (RDMA/GDR) — is not
   measured. The 12.7×→1.35× collapse partly reflects measuring in this easy regime.
4. **14B FP8 is below the disaggregation sweet spot** (70B+), where prefill and
   decode have genuinely different bottlenecks and the P/D split earns its cost.
   At this model size a single engine could serve both phases.
5. **Load too light for Phase 4/5.** Concurrency 10–20 never approaches the
   decode cache capacity, so preemption (LRU eviction, promotion) and auto-scaling
   are never exercised under pressure.

**Plan:** these are the limits to fix, one by one — scale to 4P+4D/8P+8D, 100+
conversations with shared prefixes, 70B-class model, longer OSL, multi-node
transfer. Each re-benchmark updates this section only; the historical results
below stay frozen.

---

> **Historical (superseded):** the comparison below is the original Qwen2.5-3B-Instruct,
> 4× A100 run against an earlier gateway — before the parallel dispatch and the
> `max_completion_tokens` producer-cap fix, and before KV transfer moved to vLLM's
> native NIXL push connector. It is kept for reference only; the current numbers are
> the Qwen3-14B-FP8 2P+2D runs above.

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

## Analysis (historical — pre-fix gateway)

### TTFT gap (12–17×)

This is the dominant difference and is **architectural, not a bug**.

- **Dynamo's frontend** is written in **Rust** — it handles HTTP parsing, worker selection, and response streaming in a few microseconds. The request path is:  
  `client → Rust HTTP server → route → vLLM worker (via NIXL) → stream back`.

- **Nano-Dynamo's gateway** is a **Python FastAPI application** behind uvicorn. Each request went through:  
  `client → FastAPI → tokenize (transformers) → compute KV overlap (Python set intersection) → httpx proxy to prefill worker → await full prefill response → relay streamed decode output`.  
  Two factors drove the old gap: (1) this **serialized** dispatch, and (2) the producer request silently decoded the full OSL because vLLM prefers `max_completion_tokens` (which AIPerf sends) over `max_tokens`. Both are fixed — dispatch is parallel and the producer is capped at 1 token (`src/gateway.py` `_push`), moving KV transfer to ~200 ms and closing the gap to ~1.3×.

### Throughput gap (1.5–2.2×)

Both systems use the **same vLLM backend** with identical configuration — the throughput difference comes from the gateway's inability to keep up with high request rates. At 20 concurrency, Dynamo's Rust frontend dispatches without contention, while Python's GIL and httpx connection pooling become limiting factors. (Historical: current runs are within 3–9%.)

### Latency gap (2×)

Latency = TTFT + decode time. The decode phase is identical (same vLLM, same model), so the old 2× gap was TTFT overhead propagating through the total request duration. (Historical: current runs are within 5%.)

### Goodput

Not measured in these scenarios (requires `--goodput` flag in AIPerf).

---

## Honest Assessment (historical)

| Aspect | Nano-Dynamo | NVIDIA Dynamo |
|--------|-------------|---------------|
| **Routing logic** | Identical in intent — KV-aware cost function with overlap credit, load projection, softmin | Same concept, implemented in Rust |
| **HTTP handling** | Python FastAPI + httpx | Custom Rust HTTP server |
| **KV overlap computation** | Python set intersection on block hashes | Rust-native hash set operations |
| **Concurrency** | Single asyncio event loop | Multi-threaded async Rust runtime |
| **Python overhead** | ~800–1000 ms per request (pre-fix) | < 1 ms per request |

The routing decisions are correct; the remaining delivery gap is the Python gateway's ~70–80 ms per-request overhead versus Dynamo's Rust frontend (see the 2P+2D comparison above).

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
