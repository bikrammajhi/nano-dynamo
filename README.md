# nano-dynamo

A KV-aware routing proxy for disaggregated LLM inference, written to be read: where NVIDIA Dynamo ships a compiled Rust router backed by CUDA kernels, this implements the same cost function in plain Python. The routing logic is the interesting part — [`src/gateway.py`](src/gateway.py), cost function at `gateway.py:92-240`.

> **Flex: the entire gateway is [1,298 lines of Python](src/) — 6 files, no C++, no CUDA kernels, no Rust. You can read all of it in an hour.**

## Benchmark: Nano-Dynamo vs NVIDIA Dynamo

> **TL;DR — this gateway lands within 1.35× of NVIDIA Dynamo's TTFT and within 3–9% on throughput and latency — same 4× A100s, same model, same AIPerf load.**

![Head-to-head benchmark — 2P+2D, Qwen3-14B-FP8](docs/benchmark_2p2d_qwen3_14b.png)

Regenerate with `python benchmark_charts.py` (data lives in `BENCHMARK_RESULTS.md`).

### Settings (identical on both systems)

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3-14B-FP8` |
| Hardware | 4× A100 (2 prefill + 2 decode) on Modal, CUDA 12.1 |
| Engine | vLLM 0.26 (`--enable-prefix-caching`) |
| KV transfer | NIXL, push mode (`NixlPushConnector`), side channel over localhost |
| KV config | `block_size=64`, `max_model_len=32768`, `gpu_memory_utilization=0.85` |
| Frontend | Nano: Python FastAPI gateway on :8787 · Dynamo: Rust frontend on :8000 |
| Load tool | AIPerf (OpenAI chat, streaming, `ignore_eos:true`) |
| Scenario `multi_turn` | 30 conversations × 5 turns, ISL≈256, OSL≈128, concurrency 10 |
| Scenario `mixed_workload` | 200 requests, mixed ISL/OSL (128/512/1024), concurrency 20 |

Full numbers and run links: [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md).

## Architecture

A single-process FastAPI gateway that sits in front of separately-deployed prefill and decode vLLM engines and implements Dynamo's KV-aware routing in Python.

```
 POST /v1/chat/completions
          │
          ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                    nano-dynamo gateway                      │
 │                                                             │
 │  Phase 1  DispatcherProxy._push()                           │
 │           tokenize → block hashes → kv_index overlap        │
 │                                                             │
 │  Phase 2  KvAwarePolicy.select()   (per prefill worker)     │
 │    cost = prefill_cost + decode_cost + active_req_weight    │
 │    argmin (t=0) / softmin (t>0), overlap credit             │
 │                                                             │
 │  Phase 3  MigrationManager.best_decode()  (per decode       │
 │    worker, load-based — KV is transferred, 0 credit)        │
 │                                                             │
 │  Phase 4  PreemptionManager  LRU evict + session promote    │
 │  Phase 5  ScalingManager     drain + auto-scale (advisory)  │
 └───────────────┬─────────────────────────┬──────────────────┘
                 │                         │
                 ▼                         ▼
        Prefill vLLM (P)             Decode vLLM (D)
        runs prefill, 1 token        full OSL decode
        (max_tokens=1)               (waits for blocks)
                 │                         ▲
                 └───── KV blocks, GPU-direct ─────┘
```

Every request fans out to both the prefill engine (`max_tokens=1`) and the decode engine (the full generation) in parallel. The prefill engine produces one token and pushes its KV blocks to the decode engine GPU-direct via NIXL; the decode engine runs the response once the blocks arrive; the gateway streams that response back to the client. See `_push()` at `src/gateway.py:477`.

### Phases

| Phase | What | Where |
|-------|------|-------|
| 1 | Disaggregated proxy — tokenization, `_pick()` routing, `_push()` fan-out | `gateway.py:333`, `gateway.py:477` |
| 2 | KV-aware prefill routing — Dynamo cost function with argmin/softmin | `gateway.py:92` (`KvAwarePolicy`) |
| 3 | Decode selection — load-based pick across decode workers | `kvbm.py:135` (`best_decode`) |
| 4 | Preemption & promotion — LRU eviction, promote resurrected sessions | `preemption.py:50` |
| 5 | Dynamic scaling — drain, rebalance recommendations (advisory stub) | `scaling.py:19` |

Key files, the cost function, what's implemented vs. missing, and an honest assessment live in [`docs/DESIGN.md`](docs/DESIGN.md).

## Cost old section removed

Matching what we reverse-engineered from Dynamo's `kv_router.py`:

```
raw_prefill_blocks     = (active_prefill_tokens + input_tokens) / block_size
overlap_credit_blocks  = overlap_score_credit × overlap_blocks
adjusted_prefill       = max(raw_prefill_blocks - overlap_credit_blocks, 0)
prefill_cost           = prefill_load_scale × adjusted_prefill
decode_cost            = potential_decode_blocks
request_cost           = decode_active_request_weight × active_requests
logit                  = prefill_cost + decode_cost + request_cost
```

Select argmin (temperature=0) or softmin (temperature>0). Reservoir sampling for tied minima.

## What we actually have

- Dynamo-matching cost function with configurable overlap credit, temperature, load scale, and request weight
- Per-request config overrides via `router_config_override` in the request body
- Overlap credit decay (reduces credit on overloaded workers proportional to excess queued blocks)
- Prefill active token tracking across workers (counts inflight tokens)
- Decode selection with overlap credit=0 (pure load, since KV is transferred)
- Reservoir sampling for tie-breaking (Dynamo-equivalent)
- Softmin routing at temperature>0 (range-normalized)
- KV block placement tracking (radix-tree-free hash-matching)
- Decode worker load monitoring + migration planning
- LRU preemption with session promotion
- Drain + auto-scale stubs
- Push-mode KV disaggregation via vLLM's native `NixlPushConnector` (gateway orchestrates, NIXL moves blocks GPU-direct)
- AIPerf benchmark harness (`benchmark_nano.py`) with per-phase PROF diagnostics and engine-metric breakdown
- 16 unit checks, no GPU required (`python -m temp.test_nano`)

## What we DON'T have (vs real Dynamo)

### KV event transport
Dynamo uses NATS Core / JetStream / ZMQ for real-time, distributed KV events between workers and router. nano-dynamo is single-process in-memory. Restart the gateway, lose all state.

### GPU-direct KV transfer (NIXL)
Dynamo's router integrates NIXL orchestration (NVLink/GPUDirect RDMA, zero CPU copies, multi-rail scale-out). nano-dynamo does **not** move KV itself — it delegates the transfer to vLLM's native `NixlPushConnector`, which runs on the same GPUs via UCX. On Modal (4 GPUs in one container) transfers ride intra-node NVLink; Dynamo's advantage is a production-grade, router-managed, scale-out transfer layer, not a per-request protocol difference.

### Flash Indexer
Dynamo's indexer does 170M ops/s in C++/CUDA. nano-dynamo's `KvIndex` is a hash-based Python prefix matcher.

### Multi-replica router
Dynamo syncs router state across replicas for HA. nano-dynamo is a single point of failure.

### Priority scheduling (`--router-queue-threshold`)
Dynamo supports queue-threshold-based prioritization. nano-dynamo is FCFS-only.

### AIC prefill duration model (`--router-prefill-load-model aic`)
Dynamo optionally decays only the oldest active prefill using an ML-predicted duration. nano-dynamo decays uniformly.

### Standalone indexer
Dynamo can run `dynamo.indexer` as an independent service. nano-dynamo's index is embedded in the gateway process.

### Tiered KVBM offload
Dynamo's KV Block Manager offloads to CPU → SSD → S3/Azure. nano-dynamo's "KVBM" is just block-location tracking + admission control.

### SLA planner
Dynamo has a formal SLA-based auto-scaler. nano-dynamo's ScalingManager is an advisory stub — it emits rebalance recommendations every 10 s (`src/scaling.py`) and an external orchestrator would have to act on them.

### Kubernetes-native deployment
Dynamo has CRDs, shadow-engine failover, topology-aware KV transfer. nano-dynamo is `uvicorn.run(...)`.

### Multimodal, agentic, LangChain
Dynamo v1.2+ supports multimodal encode/prefill/decode, embedding cache, per-request agent priorities, SGLang subagent KV isolation. nano-dynamo is text-only; multi-turn support is limited to `conversation_id`-based KV reuse across turns.

### `--router-track-prefill-tokens` toggle
Dynamo exposes this as a runtime flag. nano-dynamo hardcodes it at True.

## Honest assessment

nano-dynamo gets the **routing math** approximately right. The cost function, softmin, overlap credit, and argmin selection are faithful to what Dynamo's `kv_router.py` does.

Everything else — event transport, index performance, distributed consistency, router-managed scale-out KV transfer, production readiness — is absent or replaced with a toy version. The benchmark above shows the remaining per-request gap is a ~70–80 ms Python gateway overhead against Dynamo's Rust frontend.

If you want to understand the routing algorithm, read `gateway.py:92-240`. If you want to serve traffic, use real Dynamo.

## Running

```bash
# Start gateway with 1 prefill + 1 decode vLLM
python -m src.gateway \
  --prefill-ports 8100 \
  --decode-ports 8200 \
  --model Qwen/Qwen3-14B-FP8 \
  --overlap-score-credit 1.0 \
  --prefill-load-scale 1.0 \
  --router-temperature 0.0
```

### Quick validation (no GPU)

```bash
python -m temp.test_nano
```

### Modal smoke test

```bash
pip install modal
modal run temp/smoke_test.py
```

## References

Dynamo and disaggregated-inference background that informed this project.

### Talks & videos

| | Talk |
|---|---|
| [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=LDN3MIjl20g) | [Dynamo: Supporting Next-Generation AI Workloads — Olga Andreeva & Ryan McCormick, NVIDIA](https://www.youtube.com/watch?v=LDN3MIjl20g) (Linux Foundation, Open Source Summit) |
| [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=uc6TnOszzYA&t=2459s) | [Disaggregated LLM Inference: Past, Present and Future](https://www.youtube.com/watch?v=uc6TnOszzYA&t=2459s) (GPU MODE) |
| [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=OuQCSOIR-Y8) | [AI Perf Benchmarking — Dynamo and Other LLM Endpoints](https://www.youtube.com/watch?v=OuQCSOIR-Y8) (NVIDIA Developer) |
| [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=74MUe65_P1g) | [Inside NVIDIA Dynamo: Faster, Scalable AI Deployment](https://www.youtube.com/watch?v=74MUe65_P1g) (Ray Summit 2025, Anyscale) |
| [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=CLzz7ZalYD8&list=PL5B692fm6--tgryKu94h2Zb7jTFM3Go4X) | [Inference Office Hours with SGLang: Performance Optimizations for LLM Serving](https://www.youtube.com/watch?v=CLzz7ZalYD8&list=PL5B692fm6--tgryKu94h2Zb7jTFM3Go4X) (NVIDIA Developer) |
| [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=9tvJ_GYJA-o&t=14s) | [Mastering LLM Inference Optimization: From Theory to Cost-Effective Deployment — Mark Moyou](https://www.youtube.com/watch?v=9tvJ_GYJA-o&t=14s) (AI Engineer) |
| [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=pCef86jKZgM&t=882s) | [NVIDIA Dynamo: High-Performance Open Source Interface — William Arnold](https://www.youtube.com/watch?v=pCef86jKZgM&t=882s) (AER Labs) |
| [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=BUVOCqbmy3U) | [Benchmark Any LLM in 3 Steps — NVIDIA Dynamo + GenAI Perf Tutorial](https://www.youtube.com/watch?v=BUVOCqbmy3U) (Faradawn Yang) |

### Docs & articles

| | Doc / Article |
|---|---|
| [![NVIDIA](https://img.shields.io/badge/NVIDIA-%2376B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/dynamo) | [NVIDIA Dynamo — official developer page](https://developer.nvidia.com/dynamo) |
| [![NVIDIA](https://img.shields.io/badge/NVIDIA-%2376B900?style=flat-square&logo=nvidia&logoColor=white)](https://docs.nvidia.com/dynamo/dev/knowledge-base/overview) | [Dynamo Knowledge Base: Overview](https://docs.nvidia.com/dynamo/dev/knowledge-base/overview) (NVIDIA docs) |
| [![NVIDIA](https://img.shields.io/badge/NVIDIA-%2376B900?style=flat-square&logo=nvidia&logoColor=white)](https://docs.nvidia.com/dynamo/dev/digest/flash-indexer) | [Dynamo Digest: Flash Indexer](https://docs.nvidia.com/dynamo/dev/digest/flash-indexer) (NVIDIA docs) |
| [![Article](https://img.shields.io/badge/Article-%23D14836?style=flat-square&logo=readthedocs&logoColor=white)](https://blog.aifoundry.org/p/nvidia-dynamo-serving-llms-at-ai) | [NVIDIA Dynamo: Serving LLMs at AI Speed](https://blog.aifoundry.org/p/nvidia-dynamo-serving-llms-at-ai) (AI Foundry) |
| [![Article](https://img.shields.io/badge/Article-%23D14836?style=flat-square&logo=readthedocs&logoColor=white)](https://jarvislabs.ai/blog/vllm-optimization-techniques) | [vLLM Optimization Techniques](https://jarvislabs.ai/blog/vllm-optimization-techniques) (Jarvislabs) |
| [![Article](https://img.shields.io/badge/Article-%23D14836?style=flat-square&logo=readthedocs&logoColor=white)](https://jarvislabs.ai/blog/scaling-llm-inference-dp-pp-tp) | [Scaling LLM Inference: Data Parallelism, Pipeline Parallelism, Tensor Parallelism](https://jarvislabs.ai/blog/scaling-llm-inference-dp-pp-tp) (Jarvislabs) |

### Code & tools

| | Repo / Gist |
|---|---|
| [![GitHub](https://img.shields.io/badge/GitHub-%23181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ai-dynamo/dynamo) | [ai-dynamo/dynamo — NVIDIA Dynamo inference framework](https://github.com/ai-dynamo/dynamo) |
| [![GitHub](https://img.shields.io/badge/GitHub-%23181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ai-dynamo/aiperf) | [ai-dynamo/aiperf — LLM benchmarking tool](https://github.com/ai-dynamo/aiperf) |
| [![GitHub](https://img.shields.io/badge/GitHub-%23181717?style=flat-square&logo=github&logoColor=white)](https://gist.github.com/BenHamm/31c648f7d7331c94c1f3a45859db6677) | [AIPerf: Comprehensive LLM Benchmarking — BenHamm](https://gist.github.com/BenHamm/31c648f7d7331c94c1f3a45859db6677) (GitHub Gist) |

## License

MIT
