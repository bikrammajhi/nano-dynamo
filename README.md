# nano-dynamo

> A minimal reimplementation of NVIDIA Dynamo's KV-aware routing core — built for learning, not production.

![Benchmark Comparison](docs/benchmark_comparison.png)

## What is this?

**nano-dynamo** (~2,300 lines of Python) implements NVIDIA Dynamo's key innovation: **KV-cache-aware routing for disaggregated prefill/decode inference**. It shows how a radix tree, cost-function scheduler, and block pool tracker route requests to the prefill worker with the most cached KV blocks — reducing time-to-first-token without changing the model.

Everything runs on [Modal](https://modal.com) with real vLLM backends and actual NIXL GPU-to-GPU KV transfer. No mocks.


## Architecture

```
                    ┌─────────────────────┐
                    │   Client (AIPerf)   │
                    └──────────┬──────────┘
                               │ HTTP
                    ┌──────────▼──────────┐
                    │   KV Router Gateway  │
                    │                      │
                    │  KvScheduler         │ ← cost function
                    │  + KvIndexer         │ ← radix tree
                    │  + BlockPool         │ ← prefix tracking
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼─────┐ ┌───────▼───────┐ ┌─────▼─────────┐
    │ Prefill vLLM   │ │ Prefill vLLM  │ │ Decode vLLM   │
    │ (GPU 0)        │ │ (GPU 1)       │ │ (GPU 2, 3)    │
    └────────────────┘ └───────────────┘ └───────────────┘
                               │ NIXL (UCX)
                        GPU-to-GPU KV transfer
```

## Benchmark Results

**Qwen/Qwen2.5-3B-Instruct on A10G**

> "2P + 2D" = 2 prefill workers + 2 decode workers = 4 GPUs. Prefill workers process prompts and generate KV cache. Decode workers receive KV cache via NIXL and generate output tokens.

| Scenario | System | TTFT (ms) | tok/s | Latency (ms) |
|----------|--------|-----------|-------|--------------|
| multi_turn | nano-dynamo | **158** | 341 | 2574 |
| | NVIDIA Dynamo | 199 | **358** | **2324** |
| mixed_workload | nano-dynamo | **120** | 849 | 3928 |
| | NVIDIA Dynamo | 228 | **982** | **3163** |

**Key insight:** nano-dynamo wins on TTFT (21-47% faster) because HTTP routing has less overhead than Dynamo's gRPC stack. Dynamo wins on throughput (5-16%) because gRPC workers communicate more efficiently.

## Quick Start

```bash
pip install modal
modal run benchmark_nano.py --scenario multi_turn
```

```bash
# 1P + 1D (2 GPUs)
modal run benchmark_nano.py --scenario multi_turn --num-prefill 1 --num-decode 1

# 2P + 2D (4 GPUs, default)
modal run benchmark_nano.py --scenario mixed_workload

# Compare with NVIDIA Dynamo
modal run benchmark_nvidia_dynamo.py --scenario multi_turn
```

| Scenario | What it tests |
|----------|---------------|
| `multi_turn` | Multi-turn conversations with shared system prompt (KV cache reuse) |
| `mixed_workload` | Variable sequence lengths simulating real chatbot traffic |

## Codebase

| File | LOC | Purpose |
|------|-----|---------|
| `src/gateway.py` | 319 | FastAPI KV router — core routing logic |
| `src/scheduler.py` | 203 | Dynamo's cost function + temperature sampling |
| `src/radix_tree.py` | 170 | Prefix indexing with content-based hashing |
| `src/block_pool.py` | 183 | Block state tracking (free/inflight/committed) |
| `src/types.py` | 178 | Core types (WorkerLoad, Request, etc.) |
| `src/engine.py` | ~400 | Worker inference engine |
| `src/vllm_sync.py` | 227 | vLLM metrics sync + NIXL orchestrator |
| `benchmark_nano.py` | 227 | Modal benchmark runner |
| `benchmark_nvidia_dynamo.py` | 219 | NVIDIA Dynamo comparison runner |
| **Total** | **~2,300** | |

## License

MIT
