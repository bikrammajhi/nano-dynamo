# kv-prefix-router

LLM inference is slow. One reason: when multiple requests share a prefix (like a system prompt), the first request computes a KV cache, and every subsequent request recomputes it from scratch. NVIDIA Dynamo fixes this by routing requests to workers that already have the cache. This project demonstrates that **one specific idea** — prefix-aware worker selection via a radix tree and cost function — in ~900 lines of Python.

## The problem

When you send a chat message to an LLM, the system prompt gets tokenized and processed into a KV cache before the model can start generating. If two users share the same system prompt, that work happens twice. In production, this wastes GPU time and increases latency — the time before you see the first token.

The standard approach is "disaggregated inference": split the work into two phases.

- **Prefill** — Process the prompt and build the KV cache
- **Decode** — Generate tokens using that cache

Dynamo's insight: route requests to the prefill worker that already has the most cached context. Less recompute, faster responses.

## How it works

```
                         ┌─────────────────────────────────────────────────────────┐
                         │                   KV ROUTER GATEWAY                    │
   CLIENT REQUEST        │                                                         │
   ───────────────       │   ┌─────────────────────────────────────────────────┐   │
                         │   │  STEP 1: PICK PREFILL WORKER                    │   │
  "You are a helpful  ──►│   │                                                 │   │
   assistant...          │   │  Radix tree: which GPU has this prefix cached?  │   │
   Tell me a joke"       │   │  Cost function (Dynamo-compatible):             │   │
                         │   │    logit = 2 × overlap_ratio                    │   │
                         │   │          - gpu_cache_usage_pct                  │   │
                         │   │          - normalized_queue_depth               │   │
                         │   │                                                 │   │
                         │   │  Pick max logit. Done.                          │   │
                         │   └─────────────────────────────────────────────────┘   │
                         │                          │                              │
                         │                          ▼                              │
                         │   ┌─────────────────────────────────────────────────┐   │
                         │   │  STEP 2: LOCAL OR REMOTE PREFILL?               │   │
                         │   │                                                 │   │
                         │   │  effective_prefill = tokens - cached_prefix     │   │
                         │   │                                                 │   │
                         │   │  If effective_prefill > 512 tokens              │   │
                         │   │     OR queue_depth >= 4:                        │   │
                         │   │       → offload to least-loaded remote GPU      │   │
                         │   │                                                 │   │
                         │   │  Otherwise: do it locally                       │   │
                         │   └─────────────────────────────────────────────────┘   │
                         │                          │                              │
                         └──────────────────────────┼──────────────────────────────┘
                                                    │
                                                    │  Gateway proxies HTTP POST
                                                    │  to chosen prefill worker
                                                    ▼
                                    ┌──────────────────────────────────┐
                                    │  PREFILL vLLM (GPU 0)            │
                                    │                                  │
                                    │  Processes prompt, builds KV     │
                                    │  cache for the new tokens only   │
                                    │                                  │
                                    │  KV transfer → decode is         │
                                    │  handled by vLLM internally      │
                                    │  (not by kv-prefix-router's code)     │
                                    └──────────────┬───────────────────┘
                                                   │
                                                   │  vLLM internal
                                                   │  GPU→GPU transfer
                                                   │  (NixlConnector / NCCL)
                                                   ▼
                                    ┌──────────────────────────────────┐
                                    │  DECODE vLLM (GPU 1)             │
                                    │                                  │
                                    │  Receives KV cache               │
                                    │  Generates tokens                │
                                    │  Streams response to client      │
                                    └──────────────────────────────────┘
```

**Key insight: the router only picks PREFILL workers.** The decode worker and KV transfer are handled entirely by vLLM's internal NixlConnector (started via `--kv-transfer-config`). kv-prefix-router's code is an HTTP proxy that selects the prefill worker — nothing more.

**The cost function (scheduler.py:90):**

```
score        = matching_blocks / total_request_blocks     # prefix overlap ratio
cache_usage  = kv_cache_used / kv_cache_total             # GPU memory pressure
waiting      = active_requests / max_active_across_workers # queue depth

logit = overlap_weight × score - cache_usage - waiting
```

Where `overlap_weight` defaults to 2.0, matching Dynamo's `kv_router.py`. Selection is argmax with random tie-breaking — same as Dynamo.

**The disaggregation decision (prefill.py:38):**

After picking a prefill GPU, the router checks: is the remaining work too big? If `effective_prefill > 512 tokens` or the GPU's queue has `≥ 4` requests waiting, it offloads to the least-loaded remote GPU instead.

## Features

- **KV-aware routing** — cost function matches Dynamo's `kv_router.py` formula
- **Prefix index** — radix tree mapping token block hashes to worker IDs
- **Block state tracking** — per-worker pool (free → inflight → committed with dedup)
- **Disaggregation** — local vs remote prefill decision with configurable thresholds
- **OpenAI-compatible API** — `POST /v1/chat/completions` (streaming + non-streaming)
- **vLLM metrics scraping** — pull-based Prometheus `/metrics` every 2s
- **No GPU code** — routing is pure Python; KV transfer delegated to vLLM's NixlConnector

## Comparison with NVIDIA Dynamo

| Dimension | kv-prefix-router | NVIDIA Dynamo |
|---|---|---|
| **Codebase** | ~900 lines of Python, 7 files | ~100,000+ lines, Python + C++/CUDA |
| **Design goal** | Educational — illustrate the routing concept | Production — deploy at scale |
| **Protocol** | HTTP/1.1 + JSON | gRPC + protobuf (binary, ~10x smaller) |
| **Cost function** | Same: `2×overlap - cache_usage - waiting` | Same formula in `kv_router.py` |
| **Prefix index** | Pure Python radix tree | Native C++ `KvIndexer` (`dynamo._core`) |
| **KV transfer** | Delegated to vLLM's NixlConnector | `DynamoNixlConnector` with Triton kernel |
| **Metrics** | Pull (Prometheus scrape, up to 2s stale) | Push (ZMQ pub/sub, near real-time) |
| **Service discovery** | Static ports | Dynamic (etcd or file-based) |
| **Routing modes** | KV-aware only | `random`, `round-robin`, `kv` |
| **Worker scope** | Prefill only | Prefill + decode |
| **Request queue** | In-memory deque (max 1000) | NATS JetStream (distributed, persistent) |
| **Fault tolerance** | None | Leader election, failover |
| **Autoscaling** | None | K8s CRD with min/max replicas |
| **Deployment** | `python -m src.gateway` | `dynamo serve`, `dynamo deploy` to K8s |
| **Auth/TLS** | None | Full production stack |

## Project structure

```
src/
  gateway.py     276  FastAPI HTTP proxy — entry point
  scheduler.py   124  Cost function and worker selection
  radix_tree.py  144  Prefix index (radix tree of token block hashes)
  block_pool.py  154  Per-worker KV block state (free/inflight/committed)
  prefill.py      97  Disaggregation decision (local vs remote)
  types.py        67  Domain primitives (WorkerId, WorkerLoad, etc.)
  __init__.py      5  Public API exports
```

Start with `gateway.py` — it's the router. Then `scheduler.py` to understand the cost model.

## Running

```bash
# Start gateway with 1 prefill + 1 decode worker
python -m src.gateway \
  --prefill-ports 8100 \
  --decode-ports 8200 \
  --model Qwen/Qwen2.5-3B-Instruct
```

Requires vLLM instances running on the specified ports with `--kv-transfer-config` for GPU-to-GPU KV transfer.

### Smoke test (Modal, 2× A10G)

```bash
pip install modal
modal run temp/smoke_test.py
```

Launches 1 prefill + 1 decode vLLM on 2 GPUs, starts the gateway, and runs:
1. Basic chat completion (non-streaming)
2. Streaming chat completion
3. Multi-turn conversation (5 turns, KV cache reuse)
4. Parallel requests (5 concurrent)

### Full benchmark

```bash
modal run benchmark_nano.py --scenario multi_turn
modal run benchmark_nano.py --scenario mixed_workload

# Compare with NVIDIA Dynamo
modal run benchmark_nvidia_dynamo.py --scenario multi_turn
```

## Quick validation

```bash
python temp/test_nano.py
```

Runs 42 unit tests covering block pool, radix tree, scheduler, request queue, and disaggregation router — no GPU required.

## License

MIT
