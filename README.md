# nano-dynamo

> What happens when you strip NVIDIA Dynamo down to its core idea and rebuild it from scratch?

LLM inference is slow. One reason: when multiple requests share a prefix (like a system prompt), the first request computes a KV cache, and every subsequent request recomputes it from scratch. NVIDIA Dynamo fixes this by routing requests to workers that already have the cache. This project demonstrates that **one specific idea** — prefix-aware worker selection via a radix tree and cost function — in ~1,500 lines of Python. It is **not** a Dynamo reimplementation; it is a toy proxy that illustrates the routing concept.

![Benchmark Comparison](docs/benchmark_comparison.png)

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
   Tell me a joke"       │   │  Cost function: score =                         │   │
                         │   │    (tokens - cache_hits) / block_size           │   │
                         │   │    + decode_load                                │   │
                         │   │                                                 │   │
                         │   │  Pick lowest cost. Done.                        │   │
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
                                    │  (not by nano-dynamo's code)     │
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

**Key insight: the router only picks PREFILL workers.** The decode worker and KV transfer are handled entirely by vLLM's internal NixlConnector (started via `--kv-transfer-config`). nano-dynamo's code is an HTTP proxy that selects the prefill worker — nothing more. There is no NIXL integration, no GPU-direct transfer logic, and no decode-worker selection in this codebase.

**The cost function (scheduler.py:105):**

```
credit = cache_hits_on_gpu × 1.0     # each cached block reduces work
cost   = (tokens - credit) / 16      # blocks of work remaining
cost   += decode_load                 # don't overload busy GPUs
```

Lowest cost wins. With `temperature=0.0`, this is greedy — no randomness.

**The disaggregation decision (gateway.py:95):**

After picking a prefill GPU, the router checks: is the remaining work too big? If `effective_prefill > 512 tokens` or the GPU's queue has `≥ 4` requests waiting, it offloads to the least-loaded remote GPU instead.

**What the router does NOT control:**

Which decode worker receives the KV cache. That's vLLM-internal NIXL plumbing. The router's job ends at picking the prefill worker — everything after that is vLLM's NIXL connector shuttling GPU memory directly.

## Results

**Qwen/Qwen2.5-3B-Instruct on A10G, 2 prefill + 2 decode workers**

| Scenario | System | Time to first token | Throughput | End-to-end |
|----------|--------|---------------------|------------|------------|
| multi_turn | nano-dynamo | **158 ms** | 341 tok/s | 2574 ms |
| | NVIDIA Dynamo | 199 ms | **358 tok/s** | **2324 ms** |
| mixed_workload | nano-dynamo | **120 ms** | 849 tok/s | 3928 ms |
| | NVIDIA Dynamo | 228 ms | **982 tok/s** | **3163 ms** |

**Caveat: this is not a fair comparison.** nano-dynamo is a bare HTTP/1.1 proxy with zero overhead beyond the request itself. NVIDIA Dynamo includes gRPC serialization, distributed state synchronization, and full routing stack overhead. The TTFT difference reflects protocol transport overhead (HTTP/1.1 vs gRPC), not routing algorithm quality. Dynamo's higher throughput (5-16%) shows the benefit of its actual optimizations. These numbers should not be interpreted as "nano-dynamo's routing is better."

## Try it

```bash
pip install modal
modal run benchmark_nano.py --scenario multi_turn
```

```bash
# 2 GPUs (1 prefill + 1 decode)
modal run benchmark_nano.py --scenario multi_turn --num-prefill 1 --num-decode 1

# 4 GPUs (default)
modal run benchmark_nano.py --scenario mixed_workload

# Compare with NVIDIA Dynamo
modal run benchmark_nvidia_dynamo.py --scenario multi_turn
```

| Scenario | What it tests |
|----------|---------------|
| `multi_turn` | Multi-turn conversations with shared system prompt (KV cache reuse) |
| `mixed_workload` | Variable sequence lengths simulating real chatbot traffic |

## Where to start reading

| File | Lines | What it does |
|------|-------|--------------|
| `src/gateway.py` | ~260 | Entry point — the FastAPI router that ties everything together |
| `src/scheduler.py` | ~185 | Cost function: given a request and a worker, what's the score? |
| `src/radix_tree.py` | 170 | Prefix index: which workers have seen this prompt prefix before? |
| `src/block_pool.py` | 183 | Block state: free, in-flight, or committed? |
| `src/prefill.py` | ~85 | Disaggregation decision: prefill locally or offload to another GPU? |
| `src/types.py` | 178 | Data structures |
| `src/vllm_sync.py` | 227 | vLLM Prometheus metrics scraper + HTTP prefill/decode orchestration |

Start with `gateway.py` — it's the router. Then `scheduler.py` to understand the cost model.

## License

MIT
