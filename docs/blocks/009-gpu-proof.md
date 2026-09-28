# Block 9 — GPU proof: mini-dynamo serves real traffic on 2×A100

**Date:** 2026-09-27 · **Status:** 10/10 PASS · **Run:** Modal `mini-dynamo-proof`
(1 prefill GPU0 :8100 + 1 decode GPU1 :8200, Qwen3-14B-FP8, block 64, NIXL push)

## Result

```
health / models proxy ............ PASS (real HF tokenizer configs)
warmup 0/1/2 ..................... 200 (cold compile absorbed, untimed)
chat status ...................... 200
warmed TTFTs ..................... 61 / 50 / 48 ms  (< 1500 ms push regime)
turns 0/1/2 (affinity) ........... 200 / 200 / 200
kvbm/status ...................... decode_usage {8200: 0} (no load leak)
PROF lines ....................... n=9, all P0 D0
routing evidence ................. n=14 (KV_ROUTE + KVBM route lines)
```

PROF sample: `total_to_stream=5525.4 (cold) → 142.8 → 139.8 → 37.5 → 26.2 ms`.
The cold→warm curve is engine compile, not gateway overhead — steady-state
tens of ms at ISL≈11–14, concurrency 1. (Benchmark's 253 ms TTFT is ISL≈256
under concurrency 10; same regime, different load point.)

## Errors encountered (all found ONLY by running on GPUs)

| # | Error | Cause | Fix |
|---|---|---|---|
| E9.1 | First measured TTFT 6408 ms | Measured the virgin request: tokenizer download + vLLM compile, not serving latency | 3 untimed warmups before measurement (mirrors AIPerf `--warmup-request-count`) + report the cold curve, don't hide it |
| E9.2 | All traffic 502 → inhibited → 503 | `inject_transfer_metadata` omitted `tp_size`/`pp_size` — vLLM push connector can't plan transfer without producer topology | Fields added via `PrefillWorker.tp/pp_size` → `Route` → metadata (defaults 1). Cross-checked against old gateway, which sends both |
| E9.3 | PROF lines n=0 while serving 200s | `frontend.main()` never called `logging.basicConfig` → root stayed WARNING → every `log.info` (PUSH, PROF, routing) silently dropped | `basicConfig(INFO)` in the entry point (library import must NOT configure logging — documented at the call site) |

## Decisions (and why)

1. **1P+1D functional proof before any load run.** 2×A100 ≈ half the cost of
   the benchmark topology; failures here are code bugs (E9.2/E9.3 were),
   failures at load are capacity questions. Order matters for the bill.
2. **Failure log-dumps added to `modal_proof.py` first** (gateway + worker
   tails, proc exit codes on any failed check). E9.2 was diagnosed from the
   decode-side error text in the dumped mini.log, not from the status code.
3. **No benchmark claimed.** ~10 requests prove serving correctness
   (routing, caps, affinity, accounting, observability). Throughput/TTFT
   authority stays with `benchmark_nano.py` (Block 7: 253 ms) until a load
   harness points at mini-dynamo.

## Interview note

"The GPUs found three bugs my tests couldn't: a missing metadata pair that
only a real NIXL connector validates, a logging call that only matters when
stdout is a file, and a measurement that conflated compile with serving.
Each is now a regression check or a code comment — the proof paid for itself
in code, not just confidence."
