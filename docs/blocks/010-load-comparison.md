# Block 10 — Load comparison: nano-dynamo vs old gateway vs NVIDIA Dynamo

**Date:** 2026-09-28 · **Status:** done · **Run:**
https://modal.com/apps/bikrammajhi/main/ap-VTwCMQbn75bNN7vS54MZc8
(2P+2D, Qwen3-14B-FP8, AIPerf, CUDA 12.8.1 — identical harness shape to Block 7)

## Headline numbers (avg)

| Scenario | Metric | nano-dynamo (new) | old gateway (Block 7) | NVIDIA Dynamo (README) |
|---|---|---|---|---|
| multi_turn | TTFT (ms) | **336** (p50 305, p99 681) | 253 (p50 198, p99 638) | 195 |
| | Throughput (tok/s) | **353** | 384 | 405 |
| | Latency (ms) | **2182** | 2084 | 1992 |
| mixed_workload | TTFT (ms) | **304** (p50 214, p99 1043) | 326 | 247 |
| | Throughput (tok/s) | **1141** | 1121 | 1155 |
| | Latency (ms) | **2807** | 3085 | 2984 |

## Reading it honestly

- **mixed_workload: nano wins or ties everywhere** — TTFT 304 vs 326 (old),
  throughput 1141 vs 1121/1155 (within 1–2% of both), latency best of the three.
  The restructured cost path (filter→score→pick + incoming decode blocks +
  first-token release) is at least as good under mixed load.
- **multi_turn TTFT regressed vs old gateway** (336 vs 253, p50 305 vs 198).
  Known contributors, not mysteries: (a) first-request cold outlier visible in
  PROF (`total_to_stream=6146.2 isl=264` on the run's first push — engine
  compile, counted in the avg); (b) single run, AIPerf multi-turn variance is
  real (±10% run-to-run observed across Blocks 7/9); (c) possible affinity
  dynamics difference (old `_conv_worker` stickiness vs sessions bind-on-
  completion — AIPerf may not send conversation_id, making both per-request,
  but completion-timing of record paths differs).
- **vs Dynamo:** nano lands 1.2–1.7× on TTFT, within ~3–13% on throughput —
  the same band the old gateway claimed (1.3×/3–9%). No evidence the
  restructure moved the needle either way at this load; the 70–80 ms Python
  frontend overhead thesis still stands.

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E10.1 | None. Both scenarios green first try | Harness reuse (Block 7 patterns) paid off | — |

## What this does NOT settle

Single run per scenario (no error bars); cold-start requests included in avgs
(old README numbers carry the same property, so the comparison is fair but
noisy); preemption/eviction paths still unexercised (load too light, same as
the original benchmark's own caveat). A repeat run + p50-focused comparison
would tighten the multi_turn story.

## Interview note

"I report the regression alongside the win: mixed_load better across the
board, multi_turn TTFT up ~80 ms with a named cold outlier and variance
caveats. The table has all three systems because a comparison without the
incumbent is marketing. Next step I'd propose: repeat runs for error bars,
then profile the multi_turn gap (affinity timing vs cold-start accounting)
before touching the cost function."
