# Block 13 — Stress crossover probe: ISL 4k, concurrency 30

**Date:** 2026-09-28 · **Status:** done · **Setup:** same 4×A100/Qwen3-14B-FP8,
AIPerf ISL-mean 4096 / OSL 256 / conc 30 / 60 reqs, both architectures.

## Numbers

| System | TTFT avg (ms) | TTFT p50 / p90 (ms) | tok/s | Lat avg (ms) |
|---|---|---|---|---|
| nano-dynamo 2P+2D | **18,699** | 7,859 / 49,457 | **219** | 22,819 |
| Aggregated TP=4 | **2,310** | — | **815** | 9,297 |

## Verdict

Not just "doesn't win" — **collapses**. At p90 the disaggregated path takes
~50 s to first token (queueing collapse across the split pools + Python
gateway serialization at conc 30 + 64-block KV pushes per request), while the
aggregated engine degrades gracefully (2.3 s TTFT, 3.7× the throughput).
The crossover is nowhere near 14B/4k/30: the next probe must be model scale
(70B/MoE, where decode-bandwidth pressure actually exists), not more load.

## Spend decision (recommendation)

Stop GPU spend at this scale. Remaining budget (if any) goes to exactly one
run: 70B-class or MoE, moderate load, aggregated-vs-disaggregated — the only
unanswered empirical question. Everything else (ratio sweeps, ITL dashboards,
eviction-under-pressure) is engineering atop an architecture this regime
rejects. The teaching build (Blocks 0–10) is complete and stands on its own:
faithful routing math, diagram-traced code, honest numbers including the ones
that beat us.

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E13.1 | None. Both runs green first try; output-to-file procedure held | — | — |
