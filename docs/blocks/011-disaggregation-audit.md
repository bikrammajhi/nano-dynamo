# Block 11 — Disaggregation audit: what the 2P+2D numbers don't prove

**Date:** 2026-09-28 · **Status:** analysis (no code change) · **Prompted by:**
2P+2D cannot do justice to disaggregated serving; DeepSeek-V3 shows why.

## 1. The ratio is arbitrary, not measured

2P+2D divides 4 GPUs evenly — that is the entire justification. DeepSeek-V3's
deployment is heavily decode-skewed (memory-bandwidth-bound decode batches
massively on MLA's compressed cache; compute-bound prefill saturates fewer
nodes), with P:D derived from per-phase sustainable throughput. We never
measured prefill saturation vs decode saturation, so 2P+2D may be starving
decode, idling prefill, or both. With 2 prefill workers routing is a binary
choice round-robin would tie — the cost function is never actually tested.

## 2. No aggregated baseline (the missing control)

Every comparison pits two disaggregated systems against each other. Neither
was run against one aggregated vLLM engine on the same hardware. At 14B dense,
aggregated + prefix caching may match or beat disaggregated outright (no
transfer, no gateway hop, cross-phase batching). Without the control, no claim
that disaggregation earns its complexity in this regime — only that our router
is close to theirs.

## 3. Wrong model class

DeepSeek-V3 (671B MoE, 37B active, MLA) has the two properties that CREATE the
P/D win: MoE decode streams experts from HBM every step (needs many decode
nodes), MLA's ~KB/token cache makes transfer nearly free. Qwen3-14B dense sits
in the dead zone: KV/token large enough to cost real milliseconds, model small
enough for one engine to serve both phases. We measure transfer overhead where
frontier architectures eliminated it, and skip the 70B+/MoE regime where the
split pays.

## 4. Workload too light on every axis that matters

| Axis | Tested | Where disaggregation shows up |
|---|---|---|
| ISL | 128–1024 | 4k–32k (compute-dominated prefills) |
| Concurrency | 10–20 | 100+ (decode batch size is THE efficiency lever; never reported) |
| Prefix pressure | 30 convs, one prompt | 100+ sessions, skewed popularity (hot + cold tail) |
| Fabric | intra-node NVLink | contended scale-out RDMA/GDR + topology |
| Skew/failure | none | straggler prefill, mid-run worker loss |

## 5. Wrong metrics for decode-side claims

Reported: TTFT/throughput/latency. Missing: ITL (decode smoothness — the metric
disaggregation most affects), goodput/SLO attainment, prefill queueing vs
compute split, transfer-as-fraction-of-TTFT. A system can win TTFT and lose ITL
(starved decode pool); current instrumentation cannot see it.

## Ordered experiment matrix (spend in this order)

1. **Aggregated control** — same GPUs/model, prefix caching on. Cheapest run,
   highest information. Decides whether the regime justifies disaggregation.
2. **Skew/straggler leg** — slow or kill one prefill mid-run. Our code owns an
   answer here (503 taxonomy, 5 s inhibition, eligible() filtering) that
   nothing has exercised.
3. **P:D ratio sweep at fixed GPUs** — 1P+3D vs 2P+2D (4 GPUs); 2P+6D vs 4P+4D
   if 8×A100 budget exists. Report per-phase saturation, not just TTFT.
4. **Long-context + high-concurrency leg** — ISL 4k–8k, concurrency 50–100,
   skewed prefixes. Report ITL, goodput, achieved decode batch size.
5. **70B/MoE leg** — only if 1–4 show the mechanism working.

## Standing caveat for all published numbers

Blocks 7/9/10 numbers are frontend-overhead measurements (Python vs Rust
per-request cost), not routing-quality measurements. Any sentence of the form
"mini-dynamo routes within X of Dynamo" should read "mini-dynamo SERVES within
X of Dynamo at 2P+2D, light load, NVLink" until the matrix above says otherwise.
