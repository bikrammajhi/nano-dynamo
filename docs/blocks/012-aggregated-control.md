# Block 12 — Aggregated control: the regime verdict

**Date:** 2026-09-28 · **Status:** done · **Setup:** one vLLM engine, TP=4,
prefix caching on, same 4×A100 / model / AIPerf scenarios, no gateway.

## Numbers

| Scenario | Metric | Aggregated TP=4 | nano-dynamo 2P+2D | Old gateway | NVIDIA Dynamo |
|---|---|---|---|---|---|
| multi_turn | TTFT (ms) | **95** | 336 | 253–264 | 195 |
| | tok/s | **569** | 353 | 384 | 405 |
| | lat (ms) | **1043** | 2182 | 2084 | 1992 |
| mixed | TTFT (ms) | **89** | 304 | 326 | 247 |
| | tok/s | **1664** | 1141 | 1121 | 1155 |
| | lat (ms) | **1900** | 2807 | 3085 | 2984 |

## Verdict

The aggregated engine beats everything on every metric — including NVIDIA
Dynamo itself (2× TTFT, 1.4× throughput on multi_turn). In this regime
(14B dense, ≤1k ISL, concurrency ≤20, NVLink), **disaggregation does not earn
its complexity**: no transfer cost, no gateway hop, cross-phase batching wins
by 2–3.5× on TTFT and ~1.5× on throughput.

This reframes all prior blocks: 7/9/10 measured *frontend overhead between two
disaggregated systems*, a contest neither contestant wins. The audit's
suspicion (Block 11 §2) is confirmed empirically. Standing caveat upgraded to
a finding: every "within X of Dynamo" sentence to date describes the cost of
an architecture that loses to `vllm serve --tensor-parallel-size 4` here.

## What survives (and what changes)

- Survives: the code (routing math, S1–S9 structure, error taxonomy) is still
  faithful and still tested; the build quality claims are untouched.
- Changes: the *benchmark story*. Honest next questions are (a) at what model
  scale / ISL / concurrency does the control lose (70B? 8k+ ISL? 100+ conc?),
  and (b) ratio sweeps only make sense past that crossover. Spending order
  from Block 11 still holds; step 1 just answered "not here."
- Process note: first attempt's results were lost to `| head` truncation of
  the Modal output (ephemeral app logs retain build noise, not function
  output). Reran with output to file. Rule: GPU-run output always lands in a
  file first, greps second.

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E12.1 | First run's numbers lost (output truncated, app logs unrecoverable) | `\| head -30` on a 40-minute run's stdout; ephemeral apps don't retain function returns | Reran with `> file 2>&1` + grep after; documented as procedure |

## Interview note

"The most valuable run of the project is the one that beat us: a single
aggregated engine halved our best TTFT. I report it prominently because a
benchmark without its control is marketing — and now the interesting question
is where the crossover lives, which is a better thesis than any gap-closing
claim against Dynamo."
