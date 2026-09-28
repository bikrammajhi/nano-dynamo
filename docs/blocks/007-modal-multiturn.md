# Block 7 — Modal `multi_turn` 2P+2D parity rerun

**Date:** 2026-09-27 · **Status:** done · **Run:**
https://modal.com/apps/bikrammajhi/main/ap-HuCBvxTklpvoNvNCRbIUse

## What ran (and what did NOT)

`modal run benchmark_nano.py --scenario multi_turn --num-prefill 2 --num-decode 2`
against the **old gateway** (`src/gateway.py`), not mini-dynamo. This run
establishes the current-harness baseline (CUDA 12.8.1 image, today's deps) that
any future mini-dynamo GPU run must match. Mini-dynamo has served mocked
traffic only — claiming parity for it would require pointing this same harness
at `mini-dynamo/` (open future work; needs a serving entry + sessions wiring
in `create_app`).

Harness hardening applied first (see plan): snapshot-download retry ×3 with
backoff, HF token removed from the logged command (env-only auth). The user's
earlier fail-fast + log-dump + 12.8.1 image changes were kept.

## Results vs README (`BENCHMARK_RESULTS.md` 2P+2D multi_turn)

| Metric | README (12.1, 08-01) | This run (12.8.1) | Delta |
|---|---|---|---|
| TTFT avg (ms) | 264 | **253** | −4% (noise-or-better) |
| TTFT p50 / p99 (ms) | 218 / 638 | **198 / 638** | same shape |
| Throughput (tok/s) | 371 | **384** | +3.5% |
| Request latency (ms) | 2083 | **2084** | identical |
| Engine prefill TTFT | ~196 | **155** | faster prefill side |
| Engine decode TTFT / E2E | ~273 / ~2580 | **245 / 2075** | consistent |
| KV xfer avg | 5–31 ms | **15–63 ms** (P90 up to ~200 under burst) | same regime |

Verdict: **parity confirmed.** All headline numbers within ±5% across a CUDA
minor-bump and two months of dependency drift. The routing math + push-mode
orchestration reproduce.

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E7.1 | None this run. Prior HF 504s (×2, same day) did not recur | Transient HF API flakiness; endpoint latency 6–20 s even when healthy | Retry loop (already in harness); no action needed |

## Interview note

"I re-ran the head-to-head baseline before claiming anything about the
restructure: 253 ms vs 264 ms TTFT, throughput +3.5%, latency identical.
And I'm explicit about what it does NOT prove — the old gateway served this
traffic, mini-dynamo hasn't yet. The honest scope boundary is part of the result."
