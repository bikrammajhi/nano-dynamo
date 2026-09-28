# Block 8 — Live serving: mini-dynamo runs end-to-end

**Date:** 2026-09-27 · **Status:** done, verified · **Files touched:** 2 modified
(`frontend.py`: sessions auto-wire + worker-port CLI + transport-error mapping
+ key normalization), 1 created (`tests/integration/test_serve_live.py`)

## What was written

- Serving entry: `main()` takes `--prefill-ports/--decode-ports/
  --prefill-side-ports/--prefill-kv-host/--block-size`, builds `Discovery`,
  `create_app(discovery=)` now also attaches `SessionAffinity` sharing the
  router's indexer/slots, sets models source to prefill-0, pins block size.
- `tests/integration/test_serve_live.py`: fake prefill/decode FastAPI workers
  + real frontend over real HTTP/SSE on test ports. Covers: streamed chat,
  producer caps + transfer metadata asserted on the wire, multi-turn affinity
  reuse, KV locality recorded, health/models/kvbm-status, 400 path,
  decode-dead 502 → inhibited → 503.
- `examples/disaggregated.py` path verified via the same wiring.

## Decisions (and why)

1. **Sessions auto-wire in `create_app(discovery=)`.** Manual attach (Block 6
   style) is a forgotten-step bug waiting to happen; the factory owns full
   stacks, tests own partial ones.
2. **Transport failures map to `UpstreamError`.** Only non-200s mapped before;
   dead workers raised raw `ConnectError` → 500. All `httpx.HTTPError`/`OSError`
   now funnel into the 502/inhibition path (see E8.1).
3. **Single canonical inhibition key.** See E8.2.

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E8.1 | Dead decode → 500 (`httpx.ConnectError` traceback in server log) | `_post_stream` mapped HTTP statuses but not transport failures | `try/except (httpx.HTTPError, OSError)` around the eager POST → `UpstreamError`. Found only by running live — no unit test covers refused connections |
| E8.2 | Second request 502, not 503 (inhibition ignored) | Key mismatch: `dispatch` inhibits `"http://h:p"`, `eligible()` checks bare `"h:p"` — writer/reader disagreed silently | `_inhibit_key()` normalizes both paths. Class: same family as E3.2/E4.1 (two code paths sharing state with different key shapes) |
| E8.3 | None in test. Initial 500s were E8.1/E8.2 server-side, correctly surfaced | — | — |

## Verification

```
tests/ ......... 9 passed (7 unit + 2 live incl. failure→inhibit→503 chain)
```

Live trace highlights: producer wire shows both 1-token caps; decode wire
shows `remote_engine_id=prefill-0, remote_port=5600`; turn 2 reuses bound
P0/D0; dead-decode run goes 502 → inhibited → 503 with no blind retry.

## Interview note

"Running it found two bugs no amount of reading would: dead workers 500'd
because only HTTP statuses were mapped, and inhibition silently never matched
because of a scheme prefix. Both are the same lesson as the rest of the log —
state shared between two paths needs one canonical key — now with live
evidence. mini-dynamo serves: 9/9 with real HTTP and real SSE."
