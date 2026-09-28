# Block 5 — Router layer: `kv_events` + `kv_indexer` + `slot_manager` + `prefill_router`

**Date:** 2026-09-27 · **Status:** done, verified · **Files touched:** 4 created,
1 modified (`frontend.py`: Route moved home, error mapping, inhibited passing),
1 test created (`tests/unit/test_naming.py`)

## What was written

- `infra/kv_events.py` — `token_ids_to_block_hashes` ported verbatim (21→35 lines
  with header); block-flow steps + determinism comments.
- `router/kv_indexer.py` — merged index (`record`/`forget`/`overlap`/
  `find_matches_for_request`); hashing delegated to kv_events (Publisher/
  Indexer split honored).
- `router/slot_manager.py` — `add_request`/`mark_prefill_completed`/`free`/
  `snapshot` + `WorkerLoad`; first-token release semantics (fix vs old
  drain-completion release).
- `router/prefill_router.py` (~300 lines) — `PrefillRouter` (`route`,
  `route_to_prefill`, `route_to_decode`, `inject_transfer_metadata`,
  `eligible`, `_any_compatible`), `KvRouter.score` (pure, term-for-term
  official formula incl. incoming decode blocks + decay), `Route` (moved home
  from frontend), `RoutingError(empty|overloaded)`, RR + random modes,
  reservoir ties + range-normalized softmin, narrowed per-request
  `overlap_credit` (direct-API parity).
- `frontend.py` hookup: imports `Route`/`RoutingError` (single home asserted);
  `http_api_call` passes inhibited URLs, maps overloaded→503 / empty→502.
- `tests/unit/test_naming.py` — 6 tests: surface assertions per file,
  no-cost-in-frontend, no-planner-module, Route single home.

## Decisions (and why)

1. **Scoring split into `route_to_prefill`/`route_to_decode` as real methods**
   (not inlined in `route()`). The map promised both badge names; the first
   draft merged them and the naming test caught it before any behavior test ran
   — the contract working as designed.
2. **Metadata construction moved to `inject_transfer_metadata`.** Frontend
   built `d_body` inline (leftover from the stub-router era); the S6 sub-label
   belongs to the router box. `dispatch` now calls it (kv_host passed as arg;
   router never touches HTTP).
3. **Inhibited-all ⇒ overloaded (503), never empty.** See E5.1.
4. **Decode cost includes incoming blocks** (open point #4, included):
   `potential_decode = usage + len(hashes)` inside `select_decode`.
5. **Sessions hook present but dormant** (`sessions=None` → skip). The branch
   exists so Block 6 plugs in without touching `route()`; documented, not dead
   (the skip path is the tested default).

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E5.1 | `route(inhibited=[all])` raised `empty` instead of `overloaded` | `_any_compatible` excluded inhibited URLs, so full quarantine read as "nothing exists" | Spec call: quarantine/busy = unusable *existing* capacity → 503; only a zero-length pool is `empty` → 502. Matches Filtering doc intent (compatible-but-unusable ⇒ overload) |
| E5.2 | Naming test draft asserted methods that didn't exist yet (`route_to_prefill` etc.) | Wrote test from map before refactoring code to match | Refactored `route()` to delegate (decision 1) instead of weakening the test — map wins over code |

## Verification

```
pytest tests/unit/test_naming.py ............ 6 passed
BEHAVIOR: 15 checks pass   # overlap-wins, zero-credit, softmin, decode
                           # least-loaded, filter off/on, inhibited, overload +
                           # empty reasons, slot lifecycle, record/forget
E2E: turn1 200, turn2 200, overloaded -> 503 {'detail': 'no worker available (overloaded)'}
```

## Known gap (owned by Block 6, not a surprise)

Turn 2 did NOT reuse P0 (went P1 on a tie) because nothing records served
blocks into the indexer yet — `_finish_push` still carries the TODO. Scoring
is proven by the manual-record check; cross-request locality activates when
Block 6 wires record-on-completion. Stated here so it reads as plan, not bug.

## Interview note

"The router layer is where the diagram pays off: S3 and S6 are separate
methods because the official decode-leg flags make them different scoring
regimes, not one function with an if. And the inhibited-all question (E5.1)
is a genuine design decision I can defend: quarantine is unusable capacity,
so it 503s like overload instead of 502ing like misconfiguration."
