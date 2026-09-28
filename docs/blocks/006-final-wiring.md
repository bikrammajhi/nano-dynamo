# Block 6 — Sessions + workers/discovery + completion wiring (final code block)

**Date:** 2026-09-27 · **Status:** done, verified · **Files touched:** 4 created
(`router/sessions.py`, `workers/prefill_worker.py`, `workers/decode_worker.py`,
`infra/discovery.py`, `examples/disaggregated.py`), 3 modified (`frontend.py`,
`prefill_router.py`, `tests/unit/test_naming.py`)

## What was written

- `router/sessions.py` — `SessionAffinity` (`lookup` touches clock, `bind`
  commits with hashes, `invalidate`, `access`, `evict_lru` forgetting decode
  blocks + freeing usage). Hard-mode verbs match the Tuning doc (`bind`,
  `invalidate`); no idle-TTL (documented gap).
- `workers/prefill_worker.py` — `PrefillWorker` handle + `producer_body()`
  pure builder (dual 1-token cap moved OUT of `dispatch` into its owning box).
- `workers/decode_worker.py` — `DecodeWorker` handle + explicit ours/vLLM
  verb split (transfer/decode/generate are vLLM-side; relay is ours).
- `infra/discovery.py` — `Discovery` static tables (2P+2D benchmark layout),
  `register_worker` (admission-log no-op), `service_discovery`,
  `worker_discovery`, `build_router` (single construction point: urls/engines/
  ports consistent by construction).
- Completion timing contract: prefill side in `_finish_push`
  (`mark_prefill_completed` + record prefill at drain end == prefill+1 done);
  decode side in `tracked()` wrapper (`free` always; record + `bind` only on
  full consumption). `route()` only READS affinity now.
- `create_app(..., discovery=)` builds the full live stack; `/kvbm/status`
  serves real decode usage.

## Decisions (and why)

1. **Worker verbs live where they execute.** `compute_kv_cache`/`kv_transfer`/
   `decode`/`generate_tokens` run inside vLLM, so worker modules hold handles
   + the producer contract (pure, testable) instead of fake methods. The map
   deviation is recorded here rather than papered over.
2. **Bind-on-completion, not bind-on-route.** `route()` lost its `bind` call:
   committing placement before knowing the outcome pins sessions to failed
   routes. Official commit-on-dispatch, implemented as commit-on-completion
   (stronger: stream actually finished).
3. **Free-always, record/bind-on-completion.** Load accounting must not leak on
   disconnects; affinity must not pin partial streams. Two different
   correctness bars, two code paths in one `finally`.
4. **Membership constructed once, passed down.** `build_router` feeds tables
   into `PrefillRouter` args — no second copy, no drift between discovery and
   routing views.

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E6.1 | `from typing List` SyntaxError in first `discovery.py` draft | Typing slip written faster than checked | Rewrote file cleanly; enforced habit: import the new module immediately after writing (caught in seconds) |
| E6.2 | Naming test: `callable(chat_url)` fails | Properties are data, not actions | Assert existence (`isinstance(..., property)`) — and the distinction is now a documented convention: verbs callable, handles by existence |
| E6.3 | Turn-2 evict expectation wrong twice | (a) Assumed `bind` publishes blocks (it doesn't — record does); (b) forgot live sessions re-bind on every completion, staying freshest | Corrected lifecycle in test (record→bind→evict) and expectation (idle-first). Both runs taught the same lesson as E3.3: when the code disagrees twice, interrogate the test's model before the code |
| E6.4 | `route()` bound without hashes (pre-existing from Block 5) | Sessions didn't exist yet; placeholder call had wrong arity for the real API | Removed; binding lives in `_complete_request` where hashes + outcome exist |

## Verification

```
pytest tests/unit/ ............ 7 passed (naming incl. new surfaces)
E2E: turn1 200, turn2 reuses bound P0 D0, kvbm/status 200 with live usage
     prefill released, decode freed, affinity bound, invalidate ok
evict: idle-first, live survives, blocks forgotten, usage released
discovery tables == benchmark layout; producer_body caps asserted
```

## Interview note

"Block 6 is where timing became the design: prefill completion and decode
completion are different moments with different obligations (release vs
free/record/bind), and the code has two separate paths because collapsing
them leaks one side or mis-pins the other. If they ask about the trickiest
part of the build, it's this file's `finally` block."
