# Block 3 — `dispatch` + `stream_decode` (S4–S9, stub router)

**Date:** 2026-09-27 · **Status:** done, verified · **Files touched:** 1 modified
(`src/frontend/frontend.py`: +150 lines)

## What was written

- `Route` dataclass (prefill/decode URLs, idxs, engine_id, side port) —
  **temporary home**, marked for move to `router/prefill_router.py` in Block 5.
- `UpstreamError` — non-200 worker answers; carries status + body snippet.
- `http_api_call` (now async): preprocess → `router.route()` → `dispatch`.
- `dispatch`: producer body (dual 1-token cap + inert transfer params),
  decode body (transfer metadata), parallel fire, 502 path, background
  `_finish_push` (await + PROF log; release/record TODOs for Block 5/6).
- Transport merge: single `_post_stream` primitive; `_drain` discards,
  `stream_decode` relays. One timeout/close policy for both sinks.
- Route handlers now `await request.app.state.frontend...` (see E3.2).

## Decisions (and why)

1. **Parallel dispatch kept (diagram deviation, commented).** Justification
   restated at the method: decode-side PUSH_REG unblocks the transfer.
2. **`Route` lives here temporarily, loudly marked.** Alternatives were
   blocking Block 3 on the router or inventing a second shape — both worse.
   The move is a recorded Block 5 task, not forgotten debt.
3. **502, no retry, cancel producer.** Matches the official no-aggregated-retry
   rule; holding prefill open with no decode consumer is pure waste.
4. **`_finish_push` never fails the client.** Awaits producer post-response;
   logs drain failures. Release/record hooks are explicit TODOs, not silence.

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E3.1 | `SyntaxError: 'await' outside async function` on first verify | `http_api_call` kept Pass-1 sync `def` while new body awaits | `async def`; handlers already async |
| E3.2 | POST → 500: stub router injected via `app.state.frontend` ignored | Handlers closed over factory-local `frontend`, so `app.state` swap (the reason for the factory!) had no effect | Handlers use `request.app.state.frontend` (+ missing `await` added). Block 1's testability claim now actually holds |
| E3.3 | Decode-fail test → 500, then 502 in isolation (flake hunt) | Test-stub bug, not code: stub asserted `session_id == 'conv-1'` but the fail-case POST omitted `conversation_id` → AssertionError inside `route()` → 500 | Relaxed stub for the fail case; verified 502 isolated AND combined. Lesson recorded: stubs must not over-assert beyond what the case under test needs |

## Verification (stub router + stub transport, no GPU)

```
stream status: 200 | body: data: hello
producer caps + decode metadata ok   # both 1-token fields; remote_engine_id/port/flags
decode-fail status: 502 {"error":"boom"}   # prefill task cancelled
```

## Interview note

"Three bugs, three different layers: a syntax leftover (E3.1), a design claim
the code didn't honor (E3.2 — the factory existed FOR injection and didn't
support it), and a test lying about the code (E3.3). The error table is the
point: each one names the layer so the next block doesn't repeat its class."
