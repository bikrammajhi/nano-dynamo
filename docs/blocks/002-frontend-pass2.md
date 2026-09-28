# Block 2 — S2 `preprocess`: single tokenize + 400 validation

**Date:** 2026-09-27 · **Status:** done, verified · **Files touched:** 1 modified
(`src/frontend/frontend.py`: +55 lines)

## What was written

- `_get_tokenizer()` module singleton (thread-locked lazy load) + `Frontend._tokenizer()`
  accessor — ported verbatim from `gateway.py:39-50`; configs only, never weights.
- `tokenize_and_validate(body)`: messages → chat template, else raw prompt;
  **400** on missing/empty input and on zero-token output.
- `preprocess(body)`: the single S2 entry — one tokenization per request,
  feeding selection AND dispatch downstream.

## Decisions (and why)

1. **Validate before tokenizer load.** A malformed body 400s without triggering
   a HuggingFace config download. Old code inverted this cost (tokenizer work
   first, silent empty-tokens after).
2. **400 replaces silent empty-tokens.** Old `_tokenize` returned `[]` for `{}`,
   which then scored/routed a phantom request. Fail-loud at S2 keeps bad input
   out of the cost function entirely.
3. **Singleton kept, accessor added for tests.** `_tokenizer()` indirection lets
   tests stub the tokenizer without downloading weights — no `transformers`
   import at module load (lazy inside `_get_tokenizer`, as before).
4. **One call site enforced.** `preprocess` is the only caller of
   `tokenize_and_validate` (asserted in verification); the old double-call
   (`_pick` + `_push`) cannot recur without editing this file.

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E2.1 | None in code. Verification env reuse confirmed | `uv run --with` pattern from Block 1 worked unchanged | No action; pattern stable |

## Verification

```
400 paths ok (4/4)      # {}, messages:[], prompt:'', model-only
happy paths ok (chat + prompt)   # stub tokenizer, no download
single-tokenize wiring ok        # exactly one call site
GET /health -> 200               # Pass 1 intact
```

## Interview note

"S2 is where I fixed a silent-failure class, not just moved code: empty
bodies used to route phantom requests with zero tokens. Now they 400 before
the tokenizer even loads — validation ordering as a cost decision."
