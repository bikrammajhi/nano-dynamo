# Block 1 — `frontend/frontend.py` Pass 1: skeleton + S1 routes + CLI

**Date:** 2026-09-27 · **Status:** done, verified · **Files touched:** 1 created
(`src/frontend/frontend.py`, ~140 lines incl. comments)

## What was written

- Module header = diagram pointer + feature matrix (HAVE vs OUT-OF-SCOPE,
  one line of reason each) + launch line mirroring `python -m dynamo.frontend`.
- `class Frontend`: constructor validates `--router-mode` (fail-fast
  `NotImplementedError` naming the mode for the 4 unimplemented official
  modes); `http_api_call()` / `http_response()` as 501 TODOs for Pass 3.
- `create_app(model, router_mode)` factory (not a module-global app) with
  routes: `POST /v1/chat/completions`, `POST /v1/completions`,
  `GET /v1/models`, `GET /health`, `GET /status`, `GET /kvbm/status`.
  Inference/debug bodies are explicit `NotImplementedError` TODOs.
- `main()`: argparse `--model/--http-port/--host/--router-mode` + `uvicorn.run`.

## Decisions (and why)

1. **Factory over module-global app.** Tests build isolated instances
   (`create_app()` per test); a global would leak router state between tests.
   Official `main.py` wires per-process the same way.
2. **501 TODOs, not silent stubs.** An unimplemented route must fail loudly
   at call time. Verified: POST chat → 500 today (TestClient surfaces the
   NotImplementedError), 200s only where implemented.
3. **`--http-port` (official name), default 8787 (nano value).**
   Name parity with `dynamo.frontend --http-port 8000`; value avoids vLLM
   port clash. Both facts in the header comment.
4. **`/kvbm/status` kept, Prometheus rejected.** One JSON endpoint, zero new
   dependencies — recorded in the header so nobody "helpfully" adds
   `prometheus_client` later.
5. **`Frontend` holds no router/worker references yet.** Constructor takes
   only `model` + `router_mode`; wiring lands in Pass 3/4. Keeps this block
   independently verifiable.

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E1.1 | `.venv` from earlier session gone (workspace re-root moved the tree) | Environment paths shifted mid-project; `python3` has no fastapi | Verify via `uv run --with fastapi --with httpx --no-project` (uv cache only, zero repo/env mutation). Recorded here so future blocks reuse it |
| E1.2 | `TestClient` import prints a deprecation warning line | Upstream starlette re-export shim | Harmless; filtered in verification output, not in code |

## Verification

```
unimplemented mode -> NotImplementedError ok
routes: [.../health, /kvbm/status, /status, /v1/chat/completions, /v1/completions, /v1/models]
GET /health -> 200 {'status': 'ok'}
GET /status -> 200 {'model': 'Qwen/Qwen3-14B-FP8', 'router_mode': 'kv'}
POST /v1/chat/completions -> 500   (expected: Pass 3 TODO)
GET /v1/models -> 500              (expected: Pass 3 TODO)
GET /kvbm/status -> 500            (expected: Pass 4 TODO)
```

## Interview note

"Ask me for any route and I'll tell you its badge: chat/completions is S1,
health is infra, kvbm/status is the dotted metrics arrow. Unimplemented
official modes don't 404 or guess — they name themselves in the error."
