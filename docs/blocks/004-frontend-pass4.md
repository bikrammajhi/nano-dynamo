# Block 4 — Pass 4: inhibition, models proxy, PROF, prune

**Date:** 2026-09-27 · **Status:** done, verified · **Files touched:** 1 modified
(`src/frontend/frontend.py`: +60 lines)

## What was written

- `INHIBIT_SECS = 5.0` + `inhibit()` / `is_inhibited()`: timestamp quarantine
  table with lazy expiry-eviction on read (no growth over process lifetime).
  Wired into the dispatch 502 path (failed decode worker inhibited).
  Official counterpart cited: `DYN_RUNTIME_INHIBITED_DURATION_SECS`.
- `GET /v1/models`: proxied to `models_source_url` when wired, explicit 501
  when not (never connection-refused tracebacks for unwired state).
- PROF line extended with `mode=push P%d D%d` (route-aware diagnostics;
  release/record values join it in Block 5/6).
- Prune confirmed by assertion: 6 routes, no `/kvbm/preempt*`, no `/scale/*`.

## Decisions (and why)

1. **Inhibition without a health subsystem.** No probes, no registry watches —
   just failed-URL → timestamp, consulted by `eligible()` (Block 5). The
   cheapest resilience that is still real; documented as step one of the
   official resilience loop rather than a quarter-built loop.
2. **Expiry-eviction on read, not a sweeper task.** A background janitor for
   a dict is over-engineering; lazy delete keeps the table bounded with zero
   tasks. Verified: expired entry gone after one `is_inhibited` call.
3. **Discovery stays authoritative (commented).** Expiry restores; only the
   router/discovery layer removes. The table quarantines, never bans.
4. **501 for unwired, not try/except around httpx.** Unwired state is
   configuration fact, not a runtime surprise — say so in the status code.

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E4.1 | Models-proxy test → 501 despite wired URL | Same class as E3.2: `list_models`/`status` closed over factory-local `frontend`, ignoring `app.state` swaps (and `status` passed earlier only because defaults coincided — a passing test hiding the bug) | All handlers read `request.app.state.frontend`; verification now asserts the *injected* model (`'m'`), which would have caught E3.2-class rot at introduction |

## Verification

```
inhibition lifecycle ok (set/hit/expire/evict); INHIBIT_SECS = 5.0
models unwired -> 501 ok
status reads injected instance: {'model': 'm', ...}   # E4.1 regression guard
models proxied ok                                     # URL + payload asserted
prune check ok — no /kvbm/preempt, no /scale/*
```

## Interview note

"Two sibling bugs (E3.2, E4.1) from one habit: closures over construction
state instead of request state. The fix generalizes — any handler reading
config must go through `app.state`, and the status-endpoint assertion now
pins that property. I also got inhibition, the highest-value resilience per
line in the official architecture, in ~10 lines."
