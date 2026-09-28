# Block 0 — Package scaffolding

**Date:** 2026-09-27 · **Status:** done, verified · **Files touched:** 6 created, 0 modified

## What was written

Empty package tree plus the diagram source — no logic:

```
mini-dynamo/
├── docs/architecture.mmd              # diagram source (symbol authority)
└── src/frontend|router|workers|infra/ # one __init__.py per diagram layer
```

Each `__init__.py` is a single sentence naming its diagram layer, so the
first file seen in any folder states which box you're standing in.

## Decisions (and why)

1. **Flat `src/<layer>/` instead of `src/nano_dynamo/<layer>/`.**
   The extra package level spelled `nano_dynamo.router.prefill_router` —
   stutter with no diagram box to justify it. The diagram has no
   "nano_dynamo" node, so the code gets no such level. (Flattening was
   applied by hand after scaffolding; imports are `frontend.*`, `router.*`.)
2. **Folders = diagram layers, files = diagram boxes (1:1).**
   Navigation rule for interviews: layer first, then box. No folder holds
   more than 5 files; no file exists without a diagram element.
3. **`docs/architecture.mmd` committed as source, not image.**
   A picture can't be diffed; Mermaid source can. Official boxes/edges keep
   original indices and colors; nano additions are dashed `Z_*` nodes
   appended *after* official edges so `linkStyle` numbering stays stable.

## Errors encountered

| # | Error | Cause | Fix |
|---|---|---|---|
| E0.1 | `linkStyle` indices wrong on first draft (zoom edges numbered 30–35) | Mermaid assigns edge indices in declaration order; zoom edges were declared before infra edges, so they are 20–25, infra 26–35 | Renumbered; rule recorded: new edges always append after official ones |
| E0.2 | Scaffolding landed beside the repo (`../mini-dynamo/`) instead of inside it | Working directory assumed repo root one level too high; true root has `.git` + `src/gateway.py` | Moved subtree into `<repo>/mini-dynamo/`; verified old tree untouched via file listing |
| E0.3 | `ast.parse` used to "validate" the `.mmd` file → `SyntaxError` | Wrong tool for the format; `.mmd` is not Python | Dropped the check; validation for the diagram is visual render (mermaid.live) |

## Verification

- `import frontend, router, workers, infra` (with `mini-dynamo/src` on path) → ok.
- Old tree (`src/gateway.py` etc.) untouched; `git status` shows `mini-dynamo/` as the only untracked addition.

## Interview note

"The empty tree *is* the diagram. Four folders, four layers — before a
single line of logic, the project structure already tells the request story."
