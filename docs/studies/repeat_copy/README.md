# Study: repeat_copy

Prefix-keyed copy / induction circuit in a from-scratch causal transformer (GFM-nano).

Narrative and findings currently live in `docs/PLAN_repeat_copy.md` and the archived branch dump
`.archive/branches-codex-repeat-induction/codex-repeat-induction/` (not yet migrated into this
per-study layout — see `docs/studies/INDEX.md`).

**Resolved (TO_FIX.md §4):** the weight-1 control + objective sweep are run
(`scripts/eval_repeat_copy_sweep.py`, `results/repeat-copy/objective_sweep.json`). The circuit is
**forced, not unbidden**: absent at copy weight 1 (0/3 seeds) and 2, forming only at weight ≥ 4. See
`RESULTS.md` in this dir and `configs/experiments/repeat_copy_objective_sweep.toml`.
