# Study: repeat_copy

Prefix-keyed copy / induction circuit in a from-scratch causal transformer (GFM-nano).

Narrative and findings currently live in `docs/PLAN_repeat_copy.md` and the archived branch dump
`.archive/branches-codex-repeat-induction/codex-repeat-induction/` (not yet migrated into this
per-study layout — see `docs/studies/INDEX.md`).

**Open (TO_FIX.md §4):** the circuit is trained under an 8× copy-position-weighted objective
(`--copy-weight 8.0`), so "unbidden / no forcing objective" is unsupported until the **weight-1
control** and the objective sweep are run. See `configs/experiments/repeat_copy_objective_sweep.toml`.
