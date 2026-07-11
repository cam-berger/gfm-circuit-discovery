# Migration — RRR hardening pass (2026-07-11)

Light, additive rigor pass on the reading-frame + repeat-copy tracks. **No modules moved, renamed, or
deleted.** Three in-place code edits (each with a memorial test), doc-status corrections, and three new
docs. The retraction ledger and all existing controls are untouched.

## Voided / corrected claims

- **Stale test counts** — `README.md` said "~130 tests", `RESULTS.md`/`REC.md` said "102 tests passing";
  actual collected = **133**. Corrected everywhere.
- **The iff headline was over-graded SETTLED.** Downgraded to **HYPOTHESIS / open (C0)** in `RESULTS.md
  §0`, `REC.md`, and `README.md`, with the reasons (single-seed conditions F2, lower-bound pooled ceiling
  F5). The evidence and retraction ledger are unchanged; only the claim grade moved. Logged as `D36`.

## Added

| Path | Purpose |
|------|---------|
| `docs/CLAIMS.md` | Claims registry (C0–C7 + permanent retractions R-T1, R-D32/D35 + pending F2/F5/F6). |
| `docs/RRR_REFACTOR.md` | Finding → fix → memorial-test map; preserved good practices. |
| `tests/test_signatures_rigor.py` | Memorial tests for F3 (depth-selection null) and F4 (held-out probe). |

## Changed in place

| Path | Change (finding) |
|------|------------------|
| `scripts/verdict_lambda.py` | Persists the §7 core-result numbers to `results/reading-frame/fig_data/verdict_lambda.json` (F1). Was printed, never saved. |
| `src/gfmcd/signatures.py` | `signature()` now fits its probe on a 70% held-out split so far-field accuracies are out-of-sample (F4). Added `_layer_effects`, `_depth_max_null_p`, and `peak_anchor_dependence_pvalue` — a permutation max-over-depth null calibrating the 7-layer argmax (F3). `peak_anchor_dependence`'s 3-tuple return is unchanged (back-compatible). |
| `README.md`, `docs/RESULTS.md`, `docs/REC.md`, `docs/DECISIONS.md` | Test counts (F7); iff status downgrade (F8); D36 appended. |

## Not done (tracked as pending-fresh-run in CLAIMS.md)

- **F2** — ≥3-seed replication of the λ arms / 2×2 cells (needs training runs).
- **F5** — RF-limited-CNN pooled ceiling on the reading-frame arms (needs a run; repeat-copy has it).
- **F6** — verify the cell C/D accuracy split is held-out.

`verdict_lambda.py`'s new artifact writes on the next checkpointed run (checkpoints are gitignored, so it
could not be regenerated in this pass).
