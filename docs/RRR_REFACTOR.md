# Research-rigor refactor — reading-frame + repeat-copy (2026-07-11)

Audit → light additive hardening. The audit found **no contamination/correctness bug** — the tracer is
gauge-correct, nulls are proper, the test split is firewalled, and the retraction ledger is real.
Findings were validity/design/framing gaps. Each fix below has a memorial test where it is code.

| # | Finding (severity) | Fix | Memorial test |
|---|--------------------|-----|---------------|
| F8 | iff over-graded SETTLED given single-seed + lower-bound ceiling (framing) | Downgraded to HYPOTHESIS/open (C0) in RESULTS §0, REC, README; D36 appended | — (doc); tracked in `CLAIMS.md` |
| F1 | §7 "core result" printed, never saved (design) | `verdict_lambda.py` dumps `fig_data/verdict_lambda.json` | regenerates on next checkpointed run |
| F3 | 7-layer argmax with no selection null (design) | `peak_anchor_dependence_pvalue` + `_depth_max_null_p`: permutation max-over-depth null | `test_signatures_rigor.py::test_depth_null_flags_signal_and_clears_noise` |
| F4 | `signature()` fit+eval probe on overlapping data (design) | held-out 70/30 split; far accuracies out-of-sample | `test_signatures_rigor.py::test_insample_probe_inflates_vs_heldout` |
| F7 | stale test counts (~130 / 102) (hygiene) | corrected to 133 across README/RESULTS/REC | `pytest -q --co` |
| F2 | universal iff on n=1 seed (design) | flagged pending-fresh-run (needs runs) | `CLAIMS.md` F2 |
| F5 | "only if" untestable with lower-bound oracle (validity) | flagged pending-fresh-run (RF-CNN rival) | `CLAIMS.md` F5 |
| F6 | cell C/D accuracy split unverified (design) | flagged open-verify | `CLAIMS.md` F6 |

## Preserved (named so the refactor didn't erode it)

Retraction ledger with cause-of-death; the exemplary self-caught T1 calibration error (AR(1)-surrogate
null); **gauge-correct QK-product attribution** with a live-but-unread-head falsification control;
Altschul-Erikson dinuc-shuffle + G-density-matched nulls; test-split firewalls that raise; the
repeat-copy track's RF-CNN + base-rate ceiling + prefix-scramble control.

## Note on scope

This pass was run against `master` (the track-2 tree). The audit's premise that the README already
demoted the iff was read off a *different* working copy; on `master` all four docs asserted it
consistently, so F8 here is an honest **overclaim correction**, not a contradiction reconciliation.

## Verification

`PYTHONPATH=src pytest tests/test_signatures_rigor.py` green (2 memorial tests). Full collected suite =
133. The two `pending-fresh-run` items (F2, F5) need compute and are their own experiments.
