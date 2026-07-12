# Research-rigor refactor — summary (2026-07-11)

Applied the research-rigor-refactor methodology (audit → additive hardening) to this
repo. The audit found **no outstanding correctness/contamination bug** (the one of that
class — a no-op negative control — was already caught and fixed; see `CLAIMS.md` R1).
Headline numbers recompute exactly from the checkpoints. What follows is each finding,
its fix, and the memorial test that keeps it from returning.

| # | Finding (severity) | Fix | Memorial test |
|---|--------------------|-----|---------------|
| 1 | No calibration audit for the null machinery (design) | `src/gfmcd/rigor.py` (vendored) + `scripts/splicebert/calibration_audit.py`; pure-noise → uniform max-p, 0 FDR discoveries | `test_rigor_calibration.py::test_max_null_uniform_on_noise`, `::test_planted_signal_is_detected` |
| 2 | No multiplicity control in the head screen (design) | Per-head add-one p + BH q-values vs a single-head null in `run_tracing.py`; `required_null_draws` reported; effective search space (96) recorded | `test_rigor_calibration.py::test_bh_and_required_draws`, `::test_empirical_p_never_zero` |
| 3 | `run_tracing.py` wrote no provenance manifest (design) | Added `ExperimentManifest` (params, versions, checkpoint + split hashes) | re-smoke writes `run_manifest.json` (checked) |
| 4 | `val_top1` and `val_pmr` reported as two metrics but identical (framing) | Documented as one statistic (up to rank-1 ties) in `aggregate_matrix.py`; kept PMR as the registered metric | `test_metric_identity.py` (code identity + artifact tolerance) |
| 5 | Stale `summary.csv` (INCOMPLETE row) (hygiene) | Reproducible `aggregate_matrix.py`; regenerated 25/25 | `test_metric_identity.py::test_identity_holds_in_summary_artifact` |
| 6 | No claims registry (hygiene) | `CLAIMS.md` with status + evidence-to-advance + permanent retraction R1 | — (doc) |
| 7 | Missing cheap rival — RF-CNN/composition (framing, low) | Left OPEN as claim **C7**; PWM+length remains the cheap rival | — (open) |

## What was preserved (named so the refactor didn't erode it)

Freeze-and-hash (`FREEZE.md`), staged evidence with hard STOP gates, the
randomized-backbone negative control that actually fires, gene-disjoint
discovery/confirmation split, gene-grouped bootstrap, add-one finite-sample p-values,
instrument-tested-before-use (G1/AtP), sealed test split, retraction discipline.

## The audit caught its own author

When first wired, the per-head FDR null (finding 2) used 200 draws. `calibration_audit.py`
reported `n_null_sufficient: false` — the finite-sample p-floor 1/201 is too coarse for a
true hit among 96 heads to clear BH at q=0.1 (needs ≥ `required_null_draws(96,0.1)=960`).
Corrected both defaults to 1000/999. This is exactly the failure mode the tooling exists
to prevent, and it fired on the first pass.

## Verification

`pytest` full suite green (**142 passed, 16 skipped, 0 failures**); `calibration_audit.py`
**PASS** (KS-uniform maxima at sizes 16/48/96, `n_null_sufficient: true`, 0 FDR
discoveries on noise). The tracing GPU run remains the one `pending-fresh-run` item
(claim C4); the calibration audit and instrument tests license belief in the numbers it
will produce.

Rerun both:
```
uv run python scripts/splicebert/aggregate_matrix.py       # regenerate summary.csv
uv run python scripts/splicebert/calibration_audit.py      # statistical exit criterion
uv run pytest -q                                           # full suite incl. memorial tests
```
