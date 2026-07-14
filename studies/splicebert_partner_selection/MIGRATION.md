# Migration — RRR hardening pass (2026-07-11)

> ⛔ **SUPERSEDED / VOID.** This document belongs to the **retracted** v1/v2 partner-selection
> design (R2/R3) or to the pre-sealed-test protocol. It is kept for provenance only.
> **The document of record is `studies/splicebert_partner_selection/FINDINGS.md`;**
> the frozen protocol actually executed is `PREREGISTRATION_SEALED.md`.
> Retractions: `CLAIMS.md` §R1–R12.

Light, additive rigor pass. **No modules were moved, renamed, or deleted** — the repo
already followed the one-module-per-stage layout. Only additions + three in-place
edits, listed below.

## Voided numbers

- **The stale `summary.csv`** (built before the `additive-random-frozen` control was
  re-run) listed `additive-random-frozen_seed1` as `INCOMPLETE`. That artifact is
  void; regenerate with `scripts/splicebert/aggregate_matrix.py` (now 25/25 complete).
- **The pre-fix random-control NO-GO** is retracted — see `CLAIMS.md` row **R1**. Any
  summary still citing "control ≈ frozen" is void.

## Added

| Path | Purpose |
|------|---------|
| `src/gfmcd/rigor.py` | Vendored `rigor_kit` (real scipy) — max-null calibration, add-one p, BH-FDR, block permutation, calibration audit. |
| `scripts/splicebert/calibration_audit.py` | CLI: pure-noise → the real max-null/FDR path; asserts uniform maxima + ~0 discoveries (statistical exit criterion, claim SM1). |
| `scripts/splicebert/aggregate_matrix.py` | Reproducible regeneration of `summary.csv` (replaces the earlier one-off). |
| `tests/test_rigor_calibration.py` | Memorial tests: uniform on noise, planted signal detected, add-one floor, BH + null-sizing. |
| `tests/test_metric_identity.py` | Memorial test: `top1 ≡ PMR` up to rank-1 ties. |
| `studies/.../CLAIMS.md` | Claims registry with retraction row R1. |

## Changed in place

| Path | Change | Re-verified |
|------|--------|-------------|
| `scripts/splicebert/run_tracing.py` | Added per-head add-one p + BH q-values (FDR, finding 2) against a single-head null; added `ExperimentManifest` provenance (finding 3). | Re-smoked on MPS; `run_manifest.json` written. |
| `colab_results/summary.csv` | Regenerated from all 25 runs; added `status` column; `val_top1`+`val_pmr` documented as one statistic. | `test_metric_identity.py`. |

## Not done (tracked as open)

- **C7** — receptive-field-limited CNN / composition rival (finding 7): still open in
  `CLAIMS.md`. PWM+length remains the cheap rival.
