# DNABERT-2 promoter results index

The live study is the frozen 70-nt GUE `prom_core_all` assay. Claims live in
[`studies/dnabert_promoter/CLAIMS.md`](../../studies/dnabert_promoter/CLAIMS.md) as DBP1–DBP10;
corrections and archive decisions live in
[`CORRECTIONS.md`](../../studies/dnabert_promoter/CORRECTIONS.md).

Per-file evidence metadata is in [`results/MANIFEST.csv`](../MANIFEST.csv).

## Readout and data hazards

The canonical suite reads encoder Layer 2 with `selected_layer_2_probe.joblib` (sha256
`bd0d9e14…`). Files under `causal/layer3_*` use encoder Layer 3 and a different probe. They are
marked `superseded`, control no claim, and must not be combined with the canonical suite (DBPR7).

`shared/leakage_report.json` describes the raw GUE cohort. The cleaning stage removes conflicting and
cross-split exact duplicates, then asserts pairwise disjoint retained hashes. Canonical results use the
cleaned counts `47,149/5,856/5,865`; near-duplicates were not screened (DBP2).

## Active directories

| Directory | Status | Claims |
|---|---|---|
| `shared/`, `provenance/`, `frozen_claim_tests/registry/` | Instrument and data record | DBP1, DBP2 |
| `frozen/layer_probes/`, `frozen/baselines/` | Frozen representation and sequence baselines | DBP3, DBP4 |
| `behavioral/`, canonical `causal/` families | Candidate head and branch interventions | DBP5 |
| `frozen_claim_tests/circuit_accounting/` | Conditional necessity and delivery | DBP6 |
| `frozen_claim_tests/faithfulness_curve/` | Standalone faithfulness curve; no tested knee | DBP7 |
| `frozen_claim_tests/components/` | Matched/same-class/opposite-class activation transplant | DBP8 |
| `frozen_claim_tests/biological_features/` | TATA enrichment and controlled non-positive perturbation | DBP9, DBP10 |
| `causal/layer3_*` | Superseded different-readout replicate | — |

DBP7 and DBP8 are central bounds, not supporting positives: the five-node set is not standalone
faithful, and matched transfer does not exceed unrelated same-class transfer. DBP10 is a non-positive
biological test at n=71.

## Provenance

Current frozen runners use `gfmcd.dnabert.write_run_manifest`, and
`tests/test_dnabert_manifest.py` pins that forward contract. Most historical artifacts predate the
manifest integration; their blank manifest fields are retained rather than backfilled. The study also
has one seed and no frozen preregistration or calibration pair. These limitations travel with every
DBP result.

## Archived material

The notebook-only fine-tuned extension formerly labeled DBP11–DBP17, its figures, and its unmatched
frozen/fine-tuned comparison were removed from the active tree under DBPR12. The superseded top-level
`dnabert/` notebook tree was removed in the same cleanup. Both remain recoverable at Git tag
`archive/unsupported-exploratory-2026-08-22`.

The current paper uses only
[`figures/paper_figure_workflow_transfer.png`](../../figures/paper_figure_workflow_transfer.png) from this
study. The former standalone diagnostic suite was removed from the active figure tree and remains
recoverable from Git history; its controlling artifacts and statuses are unchanged here.
