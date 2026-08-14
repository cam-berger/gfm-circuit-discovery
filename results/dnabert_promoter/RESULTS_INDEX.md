# `results/dnabert_promoter/` — family index

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Artifacts of the DNABERT-2 promoter study, including the canonical frozen arc and the completed fine-tuned extension.

Claims live in
[`studies/dnabert_promoter/CLAIMS.md`](../../studies/dnabert_promoter/CLAIMS.md) as `DBP1`–`DBP17`;
withdrawals and bounded historical statements live in
[`CORRECTIONS.md`](../../studies/dnabert_promoter/CORRECTIONS.md) as `DBPR1`–`DBPR10`.

Per-file rows — `claim_id`, `status`, `evidence_level`, `polarity`, `script`, `manifest`, `figure` — are
in [`results/MANIFEST.csv`](../MANIFEST.csv); this page says which directories matter and why.

## Read this before quoting any number here

**Two readout endpoints are present in the frozen directory, and only one is canonical.** The canonical
suite reads encoder Layer 2 (`hidden_states[2]`) and scores with `selected_layer_2_probe.joblib`
(sha256 `bd0d9e14…`). Everything under `causal/layer3_*` reads encoder **Layer 3** and scores with a
*different* probe, `all_layer_probes/layer_3_probe.joblib` (sha256 `2337441e…`). The Layer-3 family is
`superseded`, controls no claim, and is retained only because results here are immutable (`DBPR7`).

The cue in a directory listing is the `layer3_` filename prefix; the cue inside a file is the
`selected_layer` field. Check one of them before comparing anything to a published number.

**`shared/leakage_report.json` is a diagnosis, not a defect left standing.** It reports the *raw* frozen
cohort: 259 exact-duplicate groups, 104 spanning splits, 64 label-conflicting. `clean_promoter_cohort.py`
then removes them — 326 rows — and asserts the retained splits are pairwise disjoint, and every canonical
frozen artifact in this tree is computed on that cleaned cohort (`DBP2`). Its report and cleaned hashes are
not committed, which is the residual gap; the counts are recoverable from
`frozen/layer_probes/all_layer_probe_results.json`.

The fine-tuned extension uses a different scoped behaviour: GUE `prom_300_all`, 300-nt sequences, and
the checkpoint classifier for the primary predictive evaluation. It therefore must not be treated as a
matched before/after fine-tuning experiment with the frozen 70-nt `prom_core_all` study (`DBP17`,
`DBPR10`).

## Directories

| Directory | Status | What it holds | Claims |
|---|---|---|---|
| `shared/` | canonical | Frozen-study raw split manifest and hashes, dataset summary, and the leakage scan that motivated the cleaning stage. | DBP2 |
| `provenance/` | infrastructure | `checkpoint_equivalence.md` — frozen model/revision identity record. | DBP1 |
| `frozen_claim_tests/registry/` | canonical | `frozen_config.json`, the pinned frozen assay configuration. | DBP1 |
| `frozen/layer_probes/` | mixed | Per-layer frozen probe fits, rankings, test statistics, and the canonical probe artifact. The 12 `all_layer_probes/layer_*_probe.joblib` files are `development`; `selected_layer_2_probe.joblib` is `canonical` and pinned by hash. | DBP3, DBP4 |
| `frozen/baselines/` | canonical | Majority, GC-composition, and k-mer 3–6 baselines. | DBP4 |
| `behavioral/` | canonical | All-layer corruption trajectories and the behavioural probe summary. | DBP5 |
| `causal/head_ablation/`, `grouped_head_ablation/`, `branch_ablation/`, `token_alignment/`, `activation_restoration/` | canonical | The **Layer-2** frozen intervention suite: per-head scan over layers 0–2, grouped and branch ablations, negative-class source controls. | DBP5 |
| `causal/layer3_*` (5 directories, 42 files) | **superseded** | The **Layer-3** frozen replicate against a different probe. Controls nothing. Do not quote. | — (`DBPR7`) |
| `frozen_claim_tests/circuit_accounting/` | canonical | Necessity and delivery fractions for the five-node frozen candidate set, validation and untouched test. | DBP6 |
| `frozen_claim_tests/faithfulness_curve/` | canonical | Standalone frozen faithfulness versus circuit size. **The compactness bound** — no knee. | DBP7 |
| `frozen_claim_tests/components/` | canonical | Frozen donor-matrix / activation-transplant statistics: matched / same-class / opposite-class / zero. | DBP8 |
| `frozen_claim_tests/biological_features/` | canonical | Frozen TATA-like enrichment, eligibility, and controlled perturbation cohort and result. Strata files are `development` — exploratory and uncorrected. | DBP9, DBP10 |
| `finetuned/shared/` | canonical | Fine-tuned checkpoint validation, dataset provenance, label mapping, and GUE `prom_300_all` dataset summary. | DBP11 |
| `finetuned/summary/` | canonical | Compact consolidated summaries of the completed fine-tuned experiment package. | DBP11–DBP16 |
| `finetuned/test/` | canonical | Full held-out fine-tuned test metrics, predictions, confusion matrices, and strict-unseen sensitivity analysis after removing exact cross-split sequence overlaps. | DBP12 |
| `finetuned/layer_localization/` | canonical | Fine-tuned layer-wise masked-mean linear-probe metrics, split metadata, and the validation-selected Layer-10 endpoint. | DBP13 |
| `finetuned/causal/cohorts/` | canonical | Locked deterministic fine-tuned causal cohorts, including the balanced 256-example test cohort used for ablation and restoration. | DBP14 |
| `finetuned/causal/targeted_ablation/` | canonical | Fine-tuned `L0H8`, `L0_MLP`, and combined early-component ablation results, including class-conditional effects. | DBP14 |
| `finetuned/causal/activation_restoration/` | canonical | Restoration of clean `L0H8` and `L0_MLP` activations in the joint-ablation background. | DBP14 |
| `finetuned/causal/component_donor_matrix/` | canonical | Nucleotide-aligned activation-source swapping and same-class versus opposite-class compatibility statistics. | DBP15 |
| `finetuned/tata/` | canonical | Fine-tuned TATA-like motif eligibility, baseline evaluation, matched perturbation controls, per-example results, and statistical summaries. | DBP16 |
| `finetuned/performance_comparison/` | bounded | Descriptive frozen/fine-tuned performance contrast. It does not identify a causal effect of fine-tuning because the two studies are not matched on task, sequence length, cohort policy, or readout. | DBP17 |

## Frozen polarity, so a null is never re-read as a positive

Three of the frozen family's load-bearing results are **not** positives, and they are the ones most likely
to be misquoted:

- `frozen_claim_tests/faithfulness_curve/*` — **bounded**. Refutes compactness at the sizes tested (DBP7).
- `frozen_claim_tests/components/donor_matrix/*statistics.json` — **bounded**. The matched-versus-same-class contrast is null; only the class contrast is large (DBP8).
- `frozen_claim_tests/biological_features/tata_perturbations/*` — **null**. No causal motif dependence at n=71 (DBP10).

## Fine-tuned evidence summary

The fine-tuned checkpoint reaches:

```text
Balanced accuracy: 0.935205
MCC:               0.870487
ROC-AUC:           0.984785
PR-AUC:            0.984351