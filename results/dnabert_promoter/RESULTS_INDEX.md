# `results/dnabert_promoter/` — family index

Artifacts of the DNABERT-2 promoter study, including the canonical frozen arc and the completed fine-tuned extension.

Claims live in
[`studies/dnabert_promoter/CLAIMS.md`](../../studies/dnabert_promoter/CLAIMS.md) as `DBP1`–`DBP17`;
withdrawals and bounded historical statements live in
[`CORRECTIONS.md`](../../studies/dnabert_promoter/CORRECTIONS.md) as `DBPR1`–`DBPR11`.

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
| `finetuned/performance_comparison/` | canonical | `frozen_hf_performance.csv` — descriptive frozen/fine-tuned performance contrast, carrying only artifact-backed rows at full precision with the GUE task and sequence length on each row. It does not identify a causal effect of fine-tuning because the two studies are not matched on task, sequence length, cohort policy, or readout. | DBP17 |

## Frozen polarity, so a null is never re-read as a positive

Three of the frozen family's load-bearing results are **not** positives, and they are the ones most likely
to be misquoted:

- `frozen_claim_tests/faithfulness_curve/*` — **bounded**. Refutes compactness at the sizes tested (DBP7).
- `frozen_claim_tests/components/donor_matrix/*statistics.json` — **bounded**. The matched-versus-same-class contrast is null; only the class contrast is large (DBP8).
- `frozen_claim_tests/biological_features/tata_perturbations/*` — **null**. No causal motif dependence at n=71 (DBP10).

## Fine-tuned evidence summary

The fine-tuned checkpoint reaches, on the official GUE `prom_300_all` test split (n=5,920):

```text
Balanced accuracy: 0.935205
MCC:               0.870487
ROC-AUC:           0.984785
PR-AUC:            0.984351
```

Removing exact test sequences that also occur in train or development leaves n=5,900 and balanced
accuracy `0.935136`, so the reported performance is not materially explained by exact overlap (`DBP12`).
Near-duplicate leakage is not screened by that analysis.

**The layer choice is a band, not a point.** `layer_localization/selected_endpoint.json` records the
layer the stated validation rule returned; `layer_localization/layer_selection_band.json` records what
that selection can bear. The argmax leads its runner-up by `0.000998` balanced accuracy, which is
`0.33` standard errors on this split: layers 7–11 are within one standard error of the argmax and
layers 5–11 within two, and the *test* argmax is Layer 8, not Layer 10. Quote the band (`DBP13`,
`DBPR11`), never Layer 10 alone, and never as a relocation relative to the frozen Layer-2 endpoint
(`DBP17`, `DBPR10`).

## Provenance: fixed forward, gap now extended forward once

**Only 5 of the ~109 frozen artifacts in this family have a sibling `*.manifest.json`**, and those five
are config echoes — they carry no `command`, `git_commit`, `created_at_utc`, seed, or input hashes. The
repository invariant is one manifest per result, carrying all of that.

That is fixed for every future *runner*: `src/gfmcd/dnabert/manifest.py` wraps the repository's own
`ExperimentManifest` and is wired into all 15 canonical frozen runners plus
`analyze_finetuned_layer_selection.py`, so a rerun writes hashed inputs/outputs, the probe sha256 that
defines the readout, the split provenance, the seed, the command, and the git state.
`tests/test_dnabert_manifest.py` pins that contract.

**The fine-tuned extension extended the gap forward rather than closing it.** Its 52 artifacts were
produced by a notebook rather than by runners, and carry neither sibling manifests nor a `script` entry;
their MANIFEST rows record that truthfully with blank `script` and `manifest` columns. This is a
disclosed, accepted debt, not an oversight — the assays are real and their inputs are pinned by
revision in `finetuned/shared/`, but they are not re-runnable from the tree until the notebook is
ported to `scripts/dnabert_promoter/`. `layer_selection_band.json` is the one fine-tuned artifact that
does have both, and is the shape the rest should take.

The **historical** frozen artifacts keep a blank `manifest` column, which `results/RESULTS_INDEX.md`
defines as the record "where one exists". Back-filling them would mean asserting provenance nobody
recorded — the run conditions were not captured at the time, and inventing them is worse than an honest
blank. They fill in when the affected assays are next re-run, which the roadmap gates on multi-seed
anyway. Note that new manifests use the repository-standard `<stem>.manifest.json` name; the five
hand-written ones use an older `<stem>_manifest.json` spelling and are immutable.

## Figures

Published figures are in [`figures/dnabert_promoter/`](../../figures/dnabert_promoter/) with captions beside
them, following the repository figure convention. **Figures 1–3 and 9–12 are canonical; 4–8 depict the
superseded Layer-3 replicate** (`DBPR7`). Figures 9–12 were added during the integration review because
`DBP6`, `DBP7` and `DBP8` — including the faithfulness curve that carries the study's central negative
result — had no published figure at all.

The six fine-tuned figures under `figures/dnabert_promoter/finetuned/` (12 files, PDF and PNG) came from the same notebook and
have no producing script and no caption files yet; they are indexed against `DBP12`–`DBP16` and share the
debt recorded above.

`results/**/figures/` is gitignored precisely so that published figures live in one place; the arc
originally force-committed 24 files under `results/dnabert_promoter/figures/frozen/`, and they were
relocated on 2026-08-11.