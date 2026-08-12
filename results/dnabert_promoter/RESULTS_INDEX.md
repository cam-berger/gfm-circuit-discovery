# `results/dnabert_promoter/` — family index

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Artifacts of the frozen DNABERT-2 promoter arc. Claims live in
[`studies/dnabert_promoter/CLAIMS.md`](../../studies/dnabert_promoter/CLAIMS.md) as `DBP1`–`DBP10`;
withdrawals live in [`CORRECTIONS.md`](../../studies/dnabert_promoter/CORRECTIONS.md) as `DBPR1`–`DBPR8`.
Per-file rows — `claim_id`, `status`, `evidence_level`, `polarity`, `script`, `manifest`, `figure` — are
in [`results/MANIFEST.csv`](../MANIFEST.csv); this page says which directories matter and why.

## Read this before quoting any number here

**Two readout endpoints are present in this directory, and only one is canonical.** The canonical suite
reads encoder Layer 2 (`hidden_states[2]`) and scores with `selected_layer_2_probe.joblib`
(sha256 `bd0d9e14…`). Everything under `causal/layer3_*` reads encoder **Layer 3** and scores with a
*different* probe, `all_layer_probes/layer_3_probe.joblib` (sha256 `2337441e…`). The Layer-3 family is
`superseded`, controls no claim, and is retained only because results here are immutable (`DBPR7`).

The cue in a directory listing is the `layer3_` filename prefix; the cue inside a file is the
`selected_layer` field. Check one of them before comparing anything to a published number.

**`shared/leakage_report.json` is a diagnosis, not a defect left standing.** It reports the *raw* cohort:
259 exact-duplicate groups, 104 spanning splits, 64 label-conflicting. `clean_promoter_cohort.py` then
removes them — 326 rows — and asserts the retained splits are pairwise disjoint, and every artifact in
this tree is computed on that cleaned cohort (`DBP2`). Its report and cleaned hashes are not committed,
which is the residual gap; the counts are recoverable from `frozen/layer_probes/all_layer_probe_results.json`.

## Directories

| Directory | Status | What it holds | Claims |
|---|---|---|---|
| `shared/` | canonical | Raw split manifest and hashes, dataset summary, and the leakage scan that motivated the cleaning stage. | DBP2 |
| `provenance/` | infrastructure | `checkpoint_equivalence.md` — model/revision identity record. | DBP1 |
| `frozen_claim_tests/registry/` | canonical | `frozen_config.json`, the pinned assay configuration. | DBP1 |
| `frozen/layer_probes/` | mixed | Per-layer probe fits, rankings, test statistics, and the canonical probe artifact. The 12 `all_layer_probes/layer_*_probe.joblib` files are `development`; `selected_layer_2_probe.joblib` is `canonical` and pinned by hash. | DBP3, DBP4 |
| `frozen/baselines/` | canonical | Majority, GC-composition, and k-mer 3–6 baselines. | DBP4 |
| `behavioral/` | canonical | All-layer corruption trajectories and the behavioural probe summary. | DBP5 |
| `causal/head_ablation/`, `grouped_head_ablation/`, `branch_ablation/`, `token_alignment/`, `activation_restoration/` | canonical | The **Layer-2** intervention suite: per-head scan over layers 0–2, grouped and branch ablations, negative-class source controls. | DBP5 |
| `causal/layer3_*` (5 directories, 42 files) | **superseded** | The **Layer-3** replicate against a different probe. Controls nothing. Do not quote. | — (`DBPR7`) |
| `frozen_claim_tests/circuit_accounting/` | canonical | Necessity and delivery fractions for the five-node candidate set, validation and untouched test. | DBP6 |
| `frozen_claim_tests/faithfulness_curve/` | canonical | Standalone faithfulness versus circuit size. **The compactness bound** — no knee. | DBP7 |
| `frozen_claim_tests/components/` | canonical | Donor-matrix (activation-transplant) statistics: matched / same-class / opposite-class / zero. | DBP8 |
| `frozen_claim_tests/biological_features/` | canonical | TATA-like enrichment, eligibility, and the controlled perturbation cohort and result. Strata files are `development` — exploratory and uncorrected. | DBP9, DBP10 |

## Polarity, so a null is never re-read as a positive

Three of this family's load-bearing results are **not** positives, and they are the ones most likely to
be misquoted:

- `faithfulness_curve/*` — **bounded**. Refutes compactness at the sizes tested (DBP7).
- `components/donor_matrix/*statistics.json` — **bounded**. The matched-versus-same-class contrast is
  null; only the class contrast is large (DBP8).
- `biological_features/tata_perturbations/*` — **null**. No causal motif dependence at n=71 (DBP10).

## Provenance: fixed forward, gap preserved backward

**Only 5 of the ~109 artifacts in this family have a sibling `*.manifest.json`**, and those five are config
echoes — they carry no `command`, `git_commit`, `created_at_utc`, seed, or input hashes. The repository
invariant is one manifest per result, carrying all of that.

That is fixed for every future run: `src/gfmcd/dnabert/manifest.py` wraps the repository's own
`ExperimentManifest` and is wired into all 15 canonical runners, so a rerun now writes hashed
inputs/outputs, the probe sha256 that defines the readout, the split provenance, the seed, the command, and
the git state. `tests/test_dnabert_manifest.py` pins that contract.

The **historical** artifacts keep a blank `manifest` column, which `results/RESULTS_INDEX.md` defines as
the record "where one exists". Back-filling them would mean asserting provenance nobody recorded — the run
conditions were not captured at the time, and inventing them is worse than an honest blank. They fill in
when the affected assays are next re-run, which the roadmap gates on multi-seed anyway. Note that new
manifests use the repository-standard `<stem>.manifest.json` name; the five hand-written ones use an older
`<stem>_manifest.json` spelling and are immutable.

## Figures

Published figures are in [`figures/dnabert_promoter/`](../../figures/dnabert_promoter/) with captions beside
them, following the repository figure convention. **Figures 1–3 and 9–12 are canonical; 4–8 depict the
superseded Layer-3 replicate** (`DBPR7`). Figures 9–12 were added during the integration review because
`DBP6`, `DBP7` and `DBP8` — including the faithfulness curve that carries the study's central negative
result — had no published figure at all.

`results/**/figures/` is gitignored precisely so that published figures live in one place; the arc
originally force-committed 24 files under `results/dnabert_promoter/figures/frozen/`, and they were
relocated on 2026-08-11.
