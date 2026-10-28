# DNABERT-2 promoter study

This is a bounded test of whether the repository's intervention workflow transfers to another genomic
model, task, and tokenizer; it is not a replication of the SpliceBERT mechanism. We read a frozen
DNABERT-2 encoder on GUE `prom_core_all` (70 nt, exact duplicates and conflicts removed) through a
masked-mean linear probe at encoder Layer 2, then tested component necessity, delivery, standalone
faithfulness, and activation transplant. Every result is single-seed, and the study froze no
preregistration (DBPR8). Exact wording is in [`CLAIMS.md`](CLAIMS.md) (DBP1–DBP10) and
[`CORRECTIONS.md`](CORRECTIONS.md).

## Findings

The frozen encoder carries promoter information that the probe reads at test balanced accuracy `0.8102`,
but its validation lead over Layer 3 is only `0.0019` on one seed, so layer identity is unresolved (DBP3, DBP4). A
five-node early set, selected on validation and confirmed on an untouched test cohort, has conditional
necessity `0.630` and conditional delivery `0.882` (DBP6). Run alone it reaches faithfulness `0.194`, and
faithfulness reaches `1.0` only once all three early MLPs are included, with no knee, so no compact
circuit is established (DBP7). Activation transplant recovers class-level promoter evidence, but
matched-source transfer does not exceed unrelated same-class transfer (`0.879` versus `0.843`; DBP8).
TATA enrichment is observational, and the readout is not shown to depend causally on TATA-like motifs
(DBP9, DBP10).

## Status

Done, within these limits: the result is a structured conditional core inside a distributed,
MLP-dominated computation. A notebook-only fine-tuned extension and the superseded top-level `dnabert/`
notebooks were archived, without numerical retraction, at tag
`archive/unsupported-exploratory-2026-08-22` (DBPR12). A future fine-tuned comparison must start as a new
study with a matched task and cohort, a frozen protocol, multiple seeds, calibration controls, producing
scripts, and manifests. The superseded Layer-3 replicate controls no claim (DBPR7).

## Files

- [`CLAIMS.md`](CLAIMS.md), [`CORRECTIONS.md`](CORRECTIONS.md), [`DECISIONS.md`](DECISIONS.md): claims,
  corrections, and design decisions.
- [`FROZEN_EXPERIMENT_REGISTRY.md`](FROZEN_EXPERIMENT_REGISTRY.md): assay definitions; not a claim
  ledger.
- [`results/dnabert_promoter/`](../../results/dnabert_promoter/RESULTS_INDEX.md): artifacts.
  [`figures/paper_figure_workflow_transfer.png`](../../figures/paper_figure_workflow_transfer.png): the
  manuscript figure.
- `scripts/dnabert_promoter/`, `src/gfmcd/adapters/dnabert.py`, `src/gfmcd/dnabert/`,
  `configs/dnabert_promoter/base.json`: runners, adapter, utilities, and frozen configuration.

Run the frozen pipeline from the repository root; environment and data requirements are in
[`docs/REPRODUCE.md`](../../docs/REPRODUCE.md).

```bash
uv run --extra dnabert python scripts/dnabert_promoter/run_environment_preflight.py
uv run --extra dnabert python scripts/dnabert_promoter/build_promoter_cohort.py
uv run --extra dnabert python scripts/dnabert_promoter/extract_frozen_representations.py
uv run --extra dnabert python scripts/dnabert_promoter/run_frozen_layer_probes.py
uv run --extra dnabert python scripts/dnabert_promoter/run_sequence_baselines.py
uv run --extra dnabert python scripts/dnabert_promoter/run_frozen_circuit_accounting.py
uv run --extra dnabert python scripts/dnabert_promoter/run_frozen_faithfulness_curve.py
uv run --extra dnabert python scripts/dnabert_promoter/run_component_donor_matrix.py
uv run --extra dnabert python scripts/dnabert_promoter/run_tata_perturbation_probe.py
```
