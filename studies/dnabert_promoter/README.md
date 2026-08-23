# DNABERT-2 promoter study

This is a bounded test of whether the repository's intervention workflow transfers to another genomic
model, task, and tokenizer. It is not a replication of the SpliceBERT mechanism.

Start with [`CLAIMS.md`](CLAIMS.md), then [`CORRECTIONS.md`](CORRECTIONS.md). The live ledger is
DBP1–DBP10. Every result is single-seed and the study has no frozen preregistration.

## Result

A frozen DNABERT-2 encoder contains promoter-discriminative information accessible to a masked-mean
linear probe. On exact-duplicate-cleaned GUE `prom_core_all` splits, the selected readout reaches test
balanced accuracy `0.8102` and ROC-AUC `0.8919`. Encoder Layer 2 is only a working endpoint: its
validation lead over Layer 3 is `0.0019` on one seed.

A five-node early set has conditional necessity `0.630` and conditional delivery `0.882` on an
untouched 128-example test cohort. The same set has standalone faithfulness `0.194`; faithfulness
reaches `1.0` only after all three early MLPs are included, with no knee in the tested curve.
Activation transplant recovers class-level promoter evidence, but matched-source transfer does not
exceed unrelated same-class transfer (`0.879` versus `0.843`, raw `p=0.252`).

The supported comparison is therefore a structured conditional core inside a distributed,
MLP-dominated computation. It does not establish a compact circuit, example-specific cargo, a TATA
mechanism, or transfer of SpliceBERT's address, payload, write, and delivery roles.

## Evidence status

| Question | Status |
|---|---|
| Data hygiene | Exact duplicates and conflicts removed; retained split hashes asserted disjoint; near-duplicates unscreened (DBP2) |
| Representation | Artifact-backed on cleaned splits (DBP3); layer identity unresolved (DBP4) |
| Component scan | Candidate ranking only; no calibration pair (DBP5) |
| Conditional core | Validation-selected and test-confirmed; non-additive (DBP6) |
| Compactness | Not supported; tested faithfulness curve is MLP-dominated and has no knee (DBP7) |
| Example specificity | Not supported; transfer is class-level under the tested assay (DBP8) |
| Biological bridge | TATA enrichment is observational; perturbation is non-positive at n=71 (DBP9, DBP10) |
| Replication | One seed; validation-to-test confirmation, not seeded replication |
| Preregistration | None frozen (DBPR8) |
| Provenance | Current runners emit manifests, but most historical artifacts predate that contract |

## Archive disposition

The notebook-only fine-tuned extension formerly labeled DBP11–DBP17 was removed from the active tree.
Its 52 artifacts lacked producing runners and sibling manifests, were not clean-checkout reproducible,
and used an unmatched 300-nt task and readout. The superseded top-level `dnabert/` notebooks and
outputs were also removed. Both remain recoverable at Git tag
`archive/unsupported-exploratory-2026-08-22`.

This is an archive decision, not a numerical retraction. A future fine-tuned comparison must begin as a
new study with a matched task and cohort, a frozen protocol, multiple seeds, calibration controls,
producing scripts, and manifests.

The Layer-3 replicate under `results/dnabert_promoter/causal/layer3_*` remains in the tree as a clearly
marked superseded result because its runners and endpoint are recoverable. It controls no claim
(DBPR7).

## Files

| Path | Purpose |
|---|---|
| [`CLAIMS.md`](CLAIMS.md) | Exact licensed wording for DBP1–DBP10 |
| [`CORRECTIONS.md`](CORRECTIONS.md) | DBPR1–DBPR12, including the archive disposition |
| [`DECISIONS.md`](DECISIONS.md) | Frozen-study design decisions and departures |
| [`FROZEN_EXPERIMENT_REGISTRY.md`](FROZEN_EXPERIMENT_REGISTRY.md) | Assay definitions and metrics; not a claim ledger |
| [`results/dnabert_promoter/`](../../results/dnabert_promoter/RESULTS_INDEX.md) | Artifact family index |
| `scripts/dnabert_promoter/` | Frozen-model runners |
| `src/gfmcd/adapters/dnabert.py`, `src/gfmcd/dnabert/` | Adapter, data, and perturbation utilities |
| `configs/dnabert_promoter/base.json` | Frozen assay configuration |
| `figures/dnabert_promoter/` | Canonical figures 1–3 and 9–12; figures 4–8 are marked superseded |

## Instrument

- Model: `quietflamingo/dnabert2-no-flashattention`, revision
  `813031b2bf86d9e960a027e2734c908009f31601`.
- Adapter contract: 12 encoder-layer outputs and no embedding output; `hidden_states[i]` is encoder
  layer `i`.
- Readout: masked mean over biological tokens at encoder Layer 2, followed by the fixed linear probe
  `selected_layer_2_probe.joblib` (sha256
  `bd0d9e14e1935e3ebd8e2ce638ccd5a721feb9dc9a0895447d9b62080ce98b3b`).
- Data: `leannmlindsey/GUE`, `prom_core_all`, 70 nt. Raw train/validation/test counts are
  `47,356/5,920/5,920`; cleaned counts are `47,149/5,856/5,865`.
- Seed: 42 throughout.
- Identity controls: exactly zero in each canonical causal family.

## Reproduction

Install the DNABERT dependencies and run the frozen pipeline from the repository root:

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

See [`REPRODUCE.md`](../../REPRODUCE.md) for environment and data requirements.
