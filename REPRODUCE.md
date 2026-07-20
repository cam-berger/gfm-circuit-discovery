# Reproducing the current scientific record

This is the canonical repository-level entry point. Study-specific commands and historical environment
snapshots remain in their manifests and reproduction maps, but new development should use the `uv`
environment below.

## 1. Environment

From the repository root:

```bash
uv sync --extra splicebert --extra dev
uv run pytest
```

Use `PYTHONHASHSEED=0` for cohort construction and any shuffle or selection path whose manifest declares
determinism. Use `TOKENIZERS_PARALLELISM=false` for model runs when deterministic, quiet tokenization is
required.

Some historical records name the original conda environment `gfm-thrust1`. Those strings are retained
as execution provenance, not as a second recommended setup. The canonical dependency specification is
now `pyproject.toml` plus `uv.lock`.

## 2. Model and external data

- **Model:** `SpliceBERT.1024nt`, post-LayerNorm BERT MLM, 6 layers × 16 heads, hidden size 512.
- **Checkpoint SHA-256:**
  `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`.
- **Primary genome/annotation:** hg38 plus the GENCODE files named and hashed in each result manifest.
- **Atlas annotation:** GENCODE v50 comprehensive, hash recorded in the Atlas census/package manifests.

Models, FASTA/GTF inputs, and large Atlas runtime shards are not committed. A rerun is comparable only
when its manifest matches the checkpoint, tokenizer, data hashes, split, parameters, seed, and code
revision of the controlling artifact.

## 3. Claim → script → artifact

### Donor graph

| Claim family | Canonical runner(s) | Controlling output |
|---|---|---|
| C1–C6 sealed behavior/circuit | `scripts/splicebert/run_sealed_test.py` and localization/path runners | `results/splicebert_coupling/sealed_test.json`, `position_profile.json`, `readout_path.json` |
| C16–C19 donor readout and signature | `run_donor_anchor.py`, `run_donor_forward.py`, `run_donor_heads.py`, `run_donor_signature.py` | matching JSON + manifest in `results/splicebert_coupling/` |
| C20–C26 address/payload and `U_info` | `run_acceptor_address.py`, `run_a0_*.py` through `run_a5_*.py` | `acceptor_address.json`, `a0_*.json` through `a5_*.json` |
| C27–C33 faithfulness, generator, semantics, geometry | `run_a6_*.py` through `run_a12_*.py` | `a6_*.json` through `a12_*.json` |
| IC1 distal arrangement | `run_ic1_distal_intron_necessity.py` | `ic1_distal_intron_necessity_v8.json` |
| C34/C35 period-3 locus and A-vs-V | runners named in the P3 manifest | `p3_carrier_locus.json`, `p3_av_decomposition.json` |

The detailed command map is
[`studies/splicebert_partner_selection/REPRODUCIBILITY.md`](studies/splicebert_partner_selection/REPRODUCIBILITY.md),
and the protocol status index is
[`studies/splicebert_partner_selection/preregistrations/README.md`](studies/splicebert_partner_selection/preregistrations/README.md).

### Position/carrier record

The canonical index is [`results/phase-lattice/RESULTS_INDEX.md`](results/phase-lattice/RESULTS_INDEX.md).
It maps position spectra, carrier deletion, mediation, attention gates, and biological source assays to
their scripts, immutable JSON, and manifests.

### Head–Context Atlas

| Stage | Canonical entry point | Durable record |
|---|---|---|
| Census and matching | `scripts/splicebert_head_context_atlas/build_atlas_context_census.py`; `build_atlas_matched_cohorts.py` | compact census/matching results + manifests |
| Discovery screen | `run_atlas_discovery_screen.py`; Colab notebook/package | merged compact matrices, run manifests, and audits |
| Analysis and shortlist | `analyze_atlas_discovery.py`; `validate_atlas_discovery_analysis.py` | summary, profiles, shortlist, audit |
| Mechanism validation | `run_atlas_mechanism_validation.py`; `run_atlas_stage2_mechanisms.py` | merged rows, summaries, diagnostics, validation audits |
| Representative DAGs | `build_atlas_representative_dags.py`; validator/exporter | audited compact DAG bundles and manifests |

Bulk `target_shards/`, `health_shards/`, packet matrices, and Colab distribution archives are ignored
because they are reproducible intermediates or retained in external run bundles. The compact artifacts
required to audit the recorded claims are version-controlled.

## 4. Safe reproduction rules

1. Read the corresponding preregistration, amendments, and manifest before execution.
2. Run smoke/preflight gates before target data.
3. Do not reopen the sealed chr8/9 test as a routine smoke check.
4. Write a new immutable artifact; do not overwrite the controlling result.
5. Record checkpoint, tokenizer, dataset hashes, split, parameters, seed, command, and Git revision.
6. Re-run the validator and the narrowest relevant tests, then `uv run pytest` when feasible.
7. Update the live claim ledger only to the strongest level the new evidence licenses; preserve any
   correction in `CORRECTIONS.md` or the appropriate decision log.
