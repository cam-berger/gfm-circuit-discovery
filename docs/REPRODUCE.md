# Reproducing the record

The environment described here supports current development and cannot rebuild a historical
environment that was recorded incompletely. Study-specific commands are in the result manifests and each
study's reproduction map.

## Review the artifacts without installing the project

Use a full Git clone with its history and annotated archive tags. Several provenance checks read old
commits with `git cat-file`, which source ZIP downloads and shallow clones cannot support. Repair a
shallow clone with `git fetch --unshallow --tags origin`, or missing tags with `git fetch --tags origin`.
Then, with Python 3.10 or newer, run this read-only check from the repository root:

```bash
python3 scripts/verify_manifest.py
python3 scripts/verify_records.py
```

Both scripts use only the standard library, as does the
[reciprocal masked-acceptor review command](#reciprocal-masked-acceptor-record), which authenticates the
compact attestations against their archived bytes. That command and `verify_manifest.py` work offline
once the required Git objects and tags are present, and neither loads a model or opens genomic data.

`scripts/verify_manifest.py` checks result-index coverage, CSV/JSON agreement, declared status and
polarity, and referenced paths. It does not validate scientific values or supply missing historical
manifests, and it checks that declared references resolve, not that every historical run has a manifest.
`verify_records.py` verifies every manifest record by content against tracked files, recovering records
whose files moved. Records that cannot verify from a clean checkout are pinned in
`tests/data/record_census.json`; pass manifest paths to see one record per line. `--pins` checks every
pin held outside result manifests (protocol freezes, packages, sidecars, attestations) the same way,
including files inside committed archives, with exceptions pinned in `tests/data/pin_census.json`.

## 1. Environment

Install `uv` and run from the repository root. The checked-in [`.python-version`](../.python-version)
selects Python 3.13, as in [CI](../.github/workflows/tests.yml). Setup needs access to the package
sources or a populated local cache; it does not download checkpoints or genome files.

```bash
uv sync --locked --extra splicebert --extra biology --extra dev
uv run --locked --extra splicebert --extra biology --extra dev pytest -ra
```

Pass the extras on the `uv run` too: a bare `uv run pytest` may resolve a smaller dependency set and fail
during collection. `--locked` requires the checked-in dependency resolution to stay unchanged, and CI runs
the same test command. `-ra` reports skips with their reasons. Tests that need local models, genomic data,
or optional packages may skip, so a passing suite is not a replay of those experiments. Test counts and
duration vary with revision, hardware, inputs, and optional packages.

A focused check of result-index integrity, figure-provenance path handling, and archived-evidence
verification:

```bash
uv run --locked --extra splicebert --extra biology --extra dev pytest -ra \
  tests/test_manifest.py tests/test_figure_provenance.py \
  tests/test_evidence_attestation.py tests/test_acceptor_component_attestation.py \
  tests/test_acceptor_inverse_coupling_attestations.py
```

C36's confirmatory analysis re-executes without a model from the committed shard archive, staging each
package-pinned input by sha256 in a temporary directory and requiring byte-identical output:

```bash
uv run --locked --extra splicebert --extra dev python scripts/splicebert/replay_p3_phase_surgery_confirmation.py
```

Set `PYTHONHASHSEED=0` for cohort construction and any shuffle or selection path whose manifest declares
determinism, and `TOKENIZERS_PARALLELISM=false` for model runs that need deterministic, quiet
tokenization. Dependencies are specified by `pyproject.toml` plus `uv.lock`. The conda environment
`gfm-thrust1` named in some historical records is execution provenance, not a second recommended setup.

### 1.1 Two dependencies the default environment does not install

`maxentpy` 0.0.4 supplies MaxEntScan scoring to `run_a9_payload_semantics.py` (C30),
`run_ic1_pilot2/3.py`, `run_ic2_feasibility.py`, `run_b2_payload_steering.py`, the atlas covariate
builders, `src/gfmcd/splicing/{specificity,splice_landscape}.py`, and the positional-boundary-coherence
Phase-I candidate scorer (`src/gfmcd/splicing/pbc_biology.py`). It is not on PyPI, but the locked
`maxent` extra in `pyproject.toml` pins the exact commit the recorded runs used: add `--extra maxent` to
your `uv sync` or `uv run`. The MaxEnt tables have sha256 `8b6d7b46…b7c00`. Library modules import
`maxentpy` lazily, so `import gfmcd` works without it, but some historical runners import it at module
level and fail immediately. `tests/test_pbc_phase1_real_data.py` skips without it, and a skip there is
not a pass; the top of that file lists what stops being checked, including the MaxEnt pin memorial.

`multimolecule` 0.2.0 is needed by the three cross-model carrier runners under `scripts/crossmodel/`
(C39/C40). It belongs in a separate environment by design; the runners record
`"environment": "gfmcd-mm (conda): python3.11 torch transformers multimolecule0.2.0"` in their manifests.
Do not add it to the main environment. Build the side environment and run those scripts there.

## 2. Model and external data

The model is `SpliceBERT.1024nt`, a post-LayerNorm BERT MLM with 6 layers, 16 heads per layer, and hidden
size 512; its checkpoint hash is in §2.1. The primary genome and annotation are hg38 plus the GENCODE
files named and hashed in each result manifest. The Head–Context Atlas uses GENCODE v50 comprehensive,
hashed in the Atlas census and package manifests.

The origin claims need four more checkpoints. For C39 these are the stage-1 sibling pair
`SpliceBERT.510nt` and `SpliceBERT-human.510nt`, which differ only in training corpus; for C40, RNA-FM and
RNA-BERT, loaded through `multimolecule` in the side environment. Their SHA-256 values are in the
manifests under `results/origin-siblings/` and `results/origin-crossmodel/`. Exact historical checkpoint
acquisition is not automated; providers and access limits are under
[Third-party assets and licenses](#third-party-assets-and-licenses). Confirm the file hash, tokenizer,
and configuration before replay.

### Genome location: `GFMCD_GTF`, `GFMCD_FASTA`, `GFMCD_FASTA_FAI`

Runners and data-gated tests read the genome location from three environment variables (R46 removed a
hard-coded absolute path to one developer's machine). An unset variable falls back to a non-existent
path.

```bash
export GFMCD_GTF=/path/to/gencode.v50.basic.annotation.gtf.gz
export GFMCD_FASTA=/path/to/hg38.fa
export GFMCD_FASTA_FAI=/path/to/hg38.fa.fai
```

Set these before interpreting the suite as a real-data replay. Genome-dependent tests skip when the files
are absent, so read the skip reasons instead of comparing against a historical pass count. Every current
primary runner also accepts `--gtf` and `--fasta`, which override the environment.

### 2.1 What you must supply, and where it goes

Models, FASTA/GTF inputs, and large Atlas runtime shards are not committed; `data/` is
documentation-only in Git (`NUCLEOSOME_MAP.md`). Supply the rest in the repo-relative layout declared in
`configs/splicebert_partner_selection/base.json`:

```
data/
  models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt/   # checkpoint sha256 2ad91428…0149d
  external/gencode/v50/GRCh38.primary_assembly.genome.fa                   # GENCODE v50 primary assembly
  external/gencode/v50/gencode.v50.basic.annotation.gtf.gz                 # GENCODE v50 basic
  processed/splicebert_partner_selection/v1/                               # written by the cohort builders
```

Historical manifests do not share one path convention. Some record the original absolute hg38 path, and
the studies use both GENCODE v50 basic and comprehensive annotations, which are not interchangeable. Go
by the input name and hash in the controlling manifest, and do not assume this layout governed every
study. A rerun is comparable only when its manifest matches the controlling artifact's checkpoint,
tokenizer, data hashes, split, parameters, seed, and code revision.

The donor-arc and register-operator manifests hash the files below, and
`src/gfmcd/splicing/pbc_frozen_inputs.py` enforces the same four values. The layout names the GENCODE
primary-assembly FASTA, but the hashed genome is UCSC `hg38.fa`; supply bytes that match these digests,
whatever the file is called. `gfmcd.models` verifies every checkpoint file before loading. Set
`GFMCD_DATA_ROOTS` to the directory holding `models/` when the weights are not inside the checkout.

| Input | sha256 |
|---|---|
| SpliceBERT.1024nt `pytorch_model.bin` | `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d` |
| `hg38.fa` | `5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51` |
| `gencode.v50.basic.annotation.gtf.gz` | `5b9ad9aa71f6ab5d55123c51f74723644a895bc0cbaa9cee8fbc11bb9dec0bf7` |
| `gencode.v50.annotation.gtf.gz` (comprehensive) | `83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453` |

### 2.2 Six limits on exact reproduction

Read these before comparing numbers.

1. Many historical artifacts do not bind a clean code revision. Some manifests record a dirty tree, omit
   a revision, or omit the dirty flag, so exact code-state replay is unavailable for those rows. Compare
   against the recorded artifact and manifest, and treat missing code identity as a provenance
   limitation; do not reconstruct it after the fact.
2. Some historical `--out` defaults point at the record. They are inventoried in
   [`tests/data/unguarded_out_defaults.json`](../tests/data/unguarded_out_defaults.json), which
   `tests/test_overwrite_guard.py` keeps from growing. Historical runners pinned by their manifests were
   deliberately not edited (R35); new runners use `gfmcd.artifact_guard.guard_output_path`, which refuses
   existing output paths. Inspect the runner and pass an explicit `--out` to a new path when supported.
3. Some live canonical artifacts have no sibling manifest, including `results/p3_carrier_locus.json`
   (C34), `results/p3_av_decomposition.json` (C35), and parts of `results/phase-lattice/` (C8, C11, C13,
   C14, C15). Their manifest cells stay blank because the cleanup does not fabricate missing provenance;
   closing these gaps requires a new rerun.
4. Regenerable binary inputs are not committed. Fitted dictionaries (`*.pt`) and activation captures
   (`*.npz`) had been force-added although `.gitignore` always excluded them, and were untracked on
   2026-07-31 (~847 MB in the checkout, 268 MiB packed). One small `.npz` was deliberately kept:
   `native_context_final_mlp_unit_evaluation_v1.npz` (108 KB), which its own manifest names as an input.
   No claim is controlled by a binary; each controlling artifact is the compact JSON twin, and the sibling
   manifests recording their SHA-256 are kept. A second-stage GTEx runner cannot run from a fresh clone
   until you regenerate the first-stage capture it consumes. Examples are
   `run_gtex_native_context_per_seed_alignment_evaluation.py`, which reads the discovery `.npz` named in
   `native_context_per_seed_alignment_discovery_v1.json` (GFS19), and the J-SAE evaluation stages, which
   read their capture files. Re-run the first-stage runner, then compare against the manifest's hash.
5. 13 canonical-row runners have no `argparse` and take no flags, so `--help` is not handled. It prints
   no usage; it starts a real run against whatever paths are compiled in, and several of those runs write
   to a path `results/MANIFEST.csv` registers as an artifact (limit 2). Read the source before invoking
   any of these, and do not probe them with `--help`: `make_fig_phase_data.py`, `make_fig_report_data.py`,
   `measure_band_mass.py`, `run_carrier_geometry.py`, `run_carrier_subbin.py`, `run_dyad_wavelength.py`,
   `run_human_carrier.py`, `run_postsoftmax.py`, `run_second_carrier.py`, `run_sibling_pspectrum.py`,
   `run_sibling_validate.py`, `run_splice_long.py`, `run_splice_profile.py`.
6. Three retained model-free capacity runners (`audit_real_panel_capacity.py`,
   `run_real_panel_capacity.py`, `run_pbc_v2_capacity.py`) keep original-machine artifact paths. These
   bind large untracked census and cell inputs used by the recorded capacity stop, so the runners are not
   portable clean-checkout reruns. Current primary genome runners use environment variables or CLI paths.
   The historical capacity result remains reviewable through its compact artifact, manifest, and audit.

Departures from frozen protocols are recorded in [`docs/PREREG_CONFORMANCE.md`](PREREG_CONFORMANCE.md),
which audits the SpliceBERT donor/carrier arc only. Read it before treating any preregistered gate as
having been enforced. The GTEx fine-tuning study's gates (`GFS*`) are self-reported by its own ledger and
`DECISIONS.md` and are not independently audited.

## 3. Claims, runners, and artifacts

### Donor graph

| Claim family | Canonical runner(s) | Controlling output |
|---|---|---|
| C1–C6 sealed behavior/circuit | `scripts/splicebert/run_sealed_test.py` and localization/path runners | `results/splicebert_coupling/sealed_test.json`, `position_profile.json`, `readout_path.json` |
| C16–C19 donor readout and signature | `run_donor_anchor.py`, `run_donor_forward.py`, `run_donor_heads.py`, `run_donor_signature.py` | matching JSON + manifest in `results/splicebert_coupling/` |
| C20–C26 address/payload and `U_info` | `run_acceptor_address.py`, `run_a0_*.py` through `run_a5_*.py` | `acceptor_address.json`, `a0_*.json` through `a5_*.json` |
| C27–C33 faithfulness, generator, semantics, geometry | `run_a6_*.py` through `run_a12_*.py` | `a6_*.json` through `a12_*.json` |
| IC1 distal arrangement | `run_ic1_distal_intron_necessity.py` | `ic1_distal_intron_necessity_v8.json` |
| C34/C35 period-3 locus and A-vs-V | `run_p3_carrier_locus.py`, `run_p3_av_decomposition.py` | `results/p3_carrier_locus.json`, `results/p3_av_decomposition.json` (no sibling `.manifest.json`; see §2.2) |

The detailed command map is
[`studies/splicebert_partner_selection/REPRODUCIBILITY.md`](../studies/splicebert_partner_selection/REPRODUCIBILITY.md),
and the protocol status index is
[`studies/splicebert_partner_selection/preregistrations/README.md`](../studies/splicebert_partner_selection/preregistrations/README.md).

### Reciprocal masked-acceptor record

The active tree keeps the compact attestations, component audit, and frozen protocols needed to review
the masked-acceptor record; large raw packages remain immutable under the archive tag. Historical runners
are not copied into current source. Their frozen authorization binds the archived execution tree, and the
stopped programs cannot authorize a new run from current master.

| Evidence family | Descriptive record | Legacy aliases |
|---|---|---|
| supported component interventions | `acceptor-coupling/components/l1h12-mean-ablation`; `acceptor-coupling/components/l4-head-noising` | `AC5`, `AC6`, `R3c v2.1` |
| bounded component limits | `acceptor-coupling/components/route-hypotheses-inconclusive`; `acceptor-coupling/components/l2h15-health-ineligible` | `AC7`, `AC8`, `R3c v2.1` |
| terminal pre-target instrument stop | `acceptor-coupling/instrument/frozen-handoff-schema-mismatch` | `AC9`, `R1c→R5` |
| natural-`GC` behavior modifier | `acceptor-coupling/natural-gc/behavior-modifier` | `AC10`, `M6` |
| natural-`GC` attention modifier | `acceptor-coupling/natural-gc/l3h1-attention-modifier` | `AC11`, `M6` |

The development record `acceptor-coupling/natural-gc/qk-score-diagnostic` (legacy `M4d`) has no claim
alias. It preserves nominal, unadjusted pre-softmax score-margin diagnostics and their limitations, not a
promoted result.

The review path needs no model or genome:

```bash
python3 scripts/verify_archived_evidence.py \
  results/splicebert_acceptor_coupling/attestations/paired_donor_component_confirmation_v1.json \
  results/splicebert_acceptor_coupling/attestations/frozen_handoff_schema_mismatch_v1.json \
  results/splicebert_acceptor_coupling/attestations/natural_gc_validation_v1.json \
  results/splicebert_acceptor_coupling/attestations/development/natural_gc_qk_score_diagnostic_v1.json
uv run --locked --extra splicebert --extra biology --extra dev pytest -ra tests/test_evidence_attestation.py \
  tests/test_acceptor_component_attestation.py \
  tests/test_acceptor_inverse_coupling_attestations.py
```

The verifier authenticates the annotated archive tag, execution provenance, raw package paths, byte sizes
and SHA-256 digests, the active audit and protocol where present, and the selected JSON facts each
compact record summarizes. It reports dirty or historical-nonfinite provenance as a limitation instead of
silently normalizing it, and it does not rescore the model. To inspect by hand, use
`git show archive/acceptor-inverse-coupling-2026-08-17:<path>`; never recreate or rewrite the frozen
package in place.

#### BG1/BG2 acceptor-boundary grammar: development preflight failed, targets not opened

The 2026-08-11 execution stopped at the complete development preflight, as the frozen chain requires.
The pre-amendment smoke is preserved as superseded; the corrected smoke passed its active mechanical
checks but was non-authorizing. The complete preflight then reported `PASS=false` on planted
sensitivity, route dose sensitivity, route resolution, and target resolution, and its disposition is
instrument-unresolved. No BG1 validation behavior or BG2 intervention was run, and no claim depends on
the development estimates. Steps 3–6 are shown so the authorization rule can be audited, not as commands
that may now be run.

```bash
# 1. Historical smoke sequence. A smoke artifact can never authorize a target.
#    The first output is superseded; the v2 output carries the repaired smoke semantics.
uv run python scripts/splicebert/run_bg1_boundary_preflight.py --smoke \
  --out results/splicebert_coupling/bg1_boundary_preflight_smoke_v2_20260811.json

# 2. Recorded terminal command: train-split supply, invariants, controls, power, FPR, and MDE.
#    It wrote bg1_boundary_preflight_v1.json with PASS=false and stopped the arc.
uv run python scripts/splicebert/run_bg1_boundary_preflight.py

# NOT AUTHORIZED AFTER THE RECORDED PREFLIGHT FAILURE:
# 3. BG1 behaviour on val (verification tier). Requires a passing, non-smoke preflight.
uv run python scripts/splicebert/run_bg1_boundary_grammar.py --split val

# 4. Independent model-free audit of BG1. Required before any BG1 wording is claim-bearing.
uv run python scripts/splicebert/audit_bg_boundary.py \
  --result   results/splicebert_coupling/bg1_boundary_grammar_val_v1.json \
  --manifest results/splicebert_coupling/bg1_boundary_grammar_val_v1.manifest.json

# 5. BG2 route recovery — only if BG1's verdict is SUPPORTED_VERIFICATION and its audit PASSes.
#    BG2 re-runs the BG1 audit itself and compares, before loading the model.
uv run python scripts/splicebert/run_bg2_boundary_payload_recovery.py --split val

# 6. Independent audit of BG2.
uv run python scripts/splicebert/audit_bg_boundary.py \
  --result   results/splicebert_coupling/bg2_boundary_payload_recovery_val_v1.json \
  --manifest results/splicebert_coupling/bg2_boundary_payload_recovery_val_v1.manifest.json
```

The protocol freezes `--enable-test-replication` (chr8/9) as disabled, and it must stay off: the
protocol's §2 records that the motif-clean cohort cannot clear the target supply gate, so no sealed-test
run is authorized. A halt at step 2 is a legitimate terminal outcome; do not relax a gate to get past it.
Before citing any BG1/BG2 gate as enforced, read [`docs/PREREG_CONFORMANCE.md`](PREREG_CONFORMANCE.md)
and the three open design questions in
[`DECISIONS_A0.md`](../studies/splicebert_partner_selection/DECISIONS_A0.md) § "BG1/BG2".

### Position/carrier record

[`results/phase-lattice/RESULTS_INDEX.md`](../results/phase-lattice/RESULTS_INDEX.md) is the canonical
index. It maps position spectra, carrier deletion, mediation, attention gates, and biological source
assays to their scripts, immutable JSON, and manifests.

### GTEx fine-tuning shift

The five published SpliceAI-GTEx fine-tunes contain a token-classification encoder and head, not an MLM
head. `scripts/splicebert_gtex_finetune_shift/prepare_gtex_mlm_transplant.py` strictly transplants each
complete encoder into the original masked LM and writes a derived Hugging Face checkpoint with an audit
report. Run `run_gtex_finetune_preflight.py` before `run_gtex_coupling_shift.py`, which requires all five
seeds, replays the exact base validation result before opening them, and stops at behavior. Exact hashes,
readout scope, commands, and the outcome-aware verification status are in
[`studies/splicebert_gtex_finetune_shift/`](../studies/splicebert_gtex_finetune_shift/README.md).

### Head–Context Atlas

| Stage | Canonical entry point | Durable record |
|---|---|---|
| Census and matching | `scripts/splicebert_head_context_atlas/build_atlas_context_census.py`; `build_atlas_matched_cohorts.py` | compact census/matching results + manifests |
| Discovery screen | `run_atlas_discovery_screen.py`; Colab notebook/package | merged compact matrices, run manifests, and audits |
| Analysis and shortlist | `analyze_atlas_discovery.py`; `validate_atlas_discovery_analysis.py` | summary, profiles, shortlist, audit |
| Mechanism validation | `run_atlas_mechanism_validation.py`; `run_atlas_stage2_mechanisms.py` | merged rows, summaries, diagnostics, validation audits |
| Representative DAGs | `build_atlas_representative_dags.py`; validator/exporter | audited compact DAG bundles and manifests |

Bulk `target_shards/`, `health_shards/`, packet matrices, and Colab distribution archives are ignored
because they are reproducible intermediates or are retained in external run bundles. The compact
artifacts needed to audit the recorded claims are version-controlled.

### Positional boundary coherence (model-free execution only)

There is no successor model outcome to reproduce. `studies/splicebert_positional_boundary_coherence/`
ships a frozen preregistration, 29 runner/audit entry points under
`scripts/splicebert_positional_boundary_coherence/`, eleven `src/gfmcd/splicing/pbc_*` and
`positional_boundary_coherence` modules, and their tests. Only the model-free portion was executed. Its
compact freshness and census audit receipts and the terminal capacity stop and audit are retained; the
2.34-GB raw candidate census is intentionally ignored and must be regenerated from its bound producer
manifest. There is no checkpoint, preflight, target, or model-outcome artifact.

| Stage | Producer | Independent authorization |
|---|---|---|
| Freshness registry | `build_freshness_registry.py` | `audit_freshness_registry.py` |
| Real-sequence census | `build_candidate_census.py` | `audit_candidate_census.py` |
| Balanced cohort | `build_composition_balanced_cohort.py` | `audit_composition_balanced_cohort.py` |
| Pre-target protocol/code conformance | `audit_protocol_code_conformance.py` | its passing artifact and manifest are required and hash-bound by the Phase-II freezer |
| P3-first attenuation development | `calibrate_carrier_factorial.py` → `run_carrier_factorial_preflight.py` | `audit_carrier_factorial_preflight.py` → `freeze_carrier_factorial_target.py` |
| Attenuation confirmation | `run_carrier_factorial_target.py` → `analyze_carrier_factorial.py` | `audit_carrier_factorial.py` |
| P3 register development | `run_p3_register_preflight.py` → `analyze_p3_register_preflight.py` | `audit_p3_register_preflight.py` |
| Route discovery and freeze | `run_p3_register_candidate_screen.py` → `freeze_p3_register_candidates.py` | `audit_p3_register_candidate_freeze.py` |
| Reserve seal | `freeze_p3_register_target.py` | the seal binds authorization, candidates, code, inputs, and every output path |
| Register confirmation | `run_p3_register_target.py` → `assemble_p3_register_target.py` → `analyze_p3_register_relocation.py` | `audit_p3_register_relocation.py` |
| Route mediation | `run_p3_register_mediation_target.py` → `assemble_p3_register_mediation_target.py` → `analyze_p3_register_mediation.py` | `audit_p3_register_mediation.py` |

Every edge is fail-closed, and the terminal Phase-I capacity check precedes the four target bindings.
Before expensive cohort matching, the balanced-cohort producer checks the necessary unique
normalized-gene capacity for all seven source slots against the total frozen request (120 + 600 + 384 =
1,104 blocks). On failure it writes a compact `PHASE1_INSTRUMENT_UNRESOLVED` artifact and manifest with
empty block and cell tables, then exits 2. The independent auditor rederives the capacity and exits 0
only when its own audit passes, returning `AUDIT_PASS_INSTRUMENT_UNRESOLVED`; target authorization
remains false. Both records bind the ignored census's byte identity, semantic hashes, producer revision,
and `candidate_census_v1.manifest.json` regeneration receipt, and both state
`clean_checkout_reexecutable=false`. Do not reduce a role or open sealed chromosomes in response.

The recorded run found donor-phase-2 capacity `809/1,104` (deficit 295) and acceptor-phase-2 capacity
`854/1,104` (deficit 250); the independent audit passed all 48 checks with
`target_execution_authorized=false`. To replay it, first regenerate `development/candidate_census_v1.json`
from the exact command and pinned inputs in its retained manifest. For byte-identical replay use producer
revision `addd7be93c321373118b02b82bc3d95eeb0bd20a`; the retained census byte SHA-256 is
`9be356ac26ff18aa750b0552684d4a94fa6b8963fee01666608b38cc7de25614`.

Four further rules apply:

1. The preregistration's own "Present status" requires a freeze package binding the final code and
   inputs, plus an independent protocol-to-code conformance audit, before any target or Phase-III reserve
   opens. Do not run a target without it.
2. Every model run records the checkpoint, base `SpliceBERT.1024nt` `pytorch_model.bin` sha256
   `2ad91428…0149d`. A mismatch is fatal, not a warning.
3. `chr8`/`chr9` stay globally sealed, and no script may fall back to them when capacity is low. `chrY`
   is excluded. Genes are excluded by the rederived freshness registry covering every C43/C44/GFS25 role.
4. Producers cannot authorize themselves. Every row of the table pairs a producer with a separate
   auditor, and no later stage may infer a positive, negative, or bounded result from a failed instrument.

These runners refuse to overwrite an existing artifact (`gfmcd.artifact_guard.guard_output_path`). Results
go to `results/splicebert_positional_boundary_coherence/` under the layout its
[`RESULTS_INDEX.md`](../results/splicebert_positional_boundary_coherence/RESULTS_INDEX.md) declares. The
PBC-focused suite is `uv run --extra splicebert --extra dev --extra maxent pytest tests/ -k pbc`; most of
its tests build tiny FASTA/GTF fixtures in `tmp_path` and need no genome or checkpoint. Without
`--extra maxent`, the real-data test file skips as a whole (§1.1).

## 4. Safe reproduction rules

1. Read the corresponding preregistration, amendments, and manifest before execution.
2. Run smoke/preflight gates before target data.
3. Do not reopen the sealed chr8/9 test as a routine smoke check.
4. Write a new immutable artifact; do not overwrite the controlling result.
5. Record checkpoint, tokenizer, dataset hashes, split, parameters, seed, command, and Git revision.
6. Re-run the validator and the narrowest relevant tests, then the full suite with the extras in §1
   when feasible.
7. Update the live claim ledger only to the strongest level the new evidence supports; preserve any
   correction in `CORRECTIONS.md` or the appropriate decision log.

## Third-party assets and licenses

The repository's original code and documentation are licensed under the root [`LICENSE`](../LICENSE)
(MIT). That license does not relicense model checkpoints, reference genomes, annotations, human-subject
data, or other third-party assets. Large models and source datasets are not redistributed in the tracked
repository; users obtain them from the upstream providers and must follow their terms. The current
manifest standard requires exact source and hash records, but historical coverage is incomplete. Consult
[`PREREG_CONFORMANCE.md`](PREREG_CONFORMANCE.md) and the relevant correction ledger before treating any
run as exactly reconstructible.

| Asset | Upstream source | Terms / license checked | Repository handling |
|---|---|---|---|
| SpliceBERT code and published checkpoints | [biomed-AI/SpliceBERT](https://github.com/biomed-AI/SpliceBERT) | BSD-3-Clause in the upstream repository | Checkpoints are downloaded separately and ignored by Git; this repository distributes only adapters, derived aggregate results, and hashes. |
| SpliceBERT GTEx analysis/fine-tuned checkpoints | [chenkenbio/SpliceBERT-analysis](https://github.com/chenkenbio/SpliceBERT-analysis) | BSD-3-Clause in the upstream repository | Checkpoints are not redistributed. No participant-level GTEx sequence or phenotype data is tracked. |
| GENCODE v50 and GRCh38 primary assembly | [GENCODE data access](https://www.gencodegenes.org/pages/data_access.html) | GENCODE states that its project data are open access | Source files are downloaded separately; manifests pin filenames and hashes. |
| GTEx data | [GTEx Portal](https://gtexportal.org/home/downloads/adult-gtex) and [dbGaP access guidance](https://www.grants.nih.gov/policy-and-compliance/policy-topics/sharing-policies/accessing-data/dbgap) | GTEx distinguishes open-access resources from protected participant-level data requiring authorization | This repository does not redistribute controlled GTEx data. Any future use of protected data requires independent authorization and compliance with its data-use limitations. |
| ENCODE data | [ENCODE data-use policy](https://www.encodeproject.org/about/data-use-policy/) | ENCODE releases data for unrestricted use immediately and requests appropriate citation | Source records are downloaded separately; only derived cohorts, aggregate results, and provenance are tracked. |
| DNABERT-2 source code | [MAGICS-LAB/DNABERT_2](https://github.com/MAGICS-LAB/DNABERT_2) | Apache-2.0 in the upstream repository | Source is not vendored. The base and promoter-fine-tuned checkpoints are downloaded separately. Their Hugging Face cards do not currently declare a model license, so the checkpoints are not redistributed under this repository's MIT license. |
| RNA-FM | [ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM) | MIT in the upstream repository | Model files are downloaded separately and are not tracked. |
| RNA-BERT | Source cited by the RNA-BERT publication/model provider | No separate license was verified during this audit | Model files are not redistributed. Users must verify the current upstream terms before obtaining or sharing them. |
| EPDnew, PolyASite, FANTOM5, and other atlas catalogs | Provider named in each study manifest/protocol | Dataset-specific terms apply; no blanket relicensing is asserted here | These inputs are not redistributed. The manuscript's DNABERT-2 comparison is a bounded historical probe study, not a live claim from these catalogs. |
| NeurIPS 2026 author kit | Supplied NeurIPS 2026 formatting bundle | No separate license statement was present in the supplied `neurips_2026.sty` header | The style, example, and checklist files used by the current draft are not tracked in this repository. Their earlier inclusion did not relicense them under the repository's MIT license. |
| Third-party Python/R/system dependencies | `pyproject.toml`, lockfile, and environment documentation | Each package retains its upstream license | Dependencies are installed from their normal package channels; their source is not vendored by this repository. |

Attribution and redistribution rules:

- Cite the original dataset and model papers in publications, in addition to following provider-specific
  attribution instructions.
- Do not commit access tokens, participant-level controlled data, or checkpoint weights.
- Before publishing an artifact bundle, verify every included file against this table and its manifest.
- If an upstream model card lacks a declared license, do not infer one from the accompanying code
  repository; distribute only repository-owned code and independently derived aggregate results.

This table documents upstream statements and repository handling; it is not legal advice.
