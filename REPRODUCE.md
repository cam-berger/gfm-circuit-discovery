# Reproducing the current scientific record

This guide separates review of the recorded evidence, software checks, and model/data replay.
Study-specific commands and historical environments remain in the result manifests and reproduction
maps. The environment below supports current development; it does not reconstruct an incompletely
recorded historical environment.

## Review the artifacts without installing the project

Use a full Git clone with its history and annotated archive tags. Several provenance checks read old
commits with `git cat-file`; source ZIP downloads and shallow clones cannot reproduce those checks.
For an existing shallow clone, run `git fetch --unshallow --tags origin`; for a full clone with missing
tags, run `git fetch --tags origin`.

With Python 3.10 or newer and Git available, run this read-only check from the repository root:

```bash
python3 scripts/verify_manifest.py
```

This uses only the Python standard library. It checks result-index coverage, CSV/JSON agreement,
declared status and polarity, and referenced paths. It does not validate scientific values or supply
missing historical manifests. The [reciprocal masked-acceptor review command](#reciprocal-masked-acceptor-record)
below also uses only the standard library and authenticates the compact attestations against their
archived bytes. Both commands work offline once the required Git objects and tags are present; neither
loads a model or opens genomic data.

## 1. Environment

For software tests, install `uv` and run from the repository root. The checked-in
[`.python-version`](.python-version) selects Python 3.13, as in
[CI](.github/workflows/tests.yml). Initial setup requires access to the package sources or a populated
local cache; checkpoints and genome files are not downloaded by these commands.

```bash
uv sync --locked --extra splicebert --extra biology --extra dev
uv run --locked --extra splicebert --extra biology --extra dev pytest -ra
```

**Pass the extras on the `uv run` too.** A bare `uv run pytest` may resolve a smaller dependency set and
fail during collection. `--locked` requires the checked-in dependency resolution to remain unchanged.
CI executes the same test command. `-ra` reports skipped tests and their reasons: tests requiring local
models, genomic data, or optional packages may skip. A passing software suite is not a replay of those
experiments. Test counts and duration depend on the revision, hardware, inputs, and optional packages.

For a focused check of result-index integrity, figure-provenance path handling, and archived-evidence
verification after installation:

```bash
uv run --locked --extra splicebert --extra biology --extra dev pytest -ra \
  tests/test_manifest.py tests/test_figure_provenance.py \
  tests/test_evidence_attestation.py tests/test_acceptor_component_attestation.py \
  tests/test_acceptor_inverse_coupling_attestations.py
```

C36's confirmatory analysis re-executes without a model from the committed shard archive. The command
stages each package-pinned input by sha256 in a temporary directory, and requires byte-identical output:

```bash
uv run --locked --extra splicebert --extra dev python scripts/splicebert/replay_p3_phase_surgery_confirmation.py
```

Use `PYTHONHASHSEED=0` for cohort construction and any shuffle or selection path whose manifest declares
determinism. Use `TOKENIZERS_PARALLELISM=false` for model runs when deterministic, quiet tokenization is
required.

Some historical records name the original conda environment `gfm-thrust1`. Those strings are retained
as execution provenance, not as a second recommended setup. The canonical dependency specification is
now `pyproject.toml` plus `uv.lock`.

### 1.1 Two dependencies the default environment does not install

| Package | Needed by | How to get it |
|---|---|---|
| `maxentpy` 0.0.4 | MaxEntScan scoring: `run_a9_payload_semantics.py` (C30), `run_ic1_pilot2/3.py`, `run_ic2_feasibility.py`, `run_b2_payload_steering.py`, the atlas covariate builders, `src/gfmcd/splicing/{specificity,splice_landscape}.py`, and the positional-boundary-coherence Phase-I candidate scorer (`src/gfmcd/splicing/pbc_biology.py`) | Not on PyPI, but `pyproject.toml` ships a locked **`maxent` extra** pinning the exact commit the recorded runs used — add `--extra maxent` to your `uv sync`/`uv run`. MaxEnt tables sha256 `8b6d7b46…b7c00`. |
| `multimolecule` 0.2.0 | the three cross-model carrier runners under `scripts/crossmodel/` (C39/C40) | A **separate** environment by design — the runners record `"environment": "gfmcd-mm (conda): python3.11 torch transformers multimolecule0.2.0"` in their manifests. Do not add it to the main env; build the side environment and run those scripts there. |

The library modules import `maxentpy` lazily, so `import gfmcd` works without it; some historical runners
import it at module level and fail immediately if it is absent. `tests/test_pbc_phase1_real_data.py`
**skips** without it — a skipped run there is not a passing one, and what stops being checked (including
the MaxEnt pin memorial) is listed at the top of that file.

## 2. Model and external data

- **Model:** `SpliceBERT.1024nt`, post-LayerNorm BERT MLM, 6 layers × 16 heads, hidden size 512.
- **Checkpoint SHA-256:**
  `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`.
- **Primary genome/annotation:** hg38 plus the GENCODE files named and hashed in each result manifest.
- **Atlas annotation:** GENCODE v50 comprehensive, hash recorded in the Atlas census/package manifests.

### Tell the code where the genome is — `GFMCD_GTF` / `GFMCD_FASTA` / `GFMCD_FASTA_FAI`

Runners and data-gated tests no longer carry a hard-coded absolute path to one developer's machine
(R46). They read three environment variables, falling back to a non-existent path when unset:

```bash
export GFMCD_GTF=/path/to/gencode.v50.basic.annotation.gtf.gz
export GFMCD_FASTA=/path/to/hg38.fa
export GFMCD_FASTA_FAI=/path/to/hg38.fa.fai
```

**Set these before interpreting the suite as a real-data replay.** Genome-dependent tests are guarded on
file existence and report as skipped when the inputs are absent. Read the skip reasons rather than
comparing against a historical pass count. Every current primary runner also accepts `--gtf` and
`--fasta`, which override the environment.

Four more checkpoints are needed for the origin claims. C39 needs the stage-1 sibling pair `SpliceBERT.510nt` and
`SpliceBERT-human.510nt`, which differ only in training corpus; C40 needs RNA-FM and RNA-BERT, loaded
through `multimolecule` in the separate side environment. Their SHA-256 values are recorded in the
manifests under `results/origin-siblings/` and `results/origin-crossmodel/`, so you can verify a candidate
file. Provider references and access limitations are listed in
[`docs/THIRD_PARTY_ASSETS.md`](docs/THIRD_PARTY_ASSETS.md); exact historical checkpoint acquisition is
not automated here. Confirm the file hash, tokenizer, and configuration before replay.

Models, FASTA/GTF inputs, and large Atlas runtime shards are not committed. A rerun is comparable only
when its manifest matches the checkpoint, tokenizer, data hashes, split, parameters, seed, and code
revision of the controlling artifact.

### 2.1 What you must supply, and where it goes

`data/` is documentation-only in Git (`NUCLEOSOME_MAP.md`). Everything below is yours to provide. The
repo-relative layout is the one declared in `configs/splicebert_partner_selection/base.json`:

```
data/
  models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt/   # checkpoint sha256 2ad91428…0149d
  external/gencode/v50/GRCh38.primary_assembly.genome.fa                   # GENCODE v50 primary assembly
  external/gencode/v50/gencode.v50.basic.annotation.gtf.gz                 # GENCODE v50 basic
  processed/splicebert_partner_selection/v1/                               # written by the cohort builders
```

Historical manifests do not use one uniform path convention: some record the original absolute hg38
path, and the studies use both GENCODE v50 basic and comprehensive annotations. Those annotations are
not interchangeable. Compare the input name and hash in the controlling manifest rather than assuming
that the repository-level example layout governed every study.

The donor-arc and register-operator manifests hash the files below. The layout above names the GENCODE
primary-assembly FASTA, but the hashed genome is UCSC `hg38.fa`; supply bytes that match these digests,
whatever the file is called. `src/gfmcd/splicing/pbc_frozen_inputs.py` enforces the same four values.

| Input | sha256 |
|---|---|
| SpliceBERT.1024nt `pytorch_model.bin` | `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d` |
| `hg38.fa` | `5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51` |
| `gencode.v50.basic.annotation.gtf.gz` | `5b9ad9aa71f6ab5d55123c51f74723644a895bc0cbaa9cee8fbc11bb9dec0bf7` |
| `gencode.v50.annotation.gtf.gz` (comprehensive) | `83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453` |

### 2.2 Six limits on exact reproduction — read before comparing numbers

1. **Many historical artifacts do not bind a clean code revision.** Some manifests record a dirty tree,
   omit a revision, or omit the dirty flag. Exact code-state replay is therefore unavailable for those
   rows. Compare against the recorded artifact and manifest, and treat missing code identity as a
   provenance limitation rather than reconstructing it after the fact.
2. **Some historical `--out` defaults point at the record.** The retained inventory in
   [`tests/data/unguarded_out_defaults.json`](tests/data/unguarded_out_defaults.json) lists runners
   whose defaults can overwrite recorded artifacts. `tests/test_overwrite_guard.py` prevents that
   inventory from growing. New runners use `gfmcd.artifact_guard.guard_output_path`, which refuses
   existing output paths; historical runners pinned by their manifests were deliberately not edited
   (R35). **Inspect the runner and pass an explicit `--out` to a new path when supported.**
3. **Some live canonical artifacts have no sibling manifest.** This includes
   `results/p3_carrier_locus.json` (C34), `results/p3_av_decomposition.json` (C35), and parts of
   `results/phase-lattice/` (C8, C11, C13, C14, C15). Their manifest cells remain blank; the cleanup does
   not fabricate missing provenance. `scripts/verify_manifest.py` checks that declared references
   resolve, not that every historical run has a manifest. Closing these gaps requires a new rerun.
4. **Regenerable binary inputs are not committed.** Fitted dictionaries (`*.pt`) and activation captures
   (`*.npz`) were untracked on 2026-07-31 (~847 MB in the checkout, 268 MiB packed; `.gitignore` had
   always excluded them and they had been force-added). **One small `.npz` was deliberately kept** —
   `native_context_final_mlp_unit_evaluation_v1.npz`, 108 KB, which its own manifest names as an
   input. No claim is controlled by a binary — every controlling artifact is the compact JSON
   twin, and the sibling manifests recording their SHA-256 are kept. But a **second-stage** GTEx runner
   that consumes a first-stage capture cannot run from a fresh clone until you regenerate it: notably
   `run_gtex_native_context_per_seed_alignment_evaluation.py` reads the discovery `.npz` named in
   `native_context_per_seed_alignment_discovery_v1.json` (GFS19), and the J-SAE evaluation stages read
   their capture files. Re-run the first-stage runner to rebuild the input, then compare against the
   recorded hash in the manifest.
5. **13 canonical-row runners have no `argparse` at all.** They take no flags, so `--help` is *not*
   handled: passing it does not print usage, it **starts a real run** against whatever paths are compiled
   in, and several of those runs write to a path `results/MANIFEST.csv` registers as an artifact (limit 2).
   The affected runners are `make_fig_phase_data.py`, `make_fig_report_data.py`, `measure_band_mass.py`,
   `run_carrier_geometry.py`, `run_carrier_subbin.py`,
   `run_dyad_wavelength.py`, `run_human_carrier.py`, `run_postsoftmax.py`, `run_second_carrier.py`,
   `run_sibling_pspectrum.py`, `run_sibling_validate.py`, `run_splice_long.py`, `run_splice_profile.py`.
   **Read the source before invoking any of them** — do not probe them with `--help`.
6. **Three retained model-free capacity runners preserve original-machine artifact paths.** They are
   `audit_real_panel_capacity.py`, `run_real_panel_capacity.py`, and `run_pbc_v2_capacity.py`. The paths
   bind large untracked census/cell inputs used by the recorded capacity stop; they are not portable
   clean-checkout reruns. Current primary genome runners use environment variables or CLI paths. The
   historical capacity result remains reviewable through its compact artifact, manifest, and audit.

Where a runner departed from its frozen protocol, the deviation is recorded in
[`docs/PREREG_CONFORMANCE.md`](docs/PREREG_CONFORMANCE.md) — which audits the **SpliceBERT donor/carrier
arc only**. The GTEx fine-tuning study's gates (`GFS*`) are self-reported by its own ledger and
`DECISIONS.md`, not independently audited. Read the conformance file before treating any preregistered
gate as having been enforced.

## 3. Claim → script → artifact

### Donor graph

| Claim family | Canonical runner(s) | Controlling output |
|---|---|---|
| C1–C6 sealed behavior/circuit | `scripts/splicebert/run_sealed_test.py` and localization/path runners | `results/splicebert_coupling/sealed_test.json`, `position_profile.json`, `readout_path.json` |
| C16–C19 donor readout and signature | `run_donor_anchor.py`, `run_donor_forward.py`, `run_donor_heads.py`, `run_donor_signature.py` | matching JSON + manifest in `results/splicebert_coupling/` |
| C20–C26 address/payload and `U_info` | `run_acceptor_address.py`, `run_a0_*.py` through `run_a5_*.py` | `acceptor_address.json`, `a0_*.json` through `a5_*.json` |
| C27–C33 faithfulness, generator, semantics, geometry | `run_a6_*.py` through `run_a12_*.py` | `a6_*.json` through `a12_*.json` |
| IC1 distal arrangement | `run_ic1_distal_intron_necessity.py` | `ic1_distal_intron_necessity_v8.json` |
| C34/C35 period-3 locus and A-vs-V | `run_p3_carrier_locus.py`, `run_p3_av_decomposition.py` | `results/p3_carrier_locus.json`, `results/p3_av_decomposition.json` (**no sibling `.manifest.json` — see §2.2**) |

The detailed command map is
[`studies/splicebert_partner_selection/REPRODUCIBILITY.md`](studies/splicebert_partner_selection/REPRODUCIBILITY.md),
and the protocol status index is
[`studies/splicebert_partner_selection/preregistrations/README.md`](studies/splicebert_partner_selection/preregistrations/README.md).

### Reciprocal masked-acceptor record

The active tree retains compact attestations, the component audit, and the frozen protocols needed to
review the masked-acceptor record. Large raw packages remain immutable under the archive tag. Historical
runners are not copied into current source: their frozen authorization binds the archived execution
tree, and the stopped programs cannot authorize a new run from current master.

| Evidence family | Descriptive record | Legacy aliases |
|---|---|---|
| supported component interventions | `acceptor-coupling/components/l1h12-mean-ablation`; `acceptor-coupling/components/l4-head-noising` | `AC5`, `AC6`, `R3c v2.1` |
| bounded component limits | `acceptor-coupling/components/route-hypotheses-inconclusive`; `acceptor-coupling/components/l2h15-health-ineligible` | `AC7`, `AC8`, `R3c v2.1` |
| terminal pre-target instrument stop | `acceptor-coupling/instrument/frozen-handoff-schema-mismatch` | `AC9`, `R1c→R5` |
| natural-`GC` behavior modifier | `acceptor-coupling/natural-gc/behavior-modifier` | `AC10`, `M6` |
| natural-`GC` attention modifier | `acceptor-coupling/natural-gc/l3h1-attention-modifier` | `AC11`, `M6` |

The development record `acceptor-coupling/natural-gc/qk-score-diagnostic` (legacy `M4d`) has no claim
alias: it preserves nominal, unadjusted pre-softmax score-margin diagnostics and their limitations, not
a promoted result.

The review path is model- and genome-free:

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

The verifier authenticates the annotated archive tag, execution provenance, raw package paths, byte
sizes and SHA-256 digests, active audit/protocol where present, and the selected JSON facts summarized
by each compact record. It preserves dirty or historical-nonfinite provenance as a limitation rather
than silently normalizing it. It does not rescore the model. For manual inspection, use
`git show archive/acceptor-inverse-coupling-2026-08-17:<path>`; never recreate or rewrite the frozen
package in place.

#### BG1/BG2 acceptor-boundary grammar — development preflight failed; targets not opened

The 2026-08-11 execution stopped at the complete development preflight, exactly as the frozen chain
requires. The pre-amendment smoke is preserved as superseded; the corrected smoke passed its active
mechanical checks but was non-authorizing. The complete preflight then reported `PASS=false` on planted
sensitivity, route dose sensitivity, route resolution, and target resolution. Its disposition is
instrument-unresolved. No BG1 validation behavior or BG2 intervention was run, and no claim depends on
the development estimates. The commands below document the chain and its terminal point; steps 3–6 are
shown to make the authorization rule auditable, not as commands that may now be run.

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

`--enable-test-replication` (chr8/9) is frozen **disabled** by the protocol and must stay off: §2 records
that the motif-clean cohort cannot clear the target supply gate, so no sealed-test run is authorized. A
halt at step 2 is a legitimate terminal outcome, not a failure to work around — do not relax a gate to get
past it. Before citing any BG1/BG2 gate as enforced, read
[`docs/PREREG_CONFORMANCE.md`](docs/PREREG_CONFORMANCE.md) and the three open design questions in
[`DECISIONS_A0.md`](studies/splicebert_partner_selection/DECISIONS_A0.md) § "BG1/BG2".

### Position/carrier record

The canonical index is [`results/phase-lattice/RESULTS_INDEX.md`](results/phase-lattice/RESULTS_INDEX.md).
It maps position spectra, carrier deletion, mediation, attention gates, and biological source assays to
their scripts, immutable JSON, and manifests.

### GTEx fine-tuning shift

The five published SpliceAI-GTEx fine-tunes contain a token-classification encoder/head, not an MLM head.
`scripts/splicebert_gtex_finetune_shift/prepare_gtex_mlm_transplant.py` strictly transplants each complete encoder into the
original masked LM and writes a derived Hugging Face checkpoint with an audit report. Run
`run_gtex_finetune_preflight.py` before `run_gtex_coupling_shift.py`; the latter requires all five seeds,
replays the exact base validation result before opening them, and stops at behavior. Exact hashes, readout
scope, commands, and the outcome-aware verification status are in
[`studies/splicebert_gtex_finetune_shift/`](studies/splicebert_gtex_finetune_shift/README.md).

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

### Positional boundary coherence — model-free execution only

**There is no successor model outcome to reproduce.** `studies/splicebert_positional_boundary_coherence/`
ships a frozen preregistration, 29 runner/audit entrypoints under
`scripts/splicebert_positional_boundary_coherence/`, eleven `src/gfmcd/splicing/pbc_*` and
`positional_boundary_coherence` modules, and their tests. The executed portion is model-free: compact
freshness/census audit receipts and the terminal capacity-stop/audit are retained, while the 2.34-GB raw
candidate census is intentionally ignored and must be regenerated from its bound producer manifest. There
is no checkpoint, preflight, target, or model-outcome artifact.

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

The terminal Phase-I capacity check precedes the four target bindings, and every edge is fail-closed:

Before expensive cohort matching, the balanced-cohort producer checks the necessary unique normalized-gene
capacity for all seven source slots against the total frozen request (120 + 600 + 384 = 1,104 blocks). A
failure writes a compact `PHASE1_INSTRUMENT_UNRESOLVED` artifact and manifest with empty block/cell tables,
then exits 2. The independent auditor rederives the capacity and exits 0 only when its own audit passes,
returning `AUDIT_PASS_INSTRUMENT_UNRESOLVED`; target authorization remains false. Both records bind the
ignored census's byte identity, semantic hashes, producer revision, and
`candidate_census_v1.manifest.json` regeneration receipt and state `clean_checkout_reexecutable=false`.
Do not reduce a role or open sealed chromosomes in response.

The recorded run found donor-phase-2 capacity `809/1,104` (deficit 295) and acceptor-phase-2 capacity
`854/1,104` (deficit 250); the independent audit passed all 48 checks with
`target_execution_authorized=false`. A clean checkout must first regenerate
`development/candidate_census_v1.json` from the exact command and pinned inputs in its retained manifest.
For byte-identical replay use producer revision `addd7be93c321373118b02b82bc3d95eeb0bd20a`; the retained
census byte SHA-256 is `9be356ac26ff18aa750b0552684d4a94fa6b8963fee01666608b38cc7de25614`.

1. **A conformance audit.** The preregistration's own "Present status" requires a freeze package binding
   the final code and inputs, plus an independent protocol-to-code conformance audit, before any target
   or Phase-III reserve opens. Do not run a target without it.
2. **Checkpoint identity.** Base `SpliceBERT.1024nt`, `pytorch_model.bin` sha256 `2ad91428…0149d`,
   recorded on every model run. A mismatch is fatal, not a warning.
3. **The seal.** `chr8`/`chr9` stay globally sealed; no script may fall back to them when capacity is
   low. `chrY` is excluded. Genes are excluded by the rederived freshness registry covering every
   C43/C44/GFS25 role.
4. **Producers cannot authorize themselves.** Every row above pairs a producer with a separate auditor,
   and no later stage may infer a positive, negative, or bounded result from a failed instrument.

Runners call `gfmcd.artifact_guard.guard_output_path`, so a rerun aimed at an existing artifact refuses
rather than overwriting. Results go to `results/splicebert_positional_boundary_coherence/` under the
layout its [`RESULTS_INDEX.md`](results/splicebert_positional_boundary_coherence/RESULTS_INDEX.md)
declares. The PBC-focused suite is
`uv run --extra splicebert --extra dev --extra maxent pytest tests/ -k pbc`; most tests need no genome or
checkpoint because they build tiny FASTA/GTF fixtures in `tmp_path`.
Drop `--extra maxent` and `tests/test_pbc_phase1_real_data.py` skips as a file.

## 4. Safe reproduction rules

1. Read the corresponding preregistration, amendments, and manifest before execution.
2. Run smoke/preflight gates before target data.
3. Do not reopen the sealed chr8/9 test as a routine smoke check.
4. Write a new immutable artifact; do not overwrite the controlling result.
5. Record checkpoint, tokenizer, dataset hashes, split, parameters, seed, command, and Git revision.
6. Re-run the validator and the narrowest relevant tests, then the full suite with the extras in §1
   when feasible.
7. Update the live claim ledger only to the strongest level the new evidence licenses; preserve any
   correction in `CORRECTIONS.md` or the appropriate decision log.
