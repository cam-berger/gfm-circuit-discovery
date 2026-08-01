# Reproducing the current scientific record

> **Tier: T1 — canonical.** Current, authoritative. Read this.

This is the canonical repository-level entry point. Study-specific commands and historical environment
snapshots remain in their manifests and reproduction maps, but new development should use the `uv`
environment below.

## 1. Environment

From the repository root:

```bash
uv sync --extra splicebert --extra biology --extra dev
uv run --extra splicebert --extra biology --extra dev pytest
```

**Pass the extras on the `uv run` too.** A bare `uv run pytest` re-syncs to the default dependency set
first, uninstalling `pytest` and `pyfaidx`, and silently reports a smaller suite. The extras form is what
`.github/workflows/tests.yml` executes. The baseline on this tree is **482 passed, 24 skipped** (it was
648/46 before the 2026-07-31 cleanup, which removed the DNABERT-2 suite along with that study, a dead
supervised-probe code island, and four memorial files whose artifact *and* runner had already been
deleted — so those tests could never fire again). All skips are data-gated or optional-dependency; none
is a failure. Treat the number as "the suite passes", not as a fixed constant.

Use `PYTHONHASHSEED=0` for cohort construction and any shuffle or selection path whose manifest declares
determinism. Use `TOKENIZERS_PARALLELISM=false` for model runs when deterministic, quiet tokenization is
required.

Some historical records name the original conda environment `gfm-thrust1`. Those strings are retained
as execution provenance, not as a second recommended setup. The canonical dependency specification is
now `pyproject.toml` plus `uv.lock`.

### 1.1 Two dependencies the default environment does not install

| Package | Needed by | How to get it |
|---|---|---|
| `maxentpy` 0.0.4 | MaxEntScan scoring: `run_a9_payload_semantics.py` (C30), `run_ic1_pilot2/3.py`, `run_ic2_feasibility.py`, `run_b2_payload_steering.py`, the atlas covariate builders, and `src/gfmcd/splicing/{specificity,splice_landscape}.py` | Not on PyPI. `uv pip install "maxentpy @ git+https://github.com/kepbod/maxentpy.git@b5e69aadd745fcd9ee65c9dd0ff21b902a63a825"` (the commit the recorded runs used). MaxEnt tables sha256 `8b6d7b46…b7c00`. |
| `multimolecule` 0.2.0 | the three cross-model carrier runners under `scripts/crossmodel/` (C39/C40) | A **separate** environment by design — the runners record `"environment": "gfmcd-mm (conda): python3.11 torch transformers multimolecule0.2.0"` in their manifests. Do not add it to the main env; build the side environment and run those scripts there. |

The library modules import `maxentpy` lazily, so `import gfmcd` works without it; the four runners above
import it at module level and will fail immediately if it is absent.

## 2. Model and external data

- **Model:** `SpliceBERT.1024nt`, post-LayerNorm BERT MLM, 6 layers × 16 heads, hidden size 512.
- **Checkpoint SHA-256:**
  `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`.
- **Primary genome/annotation:** hg38 plus the GENCODE files named and hashed in each result manifest.
- **Atlas annotation:** GENCODE v50 comprehensive, hash recorded in the Atlas census/package manifests.

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

**Two specs disagree and this is unresolved.** The config above is repo-relative and names
`GRCh38.primary_assembly.genome.fa`; 62 canonical manifests instead record the absolute
`/Users/cberg/Research/data/genomes/hg38.fa` plus `gencode.v50.basic.annotation.gtf.gz`, and 33 more
record a third file, `gencode.v50.annotation.gtf.gz` (comprehensive, not basic). The sequence content of
hg38 and the GENCODE primary assembly is equivalent for these purposes, but **the basic-vs-comprehensive
annotation split is not** — check which one a given manifest names before comparing to its artifact.

### 2.2 Four limits on exact reproduction — read before comparing numbers

1. **Code revision is not recoverable for almost every claim.** Of the 93 manifests referenced by
   canonical `results/MANIFEST.csv` rows, exactly **one** records a revision with a clean working tree
   (`results/phase-lattice/p3_phase_surgery_confirmation_v1/confirmation_analysis.manifest.json`, C36).
   66 record a commit with `git_dirty: true`, 21 record no revision at all, and 5 record a revision with
   no dirty flag. The comparability standard stated above is therefore **not attainable** for those 92;
   compare against the recorded numbers and manifests, not against a rebuilt binary-identical run.
2. **The `--out` defaults point at the record.** 56 runners default `--out` to a path that
   `results/MANIFEST.csv` lists as an artifact, and no runner checks whether the target already exists.
   Re-running any of them with defaults **overwrites the controlling artifact** — the opposite of the
   immutability rule stated in §4 and in `README.md`. `gfmcd.artifact_guard.guard_output_path` now exists
   for new runners to call, and `tests/test_overwrite_guard.py` keeps the inventory of the 56 from growing,
   but the existing runners are pinned by their manifests and were deliberately not edited (R35) — so
   **always pass an explicit `--out` to a new path when reproducing.**
3. **47 canonical result JSONs have no sibling `.manifest.json`.** The repo's own convention is that a
   result-producing run writes an immutable artifact *and* a manifest recording checkpoint, tokenizer,
   data hashes, split, parameters, seed, command, and revision. 47 of 429 canonical result JSONs predate
   or skipped that step — including `results/p3_carrier_locus.json` (C34), `results/p3_av_decomposition.json`
   (C35), and much of `results/phase-lattice/` (C8, C11, C13, C14, C15). Their `manifest` cell in
   `results/MANIFEST.csv` is empty, and `scripts/verify_manifest.py` passes because it only checks that
   *non-empty* reference paths resolve. For those claims the recorded numbers are the evidence and the
   execution environment is not recoverable. This is a disclosed gap in the record, not an oversight of
   this cleanup, and closing it would require re-running the affected experiments.
4. **Regenerable binary inputs are not committed.** Fitted dictionaries (`*.pt`) and activation captures
   (`*.npz`) were untracked on 2026-07-31 (847 MB; `.gitignore` had always excluded them and they had
   been force-added). No claim is controlled by one — every controlling artifact is the compact JSON
   twin, and the sibling manifests recording their SHA-256 are kept. But a **second-stage** GTEx runner
   that consumes a first-stage capture cannot run from a fresh clone until you regenerate it: notably
   `run_gtex_native_context_per_seed_alignment_evaluation.py` reads the discovery `.npz` named in
   `native_context_per_seed_alignment_discovery_v1.json` (GFS19), and the J-SAE evaluation stages read
   their capture files. Re-run the first-stage runner to rebuild the input, then compare against the
   recorded hash in the manifest.

Where a runner departed from its frozen protocol, the deviation is recorded in
[`docs/PREREG_CONFORMANCE.md`](docs/PREREG_CONFORMANCE.md) — read it before treating any preregistered
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

## 4. Safe reproduction rules

1. Read the corresponding preregistration, amendments, and manifest before execution.
2. Run smoke/preflight gates before target data.
3. Do not reopen the sealed chr8/9 test as a routine smoke check.
4. Write a new immutable artifact; do not overwrite the controlling result.
5. Record checkpoint, tokenizer, dataset hashes, split, parameters, seed, command, and Git revision.
6. Re-run the validator and the narrowest relevant tests, then `uv run pytest` when feasible.
7. Update the live claim ledger only to the strongest level the new evidence licenses; preserve any
   correction in `CORRECTIONS.md` or the appropriate decision log.
