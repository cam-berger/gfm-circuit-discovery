# SpliceBERT Head–Context Atlas

Across multiple overlapping biological sequence contexts, which SpliceBERT components are reused
generally, which are recruited selectively, and what computations do the validated components perform?
The goal is an evidence-graded map of *when and how* SpliceBERT reuses its circuitry—not a gallery of
graphs and not "head *h* is an acceptor-cluster neuron." The alternative-site competition study is one
bounded atlas cell, not the organizing objective. See **`RESEARCH_SPINE.md`**.

## Study identity
- **Version control:** study code, tests, notebooks, compact results, manifests, and audits are tracked;
  reproducible bulk shards/intermediates and Colab distribution archives are intentionally ignored.
- **Model/checkpoint:** `SpliceBERT.1024nt` (post-LN BERT-MLM, 6L×16H, d=512, P=1026 absolute), sha256 `2ad91428…0149d`.
- **Tokenizer / position:** single-nucleotide; absolute learned positions.
- **Objective / corpus:** masked-LM on pre-mRNA.
- **Biological computation:** native splice-site reconstruction across single-pair, clustered,
  multi-pair, cassette/complex, and matched-control contexts.
- **Model-native metric:** `donor_margin` / `acceptor_margin` (mask the core base; `logit(true) − max(others)`).
- **Primary dataset / sealed:** GENCODE v50 **comprehensive** (sha256 `83fba3e9…13a453`) + hg38.
  Discovery = autosomes ∉ {chr8,9,11,16,19}; **sealed = chr8/9**.
- **Compute:** local MPS for census/pilot/mechanical preflight; Colab CUDA credits only after a frozen,
  power-licensed wider atlas screen.

## Estimands
- **General use:** stable causal contribution across matched context families.
- **Context association:** component contribution in one biological context minus matched controls,
  adjusted for strength, distance, coding context, composition, packet geometry, and gene.
- **Mechanism:** QK routing, OV writing, MLP transformation, or mixed/path-mediated, named only after an
  identifying intervention.
- **Competition C:** retained as one secondary perturbation lens; it is not the atlas primary.

## Documents
- `ONTOLOGY.md` — frozen tag ontology (annotation-only v1; model-derived namespace kept separate).
- `ATLAS_COHORT_V1.md` — wider-atlas taxonomy, census, matching policy, selected cohort, and audit.
- `ANALYSIS_PLAN_V1.md` + `ANALYSIS_AMENDMENT_A1.md` — outcome-independent general/context estimators,
  multiplicity, stability, shortlist rules, and smoke-discovered mechanical clarifications.
- `MECHANISM_VALIDATION_PLAN_V1.md` + amendments A1/A2/A3 — pre-outcome exact group, whole-effect,
  conditional QK/V/MLP-path, untouched-validation, representative-DAG, and stopping rules.
- `colab_atlas_package/` — self-contained discovery runner, notebook, gates, validator, merge, and bundle builder.
- `RESEARCH_SPINE.md` — persistent objective, evidence ladder, drift guard, and immediate return path.
- `EXPERIMENTS.md` — prioritized working hypotheses and falsifiable follow-up tests, including a causal
  site-choice steering program from latent-direction discovery through biological validation.
- `results/splicebert_head_context_atlas/competition_branch_closure_audit_v1.json` — machine-readable,
  requirement-by-requirement closure evidence for the competition branch (indexed in `RESULTS.md`); the
  stage-gate status it closes is the table below.
- `PREREGISTRATION_C_COMPETITION.md` — confirmatory gates, controls, thresholds, stopping rules.
- `DECISIONS.md` — dated design decisions + the red-team pivot.
- `CLAIMS.md` — claim ledger (evidence level, status, scope, next bridge).
- `RESULTS.md` — index into immutable artifacts + manifests.
- `REDTEAM_SYNTHESIS_C_6lens.md` — the pre-compute adversarial design red-team (verdict + must-fixes).
  Its target was a pre-freeze design draft that is no longer in the tree; the must-fixes were folded into
  `PREREGISTRATION_C_COMPETITION.md`, which supersedes that draft as the design of record.

## Stage-gate status (2026-07-19)
| Gate | Status |
|---|---|
| Competition census + behavioral gate | **CLOSED** — behavior exists but is not competitor-specific. |
| Historical 96-head screen | **PARTIALLY VALID** — donor rows valid; acceptor rows withdrawn (coordinate defect). |
| Donor-only reanalysis + exact group/MLP closure | **DONE, exploratory** — B concentrated; C/D largely shared; no licensed competition module. |
| chr8/9 competition confirmation | **NOT LICENSED** — remains untouched. |
| Wider multi-context census + matching | **DONE, outcome-independent** — all ten role-valid contrasts balanced; independent audit 0 failures. |
| Wider all-component model preflight | **PASS** — 816 rows, every gate green; bulk path matches 2,448 reference quantities exactly. |
| Wider discovery screen | **COMPLETE + AUDITED** — 512 health packets and 11,807 target packets; every execution, merge, provenance, and split gate passes. |
| Discovery analysis/shortlist | **COMPLETE + AUDITED** — four general-use candidates advance; 20 context-associated signals are preserved but fail the frozen precision gate and do not advance. |
| Mechanism packaging + QK/V primitive | **COMPLETE + AUDITED** — candidate package sealed before internal-validation outcomes; 3-layer donor/acceptor QK/V/QKV/whole bridge errors are 0.0. |
| Whole/group validation + inference | **COMPLETE + AUDITED** — all four general-use candidates validate for both donor and acceptor reconstruction; labels remain component-level. |
| Conditional QK/OV/MLP/path inference | **COMPLETE + AUDITED UNDER A3** — all three A-heads have multi-layer downstream MLP recovery; L2MLP/B is material but its upstream attention route is instrument-limited by the empty frozen same-B head set. |
| Representative DAG + visualizer boundary | **COMPLETE + AUDITED** — eight case/negative-control bundles, explicit remainder mass, audited evidence overlays, and descriptive A interchange independently audit with zero failures; the source-hash-pinned `gfm-visualizer` import and production verifier also pass. |

A well-powered null or a diagnosed instrument limit is a valid result. No single context is allowed to
replace the atlas objective.

## Reproduce
```bash
uv sync --extra splicebert --extra dev
PYTHONHASHSEED=0 uv run python scripts/splicebert_head_context_atlas/build_census.py
PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false uv run python \
  scripts/splicebert_head_context_atlas/behavioral_pilot.py --n 400
PYTHONHASHSEED=0 uv run python scripts/splicebert_head_context_atlas/analyze_atlas_discovery.py --help
PYTHONHASHSEED=0 uv run python \
  scripts/splicebert_head_context_atlas/validate_atlas_discovery_analysis.py --help
```

See [`../../REPRODUCE.md`](../../REPRODUCE.md) for the repository-wide environment and artifact rules.

The zero-failure bundle is now exposed through `gfm-visualizer` without changing its evidence vocabulary
or claim boundary. Next, return to the atlas objective with a fresh, better-powered context study around
the preserved cluster/cassette hypotheses; do not relax the frozen precision rule. The site-choice
steering census may proceed in parallel without model outcomes. chr8/9 remains sealed and is not the
internal-validation split.

The Atlas does **not** by itself close the donor mechanism's cross-context generalization gap. Its matched
cohorts and validated components are the substrate for a separate causal transduction test: derive a
state/direction on discovery genes, inject it into gene-disjoint targets across context classes, and
require a dose-ordered native-margin change above matched-random and within-example controls. See
[`../../MECHANISM.md`](../../MECHANISM.md) and [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md).
