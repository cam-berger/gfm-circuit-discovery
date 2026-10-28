# SpliceBERT Head–Context Atlas

This study asks which SpliceBERT components are reused across overlapping sequence contexts, which are
recruited selectively, and what the validated components compute. The goal is an evidence-graded map of
*when and how* the model reuses its circuitry, not a gallery of graphs. On `SpliceBERT.1024nt` (sha256
`2ad91428…0149d`) we measured each component's effect on the masked splice-site margin (`donor_margin` or
`acceptor_margin`, `logit(true) − max(others)` at the masked core base) under zero-output (A) and
partner-scrub (B) interventions, across matched context families drawn from GENCODE v50 comprehensive.
Discovery and gene-disjoint internal validation are kept separate, and chr8/9 is sealed. Exact wording is
in the [claim ledger](CLAIMS.md).

## Findings

The alternative-site competition branch is closed: destroying a competitor moves the target margin, but a
strength- and distance-matched non-competitor produces an equal-or-larger effect (HCA4, HCA7). The
discovery screen of all 102 components yields four general-use candidates (`L2H15`, `L3H1`, `L1H12`, and
`L2MLP`) and 20 context-associated signals; the signals fail the frozen precision gate and were later
closed as under-powered leads (HCA18, HCA23). All four candidates validate as material causal relations
for both donor and acceptor reconstruction on the internal-validation cohort; these are component-level
relations, not biological function labels (HCA19). The three validated heads have specificity-controlled
downstream MLP recovery paths in both target roles, while the upstream attention route into `L2MLP` is
instrument-limited (HCA20). The representative-DAG bundles pass independent audit and are exposed through
`gfm-visualizer` without changing their evidence labels (HCA21, HCA22). HCA20 and HCA22 rest on artifacts outside this
repository with no retrievable URL or hash, so treat them as reported by the author until those are added
(see the reviewer note in the [ledger](CLAIMS.md)).

## Status

Open. The next step is a fresh, better-powered context study around the preserved cluster/cassette
hypotheses, without relaxing the frozen precision rule. The Atlas does not by itself close the donor
mechanism's cross-context generalization gap; that needs a separate causal transduction test (see
[`docs/MECHANISM.md`](../../docs/MECHANISM.md) and [`docs/ROADMAP.md`](../../docs/ROADMAP.md)).

## Files

- [`CLAIMS.md`](CLAIMS.md), [`RESULTS.md`](RESULTS.md), [`DECISIONS.md`](DECISIONS.md): claims, artifact
  index, and decisions. [`RESEARCH_SPINE.md`](RESEARCH_SPINE.md): objective and drift guard.
- [`ONTOLOGY.md`](ONTOLOGY.md) and [`ATLAS_COHORT_V1.md`](ATLAS_COHORT_V1.md): tag ontology and cohort.
- [`ANALYSIS_PLAN_V1.md`](ANALYSIS_PLAN_V1.md), [`ANALYSIS_AMENDMENT_A1.md`](ANALYSIS_AMENDMENT_A1.md),
  and [`MECHANISM_VALIDATION_PLAN_V1.md`](MECHANISM_VALIDATION_PLAN_V1.md) with amendments A1–A3: frozen
  analysis and validation rules.
- [`PREREGISTRATION_C_COMPETITION.md`](PREREGISTRATION_C_COMPETITION.md): competition-branch design of
  record. [`EXPERIMENTS.md`](EXPERIMENTS.md): working hypotheses and follow-up tests.
- [`colab_atlas_package/`](colab_atlas_package/): discovery runner and bundle builder.

Repository-wide environment rules are in [`docs/REPRODUCE.md`](../../docs/REPRODUCE.md).

```bash
uv sync --extra splicebert --extra dev
PYTHONHASHSEED=0 uv run python scripts/splicebert_head_context_atlas/build_census.py
PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false uv run python \
  scripts/splicebert_head_context_atlas/behavioral_pilot.py --n 400
PYTHONHASHSEED=0 uv run python scripts/splicebert_head_context_atlas/analyze_atlas_discovery.py --help
PYTHONHASHSEED=0 uv run python \
  scripts/splicebert_head_context_atlas/validate_atlas_discovery_analysis.py --help
```
