# Research spine — SpliceBERT Head–Context Atlas

## The original question

Across biologically distinct, overlapping sequence contexts, **which SpliceBERT components are reused
generally, which are recruited selectively, and what computation does each validated component perform?**

The unit of the atlas is not a head name and not an attractive attribution graph. It is an
evidence-graded relation:

> component × biological context × model behavior × causal mechanism

The atlas spans all 96 attention heads and all six MLPs. A sequence may carry multiple biological tags;
the design must not force mutually exclusive labels onto overlapping splice architectures.

## Non-negotiable scope

1. **Contexts, not one favored perturbation.** The initial competition study is one atlas cell. It may
   generate hypotheses, but it does not organize the program.
2. **General versus context-associated use.** Compare single donor–acceptor pairs, donor clusters,
   acceptor clusters, multiple independent pairs, cassette/complex events, and matched non-site controls.
3. **Causal validation after screening.** A head becomes mechanistically interpretable only after exact
   group intervention plus QK, OV, MLP, or path/DAG validation as appropriate.
4. **Biological and model labels stay separate.** Annotation-derived tags describe the sequence;
   model-derived measurements describe the computation. Do not let one define the other.
5. **Discovery and display remain separate.** Experiments and claim ledgers live in
   `gfm-circuit-discovery`; only validated, provenance-pinned artifacts are exposed through
   `gfm-visualizer`.

## Minimum v1 context ontology

Tags overlap and should be represented as a multi-hot vector, with continuous covariates retained rather
than discretized when possible.

| Family | Required tags / covariates |
|---|---|
| Site architecture | `single_pair`, `donor_cluster`, `acceptor_cluster`, `multiple_independent_pairs`, `cassette_exon`, `complex_event` |
| Site role | `target_donor`, `target_acceptor`, `annotated_site`, `unannotated_candidate`, `matched_non_site` |
| Choice topology | number of same-type alternatives, local degree, nearest alternative distance, target–partner distance, strength gap |
| Genomic context | CDS/UTR/intronic, strand, exon/intron length, transcript support |
| Sequence context | local GC/PPT/k-mer summaries, frame/phase, motif strength, packet geometry |
| Controls | composition/distance/strength-matched decoy, shuffled-label control, MLM-health probes |

RNA-seq-dependent labels (`junction_observed`, `expressed_unannotated`, condition-specific cryptic
activation) remain a later evidence layer; annotation support must not be renamed as expression support.

## Atlas estimands

For component `h`, behavior `y`, and context tag `k`:

- **General-use score:** stable causal contribution to `y` across matched context families.
- **Context-association score:** contribution difference for `k` versus matched controls, adjusted for
  strength, distance, coding status, composition, and gene.
- **Selectivity profile:** the vector of context-association scores across all tags, with uncertainty.
- **Mechanism label:** QK routing, OV writing, MLP transformation, or mixed/path-mediated—assigned only
  after an intervention that identifies that distinction.

Screening ranks candidates; it does not name functions. Exact joint patches test distributed circuits;
single-head effects are never summed and called faithfulness.

## Evidence ladder

1. **Census:** multi-label cohort counts, common support, gene-disjoint split, projected uncertainty.
2. **Behavioral gate:** the context contrast changes a native MLM behavior without broadly damaging MLM
   health.
3. **All-component screen:** 96 heads + six MLPs, corrected donor and acceptor coordinate contracts.
4. **Matched association analysis:** hierarchical/gene-clustered model comparing context tags while
   controlling confounds; stability across independently seeded discovery draws.
5. **Mechanistic shortlist:** exact ranked-group, QK/OV, MLP/path interventions and representative
   measured attribution DAGs.
6. **Untouched confirmation:** freeze a small family before chr8/9; use max-T/FWER for selection claims.
7. **Visualization:** publish only the validated atlas table, examples, uncertainty, and DAG artifacts.

## Closed competition branch

The initial intron-mate alternative-site study is now a bounded exploratory atlas cell:

- The behavioral perturbation is **not competition-specific**: destroying a matched non-competitor moves
  the target at least as much as destroying the annotated competitor.
- On coordinate-valid donor targets, B (partner coupling) is distinct and concentrated: L3H1 alone
  reproduces ~84% of the signed effect in the 400-gene exact diagnostic; ten ranked heads reproduce ~99%.
- Donor C and D head-magnitude profiles are nearly identical (`r≈0.94`). A post-hoc top-three exact group
  shows a small C>D magnitude excess, but broader sets and the all-head behavioral contrast are
  inconclusive. This is a hypothesis-generating selectivity candidate, not a competition module.
- No individual MLP or all-MLP C>D contrast excludes zero. MLP and attention patches both mediate B at
  different causal cuts; their effects must not be added.
- Historical acceptor rows from `head_screen_v1` are invalid because the wrong nucleotide was masked.
  Acceptor-side head claims are withdrawn until a corrected screen is run.

The branch therefore contributes a positive-control circuit (donor partner coupling), a generic
same-type-site perturbation family, and a design lesson. It does **not** justify spending chr8/9 or more
Colab credits on competition confirmation.

## Immediate return path

The representation layer is complete and mechanically audited:

1. **Done:** multi-label annotation census and common-support accounting;
2. **Done:** corrected donor-G / acceptor-G sequence contracts and regression tests;
3. **Done:** gene-disjoint matched cohorts for ten role-valid contrasts across six overlapping families;
4. **Done:** discovery-only, all-96-head/all-six-MLP mechanical model preflight with genuinely masked
   non-site MLM-health probes; bulk runner reproduced all 2,448 reference quantities exactly;
5. **Done:** the hash-verified Colab execution completed 512 health and 11,807 target packets; immutable
   merge and independent analysis audit pass.
6. **Done:** the frozen general/context analysis identified four validation-ready general-use candidates
   and preserved 20 context-associated signals that fail the separate precision gate.
7. **Done:** the candidate package was instantiated before internal-validation outcomes were opened; all
   four general-use candidates validate in both target roles under the frozen familywise intervals.
8. **Done:** the A3-amended conditional analysis independently validates multi-layer downstream MLP
   recovery for all three general A-heads; L2MLP/B is material but its upstream route is instrument-limited.
9. **Done:** eight representative case/control DAG bundles independently audit with zero failures and are
   marked `export_eligible=true`; the exact payload now passes a source-hash-pinned, fail-closed import and
   production build in `gfm-visualizer` (HCA22).
10. **Next experiment:** improve power for the preserved cluster/cassette context hypotheses without
   loosening the estimand, controls, or precision gate.

The selected cohort is `atlas_context_matched_v7`; its independent audit reports zero failures. See
`ATLAS_COHORT_V1.md`. Thin internal-validation arms remain exploratory until outcome-specific MDEs are
known; lower sample count never relaxes the balance or coordinate gates.

The self-contained Colab bundle is under `colab_atlas_package/dist/` (ignored by git because it contains
the 75 MB checkpoint). It includes no validation/sealed packet files and requires no genome download.
The full matrix returns to this repository for analysis under `ANALYSIS_PLAN_V1.md` plus
`ANALYSIS_AMENDMENT_A1.md`; no internal-validation file is loaded until the discovery shortlist and exact
mechanism package are frozen.

## Drift check for every proposed experiment

Before execution, answer:

1. Which atlas cell does this populate?
2. What matched context makes “selective” identifiable?
3. What behavioral gate establishes that there is something to attribute?
4. What result would retire the hypothesis?
5. Does this require new compute, or can an existing artifact answer it?

If question 1 has no clear answer, defer the experiment. If one context begins replacing the atlas, return
to this document.
