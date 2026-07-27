# Atlas preregistration

> **Tier: T2 — provenance. DRAFT, NOT FROZEN, NOT RUN.**

The machine-readable lock candidate is
[`../../configs/dnabert2_atlas/preflight_v1.draft.json`](../../configs/dnabert2_atlas/preflight_v1.draft.json).
Its hash must be frozen by the study owner before a preflight report can set `frozen=true`.
No threshold below is confirmatory until that occurs.

## Question

Which declared biological site classes have positive reconstruction advantage over all required
controls, which target-preserving context regions mediate that advantage, and which model
components causally support it?

## Units and splits

- Unit of assignment and inference: gene/homology cluster.
- Development: 30% deterministic cluster hash.
- Local-frequency baseline training: 20%, cluster-disjoint and never used as target examples.
- Confirmation: 50%, sealed until behavior and instrument gates pass.
- Strand, site class, source release, and token geometry are mandatory strata/diagnostics.

## Preflight

Mandatory gates are positive control, negative control, sensitivity, resolution, coordinate and
split invariants, matching balance, token geometry, and model health. A report qualifies only when
all are `PASS`, its config hash matches exactly, it is marked frozen, and it used no confirmation
outcomes. It also binds the collection, control matches, and baseline-training corpus by SHA-256;
the target runner rejects any byte-different input.

The positive control is a graded, target-copy plant into a non-overlapping context region. This
tests whether the masked target score can detect context-provided sequence information; it is not a
biological model claim. Negative controls include sham exchanges and equal-length irrelevant
regions.

## Primary behavior

- Metric: simultaneous-mask true-token log probability per BPE-covered nucleotide.
- Empirical contrasts: same motif unannotated, composition/token matched, same-sequence non-site.
- Baselines: held-out nucleotide n-gram orders 0, 1, and 2.
- Multiple-testing family: every site class × every declared primary contrast in the census.
- P-values: add-one permutation values; Holm correction across the complete family.
- Estimation: cluster-first effects and cluster bootstrap intervals.

Site-class-specific SESOIs and minimum cluster counts remain deliberately unset in the draft config.
They require a development-only variance and positive-control calibration. The owner must either
freeze them before target effects or stop the class as underpowered.

The first bounded census slice is frozen before outcomes to GENCODE v50 basic protein-coding
genes on the primary chromosomes. It samples at most 512 gene clusters per site class per split,
with one target per gene × class, and evaluates splice donor, splice acceptor, start codon,
contiguous Kozak span, and stop codon supply. A class lacking all three empirical control
families is quarantined before model scoring. Promoter, branch-point/tract, poly(A), TF-binding,
and assay-supported sources remain deferred rather than being silently represented by motif calls.

## Context-source gate

Candidate region exchange must preserve target sequence, complete sequence length, target token
indices, target offset map, and token cover. Each biological region is tested against separately
matched irrelevant regions of equal nucleotide length. The full scanned region × site-class family
is corrected together.

## Internal and causal stages

Candidate set size, nomination union/intersection rule, ablation null draws, protected-health
thresholds, mediation SESOI, and faithfulness grid will be frozen in an amendment after behavioral
and context qualification but before internal confirmation. Reusing discovery outcomes to set those
thresholds without an amendment is prohibited.

## Possible outcomes

- `SUPPORTED: BEHAVIOR` — excess reconstruction passes and replicates; internal mechanism unknown.
- `SUPPORTED: CONTEXT` — a token-invariant biological region effect passes matched irrelevant-region
  controls; model component unknown.
- `SUPPORTED: NECESSITY` — a frozen component intervention exceeds matched nulls with protected
  health; direction/mediation unknown.
- `SUPPORTED: MEDIATION` or `DIRECTED CIRCUIT` — only after the corresponding interventions.
- `NULL` — a powered, qualified assay rejects the predeclared effect.
- `INSTRUMENT_UNRESOLVED` — controls, sensitivity, resolution, geometry, or invariants fail.
