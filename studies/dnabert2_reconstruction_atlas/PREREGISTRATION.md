# Atlas preregistration

> **Tier: T2 — provenance. V1 DEVELOPMENT COMPLETE; MULTISOURCE V4 FROZEN BEFORE
> MULTISOURCE MODEL OUTCOMES.**

The historical machine-readable lock candidate is
[`../../configs/dnabert2_atlas/preflight_v1.draft.json`](../../configs/dnabert2_atlas/preflight_v1.draft.json).
The controlling lock is
[`../../configs/dnabert2_atlas/preflight_v1.frozen.json`](../../configs/dnabert2_atlas/preflight_v1.frozen.json),
frozen from calibration v3 before any empirical target-control outcome. A preflight report may
set `frozen=true` only for that exact hash.

The superseded seven-nucleotide multisource lock is
[`../../configs/dnabert2_atlas/preflight_v2.frozen.json`](../../configs/dnabert2_atlas/preflight_v2.frozen.json)
with the corresponding
[`site_classes.v2.json`](../../configs/dnabert2_atlas/site_classes.v2.json) and
failed outcome-blind supply record
[`../../results/dnabert2_reconstruction_atlas/collection_qualification_v2.json`](../../results/dnabert2_reconstruction_atlas/collection_qualification_v2.json).
The five-nucleotide 128-candidate supply lock is
[`../../configs/dnabert2_atlas/preflight_v3.frozen.json`](../../configs/dnabert2_atlas/preflight_v3.frozen.json)
with [`site_classes.v3.json`](../../configs/dnabert2_atlas/site_classes.v3.json) and
[`SOURCE_MANIFEST_V2.json`](SOURCE_MANIFEST_V2.json). It inherits the already qualified
instrument thresholds unchanged. Its supply audit is
[`../../results/dnabert2_reconstruction_atlas/collection_qualification_v3.json`](../../results/dnabert2_reconstruction_atlas/collection_qualification_v3.json).
The controlling multisource lock is
[`../../configs/dnabert2_atlas/preflight_v4.frozen.json`](../../configs/dnabert2_atlas/preflight_v4.frozen.json).
It changes only the local candidate bound to 512 and requires a new v4 qualified preflight bound
to the exact v4 inputs.

The later context-margin amendment is controlled by
[`../../configs/dnabert2_atlas/context_margin_v1.frozen.json`](../../configs/dnabert2_atlas/context_margin_v1.frozen.json),
frozen from control-only calibration before any structured-versus-irrelevant target effect.
Its qualified preflight is
[`../../results/dnabert2_reconstruction_atlas/context_margin_v1_preflight_qualified_rerun.json`](../../results/dnabert2_reconstruction_atlas/context_margin_v1_preflight_qualified_rerun.json).
Splice acceptor and donor were frozen as instrument-unresolved before target scoring.

The promoter-specific classifier amendment is controlled by
[`../../configs/dnabert2_atlas/promoter_classifier_v2.frozen.json`](../../configs/dnabert2_atlas/promoter_classifier_v2.frozen.json).
It preserves the failed v1 preflight, keeps all biological SESOIs unchanged, and freezes the
replacement null/model-health instrument before any GENCODE target outcome. Its qualified report
is
[`../../results/dnabert2_reconstruction_atlas/promoter_classifier_preflight_v2.json`](../../results/dnabert2_reconstruction_atlas/promoter_classifier_preflight_v2.json).
The EPDnew cohort is excluded because its 300-nt windows are members of GUE.

## Question

Which declared biological site classes have positive reconstruction advantage over all required
controls, which target-preserving context regions mediate that advantage, and which model
components causally support it?

## Units and splits

- V1 unit of assignment and inference: gene/homology cluster.
- V2 unit of assignment and inference: shared 1-Mb genomic locus block across all source classes;
  complete target windows crossing block boundaries are excluded.
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
biological model claim. After calibration v1 exposed arbitrary-distance placement as a confound,
the plant region is the nearest downstream BPE-geometry-eligible region after a four-nucleotide
buffer from the scored cover. Negative controls include sham exchanges and equal-length irrelevant
regions.

## Primary behavior

- Metric: simultaneous-mask true-token log probability per BPE-covered nucleotide.
- Empirical contrasts: same motif unannotated, composition/token matched, same-sequence non-site.
- Baselines: held-out nucleotide n-gram orders 0, 1, and 2.
- Multiple-testing family: every site class × every declared primary contrast in the census.
- P-values: add-one permutation values; Holm correction across the complete family.
- Estimation: cluster-first effects and cluster bootstrap intervals.

The frozen primary SESOI is 0.02 nats per covered nucleotide. V1 requires 350 gene clusters per
target split; multisource v4 requires 250 locus blocks per target split from a maximum of 384.
Calibration-v3 thresholds are 0.05 for the four-copy positive
effect, 0.02 for irrelevant-exchange equivalence, 0.015 for the null-noise MDE, and 0.01
for the null-noise CI half-width.

The first bounded census slice is frozen before outcomes to GENCODE v50 basic protein-coding
genes on the primary chromosomes. It samples at most 512 gene clusters per site class per split,
with one target per gene × class, and evaluates splice donor, splice acceptor, start codon,
contiguous Kozak span, and stop codon supply. A class lacking all three empirical control
families is quarantined before model scoring. Promoter, branch-point/tract, poly(A), TF-binding,
and assay-supported sources remain deferred rather than being silently represented by motif calls.

Multisource v4 admits 13 classes: the four v1 classes plus EPDnew TSS-centered promoter cores,
PolyASite signal hexamers and representative cleavage nucleotides, CTCF/REST/CEBPB ENCODE ChIP
five-nucleotide peak cores, Mercer experimentally mapped branch nucleotides, GENCODE-derived contiguous
polypyrimidine tracts, and ENCODE Registry v4 distal enhancer-like cCRE cores. PolyASite entries
must be terminal-exon annotated, observed in at least 10% of samples, supported by at least two
protocols, and carry a declared canonical signal 5–60 nt upstream. TF clusters require score ≥500
and at least two contributing experiments. Mercer hg19 calls require unique hg38 liftover, exact
reverse round trip, and source-base agreement.

All v4 classes are split together by locus, sampled at 384 blocks per class per split, and searched
at 512 outcome-blind candidate positions per window prioritized by target-word nucleotide
composition, plus exhaustive target-word occurrences.
Confirmation remains sealed for every class that does not qualify in development.

## Context-source gate

Candidate region exchange must preserve target sequence, complete sequence length, target token
indices, target offset map, and token cover. Each biological region is tested against separately
matched irrelevant regions of equal nucleotide length. The full scanned region × site-class family
is corrected together.

Context-margin v1 instantiates this with a fixed per-class proximal span and a distal span on the
same side or sides. Both are mononucleotide-composition preserving and exactly matched for changed
base count. The metric is the true score minus log-sum-exp over the common fair Hamming-one
alternative family, divided by covered nucleotides. The primary contrast is distal-corruption
margin minus proximal-corruption margin. The frozen SESOI is +0.01; each split also requires a
positive cluster-bootstrap lower bound and Holm-adjusted add-one `p≤0.05`. Because the distal null
is not distance-matched, the licensed claim is proximal-context dependence, not a resolved
biological motif or cellular mechanism.

### Promoter-classifier amendment

The selected checkpoint is the human GUE `prom_300_all` two-class head. Its model-native
measurement is promoter logit minus nonpromoter logit on the same 300-nt input, so alternative
BPE segmentation is not a fairness problem. Targets are GENCODE v50 protein-coding MANE Select
TSSs with exact forward/reverse membership in every GUE split excluded. Controls are nearby
pseudo-TSS windows with no GENCODE transcript TSS within 300 nt, chosen before outcomes by
GC/CpG balance. Shared 1-Mb locus blocks define 384 development and 384 confirmation examples.

The clean endpoint is true-TSS margin minus pseudo-TSS margin, with SESOI +0.50. The context
endpoint is the true-TSS clean-minus-shuffle loss minus the pseudo-TSS clean-minus-shuffle loss,
with SESOI +0.25. Shuffles preserve the five-nucleotide anchor, input length, intervened-span
mononucleotide composition, and exact target/control Hamming dose. Both endpoints must have
positive 95% block-bootstrap lower bounds and pass Holm-adjusted add-one sign inference in each
split. Confirmation opens only if both development endpoints qualify.

## Internal and causal stages

Candidate set size, nomination union/intersection rule, ablation null draws, protected-health
thresholds, mediation SESOI, and faithfulness grid will be frozen in an amendment after behavioral
and context qualification but before internal confirmation. Reusing discovery outcomes to set those
thresholds without an amendment is prohibited.

### Promoter clean-behavior head atlas v1 — frozen amendment

This amendment localizes only D4's replicated clean supervised promoter contrast. H6's failed
target-preserving context effect stays failed and is not an entry gate or an outcome here.

- Cohort: 192 outcome-blind SHA-256-ranked, unique 1-Mb locus blocks per split from the existing
  exact-GUE-excluded GENCODE MANE TSS/pseudo-TSS cohort.
- Node: one head's context vector before the shared attention output projection, at the unpadded
  `[CLS]` row that feeds the classifier pooler.
- Candidate measurements: target-from-control patch loss, control-from-target patch gain, their
  corresponding first-order attributions, and native `[CLS]` attention mass over five frozen
  TSS-relative nucleotide bands.
- Selection: require positive development means in all four attribution/exact directions; select
  the eight best mean-fractional-rank heads, capped at two per layer.
- Exact validation: patch the eight heads jointly in each direction and compare against 63
  distinct candidate-excluding random sets with the identical layer profile.
- Primary gates: necessity and sufficiency each must be at least +0.60 logits, have a positive
  95% locus-block bootstrap lower bound, and pass Holm-adjusted add-one random-set inference at
  familywise `α=0.05`.
- Confirmation: opens only after a qualified GUE-only head preflight and a development pass. It
  reuses the exact eight development heads and unchanged thresholds.
- Individual heads: confirmation estimates and sign tests are descriptive candidate evidence;
  the group matched-null result does not silently promote each member.

The calibration and preflight GUE slices are disjoint. Their role is instrument qualification,
not evidence about the GENCODE target cohort. The complete machine-readable lock is
`configs/dnabert2_atlas/promoter_head_atlas_v1.frozen.json`.

## Possible outcomes

- `SUPPORTED: BEHAVIOR` — excess reconstruction passes and replicates; internal mechanism unknown.
- `SUPPORTED: CONTEXT` — a token-invariant biological region effect passes matched irrelevant-region
  controls; model component unknown.
- `SUPPORTED: NECESSITY` — a frozen component intervention exceeds matched nulls with protected
  health; direction/mediation unknown.
- `SUPPORTED: MEDIATION` or `DIRECTED CIRCUIT` — only after the corresponding interventions.
- `NULL` — a powered, qualified assay rejects the predeclared effect.
- `INSTRUMENT_UNRESOLVED` — controls, sensitivity, resolution, geometry, or invariants fail.
