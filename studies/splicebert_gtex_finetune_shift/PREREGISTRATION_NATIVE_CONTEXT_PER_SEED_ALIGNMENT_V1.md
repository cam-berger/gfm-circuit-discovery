# Preregistration: per-seed final-MLP codes and outcome-blind causal alignment

> **Tier: T2 — provenance.** Frozen before fitting a seed-specific basis or opening
> any aligned causal intervention. GFS18 and all outcomes in its discovery and
> confirmation artifacts are known. This follow-up deliberately reuses those
> already-opened gene partitions and is therefore exploratory/developmental rather
> than an independent confirmation.
>
> **Pre-outcome cohort-tag correction (2026-07-29):** the first discovery invocation
> stopped at the exact example-ID invariant before loading any checkpoint or
> collecting any outcome. The cohort namespace reproduced the gene partition, but the
> newly named cohort tag changed within-gene example sampling. The runners now reuse
> the exact GFS18 discovery and confirmation cohort tags, as Section 2 already
> required. No rank, intervention, threshold, or statistical family changed.

## 1. Question and evidence boundary

GFS18 rejected a single seed-2020 low-rank basis as a substantial five-seed mechanism
in raw coordinates. Its seed-stratified result nevertheless motivates a narrower
question: does each independently fine-tuned checkpoint implement the same compact
functional code in a checkpoint-specific rotation?

For each seed 2020--2024 and both established classifier-native sequence effects, this
study will:

1. fit a seed-specific final-MLP-write basis on discovery genes;
2. measure its causal effect on gene-disjoint evaluation genes;
3. measure raw cross-seed basis transport;
4. fit an outcome-blind orthogonal map between discovery representations and freeze it;
5. measure aligned cross-seed basis and instance-vector transport on evaluation genes.

The endpoints and counterfactuals are unchanged:

- donor margin at `donor_pos - 1`, with `[-80,-48)` relative to the donor shuffled;
- acceptor margin at `next_acceptor_pos`, with `[+16,+48)` relative to the acceptor
  shuffled.

The fitted maps never use classifier margins, labels, gradients, classifier weights,
or evaluation examples. An alignment is a model-coordinate relation, not a biological
feature.

## 2. Frozen cohorts and checkpoint family

The study reconstructs the two partitions from GFS18 exactly:

`SHA256("gtex-native-context-joint-lowrank-v1:" + gene_id) mod 100`.

- buckets `0--49`: discovery, `n=128`;
- buckets `50--99`: evaluation, `n=200`.

The original six prior-exposed result families remain excluded. Discovery example IDs
must exactly equal `native_context_lowrank_discovery_v1.json`; evaluation IDs must
exactly equal `native_context_joint_lowrank_confirmation_v1.json`. The partitions must
have zero gene overlap. Chromosomes 8/9 and 11/16/19 remain unopened.

All five released checkpoints (2020--2024) are required. There is no best-seed
selection. The discovery artifact and compressed arrays must be written immutably
before evaluation.

## 3. Preflight and stopping gates

Before interpretation:

1. the exact final-layer mechanism preflight and both 32-nt region licenses replay;
2. architecture is six layers, 16 heads, width 512, eager attention;
3. every seed/endpoint discovery behavior and the five-seed evaluation behavior have
   mean at least `0.05`, interval above zero, corrected `q<=0.05`, and positive sign in
   all applicable seeds;
4. all examples, gene IDs, corruptions, target coordinates, and arrays align exactly
   across checkpoints;
5. each fitted basis is row-orthonormal within `1e-8`;
6. every Procrustes map is orthogonal within `1e-8`;
7. an aligned basis remains row-orthonormal within `1e-8`;
8. rank-zero projection is an exact no-op and rank-512 projection reproduces the
   complete target delta within `1e-6`;
9. direct NumPy composition of final LayerNorm and classifier margin reproduces the
   captured native target margin within `1e-5`.

Failure freezes interpretation as instrument unresolved.

## 4. Discovery fits

For every seed and endpoint, examples are averaged within gene before fitting. Let

`X_s = mean_gene(MLP_write_clean - MLP_write_corrupt)`

for checkpoint `s`. Fit the uncentered SVD and freeze its first 64 right-singular
vectors. The causal rank grid is:

`r in {1, 2, 4, 8, 16, 32, 64}`.

For every ordered source/target checkpoint pair, fit the row-vector orthogonal
Procrustes map

`R_s->t = argmin_R ||X_s R - X_t||_F`, subject to `R^T R = I`.

The fitted source basis in target coordinates is `U_s R_s->t`. Discovery reports
singular-value energy, raw/aligned principal-angle cosines, relative alignment error,
and paired-row cosine. These are representation evidence only.

## 5. Evaluation interventions

At each target checkpoint, final-layer computation is held fixed. For target delta
`d_t`, causal subspace patching uses:

- **own:** projection onto target checkpoint basis `U_t`;
- **raw cross-seed:** projection onto source basis `U_s` without a map;
- **aligned cross-seed:** projection onto `U_s R_s->t`;
- **matched null:** projection onto each of three deterministic Haar-random target
  subspaces of equal rank.

Noising replaces the clean write by `m_clean - P d_t`; recovery replaces the corrupt
write by `m_corrupt + P d_t`. Effects are read through the target checkpoint's exact
LayerNorm and classifier.

Instance-vector transport is separate. On matched evaluation examples, inject:

- raw source delta `d_s`;
- aligned source delta `d_s R_s->t`;
- three deterministic signed-coordinate permutations of `d_s`, preserving vector
  norm and intervention size.

Noising subtracts the transported vector from the target clean write; recovery adds it
to the target corrupt write. Raw and signed-permutation controls can produce negative
effects; values are not clipped or rescaled.

## 6. Statistics and multiplicity

All intervals use 5,000 gene-grouped bootstrap draws. All inferential p-values use
9,999 exact add-one gene sign flips. Seed-family results average intervention values
within evaluation example across the relevant four off-diagonal source checkpoints
and then across the five target checkpoints; genes remain the resampling unit.

Families:

1. 28 own-basis effects: two endpoints x seven ranks x two directions;
2. 16 primary raw/aligned cross-seed basis effects at ranks 4 and 64;
3. 16 corresponding candidate-minus-random basis contrasts;
4. eight aligned-minus-raw basis contrasts;
5. eight raw/aligned instance-vector effects;
6. eight corresponding candidate-minus-signed-permutation contrasts;
7. four aligned-minus-raw instance-vector contrasts.

Each family receives Benjamini--Hochberg correction over exactly the enumerated tests.
All other 5x5 transport matrices, ranks, geometry, and seed-stratified values are
descriptive.

Causal effect ratios are ratios of gene means to the target checkpoint's complete
MLP-patch effect, with gene-grouped bootstrap intervals. They are not additive
mediation fractions.

## 7. Frozen decision rules

The code is **compact within checkpoints** if the first rank at or below 16 whose
damage and recovery ratio lower intervals both reach `0.50` is found. If no rank
through 64 reaches this threshold, the tested linear code is not compact.

A **rotated shared low-rank code** requires, at the same rank at or below 16:

1. own-checkpoint compactness;
2. aligned cross-seed damage and recovery effects positive, corrected `q<=0.05`, and
   ratio lower intervals at least `0.50`;
3. aligned-minus-raw and aligned-minus-random contrasts both have intervals above zero
   and corrected `q<=0.05`.

Aligned instance-vector transport is stronger evidence of example-conditioned
cross-checkpoint equivalence. It confirms only if both directions are positive with
corrected `q<=0.05`, exceed signed-permutation controls and raw transport with
intervals above zero and corrected `q<=0.05`, and reach ratio lower intervals `>=0.50`.

Interpretation:

| Outcome | Licensed interpretation |
|---|---|
| own compact; aligned basis succeeds | checkpoint-specific rotations of a shared compact subspace are causally adequate |
| aligned instance vectors also succeed | an outcome-blind map transports example-conditioned causal writes |
| own compact; alignment fails | compact but non-equivalent checkpoint-specific implementations |
| own rank 64 remains below 50% | the tested final-MLP change is intrinsically broader or nonlinear |

Because these partitions have known prior outcomes, every positive remains
exploratory and requires a new gene-disjoint replication before an established claim.
No outcome licenses a biological motif, a unique algorithm, or arbitrary
behavior-directed steering.
