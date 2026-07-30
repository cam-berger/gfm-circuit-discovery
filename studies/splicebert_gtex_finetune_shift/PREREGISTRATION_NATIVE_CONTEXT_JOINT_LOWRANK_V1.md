# Preregistration: joint L5 routes, head→MLP mediation, and low-rank task structure

> **Tier: T2 — provenance.** Frozen before evaluating a joint-head, path-specific,
> or low-rank projection outcome. GFS14–GFS17 and their individual-head values are
> known. They fix the regions and head identities but do not determine a joint effect
> or representation direction.
>
> **Pre-outcome architecture correction (2026-07-29):** the released SpliceBERT
> configuration has hidden width 512, not 768. The first discovery invocation stopped
> at the architecture invariant before collecting or writing any outcome. Every
> occurrence below was corrected from 768 to 512; ranks, thresholds, splits, and
> statistical tests are unchanged.

## 1. Questions and evidence boundary

This package asks three linked questions about each confirmed 32-nt context effect:

1. **Joint structure:** how much behavior do the three frozen top-ranked L5 heads carry
   together, and are their effects redundant or complementary?
2. **Directed mediation:** does the joint head-set effect propagate through the final
   MLP write, rather than only through the residual stream around the MLP?
3. **Low-rank structure:** does a train-derived low-dimensional subspace of the final
   MLP-write change causally recover the classifier effect on fresh genes and across
   all five fine-tuning seeds?

The endpoints and input corruptions remain unchanged:

- donor margin at `donor_pos - 1`, with `[-80,-48)` relative to the donor shuffled;
- acceptor margin at `next_acceptor_pos`, with `[+16,+48)` relative to the acceptor
  shuffled.

The frozen head sets are:

- donor: `{L5H14, L5H8, L5H12}`;
- acceptor: `{L5H11, L5H14, L5H8}`.

These are top-three confirmation sets, not claims that every other head is null.

## 2. New outcome-independent cohorts

Every gene appearing in any prior train-stage GTEx mechanism/context result, including
both GFS17 census artifacts, is excluded. Remaining annotation-eligible
chromosome-training genes are assigned using

`SHA256("gtex-native-context-joint-lowrank-v1:" + gene_id) mod 100`.

- buckets `0–49`: direction/subspace discovery, seed 2020, `n=128`;
- buckets `50–99`: joint/path/subspace confirmation, all seeds 2020–2024, `n=200`.

Both partitions must have zero gene overlap with each other and every excluded
artifact. Chromosomes 8/9 and 11/16/19 remain unopened. “Fresh” means unseen by this
analysis program, not demonstrated absence from the checkpoint's training corpus.

The confirmation cohort remains closed until the discovery artifact has frozen the
directions and bases.

## 3. Instrument and behavior gates

The package requires:

1. the exact final-layer mechanism preflight to remain `PASS`;
2. both complete 32-nt context regions to remain licensed;
3. the GFS17 discovery and confirmation artifacts to replay the frozen head identities;
4. six layers, 16 heads, eager attention, exact target coordinates, and finite tensors;
5. each endpoint's clean-minus-corrupt behavior on its current cohort to have mean at
   least `0.05`, interval above zero, corrected `q≤0.05`, and positive sign in every
   checkpoint used at that stage;
6. every fitted basis to be orthonormal within `1e-8`, every direction to have unit
   norm within `1e-8`, a zero-rank projection to be an exact no-op, and a complete
   512-dimensional projection to reproduce the full clean-minus-corrupt MLP-write
   delta within `1e-6`.

Any failure stops interpretation. No direction, rank, or head set may be reselected.

## 4. Joint head-set intervention and matched controls

Every non-empty subset of the frozen three-head set is evaluated at the classified
target token: three singletons, three pairs, and the full triple. Noising patches the
corrupt head contexts into the clean run; denoising patches clean contexts into the
corrupt run.

The three already-frozen candidate-specific control triples from GFS17 are the
outcome-independent control blocks. For intervention size `k∈{1,2,3}`, the first `k`
heads in each ordered control triple define three same-size control sets. A candidate
subset is compared with the mean effect of the three control sets of the same size.
No intervention sizes are pooled.

All 28 endpoint × candidate-subset × direction tests form one BH family. Their 28
candidate-minus-matched-control contrasts form a second family. Every test uses 5,000
gene-grouped bootstrap draws and 9,999 exact add-one gene sign flips.

The full triple is a confirmed joint carrier only if both directions:

- have mean at least `0.05`, interval above zero, corrected `q≤0.05`, and positive
  estimates in all five seeds; and
- exceed the mean same-size controls with interval above zero and corrected `q≤0.05`.

Completeness is the ratio of gene means to the full behavioral effect. A complete-set
claim requires both damage and recovery ratio intervals to have lower bound at least
`0.80`; otherwise the measured ratio is a bound. Leave-one-out marginal changes and
`joint - sum(singletons)` are descriptive nonlinear interaction contrasts, not
additive mediation fractions.

## 5. Directed head→MLP mediator transplant

The full frozen head set and each of the three frozen size-three control sets enter the
path assay.

For noising:

1. patch corrupt heads into the clean final-layer run and capture the resulting target
   MLP write;
2. replay the unpatched clean run while clamping only the target MLP write to that
   head-induced counterfactual write.

The clean-margin loss in step 2 is the **MLP-mediated head-set damage**. Denoising is
the symmetric clean-head-into-corrupt construction followed by transplanting the
resulting MLP write into the unpatched corrupt run.

A controlled direct effect is also measured by applying the head patch while clamping
the MLP write to its baseline clean value (noising) or corrupt value (denoising).
Because residual connections and LayerNorm are nonlinear, total, mediated, and direct
effects are never required to add.

The four candidate mediated-direction tests form one BH family across endpoints. The
four candidate-minus-control contrasts form a second family. Directed mediation
confirms only if both directions are at least `0.05`, have intervals above zero,
corrected `q≤0.05`, are positive in 5/5 seeds, and exceed the three same-size control
paths with interval above zero and corrected `q≤0.05`.

## 6. Train-derived low-rank structure

### 6.1 Frozen discovery construction

For each endpoint on the fresh seed-2020 discovery cohort, record at the classified
target:

`Δm_i = MLP_write(clean_i) - MLP_write(corrupt_i)`.

Examples are averaged within gene before fitting so genes, not exons, determine the
geometry.

Two objects are frozen per endpoint:

1. **Uncentered SVD basis:** the first 64 right singular vectors of the gene × 512
   `Δm` matrix. The preregistered rank curve is `r∈{1,4,16,64}`.
2. **Task-covariance direction:** the unit vector
   `u_task ∝ Xᵀy`, where `X` is the gene-level `Δm` matrix and `y` is the corresponding
   gene-level clean-minus-corrupt classifier-margin effect. Its sign is oriented so
   `mean(X)·u_task > 0`.

No discovery causal projection is used to choose a rank or direction. The bases,
singular values, energy curve, task direction, and cohort identities are written to an
immutable artifact before confirmation.

### 6.2 Causal confirmation

For a frozen basis `U_r`, each confirmation example receives only the instance-specific
projected MLP-write change

`P_r Δm_i = U_rᵀ U_r Δm_i`.

- noising clamps the clean target MLP write to `m_clean - P_rΔm`;
- denoising clamps the corrupt target MLP write to `m_corrupt + P_rΔm`.

The task-covariance direction is tested identically as a separate rank-one structure.
The entire clean-minus-corrupt MLP write is the positive ceiling. Three deterministic
Haar-random subspaces of the same dimensionality are the matched nulls for each tested
structure. The random bases are generated from frozen endpoint/rank/draw tags and do
not inspect activations.

The 20 endpoint × structure × direction effects form one BH family; the 20
candidate-minus-mean-random contrasts form another. A structure confirms when both
directions have interval above zero, corrected `q≤0.05`, positive estimates in all
five seeds, and exceed the matched random mean with interval above zero and corrected
`q≤0.05`. The task direction additionally requires mean effect at least `0.05` in
both directions.

The minimal confirmed SVD rank reaching a ratio-to-full-MLP lower interval bound of
`0.50` in both directions is reported as the 50%-recovery rank. No untested
intermediate rank is interpolated.

## 7. Geometry and licensed outcomes

The following are descriptive unless paired with the causal gates above:

- cosine between donor and acceptor task directions;
- principal-angle cosines between donor and acceptor SVD subspaces;
- cosine between each frozen direction and each seed's exact classifier
  class-vs-no-site weight difference;
- singular-value energy concentration.

Licensed conclusions:

| Outcome | Licensed interpretation |
|---|---|
| joint triple confirms, low ratio | a structured causal core in a broader distributed computation |
| joint ratio lower bound ≥0.80 | the frozen triple is near-complete at one-head target-token resolution |
| mediated path confirms | the frozen head set causally changes classification through the final MLP write |
| task direction confirms | a train-derived one-dimensional MLP-write direction carries classifier-relevant context across fresh genes and seeds |
| low SVD rank reaches 50% | the final MLP context transformation contains a causally supported low-rank component |

None of these outcomes identifies a biological motif or spliceosomal mechanism. A
direction is a model-coordinate object, not a biological variable.
