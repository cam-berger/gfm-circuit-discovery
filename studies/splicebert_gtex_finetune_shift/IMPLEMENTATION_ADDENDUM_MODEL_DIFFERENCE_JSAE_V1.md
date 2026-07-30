# Implementation addendum: base-to-tuned model-difference J-SAE v1

**Status:** frozen before fitting the model-difference dictionaries
**Date frozen:** 2026-07-30
**Parent evidence:** GFS9, GFS18--GFS21, and GFS26--GFS33 in `CLAIMS.md`
**Method references:** Farnik et al. (2025) J-SAE; model-difference
decomposition and signed shared-support dictionaries are project-specific

## Question and estimand

This assay asks:

> Can a sparse feature graph explain only the layer-5 computation that changed
> between base SpliceBERT and each GTEx-fine-tuned checkpoint?

It does not reconstruct the base or tuned state.  For a matched sequence state,
let:

- `x0` be the base layer-5 MLP input;
- `x1` be the tuned layer-5 MLP input;
- `f0` be the base final MLP;
- `f1` be the tuned final MLP;
- `y_ab = f_b(x_a)`.

The input target is the zero-anchored representation shift:

`delta_x = x1 - x0`.

The native MLP-write shift is:

`delta_y = y_11 - y_00`.

Its exact two-player Shapley decomposition is:

`phi_state = 0.5 * ((y_10 - y_00) + (y_11 - y_01))`

`phi_parameter = 0.5 * ((y_01 - y_00) + (y_11 - y_10))`.

Therefore `phi_state + phi_parameter = delta_y` to floating-point precision.
“State” means the representation delivered to the final MLP by upstream
layers; “parameter” means the final MLP parameters.  Both changed during
fine-tuning.  The allocation is an exact factorial description, not a claim
that the two players vary independently in nature.

Also report the ordered tuned-path decomposition:

`state_tuned_path = y_11 - y_01`

`parameter_tuned_path = y_01 - y_00`,

the reverse base-path decomposition, and the interaction
`y_11 - y_10 - y_01 + y_00`.  These are sensitivity diagnostics; the Shapley
allocation is primary.

## Local computation shift

For state `a` and parameter set `b`, define:

`A_ab = d f_b(x) / d x | x=x_a`.

The native local operator shift is `delta_A = A_11 - A_00`.  Its exact
Shapley blocks are:

`Phi_A_state = 0.5 * ((A_10 - A_00) + (A_11 - A_01))`

`Phi_A_parameter = 0.5 * ((A_01 - A_00) + (A_11 - A_10))`.

After projection through active input decoder directions and output encoder
directions, the corresponding feature graphs are `Phi_J_state`,
`Phi_J_parameter`, and:

`delta_J = Phi_J_state + Phi_J_parameter = J_11 - J_00`.

The graph penalty uses the per-edge group magnitude
`sqrt((Phi_J_state**2 + Phi_J_parameter**2) / 2)`.  This prevents opposite
state and parameter shifts from cancelling into false sparsity.

The finite write shift and local derivative shift are separate objects.
Neither is described as the integral of the other.

## Architecture

Fit one independent base-to-tuned instrument per tuned checkpoint.  There is
no standalone base dictionary, six-view objective, shared/private block, or
cross-seed feature identity constraint.

The frozen architecture is:

- native input/output width: 512;
- final-MLP intermediate width: 2,048;
- input difference features: 512;
- output difference features: 128;
- signed TopK-16 activity at both sites;
- no encoder bias;
- no decoder bias;
- unit-norm decoder rows;
- no AuxK;
- three initializations per tuned checkpoint;
- fixed 40 epochs with no early stopping.

Signed TopK selects the largest absolute linear preactivations and retains
their signs.  Zero input therefore reconstructs as exact zero.

The output state and parameter blocks share one encoder and decoder.  Their
joint support is selected by
`sqrt(code_state**2 + code_parameter**2)`, after which the two signed
amplitudes are decoded separately.  Thus at most 16 output directions jointly
explain both contributors to the write shift.

All normalization is zero anchored.  Training-only scalar RMS scales may
change units, but no mean is removed: the mean base-to-tuned displacement is
part of the estimand.

## Frozen data and weighting

Use only the hash-pinned label-free activation package from GFS21/GFS26--GFS33:

- activation report SHA-256
  `142ce4ee5976970b9b296685ef403cce30e56f440a87d7222a1933fe09d96b5b`;
- activation arrays SHA-256
  `fa6c972f4765d5c57354abeb262b3e62d6ac3ff8c69e5850a1511c7cf054d559`;
- 7,991 states;
- 6,530 training states from 105 genes;
- 1,461 validation states from 23 disjoint genes;
- 6,320 training and 1,415 validation reference/changed pairs.

Every fit loads the base plus exactly one tuned checkpoint.  Exact weight
hashes are inherited from GFS33 and must be checked before construction.

Ordinary-state exposure is balanced hierarchically by:

1. endpoint;
2. state family (`clean`, `target_shuffle`, `sham_shuffle`, `max_recode`,
   `synonymous_single`);
3. gene;
4. state within gene.

Reference/changed exposure is balanced by endpoint, pair family, gene, then
pair.  All powers used to normalize losses and all omnibus held-out metrics use
the same hierarchy.  No per-example normalization is permitted.

Per-state graph summaries use explicit hierarchy weights
`1 / (n_strata * n_genes_in_stratum * n_states_in_gene)`.  E50/E90/E99 use
weighted quantiles; effective edge count, graph RMS, sign prevalence, and the
saved aggregate graph use the corresponding weighted distributions.  Raw-row
medians may be reported only as non-gating diagnostics.

## Loss and matched graph control

For any zero-anchored target `q`, fidelity is scored with fraction of shift
energy explained:

`FSEE = 1 - E_Q ||q - q_hat||^2 / E_Q ||q||^2`.

Centered FVE and cosine are reported as diagnostics only.

The level loss is the equal mean of power-normalized reconstruction errors for:

- `delta_x`;
- `phi_state`;
- `phi_parameter`;
- `delta_y`.

The contrast loss applies the same four terms to the within-sequence
difference of differences:

`q_changed - q_reference`.

For each active input decoder direction, the native state, parameter, and
total Shapley operator actions are compared with the actions obtained by
decoding the corresponding feature graph through the output decoder.  The
three operator MSEs are normalized by the larger of their pooled mini-batch
native power and 5% of pooled total-operator power, then averaged as
`L_operator`.  This is a block-level resolution floor, never per-example
normalization.

The primary condition is:

`L_level + L_contrast + L_operator + 0.25 * L_delta_J`.

Its matched control is:

`L_level + L_contrast + L_operator`.

Both conditions start from a byte-identical initialization and use identical
ordinary and paired schedules.  Adam uses betas `(0.9, 0.999)`, peak learning
rate `5e-4`, 1% warmup, 20% terminal decay, 5% graph warmup, and gradient norm
cap 1.  Each step contains 64 ordinary states and 32 complete pairs.  There
are 103 steps per epoch.

## Preflight

Before the first real fit, the runner must pass a deterministic planted
instrument with:

- exact four-cell write and Jacobian Shapley closure;
- analytic-versus-autograd agreement;
- recovery of a known aligned model-difference edge's location, sign, and
  decoded native operator action;
- amplitude checks immediately below and above the registered graph-resolution
  floor;
- an identical-model zero-difference negative control;
- a dense planted operator that must not appear one-edge sparse;
- decoder-direction finite-difference recovery at the registered dose;
- exact zero-shift reconstruction;
- negative signed activation, common state/parameter output support, and
  decoder-norm invariants.

On the real artifacts, before optimization:

- base and tuned native MLP replay error must be `<=5e-4`;
- all four cross cells must be finite;
- float64 write-Shapley closure must be `<=1e-10`;
- float64 Jacobian-Shapley closure must be `<=1e-10` on the checked rows;
- base/tuned state ordering, splits, counts, and hashes must match;
- no classifier parameter, class label, margin, prior feature ranking, or
  biological outcome may be loaded.

An anomalous preflight stops before training.

## Frozen representation gates

The seed-2020 initialization-0 primary and matched control form the pilot.  If
the pilot fails, preserve it and stop the other seeds and initializations.

Every primary fit must satisfy:

- input `delta_x` validation FSEE `>=0.90`;
- `phi_state`, `phi_parameter`, and total `delta_y` validation FSEE `>=0.95`;
- input contrast FSEE `>=0.85` in every endpoint/family cell;
- each output-block and total contrast FSEE `>=0.90` in every
  endpoint/family cell;
- reconstructed-`delta_x` four-cell cross-replay FSEE `>=0.90` for state,
  parameter, and total write blocks;
- decoded total operator-action FSEE `>=0.90`;
- decoded state and parameter operator-action FSEE `>=0.85` when the block
  carries at least 5% of total native operator power; a lower-power block
  instead passes only when its error power is at most 5% of total native
  operator power and is reported below resolution;
- no level or cross-replay fidelity more than `0.03` below the matched
  no-Jacobian condition;
- no operator fidelity more than `0.03` below the matched no-Jacobian
  condition for a resolved block, with the same absolute-error rule for an
  unresolved block;
- exact signed L0 of 16 on nonzero held-out examples at both sites;
- training dead-feature fraction `<=0.50` at both sites;
- decoder norm error `<=1e-5`;
- analytic-versus-autograd relative Frobenius error `<=1e-5`;
- for each resolved state, parameter, and total graph, decoder-direction
  finite-difference correlation `>=0.95`, eligible-edge sign agreement
  `>=0.95`, and relative Frobenius error `<=0.10`; a block whose analytic RMS
  is below `1e-5` instead requires absolute RMS error `<=1e-5` and observed
  RMS `<=2e-5`, and is reported below resolution;
- median state/parameter group-graph E90 at least 15% below the matched
  control;
- median effective state/parameter group-edge count at least 10% below the
  matched control;
- median normalized group-graph RMS above the planted `1e-5` resolution
  floor.
- median normalized signed-total-`delta_J` RMS above the same planted `1e-5`
  resolution floor.

Across each checkpoint, all three initialization pairs must pass
sign-aligned reciprocal feature matching at input and output:

- validation activation-density floor `0.001`;
- reciprocal fraction `>=0.25`;
- mean reciprocal score `>=0.50`.

Matching uses the frozen hierarchy weights.  A single common `+/-` orientation
must align both signed activation correlation and decoder cosine; their
absolute values may not be optimized independently.

Every initialization of all five tuned checkpoints must pass before feature
identity, classifier behavior, or biology is opened.

## Stopping and claim boundary

Checkpoint order is seed 2020, then 2021--2024.  Conditions and
initializations run sequentially; only one fit occupies the accelerator at a
time.

Execute the seed-2020 initialization-0 pilot first with `--pilot-only`.  Its
result, manifest, sparse weights, and validation fingerprints are written
before any other fit.  A passing pilot may be supplied to the unchanged runner
with both resume paths; resume requires exact activation, checkpoint, runner,
core model-difference implementation, protocol, and pilot-model hashes,
replays the preflights, and skips only that completed replicate.  Executed and
merely planned initialization seeds are recorded separately.  This prevents a
later interruption from erasing the decisive pilot or falsely claiming that
unrun fits executed.

A complete pass licenses only:

> Across five independently fine-tuned checkpoints, a zero-anchored signed
> sparse coordinate system preserves the held-out base-to-tuned layer-5
> representation shift and the exact state-versus-final-MLP-parameter
> contributions to the write and local Jacobian shift, while concentrating
> the projected derivative-change graph relative to matched no-penalty
> controls.

It does not establish causal use by the native classifier, identical features
across seeds, acquired/amplified/rewritten biological features, a complete
circuit, monosemanticity, or steering.  Those require a new analysis-sealed
cohort, exact feature transplants, matched feature/edge nulls, and
target-versus-sham confirmation.
