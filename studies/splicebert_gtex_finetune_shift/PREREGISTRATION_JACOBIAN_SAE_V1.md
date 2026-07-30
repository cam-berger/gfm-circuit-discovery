# Preregistration: layer-5 Jacobian sparse autoencoder instrument v1

**Status:** frozen before any J-SAE feature/task association or intervention result  
**Date frozen:** 2026-07-30  
**Evidence target:** instrument first; candidate generation only after every gate passes

## 1. Question and licensed scope

Can the fine-tuned SpliceBERT layer-5 MLP be represented by a sparse graph from
input features to output features while preserving the native splice-classification
computation?

This is a new representation instrument motivated by Jacobian sparse autoencoders
(J-SAEs), not a continuation of the failed GFS21 shared-index transcoder. The primary
object is one checkpoint-specific pair of TopK SAEs around the frozen layer-5 MLP:

`x -- input SAE -- input features -- native MLP Jacobian -- output features -- output SAE`.

The J-SAE may generate hypotheses about feature identity and feature-to-feature edges.
It cannot establish biological meaning, necessity, mediation, or monosemanticity
without the frozen interventions in Sections 7–9.

References:

- Farnik et al., *Jacobian Sparse Autoencoders: Sparsify Computations, Not Just
  Activations*, ICML 2025:
  <https://proceedings.mlr.press/v267/farnik25a.html>
- Authors' MIT-licensed implementation:
  <https://github.com/lucyfarnik/jacobian-saes>
- Gao et al., *Scaling and evaluating sparse autoencoders* (TopK/BatchTopK context):
  <https://arxiv.org/abs/2406.04093>
- Marks et al., *Sparse feature circuits*:
  <https://arxiv.org/abs/2403.19647>

## 2. Why this differs from GFS21

GFS21 forced a common feature index across the base checkpoint and five independently
fine-tuned checkpoints. All fine-tuned shared/tuned blocks collapsed while private
blocks reconstructed the views. That result rejects that parameterization, not sparse
computation.

The present instrument:

1. fits each fine-tuned checkpoint independently;
2. gives the MLP input and output separate dictionaries;
3. sparsifies the active feature-to-feature Jacobian explicitly;
4. treats post-hoc cross-seed matching as validation, never as a fitted objective;
5. uses a capacity- and TopK-matched paired-SAE control with zero Jacobian penalty.

No claim may call a feature acquired, shared, seed-convergent, biological, or causal
from reconstruction or graph sparsity alone.

## 3. Frozen data boundary

Instrument development uses only the existing label-free activation artifact:

`results/splicebert_gtex_finetune_shift/sparse_transcoder_activation_capture_v1.json`

and its hash-bound NPZ. It contains 7,991 target-token states from 128 discovery genes:
clean sequences, exact target/sham shuffles, synonymous single-codon states, and two
maximum synonymous-recoding draws. Training may read the ordered layer-5 MLP input and
exact MLP write. It may not read a classifier weight, class label, logit, margin,
feature attribution, GFS18/GFS20 ranking, or external assay outcome.

Use the already frozen gene split:

- namespace `gtex-native-biological-features-transcoder-train-v1`;
- validation gene if its SHA256 bucket is `<20`;
- all other discovery genes train;
- every state from a gene stays in one split.

The training-gene mean and one scalar RMS standard deviation are fit independently
for the MLP input and write. Scalar rather than coordinate-wise scaling preserves
native directions and makes the Jacobian units explicit.

## 4. Frozen mathematical object

For checkpoint `s`, let the exact layer-5 MLP be

`f_s(x) = W_out GELU(W_in x + b_in) + b_out`.

Both SAEs have width 512, TopK activity 8, nonnegative activations, unit-norm decoder
directions, and independent encoder/decoder weights. The active feature Jacobian is
computed exactly, not estimated:

`J_active = E_out(active) W_out diag(GELU'(W_in x+b_in))
            W_in D_in(active)`.

The derivative is evaluated at the **native MLP input `x`**, matching the released
J-SAE code. This is distinct from evaluating at the input-SAE reconstruction and is
reported as a frozen implementation choice. Only the active `8 x 8` block is
materialized.

The primary training loss is:

`MSE(x_norm, x_hat_norm) + MSE(m_norm, m_hat_norm)
 + lambda_J * mean(|J|)`.

This matches the paper's primary L1 objective. Because raw thresholds and raw L1 can
be scale-confounded, model selection is based on scale-normalized edge-mass counts
and is always matched to a zero-penalty fit with the same reconstruction budget.
L1/RMS is reported as a required sensitivity metric.

## 5. Instrument-development scan

The only development checkpoint is fine-tuned seed 2020. The only architecture is
`width=512, TopK=8`. Use one matched initialization and scan:

`lambda_J in {0, 0.015625, 0.0625, 0.25}`.

The nonzero values are quarter, nominal, and fourfold versions of the authors'
`0.5*k^2/d_resid` heuristic for `k=8`, `d_resid=512`. The zero condition is the
primary paired-SAE negative control. Frozen optimization:

- seed `1,512,008`;
- Adam, betas `(0.9,0.999)`, peak learning rate `5e-4`;
- batch 128;
- maximum 40 epochs;
- early-stopping patience 6;
- minimum validation reconstruction improvement `1e-5`;
- gradient-norm clipping 1.0;
- 1% linear learning-rate warm-up and 20% terminal linear decay;
- 5% linear Jacobian-coefficient warm-up;
- inverse-gene-state sampling so every training gene has equal expected weight;
- checkpoint by lowest validation input-plus-output normalized MSE.

Selection is lexical among nonzero coefficients that pass every representation gate,
then maximum relative reduction in median held-out `E90` edge count. Ties use higher
output FVE, then smaller coefficient. No endpoint, variant family, sequence motif, or
downstream classifier result enters selection.

The selected coefficient is frozen before the five-seed replication or any
task-feature association is opened.

## 6. Frozen instrument gates

### 6.1 Mechanical and positive controls

1. Replaying the frozen MLP on every captured input matches the captured write with
   maximum absolute error `<=5e-4`.
2. Analytic active Jacobians match autograd on 16 held-out rows with maximum absolute
   error `<=2e-5` and maximum relative error `<=2e-4`.
3. Every output is finite; train and validation genes are disjoint; every decoder
   direction has norm `1 +/- 1e-5`; observed L0 is at most 8.
4. At most 50% of features are dead on training states for either SAE.

### 6.2 Representation and sensitivity gates

On held-out validation genes, a nonzero condition passes only if:

- input FVE `>=0.85`;
- output-write FVE `>=0.95`;
- native MLP applied to the reconstructed input has output FVE `>=0.90`;
- each of those three FVEs is no more than `0.03` below its matched zero-penalty
  baseline;
- median active-Jacobian `E90` is at least 15% smaller than baseline;
- median `E90` is non-increasing from `lambda=0` through the candidate coefficient.

`E90` is the minimum number of active edges explaining 90% of squared active-Jacobian
mass. E50 and E99, mean absolute Jacobian, L1/RMS, feature density, and dead-feature
fractions are reported diagnostics, not alternative gates.

If no nonzero coefficient passes, the artifact status is
`INSTRUMENT_UNRESOLVED`; task associations and interventions remain unopened.
Thresholds are not relaxed.

## 7. Five-seed representation replication

After a seed-2020 coefficient passes and is frozen:

1. fit three initializations independently for each tuned seed 2020–2024;
2. require every fit to pass the same mechanical and representation floors;
3. compare each to its initialization-matched zero-penalty paired SAE;
4. require the median E90 improvement to have the frozen sign in all five seeds;
5. post-hoc match features using held-out activation correlation plus decoder cosine;
6. estimate graph-level stability by reciprocal matched nodes and edges.

Exact raw feature identity is not required across seeds. A cross-seed feature family
requires reciprocal matching in at least four of five seeds and must later reproduce
its sequence association and causal sign. Failure of matching licenses
checkpoint-specific sparse graphs only.

## 8. Frozen downstream behavioral preflight

Dictionary weights and coefficient selection are immutable before this stage.
Use held-out dictionary-validation genes and the native three-class head.

For clean, target-shuffle, and sham-shuffle states, replace the exact final-MLP write
by the complete output-SAE reconstruction while preserving the native residual,
LayerNorm, and classifier. Every checkpoint/initialization/endpoint must satisfy:

- mean absolute native-versus-reconstructed margin error `<=0.05`;
- native target-shuffle damage `>=0.05`;
- reconstructed target-shuffle damage is positive;
- reconstructed/native target-shuffle damage ratio `>=0.90`;
- sham-shuffle error does not exceed target-shuffle error by more than `0.02`.

Failure stops biological feature interpretation even if reconstruction and graph
sparsity pass.

## 9. Candidate generation and causal bridge

Only after Sections 6–8 pass:

1. Rank output features by clean-minus-target-shuffle activation and decoded
   classifier-aligned write, separately by endpoint and checkpoint.
2. Rank input→output edges by the paired-state change in signed Jacobian contribution,
   not by absolute edge weight alone.
3. Generate sequence hypotheses using held-out activation enrichment, k-mer/motif
   contrasts, positional edit maps, and the existing exact target/sham controls.
4. Freeze at most eight output features and sixteen directed edges per endpoint.

Each selected feature/edge family is compared with three matched null families:
activation frequency, decoder/write norm, and graph degree. Nulls are never pooled
across intervention sizes.

Required causal tests:

- **output-feature necessity:** subtract only its native output-feature decoded write;
- **output-feature recovery:** add the clean-state feature delta to the changed state;
- **input-feature path patch:** patch clean versus changed input-feature activation,
  run the exact frozen MLP, and measure the selected output feature and classifier;
- **edge mediation:** block the selected input-feature direction while retaining
  matched input reconstruction error, then test loss of the selected output-feature
  and classifier effect;
- **natural-range steering:** set activation only to observed 10/25/50/75/90 percentiles.

Inference uses gene-grouped bootstrap intervals, exact add-one permutation p-values,
and Benjamini-Hochberg correction over the complete frozen feature/edge family. A
biological label requires a sequence intervention that changes both the feature and
native classifier in the frozen sign, bidirectional feature patching, superiority to
every matched null family, and replication across all five checkpoints. Otherwise the
result remains a model-native feature descriptor.

## 10. Stop rules and strongest possible claim

- Any anomalous control freezes interpretation and triggers instrument diagnosis.
- An unresolved J-SAE does not weaken GFS11–GFS20 native-component results.
- A passing J-SAE licenses: “a causally faithful sparse coordinate system for this
  checkpoint's layer-5 MLP computation.”
- It does **not** by itself license: “the model learned a spliceosomal factor,”
  “the feature is monosemantic,” “the graph is complete,” or “the same feature exists
  in every seed.”

Every run writes an immutable JSON result, model artifact, and manifest containing
activation/checkpoint hashes, parameters, seeds, code revision, and command.
