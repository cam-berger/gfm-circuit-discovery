# Preregistration: layer-5 Jacobian sparse autoencoder instrument v2

**Status:** frozen after preserving the outcome-blind v1 failure and before rerun  
**Date frozen:** 2026-07-30  
**Parent protocol:** `PREREGISTRATION_JACOBIAN_SAE_V1.md`

## 1. Reason for the repair

V1 passed every mechanical and algebraic control. Across the four matched
seed-2020 fits, held-out input FVE was `0.910–0.913`, output-write FVE was
`0.995–0.998`, and reconstructed-input MLP output FVE was `0.971–0.992`.
The Jacobian penalty behaved with the intended sensitivity: median E90 fell
monotonically from 20 active edges at zero penalty to 4 at coefficient 0.25.

V1 stopped solely because the training-state dead-feature fractions were
`0.740–0.764` for the input SAE and `0.879–0.924` for the output SAE, above the
frozen 0.50 ceiling. No classifier result, endpoint association, variant-family
association, feature example, or feature intervention was opened.

This is a known TopK-SAE optimization failure mode. V2 adds the AuxK residual
reconstruction loss used to revive dead latents in:

- Gao et al., *Scaling and evaluating sparse autoencoders*:
  <https://arxiv.org/abs/2406.04093>
- the authors' MIT-licensed reference implementation:
  <https://github.com/openai/sparse_autoencoder>

## 2. Only permitted change

Every v1 data boundary, split, checkpoint, architecture, TopK count, Jacobian
definition, coefficient scan, initialization, optimization schedule, representation
threshold, graph threshold, selection rule, and stop rule remains frozen.

The sole change is an auxiliary loss for each of the input and output SAEs:

1. Maintain the number of consecutive training steps since each feature last had an
   activation greater than `1e-3`.
2. A feature becomes AuxK-eligible after one complete training epoch without firing
   (`ceil(6530/128)=52` steps).
3. Among eligible features, take the top 256 positive preactivations per example, or
   all eligible features if fewer than 256 exist.
4. Decode those auxiliary activations and fit the detached residual of the primary
   TopK reconstruction.
5. Normalize auxiliary MSE by the target-residual variance.
6. Add input and output auxiliary losses with coefficient `1/32` each.

AuxK is training-only. Deployment, Jacobian construction, graph metrics, feature
frequency, causal patching, and biological evaluation use exactly the primary TopK=8
features.

## 3. Unchanged gates

V2 is `INSTRUMENT_PASS` only if a nonzero Jacobian coefficient meets every v1 gate,
including:

- training-state dead-feature fraction `<=0.50` for both dictionaries;
- held-out input FVE `>=0.85`;
- held-out output-write FVE `>=0.95`;
- held-out reconstructed-input MLP output FVE `>=0.90`;
- no FVE more than `0.03` below the zero-penalty AuxK control;
- median E90 at least 15% below the zero-penalty AuxK control;
- E90 non-increasing through the candidate coefficient;
- analytic/autograd and decoder-norm controls pass.

If V2 fails, feature semantics, downstream behavior, five-seed fitting, and steering
remain closed. No further repair may reuse the name v2 or relax these thresholds.

## 4. Provenance and claim limit

The immutable v2 JSON, model artifact, and manifest must bind both protocols, the v1
failure artifact, activation arrays, checkpoint, command, code revision, and seed.

A pass licenses only a seed-2020 representation instrument and authorizes the already
frozen v1 downstream behavioral preflight. It does not license a biological feature,
cross-seed identity, causal edge, or steering claim.
