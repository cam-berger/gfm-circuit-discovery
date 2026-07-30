# Implementation addendum: J-SAE native-head behavioral preflight v1

**Status:** frozen before loading any classifier parameter  
**Date frozen:** 2026-07-30  
**Controlling protocol:** `PREREGISTRATION_JACOBIAN_SAE_V1.md` Section 8  
**Prerequisite:** `jacobian_sae_replication_v1.json` is `INSTRUMENT_PASS`

## Frozen state set

Use only the 23 held-out dictionary-validation genes and, separately by endpoint:

- clean states paired to `target_shuffle`;
- the same clean roots paired to `sham_shuffle`.

No synonymous state, maximum-recoding state, feature identity, activation example, or
graph edge enters this preflight.

For each checkpoint seed 2020–2024 and each dictionary initialization 0–2, load the
frozen coefficient-0.25 output SAE. Reconstruct the complete native layer-5 MLP write
from its native write activation. Keep the exact native layer-5 residual input,
LayerNorm, and three-class classifier unchanged.

All 15 reconstructed-write arrays must be computed and frozen in memory before the
first native classifier parameter is loaded. There is no optimizer, model selection,
feature selection, or threshold adaptation in this runner.

## Metrics

For each checkpoint, initialization, and endpoint:

1. Compute native and reconstructed class-versus-background margins for the union of
   clean, target-shuffle, and sham-shuffle rows.
2. `margin_mae` is their mean absolute difference on that union.
3. `target_margin_mae` and `sham_margin_mae` are the corresponding errors on the
   unique rows in each clean/counterfactual pair family.
4. Target-shuffle effect is mean clean margin minus target-shuffle margin.
5. Effect-recovery ratio is reconstructed target-shuffle effect divided by native
   target-shuffle effect.

Every one of the 30 checkpoint × initialization × endpoint cells must satisfy:

- `margin_mae <=0.05`;
- native target-shuffle effect `>=0.05`;
- reconstructed target-shuffle effect `>0`;
- effect-recovery ratio `>=0.90`;
- `sham_margin_mae <= target_margin_mae + 0.02`.

No upper recovery-ratio bound is imposed by the parent protocol, but ratios above 1.10
are reported as diagnostics.

## Stop and claim boundary

Any failed cell sets `INSTRUMENT_UNRESOLVED` and keeps feature/edge discovery,
biological interpretation, and steering closed.

A complete pass licenses only that the frozen J-SAE output reconstruction is faithful
enough to begin the preregistered task-feature/edge candidate-generation stage. It
does not make any individual feature or edge causal.
