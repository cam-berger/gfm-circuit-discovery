# Implementation addendum: J-SAE residual-preserving delta preflight v2

**Status:** frozen before computing any residual-preserving delta intervention  
**Date frozen:** 2026-07-30  
**Controlling protocols:** `PREREGISTRATION_JACOBIAN_SAE_V1.md` Section 8 and
`PREREGISTRATION_NATIVE_BIOLOGICAL_FEATURES_V1.md` Section 8  
**Prerequisites:** `jacobian_sae_replication_v1.json` is `INSTRUMENT_PASS`; the
complete-write failure in `jacobian_sae_behavior_preflight_v1.json` is preserved

## Why this is a separate instrument

The v1 preflight replaced the complete native layer-5 MLP write with an SAE
reconstruction. It preserved every target-shuffle effect sign and at least 90% of
every effect, but failed the frozen absolute-margin gate in all 30 cells. That failure
remains binding for complete-write replacement.

The biological confirmation protocol had already specified a different causal
interface: retain the native write, native reconstruction error, residual stream,
LayerNorm, and classifier, and add only a decoded feature-activation difference. This
preflight tests that interface before inspecting an individual feature, graph edge,
sequence association, or steering result. Classifier-level behavior from v1 is
necessarily known, so this is a post-failure instrument repair, not an outcome-fresh
replication.

## Frozen states and interventions

Use the same 23 held-out dictionary-validation genes, target-shuffle pairs, sham-
shuffle pairs, five checkpoints, and three frozen coefficient-0.25 dictionary
initializations as v1.

For each clean/counterfactual pair, endpoint, checkpoint, and initialization:

1. **Dense damage** keeps the clean residual and replaces the clean native MLP write
   with the counterfactual native write.
2. **Sparse damage** keeps the clean residual and clean native MLP write, then adds
   `decoded(counterfactual)-decoded(clean)`.
3. **Dense recovery** keeps the counterfactual residual and replaces its native MLP
   write with the clean native write.
4. **Sparse recovery** keeps the counterfactual residual and native write, then adds
   `decoded(clean)-decoded(counterfactual)`.

The dense interventions are the correct reference because they isolate the final-MLP
write path from residual-stream changes. The sparse intervention is additive over all
64 output features: the decoder bias and normalization center cancel in a paired
difference. No feature identity or support is recorded.

All decoded writes and feature codes must be frozen in memory before loading the first
classifier. There is no optimizer, model selection, threshold adaptation, or feature
selection.

## Mechanical gates

For both target and sham pairs in every checkpoint × initialization:

- the maximum discrepancy between a decoded-write difference and the corresponding
  all-feature decoder sum is at most `2e-5`;
- a zero feature delta leaves the native write unchanged to at most `1e-12`.

## Behavioral gates

For every checkpoint × initialization × endpoint target-shuffle cell:

- mean dense damage and mean dense recovery are each at least `0.05`;
- mean sparse damage and mean sparse recovery are positive;
- `abs(sparse-dense)/abs(dense) <= 0.10` separately for damage and recovery;
- the absolute sparse-versus-dense error on sham pairs is no greater than the
  corresponding target-pair error plus `0.02`, separately for damage and recovery.

All 30 cells and every mechanical check must pass.

## Stop and claim boundary

Any failed gate sets `INSTRUMENT_UNRESOLVED` and leaves individual-feature discovery,
edge interpretation, biological semantics, and steering closed.

A complete pass licenses only the preregistered residual-preserving candidate-
generation and causal-patching stages. It establishes that the complete decoded
feature delta approximates the native final-MLP write delta at the classifier, while
the native reconstruction error remains fixed. It does not establish that any
individual feature or edge is stable, causal, biological, or sufficient.
