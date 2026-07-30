# Implementation addendum: fresh-cohort J-SAE delta confirmation v1

**Status:** frozen before constructing the fresh cohort or reading any result on it  
**Date frozen:** 2026-07-30  
**Controlling protocols:** `PREREGISTRATION_JACOBIAN_SAE_V1.md` Section 8 and
`PREREGISTRATION_NATIVE_BIOLOGICAL_FEATURES_V1.md` Section 8  
**Development results preserved:** `jacobian_sae_behavior_preflight_v1.json` and
`jacobian_sae_delta_preflight_v1.json`

## Rationale

The original 23 dictionary-validation genes are now outcome-exposed. They may diagnose
the instrument but cannot authorize a repair designed after their failures. This
confirmation therefore retains the original trained encoders, decoders, sparse graph,
and thresholds and evaluates them on a newly frozen gene-disjoint cohort.

No decoder refit is allowed. The original output writers remain the scientific object.
The dense reconstruction remainder is represented explicitly as an uninterpreted
error block rather than hidden by post-hoc calibration.

## Outcome-blind cohort

Before loading a model:

1. Scan the existing `results/splicebert_gtex_finetune_shift` JSON tree with the
   canonical Ensembl-gene extractor.
2. Exclude every recorded gene, comparing Ensembl identifiers without version
   suffixes.
3. Exclude the repository's frozen forbidden chromosomes.
4. From eligible GTEx training examples, retain one deterministic example per gene,
   order by SHA-256 of namespace, gene, and example ID, and freeze the first 128.

For each gene and endpoint, create exactly three states: clean, target-window
dinucleotide shuffle, and the opposite-side sham-window shuffle. Windows, anchors,
draws, target-token positions, and sequence construction are identical to the earlier
biological-feature package. The package contains 768 states from 128 genes.

The package records the complete prior-gene exclusion inventory and aggregate hash.
No classifier output, SAE feature, activation, margin, or biological annotation is
used for selection.

## Activation capture

Capture only the five tuned checkpoints, sequentially. At the target token record:

- exact input to layer-5 `BertIntermediate`;
- exact layer-5 dense MLP write before residual addition and LayerNorm.

The existing mechanical tolerances remain:

- MLP-input equality error `<=2e-5`;
- dense-write replay error `<=5e-4`.

No classifier output or parameter direction is read during capture.

## Frozen confirmation

For every checkpoint and each of the three original coefficient-0.25 dictionary
initializations, decode all writes and feature codes before extracting the classifier
direction. The intervention and thresholds are unchanged from the residual-preserving
development assay:

- dense target damage and recovery each `>=0.05`;
- sparse target damage and recovery positive;
- sparse-versus-dense relative error `<=0.10` in each direction;
- sham absolute error no larger than target absolute error plus `0.02`;
- all-feature decoder-sum discrepancy `<=2e-5`;
- float64 zero-delta identity discrepancy `<=1e-12`.

All 30 checkpoint × initialization × endpoint cells and all 15 mechanical cells must
pass.

## Explicit error-block accounting

For each pair define:

`dense_delta = native_write(changed) - native_write(clean)`

`feature_delta = decoded_write(changed) - decoded_write(clean)`

`error_delta = dense_delta - feature_delta`.

Evaluate feature-only, error-only, and joint patches through the untouched native
residual, LayerNorm, and classifier. The joint patch must reproduce the dense patch to
`1e-10` margin units. Because LayerNorm is nonlinear, allocate the dense behavioral
effect between feature and error blocks using the exact two-player Shapley value,
averaged over both intervention orders.

For target shuffles, the absolute mean error-block Shapley allocation divided by the
absolute dense effect must be at most `0.10`, separately for damage and recovery in
every cell. Sham allocations are recorded diagnostically and are not treated as sparse
features.

## Stop and claim boundary

Any failed gate sets `INSTRUMENT_UNRESOLVED`; individual feature identity, edge
interpretation, semantics, and steering remain closed.

A complete pass establishes only that the original J-SAE output-feature block
generalizes to unseen genes as a faithful residual-preserving coordinate system for
the target-shuffle final-MLP path, with at most 10% mean task effect assigned to its
dense error block. It licenses frozen feature/edge candidate generation, not an
individual causal or biological feature claim.
