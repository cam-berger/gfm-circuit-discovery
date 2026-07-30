# Implementation addendum: five-seed Jacobian-SAE replication v1

**Status:** frozen before replication fits  
**Date frozen:** 2026-07-30  
**Controlling representation protocol:** `PREREGISTRATION_JACOBIAN_SAE_V3.md`

## Fixed target

Replicate only the v3-selected architecture and coefficient:

- independent J-SAE per fine-tuned checkpoint seed 2020–2024;
- input/output feature widths 256/64;
- TopK 8;
- AuxK exactly as v3;
- Jacobian coefficient 0.25;
- native-input analytic Jacobian;
- same 105-gene/23-gene training/validation split and normalization rule.

For every checkpoint and initialization, fit a capacity-, TopK-, data-, optimizer-,
schedule-, and initialization-matched zero-Jacobian-penalty pair. Checkpoints and fits
run sequentially. No classifier parameter or task outcome is loaded.

Dictionary initializations are `{0,1,2}` with seed:

`2,000,000 + 10,000*checkpoint_seed + initialization`.

## Per-fit gates

Every one of the 15 nonzero fits must satisfy:

- training input and output dead-feature fraction `<=0.50`;
- validation input FVE `>=0.85`;
- validation output-write FVE `>=0.95`;
- validation reconstructed-input MLP output FVE `>=0.90`;
- each FVE no more than `0.03` below its initialization-matched zero-penalty fit;
- median validation E90 at least 15% below its matched zero-penalty fit;
- analytic/autograd and unit-decoder controls pass.

All metrics and failed fits are preserved. The v3 seed-2020 model is not substituted
for its replication fit.

## Within-checkpoint optimizer stability

Match features separately for the input and output dictionaries on held-out
validation states. A feature is eligible if it is positive on at least 0.1% of
validation states. Similarity is the equal-weight mean of:

1. nonnegative activation Pearson correlation on identical states;
2. nonnegative decoder-direction cosine in the same checkpoint's native coordinates.

For all three initialization pairs, both SAE sites, and every checkpoint require:

- reciprocal-match fraction `>=0.25`;
- mean reciprocal combined score `>=0.50`.

This is optimizer stability, not biological convergence.

## Cross-checkpoint matching

Because prior evidence establishes checkpoint-specific rotations, raw decoder cosine
is not a replication gate across checkpoints. Report, on identical held-out states:

- reciprocal activation-correlation matches;
- raw-coordinate activation-plus-decoder matches as a sensitivity;
- all ten checkpoint pairs, separately for input and output features.

No cross-seed feature family is promoted in this representation-only run. A later
feature family must reproduce a frozen sequence response and causal classifier sign
in all five seeds.

## Graph stability diagnostic

For every fit, accumulate the mean absolute active Jacobian into the complete
`64 x 256` output-feature by input-feature matrix on validation states. Use the
feature mappings above to report top-64 edge overlap and aligned matrix cosine for
each within-checkpoint initialization pair. These are required diagnostics but not
additional gates because no independent threshold was calibrated before this run.

## Stop and claim boundary

Five-seed representation replication passes only if all 15 per-fit gates and all 30
within-checkpoint site-stability comparisons pass. Failure leaves downstream native
heads, feature examples, sequence associations, and steering closed.

A pass authorizes only the already frozen native-head behavioral preflight. It does
not establish causal edges, cross-seed biological features, or classifier mediation.
