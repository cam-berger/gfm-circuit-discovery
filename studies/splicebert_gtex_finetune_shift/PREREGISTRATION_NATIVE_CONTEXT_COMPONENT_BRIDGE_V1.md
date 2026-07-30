# Preregistration: frozen final-block bridge for confirmed context regions

> **Tier: T2 — provenance.** Frozen after the context-region confirmation and
> source-to-target depth trace, but before any component patch was evaluated for either
> context shuffle. This is a targeted mechanistic transfer/verification assay on the
> existing validation cohort, not a new independent confirmation cohort.

## Question

The context-localization study independently established two 32-nt sequence-order
effects and localized most source-to-target integration to the final encoder block:

- donor endpoint: `[-80,-48)` relative to `donor_pos`;
- acceptor endpoint: `[+16,+48)` relative to `next_acceptor_pos`.

The earlier paired-acceptor study independently selected and confirmed `L5H6`, its
controls `{L5H7,L5H11,L5H9}`, and the final MLP. This bridge asks:

> Does that already-frozen `L5H6 → final-MLP` pathway carry either newly confirmed
> sequence-context effect, or is final-block integration implemented by a different
> distributed route?

No head is rescanned or reselected.

## Domain and endpoints

- Same `n=200` validation examples and five published seeds 2020–2024 as
  `native_context_confirmation_v1.json`.
- Same draw-0 dinucleotide-preserving full-region corruptions. The confirmation artifact
  must replay within `2e-5` for every seed and endpoint before component effects open.
- Donor and acceptor native class-vs-no-site margins use their documented label
  coordinates.
- Unit of resampling and sign flipping is gene.

This reuses the confirmed validation cohort and is therefore labeled **mechanistic
verification**, even though component identities and all thresholds are frozen before
the run.

## Exact interventions

At the classified target token in final encoder layer 5:

1. **Head noising:** patch the context-corrupt head output into the clean run. Damage is
   `clean margin - patched margin`.
2. **Head denoising:** patch the clean head output into the context-corrupt run. Recovery
   is `patched margin - corrupt margin`.
3. Repeat both directions for frozen `L5H6` and each of `{L5H7,L5H11,L5H9}`.
4. **Final-MLP noising/denoising:** analogously patch the pre-residual MLP write at the
   target.
5. **Controlled direct `L5H6` effect:** noise `L5H6` in the clean run while clamping the
   target's final-MLP write to its clean value. The MLP-dependent contrast is ordinary
   `L5H6 damage - controlled direct damage`. It is an interaction contrast, not an
   additive mediation percentage.

Exact preflight from `native_mechanism_preflight_v1.json` remains required. Empty,
identity, direct-final-layer replay, whole-head, and MLP patch mechanics are not
redefined here.

## Inference family and gates

One 28-test family contains, for each of two endpoints:

- four heads × two directions = 8;
- `L5H6 - mean(three frozen controls)` × two directions = 2;
- final MLP × two directions = 2;
- controlled direct `L5H6` damage = 1; and
- MLP-dependent `L5H6` damage contrast = 1.

Every test uses a two-sided exact add-one gene sign-flip p-value; BH correction is over
all 28. Family means average each example's paired effect across the five checkpoints
before gene resampling. Per-example component/behavior ratios are forbidden.

`L5H6` transfers for an endpoint only if:

1. noising damage and denoising recovery are each at least `0.05`, have 95% intervals
   above zero, BH `q<=0.05`, and are positive in all five seeds; and
2. both candidate-minus-control contrasts are at least `0.05`, have intervals above
   zero, and BH `q<=0.05`.

The final MLP transfers if both directions have intervals above zero, BH `q<=0.05`,
positive means in all five seeds, and ratio-of-gene-means recovery at least `0.50`.

If `L5H6` transfers, the interaction is called MLP-dependent only when
`L5H6 damage - controlled direct damage >=0.05`, its interval is above zero, and
`q<=0.05`. Otherwise it remains descriptive.

## Licensed outcomes

- `L5H6` + MLP pass: the new context effect converges on the already-established
  final-block route.
- MLP pass, `L5H6` fail: context is transformed by the same final task stage but reaches
  it through a different or more distributed attention route.
- Both fail with healthy replay: final-state integration does not imply mediation by
  these frozen components; a train-only final-head discovery is required.

Even a positive bridge does not identify the biological grammar in the 32-nt region.
