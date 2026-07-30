# Preregistration: asymmetric-capacity layer-5 Jacobian SAE instrument v3

**Status:** frozen after preserving v1/v2 and before v3 outcomes  
**Date frozen:** 2026-07-30  
**Parents:** `PREREGISTRATION_JACOBIAN_SAE_V1.md`,
`PREREGISTRATION_JACOBIAN_SAE_V2.md`

## 1. Outcome-blind diagnosis

V1 and V2 never opened a classifier output, endpoint/variant association, feature
example, or intervention. Both show the same label-free instrument pattern:

- native MLP replay and analytic Jacobian controls pass;
- reconstruction and reconstructed-input MLP fidelity pass;
- the Jacobian penalty monotonically concentrates active edge mass;
- the symmetric 512/512 dictionaries use very different global supports.

In v2, the zero-penalty input/output dictionaries have 137/95 live training features.
At the nominal nonzero coefficient 0.0625 they have 131/52 live features. Thus the
same 512-feature capacity is not empirically justified on both sides of this narrow
target-token corpus. AuxK improves coverage but cannot make unused capacity
identifiable.

V3 tests whether a capacity matched to the observed, still outcome-blind support
resolves the instrument without changing the active graph size or any pass threshold.

## 2. Frozen v3 change

The input SAE has 256 features. The output SAE has 64 features. Both retain
per-example TopK=8 activity. The active Jacobian remains exactly `8 x 8`; J-SAE
mathematics does not require equal dictionary widths.

V3 retains the v2 AuxK repair independently for both dictionaries:

- eligibility after 52 consecutive non-firing steps at activation `>1e-3`;
- top 256 eligible input features, bounded by the 256-feature input width;
- top 64 eligible output features, bounded by the 64-feature output width;
- normalized residual loss coefficient `1/32` per dictionary.

Every other v1/v2 choice is unchanged: data, split, seed-2020 checkpoint, coefficient
scan `{0,0.015625,0.0625,0.25}`, initialization, optimizer, schedule, native-input
Jacobian definition, autograd control, metrics, selection rule, and stop rules.

The asymmetric widths were chosen from the unopened v2 training support with roughly
twofold headroom at the nominal coefficient. They are not scanned. This prevents
capacity selection from becoming a hidden multiple-comparison axis.

## 3. Unchanged gates

A nonzero fit must still satisfy all of:

- training dead-feature fraction `<=0.50` on both dictionaries;
- held-out input FVE `>=0.85`;
- held-out output-write FVE `>=0.95`;
- held-out reconstructed-input MLP output FVE `>=0.90`;
- each FVE no more than `0.03` below the width-matched zero-penalty control;
- median E90 at least 15% below that control and non-increasing through the
  candidate coefficient;
- analytic/autograd and unit-decoder controls pass.

Model selection is unchanged. If no nonzero condition passes, v3 is
`INSTRUMENT_UNRESOLVED` and all downstream behavior/semantics stay closed.

## 4. Claim boundary

A v3 pass establishes only that this seed-2020 target-token MLP admits a
capacity-matched, causally testable sparse coordinate graph under the frozen
representation gates. It authorizes the v1 downstream behavioral preflight, not a
biological feature, causal edge, cross-seed family, or steering claim.
