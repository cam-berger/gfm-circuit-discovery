# Frozen preregistrations — DNABERT-2 reconstruction atlas

> **Tier: T1 — frozen.** These are the protocols as they were frozen before their runs. They are
> not edited to match outcomes. Where a protocol was amended, the amendment is dated and marked
> pre-outcome inside the file; where a design was retired, the file says so at the top and is kept
> as a design record.

Moved here from the study root on 2026-07-29. Result artifacts written **before** that date embed
the old flat path (`studies/dnabert2_reconstruction_atlas/PREREGISTRATION_*.md`) in their
provenance blocks — those artifacts are immutable and were deliberately not rewritten, so an old
path resolves to this directory.

| Protocol | Claim | Status |
|---|---|---|
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | D1–D11 | Frozen; the study's base protocol and its amendments |
| [`PREREGISTRATION_D12_DOWNSTREAM_DOMAIN.md`](PREREGISTRATION_D12_DOWNSTREAM_DOMAIN.md) | D12 | Frozen, run. Amendment 1 pre-outcome. One of five endpoints qualified |
| [`PREREGISTRATION_D13_COMPOSITION_VS_POSITION.md`](PREREGISTRATION_D13_COMPOSITION_VS_POSITION.md) | — | **Retired unrun.** Kept as a design record; its control was confounded with its own result |
| [`PREREGISTRATION_D14_ARRANGEMENT_VS_COMPOSITION.md`](PREREGISTRATION_D14_ARRANGEMENT_VS_COMPOSITION.md) | D14 | Frozen, run. Replaced D13 and answered the arrangement half |
| [`PREREGISTRATION_D15_ATTENTION_CAUSALITY_DISSOCIATION.md`](PREREGISTRATION_D15_ATTENTION_CAUSALITY_DISSOCIATION.md) | D15 | Frozen, run. Amendment 1 pre-outcome (a memorial that could not fail; population memorials gated at smoke scale) |
| [`PREREGISTRATION_D16_FINETUNING_ATTENTION_SHIFT.md`](PREREGISTRATION_D16_FINETUNING_ATTENTION_SHIFT.md) | D16 | Frozen, run. Interpretation rule fixed before any base-model number was read |
| [`PREREGISTRATION_D17_L9H7_ROTATION_PROBE.md`](PREREGISTRATION_D17_L9H7_ROTATION_PROBE.md) | D17 | Frozen, run. Its own decision rule was later found architecture-forced for L11H2 — recorded in the protocol and in `CLAIMS.md`, not silently repaired |

**Conformance caveat.** These protocols have **not** been through the independent
protocol-vs-execution audit in [`docs/PREREG_CONFORMANCE.md`](../../../docs/PREREG_CONFORMANCE.md),
which covers the SpliceBERT arc only. Known departures here are self-reported in
[`../DECISIONS.md`](../DECISIONS.md) and in the amendment sections above.
