# Preregistration: named positional carriers on native task margins

> **Tier: T2 — provenance.** Frozen before any named-carrier intervention on a
> fine-tuned classifier.

## Question and scope

The weight-only spectrum assay establishes that period 3 (bins `339–345`) and
≈10.63 (bins `94–100`) survive GTEx fine-tuning, but its matched-frequency
control rule cannot form three comparably powerful non-carrier bands. This
exploratory gate asks directly whether the released classifier's native donor
margin, acceptor margin, or paired-acceptor donor coupling changes when either
named carrier is removed.

This is a screening assay on a reused cohort. It can motivate a separately
sealed confirmation, but it cannot silently replace the unavailable
frequency-matched-control design.

## Frozen models and cohort

- Base and five fine-tuned checkpoints must exactly match the hashes in
  `positional_spectrum_v1_target_gate_correction.json`.
- Reuse the first 64 examples, in recorded order, from
  `positional_band_surgery_v1.json`.
- Reconstruct those exact example IDs from the same GENCODE basic annotation
  and hg38 FASTA; gene IDs and chromosomes must replay exactly.
- No outcome-based example selection is allowed.

## Interventions

For each checkpoint and named seven-bin band:

1. intact table;
2. full deletion of the band;
3. 50% deletion of the band;
4. replacement of the tuned band with the frozen base-model band;
5. three same-band phase controls.

Each phase control multiplies every nominated complex rFFT coefficient by
`exp(±iπ/3)`. The three frozen sign patterns are all-positive, alternating
`+−+−+−+`, and `++−+−−+`. Because
`|exp(±iπ/3)−1|=1`, every phase intervention has exactly the same table-space
RMS dose as deleting the band, while preserving every nominated per-bin
coefficient magnitude and total band power. These are structured
same-carrier controls, not assumptions that phase is irrelevant.

## Endpoints and estimands

Native endpoints:

- donor class-vs-no-site margin at the last exonic boundary base;
- acceptor class-vs-no-site margin at the annotated downstream acceptor;
- donor coupling: clean donor margin minus paired `AG→CC` donor margin.

For each band and endpoint:

- deletion damage: intact minus deleted;
- half-dose damage: intact minus 50%-deleted;
- base-band rescue: base-replaced minus deleted;
- mean phase damage: intact minus the mean of three phase controls;
- deletion-minus-phase: deletion damage minus mean phase damage.

The 50% arm is a descriptive dose check. Base replacement tests functional
interchangeability with the pretrained carrier. Deletion-minus-phase asks
whether removing the carrier is more damaging than an equal-RMS displacement
that preserves its power.

## Hard instrument gate

Every seed must pass:

1. captured device table exactly equals the raw checkpoint table;
2. all tables and logits are finite;
3. every edit changes no non-nominated Fourier coefficient by more than
   `1e-10` relative to maximum checkpoint coefficient magnitude;
4. each phase control preserves nominated band power within `1e-10` relative;
5. each phase-control RMS dose matches full deletion within `1e-10` relative;
6. intact local `GT→CC` donor destruction is at least `5.0` margin units;
7. every intervention retains at least `80%` of intact local-donor
   destruction.

An anomaly stops interpretation and is preserved.

## Inference and screening rule

Average the five checkpoint effects within example, then use 5,000
gene-grouped bootstrap draws and 9,999 gene sign flips with exact add-one
p-values. One BH family spans:

`2 bands × 3 endpoints × {deletion, base rescue, deletion-minus-phase} = 18`.

A band/endpoint passes the native-use screen only if deletion damage and
base-band rescue:

- are positive in all five checkpoints;
- each have a 95% interval above zero;
- each have BH `q≤0.05`; and
- each have mean effect at least `0.05` margin units.

Deletion specificity additionally requires deletion-minus-phase to have an
interval above zero and BH `q≤0.05`. Failure is a bounded/null screen for the
tested endpoint, not evidence that the carrier has no function elsewhere.
