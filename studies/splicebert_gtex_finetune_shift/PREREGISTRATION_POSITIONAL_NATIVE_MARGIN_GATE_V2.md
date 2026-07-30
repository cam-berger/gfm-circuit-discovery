# Preregistration: health-preserving positional carriers on native margins

> **Tier: T2 — provenance.** Frozen after the V1 health diagnosis and before
> opening any named-carrier outcome on the V2 cohort.

## Why V2 is required

The V1 instrument and all phase controls passed mechanically, but full
period-3 deletion retained less than the frozen 80% local `GT→CC` response in
seeds 2020, 2021, 2022, and 2024. Full period-10 deletion passed that health
check in every seed. V1 is preserved as
`STOPPED_INTERVENTION_PREFLIGHT`; its endpoint inference remains sealed.

V2 tests the largest already-observed health-preserving dose for each band:

- period 3: retain 50% of bins `339–345`;
- ≈10.63 nt: retain 0% of bins `94–100`.

The choice uses only V1 intervention-health results. No V2 outcome has been
opened.

## Frozen models and outcome-fresh cohort

- Base and five fine-tuned checkpoints must exactly match the hashes in
  `positional_spectrum_v1_target_gate_correction.json`.
- Use examples 65–128 (zero-based slice `[64:128]`), in recorded order, from
  `positional_band_surgery_v1.json`.
- Reconstruct those exact example IDs from the same GENCODE basic annotation
  and hg38 FASTA; gene IDs and chromosomes must replay exactly.
- These genes appeared in the earlier scan-candidate study, but their named
  period-3 and period-10 intervention outcomes have never been opened.

## Interventions and matched controls

For each checkpoint and named band:

1. intact table;
2. the frozen band-specific attenuation;
3. replacement of the tuned band with the frozen base-model band;
4. three equal-RMS, same-band phase controls.

For removal fraction `r`, each phase control rotates the nominated complex
rFFT coefficients by `±θ`, where `θ = 2 asin(r/2)`. Therefore
`|exp(iθ)-1| = r`: phase surgery has exactly the attenuation's table-space RMS
dose while preserving every nominated coefficient magnitude and total band
power. Sign patterns remain all-positive, alternating `+−+−+−+`, and
`++−+−−+`.

## Endpoints and estimands

Native endpoints:

- donor class-vs-no-site margin at the last exonic boundary base;
- acceptor class-vs-no-site margin at the annotated downstream acceptor;
- donor coupling: clean donor margin minus paired `AG→CC` donor margin.

For each band and endpoint:

- attenuation damage: intact minus attenuated;
- base-band rescue: base-replaced minus attenuated;
- mean phase damage: intact minus the mean of three phase controls;
- attenuation-minus-phase: attenuation damage minus mean phase damage;
- base rewrite: intact minus base-replaced (descriptive).

## Hard instrument gate

Every seed must pass:

1. captured device table exactly equals the raw checkpoint table;
2. all tables and logits are finite;
3. every edit changes no non-nominated Fourier coefficient by more than
   `1e-10` relative to maximum checkpoint coefficient magnitude;
4. each phase control preserves nominated band power within `1e-10` relative;
5. each phase-control RMS dose matches its attenuation within `1e-10`
   relative;
6. intact local `GT→CC` donor destruction is at least `5.0` margin units;
7. every intervention retains at least `80%` of intact local-donor
   destruction.

An anomaly stops interpretation and is preserved.

## Inference and screening rule

Average the five checkpoint effects within example, then use 5,000
gene-grouped bootstrap draws and 9,999 gene sign flips with exact add-one
p-values. One BH family spans:

`2 bands × 3 endpoints × {attenuation, base rescue, attenuation-minus-phase} = 18`.

A band/endpoint passes the native-use screen only if attenuation damage and
base-band rescue:

- are positive in all five checkpoints;
- each have a 95% interval above zero;
- each have BH `q≤0.05`; and
- each have mean effect at least `0.05` margin units.

Attenuation specificity additionally requires attenuation-minus-phase to have
an interval above zero and BH `q≤0.05`. Passing licenses confirmatory
behavioral evidence for this cohort and intervention. It does not by itself
identify which downstream layers mediate the positional signal.
