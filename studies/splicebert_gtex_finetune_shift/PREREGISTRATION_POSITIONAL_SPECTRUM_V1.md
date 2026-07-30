# Preregistration v1: positional-carrier change after GTEx fine-tuning

> **Tier: T2 — provenance.** Frozen before reading any fine-tuned positional
> Fourier coefficient, spectrum, carrier power, carrier subspace, or
> positional-band intervention outcome.

## 1. Question and evidence boundary

The final multi-species `SpliceBERT.1024nt` position table contains two
previously established high-frequency structures:

- the causal period-3 carrier in Fourier bins `339–345`;
- a distinct carrier in bins `94–100`, whose final-checkpoint continuous
  wavelength is `10.652435732380217` nt and whose cross-checkpoint name is
  **≈10.63**.

The GTEx checkpoints fine-tune the complete encoder, including the learned
absolute position table. Existing GFS preflight records a large table-level
parameter displacement but never decomposed that displacement by frequency.

This study asks:

1. Are the two frozen carrier bands retained, amplified, attenuated, or
   rewritten in embedding-space coordinates across all five GTEx seeds?
2. Which other contiguous frequency bands consistently gain or lose normalized
   power?
3. Does the released native splice classifier causally use a surviving or
   reproducibly changed band?

Spectrum, subspace overlap, attention use, and native-classifier causality are
separate evidence levels. A weight-space peak is a candidate generator only.

## 2. Frozen sources and named bands

Base checkpoint:

- `SpliceBERT.1024nt/pytorch_model.bin`
- SHA256 `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`
- position table shape `1026 × 512`

Fine-tuned checkpoint SHA256 values:

- 2020: `2552486a0870e2466a223fcce34f0477c40526a09172622619f0d1672a1af45e`
- 2021: `2cb06d2a17d941857485329604d526b465eabda319632c932000b85d8d40cda6`
- 2022: `556e38c92319ce87fb15ca390743ae20c539f4c5400c4cca82a83cdf3d46bd0a`
- 2023: `03c25882ae3e4b48198265bf8447ef543308e40ec9fd700b4834d829dcb6100b`
- 2024: `d1caccec4db1784537178e355351a3e0b67cde4e70185bf6b6a6d63f792198f1`

Prior carrier records, read before this freeze:

- `carrier_subbin.json`, SHA256
  `8a79ff0ba261020a439fc54797679a24382dcf0254f055b4450efd96c29bf7fb`;
- `carrier_geometry.json`, SHA256
  `2a46ebffc44b82f625adeba0ea0ff2bdada6a0ff8e8cb1233e8c3d678db248ef`;
- `fig_phase_data.json`, SHA256
  `91d7f51e113fecd7d8972d6b3a1f87278de9c3efe22fbc600f82e9584ebe9316`.

Frozen bands are:

- `period3 = [339,340,341,342,343,344,345]`;
- `period10 = [94,95,96,97,98,99,100]`.

The ≈10.63 label is nomenclature, not a claim that its exact period equals
10.630 in every checkpoint. Its primary band is the already frozen seven-bin
final-checkpoint carrier. A continuous leakage-aware fit at the frozen base
period is a sensitivity descriptor.

## 3. Spectrum and geometry estimands

For a position table `P`, subtract each embedding dimension's row mean and
compute the length-1026 real FFT. Per-bin power is the sum of squared complex
coefficient magnitude over all 512 embedding dimensions.

For every table and frozen band, record:

- raw band power and band/median-bin power;
- fraction of total non-DC centered spectral energy;
- the exact-bin peak and leakage-aware continuous-period power;
- rank and singular-value energy of the real band coefficient matrix.

The continuous-frequency peak is the maximum on a fixed `0.001`-nt grid:
`[2.95,3.05]` for period 3 and `[10.4,10.9]` for ≈10.63. Numerical rank uses
NumPy's default matrix-rank tolerance. The fine-tuning update-energy
denominator is the total centered, non-DC update energy.

For every fine-tuned seed relative to base, record:

- raw and normalized band-power ratios;
- fraction of fine-tuning update energy in the band;
- raw cosine between base and tuned band reconstructions;
- symmetric projection overlap and principal-angle cosines between the
  corresponding real coefficient subspaces.

Terminology thresholds:

- **power retained:** normalized band-power ratio `≥0.5`;
- **power amplified:** normalized ratio `≥2.0`;
- **power strongly attenuated:** normalized ratio `≤0.1`;
- **coordinate retained:** symmetric subspace overlap `≥0.5`;
- **coordinate rewritten:** power retained but symmetric overlap `<0.25`.

Family wording requires the condition in all five seeds. Otherwise report the
five quantitative values without a categorical family label.

## 4. Full-spectrum scan and frozen candidate rule

Scan every seven-bin contiguous window whose start is `3–507`. For each seed,
compare the window's fraction of total non-DC power with base. Report all
windows and perform two-sided exact sign flips over the five seed log-ratios,
with add-one p-values and BH correction across the complete scanned family.
The five-checkpoint family is not treated as five independent biological
samples; with five signs, this inference is resolution-limited and is not
expected to establish small effects.

Candidate generation uses no classifier outcome. Exclude windows overlapping
either frozen carrier, its period-6/period-1.5 harmonics, or an already selected
window. The period-6 exclusion is bins `168–174`; period 1.5 aliases to the
already excluded period-3 band. Freeze at most:

- two nonoverlapping gains; and
- two nonoverlapping losses.

A scan window is eligible only if all five seed log-ratios have the same sign
and the median absolute normalized power change is at least two-fold. Rank by
median absolute log2 change, breaking ties by lower start bin.

The two named carriers enter causal testing regardless of their spectrum
category. Eligible scan candidates enter only after this weight-only freeze.

## 5. Hard spectrum preflight

No target spectrum may be interpreted unless all checks pass:

1. exact checkpoint hashes, tensor key, `1026 × 512` shape, and finiteness;
2. centered FFT/IRFFT roundtrip maximum error `≤1e-10`;
3. Parseval relative error `≤1e-10`;
4. identity comparison gives power ratio, reconstruction cosine, and subspace
   overlap within `1e-10` of one;
5. a fixed orthogonal rotation of embedding dimensions preserves every
   non-DC per-bin power within relative error `≤1e-10`;
6. base period-3 band/median replays `802.4380295` within `1e-4` relative;
7. base ≈10.63 band/median replays `851.8638377` within `1e-4` relative;
8. the base sub-bin fit replays wavelength `10.6524357` within `0.01` nt;
9. planted rank-2 tones at period 3 and 10.6524357 recover their nominated
   periods within `0.01` nt and their coefficient subspaces with overlap
   `≥0.99`;
10. a fixed row-permuted negative control does not exceed the planted positive
    control at either target.

An anomaly preserves a stopped artifact and closes classifier surgery.

## 6. Gated native-classifier surgery

Only a complete spectrum artifact may freeze the surgery package. No
classifier output may be read while selecting target or control bands.

The behavior cohort contains 200 deterministic, one-example-per-gene records
from the annotation `train` chromosome split under namespace
`gtex-positional-spectrum-native-v1`. It excludes every gene already present
in a GTEx-shift JSON artifact and chromosomes `chr8`, `chr9`, `chr11`,
`chr16`, `chr19`, `chrX`, `chrY`, and mitochondrial contigs. This cohort is
fresh to the analysis but is not claimed to be absent from checkpoint
training.

For every checkpoint and target band, freeze three same-width, nonoverlapping
control bands matched to target power within ±25%, excluding every target,
named carrier, period-3 harmonic, and previously used control. If three
controls cannot be formed, that target stops. Targets are processed in this
fixed order: period 3, ≈10.63, ranked gains, then ranked losses. Eligible
controls are ranked by absolute log power mismatch, breaking ties by lower
start bin.

Interventions on each fine-tuned position table:

- identity/no-op;
- delete the target band;
- replace the tuned target band with the base target band;
- the corresponding deletion and base-band replacement for every frozen
  control.

Primary behavior is native donor coupling:

`donor margin(clean) - donor margin(paired AG→CC)`.

Per-example target estimands are:

- necessity: intact coupling minus target-deletion coupling;
- base-band rescue: base-band-replacement coupling minus target-deletion
  coupling;
- rewrite contrast: intact coupling minus base-band-replacement coupling.

Controls are the mean of the three matched bands at the same intervention.
Also record clean donor margin, clean donor-positive fraction, and local
`GT→CC` donor-margin destruction.

Surgery preflight, mandatory in every seed:

1. identity logits differ by at most `2e-5`;
2. FFT deletion changes no coefficient outside the nominated conjugate band
   above `1e-10` relative and IRFFT is real;
3. replacing every Fourier coefficient with its original value exactly
   reconstructs the intact table and logits;
4. local donor destruction is at least `5.0` nats in the intact model;
5. every target and control arm retains at least `80%` of intact local-donor
   destruction and at least `80%` of the intact clean-donor-positive fraction;
6. all margins and logits are finite.

Inference averages the five checkpoint effects within example, then uses
gene-grouped 5,000-draw bootstrap intervals and 9,999 exact add-one gene sign
flips. One BH family spans every tested target × estimand × target-minus-control
contrast. Per-seed estimates and all-five-positive signs are reported.

A band is licensed as a causal native-classifier carrier only when necessity
and base-band rescue are positive in all five seeds, each has a 95% interval
above zero, each exceeds the matched controls, every associated BH q-value is
`≤0.05`, and both mean effects are at least `0.05` margin units. A failed gate
is a bounded or null result for this behavior, not evidence that the band has
no other function.

## 7. Stopping and reporting

- Spectrum alone licenses only position-table retention, attenuation,
  amplification, or coordinate-rewrite wording.
- Surgery licenses necessity/rescue only for the native donor-coupling
  behavior and tested cohort.
- No frequency is called biological, codon-derived, nucleosomal, or a splice
  grammar from this assay.
- No threshold may change after a fine-tuned spectrum or classifier outcome is
  opened.
