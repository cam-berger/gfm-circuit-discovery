# Implementation addendum: larger paired-delta J-SAE v1

**Status:** frozen before fitting any larger dictionary
**Date frozen:** 2026-07-30
**Parent evidence:** GFS26–GFS32 in `CLAIMS.md`
**Method references:** Farnik et al. (2025) J-SAE; the paired-delta term is a
project-specific, label-free repair motivated by the preserved sham-calibration
failures

## Scientific objective

The original 256-input-feature/64-output-feature TopK-8 J-SAE reproduced the
target-shuffle final-MLP path on 128 fresh genes, but nuisance/sham calibration
failed in four cells. This assay asks whether a larger sparse coordinate system,
trained to preserve both states and paired changes, resolves that bounded failure
without changing the classifier, selecting features, or fitting a post-hoc decoder.

This is an outcome-informed instrument repair. Training remains label-free: no
classifier parameter, margin, class label, feature identity, biological score, or
outcome-derived weight enters the objective.

## Architecture

Fit independent dictionary pairs around the frozen layer-5 MLP for:

- base SpliceBERT;
- tuned checkpoints 2020–2024.

The architecture is fixed:

- native input/output width: 512;
- MLP intermediate width: 2,048;
- input features: 512;
- output features: 128;
- TopK active features: 16 at both sites;
- exact GELU derivative and exact active `16×16` Jacobian;
- float32 training;
- unit-norm decoder rows;
- native-input Jacobian evaluation.

This is a capacity-limited sparse feature pair, not an overcomplete language-model
SAE.

## Frozen data

Use only the original label-free activation artifact. Preserve its deterministic
105-gene/23-gene split:

- 6,530 ordinary training states;
- 1,461 representation-validation states;
- 6,320 training reference/changed pairs;
- 1,415 validation pairs.

The runner must pin, before fitting:

- activation-report SHA-256
  `142ce4ee5976970b9b296685ef403cce30e56f440a87d7222a1933fe09d96b5b`;
- activation-array SHA-256
  `fa6c972f4765d5c57354abeb262b3e62d6ac3ff8c69e5850a1511c7cf054d559`;
- the exact six checkpoint-weight hashes embedded in the runner;
- 7,991 total states and all state/pair/gene split counts above.

Directory labels are not checkpoint identity. Any source mismatch stops before
the first dictionary is constructed.

Pairs have orientation `changed-reference` and are grouped into:

- target shuffle;
- sham shuffle;
- maximum recoding (draws 0 and 1 combined);
- synonymous single.

The fresh-v1 128-gene cohort is not used in training or representation selection.

## Balanced exposure

Every optimization step contains:

- 64 ordinary states, sampled by the existing inverse-gene-state epoch;
- 32 complete reference/changed pairs, contributing 64 additional encoded rows.

There are 103 complete optimization steps per epoch. The ordinary sampler draws
6,592 rows per epoch: its complete 6,530-row gene-balanced epoch plus 62
deterministic gene-balanced padding draws. There is no two-state terminal batch.

The pair schedule balances endpoint and pair family, then genes within each
endpoint/family cell. It cycles deterministically without replacement within each
gene cell. Target and sham receive identical exposure. The fixed 40-epoch schedule
must encounter every training pair at least once.

Ordinary reconstruction and Jacobian losses use only ordinary rows. Repeated clean
roots from the pair branch therefore cannot silently reweight those terms.

## Loss

Scalar centering/scaling is fit on ordinary training states exactly as in the
smaller J-SAE. Define:

`L_state = 0.5 * (MSE(input) + MSE(output))`.

For each site, freeze one checkpoint-specific training-pair delta power. It is the
mean normalized squared `changed-reference` delta under equal endpoint/family and
gene exposure. The paired loss is:

`L_delta = 0.5 * (MSE(delta_input)/power_input +
                  MSE(delta_output)/power_output)`.

Use global site powers, never pair-specific or outcome-specific normalization.

The active-Jacobian loss is `mean(abs(J_active))`. The primary objective is:

`L_state + 1.0*L_delta + 0.25*L_J + L_AuxK`.

AuxK is identical in every condition:

- coefficient `1/32`;
- maximum 64 auxiliary input features;
- maximum 64 auxiliary output features;
- dead after one ordinary-state epoch without firing;
- auxiliary activations never enter the Jacobian or reported feature statistics.

AuxK may not exceed 25% of the mean total loss over the last five epochs.

## Full factorial controls

For each tuned checkpoint and initialization, train four conditions from
byte-identical SAE initialization and identical ordinary/pair schedules:

| Condition | Paired delta | Jacobian |
|---|---:|---:|
| `delta_jacobian` | 1.0 | 0.25 |
| `delta_no_jacobian` | 1.0 | 0 |
| `no_delta_jacobian` | 0 | 0.25 |
| `plain_topk` | 0 | 0 |

For the base checkpoint, train the primary and its matched
`delta_no_jacobian` graph control. The base arm is a representation comparator;
it has no native splice-classifier behavioral endpoint.

Use three dictionary initializations, Adam `(0.9,0.999)`, peak learning rate
`5e-4`, 1% warmup, 20% terminal decay, 5% Jacobian warmup, gradient norm cap 1,
and exactly 40 epochs. There is no early stopping or outcome-based checkpoint
selection.

Checkpoint order is seed 2020 pilot, then the remaining tuned checkpoints, then
base. Only one checkpoint and one live fit may occupy the accelerator at a time.
If the seed-2020 initialization-0 primary representation gate fails, preserve the
pilot and stop before the family.

## Representation metrics and gates

Every primary fit must satisfy:

- exact MLP replay `<=5e-4`;
- analytic-versus-autograd active Jacobian within the inherited tolerances;
- small-dose decoder-direction finite difference:
  correlation `>=0.95`, sign agreement `>=0.95`, relative Frobenius error
  `<=0.10`;
- decoder norm error `<=1e-5`;
- training dead-feature fraction `<=0.50` at both sites;
- validation input FVE `>=0.95`;
- validation output-write FVE `>=0.95`;
- reconstructed-input MLP-output FVE `>=0.90`;
- none of those three FVEs more than `0.03` below matched
  `delta_no_jacobian`;
- target/sham input-delta FVE `>=0.90` and output-delta FVE `>=0.95` for each
  endpoint;
- maximum-recode/synonymous input-delta FVE `>=0.85` and output-delta FVE
  `>=0.90` for each endpoint;
- target/sham output-delta FVE no more than `0.02` below
  `no_delta_jacobian`;
- median E90 at least 15% lower than matched `delta_no_jacobian`;
- median effective active-edge count at least 10% lower than matched
  `delta_no_jacobian`;
- late AuxK fraction `<=0.25`.

Report E50/E90/E99, effective active-edge count, top-10% edge mass,
scale-normalized Jacobian statistics, signed-edge prevalence, exact L0 and active
fraction. The complete 2×2 factorial is descriptive; only frozen matched
contrasts gate.

Within each checkpoint, all three initialization pairs must pass the inherited
activation/writer reciprocal-match gates at input and output:

- reciprocal fraction `>=0.25`;
- mean reciprocal score `>=0.50`;
- validation activation-density floor `0.001`.

All tuned checkpoints and the base comparator must pass their registered
representation gates before any new classifier cohort is constructed.

## Behavioral firewall

No classifier parameter or margin may be loaded by this runner. A complete
representation pass freezes the primary models, feature maps, graph summaries,
and thresholds, and only then authorizes construction of namespace
`gtex-jacobian-sae-fresh-delta-confirmation-v2`.

The v2 cohort must:

- snapshot and hash the complete prior-result JSON inventory before and after
  selection;
- exclude every canonicalized Ensembl gene in that inventory;
- explicitly have zero overlap with the original corpus and fresh-v1 cohort;
- exclude the frozen chromosomes;
- contain 128 deterministic feasible genes and 768 clean/target/sham states;
- be constructed and hashed before model capture.

Fresh confirmation reuses every GFS32 threshold unchanged and computes decoder
algebra in float64. Any representation, mechanical, target, sham, joint replay,
or error-block failure closes feature identity, edge interpretation, semantics,
and steering.

## Claim boundary

A representation pass licenses only:

> Across checkpoint-specific SpliceBERT MLPs, a capacity-limited paired-delta
> J-SAE preserves held-out states and counterfactual differences while
> concentrating the active local Jacobian relative to matched TopK controls.

The base comparator is required before describing a graph statistic as associated
with fine-tuning. A fresh behavioral pass is required before candidate features
or edges may be opened. Biological meaning still requires exact feature
interventions, matched feature/edge nulls, sequence destruction/rescue, and
independent confirmation.
