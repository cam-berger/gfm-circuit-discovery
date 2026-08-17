# SpliceBERT acceptor-coupling claim ledger

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Point-of-contact record for the strongest wording currently licensed. One claim, one evidence level, one
controlling artifact set, one scope boundary per row. `AC*` rows govern the **reverse** direction — a
masked **acceptor** readout. They are deliberately kept out of
[`../splicebert_partner_selection/CLAIMS.md`](../splicebert_partner_selection/CLAIMS.md), every row of
which is scoped to a masked **donor**; filing reverse rows there would let `C1`–`C42` silently acquire a
second meaning.

Status vocabulary matches the donor ledger: **established** / **bounded** / **open** / **retracted** /
**closed**.

## Claim record

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| AC1 | **On the R1b validation cohort, destroying the paired donor `GT` moved the masked-acceptor-`G` margin by `+0.9763 [+0.9060, +1.0526]`, while the distance-matched decoy-readout and decoy-partner effects were `-0.0009 [-0.0079, +0.0056]` and `+0.0147 [+0.0019, +0.0293]`.** The paired differences were `+0.9772 [+0.9058, +1.0560]` and `+0.9616 [+0.8896, +1.0399]`; the control `MDE80` was `0.0196` against the `0.05` band. R1b returned **`GO`**, while the same measurements return **`INSTRUMENT_UNRESOLVED`** under R1's original power clause. These are the observed results of a validation **gate**, not an established or confirmatory claim. | **bounded: behaviour gate — verification tier** (corrected by `ACR8`) | `results/splicebert_acceptor_coupling/r1b_reverse_coupling.json` (+manifest); `preregistrations/PREREGISTRATION_R1b_REVERSE_COUPLING_GATE.md`. Superseded gate: `r1_reverse_coupling.json` (R1, `INSTRUMENT_UNRESOLVED`), retained as provenance. | **Every `[a, b]` in this row is a gene-clustered bootstrap confidence interval on draw `r1b-d0`.** Validation split (chr11/16/19), n=974 paired / 645 genes. R1/R1b did not score the later R1c masked-acceptor target outcomes; chr8/9 itself had already been used by the opposite-direction donor program (`C1`, `ACR18`). The frozen R1 protocol says explicitly that this is “a GATE, not a claim” and that `GO` only authorises a mechanistic battery. R1b inherited that scope. Its forward anchor (`+0.9682`) is a same-cohort validation check, not donor `C1`'s chromosome-held-out cohort. The label-permutation memorial also flipped rows rather than gene clusters, contrary to the protocol (`ACR8`), so “all nine memorials pass” is only the artifact's self-report. Behaviour rung only; no component identified. |
| AC2 | **On the already-observed R1b draw-0 validation cohort, the reverse endpoint responded to destruction of either donor base, sub-additively, with a larger response to the `T`-only interventions.** Against the both-bases reference (`AC`, ρ≡1.000), destroying `G` alone gave `ρ = 0.672 [0.648, 0.699]`; destroying `T` alone gave `0.857 [0.821, 0.892]` (`GA`) and `0.834 [0.807, 0.860]` (`GG`). The two single-base ratios sum to `≈1.52`. A related `G`-only dose pattern appears on the R1/R1b cohorts (`AT` 0.679/0.673/0.672, `CT` 0.690/0.675/0.663, `TT` 0.678/0.653/0.641). | **bounded: post-outcome causal operator screen — verification tier** | `r2_donor_base_decomposition.json`; `r1b_reverse_coupling.json` and `r1_reverse_coupling.json` `operators`; corrected by `ACR4` and re-bounded by `ACR12` | R2 was a corrective analysis written after R1b's outcome on the same validation cohort; it is neither held out nor independently confirmatory. It refutes the earlier claim that rewriting `seq[d+1]` alone produces `ρ≈0.99`, but does not isolate a biological contribution of either nucleotide. `GA`/`GG` both retain `G` and occupy different admissibility sub-cohorts. Any cross-direction comparison must match substitution count. Slash-separated figures are cohort values, not intervals. |
| AC3 | **Retracted: the available draws do not establish that the `CC`-admissible stratum has weaker reverse coupling than the full cohort.** The nominal signed shortfall is `+18.7%` in R1, `+4.0%` in R1b draw 0, `+3.2%` in draw 1, and **`-4.8%` in draw 2**, where the stratum value (`1.0534`) exceeds the full-cohort value (`1.0048`). | **retracted: direction contradicted within the recorded draw family** (`ACR7`, superseding the disposition in `ACR3`) | `r1b_reverse_coupling.json` and `r1_reverse_coupling.json`, `memorials.selection_audit_vs_DONA_stratum` | The contrast was an unpowered memorial, not a frozen endpoint, and has no gene-clustered interval or familywise decision. It may be reported only as descriptive selection sensitivity over the four recorded cohorts. `DONA`'s `NOT_INVARIANT` verdict and AC2's operator response do not depend on AC3. |

## AC4 — paired component-screen record

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| AC4 | **Instrument-unresolved: R3 records an exploratory paired mean-ablation screen, but its frozen decision path did not validly establish a component partition or rank identification.** The raw point estimates nominate `L3H1` (`frac_D=0.821`, `frac_A=0.885`), `L2H15` (`0.473` / `0.505`), and `L1H12` (`0.211` / `0.259`) for follow-up. The artifact's nominal 73/17/4/2 `SHARED`/`UNRESOLVED`/`FORWARD_SPECIFIC`/`REVERSE_SPECIFIC` partition is **not a licensed familywise equivalence partition**. | **open: causal candidate screen; inferential partition retracted** (`ACR9`, `ACR11`, superseding the dispositions in `ACR5`–`ACR6`) | `results/splicebert_acceptor_coupling/r3_paired_component_scan.json` (+manifest); `preregistrations/PREREGISTRATION_R3_PAIRED_COMPONENT_SCAN.md` | R3 checked specificity dispersion in only one direction; its permutation null was zero plus scalar noise rather than permuted observed component effects; its planted positive bypassed the actual classification path; `SHARED` used unadjusted percentile intervals; and reverse health scored a different masking policy. The preregistered noising estimand was also absent. Under the frozen rule, failed or incapable memorials make the result `INCONCLUSIVE`. The raw intervention measurements remain useful for candidate generation only. `L5H11`/`L4H2` had near-zero or negative reverse point estimates; that is not evidence that they “contribute nothing.” No MLP, QK, OV, path, mediation, semantic-transfer, or faithfulness result follows. |

## AC5–AC8 — R3c v2.1 held-out component record

R3c v2.1 used a clean, target-bindable bucket-0 preflight and then opened the gene-disjoint bucket-1
target on validation chr11/16/19. The immutable package is the
[`v2.1 preflight`](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.json)
([manifest](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.manifest.json))
and [r3c_acceptor_head_target_v2_1.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.json)
([r3c_acceptor_head_target_v2_1.manifest.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.manifest.json)).
Their SHA-256 values are respectively `c0c9306fe894ce73bfd981a14c85e2aec6660da33f01d9716cee4327b3ff2d3e`,
`4a53f96b365c7bed227c8e0c995cee65a022620994e8898bab988b354fce41bb`,
`ebd4bfd8c10cc0ccb218b3f1ceca634605903d8afbb5617de312087100627cc6`, and
`dc644901224227f03e683f7416587dccf3eedd700052b211089242b9b8d69cbe`. `ACR16` and `ACR17`
preserve the v1 health-revival and v2 authorization-order failures that preceded this package. An
independent [r3c_acceptor_head_target_v2_1.audit.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.audit.json),
SHA-256 `0363d02dced48d0d1a8ba56f1862be4a0cbb0b82eb508f9aeb87b8c61b05edb3`, reports
`AUDIT_PASS_MIXED_SCIENTIFIC_OUTCOME` after two exact replays without rescoring the model, reopening
the target cohort, or accessing chr8/9.

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| AC5 | **On R3c's held-out within-validation bucket-1 target, within-example token-mean ablation supports an intervention-specific `L1H12` component pattern at the masked acceptor.** The partner, pseudo, and partner-minus-pseudo drops were `+0.2407 [+0.1615,+0.3199]`, `-0.0040 [-0.0229,+0.0149]`, and `+0.2447 [+0.1629,+0.3265]`; `L1H12_mean_component` was **`SUPPORTED`** with seven-path Holm-adjusted `p=0.00175`. | **bounded: held-out within-validation component intervention** | [r3c_acceptor_head_target_v2_1.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.json) ([r3c_acceptor_head_target_v2_1.manifest.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.manifest.json); [r3c_acceptor_head_target_v2_1.audit.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.audit.json)); bound [`r3c_acceptor_head_preflight_v2_1.json`](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.json) ([manifest](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.manifest.json)); [frozen v2.1 protocol](preregistrations/DRAFT_R3C_ACCEPTOR_HEAD_CONFIRMATION.md) | n=500 rows / 304 genes in bucket 1, gene-disjoint from bucket 0 after excluding historical R3 genes; validation chr11/16/19 only. Intervals are whole-gene simultaneous confidence intervals over the 62-row family. The endpoint masks only `next_acceptor_pos-1`; no masked-donor endpoint exists. This identifies a bounded response to this head-output intervention, not an address, QK/OV role, write, delivery path, mediator, or mechanism. |
| AC6 | **On the same target, same-example corrupt-to-clean activation noising supports intervention-specific component patterns for `L4H11`, `L4H9`, and `L4H7`.** Partner / pseudo / partner-minus-pseudo drops were respectively `+0.0952 [+0.0753,+0.1150]` / `+0.0018 [-0.0030,+0.0065]` / `+0.0934 [+0.0725,+0.1143]`; `+0.0917 [+0.0731,+0.1102]` / `+0.0011 [-0.0035,+0.0058]` / `+0.0905 [+0.0708,+0.1103]`; and `+0.0790 [+0.0603,+0.0976]` / `+0.0002 [-0.0045,+0.0049]` / `+0.0788 [+0.0593,+0.0982]`. Each named component path was **`SUPPORTED`**, with Holm-adjusted `p=0.00175`. | **bounded: held-out within-validation component noising interventions** | [r3c_acceptor_head_target_v2_1.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.json) ([r3c_acceptor_head_target_v2_1.manifest.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.manifest.json); [r3c_acceptor_head_target_v2_1.audit.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.audit.json)); bound [`r3c_acceptor_head_preflight_v2_1.json`](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.json) ([manifest](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.manifest.json)); [frozen v2.1 protocol](preregistrations/DRAFT_R3C_ACCEPTOR_HEAD_CONFIRMATION.md) | Same n=500 / 304-gene validation target and simultaneous family as AC5. The source is the aligned corrupt version of the **same example**, separately for paired and pseudo edits. These are component-level noising patterns only. They do not show that the heads deliver a state, mediate a path, or form a distinct acceptor route. |
| AC7 | **Open: R3c did not confirm either the reused-`L3H1` conjunction or the distinct-route conjunction.** Both are **`INCONCLUSIVE_METHOD_DISAGREEMENT`**: their Holm-adjusted path p-values pass (`0.02724` and `0.00200`), but the mandatory strict simultaneous-interval conjunctions do not. For `L3H1`, the pseudo endpoints remain unresolved under both mean ablation (`+0.0228 [-0.0245,+0.0700]`) and noising (`+0.0234 [-0.0237,+0.0706]`). For the route contrast, the noising delivery-gap partner endpoint is `+0.0676 [+0.0500,+0.0852]`, while its specific endpoint is unresolved at `+0.0667 [+0.0482,+0.0852]`. | **open: two prespecified paths inconclusive by method disagreement** | [r3c_acceptor_head_target_v2_1.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.json) ([r3c_acceptor_head_target_v2_1.manifest.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.manifest.json); [r3c_acceptor_head_target_v2_1.audit.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.audit.json)); [frozen v2.1 protocol](preregistrations/DRAFT_R3C_ACCEPTOR_HEAD_CONFIRMATION.md) | A passing p-value does not override the frozen strict-CI concordance rule. This is not evidence of absence, but it licenses **no reused donor component, distinct route, delivery, or mediation claim**. The supported AC5/AC6 paths may not be substituted post-outcome for these failed registered conjunctions. |
| AC8 | **Open: `L2H15_mean_component` is `HEALTH_FAILED`, not supported and not null.** Its target scientific endpoints were partner `+0.5350 [+0.4040,+0.6659]`, pseudo `-0.0018 [-0.0372,+0.0335]`, and specific `+0.5368 [+0.3985,+0.6752]`, and its current bucket-1 health check passed `34/64`; nevertheless, the reviewed v1 bucket-0 ceiling failed `31/64`, so the monotone authorization rule kept the path suppressed. | **open: health-ineligible prespecified component path** | [r3c_acceptor_head_target_v2_1.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.json) ([r3c_acceptor_head_target_v2_1.manifest.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.manifest.json); [r3c_acceptor_head_target_v2_1.audit.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.audit.json)); bound [`r3c_acceptor_head_preflight_v2_1.json`](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.json) ([manifest](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.manifest.json)); corrected by `ACR16` | Target health can shrink the preflight-authorized set but cannot revive a path suppressed by the reviewed development ceiling. The numerical endpoints are preserved as diagnostics and must not be quoted as component support. A new diagnosis and prospectively frozen successor would be required to revisit `L2H15`. |

## AC9 — terminal frozen R1c→R5 adjudication

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| AC9 | **Closed — `INSTRUMENT_UNRESOLVED` before the R1c target.** The prospectively frozen R1c v4 validation preflight was a genuine `PREFLIGHT_GO`: all 13 gates passed on 400 rows / 321 genes, and independent replay reproduced its row algebra, 4,000-draw inference, 10,000-draw raw/trimmed gene bootstrap, and all 1,000 calibration trials. The exact frozen R1c→R5 handoff nevertheless rejects that authentic package: R1c `/3` rows serialize six `margin_levels` (`P_clean`, `P_under_D`, `P_under_N`, `X_clean`, `X_under_D`, `X_under_N`), while the bound R5 consumer asserts that the same field has exactly the three `P_*` keys before applying R1c's intended three-key common-leg projection. Because repairing the consumer or rewriting the rows would alter the frozen joint stack after observing the validation outcome, the one-run/no-redesign stop rule closes the chain. | **closed: instrument/package handoff — terminal `INSTRUMENT_UNRESOLVED`** (`ACR19`) | [`r1c_reverse_behavior_preflight_v4.json`](../../results/splicebert_acceptor_coupling/r1c_reverse_behavior_preflight_v4.json), SHA-256 `1b27c4132c71a576c435b7b71f32831af9a44fbca6325931f35716ae4133cbdd`; [manifest](../../results/splicebert_acceptor_coupling/r1c_reverse_behavior_preflight_v4.manifest.json), SHA-256 `7412f3159e82f6ab0bbaa95e2796f41aa038eef4233efa7f09ec65867e39c45c`; verifier attestation SHA-256 `04be60c2239c98eefb01521f7a8a94a68f4d2cc4db5db8f1dddb5622f7378ce1`; [frozen R1c protocol](preregistrations/DRAFT_R1C_SEALED_REVERSE_BEHAVIOR_CONFIRMATION.md); `tests/test_acceptor_mechanism_cohort.py::test_reviewed_r1c_preflight_is_authentic_but_frozen_r5_handoff_rejects_its_six_margin_rows` | Validation chr11/16/19 only. This program did not open or score the R1c masked-acceptor target outcomes on chr8/9; the chromosomes were not globally virgin (`ACR18`). The R1c target preflight seal stayed unset, every shared R1c package pin stayed unset, and no R5a, R5b, or R5c scored artifact exists. R3d Path A was the sole admitted component prior in the frozen design; it was never activated downstream. `L2H15`, low-rank, H7, the natural donor-class candidates, early heads, and MLPs did not enter any gate. This is neither positive nor null evidence about paired-donor behavior, L3H1 reading/writing, L4H11+H9 delivery, or a biological mechanism. |

## AC10–AC11 — natural donor-class modifier record

These rows answer a narrower, orthogonal question: whether a natural minor-class `GC-AG` donor
modifies the masked-acceptor behavior and L3H1 attention geometry relative to a matched canonical
`GT-AG` donor. They do not repair AC9 or complete the canonical read→write→delivery chain.

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| AC10 | **On the donor-class-development-held-out validation cohort, masked-acceptor coupling was substantially weaker for natural `GC-AG` donors than for matched `GT-AG` donors.** The prespecified ratio was `rho=0.2571`, with one-sided upper bound `0.5402` below the registered `0.60` boundary. | **bounded: held-out behavior verification** (`ACR20`, `ACR21`) | Recovered first-run [`m6_validation_confirmation.json`](../../results/splicebert_acceptor_coupling/m6_validation_confirmation.json), SHA-256 `80150013b084cb659c7e3a0e1739573fd1a12e0c95c8cc433067ec9929d4051a`; [manifest](../../results/splicebert_acceptor_coupling/m6_validation_confirmation.manifest.json), SHA-256 `4e470908b93885133a817a12eee2b074c8008f3d0b3aafbfb4ac596ae27216df`; original [M6 protocol](preregistrations/PREREGISTRATION_M6_VALIDATION_CONFIRMATION.md), SHA-256 `da326e4d…f46a` | Validation chr11/16/19; 67 matched pairs anchored by 56 treatment genes, the entire M6 reserve. The first run was clean at `42954a6` and numerically replicated the train direction, but donor class is quasi-observational; matching covers selected flanks, gap, and donor transcript support, not conservation, expression, or splicing efficiency. The checkpoint was not byte-bound, the stored cohort hash is order-insensitive and omits comparator pairing, the bootstrap does not preserve every matched pair, and row-level draws are absent. Therefore this is verification-grade bounded evidence, not clean confirmation. |
| AC11 | **On the same cohort, L3H1 placed substantially less donor-directed attention mass on natural `GC-AG` donors than on matched `GT-AG` donors.** The prespecified attention ratio was `0.3755`, with one-sided upper bound `0.5809` below the registered `0.70` boundary. | **bounded: held-out attention-geometry verification** (`ACR21`) | Same recovered first-run M6 artifact, manifest, and original protocol as AC10 | Same 67 pairs / 56 treatment genes and the same provenance/inference limits as AC10. This establishes a bounded attention-geometry modifier only. It does not show that L3H1 is necessary, what it writes after `W_O`, whether it mediates the behavior, or whether L4H11/H9 delivers that state. |

## Terminal successor status

The canonical R1c→R5 program is closed under AC9. It is not awaiting another pin or run:

| Program | Cohort boundary | Strongest licensed status |
|---|---|---|
| **R1c v4** | The only scored successor cohort was the 400-row validation preflight on chr11/16/19. The planned scientific target remained sealed on chr8/9. | **Validation instrument passed; frozen downstream handoff failed.** This is AC9's terminal instrument outcome, not behavior evidence. No repair, rerun, or target pin is authorised under this closed program. |
| **R5a/R5b/R5c** | No scored cohort was opened. | **Not run because the mandatory frozen predecessor handoff failed.** Their source remains a model-free record of the intended causal adjudication, not evidence about the model. |

The earlier R3b document remains a never-frozen, unrun design record; its reciprocal donor endpoint and
head+MLP family are not the current acceptor-only successor. R3c v2.1 creates only AC5–AC8; it does
not alter AC1's bounded gate status, AC2's post-outcome status, AC3's retraction, or AC4's unresolved
partition.

## Integrated interpretation

The combined record has one coherent progression. Canonical `GT-AG` masked-acceptor dependence is
large but bounded at the validation-gate level (AC1); both donor bases affect it sub-additively in a
post-outcome operator screen (AC2); and the original stratum claim is retracted (AC3). R3 nominates
components without a valid partition (AC4). R3c then supports bounded `L1H12` mean-ablation and
`L4H11/H9/H7` noising patterns (AC5–AC6), while leaving `L3H1` reuse and a distinct route inconclusive
(AC7) and `L2H15` health-ineligible (AC8). The prospectively frozen R1c→R5 mechanism program stops at
its authentic package handoff before target science (AC9).

The natural donor-class arc adds a modifier, not a missing mechanism. AC10–AC11 show that `GC-AG`
donors have both weaker reverse coupling and lower L3H1 donor attention than matched `GT-AG` donors
on the M6 validation cohort. M1–M5 supply adaptive development candidates—early L0/L1 heads and an
`MLP1` asymmetry—but no registered transport equivalence, post-`W_O` write, mediation, or delivery
result (`ACR22`). The two arcs must therefore not be composed into a synthetic successful circuit.
The strongest complete conclusion is: **reverse masked-acceptor dependence and natural donor-class
heterogeneity reach behavior/attention evidence; causal construction of the margin remains unresolved.**

## AC1 draw stability — per-draw values, NOT intervals

Kept out of the AC1 row on purpose. R40 records what happens when across-draw spread is written in
`[a, b]` notation: C41's per-draw ranges were readable as confidence intervals and understated the
uncertainty. **Nothing in this table is an interval.** Each column is one resampled cohort.

| quantity | `r1b-d0` | `r1b-d1` | `r1b-d2` | SD |
|---|---|---|---|---|
| `Y_A` reverse coupling | 0.9763 | 1.0987 | 1.0048 | 0.064 |
| control — decoy-`AG` readout | −0.0009 | −0.0011 | +0.0031 | 0.0024 |
| control — decoy-`GT` corruption | +0.0147 | +0.0156 | +0.0144 | 0.0006 |
| paired difference vs readout | 0.9772 | 1.0998 | 1.0016 | 0.065 |
| paired difference vs partner | 0.9616 | 1.0831 | 0.9903 | 0.064 |
| control `MDE80` | 0.0196 | 0.0191 | 0.0213 | — |
| **verdict class** | **GO** | **GO** | **GO** | — |

The three draws are **resampled from one validation pool and overlap by construction**. They measure
stability under cohort selection; they are **not** independent replications, do not widen the
population, and license nothing about generalisation.

## What is NOT claimed

- **No mediated reverse mechanism.** AC5 and AC6 establish only bounded, intervention-specific component
  patterns. AC7 leaves `L3H1` reuse and the distinct route inconclusive, while AC8 leaves `L2H15`
  health-ineligible. Nothing identifies a mediator, reverse address, payload, QK/OV role, MLP write,
  delivery, edge/path wiring, semantic transfer, or faithful component set. The donor-side `L3H1`
  account is **not** shown to run backwards.
- **No symmetry claim.** AC1's validation-cohort forward anchor and reverse endpoint are similar in
  magnitude, but that is neither C1's sealed cohort nor evidence that the directions share components,
  computation, or wiring. R4 has not produced a licensed edge-level result.
- **No R1c target evidence.** AC9 closes this adjudication before the masked-acceptor target outcomes
  on chr8/9 were scored; it does not reserve or authorise another attempt under the same program.
  chr8/9 had already been used by donor `C1`, so this is an endpoint-outcome holdout rather than a
  globally virgin chromosome seal (`ACR18`). R3c's held-out bucket is within validation chr11/16/19.
- **Nothing about splicing.** AC1–AC11 are model-internal; no external biological assay enters those records.

## Interpretation guardrails

- **R1b's `GO` is a gate verdict, not claim promotion.** The same R1b measurements become
  `INSTRUMENT_UNRESOLVED` under R1's rule, and the earlier R1 run also returned that verdict. Quote both
  readings and the frozen “GATE, not a claim” scope. The row-level permutation defect means the ninth
  memorial is not protocol-conformant (`ACR8`).
- **R1b identified no component; AC4 only nominates candidates.** R3's raw mean-ablation values make
  `L3H1`, `L2H15`, and `L1H12` reasonable follow-ups, but the nominal rank and shared/specific partition
  are retracted after the calibration-path audit (`ACR9`). Before R3, the only head screen at an
  acceptor readout had been withdrawn as coordinate-invalid
  (`../splicebert_head_context_atlas/DECISIONS.md`, 2026-07-19: *"the reciprocal acceptor computation is
  now untested, not null"*).
- **Keep R3c's four dispositions separate.** Quote only `L1H12_mean_component` and the
  `L4H11`/`L4H9`/`L4H7` noising component paths as supported. Quote `L3H1_reused_component` and
  `distinct_acceptor_route_candidate` as `INCONCLUSIVE_METHOD_DISAGREEMENT`, and
  `L2H15_mean_component` as `HEALTH_FAILED`. Same-example noising is not delivery. R5a/R5b/R5c remain
  model-free draft methods, and the supported secondary paths cannot replace their registered upstream
  gates.
- **AC9 is terminal, not a request to repair and retry.** The authentic R1c validation package passed
  its own frozen verifier, but the frozen downstream consumer rejects it. Changing either side after
  observing that package would create a new method. Under the one-run goal, the only licensed outcome
  is `INSTRUMENT_UNRESOLVED`; no R1c target or R5 stage may be inferred from the preflight estimates.
- **AC10–AC11 are modifier evidence, not a reverse circuit.** They neither repair AC9 nor license
  early-head, MLP, address, post-`W_O` write, delivery, mediation, or canonical-margin construction.
- **`[a, b]` is a confidence interval.** Spread across draws or seeds is written "range" or "SD".
- **C42 does not transfer to this direction.** Its row scopes it to the acceptor operator; AR1's `DONA`
  returned `NOT_INVARIANT` and AC2 explains why. Never cite C42 in support of a donor-side operator.
- The readout-specificity control reads at a spurious decoy `AG` whose clean margin is `+0.179` against
  a real acceptor's `+3.668`. It sits near floor by construction. This weakness is **inherited from
  C2's own idiom** and applies to both directions equally.
