# SpliceBERT acceptor-coupling claim ledger

Point-of-contact record for the strongest wording currently licensed. One claim, one evidence level, one
controlling artifact set, one scope boundary per row. `AC*` rows govern the **reverse** direction — a
masked **acceptor** readout. They are deliberately kept out of
[`../splicebert_partner_selection/CLAIMS.md`](../splicebert_partner_selection/CLAIMS.md), every row of
which is scoped to a masked **donor**; filing reverse rows there would let `C1`–`C42` silently acquire a
second meaning.

Status vocabulary matches the donor ledger: **established** / **bounded** / **open** / **retracted** /
**closed**.

## Reader-facing evidence catalog

Descriptive IDs are the preferred names in new prose. The `AC*` codes remain compatibility aliases for
existing manifests, frozen records, and citations; they do not encode an evidence level or sequence.

| Evidence ID | Plain-language title | Legacy alias | Current disposition |
|---|---|---|---|
| `acceptor-coupling/behavior/validation-gate` | Reverse-coupling validation gate | `AC1` | bounded gate observation |
| `acceptor-coupling/operator/donor-base-screen` | Donor-base intervention screen | `AC2` | bounded post-outcome screen |
| `acceptor-coupling/selection/cc-stratum-retracted` | `CC`-admissible stratum comparison | `AC3` | retracted |
| `acceptor-coupling/components/exploratory-paired-screen` | Exploratory paired head screen | `AC4` | instrument unresolved; candidates only |
| `acceptor-coupling/components/l1h12-mean-ablation` | Held-out `L1H12` mean-ablation pattern | `AC5` | supported, bounded to this intervention |
| `acceptor-coupling/components/l4-head-noising` | Held-out `L4H11`/`L4H9`/`L4H7` noising patterns | `AC6` | supported, bounded to these interventions |
| `acceptor-coupling/components/route-hypotheses-inconclusive` | Reused-`L3H1` and distinct-route tests | `AC7` | inconclusive by method disagreement |
| `acceptor-coupling/components/l2h15-health-ineligible` | `L2H15` component test | `AC8` | health failed; no scientific verdict |

The identifier format is defined in
[`docs/NOMENCLATURE.md`](../../docs/NOMENCLATURE.md#reader-facing-evidence-identifiers).

## Behavior, operator, and exploratory component records

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| AC1 | **On the R1b validation cohort, destroying the paired donor `GT` moved the masked-acceptor-`G` margin by `+0.9763 [+0.9060, +1.0526]`, while the distance-matched decoy-readout and decoy-partner effects were `-0.0009 [-0.0079, +0.0056]` and `+0.0147 [+0.0019, +0.0293]`.** The paired differences were `+0.9772 [+0.9058, +1.0560]` and `+0.9616 [+0.8896, +1.0399]`; the control `MDE80` was `0.0196` against the `0.05` band. R1b returned **`GO`**, while the same measurements return **`INSTRUMENT_UNRESOLVED`** under R1's original power clause. These are the observed results of a validation **gate**, not an established or confirmatory claim. | **bounded: behaviour gate — verification tier** (corrected by `ACR8`) | `results/splicebert_acceptor_coupling/r1b_reverse_coupling.json` (+manifest); `preregistrations/PREREGISTRATION_R1b_REVERSE_COUPLING_GATE.md`. Superseded gate: `r1_reverse_coupling.json` (R1, `INSTRUMENT_UNRESOLVED`), retained as provenance. | **Every `[a, b]` in this row is a gene-clustered bootstrap confidence interval on draw `r1b-d0`.** Validation split (chr11/16/19), n=974 paired / 645 genes. R1/R1b did not open the masked-acceptor target outcomes on chr8/9; donor-side `C1` had already used those chromosomes (`ACR18`). The frozen R1 protocol says explicitly that this is “a GATE, not a claim” and that `GO` only authorises a mechanistic battery. R1b inherited that scope. Its forward anchor (`+0.9682`) is a same-cohort validation check, not C1's chromosome-held-out cohort. The label-permutation memorial also flipped rows rather than gene clusters, contrary to the protocol (`ACR8`), so “all nine memorials pass” is only the artifact's self-report. Behaviour rung only; no component identified. |
| AC2 | **On the already-observed R1b draw-0 validation cohort, the reverse endpoint responded to destruction of either donor base, sub-additively, with a larger response to the `T`-only interventions.** Against the both-bases reference (`AC`, ρ≡1.000), destroying `G` alone gave `ρ = 0.672 [0.648, 0.699]`; destroying `T` alone gave `0.857 [0.821, 0.892]` (`GA`) and `0.834 [0.807, 0.860]` (`GG`). The two single-base ratios sum to `≈1.52`. A related `G`-only dose pattern appears on the R1/R1b cohorts (`AT` 0.679/0.673/0.672, `CT` 0.690/0.675/0.663, `TT` 0.678/0.653/0.641). | **bounded: post-outcome causal operator screen — verification tier** | `r2_donor_base_decomposition.json`; `r1b_reverse_coupling.json` and `r1_reverse_coupling.json` `operators`; corrected by `ACR4` and re-bounded by `ACR12` | R2 was a corrective analysis written after R1b's outcome on the same validation cohort; it is neither held out nor independently confirmatory. It refutes the earlier claim that rewriting `seq[d+1]` alone produces `ρ≈0.99`, but does not isolate a biological contribution of either nucleotide. `GA`/`GG` both retain `G` and occupy different admissibility sub-cohorts. Any cross-direction comparison must match substitution count. Slash-separated figures are cohort values, not intervals. |
| AC3 | **Retracted: the available draws do not establish that the `CC`-admissible stratum has weaker reverse coupling than the full cohort.** The nominal signed shortfall is `+18.7%` in R1, `+4.0%` in R1b draw 0, `+3.2%` in draw 1, and **`-4.8%` in draw 2**, where the stratum value (`1.0534`) exceeds the full-cohort value (`1.0048`). | **retracted: direction contradicted within the recorded draw family** (`ACR7`, superseding the disposition in `ACR3`) | `r1b_reverse_coupling.json` and `r1_reverse_coupling.json`, `memorials.selection_audit_vs_DONA_stratum` | The contrast was an unpowered memorial, not a frozen endpoint, and has no gene-clustered interval or familywise decision. It may be reported only as descriptive selection sensitivity over the four recorded cohorts. `DONA`'s `NOT_INVARIANT` verdict and AC2's operator response do not depend on AC3. |

## Exploratory paired component screen (legacy alias `AC4`)

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| AC4 | **Instrument-unresolved: R3 records an exploratory paired mean-ablation screen, but its frozen decision path did not validly establish a component partition or rank identification.** The raw point estimates nominate `L3H1` (`frac_D=0.821`, `frac_A=0.885`), `L2H15` (`0.473` / `0.505`), and `L1H12` (`0.211` / `0.259`) for follow-up. The artifact's nominal 73/17/4/2 `SHARED`/`UNRESOLVED`/`FORWARD_SPECIFIC`/`REVERSE_SPECIFIC` partition is **not a licensed familywise equivalence partition**. | **open: causal candidate screen; inferential partition retracted** (`ACR9`, `ACR11`, superseding the dispositions in `ACR5`–`ACR6`) | `results/splicebert_acceptor_coupling/r3_paired_component_scan.json` (+manifest); `preregistrations/PREREGISTRATION_R3_PAIRED_COMPONENT_SCAN.md` | R3 checked specificity dispersion in only one direction; its permutation null was zero plus scalar noise rather than permuted observed component effects; its planted positive bypassed the actual classification path; `SHARED` used unadjusted percentile intervals; and reverse health scored a different masking policy. The preregistered noising estimand was also absent. Under the frozen rule, failed or incapable memorials make the result `INCONCLUSIVE`. The raw intervention measurements remain useful for candidate generation only. `L5H11`/`L4H2` had near-zero or negative reverse point estimates; that is not evidence that they “contribute nothing.” No MLP, QK, OV, path, mediation, semantic-transfer, or faithfulness result follows. |

## Held-out component confirmation (legacy run `R3c v2.1`)

The clean preflight bound a gene-disjoint target before it was opened. The target contains 500 rows / 304
genes from bucket 1 on validation chromosomes 11, 16, and 19, with no bucket-0 or historical-R3 gene
overlap. The compact
[attestation](../../results/splicebert_acceptor_coupling/attestations/paired_donor_component_confirmation_v1.json)
is the active review surface: it pins the exact archived preflight, target, manifests, execution commits,
byte sizes, and SHA-256 values. The independent
[audit](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.audit.json) reports
`AUDIT_PASS_MIXED_SCIENTIFIC_OUTCOME` after two exact decision replays without rescoring the model or
reopening the target cohort.

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first missing bridge |
|---|---|---|---|---|
| AC5 | **Within-example token-mean ablation supports an intervention-specific `L1H12` pattern at the masked acceptor.** Partner, pseudo, and partner-minus-pseudo drops were `+0.2407 [+0.1615,+0.3199]`, `-0.0040 [-0.0229,+0.0149]`, and `+0.2447 [+0.1629,+0.3265]`; the seven-path Holm-adjusted `p` was `0.00175`. | **bounded: head-output component intervention** | [compact attestation](../../results/splicebert_acceptor_coupling/attestations/paired_donor_component_confirmation_v1.json); [decision audit](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.audit.json); [frozen protocol](preregistrations/DRAFT_R3C_ACCEPTOR_HEAD_CONFIRMATION.md) | Same 500-row / 304-gene validation target and 62-row simultaneous-inference family described above. The intervention changes a head output and reads native behavior, but does not identify an address, QK role, post-`W_O` write, directed edge, delivery path, mediator, or mechanism. |
| AC6 | **Same-example corrupt-to-clean activation noising supports intervention-specific patterns for `L4H11`, `L4H9`, and `L4H7`.** Their partner / pseudo / partner-minus-pseudo drops were respectively `+0.0952 [+0.0753,+0.1150]` / `+0.0018 [-0.0030,+0.0065]` / `+0.0934 [+0.0725,+0.1143]`; `+0.0917 [+0.0731,+0.1102]` / `+0.0011 [-0.0035,+0.0058]` / `+0.0905 [+0.0708,+0.1103]`; and `+0.0790 [+0.0603,+0.0976]` / `+0.0002 [-0.0045,+0.0049]` / `+0.0788 [+0.0593,+0.0982]`. Each path's Holm-adjusted `p` was `0.00175`. | **bounded: head-output component interventions** | Same attestation, audit, and frozen protocol as AC5 | The activation source is the aligned corrupt version of the **same example**, separately for paired and pseudo edits. These are component-level noising patterns. No state identity, post-`W_O` write, directed edge, delivery, mediation, or distinct acceptor route is established. |
| AC7 | **Open: neither the reused-`L3H1` conjunction nor the distinct-route conjunction was confirmed.** Both are `INCONCLUSIVE_METHOD_DISAGREEMENT`: their adjusted path p-values pass (`0.02724` and `0.00200`), but their mandatory strict simultaneous-interval conjunctions do not. `L3H1` pseudo endpoints remain unresolved under both mean ablation (`+0.0228 [-0.0245,+0.0700]`) and noising (`+0.0234 [-0.0237,+0.0706]`). The noising route contrast's specific endpoint is unresolved at `+0.0667 [+0.0482,+0.0852]`. | **open: two prespecified component/route paths inconclusive** | Same attestation, audit, and frozen protocol as AC5 | A passing p-value does not override the frozen strict-interval rule. This is not evidence of absence, but it licenses no reused donor component, distinct route, directed edge, delivery, or mediation claim. Supported secondary paths cannot replace these registered conjunctions after outcome inspection. |
| AC8 | **Open: `L2H15_mean_component` is `HEALTH_FAILED`, not supported and not null.** Its diagnostic partner, pseudo, and specific endpoints were `+0.5350 [+0.4040,+0.6659]`, `-0.0018 [-0.0372,+0.0335]`, and `+0.5368 [+0.3985,+0.6752]`. Target health passed `34/64`, but the reviewed bucket-0 ceiling had failed `31/64`, so monotone authorization kept the path suppressed. | **open: health-ineligible prespecified component path** (`ACR16`) | Same attestation, audit, and frozen protocol as AC5 | A target check can shrink the preflight-authorized set but cannot revive a suppressed path. The numerical endpoints are diagnostics and must not be quoted as component support. A new prospective diagnosis would be required to revisit `L2H15`. |

## Reverse-coupling gate draw stability (legacy alias `AC1`; per-draw values, not intervals)

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

- **No mediated reverse mechanism.** AC5 and AC6 support bounded, intervention-specific component
  patterns; AC7 leaves `L3H1` reuse and a distinct route inconclusive, and AC8 is health-ineligible.
  None identifies a reverse address, payload, QK role, post-`W_O` write, directed edge, delivery path,
  mediator, semantic transfer, or faithful component set. The donor-side `L3H1` account is **not** shown
  to run backwards.
- **No symmetry claim.** AC1's validation-cohort forward anchor and reverse endpoint are similar in
  magnitude, but that is neither C1's sealed cohort nor evidence that the directions share components,
  computation, or wiring. R4 has not produced a licensed edge-level result.
- **No chromosome-held-out acceptor evidence.** AC5–AC8 use a gene-disjoint bucket within validation
  chromosomes 11/16/19. They do not score a masked-acceptor target on chr8/9.
- **Nothing about splicing.** AC1–AC8 are model-internal; no external biological assay enters those claims.

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
- **Keep the held-out dispositions separate.** Only the `L1H12` mean-ablation pattern and the
  `L4H11`/`L4H9`/`L4H7` noising patterns are supported, each at the bounded component-intervention
  scope. Reused `L3H1` and the distinct-route conjunction are
  `INCONCLUSIVE_METHOD_DISAGREEMENT`; `L2H15` is `HEALTH_FAILED`. Same-example noising is not delivery.
- **`[a, b]` is a confidence interval.** Spread across draws or seeds is written "range" or "SD".
- **C42 does not transfer to this direction.** Its row scopes it to the acceptor operator; AR1's `DONA`
  returned `NOT_INVARIANT` and AC2 explains why. Never cite C42 in support of a donor-side operator.
- The readout-specificity control reads at a spurious decoy `AG` whose clean margin is `+0.179` against
  a real acceptor's `+3.668`. It sits near floor by construction. This weakness is **inherited from
  C2's own idiom** and applies to both directions equally.
