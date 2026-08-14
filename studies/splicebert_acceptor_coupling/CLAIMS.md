# SpliceBERT acceptor-coupling claim ledger

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
| AC1 | **On the R1b validation cohort, destroying the paired donor `GT` moved the masked-acceptor-`G` margin by `+0.9763 [+0.9060, +1.0526]`, while the distance-matched decoy-readout and decoy-partner effects were `-0.0009 [-0.0079, +0.0056]` and `+0.0147 [+0.0019, +0.0293]`.** The paired differences were `+0.9772 [+0.9058, +1.0560]` and `+0.9616 [+0.8896, +1.0399]`; the control `MDE80` was `0.0196` against the `0.05` band. R1b returned **`GO`**, while the same measurements return **`INSTRUMENT_UNRESOLVED`** under R1's original power clause. These are the observed results of a validation **gate**, not an established or confirmatory claim. | **bounded: behaviour gate — verification tier** (corrected by `ACR8`) | `results/splicebert_acceptor_coupling/r1b_reverse_coupling.json` (+manifest); `preregistrations/PREREGISTRATION_R1b_REVERSE_COUPLING_GATE.md`. Superseded gate: `r1_reverse_coupling.json` (R1, `INSTRUMENT_UNRESOLVED`), retained as provenance. | **Every `[a, b]` in this row is a gene-clustered bootstrap confidence interval on draw `r1b-d0`.** Validation split (chr11/16/19), n=974 paired / 645 genes; **sealed chr8/9 not opened or spent**. The frozen R1 protocol says explicitly that this is “a GATE, not a claim” and that `GO` only authorises a mechanistic battery. R1b inherited that scope. Its forward anchor (`+0.9682`) is a same-cohort validation check, not C1's sealed cohort. The label-permutation memorial also flipped rows rather than gene clusters, contrary to the protocol (`ACR8`), so “all nine memorials pass” is only the artifact's self-report. Behaviour rung only; no component identified. |
| AC2 | **On the already-observed R1b draw-0 validation cohort, the reverse endpoint responded to destruction of either donor base, sub-additively, with a larger response to the `T`-only interventions.** Against the both-bases reference (`AC`, ρ≡1.000), destroying `G` alone gave `ρ = 0.672 [0.648, 0.699]`; destroying `T` alone gave `0.857 [0.821, 0.892]` (`GA`) and `0.834 [0.807, 0.860]` (`GG`). The two single-base ratios sum to `≈1.52`. A related `G`-only dose pattern appears on the R1/R1b cohorts (`AT` 0.679/0.673/0.672, `CT` 0.690/0.675/0.663, `TT` 0.678/0.653/0.641). | **bounded: post-outcome causal operator screen — verification tier** | `r2_donor_base_decomposition.json`; `r1b_reverse_coupling.json` and `r1_reverse_coupling.json` `operators`; corrected by `ACR4` and re-bounded by `ACR12` | R2 was a corrective analysis written after R1b's outcome on the same validation cohort; it is neither held out nor independently confirmatory. It refutes the earlier claim that rewriting `seq[d+1]` alone produces `ρ≈0.99`, but does not isolate a biological contribution of either nucleotide. `GA`/`GG` both retain `G` and occupy different admissibility sub-cohorts. Any cross-direction comparison must match substitution count. Slash-separated figures are cohort values, not intervals. |
| AC3 | **Retracted: the available draws do not establish that the `CC`-admissible stratum has weaker reverse coupling than the full cohort.** The nominal signed shortfall is `+18.7%` in R1, `+4.0%` in R1b draw 0, `+3.2%` in draw 1, and **`-4.8%` in draw 2**, where the stratum value (`1.0534`) exceeds the full-cohort value (`1.0048`). | **retracted: direction contradicted within the recorded draw family** (`ACR7`, superseding the disposition in `ACR3`) | `r1b_reverse_coupling.json` and `r1_reverse_coupling.json`, `memorials.selection_audit_vs_DONA_stratum` | The contrast was an unpowered memorial, not a frozen endpoint, and has no gene-clustered interval or familywise decision. It may be reported only as descriptive selection sensitivity over the four recorded cohorts. `DONA`'s `NOT_INVARIANT` verdict and AC2's operator response do not depend on AC3. |

## AC4 — paired component-screen record

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| AC4 | **Instrument-unresolved: R3 records an exploratory paired mean-ablation screen, but its frozen decision path did not validly establish a component partition or rank identification.** The raw point estimates nominate `L3H1` (`frac_D=0.821`, `frac_A=0.885`), `L2H15` (`0.473` / `0.505`), and `L1H12` (`0.211` / `0.259`) for follow-up. The artifact's nominal 73/17/4/2 `SHARED`/`UNRESOLVED`/`FORWARD_SPECIFIC`/`REVERSE_SPECIFIC` partition is **not a licensed familywise equivalence partition**. | **open: causal candidate screen; inferential partition retracted** (`ACR9`, `ACR11`, superseding the dispositions in `ACR5`–`ACR6`) | `results/splicebert_acceptor_coupling/r3_paired_component_scan.json` (+manifest); `preregistrations/PREREGISTRATION_R3_PAIRED_COMPONENT_SCAN.md` | R3 checked specificity dispersion in only one direction; its permutation null was zero plus scalar noise rather than permuted observed component effects; its planted positive bypassed the actual classification path; `SHARED` used unadjusted percentile intervals; and reverse health scored a different masking policy. The preregistered noising estimand was also absent. Under the frozen rule, failed or incapable memorials make the result `INCONCLUSIVE`. The raw intervention measurements remain useful for candidate generation only. `L5H11`/`L4H2` had near-zero or negative reverse point estimates; that is not evidence that they “contribute nothing.” No MLP, QK, OV, path, mediation, semantic-transfer, or faithfulness result follows. |

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

- **No mediated reverse mechanism.** AC4 nominates heads from an exploratory mean-ablation screen; its
  rank and partition decisions are not licensed. It identifies no mediator, reverse address, payload,
  QK/OV role, MLP write, edge/path wiring, semantic transfer, or faithful component set. The donor-side
  `L3H1` account is **not** shown to run backwards.
- **No symmetry claim.** AC1's validation-cohort forward anchor and reverse endpoint are similar in
  magnitude, but that is neither C1's sealed cohort nor evidence that the directions share components,
  computation, or wiring. R4 has not produced a licensed edge-level result.
- **No sealed-split evidence.** chr8/9 is unspent, deliberately, so a later confirmatory reverse-coupling
  claim has a genuine seal to spend.
- **Nothing about splicing.** AC1–AC4 are model-internal; no external biological assay enters those claims.

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
- **`[a, b]` is a confidence interval.** Spread across draws or seeds is written "range" or "SD".
- **C42 does not transfer to this direction.** Its row scopes it to the acceptor operator; AR1's `DONA`
  returned `NOT_INVARIANT` and AC2 explains why. Never cite C42 in support of a donor-side operator.
- The readout-specificity control reads at a spurious decoy `AG` whose clean margin is `+0.179` against
  a real acceptor's `+3.668`. It sits near floor by construction. This weakness is **inherited from
  C2's own idiom** and applies to both directions equally.
