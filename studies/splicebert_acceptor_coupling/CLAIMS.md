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

## Live claims

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| AC1 | **Destroying a donor's `GT` moves its partner acceptor's masked-`G` margin by `+0.9763 [+0.9060, +1.0526]`, partner- and site-specifically.** The same corruption read at a distance-matched decoy `AG` gives `-0.0009 [-0.0079, +0.0056]`; a distance-matched decoy `GT` corruption read at the exon's own acceptor gives `+0.0147 [+0.0019, +0.0293]`; paired differences `+0.9772 [+0.9058, +1.0560]` and `+0.9616 [+0.8896, +1.0399]`; control `MDE80` `0.0196` against the `0.05` band. Measured on the same introns whose forward direction defines C1, forward anchor `+0.9682` reproducing AR1's val `+0.9909`. All **nine** memorials PASS and the preregistered verdict is **`GO`**. Cohort-selection stability is a separate statement — see the stability table below, which is **not** in interval notation. | **established: behaviour — verification tier, NOT confirmatory** (val only; every endpoint seen before freezing; see scope) | `results/splicebert_acceptor_coupling/r1b_reverse_coupling.json` (+manifest); `preregistrations/PREREGISTRATION_R1b_REVERSE_COUPLING_GATE.md`. Superseded gate: `r1_reverse_coupling.json` (R1, `INSTRUMENT_UNRESOLVED`), retained as provenance. | **Every `[a, b]` in this row is a gene-clustered bootstrap confidence interval on draw `r1b-d0`.** Validation split (chr11/16/19), n=974 paired / 645 genes; **sealed chr8/9 not opened and not spent by this study** (globally the split WAS opened, for the donor arc's `C1`/`C2`/`C3`; see `ACR18`). Verification tier by construction: the design was established by disclosed val pilots whose endpoints were seen, and R1b's `MDE80` clause was relocated knowing that clause was what blocked `GO` (R1b §7). Under R1's original clause 3 this same run reads `INSTRUMENT_UNRESOLVED` (`verdict_under_R1_rule`); both readings are in the artifact. Behaviour rung only. **Identifies no component; says nothing about whether L3H1 transfers.** |
| AC2 | **The reverse coupling responds to destruction of *either* base of the donor `GT`, strongly sub-additively, and the `T` contributes more than the `G`.** Against a both-bases reference (`AC`, ρ≡1.000): destroying the **`G` alone** gives `ρ = 0.672 [0.648, 0.699]`; destroying the **`T` alone** gives `0.857 [0.821, 0.892]` (`GA`) and `0.834 [0.807, 0.860]` (`GG`). The two singles sum to `≈1.52` against a both-bases value of `1.000`, i.e. the operator **saturates**. The `G`-only dose-response replicated on three cohorts (`AT` 0.679/0.673/0.672, `CT` 0.690/0.675/0.663, `TT` 0.678/0.653/0.641). | established: operator dose-response and single-base decomposition (verification tier, held-out val) | `r2_donor_base_decomposition.json`; `r1b_reverse_coupling.json` and `r1_reverse_coupling.json` `operators`; corrected by `ACR4` | **Supersedes the original wording**, which attributed the split to whether `seq[d+1]` is rewritten; that was **not identifiable** from its own nine-alt panel and is **refuted** by `GA`/`GG` (`ACR4`). **Any symmetry statement between the two directions must match substitution counts** — `reverse_coupling.matched_operators` enforces it. Reconciles AR1's `DONA` (`ρ≈0.62`) and E11's `Retention_A` (3% vs 22%). **Residual confound:** `GA`/`GG` are the only admissible `T`-only alts and both retain the `G`, so "keeps `G`" is not separable from "destroys `T` alone"; they also sit on different admissibility sub-cohorts (37%/84%). Slash-separated figures are per-cohort values, **not** intervals. |
| AC3 | **AR1's `DONA` arm measured the reverse coupling on a stratum where it is smaller, so `DONA` understated it — but the size of the shortfall is not pinned down.** On the `CC`-admissible stratum (`seq[d-1] != "G"`; 17.7–19.6% of sites, the only sites `DONA` could measure) `Y_A` runs **below** the full-cohort value on every draw examined, by **4% to 19%** (R1: `0.8294` vs `1.0200`; R1b draw 0: `0.9372` vs `0.9763`). | **bounded — direction established, magnitude unstable across draws** (corrected by `ACR3`) | `r1b_reverse_coupling.json` and `r1_reverse_coupling.json`, `memorials.selection_audit_vs_DONA_stratum` | The **direction** reproduces; the **magnitude does not**, and the originally published "19%" was a single draw. Quote the range, never the point. Does not retract `DONA`: its verdict (`NOT_INVARIANT`) stands and is corroborated by AC2. A stable estimate would need the stratum contrast run as its own powered endpoint, which no protocol here did. |

## AC4 — the paired component partition

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| AC4 | **The two directions share their top components and differ in a small periphery.** On 974 exons / 645 genes measured at **both** readouts in one process, mean-ablating each of the 96 heads and taking the paired within-direction-normalised difference `Δ(c) = frac_A(c) − frac_D(c)` (Holm over 96) partitions them **73 `SHARED` / 17 `UNRESOLVED` / 4 `FORWARD_SPECIFIC` / 2 `REVERSE_SPECIFIC`**. **`L3H1` is rank 1 in BOTH directions** (`frac_D = 0.821`, `frac_A = 0.885`) and `L2H15` rank 2 in both (`0.473` / `0.505`), with `L1H12` third (`0.211` / `0.259`). The direction-specific set is small and layer-localised: `FORWARD_SPECIFIC` `L2H4` (`Δ=−0.124`), `L4H2` (`−0.096`), `L5H11` (`−0.087`), `L1H8` (`−0.077`); `REVERSE_SPECIFIC` `L2H5` (`Δ=+0.186`, the largest of any component) and `L4H12` (`+0.092`). Two of C18's three forward transporters (`L5H11`, `L4H2`) contribute **nothing** to the reverse direction (`frac_A` `−0.006` / `−0.013`); the third (`L4H0`) is `SHARED`. All nine memorials PASS, including the forward arithmetic anchor (`L3H1` max at `0.821 > L2H15 0.473 > 0`, reproducing `donor_heads.json`'s `sane` check) and a specificity null 39× tighter than the signal (SD `0.0026` vs `0.1026`). | established: causal screen / paired component partition — **verification tier** | `results/splicebert_acceptor_coupling/r3_paired_component_scan.json` (+manifest); `preregistrations/PREREGISTRATION_R3_PAIRED_COMPONENT_SCAN.md` | **`UNRESOLVED` is NOT `SHARED`** — `L3H1`'s own `Δ = +0.0640 [−0.0039, +0.1318]` straddles zero but reaches past the `0.05` band, so **the top head of both directions is not certified equivalent**, only "rank 1 in both with no evidence of difference at this precision". Same for `L2H15`. **Powered to identify rank 1 and to detect a set; NOT powered to order ranks 2+** (median `|Δ| = 0.016`; ordering a single adjacent pair needs ~10⁴ paired exons). **MLP writes are out of scope** — no mean-ablation primitive exists for them and `HCA18` reports MLP writes dominating its acceptor-cluster signals, so coverage is incomplete by construction. Mean-ablation only; the noising second estimand (prereg §5.4) has **not** been run, and forward `L2H15` differs 45× between the two idioms. Validation split; **chr8/9 unopened by this study** (globally opened for `C1`; see `ACR18`). **Attribution RANKS; intervention DECIDES — this licenses no mediation claim in either direction.** |

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

- **No mechanism.** Nothing here identifies a head, a mediator, an address, a payload, or a delivery
  path for the reverse direction. The donor circuit's `L3H1` account is **not** shown to mirror, and no
  outcome here predicts that it will.
- **No symmetry claim.** AC1 and C1 are comparable in magnitude on the same introns, but that is a
  *behavioural* co-occurrence measured under one operator pairing. Whether the two directions are one
  computation or two is exactly what the mirrored mechanistic battery would test, and it has not run.
- **No sealed-split evidence.** chr8/9 is unspent *by this study*, deliberately. It is NOT globally
  virgin: the donor arc opened it for `C1`/`C2`/`C3` (`results/splicebert_coupling/sealed_test.json`,
  `"split": "test"`). Reuse for an acceptor endpoint is defensible -- that read was behaviour-level, in
  the opposite direction, and identified no components -- but it must be disclosed in any protocol that
  spends it, never inherited. See `ACR18`. Measured, not assumed: the split holds only 22 `GC-AG` rows
  across 20 genes.
- **Nothing about splicing.** Model-internal throughout; no external biological assay enters this study.

## Interpretation guardrails

- **The gate returned `GO` on R1b and `INSTRUMENT_UNRESOLVED` on R1, from the same measurements.** The
  difference is entirely the `MDE80` clause, and R1b's relocation of it was written knowing that clause
  was what blocked `GO` (R1b §7, `ACR1`). Quote AC1's verdict **with** `verdict_under_R1_rule`, never
  the `GO` alone. What licenses the `GO` is not the rule change on its own but that the relocated gate
  clears its band by ~2.5× on all three draws while R1's could not have been cleared at any `n`.
- **`GO` authorises designing the mirrored mechanistic battery. It identifies nothing.** No component,
  no head, no mediator. In particular it does **not** show that `L3H1` transfers, and the one prior
  attempt to screen heads at an acceptor readout was withdrawn as coordinate-invalid
  (`../splicebert_head_context_atlas/DECISIONS.md`, 2026-07-19: *"the reciprocal acceptor computation is
  now untested, not null"*).
- **`[a, b]` is a confidence interval.** Spread across draws or seeds is written "range" or "SD".
- **C42 does not transfer to this direction.** Its row scopes it to the acceptor operator; AR1's `DONA`
  returned `NOT_INVARIANT` and AC2 explains why. Never cite C42 in support of a donor-side operator.
- The readout-specificity control reads at a spurious decoy `AG` whose clean margin is `+0.179` against
  a real acceptor's `+3.668`. It sits near floor by construction. This weakness is **inherited from
  C2's own idiom** and applies to both directions equally.
