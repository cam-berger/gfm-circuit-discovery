# Corrections — SpliceBERT acceptor coupling

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Permanent record of what was wrong, bounded, or mis-specified. Nothing here is edited away; wording that
was retracted stays retracted.

**`ACR*` numbers are local to this study and do not continue the donor study's own correction sequence.**
They were originally written with an `AR1-` prefix and renamed for two reasons: that form was
confusable with the donor study's `PREREGISTRATION_AR1_CORRUPTION_OPERATOR_SENSITIVITY` and its `DONA`
arm, and its dash-joined numerals parsed as a stale correction-range assertion under
`tests/test_ledger_consistency.py::test_no_document_understates_the_correction_range`. The rename is
cosmetic; no correction's content changed.

## ACR1 — R1's power gate is pointed at the effect instead of at the nulls, and returns `INSTRUMENT_UNRESOLVED` on a sound run (2026-08-05)

**Outcome in one line: a specification defect in a frozen decision rule, found by executing it, and
deliberately NOT repaired post-hoc.**

`PREREGISTRATION_R1_REVERSE_COUPLING_GATE.md` §6.3 reads: *"Observed `MDE80` on either paired difference
`> SESOI` → `INSTRUMENT_UNRESOLVED`."* On the settled run that clause fires:

| quantity | interval | `MDE80` |
|---|---|---|
| paired difference vs readout control | `[+0.9628, +1.1113]` | `0.106` |
| paired difference vs partner control | `[+0.9334, +1.0816]` | `0.105` |
| control — decoy-acceptor readout | `[-0.0327, -0.0063]` | `0.019` |
| control — decoy-`GT` corruption | `[-0.0005, +0.0270]` | `0.029` |

**Why it is a defect.** A power gate exists so that a null is not an underpowered null. The
discrimination that needs power here is *"is this control null, or is it SESOI-sized?"* — and on the
control arms the instrument resolves that with room to spare (`0.019` and `0.029` against a `0.05`
band). The paired differences are not a null being tested; they are a heavy-tailed effect of size `1.0`
(`coupling.py` records the forward effect as mean `0.436` / median `0.051`), whose bootstrap SD scales
with the effect itself. Demanding `MDE80 ≤ 0.05` there asks the study to resolve a `0.05` shift inside a
`1.0` heavy-tailed effect — precision the question never required and that no achievable `n` on this
cohort would buy.

**Why it was not repaired.** The same preregistration had already been corrected once post-hoc (§7.1:
the `GO` conjunct requiring each control's CI inside `±SESOI`, removed after pilot 2 breached it by
`0.0009`). A **second** post-hoc correction, again in the direction that enables `GO`, would make the
document worthless as a preregistration regardless of how sound each individual argument was. The rule
was executed as frozen and the verdict is reported as it fell.

**What the run establishes despite the verdict.** All eight memorials PASS, including the arithmetic
instrument anchor (forward `Y_D = +0.9436` against AR1's val `+0.9909`, tolerance `±0.15`), zero planted
motifs across 970 constructs, planted-positive, and a permuted-label FPR. `AC1`–`AC3` carry the
measurements with the verdict stated in the same row rather than in a footnote.

**Disposition.** Closing this requires a **new** preregistration (`R1b`) that gates `MDE80` on the
control arms, frozen before it runs, executed on a fresh val draw. It is not an amendment: R1 is frozen
and hash-verifiable, and §6.3 stays as written.

**One thing this cost nothing.** The §7.1 correction turned out to be **immaterial on the settled
cohort**: the decoy-`GT` control landed at `+0.0129 [-0.0005, +0.0270]`, inside `±SESOI`, so the drafted
rule and the frozen rule return the same verdict. The artifact records
`verdict_under_drafted_rule = INSTRUMENT_UNRESOLVED` alongside `verdict`. The disclosure in §7.1 stands
regardless — the rule was still touched after its governing number was seen, and a correction that
happens not to matter is not a correction that was licensed.

## ACR4 — AC2's mechanism clause was not identified by its own panel, and is refuted once the missing cell is run (2026-08-05)

**Published wording, now retracted:** *"The reverse-direction operator's effect is governed by whether
`seq[d+1]` is rewritten, **and by nothing else**… every alt that rewrites it lands at `ρ ≈ 0.99`."*

**The panel could not have identified that.** Across all nine alts in `OPERATORS`, two coincidences held
without exception and neither was noticed:

1. `z[1] == "T"` **iff** `edit_size == 1`. "Keeps the `T`" and "changes one base" were the same set.
2. Every alt that rewrote the `T` **also destroyed the `G`** (all nine have `z[0] != "G"`). "Rewrites the
   `T`" and "destroys both bases" were also the same set.

So the panel contained **two of three reachable cells** — *destroy both* (`ρ≈0.99`) and *destroy `G`
only* (`ρ≈0.67`) — and three different mechanisms predicted identically. The missing cell is *destroy the
`T` while keeping the `G`*, which requires `z[0]=="G"`: `GA` (admissible where `seq[d+2] != "G"`, 37.0%
of val donors) and `GG` (where `seq[d+2] != "T"`, 84.0%). `GC` is excluded because it **is** the
non-canonical U2 donor — 0% admissible under R44's own rule, which is a definition, not a power failure.

**The missing cell, run (R2, `r2_donor_base_decomposition.json`, R1b's own draw-0 cohort):**

| alt | destroys | `edit_size` | n | ρ vs `AC` |
|---|---|---|---|---|
| `AC` | `G` and `T` | 2 | 974 | `1.000` (reference, forced) |
| `AT` | `G` only | 1 | 974 | `0.672 [0.648, 0.699]` |
| `GA` | `T` only | 1 | 360 | `0.857 [0.821, 0.892]` |
| `GG` | `T` only | 1 | 820 | `0.834 [0.807, 0.860]` |

**What survives.** Edit size is **not** the explanation: `AT`, `GA` and `GG` are all one substitution and
split `0.672` against `~0.845`, non-overlapping. The confound noted at (1) is broken in AC2's favour.

**What is refuted.** "Rewriting `seq[d+1]` lands at `ρ ≈ 0.99`" is **false**. `GA` and `GG` rewrite the
`T` and land at `~0.845`. Reaching `1.000` requires destroying **both** bases, not the `T`.

**The corrected reading, which is stronger than the wrong one.** The reverse coupling responds to
destruction of **either** base of the donor `GT`, **strongly sub-additively**: `G` alone `0.672`, `T`
alone `~0.845`, both `1.000` — the singles sum to `1.52`, so the operator saturates. And the **`T`
contributes more than the `G`** (`0.845` vs `0.672`), which is not the ordering the motif's canonical
name suggests. AC2 is rewritten to this and its status is unchanged at established, because the
dose-response itself replicated three times; only its *mechanism attribution* was wrong.

**Residual confound, stated rather than buried.** `GA`/`GG` are the only admissible `T`-only alts and both
**retain the `G`**; a "destroy the `T` and also the `G`" cell is by definition the both-bases arm, so
"keeps `G`" is not separable from "destroys `T` alone" on this operator family. `GA` and `GG` also sit on
different admissibility sub-cohorts (37% vs 84%) and are reported separately for that reason; their
agreement (`0.857` vs `0.834`) is the only cross-check available.

**Where this came from.** The confound was found by an adversarial design review, not by the runs. Both
R1 and R1b passed all their memorials with the defective wording in place, because **no memorial tested
whether the panel identified the claim the panel was quoted for.** That is the same shape as `ACR3`: a
quantity no gate was watching.

## ACR3 — AC3's "19%" was one draw; the selection effect's direction holds but its magnitude does not (2026-08-05)

**Published wording, now corrected:** *"On the `CC`-admissible stratum — 17.7% of sites — `Y_A = +0.8294`
against `+1.0200` on the full cohort: **19% larger** once the R44 admissibility filter is escaped."*

R1b's draw 0 recomputed the same contrast on a fresh cohort and returned `0.9372` on the stratum against
`0.9763` on the full cohort — a **4.0%** shortfall, not 19%.

| run | stratum `Y_A` | full-cohort `Y_A` | shortfall | stratum share |
|---|---|---|---|---|
| R1 | `0.8294` | `1.0200` | **18.7%** | 17.7% |
| R1b draw 0 | `0.9372` | `0.9763` | **4.0%** | 19.6% |

**The direction reproduces; the magnitude does not.** `AC3` is rewritten to claim the direction and to
quote the range `4–19%`, and its status drops from `established` to `bounded`.

**Why this happened, and it is not a surprise in hindsight.** The stratum contrast was carried as a
**memorial**, not as an endpoint. Memorials in this study are checked for pass/fail, never powered: the
stratum is ~19% of the cohort (n≈172–190), it was never given a CI, and no draw-stability requirement
touched it — §5.9 gates the *verdict class*, and this quantity does not enter the verdict. A number that
no gate protects is a number that can drift, and this one drifted by a factor of four and a half.

**The general lesson, which is the reusable part.** R1b added draw stability precisely because R1's
partner control had flipped significance between draws — and that instinct was right, but it was applied
only to the decision layer. Every quantity a claim row quotes needs the same treatment or an explicit
"single-draw, unpowered" label. On R1b's own three draws the partner control turned out extremely stable
(`0.0147 / 0.0156 / 0.0144`, SD `0.0006`) — so **pilot 2's `+0.0284` was the outlier, and the instability
that motivated §5.9 was itself a small-sample artefact.** The memorial that drifted was a different one,
and it drifted because nothing was watching it.

**Disposition.** A stable estimate needs the stratum contrast run as its own powered endpoint with a CI
and multiple draws. **Not pursued** — it bounds a limitation of a superseded secondary arm (`DONA`) and
changes nothing about `AC1` or `AC2`. Recorded as a bounded row, not a live loose end.

## ACR2 — pilot 1's other-acceptor control was mis-constructed and its number must not be quoted (2026-08-05)

Pilot 1 built its site-specificity control from `real_acc[0]` — the first *real annotated* acceptor in
the window — with no distance matching. An audit over 1,200 val exons put those sites at a median offset
of `+157` from the donor against the partner's median gap of `95`, with **36.9% upstream of the donor**.
Its reported `+0.0434 [+0.0023, +0.0947]` therefore mixes partner identity with a large and
signed distance mismatch.

Superseded by the C2 idiom imported from `run_sealed_test.py`, which builds both controls from
**spurious distance-matched motifs** rather than real annotated sites (`matched_decoy_acceptor` for the
readout arm, `matched_decoy_donor` for the corruption arm). **`+0.0434` must not be quoted.** It appears
in `DECISIONS.md` as provenance only, labelled there.

## ACR18 — "sealed chr8/9 not opened and not spent" is study-local and reads as globally virgin (2026-08-17)

**Numbering note.** This entry is `ACR18`, not `ACR5`. Numbers `ACR5`–`ACR17` are allocated to the
uncommitted `AC5`–`AC8` package on `berger/acceptor_coupling` and would collide on merge. The gap is
deliberate and is itself information: it records that thirteen corrections exist on another branch.

`AC1`'s scope column says **"sealed chr8/9 not opened and not spent"**, `AC4`'s says **"chr8/9
unopened"**, and the *What is NOT claimed* section says **"chr8/9 is unspent, deliberately."** Every one
of those is true *of this study* and false as a statement about the split.

chr8/9 was opened, and spent, by the donor arc. `results/splicebert_coupling/sealed_test.json` carries
`"split": "test"` and is the controlling artifact for `C1` (`+0.872 [0.771, 0.969]`), `C2`, and `C3`.
Its scope column says so plainly — "Canonical short introns, sealed chr8/9 split".

**What is wrong:** nothing numeric. No acceptor-study measurement touched chr8/9, and none of `AC1`–`AC8`
is affected. What is wrong is that a reader arriving at the acceptor ledger will conclude the split is
untouched and available as a fresh confirmatory cohort. It is not. It has been read once, by a
behaviour-level assay in the opposite direction that identified no components — which is a *defensible*
basis for reuse, but it must be disclosed and reasoned about, never inherited silently.

**Disposition:** the three passages are amended to say "not opened *by this study*" with a pointer here.
The claims themselves, their numbers, and their evidence levels are untouched.

**Consequence for planning:** any acceptor-side confirmatory package that proposes to spend chr8/9 must
state the `C1` overlap in its protocol. Separately, and measured rather than assumed: the split holds
only **22** `GC-AG` rows across **20** genes, so it cannot confirm a donor-class result at any power.

## ACR19 — every claim in this repository is implicitly scoped to `GT-AG` introns (2026-08-17)

`circuit.eligible` has always filtered `x.sequence[donor_pos:donor_pos+2] == "GT"` (`circuit.py:230`).
`coupling.DONOR_MOTIFS` admits `("GT", "GC")` and `build_coupling_dataset` materialises both, so the raw
cohort holds **3,627** `GC-AG` rows (**1.52%**) — and `eligible` discarded every one before any study saw
them.

**What is wrong:** again, nothing numeric, and this is the good direction of error. No cohort in this
repository is contaminated by the minor class; they are `GT-AG`-pure by construction. But no claim
*covers* the minor class either, and none says so. `AC1`–`AC8`, the `R*` reverse series, and the donor
arc's `C*` claims are all scoped to canonical introns, silently.

**Disposition:** `eligible` gains an optional `donor_motifs` parameter defaulting to `("GT",)`, so every
existing caller is byte-identical — verified by comparing ordered `example_id` lists against a literal
copy of the historical predicate on train, val, and test. Widening it is now an explicit, recorded
choice at each call site rather than an invisible default.

**Scope of this entry:** it is filed here because the acceptor study found it, but the limitation is
repo-wide. The donor study's ledger carries the same implicit scoping and needs its own `R`-numbered
entry; that is the donor study's to write, not this one's.
