# Acceptor mechanism dissection — program design (M1–M5 + capstone)

> **Tier: T2 — design record.** Not a preregistration. Nothing here authorises a run. Each stage
> requires its own frozen study-local protocol, a `--smoke`, and a preflight before it may execute.
> Numbers quoted from the existing record are citations; numbers describing this program's stages are
> design parameters.

**Date:** 2026-08-16 · **Study:** `splicebert_acceptor_coupling` · **Arc:** `M1`–`M5`

## 1. Purpose and success criterion

Build a **standalone mechanism account of the masked-acceptor readout** — address, payload, delivery,
faithfulness — at the depth the donor arc reached, whether or not it mirrors the donor circuit.

Success is a defensible account of how the acceptor readout uses its partner donor, with each element
resting on an intervention rather than an attribution, and one confirmatory package behind it. Mirror
symmetry with the donor arc is an *output* of this program, not its objective.

This explicitly is **not** scoped as a symmetry verdict, a methodology-transfer demonstration, or a
biological claim.

## 2. What the program starts from

**Established or gated (`AC*`, citable):**

- **AC1** — destroying the paired donor `GT` moves the masked-acceptor-`G` margin `+0.9763
  [+0.9060, +1.0526]`; distance-matched decoy readout `-0.0009 [-0.0079, +0.0056]` and decoy partner
  `+0.0147 [+0.0019, +0.0293]`. Behaviour gate, verification tier, chr11/16/19, n=974 / 645 genes.
- **AC2** — operator decomposition on the same draw: destroying the `T` alone recovers `ρ = 0.857
  [0.821, 0.892]` (`GA`) and `0.834 [0.807, 0.860]` (`GG`); destroying the `G` alone recovers
  `0.672 [0.648, 0.699]` (`AT`). The two single-base ratios sum to ≈`1.52`, i.e. **sub-additive with the
  `T` load-bearing**. Post-outcome screen, not confirmatory.
- **AC5** — `L1H12` under within-example token-mean ablation: specific `+0.2447 [+0.1629, +0.3265]`,
  Holm `p = 0.00175`. Held-out within-validation component intervention.
- **AC6** — `L4H11` `+0.0934 [+0.0725, +0.1143]`, `L4H9` `+0.0905 [+0.0708, +0.1103]`, `L4H7`
  `+0.0788 [+0.0593, +0.0982]` under same-example activation noising. Same tier.
- **AC7** — `L3H1` reuse and the distinct-route conjunction are both
  `INCONCLUSIVE_METHOD_DISAGREEMENT`. No binder is identified.
- **AC8** — `L2H15` is `HEALTH_FAILED` under the monotone rule and is excluded from this program.

**Development-tier input from a parallel branch (not verified here, not citable as evidence):** a
region-tiling scan reports the last 9 nt before the `AG` at `+1.98` (val, 388 genes) / `+1.76` (train,
1,051 genes), the partner donor `GT` at `+1.01` / `+0.91`, the 9 nt just inside the donor at
`+0.09` / `+0.06`, and intron body 20–100 nt upstream at ≤ `+0.09`. Its donor arm is numerically
consistent with AC1 (`+1.01` lies inside AC1's interval).

Two caveats govern how this program uses that scan:

1. **It is structurally blind to phase-based addressing.** Its 3-nt window substitutions are
   length-preserving, so every downstream position retains its index and period-3 phase is exactly
   preserved. Over a ~100 nt span such an edit perturbs roughly one cycle in ~33. An "inert intron
   body" is therefore the *predicted* observation under an intact carrier, not evidence against one.
   The scan's null must not be read as contradicting `C34`.
2. **Its regions are not on one scale.** `donorGT` is a 2-substitution edit; the windows are 3.
   `reverse_coupling.matched_operators()` exists precisely so edit size cannot be mistaken for
   computation. No local-vs-long-range ratio may be published without matched edit sizes.

Used correctly, the scan contributes a **spatial prior**: the long-range input is the donor `GT`
dinucleotide, not the donor's 5′ region. Combined with AC2's operator decomposition of that same
dinucleotide, this makes the acceptor-side payload plausibly a very low-information object — which is
the substantive reason to expect the reverse direction to be a better place to attack payload
semantics than the donor side, where `C30` is closed-bounded and `C38` is inconclusive.

## 3. Architecture

### 3.1 Placement

A new arc inside the existing acceptor study — **not** a fourth study directory.

| kind | location |
|---|---|
| protocols | `studies/splicebert_acceptor_coupling/preregistrations/PREREGISTRATION_M*.md` |
| runners | `scripts/splicebert_acceptor_coupling/run_m*_*.py` |
| results | `results/splicebert_acceptor_coupling/` + sibling `*.manifest.json` |
| claims | `AC*` rows in the study ledger |
| decisions | `studies/splicebert_acceptor_coupling/DECISIONS.md` |

Stages are named `M1`–`M5` so they never collide with the `R1`/`R3`/`R5` behaviour-and-screen lineage.

### 3.2 Cohorts and evidence tiers

| stage | question | cohort | tier |
|---|---|---|---|
| **M1** payload | motif *presence* or motif *identity*? | train | development |
| **M2** address | does the span carry geometry the tiling scan cannot see? | train | development |
| **M3** delivery | are `L4H11`/`L4H9`/`L4H7` transporters? | train | development |
| **M4** faithfulness | is the reverse also irreducibly distributed? | train | development |
| **M5** binder / `U_info` | which head binds; what subspace is written? | train, gated on R3d | development |
| **capstone** | end-to-end test of the assembled account | **chr8/9** | **confirmatory** |

Splits are `CouplingConfig`'s: val = chr11/16/19, test = chr8/9 (sealed), train = everything else.

Three properties justify this allocation:

**Train is a legitimate held-out cohort for these components.** `L1H12` and the `L4` triplet were
selected on validation genes — R3's screen ran there, which is why the R3c census excludes 3,873 rows
across 645 historical genes. Testing val-selected components on train is a held-out test of that
selection.

*Disclosure:* the tiling scan already read train, so M1's design is informed by train data and M1 is
**not blind**. Acceptable at development tier; it must appear in M1's protocol.

**Val is not used by this program.** The remaining validation budget is 860 R3c-eligible genes, of
which the R3d adjudication consumes 1,201 rows / 556 genes (ordered gene hash
`7cea952bb066cbfa6855681101f80ecbc1b0971e341ebd50a45a36adeee19988`). R3d and this program therefore do
not compete for cohort.

**One seal spend, at the end.** Development tier constrains the cohort, not the discipline: every
stage still gets a frozen protocol, falsifiable memorials, calibration gates, and a manifest. Only the
capstone reads chr8/9, and only once.

### 3.3 Dependency graph

```
M1 payload ──┐
M2 address ──┼──> capstone (chr8/9, confirmatory)
M3 delivery ─┤
M4 faithful ─┘
M5 binder/U_info ── gated on R3d; joins the capstone ONLY if R3d resolves the binder
```

M1–M4 are mutually independent and binder-agnostic; they may run in any order or concurrently. The
recommended order is **M1 → M2 → M3 → M4**, front-loading payload and address because M2's outcome
determines whether the account contains an addressing element at all.

## 4. M1 — payload

**Question.** The long-range input is the partner donor. Does the acceptor readout depend on the donor
being a *recognisable splice donor*, or only on the canonical `GT` bases being present?

**Dropped arm.** Whether a donor *anywhere* would substitute is already answered by AC1's decoy-partner
control (`+0.0147 [+0.0019, +0.0293]` against a paired `+0.9763`). The signal is address-bound.

### 4.1 Primary — real `GC-AG` introns, not synthetic constructs

An earlier draft of this section proposed writing `GC` over a native `GT` donor. That design is
**withdrawn as the primary**, because the construct does not resemble what it models. Measured on M1's
own row sets: synthetic `GC` constructs carry `+5 G` at **63.2%** and the joint `−2A/−1G/+5G` context at
**28.8%**, against real `GC` donors in the same flank cell at **90.9%** and **75.9%**. Real `GC` donors
carry a compensating consensus that overwriting a `GT` does not create — unsurprising, since a weaker
core motif needs stronger flanking support. A weak response to a synthetic `GC` would therefore be
uninterpretable: it could mean the model does not treat `GC` as a donor, or that it does but requires the
context the construct lacks.

Real `GC-AG` introns are available and were simply never used. `coupling.DONOR_MOTIFS = ("GT", "GC")`, so
the dataset builder admits both, and the raw cohort holds **3,627** `GC-AG` rows (**1.52%**). Every one
is then discarded by `circuit.eligible`, which hardcodes `x.sequence[donor_pos:donor_pos+2] == "GT"`
(`circuit.py:230`). One consequence is worth recording on its own: **no prior cohort in this repository
— R1, R1b, R2, R3, R3c, AC1–AC8 — carries any `GC-AG` contamination.** They are GT-only by construction.

The primary estimand is therefore an intervention on **native** donors:

| arm | native donor | operator | post-intervention state |
|---|---|---|---|
| treatment | `GC` (real `GC-AG` intron) | `GG` | `GG` |
| comparator | `GT` (real `GT-AG` intron) | `GG` | `GG` |

Both arms land on an **identical** post-intervention state, so the only difference between them is the
motif the model was reading beforehand — which is the estimand exactly. Contrast:
`ρ = mean(Y_GC) / mean(Y_GT)`, where `Y = clean − destroyed` on the masked-acceptor-`G` margin.

- **ρ ≈ 1** ⇒ the readout depends on a `GC` donor as much as a `GT` donor; the payload carries donor
  identity, not literal `GT`.
- **ρ ≈ 0** ⇒ destroying a real `GC` donor does not move the readout; the model tracks canonical `GT` only.

### 4.2 Why the operator is `GG`, and why `AC` would have been wrong

`GT` and `GC` share the `G` at `d` and differ at `d+1`. For an alt `XY` the edit size is
`[X≠G] + [Y≠T]` on `GT` and `[X≠G] + [Y≠C]` on `GC`; these are equal **iff `Y ∉ {T, C}`**. So only alts
ending in `A` or `G` are edit-size matched across the two motifs. The obvious choice, `AC`, is
**2 substitutions on `GT` but 1 on `GC`** — it would confound motif class with edit size, which is
precisely what `reverse_coupling.matched_operators()` exists to prevent.

Exhaustive evaluation of all 16 alts on the real train cohort leaves two viable matched candidates:

| alt | edit size | GC rows / genes | GT admissibility | filter interacts with `+3`? |
|---|---|---:|---:|---|
| `AA` | 2 on both | 208 / 175 | 0.547 | **yes** — excludes `+3 G` |
| **`GG`** | **1 on both** | **182 / 153** | **0.837** | no — excludes only `+3 T/C` |

`AG` is inadmissible everywhere (0 of 16 flank contexts): its written register *is* the acceptor motif.

`GG` is frozen as the primary operator. `AA`'s filter excludes `+3 G`, which is 45% of `GT` rows but only
6% of `GC` rows — it selects hard on the very confounder under test. `GG` excludes only `+3 T`/`+3 C`,
rare in both arms. `GG` destroys the distinguishing base while holding the shared `G` fixed; AC2 records
that destroying the `T` alone already recovers `0.834 [0.807, 0.860]` of full destruction on `GT`, so the
partial operator retains most of the signal. `AA` is retained as a secondary robustness arm with its
differential filtering disclosed.

### 4.3 Cohort, and the matching that is not optional

Train split (all chromosomes outside chr11/16/19 and the sealed chr8/9), operator-admissible on both arms:

| | rows | genes |
|---|---:|---:|
| `GC` treatment arm | **182** | **153** |
| `GT` comparator pool | 12,940 | 4,513 |
| within-gene paired subset | 132 | 107 |

**Matching is mandatory.** After `GG` filtering the arms remain grossly unbalanced on `+3`: `GC` is
**A94/G6**, `GT` is **A50/G50**. An unmatched contrast would confound donor class with flanking context.
The comparator set is matched on `+3`, `−1`, `+5`, and the donor–acceptor gap; the realized balance table
is emitted in the artifact, not assumed.

`140` of the `189` `GC` genes also carry a `GT-AG` intron, so a **within-gene paired** contrast is
available on 107 genes (73% of the arm) and is the secondary robustness analysis — it controls
gene-level expression and context exactly, at some cost in n.

Projected gene-clustered half-width, scaling AC1's `0.0733` at 645 genes: **≈0.152** on 153 genes,
**≈0.180** on the 107 paired genes. The question spans `ρ ≈ 0` to `ρ ≈ 1`, so this resolves the headline
comfortably; it does **not** resolve fine gradations near `ρ = 1`, and no such gradation may be claimed.

**Reserves.** Validation holds `79` `GC` rows / `65` genes for a held-out check. The sealed chr8/9 split
holds only **22 rows / 20 genes** — below anything useful — so **M1 cannot be confirmed on the seal**, and
that is an argument for leaving it shut rather than spending it here.

### 4.4 Secondary — the synthetic construct arm

The synthetic `GC`-write arm is retained as a secondary, since it is interventional where the primary is
quasi-observational (donor class is a property of the intron, not something we assign). Its
construct-context gap from §4.1 is reported with any result, and it may not be read as confirming the
primary when it disagrees — the two fail in different directions, so agreement is evidence and
disagreement is a finding.

### 4.5 Required code change

`circuit.eligible` hardcodes `== "GT"`. It gains an optional motif parameter **defaulting to GT-only**, so
that every existing caller keeps byte-identical behaviour; a test asserts that equivalence directly
against the current implementation rather than assuming it.

### 4.6 Memorials

The `GG` arm reproduces AC2's recorded `ρ = 0.834 [0.807, 0.860]` on `GT` within tolerance (an unforced
cross-artifact anchor on the measurement path); `alt == "GT"` on a `GT` donor is an exact no-op returning
zero coupling; `edit_size` equals 1 on both arms for `GG`; the two arms' post-intervention states are
byte-identical at the donor dinucleotide; planted-positive sensitivity; permuted-label FPR. Any memorial
that is architecture-forced is labelled `forced: true` and is not evidence (R7/R21).

## 5. M2 — address

**Question.** Does the donor–acceptor span carry addressing geometry that a length-preserving scan
cannot detect?

**Primary — phase surgery.** Length-changing edits placed in the **20–100 nt upstream intron body** —
the zone the tiling scan reported as inert (≤ `+0.09`). Placing them there is deliberate: it is exactly
the region whose emptiness is contested, and the region a length-preserving scan cannot interrogate.
Edits avoid the last 19 nt before the `AG` (the strong local reader, and the PPT) and the 9 nt flanking
the donor, so the manipulation cannot be confounded with local content damage.

| edit | length change | downstream period-3 phase | interpretation if it degrades |
|---|---:|---|---|
| 1-nt indel | ±1 | shifted by 1 | carrier is read |
| 2-nt indel | ±2 | shifted by 2 | carrier is read |
| **3-nt indel** | ±3 | **unchanged** | **control** — distance, not phase |

The 3-nt arm is what makes the design self-calibrating: it matches the others in length and separation
change while leaving phase intact. Degradation at 1 and 2 but not 3 implicates the carrier;
degradation proportional to length across all three implicates distance; no degradation anywhere means
the carrier is not read in this direction — a genuine asymmetry against `C34`, and its own claim.

Instruments exist on the donor side and are re-pointed, not rewritten:
`build_p3_phase_surgery_cohort`, `match_p3_phase_surgery_contexts`,
`analyze_p3_phase_surgery_confirmation`, `analyze_p3_phase_surgery_controls`.

**Secondary — offset versus absolute (`C32` analogue).** Vary separation with absolute position
matched, and the converse.

*This arm is secondary deliberately.* `ED1` ran this shape and returned `GATES_FAILED`: its raw
architecture × direction interaction was large (`+0.673 [0.629, 0.714]`) but definitionally entangled
with separation, its distance-clean half straddled zero (`+0.0136 [−0.0129, +0.0402]`), and **two of
four calibration gates failed** — `planted_confound_null` at a false-positive rate of `0.285` against a
`0.1` target. M2a's calibration gates must be built and passed before the contrast is read.

**Memorials.** The 0-nt identity check is architecture-forced and must be labelled `forced: true`
(R7/R21) — it is not evidence. The unforced set: independent verification that downstream positions
mod 3 are genuinely unchanged under the 3-nt control; a planted phase effect recovered at known
magnitude; permuted-label FPR.

## 6. M3 — delivery

**The circularity constraint.** AC6 nominated `L4H11`/`L4H9`/`L4H7` under same-example activation
noising, and the record states that same-example noising is **not** delivery. M3 must therefore use a
strictly stronger intervention than the one that selected the heads.

**Primary — clean-state restoration sufficiency.** On the corrupt run (donor `GT` destroyed), patch the
candidate head's output from the clean run. Recovery of the clean margin supports the head carrying the
donor-derived signal; failure to recover places it downstream or parallel. This is a sufficiency test on
the head's output and is independent of the noising screen.

**Gating control — matched-size random head sets.** Three randomly drawn heads, matched for count and
layer distribution, must fail the same restoration test. If random sets restore, the instrument is
measuring capacity rather than identity and the stage is `INCONCLUSIVE`.

**Secondary — source localization.** Where the head reads from at the acceptor token
(`run_a7_source_localization` re-pointed). A transporter should weight the donor position or an
intermediate carrying it.

**Standing interpretive limit.** A supported M3 licenses "carries the donor-derived signal to the
readout." It does not license "delivery" in the donor arc's sense until a binder and a write are
identified — that is M5.

## 7. M4 — faithfulness

Restore `K` components ordered by attribution; plot recovered margin against `K`; test for a knee.

**Component universe includes MLPs.** The donor payload ran through MLP0/1/2 (`C29`, `C31`); an
attention-only curve would misrepresent the computation.

**Exact anchors, both already implemented in `run_a12_faithfulness_size`:** `F(ALL)` equals the clean
margin and `F(∅)` equals the corrupt margin. Both sit on the measurement path and can fail.

**Pre-registered expectation.** Most likely no knee, reproducing `C33`. This is recorded in advance so
the outcome cannot be spun either way afterwards. A replicated absence of a knee in the reverse
direction upgrades `C33` from a property of one direction toward a property of how this checkpoint does
partner association generally — a more valuable result than a knee would be.

## 8. M5 — binder and `U_info` (deferred)

Gated on R3d resolving the binder. Address × payload multiplication site, then the low-rank subspace
written at it (`run_a5_uinfo_replication` re-pointed). If R3d does not resolve `L3H1`, M5 requires its
own binder screen on train before it can proceed, and it does not enter the capstone.

The current structural gap motivates this stage: supported components sit at layer 1 (`L1H12`) and
layer 4 (`L4` triplet), with layers 2–3 — where a binder belongs — occupied by exactly the two
components that returned unresolved (`L3H1`) and health-failed (`L2H15`).

## 9. Capstone

**What is fixed now:** form, cohort, and prerequisites. **What is not:** the endpoints, which depend on
what M1–M4 find. The capstone protocol is written after the map exists and before chr8/9 is read.

**Form.** One preregistered conjunction on chr8/9 with a small fixed endpoint set — approximately: the
long-range donor-`GT` behaviour anchor; M1's verdict direction; M2's verdict; M3 restoration with
random controls null; M4 curve shape. Decision by **test-inverted bounds** with a frozen α allocation,
so the interval inverts the same test that decides each endpoint and disagreement is unreachable by
construction. This carries forward the R3d lesson.

**Prerequisite 1 — correct the seal wording.** chr8/9 was opened once, for `C1`
(`results/splicebert_coupling/sealed_test.json`, "sealed chr8/9 split", `+0.872 [0.771, 0.969]`). The
acceptor ledger's "sealed chr8/9 not opened or spent" is study-local phrasing that reads as globally
virgin. An `ACR` entry must record the true state and the C1 overlap before any capstone cites the
seal. The overlap is defensible — C1 was a behaviour endpoint in the other direction and selected no
components — but it must be disclosed rather than inherited.

**Prerequisite 2 — hard early guard.** The runner refuses every caller-supplied path (model, GTF,
FASTA, output, cohort, preflight artifact) before inspecting any of them, absent a valid preflight
artifact. This is `ACR17`, which cost R3c a run.

**Scope discipline.** M5 enters the capstone only if R3d has landed and the binder is identified. One
seal spend carries only what is solid.

## 10. Shared primitives and testing contract

**Additive extraction only.** New primitives live in `src/gfmcd/circuits/`; acceptor runners call them;
**donor runners are left byte-identical.** Several frozen donor preregistrations cite those runner paths
as evidence and some are hash-pinned, so in-place refactoring would break verification packages. The
duplication is the deliberate price of not touching the frozen record.

**Comparability test.** A fixture test asserts each new primitive reproduces the donor arc's published
numbers on a stored fixture. This recovers estimand comparability with the donor arc without editing
donor code — the property the refactor would have provided.

**Per-stage contract.** Every stage emits an immutable result plus a sibling `*.manifest.json`; no
runner's `--out` defaults to a registered artifact; every claim resolves to a committed artifact.

**Verification before any claim:**

```
uv run --extra splicebert --extra biology --extra dev pytest
uv run python scripts/verify_manifest.py
python3 .claude/skills/gfm-repo-discipline/scripts/check_repo.py
```

with `GFMCD_GTF` and `GFMCD_FASTA` set — without them 13 memorials skip silently and the suite still
reports green.

## 11. Risks

| risk | mitigation |
|---|---|
| M1 hits `C38`'s gene-heterogeneity wall | variance-decomposition preflight with a `NOT_POWERABLE` exit before the contrast runs |
| M1's `GC` arm collides with motif hygiene | carve-out filters `planted_donor_motifs()` by register key: `written` accepted for the `GC` arm only; `upstream`/`downstream` disqualifying on every arm; dispositions serialized per row |
| `GC` admissibility differs from the twins, biasing the contrast | realized admissible fraction reported per arm; contrast restricted to rows admissible in all three arms, so `GC`, `GA`, `GG` are compared on one common row set |
| M2a repeats `ED1`'s calibration failure | calibration gates built and passed before the contrast is read; M2a is secondary to phase surgery |
| M3 is circular on the noising screen | restoration sufficiency, a strictly stronger intervention than the one that selected the heads |
| Random head sets restore in M3 | matched-size random controls gate the stage |
| R3d fails, stranding M5 | M1–M4 are binder-agnostic; the capstone excludes M5 |
| Tiling scan's null misread as refuting `C34` | §2 caveat 1 is normative for this program |
| Capstone spends the seal on a weak account | endpoints frozen only after M1–M4 report; M5 excluded unless resolved |

## 12. What this program does not claim

No biological mechanism. No spliceosomal interpretation. No claim that the reverse direction mirrors or
does not mirror the donor circuit — symmetry is an output, not a hypothesis under test. No minimality
claim for any component set. No promotion of the tiling scan's development-tier numbers to any rung of
the evidence ladder. `SpliceBERT.1024nt` only; checkpoint sha256 `2ad91428…0149d`.
