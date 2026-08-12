# Decision log — SpliceBERT positional proxy modifiability

> **Tier: T2 — provenance.** Prospective study; no model outcome exists and no post-outcome decision has been
> made.

The frozen choices belong in [`PREREGISTRATION.md`](PREREGISTRATION.md). Add a dated entry here only after
data or diagnostics force a decision, amendment, stop, or scope change. Each entry must name the evidence
available at the time, the alternatives considered, the affected files, and the effect on claim scope. This
log never silently amends the preregistration or a claim ledger.

---

## 2026-08-11 — why this successor exists

**Decision.** Open a new study whose primary estimands are **between-checkpoint** contrasts under a corpus
manipulation, rather than continue to pursue a standalone within-checkpoint `Ψ` on either existing panel.

**Evidence available at the time.** No model outcome. This decision rests entirely on two properties of the
existing designs, both of which are design facts rather than measurements, and neither of which depends on any
number produced by the closed positional boundary-coherence study.

1. **The real panel cannot support a standalone `Ψ` at the `0.05` SESOI.** `Ψ_real` is a within-checkpoint
   matched-minus-mismatched contrast on authentic modules. Resolving it against the frozen `0.05` materiality
   scale on its own is a sample-size problem this program has not solved: a `Ψ` that is statistically resolved
   but sub-SESOI licenses only a rough, assay-scoped statement, and that is the ceiling of the design, not a
   result that a larger cohort of the same kind converts into mechanism.
2. **The sequence-only designed panel cannot identify the carrier.** At the frozen 120-nt donor→acceptor span,
   120 is a multiple of 3, so designed-register agreement and joined-window period-3 **amplitude** are
   algebraically the same quantity: matched registers add constructively, mismatched registers add
   destructively. A translation-invariant periodicity detector with no positional coordinate reproduces the
   entire `Ψ_design` ranking. No composition-balance objective can separate them, because they are not two
   correlated features.

**Alternatives considered, and why each was rejected.**

| Alternative | Rejected because |
|---|---|
| Scale the real panel until `Ψ_real` clears `0.05` on its own | The ceiling is the design's, not the cohort's; and even a material `Ψ_real` would remain a behaviour-rung statement with no carrier attribution. |
| Report `Ψ_design` from the designed panel as carrier evidence | Property 2. `Ψ_design` is exactly `S_0`, the `r = 0` member of the register family, and it is confounded by construction. It is retained as a **diagnostic only** and is retired as an identifier. |
| Change the span so that its length is not a multiple of 3 | Changes the assay geometry, breaks comparability with every inherited window and coordinate convention, and moves the acceptor off the frozen position — a new instrument, not a fix to this one. |
| Reopen or relax the closed study's register instrument | That study is closed and read-only, and its development readings are blacklisted as calibration. Reopening a sealed instrument to reach a different verdict is precisely what the seal exists to prevent. |

**What the successor does instead.** Two co-primary contrasts, each covering the other's blind spot:

- `T_real = Ψ_L1 − Ψ_L2` on fresh authentic modules. Because it is a difference **between checkpoints on
  identical sequences**, every sequence-only confound — including the amplitude coupling above — enters both
  checkpoints identically and cancels. It cannot attribute the change to the positional carrier.
- `T_clock = Ω3_L1 − Ω3_L2` on the designed panel under the identical-sequence carrier intervention. `Ω3` is
  exactly zero under any readout that does not depend on the imposed carrier register `r`, so it is
  carrier-specific by construction. It cannot speak about authentic modules.

The end-to-end claim requires both. Either may stand alone at a lower rung without upgrading the other.

**Affected files.** [`PREREGISTRATION.md`](PREREGISTRATION.md) (new, frozen 2026-08-11); this log.

**Effect on claim scope.** None yet, in either direction. No claim ledger row exists for this study, and no
existing claim in any ledger is changed, bounded, or supported by this decision. In particular this decision
says nothing about `C43`, `C44`, or the closed study's status, and the predecessor's reserved `P3_*` outcome
labels remain unearned and unavailable.

## 2026-08-12 — base-checkpoint behavioural decomposition: the positional surrogate is not sufficient

**Evidence available.** Base checkpoint only. No lineage was trained; no seal was spent. All
readings use one pipeline: masked-donor margin, `AG→AC` acceptor corruption, gene-clustered
intervals, authentic validation parents (chr11/16/19; chr8/chr9 never opened).

**Positive control, run first.** The pipeline reproduces the annotated-phase preference on
authentic chimeric joins: `Ψ = +0.1747 [+0.1091, +0.2474]`, 324 cells, matched `+0.1953` against
mismatched `+0.0193` / `+0.0219`. This is a fresh measurement in this pipeline, not a re-audit of
`C43`, and it is smaller than `C43`'s `+0.3901 [+0.3553, +0.4252]`; the two are **not pooled**.
Without this control every comparison below would have been cross-pipeline and confounded.

**Finding 1 — designed register does not reproduce annotated phase.** Three constructs, two
operators, both settings of chimerism: circular rotation within gene `−0.0310 [−0.0611, −0.0066]`;
paired indel within gene `−0.0398 [−0.0747, −0.0027]`; paired indel across genes
`−0.0280 [−0.0463, −0.0110]`. All small, **negative**, roughly thirteen times too small, intervals
excluding both zero and the annotated value. Manufacturing the arithmetic of a frame does not
manufacture the preference.

**Finding 2 — raw period-3 energy does not carry `Ψ` either.** On annotated-phase cells, matched
joins carry *less* joined-span period-3 power than mismatched (162.5 vs 167.5, ratio 0.970); `Ψ` is
flat across power tertiles (`+0.278` / `+0.284` / `+0.206`); and after regressing standardised
power out of the coupling, `Ψ = +0.2552 [+0.1914, +0.3196]` — unchanged from `+0.2563`. A
translation-invariant periodicity detector does not explain the behaviour.

**Consequence.** `Ψ` requires the frame relation **and** authentic sequence content together. A
period-3 positional coordinate is a frame-arithmetic device and cannot know which frame the
*content* prefers; a positionless energy detector cannot see frame at all. So the learned carrier
is **not sufficient** to account for `C43`. It may still be one input, supplying the offset over
which content statistics are compared.

**What this does NOT say.** It does not refute `C43`, which measured something real and
replicated. It does not touch `C36`, which establishes the carrier is causally *used* in
phase-aligned coupling — only the carrier's *sufficiency* for `C43` is newly bounded. It is base
checkpoint, one readout, one geometry, one span, and three constructs are not an exhaustive
operator space: a construct that also moved the correlated content features would be expected to
reproduce `Ψ`, and that is the next experiment rather than a refutation of this one.

**Effect on claim scope.** No claim row is added. This is base-checkpoint behavioural evidence
about what `Ψ` responds to, and it licenses no mechanism, route, or biological statement.

**Decision.** The lineage-training programme does **not** proceed. Its end-to-end chain was
corpus structure → learned positional surrogate → masked-token behaviour, and the middle term is
now shown insufficient to carry the behaviour on its own. Training paired lineages would test a
link the behaviour does not use unaided. The frozen step-6 stop rule — no training without a
passing relocation preflight — is what surfaced this before fifteen checkpoints were spent.

**Successor, well-posed and cheaper.** Hold annotated phase fixed and vary, one at a time, the
features that co-vary with it: codon usage bias, exonic splicing enhancer content, wobble-position
composition, in-frame stop load. Whichever carries `Ψ` is the real target. No training, no carrier
surgery, no sealed cohort. Finding 2 above is its first step and has already eliminated one
candidate.

## 2026-08-12 (later) — necessity test: the carrier IS in the causal path. Prior entry's
## recommendation is amended.

**What was missing.** The entry above tested only SUFFICIENCY — whether a designed register
offset reproduces `Ψ` (no), and whether raw periodic energy explains it (no). It then concluded
that training "would test a link the behaviour does not use unaided". **NECESSITY was never
tested, and that conclusion was stronger than the evidence supported.**

**The test.** Delete the period-3 band from the learned position table (`C8`'s operator, rFFT bins
339–345) and re-measure `Ψ` on the same annotated-phase cells. Three power-matched control bands
of equal width, chosen with the frozen exclusion of periods near 3.0, 1.5 and 6.0 so no period-3
harmonic can serve as a control.

| condition | coupling | `Ψ` | `Ψ` retention |
|---|---|---|---|
| intact | +0.0492 | **+0.25625** | — |
| **period-3 band deleted** | +0.0125 | **+0.00501** | **2.0%** |
| control band 95–101 | +0.0448 | +0.24507 | 96% |
| control band 96–102 | +0.0486 | +0.23822 | 93% |
| control band 19–25 | +0.0497 | +0.30097 | 117% |
| control mean | — | — | **102%** |

**Deleting the carrier abolishes `Ψ`; deleting equal-power bands elsewhere does not touch it.**
The specificity is not subtle.

**The R12/R13 caveat, stated rather than waved away.** Period-3 deletion also cut general coupling
to 25% of intact, and this repository forbids reading a mechanism from a destroyed behaviour. Two
facts bear against dismissing the result on that ground: the control bands retained BOTH coupling
(91–101%) and `Ψ` (102%), so the damage is band-specific rather than generic; and `Ψ`'s loss
(98%) is much larger than coupling's (75%). It remains confounded, and the clean version is a
DOSE-GRADED attenuation — if `Ψ` falls faster than coupling across doses, necessity is
established without destroying anything.

**The picture that now fits all the evidence.** The carrier is **necessary but not sufficient**.
Not sufficient: three designed constructs across two operators give ≈ −0.03 where authentic
annotated phase gives +0.175. Necessary: removing it abolishes `Ψ` with band specificity. It is
one required input to a content-dependent frame-compatibility computation — supplying the offset
over which content statistics are compared — and not the whole mechanism.

**Amendment to the previous decision.** The prior entry said the training programme's middle term
is "insufficient to carry the behaviour" and should not proceed. The first half stands; **the
inference drawn from it does not**. The chain corpus → carrier → behaviour requires the carrier to
be causally IN the path, and necessity is exactly that. Training is therefore not futile.

**Revised gate before any lineage is trained.** Run the dose-graded attenuation to separate
carrier necessity from general degradation. If `Ψ` falls faster than coupling, the programme
proceeds on a supported premise. If both fall together, the necessity reading collapses and the
prior entry's decision stands unamended.

## 2026-08-12 (final) — the mechanism question is answered: necessary, but not the register code

**Two clean results, both on ANNOTATED-phase cells where `Ψ` is real (+0.256), both at doses that
leave the behaviour intact so R12/R13 does not apply.**

**1. Dose-graded necessity — PASSES.** Scaling the period-3 band:

| band | dose | coupling retention | `Ψ` retention | ratio |
|---|---|---|---|---|
| P3 | 0.75 | 141% | 61.7% | 0.437 |
| **P3** | **0.50** | **114.3%** | **10.7%** | **0.093** |
| P3 | 0.25 | 40.7% | 3.3% | 0.080 |
| control | 0.50 | 76.7% | 111.2% | 1.451 |

At half attenuation the general behaviour is fully intact — 114% of baseline — while `Ψ` collapses
to 10.7%. A power-matched control band does the opposite. Ratio 0.087 for P3 against 1.348 for the
control. **The carrier is necessary, established without destroying anything.**

**2. Relocation — the preference does NOT move.** Same cells, carrier register imposed at
r ∈ {−1,0,+1} with the token sequence identical across r:

| dose | `Ω3` | 95% CI | argmax matches prediction |
|---|---|---|---|
| 25% | +0.00136 | [−0.00389, +0.00576] | 1/3 |
| 50% | +0.00395 | [−0.00800, +0.01540] | 1/3 |

The preferred class stays at δ=0 for every r. The single "match" is r=0, which is FORCED — both
handedness conventions predict δ=0 there — so it counts for nothing. Coupling is preserved
throughout (0.049–0.058 against 0.049 intact). Relocation does, however, WEAKEN the ridge:
G0 falls +0.2418 → +0.2168 (r=−1) → +0.1513 (r=+1).

**What this settles.** The period-3 carrier is **necessary for `Ψ` but does not encode which
register is preferred.** Disrupting it degrades the computation; relocating it does not relocate
the preference. Combined with the earlier results — designed register offsets do not reproduce
`Ψ`, and raw period-3 energy does not explain it — the carrier is a **required supporting input to
a content-dependent frame-compatibility computation**, not a positional representation of the
phase relation itself.

**Consequence for the end-to-end goal.** The intended chain was corpus → learned positional
surrogate → masked-token behaviour, licensing "SpliceBERT uses a trainable positional surrogate
for a real biological relation". The carrier is in the path, but it is **not a surrogate for the
relation** — it does not carry the register identity. A corpus manipulation that moved the carrier
would be expected to degrade `Ψ`, not to re-target it, and degradation is not evidence of a
positional surrogate.

**Decision.** The lineage-training programme does not proceed on its original claim. What the
evidence supports is narrower and worth stating on its own: `Ψ` requires the carrier AND authentic
content together, and the carrier's role is enabling rather than representational.

**Successor.** Identify what carries the register identity — hold annotated phase fixed and vary
codon usage bias, exonic splicing enhancer content, wobble composition and in-frame stop load one
at a time. No training, no seal. If a training study follows, its endpoint should be `Ψ`
degradation under carrier-targeted corpora, not ridge relocation.

## 2026-08-12 — paired-lineage pilot: the behaviour IS trainable, but not through the carrier

**Authorization.** The user directed this run explicitly after being shown that its original
premise is falsified. Recorded here as their decision, not an inference from the evidence.

**Scope, first, because it bounds everything below.** ONE SEED PER ARM. The replicate for a corpus
intervention is the training SEED, not the evaluation gene. Every interval here is a
gene-clustered interval over evaluation-cell resampling for one lineage pair. **None covers
training-seed variability and none may be quoted as a training-process claim.** 2,000 steps,
3,440 windows per arm.

**Gate passed before training was allowed.** Checkpoint `2ad91428…0149d` verified. The base arm
reproduced the record: `Ψ = +0.25625` against +0.256, coupling `+0.04918` against +0.049, 479
cells. Had it drifted, a `ΔΨ` would have meant nothing.

**Design.** L1 inserts `3k` exonic bases and deletes `3k` deep in the intron — length preserved,
reading frame unchanged. L2 uses `3k+1` / `3k+2`, so the downstream exon's frame no longer
continues the upstream one. Paired indel, not rotation, because rotation scrambles codon coherence
and measurably destroys the effect. Same source windows, order, masking RNG, optimizer schedule,
step count and token budget; **the only difference between corpora is the offset mod 3.**

**Result.**

| arm | `Ψ` | 95% CI |
|---|---|---|
| base | +0.25625 | reproduced |
| **L1 phase-preserving** | **+0.64208** | [+0.5212, +0.7715] |
| **L2 phase-decoupled** | **+0.29748** | [+0.2137, +0.3948] |

| contrast | estimate | gene-clustered CI |
|---|---|---|
| **`ΔΨ` = L1 − L2** | **+0.3446** | acceptor [+0.2593, +0.4335] · donor [+0.2576, +0.4413] |
| L1 − base | +0.3858 | [+0.2878, +0.4935] |
| L2 − base | +0.0412 | [−0.0277, +0.1224] — covers zero |

**Reading.** Phase-preserving continuation **strengthened** the boundary preference by 2.5×.
Phase-decoupled continuation left it at baseline. So controlled continued pretraining does change
the masked-token boundary preference, in the predicted direction.

**And it did not go through the carrier.** Period-3 band power of the learned position table:
L1 198,761, L2 198,592 — **0.085% apart**. MLM health identical (0.4736 / 0.4739), so no
differential degradation. The corpus manipulation moved the behaviour while leaving the carrier
untouched.

**This confirms the base-checkpoint finding rather than contradicting it.** The carrier is
necessary for `Ψ` but does not encode register identity; a corpus manipulation therefore changes
`Ψ` through content, not through the carrier. The pilot shows exactly that pattern independently.

**Status against the original two-part goal.** Part one — training changes the boundary preference
on fresh genomic sequences — is **demonstrated at pilot strength**. Part two — that the change
depends on the learned period-3 coordinate — is **refuted**: the coordinate did not change.
The end-to-end "trainable positional surrogate" claim is therefore not available, and the honest
statement is narrower: *the boundary preference is trainable by corpus phase structure, through a
route that is not the period-3 positional carrier.*

**Also worth recording.** Overall coupling went negative in both arms (L1 −0.134, L2 −0.004
against base +0.049) while `Ψ` rose. Continued pretraining changed the model well beyond this
endpoint, and that is not characterised here.

**What a confirmatory version needs.** At least six paired seeds, so intervals cover training-seed
variability; the frozen exclusion and balance gates; and a preregistered endpoint. Nothing in this
entry substitutes for that.

## 2026-08-12 (same day, bounding the entry above) — "the carrier did not change" is not what was measured

**What forced this entry.** The entry above reads equal period-3 **band power** across arms
(198,761 vs 198,592, 0.085% apart) as showing that the corpus manipulation "did not go through the
carrier", and states that part two of the original goal is **refuted** because "the coordinate did
not change". That is stronger than the readout supports, and the gap is demonstrable rather than a
matter of taste.

**The demonstration.** `period3_band_power` is `sum(|rFFT|²)` over bins 339–345 of the mean-centred
position table. Applying this study's own carrier operator — `positional_spectrum.phase_rotate_band`,
the same function the relocation test used — at 90° to a random table leaves band power **exactly**
unchanged (relative change `1.2e-16`) while the table is materially different. **A carrier
relocation is invisible to this readout by construction**, because a phase rotation preserves every
coefficient's modulus. Memorialised in `tests/test_lineage_pretraining.py::
test_period3_band_power_is_blind_to_a_phase_rotation` so the limitation cannot be forgotten by the
next reader of the number.

**What the pilot's band-power reading therefore does and does not license.**

* **Licensed:** the period-3 carrier's *amplitude* is unchanged, equally, in both arms — so the arms
  are not separated by a change in carrier strength, and neither arm degraded or amplified the
  carrier.
* **Not licensed:** "the carrier did not change", "the coordinate did not change", or that the
  corpus effect demonstrably bypasses the carrier. A phase-only change, or a change in how
  downstream heads *read* an unchanged carrier, would produce exactly the observed reading. Neither
  was measured here.

**Effect on claim scope.** None — the entry above adds no claim row either, and no ledger row cites
it. What changes is the wording available for it: the honest statement is *"the boundary preference
moved without any change in period-3 carrier amplitude"*, not *"without going through the carrier"*.
The `Ω3` relocation readout, which **is** phase-sensitive by construction, is what could settle the
route; it was not run on the trained arms.

**Affected files.** This log; `results/splicebert_positional_proxy_modifiability/RESULTS_INDEX.md`
(its carrier paragraph is bounded to match); `tests/test_lineage_pretraining.py` (new memorial).
The artifact `lineage_pilot_v1.json` is **not** edited — it records `period3_band_power` and its
bins, which is a correct measurement; only the prose reading of it was overstated.

## 2026-08-12 (three-arm carrier dependence) — the pilot's rise is GENERIC CONTINUATION, and the
## carrier question is answered on both halves

**Authorization and scope change.** The user directed a three-arm engineering pilot in place of a
re-run of the two pilot arms. Recorded here as their decision. It is an **engineering pilot** with
**two training seeds**, one lineage triple each: *observed in two exploratory lineage triples*, not
demonstrated. Every interval below covers **evaluation-parent resampling only**; across-seed figures
are a **range** over two draws and bound nothing.

**Artifact.** `results/splicebert_positional_proxy_modifiability/lineage_carrier_dependence_v1.json`
(+ sibling manifest), from
`scripts/splicebert_positional_proxy_modifiability/run_lineage_carrier_dependence.py`. Trained
weights and every per-cell array are persisted outside the repository with sha256 in the artifact.

**Gate passed first.** Checkpoint `2ad91428…0149d` verified; base `Ψ = +0.25625` against the
recorded +0.256, coupling +0.04918 against +0.049, 479 cells — identical to the pilot's base reading
in every printed digit.

**Design changes, each closing a hole.** (1) a third arm **L0**, native continued pretraining on the
*unedited* windows, same crop plan, masking draws, schedule, step count and token budget; (2)
**`k = 10`** so the arms differ by 1–2 bases in 30 rather than 1–2 in 3 — this **reduces and does not
remove** the edit-size confound (75,930 against 79,707 inserted bases, 5.0% apart); (3) a
**phase-sensitive** complex-Fourier carrier readout; (4) a **difference-of-differences** endpoint
against power-matched control bands; (5) health probes at every dose — coupling distribution, masked
MLM top-1, masked donor-G top-1, masked acceptor-G top-1.

**Finding 1 — the pilot's central reading does not survive the control arm.**

| arm | seed 20260812 | seed 20260813 |
|---|---|---|
| **L0 native, no edit** | **+0.65616** [+0.52317, +0.80166] | **+0.73215** [+0.58131, +0.89119] |
| L1 phase-preserving | +0.57863 [+0.46520, +0.70041] | +0.60421 [+0.46663, +0.75517] |
| L2 phase-decoupled | +0.38725 [+0.29050, +0.49293] | +0.45628 [+0.33578, +0.58509] |

`L0 > L1 > L2` in both seeds, and `L1 − L0` is negative and excludes zero in both (−0.0775
[−0.1371, −0.0186]; −0.1279 [−0.2003, −0.0618]). **The rise in `Ψ` under continued pretraining is
generic continuation, not phase-preservation.** The edit-matched `L1 − L2` contrast is still positive
and material (+0.1914 [+0.1330, +0.2469]; +0.1479 [+0.0869, +0.2096]), so residue mod 3 does move
`Ψ` — but between two edited arms that both sit *below* native, i.e. the phase-decoupled edit
suppresses the generic rise rather than the phase-preserving edit causing it.

**This bounds the 2026-08-12 pilot entry.** Its sentence "Phase-preserving continuation
**strengthened** the boundary preference by 2.5×" attributed to phase what a no-edit arm reproduces
and exceeds; it is withdrawn as an attribution and survives only as "continued pretraining on these
windows raised `Ψ`". Its sentence "Phase-decoupled continuation left it at baseline" is withdrawn
twice over: it was a non-detection read as an equivalence, and at `k = 10` `L2 − base` is in fact
resolved and material (+0.1310 [+0.0516, +0.2188]; +0.2000 [+0.0994, +0.3051]). The pilot artifact
is **not** edited — its numbers are correct measurements of a two-arm design; what is withdrawn is
the causal attribution the prose drew from them.

**Finding 2 — training did not move the carrier, and this time it was measured.** Complex rFFT bins
339–345 of the mean-centred position table, per bin and per hidden dimension: every arm-vs-base
per-bin rotation is under **1.1°** (under **0.009 nt** of equivalent translation), coherence stays
above 0.998, band power sits within +0.52% to +0.85% of base, per-bin energy total variation is
≤ 0.0044 and per-dimension ≤ 0.0021 with top-8 dimension overlap 1.000. L1-vs-L2 rotation is ≤ 0.34°.
The arms are separated by neither carrier amplitude, nor carrier phase, nor carrier redistribution —
which is a **stronger and differently-supported statement** than the pilot's phase-blind band power,
and it is the readout the 2026-08-12 bounding entry above said was missing.

**Finding 3 — every checkpoint's `Ψ` depends on the carrier, the trained ones included.** `Ψ`
retention under period-3 attenuation at dose 0.50: base 10.7%, L0 14.7%/18.1%, L1 **15.8%/17.8%**,
L2 16.1%/19.5%; the three power-matched control bands at the same dose retain 97–117% everywhere.
**L1's elevated preference collapses under carrier attenuation the way the base's does — training
did not move the behaviour onto a different substrate.**

**Finding 4 (primary endpoint) — the training contrast itself is carrier-dependent.** The difference
of differences `[Ψ_L1 − Ψ_L2]_intact − [Ψ_L1 − Ψ_L2]_attenuated` at dose 0.50 is **+0.1623
[+0.1029, +0.2186]** and **+0.1295 [+0.0741, +0.1868]** for period-3 — a gap retention of 15.2% and
12.4% — against `DoD` between −0.025 and +0.018 and gap retention 89–113% for every control band. At
the milder 0.75 dose the gap largely survives (77.1%, 91.6%), so the dependence appears between the
0.75 and 0.50 rungs.

**R12/R13, applied.** At dose 0.50 the trained arms hold masked MLM at 97–99% of intact and the
acceptor-AG motif probe at 0.797–0.814 (intact 0.785–0.827), while coupling **magnitude** retention
is 28–38% — band-specifically, since control bands at the same dose hold magnitude at 98–109%. That
rung is read. **The 0.25 and 0.00 rungs are not read**: the AG probe falls to 0.58–0.72 and coupling
magnitude to 15–25%, which is a destroyed behaviour.

**This also qualifies the 2026-08-12 (final) base-checkpoint entry.** That entry's dose table is
reproduced here to the digit from a committed runner (61.7% / 10.7% / 3.3%, control 111.2%, signed
coupling 114.3% and 76.7%). But its reading of the 114% signed-mean ratio as "the general behaviour
is fully intact" and its conclusion "established without destroying anything" are **too strong**: at
that same dose the base's coupling magnitude retention is 34.8% and its acceptor-AG probe falls
13 points, from 0.7474 to 0.6159. The **band specificity is untouched** — every control band retains
`Ψ`, magnitude, MLM and the motif probe — so the supportable statement is that period-3 attenuation
degrades `Ψ` *faster* than the general behaviour, band-specifically, with the general behaviour
measurably degraded too.

**`Ω3` was not run**, and no capability preflight for it was run. Imposed relocation is a different
question from trained phase movement and neither stands in for the other.

**Effect on claim scope.** No claim row is added and none is changed. `PPM1` is untouched. This is a
two-seed engineering pilot; it licenses no mechanism, route, or biological statement, and the
`ΔΨ`-as-reinforcement reading of the earlier pilot is withdrawn as an attribution.

**Affected files.** This log; `results/splicebert_positional_proxy_modifiability/RESULTS_INDEX.md`;
`studies/splicebert_positional_proxy_modifiability/RESULTS.md`; `README.md`; `results/MANIFEST.csv`
and `results/MANIFEST.json`; new `src/gfmcd/splicing/carrier_phase.py`, new runner, and new/extended
tests in `tests/test_carrier_phase.py` and `tests/test_lineage_pretraining.py`.

## 2026-08-12 — pilot wording tightened, and a permanent preflight rule

**Three wording corrections, adopted.** The pilot entry above was still overclaiming in three
places. The licensed summary is now:

> In one exploratory matched lineage pair, phase-preserving continuation exceeded phase-decoupled
> continuation by `ΔΨ = +0.3446` on a held-out-from-training — but previously examined —
> development panel. The point difference was primarily associated with an increase under L1;
> L2 versus base remained unresolved. Total period-3 band power was nearly identical across the
> trained arms, but carrier phase, internal redistribution, and causal use were not measured.

| was | is |
|---|---|
| "demonstrated at pilot strength" | **"observed in one exploratory lineage pair"** — the bootstrap covers evaluation-parent uncertainty, never training-seed variability |
| "L2 left it at baseline" | **unresolved** — its interval includes zero but also effects above the 0.05 SESOI. Non-detection, not equivalence |
| "carrier amplitude unchanged" | **"aggregate period-3 band power was similar"** — a scalar sum can conceal per-frequency AND per-dimension redistribution, not only phase rotation |

The third is the subtlest and I had not caught it: withdrawing "the carrier did not change" in favour
of "amplitude unchanged" was still too strong, because band power is a sum.

**Sequencing decision.** No six-seed confirmation yet. A corrected 2–3-pair ENGINEERING pilot comes
first, carrying: an **L0 native/no-edit continuation arm** — without it L1's rise could reflect
generic continuation rather than phase structure; `k ≈ 10` to shrink the edit-size mismatch that
`k = 1` leaves between arms (reduced, not removed); edit-length and content controls; complex
Fourier phase and per-dimension amplitude read from the weights; the causal endpoint as a
DIFFERENCE OF DIFFERENCES, `[Ψ_L1 − Ψ_L2]_intact − [Ψ_L1 − Ψ_L2]_P3-attenuated`, against
power-matched control bands; `Ω3` only if its own capability preflight passes, and kept separate
because imposed relocation and trained phase change are different questions; and health gates
stronger than a flat MLM accuracy, since overall coupling moved substantially under training.

Only if that is capable: freeze a fresh panel, run ≥6 paired seeds (8 safer), with independent audit.

**No claim row.** Not yet, on any of tonight's work.

**The successor decomposition proceeds in parallel**, explicitly exploratory, and must not be mixed
into the confirmatory family.

## PERMANENT PREFLIGHT RULE — axis capability

Seven defects this session shared one shape: **a measurement that cannot vary along the axis it is
supposed to detect will read as confirmation.**

- a blacklist filter comparing unversioned IDs against versioned ones — could never fire
- `gate_designed_register` inferring register from a length delta, against a length-preserving operator
- `gate_period3`'s expected-phase model, and separately its whole-sequence amplitude measurement, both encoding insertion geometry against a rotation operator
- `clean_checkout_reexecutable` computing "inputs exist on this machine", the condition under which it is FALSE
- a leakage screen run on chromosomes where no evaluation gene was at risk, reporting `0 dropped`
- a spike-in capability control that exercised the estimator but not the causal path it certified
- period-3 **band power** used to argue about carrier **phase**, when a 90° rotation leaves it unchanged to 1.2e-16

**Rule, binding on every future measurement in this study:** ship a positive transformation along
the exact axis the measurement is meant to detect, and show the measurement responds to it. Software
tests show the code runs. Axis-capability tests show the instrument can answer the scientific
question. A green suite establishes only the former, and three of the seven above were introduced by
the person most convinced the record was sound.

## 2026-08-12 — closing disposition: what survives, and why the end-to-end case does not

**The two statements that survive**, and the only two this arc licenses:

1. **SpliceBERT genuinely favours donor–acceptor joins whose ANNOTATED registers match.**
   `Ψ = +0.175 [+0.109, +0.247]` and `+0.25625` on a second draw, reproducing `C43`'s
   `+0.3901 [+0.3553, +0.4252]` in an independent pipeline and cohort. Boundary: the masked-donor
   margin under `AG→AC` in CHIMERIC constructs at one geometry — not splice prediction, not
   spliceosomal recognition.

2. **Attenuating the learned period-3 component degrades that preference, band-specifically and
   disproportionately, so the component APPEARS NECESSARY.** At dose 0.50 on base: period-3 gives
   coupling magnitude retention 34.8% with `Ψ` retention 10.7%; power-matched control bands give
   magnitude 107% with `Ψ` 111%. At the gentler 0.75 dose, coupling magnitude 78% against `Ψ` 62%.
   **Corrected caveat:** an earlier reading of "coupling 114%, behaviour intact" was a SIGNED-MEAN
   artifact — the signed mean can rise while magnitudes collapse. General coupling IS degraded at
   the doses where the effect is largest. The claim rests on band specificity and on `Ψ` falling
   faster, never on the behaviour being unharmed.

**Why the end-to-end "trainable positional surrogate" case is NOT established.** The intended chain
was corpus structure → learned positional surrogate → masked-token behaviour. The engineering pilot
measured the middle term directly, and it does not move:

- every arm-vs-base per-bin phase rotation `< 1.1°` (`< 0.009` nt equivalent), coherence `> 0.998`,
  band power within `+0.85%`, per-dimension TV `≤ 0.0021`, top-8 dimension overlap `1.000`;
- **L1 vs L2 `≤ 0.34°`**.

Training moved neither the carrier's amplitude, nor its phase, nor its distribution across
dimensions. So what the difference-of-differences shows — `L1 − L2` collapsing under period-3
attenuation, `+0.162 [+0.103, +0.219]` and `+0.130 [+0.074, +0.187]` against `−0.025…+0.018` for
every control — is that the behaviour DEPENDS on an UNCHANGED coordinate. That is
**necessary-but-not-sufficient**, and it is not the same thing as a coordinate that training
re-shapes. **A surrogate that does not move under the manipulation is not a trainable surrogate.**

**And the training story itself inverted once L0 existed.** `Ψ`: L0 `+0.656/+0.732` **>** L1
`+0.579/+0.604` **>** L2 `+0.387/+0.456` **>** base `+0.256`. `L1 − L0` is NEGATIVE and excludes
zero in both seeds — the rise above base is GENERIC CONTINUATION, not phase-preservation. What
survives is `L1 − L2` (`+0.191/+0.148`): residue mod 3 moves `Ψ`, as SUPPRESSION between two edited
arms that both sit below native. The earlier pilot's headline, which lacked L0, was wrong.

**Claim rows: none.** Neither surviving statement is entered in `CLAIMS.md`. Both are
base-checkpoint or two-seed engineering readings; the register-preference statement duplicates
`C43`'s territory and would need its own confirmatory framing; the necessity statement is one
operator, one cohort, and carries the health caveat above.

**Open, and correctly so:** whether anything is a trainable surrogate for this relation; what
carries register identity, given designed offsets do not reproduce `Ψ`; and `PPM1`'s paired-`ΔΨ`
question, which two seeds cannot answer.
