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
