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
