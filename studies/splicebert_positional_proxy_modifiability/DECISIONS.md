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
