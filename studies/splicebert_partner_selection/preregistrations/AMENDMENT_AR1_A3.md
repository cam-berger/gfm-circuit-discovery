# AMENDMENT AR1-A3 — twin representative must be the incumbent; reference twins excluded (2026-08-02)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Corrects the representative rule introduced by `AMENDMENT_AR1_A2.md`. Written **after
re-analysing the train pilot under A2 (`ar1_trainpilot2_a2`), before any val access.**
The frozen prereg body is not edited.

## A2's rule produced a verdict that could not fail

A2 §"Twin deduplication" specified *"one representative is kept — lexicographically first"*.
On DONM the `+1 = C` construct group is `{AC, CC, GC, TC}`, so the lexicographic rule elected
**`AC`** and dropped `CC`. But `CC` is the **incumbent reference** the whole ratio `ρ` is
defined against. The surviving panel was therefore `['AC']` — a single alt that is
*byte-identical to the reference*, giving `ρ = 1.000 [1.000, 1.000]`, `EQUIVALENT`, and an
arm verdict of **`ALT_INVARIANT`**.

That is the incumbent compared with itself. It is an architecture-forced identity dressed as
a verdict — the exact R7/R21 failure A2 was written to remove, reintroduced one layer up, and
strictly worse than the `NOT_INVARIANT` it replaced because it reads as a pass.

**A2's stated expectation was also wrong.** A2 predicted the DONM panel would become
`{+1 = C, +1 = A, +1 = G}` and return `INCONCLUSIVE`. It overlooked that the §4 admissibility
filter (≥95%) independently removes the `A` group (`AA`, 54.8%) and the `G` group (`AG`,
93.1%). The prediction was not checked against the filter before being written down.

## Corrected rule (replaces A2's representative clause)

1. **Representative preference.** Within a byte-identical construct group, the representative
   is `CORRUPT_ALT` if the group contains it; otherwise the lexicographically first member.
   The incumbent is never deduplicated away.
2. **Reference-twin exclusion.** Any `z` whose visible construct is byte-identical to the
   reference construct is excluded from the panel and reported under `reference_twins`. Such
   an alt yields `ρ ≡ 1` by construction and can carry no information about operator
   sensitivity.
3. An empty panel yields **`INCONCLUSIVE` / `NO_ADMISSIBLE_PANEL`** — never `ALT_INVARIANT`.
   Equivalence is a claim; it requires something to have been compared.

## Corrected expectation for DONM

Under the corrected rule the DONM panel is **empty**: the reference group collapses to `CC`
itself, and the only two genuinely distinct constructs (`+1 = A`, `+1 = G`) both fall below
the 95% admissibility floor. DONM therefore returns **`NO_ADMISSIBLE_PANEL`** and issues no
verdict.

That is the honest result, and it is informative rather than empty: **under a masked-donor
readout there is exactly one motif-hygienic donor-destruction construct.** The operator has
no admissible alternative to be robust to. Any `A_donor`-style anchor on this readout is
therefore not "one arbitrary choice among four" — it is the only clean choice, and the
descriptive table (`+1 = C` → Δ 2.598, `+1 = A` → 3.392, `+1 = G` → 3.606) shows the
non-hygienic alternatives differ from it by ~31–39%.

## Gate scoping is per-arm (clarifies §6 against §4)

Implementing rule 3 surfaced a scoping bug: the runner conjoined **all** arms' calibration
gates into one global pass/fail, so DONM's structurally empty panel — which correctly makes
its plant gates inapplicable, returning `NO_PANEL` — voided ACC and DONA as well, and the run
reported `GATES_FAILED` with three uninterpretable arms.

§4 already states the decision rule is applied **"per arm"**; each arm is a separate estimand
on a separate cohort. The global conjunction was an implementation choice, not a
preregistered requirement. Gates are therefore scored **per arm**:

- an arm is licensed iff no non-forced memorial failed **and its own** three plant gates plus
  its permuted-label FPR pass;
- an arm with `NO_ADMISSIBLE_PANEL` reports its plant gates as **`NOT_APPLICABLE`**, not
  `FAIL` — there is nothing to plant into — and its verdict stands as
  `NO_ADMISSIBLE_PANEL`;
- one arm's failure never licenses or voids another.

## Unchanged

The ±25% bound, common support, the memorial set, the A1 calibration gates, and §8. ACC and
DONA are unaffected — both have no twins and no no-ops, so A2/A3 are inert on them (confirmed
on `ar1_trainpilot2_a2`: the reduction fired only on DONM).
