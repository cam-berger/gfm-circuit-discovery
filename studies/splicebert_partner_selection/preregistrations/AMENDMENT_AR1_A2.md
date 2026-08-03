# AMENDMENT AR1-A2 — DONM panel deduplicated; no-op constructs excluded (2026-08-02)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Amends `PREREGISTRATION_AR1_CORRUPTION_OPERATOR_SENSITIVITY.md` §2/§4 panel construction.
Follows `AMENDMENT_AR1_A1.md`. Written **after the amended train pilot
(`ar1_trainpilot2`), before any val access.** The frozen prereg body is not edited.

## What the pilot exposed

With all twelve calibration gates passing, DONM returned **`NOT_INVARIANT`** — and the
verdict is **entirely architecture-forced**, exactly the R7/R21 failure the prereg set out to
avoid. The n=1000 train table collapses into **four** distinct constructs, grouped by `z[1]`:

| `z[1]` | alts | mean Δ | ρ |
|---|---|---|---|
| C | AC, CC, GC, TC | 2.598 | 1.000 |
| A | AA, CA, GA, TA | 3.392 | 1.327 |
| G | AG, CG, GG, TG | 3.606 | 1.356 |
| **T** | AT, CT, TT | **0.000** | **0.000** |

Every row within a group is *bit-identical*, and the `z[1] = T` group is bit-identical to
**clean** — the native donor `+1` base is `T`, so `GT → AT/CT/TT` leaves the visible sequence
unchanged once `d` is masked. Those constructs are not corruptions at all. They entered the
verdict panel (admissible 100%, common support 1000), returned `ρ = 0`, were scored
`DIFFERENT`, and drove the arm's verdict.

**A verdict driven by a forced identity is not a measurement.** §2 anticipated the
degeneracy and recorded it as a forced memorial, but did not stop it reaching the decision
rule. That is the defect.

## Respecified panel construction (applies to every arm; binds only DONM in practice)

Before the §4 rule is applied, the per-arm panel is reduced to **distinct realised
constructs**:

1. **No-op exclusion.** Any `z` whose built sequence is byte-identical to clean *as the model
   sees it* (i.e. after masking) is removed from the panel and reported under
   `excluded_noop`, with its identity to clean asserted as a memorial rather than scored as
   an effect.
2. **Twin deduplication.** Where several `z` produce byte-identical visible constructs, one
   representative is kept — lexicographically first, deterministic — and the rest are
   reported under `deduplicated_twins`. Byte-identity is *verified* on the encoded input ids,
   not assumed from the masking argument.

Both reductions are computed from the constructs themselves, so they apply automatically and
would be no-ops on an arm without degeneracy (ACC and DONA are unaffected: the pilot's ACC
and DONA panels contain no twins and no no-ops).

## Consequence for the DONM verdict

DONM's panel becomes the three genuine constructs `{+1 = C, +1 = A, +1 = G}`. On the train
pilot that yields ρ = 1.000 / 1.327 [1.286, 1.371] / 1.356 [1.322, 1.390] — i.e.
**`INCONCLUSIVE`**, both non-reference CIs straddling the 1.33 bound. That is the honest
outcome and it replaces a `NOT_INVARIANT` that meant nothing.

It also states the substantive caution plainly: **under a masked-donor readout the entire
donor corruption is carried by the `+1` base**, and moving that base from `C` to `A`/`G`
changes the measured effect by ~33% — at the edge of the ±25% bound. Any `A_donor`-style
anchor built on a single hardcoded donor alt inherits this.

## Unchanged

The ±25% bound, the common-support rule, the memorial set, the A1 calibration gates, and §8's
statement of what a PASS does not license all stand. No arm's estimand changes; only which
constructs are eligible to carry a verdict.

## Runner change

`--from-scores` is added so a panel-rule change can be re-analysed against the persisted
`ar1_scores_<tag>.json` without re-running the model. The re-analysis of `ar1_trainpilot2`
under this amendment is written to
`results/splicebert_coupling/development/ar1_trainpilot2_a2.json`.
