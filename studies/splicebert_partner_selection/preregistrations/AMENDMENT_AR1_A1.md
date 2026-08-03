# AMENDMENT AR1-A1 — calibration gates respecified (2026-08-02)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Amends `PREREGISTRATION_AR1_CORRUPTION_OPERATOR_SENSITIVITY.md` §6 and §7. **The frozen
prereg body is NOT edited** — this file is the amendment of record, per the repo's
amendment discipline. Written **after the train pilot, before any val access.**

## What triggered it

The train pilot (`results/splicebert_coupling/development/ar1_trainpilot.json`, n=400, 365
genes, all six memorials PASS) returned **GATES_FAILED** on two of the nine calibration
gates. Neither failure was a model result; both were design defects the pilot exists to
catch. Per prereg §6 no verdict was read, and `arms` was written as `null`.

## Defect 1 — the planted-difference gate tested an unknown quantity

§6 planted a difference by scaling alt `z`'s effect by 1.35. The quantity actually tested is
then `1.35 × ρ_true(z)`, where `ρ_true(z)` is exactly the unknown the experiment is trying to
measure. Whenever `ρ_true(z) ≲ 0.985` the plant lands *inside* `[0.75, 1.33]`, so the gate
returns `EQUIVALENT` and fails — which is what happened on ACC (`planted_target_alt = "AC"`).

*(Measured after this amendment was written, on `ar1_trainpilot2`: `ρ_true(AC) = 0.888`, so
the failed plant tested 1.35 × 0.888 = **1.199**, comfortably inside the band. The mechanism
is as described; the amendment was authored before the value was in hand.)*

**A gate whose meaning depends on the unmeasured estimand is not a calibration.**

## Defect 2 — the gate guarded the wrong error

An equivalence design has an asymmetric error structure. The error that matters here — the one
that would let "the alts agree" be asserted falsely — is **false equivalence**: declaring
`EQUIVALENT` when the truth lies outside the bound. §6 instead demanded `DIFFERENT` at a
planted truth of **1.35**, which is 1.5% beyond the 1.33 bound; no achievable CI separates
those, so the gate was close to unpassable by construction and tested nothing useful.

## Defect 3 — DONA is structurally underpowered against the incumbent

`ρ` is computed on common support (§3.1), and on the **donor** side the incumbent `CC` is
admissible on only ~20% of examples (it plants a `GC` at `[d-1,d]` wherever `seq[d-1] == "G"`,
which is ~80% of donors). Common support with `CC` is therefore ~20% of `n`, which at n=400
is ~80 — below the `MIN_SUPPORT = 100` floor, so every DONA alt returned
`INSUFFICIENT_SUPPORT`. This is a *sample-size* consequence of a real property of the
incumbent operator, not a defect in the estimand.

## Respecified gates (replace §6 in full)

The plant is now constructed so its true ratio is **known by construction and independent of
`ρ_true`**, while carrying the empirical per-example noise of a real alt-vs-CC comparison:

    Δ_plant = Δ_z × (τ / ρ_true(z))        ⇒  point-estimate ρ(plant) = τ exactly

`z` is the first alt of the arm's verdict panel in sorted order (deterministic; recorded as
`noise_donor_alt`). A purely synthetic `Δ_plant = Δ_CC × τ` was rejected: its bootstrap ratio
is identically `τ` in every draw, so the gate could not fail — an R7/R21 architecture-forced
"memorial".

| gate | planted truth τ | requirement | guards |
|---|---|---|---|
| `sensitivity_no_false_equivalence` | **1.35** | must **NOT** return `EQUIVALENT` | the false-equivalence error — the one that would let §8 be asserted wrongly |
| `sensitivity_detect` | **1.60** | must return `DIFFERENT` | the positive-detection path |
| `specificity_planted_1.00` | **1.00** | must return `EQUIVALENT` | spurious non-equivalence |
| `permuted_label_fpr` | — | ≤ 0.05 over 200 permutations | unchanged from §6 |

All four run at the full `N_BOOT = 2000`; the reduced `N_BOOT_PERM = 400` now applies **only**
inside the permutation loop, and is reported.

## Respecified n (replaces the §7 sentence on pilot-set n)

The confirmatory `n` is set so the **DONA** arm's expected common support clears
`MIN_SUPPORT`, i.e. `n ≥ MIN_SUPPORT / f_CC` where `f_CC` is the measured donor-side
admissible fraction of `CC`. With `f_CC ≈ 0.20` this gives `n ≥ 500`; the confirmatory run
uses **n = 1000** (or the full val eligible pool if smaller, recorded either way). ACC is
unaffected — `CC` is admissible on ~99.5% there.

If val's eligible pool cannot reach the DONA floor, **DONA is reported as
`INSUFFICIENT_SUPPORT` and no donor-arm verdict is issued.** It is not rescued by lowering the
floor after seeing val.

## Unchanged

§0–§5 and §8–§9 stand as frozen, including the ±25% bound `[0.75, 1.33]`, the common-support
rule, the memorial set, and — in particular — §8's statement of what a PASS does not license.

## Runner changes accompanying this amendment

1. Raw per-construct scores are persisted to
   `results/splicebert_coupling/development/ar1_scores_<tag>.json`, so a re-analysis never
   requires a re-run (the pilot's `arms: null` left no way to inspect `ρ` without re-scoring —
   that cost one full pilot).
2. The descriptive per-alt table is now always computed and stored, carrying an explicit
   `GATES_FAILED — descriptive only, no interpretation` label when gates fail, rather than
   being dropped.
