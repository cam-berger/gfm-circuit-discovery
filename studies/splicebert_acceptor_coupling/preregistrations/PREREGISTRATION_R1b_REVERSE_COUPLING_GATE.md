# Preregistration R1b — reverse-direction coupling gate, corrected power gate + draw stability

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN 2026-08-05, before the R1b runner's first execution.** Freeze covers §1–§8. This is a
**new preregistration, not an amendment.**
[`PREREGISTRATION_R1_REVERSE_COUPLING_GATE.md`](PREREGISTRATION_R1_REVERSE_COUPLING_GATE.md) stays
frozen, hash-verifiable, and unedited; its §6.3 defect is recorded in
[`../CORRECTIONS.md`](../CORRECTIONS.md) as `ACR1` rather than repaired in place.

---

## 0. Why this document exists

R1 executed cleanly — all eight memorials passed, the arithmetic instrument anchor reproduced, and the
reverse coupling came in at `+1.0200 [+0.9472, +1.0916]` with both specificity contrasts intact — and
then returned **`INSTRUMENT_UNRESOLVED`**, because its §6.3 gated `MDE80` on the **paired differences**
rather than on the **control arms**.

That is a power gate pointed at the wrong quantity. A power gate exists so that a null is not an
underpowered null. The null under test here is *"is this control null, or is it SESOI-sized?"*, and it
lives on the control arms, where R1 measured `MDE80` of `0.019` and `0.029` against a `0.05` band. The
paired differences are not a null; they are a heavy-tailed effect of size `≈1.0` whose bootstrap SD
scales with the effect itself (`coupling.py` records the forward effect as mean `0.436` / median
`0.051`). R1's paired-difference `MDE80` was `0.106`, and **no achievable `n` on this cohort would have
brought it under `0.05`** — the gate was unpassable by construction, which is the same defect shape
E11's preflight-resolution gate had (its development third's own `MDE80` could never clear a target
threshold) and that IC2's freeze-audit flagged as a must-fix.

**R1b changes exactly two things.** Everything else — SESOI, cohort, operator, controls, estimand,
verdict class names, precedence, and all eight memorials — is imported from R1 unchanged.

1. **§6.3 gates `MDE80` on the two control arms.**
2. **Three seeded draws, with verdict stability as a new memorial** (§2.1, and see §5.9).

## 1. The second change, and the observation that forced it

R1's decoy-`GT` partner control landed at `+0.0129 [-0.0005, +0.0270]`. Pilot 2's, on a different draw
from the same pool, landed at `+0.0284 [+0.0100, +0.0509]`. **The two draws disagree on whether that
control excludes zero.** Both are small and both are far below the effect, so no verdict turned on it —
but the disagreement is exactly the instability `CLAUDE.md` convention 4 warns about: *within-sample
held-out is not sampling stability; effect-existence needs multiple seeded draws.* R1 ran one draw.

A gate whose specificity arm flips significance between draws has not demonstrated specificity; it has
demonstrated one draw. R1b therefore runs **three** and requires the verdict class to be identical on
all three.

**What three draws do and do not buy, stated in advance.** They are resampled from the same validation
pool by a different `load_examples` sampling tag, so they **overlap by construction and are not
independent replications.** They measure the stability of the verdict under cohort selection, which is
the specific failure observed. They do not widen the population, do not substitute for the sealed split,
and do not license any statement about generalisation. Draw-to-draw spread is reported as **range or
SD, never as `[a, b]`** — that notation means a confidence interval in this repository and nothing else.

## 2. Design — imported from R1 §2 unchanged

| | mask | corrupt | quantity |
|---|---|---|---|
| forward (C1; anchor only, not evidence produced here) | `donor_pos` | `[p-2, p)` with `CC` | `Y_D` |
| **reverse** | `next_acceptor_pos - 1` | `[d, d+2)` with `AC` | `Y_A` |

Both readouts mask a `G`; `CouplingTask.margin` is used unchanged. Cohort: existing `CouplingConfig`
eligibility, val split (chr11/16/19), `n = 1000` requested per draw, paired on exons having **both**
matched comparators. Operator `MOTIF_CLEAN_DONOR_ALT = "AC"`, admissibility re-verified per construct
by `planted_donor_motifs` (R44). Controls are C2's own idiom: `matched_decoy_acceptor(x, 40)` for the
readout arm, `matched_decoy_donor(x)` for the corruption arm — both **spurious distance-matched
motifs**, not real annotated sites. Inference: gene-clustered bootstrap, `n_boot = 2000`, seed 0,
ratio-of-means, population unconditional (R10). `SESOI = 0.05`.

### 2.1 Draws

Sampling tags `r1b-d0`, `r1b-d1`, `r1b-d2`, fixed here. **Draw 0 is the primary** and is the only draw
that carries the verdict; draws 1 and 2 exist solely to test that the verdict class does not move. The
§4 operator sweep runs on **draw 0 only** — `AC2`'s dose-response has already replicated independently
across pilot 2 and R1, and re-running nine alts three times buys nothing.

## 3. Endpoints

Unchanged from R1 §3: `Y_A`; `Y_A^readout` (same corruption, decoy-acceptor readout); `Y_A^partner`
(decoy-`GT` corruption, own acceptor readout); `Y_D` as the instrument anchor. **Primary:** the paired
differences `Y_A − Y_A^readout` and `Y_A − Y_A^partner`. Secondary: margin levels, selectivity ratio,
the operator sweep, and the across-draw range of every endpoint.

## 4. Operator sweep

Unchanged from R1 §4, on draw 0. Alts `{AC, AT, CC, CT, TC, TT, CG, TG, AA}`, admitted per example,
`n ≥ 50`, `ρ` against `AC`. **The `z[1]` prediction is no longer a prediction** — it was stated in
advance in R1 §4 and met twice (pilot 2 and R1's independent draw, no CI overlap between groups). It is
carried here as a **replication check**, and a third failure to reproduce it would be a finding.

## 5. Memorials — R1's eight, plus one

Memorials 1–8 are imported from R1 §5 verbatim: instrument anchor (`Y_D` within `±0.15` of AR1's val
`0.9909`); zero planted motifs; edit-size match; readout health at both sites above `1.0`; comparator
retention published; selection audit against `DONA`'s `CC`-admissible stratum; planted positive at
`3 × SESOI` in `≥ 18/20` seeds; permuted-label FPR `≤ 0.05` as a **rate of positive decision-layer
calls**, not a mean absolute effect.

**5.9 — Draw stability (new, and it can fail).** The verdict class computed independently on each of
the three draws must be **identical**. If draws disagree, the gate returns `INCONCLUSIVE` and the
disagreement is reported with all three verdicts. This memorial is **not** satisfied by the endpoints
being similar; it is evaluated on the decision layer, because the decision layer is what R1's control
instability threatened.

**The comparison basis is the six memorials every draw runs** (anchor, construction, edit-size,
readout health, comparator retention, selection audit). The two calibration arms — planted positive
and permuted-label FPR — calibrate **the decision layer itself rather than the cohort**, and run once,
on draw 0, following E11's precedent of calibrating a decision layer once per instrument. Including
them would make draw 0's verdict class incomparable to the other two by construction, which is not a
stability test but an artefact. The runner emits `verdict_on_shared_memorials` per draw for this
comparison and `verdict` (full memorial set) for draw 0's primary. **This basis was fixed by the
`--smoke` run, before any settled endpoint was computed** — see §7.

`comparator_retention_published` remains labelled `forced: true` — it records a number and cannot fail
— and is therefore **not evidence** (R7/R21).

## 6. Decision rule, frozen

Evaluated in order on **draw 0**; the first matching class is the verdict. Identical to R1 §6 except
clause 3.

1. Any memorial in §5 fails → **`INCONCLUSIVE`**.
2. `Y_A`'s CI lower bound `≤ SESOI` → **`REVERSE_INERT`**. Evaluated before any specificity contrast,
   deliberately: a near-zero effect makes both controls trivially null and would read as perfect
   specificity.
3. **Observed `MDE80` on either CONTROL ARM `> SESOI` → `INSTRUMENT_UNRESOLVED`.** The controls are
   where a null is being asserted, so they are where the study must be able to detect a SESOI-sized
   effect. *(This is the corrected clause; R1 gated the paired differences here.)*
4. Either control arm is **materially** non-null — its CI excludes zero **and** its nearer edge is
   `≥ SESOI` → **`NO_GO — NONSPECIFIC`**.
5. **Both** paired differences have CI lower bound `> SESOI` → **`GO`**. The mirrored mechanistic
   battery is authorised.
6. Otherwise → **`INSTRUMENT_UNRESOLVED`**.

`GO` is a conjunction over both contrasts. Classes are mutually exclusive and exhaustive; `NO_GO`
precedes `GO`, so a material control failure cannot be outvoted by a large paired difference.

**R1's clause 3 is also computed and reported** as `verdict_under_R1_rule`, so the two protocols stay
directly comparable and nothing is hidden by the correction.

## 7. Disclosure — what was seen before this was frozen

**Everything R1 measured was seen**: two val pilots and the settled R1 run, all endpoints, on the same
validation pool this gate resamples. §7.1 of R1 additionally records a post-hoc rule change there, which
turned out immaterial (both rules gave the same verdict on R1's cohort).

**What that costs, without mitigation.** R1b is **not** an independent test of R1's numbers. Its cohort
is drawn from the same pool, its SESOI was chosen with pilot magnitudes in view, and its corrected
clause 3 was written knowing that clause 3 was what blocked `GO`. **A reader is entitled to regard the
`MDE80` relocation as outcome-motivated**, and the honest defence is not that it was blind — it was not
— but that the relocated gate is (i) the quantity where the null under test actually lives, (ii)
passable in principle where R1's was not, and (iii) frozen here *before* R1b runs, with R1's clause
computed alongside so both readings survive.

This is why the tier is **verification, unconditionally**, and why **chr8/9 stays sealed.** The seal is
the thing being protected: a confirmatory reverse-coupling claim, if one is ever warranted, needs an
unspent split, and neither R1 nor R1b spends it.

### 7.1 What the smoke run changed, before freezing

A 24-example `--smoke` was run before this document was frozen, per `CLAUDE.md`'s prereg → smoke →
confirmatory order and E11's precedent that freezing a design the instrument cannot execute spends a
seal for nothing. It exposed one defect in §5.9, fixed above: **the three draws were being compared on
different memorial sets.** Draws 1 and 2 do not run the two calibration arms, so draw 0 could return
`INCONCLUSIVE` from a calibration failure while the others returned a different class from the
endpoints — a disagreement guaranteed by the design rather than by cohort selection. The stability
comparison is now computed on the six shared memorials.

**No settled endpoint was seen when this was fixed.** The smoke withholds endpoints from stdout by
construction, and 24 examples cannot estimate any §3 quantity — its `MDE80` on the controls runs
roughly `6×` the target's. The fix removes an artefact that would have made §5.9 fail for a reason
unrelated to what it tests; it does not touch a threshold, a direction, or a verdict class.

## 8. Outputs, and what is out of scope

`scripts/splicebert_acceptor_coupling/run_r1b_reverse_coupling.py`;
`results/splicebert_acceptor_coupling/r1b_reverse_coupling.json` + `<stem>.manifest.json` with
checkpoint, GTF and FASTA as hashed inputs; `status = "VERIFICATION"` unconditionally — this protocol
cannot emit `CONFIRMATORY`. A `REFLECTION` block states pass/fail against every §5 memorial and §6
class verbatim, the across-draw ranges, `verdict_under_R1_rule`, and any quantity that could not have
failed.

**Out of scope and not implied by any outcome:** which components mediate the reverse direction, the
address geometry, a payload, a `U_info` analogue, faithfulness, carrier involvement, the sealed split,
and every biological reading. **A `GO` authorises designing the mechanistic battery. It identifies
nothing, and in particular it says nothing about whether `L3H1` transfers.**
