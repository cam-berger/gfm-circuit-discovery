# B3 confirmatory control redesign — health + MF2 (SIGNED OFF, pre-committed)

**Status: OWNER-SIGNED-OFF 2026-07-23 (pre-run).** The owner approved the reserved decisions —
**D-4(ii)** fixed margin-gap-band fallback, **D-1** M8-floor amendment + mandatory R-h2 positive-control,
**D-7(i)** authorize one verification-level re-run. The frozen B3 prereg is reconciled onto this branch and
carries the binding **pre-run amendment** (`PREREGISTRATION_B3_CROSS_EXAMPLE_TRANSDUCTION.md`, §A0–A5); that
amendment is the decision-of-record and this doc is the executor detail behind it. Tier T2 (decision/
provenance). Companion to `docs/B2_B3_EXECUTION_DISPOSITION_2026-07-23.md`. The re-run stays
**verification-level** (val numbers were seen); the push is held.

This spec was produced by an executor design pass and hardened by an independent adversarial review of each
control (forking-paths / gate-gaming / implementability / prereg-consistency). All three drafts returned
**NEEDS_REPAIR**; the repairs are folded in below, and the residual free choices are surfaced in §6 rather
than resolved silently.

---

## 0. The one thing to read first — what a re-run can and cannot do

The B3 confirmatory already ran once on `val` (n=260), deterministically (`SEED=0`, `PYTHONHASHSEED=0`), and
its numbers were seen. That has two hard consequences the owner must weigh **before** authorizing any re-run:

1. **A re-run cannot yield POSITIVE, and its likely verdict is predictable.** The primary transfer is large
   and favorable (`R_xfer(α=1)=0.912`), but the **margin-mismatched stratum equalled-or-exceeded the matched
   one** (1.094 vs 0.912). Under the prereg-mandated *paired* MF2 test (§3), the point estimate is fixed at
   `Δ_MF2 ≈ 0.912 − 1.094 = −0.182 < 0`. A content-specific POSITIVE requires `Δ_MF2` CI-low `> 0`, which a
   negative point estimate cannot reach on this cohort. **The honest expected outcome is therefore Endpoint C
   (transfer is *position*-specific but not shown to be *content*-specific) or INCONCLUSIVE — not POSITIVE.**
2. **Any val re-run is VERIFICATION-LEVEL, not fresh confirmation** (IC1-A2 / R22 discipline: a control fix
   applied after the primary was inspected is outcome-aware verification). A clean *confirmatory* C-claim of
   any tier requires the frozen redesign run untouched on the **sealed chr8/9 cohort — which is NOT to be
   opened**.

So the redesign's real job is **not** to rescue a positive (impossible and unwarranted). It is to convert the
current *messy* INCONCLUSIVE (three mis-specified controls) into a **clean, defensible characterization**: the
cross-example transfer is large, dose-monotone, and position-specific, but — pending the paired MF2 contrast —
**most likely not content-specific**. §6-D asks the owner whether that verification-level characterization is
worth a re-run at all, or whether B3 should be reported directly at "INCONCLUSIVE, with a documented lean
against content-specificity."

---

## 1. Context (where B3 stands)

- Branch `b3-confirmatory-redesign` @ `6ebc0af` (unpushed). Committed there: R-b3fix1/2 correctness fixes, the
  scramble redesign (feature-coordinate permutation — 1 of 3 blocking controls, **done**), B2 retirement, and
  the disposition doc.
- B3 confirmatory (val n=260) = **INCONCLUSIVE**. Strong primary: `R_xfer(α=1)=0.912`, dyadic CI [0.757,
  1.112]; dose-monotone 0.534→0.912→1.178 (β=0.271); **decoy collapses** (R≈−0.06). Blocked by three
  mis-specified controls: scramble (**fixed**), **health**, **MF2**.
- This doc specifies **health** and **MF2**, to be frozen together with scramble as a *set* before one re-run.

---

## 2. Health gate redesign (under-injection, baseline-relative)

> **UPDATE (2026-07-23, prereg A6): health DOWNGRADED to a reported diagnostic.** The R-h2 smoke
> positive-control did not fire — a 50×-destructive single-token L3H1 clamp leaves distant non-site
> health unchanged (0.540 vs 0.538), a verified architectural property (local injection does not
> propagate to distant probes). Health is therefore **measured + reported** (no-global-breakage
> diagnostic, weak evidence per R-h3) but is **not** a POSITIVE gate and R-h2 is **not** a SMOKE_PASS
> gate. The design below stands as the measurement; only its gating role is removed.

**Defect.** `result["mlm_health"] = task.mlm_health(X[:40])` (driver:434) measures **baseline** masked
accuracy with **no injection**, gated `> 0.8` (driver:439) against a model baseline of ~0.56 — so the gate can
never pass and never measures injection damage. Doubly wrong (disposition §"Still open").

**Redesign (prereg M8: "each interpreted arm keeps ≥0.5 MLM headroom over chance, MASKED probes; below-floor
arm dropped").** Measure top-1 masked-nucleotide accuracy at distant **NON-site** positions **under** the
payload injection, and threshold **relative to the model's own in-run baseline**.

- **Mechanism (bespoke helper, NOT `mlm_health`'s `ctx`).** `mlm_health`'s `ctx` argument is a zero-arg
  context factory called once per batch; it cannot carry the per-batch `enc`/`dtok`/`od` the injection needs
  (confirmed by the adversarial review and by direct inspection). Use a new helper `mlm_health_injected(...)`
  that replicates `CouplingTask.mlm_health`'s probe selection **verbatim** (same `stable_rng("health-0",0)`,
  `n_probe=40`, MASKED probes, non-site exclusion `|p−donor|>2 ∧ |p−acceptor|>3`) but runs the forward inside
  `inject_capture(ad, enc, dtok, od)` (which already sizes its patch from `enc` shape and handles per-batch
  `dtok`/`od`). Because the probe RNG/seed/target-set are identical, `acc_base` (inject off) and `acc_inj`
  (inject on) score the **same positions** → a paired competence comparison. Full code in Appendix A.
- **Threshold (baseline-relative).** `chance = 0.25` (circuit.py's ln¼ note); `acc_base` measured in-run (not
  hardcoded); `floor = 0.25 + 0.5·(acc_base − 0.25)` (≈ 0.405 at `acc_base=0.56`). An arm **passes** iff
  `acc_inj ≥ floor`, i.e. it retains ≥50% of the model's baseline headroom over chance. Below-floor arm is
  **dropped**; since POSITIVE requires the primary healthy AND all controls collapse AND `mm_ok`, a dropped
  arm **blocks** POSITIVE.
- **Arms & dose.** Primary checked at **max dose α=1.5** (worst-case damage; mandatory). Scramble/decoy/MF2
  arms at their operating α=1.0.

**Repairs folded in from the adversarial review (these are not optional):**
- **R-h1 — decoy-arm bug.** The decoy forward injects at the *decoy* donor (`inject_margins`:241–242), so the
  decoy health arm must honor `p.get("decoy")` and inject at `E[t]["decoy"]+off`, not the true donor.
  Otherwise it certifies a different intervention than it runs. (Appendix A.)
- **R-h2 — positive-control (make the gate demonstrably failable).** A single-token, single-head (L3H1) output
  clamp barely perturbs 40 *distant* non-site probes, so in practice `acc_inj ≈ acc_base` and the gate risks
  being an **unfireable pass** — the exact R7/R21 pathology it exists to fix. Add a smoke-time
  positive-control: inject a **known-destructive `od`** (e.g. a large-norm random head output) and assert it
  **drops `acc_inj` below `floor`**. This mirrors the project's planted-positive/permuted-FPR discipline and
  proves the gate *can* fire.
- **R-h3 — honest low-power disclosure.** Even with R-h2, a genuine PASS under a localized one-token edit is
  **weak** evidence that the injection is non-damaging. The re-run pilot must report the actual
  `acc_inj − acc_base` gap; the result must present a PASS as "no *detectable* damage at distant probes,"
  **not** a strong "injection did not break the model" certification.

**Can it fail?** Yes: if α=1.5 injection pushes L3H1's donor output off-manifold and distant masked accuracy
collapses toward chance (`acc_inj=0.30 < floor 0.405`), the primary arm is dropped and POSITIVE is blocked
even if every other gate passes. R-h2 guarantees the mechanism is exercised.

---

## 3. MF2 margin-mismatched gate redesign (paired within-target difference)

**Defect (a prereg violation, not just a power problem).** The shipped gate `mm_ok = prim_CI_low > mm_CI_high`
(driver:428) compares the CI-low of one arm to the CI-high of another — a **non-overlap-of-two-arm-CIs test**,
which **prereg §6 explicitly forbids** ("Every arm-vs-arm test is the PAIRED within-target difference CI, not
non-overlap of two arm CIs"). It also silently collapses an *unfavorable-at-power* result (mismatched ≥
matched) — which §4/§5 define as **evidence AGAINST content-specificity → Endpoint C** — into a plain gate
failure, losing the scientific signal.

**Redesign (paired, one-sided, three-way).** The matched primary (`match_pairs(E, rng)`) and the mismatched
stratum (`match_pairs(E, rng′, mismatch=True)`) iterate the **same target list** under the identical
strand/coding/|Δpdist|≤200 filter; only source *selection* differs. So each target `i` carries both a matched
and a mismatched source with the **same target-corrupt run, same clean address A⁺, same self-gain
denominator**. Define the paired per-target contrast at α=1:

> `Δ_MF2 = ⟨ M_xfer^matched(i) − M_xfer^mismatched(i) ⟩ / ⟨ M_self(i) ⟩` — a single ratio-of-means, dyadic CI
> via `dyadic_ratio_ci` on the paired numerator. The target-level main effect (baseline coupling, self-gain,
> matched covariates) cancels; only the source-content contrast survives.

**Pre-committed three-way interpretation** (`[d_lo, d_hi]`, `hw=(d_hi−d_lo)/2`, `SESOI_R=0.20`):
1. **CONTENT-SPECIFIC (CONFIRMED)** ⟺ `d_lo > 0`. (Confound-*direction* test; the effect-*size* bar
   `R_xfer > SESOI_R` is enforced separately, so a second SESOI hurdle here would double-count.) `mf2_confirmed`
   replaces `mm_ok` in the POSITIVE conjunction.
2. **REFUTED → Endpoint C** ⟺ `d_hi < 0` (decisive mismatched ≥ matched) **OR** (`d_lo ≤ 0 ≤ d_hi` AND
   `hw < SESOI_R`) (powered equivalence at 0). An honest non-identifiable finding ("recovery not separable
   from a margin/norm push"), a **new higher-precedence verdict**, never powered away.
3. **INCONCLUSIVE on this control** ⟺ `d_lo ≤ 0 ≤ d_hi` AND `hw ≥ SESOI_R`. Underpowered; reported at achieved
   precision.

**Power.** The paired difference has variance `Var_m + Var_mm − 2·Cov` with `Cov` large-and-positive (shared
target, self-gain, matched covariates), so it collapses relative to the `Var_m + Var_mm` the forbidden
two-CI test effectively used. N is sized from a **re-run `--pilot`** (discovery/train split) that runs the
mismatched arm and emits `hw(Δ_MF2)` and `N_req = n_pairs·(hw/SESOI_R)²`. **Forbidden:** enlarging N,
redrawing the cohort, or reshaping the estimator to shrink the mismatched arm — the pilot sizes variance only;
the val cohort, the paired estimator, and the sign of `Δ_MF2` are frozen.

**Repairs folded in from the adversarial review:**
- **R-m1 — implementation.** There is no `mm.matched_counterpart`; persist the primary α=1.0 per-pair `xfer`
  from the dose loop and difference it element-wise against the mismatched arm (both index-aligned by the
  shared target order). (Appendix B.)
- **R-m2 — conservative dyadic clustering (DEFAULT).** The paired numerator reuses two source genes per
  target; cluster on **both** matched and mismatched source genes (three-way `w[ti]·w[si]·w[si2]`), not
  matched-source-only. It only **widens** the CI (never flips sign, never eases CONFIRMED), so it is
  anti-forking-safe — and, given the negative point estimate, it guards against **over**-claiming a decisive
  Endpoint C when the honest call is INCONCLUSIVE. Adopted as default (a §6 sign-off item since it is a new
  estimator).
- **R-m3 — the Endpoint-C branch may be unreachable at current power (the key strategic risk).** The
  mismatched stratum picks the **maximally** margin-mismatched source (`argmax`; seen half-width ≈0.84 ≫
  SESOI 0.20). Even with the paired variance reduction, `hw(Δ_MF2)` may exceed SESOI_R, in which case MF2 can
  only ever return **INCONCLUSIVE** — the advertised "evidence against" branch never fires. **Pre-committed
  fallback:** if the re-run `--pilot` shows `hw(Δ_MF2) ≥ SESOI_R`, replace the `argmax`-extreme selection with
  a **powered fixed margin-gap band** (a controlled-mismatch stratum with lower variance and adequate n) so the
  honest-negative branch is reachable. Which of {accept likely-INCONCLUSIVE MF2, adopt the fixed band} to
  pre-commit is a §6 decision.
- **R-m4 — compliance (adjacent).** The primary-R denominator `self_num[:len(xfer)]` (driver:384) is a
  *positional* slice while `xfer` is `pairs`-indexed; it is misaligned if any target is dropped by
  `match_pairs`. Verify the frozen val cohort has zero skips (then it is moot on the seen 0.912); if
  correcting it in the same pass, flag it as compliance that could move the seen `R_xfer`.

**Can it refute?** Yes, and this is the anti-forking proof: on the seen data `Δ_MF2 ≈ −0.182 < 0`, so
CONFIRMED (`d_lo>0`) is unreachable. The gate can only return **Endpoint C** (if powered below 0) or
**INCONCLUSIVE** (if the CI straddles 0 too widely) — the seen unfavorable direction drives the verdict, it is
not powered away.

---

## 4. Verdict formula (assembled)

```python
mf2_confirmed = (mf2["dyadic_ci"][0] > 0.0)                       # §3 branch 1
mf2_refuted   = (mf2["dyadic_ci"][1] <= 0.0) or \
                (mf2["dyadic_ci"][0] <= 0 <= mf2["dyadic_ci"][1] and mf2["hw"] < SESOI_R)   # §3 branch 2
positive = (prim["dyadic_ci"][0] > SESOI_R and beta > 0 and monotone
            and all(sp[c]["collapses"] for c in sp)
            and mf2_confirmed
            and calib["permuted_includes_0"] and health_ok)       # health_ok per §2 (replaces mlm_health>0.8)
verdict = ("POSITIVE_transfer"       if positive
           else "ENDPOINT_C_not_content_specific" if mf2_refuted   # higher precedence than NULL/INCONCLUSIVE
           else "NULL_no_transfer"    if prim["dyadic_ci"][1] < SESOI_R
           else "INCONCLUSIVE_underpowered")
result["evidential_tier"] = "verification"   # never "confirmatory" on this seen val cohort (§0.2)
```

---

## 5. Outcome → claim map (C37 discipline — never stamp a claim past the artifact's verdict)

| Re-run verdict | Claim it licenses | Tier | Sealed cohort needed for a clean confirmatory C-claim? |
|---|---|---|---|
| **POSITIVE_transfer** (full family + health + `mf2_confirmed`) | "Verification-level evidence that the donor-side payload state is a transferable causal variable: dose-ordered, content-specific recovery above random/scramble/decoy/margin-mismatched nulls." | Verification-level transferable OV/mediation→generalization (Endpoint A). **Not** a confirmatory C-number. | **Yes** |
| **ENDPOINT_C_not_content_specific** (`mf2_refuted`, or `R_xfer≈R_rand`) | "Cross-example transduction of the donor payload is **not identifiable as content-specific** on this cohort — recovery is not separable from a margin/norm push (transfer is position- but not content-specific)." | Non-identifiable-as-content-specific (Endpoint C); honest negative-identifiability. | **No** — a negative-identifiability finding is licensed at verification tier without the seal. |
| **NULL_no_transfer** (`R_xfer` CI-high < SESOI at HW < SESOI) | "The payload state does not demonstrably transfer at MDE≈X." | Floored null (Endpoint B). **Contradicts the seen 0.912** → if it occurs, treat as a pipeline red flag, not science. | Yes (publication-grade null needs an untouched cohort). |
| **INCONCLUSIVE_underpowered** | No C-claim; report at achieved precision. | No claim. | Moot. |

The **most probable** row, given the seen deterministic data, is **Endpoint C** (or INCONCLUSIVE if MF2 stays
underpowered) — see §0.

---

## 6. Decisions reserved for the owner (please sign off before freeze)

These are genuine free choices made with the blocking direction already known; the executor does not resolve
them.

- **D-1 — M8 reinterpretation (memorial edit, not compliance).** Literal M8 ("≥0.5 headroom over chance" =
  absolute 0.75) is unsatisfiable at the 0.56 baseline — the same disease as the shipped 0.8. This spec reads
  it as **retain ≥50% of baseline headroom** (floor ≈0.405, tolerating ~a 27% relative-accuracy loss).
  *Executor recommendation:* adopt the relative reading and **amend M8's wording** in the tracked prereg;
  disclose the leniency. Approve, or set a different fraction.
- **D-2 — MF2 CONFIRMED margin:** `d_lo > 0` (confound-direction; *recommended*) vs `d_lo > SESOI_R` (stronger
  separation). Moot on seen data (negative point), but must be frozen pre-run.
- **D-3 — MF2 dyadic clustering:** adopt the conservative three-way (both source genes) as default (*recommended*,
  anti-forking) — confirm.
- **D-4 — MF2 mismatch-stratum power (the important one, R-m3):** (i) keep `argmax`-extreme selection and
  accept that MF2 likely returns INCONCLUSIVE, or (ii) pre-commit the **fixed margin-gap band** fallback,
  triggered when the re-run pilot shows `hw(Δ_MF2) ≥ SESOI_R`, so the Endpoint-C branch is reachable.
  *Executor recommendation:* (ii) — pre-commit the fallback so an honest negative is actually attainable.
- **D-5 — Endpoint-C routing:** confirm that a powered equivalence at 0 (`hw < SESOI_R`, straddling) routes to
  **Endpoint C** (content not separable), not a separate "weak-but-present" bucket. (Not verbatim in the prereg.)
- **D-6 — health specifics:** substrate = target `co_ids` (*recommended*, experiment-faithful) vs clean
  sequence; whether an unhealthy **MF2/mismatched** arm (high-norm by construction) **blocks** POSITIVE or is
  reported "uninterpretable at dose" while the primary gate governs; the health positive-control `od` (R-h2).
- **D-7 — is the re-run worth it at all? (§0).** Given the deterministic val cohort and the predictable
  lean toward Endpoint C, decide: (i) authorize one verification-level re-run to produce a clean Endpoint-C
  (or INCONCLUSIVE) characterization, or (ii) report B3 directly as "INCONCLUSIVE with a documented lean
  against content-specificity" without another val run, and reserve any confirmatory work for the sealed
  cohort (out of scope now).

---

## 7. Hygiene, sequencing, and commit-hold

- **Prereg reconciliation (blocking).** The frozen B3 prereg is **not tracked on `b3-confirmatory-redesign`**
  (last touched by `2baf31e`). This redesign must be reconciled onto the tracked design-of-record **as a
  pre-run amendment** before any B3 result is committed. No result may be committed against an amendment
  absent from tracked history.
- **Amendment logging (honest).** Log `Amendment (2026-07-23, pre-run)` and state plainly: no NEW confirmatory
  data was read for the redesigned health/MF2 controls, **and** the mis-specified-control numbers (scramble
  0.978, MF2 1.094 vs 0.912, health 0.560) **were** seen. Concealing the second would itself be a
  forking-paths violation.
- **One re-run, no per-control iteration.** After the re-run emits its verdict, no further control adjustment
  and no second val re-run. If a control fails, the licensed next step is the sealed cohort or an
  owner-approved fresh design — never another val tweak.
- **Smoke first.** Re-pilot the new `mlm_health_injected` plumbing (incl. the R-h2 positive-control) and the
  MF2 paired arm under `--smoke` before freeze; give the pilot a `_pilot` `--out` suffix (confirm needs
  `--force`).
- **Commit-hold.** Hold this doc, the prereg amendment, and the code edits for explicit owner sign-off on §6.
  The confirmatory re-run runs only after the reconciled amendment is committed. Push stays held.

---

## Appendix A — health code (schematic; see workflow artifact for the full helper)

New `mlm_health_injected(ad, task, E, X, pairs, dev, off, Wv, edit_fn, alpha, inject=True, decoy=False,
n_probe=40, seed=0)` mirrors `CouplingTask.mlm_health` probe selection on the target `co_ids` substrate; masks
the donor and the probes; runs the forward inside `inject_capture(ad, enc, dtok, od)` with
`od = head_output(base, Ap, deliver(Wv, edit_fn(E,p,dev,alpha)))`; **honors `p.get("decoy")` for `dtok`
(R-h1)**. Verdict block replaces driver:433–439 with the `acc_base`/`floor`/per-arm loop and `health_ok`
(§2/§4). Add smoke-time R-h2 positive-control.

## Appendix B — MF2 code (schematic)

Persist the primary α=1.0 per-pair `xfer` from the dose loop; compute `D = xfer_matched − (mmc − mmk)`
element-wise (shared target order); `mf2 = dyadic_ratio_ci(D, self_num, gt, gs[, gs2])` with the three-way
clustering (R-m2); derive `mf2_confirmed`/`mf2_refuted`/`hw` per §3–§4. Replace `mm_ok` (driver:428) and the
verdict (driver:436–442).
