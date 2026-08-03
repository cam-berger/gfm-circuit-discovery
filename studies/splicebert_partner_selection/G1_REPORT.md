# G1 — cross-example transduction of the donor-side delivered state: REPORT

> **Tier: T2 — provenance.** One-page executor report (research-handoff schema). Canonical claim wording
> lives in `CLAIMS.md` (C41). Full protocol: the FROZEN
> [G1 prereg](preregistrations/PREREGISTRATION_G1_CROSS_EXAMPLE_TRANSDUCTION.md); design audit:
> [red-team synthesis](redteam/REDTEAM_SYNTHESIS_G1.json).

## Bottom line

**The donor-side delivered state (C16/C26) transfers across gene-disjoint, context-mismatched examples as a
gene-generic readout axis — not as example-specific information.** On the sealed Atlas internal-validation
split originally reserved for confirmation (`H_conf`, 551 genes, gene- and window-disjoint from the sources), a U_info state derived
on gene-disjoint single-pair short-intron source examples moves a target's native donor margin: **large**
(τ_REAL ≈ 0.65), **dose-monotone** (β = 0.27), and **content-dependent** (REAL ≫ matched-random +0.62, ≫
scrambled-content +0.68). But the **example-specific part does not beat a norm-matched gene-generic axis**
(Δ_spec = τ_REAL − τ_MEAN = −0.33), and — the decisive test — the **example-specific component is causally
inert for the donor margin**: injected alone in the linear regime it is an equivalence-null within ±SESOI,
even though it is ~78 % of the source state by norm; the generic-aligned component carries the entire transfer.

This **bounds the interventional reach of the C16/C26 delivered state to a shared readout direction** and
answers the `docs/METHODOLOGY.md` cross-context-transduction requirement *negatively for example-specificity*
— coherent with C26's 0.970 cross-gene rank-1 overlap. It is a **positive mechanistic finding + a material
scope correction**, and the prereg's designated "GENERIC READOUT AXIS" outcome (a legitimate, reportable
result), reached at **verification tier — not confirmatory** (R28/R32: the frozen §9 `Global PASS = AND(...)`
was never completed — the planted-confound arm was not run, and planted-positive was applied as a point check
rather than the preregistered slope-CI-covers-1).

## Result (H_conf; mean over the 6 clean draws of 8, ranges in brackets)

| Quantity | Value | Reading |
|---|---|---|
| τ_REAL (α=1) | **+0.65** [0.59, 0.72] | specific-source state transfers, large |
| τ_MEAN (α=1) | **+0.98** [0.92, 1.06] | gene-generic axis transfers *more* |
| τ_SELF (α=1) | +0.86 [0.86, 0.87] | own-state ceiling — **reproduces C26's 0.875** (memorial M3) |
| **Δ_spec = τ_REAL − τ_MEAN** | **−0.33** [−0.345, −0.276] | example-specific does **not** beat the generic axis |
| τ_PAR (⟂-decomp, low α) | +0.18 | generic-aligned component carries the transfer |
| **τ_PERP_matched (low α)** | **0.00** [−0.015, +0.010] | example-specific component **causally inert** (equivalence-null) |
| REAL − RANDOM | +0.62 | ≫ matched-random floor (transfer is real) |
| REAL − SCRAMBLE | +0.68 | ≫ scrambled-content (transfer is content-dependent) |

- **Sampling stability (R22, the load-bearing discipline):** across **8 independently-resampled source
  draws** (each deriving its own `U_D`/`s̄`, resampled with replacement), Δ_spec SD = **0.023 ≤ 0.025**
  (trigger) → **`stability_ok = True`**; PERP_matched SD = 0.008. Every draw's Δ_spec is strongly negative
  (−0.28 to −0.35). This is the certification the predecessor B3/A9 blind spot lacked.
- **Gates:** M1 head-readout = 0.0 exact; M3 (SELF vs C26 0.875) pass; M4 gene-disjoint (computed);
  M8 OOD ratio ∈ [0.5,2.0] at every arm×dose (`ratio_in_band = 1.0`); cross-gene permuted-label FPR = 0.0;
  planted-positive monotone.

## REFLECTION (re-anchored to §0 INTENT)

*INTENT:* "Does the donor-side delivered state behave as a transferable **example-specific** information-bearing
state … or does any cross-example effect reduce to a push along a **gene-generic readout axis**?"

**Answer: the latter, at the donor-margin readout.** Cross-example transfer is real, dose-ordered, and
content-dependent — but it is carried by a gene-generic axis; the example-specific structure of the delivered
state (the majority of it by norm) does not move the donor margin. The metric did not drift: Δ_spec + the
parallel/perp decomposition measure exactly the example-specific-vs-generic contrast the intent demands, and
the low-α PERP arm (added by the design red-team) rules out the norm-wastage "denoising artifact" reading of
Δ_spec < 0. No payload semantics are named (R22 stands); no A×V binding is claimed (R23); the value-null
statements are within-intervention (R24).

## Caveats (scope of exactly what is licensed — from the adversarial verification)

1. **Locus/method scope (R24-qualified).** The null bounds **only the C16/C26 delivered FINAL-RESIDUAL
   state**, via **additive** injection through the **short-source rank-8 U_D**. It does **not** bound C17
   (routing), C18 (transporters), or C23 (address+payload joint) — the estimand intervenes on none of them —
   nor earlier-layer or non-additive example-specificity. (C34/C35 record that the *address* routes
   example-specifically through L3H1 attention; that is untouched here.)
2. **"Inert for the donor margin, in the linear (α=0.25) regime," not "uninformative."** PERP ≈ 0 shows the
   example-specific component does not move the *donor-margin readout*; it cannot distinguish "off-axis
   content is uninformative noise" from "off-axis content is information the margin readout cannot express."
3. **Δ_spec's negative sign is geometry, not content-harm.** It is denoising/norm-wastage-consistent
   (`τ_REAL/τ_MEAN` = 0.63–0.71 *exceeds* `E[cos]` = 0.36–0.43 every draw, consistent with α=1 saturation of
   the all-on-axis MEAN arm); **`|Δ_spec|` is not an effect size** and must not be read as example-specific
   content being anti-informative.
4. **The RANDOM specificity floor is noisy in this cohort — disclose explicitly.** A norm-matched random
   rank-8 direction reached **τ ≈ 0.14 in 2/8 draws** (M6 breach → those draws `INSTRUMENT_LIMITED`) and 0.084
   in a third that the code's `est ≤ 0.05 + hw` gate passed; the prereg's *literal* `RANDOM τ ≤ 0.05` would
   flag **3/8** → **5/8 GENERIC**. Either way the headline direction is unchanged and **REAL − RANDOM is
   robustly positive in every draw (min +0.51)**, but the per-draw-abstention-then-modal-vote handling is a
   disclosed adaptation of the prereg's literal "M6 breach ⇒ STOP." Headline: **modal 6/8 GENERIC (5/8 under
   prereg-literal M6), 2–3/8 instrument-limited, 0/8 TRANSDUCING.**
5. **Stability is point-stable at modest precision.** SD(Δ_spec) = 0.023 (population; 0.024 unbiased) passes
   the 0.025 trigger, but on 8 draws (7 df) the true-SD 95% upper bound ≈ 0.05 ≈ SESOI — report as
   point-stable, not tightly stable.
6. **SCRAMBLE / SRC_SHAM absolute τ are instrument-limited** (HW > SESOI); the content-null conclusion rests
   on the *contrasts* (REAL − SCRAMBLE, REAL − RANDOM) and the PERP arm, not on those arms' absolute
   equivalence. SRC_SHAM no-decoy fallback is excluded (`frac_sham_ok = 1.0`).

## Verification

- **Design (step 2):** 4-lens adversarial red-team → FIXABLE_THEN_RUNNABLE, survivor=YES, 0 FATAL; 9
  must-fixes applied + re-validated (`REDTEAM_SYNTHESIS_G1.json`).
- **Result (step 5):** independent adversarial re-derivation → **`RESULT_STANDS_WITH_CAVEATS`**. Arithmetic
  exact (Δ_spec = τ_REAL − τ_MEAN to 1e-9 all draws); PERP identification sound and not saturation-forced
  (PAR moves +0.18 at the same α while a 2.5×-larger off-axis push moves ≈0); R22/R23/R24 genuinely honored;
  M1/M3/M4/M8 are real failable on-path gates. No attack refuted the GENERIC verdict; the 5 caveats above are
  the required scope, all folded into the C41 CLAIMS wording.
