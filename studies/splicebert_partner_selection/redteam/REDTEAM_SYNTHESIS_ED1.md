# ED1 design red-team — synthesis (pre-compute, 2026-07-25)

> **Tier: T2 — provenance.**

Four lenses run against `PREREGISTRATION_ED1_ARCHITECTURE_BOUNDARY_PAIRING.md` **v1 draft, before any
confirmatory compute**. Verdict: **FIXABLE — 9 must-fix items, no fatal**. The estimand is well-posed and
the identification asymmetry (§4.1) is sound; the failures are in the *inference layer*, the
*adjustment set*, and one *unit-scale* error that would have invalidated every sealed threshold.

Overall: **the design survives with the fixes below applied.** They are applied in v2.

---

## FATAL — none

The design was checked for the three failure classes that killed or bounded earlier arcs:

- **Architecture-forced answer (R23 class).** `Γ` is *not* forced: `f` is an MLM logit functional with
  no algebraic identity making the 2×2 additive, unlike `O = A·V` which forced `Δ_bind`. The `Γ`
  regrouping identity IS forced and is labelled as such rather than sold as a memorial.
- **Definitional entanglement (A9b class).** Present, located, and *bounded*: `L_down` is the
  cross-intron separation, so the `C_intron` half of the interaction is entangled with distance. The
  design does not paper over it — the primary is an intersection–union test whose second component
  (`E_switch`) is the distance-clean half, so a positive verdict cannot rest on the entangled half.
- **Vacuous memorial (R21 class).** `identity_noop_exact` and `did_regrouping_identity` are both
  forced-by-construction and are explicitly labelled; the load-bearing memorials are
  `variant_construction_exact`, `readout_anchor`, `batch_invariance` and `native_coupling_present`.

---

## MUST-FIX (applied in v2)

### Lens: STATISTICS

**M1 — one-way gene clustering is wrong; the data are DYADIC.** *(severity: high — this is the
A9b/R22 error class recurring in a new costume.)*
A matched pair joins a *treated* unit from gene `g_t` and a *control* unit from gene `g_c`, and a gene
appears as treated in some pairs and control in others. Pair differences therefore share clusters on
**both** ends; a one-way gene bootstrap under-covers exactly as permutation-on-correlation did in A9b.
**Fix:** primary CI from a **two-way (dyadic) multiplier bootstrap** — iid mean-1 Exp(1) weights per
gene, pair weight `(ω_{g_t} + ω_{g_c})/2`. Report the pigeonhole variant (resample genes; keep pairs
whose *both* genes were drawn) and the naive one-way interval as comparisons, and report the
within-gene-pair fraction.

**M2 — BCa is not defined for a multiplier bootstrap.** **Fix:** percentile interval from the dyadic
multiplier bootstrap as primary; BCa reported only for the one-way comparison interval.

**M3 — heavy tails are not disclosed as an estimand property.** `C_exon` has mean 0.288 / median 0.045 /
max 4.06 on train: the population mean is carried by roughly a tenth of the units. That is a legitimate
estimand but must not read as a typical-donor effect. **Fix:** report the share of the effect
contributed by the top decile, alongside the already-planned winsorized and rank readouts.

### Lens: CONFOUND

**M4 — the adjustment set omits in-window site density.** *(severity: high — this is the strongest
surviving alternative explanation.)*
A down-SHORT window contains the downstream acceptor at ≤200 nt **and** the next exon's donor, and often
a further exon; a down-LONG window is mostly intron. "The donor depends less on its upstream acceptor
when more real splice sites compete in-window" is a *generic site-density* account that reproduces a
positive `E_switch` without any exon/intron-definition switch. Measured on val, the imbalance is large:
`n_real_don` **SMD +0.606**, `n_real_acc` **+0.387**, and the v1 matching leaves it at +0.494 / +0.314.
**Fix (as first prescribed):** add both to the matching covariates.

> ### M4′ — CORRECTION to M4's prescribed fix (found while applying it)
>
> **The problem M4 identified is real; the fix it prescribed is wrong.** In-window site density is
> substantially a **DESCENDANT of the exposure**, not a confounder: a short downstream intron
> mechanically pulls the next exon into the 1024-nt window (`corr(log L_down, n_real_don) = −0.31`,
> monotone across every `L_down` quintile: 2.07 → 0.95). Putting it in the *primary* adjustment set
> would estimate a **controlled direct effect while reporting it as the total architecture effect**, and
> would open a collider path on any unmeasured common cause of density and coupling.
>
> **Replacement rule.** Density is excluded from `BASE_COVARIATES` and handled as a preregistered
> **secondary D1**: the primary estimates the *total* effect (density free, its imbalance reported as an
> exposure property and explicitly NOT gated), and D1 exact-matches on `n_real_don` and `n_real_acc`
> (+ a `donor_pos ≤ 80` caliper to restore common support) to estimate the *controlled direct* effect.
> Achieved on val, outcome-blind: **D1 = 1,245 pairs / 1,241 genes, worst |SMD| 0.096** — so the rival
> IS adjudicable, at ~36 % of the primary's pairs.
>
> **Consequence for the claim.** D1 is what separates "partner-specific architecture switch" from
> "generic dilution by competing sites". If `E_switch` survives in D1, the site-density account is
> excluded; if it collapses, the licensed claim must be bounded to *architecture including its
> site-density consequences*, and must say so.

**M5 — `gene_type` is unmatched.** Short introns are not distributed uniformly across biotypes.
**Fix:** exact matching on `gene_type` (keeps the whole cohort and guarantees balance, unlike
restricting to protein-coding).

**M6 — repeat content is uncontrolled.** Long introns carry more interspersed repeats; GC captures only
part of this. Not fixable within the design. **Fix:** declare as a residual caveat, not as controlled.

### Lens: IDENTIFIABILITY

**M7 — the down-LONG arm is a SELECTED sample and the prereg under-states it.** `next_acceptor_pos` is
only defined when the downstream acceptor lands inside the 1024-nt window, so down-LONG units are
conditioned on `L_down ≲ 1020 − donor_pos` — selection on a function of the exposure *and* window
position. **Fix:** state it as an explicit scope restriction (the estimand is defined *within* the
in-window subdomain; `donor_pos`/`acc_pos` are matched so the contrast is unbiased *within* it, but it
does not extrapolate to deep-intronic architecture).

**M8 — `Γ_sham` under-controls saturation.** `Γ_sham` is built from a sham whose individual effect is
≈0, so it cannot probe the region of the `f` nonlinearity that the large partner effect visits. It is a
*lower-bound* control, precisely as IC1's PPT sham was for AG-channel damage. **Fix:** label it as such
and forbid `Γ_specific` from carrying any verdict.

### Lens: ESTIMAND ↔ INTENT

**M9 — UNIT-SCALE ERROR: sealed thresholds were derived on the WRONG observable.** *(severity: high —
would have silently invalidated SESOI, the power calculation, and a memorial.)*
SESOI = 0.10, `sd_between` = 0.702 and the `native_coupling_present` floor of 0.30 were all taken from
`exon_vs_intron.json`, which used the **max-margin** observable `logit(G) − max(other)`. The prereg
freezes the **linear** observable `logit(G) − mean(other)`, which is systematically larger and on a
different scale. Thresholds sealed in one metric cannot gate a run in the other. **Fix:** re-derive
SESOI, `sd_between`, the MDE and the `native_coupling_present` floor from a **TRAIN** pilot computed
with the linear observable, *before* freezing. No val data is touched.

---

## Build-stage findings (pre-val, outcome-blind)

Found while implementing and smoke-testing, **before any val data was scored**. Both are corrected in
the runner; the smoke artifact `ed1_smoke.json` is preserved with the defective gates for the record.

**B1 — `permuted_label_fpr` was VACUOUS (R21 class).** The implementation re-centred the sign-flipped
vector (`raw*s − mean(raw*s)`), forcing the statistic to exactly zero, so the bootstrap CI always
contained 0 and the gate could never fire. It reported a suspiciously perfect `fpr = 0.000` on the smoke
run — a gate that cannot fail validates nothing. **Fix:** do not re-centre; sign-flipping a
null-centred vector already yields the correct randomization null.

**B2 — `sensitivity_decision` planted on top of the real effect.** It tested `raw + SESOI`, i.e.
detection of *(true train effect + SESOI)*, inflating apparent power. **Fix:** plant on a null-centred
cohort (`raw − mean(raw) + SESOI`) so the gate measures detection of exactly SESOI at the achieved n
and dispersion.

**B4 — the frozen PRIMARY INFERENCE was anti-conservative by ~4× in FPR.** *(severity: high — this is
the single most consequential pre-run catch, and it invalidates nothing only because it was found before
any confirmatory data existed.)*
M1 correctly replaced one-way gene clustering with a dyadic scheme, but the implementation was a
Bayesian-bootstrap **weighted mean** with `w_p = (ω_{g_a} + ω_{g_b})/2`, `ω ~ Exp(1)`. Averaging two
Exp(1)s gives `Var(w) = 0.5`, so the bootstrap reproduces only half the sampling variance and intervals
come out ~√2 too narrow. Predicted consequence: an α=0.05 rule behaving like `1.96/√2 = 1.39` ⇒ p ≈ 0.17.
**Observed by true-null simulation: FPR = 0.193 / 0.207 / 0.193 / 0.287 at n_pairs = 3000 / 1000 / 200 /
20 — flat in n, i.e. a systematic bias, not a small-sample effect.**
**Fix:** additive mean-zero multipliers, `T* = mean_p[(e_{g_a} + e_{g_b})/√2 · (v_p − v̄)]`, `e_g ~ N(0,1)`,
basic (pivotal) interval. Its variance is exactly the two-way cluster-robust variance with the diagonal
normalized to the iid case. **Re-verified: FPR = 0.065 / 0.055 (n = 3000 / 1000), power 0.79 at a true
effect of 0.10.**

*A first simulation of this appeared to show FPR = 0.000 everywhere and was itself wrong* — it
sample-centred the data before testing, so the point estimate was identically 0 and the CI trivially
contained it (the B1 defect again). The true-null generator is what exposed the real behaviour.

**B5 — the wild-sign-flip fallback was uselessly conservative (FPR 0.000) for two compounding reasons.**
It used a *multiplicative* `s_a · s_b` null (too coarse — it collapses toward one global sign) and it
centred `v` before computing `obs = |mean(v)|`, making the observed statistic identically zero so
`p = 1` always. Both fixed (additive multiplier; centre only the null deviations): FPR now 0.045 / 0.055.

**B6 — self-pair variance inflation.** A unit loading on the *same* cluster at both ends (the per-cell
summaries pass `ga == gb`; genuine within-gene matched pairs also occur) received multiplier
`2e/√2`, i.e. variance 2 instead of 1, inflating those CIs by √2. Fixed by normalizing self-pairs to
`e`. Verified: one-way clustered-mean FPR 0.065.

**B3 — factor-2 compute waste.** Linear and max-margin readouts were obtained from two separate forward
passes over identical inputs. Replaced by `CouplingScorer.margins_both` (one pass, two reductions);
numerically identical, halves the confirmatory run.

## POST-RUN: the one gate that failed on the confirmatory (2026-07-25)

**`planted_confound_null` FAILED on val — 28.5 % vs a ≤10 % target.**

```
raw gc_intron gap (SHORT − LONG)   +0.0777
residual gap after matching        −0.0048      matching removes 93.8 %
preregistered beta = 3*SESOI/gap    4.09
implied residual bias              −0.0197 nats = 1.45 x SE(0.0135)
                                                -> predicted rejection ~29 %, observed 28.5 %
```

**Two independent defects in the gate, not one.**

1. **It gates DETECTABILITY, not MAGNITUDE.** A rejection-rate criterion necessarily fails at large n
   whenever *any* residual imbalance survives: detectability grows with n while the bias does not
   shrink. Such a gate cannot be satisfied by a well-powered study and so cannot mean what it claims.
   **Replacement rule: gate the magnitude of the implied residual bias — e.g. `|bias| ≤ 0.25 × SESOI` —
   and report the rejection rate as a diagnostic only.**
2. **β pinned to a multiple of SESOI.** `β = 3 × SESOI / raw_gap` plants a confound whose strength is
   unrelated to the data's actual confounding (here ~5× the empirical `gc_intron → C_exon` slope of
   −0.81 nats/unit GC). Calibrate β to the empirically estimated confound slope **on the pilot split**,
   where it is still outcome-blind.

Both are fixes forward, found after the primary was seen. **Neither licenses relabelling this run**,
and the artifact's `GATES_FAILED` verdict is left exactly as emitted.

**What survives, using only PREREGISTERED quantities.** The temptation is to rescue the run with the
empirical GC slope — but that slope was fitted after the outcome was seen, so it is the weaker
argument. The stronger one needs nothing post-hoc: take the frozen, deliberately severe plant, and ask
how large an *active* effect could still hide behind it
(`results/splicebert_coupling/ed1_sensitivity_bound.json`):

```
one-sided 95 % UCB, E_switch alone                      +0.0361
one-sided 95 % UCB, E_switch + |bias|  (joint resample)  +0.0557
sign-flip CALIBRATED UCB, E_switch   (c* = 0.0224)      +0.0360
sign-flip CALIBRATED UCB, + |bias|                      +0.0577
                                        CONSERVATIVE BOUND  +0.0577  <  SESOI 0.106
```

Both terms are drawn on the **same dyadic resample** so the bias uncertainty is propagated rather than
added as a constant, and the interval is calibrated on the design's own within-pair sign-flip null
because the measured permuted-label FPR was 0.060 rather than the nominal 0.050.

**The asymmetry that decides the claim.** A bias envelope of this size **cannot manufacture** a
SESOI-sized effect, so *active* cross-exon strengthening at SESOI scale is excluded. It **can**
manufacture a positive of the observed size (+0.0136) — which is also smaller than `ΔΔ_sham` (+0.0213)
— so a positive would not have been licensable. The frozen verdict remains **NOT LICENSED AS A SWITCH**
(prereg §6 row 2: `ΔΔ` excludes 0, `E_switch` includes 0); the bound above is a **post-run robustness
interpretation**, not a preregistered outcome, and is labelled as such wherever it appears.

**Scope the bound honestly.** It bounds *active strengthening of the cross-exon channel*. It does
**not** exclude exon-like relative **dominance** produced by a constant cross-exon channel plus decay
of the cross-intron channel — which is exactly what the raw 2×2 cells show operationally (cross-intron
dominant in SS, cross-exon dominant in LL).

## Checked and NOT changed

- **Direction-constant confounds** (exonic vs intronic intervening sequence, upstream vs downstream of
  the readout, the own acceptor being the exon's 3′ boundary) — these are why the primary is an
  interaction; they cancel there and would only bias a raw `C_exon − C_intron` difference, which the
  design already refuses to treat as a result.
- **`clean` excluded from matching** — correct; it is a component of every outcome (IC1 A2 collider).
- **Sign preregistration** — correct; a CI excluding 0 in the wrong direction is recorded as a
  falsification, not relabelled as an effect.
- **Within-pair sign-flip permutation** — a valid exact randomization null for `permuted_label_fpr`, and
  it doubles as a calibration check on the dyadic bootstrap. Kept.
- **No circuit-level readout in ED1.** Adding an L3H1 attention readout would be evidence-ladder level 3
  sitting next to a level-5 result, inviting exactly the promotion the project forbids. Localization
  stays a separate, contingent prereg.
