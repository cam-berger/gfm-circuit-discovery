# ED1 — running decision log (architecture-conditioned boundary pairing)

> **Tier: T2 — provenance.** Rationale for the ED1 arc: what was decided, why, and what was rejected.
> Claims live in [`CLAIMS.md`](CLAIMS.md); the frozen protocol is
> [`preregistrations/PREREGISTRATION_ED1_ARCHITECTURE_BOUNDARY_PAIRING.md`](preregistrations/PREREGISTRATION_ED1_ARCHITECTURE_BOUNDARY_PAIRING.md);
> the design audit is [`redteam/REDTEAM_SYNTHESIS_ED1.md`](redteam/REDTEAM_SYNTHESIS_ED1.md).

## The question

The sealed behaviour (+0.87) shows a donor reading its **downstream** acceptor across a **short intron**
— intron-definition geometry. Classical splicing says that geometry should *flip* once the intron
exceeds ~200–250 nt (Robberson 1990; Amit 2012): the donor should instead pair with its **upstream**
acceptor across the **exon**. Does SpliceBERT show the flip?

The decisive quantity is an **architecture × partner-direction interaction**, not a raw upstream-vs-
downstream difference — the raw difference is confounded by separation, intervening sequence type, and
readout side all at once.

## D1 — why the raw direction difference is not a result (2026-07-25)

`C_exon` and `C_intron` are measured at different separations, across different sequence types (exonic
vs intronic), on opposite sides of the readout. These confounds are **direction-constant**: they cancel
in an interaction across architectures but not in a raw difference. Everything downstream is therefore
an interaction. *Rejected:* reporting `C_exon` vs `C_intron` at "matched distance" as the headline —
that is what the contaminated development artifact
(`results/splicebert_coupling/development/exon_vs_intron.json`, n=700, TRAIN) already did.

## D2 — the distance trap, and why the primary is an intersection–union test

`L_down` **is** the cross-intron separation. Stratifying on downstream-intron length is therefore
definitionally entangled with distance on the `C_intron` side — "C_intron falls when the intron is
long" is exactly what plain distance decay predicts, and nothing inside this design can fully separate
them. It is **not** entangled on the `C_exon` side: `C_exon`'s separation is `L_exon`, matched to
±10 nt across strata.

**Decision:** make the primary an **intersection–union test** over `{ΔΔ, E_switch}`. `ΔΔ` is the
interaction the question asks for; `E_switch` is its distance-clean half. A positive verdict requires
both, so the claim can never rest on the entangled half alone. Under a conjunctive null the max
p-value controls type I error at α, so no multiplicity correction is applied.

## D3 — splits (2026-07-25)

The development run filtered to `split == "train"`, so **val (chr11/16/19) is genuinely untouched** for
this question. Design, variance estimates and every sealed threshold come from TRAIN — that is what a
pilot split is for — and val is the confirmatory. **Test (chr8/9) stays sealed and was not read.**

## D4 — pairing does NOT reduce variance here (2026-07-25)

Measured on the development artifact: `corr(C_exon, C_intron) = −0.064`. Within-example differencing
therefore *adds* variance (`sd(C_exon − C_intron) = 1.10` vs `sd(C_exon) = 0.71`). This killed the
obvious "paired direction contrast" design and is why the estimand set is built around across-strata
contrasts of each direction separately, with the interaction formed at the end.

## D5 — matching is load-bearing, and what may NOT be matched (2026-07-25)

Short introns sit in GC-rich compact genes: `gc_intron` SMD **+0.743** unmatched. Frozen deterministic
1:1 caliper matching brings the primary axis to **3,472 pairs / 1,933 genes, worst |SMD| 0.092**.

Deliberately excluded from the adjustment set:
- **`clean` (the native margin)** — a component of every outcome (`C = f_N − f_perturbed`); matching on
  it is the collider error IC1 removed in its Amendment A2. `s_don` is matched instead.
- **The exposed flank** — calipering the variable being contrasted does not add conservatism, it
  deletes the contrast. An early pilot that calipered `L_up` on the *upstream* axis collapsed it from
  3,727 possible pairs to 102.
- **In-window site density** — see D6.

## D6 — the red-team's own fix was wrong (M4 → M4′) (2026-07-25)

The design red-team correctly identified in-window **site density** as the strongest rival account ("the
donor leans less on its upstream acceptor when more real sites compete"), measured at `n_real_don`
SMD +0.61. It prescribed adding density to the primary adjustment set. **That prescription is wrong.**

Density is substantially a **descendant of the exposure** — a short downstream intron mechanically pulls
the next exon into the 1024-nt window (`corr(log L_down, n_real_don) = −0.31`, monotone 2.07 → 0.95
across `L_down` quintiles). Adjusting for it in the primary would report a *controlled direct effect* as
a *total effect* and open a collider path.

**Replacement:** density is excluded from the primary (its imbalance reported as an exposure property)
and handled as secondary **D1**, exact-matched (1,245 pairs / 1,241 genes, worst |SMD| 0.096). D1 is
what adjudicates *partner-specific switch* vs *generic dilution*, and by frozen rule it constrains the
verdict wording.

## D7 — the frozen inference was anti-conservative by ~4× (build stage, pre-val)

Matched pairs are **dyadic** — each joins a treated gene and a control gene, and genes recur on both
ends — so one-way gene clustering under-covers (the A9b/R22 error). The replacement was a multiplier
bootstrap with `w_p = (ω_{g_a} + ω_{g_b})/2`, `ω ~ Exp(1)`. But `Var(w) = 0.5` there, so intervals came
out ~√2 too narrow: **true-null simulation gave FPR 0.193 / 0.207 / 0.193 / 0.287 at n = 3000 / 1000 /
200 / 20 — flat in n, i.e. a bias, not a small-sample effect.**

**Fix:** additive mean-zero multipliers, `T* = mean_p[(e_{g_a} + e_{g_b})/√2 · (v_p − v̄)]`, whose
variance is exactly the two-way cluster-robust variance. Re-verified **FPR 0.065 / 0.055, power 0.79**
at a true effect of 0.10. A self-pair normalization was added so a unit loading on one cluster twice
gets variance 1, not 2. Locked by `tests/test_ed1_inference_calibration.py`.

Two further build-stage defects, both caught before any confirmatory data existed: the
`permuted_label_fpr` gate was **vacuous** (it re-centred the sign-flipped vector, forcing the statistic
to zero — it reported a perfect `fpr = 0.000`), and `sensitivity_decision` planted SESOI *on top of* the
real effect. Both fixed; see red-team B1/B2/B4/B5.

## D8 — units, and a threshold error that would have invalidated every gate

v1 sealed SESOI, `sd_between` and the native-coupling floor from the development artifact, which used
the **max-margin** observable — then froze a **linear**-observable protocol. Thresholds in one metric
cannot gate a run in the other. All scale constants are re-derived from a TRAIN pilot computed with the
linear observable, and **SESOI is anchored to the coupling scale (⅓ × mean `C_exon`), never to the
observed architecture difference**, so it cannot be tuned by what the pilot showed.

The linear observable is itself a decision: every estimand here is a difference-in-differences, and
under `max` the arg-max base can switch between the states being differenced, so a DiD would measure
arg-max switching. Max-margin is reported as a secondary robustness readout.

## D10 — what the TRAIN pilot showed, and the two gate amendments it forced (2026-07-25)

Pilot: 4,000 units (deterministic stride) → 729 matched pairs / 1,382 genes, linear observable.
**This is TRAIN and licenses nothing**; it exists to seal constants and exercise the gates at realistic n.

**Scale sealed.** `C_intron | down-SHORT` = 0.663 nats; n-weighted mean `C_exon` over the 2×2 = 0.319 ⇒
**SESOI = 0.106**, floor = 0.265. Projected val MDE = **0.043 ≤ SESOI**, so a bounded null is licensable.

**The estimands separate exactly as the design anticipated.** `ΔΔ = +0.597 [+0.513, +0.684]` but
`E_switch = +0.043 [−0.023, +0.110]` — the interaction is carried almost entirely by the
distance-entangled cross-intron half (`I_switch = −0.554`). This is precisely the case the
intersection–union test exists to catch: on `ΔΔ` alone one would report a large architecture ×
direction interaction, and it would be indistinguishable from cross-intron distance decay.

**Matching moved the number a lot.** The unmatched development artifact implied `E_switch ≈ +0.13`
(max-margin). Matched, on the linear observable, it is +0.043 with a CI covering 0 — **most of the
apparent cross-exon architecture effect was gene-composition confounding.** The clean upstream axis
agrees: +0.008 [−0.062, +0.075]. `ΔΔ_sham = +0.011 [−0.010, +0.031]` confirms the machinery is
partner-specific, so the null is a real null and not a dead assay.

**Amendment A1 (pre-run, outcome-blind).** Neither inference method met the ≤0.05 FPR gate on real data
(bootstrap 0.087, wild 0.067) — heavy tails (top-decile share of `E_switch` = 0.573) degrade interval
coverage. The gate is now **graded**, on an asymmetry worth stating plainly: over-rejection inflates a
*positive* claim but makes a *null* conclusion conservative. So FPR ≤ 0.05 is clean; 0.05–0.10 lets the
run proceed but requires any positive verdict to clear a **null-calibrated critical value `p* = 0.030`**
(the 5th percentile of the bootstrap p under this design's own sign-flip null); >0.10 stops the run.
Also amended: the stability trigger now watches `E_switch` (SD 0.028, no trigger) rather than `ΔΔ`,
whose SD of 0.067 fires an absolute threshold trivially when the effect is ~6× SESOI.

## D11 — RESULT (val confirmatory, 2026-07-25): NOT LICENSED AS A SWITCH

Cohort: 15,885 eligible units on chr11/16/19 → **3,472 matched pairs / 1,933 genes**. All seven
memorials PASS (`balance_achieved` 0.0922; `native_coupling_present` 0.836 vs floor 0.265;
`batch_invariance` exactly 0; DiD residual 7.8e-16). Achieved **MDE 0.0386 ≤ SESOI 0.106**.

| estimand | val |
|---|---|
| `ΔΔ` (architecture × direction) | **+0.673** [+0.629, +0.714] |
| **`E_switch` (distance-clean)** | **+0.0136 [−0.0129, +0.0402]** |
| `I_switch` (distance-entangled) | −0.659 [−0.693, −0.625] |

**Frozen verdict (prereg §6, row 2): `ΔΔ` excludes 0 and is > 0 while `E_switch` includes 0 ⇒ NOT
LICENSED AS A SWITCH.** The canonical artifact records `GATES_FAILED` (one calibration gate failed,
§D12) and that verdict stands unedited. *This is deliberately NOT the BOUNDED_NULL row: that row
requires `ΔΔ`'s CI to include 0, and it does not. An earlier draft of this log called the result a
"gate-qualified bounded null" — that was an over-label against the frozen table and is corrected here.*

**What the result does say.** The interaction is real and large, and all of it sits in the cross-intron
half, whose stratifier *is* its own separation. The distance-clean half shows no detectable
architecture-driven strengthening, by every available route: upstream axis +0.0126 [−0.0125, +0.0367];
site-density-matched D1 +0.0239 [−0.0189, +0.0678] (1,245 pairs, balanced); within-gene +0.0152
[−0.0200, +0.0491] (1,178 genes); max-margin +0.0163 [−0.0117, +0.0441]. `ΔΔ_sham = +0.0213
[+0.0120, +0.0306]` is **larger than `E_switch` itself**, so the residual positive is within the range
the machinery produces for a site that is not a partner at all.

**What it explicitly does NOT say.** The raw cells do show the operational exon-like pattern —
cross-intron dominates in SS (0.904 vs 0.203), cross-exon dominates in LL (0.268 vs 0.110). ED1 shows
that this dominance flip is achievable **without any detectable recruitment or strengthening of the
cross-exon channel**: a constant cross-exon channel plus decay of the cross-intron channel reproduces
it. ED1 therefore **does not exclude an emergent exon-like regime produced by a generic
distance-dependent linker.** Bounding *active strengthening* is not the same as excluding *relative
dominance*.

**Claim of record (wording adopted from Cam, 2026-07-25):**

> On the held-out, in-window 300–800-nt subdomain, long downstream introns did not materially increase
> the donor's dependence on its upstream acceptor (`E_switch = +0.014 [−0.013, +0.040]`; SESOI 0.106).
> The large architecture-by-direction interaction was carried by the distance-entangled loss of
> cross-intron dependence. This gate-qualified verification bounds active architecture-conditioned
> strengthening of the cross-exon channel, but does not exclude exon-like relative dominance emerging
> from a generic distance-dependent mechanism or behavior in deeper introns outside SpliceBERT's
> context window.

**Contingent circuit prereg (does L3H1 retarget?) stays CLOSED.** Its trigger was a positive ED1, and
there is no behavioural evidence licensing it.

## D12 — the `planted_confound_null` failure, and the sensitivity bound that survives it

`planted_confound_null` **FAILED**: 28.5 % vs a ≤10 % target.

```
raw gc_intron gap (SHORT − LONG)   +0.0777
residual gap after matching        −0.0048      matching removes 93.8 %
preregistered beta = 3*SESOI/gap    4.09
implied residual bias              −0.0197 nats = 1.45 x SE(0.0135)
                                                -> predicted rejection ~29 %, observed 28.5 %
```

**The gate is mis-specified, and not only in its beta.** A *rejection-rate* gate necessarily fails at
large n whenever any residual imbalance remains, because detectability grows with n while the bias does
not shrink. **Replacement rule for future runs: gate the MAGNITUDE of the implied residual bias (e.g.
`|bias| ≤ 0.25 × SESOI`), not whether it is statistically detectable.** Pinning β to a multiple of
SESOI is a second, separate flaw — it plants a confound whose strength is unrelated to the data's
actual confounding. Both are fixes forward; neither licenses relabelling this run.

**The sensitivity bound (post-run robustness, `results/splicebert_coupling/ed1_sensitivity_bound.json`).**
This deliberately uses **only preregistered quantities** — no post-outcome empirical GC slope — so it
cannot be accused of being fitted to the answer. Under the frozen, deliberately severe plant:

```
one-sided 95 % UCB, E_switch alone                    +0.0361
one-sided 95 % UCB, E_switch + |bias|  (joint resample) +0.0557
sign-flip CALIBRATED UCB, E_switch  (c* = 0.0224)     +0.0360
sign-flip CALIBRATED UCB, + |bias|                    +0.0577
                                       CONSERVATIVE BOUND  +0.0577  <  SESOI 0.106
```

Both terms are bootstrapped on the **same dyadic resample**, so the bias uncertainty is propagated
rather than added as a constant; the interval is additionally calibrated on the design's own
within-pair sign-flip null, because the measured permuted-label FPR was 0.060, not the nominal 0.050.
**Reading:** an *active* strengthening of the cross-exon channel of SESOI size is excluded even under
the severe plant. A *positive* of the observed size (+0.014) would still not have been licensable — it
sits below the bias envelope and below `ΔΔ_sham`.

## D13 — the distance profile is SUPPORTIVE, not decisive; and the rebound is a window artifact

S4 is a preregistered **descriptive** secondary built from **marginal bins**, not a cross-direction
matched comparison. It is convergent evidence, not a test. Read at that weight.

```
separation   C_exon vs L_exon   C_intron vs L_down
  60–100          +0.355              +1.059
 140–200          +0.070              +0.496
 200–250          +0.035              +0.262
 250–300          +0.016              +0.132
 300–400          +0.019              +0.044
 400–550          +0.011              +0.093   <-- rebound
 550–900             --               +0.230   <-- rebound
```

**"Steep generic positional decay" is too simple** and is withdrawn as a summary. The supported wording
is **strong distance dependence through 300–400 nt, with an unresolved long-distance rebound** —
except the rebound is now diagnosed, and it is not biology:

| `L_down` bin | n | `C_intron` | `donor_pos` | **`next_acc`** | `L_exon` |
|---|---|---|---|---|---|
| 300–400 | 1895 | +0.044 | 329 | 676 | 130 |
| 400–550 | 2179 | +0.093 | 324 | **793** | 125 |
| 550–900 | 1690 | +0.230 | 283 | **920** | 105 |

The rebound tracks the destroyed acceptor sliding into the last ~100 nt of the 1024-nt window, where it
has truncated downstream context: **34.8 % of the ≥400 nt bin has `next_acc` > 900, against 1.0 % at
250–400 nt.** Those examples are also structurally different (shorter exons, smaller `donor_pos`) —
both direct consequences of the in-window eligibility rule (caveat M7). **The long-distance arm of the
marginal curve is a window-geometry artifact and must not be read as a long-range biological effect.**

## D9 — what ED1 deliberately does NOT do

No circuit-level readout. Adding an L3H1 attention measurement would put evidence-ladder level 3 next to
a level-5 behavioural result and invite exactly the promotion the project forbids. **Localization — does
L3H1 retarget upstream, or does a second core appear? — is a separate prereg, licensed only by a
positive ED1.**
