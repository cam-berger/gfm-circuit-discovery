# Pre-registration — ED1: is donor→acceptor coupling ARCHITECTURE-CONDITIONED (exon vs intron definition)?

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: v2 (2026-07-25) — FROZEN after the 4-lens design red-team; must-fixes M1–M9 applied
(`redteam/REDTEAM_SYNTHESIS_ED1.md`). Sealed scale constants (§5) are set from a TRAIN pilot run on the
LINEAR observable before any val compute; no val data is read until they are fixed.**
Behaviour-first: full-model coupling assay only. Circuit localization (does L3H1 retarget?) is a
**separate, contingent** later prereg and is explicitly NOT part of this protocol.

**Red-team changes from v1.** (M1) primary inference switched from a one-way gene bootstrap to a
**two-way dyadic multiplier bootstrap** — matched pairs load on two gene clusters, so one-way clustering
under-covers exactly as in A9b/R22. (M2) percentile, not BCa. (M3) tail concentration reported. (M4)
`n_real_acc` / `n_real_don` added to the adjustment set to close the in-window **site-density**
alternative. (M5) `gene_type` exact-matched. (M6) repeat content declared a residual caveat. (M7) the
in-window selection of the down-LONG arm stated as a scope restriction. (M8) `Γ_sham` labelled a
lower-bound control. (M9) **all sealed scale constants re-derived on the linear observable** — v1's were
taken from a max-margin artifact and were in the wrong units.

**Lineage.** `scripts/splicebert/run_exon_vs_intron.py` +
`results/splicebert_coupling/development/exon_vs_intron.json` (n=700, TRAIN) are treated as
**contaminated discovery**: they compare directions without a frozen architecture interaction, without
matching, and without a specificity control. Their mechanics are reused; their numbers are used ONLY to
size variance and set sealed thresholds, never as evidence. Cohort/identifiability pilots (outcome-blind,
annotation-only): `ed1_cohort_pilot_{train,val}.json`, `ed1_matching_pilot_val.json`.

---

## 0. INTENT (north star — re-read at every gate)

> **Does SpliceBERT's donor→acceptor dependence SWITCH from the cross-intron partner to the cross-exon
> partner as genomic architecture crosses the classical intron-definition → exon-definition transition
> (~200–250 nt) — i.e. is there an architecture × partner-direction INTERACTION, as opposed to a
> direction main effect, a generic distance decay, or a gene-composition confound?**

The geometry, for one donor:

```
upstream acceptor -- EXON -- donor -- INTRON -- downstream acceptor
      (a)                     (d)                   (next_acc)
      |<--- L_exon --->|       |<--- L_down --->|

C_exon    = donor-margin loss when the UPSTREAM acceptor is destroyed  (across the EXON)
C_intron  = donor-margin loss when the DOWNSTREAM acceptor is destroyed (across the INTRON)
```

**Honest boundaries, stated up front.**
(i) This is an analogy in *geometry and architecture-dependence*, never a claim that SpliceBERT
reproduces spliceosome assembly. (ii) A positive result licenses **"an architecture-conditioned
boundary-pairing computation analogous in geometry to intron/exon definition"** — nothing about U1/U2,
cross-exon complexes, or splicing chemistry. (iii) The cross-intron arm requires the downstream acceptor
to be **inside the 1024-nt window**, so `L_down` is structurally capped at ~800 nt: the "LONG" arm means
*300–800 nt*, not the genomic median (~1,500 nt). It straddles the classical transition, which is what
the question needs, but the finding does not extrapolate to deep-intronic architectures.

---

## 1. Domain, observable, eligibility, splits

- **Observable (frozen, LINEAR):** `f = logit(G) − mean{logit A,C,T}` at the masked donor `G`
  (`donor_pos`, the intron-start GT). Linear, **not** the max-margin task metric: every estimand below
  is a difference-in-differences, and under `max` the arg-max base can switch between the states being
  differenced, so a DiD would measure arg-max switching. (Same reasoning IC1 froze.) The max-margin
  version is computed and reported as a **secondary robustness readout only**.
- **Perturbation:** real-base substitution of the `AG` at a target acceptor with each of
  `ACCEPTOR_ALTS = (CC, TT, CA, TC)` — never `[MASK]` (off-manifold). Every state is averaged over all
  four; in double states the SAME alt index `k` is used at both sites. **Corruption dose is therefore
  identical across directions and strata by construction** (2 bp, same alphabet, same count).
- **Eligibility** (`gfmcd.splicing.exon_intron.eligible`): internal exon; canonical `GT` donor; own
  acceptor `AG`; downstream acceptor annotated, `AG`, and IN-WINDOW; upstream intron annotated.
  Val yield: **15,885 units / 2,520 genes**.
- **Splits.** Design, variance estimates and every sealed threshold come from **TRAIN** (already
  contaminated by the development run — that is exactly what a pilot split is for). **Confirmatory =
  VAL (chr11, chr16, chr19)**, which the development run never touched (it filtered to `split=="train"`).
  **TEST (chr8, chr9) stays SEALED** and is not read.
- **Inference unit = gene** (exons of one gene are not independent). Genes may appear in both arms;
  clustering handles it.

## 2. Architecture strata (frozen from the literature, NOT from these data)

Cross-exon recognition operates out to ~300 nt (Robberson 1990); systematic constructs place the
intron→exon definition transition at ~200–250 nt (Amit 2012). Therefore:

| band | rule |
|---|---|
| **SHORT (S)** | flanking intron ≤ **200 nt** — intron-definition regime |
| **LONG (L)** | flanking intron ≥ **300 nt** — exon-definition regime |
| buffer | 201–299 nt — the transition band, **excluded** from the 2×2 (scored, reported in §4.5 only) |

The **2×2 flanking architecture** is `(upstream band, downstream band) ∈ {SS, SL, LS, LL}`.
Val counts: SS 2,416 / SL 1,311 / LS 4,429 / LL 3,937.

## 3. States and per-example estimands

Seven states per example (`u` = distance-matched non-partner sham upstream, `d` = downstream sham):

| state | construction | forwards |
|---|---|---|
| `N` | native | 1 |
| `U⁻` | own acceptor destroyed | 4 |
| `D⁻` | downstream acceptor destroyed | 4 |
| `UD⁻` | both destroyed (same `k`) | 4 |
| `u⁻` | upstream **sham** AG destroyed | 4 |
| `d⁻` | downstream **sham** AG destroyed | 4 |
| `D⁻u⁻` | downstream acceptor **and** upstream sham destroyed | 4 |

```
C_exon        = f_N − f_U                 C_intron       = f_N − f_D
C_exon_sham   = f_N − f_u                 C_intron_sham  = f_N − f_d
S_exon        = C_exon   − C_exon_sham    S_intron       = C_intron − C_intron_sham
Γ_partner     = (f_D − f_UD) − (f_N − f_U)      # extra reliance on the EXON partner once the
Γ_sham        = (f_D − f_Du) − (f_N − f_u)      #   INTRON partner is gone; sham = same background
Γ_specific    = Γ_partner − Γ_sham
```

**Sham policy (frozen):** `ShamPolicy(min_sep=25, max_log2_ratio=1.0, k=1)` — a non-partner `AG` on the
same side of the donor, separation within a factor of 2 of the true partner's, and ≥25 nt clear of the
donor, the own acceptor and the downstream acceptor so a sham cannot damage the site it controls for.
Best-distance-matched candidate, deterministic ordering, no RNG. Availability (val): 96.2 % both sides.

## 4. Identification — where the confounds go

**4.1 The one distance trap, stated explicitly.** `L_down` **is** the cross-intron separation. So
stratifying on downstream-intron length is *definitionally entangled with distance on the `C_intron`
side*: "C_intron falls when the intron is long" is exactly what plain distance decay predicts, and no
control inside this design can fully separate them. It is **not** entangled on the `C_exon` side —
`C_exon`'s separation is `L_exon`, which is matched to ±10 nt across strata. **That asymmetry is the
entire identification strategy**, and §6 is written so the claim can only rest on the clean half.

**4.2 Direction-constant confounds** (exonic vs intronic intervening sequence; upstream vs downstream of
the readout; the own acceptor being the exon's own 3′ boundary) differ between directions but not
between architectures, so they **cancel in the interaction** and only bias a raw `C_exon − C_intron`
difference. That is why the primary is an interaction, and why the raw difference is not a result.

**4.3 Gene-composition confound — load-bearing, and handled by outcome-blind matching.** Short introns
sit preferentially in GC-rich compact genes: unmatched val `gc_intron` 0.621 (SHORT) vs 0.544 (LONG),
**SMD +0.743**. Frozen matching (`exon_intron.axis_spec`, `match_pairs`): deterministic greedy 1:1
nearest-neighbour without replacement, Mahalanobis-standardized Euclidean distance, treated units
processed in `example_id` order (no RNG, no hash-order dependence), on

`L_exon, gc_exon, gc_intron, s_acc, s_next, s_don, log1p(n_iso_don), log1p(n_iso_acc), donor_pos,
acc_pos, log1p(opposite flank length)`

with hard calipers `L_exon ≤ 10 nt`, `gc_intron ≤ 0.03`, `gc_exon ≤ 0.05`, `opposite flank log ≤ 0.5`,
and **exact agreement on the opposite flank's band and on `gene_type`** (M5 — short introns are not
distributed uniformly across biotypes; exact matching keeps the whole cohort where restricting to
protein-coding would not). The **exposed** flank is never a covariate or a caliper on its own axis:
calipering it does not add conservatism, it deletes the contrast (an early pilot that did so collapsed
the upstream axis from 3,727 possible pairs to 102).

*Achieved, outcome-blind, on val:* **downstream axis 3,472 pairs / 1,933 genes, worst |SMD| 0.092**;
**upstream axis 3,599 pairs / 1,707 genes, worst |SMD| 0.056**. Both BALANCED.

**4.3b In-window SITE DENSITY is a descendant of the exposure, not a confounder (red-team M4′).**
A short downstream intron mechanically pulls the next exon into the 1024-nt window, so the count of
other annotated in-window sites falls monotonically with `L_down`
(`corr(log L_down, n_real_don) = −0.31`; 2.07 → 0.95 across `L_down` quintiles). Adjusting for it in the
**primary** would silently convert the estimand from a total effect into a controlled direct effect and
open a collider path. Therefore:

- **Primary:** density is NOT matched and NOT gated; its post-match SMD (val: `n_real_don` +0.494,
  `n_real_acc` +0.314) is reported as a property of the exposure.
- **Secondary D1 (controlled direct effect):** the same contrast with `n_real_don` and `n_real_acc`
  **exact-matched** (plus a `donor_pos ≤ 80` caliper to restore common support). Outcome-blind achieved
  on val: **1,245 pairs / 1,241 genes, worst |SMD| 0.096.**
- **Interpretive rule, frozen:** D1 is what adjudicates *partner-specific architecture switch* vs
  *generic dilution by competing in-window sites*. If `E_switch` survives in D1, the site-density
  account is excluded. **If it collapses in D1, the §6 verdict wording must be bounded to
  "architecture, including its site-density consequences"** — the top-row claim is not licensed in its
  partner-specific form.

**4.4 `clean` is NOT a matching covariate.** The native margin `f_N` is a component of every outcome
(`C = f_N − f_perturbed`); matching on it would condition on part of the outcome — the collider error
IC1 removed in its Amendment A2. Donor motif strength `s_don` (exogenous, genomic, the main sequence
determinant of `f_N`) is matched instead. `f_N` is reported as a diagnostic only.

**4.5 Buffer band.** The 201–299 nt units are scored but excluded from the 2×2; they feed only the
continuous distance profile (§7 S4).

## 5. PRIMARY (confirmatory) — one intersection–union test

Matched-pair differences on the frozen downstream-axis cohort (LONG − SHORT):

```
E_switch = mean_pairs Δ C_exon        I_switch = mean_pairs Δ C_intron
ΔΔ       = E_switch − I_switch                                   <-- the architecture × direction interaction
```

- **Predicted signs:** `ΔΔ > 0` and `E_switch > 0` (exon definition engages as the downstream intron
  lengthens past the transition).
- **Test:** **two-way (dyadic) multiplier bootstrap**, 2,000 resamples, **percentile** interval,
  two-sided **α = 0.05**, on each of `ΔΔ` and `E_switch`. A matched pair joins a treated unit from gene
  `g_t` and a control unit from gene `g_c`, and genes recur on both ends, so a one-way gene bootstrap
  under-covers — the A9b/R22 dyadic-dependence error. Weights: iid mean-1 Exp(1) per gene, pair weight
  `(ω_{g_t} + ω_{g_c})/2`. The **within-pair sign-flip randomization p** is reported alongside as the
  exact null the matched design implies.
- **Inference-selection rule (frozen, outcome-blind, decided on the TRAIN pilot only).** A decision rule
  whose false-positive rate is not ≈α is not a decision rule. The `permuted_label_fpr` gate is evaluated
  on the TRAIN pilot at realistic n, and the primary inference is the **first** method in this frozen
  order whose FPR ≤ 0.05: (1) dyadic multiplier bootstrap percentile CI; (2) **two-way wild sign-flip**
  (iid sign per gene, pair multiplier = the product of its two genes' signs — pairs sharing a gene move
  together, as they must); (3) gene-block permutation. The choice is made **before any val data is
  scored** and is recorded in §8. *(Motivation: at 20 pairs the bootstrap over-rejected at 0.25.)*
- **This is an intersection–union test.** The switch is licensed only if **both** reject in the
  predicted direction; the conjunctive null is controlled at α by the maximum p-value, so **no
  multiplicity correction is applied or needed**. `ΔΔ` alone cannot distinguish a switch from
  cross-intron distance decay (§4.1), which is precisely why `E_switch` is inside the primary rather
  than beside it.
- **SESOI — anchored to the COUPLING SCALE, never to the observed switch.** SESOI = **⅓ × mean
  `C_exon`** measured on the TRAIN pilot **with the linear observable**. Anchoring to the coupling
  magnitude (a property of the phenomenon) rather than to the observed architecture difference (the
  quantity under test) keeps the threshold non-circular: a switch worth less than a third of the
  cross-exon coupling it modulates is not a mode change. Sealed numerically in §8 before any val
  compute.
- **Power:** `sd_between(C_exon)` is likewise re-derived from the TRAIN pilot on the linear observable
  (v1's 0.702 came from a max-margin artifact — red-team M9). A bounded null is licensable only if the
  achieved MDE ≤ SESOI; the achieved MDE is recomputed from val and reported **regardless of outcome**.

## 6. DECISION RULE (frozen before any val outcome is computed)

Preconditions: all §8 memorials PASS **and** all §9 calibration gates PASS **and** `balance_achieved`.
Otherwise: **no biological interpretation**, regardless of the numbers.

| `ΔΔ` | `E_switch` | Licensed conclusion |
|---|---|---|
| CI excludes 0, **> 0**, `|ΔΔ| ≥ SESOI` | CI excludes 0, **> 0** | **ARCHITECTURE-CONDITIONED BOUNDARY PAIRING.** SpliceBERT's donor→acceptor coupling switches direction with flanking architecture, analogous in geometry to intron/exon definition. → licenses the contingent circuit prereg (does L3H1 retarget?). |
| CI excludes 0, > 0 | CI **includes** 0 | **NOT licensed as a switch.** The interaction is carried by the distance-entangled cross-intron half; consistent with cross-intron distance decay plus a direction main effect. Report as such. |
| CI excludes 0, > 0, `|ΔΔ| < SESOI` | CI excludes 0, > 0 | **Small-but-detectable** architecture dependence; sub-SESOI, not a mode switch. |
| CI includes 0, wholly within ±SESOI | any | **BOUNDED NULL:** no architecture-conditioned switch ≥ SESOI in this subdomain ⇒ the biological analogy is **bounded to short-intron coupling or generic positional dependence** (the user's third outcome). Report achieved MDE. |
| CI includes 0, extends beyond ±SESOI | any | **Inconclusive** (underpowered). |
| `ΔΔ < 0` with CI excluding 0 | — | **Reverse switch** — report as a falsification of the predicted direction; do not re-label. |

Sign is preregistered: a CI excluding 0 in the *wrong* direction is a falsification, never a
"significant effect".

## 7. SECONDARY (preregistered, reported, NEVER licensing)

- **S1 — full 2×2 profile.** `C_exon`, `C_intron`, `S_exon`, `S_intron` in each of SS/SL/LS/LL with
  gene-clustered CIs; the primary contrast re-run within each upstream band (val: 2,404 pairs at up-L,
  1,068 at up-S).
- **S2 — UPSTREAM-flank axis (the clean architecture manipulator).** `C_exon|up-L − C_exon|up-S` on the
  frozen upstream-matched cohort. Upstream intron length is **not** a separation for either coupling, so
  this axis carries no distance entanglement at all. Predicted > 0 (a long upstream intron forces the
  upstream acceptor to be exon-defined with this donor). *Reported, not licensing, because the classical
  transition is stated in downstream-intron terms.*
- **S3 — specificity.** `ΔΔ_sham` (same contrast built from `C_exon_sham` / `C_intron_sham`) must be
  ≈0; `S_exon` / `S_intron` are the sham-subtracted partner-specific couplings and the primary contrast
  is re-run on them.
- **S4 — distance profile / transition location.** `C_exon` vs `L_exon` and `C_intron` vs `L_down` over
  the full scored range including the buffer. Is there a transition near 200–250 nt in the cross-intron
  curve that is absent from the cross-exon curve? Descriptive; no change-point inference is claimed.
- **S5 — within-molecule substitution (Γ).** `Γ_specific`, and its architecture dependence. Tests
  causally, inside a single molecule, whether destroying the cross-intron partner *increases* reliance
  on the cross-exon partner. **Coherence prediction, preregistered:** if the effect is an architecture
  switch rather than a content difference, `Γ_partner|down-SHORT ≈ E_switch` — i.e. removing the
  cross-intron partner in a short-intron molecule should recapitulate the natural long-intron
  architecture. Γ is a genuine nonlinearity (nothing in `f` forces additivity), *unlike* the
  architecture-forced `Δ_bind` of R23 — but a floor/saturation effect can also produce `Γ ≠ 0`, which is
  what `Γ_sham` is subtracted for. **Caveat (red-team M8):** `Γ_sham` is built from a sham whose own
  effect is ≈0, so it cannot probe the region of the `f` nonlinearity that the large partner effect
  visits — it is a **lower-bound** saturation control, exactly as IC1's PPT sham was for AG-channel
  damage. `Γ_specific` therefore carries no verdict.
- **S5b — D1, the site-density controlled direct effect** (§4.3b). Adjudicates partner-specific switch
  vs generic dilution. Its interpretive rule is frozen in §4.3b and constrains the §6 wording.
- **S6 — within-gene robustness.** Restricted to genes contributing both a SHORT and a LONG unit, so
  gene-level expression/isochore context is held exactly fixed.
- **S7 — sampling stability (R22).** The matched cohort is split into 3 disjoint gene blocks; `ΔΔ` and
  `E_switch` are reported per block. **Instability trigger: across-block SD > 0.5 × SESOI = 0.05** ⇒
  report as heterogeneous and do not lean on the pooled mean. (Held-out ≠ sampling stability.)
- **S8 — robustness readouts.** Max-margin observable; 5 %-winsorized means; the within-pair sign-flip
  p. Reported alongside the primary; the primary is the untransformed mean.
- **S9 — tail concentration (red-team M3).** `C_exon` is heavy-tailed (train max-margin: mean 0.288,
  median 0.045, max 4.06), so the population mean is carried by roughly a tenth of the units. The share
  of `E_switch` contributed by its top decile is reported so the result is not read as a typical-donor
  effect. **No secondary can upgrade the §6 verdict.**

## 8. MEMORIALS (hard preflight; run STOPS on failure)

Each is **on the measurement path** and **can fail** (R7/R21); forced ones are labelled as forced.

| memorial | check | can fail on |
|---|---|---|
| `variant_construction_exact` | every constructed state differs from its parent in **exactly** the intended 2 bp, at the intended index, is length-preserved, bit-identical elsewhere, and no longer presents `AG` at the target; `D⁻u⁻` differs from `D⁻` in exactly the sham 2 bp; the sham index used in `u⁻` and `D⁻u⁻` is identical | any off-by-one in window/position mapping, sham re-selection between states |
| `readout_anchor` | the masked token index is `donor_pos+1`; the token there **before** masking decodes to the native base `G`; `seq[d:d+2]` is `GT` | tokenizer/offset drift, `[CLS]` convention change |
| `batch_invariance` | `f_N` recomputed in an independently-composed second batch reproduces the first to `<1e-4` | batch-composition / padding non-determinism on MPS |
| `native_coupling_present` | `C_intron` on the down-SHORT stratum ≥ **0.265** (§9b; TRAIN-sealed at 0.4× the observed 0.663 on the linear observable) | any break anywhere in dataset → variant → scoring → contrast |
| `balance_achieved` | worst post-match `|SMD|` < **0.10** on the frozen matched cohort | matching failure / insufficient overlap |
| `determinism_hashseed` | cohort membership, matching, sham choice, bootstrap and sign-flip p identical across `PYTHONHASHSEED ∈ {0,1,17}` — enforced by `tests/test_ed1_determinism.py` (subprocess per seed), plus `PYTHONHASHSEED=0` pinned at launch | the IC1-A2 `set()`-iteration reproducibility class of bug |
| `identity_noop_exact` | substituting the native dinucleotide back gives `ΔC == 0` exactly | *(labelled FORCED-by-construction: a determinism/plumbing check, NOT a completeness proof — R21)* |
| `did_regrouping_identity` | `Γ` computed by the two algebraic groupings agrees `<1e-9` | *(labelled FORCED: linear-`f` algebra. Reported, validates nothing on its own)* |

## 9. CALIBRATION (hard preflight; identification claim ⇒ sensitivity AND specificity)

| gate | construction | pass |
|---|---|---|
| `sensitivity_decision` (planted positive) | plant exactly `+SESOI` on a **null-centred** cohort with the real dispersion (planting on top of the observed effect would test detection of *effect + SESOI*, red-team B2); run the **frozen** test | rejects in ≥ **80 %** of draws |
| `permuted_label_fpr` (specificity) | sign-flip the architecture label **within matched pairs** (the exact randomization null of a matched design), re-run the frozen test. **Not re-centred** — re-centring forces the statistic to zero and makes the gate vacuous (red-team B1) | **graded, per Amendment A1:** ≤0.05 clean; ≤0.10 proceeds with the calibrated `p*` applied to positive verdicts; >0.10 STOPS |
| `planted_confound_null` (confound-specificity) | plant an outcome driven **purely** by `gc_intron` (β chosen so the UNMATCHED estimator rejects) | the **matched** estimator rejects ≤ **10 %** — i.e. matching actually blocks the gene-composition backdoor |
| ~~`sensitivity_onpath` (through-model)~~ | **NOT IMPLEMENTED — declared, not run.** A planted consensus 3′ss would test the same substitutability that `Γ` already measures causally, and a null result would be ambiguous between "pipeline blind" and "model genuinely does not switch" — the very ambiguity a sensitivity gate must not have. Its role is carried by `native_coupling_present` (which certifies the whole dataset → variant → scoring → contrast path against a large known effect) and by `sensitivity_decision`. | — |

`--smoke` = memorials + `sensitivity_decision` + `permuted_label_fpr` + `planted_confound_null` on
tiny-n **TRAIN** data. **Val is sealed during smoke.**

## 9b. SEALED constants (from the TRAIN pilot `ed1_trainpilot.json`, n=4,000 units / 729 pairs)

All on the **linear** observable (red-team M9). No val data was read to set any of these.

| Parameter | SEALED | Basis |
|---|---|---|
| **SESOI** | **0.106 nats** | ⅓ × n-weighted mean `C_exon` over the 2×2 = 0.3192/3. Anchored to the coupling scale, **not** to the observed switch. |
| **`native_coupling_present` floor** | **0.265** | 0.4 × observed `C_intron` on down-SHORT (0.663) |
| `sd_between(C_exon)`, gene level | 0.698 | pilot |
| pilot achieved MDE | 0.0940 at 729 pairs | pilot |
| **projected MDE at val (3,472 pairs)** | **0.0431 ≤ SESOI** | ⇒ **a bounded null is licensable** |
| inference | dyadic multiplier bootstrap (additive multipliers) | §5 selection rule |
| **null-calibrated critical `p*`** | **0.030** | 5th percentile of the bootstrap p under the design's own sign-flip null |
| balance | worst \|SMD\| 0.092 (val, outcome-blind) | matching pilot |

*Pilot diagnostics, for context only — the pilot is TRAIN and licenses nothing:* `ΔΔ = +0.597`,
`E_switch = +0.043 [−0.023, +0.110]` (**includes 0**), `I_switch = −0.554`, upstream axis
`+0.008 [−0.062, +0.075]`, `ΔΔ_sham = +0.011 [−0.010, +0.031]`, top-decile share of `E_switch` 0.573.

---

## Amendment A1 (2026-07-25, PRE-RUN, outcome-blind — no val data scored)

**Trigger.** §5's inference-selection rule says to take the first method whose permuted-label FPR
≤ 0.05. On the TRAIN pilot **neither** qualified: dyadic bootstrap 0.087, two-way wild 0.067. The rule
did not cover "none passes", so it is extended here — on TRAIN, before any val compute.

**Why the FPR is above nominal.** `C_exon` is heavy-tailed (top-decile share 0.573), and bootstrap
interval coverage degrades under heavy tails. This is a property of the statistic, not a bug — the
implementation was already verified against a true-null simulation (FPR 0.065/0.055 on Gaussian-plus-t
data).

**The extension, with its asymmetry made explicit.** Over-rejection **inflates a positive claim** but
makes a **null conclusion conservative** — a test that over-rejects and *still* fails to reject is
stronger evidence for the null, not weaker. So the gate is graded rather than binary:

- FPR ≤ 0.05 → clean; nominal α applies.
- 0.05 < FPR ≤ 0.10 → run proceeds; **any POSITIVE verdict must additionally clear the
  null-calibrated critical value `p ≤ p*`**, where `p*` is the 5th percentile of the bootstrap
  p-value under this cohort's own within-pair sign-flip null (empirical FPR = 5 % by construction).
  A NULL verdict is unaffected.
- FPR > 0.10 → STOP.

**Also amended (same pass, same justification):** the sampling-stability trigger is evaluated on
**`E_switch`**, the statistic the verdict turns on, not on `ΔΔ`. An absolute SD threshold of
0.5 × SESOI fires trivially against a `ΔΔ` that is ~6 × SESOI. Pilot: SD(`E_switch`) = 0.028 < 0.053
(no trigger); SD(`ΔΔ`) = 0.067.

**Legitimacy.** Decided on the TRAIN pilot; no val outcome computed or inspected; the cohort, the
perturbation, the estimands and the decision table are UNCHANGED. The val run remains **confirmatory**.

## 10. Design red-team (REQUIRED before freeze)

Four independent lenses — identifiability, statistics, confound, estimand↔intent — must be run against
this document and their must-fixes applied **before** any val compute. Recorded in
`studies/splicebert_partner_selection/redteam/REDTEAM_SYNTHESIS_ED1.md`.

## 11. Scope, outputs, provenance

- **Establishes at most:** a *behavioural*, architecture-conditioned coupling asymmetry in a frozen MLM.
  **Does NOT establish:** spliceosome assembly, U1/U2 interactions, cross-exon complexes, that L3H1 (or
  any component) implements it, or anything about introns >800 nt.
- **Residual, uncontrolled caveats (declared, not fixed):**
  (i) **Repeat content** — long introns carry more interspersed repeats; GC matching captures only part
  of this, and the design has no handle on the rest (red-team M6).
  (ii) **Selection of the down-LONG arm** — `next_acceptor_pos` exists only when the downstream acceptor
  is in-window, so down-LONG units are conditioned on `L_down ≲ 1020 − donor_pos`, a function of both
  the exposure and window position (red-team M7). `donor_pos`/`acc_pos` are matched, so the contrast is
  unbiased *within* the in-window subdomain, but the estimand is **defined on that subdomain** and does
  not extrapolate to deep-intronic architecture.
  (iii) **Saturation** — bounded only to the lower-bound quality of `Γ_sham` (M8).
- **Deferred/contingent:** circuit localization (does L3H1 retarget upstream? is there a second core?)
  is a separate prereg, licensed only by the top row of §6.
- **Outputs:** `scripts/splicebert/run_ed1_architecture_boundary_pairing.py`; result JSON + manifest +
  preflight JSON; matched-cohort membership; per-cell tables; all CIs gene-clustered BCa.
- **Provenance:** checkpoint sha256 `2ad91428…0149d`; SpliceBERT.1024nt, 6L×16H, hidden 512;
  GENCODE v50 basic; hg38; PWM `colab_results/splicebert_run/baselines/local_pair_baseline.pkl`;
  `PYTHONHASHSEED=0` pinned at launch; all thresholds and sham/matching rules frozen above.
