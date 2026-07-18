# Pre-registration — IC1: does distal intron interstitial ARRANGEMENT modulate donor–acceptor coupling?

**Status: v8 (2026-07-17). Amendment A1 (outcome-blind) corrected the mis-calibrated distal-OOD gate.
Amendment A2 then fixes a shuffler REPRODUCIBILITY bug + removes a tautological verdict-precondition — but
these were made AFTER ΔC=+0.033 was seen on the v7 run, so v8 is an OUTCOME-AWARE DETERMINISTIC
VERIFICATION, NOT a confirmatory analysis (seeing ΔC forecloses confirmatory status regardless of
transparency). A publication-grade confirmatory null would require a genuinely untouched, disjoint gene
cohort/dataset (future).** Behaviour-first: full-model coupling assay only; circuit localization
(L3H1 / payload / `U_info`) is a
separate, contingent later prereg. Opened after R22 (payload semantic identity unresolved; next target =
conditional splice competitiveness).

**Lineage:** v1 → 5-lens red-team (11 fixes) → v2 → train pilots (`ic1_pilot.json`) → v3 → 4-lens
freeze-audit → v4 → pilot2 seals (`ic1_pilot2.json`) → v5 (conservative-scramble primary + deterministic
sham) → **3-lens v5 re-audit** → **v6**: dispersion **demoted to a secondary reported readout** (the PPT
sham cannot bracket AG-channel state-dependent damage, so it must not FWER-gate); the **Δf⁺ regression
deleted** (it conditions on a component of ΔC); **N reconciled to ~1264**; everything scoped to the
**conservative-constructible** subdomain. All thresholds sealed from the DISJOINT train split; **no val data
read before the run.** Every gate can FAIL and is on the measurement path (R7/R21); no post-hoc/outcome
selection (R22).

---

## 0. INTENT (north star — re-read at every gate)

> **Does the ARRANGEMENT (ordering) of distal intron sequence — mono+di composition, both splice-site
> contexts, the branchpoint/PPT zone (to −70), intron length, positional address (up to triplet-phase),
> AND created cryptic-site competition all held ~fixed — change the population-average donor–acceptor
> coupling, in the conservative-constructible minimal-intron subdomain (ACC_UP=70, L≳107)?**

Honest boundaries: (i) mono+di composition held EXACTLY fixed → tests dinuc-composition-controlled ORDERING
(incl. its k≥3-mer/phase content), never "distal content." (ii) The **conservative primary arm** bounds
cryptic-site change BY CONSTRUCTION (§2), so the claim is clean, not stratum-contingent. (iii) The
confirmatory estimand is the **population-average directional** effect (signed mean); an equivalence result
supports only *"no meaningful average directional modulation ≥0.05 in the constructible subdomain"* — NOT
"arrangement doesn't matter" (individual introns may still respond in opposing directions; that is the
demoted dispersion question, not confirmatorily testable here). Scope excludes the protected BPS/PPT, the
cryptic-rich introns that fail conservative construction (~6%, reported separately, §7), and
deep-intronic/exon-definition regimes.

---

## 1. Domain, observable, eligibility, protected zones, DISTAL window

- **Observable (frozen LINEAR functional):** `f = logit(G) − mean{logit A,C,T}` at the masked donor
  (`donor_pos`, intron-start GT). Linear (NOT the max-margin task metric) — the DiD requires linearity.
  `A⁺` native; `A⁻` = partner AG→CC at `[a−2,a−1]` (`a=next_acceptor_pos`).
- **Domain:** canonical internal introns (`GT…AG`), native pair, on-strand, in-window; `L∈[60,150]`.
- **Protected zones (bit-identical):** donor 5′SS `[d₀−3 .. d₀+6]`; acceptor `[a−70 .. a+3]` (**ACC_UP=70**,
  SEALED — protects 3′SS+PPT+branchpoint to −70, no BP predictor). Robustness subset **ACC_UP=100** (L≳137)
  — required to sustain any positive claim (else *provisional, not confidently distal*).
- **DISTAL window** `W = [d₀+7 .. a−71]` ⇒ minimal `L ≳ 107`.
- **Inference unit = homology cluster** (≥80% id). Pilot near-dup 0.5% ⇒ **cluster ≈ gene**.
- **Eligibility (else EXCLUDE):** (1) `ℓ_dist ≥ 30`; (2) **dose floor** (§5); (3) **conservative-
  constructible** — the intron admits `N_SEED` valid conservative shuffles within the frozen sampler budget
  (§2/§5). **Sealed primary N ≈ 1264 clusters** = (dose-pass 0.61) × (conservative-yield 0.94) × 2205.

---

## 2. Perturbation — 4-arm structure (`N_SEED=16` per intron per arm)

Perturb only `W`; protected zones bit-identical; **whole-intron dinuc multiset preserved exactly** (all
arms; unit-tested). Run on the SAME introns. Seeds preregistered (content-independent).

1. **CONSERVATIVE SCRAMBLE — PRIMARY** (`ic1_pilot3.json`): anchored (GT/AG-position-fixed) dinuc shuffle
   with cryptic change **bounded BY CONSTRUCTION** via rejection: `|Δmax score3|≤2`, `|Δmax score5|≤2`,
   `|Δmax poly-Y|≤2`, no NEW 23-mer score3>6.0, count(score3>6.0) preserved, + dose floor. **Frozen
   sampler:** per-intron **max_attempts=400**; proposal RNG = `default_rng((gene_hash*17)+seed+1)`
   (deterministic given intron, seed, budget); accept the FIRST proposal passing all constraints (tie-break
   = first-accepted); **exclusion = budget-exhaustion** (no valid shuffle in 400 attempts). Pilot: yield
   0.94 (constraints do NOT collapse dose — normHam 0.36), max observed `|Δscore3|`=1.99.
2. **ANCHORED — SECONDARY:** GT/AG positions fixed, cryptic PPT/3′SS free.
3. **UNCONSTRAINED — SECONDARY:** whole-`W` dinuc shuffle (max cryptic creation/destruction).
4. **Arm CONTRASTS (secondary):** paired `anchored−conservative` and `unconstrained−anchored` (§6);
   estimate cryptic-competition — a **lower bound** (computable only on the conservative-constructible set,
   which excludes the cryptic-richest introns).

---

## 3. Estimand, primary, and supporting diagnostics

Per intron, per seed: `ΔC = [f(I',A⁺) − f(I',A⁻)] − [f(I,A⁺) − f(I,A⁻)]`.
- **PRIMARY (confirmatory):** the signed **`ΔC̄`** (seed-mean → homology-cluster mean). One test (§4).
- **Symmetric main effects (validity diagnostic, NO conditioning):** report `Δf⁺=f(I',A⁺)−f(I,A⁺)` and
  `Δf⁻=f(I',A⁻)−f(I,A⁻)`; the pure-damage null IS `Δf⁺=Δf⁻`. **No regression of ΔC on Δf⁺** (that conditions
  on a component of ΔC ≡ Δf⁺−Δf⁻ — a collider); damage is controlled by the sham DiD below, not by adjusting
  on outcome components.
- **SHAM DiD (validity diagnostic):** deterministic non-partner PPT purine-swap `A⁻' = substitute(a−8,'AAAA')`
  (pilot3 best population match, median f-shift **0.56×** — PPT swaps cap there; AG left intact).
  `E[ΔC] − E[ΔC_sham]` and the gate **`ΔC_sham` CI ⊆ `[−0.05,+0.05]`** (pilot mean 0.005). This certifies
  state-independence only **at the sham corruption (AG intact, ~0.56× damage)** — an incomplete lower bound
  on full-corruption state-dependence.
- **DISPERSION (SECONDARY, reported — NOT confirmatory):** per-intron across-seed **RMS** of `ΔC`,
  excess over the sham `= RMS(ΔC) − RMS(ΔC_sham)`, with its cluster BCa CI (and `RMS_sham` reported with its
  own CI). **Explicit caveat:** the PPT sham **cannot exclude AG-channel state-dependent damage** (it leaves
  the AG intact, whereas `A⁻` destroys it), so excess dispersion is **not** used to establish arrangement
  sensitivity or to rescue a null signed mean. Pilot3 (train): `RMS(ΔC)`≈0.112, `RMS_sham`≈0.081 (~0.72×),
  excess≈0.031. A valid AG-channel dispersion control is a **separate future methods problem** (Cam).
- **Dose-response (secondary):** `ΔC` vs realized shuffled-fraction, covaried on `ℓ_dist`, `L`.

---

## 4. Inference (SEALED — one confirmatory test)

- **Unit:** homology cluster (≥80% id ⇒ ≈ gene). **Confirmatory N = the conservative-constructible census
  ≈ 1264 clusters** (dose ∧ construction; frozen, no run-time N choice). The all-arm-common ~1264 set is also
  the basis for the secondary paired contrasts.
- **ONE confirmatory test:** the signed cluster-mean **`ΔC̄`**, **two-sided α=0.05**, **cluster (wild/multiplier)
  bootstrap, BCa CI**. (No max-statistic; `z_disp` is REMOVED from all pass/fail logic.) Secondary arm,
  dose-response, sham DiD, dispersion, ACC_UP=100 robustness = reported, not FWER-gated.
- **Power (SEALED):** signed `MDE_census` ≈ **0.013** ≤ SESOI at N≈1264 (`sd_between`=0.162) ⇒ the signed
  null is licensable. (Secondary dispersion MDE ≈ 0.005 — pilot3 0.0036 at N=2069, recomputed at N≈1264;
  reported, not licensing.)
- **Sampling stability:** census default; if sub-sampled, multiple independently-seeded draws + stability;
  **instability trigger** (SEALED): across-draw SD(`ΔC̄`) > 0.5×SESOI = 0.025 → heterogeneity branch (§6).

---

## 5. Gates / memorials (HARD preflight; run STOPS on failure unless noted)

- **`protected_bit_identical`**, **`length_position_preserved`**, **`whole_intron_dinuc_preserved`** (anchor
  GT/AG positions + whole-window multiset; unit-tested).
- **`perturbation_dose`** (on-path, can fail): all `N_SEED` distinct; median altered ≥ 8; median normHam ≥
  0.25; exclude near-identity before inference.
- **`conservative_construction`** (on-path, can fail; PRIMARY arm): every accepted shuffle satisfies the §2
  cryptic bounds (verifies the sampler; pilot max `|Δscore3|`=1.99); exclusion = budget-exhaustion (frozen
  400 attempts). **Report the excluded (~6%) vs retained covariate profiles** (max poly-Y, count(score3>6),
  max score3, `L`, `ℓ_dist`); **escalation:** if excluded fraction > 10% OR standardized mean-diff on those
  covariates > 0.5, label the primary **scope-restricted**, not domain-general.
- **`distal_ood` (v7 corrected — Amendment A1):** INCREMENTAL, WITHIN-STATE `ΔNLL_s = NLL(I',A_s) −
  NLL(I,A_s)`, `s∈{+,−}` (acceptor corruption cancels ⇒ cannot fail a distal gate). NONINFERIORITY to a
  NATURAL-to-NATURAL null: `band_s` = p95 of natural-swap `ΔNLL_s` on the cohort; PASS iff conservative
  in-band rate `≥` natural in-band rate `− δ_OOD`, per state, **gene-clustered** (same aggregation for
  natural & shuffled). `δ_OOD = 0.044` frozen from `ic1_pilot4_ood` (natural-rate gene-clustered SEM — NOT
  tuned to any shuffle rate). Demonstrated: passes natural / identity / conservative-pilot, FAILS a planted
  alt-GC OOD. `|Δf|` = damage report only.
- **`sensitivity_onpath`** (SEALED PASS): plant `"T"×18+"AGG"` mid-`W`; through-model `|ΔC_plant|`≥SESOI —
  pilot absmedian 0.087, 69%≥0.05 ⇒ the pipeline detects a distal coupling change of the target size ⇒
  signed null is on-path licensable. In `--smoke`.
- **`identity_noop_exact`** (determinism/plumbing per R21): identity ⇒ `ΔC==0` exact.
- **`native_coupling_present`**: baseline coupling of expected sign/scale on the eligible subdomain.

`--smoke` = mechanics + dose + conservative-construction + `sensitivity_onpath` on tiny-n.

---

## 6. Decision rule (SEALED; SESOI=0.05 nats; signed mean only; scope = conservative-constructible, L≳107)

Preconditions for ANY primary interpretation: §5 gates PASS **AND** `ΔC_sham` mean CI ⊆ `[−0.05,+0.05]` and
the `Δf⁺=Δf⁻` symmetry holds (else **primary interpretation compromised**, regardless of the signed result).

| Signed-mean result | Licensed conclusion |
|---|---|
| Test rejects (two-sided α=0.05) **and** `|ΔĈ|≥0.05` **and** holds on the ACC_UP=100 subset | **Conservative interstitial rearrangement changes AVERAGE donor–acceptor coupling** in the subdomain (→ next prereg: localize). If it fails the ACC_UP=100 subset: *provisional, not confidently distal*. |
| `ΔC̄` CI ⊆ `[−0.05,+0.05]` | **No population-average DIRECTIONAL effect ≥0.05** in the conservative-constructible L≳107 subdomain (NOT "arrangement doesn't matter"). Achieved MDE reported. → IC2/IC5. |
| Signed null **but** excess dispersion elevated | **Possible heterogeneous sensitivity — CONFOUNDED with AG-state-dependent damage**; not licensed as arrangement sensitivity; a valid AG-channel control is required follow-up. |
| Signed CI straddles ±0.05 | Inconclusive. |

- **Health/OOD failure** (global): no biological interpretation; redesign.
- **Heterogeneous** (across-draw SD(ΔC̄) > 0.025): characterize intron-specific dependence before any mean claim.
- **Cryptic-site competition (secondary):** paired `anchored−conservative`, `unconstrained−anchored` cluster
  bootstrap (not difference-of-significance); a **lower bound** scoped to the constructible set.

---

## 7. Scope, caveats, outputs, deferred

- **Behaviour-first**; circuit localization deferred.
- **Caveats:** (i) dinuc-composition-controlled ORDERING, not composition; (ii) scoped to the
  **conservative-constructible, L≳107** subdomain (robustness ACC_UP=100 ⇒ L≳137) — intron-definition
  substrate; the excluded cryptic-rich ~6% are reported separately and NOT extrapolated to; (iii) **C8
  period-3 clock:** dinuc-shuffle scrambles triplet phase, so "position held fixed" holds only to
  triplet-phase — report `ΔC` vs distal triplet-periodicity; (iv) dispersion is secondary and its sham floor
  cannot exclude AG-channel damage.
- **`ΔC_meaningful`=SESOI=0.05 nats**; always report the achieved MDE.
- **Adversarial verification is a MANUSCRIPT-STAGE requirement.**
- **Outputs:** `scripts/splicebert/run_ic1_distal_intron_necessity.py`; JSON + manifest + preflight;
  eligible-`L` + exclusion histograms; **conservative-arm rejection yield + constraint-pass counts**; the
  **excluded-6% cryptic covariate profile**; `ΔC̄` (primary, BCa CI), `Δf⁺/Δf⁻`, sham DiD, **dispersion
  excess-over-sham with CI (secondary)**, per-arm + paired contrasts, ACC_UP=100 robustness,
  seed/draw-stability — all with homology-cluster BCa CIs.
- **Provenance:** checkpoint sha256 `2ad91428…0149d`; MaxEnt pinned (`maxentpy 0.0.4`, `8b6d7b46…b7c00`);
  anchored/conservative shuffler unit-tested; sampler budget (400 attempts) + RNG stream + tie-break frozen;
  all thresholds + seeds preregistered.

---

## 8. SEALED thresholds (from disjoint train pilots `ic1_pilot{,2,3}.json`)

| Parameter | SEALED | Pilot basis |
|---|---|---|
| ACC_UP primary / robustness | **70 / 100** | 2205 / 560 genes; BP protected to −70 |
| PRIMARY arm | **conservative scramble** (cryptic bounded by construction) | yield 0.94 |
| Confirmatory **N** | **≈1264 clusters** (dose 0.61 × yield 0.94 × 2205) | reconciles the 2205/2069/1345 figures |
| N_SEED | **16** | 8↔16 dispersion corr 0.90 |
| SESOI | **0.05 nats** | above non-partner floor (Cam) |
| dose floor | median-altered ≥8; normHam ≥0.25; all 16 distinct | ~61% pass (pilot1, n_seed=8 est.) |
| sampler budget | **400 attempts**, frozen RNG stream, first-accept tie-break, exclusion=budget-exhaustion | yield 0.94 at 400 |
| confirmatory test | **signed `ΔC̄` only**, two-sided α=0.05, cluster **BCa** bootstrap (no z_disp/max-stat) | Cam (Option A) |
| signed MDE_census | **0.013 ≤ 0.05** at N≈1264 | sd_between 0.162 (pilot3) |
| dispersion (SECONDARY, reported) | excess over sham + CI; MDE≈0.005 (pilot3 0.0036 @N=2069 → N≈1264) | RMS_sham 0.081 vs RMS 0.112; **cannot exclude AG-channel damage** |
| sham `A⁻'` (deterministic) | `a−8→'AAAA'` (0.56×, AG intact); DiD diagnostic + gate ΔC_sham CI⊆[−.05,.05] | mean 0.005 (pilot3) |
| on-path positive | plant `"T"×18+"AGG"` mid-W; \|ΔC_plant\|≥0.05 | absmedian 0.087, 69%≥.05 ⇒ PASS |
| cryptic control (primary) | conservative arm bounds by construction | max \|Δscore3\|=1.99 |
| OOD (v7, Amendment A1) | incremental within-state ΔNLL, noninferiority to natural-to-natural null; δ_OOD=0.044 (gene-clustered SEM) | conservative in-band 1.00 vs natural 0.95; demonstrated (pilot4) |
| homology | cluster ≈ gene | near-dup 0.5% |
| instability trigger | across-draw SD(ΔC̄) > 0.025 | 0.5×SESOI |

**Resolved re-audit items:** dispersion demoted to secondary (no z_disp FWER breach); Δf⁺ regression deleted
(collider); N reconciled to ~1264 + MDE recomputed (0.013, null still licensable); primary scoped to the
conservative-constructible subdomain with the excluded-6% reported separately; sampler budget frozen;
RMS_sham corrected to 0.081; stale no-new-motif language removed.

**No further sham pilot for IC1** (Cam): a valid AG-channel dispersion control is a separate future methods
problem. **Remaining before GO:** one final mechanical audit of estimands/thresholds/output-names; then the
confirmatory-script build. No val data is read before the run.

---

## Amendment A1 (2026-07-17) — outcome-blind correction of the distal-OOD gate

**What triggered it.** The v6 confirmatory ran on val and FAILED preflight on `distal_ood` (excl_frac
7.45% vs a flat 5% cutoff). Per §5 the run **did not compute or write any primary ΔC statistic or verdict**,
and none was inspected. The gate failure is demonstrable **without reference to the scientific outcome**.

**Why the original gate was invalid (mathematically miscalibrated).** The v6 disposition compared a
COMPOUND per-intron / per-cell exclusion rate against the **p95** sequence-level band used to DEFINE
"in-band." Because ~5% of even natural sequences exceed a p95 band by construction, and the compound
(multi-cell, p90-of-tail) structure inflates that base rate, the gate **fails in-distribution perturbations
by construction** (an R21-class gate that cannot serve its purpose). The observed 7.45% is squarely the
in-distribution base rate; the pilot directly shows the conservative perturbation is in-distribution
(shuffled median NLL ≈ natural; ~95–100% in-band), and the conservative shuffle is *less* perturbing than
swapping in another natural intron's window.

**The correction (identified during the pre-run v5 re-audit; calibrated in `ic1_pilot4_ood`).** Measure the
shuffle's INCREMENTAL, WITHIN-STATE OOD effect `ΔNLL_s = NLL(I',A_s) − NLL(I,A_s)` (acceptor corruption
cancels), and gate by NONINFERIORITY to a NATURAL-to-NATURAL null: conservative in-band rate `≥` natural
in-band rate `− δ_OOD`, per acceptor state, gene-clustered, same aggregation for natural and shuffled.
`δ_OOD = 0.044` is fixed EXCLUSIVELY from the natural-rate gene-clustered SEM on the disjoint pilot — never
from the observed 7.45%. **Demonstrated (`ic1_pilot4_ood`, disjoint TRAIN):** passes natural(out-of-sample)
0.95, identity 1.00, and the conservative pilot 1.00 in-band; **FAILS** a planted alt-GC OOD (0.77);
threshold 0.906.

**Legitimacy.** ΔC remains unseen/uncomputed; the failure is outcome-independent; the replacement rule was
pre-identified in the v5 audit; the perturbation, cohort, estimand, and statistical test are UNCHANGED.
**Recordkeeping:** the original failed artifact is preserved (`preflight_ic1_v6_FAILED.json`); the rerun
uses the SAME cohort, RNG streams, perturbations, and thresholds and is written to a NEW artifact
(`ic1_distal_intron_necessity_v7.json`), not overwriting the failed run. The final result is an
**outcome-blind, protocol-amended confirmatory analysis.**

---

## Amendment A2 (2026-07-17) — reproducibility + verdict-precondition fixes; OUTCOME-AWARE (ΔC seen)

**Trigger.** After the v7 run, reconciling a cohort-size discrepancy (v6=255, v7=261; a diagnostic pass
gave 259) revealed a REPRODUCIBILITY bug, and inspecting the v7 verdict revealed a mis-specified precondition.
**Both were found AFTER `ΔC̄ = +0.033` (CI [+0.011, +0.055]) had been computed and inspected on v7.** Seeing
ΔC forecloses outcome-blind confirmatory status, so v8 is labelled an **outcome-aware deterministic
verification**, NOT a confirmatory analysis.

**Bug 1 — non-reproducible cohort.** `dinuc_shuffle` iterated `set(seq)` (a set of characters), whose
iteration order is `PYTHONHASHSEED`-dependent, so the Eulerian vertex order — and hence every shuffle
realization — varied across processes. This flipped the ~20 introns on the conservative-constructibility
boundary, giving a non-reproducible cohort (255–261). **Confirmed:** with a pinned seed the cohort is
deterministic (`PYTHONHASHSEED=0`→260 twice, `=7`→260); only unpinned runs vary. **Fix:** `sorted(set(seq))`
(hashseed-independent) + pin `PYTHONHASHSEED=0` at launch (defense in depth) + a determinism test
(`test_ic1_determinism.py`) asserting identical shuffles, cohort membership, and counts across
`PYTHONHASHSEED ∈ {0,1,2,17}`. NOTE (Cam): the sort changes Eulerian traversal broadly, so the v8 cohort is a
NEW deterministic cohort, not v7's 261; the SEALED population thresholds (pilots) are robust to shuffle
realization and are reused unchanged.

**Bug 2 — tautological verdict-precondition.** §6 (v6/v7) required the "Δf⁺=Δf⁻ symmetry" as a hard
precondition. But `ΔC ≡ Δf⁺−Δf⁻` by construction, so population symmetry `|E[Δf⁺]−E[Δf⁻]| = |ΔC̄|` is the
primary test itself, and the per-intron `mean|Δf⁺−Δf⁻|` is a dispersion measure — neither is a valid
independent gate. **Fix:** DELETE the symmetry (and sham) verdict-precondition; keep Δf⁺, Δf⁻, the sham
DiD, and the symmetry measure as **reported diagnostics only**. The verdict is classified purely by the
signed-mean CI:
- **material:** CI excludes 0 AND `|ΔĈ| ≥ 0.05`.
- **small-but-detectable:** CI excludes 0 AND `|ΔĈ| < 0.05`.
- **bounded null:** CI includes 0 AND entirely within `[−0.05,+0.05]`.
- **inconclusive:** CI includes 0 AND extends outside `[−0.05,+0.05]`.

**Interpretive note (Cam).** The v7 sham DiD ≈ ΔC̄ is CONVERGENT diagnostic evidence of a *nonspecific*
perturbation response, but does NOT prove non-partner-specificity (the PPT sham does not reproduce the AG
channel exactly). Report it as such, not as proof.

**Recordkeeping.** v6 (`preflight_ic1_v6_FAILED.json`) and v7 (`ic1_distal_intron_necessity_v7.json`,
`preflight_ic1.json`) artifacts are PRESERVED; v8 is written separately
(`ic1_distal_intron_necessity_v8.json`, `preflight_ic1_distal_intron_necessity_v8.json`). v8 reuses the
frozen cohort RULES and RNG streams under the fixed deterministic shuffler.

**v8 label:** *Outcome-aware deterministic verification of IC1 following gate and reproducibility
corrections.* If v8 lands near +0.03 with an equivalence-compatible CI and a sham-sized response, record the
bounded/small-detectable diagnostic conclusion and proceed to IC2 (branchpoint/PPT + acceptor-proximal) then
IC5 (competing acceptors); return to distal arrangement only if v8 shows a material, partner-specific effect.
