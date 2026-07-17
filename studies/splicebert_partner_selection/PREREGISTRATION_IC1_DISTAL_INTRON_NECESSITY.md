# Pre-registration — IC1: does distal intron interstitial ARRANGEMENT modulate donor–acceptor coupling?

**Status: CORE DESIGN FROZEN v6 (2026-07-17), pending one final mechanical audit then Cam's confirmatory
GO.** Behaviour-first: full-model coupling assay only; circuit localization (L3H1 / payload / `U_info`) is a
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
- **`distal_ood`**: distal-window masked-LM NLL over `W`, per-intron **p90 tail**, within the empirical band
  = **p95 of NATURAL distal NLL** (≈1.41 nats), re-derived on confirmatory naturals; all four cells, gating
  `(I',A⁻)`; per-intron exclusion, > 5% excluded ⇒ global redesign. `|Δf|` = damage report only.
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
| OOD | band=p95 natural distal NLL (~1.41); p90 tail; 4 cells; >5% excl ⇒ redesign | ~95% in-band |
| homology | cluster ≈ gene | near-dup 0.5% |
| instability trigger | across-draw SD(ΔC̄) > 0.025 | 0.5×SESOI |

**Resolved re-audit items:** dispersion demoted to secondary (no z_disp FWER breach); Δf⁺ regression deleted
(collider); N reconciled to ~1264 + MDE recomputed (0.013, null still licensable); primary scoped to the
conservative-constructible subdomain with the excluded-6% reported separately; sampler budget frozen;
RMS_sham corrected to 0.081; stale no-new-motif language removed.

**No further sham pilot for IC1** (Cam): a valid AG-channel dispersion control is a separate future methods
problem. **Remaining before GO:** one final mechanical audit of estimands/thresholds/output-names; then the
confirmatory-script build. No val data is read before the run.
