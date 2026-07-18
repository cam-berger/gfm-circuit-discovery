# Pre-registration — IC2: does SpliceBERT rank COMPETING candidate acceptors? (behaviour-level)

**Status: DRAFT v3 (post-red-team) for re-audit → pilot-calibrate → FREEZE. DO NOT RUN until frozen.**
Compact acceptor-competition study (supersedes the preserved, not-pursued IC2a); intended FINAL SpliceBERT
experiment before transfer to a second GFM. **Behaviour-level** — mechanistic names (address/payload) are
deferred to a separate mediation study. Carries all IC1 lessons.

**v3 corrections (4-lens red-team → all fixes applied):** the **sham is REINSTATED** (slot- AND
strength-resolved; the reciprocal swap cancels only window-specific AG damage, not the slot-generic
gradient); **κ tested vs a SATURATION baseline** (the sealed max-margin is saturating, so κ≠0 from
curvature alone) + a non-competing control gate; **on-path = partner-SPECIFIC contrast** (not generic
corrupt); **WINDOW = full 3′SS (BP+PPT+AG)**; **near-slot floor + mod-3 phase-balance** (C8); **D_m matched
on MODEL AG-damage** (not score3, per R22); **strength-contrast composition-balanced/DML** (A9b);
**sup-t bands** (not "simultaneous BCa"); **κ-driven power**; **pre-set numeric cohort thresholds**;
**natural alt-3′SS PRIMARY (scoped), synthetic secondary**; decision **gates competition on κ** and
**ranking on specificity**; operational naming (score3-axis, not "strength").

---

## 0. INTENT

> **At loci where two candidate acceptors compete, does the donor's coupling depend on candidate SLOT
> (near/far), WINDOW (the local 3′SS unit), and their interaction — and do the candidates genuinely
> COMPETE (interact beyond saturation and beyond separate generic-AG damage)?**

Behaviour-level; SLOT/WINDOW are factor names, not "address/payload". Competition ⇒ κ beyond the
saturation baseline. All ranking claims are net of a matched sham (generic AG damage).

---

## 1. Observable, cohort, WINDOW definition

- **Observable (SEALED, PRIMARY):** `f = logit(G) − max{A,C,T}` at the masked donor, with the `{A,C,T}`
  **reference base PINNED to the native-cell argmax across all four cells** (`f11/f10/f01/f00`) to stop
  reference-switching non-additivity. Secondary robustness: linear `f_lin = G − mean{A,C,T}` and a
  **link/logit-probability scale** (for the κ sensitivity check).
- **WINDOW = the FULL 3′SS unit** (branchpoint + polypyrimidine tract + AG, `≈[·−40 .. ·+2]`), NOT a local
  AG. A construction gate requires each built acceptor to be **viable in BOTH slots** (`score3` ≥ a
  pilot-frozen floor).
- **Cohort — NATURAL alt-3′SS loci PRIMARY (`⟐`):** donor contexts with **two ANNOTATED (both-used)
  acceptors** (disambiguated from merely "plausible"/cryptic). This is the natural home of competition;
  **scope: "competitive ranking at alternative-acceptor loci,"** NOT constitutive donors. **Representa-
  tiveness report + gate:** compare the cohort's gene-class / donor & acceptor `score3` distributions to
  the A0–A12 donor population; report divergence and scope accordingly. **Numeric adequacy thresholds
  (min TRAIN N, max κ-MDE) are pre-set BEFORE the census returns (§8).**
- **Synthetic scaffold — SECONDARY robustness (conditional):** THREE anchors (native / single-candidate /
  two-candidate); a **GLOBAL OOD gate** (whole-donor-context junction-spanning NLL vs a natural reference)
  because inserting a second full acceptor is a novel global architecture the per-window gate cannot catch.
  Scoped to "ranking in a constructed assay."
- **Slots:** near/far, **near floor ≥ ~80 nt** (above minimum functional intron length; from the native
  two-acceptor distance distribution), **positions jittered / phase-balanced mod-3** (C8 period-3 clock),
  reported.

---

## 2. Factorial + SHAM (reinstated) `⟐ RED-TEAM`

Per locus, place windows into `{near,far}` in both reciprocal arrangements; cells `f11/f10/f01/f00`;
corruption = `AG→CC`. **Reliance** `ρ(X) = f11 − f(X corrupt, other intact)`.
- **SHAM arm (slot- AND strength-resolved):** at each slot, a **non-acceptor** AG-corruptible window
  `Sh` (matched local composition, NOT a plausible acceptor) gives `ρ_sham(slot)` = the **slot-generic AG
  sensitivity gradient**. Ranking estimands are DiDs net of it.
- **Two DECOY STRATA (not merged):** `D_m` = decoy matched to `P` on the **MODEL's own single-candidate
  AG-damage f-drop** (per R22, NOT MaxEnt score3); `D_c` = **composition-balanced** strength-contrast decoy
  (GC + pyrimidine fraction matched; residual score3 gap netted via A9b-style cross-fitted DML).

---

## 3. Estimands (frozen)

**PRIMARY — behavioural, DiD net of sham, sup-t simultaneous inference:**
- **SLOT advantage** `= mean_W{[ρ(W@near) − ρ(W@far)] − [ρ_sham(near) − ρ_sham(far)]}` (net the slot-generic AG gradient).
- **WINDOW advantage — per stratum, NOT merged:** *annotation/compatibility* `= mean_slot[ρ(P) − ρ(D_m)]`
  (AG-damage-matched ⇒ residual is compatibility); *score3-axis* `= mean_slot[ρ(D_c strong) − ρ(D_c weak)]`
  (composition-adjusted).
- **SLOT × WINDOW interaction** (per stratum).

**COMPETITION (CENTRAL; κ vs SATURATION baseline; change-blocker):**
`κ = f11 − f10 − f01 + f00`. Because `f` saturates, define **`κ_comp = κ_observed − κ_null(saturation)`**,
where `κ_null` is predicted from a **single-candidate graded dose-response** (estimating the readout
curvature `g`). **Gate:** a **far-apart NON-COMPETING two-candidate control** must give `κ ≈ κ_null`
(no excess). Competition is licensed only by `κ_comp` beyond baseline (and a link-scale sensitivity check).

**SECONDARY (corruption-caveated):** absolute support `S(X)`, reliance `ρ(X)`; **partner-win fraction** —
in the swapped arrangement, computed **only within the AG-damage-MATCHED stratum**, net the sham, with a
**within-pair label-permutation null** (permute which window is the annotated partner, holding slot &
strength fixed; NOT a 0.5 binomial), scaffold-clustered CI, own meaningful-fraction threshold.

---

## 4. Inference (frozen) `⟐ RED-TEAM`

- **Unit / dependence:** homology cluster (≈ scaffold gene). Decoys from other genes ⇒ DYADIC: **two-way
  clustering on {scaffold gene, decoy-source gene}**, OR unique per-scaffold decoy assignment — the pilot
  reports cluster counts PER MARGIN and picks per a frozen rule (wild-cluster bootstrap + small-cluster
  correction if a margin < ~30 clusters).
- **SUP-t simultaneous bands** (NOT "simultaneous BCa"): studentize each confirmatory component by its
  two-way wild-cluster bootstrap SE; resample WHOLE scaffold clusters carrying ALL cells/slots/strata;
  critical value `c` s.t. `P(max_j |t_j*| ≤ c)=0.95`. The confirmatory family = **the EXACT enumerated set**
  {SLOT, WINDOW_annotation, WINDOW_score3, SLOT×WINDOW (per stratum), κ_comp}; partner-win fraction carries
  its OWN reserved α (different scale).
- **Power BINDS ON κ_comp** (highest-variance, strictest-penalty central estimand): pre-register N for 80%
  simultaneous power at the per-order SESOI for κ_comp; the pilot reports κ_comp's detection-MDE AND
  equivalence-SE.
- **Per-order SESOIs** (pilot-derived): separate thresholds for first-order reliance contrasts
  (SLOT/WINDOW) and second-order double-differences (κ_comp, SLOT×WINDOW).

---

## 5. Gates / memorials (HARD preflight; STOP on failure) `⟐ RED-TEAM`

- **`determinism`** (`sorted(set())`+pinned hashseed; identical across `PYTHONHASHSEED∈{0,1,2,17}`).
- **`native_anchor(s)`** (native reproduces coupling; synthetic reports all 3 anchors + the single→two cost).
- **`corruption_validity`** (`AG→CC` collapses `score3`; `f00<f11`).
- **`window_viability`** (each built acceptor `score3 ≥` floor in BOTH slots).
- **`distal_ood` (incremental within-state, IC1 A1)** + **`global_ood`** (whole-context NLL vs natural class;
  binds the synthetic arm).
- **`noncompeting_kappa_control`**: the far-apart control gives `κ ≈ κ_null(saturation)` — else the
  saturation model is wrong and κ_comp is uninterpretable.
- **`sensitivity_onpath` (partner-SPECIFIC, C1 ~37×):** native-partner-corrupt **minus a matched non-partner
  sham corrupt** produces a detectable partner-specific drop — proves the pipeline detects candidate
  SUPPORT, not generic AG damage (native-vs-corrupt alone is explicitly NOT the control).
- **`sham_exchangeability`** (the sham matches dose/OOD/composition of the candidate perturbations).
- **`native_coupling_present`**, **`identity_noop`**.

`--smoke` = mechanics + determinism + native_anchor + corruption_validity + partner-specific-on-path on tiny-n.

---

## 6. Decision rule (frozen; specificity- and κ-gated) `⟐ RED-TEAM`

Gates PASS. **Competition is licensed ONLY if `κ_comp` (net saturation) is material AND the non-competing
control is clean.** Ranking language is licensed ONLY for components that (i) survive the sham DiD
(specificity) AND (ii) occur with `κ_comp` (or SLOT×WINDOW) material (genuine competition):
- **material** (sup-t CI excludes 0 AND `|·| ≥` its per-order SESOI) / **detectable-but-immaterial**
  (excludes 0, `|·| <` SESOI — NOT called "ranking" unless specific+competitive) / **bounded-null**
  (sup-t CI within ±SESOI) / **inconclusive**.
- **Interpretation (behaviour-level):** SLOT material+specific+competitive → coupling ranks competitors by
  **slot (position)**; WINDOW_score3 → ranks along the **MaxEnt-score3 axis** (NOT "strength" — R22);
  WINDOW_annotation → **compatibility beyond matched AG-damage**; κ_comp → **genuine competition**;
  partner-win (own scale) → partner wins after slot+window exchanged.
- **No competition** (κ_comp bounded-null / control fails): report SLOT/WINDOW as **marginal single-candidate
  position/sequence sensitivities of candidate support**, NOT ranking of competitors.
- **No material ranking** requires ALL sup-t CIs within their per-order equivalence bounds.
- **Established ranking** → contingent L3H1/payload mediation follow-up (§7). **All-bounded / inadequate** →
  **transfer to a second GFM** (no further SpliceBERT subdivision).

---

## 7. Scope, deferred, transfer

- **Behaviour-first.** Address/payload naming ONLY in a SEPARATE contingent mediation prereg (A8
  reconstruct-inject: deliver a candidate's payload through the other's address; QK/`U_info`).
- **Intended LAST SpliceBERT experiment** → then **transfer the method to a SECOND GFM** (method, not
  catalogue — CLAUDE.md mission).
- **Biological bridge (if ranking established):** model ranking vs MPSA competitive-splicing outcomes for
  natural near/far or score3-axis acceptor pairs.

---

## 8. Pre-freeze pilots (disjoint TRAIN; no val read; RESOURCE-CONSIDERATE — shared laptop) `⟐`

Pre-set BEFORE the census: **min TRAIN natural-cohort N and max κ_comp-MDE** (numeric, TRAIN-only) that
select natural-vs-synthetic; pre-commit to the selected cohort.
1. **NATURAL alt-3′SS census FIRST** (+ representativeness vs A0–A12); decide cohort by the pre-set rule.
2. **Construction** (natural slots or synthetic 3-anchor): native_anchor, corruption_validity,
   window_viability in both slots, single→two cost; slot geometry (≥80nt, mod-3-balanced).
3. **Sham + decoy strata:** slot/strength-resolved sham; `D_m` matched on **model AG-damage**; `D_c`
   composition-balanced; report cluster counts per margin.
4. **κ saturation baseline** (single-candidate dose-response ⇒ `g`, `κ_null`) + **non-competing control**;
   corrected **distal_ood + global_ood** demonstration (natural/identity pass; planted-OOD fails) + `δ_OOD`.
5. **Partner-specific on-path (C1 ~37×)** sensitivity; **per-order SESOIs**; **κ_comp-binding power/N** +
   sup-t multiplicity; partner-win permutation-null power.
6. **Determinism test** across hash seeds.
- **Resource note:** shared laptop — modest batches, NO parallel heavy MPS jobs, check load before launch.

**Provenance:** checkpoint sha256 `2ad91428…0149d`; MaxEnt pinned (STRATIFIER only, not ground truth);
shuffler/decoy-pool/slot geometry pinned; thresholds + seeds preregistered after pilots. No val read pre-freeze.
