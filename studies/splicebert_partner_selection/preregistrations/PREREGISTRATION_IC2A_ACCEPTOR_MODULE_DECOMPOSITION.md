# Pre-registration — IC2a: which 3′ acceptor-recognition module changes PARTNER-SPECIFIC donor coupling?

**Status: DRAFT — NOT PURSUED (2026-07-17). Preserved as future work; DO NOT red-team, pilot, or freeze.**
Cam redirected the arc away from further proximal-region subdivision toward the compact **acceptor-
competition** study (`PREREGISTRATION_IC2_ACCEPTOR_COMPETITION.md`), and then to transfer to a second GFM
over further SpliceBERT subdivision. This draft is a faithful decomposition design retained in case the
competition study motivates returning to proximal-module dissection. — (original framing below) — Second
experiment of the intron-coupling arc, opened after **IC1** found distal interstitial arrangement is not a
meaningful partner-specific coupling driver (small +0.032, sub-SESOI, sham-reproduced). IC2a turns to the
region IC1 PROTECTED — the acceptor-proximal 3′ recognition module — and asks which of its
biologically-distinct sub-modules carries the effect. **Behaviour-first** (full-model coupling assay);
payload mediation is the separate contingent **IC2b**, and competing-acceptor semantics is the separate **IC5**.

Carries every IC1 lesson: deterministic shuffler (`sorted(set())` + pinned `PYTHONHASHSEED` + a determinism
test), CI-based classification, incremental-within-state OOD noninferiority gate, partner-specificity (sham
DiD) as the PRIMARY estimand (IC1's lesson), and strict outcome-blind discipline (design + pilot + audit +
freeze before any val read).

---

## 0. INTENT (north star)

> **Does disrupting the branchpoint ensemble, the polypyrimidine tract (PPT), or the AG-exclusion /
> internal-AG region change donor–acceptor coupling SPECIFICALLY through the paired acceptor?**

Behaviour-level. The unit of interest is the **partner-specific** coupling change (§3), not raw damage.
IC2a identifies WHICH proximal module matters; IC2b (contingent) tests whether that module is compressed
into the acceptor-local payload; IC5 (separate) tests conditional competitiveness.

---

## 1. Domain, observable, coupling, and PARTNER-SPECIFICITY

- **Domain (NOT IC1's L≳107):** canonical internal introns `L∈[60,150]` (the original validated domain) —
  IC2a targets the acceptor-proximal region present in essentially all eligible introns, so it does not
  inherit IC1's distal-window length restriction. A module whose construction needs a narrower population
  (e.g. a minimum BP-to-AG span) narrows only THAT module's cohort, stated explicitly.
- **Observable (frozen LINEAR):** `f = logit(G) − mean{logit A,C,T}` at the masked donor. **Coupling**
  `C(x) = f(x, A⁺) − f(x, A⁻)`, `A⁻` = partner AG→CC at `[a−2,a−1]` (the sealed +0.87 behaviour).
- **PARTNER-SPECIFIC effect (PRIMARY estimand, per module R):** for perturbation `T_R`,
  `S_R = [C(T_R(x)) − C(x)] − [C(T_R^sham(x)) − C(x))]`, i.e. the perturbation's coupling change **net of a
  matched non-partner sham** (§3). Cluster-mean, BCa CI, classified by CI (§6). `S_R ≠ 0` ⇒ the module
  changes coupling *through the paired acceptor*; `S_R ≈ 0` with a non-zero raw effect ⇒ nonspecific damage.

---

## 2. Three biologically-distinct modules (perturbations + controls) `⟐ RED-TEAM`

Each module has its own perturbation `T_R` and matched non-partner sham `T_R^sham`. All perturbations:
protected-zone-aware, dose-matched, deterministic (sorted-set shuffler, pinned hashseed).

| Module | Perturbation `T_R` (disrupt) | Held FIXED (controls) |
|---|---|---|
| **BRANCHPOINT ensemble** | disrupt credible branch-A / `yUnAy` capacity at ALL predicted BPs (ensemble, not one) — e.g. branch-A→C and context de-optimization | PPT pyrimidine content, AG landscape, `L`, mono+di composition, 3′SS MaxEnt score3 |
| **PPT** | (a) COMPOSITION arm: reduce pyrimidine fraction/run structure; (b) ORDERING arm: shuffle PPT pyrimidine positions preserving count (composition-controlled) | branchpoint candidates, all AGs, edit dose, position |
| **AG-exclusion / internal-AG** | introduce OR remove matched internal `AG`s in the BP→terminal-AG zone | BPS, PPT, edit dose, positional zone, composition |

**Branchpoint ensemble handling `⟐ RED-TEAM`.** Do NOT treat one predicted BP as truth — human introns
carry multiple, sometimes tissue-specific branchpoints (Pineda & Bradley 2018). Use a **BP-predictor
ensemble** (LaBranchoR + BPP) over `[a−ℓ_bp .. a−δ]`; perturb ALL credible BPs (score above a frozen
threshold), and **stratify by predicted BP redundancy** (single vs multiple credible BPs) — the effect may
live only where BP capacity is non-redundant. The BP→AG span is mechanistically coupled to acceptor
selection (AG-exclusion-zone + branchpoint-scanning), so the branchpoint and AG-exclusion modules are
reported together where they overlap.

---

## 3. Per-module SHAM + the honest fallback `⟐ RED-TEAM`

For each module, `T_R^sham` matches **edit dose, positional zone, local model disruption (distal-window
NLL shift), and motif-score change** WITHOUT perturbing the partnered-acceptor module — a non-partner edit
of equal "damage geometry" (the AG-channel lesson from IC1: the sham should engage a comparable readout
where possible). Sham exchangeability is **demonstrated on the disjoint pilot** (matched dose/OOD/motif
distributions), analogous to IC1's `ic1_pilot4_ood`.

**Fallback (honest, per Cam):** if **no exchangeable sham exists** for a module (e.g. the branchpoint has
no non-partner analogue of comparable salience), keep the **raw coupling effect** `C(T_R)−C` PRIMARY for
that module and **narrow the interpretation** ("changes coupling; partner-specificity not isolable"),
rather than forcing an invalid subtraction. State per module, pre-freeze, whether a valid sham exists.

---

## 4. Inference — omnibus FWER, then gatekept module contrasts `⟐ RED-TEAM`

- **Unit:** homology cluster (≥80% id ⇒ ≈ gene). Deterministic cohort (sorted-set shuffler + pinned
  `PYTHONHASHSEED`; determinism test required, IC1 A2).
- **ONE global omnibus test first:** a single **FWER-controlled** omnibus across the three modules'
  `S_R` (bootstrap max-statistic over the module statistics; two-sided α=0.05). **Only if the omnibus
  rejects** are the planned per-module `S_R` contrasts promoted/interpreted (hierarchical gatekeeping) —
  this avoids three independent fishing expeditions. Report all three regardless, labelled exploratory
  unless promoted.
- **CIs:** cluster (wild/multiplier) bootstrap, **BCa**. **SESOI = 0.05 nats** (inherited).
- **Sampling:** census of the eligible deterministic cohort; if sub-sampled, multiple independently-seeded
  draws + stability.

---

## 5. Gates / memorials (HARD preflight; STOP on failure) `⟐ RED-TEAM`

- **`determinism`**: `test_ic2a_determinism.py` — identical perturbations, cohort membership, and results
  across `PYTHONHASHSEED ∈ {0,1,2,17}` (IC1 A2 bug class).
- **`protected_and_control_invariants`**: per module, the FIXED quantities (§2) are preserved (bit-identical
  or within a frozen tolerance): composition, `L`, position, the held MaxEnt scores, held motif landscapes.
- **`dose`**: per-perturbation edit dose ≥ floor; near-identity excluded.
- **`distal_ood` (corrected, IC1 A1 form)**: incremental within-state `ΔNLL_s` noninferiority to a
  natural-to-natural null; `δ_OOD` frozen from the disjoint pilot; per-state, gene-clustered.
- **`sham_exchangeability`**: each module's sham matches the perturbation's dose/OOD/motif distributions
  (pilot-demonstrated); else the module drops to the raw-effect fallback (§3).
- **`sensitivity_onpath`**: a planted, biologically-strong version of each module perturbation (e.g. a
  full BP knockout, a full PPT ablation, a strong planted AG) registers `|S_R|` or `|ΔC|` ≥ SESOI through
  the model — the null is not licensable for a module whose strong version the pipeline cannot detect.
- **`native_coupling_present`**, **`identity_noop`** (plumbing).

`--smoke` = mechanics + determinism + dose + control-invariants + on-path on tiny-n.

---

## 6. Decision rule (frozen; per module, gatekept by the omnibus)

Omnibus rejects (α=0.05 FWER) → interpret per-module `S_R` (else all modules exploratory / null-of-omnibus):
- **Material partner-specific effect (module R):** `S_R` CI excludes 0 AND `|Ŝ_R| ≥ 0.05` → **module R
  changes coupling through the paired acceptor** → promote to **IC2b** (payload-mediation) for that module.
- **Small-but-detectable:** `S_R` CI excludes 0 AND `|Ŝ_R| < 0.05`.
- **Bounded null:** `S_R` CI ⊆ `[−0.05,+0.05]` (and includes 0) → module R not a meaningful partner-specific
  driver.
- **Inconclusive:** CI includes 0 and extends beyond ±0.05.
- **Raw-effect fallback module** (no valid sham): classify `C(T_R)−C` by the same CI scheme, interpretation
  narrowed to "coupling change, partner-specificity not isolable."
- **OOD / determinism / on-path failure:** no interpretation for that module; redesign.

Report the BP-redundancy stratification (single- vs multi-BP) for the branchpoint module.

---

## 7. Scope, deferred, biological bridge

- **Behaviour-first**; **IC2b (payload mediation)** is a SEPARATE, contingent prereg run ONLY for a module
  with a supported partner-specific effect: perturb the proximal sequence → confirm altered coupling →
  **reconstruct-inject the clean 4-token acceptor-window value/payload while leaving the perturbed
  sequence-derived address intact** (A8 machinery) → does it rescue coupling and donor-side `U_info`?
  (payload-compression vs address-change vs distributed-route vs general-damage, per the IC2b table).
- **IC5 (competing acceptors)** stays separate — do not combine regional semantics + competition + mediation.
- **Biological bridge (after a positive IC2a module):** test whether model `S_R` sensitivity predicts
  MPSA-measured splicing effects for natural variants in that module (near-exon intronic + branch-site
  MPSA), a stronger validation than MaxEnt correlation.
- **Sequence:** IC2a (which module) → IC2b (payload compression of the positive module) → IC5 (competition).

---

## 8. Pre-freeze pilots required (disjoint TRAIN; no val read)

1. **BP-predictor ensemble** calibration: run LaBranchoR + BPP, set the credible-BP score threshold, report
   the single-vs-multi-BP redundancy strata and eligible counts.
2. **Perturbation + control-invariant** construction per module (deterministic); demonstrate the FIXED
   quantities are preserved and dose is adequate (no collapse), like IC1 pilot3.
3. **Per-module sham exchangeability** (dose/OOD/motif matching) + the corrected OOD gate demonstration
   (natural/identity/perturbation pass; planted-OOD fails), like IC1 pilot4.
4. **On-path sensitivity** (strong-version detection) + **MDE / N** per module and for the omnibus.
5. **Determinism test** across hash seeds.

**Provenance:** checkpoint sha256 `2ad91428…0149d`; MaxEnt pinned; BP predictors + shuffler pinned; all
thresholds + seeds preregistered after the pilots. No value read from the confirmatory val set before freeze.
