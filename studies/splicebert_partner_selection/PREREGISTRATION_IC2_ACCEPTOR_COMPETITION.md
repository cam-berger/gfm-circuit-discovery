# Pre-registration — IC2: does SpliceBERT rank COMPETING candidate acceptors? (behaviour-level)

**Status: DRAFT v2 (post-Cam-revision) for red-team → pilot-calibrate → FREEZE. DO NOT RUN until frozen.**
The compact acceptor-competition study (supersedes the preserved, not-pursued IC2a decomposition); intended
FINAL SpliceBERT experiment before transfer to a second GFM. **Behaviour-level throughout** — mechanistic
names (address, payload) are DEFERRED to the contingent L3H1/payload mediation follow-up. Carries all IC1
lessons: sealed readout, deterministic constructs (`sorted(set())` + pinned `PYTHONHASHSEED` + determinism
test), CI-based classification, corrected incremental-within-state OOD gate, outcome-blind discipline.

**v2 corrections (Cam):** (1) restore the SEALED max-margin readout; (2) behaviour-level language (SLOT /
WINDOW / SLOT×WINDOW, not address/payload); (3) separate annotation from strength (two decoy strata);
(4) candidate corruption is NOT partner-specific by construction — the four-cell interaction κ is the
central competition test; (5) prefer a NATURAL dual-acceptor cohort, synthetic only if needed (3 anchors);
(6) two-way clustering for decoy-source dependence; (7) simultaneous inference, not unadjusted-after-omnibus;
(8) C1-based (not MaxEnt) on-path positive control.

---

## 0. INTENT (north star)

> **When two candidate acceptors compete in one donor context, does the donor's coupling depend on the
> candidate SLOT (near/far position), the candidate WINDOW (local acceptor sequence), and their
> INTERACTION — and do the two candidates COMPETE (interact) beyond their separate effects?**

Behaviour-level. SLOT/WINDOW/SLOT×WINDOW are descriptive factor names; whether SLOT ≈ "positional address"
and WINDOW ≈ "acceptor-local payload" is the SEPARATE, contingent mediation study (§7). No mechanistic
claim here.

---

## 1. Observable, cohort (NATURAL preferred), domain

- **Observable (SEALED, PRIMARY):** `f = logit(G) − max{logit A, logit C, logit T}` at the masked donor —
  the established donor margin (the sealed +0.87 behaviour metric). Secondary robustness readout: the
  linear `f_lin = logit(G) − mean{A,C,T}`; report but do not base the verdict on it.
- **Cohort — NATURAL dual-acceptor PREFERRED (`⟐ RED-TEAM`, pilot-decided first).** Census donor contexts
  with **two plausible/annotated candidate acceptors** at a near and a far position (both `GT…AG`,
  in-window, `L∈[60,150]`-scale). **If the disjoint-TRAIN yield and MDE are adequate, the natural
  two-acceptor cohort is PRIMARY** (no synthetic construction). Corruption is `AG→CC` per candidate.
- **Synthetic scaffold — CONDITIONAL** (only if natural yield is inadequate). Requires **THREE anchors:**
  (a) the true unmodified NATIVE sequence, (b) a reconstructed SINGLE-candidate sequence, (c) the
  TWO-candidate synthetic sequence. The single→two-candidate step is where the second candidate enters, so
  a "native anchor" that already contains a second candidate is NOT native. **Synthetic results are scoped
  to "ranking in a CONSTRUCTED competition assay," NOT biological splice-site selection.**

---

## 2. Factorial — 2 slot × 2 window × candidate-corruption, reciprocal swap `⟐ RED-TEAM`

Per donor context, place windows into slots `{near, far}` in BOTH reciprocal arrangements, read `f` under
all candidate intact/corrupt states:
- Cells: `f11` (both intact), `f10` (near intact/far corrupt), `f01` (near corrupt/far intact), `f00`
  (both corrupt). **Reciprocal swap:** the same two windows exchanged across near/far.
- **Reliance** `ρ_near = f11 − f01`, `ρ_far = f11 − f10` (drop from removing one candidate while the other
  is intact). **Absolute reliance carries generic AG-channel damage (IC1) — it is SECONDARY (§3).**
- **Two DECOY STRATA (do NOT merge; change 3):** for each donor with annotated partner `P`:
  - **Strength-MATCHED real acceptor `D_m`** (`score3` matched to `P`) → the WINDOW contrast `P` vs `D_m`
    isolates **annotation / compatibility** advantage (strength and AG-damage matched).
  - **Strength-CONTRAST real acceptor `D_c`** (`score3` far from `P`) → the WINDOW contrast strong-vs-weak
    isolates **local-window-strength** advantage.
- Deterministic construction: frozen decoy pool + slot geometry + RNG (`sorted(set())`, pinned hashseed).

---

## 3. Estimands (frozen)

**PRIMARY — behavioural factorial with SIMULTANEOUS inference (change 7):**
- **SLOT advantage** = reliance contrast for the SAME window across slots: `mean_window[ρ(window@near) −
  ρ(window@far)]`. (The window-specific generic AG-damage CANCELS in this within-window across-slot
  contrast ⇒ robust to the IC1 AG-channel confound.)
- **WINDOW advantage — reported per stratum, NOT merged:**
  - *annotation/compatibility* = `mean_slot[ρ(P) − ρ(D_m)]` (matched-decoy arm);
  - *local-window-strength* = `mean_slot[ρ(strong) − ρ(weak)]` (contrast-decoy arm).
- **SLOT × WINDOW interaction** = the non-additive slot×window term (per stratum).

**COMPETITION evidence (CENTRAL; change 4):** the four-cell interaction
`κ = f11 − f10 − f01 + f00` — whether the two candidates interact BEYOND their separate corruption effects
(a double-difference robust to per-candidate generic damage). `κ ≠ 0` ⇒ genuine competition/non-additivity.

**SECONDARY (with explicit corruption-control caveats):** absolute candidate support
`S(X) = f(only X intact) − f00` and reliance `ρ(X)`; and the **partner-win fraction** — in the swapped
arrangement (`P@far, D@near`), the fraction of donors relying more on `P` than the near decoy — on its OWN
scale, with its OWN null, CI, and meaningful-effect threshold (change 7), NOT the nats-SESOI scale.

---

## 4. Inference (frozen; change 6, 7)

- **Unit:** homology cluster (≥80% id ≈ scaffold gene). **Decoy-source dependence:** decoys drawn from
  other genes make observations DYADIC — either assign each decoy UNIQUELY per scaffold, or use **two-way
  clustering on {scaffold gene, decoy-source gene}** (bootstrap). Determinism test required (IC1 A2).
- **Simultaneous inference:** report **simultaneous (max-statistic) BCa CIs** across the primary factorial
  effects {SLOT, WINDOW_annotation, WINDOW_strength, SLOT×WINDOW} and κ — a global omnibus rejection does
  NOT license unadjusted per-component interpretation; use the simultaneous CIs (or multiplicity-adjusted
  component tests) for EACH component. `SESOI = 0.05` nats (sealed-margin scale).
- **Partner-win fraction:** separate binomial/permutation null + CI + a pre-set meaningful fraction.
- **Sampling:** census of the eligible deterministic cohort; else multiple seeded draws + stability.

---

## 5. Gates / memorials (HARD preflight; STOP on failure) `⟐ RED-TEAM`

- **`determinism`** (IC1 A2): identical constructs/cohort/results across `PYTHONHASHSEED ∈ {0,1,2,17}`.
- **`native_anchor(s)`:** the true-native construct reproduces the sealed coupling of expected sign/scale;
  for the synthetic path, ALL THREE anchors (native / single-candidate / two-candidate) are reported and
  the single→two-candidate delta is the construction cost (scoped, not native).
- **`corruption_validity`:** `AG→CC` collapses each candidate's `score3`; `f00 < f11` on the eligible set.
- **`distal_ood` (corrected, IC1 A1):** the (natural or synthetic) two-candidate construct's window NLL is
  noninferior to a natural-to-natural null (incremental within-state, per-state, gene-clustered; `δ_OOD`
  pilot-frozen) — a construct that is OOD cannot be interpreted; synthetic constructs are the higher risk.
- **`sensitivity_onpath` (C1-based, change 8):** an intact NATIVE partner vs its EXACT corrupt version
  produces a detectable candidate-support drop (`|Δ| ≥ SESOI`) through the model — a KNOWN on-path contrast
  from C1, proving the pipeline detects candidate support. (MaxEnt strong-vs-weak is the HYPOTHESIS, not a
  sensitivity control; MaxEnt is an operational STRATIFIER only, not ground truth.)
- **`native_coupling_present`**, **`identity_noop`**.

`--smoke` = mechanics + determinism + native_anchor + corruption_validity + C1-on-path on tiny-n.

---

## 6. Decision rule (frozen; simultaneous CIs; behaviour-level)

Gates PASS → classify EACH primary component by its **simultaneous** CI (SESOI=0.05):
- **material** (simultaneous CI excludes 0 AND `|·|≥0.05`) / **small-but-detectable ranking evidence**
  (excludes 0, `|·|<0.05`) / **bounded-null** (simultaneous CI ⊆ ±0.05, includes 0) / **inconclusive**.
- **Interpretation (behaviour-level):**
  - SLOT material → coupling ranks candidates by **slot (position)**.
  - WINDOW_strength material → ranks by **local-window strength**; WINDOW_annotation material → ranks by
    **annotated-partner/compatibility** beyond matched strength.
  - SLOT×WINDOW material → **non-additive** slot/window ranking.
  - `κ` material → genuine **competition** (candidates interact).
  - partner-win fraction (own scale) → whether the annotated partner wins after slot+window are exchanged.
- **A small-but-detectable component is still ranking evidence** — just not MATERIAL ranking.
- **"No material ranking"** requires ALL simultaneous CIs to fall within their equivalence bounds (±0.05).
- **Omnibus failure is NOT "no ranking"** — report the simultaneous CIs; only all-bounded licenses a null.
- **Established ranking** (any primary component material, gates clean) → promote to the contingent
  L3H1/payload mediation follow-up (§7). **All-bounded null / inadequate** → do not subdivide further;
  **transfer to a second GFM.**

---

## 7. Scope, deferred, transfer

- **Behaviour-first; mechanistic terminology deferred.** Only after §6 establishes a behavioural ranking:
  a SEPARATE contingent prereg tests whether SLOT ≈ positional address and WINDOW ≈ acceptor-local payload,
  via A8 reconstruct-inject (deliver a candidate's payload through the other's address; QK/`U_info` readouts)
  — that mediation study is where "address"/"payload" names are earned.
- **Intended LAST SpliceBERT experiment.** After IC2, **prioritize TRANSFER of the method to a SECOND GFM**
  over further SpliceBERT subdivision (method, not catalogue — CLAUDE.md mission).
- **Biological bridge (if a ranking is established):** whether model ranking predicts MPSA competitive-
  splicing outcomes for natural near/far or strong/weak acceptor pairs.

---

## 8. Pre-freeze pilots required (disjoint TRAIN; no val read; RESOURCE-CONSIDERATE — laptop shared)

1. **NATURAL dual-acceptor census FIRST:** count donor contexts with two plausible/annotated acceptors;
   report yield + MDE. Decide natural-vs-synthetic cohort BEFORE any construct work.
2. If synthetic needed: scaffold + THREE-anchor construction; demonstrate `native_anchor`,
   `corruption_validity`, the single→two-candidate construction cost, adequate count.
3. **Slot geometry** (near/far) frozen from the native two-acceptor distance distribution; **decoy pool +
   matched/contrast `score3` strata** frozen (deterministic, unique-assignment or dyadic-clustered).
4. **Corrected OOD gate** demonstration (natural/identity pass; planted-OOD fails) + `δ_OOD`.
5. **C1-based on-path sensitivity** + **MDE / N** for SLOT, WINDOW(×2), SLOT×WINDOW, κ, and the
   simultaneous-inference multiplicity; partner-win-fraction power on its own scale.
6. **Determinism test** across hash seeds.
- **Resource note:** the laptop is shared with another agent — pilots use modest batch sizes, avoid parallel
  heavy GPU/MPS jobs, and check load before launching.

**Provenance:** checkpoint sha256 `2ad91428…0149d`; MaxEnt pinned (stratifier only); shuffler/decoy-pool +
slot geometry pinned; all thresholds + seeds preregistered after the pilots. No val data read before freeze.
