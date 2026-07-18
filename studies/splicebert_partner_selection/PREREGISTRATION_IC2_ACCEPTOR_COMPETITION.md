# Pre-registration — IC2 v5.1: does donor support depend on candidate SLOT or WINDOW? (behaviour-level)

**Status: DRAFT v5.1 (post-re-audit) for freeze-audit → freeze → confirmatory. DO NOT RUN confirmatory until frozen.**
Compact two-candidate assay; **final pre-specified behavioural SpliceBERT experiment** — after v5.1, **transfer
to a second GFM regardless of outcome** (a material result additionally licenses a SEPARATE tracing prereg;
branches non-exclusive). Behaviour-level; address/payload naming deferred to that mediation study.

**Lineage.** v4 → 4-lens red-team (all blocker) → v5 narrowing (retired the floored-sham `β_slot,specific`
and the between-gene spacing reference arm; κ demoted) → 2-lens re-audit (both **major-fixes, no blocker**) →
**v5.1** applies those fixes: a **planted-identity positive control** for the binding equivalence; **mandatory
per-window slot terms** gating the occupant-generic claim; **N pinned from the CLUSTERED SE** with a feasibility
contingency; **total** decision table; **"reliance-matched"** (not "matched-strong", R22); per-slot OOD reported.

**Pilot basis** (`ic2_feasibility.json`, n=50 TRAIN, reciprocal swap): SLOT advantage **+0.349** (native vs
displaced, window-averaged; both the native occupant P *and* a foreign window get it), WINDOW-identity
advantage **−0.0002** (reliance-matched P vs D, slot-averaged), argmax stable **90%**, generic-AG floor
gradient **+0.034**, native coupling **+0.52**.

---

## 0. INTENT + PRIMARY CLAIM (behaviour-level, narrowed)

> **In a constructed two-candidate assay, does the donor's masked G-margin depend on which scaffold SLOT
> holds a competent acceptor, and on WHICH reliance-matched acceptor WINDOW occupies a fixed slot?**

**Licensed claim IF the confirmatory pattern confirms (frozen wording):** *"In a constructed two-candidate
assay, donor support depends strongly on which scaffold slot contains a competent acceptor — for a foreign
window as well as the native one — while exchanging two reliance-matched acceptor windows between those slots
has no material effect."* (The "foreign window as well" clause is licensed ONLY if the `D_m` per-window slot
term is itself material — §4/§7; else the claim narrows to an occupant-specific displacement penalty.)

**Explicit, frozen limitation (reported with the claim):** *"The experiment does NOT distinguish a general
donor–acceptor spacing rule from native-boundary privilege, local slot context, or the penalty incurred by
displacing an acceptor from its natural position."*

**NOT claimed:** "positional address dominates payload"; a spacing rule; biological acceptor selection;
splice-strength payload semantics (R22). **The model emits a donor SCORE `f`, not a splice-choice
probability.** Verb is **"depends on"**, never "allocates". "Reliance-matched" = matched on the model's own
f-drop (§3), NOT MaxEnt "strength" (R22 withdrew that naming).

---

## 1. Observable + READOUT AUDIT (frozen)

- **PRIMARY observable: SEALED max-margin** `f = logit(G) − max{A,C,T}` at the masked donor (piecewise-linear;
  nonlinear only when the non-G argmax switches).
- **Support** `S(X,p) := ρ(X@p) = f(both intact) − f(X@p corrupted, partner intact)` — two-candidate reliance
  of window `X` at slot `p` (competition-inclusive; sign-agnostic — `S<0`, a rival depressing `f`, is retained).
- **`argmax_stability` GATE (≥90%, pilot-frozen).** Full-set sealed-margin is the ONLY confirmatory readout;
  the stable-subset recomputation is a **diagnostic** (argmax-switching correlates with response magnitude →
  the subset is attenuation-biased; a full-vs-subset gap does NOT impeach the full-set result).
- **Fixed-class robustness decomposition:** report `G−A, G−C, G−T` sign-concordance with the sealed DiD (pilot
  ~84%). Decomposition only; probabilities (real saturation) are not used.

---

## 2. Cohort & construction (SYNTHETIC constructed-assay PRIMARY)

- **SYNTHETIC = PRIMARY, scoped** to "a constructed two-candidate assay" (natural yield inadequate: 142 loci,
  0 at ≥80 nt).
- **Scaffold pool:** TRAIN internal introns, gene-deduplicated, `L≥110` (pilot pool 1,985); held-out gene split.
- **Two slots (CONFIRMATORY):** native `p_nat` and ONE frozen displaced `p_dis`, at a donor-distance **a
  multiple of 3 from `p_nat`** (holds C8 phase) that **clears all protected zones** (upstream edge `> d0+8`;
  frozen default `p_dis = p_nat − 60`). Retained-pool count at `p_dis` reported.
- **WINDOW = full 3′SS** (BP+PPT+AG, `[·−37..·+2]`, 40 nt), 40→40 in-place; passes `window_viability` in every
  slot and adds no new cross-junction cryptic 3′SS.
- **Slot-SWEEP = EXPLORATORY only** (extra distances, phase-balanced): descriptive support-vs-distance curve,
  **not** causal identification, cannot license a spacing-rule claim.
- **Natural alt-3′SS (142 loci) = PURELY DESCRIPTIVE CONTEXT** (15 nt median spacing → windows overlap →
  geometrically incompatible); **cannot corroborate/override any synthetic conclusion.**

---

## 3. Design — reciprocal swap + generic-AG floor-check (+ secondary strength) `⟐`

Per scaffold, two windows at `{p_nat, p_dis}` in **both arrangements**; cells = both-intact + each-single-
corrupt (`AG→CC`). **Config N:** `P@p_nat, D_m@p_dis`. **Config W:** `D_m@p_nat, P@p_dis`.
- **`D_m` = RELIANCE-MATCHED decoy:** matched to `P` on the **model's own single-candidate AG-damage f-drop**
  (R22-safe; NOT score3-as-semantics). In-context validity + a planted-identity positive control gate it (§6).
- **Generic-AG arm (`Sh`) = COMPETENCE/FLOOR CHECK ONLY:** composition-matched non-acceptor AG window;
  verifies support tracks competence. **`ρ_sham` REPORTED, never subtracted, never thresholded-small.**
- **3b. STRENGTH axis = REPORTED SECONDARY** (NOT confirmatory, NOT mediation-gating): `ρ(strong)` vs
  `ρ(D_w weak-but-competent)` + slot×strength, **composition-adjusted CIs** (score3 ≈ PPT-pyrimidine =
  A9b/R22). Cannot on its own license mediation or a strength/payload claim.

---

## 4. Estimands (frozen)

**CONFIRMATORY family (EXACTLY m = 3; studentized sup-t simultaneous CIs — §5):**
1. **SLOT effect** `β_slot|competent = mean_{X∈{P,D_m}} [ S(X,p_nat) − S(X,p_dis) ]`. *(Pilot: +0.349.)*
   **MANDATORY per-window reporting (not optional):** the two terms separately —
   `τ_P = S(P,p_nat)−S(P,p_dis)` (native occupant; carries occupant-specific native-position memory) and
   `τ_D = S(D_m,p_nat)−S(D_m,p_dis)` (**foreign** occupant; occupant-generic slot/boundary/context) — each with
   its own sup-t CI. *(Pilot: both ≈ 0.35.)*
2. **WINDOW-identity effect** `β_window|slot = mean_{p} [ S(P,p) − S(D_m,p) ]`, **EQUIVALENCE-tested within
   ±0.05** (studentized CI ⊆ [−0.05,+0.05]). *(Pilot: −0.0002.)* Interpretable as identity-INVARIANCE only if
   the **planted-identity positive control fires** (§6); otherwise reported as "no identity effect DETECTABLE
   at matched reliance."
3. **SLOT × WINDOW interaction** `= ½(τ_P − τ_D)`.

**Equivalence bound (component 2):** ±0.05 nats = **12.5% of native P-reliance (0.402)** and **~14% of the SLOT
effect (0.349)** — a substantively negligible support change.

**SECONDARY (reported; NOT in the family, NOT mediation-gating):** strength `ρ(strong)−ρ(D_w)` + slot×strength
(composition-adjusted); κ nonadditivity (characterization only); `ρ_sham` floor magnitude; **per-slot `δ_OOD`
point + CI reported alongside `β_slot`** (bounds the OOD/displacement share of the RAW slot effect); the
exploratory sweep curve; the descriptive natural-loci context.

---

## 5. Inference (frozen) `⟐`

- **Dependence — TWO-WAY CLUSTERING PRIMARY** on `{scaffold gene, decoy-source gene}` (unique decoys NOT
  feasible: each scaffold needs ≥1 reliance-matched + ≥1 weak decoy-source ≠ scaffold, viable in both slots).
- **SUP-t over m = 3:** studentize each component by its two-way wild-cluster bootstrap SE (resample WHOLE
  scaffold clusters); `c` s.t. `P(max_{j≤3}|t_j*| ≤ c)=0.95`. No omnibus-then-unadjusted. Component 2's
  equivalence uses the same studentized band (CI ⊆ ±0.05, a two-sided-CI-subset TOST).
- **Power / N — 80% SIMULTANEOUS power on the actual TWO-WAY-CLUSTERED simultaneous-band SE (NOT the IID MDE
  table).** The **binding component is the WINDOW equivalence** (component 1's +0.349 has ~5× margin). N is
  pinned from the clustered SE using a **decoy-source ICC estimated on TRAIN** (dedicated clustering census,
  §8.2), bound to the **upper CI of the design effect**. **State a max feasible N and a CONTINGENCY:** if 80%
  simultaneous equivalence power is unreachable at max N, the WINDOW result is reported as **"bounded-null at
  the achieved precision," NOT "equivalent/interchangeable"** (the equivalence claim is withheld, not faked).
- **Per-order SESOIs (pinned, substantive-anchored, NOT pilot-SE-derived):** first-order (SLOT, WINDOW, `τ_P`,
  `τ_D`) = **0.05 nats** (12.5% of native reliance 0.402); second-order (interaction) pinned separately as a
  stated fraction of native reliance and **verified achievable at N** (else interaction is reported-only).

---

## 6. Gates / memorials (HARD preflight; STOP on failure) `⟐`

- **`determinism`** — `sorted(set())` + pinned `PYTHONHASHSEED`; identical across `{0,1,2,17}`; test required.
- **`argmax_stability`** — ≥90% (§1).
- **`native_anchor`** — native construct reproduces coupling as a CI/band able to fail (pilot +0.52; the
  single→two cost is measured against a reference with a **SHAM at the displaced slot**, isolating competition).
- **`corruption_validity`** — AGGREGATE check (median single-candidate f-drop > 0 AND score3 collapse); NOT a
  per-construct `ρ>0` requirement (competitive `ρ<0` retained).
- **`window_viability`** — every built acceptor `score3 ≥` floor in EVERY slot; no new cross-junction cryptic 3′SS.
- **`ood` (incremental within-state, IC1 A1) — PER SLOT-DISTANCE:** real constructs at `p_nat`, `p_dis` each
  noninferior to a natural null; `p_dis` additionally noninferior to `p_nat`; `δ_OOD` per distance;
  planted-OOD FAILS. **The per-slot `δ_OOD` magnitude (point + CI) is REPORTED alongside `β_slot`** (§4).
- **`planted_identity_positive` (NEW — sensitivity for the binding equivalence):** a window pair matched on
  single-candidate f-drop but **deliberately composition/motif-DIVERGENT** (identity differs, reliance held)
  must yield `|β_window| ≥ 0.05` through the model. **An equivalence PASS on `β_window|slot` is interpretable
  as identity-invariance ONLY if this control fires;** the `strength_strata_valid` threshold is tied to this
  planted-identity MDE (not merely to a decorrelation sign). If the control does not fire, the equivalence is
  reported as "undetectable at matched reliance," not "interchangeable."
- **`strength_strata_valid`** — `D_m` matched on model AG-damage AND verified in-context (`|S(P)−S(D_m)|` vs a
  common competitor within ±0.05 pre-swap; single-candidate f-drop vs two-candidate `ρ` decorrelate so the
  equivalence retains power). `D_w` composition-balanced vs strong (residual netted A9b-style).
- **`sensitivity_onpath` (partner-SPECIFIC, C1 ~37×)** — native-partner-corrupt MINUS matched non-partner sham.
- **`generic_AG_floor_check`** — SMOKE-ONLY (`ρ(competent) − ρ_sham > 0` is near-non-falsifiable and is NOT a
  binding gate); the BINDING conditions are the sham's mechanical guards (AG-corruptible, no cryptic 3′SS,
  composition-matched, well-estimated, seed-stable). `ρ_sham` magnitude REPORTED.

`--smoke` = mechanics + determinism + native_anchor + corruption_validity + argmax_stability +
partner-specific-on-path + generic-AG-floor(>0) on tiny-n.

---

## 7. Decision rule (frozen; TOTAL over slot × window outcomes) `⟐`

Gates PASS (incl. `planted_identity_positive` for any equivalence interpretation). Each component:
**material** (sup-t CI excludes 0 AND `|·| ≥` SESOI) / **detectable-but-immaterial** / **equivalent/bounded-
null** (CI ⊆ ±SESOI) / **inconclusive**. **`β_slot|competent` material licenses the occupant-GENERIC slot
claim ONLY if `τ_D` (foreign window) is itself ≥ SESOI**; if only `τ_P` moves, the claim narrows to *"displacing
the native acceptor from its native slot reduces its own donor support"* (occupant-specific displacement penalty).

**Total table** (rows cover every {SLOT}×{WINDOW} outcome):
| SLOT (`β_slot`, with τ_D check) | WINDOW (`β_window`) | Conclusion |
|---|---|---|
| material **and τ_D ≥ SESOI** | equivalent | **SLOT-dependent (occupant-generic), WINDOW-identity-invariant** → §0 claim + §0 limitation |
| material but **τ_D < SESOI** | equivalent | occupant-SPECIFIC displacement penalty; NOT occupant-generic slot-dependence |
| material (either τ form) | material | support depends on BOTH slot and identity (report both; interaction qualifies) |
| material | inconclusive/immaterial | license follow-up; **withhold** the "window-invariant" clause |
| equivalent/bounded-null | material | WINDOW-identity-dependent, no material slot effect |
| equivalent/bounded-null | equivalent | no material dependence on slot or identity |
| inconclusive | any | inconclusive on slot; report window per its cell |
| any material component | — | additionally licenses the contingent tracing prereg |
| all inconclusive | | pilot instability / no signal → stop |

**Proceed to mechanistic tracing IF `β_slot|competent` or `β_window|slot` is material.** Given the pilot slot
effect (~5× SESOI) this branch is **expected to fire**, not merely contingent; the tracing prereg (L3H1
selection / payload mediation, A8) **may NOT introduce new behavioural SpliceBERT contrasts** — it is
mechanistic only. **Transfer to the second GFM proceeds regardless of outcome** (non-exclusive with tracing).

---

## 8. Pre-freeze pilots (disjoint TRAIN; no val read; RESOURCE-CONSIDERATE — shared laptop) `⟐`

Feasibility pilot **DONE** (`run_ic2_feasibility.py`). Remaining:
1. **Displaced-slot geometry + per-distance OOD** — `p_dis = p_nat−60` (phase-preserving, zone-clear);
   retained-pool census; incremental `ΔNLL` per distance (native/identity pass, planted-OOD fails); `δ_OOD`
   point+CI per distance; `p_dis` noninferior to `p_nat`.
2. **Decoy strata + CLUSTERING CENSUS** — `D_m` reliance-matched + in-context validity + **planted-identity
   positive control** MDE; `D_w` composition-balanced. **Dedicated TRAIN clustering census large enough to
   estimate the decoy-source (and scaffold) ICC with a usable CI** → design effect → N pinned to its upper CI.
3. **Inference** — clustered-SE TOST N for 80% equivalence power; **max feasible N + contingency** (§5); sup-t
   `c` + `m=3`; first/second-order SESOIs; confirm interaction achievable at N (else reported-only).
4. **Sensitivity on-path** (C1 ~37×); **determinism test** across hash seeds.
- **Resource note:** shared laptop — modest batches, NO parallel heavy MPS jobs, check load before launch.

**Provenance:** checkpoint sha256 `2ad91428…0149d`; MaxEnt pinned (operational STRATIFIER only, R22);
shuffler / decoy-pool / slot geometry pinned; SESOIs + `m` + N + floor + seeds preregistered after the pilots,
with a freeze manifest; no val read before freeze. Feasibility artifact:
`results/splicebert_coupling/ic2_feasibility.json`.
