# Pre-registration — IC2 v5.2: does donor support depend on candidate SLOT, and is matched reliance transportable? (behaviour-level)

**Status: DRAFT v5.2 (calibration-ready) for freeze-audit → freeze → confirmatory. DO NOT RUN confirmatory until frozen.**
Compact two-candidate assay; **final pre-specified behavioural SpliceBERT experiment** — after v5.2, **transfer
to a second GFM regardless of outcome** (a material result additionally licenses a SEPARATE tracing prereg;
branches non-exclusive). Behaviour-level; address/payload naming deferred to that mediation study.

**Lineage.** v4 → 4-lens red-team (all blocker) → v5 narrowing (retired the floored-sham `β_slot,specific`
and the between-gene spacing reference arm; κ demoted) → 2-lens re-audit (both **major-fixes, no blocker**) →
v5.1 (per-window slot terms, clustered-N, total table, "reliance-matched") → **v5.2** (this): the WINDOW
equivalence was **partly true by construction** (`D_m` matched to `P` on the model's own reliance), so it is
**retired** and replaced by the **TRANSPORT DiD `β_transport = τ_D − τ_P`** (matching on a FROZEN INDEPENDENT
reference; the DiD nets the matched native context out → nothing forced to 0); claim → **"matched donor
reliance is positionally transportable"** (NOT "windows interchangeable"); the identity control becomes a
**cross-fitted (TRAIN-A freeze → TRAIN-B test) WORDING gate**; a **decoy-source cluster floor + reuse cap** is
frozen (power is set by clusters, not construct count).

**Pilot basis** (`ic2_feasibility.json`, n=50 TRAIN, reciprocal swap): SLOT advantage **+0.349** (native vs
displaced, window-averaged; both the native occupant P *and* a foreign window get it), WINDOW-identity
advantage **−0.0002** (reliance-matched P vs D, slot-averaged), argmax stable **90%**, generic-AG floor
gradient **+0.034**, native coupling **+0.52**.

---

## 0. INTENT + PRIMARY CLAIM (behaviour-level, narrowed)

> **In a constructed two-candidate assay, does the donor's masked G-margin depend on which scaffold SLOT
> holds a competent acceptor (native AND foreign windows), and does a reliance-matched pair's support
> difference SURVIVE DISPLACEMENT across slots (positional transportability)?**

**Licensed claim IF the confirmatory pattern confirms (frozen wording):** *"In a constructed two-candidate
assay, donor support depends strongly on which scaffold slot contains a competent acceptor — for a foreign
window as well as the native one — and matched donor reliance is positionally transportable across slots."*
(The "foreign window as well" clause is licensed ONLY if the `D_m` per-window slot term `τ_D` is itself
material — §4/§7; else the slot claim narrows to an occupant-specific displacement penalty. The transport
clause replaces the retired "windows are interchangeable" framing — see §4.)

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
- **`D_m` = RELIANCE-MATCHED decoy:** matched to `P` on the **model's own single-candidate AG-damage f-drop
  measured in a FROZEN, INDEPENDENT single-candidate reference context** (R22-safe; NOT score3-as-semantics).
  The match is **NOT re-imposed in the two-candidate confirmatory context** (that would force the tested
  difference toward 0). Because matching-in-the-reference partially pulls the native two-candidate difference
  toward 0, the confirmatory WINDOW estimand is the **transport DiD** `β_transport` (§4), which nets that
  native context out. A cross-fitted discriminability control gates its interpretation (§6).
- **Generic-AG arm (`Sh`) = COMPETENCE/FLOOR CHECK ONLY:** composition-matched non-acceptor AG window;
  verifies support tracks competence. **`ρ_sham` REPORTED, never subtracted, never thresholded-small.**
- **3b. STRENGTH axis = REPORTED SECONDARY** (NOT confirmatory, NOT mediation-gating): `ρ(strong)` vs
  `ρ(D_w weak-but-competent)` + slot×strength, **composition-adjusted CIs** (score3 ≈ PPT-pyrimidine =
  A9b/R22). Cannot on its own license mediation or a strength/payload claim.

---

## 4. Estimands (frozen)

**CONFIRMATORY family (EXACTLY m = 3; studentized sup-t simultaneous CIs — §5).** Atomic per-window slot terms:
`τ_P = S(P,p_nat)−S(P,p_dis)` (**NATIVE** occupant; carries occupant-specific native-position memory) and
`τ_D = S(D_m,p_nat)−S(D_m,p_dis)` (**FOREIGN** occupant; occupant-generic slot/boundary/context). *(Pilot: both ≈ 0.35.)*
1. **`τ_P`** (material?) — native-occupant displacement effect.
2. **`τ_D`** (material?) — foreign-occupant slot effect. **The occupant-GENERIC slot claim requires `τ_D`
   material** (a foreign window also shows the slot effect). `β_slot|competent = ½(τ_P+τ_D)` is REPORTED as the
   summary slot magnitude *(pilot +0.349)*.
3. **TRANSPORT contrast** `β_transport = [S(P)−S(D_m)]_{p_dis} − [S(P)−S(D_m)]_{p_nat} = τ_D − τ_P`,
   **EQUIVALENCE-tested within ±0.05** (studentized CI ⊆ [−0.05,+0.05]). *(Pilot: ≈ 0.)* Because `D_m` is
   matched to `P` on a FROZEN INDEPENDENT single-candidate reference (§3) and this DiD nets the (partially
   matched) native context against the free displaced context, **no part of `β_transport` is zero by
   construction**. It asks whether matched donor reliance **SURVIVES DISPLACEMENT (positional
   transportability)**, NOT whether windows are "interchangeable". Interpretable as transportability ONLY if
   the **cross-fitted discriminability control fires** (§6); else reported as "no non-transport DETECTABLE at
   matched reliance." (β_transport is exactly the slot×window interaction; the two are one estimand.)

**Equivalence bound (β_transport):** ±0.05 nats = **12.5% of native P-reliance (0.402)** — a substantively
negligible position-dependent identity shift.

**SECONDARY (reported; NOT in the family, NOT mediation-gating):** strength `ρ(strong)−ρ(D_w)` + slot×strength
(composition-adjusted); κ nonadditivity (characterization only); `ρ_sham` floor magnitude; **per-slot `δ_OOD`
point + CI reported alongside `β_slot`** (bounds the OOD/displacement share of the RAW slot effect); the
exploratory sweep curve; the descriptive natural-loci context.

---

## 5. Inference (frozen) `⟐`

- **Dependence — TWO-WAY CLUSTERING PRIMARY** on `{scaffold gene, decoy-source gene}` (unique decoys NOT
  feasible: each scaffold needs ≥1 reliance-matched + ≥1 weak decoy-source ≠ scaffold, viable in both slots).
- **DECOY-SOURCE CLUSTER FLOOR + REUSE CAP (frozen; power is set by these, NOT the construct count):** freeze a
  **minimum number of independent decoy-source genes** and a **max reuse per decoy-source gene**, pinned from
  §8.2's cluster census. The effective sample size and thus power are governed by the number and BALANCE of
  `{scaffold, decoy-source}` clusters; nominal construct count is not a power input.
- **SUP-t over m = 3:** studentize each of `{τ_P, τ_D, β_transport}` by its two-way wild-cluster bootstrap SE
  (resample WHOLE scaffold clusters); `c` s.t. `P(max_{j≤3}|t_j*| ≤ c)=0.95`. No omnibus-then-unadjusted.
  `β_transport`'s equivalence uses the same studentized band (CI ⊆ ±0.05, a two-sided-CI-subset TOST).
- **Power / N — 80% SIMULTANEOUS power on the actual TWO-WAY-CLUSTERED simultaneous-band SE (NOT the IID MDE
  table).** The **binding component is the `β_transport` equivalence** (`τ_P,τ_D` ≈ +0.35 have ~7× margin).
  N (and the decoy-source cluster floor) are pinned from the clustered SE using a **decoy-source ICC estimated
  on TRAIN** (§8.2), bound to the **upper CI of the design effect**. **State a max feasible N/cluster count and
  a CONTINGENCY:** if 80% simultaneous equivalence power is unreachable, `β_transport` is reported as
  **"bounded-null at the achieved precision," NOT "transportable"** (the transport claim is withheld, not faked).
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
- **`transport_discriminability` (CROSS-FITTED; sensitivity for the binding equivalence):** on **TRAIN-A**,
  discover and FREEZE a motif-divergent, reliance-matched pair (or a deterministic selection rule); on
  **disjoint TRAIN-B**, test whether that frozen control yields `|β_transport| ≥ 0.05`. **No searching until
  something fires on one dataset.** **This is a WORDING gate, NOT an experiment gate:** if the frozen control
  fires on TRAIN-B, an equivalence PASS on the real `β_transport` is interpretable as positional
  transportability; if it does NOT fire, the real result is reported as "no non-transport DETECTABLE at matched
  reliance," and the rest of the experiment (`τ_P`, `τ_D`, the SLOT arm) proceeds unaffected.
- **`strength_strata_valid`** — `D_m` matched on the model's single-candidate f-drop in a **FROZEN INDEPENDENT
  reference context**; the match is **NOT re-imposed in the two-candidate confirmatory context** (§3, so the
  tested difference is not forced toward 0). Verify only that single-candidate f-drop vs two-candidate `ρ`
  **DECORRELATE** (so `β_transport` is non-degenerate). `D_w` composition-balanced vs strong (residual netted A9b).
- **`sensitivity_onpath` (partner-SPECIFIC, C1 ~37×)** — native-partner-corrupt MINUS matched non-partner sham.
- **`generic_AG_floor_check`** — SMOKE-ONLY (`ρ(competent) − ρ_sham > 0` is near-non-falsifiable and is NOT a
  binding gate); the BINDING conditions are the sham's mechanical guards (AG-corruptible, no cryptic 3′SS,
  composition-matched, well-estimated, seed-stable). `ρ_sham` magnitude REPORTED.

`--smoke` = mechanics + determinism + native_anchor + corruption_validity + argmax_stability +
partner-specific-on-path + generic-AG-floor(>0) on tiny-n.

---

## 7. Decision rule (frozen; TOTAL over slot × window outcomes) `⟐`

Gates PASS (incl. `transport_discriminability` firing on TRAIN-B for any transportability interpretation). Each
component: **material** (sup-t CI excludes 0 AND `|·| ≥` SESOI) / **detectable-but-immaterial** /
**equivalent/bounded-null** (CI ⊆ ±SESOI) / **inconclusive**. **The occupant-GENERIC slot claim requires `τ_D`
material** (foreign window shows the slot effect); if only `τ_P` moves, the slot claim narrows to *"displacing
the native acceptor from its native slot reduces its own donor support"* (occupant-specific displacement penalty).

**Total table:**
| SLOT (`τ_P`, `τ_D`) | TRANSPORT (`β_transport`) | Conclusion |
|---|---|---|
| both material | equivalent (& control fired) | **occupant-generic SLOT-dependent; matched reliance POSITIONALLY TRANSPORTABLE** → §0 claim + limitation |
| both material | equivalent (control did NOT fire) | slot-dependent; transport "not detectable at matched reliance" (transport clause withheld) |
| both material | material | slot-dependent AND matched reliance does NOT transport (position modulates identity) — report both |
| only `τ_P` material | any | occupant-SPECIFIC displacement penalty; withhold the occupant-generic slot claim |
| both bounded-null | material | no material slot effect; matched reliance non-transportable |
| both bounded-null | equivalent | no material slot dependence (transport per control) |
| inconclusive on τ | any | inconclusive on slot; report transport per its cell |
| any material component (`τ_P`, `τ_D`, `β_transport`) | — | additionally licenses the contingent tracing prereg |
| all inconclusive | | pilot instability / no signal → stop |

**Proceed to mechanistic tracing IF any confirmatory component (`τ_P`, `τ_D`, or `β_transport`) is material.** Given the pilot slot
effect (~5× SESOI) this branch is **expected to fire**, not merely contingent; the tracing prereg (L3H1
selection / payload mediation, A8) **may NOT introduce new behavioural SpliceBERT contrasts** — it is
mechanistic only. **Transfer to the second GFM proceeds regardless of outcome** (non-exclusive with tracing).

---

## 8. Pre-freeze pilots (disjoint TRAIN; no val read; RESOURCE-CONSIDERATE — shared laptop) `⟐`

Feasibility pilot **DONE** (`run_ic2_feasibility.py`). Remaining (Cam-directed order):
1. **`D_m` feasibility** — for each scaffold `P`, find a reliance-matched decoy on a **FROZEN INDEPENDENT
   single-candidate reference**; report match residual, yield, and the realized independent decoy-source pool.
2. **Cross-fitted discriminability calibration** — TRAIN-A: freeze a motif-divergent reliance-matched selection
   rule; disjoint TRAIN-B: test `|β_transport| ≥ 0.05` for it (the `transport_discriminability` WORDING gate).
   No search-until-fires.
3. **ICC / design-effect pilot** — dedicated TRAIN clustering census large enough to estimate the decoy-source
   (and scaffold) ICC with a usable CI → design effect → **decoy-source cluster floor + reuse cap + N** pinned
   to the upper CI; clustered-SE TOST N for 80% `β_transport` equivalence power + **max feasible N + contingency**.
4. **Displaced-slot geometry + per-distance OOD** — `p_dis = p_nat−60` (phase-preserving, zone-clear);
   retained-pool census; incremental `ΔNLL` per distance (native/identity pass, planted-OOD fails); `δ_OOD`
   point+CI; `p_dis` noninferior to `p_nat`.
5. **Sensitivity on-path** (C1 ~37×); **sup-t `c` + `m=3` + SESOIs**; **determinism test** across hash seeds.
**If 1–3 pass → one final freeze-audit → request GO.**
- **Resource note:** shared laptop — modest batches, NO parallel heavy MPS jobs, check load before launch.

**Provenance:** checkpoint sha256 `2ad91428…0149d`; MaxEnt pinned (operational STRATIFIER only, R22);
shuffler / decoy-pool / slot geometry pinned; SESOIs + `m` + N + floor + seeds preregistered after the pilots,
with a freeze manifest; no val read before freeze. Feasibility artifact:
`results/splicebert_coupling/ic2_feasibility.json`.
