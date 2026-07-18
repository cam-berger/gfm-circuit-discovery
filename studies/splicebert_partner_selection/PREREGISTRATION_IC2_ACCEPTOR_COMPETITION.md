# Pre-registration — IC2 v5: does donor support depend on candidate SLOT or WINDOW? (behaviour-level)

**Status: DRAFT v5 (post-red-team, narrowed) for freeze-audit → freeze → confirmatory. DO NOT RUN confirmatory until frozen.**
Compact two-candidate assay; **final pre-specified behavioural SpliceBERT experiment** — after v5, **transfer
to a second GFM regardless of outcome** (a material result additionally licenses a SEPARATE tracing prereg;
the branches are non-exclusive). Behaviour-level; address/payload naming deferred to that mediation study.

**v5 supersedes v4** (4-lens red-team → all blockers). The v4 `β_slot,specific` (slot effect netted by a
generic-AG sham) is **RETIRED**: the sham is FLOORED (a non-competent window has ~0 reliance, so displacing
it costs ~0), so it cannot estimate the displacement/OOD penalty and **must not be subtracted**. A
native-spacing reference arm was also rejected — comparing different genes whose native acceptors sit at
different distances confounds distance with gene background/context/phase/architecture (`E[support|d_native]
≠ causal effect of distance`). **v5 therefore does NOT attempt to identify a spacing rule.** It measures only
the two within-construct contrasts the reciprocal swap identifies cleanly, and states the limitation explicitly.

**Pilot basis** (`ic2_feasibility.json`, n=50 TRAIN, reciprocal swap): SLOT advantage **+0.349** (native slot
vs a displaced slot, window-averaged), WINDOW-identity advantage **−0.0002** (matched-strong P vs D,
slot-averaged), argmax stable on **90%**, generic-AG floor gradient **+0.034**, native coupling **+0.52**.

---

## 0. INTENT + PRIMARY CLAIM (behaviour-level, narrowed)

> **In a constructed two-candidate assay, does the donor's masked G-margin depend on which scaffold SLOT
> holds a competent acceptor, and on WHICH matched-strong acceptor WINDOW occupies a fixed slot?**

**Licensed claim IF the pilot pattern confirms (frozen wording):** *"In a constructed two-candidate assay,
donor support depends strongly on which scaffold slot contains a competent acceptor, while exchanging two
matched-strong acceptor windows between those slots has no material effect."*

**Explicit, frozen limitation (reported with the claim):** *"The experiment does NOT distinguish a general
donor–acceptor spacing rule from native-boundary privilege, local slot context, or the penalty incurred by
displacing an acceptor from its natural position."*

**NOT claimed:** "positional address dominates payload"; a spacing rule; biological acceptor selection. **The
model emits a donor SCORE `f`, not a splice-choice probability** — "support" = donor-margin dependence, not
splicing outcome. Verb is **"depends on"**, never "allocates".

---

## 1. Observable + READOUT AUDIT (frozen)

- **PRIMARY observable: SEALED max-margin** `f = logit(G) − max{A,C,T}` at the masked donor (piecewise-linear
  in the logits; nonlinear only when the non-G argmax switches).
- **Support** `S(X,p) := ρ(X@p) = f(both intact) − f(X@p corrupted, partner intact)` — the two-candidate
  reliance of window `X` at slot `p` (competition-inclusive; sign-agnostic — `S<0` = a rival depressing `f`
  is retained, not gated out).
- **`argmax_stability` GATE (≥90%, pilot-frozen):** the non-G argmax is stable across a construct's cells on
  ≥90% of constructs (pilot 90%). **Full-set sealed-margin is the ONLY confirmatory readout**; recomputation
  on the stable subset is a **diagnostic** — argmax-switching correlates with response magnitude, so the
  subset is attenuation-biased and a full-vs-subset gap does NOT impeach the full-set result.
- **Fixed-class robustness decomposition:** report `G−A, G−C, G−T`; each confirmatory DiD's sign-concordance
  fraction with the sealed DiD is reported (pilot ~84%). Decomposition only; not a replacement primary
  (probabilities would add real saturation and are not used).

---

## 2. Cohort & construction (SYNTHETIC constructed-assay PRIMARY)

- **SYNTHETIC = PRIMARY, scoped:** natural dual-acceptor yield is inadequate (census: 142 loci, **0 at ≥80 nt
  spacing**). All conclusions are scoped to **"a constructed two-candidate assay."**
- **Scaffold pool:** TRAIN internal introns, gene-deduplicated, `L≥110` so a native slot + one displaced slot
  (two 40 nt 3′SS windows) + a donor-protected zone fit (pilot pool 1,985). Held-out gene split preserved.
- **Two slots (CONFIRMATORY): native slot `p_nat` (= the gene's native acceptor position) and ONE frozen
  displaced slot `p_dis`,** at a donor-distance **that is a multiple of 3 from `p_nat`** (holds the C8
  period-3 phase constant across the contrast) and **clears every protected zone by construction** (upstream
  window edge `> d0 + 8`; pilot's feasible `p_dis = p_nat − 60`, phase-preserving, is the frozen default).
  The retained-pool count at `p_dis` is reported (per-distance census, IC1-style).
- **WINDOW = a full 3′SS unit** (branchpoint + PPT + AG, `[·−37..·+2]`, 40 nt), 40→40 in-place replacement
  (no coordinate shift). Each built acceptor passes `window_viability` in EVERY slot it occupies, and
  introduces **no new cryptic 3′SS across the junction** (added constraint).
- **Slot-SWEEP = EXPLORATORY only** (extra displaced distances, phase-balanced/within-phase reported): a
  descriptive support-vs-distance curve. **Not causal identification; cannot license a spacing-rule claim.**
- **Natural alt-3′SS (142 loci) = PURELY DESCRIPTIVE CONTEXT, NOT external validity:** at 15 nt median
  spacing the two 40 nt windows overlap (geometrically incompatible with the assay). Reported as descriptive
  colour on natural dual-acceptor donors; **cannot corroborate, override, or rescue any synthetic conclusion.**

---

## 3. Design — reciprocal swap + generic-AG floor-check (+ secondary strength) `⟐ RED-TEAM`

Per scaffold, two competent windows are placed at `{p_nat, p_dis}` in **both reciprocal arrangements**;
cells = both-intact and each-single-corrupt (`AG→CC` at the slot's `[·−2,·−1]`).
- **Config N:** `P@p_nat , D_m@p_dis`. **Config W (swap):** `D_m@p_nat , P@p_dis`. → gives `S(P,p_nat), S(D_m,p_dis)`
  (N) and `S(D_m,p_nat), S(P,p_dis)` (W). `D_m` = a matched-strong decoy (§3b).
- **Generic-AG arm (`Sh`) = COMPETENCE / FLOOR MANIPULATION CHECK ONLY:** a composition-matched non-acceptor
  AG-corruptible window. It verifies the support readout responds to competence (a floored, non-competent
  window gets near-floor support). **It is NOT an adjustment and NOT a specificity estimator — `ρ_sham` is
  REPORTED, never subtracted and never thresholded-small.**
- **`D_m` (matched-strong decoy):** matched to `P` on the **model's own single-candidate AG-damage f-drop**
  (operational, R22-safe — NOT score3 as semantics). Validity is calibrated in-context (§6 `strength_strata_valid`).

**3b. STRENGTH axis = REPORTED SECONDARY (NOT confirmatory; NOT mediation-gating):** `ρ(strong)` vs
`ρ(D_w weak-but-competent)` and slot×strength. MaxEnt `score3` strata are largely PPT-pyrimidine composition
(the A9b axis / R22), so this is reported with **composition-adjusted CIs** and explicitly cannot, on its
own, license mediation or a "strength/payload" claim. Demoted per red-team (second-order, composition-confounded).

---

## 4. Estimands (frozen)

**CONFIRMATORY family (EXACTLY m = 3; studentized sup-t simultaneous CIs — §5):**
1. **SLOT effect** `β_slot|competent = mean_{X∈{P,D_m}} [ S(X,p_nat) − S(X,p_dis) ]`
   `= ½[(S(P,p_nat)+S(D_m,p_nat)) − (S(P,p_dis)+S(D_m,p_dis))]`. *(Pilot: +0.349.)* — do competent windows
   receive different support at the native vs displaced slot? **RAW, not sham-netted.**
2. **WINDOW-identity effect** `β_window|slot = mean_{p∈{p_nat,p_dis}} [ S(P,p) − S(D_m,p) ]`, **EQUIVALENCE-
   tested within ±0.05** (TOST / studentized CI ⊆ [−0.05,+0.05]). *(Pilot: −0.0002.)* — does exchanging two
   matched-strong windows at a fixed slot change support?
3. **SLOT × WINDOW interaction** `= ½[(S(P,p_nat)−S(P,p_dis)) − (S(D_m,p_nat)−S(D_m,p_dis))]`.

**Equivalence bound justification (component 2):** ±0.05 nats = **12.5% of native P-reliance (0.402)** and
**~14% of the SLOT effect (0.349)** it is contrasted against — a substantively negligible support change.

**SECONDARY (reported; NOT in the sup-t family, NOT mediation-gating):** strength `ρ(strong)−ρ(D_w)` +
slot×strength (composition-adjusted); κ nonadditivity `f11−f10−f01+f00` (characterization only, attribution
deferred); `ρ_sham` floor magnitude; the exploratory slot-sweep curve; the descriptive natural-loci context.

---

## 5. Inference (frozen) `⟐ RED-TEAM`

- **Dependence — TWO-WAY CLUSTERING PRIMARY** on `{scaffold gene, decoy-source gene}` (unique decoys are NOT
  feasible: each scaffold needs ≥1 matched-strong + ≥1 weak decoy-source gene ≠ scaffold, both viable in both
  slots). The pilot reports realized cluster counts **per stratum**; a margin < ~30 clusters → wild-cluster
  bootstrap + small-cluster correction.
- **SUP-t simultaneous bands over the m = 3 family:** studentize each component by its two-way wild-cluster
  bootstrap SE (resample WHOLE scaffold clusters carrying all cells/slots); critical value `c` with
  `P(max_{j≤3}|t_j*| ≤ c)=0.95`. **No omnibus-then-unadjusted interpretation.** Component 2's equivalence uses
  the same studentized bootstrap (CI ⊆ ±0.05).
- **Power / N — 80% SIMULTANEOUS power on the CLUSTERED SE (not the 50%-power IID MDE table):** N bound to the
  **BINDING component = the WINDOW equivalence** (component 1's +0.349 has a ~5× margin and does not bind).
  Pilot: WINDOW SD 0.33 → recompute clustered-SE TOST N for 80% equivalence power (target region N≈1,500–2,500,
  pinned after the cluster-count census). Freeze N to the equivalence.
- **Per-order SESOIs (pinned, substantive-anchored — NOT pilot-SE-derived):** first-order (SLOT, WINDOW) SESOI
  = **0.05 nats** (12.5% of native reliance 0.402); second-order (interaction) SESOI pinned separately as a
  stated fraction of native reliance and **verified achievable at the frozen N** (else the interaction is
  reported-only). No SESOI is set from the pilot SE alone (IC1 anti-tautology lesson).

---

## 6. Gates / memorials (HARD preflight; STOP on failure) `⟐ RED-TEAM`

- **`determinism`** — `sorted(set())` + pinned `PYTHONHASHSEED`; identical constructs/cohort/results across
  `PYTHONHASHSEED∈{0,1,2,17}` (IC1 A2); determinism test required.
- **`argmax_stability`** — non-G argmax stable on ≥90% of constructs (§1).
- **`native_anchor`** — native single-acceptor construct reproduces coupling, as a CI/band able to fail
  (pilot +0.52; note vs the +0.87 headline is a long-intron-subset difference). The single→two-candidate cost
  is measured against a reference with a **SHAM at the displaced slot** (matching the 40→40 edit) so the cost
  isolates competition, not a scaffold edit.
- **`corruption_validity`** — AGGREGATE manipulation check (median single-candidate f-drop > 0 AND `score3`
  collapse on the built window); explicitly NOT a per-construct `ρ>0` requirement (competitive `ρ<0` retained).
- **`window_viability`** — every built acceptor `score3 ≥` floor in EVERY slot; no new cross-junction cryptic 3′SS.
- **`ood` (incremental within-state, IC1 A1) — PER SLOT-DISTANCE:** real-acceptor constructs at `p_nat` and
  `p_dis` each noninferior to a natural null; **`p_dis` additionally noninferior to `p_nat`-slot OOD** (catches
  a displacement-dependent OOD gradient the shuffled sham cannot reproduce); `δ_OOD` re-derived per distance;
  planted-OOD construct must FAIL.
- **`generic_AG_floor_check`** — the `Sh` arm's support is near floor and aggregate `ρ(competent) − ρ_sham > 0`
  (competent windows out-support a floored AG), validating that support tracks competence. Gate on the sham's
  **mechanical validity + estimation quality only** (AG-corruptible, composition-matched, well-estimated,
  seed-stable); **`ρ_sham` magnitude is REPORTED, never thresholded.**
- **`sensitivity_onpath` (partner-SPECIFIC, C1 ~37×)** — native-partner-corrupt MINUS a matched non-partner
  sham corrupt gives a detectable partner-specific drop (detects candidate SUPPORT, not generic AG damage).
- **`strength_strata_valid` (secondary arm)** — `D_m` matched on model AG-damage AND verified **in-context**
  on the pilot: `|S(P)−S(D_m)|` vs a common competitor is within the ±0.05 band BEFORE the identity swap, and
  single-candidate f-drop vs two-candidate `ρ` **decorrelate** enough (competition variance ≫ matching
  tolerance) that the equivalence retains power — else the WINDOW equivalence is flagged mechanically induced
  and reported as uninformative. `D_w` composition-balanced vs strong (residual netted A9b-style).

`--smoke` = mechanics + determinism + native_anchor + corruption_validity + argmax_stability +
partner-specific-on-path on tiny-n.

---

## 7. Decision rule (frozen) `⟐ RED-TEAM`

Gates PASS. Each confirmatory component: **material** (sup-t CI excludes 0 AND `|·| ≥` its per-order SESOI) /
**detectable-but-immaterial** / **bounded-null / equivalent** (CI ⊆ ±SESOI) / **inconclusive**.

**Frozen estimand→claim table (behaviour-level):**
| Pattern | Conclusion |
|---|---|
| `β_slot|competent` material AND `β_window|slot` equivalent (⊆±0.05) | **SLOT-dependent, WINDOW-identity-invariant** → the licensed §0 claim + the §0 limitation |
| `β_slot|competent` material AND `β_window|slot` material | support depends on BOTH slot and window identity (report both; interaction qualifies) |
| `β_slot|competent` bounded-null AND `β_window|slot` equivalent | **no material support dependence** on slot or matched-strong identity |
| any material component | (see rows above) — a material SLOT **or** WINDOW effect licenses the contingent tracing follow-up |
| all inconclusive | pilot instability / no behaviour-level signal → stop |

**Proceed to mechanistic tracing ONLY IF a confirmatory component (`β_slot|competent` or `β_window|slot`) is
material.** The tracing follow-up (L3H1 selection / payload mediation, A8 reconstruct-inject) is a SEPARATE
contingent prereg, not part of v5. **Transfer to the second GFM proceeds regardless of outcome** (non-exclusive
with any tracing licensed above).

---

## 8. Pre-freeze pilots (disjoint TRAIN; no val read; RESOURCE-CONSIDERATE — shared laptop) `⟐`

Feasibility pilot **DONE** (`run_ic2_feasibility.py`). Remaining pre-freeze calibration:
1. **Displaced-slot geometry + per-distance OOD** — `p_dis = p_nat−60` (phase-preserving, protected-zone-clear);
   retained-pool census; corrected incremental `ΔNLL` per distance (native/identity pass; planted-OOD fails);
   `p_dis` noninferior to `p_nat`; `δ_OOD` per distance.
2. **Decoy strata + clustering** — `D_m` matched on model AG-damage + in-context validity (decorrelation +
   pre-swap equivalence); `D_w` composition-balanced; realized two-way cluster counts PER stratum (verify the
   equivalence-binding N is reachable).
3. **Inference** — clustered-SE TOST N for 80% equivalence power; sup-t critical value + `m=3`; first/second-
   order SESOIs (substantive-anchored); confirm interaction achievable at N (else demote to reported-only).
4. **Sensitivity on-path** (C1 ~37×) + **generic-AG floor check**; **determinism test** across hash seeds.
- **Resource note:** shared laptop — modest batches, NO parallel heavy MPS jobs, check load before launch.

**Provenance:** checkpoint sha256 `2ad91428…0149d`; MaxEnt pinned (operational STRATIFIER only, R22);
shuffler / decoy-pool / slot geometry pinned; SESOIs + `m` + N + floor + seeds preregistered after the pilots,
with a freeze manifest; no val read before freeze. Feasibility artifact:
`results/splicebert_coupling/ic2_feasibility.json`.
