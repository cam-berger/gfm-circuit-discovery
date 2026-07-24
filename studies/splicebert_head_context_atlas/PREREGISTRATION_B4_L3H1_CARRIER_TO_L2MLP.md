# PREREGISTRATION — B4: L3H1/carrier -> L2MLP bridge (HCA20 residual) (v1)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: NOT FROZEN — the 5-lens design red-team (2026-07-22) adjudicated this draft DEFER (survivor=NO).**
See [`REDTEAM_SYNTHESIS_B4.md`](REDTEAM_SYNTHESIS_B4.md) (same directory). NON-IDENTIFIABLE on a frozen post-LN
checkpoint (verified across three convergent lenses), like P3.4. The only carrier-deletion primitive edits the whole
position-embedding matrix globally at the source; post-LN renormalization leaves no additive carrier component to
subtract at a mid-layer edge; the only implementable deletion injects a globally-carrier-deleted layer-2 residual into
L2MLP's input, bundling the direct carrier→L2MLP removal with comb-collapse content routing through L2H15 (the layer-2
comb head feeding L2MLP in the same block, amplitude →3.4% when the band is deleted). So ΔΔ_bridge = direct +
through-comb; the "edge distinct from global C8" does not exist. Blocker is IDENTIFIABILITY, not power or cohort
location (the paired second-difference off the HCA20 base already clears SESOI; in-repo
atlas_context_matched_v7.validation.packets.jsonl supplies 1486 gene-labeled packets). The only locally-runnable
identifiable variant is a coarse TOTAL upstream carrier-dependence of L2MLP's partner write — a localized-C8 read, not
a bridge — which belongs with B3's carrier/address thread beside C34/C35. A real cross-estimand bridge needs a
training-time / subspace-freeze design if the Atlas becomes a headline. This v1 is retained as the **design of
record**; it must **NOT be run** as the bridge study it names — the disposition and the verbatim must-fixes are
recorded in "## Closure disposition & must-fixes (2026-07-22 design red-team)" at the foot of this file. **Not
executed this session:** the SpliceBERT.1024nt weights and the Atlas data are absent from this worktree, so any model
run must be launched from the main checkout.

This is the design-of-record for the HCA20 residual: the atlas validated L2MLP's partner-conditioned
`B` relation as material (donor **+0.344 [0.309,0.379]**; acceptor **+0.320 [0.286,0.354]**, HCA20) but
its **upstream attention route is instrument-limited** — "the frozen same-B validated attention set is
empty," and the A-estimand QK/V is non-identifiable at population scale. HCA20's own next-test line
(`CLAIMS.md`, `return_to_planner.json::next_actions[2]`) reads: *"a new frozen cross-estimand
intervention is required to test an L3H1/carrier→L2MLP bridge; Stage 2 did not identify it."* B4
formalizes that intervention so it can be red-teamed **before** any compute is spent.

**Founding correction (B4-1) — the literal "L3H1 → L2MLP" edge is architecture-forced to zero and is
NOT the identifiable estimand.** See §B4-1 below. This draft carries that correction into the design:
the runnable, identifiable core is a **shared-upstream period-3-carrier → L2MLP-input path patch**, not
a serial L3H1-output → L2MLP-input edge.

Provenance: SpliceBERT.1024nt sha256 `2ad91428…0149d`, post-LN BERT-MLM 6L×16H, hidden 512, **absolute**
learned position embeddings P (1026×512, `ad.model.bert.embeddings.position_embeddings`). Env
`gfm-thrust1`; compute **MPS** (no local CUDA). Deterministic cohorts (`PYTHONHASHSEED=0`,
`sorted(set())`).

The atlas boundary this study must not cross (HCA20): all recovery cuts are **overlapping controlled
recovery cuts, NOT additive mediation percentages, unique serial paths, binding, or biological function
labels.**

---

## B4-1 (this draft's founding correction) — the "L3H1 → L2MLP" edge is backward and forced

- **Layer ordering (verified).** The four validated general-use components (HCA18/HCA19) are heads
  **L1H12 (layer 1), L2H15 (layer 2), L3H1 (layer 3)** and the write **L2MLP (layer 2)**. In a
  feed-forward post-LN transformer, a layer-3 head's output enters the residual stream **after** the
  layer-2 MLP sub-block has already read and written. **There is no forward edge L3H1(L3) → L2MLP(L2).**
- **This is exactly why HCA20's same-B attention set is empty.** Amendment A2.5 defines the upstream
  attention group as validated attention candidates in the same `estimand × group` and **"strictly no
  later than the MLP layer."** For L2MLP that admits only layer ≤2 heads (L1H12, L2H15); both were
  validated on the **A**-estimand (zero-output), never as **same-B** sources, so `A2.5` correctly
  reports `upstream_path_instrument_limited` (`run_atlas_stage2_mechanisms.py`,
  `MECHANISM_VALIDATION_AMENDMENT_A2.md::A2.5`).
- **A naive "L3H1-output → L2MLP-input" path patch is an R23-class forced identity.** Patching a
  layer-3 activation into a layer-2 input either (a) does nothing on the forward pass (the L2 MLP has
  already run) or (b) requires an artificial re-injection that the residual algebra makes uninterpretable
  — a route "forced by the residual algebra proves nothing" (branch guard; cf. R23 bilinear-forced
  `Δ_bind`, R15 position-already-in-V, R21 forced full-restore identity).
- **The identifiable reframe (this design).** L3H1 and L2MLP are **siblings**: both are downstream
  readers of the **period-3 positional carrier**, which is present in the residual stream from the input
  positional embeddings (layer 0) and is therefore a genuine forward input to **both** L2MLP's layer-2
  input **and** (later) L3H1. The biologically meaningful, forward-consistent question HCA20 gestures at
  is: **does L2MLP's partner-conditioned payload write require the same period-3 positional address that
  L3H1 reads?** That is a **carrier → L2MLP-input** path patch (a real edge), with L3H1 named only as the
  *co-reader that motivates the carrier as the shared coordinate* — never as an upstream source patched
  into L2MLP. Any wording that keeps "L3H1 → L2MLP" as a serial edge is withdrawn here (B4-1).

Consequence for scope: B4 can, at most, reach *"L2MLP's partner-conditioned write is / is not gated by
the period-3 positional carrier in its input"* — an **OV/mediation-level, carrier-conditioned** claim. It
**cannot** establish a serial L3H1→L2MLP wire, binding, or an additive path fraction.

---

## 0. INTENT (north star — re-read at every gate)

*Is the period-3 positional carrier — the relative-offset routing coordinate that L3H1 reads (C32/C34/
C35) — also a causal input to L2MLP's partner-conditioned payload write (HCA20)? Concretely: when the
carrier is removed from L2MLP's own layer-2 input (and only there), does L2MLP's partner-conditioned `B`
effect on the reconstruction margin collapse, above an energy-matched carrier control and a generic-MLP
decoy?* If yes, the positional-addressing machinery and the payload-writing MLP share a common positional
coordinate — tightening the address × payload story (C23) at Atlas scale. If no (equivalence), L2MLP's
partner write is carrier-independent and the "bridge" is refuted.

Model-native observable: the **carrier × partner interaction on L2MLP's write**,
`ΔΔ = [B-effect | carrier intact at L2MLP input] − [B-effect | carrier deleted at L2MLP input]`,
where `B-effect` is L2MLP's partner-conditioned contribution to the masked-base hard margin
(HCA20's `L2MLP` partner-scrub relation), read out as coupling/margin on the reconstruction target.

**This does not touch the sealed test (chr8/9); held-out on a disjoint gene split.**

---

## 1. ESTIMANDS (all contrasts / interactions — never a raw single-arm difference)

Carrier conditions on the **L2MLP input edge only** (path-restricted): **I** (carrier intact),
**D** (`make(BAND)`, BAND=`range(339,346)`, the exact C8 clock — `run_carrier_surgery.py::make`, L82/L118),
applied to the positional-embedding contribution reaching L2MLP's MLP sub-block input **and nowhere else**
(other layers' carrier untouched — this is the edge/path restriction that separates B4 from re-measuring
C8 globally). **E** (energy-matched control band distribution — width-7 bands, spectral power within 25%
of BAND, periods 3/1.5/6 excluded, per-band comb-read ≥95%; the P3 specificity control,
`run_carrier_surgery.py` control-band construction, L125+).

Partner conditions (HCA20 `B`): **P0** (partner intact) vs **Pscrub** (annotated partner motif destroyed
in the same masked packet — `run_atlas_model_preflight.py::partner_destroyed`, L155; the legitimate
same-packet `B` source per A2.2).

L2MLP write: restore/zero L2MLP's native residual write in the same forward
(`run_atlas_stage2_mechanisms.py::_run_whole_restore`, L305-313), i.e. the exact HCA19/HCA20 `B`
partner-scrub relation, re-used verbatim.

### B4.a — carrier-gating of the L2MLP partner write (PRIMARY, one-sided positive)
Per target, in each carrier condition `c∈{I,D}`, the **L2MLP partner-conditioned effect**
`B_c = margin_hard(P0, L2MLP-native) − margin_hard(Pscrub, L2MLP-native)` under carrier `c` on the L2MLP
input edge. Estimand:
- **B4.a `ΔΔ_bridge = B_I − B_D`** — the paired within-target carrier × partner second difference.
  **ΔΔ_bridge family-adjusted CI-low > SESOI → the carrier gates L2MLP's partner write** (the bridge is
  material at OV/mediation level).
- **B4.a equivalence:** `ΔΔ_bridge ⊂ ±SESOI` (CI half-width < SESOI) → **L2MLP's partner write is
  carrier-independent** — the bridge is refuted. (Power-gated; see §7 — likely unreachable at MPS.)
- else **inconclusive**.

### B4.b — carrier-specificity vs the energy-matched control (the identifiability fix)
Repeat B4.a with the carrier deleted → **E control band distribution** rather than BAND. `ΔΔ_bridge`
counts as **carrier-specific** only if it **exceeds the whole E distribution** (add-one p over ≥N control
bands), never merely a single junk band (P3 N1/confound-specificity discipline). Reported as the
carrier-specificity of B4.a; a non-specific `ΔΔ_bridge` (indistinguishable from E) → the effect is
generic positional/energy disruption of L2MLP, **not** carrier-specific gating → Endpoint B.

### B4.c — decoy-MLP null (generic-write control)
Repeat the B4.a path patch with the carrier deleted at the input edge of a **decoy MLP** of the same
layer-depth class that is **not** a validated `B` write (a frozen importance-matched MLP control, per the
A2.4/A2.5 control discipline). Estimand `ΔΔ_decoy`; require `ΔΔ_bridge − ΔΔ_decoy` CI-low > SESOI for the
carrier-gating to be **specific to L2MLP's partner write** and not a property of every position-reading
MLP. A non-separable decoy → Endpoint B.

**Note (never reported as binding, R23):** none of B4.a–c is a binding test, an additive path fraction,
or a serial-wire claim. They are overlapping carrier-conditioned recovery cuts on one validated `B`
write (HCA20 boundary).

---

## 2. METHOD (reused machinery + the one net-new plumbing piece)

**Reused / validated (exact paths):**
- Carrier deletion + construction anchor + energy-matched control bands + comb-read:
  `scripts/splicebert/run_carrier_surgery.py` (`make`, `BAND`, control-band builder, comb amplitude).
- L2MLP `B` partner-scrub write + whole-restore + health, exactly as HCA19/HCA20:
  `scripts/splicebert_head_context_atlas/run_atlas_stage2_mechanisms.py`
  (`_run_whole_restore` L305-313, `score`, health probes), with the same-packet `B` source
  `run_atlas_model_preflight.py::partner_destroyed` (L155).
- Coupling / hard margin / masked-MLM health: `src/gfmcd/splicing/circuit.py` (`CouplingTask`,
  health probes), `src/gfmcd/splicing/unembed.py::margin_hard` (L19).
- Gene-clustered CIs and ratio CIs: `src/gfmcd/splicing/circuit.py::gboot` (L287),
  `gboot_ratio` (L303); dyadic (two-way) resampling for any interchange reuse (R22).
- Path/edge-patching primitives (the ACDC family the reviewer names): directed path patching
  `scripts/splicebert/run_path_decomposition.py`; readout-path `scripts/splicebert/run_readout_path.py`;
  faithfulness scaffolding `scripts/splicebert/run_a6_circuit_faithfulness.py`;
  attribution `src/gfmcd/circuits/attribution.py`; perturbation `src/gfmcd/splicing/perturb.py`.

**Net-new plumbing (piloted in `--smoke`; must pass before any freeze):**
- **Edge-restricted carrier deletion at the L2MLP input** — a forward hook that subtracts the period-3
  carrier component of the positional-embedding contribution **only** in the residual stream feeding
  L2MLP's MLP sub-block, leaving the carrier intact at every other layer/edge. This is the single load-
  bearing new piece and the whole identifiability story: it must be validated to (a) reproduce the
  global C8 clock deletion when applied at layer 0 (positive control), and (b) leave all non-L2MLP reads
  bit-identical when applied at the L2MLP edge (the edge-restriction anchor, M-edge below). Built by
  composing `run_carrier_surgery.py::make` with the path-patch hooks in `run_path_decomposition.py`.

**EXCLUDED (R14):** no ±1/±2 head-roll or attention phase-shift assay (stride-3 geometry manufactures
effects). **EXCLUDED (B4-1):** no L3H1-output → L2MLP-input patch (backward edge, forced identity).

---

## 3. MEMORIALS (on the measurement path; each CAN fail)

| # | Memorial | Passes iff | Note |
|---|---|---|---|
| M-c8 | global carrier deletion reproduces C8 | applied at layer 0, `make(BAND)` collapses coupling `C(D)/C(I) ≤ 0.05`; comb amplitude → ~3.4% | positive control for the deletion op (failable) |
| M-edge | **edge restriction is real, not global** | carrier deleted at the L2MLP input edge leaves **every non-L2MLP** residual read bit-identical to intact (max abs err <1e-6); only L2MLP's input differs | **the decisive anchor** — if this fails B4 is just C8 re-measured (failable) |
| M-recon | L2MLP write restore is exact | `_run_whole_restore` reproduces native L2MLP margin under P0/carrier-I to <1e-4 (HCA19 identity) | plumbing bridge, failable |
| M-Bpos | the HCA20 `B` effect reproduces | carrier-I `B_I` gene-CI-low > SESOI and matches HCA20 sign/scale (donor ≈+0.34) | reproduces the validated whole effect; failable |
| M-energy | energy-matched control inert | E-band deletion at the L2MLP edge does not collapse `B` (`B(E)/B(I) ≥ 0.90`), per-band comb-read ≥95% | E distribution; failable |
| M-health | MLM health (ALL interpreted arms) | every interpreted arm keeps ≥0.5 MLM headroom over chance (R13); below-floor arm dropped | health gate |
| M-forced | forced-identity lint | assay contains **no** layer-3→layer-2 injection; no O=A·V 2×2 `Δ_bind`; no full-restore treated as evidence | construction lint (B4-1/R23/R21) |
| M-det | determinism | PYTHONHASHSEED pinned; `sorted(set())`; cross-seed cohort identity | reproducibility (IC1 A2) |

`M-edge` is the memorial that can most easily fail and is the one on which the entire identifiability of
B4 rests: without a verified edge restriction, `ΔΔ_bridge` is not a bridge estimand at all — it is the
global C8 carrier effect re-read at L2MLP, which proves nothing new.

---

## 4. CALIBRATION (sensitivity + specificity; hard preflight gates)

**Planted-positive (each travels the real channel):**
- **PP-c8 (deletion sensitivity):** global `make(BAND)` → coupling collapse (M-c8) — the carrier channel
  is live.
- **PP-B (write sensitivity):** the HCA19/HCA20 `B` planted positive — L2MLP partner-scrub moves the
  margin by the validated amount under carrier-I (M-Bpos) — the `B` readout is calibrated.
- **PP-bridge (interaction sensitivity):** an **injected** carrier-gated write — construct a synthetic
  arm where L2MLP's partner-conditioned write is *forced* to depend on the carrier (e.g. multiply the
  partner-scrub source by the carrier-deleted mask at the L2MLP edge). `ΔΔ_bridge` must recover this
  planted interaction; if it cannot, B4.a's null is uninterpretable (insensitive vs no-bridge). **Piloted
  at `--smoke`; a dead PP-bridge blocks freeze** (the P3.4 lesson — a planted positive that does not
  travel a real channel makes equivalence meaningless).

**Null / specificity (must not fire above target FPR):**
- **N-energy:** E-band deletion at the L2MLP edge does not abolish `B` (M-energy; add-one p over the E
  distribution, not a single band — R12).
- **N-decoy:** the decoy-MLP path patch `ΔΔ_decoy` does not reach the bridge floor (B4.c).
- **N-permute:** permute partner-scrub↔intact labels within gene → `ΔΔ_bridge` ≈ 0 (specificity of the
  interaction).

**Confound-specificity (the identifiability fix):** the generic-damage counterfactual is the **E-band
distribution spanning importance** plus the **decoy MLP**; `ΔΔ_bridge` counts as carrier-specific
L2MLP-gating only if it **exceeds both** (add-one p vs E, CI-low > SESOI vs decoy), never merely a junk
band or generic position read.

---

## 5. IDENTIFIABILITY

- **B4-1 (founding):** the serial L3H1→L2MLP edge is backward/forced (see header). The identifiable
  target is the shared-upstream **carrier → L2MLP-input** edge; the reachable claim is a carrier-
  conditioned mediation on L2MLP's write, **never** a serial wire, binding, or additive fraction.
- **Edge-vs-global confound (M-edge).** The carrier enters every layer's residual by construction, so a
  *global* deletion changes L2MLP's input trivially and is just C8. The whole bridge claim depends on the
  **edge restriction** being real: only the L2MLP input edge is perturbed, all other reads bit-identical
  (M-edge <1e-6). If M-edge fails → B4 is **non-identifiable** (it cannot separate a bridge from the
  global carrier effect) and is reported as such, not forced.
- **Position-read vs partner-gating (the specificity core).** L2MLP reads position regardless; a generic
  position-dependence is *not* the claim. The estimand is the **carrier × partner interaction**
  (`ΔΔ_bridge`, a second difference) above the **energy-matched** carrier control (B4.b) and a
  **decoy-MLP** control (B4.c). Only a `ΔΔ_bridge` that clears both is carrier-specific partner-gating.
  A `ΔΔ_bridge` indistinguishable from E or decoy → generic disruption → Endpoint B.
- **R23 (no binding).** No O=A·V 2×2 interaction is computed; `ΔΔ_bridge` is a carrier×partner second
  difference on a *single* MLP write, interpreted as conditional mediation only (M-forced lint).
- **R15 (position already in the residual).** The carrier is a positional coordinate that is already
  inside the residual stream L2MLP reads; "carrier gates the partner write" is a statement about a shared
  positional input, not about the carrier *being* the payload.
- **New frozen cross-estimand freeze required (HCA20/A2.5).** HCA20 explicitly requires "a new frozen
  cross-estimand intervention"; the validated L2MLP `B` cohort/packets (a3_v2) live on **Google Drive**,
  not in-repo. B4 needs its **own** frozen Atlas cohort + gene-disjoint split (discovery/confirm), not a
  carve-out of the existing validated pool — a freeze that does not yet exist.
- **Held-out ≠ sampling stability (R22).** Any selected window/threshold/decoy set is fixed on a
  discovery split and confirmed on a **disjoint, untouched** gene split; gene-clustered CIs; interchange
  reuse (a gene as both partner-intact and partner-scrub source) requires **two-way dyadic** resampling;
  effect-existence via CI-excludes-SESOI, never point-estimate sign.

---

## 6. DECISION RULE (frozen before data; result → conclusion)

**SESOI:** SESOI_bridge = **0.05 nat** (the atlas material floor used throughout HCA19/HCA20 and the P3
SESOI_C; ~5–10% of the C8 coupling effect). A NULL/equivalence call requires **CI half-width < SESOI**
(else "inconclusive"). B4.a is run **one-sided positive** first (the name-changing, feasible direction);
equivalence is reported only if powered (§7).

**Families, correctly composed:** (i) **positive** family — one-sided `H0: ΔΔ_bridge ≤ SESOI`, max-T
gene-bootstrap adjusted lower bound > SESOI (B4.a, B4.b add-one p vs E, B4.c decoy separation share
genes ⇒ max-T valid); (ii) **equivalence** family — TOST at family-adjusted α, adjusted CI ⊂ ±SESOI.
Every "carrier D vs E" and "candidate vs decoy" test is the **paired within-target difference** CI, not
non-overlap of two arm CIs.

**Endpoints:**
- **A (bridge material):** *"L2MLP's partner-conditioned payload write is causally gated by the period-3
  positional carrier in its layer-2 input: removing the carrier at the L2MLP input edge (and nowhere
  else) reduces the partner-conditioned `B` effect by ΔΔ_bridge > SESOI, above the energy-matched carrier
  distribution and a decoy-MLP control, health-gated."* — the address × payload story (C23) shares a
  positional coordinate at Atlas scale. **No serial-wire, binding, or additive-fraction claim.**
- **B (unresolved / carrier-independent / non-identifiable):** reached on (a) `ΔΔ_bridge` equivalence
  inside ±SESOI (carrier-independent write — bridge refuted), (b) non-specific `ΔΔ_bridge`
  (indistinguishable from E or decoy — generic disruption), (c) **M-edge failure** (edge restriction not
  achieved → cannot separate from global C8 → non-identifiable), or (d) underpowered CIs.

Mixed/partial patterns, any inconclusive family member, or M-edge failure → **Endpoint B**. A serial
L3H1→L2MLP wire, binding, and additive fractions are **non-identifiable (B4-1/R23) → never claimed.**

---

## 7. POWER / MDE & HELD-OUT

- **The power wall (the reason this is likely DEFER).** `ΔΔ_bridge` is a **second difference** (carrier ×
  partner). The MPS anchor (verified, `results/phase-lattice/carrier_surgery.json`): C8 coding ΔL
  gene-clustered CI half-width is **0.0198 nats at n=384** for a *raw mean*; a second-difference contrast
  inflates variance ~2–4×, so a **powered equivalence** (CI-HW < SESOI 0.05) on B4.a needs on the order
  of **hundreds to >1000 gene-clustered targets** — the same wall that deferred P3.4. The validated
  L2MLP `B` cohort (donor n=2,013/1,112 genes) *would* clear it, **but it lives on Drive** and a B4
  freeze needs its own disjoint split. **Every NULL/Endpoint-B call must be a powered equivalence, not a
  wide CI (IC1/R22).** A discovery-pilot MDE gate must pre-declare B4.a inconclusive **before** spending
  compute if the projected CI-HW > SESOI at the achievable n.
- **Feasible direction:** the **one-sided positive** search (B4.a CI-low > SESOI) can fire locally if the
  bridge is large; the equivalence/null direction is the part that defers.
- **Held-out:** discovery split (a subset of val genes) fixes SESOI, decoy pool, window, health floor;
  confirmatory on a **disjoint, untouched** val-gene split; **untouched-gene confirmation** after any
  selection. Sealed test (chr8/9) **not opened.** Deterministic cohort; gene-clustered CIs; dyadic for
  interchange reuse.

## 8. GATES
- **Global PASS = AND(** M-c8, M-edge, M-recon, M-Bpos, M-energy, M-health, M-forced, M-det, PP-c8, PP-B,
  PP-bridge, N-energy, N-decoy, N-permute, confound-specificity **).** Run STOPS on failure; below-floor
  arms (M-health) dropped; **M-edge failure demotes B4 to non-identifiable (Endpoint B), not a null.**
- **`--smoke`:** the net-new edge-restricted carrier hook (M-edge <1e-6 off-edge; M-c8 at layer 0),
  the L2MLP restore identity (M-recon), the PP-bridge planted interaction, and determinism (M-det) on a
  tiny cohort, **val sealed** — the "pilot the edge-patch plumbing and prove PP-bridge travels a real
  channel" step; must pass before any freeze.

## 9. SCOPE
- **Reaches (if run and PASS):** OV/mediation + behaviour level for whether the period-3 carrier is a
  **shared positional input** gating L2MLP's partner-conditioned write. One rung above the atlas's
  component-level `B` relation (HCA20), still below biology.
- **Does NOT establish:** a **serial L3H1 → L2MLP wire** (B4-1, backward edge, forced); **binding**
  (R23); an **additive path fraction** or a **unique serial circuit** (HCA20 boundary); a **biological
  function label**; anything about the A-estimand population QK/V (non-identifiable at population scale,
  HCA20/A2.3).
- **Provenance:** checkpoint sha `2ad91428…0149d`; BAND=`range(339,346)`; seeds + PYTHONHASHSEED to the
  manifest; results write a manifest beside output; sealed chr8/9 untouched.
- **Amendments:** pre-run only, logged `Amendment (date, pre-run, no data seen)`; once the primary is
  inspected a fix is *outcome-aware verification*, not confirmatory (IC1 A2).

---

## PROVISIONAL DISPOSITION (pre-red-team; author's honest read) — DEFER / ride with B3

- **Why not immediately runnable:** (1) **B4-1** — the branch's literal name ("L3H1 → L2MLP") is an
  architecture-backward, forced edge; only the reframed **carrier → L2MLP-input** path patch is
  identifiable, and it needs a verified edge restriction (M-edge) that is net-new plumbing. (2) HCA20/
  A2.5 **explicitly require a new frozen cross-estimand freeze**; the validated L2MLP `B` cohort
  (a3_v2 shards, ~4,019 targets, DAG bundle) is on **Google Drive**, so a powered, HCA20-scale run is a
  **Colab/CUDA** job on a new Atlas freeze — not local MPS. (3) The bridge estimand is a **second
  difference**; a powered equivalence is unreachable at MPS budget (the P3.4 wall), so only a one-sided
  positive search is locally feasible.
- **A genuinely local narrow version exists but is limited:** a one-sided positive `ΔΔ_bridge` search on
  the in-repo validation-side cohort JSONL, using local carrier + L2MLP-restore machinery, could fire if
  the bridge is large — but it cannot confirm the carrier-independent (equivalence) reading and still
  needs its own disjoint discovery/confirm freeze. This is the "**rides with B3**" option: fold the
  carrier-conditioned L2MLP check into the B3 carrier/address thread rather than standing B4 up as a
  separate Atlas-headline study.
- **Recommendation:** **DEFER** B4 (or fold into B3). Stand it up as a frozen, Colab-scale study **only
  if the Head–Context Atlas becomes a headline deliverable**; otherwise the HCA20 `B` write is already a
  supported component relation and the address × payload sharing is adequately covered by the donor-
  circuit C23/C34/C35 carrier account. The concrete blocker to a freeze is the pair {verified edge-
  restricted carrier hook + PP-bridge that travels a real channel} **and** a new gene-disjoint Atlas
  cohort sized past the second-difference power wall.

---

## Closure disposition & must-fixes (2026-07-22 design red-team)

**Disposition: DEFER (survivor=NO).** Red-team synthesis: [`REDTEAM_SYNTHESIS_B4.md`](REDTEAM_SYNTHESIS_B4.md) (same
directory; 5 independent adversarial lenses, run before any confirmatory compute; formalize-audit-verify steps 1–2).
The draft's own instinct (DEFER / ride with B3) is correct, but the LEAD blocker is **architectural
non-identifiability of a direct carrier→L2MLP edge on a frozen post-LN checkpoint** (verified across three convergent
lenses, like P3.4), not the second-difference power wall or the Drive-only cohort. The only implementable carrier
deletion is global at the P-weight source; post-LN renormalization leaves no additive carrier component to subtract at
a mid-layer edge, so ΔΔ_bridge bundles the direct removal with comb-collapse routing through L2H15 — the "edge
distinct from global C8" does not exist. No amount of Colab/CUDA compute, larger n, or new Atlas freeze can fix this.

**survivor = NO — not executed this session.** The only locally-runnable identifiable variant is a coarse TOTAL
upstream carrier-dependence of L2MLP's write (a localized-C8 read, belonging with B3's carrier/address thread), not
the bridge study this prereg names; a real cross-estimand bridge needs a training-time / subspace-freeze design if the
Atlas becomes a headline. This session is documentation finalization only. **Operational note:** the SpliceBERT.1024nt
weights and the Atlas data are absent from this worktree, so any model run must be launched from the main checkout.

**Must-fixes (verbatim, from the synthesis):**

1. Promote the identifiability non-identifiability (direct carrier->L2MLP edge undefined on a frozen post-LN checkpoint) to the LEAD blocker in the disposition; demote power-wall + Drive-cohort to secondary.
2. Restate the honest reachable estimand as the TOTAL upstream carrier effect on L2MLP's write (a localized-C8 read) and withdraw the 'genuinely local narrow bridge version exists' escape hatch; the only implementable carrier deletion is global at the P-weight source.
3. Strip 'shared with / same address L3H1 reads' from Endpoint A (R20 identity-vs-geometry): the assay never touches L3H1, so a shared Fourier band is not a demonstrated shared/used address.
4. Reclassify M-edge as a plumbing/locality gate, NOT the identifiability anchor; no memorial in the design can test the direct-vs-mediated carrier confound (post-LN mixing makes an on-edge purity memorial inconstructible).
5. Correct Section 7: drop the borrowed P3.4 '2-4x variance / same wall' number (ΔΔ_bridge is a PAIRED second difference); label the blocker cohort-location, not a fundamental power floor.
6. Remove the dyadic-bootstrap requirement (B4.a-c are within-target paired contrasts with no interchange -> one-way gene gboot is exact); note the in-repo packets cohort supports local gene-clustering.
7. Annotate M-recon as a forced full-restore identity (R21, plumbing-only); pin B4.b control-band N>=39 (current carrier_surgery.json ships only 21 -> under-powered).
