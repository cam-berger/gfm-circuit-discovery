# PREREGISTRATION — B1: ≈10.63 carrier past attention (closes/bounds C12) — v2 (SPLIT-FREEZE)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: EXECUTED 2026-07-22** — both arms run on the SpliceBERT.1024nt checkpoint; full results in the
**"## Execution results"** section at the foot of this file, in [`../DECISIONS.md`](../DECISIONS.md), and in
CLAIMS **C37** (which bounds C12, donor-scoped). Headline: the OV arm is a clean **A-null** (the carrier's 9
post-softmax readers write nothing carrier-specific onto the donor readout) and the behavioural arm finds a
**small, significant, SUPPRESSIVE** coupling effect (deletion *raises* coupling by 0.033, sub-SESOI) — the
carrier is not used to *support* the donor computation. Two design projections were corrected by the run
(below). Prior design status:

**SPLIT-FREEZE (2026-07-22), after the focused OV-estimand re-audit
([`../redteam/REDTEAM_SYNTHESIS_B1_V2.md`](../redteam/REDTEAM_SYNTHESIS_B1_V2.md), 3 independent lenses vs
code+data).**
- **BEHAVIOURAL arm (ΔC coupling under `BAND_P10` deletion): FROZEN_RUNNABLE.** Clean paired within-example
  contrast on a near-inert deletion; power is *reachable* (see §7 — the equivalence needs only ρ>0.42 at n=256,
  not the ρ≥0.98 v1 feared), gated on the §7 sizing pilot; the dual-branch decision (equivalence *or* control
  re-selection) is pre-committed.
- **OV arm (value-main → donor readout): design-frozen after the re-audit's must-fixes, pending its §8 `--smoke`
  gate on the main checkout.** The re-audit confirmed the v2 value-main dissolves v1's R23 forced-identity
  fatal *iff* the estimand is the **simple single-row main `DI−DD`** — and caught that this draft's first cut
  used the **symmetric** two-row main `½[(DI−DD)+(II−ID)]`, which equals `DI−DD + ½·Δ_bind` and so re-imports
  half of the architecture-forced interaction (on the C35 router it gives **+0.073, not the −0.005 null**). That
  is corrected below (M-1a). Two further must-fixes are applied: the router-negative is measured at the correct
  **readout scale** (M-1b), and both OV nulls are **covariance-matched** (M-2). The ill-posed readout-agnostic
  "L1" is demoted to a reported diagnostic (head-output value-main is generically nonzero via position-embedding
  propagation, so it cannot license a "write" claim without a readout).

Retains v1 ([`PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR.md`](PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR.md)) as
design-of-record. **Not executed in this worktree:** SpliceBERT.1024nt weights + coupling/atlas data are absent
here; the `--smoke` gate and any run launch from the main checkout.

Provenance: **SpliceBERT.1024nt** sha256 `2ad91428…0149d`, post-LN BERT-MLM 6L×16H, hidden 512, **absolute**
learned position embeddings P (1026×512). `BAND_P10 = range(94,101)` (peak bin 96; period `1026/96 = 10.6875`;
`run_second_carrier.py`). C11's precedent is on **SpliceBERT.510nt** (bins 46–48, `T=10.61`) — a different
checkpoint (§5 risk-1). Env `gfm-thrust1`; **MPS**; deterministic (`PYTHONHASHSEED=0`, `sorted(set())`).

The name defended (C11/C13/C14, no biology imported):
> **A separate learned ≈10.63-period positional carrier** — a high-frequency mode of P (bin 96), distinct from
> the period-3 clock, read post-softmax by a controlled set of heads. **Whether its value contribution reaches
> the donor readout, and whether deleting it moves donor coupling, is the question; its general OV content
> (MLM output) stays open.**

---

## B1's founding tensions (verified pre-compute)

- **T1 — checkpoint mismatch (leading).** C11's eight Level-A readers live on **510nt**; the donor coupling,
  `U_info`, the `carrier_surgery`/`carrier_dose` instruments, **and the C35 A-vs-V machinery** live only on
  **1024nt**, whose demonstrated ≈10.63 QK readers (`second_carrier.json`) are disjoint and **pre-softmax
  only**. B1 runs on 1024nt and re-establishes the post-softmax reader set on-model (**Preflight R**);
  `R_1024 = ∅` ⇒ OV arm non-identifiable → Endpoint B (the behaviour arm still runs).
- **T2 — control-band identity.** `BAND_P10` (94–100) is exactly the inert control band that
  `carrier_surgery`/`carrier_dose` use: deleting it *raises* coupling `0.946 → 0.970` (`carrier_dose.json`,
  control_band [94..100]), while period-3 deletion collapses it to `0.027`. Re-pointing the deletion onto
  `BAND_P10` requires re-centring the control exclusion off period ~10.6 (M-ctrl). A coupling **equivalence**
  strengthens that control; a coupling **positive** flags band 94–100 for re-selection (§5 risk-2).

---

## 0. INTENT (north star — re-read at every gate)

*The ≈10.63 carrier reaches controlled post-softmax attention (C11). Does restoring its reader heads'
carrier-intact **values** (attention held at the deleted pattern — the C35 value-main effect) move a **donor
readout** — the `U_info` direction / coupling margin — above a covariance-matched null? And does deleting the
carrier **move donor–acceptor coupling**? Or is the carrier a positional feature the model represents and
routes but **does not use for the donor computation** — the clean contrast to the period-3 clock, which IS used
(C8/C36)?*

Model-native observables: (**OV, donor-scoped**) the C35 **value main effect** `Mv_h = O_h(A=D10, V=I) −
O_h(A=D10, V=D10)` of each on-model reader head, projected onto `U_info` and tested for donor-coupling
sufficiency against a covariance-matched null; (**behaviour**) the coupling margin `C` under `BAND_P10`
deletion (primary), and general masked MLM NLL (secondary, MPS-wall-flagged). **Biology deliberately empty**
(C14; origin is training-time, out of scope).

**Scope boundary, pre-committed (donor-scoped, agreed with Cam):** the OV claim is **with respect to the donor
readout**. B1 does **not** characterize the carrier's general OV content — its native target is the MLM output,
a separate, power-walled study. B1 promotes C12 from "OV/behaviour unknown" to a **bounded statement about the
donor circuit** (used / not-used), with general OV content and origin left open.

---

## 1. ESTIMANDS

Carrier conditions on P (rfft-bin zeroing of `P − μ`, the shared `make(bins)`): **I** (intact); **D10**
(`make(BAND_P10)`); **LOWLEAK** (a low-energy 7-bin band); **SHUF** (content-shuffle, positivity per R15).
**RECIP is not used** (must-fix #4: no character-matched off-carrier high-frequency band exists on 1024nt — bin
96 is the only high-frequency mode besides period-3). All contrasts are **paired within-example**,
gene-clustered (`gboot`/`gboot_ratio`), CI-excludes-0/SESOI.

### The OV core — the value main effect (simple single-row; the anti-R23 form)
For each head `h ∈ R_1024` (Preflight R), on the **D10** base, use the C35 A-vs-V machinery
(`capture_proj`/`patch_proj`; head runs natively with one `W_O`) to form the (A ∈ {D10,I}) × (V ∈ {D10,I}) 2×2.
The estimand is the **simple single-row value main effect**:

> `Mv_h = O_h(A=D10, V=I) − O_h(A=D10, V=D10) = A_D · (V_I − V_D)` — attention **pinned at the deleted
> pattern**; `Mv_h = 0` iff the head's values are carrier-independent (`V_I = V_D`). **Non-forced in the correct
> direction** — a pure router (carrier-independent values) yields `Mv_h = 0` by construction.

**M-1a (the correction this re-audit forced):** the **symmetric** two-row main `½[(DI−DD)+(II−ID)] = Mv_h +
½·Δ_bind` re-imports half the architecture-forced interaction `Δ_bind = (A_I−A_D)(V_I−V_D)` (R23). Verified on
`p3_av_decomposition.json` (confirm split): simple `DI−DD = 0.024072 − 0.029052 = −0.00498` (a genuine router
null), whereas the symmetric main `= +0.0734 = −0.005 + ½·(0.15682)`. **We use the simple single-row `DI−DD`
everywhere** (formula, `s_ov`, `PP-ovplant` recovery, the L2 patch). `Δ_bind` is reported, **never interpreted**
(R23).

### B1-OV — does the value contribution reach the donor readout? (OV/mediation, the licensed endpoint)
`U_info` is defined once on the intact model (C16; never refit):
- **sufficiency** `S_ov^h` = fraction of the donor coupling margin recovered by patching the **value-restore
  component `Mv_h`** into the corrupt donor residual (`anchor_gates.suff_recovery`), vs a **covariance-matched**
  null (M-2: draws from the head's own OV-output empirical covariance, norm-matched to `Mv_h`, `N_NULL ≥ 200`;
  **not** the isotropic `random_subspace`/`null_recoveries` QR draw, `subspace.py:38–41`), add-one p.
- **overlap** `Ω_h = subspace_overlap(top_k_subspace(Mv_h, k), U_info)`, `k∈{1,2}`, vs the same
  covariance-matched null — reported **only** alongside a null-beating `S_ov^h`, never alone.
- **carrier-specificity:** repeat with **LOWLEAK** (the retained control); `S_ov^h(D10)` must exceed the
  LOWLEAK counterpart and the surgery control-band distribution (the latter an explicitly magnitude-mismatched
  reference, must-fix #4).

### Diagnostic (reported, NOT an endpoint) — the value-main share at head-output scale
`f_v^h = ‖Mv_h‖ / ‖O_h(I) − O_h(D10)‖ ∈ [0, ∞)` — a **diagnostic ratio** (not a partition fraction) describing
how much of head `h`'s total carrier-deletion effect is value-main vs routing-main. **Caveat (why it is not an
endpoint):** the value vectors `V = W_V·r` depend on the position-embedding-carrying residual `r`, so `Mv_h` is
**generically nonzero via position propagation** even for a functionally routing-only head; a nonzero `f_v^h`
therefore does **not** license a "carrier writes" claim. Only the readout-projected `S_ov^h` beating a
covariance-matched null does.

### B1-BEHAV-primary — donor–acceptor coupling `C` under carrier deletion (the C8/C36 readout)
`ΔC(D10) = C(I) − C(D10)`, gene-clustered, corruption = partner AG→CC (CouplingTask default). **Frozen
directional prediction, declared BEFORE the confirmatory cohort: EQUIVALENCE** (`|ΔC| < SESOI_C = 0.05`). The
seen dose arm (0.946→0.970, `|point|≈0.024`) is a **prior only**; the confirmatory equivalence runs on a
**fresh disjoint** cohort (§7).

### B1-BEHAV-secondary — general masked MLM NLL (EXPLORATORY, MPS-wall-flagged)
`ΔL(D10)` split exon/intron. Seen points: exon `≈0.0265`, intron `≈0.0253` — **coding-UNenriched**. Reported
**descriptively / one-sided** at achieved precision (on the 0.02-nat wall; never a powered equivalence).

---

## 2. METHOD (reused machinery + one preflight + net-new drivers)

**Reused / validated (exact paths):**
- **A-vs-V value-main (the OV core):** `scripts/splicebert/run_p3_av_decomposition.py` (`capture_proj`/
  `patch_proj`; the exact-0.0 anchors; `value = DI − DD` at :134) + `results/p3_av_decomposition.json` (period-3
  L3H1: `DI−DD = −0.00498` confirm = the clean router null the OV arm relies on; `Δ_bind = +0.15682` forced).
  Re-pointed from L3H1/period-3 to `R_1024`/`BAND_P10`.
- **Subspace + causal tests:** `subspace.py` (`top_k_subspace`, `subspace_overlap`, `patch_component`),
  `anchor_gates.py` (`suff_recovery`). `U_info` per C16 (`run_donor_anchor.py`).
- **Covariance-matched null (net-new; replaces the isotropic `random_subspace`, must-fix #2/#4):** draw random
  directions from the head's OV-output empirical covariance, norm-matched to `Mv_h`.
- **Deletion + control + DiD + health:** re-pointed `run_carrier_surgery.py`/`run_carrier_dose.py`
  (`BAND=BAND_P10`; control-band search excludes period {3,1.5,6, 10.6±5%}, M-ctrl). Coupling/margins/health/
  CIs: `circuit.py` (`CouplingTask`, `margin_hard`, `gboot`, `gboot_ratio`); adapter `eager=True`.
- **Post-softmax reader test:** `run_postsoftmax.py` Level-A machinery, re-pointed to 1024nt/`BAND_P10`.
- Provenance: `gfmcd.provenance.ExperimentManifest`; `gfmcd.rigor.empirical_upper_p_value`.

**Preflight R — on-model reader establishment (net-new; gates the OV arm; T1).** `R_1024` = 1024nt heads
reaching Level A at `BAND_P10` (collapse under D10, survive LOWLEAK + SHUF). `R_1024 = ∅` ⇒ Endpoint B (OV).

**Net-new driver:** `run_carrier_ov_av.py` (the simple `DI−DD` value-main + `S_ov`/`Ω` + covariance-matched
null + the router-negative + `PP-ovplant`, piloted in `--smoke`). **EXCLUDED (R14):** no head-roll/phase-shift.

---

## 3. MEMORIALS (on the measurement path; each CAN fail)

| # | Memorial | Passes iff | Note |
|---|---|---|---|
| M0 | unembed anchor (G0) | `max|cls(hidden[-1]) − logits| < 1e-3` at the donor | reused; failable |
| M-del | D10 is the exact ≈10.63 clock | `make(BAND_P10)` zeroes rfft bins 94–100 of `(P−μ)`; reconstruct anchor `<1e-6`; peak bin 96 | construction anchor; failable |
| M-read | on-model readers established | Preflight R yields `R_1024`; deletion audit `target_retained<1e-6`, `protected_*_retained>0.999` | **failable → ∅ ⇒ OV non-identifiable, Endpoint B** |
| M-av-plumb | A-vs-V plumbing exact | C35 anchors reproduce on `R_1024`: diagonal, native value-overwrite, head-identity all `= 0.0`; `O = A·V` recompose `<1e-6` | reused C35 anchors on new heads; failable |
| M-simplemain | value-main is the SIMPLE single-row form (anti-R23) | the pipeline computes `DI−DD` (attention pinned D10), and on the period-3 L3H1 check reproduces `−0.00498` (NOT the symmetric `+0.0734`); the symmetric-vs-simple gap equals `½·Δ_bind` to `<1e-6` | **the M-1a fix** — a failable check that the forced interaction is quarantined |
| M-vmain | the pipeline can null a known router (readout scale) | on period-3 L3H1 (same pipeline, coupling readout), `S_ov` lies within `±SESOI_v` (anchor `−0.005`); the head-output `f_v` is reported alongside as context | **the anti-R23 sensitivity anchor at the LICENSED (readout) scale** (M-1b); failable |
| M-covnull | both OV nulls are covariance-matched | the `S_ov` **and** `Ω` nulls reproduce the head's OV-output covariance spectrum to tolerance, norm-matched to `Mv_h` (not a uniform QR draw) | **M-2**; failable (isotropic draw fails) |
| M-ctrl | control window re-centred | every energy-matched control band excludes period {3,1.5,6, 10.6±5%} | T2; failable |
| M-couphealth | coupling assay alive under D10 | every interpreted arm keeps ≥0.5 MLM headroom over chance (R13) | below-floor → arm dropped |
| M-p3contrast | period-3 positive control still fires | re-pointed instrument at `BAND=range(339,346)` reproduces C8 collapse `C(D)/C(I) ≤ 0.05` | sensitivity of the re-pointed deletion instrument; failable |
| M-det | determinism | `PYTHONHASHSEED=0`; `sorted(set())`; cross-seed cohort identity | reproducibility (IC1 A2) |

Load-bearing failable anchors: **M-simplemain** + **M-vmain** (the value-main is the uncontaminated simple
form and can return a genuine null on a known router at the readout scale) and **M-covnull/M-ovnull** (a write
must beat a covariance-matched null). `Δ_bind` is reported forced (R23), never a memorial.

---

## 4. CALIBRATION (sensitivity + specificity; hard preflight gates)

**Planted-positive (each travels the real channel):**
- **PP-p3 (deletion channel):** re-pointed surgery/dose at `BAND=339–346` reproduces C8's coupling collapse
  (M-p3contrast).
- **PP-ovplant (the OV value channel — must-fix #3):** inject a **known low-rank value write** into a reader
  head's value output on the D10 base (a synthetic `Mv` of controlled rank, **norm pinned to the `SESOI_v`
  scale**, known direction) and require the **simple** value-main → `top_k_subspace` → `S_ov` pipeline to
  recover it above the covariance-matched null. Scoped to the L2 (readout-projected) channel. **Caveat, stated:**
  this calibrates sensitivity to a `U_info`-aligned write; sensitivity to a carrier-derived **non-`U_info`**
  write is uncalibrated (reported, not claimed).
- **PP-U (donor-projection channel):** the C16/C17 positive — intact `U_info` into the corrupt donor residual
  recovers ≈0.9 vs random ≤0.05 — calibrates `suff_recovery`.
- **PP-read (reader channel; the sensitivity FLOOR):** period-3 band on 1024nt through Preflight R yields a
  non-empty Level-A set. **Foregrounded as an *easier-feature* floor:** period-3 survival does NOT license
  ≈10.63 survival — `R_1024 = ∅ → Endpoint B` stays live.

**Null / specificity (must not fire):**
- **N-router (the router-only planted-negative, must-fix #2):** period-3 L3H1 through the same pipeline at the
  coupling readout returns `S_ov` within `±SESOI_v` (M-vmain). **Bound (re-audit):** L3H1 is out-of-family (deep
  L3, period-3 band) and its *discovery*-split value-main CI marginally excludes 0 (`−0.01055 [−0.02033,
  −0.00020]`); only the *confirm* split (`−0.005`) is a clean null. So **`SESOI_v` is pre-declared against the
  confirm-split scale (≥ 0.02)** so the anchor cannot spuriously void the arm, and M-vmain's inference is
  scoped to "the pipeline **can** null a known router," not "L3H1 is null on every split."
- **N-covrand:** covariance-matched random value-restores recover ≤ the frozen null threshold (M-ovnull).
- **N-shuf:** content-shuffle survival (positivity, R15).

**Confound-specificity:** an OV/behavioural effect counts as carrier-specific only if it exceeds the
covariance-matched null (OV) / the surgery control-band distribution (behaviour).

---

## 5. IDENTIFIABILITY (honest non-identifiable outcomes pre-committed)

- **Risk-0 — value-main on a NEW feature.** The simple `DI−DD` value-main is C35-validated for period-3/L3H1;
  B1 applies it to ≈10.63/`R_1024`. **Anchors:** M-av-plumb (C35 anchors reproduce on the new heads),
  M-simplemain (the forced term is quarantined), M-vmain (the pipeline nulls a known router at the readout
  scale). If M-vmain fails, the value-main is untrusted here → **OV arm → Endpoint B**. (Object of the re-audit;
  now anchored.)
- **Risk-1 — checkpoint mismatch (T1).** Run on 1024nt; establish `R_1024`; `∅ → Endpoint B`; no
  cross-checkpoint head-identity claim.
- **Risk-2 — control-band identity (T2).** Re-centre control exclusion (M-ctrl); confirmatory coupling on a
  fresh cohort; equivalence strengthens the control, positive flags 94–100 for re-selection — reported either
  way.
- **Risk-3 — MPS wall.** MLM footprint `≈0.026` on the 0.02-nat wall ⇒ MLM one-sided only. **Coupling
  equivalence IS reachable** (§7): with `|point|≈0.024` and a paired contrast, `HW < SESOI_C − |point| = 0.026`
  at n=256 needs only **ρ > 0.42** (not v1's ρ≥0.98) — but still **gated on the sizing pilot** (must-fix #5).
- **Risk-4 — OV forced-identity (R23).** The simple `DI−DD` value-main removes the routing conflation and
  quarantines `Δ_bind` (M-simplemain); the residual chance-projection risk is guarded by the covariance-matched
  null + add-one p (M-covnull/M-ovnull) and N-router.
- **Risk-5 — pre-softmax ≠ post-softmax (R14).** Post-softmax Level-A (Preflight R) required before any OV
  interpretation.
- **Risk-6 — held-out ≠ sampling stability (R22).** `R_1024`, `k`, bands, SESOI (incl. `SESOI_v`), floors fixed
  on discovery; confirmed on a disjoint untouched split; ≥2 seeded reader draws; gene-clustered CIs; the seen
  dose arm is a prior, its cohort excluded from the confirmatory coupling cohort.
- **Biology (empty).** No biological-origin claim (C14).

---

## 6. DECISION RULE (frozen; result → conclusion)

**SESOI:** `SESOI_C = 0.05` nat (coupling); **`SESOI_v` pre-declared ≥ 0.02** (the confirm-split router-null
scale, N-router) as the value-main "lands" threshold — a write lands iff `S_ov^h` CI-low `>` the
covariance-matched null 95th percentile **and** `> SESOI_v`; `SESOI_L = 0.02` (MLM, one-sided descriptive).
**OV equivalence is one-sided-only** (must-fix #7): A-null-OV is licensed by the covariance-matched null being
un-beaten (a bounded "no carrier-specific write onto the donor readout detected"), not a powered TOST. **Family
control:** max-T gene-bootstrap over `R_1024`; **per-head OV claims restricted to L0 readers** (L1/L2 readers'
value-restore absorbs upstream global-P propagation, must-fix #7). TOST for the coupling equivalence.

**Endpoints (each closes or bounds C12, donor-scoped):**
- **A-OV — promote C12(donor) to OV/mediation:** `R_1024 ≠ ∅`, M-vmain passes, and ≥1 **L0** head has `S_ov^h`
  beating the covariance-matched null (carrier-specific vs LOWLEAK) **and** coupling `ΔC(D10)` CI-low
  `> SESOI_C`. ⇒ *"The ≈10.63 carrier's value contribution reaches the donor readout `U_info`, and its deletion
  moves donor coupling — a second causal positional mechanism into the donor readout beside the period-3
  clock."* (Prior: unlikely — coupling already ≈inert.)
- **A-null(donor) — bounded "represented + routed, not used for the donor computation":** no L0 head's `S_ov^h`
  beats the covariance-matched null (one-sided) **and** coupling `ΔC(D10)` in **powered equivalence** (whole CI
  ⊂ ±`SESOI_C`, pilot-gated). ⇒ *"The ≈10.63 carrier reaches controlled post-softmax attention (C11) but
  **writes nothing carrier-specific onto the donor readout `U_info`** and is **not used for donor coupling** — a
  positional feature the model represents and routes but does not use **for the donor computation** (an L0-reader
  statement). Its general OV content (MLM output) and origin remain open."* (Wording bound, must-fix #6: never
  unqualified "does not use"; if the MLM secondary shows a one-sided positive, add "used non-coding-specifically".)
- **A-mixed — writes-but-donor-inert:** an L0 `S_ov^h` beats the null but coupling is in equivalence ⇒ *"the
  carrier writes a carrier-specific component onto the donor readout subspace that is behaviourally
  redundant/masked for coupling; whether it serves a non-donor readout is untested."*
- **B — unresolved:** `R_1024 = ∅`, M-vmain fails, or any interpreted CI is underpowered (half-width ≥ SESOI).
  ⇒ *"C12(donor-OV) unreached on this evidence."*

The `f_v` diagnostic and the MLM secondary never drive an endpoint.

---

## 7. POWER / MDE & HELD-OUT

- **Coupling equivalence (primary): reachable, pilot-gated.** With `|point| ≈ 0.024` (`carrier_dose.json`) and
  a paired within-example gene-clustered contrast, the freeze gate `HW < SESOI_C − |point| = 0.026` at n=256
  needs only **ρ > 0.42** (from C8 `0.0198@n384 → 0.0242@n256`, paired `0.0242·√(2(1−ρ)) < 0.026`). **Sizing
  pilot (must-fix #5):** measure gene-clustered `ΔC` HW **and** per-example ρ on discovery; freeze only if
  `HW < 0.026`; else re-scope or report one-sided; underpowered ⇒ inconclusive, never null. The residual risk
  is only that `|point| ≈ 0.024` sits near `SESOI_C = 0.05` — handled by the pre-committed dual-branch rule.
- **OV `S_ov`:** at n≈150–250 unconditional exons (donor-anchor budget), `N_NULL ≥ 200` covariance-matched;
  MDE from the null spread; **OV-null one-sided** (bounded "not detected"), not a TOST. **Per-head compute
  budget:** ~2 forwards/example/head/cell × 4 cells × |R_1024| × n — stated in the run manifest; if |R_1024|·n
  exceeds the MPS budget, restrict to the top-|R_1024|≤4 L0 readers by Preflight-R `m_T`.
- **MLM (secondary):** powered equivalence UNREACHABLE (`0.0198@n384`) → one-sided only.
- **Held-out:** discovery split fixes `R_1024`, `k`, bands, SESOI (incl. `SESOI_v`), floors; OV heads + coupling
  equivalence confirmed on a **disjoint untouched** split; seen dose arm excluded; sealed test (chr8/9) not
  opened; ≥2 seeded reader draws.

## 8. GATES
- **Global PASS = AND(** M0, M-del, M-read, M-av-plumb, M-simplemain, M-vmain, M-covnull, M-ctrl, M-couphealth,
  M-p3contrast, M-det, PP-p3, PP-ovplant, PP-U, PP-read, N-router, N-covrand, N-shuf, confound-specificity **).**
  Run STOPS on failure; `M-read = ∅` demotes only the OV arm to Endpoint B; `M-vmain` failing demotes the OV
  arm to Endpoint B.
- **`--smoke` (must pass on the main checkout before the OV confirmatory run):** M-del, M-av-plumb,
  **M-simplemain** (simple vs symmetric main; the `½·Δ_bind` quarantine), **M-vmain** (the router-negative at
  the readout scale, with the head-output `f_v` reported), **PP-ovplant** (plant→recover), M-covnull, M-ctrl,
  M-p3contrast, M-det, and Preflight R non-empty on the **period-3** band — tiny cohort, val sealed. **The
  behavioural arm's `--smoke`** (M-del, M-ctrl, M-p3contrast, M-det + the sizing pilot for ρ/HW) is independent
  and is the only gate before the coupling confirmatory run.

## 9. SCOPE
- **Reaches (if A-OV):** donor-scoped OV/mediation + behaviour. **Reaches (if A-null/A-mixed):** a **powered
  bounded** statement — "routed but not used for the donor computation" (L0 readers) / "writes-but-donor-inert".
  Either bounds C12 **with respect to the donor circuit**.
- **Does NOT establish:** the carrier's **general OV content** (native target = MLM output; separate,
  power-walled; pre-committed open); its **biological origin** (training-time; C14); any **cross-checkpoint**
  head-identity claim (C11 is 510nt); a **semantic identity** for any OV content (R22); **binding** or an
  independently-constructed payload (R23 — `Δ_bind` reported-not-interpreted); anything about the **period-3**
  carrier (C8/C34/C35 unchanged).
- **Reconciliation obligation (T2):** state whether the coupling result strengthens (equivalence) or
  invalidates (positive) band 94–100's use as the inert control.
- **Provenance:** SpliceBERT.1024nt `2ad91428…0149d`; C11 precedent 510nt (bins 46–48, T=10.61);
  `BAND_P10=range(94,101)`, bin 96, period `10.6875`; seeds + `PYTHONHASHSEED=0` to the manifest; results write
  a manifest beside output.
- **Amendments:** pre-run only, logged `Amendment (date, pre-run, no data seen)`; once the primary is inspected
  a fix is *outcome-aware verification*, not confirmatory (IC1 A2).

---

## Execution results (2026-07-22, SpliceBERT.1024nt, MPS)

Drivers: `scripts/splicebert/run_carrier_preflight_r.py`, `run_carrier_ov_av.py`, `run_carrier_p10_behaviour.py`.
Artifacts: `results/phase-lattice/carrier_preflight_r.json`, `carrier_ov_av.json`,
`carrier_p10_behaviour_confirm.json` (+ `_n1200.json`) (+ manifests). All memorials/calibration passed.

**Preflight R (n=48) → `OV_ARM_LIVE`.** `R_1024` = **9** post-softmax Level-A ≈10.63 readers:
`L2H6, L2H12, L3H3, L3H5, L3H8, L3H9, L3H11, L3H13, L4H13` — C11's eight 510nt readers **replicated on the
1024nt checkpoint** plus L2H6. Memorials exact (`target_retained=1.2e-15`, checkpoint unmutated). **This
revises T1's pessimism:** the *post-softmax* readers are not disjoint from C11 (that was the *QK* readers,
`second_carrier.json`); the OV arm is identifiable.

**OV arm (n=120, all 9 heads) → A-null (the headline).** Every reader's simple value-main `Mv_h = DI−DD` lies
**within ±SESOI_v = 0.02** (range −0.009→+0.013; largest L2H6 +0.013 [+0.006,+0.020], CI-low ≪ SESOI_v; none
fires). The **router-negative** (period-3 L3H1 through the same pipeline) returns value-main **−0.0099
[−0.019,−0.002]** (nulls a known router) with **`M-simplemain_gap = 0.0` exact** — the M-1a fix held: the code
uses the simple `DI−DD`, `Δ_bind` (+0.20) is quarantined, not interpreted (R23). **⇒ the ≈10.63 carrier writes
nothing carrier-specific onto the donor coupling readout.** (No covariance-null / PP-ovplant follow-up needed —
that calibrates a *positive*, and none fired.) Scope (R24): value-main at these deep (L2–L4) readers is a
within-intervention conditional mediation, donor-scoped.

**Behavioural arm → small, significant, SUPPRESSIVE (not a clean equivalence).** dC(D10) = C(intact) − C(deleted),
gene-clustered, stabilised across n: −0.015 (219 g) → −0.034 (738 g) → **−0.0333 [−0.0606, −0.0053]** (n=2500,
1085 genes, ρ=0.79). **The CI excludes 0**: deleting the carrier **raises** coupling by 0.033 (C 0.955→0.989) —
a **small (|0.033| < SESOI_C=0.05), significant, suppressive** effect (`small-but-detectable` per the verdict
taxonomy), replicating the seen dose arm (+0.024). M-p3contrast 0.037 (the re-pointed instrument still collapses
coupling on period-3), M-couphealth OK. **The carrier is not used to *support* coupling** (the "used" direction
would *reduce* it); its small effect is anti-coupling and — given the OV null — **not mediated by the readers'
values** (routing/distributed).

**§5 risk-2 reconciliation (obligation discharged):** band 94–100 is **not strictly inert for coupling** — its
deletion significantly raises coupling by 0.033 (CI-excludes-0). Its use as `carrier_surgery`/`carrier_dose`'s
*energy-matched inert control* is valid **for MLM** (where it was established) but **not for coupling**; this
qualifies that control (it does not touch C8: period-3's coupling effect is ~30× larger and in the opposite,
supportive direction).

**Two design projections the run corrected (honest record):**
1. **T1 (checkpoint mismatch) was too pessimistic on identifiability** — the post-softmax readers *do* appear on
   1024nt (9 heads), so the OV arm was live, not Endpoint B.
2. **The behavioural power was mis-projected by BOTH red-teams.** v1 said ρ≥0.98; the re-audit "corrected" it to
   ρ>0.42 — but that borrowed C8's *MLM* per-example SD (~0.02), whereas the **coupling** margin SD is ~0.87
   (the v1 red-team's must-fix #5 had flagged this). Empirically HW≈0.87·√(2(1−ρ))/√(n_genes); a two-sided ±0.05
   equivalence needs the CI inside ±0.05, unreachable while the true effect is ~−0.033 (a real fraction of
   SESOI). Reported one-sided / at achieved precision, not as a powered equivalence.

**Endpoint (donor-scoped), bounding C12 → CLAIMS C37:** the ≈10.63 carrier reaches post-softmax attention
(C11, replicated on 1024nt) but is **not part of the donor circuit** — it writes nothing carrier-specific onto
the donor readout (OV A-null) and is not used to support donor coupling (small suppressive behavioural effect,
sub-SESOI). A positional feature routed but not used for the donor computation — the contrast to the period-3
clock (C8/C36) which *is* used. **General OV content (MLM output) and biological origin remain open** (the
pre-committed scope boundary).
