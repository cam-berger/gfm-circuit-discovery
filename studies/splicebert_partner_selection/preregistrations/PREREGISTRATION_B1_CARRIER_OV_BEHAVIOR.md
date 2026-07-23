# PREREGISTRATION — B1: ≈10.63 carrier past attention (closes C12) (v1)

**Status: NOT FROZEN — the 5-lens design red-team (2026-07-22) adjudicated this draft FIXABLE_THEN_RUNNABLE (survivor=NO).**
See [`../redteam/REDTEAM_SYNTHESIS_B1.md`](../redteam/REDTEAM_SYNTHESIS_B1.md). Behavioural arm (ΔC coupling from
re-pointing the deletion instrument to the ≈10.63 plane) is clean and identifiable, but the OV-write arm is
NON-IDENTIFIABLE as written — dO_h = A_I·V_I − A_D·V_D conflates the carrier's already-established QK/routing role
(C11/C35) with the OV column (R23-class ladder column-jump). Fixable on the frozen checkpoint via the P3.3 A-vs-V
decomposition; requires a coupling-equivalence power pilot (ΔC is a second difference needing per-example ρ≥0.98 at
n=256) and a checkpoint-alignment fix (C11's 8 post-softmax readers are on SpliceBERT.510nt; U_info/coupling are on
1024nt). Realistic deliverable = a behaviour-primary bounded null + an OV positive-only search, not the OV-write
headline the draft leads with. This v1 is retained as the **design of record**; it must **NOT be run** as written —
the disposition and the verbatim must-fixes are recorded in "## Closure disposition & must-fixes (2026-07-22 design
red-team)" at the foot of this file. **Not executed this session:** the SpliceBERT.1024nt weights and the
coupling/atlas data are absent from this worktree, so any model run must be launched from the main checkout.

This draft takes the separate multispecies ≈10.63 positional carrier **one column right on the evidence
ladder** — from controlled post-softmax attention (C11: `A`) to **OV/mediation + behaviour** — and closes the
open claim C12 ("its OV content, behavioural function, and biological origin are UNKNOWN"). It reuses the
`U_info`/donor-anchor SVD-of-a-contrast machinery (C16) re-anchored on the carrier plane, and the
`carrier_surgery`/`carrier_dose` deletion instruments re-pointed from the period-3 band to the ≈10.63 band.
Two structural seams — a **checkpoint mismatch** and a **control-band identity** — are surfaced head-on in §5;
they are why this is a draft for red-team, not a freeze.

Provenance: **SpliceBERT.1024nt** sha256 `2ad91428…0149d`, post-LN BERT-MLM 6L×16H, hidden 512, **absolute**
learned position embeddings P (1026×512, `ad.model.bert.embeddings.position_embeddings`). The ≈10.63 band on
this checkpoint is `BAND_P10 = range(94, 101)` (bins 94–100; peak bin 96; period `1026/96 = 10.6875`), named
in `scripts/splicebert/run_second_carrier.py`. The C11 reader precedent was established on a **different**
checkpoint — **SpliceBERT.510nt** (multispecies), carrier bins `[46,47,48]`, `T=10.61` — see §5 risk-1. Env
`gfm-thrust1`; compute **MPS** (no CUDA). Deterministic cohorts (`PYTHONHASHSEED=0`, `sorted(set())`).

The name defended throughout (matching C11/C13/C14 wording, no biology imported):
> **A separate learned ≈10.63-period positional carrier:** a high-frequency mode of P (bin 96 on the 1024nt
> checkpoint; 64% of the energy-matched "control band" 94–100), distinct from the period-3 codon clock, read
> post-softmax by a controlled set of heads. **Its OV content and behavioural use are the open question.**

---

## B1's founding tensions (verified pre-compute; the red-team's gifts)

- **T1 — checkpoint mismatch (leading).** C11's eight Level-A readers (`L2H12, L3H3, L3H5, L3H8, L3H9,
  L3H11, L3H13, L4H13`; `postsoftmax.json → models.multispecies`, `T=10.61`, carrier `[46,47,48]`) live on
  **SpliceBERT.510nt**. The donor-coupling behaviour (C1–C36), `U_info` (C16), and the `carrier_surgery`/
  `carrier_dose` instruments live **only on SpliceBERT.1024nt**. The two checkpoints' head indices do **not**
  transfer: on 1024nt the demonstrated ≈10.63 QK readers (`second_carrier.json`) are `L0H5/H4/H3/H14/H12/H10/
  H2, L1H14, L2H8/H10/H7` — **disjoint** from the 510nt L2/L3/L4 set, and **pre-softmax only**. B1 therefore
  runs entirely on 1024nt and **re-establishes** the post-softmax reader set on-model (Preflight R, §2); C11 is
  cited as motivating precedent, never as a transferable head list.
- **T2 — control-band identity.** `BAND_P10` (94–100) is **exactly** the energy-matched *inert control band*
  used by `carrier_surgery`/`carrier_dose`: in `carrier_surgery.json` the band 94–100 is the **worst-damage**
  control (exon ΔL `0.0265`, intron `0.0253`), and `carrier_dose.py` selects it as `CTRL`. Re-pointing the
  deletion instrument onto `BAND_P10` requires **re-centring the control-band exclusion window off period
  ~10.6** (§2). Pre-existing (already-seen) dose evidence: deleting `BAND_P10` fully leaves coupling at
  **+0.970** (intact **+0.946**) while period-3 deletion collapses it to **+0.0265** — i.e. the band is inert
  *for coupling*. This **strengthens** C8's control (deleting a genuinely-read carrier still does not move
  coupling), and B1 must not present a coupling result that quietly invalidates it. See §5 risk-2.

---

## 0. INTENT (north star — re-read at every gate)

*The ≈10.63 carrier reaches controlled post-softmax attention (C11). Does that attention **write anything onto
a model readout** (OV/mediation), and does deleting the carrier **move any pre-declared native model output**
(behaviour)? Or is it a positional feature the model **represents and routes but does not use** — the natural
contrast to the period-3 clock, which IS used (C8/C36)?*

Model-native observables: (OV) the carrier-driven OV component of each on-model reader head, SVD'd and tested
for overlap with / sufficiency for the donor readout direction `U_info` against an energy-matched random-rank
null (the C16 add-one-p pattern); (behaviour) the donor–acceptor **coupling margin** `C` (primary) and general
masked **MLM true-base NLL** (secondary) under `BAND_P10` deletion, with the same reciprocal / low-leak /
content-shuffle controls as C11. **Biology is deliberately empty** (the helical/nucleosomal reading is already
refuted by C14: real `10.298` vs model `10.6875`, model fit to ±0.005; origin is a training-time question,
out of scope on a frozen checkpoint).

---

## 1. ESTIMANDS

Carrier conditions on P (all via rfft-bin zeroing of `P − μ`, the `make(bins)` construction shared by
`run_carrier_surgery.py` / `run_second_carrier.py`): **I** (intact); **D10** (`make(BAND_P10)`,
BAND_P10=`range(94,101)`); **RECIP** (adjacent energy-matched 7-bin band, period off ~10.6, the C11 reciprocal
analogue); **LOWLEAK** (a low-energy 7-bin band, the C11 low-leak analogue); **SHUF** (content-shuffle, a
positivity check per R15, never an artefact criterion). All contrasts are **paired within-example**,
gene-clustered (`gboot`/`gboot_ratio`), CI-excludes-0/SESOI (never point-estimate sign).

### B1-OV — does the carrier's attention write onto the donor readout? (OV/mediation)
For each head `h` in the on-model Level-A reader set `R_1024` (Preflight R):
- **Carrier-driven OV delta** `ΔO_h = O_h(I) − O_h(D10)`, where `O_h` is head `h`'s attention-weighted,
  `W_O`-projected output contribution to the residual stream at the read positions (captured natively, one
  `W_O` via the module's `attention.output.dense`, `eager=True`). `ΔO_h` is the carrier-specific OV content —
  the exact analogue of the donor-anchor contrast `delta = H_cl − H_co`.
- **Carrier-OV subspace** `U_ov^h = top_k_subspace(ΔO_h, k)` (`subspace.top_k_subspace`, C16 machinery),
  `k∈{1,2}` (the carrier is rank-2). **U_info is defined once on the intact model** (C16; never refit under
  D10) and only projected onto.
- **Overlap estimand** `Ω_h = subspace_overlap(U_ov^h, U_info)` vs the random-subspace baseline
  (50 random orthonormal pairs, `random_subspace`), exactly as donor-anchor D2.
- **Sufficiency estimand** `S_ov^h` = fraction of the donor coupling margin recovered by patching the
  carrier-OV component into the corrupt donor residual (`anchor_gates.suff_recovery` with `U=U_ov^h`), tested
  against `null_recoveries` (energy-matched random rank-k, `N_NULL≥200`) via `empirical_upper_p_value`
  (`p_add_one`). **This add-one-p null is the R23-class forced-identity guard** (see §5 risk-4).
- **Carrier-specificity:** repeat `ΔO_h` with the **RECIP** deletion instead of D10; require `ΔO_h(D10)`'s
  overlap/sufficiency to **exceed** the RECIP (and LOWLEAK) counterpart (add-one p over the control-band set).

### B1-BEHAV-primary — donor–acceptor coupling `C` under carrier deletion (the C8/C36 readout)
`C = margin_hard(clean) − margin_hard(corrupt)`; corruption = partner AG→CC at `[a−2,a−1]` (real bases, the
CouplingTask default). `ΔC(D10) = C(I) − C(D10)`, gene-clustered. **Frozen directional prediction (declared
BEFORE the confirmatory cohort is opened): EQUIVALENCE** — deleting the ≈10.63 carrier does **not** move
coupling beyond `SESOI_C`. Rationale (prior, not evidence): the already-seen `carrier_dose` control arm
(0.946→0.970). Because that arm was **already seen**, the confirmatory equivalence is run on a **fresh,
disjoint gene cohort** (§7); the seen number is a prior, not the test.

### B1-BEHAV-secondary — general masked MLM true-base NLL (EXPLORATORY, MPS-wall-flagged)
`ΔL(D10) = L_masked(D10) − L_masked(I)`, per masked base, split exon / intron
(`run_carrier_surgery.evaluate` machinery). Known already-seen point estimates: exon `≈0.0265`, intron
`≈0.0253` — **coding-UNenriched** (contrast to period-3, whose coding ΔL CI is `[0.138,0.180]` vs intron
`[0.029,0.055]`). Reported **descriptively / one-sided** at achieved precision; **not** a confirmatory
equivalence, because `≈0.026` sits on the 0.02-nat MPS wall (§5 risk-3, §7).

---

## 2. METHOD (reused machinery + one preflight + net-new drivers)

**Reused / validated (exact paths):**
- SVD-of-a-contrast + causal subspace tests: `src/gfmcd/splicing/subspace.py`
  (`top_k_subspace`, `project`, `patch_component`, `subspace_overlap`, `random_subspace`, `energy_spectrum`);
  `src/gfmcd/splicing/anchor_gates.py` (`suff_recovery`, `nec_fraction`, `null_recoveries`, `alive_margin`);
  driver template `scripts/splicebert/run_donor_anchor.py` (G0 unembed anchor, `U_info` construction,
  add-one-p null). `U_info` reused as defined by C16 (top-SVD of clean−corrupt final donor residual).
- Deletion + energy-matched control + DiD + health: `scripts/splicebert/run_carrier_surgery.py` (`make`,
  control-band search, `evaluate`); dose-response: `scripts/splicebert/run_carrier_dose.py` (`scaled`,
  `coupling`, `mlm`). Both **re-pointed** `BAND ← BAND_P10 = range(94,101)`.
- Post-softmax controlled-reader test: `scripts/splicebert/run_postsoftmax.py` (`profiles`, `deletion_audit`,
  `m_of`, BH-FDR Level-A criterion: `significant AND kept_carrier<0.50 AND kept_reciprocal>0.70 AND
  kept_low_leak>0.70`; content-shuffle a positivity check, R15).
- Coupling / margins / health / CIs: `src/gfmcd/splicing/circuit.py` (`CouplingTask`, `margin_hard`, `gboot`,
  `gboot_ratio`, `ratio_of_means`). Adapter `eager=True` (SDPA silently zeroes attentions).
- Provenance: `gfmcd.provenance.ExperimentManifest`; `gfmcd.rigor.empirical_upper_p_value`.

**Preflight R — on-model reader establishment (net-new driver; gates the whole OV arm; T1 fix).**
Re-point `run_postsoftmax.py`'s Level-A machinery to **SpliceBERT.1024nt** at `BAND_P10`, with a 1024nt-native
**reciprocal** band (adjacent energy-matched, period off ~10.6; fixed on discovery), **low-leak** band, and
content-shuffle. Output `R_1024` = the set of heads reaching Level A (collapse under D10, survive RECIP +
LOWLEAK, survive SHUF). **If `R_1024 = ∅` → the OV arm is non-identifiable → Endpoint B for OV** (the behaviour
arm still runs). `R_1024` is **not** assumed equal to C11's 510nt eight heads.

**Net-new drivers (piloted in `--smoke`, must pass before any freeze):**
- **`run_carrier_ov.py`** — for each `h∈R_1024`: capture `O_h` under I / D10 / RECIP (native forward, one
  `W_O`); build `ΔO_h`; `top_k_subspace`; `subspace_overlap` vs random; `suff_recovery` vs `null_recoveries`
  (add-one p). Writes a manifest.
- **Re-pointed `run_carrier_surgery.py` / `run_carrier_dose.py`** — `BAND=BAND_P10`; control-band search
  **excludes period 3/1.5/6 (existing) AND period ~10.6 ±5%** so the ≈10.63 band is never both treatment and
  control (T2 fix). Coupling + exon/intron MLM readouts as in C8.

**EXCLUDED (R14):** no ±1/±2 head-roll or attention phase-shift assay (stride manufactures effects). D10 is a
band **deletion** (bins zeroed), never a comb-roll.

---

## 3. MEMORIALS (on the measurement path; each CAN fail)

| # | Memorial | Passes iff | Note |
|---|---|---|---|
| M0 | unembed anchor (G0) | `max|cls(hidden[-1]) − logits| < 1e-3` at the donor | reused donor-anchor anchor; failable |
| M-del | D10 is the exact ≈10.63 clock | `make(BAND_P10)` zeroes rfft bins 94–100 of `(P−μ)`; element-wise reconstruct anchor `<1e-6`; peak bin 96 | construction anchor; failable (wrong-axis/off-by-one) |
| M-read | on-model readers established | Preflight R yields `R_1024`; deletion audit `target_retained<1e-6`, `protected_*_retained>0.999` | **failable → if ∅, OV arm non-identifiable, reported as such** |
| M-ovbase | OV capture identity | recomposed head output from captured `O_h` reproduces the native residual contribution to `<1e-6` (diagonal anchor) | R21-fix; validates the OV path is the real one |
| M-ovnull | OV recovery beats matched random | `S_ov^h` CI-low `>` `null_recoveries` max **AND** `p_add_one ≤ 0.05` | **the R23-class forced-identity guard**; failable → OV write not carrier-specific |
| M-ovspec | OV is carrier-specific | `Ω_h(D10)`/`S_ov^h(D10)` exceed the RECIP **and** LOWLEAK counterparts (add-one p over control bands) | specificity; failable |
| M-band | ≈10.63 band energy | `BAND_P10` power within the surgery control-band tolerance; bin 96 is its peak (64% of band energy) | descriptive anchor; failable |
| M-ctrl | control window re-centred | every energy-matched control band used by the re-pointed surgery/dose has period **not** in {3,1.5,6, 10.6±5%} | **T2 fix**; failable (would re-introduce the treated band as its own control) |
| M-couphealth | coupling assay alive under D10 | every interpreted arm keeps ≥0.5 MLM headroom over chance (R13) | below-floor → arm dropped |
| M-p3contrast | period-3 positive control still fires | re-pointed instrument, run at `BAND=range(339,346)`, reproduces C8 collapse `C(D)/C(I) ≤ 0.05` | sensitivity of the re-pointed instrument; failable |
| M-det | determinism | `PYTHONHASHSEED=0`; `sorted(set())`; cross-seed cohort identity | reproducibility (IC1 A2) |

---

## 4. CALIBRATION (sensitivity + specificity; hard preflight gates)

**Planted-positive (each travels the real channel):**
- **PP-p3 (deletion channel):** the re-pointed surgery/dose at `BAND=339–346` reproduces C8's coupling
  collapse (M-p3contrast) — proves the re-pointed instrument can still detect a *used* carrier.
- **PP-U (OV channel):** the C16/C17 planted positive — patch the intact `U_info` component into the corrupt
  donor residual recovers ≈0.9 vs a random rank-8 subspace ≤0.05 — calibrates `suff_recovery` + `null_recoveries`.
- **PP-read (reader channel):** the period-3 band on 1024nt run through Preflight R yields a **non-empty**
  Level-A reader set (the period-3 clock is demonstrably read post-softmax) — proves Preflight R can DETECT a
  read carrier on 1024nt before it is asked about ≈10.63.

**Null / specificity (must not fire above target FPR):**
- **N-recip:** RECIP/LOWLEAK deletions do not collapse the reader `m_T` (C11 survive_min>0.70) and do not
  produce OV overlap/sufficiency above random (M-ovspec).
- **N-randsub:** energy-matched random rank-k subspaces recover ≤ the frozen null threshold (`null_recoveries`,
  ≥200 draws) — the OV write must beat them (M-ovnull).
- **N-shuf:** content-shuffle survival (positional carriers survive; a content artefact would not) — positivity,
  R15, not an artefact criterion.

**Confound-specificity (importance-matched, the identifiability fix):** an OV or behavioural effect counts as
carrier-specific only if it **exceeds the energy-matched control-band set** (RECIP + LOWLEAK + the surgery
control distribution), never merely a junk band.

---

## 5. IDENTIFIABILITY (the honest non-identifiable outcomes are pre-committed)

- **Risk-1 — checkpoint mismatch (leading; T1).** C11's readers are 510nt; readout/behaviour/`U_info` are
  1024nt; indices do not transfer (verified). **Mitigation:** run entirely on 1024nt; establish `R_1024`
  on-model (Preflight R). **Honest non-identifiable outcome:** if no 1024nt head reaches post-softmax Level A
  at `BAND_P10`, the OV arm is **non-identifiable → Endpoint B (OV)**; B1 then reports only the behavioural
  contrast and explicitly scopes C12-OV as unreached. B1 makes **no** cross-checkpoint head-identity claim.
- **Risk-2 — control-band identity (T2).** `BAND_P10` is C8's inert control band. **Mitigation:** re-centre the
  control exclusion (M-ctrl); run the confirmatory coupling arm on a **fresh** cohort (the 0.946→0.970 arm was
  seen). **Reconciliation, pre-committed:** a coupling **equivalence** for D10 is *consistent with and
  strengthens* the existing control (deleting a genuinely-read carrier still does not move coupling; this is
  `run_second_carrier.py`'s own argument). A coupling **positive** for D10 would instead mean the band is *not*
  inert for coupling → B1 must then flag that `carrier_surgery`/`carrier_dose`'s use of 94–100 as an inert
  control is **only inert for MLM, not for coupling**, and that C8's control needs re-selection. Either way the
  outcome is reported against this control's role explicitly.
- **Risk-3 — MPS wall.** The ≈10.63 MLM footprint (`≈0.026`) sits on the 0.02-nat wall (verified anchor: C8
  raw-mean gene-clustered CI half-width `0.0198` at n=384). A **powered MLM equivalence is unreachable** at
  budget → the MLM arm is **one-sided / at-achieved-precision only**, never a confirmatory null. The **coupling**
  equivalence (SESOI_C `0.05`, robust denominator ~0.68–0.9; dose n=256 already cleanly resolved) **is** reachable.
- **Risk-4 — OV forced-identity (R23-class).** Projecting any rank-2-plane-driven OV delta onto `U_info` can be
  non-zero by construction. **Mitigation:** the energy-matched random rank-k null (M-ovnull, `null_recoveries`
  + add-one p, the exact C16 pattern) — the carrier-OV subspace must **beat matched random**, else the write is
  not carrier-specific → OV null. `Ω_h` alone (overlap) is never interpreted without `S_ov^h` (causal) beating
  the null.
- **Risk-5 — pre-softmax ≠ post-softmax (R14 in a new costume).** `second_carrier.json` readers are QK-logit
  amplitudes; a huge smooth logit wave can still softmax to a spike. Post-softmax Level-A (Preflight R) is
  required **before** any OV interpretation; QK readers are used only to seed the head search, not to license
  a reader claim.
- **Risk-6 — held-out ≠ sampling stability (R22).** `R_1024`, `k`, control bands, SESOI, and any window are
  fixed on a **discovery** gene split; confirmed on a **disjoint untouched** split; the reader selection uses
  ≥2 independently seeded draws (effect-existence via CI-excludes-0/SESOI, gene-clustered); the coupling assay
  reuses genes as clean/corrupt so uses gene-clustered (`gboot`) CIs, not permutation-on-correlation.
- **Biology (deliberately empty).** C14 already distinguishes the real nucleosomal `10.298` from the model
  `10.6875`; B1 makes **no** biological-origin claim (training-time, out of scope on a frozen checkpoint).

---

## 6. DECISION RULE (frozen; result → conclusion)

**Per-family SESOI (own scale, discovery-fixed):** `SESOI_C = 0.05` nat (≈5–10% of the C8 effect; matches
IC1/P3); `SESOI_ov` = the C16 sufficiency scale (a rank-k random subspace's 95th-percentile recovery,
discovery-fixed; a write "lands" iff `S_ov^h` CI-low `>` that AND `> SESOI_ov`); `SESOI_L = 0.02` nat (the MLM
wall — used only for the descriptive one-sided read, never a powered equivalence).

**Family control:** the OV heads in `R_1024` share genes → max-T gene-bootstrap for the positive family; TOST
at family-adjusted α for the coupling equivalence. Every "D10 vs control-band" test is the **paired
within-example difference** CI, not non-overlap of two arm CIs.

**Endpoints (all legitimate; each closes or bounds C12):**
- **A-OV — promote C12 to OV/mediation:** `R_1024 ≠ ∅` **and** ≥1 head has `S_ov^h` beating the matched-random
  null (M-ovnull) **and** carrier-specific (M-ovspec) **and** coupling `ΔC(D10)` CI-low `> SESOI_C` (a *used*
  behavioural readout). ⇒ *"The ≈10.63 carrier's post-softmax attention writes carrier-specific OV content onto
  the donor readout `U_info`, and its deletion moves donor coupling — a **second causal positional mechanism**
  into the donor readout beside the period-3 clock."* (Prior: unlikely — coupling already ≈inert.)
- **A-null — bounded "represented + routed, not used":** OV sufficiency in **powered equivalence** to the
  matched-random null (or `R_1024=∅`) **and** coupling `ΔC(D10)` in **powered equivalence** (whole
  gene-clustered CI ⊂ ±`SESOI_C`). ⇒ *"The ≈10.63 carrier reaches controlled post-softmax attention (C11) but
  writes no carrier-specific content onto the donor readout and is not used for donor coupling — a positional
  feature the model represents and routes but does not use, the clean contrast to the period-3 clock (C8/C36)."*
- **A-mixed — writes-but-inert:** OV write lands (M-ovnull passes) **but** coupling is in powered equivalence.
  ⇒ *"The carrier writes a carrier-specific component onto the readout subspace that is behaviourally
  redundant/masked for coupling"* (honest; reported, not over-read).
- **B — unresolved:** `R_1024` non-identifiable, OV forced-identity guard fails to separate, or any
  interpreted CI is underpowered (half-width ≥ SESOI). ⇒ *"C12-OV unreached on this evidence."*

Mixed/partial or any inconclusive family member on the primary path → **Endpoint B** for that arm. The MLM
secondary never drives an endpoint (risk-3); it is reported descriptively as coding-enrichment context.

---

## 7. POWER / MDE & HELD-OUT

- **Power:** per-family MDE from a discovery pilot. **Coupling equivalence** (primary): reachable — SESOI_C
  `0.05`, robust denominator, `carrier_dose` n=256 already resolved D10 cleanly; sized so CI half-width `<
  0.05`. **OV sufficiency:** the second-difference-free `suff_recovery` at n≈150–250 unconditional exons
  (donor-anchor budget) with N_NULL≥200; MDE from the C16 null spread. **MLM** (secondary): a powered
  equivalence at `SESOI_L=0.02` is **UNREACHABLE** at MPS budget (CI half-width `0.0198` at n=384) → reported
  one-sided / at achieved precision, never null (IC1/R22/P3.4 lesson). Any NULL/equivalence call that is not a
  **powered** equivalence is reported **inconclusive**, never null.
- **Held-out:** discovery split (subset of val genes) sets `R_1024`, `k`, control bands, SESOI, health floor;
  the OV heads and coupling equivalence are confirmed on a **disjoint, untouched** val-gene split; **the
  already-seen `carrier_dose` control arm (0.946→0.970) is a prior only** and its cohort is excluded from the
  confirmatory coupling cohort (outcome-aware ≠ confirmatory). Sealed test (chr8/9) **not opened**.
  Deterministic cohort; gene-clustered CIs; ≥2 seeded draws for the reader selection.

## 8. GATES
- **Global PASS = AND(** M0, M-del, M-read, M-ovbase, M-ovnull, M-ovspec, M-band, M-ctrl, M-couphealth,
  M-p3contrast, M-det, PP-p3, PP-U, PP-read, N-recip, N-randsub, N-shuf, confound-specificity **).** Run STOPS
  on failure; below-floor arms (M-couphealth) dropped; **M-read failing (∅ readers) demotes only the OV arm to
  non-identifiable**, not the whole run.
- **`--smoke`:** M-del (construction anchor), M-ovbase (OV capture identity), M-ctrl (re-centred control
  window), M-p3contrast (period-3 positive control on the re-pointed instrument), M-det, and Preflight R
  producing a non-empty set on the **period-3** band (PP-read) — tiny cohort, **val sealed**. The "pilot the
  re-pointing + OV plumbing and freeze" step; must pass before freeze.

## 9. SCOPE
- **Reaches (if A-OV):** OV/mediation + behaviour for the ≈10.63 carrier's write onto the donor readout and its
  use in donor coupling. **Reaches (if A-null/A-mixed):** a **powered bounded** behavioural characterization
  ("routed but not used" / "writes-but-inert"). Either closes or bounds C12; C11 (attention) is the entry point.
- **Does NOT establish:** the carrier's **biological origin** (training-time; C14 already refutes the
  nucleosomal reading); any **cross-checkpoint** head-identity claim (C11 is 510nt, B1 is 1024nt); a **semantic
  identity** for whatever OV content is found (R22 discipline — no "means concept X" from a direction/overlap);
  **binding** or an independently-constructed payload (R23); anything about the **period-3** carrier's role
  (C8/C34/C35 unchanged — B1 cannot redefine them).
- **Reconciliation obligation (T2):** B1 must state explicitly whether its coupling result strengthens
  (equivalence) or invalidates (positive) the existing use of band 94–100 as `carrier_surgery`/`carrier_dose`'s
  inert control, and if positive must flag that control for re-selection.
- **Provenance:** SpliceBERT.1024nt sha256 `2ad91428…0149d`; C11 precedent SpliceBERT.510nt (multispecies,
  bins 46–48, T=10.61); `BAND_P10=range(94,101)`, bin 96, period `10.6875`; seeds + `PYTHONHASHSEED=0` to the
  manifest; results write a manifest beside output.
- **Amendments:** pre-run only, logged `Amendment (date, pre-run, no data seen)`; once the primary is inspected
  a fix is *outcome-aware verification*, not confirmatory (IC1 A2).

---

## Closure disposition & must-fixes (2026-07-22 design red-team)

**Disposition: FIXABLE_THEN_RUNNABLE (survivor=NO).** Red-team synthesis:
[`../redteam/REDTEAM_SYNTHESIS_B1.md`](../redteam/REDTEAM_SYNTHESIS_B1.md) (5 independent adversarial lenses, run
before any confirmatory compute; formalize-audit-verify step 2). The behavioural arm (ΔC coupling under BAND_P10
deletion, re-pointed to the ≈10.63 plane) is clean and identifiable; the OV-write arm is non-identifiable as
written — the `dO_h = A_I·V_I − A_D·V_D` contrast conflates the carrier's already-established QK/routing role
(C11/C35) with the OV column (an R23-class ladder column-jump). Both defects are fixable on the frozen checkpoint,
so the branch becomes runnable after the must-fixes below.

**survivor = NO — not executed this session.** The OV estimand needs the P3.3 A-vs-V redesign and the primary
coupling-equivalence needs a sizing pilot before any freeze; per formalize-audit-verify it must not be run
confirmatorily until those must-fixes land, and this session is documentation finalization only. **Operational
note:** the SpliceBERT.1024nt weights and the coupling/atlas data are absent from this worktree, so any model run
must be launched from the main checkout.

**Must-fixes (verbatim, from the synthesis):**

1. Redesign the OV estimand to isolate the value channel: import the P3.3/C35 A-vs-V decomposition (results/p3_av_decomposition.json + driver) and attribute an OV write ONLY to the value-restore contribution (intact-carrier values on the D10 base, attention held at D10). The as-written dO_h = O_h(I)-O_h(D10) = A_I*V_I - A_D*V_D conflates the carrier's ALREADY-established QK/routing role (C11/C35) with the OV column, so a positive result relabels a QK effect as an OV write (R23-class fatal). Relabel every OV claim the decomposition does not license as a mediation-through-head-h (Attention column) claim.
2. Add a router-only planted-negative: run the identical OV pipeline on period-3 L3H1 (established routing-only per C35/R24) and REQUIRE S_ov in equivalence with the null; a firing router voids the OV arm's ability to reject a pure router.
3. Add a live-channel planted-positive that travels dO_h -> top_k_subspace -> S_ov and MUST fire (plant a known low-rank write into a reader head's OV output and recover it), OR drop the OV-equivalence (A-null-OV) endpoint and scope OV to positive-only. As written PP-U only calibrates on U_info itself, so the modal A-null outcome is a dead-instrument-vs-true-null ambiguity (P3.4 PP-completion class).
4. Drop the 'energy-matched RECIP' specificity control: bin 96 is the ONLY high-frequency mode besides period-3 on 1024nt (run_second_carrier.py), so no character-matched off-carrier band exists; replace the isotropic random_subspace null (subspace.py:38-42, a uniform QR draw, NOT energy-matched despite the wording) with a covariance-matched OV-output null, and use the surgery control-band distribution as an explicitly magnitude-mismatched reference.
5. Size the PRIMARY coupling-equivalence in the pilot: compute gene-clustered ΔC CI half-width AND per-example ρ in the pilot; gate equivalence on HW < (SESOI_C - |point|); pre-declare INCONCLUSIVE if unmet; re-scope to ~700+ genes or one-sided if ρ<0.98.
6. Bound the A-null wording: the arm's own seen MLM data (band 94-100 exon 0.0265 / intron 0.0253 vs ~0 for inert bands) shows the carrier IS used, so claim only 'not used for donor COUPLING and writes nothing carrier-specific onto U_info', never unqualified 'does not use'; report any one-sided MLM positive as 'used non-coding-specifically' (the true contrast to period-3's coding-enriched use).
7. Add OV-equivalence MDE and control FWER: pre-declare OV-null one-sided-only (equivalence unreachable at budget); apply max-T/BH across the ~8-head family; restrict per-head OV claims to L0 readers; and fix the launch path (data/models absent in worktree).
