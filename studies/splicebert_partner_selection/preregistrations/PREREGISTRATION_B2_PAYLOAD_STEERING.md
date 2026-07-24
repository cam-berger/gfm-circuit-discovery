# PREREGISTRATION — B2: Payload semantics (closes headline limitation C30) (v1)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: NOT FROZEN — the 5-lens design red-team (2026-07-22) adjudicated this draft FIXABLE_THEN_RUNNABLE (survivor=NO).**
See [`../redteam/REDTEAM_SYNTHESIS_B2.md`](../redteam/REDTEAM_SYNTHESIS_B2.md). The SEMANTICS arm that would actually
close C30 is NON-IDENTIFIABLE on a frozen checkpoint at MPS budget — a one-sided positive search re-running the exact
R22 failure (held-out r −0.512→+0.017), equivalence unreachable. The core steering direction u_choice is a
position/distance axis dominated by learned PE geometry (R20 re-entry): it is a diff-in-means between two
different-position acceptors. EXISTENCE is near-tautological; the SPECIFICITY discriminators are null/wrong-signed at
pilot. Honest outcome: C30's positive semantic identity is NOT closable by frozen-checkpoint steering at MPS budget —
the paper should ship with C30 scoped out. Retain as hardened design-of-record; a heavily re-scoped geometry-controlled
EXISTENCE+partner-specificity positive search is the only runnable remnant, SEMANTICS deferred to a
training-time/external-assay design. This v1 is retained as the **design of record**; it must **NOT be run** as
written — the disposition and the verbatim must-fixes are recorded in "## Closure disposition & must-fixes (2026-07-22
design red-team)" at the foot of this file. **Not executed this session:** the SpliceBERT.1024nt weights and the
coupling data are absent from this worktree, so any model run must be launched from the main checkout.

This design targets the project's **headline scientific limitation, C30**: the L3H1-delivered acceptor
payload is *acceptor-local* (C24) and *not bulk composition* (C30/A9b, identifiable), but its **positive
semantic identity is OPEN**. "Splice-site strength" was **withdrawn (R22)**: absolute local MaxEnt was
sample-unstable across independently seeded draws (held-out `r` −0.512 → +0.017) and non-significant under
two-way gene-dyadic inference. B2 replaces the failed **correlational strength probe** with **causal
steering on paired site-choice contrasts** (the reviewer/ROADMAP Track 2 = the Head–Context Atlas WH7
program). It is deliberately staged behind a strict firewall — **EXISTENCE → SPECIFICITY → SEMANTICS** —
because a direction that moves behaviour is **not** necessarily the concept direction (the R20 trap, now at
the steering layer). The pre-run posture (see §7, §9) is that the **existence + specificity core is
identifiable and MPS-affordable**, while the **semantics arm is likely NON-IDENTIFIABLE on a frozen
checkpoint** and is scoped to a one-sided positive search with a pre-declared honest null.

Provenance: SpliceBERT.1024nt sha256 `2ad91428…0149d`, post-LN BERT-MLM 6L×16H, hidden 512, **absolute**
learned position embeddings P (1026×512, `ad.model.bert.embeddings.position_embeddings`). Env
`gfm-thrust1`; compute **MPS** (no CUDA). Deterministic cohorts (`PYTHONHASHSEED=0`, `sorted(set())`).

The object under test (the C30 payload, restated at the steering layer):
> **A transferable acceptor-local "choice" direction** `u_choice` — the residual-stream axis that
> distinguishes a model-preferred acceptor site from a matched less-preferred acceptor site sharing the
> same intron-mate partner. Whether such a direction (a) exists causally, (b) is partner-specific, and (c)
> is *named* by any external splice variable, are three separate questions on three ladder rungs.

---

## 0. INTENT (north star — re-read at every gate)

*What biological/model variable does the acceptor-local payload REPRESENT — i.e., can we identify the
positive semantic identity of the payload whose composition-exclusion (A9b/C30) is established but whose
meaning is unresolved (R22)?*

The intent is a **naming** claim about the payload. B2 operationalizes it as: **learn the direction that
separates preferred from less-preferred acceptor sites in a same-partner choice set (difference-in-means,
Marks & Tegmark 2024), causally add it (ActAdd / Contrastive Activation Addition), and only then ask
whether an external variable predicts its causal dose response.** The naming claim lives entirely in the
**SEMANTICS** arm; EXISTENCE and SPECIFICITY establish only that there *is* a causal, partner-specific
choice direction to be named. No arm may silently inherit the label of the next (Atlas "Evidence labels").

Model-native observable: the coupling margin `B_i(p) = margin_i(native) − margin_i(p destroyed)` and the
relative coupling `R_ij(p) = B_i(p) − B_j(p)` within a two-site choice set (Atlas §"Primary model-native
estimand"), under a **dose-graded residual injection** `h_l(j) ← h_l(j) + α·u_choice` at the less-preferred
acceptor window.

**This is the intended attempt to close C30. It does not touch the sealed causal splice claims
(C34/C35); it does not touch the sealed test split (chr8/9).** Held-out on a disjoint val-gene split.

---

## 1. ESTIMANDS (three staged, on three ladder rungs; each measures the INTENT-slice, not a proxy)

Cohort: intron-mate **alt-3'SS acceptor competition** sets (`competition.py` `site_type='alt3ss_acceptor'`
— two+ acceptors receiving one shared upstream donor, mutually exclusive by construction). This arm is
chosen because the C30 payload is *acceptor-local* (C24/C30): steering an acceptor's payload state is the
direct analogue of the object whose semantics is unresolved. `alt5ss_donor` is reported as a
generalization arm only.

Site labelling (net-new, DISCOVERY genes only): for each acceptor site `i` in a choice set with shared
partner `p`, compute native `B_i(p)`; the **preferred** site is the higher-`B` acceptor, **less-preferred**
the lower-`B`. Labels are **model-native behaviour**, never MaxEnt (R20 firewall). Direction:
`u_choice = mean_win(r_pref) − mean_win(r_less)` over the 4-token acceptor window residual at layer `l`
(cross-fitted, gene-disjoint folds).

### B2.1 — EXISTENCE (rung: mediation / behaviour) — is there a causal choice direction?
Injection `h_l(j) ← h_l(j) + α·u_choice` at the **less-preferred** acceptor window `j`, over a symmetric
dose grid `α ∈ {−α₂,−α₁,0,+α₁,+α₂,…}` (norm-referenced to the intra-set state gap). Estimand:
- **`E_exist = dose-slope of B_j(p) in α`**, sign-specific (positive dose raises `B_j`), evaluated as the
  paired within-example lower bound of `ΔB_j(+α) = B_j(+α) − B_j(0)` at the top declared dose, **relative to
  the equal-norm random-orthogonal direction family** (max-statistic corrected). Units: nats of donor
  coupling margin. `α=0` must reproduce the un-injected forward (M-α0).

### B2.2 — SPECIFICITY (rung: mediation / behaviour) — does it move the *intended paired site*, partner-specifically?
Estimand = the paired change in **relative** coupling and its partner-conditioning:
- **`E_rel = Δ R_ij(p) = [B_i(+α)−B_j(+α)] − [B_i(0)−B_j(0)]`** at `j` — the injection must *redistribute*
  relative coupling toward `j`, not merely raise every margin (Atlas: "a raw rise in `margin_j` … is
  insufficient").
- **`E_partner = ΔR_ij(native p) − ΔR_ij(matched non-partner p′)`** — the steering must be **partner-specific**
  (distance/strength-matched non-partner and destroyed-partner arms; Atlas S4). If injection increases every
  site's margin or works without the partner → it is a site-confidence/readout direction, **not** a
  choice/payload direction (report as such).
- **`E_decoy = ΔR at the intended acceptor − ΔR at a distance/phase-matched decoy acceptor`** (Atlas S2
  control). Decoy-sparsity is a known risk (see §7).

### B2.3 — SEMANTICS (rung: representation → naming; the C30 target; PRE-DECLARED possibly non-identifiable)
Only entered if B2.1 **and** B2.2 pass. Estimand = whether an **external** variable predicts the *causal
dose response per example*, i.e. the per-example steering gain `g_e = ∂B_j/∂α` (paired slope):
- **X1 = V2b** (local, payload-extent-matched acceptor MaxEnt; `run_a9_payload_semantics.scorers` — the
  correct external splice-strength proxy under C24, the same variable R22 withdrew as an *absolute*
  correlate).
- **X2 = competitor-MaxEnt gap** `MaxEnt(pref) − MaxEnt(less)` (the **post-R22 conditional-competitiveness**
  target — "is the payload conditional site quality within the choice set?").
- **X3 = annotated usage** (transcript-support count `n_tx_target_junction` / MANE flag from
  `competition.py`; a provenance-pinned junction/PSI resource is deferred to the biological bridge, §9).
Estimand for each: the **dyadic-adjusted association of `X` with `g_e`**, out-of-sample (CV-R²), stable
across **≥3 independently-seeded discovery draws**, reported as **CI-excludes-0 under two-way (dyadic)
gene-cluster bootstrap** (R22) — never point-estimate sign, never within-sample held-out alone.

---

## 2. METHOD (reuse validated machinery by exact path; net-new plumbing piloted in `--smoke`)

**Reused / validated (exact paths):**
- **Cohort (annotation):** `src/gfmcd/splicing/competition.py` — `build_competition_index`,
  `enumerate_competition_examples`, `CompetitionConfig`, `split_for` (intron-mate, alt-3'SS acceptor).
- **Cohort (sequence windows):** `src/gfmcd/splicing/competition_seq.py::build_window` (fixed-length,
  transcript-oriented, real-genome filler, motif-checked; never PAD/`[MASK]`).
- **Behaviour + destroy:** `src/gfmcd/splicing/coupling.py` — `destroy_acceptor`, `destroy_donor`,
  `perturb_random` (the sham); `B_i(p)` / `R_ij(p)` exactly as `behavioral_pilot.py` computes them.
- **Model / margins / CIs:** `src/gfmcd/splicing/circuit.py` — `CouplingTask`, `margin`, `load_adapter`
  (`eager=True` — SDPA silently zeroes attentions), `pick_device`, `gboot_ratio`, `load_examples`.
- **Payload injection primitives:** `src/gfmcd/splicing/payload.py` — `value_weight_head`, `deliver`
  (`δV = δr W_Vᵀ`, post-LN value-linearity), `head_output`, `inject_margin_donor`; and A9's
  `run_a9_payload_semantics.py::inject_capture` + the **`synth=(Vplant,d0,β)` pure-signal branch** of
  `interchange` (the planted-positive channel, §4).
- **Control directions:** `src/gfmcd/splicing/subspace.py::top_k_subspace` for `U_info`; unembedding /
  site-identity directions for the control family (Atlas S2).
- **External variable:** `run_a9_payload_semantics.py::scorers` (V2b local MaxEnt; MaxEnt tables pinned
  sha256 `8b6d7b46…b7c00`, `maxentpy 0.0.4`).
- **Power anchors:** `results/splicebert_head_context_atlas/census_v1.json`, `behavioral_pilot_v1.json`,
  `specificity_pilot_v1.json`; `results/splicebert_coupling/ic2_calibration.json`.

**Net-new machinery (piloted in `--smoke`; must pass before freeze):**
1. **Model-preference labelling** of each choice set (native `B_i(p)` per candidate site → preferred /
   less-preferred). *The biggest engineering lift* (the annotation cohort is unlabelled by model preference).
2. **Difference-in-means choice-direction estimator** over paired contrasts, **cross-fitted, gene-disjoint**
   (folds by gene; the direction is frozen before any confirm-split forward).
3. **General dose-grid residual-injection hook** `h_l(j) += α·u_choice` at an arbitrary (window, layer)
   — A9 only injects through the L3H1 value path; B2 needs a residual add at the acceptor window on a chosen
   layer, with an exact `α=0` identity anchor (M-α0).
4. **Random-direction control family**: equal-norm random-orthogonal + equal-variance principal-component
   directions, with **max-statistic** family correction (Atlas S2 controls).
5. **OOD / residual-norm gate** at the injection site + **non-site MLM-health** probe (R13).
6. **Two-way (dyadic) gene-cluster bootstrap** over sender/receiver genes for the SEMANTICS association
   (R22; reuse A9b's dyadic estimator if present, else implement to the A9b spec).

**EXCLUDED:** no absolute-margin-only readout as evidence (Atlas: diagnostics, not `R_ij`); no
single-pair packets without a choice set; no MaxEnt-defined "preferred" labels (R20 firewall).

---

## 3. MEMORIALS (on the measurement path; each CAN fail; forced anchors LABELED)

| # | Memorial | Passes iff | Note |
|---|---|---|---|
| M-α0 | zero-dose identity | injecting `α=0` reproduces the un-injected forward margin to **<1e-6** | on-path; a buggy hook perturbs even at 0. **Not forced.** |
| M-lin | value-path linearity (if L3H1-value locus used) | `V₊−V₋ == W_V·Δr` to **<1e-4** (`deliver`) | on-path exact anchor (A2b/A9 `value_linearity`≈1.9e-6). Failable. |
| M-dose | dose monotonicity on a KNOWN direction | on the planted `d0` signal (§4), `B_j` strictly increases across the dose grid | the real sensitivity mechanic; failable |
| M-label | preference labels are model-native + generalize | discovery `B`-ordering of the two sites predicts the **confirm-split** ordering (sign agreement > chance, CI-excludes-chance) | guards the R20 trap (labels ≠ MaxEnt); failable |
| M-full | whole-state transplant reconstructs preferred `B` | injecting the exact `r_pref − r_less` window state at full dose ≈ the preferred site's `B` | **LABELED FORCED (R21-class)** — whole-window overwrite is near-identity plumbing, NOT evidence; the genuine test is M-dose with the *partial* direction |
| M-health | MLM health under every interpreted dose | non-site masked-probe keeps ≥0.5 headroom over chance (R13) at each `α` | below-floor dose → dropped |
| M-ood | injection stays on-manifold | injected residual-norm / OOD stat within a pre-set band of native at every interpreted `α` | off-manifold dose → dropped |
| M-both | both strands present | `len(strands)==2` | cohort sanity |
| M-det | determinism | `PYTHONHASHSEED=0`; `sorted(set())`; cross-seed cohort identity | reproducibility (IC1 A2) |

---

## 4. CALIBRATION (sensitivity + specificity; hard preflight gates)

**Planted-positive (travels the REAL channel):** reuse A9's `interchange(synth=(Vplant,d0,β))` branch —
deliver a **pure synthetic payload** `dr = Vplant·d0` where `d0` is the population-mean recovering payload
edit and `Vplant` is a known per-example scalar. The **entire B2.1→B2.3 pipeline must recover `Vplant`** as
the dose-response predictor (A9 achieved `r≈0.98`; the classifier the verdict uses must detect it, else it
is underpowered). Confirms the semantics regression **can** name a payload variable that is *really* there.

**Null / specificity (must not fire above target FPR):**
- **N-perm-label:** permute the preferred/less-preferred labels within gene → `u_choice` has **no**
  dose-ordered `ΔB_j` above the random-direction max-stat family.
- **N-perm-X:** permute the external variable `X` across examples → SEMANTICS association FPR ≤ 0.05
  (reuse A9's permutation-calibrated `r_thresh` on even perms, MEASURE FPR out-of-sample on odd perms).
- **N-random-dir:** equal-norm random-orthogonal and equal-variance-PC directions do **not** produce
  `ΔR_ij` above the corrected family (the core steering-non-identifiability guard).
- **N-control-dir:** `U_info`, `U_read`, unembedding, and site-identity directions are reported alongside;
  if a *generic* readout direction reproduces the effect, the choice direction is not distinguished.

**Confound-specificity (the identifiability fix):** the effect counts as a *choice* effect only if it (i)
exceeds the whole random-direction family (max-T), (ii) redistributes `R_ij` (not every-margin lift), and
(iii) is partner-specific (`E_partner` CI-excludes-SESOI). The SEMANTICS association counts as *naming*
only at adequate common support with **CV-R² > 0** and multi-draw dyadic CI-excludes-0.

---

## 5. IDENTIFIABILITY

- **Steering non-identifiability (R20 at the steering layer — the founding risk).** A behaviour-moving
  direction is **not** necessarily the concept direction. Firewall EXISTENCE→SPECIFICITY→SEMANTICS +
  random-direction max-stat family + `U_info`/`U_read`/unembedding/site-identity control directions.
  **Pre-declared honest non-identifiable outcome:** if B2.1+B2.2 pass but no external variable predicts the
  dose response under multi-draw dyadic inference (CI includes 0 / CV-R²≤0), the payload's semantic identity
  remains **UNRESOLVED** — but now **bounded strictly tighter than C30**: *"a causal, transferable,
  partner-specific acceptor-local choice direction exists and is NOT named by MaxEnt, competitor-MaxEnt gap,
  or annotated usage."* This is a legitimate, publishable stop (the paper can ship with C30 scoped out).
- **Absolute-position shortcut (R20/C22).** The direction must **not** be an absolute-position axis: require
  transfer across held-out position and distance strata (Atlas S1.2); residualize target/partner absolute
  position, packet geometry, GC, PY, CDS/UTR, composition, transcript support before labelling "preferred".
- **Held-out ≠ sampling stability (R22).** Direction, layer, dose grid, SESOI, control pools frozen on
  discovery; B2.1/B2.2 confirmed on a **disjoint untouched** val-gene split; **B2.3 effect-existence requires
  ≥3 independently-seeded discovery draws** (ideally demonstrably disjoint gene sets) — the exact failure
  mode A9 hit (held-out `r` −0.512→+0.017). Report CI-excludes-SESOI, never point sign.
- **Dyadic dependence (R22).** Choice-direction contrasts and interchange reuse genes as sender AND
  receiver → **two-way (dyadic) cluster bootstrap**; permutation-on-correlation under-covers.
- **Common support / small-n (A9b lesson).** Use adjusted / CV-R² not raw R²; prefer matching / cross-fitted
  DML over linear residualization for the X→g_e association; predefine an overlap check and report
  non-identifiable if support fails.
- **Object-mapping caveat.** The WH7 choice direction (candidate-site residual) and the C24/C30 payload
  (L3H1-delivered acceptor value) are argued to coincide for alt-3'SS acceptor competition (both acceptor-
  local), but this coincidence is an *assumption*; if injection acts through a different residual path
  (checked via `U_info`/L3H1-value locus comparison), the direction is causally usable but not shown to be
  the C30 payload — reported as such.

---

## 6. DECISION RULE (frozen; result → conclusion)

**SESOI (own scale, discovery-fixed):** `SESOI_B = 0.05` nat (matching IC1 / P3 / behavioral-pilot
`sesoi_c=0.05`) for coupling-margin contrasts; the SEMANTICS association uses a CV-R² floor > 0 plus
CI-excludes-0. A NULL/equivalence call requires **CI half-width < SESOI** (else **inconclusive**).

- **B2.1 EXISTENCE — positive** iff `ΔB_j(+α)` is monotone, sign-specific, and its family-adjusted (max-T
  over the random-direction family, gene-clustered) lower bound **> SESOI_B** at ≥2 positive doses, with M-α0
  / M-health / M-ood holding; **null/equivalence** iff the adjusted CI ⊂ ±SESOI_B with HW < SESOI_B.
- **B2.2 SPECIFICITY — positive** iff `E_rel` redistributes `R_ij` (paired CI-excludes-SESOI) **and**
  `E_partner` CI-excludes-SESOI (partner-specific) **and** intended-site `ΔR` > decoy-site `ΔR`. An
  every-margin lift or partner-independent effect → **site-confidence/readout direction, not a choice
  direction**.
- **B2.3 SEMANTICS — positive (payload NAMED)** iff, for some `X∈{V2b, competitor-gap, usage}`, the
  dyadic-adjusted `X→g_e` association **CI-excludes-0 across ≥3 independently-seeded draws** AND out-of-sample
  **CV-R² > 0**. **This is a one-sided positive search only** (equivalence is unreachable, §7). Non-robust /
  CI-includes-0 → **UNRESOLVED-but-bounded** (§5 honest outcome).

**Endpoints:**
- **A (C30 CLOSED):** *"The acceptor-local payload represents {conditional site competitiveness (X2) /
  local splice quality (X1) / annotated usage (X3)}: a causal, partner-specific choice direction whose dose
  response is predicted by X under multi-draw dyadic inference."* — reached iff B2.1+B2.2+B2.3 all positive,
  coherent, health-/OOD-/specificity-gated.
- **B (C30 TIGHTENED, not named):** *"A causal, transferable, partner-specific acceptor-local choice
  direction exists but is not named by MaxEnt / competitor-gap / usage — payload semantic identity remains
  open, now bounded below the composition floor AND the strength/competitiveness/usage floor."* — reached iff
  B2.1+B2.2 positive, B2.3 null/inconclusive. **Strictly stronger negative than the current C30.**
- **C (no steering / readout correlate):** *"The choice direction predicts held-out preference but injection
  does not causally move relative partner coupling"* — reached iff B2.1 fails; C30 unchanged; the payload
  is a readout correlate, not a steering variable (Atlas WH7 falsifier).

Mixed/partial patterns, or any inconclusive gate on the path, → **Endpoint B** (never a null from a wide CI).

---

## 7. POWER / MDE & HELD-OUT

- **Cohort supply is AMPLE (not the constraint).** `census_v1.json`: usable alt-3'SS+alt-5'SS competition
  examples = **11,275 (1,567 genes) in val**, **32,054 (4,864 genes) in train/discovery**, 3,014 (444 genes)
  in sealed chr8/9; median cluster size 3. Direction estimation and existence are supply-rich.
- **The signed site-choice BEHAVIOUR is weak (the real constraint).** `behavioral_pilot_v1.json`: the signed
  competition dependence `M_comp` = **+0.006 [−0.026,+0.040]** (BOTH, bounded-null); alt-3'SS
  **+0.034 [0.001,0.078]** (small-but-detectable); `M_comp_abs` = **+0.111 [0.082,0.142]** (material but
  sign-agnostic). Positive control `B` (partner coupling) = **+0.175 [0.124,0.229]** (material — instrument
  works). `SD(M_comp)` = **0.306** (example) / **0.317** (gene). ⇒ There is *some* signed choice behaviour to
  steer (alt-3'SS), but it is small; **EXISTENCE is a change in this weak signed relative-coupling.**
- **EXISTENCE is POWERED at budget.** A single-direction dose-slope contrast at n≈400 paired sites has
  CI-HW ≈ `1.96·0.31/√400 ≈ 0.030 < SESOI_B 0.05` (gene-clustered inflates modestly; size to CI-HW < SESOI on
  a discovery pilot). The 96-head multiplicity that holed the atlas head-screen does **not** apply (B2 tests
  one frozen direction, not 96 heads).
- **SPECIFICITY is MARGINAL and decoy-limited.** `specificity_pilot_v1.json`: the intended-vs-decoy contrast
  `abs_M_comp − abs_M_decoy` = **−0.047 [−0.118,+0.011]** (INCONCLUSIVE), with **425 windows lacking a usable
  decoy** (decoy-sparse). The partner-specific arm (`E_partner`) is the affordable specificity test; the
  decoy arm (`E_decoy`) may report at achieved precision only.
- **SEMANTICS equivalence is UNREACHABLE (defer/one-sided only).** `ic2_calibration.json`: the transport DiD
  `β_transport` SD = **0.774**, so a ±0.05 equivalence is unreachable even at the ~1,985-gene cap (IC2's
  finding). The dose-response × external-variable association is a comparably high-variance object; its
  **equivalence (null) confirmation is off the table at MPS budget**. B2.3 is therefore a **one-sided
  positive search** whose *robust positive* requires multi-draw dyadic — exactly where A9 failed (R22). Anchor:
  C8 coding-ΔL gene-clustered CI-HW = 0.0198 at n=384 (the MPS equivalence wall).
- **Held-out:** discovery = train autosomes ∉ {chr8,9,11,16,19} (fixes direction, layer, dose grid, SESOI,
  control/decoy pools, matching calipers); B2.1/B2.2 confirmatory on a **disjoint untouched** val split
  (chr11/16/19); **sealed test chr8/9 NOT opened**; B2.3 existence needs ≥3 independently-seeded discovery
  draws. Deterministic cohort; gene-clustered (dyadic for cross-gene contrasts) CIs.
- **Forward-pass budget (rough):** labelling ≈2·(#candidate sites); existence ≈ (#doses)·(#paired sites)·2;
  the random-direction family is the cost driver (subsample to a corrected family, not exhaustive). Order
  **2×10⁵–1×10⁶** forward passes total; the **existence+specificity core is affordable on MPS**, the full
  semantics multi-draw sweep is the heavy tail.

## 8. GATES

- **Global PASS = AND(** M-α0, M-lin, M-dose, M-label, M-health, M-ood, M-both, M-det, planted-positive
  sensitivity, N-perm-label, N-perm-X, N-random-dir, confound-specificity **).** Run STOPS on failure;
  below-floor / off-manifold doses (M-health, M-ood) are dropped, not interpreted. **M-full is FORCED and is
  never counted as evidence.**
- **Staged gating:** B2.2 runs only if B2.1 PASSes; B2.3 runs only if B2.1 **and** B2.2 PASS. A failed B2.1
  → Endpoint C; a passed B2.1+B2.2 with failed/non-robust B2.3 → Endpoint B.
- **`--smoke`:** the net-new plumbing on a tiny cohort, **val sealed** — the injection hook `α=0` identity
  (M-α0), dose-monotone on the planted `d0` (M-dose), preference-label ordering (M-label), determinism
  (M-det), the random-direction sampler + max-stat family producing a calibrated threshold, and the dyadic
  bootstrap achieving nominal coverage on a synthetic dyadic dataset. Must pass before freeze.

## 9. SCOPE

- **Reaches (if positive):** OV/mediation + behaviour level for a **causal, partner-specific acceptor-local
  choice direction** (B2.1/B2.2); representation→naming for the payload's semantic identity (B2.3, only if
  the one-sided positive fires robustly).
- **Does NOT establish:** a **biological** splice-choice claim (needs a provenance-pinned junction/PSI
  resource and a prospective minigene/MPSA perturbation — Atlas S5 / WH8, deferred); that a behaviour-moving
  direction **is** the concept direction absent the semantics arm (R20); that the choice direction **is** the
  C24/C30 payload if injection acts through a different residual path (checked, reported); any redefinition of
  the sealed carrier claims **C34/C35** (different object, different intervention) or the sealed behaviour
  C1–C2.
- **Pre-declared likely outcome (honest):** **Endpoint B** — existence+specificity plausibly runnable and
  informative, but the semantics arm is **likely non-identifiable on a frozen checkpoint** (equivalence
  unreachable per IC2 SD; positive search re-runs the R22 multi-draw/dyadic wall). A **minimal
  existence+specificity B2** is runnable now with semantics scoped out; the full naming claim may defer to a
  training-time / external-assay design.
- **Deferred, documented:** (a) biological bridge (junction usage / PSI + minigene, Atlas S5); (b) sequence
  realization (WH8 — constrained edits inducing the direction); (c) `alt5ss_donor` generalization arm.
- **Provenance:** checkpoint sha256 `2ad91428…0149d`; seeds + `PYTHONHASHSEED=0` to the manifest; MaxEnt
  tables sha256 `8b6d7b46…b7c00`; results write a manifest beside output (never overwrite the record).
- **Amendments:** pre-run only, logged `Amendment (date, pre-run, no data seen)`; once the primary is
  inspected a fix is *outcome-aware verification*, not confirmatory (IC1 A2).

---

## Closure disposition & must-fixes (2026-07-22 design red-team)

**Disposition: FIXABLE_THEN_RUNNABLE (survivor=NO).** Red-team synthesis:
[`../redteam/REDTEAM_SYNTHESIS_B2.md`](../redteam/REDTEAM_SYNTHESIS_B2.md) (5 independent adversarial lenses, run
before any confirmatory compute; formalize-audit-verify step 2). The SEMANTICS arm — the only arm that closes C30 —
is non-identifiable on a frozen checkpoint at MPS budget: a one-sided positive search re-runs the exact R22 failure
(held-out r −0.512→+0.017), and equivalence is unreachable. The steering direction `u_choice` is a position/distance
axis dominated by learned PE geometry (R20 re-entry), so EXISTENCE is near-tautological and the SPECIFICITY
discriminators are null/wrong-signed at pilot. Honest outcome: C30 should be scoped out of the paper; only a heavily
re-scoped, geometry-controlled EXISTENCE + partner-specificity positive search is the runnable remnant.

**survivor = NO — not executed this session.** SEMANTICS is deferred to a training-time / external-assay design; the
runnable remnant requires the redesign in the must-fixes below (not footnotes) before any freeze, and this session is
documentation finalization only. **Operational note:** the SpliceBERT.1024nt weights and the coupling data are absent
from this worktree, so any model run must be launched from the main checkout.

**Must-fixes (verbatim, from the synthesis):**

1. FATAL-2 (calibration off-channel): rebuild the planted-positive to inject a known scalar-scaled direction through B2's OWN net-new residual-add hook h_l(j)+=alpha*u_choice with a full re-propagated forward, requiring end-to-end recovery of the planted scalar; move M-alpha0 (exact zero-dose identity) and M-dose (monotonicity) onto that hook; dry-run the hook PP in --smoke. (run_a9 L218-227 synth branch injects via deliver()=dr@W_V^T on the L3H1 value path, NOT the interpreted channel.)
2. FATAL-1 (position/distance entanglement = R20 re-entry): caliper-match preferred vs less-preferred acceptors WITHIN each contrast on target_partner_distance AND absolute window position (and exon-length/CDS/GC); ADD an explicit absolute-position + donor-distance direction to the control family; add an on-path memorial that u_choice is ~orthogonal to the absolute-position subspace / survives a held-out position stratum. Resolve the raw-B-labels vs residualize-then-label contradiction.
3. Strip E_rel from the confound-specificity AND and LABEL IT FORCED (near-identity with −ΔB_j under single-window injection at j); rest specificity solely on E_partner and E_decoy.
4. Power E_partner explicitly as a difference-in-differences: reaching CI-HW<SESOI 0.05 needs ~900 gene-clustered sites (not the ~400 budgeted); add a hard preflight discovery-MDE gate that declares the contrast inconclusive BEFORE confirmatory spend if projected gene-clustered CI-HW > SESOI.
5. Add a hard preflight behaviour-existence gate: a partner-specific signed choice net of a matched sham must exist on a FRESH discovery draw before steering is licensed (currently inconclusive: alt3ss M_comp_minus_sham +0.0146 [−0.051,+0.088]).
6. Measure SD(ΔB_j) directly on a discovery injection pilot and re-project EXISTENCE power from it (not SD(M_comp)=0.31); fold the random-direction max-T penalty into the projected lower bound; declare EXISTENCE inconclusive if the on-manifold dose ceiling < SESOI_B.
7. Family-correct (FWER) the three SEMANTICS external variables X1/X2/X3; pre-register alpha at the family level.
8. Scope SEMANTICS out / defer: equivalence is unreachable at MPS budget and the one-sided positive re-runs the R22 multi-draw/dyadic wall; recalibrate SESOI_B to the phenomenon scale (~0.02–0.03); add a quantitative label-signal floor (restrict to sets whose native B_i−B_j CI-excludes-0).
9. Operational: run from the main checkout or symlink data/ (data/models is absent in this worktree); pin --model to an absolute path in the manifest.
