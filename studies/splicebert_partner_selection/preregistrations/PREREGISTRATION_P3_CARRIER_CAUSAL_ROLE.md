# PREREGISTRATION — P3: the causal role of the period-3 positional carrier in donor coupling (v2)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN-CANDIDATE v2 (post-4-lens design red-team). Not yet run confirmatorily.** v1 (the
pre-red-team draft) is superseded; the audit trail is `REDTEAM_SYNTHESIS_P3_4lens.md`. All four lenses
returned `fixable`, with **two estimand-level FATALs** that **retire the flagship "address × payload
binding" endpoint as non-identifiable (logged R23)**. This v2 reframes to what is identifiable and states
plainly what is deferred. Formalizes **P3.1–P3.3 as one prereg** per Cam's memo; **P3.4 (carrier origin) is a
separate, later-gated study** — no origin claim is reachable here.

Provenance: SpliceBERT.1024nt sha256 `2ad91428…0149d`, post-LN BERT-MLM 6L×16H, hidden 512, absolute
learned position embeddings P (1026×512, `ad.model.bert.embeddings.position_embeddings`). Env
`gfm-thrust1`; compute **MPS** (no CUDA). Deterministic cohorts (`PYTHONHASHSEED=0`, `sorted(set())`).

The name defended throughout (not "carrier wave" — spatial, not temporal; not a codon-frame or
splice-pairing code):
> **A learned period-3 positional carrier:** a rank-2 cyclic coordinate attention heads read as
> relative-offset classes.

---

## R23 (this prereg's founding correction) — "binding" and "independently-constructed payload" withdrawn

- **Δ_bind is architecture-forced.** L3H1's output is bilinear `O = A·V`, so the 2×2 DiD equals the identity
  `(A_I−A_D)(V_I−V_D)` — non-zero for *any* perturbation moving both A and V, with no carrier-specificity.
  A positive `Δ_bind` is therefore **not** evidence of "binding."
- **payload_source has no delivery-independent, carrier-uncontaminated observable** (carrier feeds the
  generator; MLP2 is dual address+payload, C29). "Independently-constructed payload" is **not licensable**.
- **Consequence:** P3.3 is reframed to an *A-vs-V routing decomposition on coupling* (conditional mediation,
  never "binding"); construction-vs-binding → **Endpoint B (unresolved)**. Coheres with R15 (position was
  already inside the value vectors), R21 (forced identities are not evidence), R22 (held-out ≠ sampling
  stability; semantic identity unresolved).

---

## 0. INTENT (north star — re-read at every gate)

*Does the learned period-3 positional carrier causally support splice-site RECOGNITION, positional
ROUTING, or payload construction/binding in SpliceBERT's donor–acceptor coupling circuit — and WHERE
(which sequence positions; which of L3H1's A-vs-V factors) must it be present?*

Model-native observable: the chain `local recognition → L2H15/L3H1 targeting (S_A) → donor-side U_info →
coupling (C)` under **intact / period-3-deleted / energy-matched-inert** carriers (P3.1), under
**position-selective restoration** (P3.2), and under the **L3H1 A-vs-V factorial** (P3.3, reframed). This is
the intended **last SpliceBERT causal study of the carrier**.

---

## 1. ESTIMANDS

Carrier conditions on P: **I** (intact), **D** (`make(BAND)`, BAND=`range(339,346)`; reproduces C8),
**E** (energy-matched control **band distribution** — width-7, spectral power within 25% of BAND, periods
3/1.5/6 excluded, **and** per-band comb-read ≥95% at runtime). Contrasts are the arm difference
`Δ(D)=X(I)−X(D)`, its specificity vs the **E distribution** (add-one p), and — for P3.2/P3.3 — normalized
recovery / factorial contrasts below.

### P3.1 — the chain (CONTEXT; routing is the maintained hypothesis, C32/C22)
- **L2 Targeting `S_A`** = `attn_to_window(attentions[3][:,1], donor_tok, acc_win)` (offsets −3..0). `ΔS_A(D)`.
- **L3 Payload `U_info`** — **defined once on the INTACT model** (top-k SVD of the clean−corrupt final donor
  residual, C16), then only **projected** onto (never refit in D — U_info(D) is an SVD of ~0 under M1 and is
  degenerate). What-vs-whether comparisons only in **non-degenerate (partial-recovery) arms**; estimability
  gated by an `energy_spectrum` floor.
- **L4 Coupling `C`** = `margin_hard(clean) − margin_hard(corrupt)`; corruption = partner AG→CC at `[a−2,a−1]`
  (real bases, never `[MASK]`). `ΔC(D)` reproduces C8 (0.946→0.027).
- **L1 Recognition `D_rec` — DEFERRED** (see §9): a valid recognition assay needs a strong-motif pseudo-site
  decoy (new machinery), a position-invariant contrast, importance-matched E, and ~1000+ matched pairs;
  foundation sanity shows `real_acceptor_decoy` available for only 1/16 examples (decoy-sparse). Not run here.
- **P3.1 map (non-exhaustive; C32 makes routing maintained):** P3.1 tests only whether recognition
  *additionally* collapses; with D_rec deferred, P3.1 confirms {ΔC(D) large, ΔS_A(D) large} — i.e. the
  routing address collapses — and provides chain context, **not** a novel recognition/routing dissociation.

### P3.2 — the carrier's causal LOCUS (CONFIRMATORY centerpiece; Cam's "where must it be?")
In the **deleted** model, restore the carrier delta `Δ = P_intact − P_deleted` (removed period-3 component,
token-position-indexed) at per-example window `a`:
1. **donor window** (donor ± w) 2. **acceptor window** (acc ± w) 3. **donor + acceptor**
4. **mod-3-phase-matched irrelevant window** (splice-free, distance/GC-matched, `(pos−donor)%3` matched — the
   specificity null) 5. **decoy-acceptor window** (non-partner AG at matched distance+phase — splice-like local
   content, wrong identity) 6. **all positions** (full-restore **memorial**, forced ≈ intact; R21).

Readout: **coupling C only** (robust denominator `C_I−C_D ≈ 0.68`). `R_C(a) = (C_a − C_D)/(C_I − C_D)`, as a
**ratio of means** with `gboot_ratio` (never mean-of-ratios). Per-example difference `C_a − C_D` also reported.

### P3.3 — A-vs-V routing decomposition at L3H1 (SECONDARY; conditional mediation, NOT binding)
Factorial on L3H1 via `capture_proj`/`patch_proj`: patch head-1's `query,key` to carrier-x's captured
projections (⇒ attention pattern `A_x = softmax(Q_x·K_x)`) and `value` to carrier-y's (⇒ `V_y`), on the
**deleted** base network, at all nucleotide positions. Cells `(A_x,V_y)`, x,y∈{I,D}. Readout: **coupling C**
(S_A removed — `S_A=f(A)` is forced; Δ_bind(S_A)≡0) **and** the **pre-saturation** U_info-projection magnitude
(hard margin saturates). Per-example **paired** contrasts:
- routing main effect `= C(A_I,V_D) − C(A_D,V_D)`; value main effect `= C(A_D,V_I) − C(A_D,V_D)`;
- `Δ_bind_i = C(A_I,V_I) − C(A_I,V_D) − C(A_D,V_I) + C(A_D,V_D)` (one gene-resample over the per-example vector).
- **Carrier-specificity of the interaction (the R23 fix):** repeat the factorial toward each **E control band**;
  require `Δ_bind(D)` CI-**above the E-band distribution** (add-one p). Also report the bilinear floor
  `‖(A_I−A_D)(V_I−V_D)‖` at L3H1's output.
- **Axis is "A vs V", NOT "address vs payload"** (V carries position — R15). Interpreted as conditional
  mediation only: patching intact A or V does not prove those states are independently generated.

---

## 2. METHOD (reused machinery + two new, piloted plumbing pieces)

Reused/validated: deletion + energy-matched control + DiD + health (`run_carrier_surgery`); rescue +
size-matched nulls (`run_carrier_mediation`, `hf_bert`); targeting (`address.attn_to_window`); U_info
(`subspace`, `unembed.margin_hard`); coupling/margins/health (`circuit.CouplingTask`); ratio CI
(`circuit.gboot_ratio`). Adapter loaded **`eager=True`** (SDPA silently zeroes attentions).

New plumbing (piloted in smoke, must pass before freeze):
- **P3.2 position-selective restoration** — a **batch-aware** forward hook on `position_embeddings` adding
  `Δ` at per-example window token rows (validated: element-wise construction anchor + monotone dose-response).
- **P3.3 A-vs-V** — `capture_proj(ad,3,{query,key,value})` per batch under I/D, then nested
  `patch_proj(ad,3,1,·)` at all nucleotide positions. Patching *inputs* and letting the module run natively
  avoids manual-softmax / double-√d / double-`W_O` (one `W_O` via the module's `attention.output.dense`).

**EXCLUDED (R14):** no ±1/±2 head-roll or phase-shift assay (stride-3 geometry manufactures effects).

---

## 3. MEMORIALS (on the measurement path; each CAN fail)

| # | Memorial | Passes iff | Note |
|---|---|---|---|
| M1 | deletion collapses coupling | `C(D)/C(I) ≤ 0.05` (C8) | validates deletion; failable |
| M1_Y | per-Y deletion-effect | denom `Y_I−Y_D` gene-CI excludes 0 (Fieller-safe) AND ≥ SESOI_Y | **R_Y reported only if M1_Y passes**; else difference-scale. **R_C only** (robust) |
| M2 | energy-matched inert | `C(E)/C(I) ≥ 0.90`, per-band comb-read ≥95% | E distribution; failable |
| M3 | health gate (ALL states) | every interpreted arm/**hybrid**/**cell** keeps ≥0.5 MLM headroom over chance (R13) | below-floor → dropped |
| M4 | A/V baseline identity | patched `(A_D,V_D)` reproduces unpatched deleted to **<1e-6** | diagonal anchor |
| M4′ | **off-diagonal native anchors** | recomposed `A_D@V_I` == deleted-forward-with-value-overwrite, `A_I@V_D` == intact likewise, **<1e-6**; **head-identity**: patched `(A_D,V_D)` head-1 == `capture_head_outputs(deleted)[3][:,1]` | the R21-fix; validates the decisive path + pins head-1=L3H1 |
| M5 | full-restore | `R_Y(all) ≈ 1` | **LABELED FORCED** (R21) |
| M5′ | **P_hybrid construction** | element-wise `P_hybrid[pos]==P_intact[pos]` in window, `==P_deleted` else | failable (off-by-one/wrong-axis) |
| M5″ | **monotone dose-response** | `R_C(w)` empty~0 → all~1, strictly increasing in w | the real selective-restoration test |
| M6 | null-restore inert | mod-3-phase-matched irrelevant `R_C ≈ 0`; size-matched null-head restorations < real (R12) | specificity |
| M8 | no phase-roll | assay has no comb-roll op | construction lint (not evidence) |
| M9 | determinism | PYTHONHASHSEED pinned; `sorted(set())`; cross-seed cohort identity | reproducibility (IC1 A2) |

---

## 4. CALIBRATION (sensitivity + specificity; hard preflight gates)

**Planted-positive (each travels the real channel):**
- **PP1 (chain):** deletion→coupling collapse (M1).
- **PP-SA (targeting):** a manipulation known to move `S_A` (corrupt/relocate the acceptor window) must change
  `attn_to_window` by the expected amount — calibrates the S_A readout.
- **PP-U (delivery):** the C16/C17 planted positive — patch the clean-U component into corrupt recovers ~0.9
  vs a random rank-8 subspace ≤0.05 — calibrates the U_info readout.
- **PP3 (restoration plumbing):** restore-all → `R≈1` — **LABELED FORCED** (M5); the genuine sensitivity test
  is M5″ (monotone dose-response).

**Null / specificity (must not fire above target FPR):**
- **N1:** energy-matched control **distribution** does not collapse C/S_A (M2; add-one p, not a single band — R12).
- **N2:** mod-3-phase-matched irrelevant window `R_C ≈ 0`; decoy-acceptor window `R_C` < real acceptor window.
- **N3:** size-matched null-head restorations ≥400 draws (non-exhaustible); exact floor reported for exhaustible.

**Confound-specificity (importance-matched, the identifiability fix):** the generic-damage counterfactual is
the **E-band distribution spanning importance**; a treatment effect (ΔS_A, or the P3.3 Δ_bind) counts as
carrier-specific only if it **exceeds the whole E distribution** (add-one p), never merely a junk band.

---

## 5. IDENTIFIABILITY

- **R23 (fatal-1, fatal-2):** binding + independent-payload withdrawn (see header). P3.3 = conditional
  mediation; construction-vs-binding = Endpoint B.
- **R15 (position-in-V):** the A/V axis is not an address/payload axis; the value main effect is not a pure
  payload effect. Reported as "A vs V," caveated.
- **U_info(D) degeneracy:** U_info defined once on intact, only projected; overlap/what-vs-whether only where
  coupling is non-degenerate; `energy_spectrum` floor gate.
- **Routing not falsifiable (C32):** stated as maintained; P3.1 tests only additional recognition collapse.
- **P3.2 artificial-states / period-3 phase:** irrelevant control **mod-3-phase-matched**; decoy-acceptor
  control; `R_C` only where the denominator is robust (M1_C).
- **Held-out ≠ sampling stability (R22):** any selected window/path/threshold confirmed on an untouched,
  disjoint gene split; gene-clustered CIs; effect-existence uses CI-excludes-0 on the paired difference.

---

## 6. DECISION RULE (frozen; result → conclusion)

**Per-Y SESOI (own scale, discovery-fixed):** SESOI_C ≈ 0.05 nat (~5–10% of the C8 effect, matching IC1);
SESOI_{S_A} as a fraction of intact acceptor-window mass; **R_C dual threshold** — a window "recovers" iff
`R_C ≥ 0.2` **AND** absolute `C_a−C_D ≥ absolute-SESOI_C`. A NULL/equivalence call requires **CI half-width <
SESOI** (else "inconclusive").

**Two families, correctly composed:** (i) **positive** family — one-sided `H0: effect ≤ SESOI`, **max-T**
gene-bootstrap adjusted lower bound > SESOI (cells share genes ⇒ max-T valid + powerful); (ii)
**equivalence** family — TOST at the family-adjusted α, adjusted CI ⊂ ±SESOI. Every "arm D vs E" test is the
**paired within-example difference** CI, not non-overlap of two arm CIs.

**P3.2 locus map (on `R_C`):** donor-window recovers → query/**address origin**; acceptor-window recovers →
key-side address or acceptor-state; both required → **relational** coordinate; only all-positions (all local
`R_C≈0`) → **distributed upstream organization** (all-positions ≈1 is memorial M5, not evidence).

**P3.3 A-vs-V (on coupling):** routing main effect (A_I,V_D restores C) dominant → carrier's coupling
contribution routes predominantly through the **attention pattern**; value main effect dominant → through the
**values** (caveat R15); carrier-specific `Δ_bind` (CI-above the E distribution) → a super-additive A×V
contribution **beyond the bilinear floor** — reported as conditional mediation, **not** binding.

**Endpoints (reframed; both legitimate; either closes the SpliceBERT carrier account):**
- **A (locus resolved):** *"The learned period-3 positional carrier causally supplies a relative positional
  routing coordinate whose causal locus is {donor / acceptor / relational}, and whose contribution to donor
  coupling routes predominantly through L3H1's {attention pattern / values}."* — reached iff P3.2 localizes
  recovery AND (if run) P3.3 resolves the A-vs-V split, health-gated and specificity-passing. **No "binding"
  claim; no payload-semantics claim.**
- **B (unresolved / distributed):** *"The carrier is necessary for coupling and read by the identified
  circuit, but its causal locus and its recognition-vs-routing-vs-binding contribution remain unresolved."*
  — reached on only-global recovery, mixed/partial patterns, or underpowered CIs.

Mixed/partial patterns and any inconclusive family member → **Endpoint B**. Construction-vs-binding is
**non-identifiable (R23) → always Endpoint B**.

---

## 7. POWER / MDE & HELD-OUT

- **Power:** per-Y MDE derived from a discovery pilot. Binding constraints: the P3.3 `Δ_bind` interaction
  (second difference; ~4× a main-effect n; paired reduces this) and the E-distribution comparison. **Every
  NULL/Endpoint-B call must be a powered equivalence (CI half-width < SESOI), not a wide CI** (IC1/R22 lesson).
  Cohort sized to the compute budget; if a contrast is underpowered it is reported *inconclusive*, never null.
- **Held-out:** discovery split (subset of val genes: chr11/16/19) sets k, w, SESOI, decoy/window pools, health
  floor; confirmatory on a **disjoint, untouched** val-gene split; **untouched-gene confirmation** after any
  window/path selection. Sealed test (chr8/9) **not opened**. Deterministic cohort; gene-clustered CIs.

## 8. GATES
- **Global PASS = AND(** M1..M9, M1_Y, M4′, M5′, M5″, PP1, PP-SA, PP-U, N1..N3, confound-specificity **).** Run
  STOPS on failure; below-floor states (M3) dropped.
- **`--smoke`:** the new-plumbing pilots (M4/M4′ off-diagonal + head-identity; M5/M5′/M5″; N2 inertness;
  determinism) on a tiny cohort, **val sealed** — the "pilot the restoration plumbing" step; must pass before freeze.

## 9. SCOPE
- **Reaches:** OV/mediation + behaviour level for the carrier's causal **locus** and its **A-vs-V routing
  decomposition** in donor coupling.
- **Does NOT establish:** address × payload **binding** (R23, non-identifiable); an **independently-constructed
  payload** (R23); **recognition** (D_rec deferred — needs a pseudo-site-decoy study); the carrier **origin**
  (P3.4, separate); a **biological reading-frame** claim.
- **Deferred, documented:** (a) **D_rec recognition** — pseudo-site decoy generator + position-invariant
  contrast + importance-matched E + ~1000 pairs; (b) **construction-vs-binding** — non-identifiable here.
- **Provenance:** checkpoint sha; seeds + PYTHONHASHSEED to the manifest; results write a manifest beside output.
- **Amendments:** pre-run only, logged `Amendment (date, pre-run, no data seen)`; once the primary is inspected a
  fix is *outcome-aware verification*, not confirmatory (IC1 A2).
