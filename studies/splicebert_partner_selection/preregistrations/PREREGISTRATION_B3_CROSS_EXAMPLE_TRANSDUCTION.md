# PREREGISTRATION — B3: Cross-example transduction (the generalization gap) (v1)

**Status: NOT FROZEN — the 5-lens design red-team (2026-07-22) adjudicated this draft FIXABLE_THEN_RUNNABLE (survivor=NO).**
See [`../redteam/REDTEAM_SYNTHESIS_B3.md`](../redteam/REDTEAM_SYNTHESIS_B3.md). Two convergent fatals, both fixable
(this is NOT a P3.4-style in-principle dead end — the cross-input interchange IS identifiable and MPS-feasible after a
bounded redesign). FATAL-1 object/space contradiction: §1 injects Π_U of the source's FINAL donor residual ([8,512])
"through the L3H1 head-output path," but that path delivers a [B,32] head output of the LAYER-3 acceptor-window
residual — distinct objects at different layers (C24: payload is NOT concentrated in U_info; L3H1 converts). FATAL-2
backdoor: U_info IS the clean−corrupt axis by construction, so a generic clean-donor offset rides through all four
controls; needs a same-source-CORRUPT control (clean-source > corrupt-source, not > empty). Recommended v2: transplant
the acceptor payload cross-example, DROP Π_U, own R22 as transferability-of-STATE (not semantics), add the
corrupt-source control, add a net-new two-way source×target dyadic ratio bootstrap, and run a discovery PILOT
measuring the cross-example numerator + its dyadic HW before freeze. This v1 is retained as the **design of record**;
it must **NOT be run** as written — the disposition and the verbatim must-fixes are recorded in "## Closure disposition
& must-fixes (2026-07-22 design red-team)" at the foot of this file. **Not executed this session:** the
SpliceBERT.1024nt weights and the coupling/atlas data are absent from this worktree, so any model run must be launched
from the main checkout.

This v1 is the design of record submitted to a pre-compute 4-lens red-team; it must **NOT** be run
confirmatorily until (a) the red-team clears the identifiability items in §5, (b) the `--smoke` plumbing
pilot in §8 passes, and (c) the net-new cohort-adapter in §2 is validated against the coupling loader. It
promotes the **already-validated** donor-side readout `U_info` (C16/C26) from a *within-example mediator*
to a candidate *transferable, interventionally-general direction* — the first missing bridge in MECHANISM
(`docs/ROADMAP.md` Track 1). Because the transplanted object is validated, B3 is **lower interpretive-risk
than a new-direction steering study (B2)**: it adds infrastructure (cohort matching, four controls, dyadic
inference), not a new semantic claim.

Provenance: SpliceBERT.1024nt sha256 `2ad91428…0149d`, post-LN BERT-MLM 6L×16H, hidden 512, **absolute**
learned position embeddings P (1026×512). Env `gfm-thrust1`; compute **MPS** (no CUDA). Deterministic
cohorts (`PYTHONHASHSEED=0`, `sorted(set())`). Adapter loaded **`eager=True`** (SDPA silently zeroes
attentions; B3 reads attentions for the address hold, C22/A8).

The object under test (defended, validated, NOT redefined):
> **`U_info`** — a stable rank-8 direction in SpliceBERT's **final donor residual**, defined as the top-k
> right-singular subspace of the clean−corrupt donor-residual difference (`subspace.top_k_subspace`, C16),
> whose clean component causally mediates donor–acceptor coupling (rank-8 sufficiency ≈0.92 vs matched
> random ≤0.05, C16/C17) and which **generalizes across disjoint genes** (rank-1 overlap 0.970, held-out
> rank-8 sufficiency **+0.875**, random ≤0.05; C26 `a5_uinfo_replication.json`). **`U_info` is the
> donor-side DELIVERED representation, NOT a semantic "partner identity" (C16 scope note; R20).**

---

## B3.0 (founding guards) — what this study may and may NOT claim

- **R20 (identity-vs-geometry):** `U_info` is a donor-side readout direction. B3 tests whether that STATE
  transfers as a causal variable (Geiger et al. 2023/2025 interchange across inputs); it does **NOT** claim
  `U_info` semantically *means* "the partner is intact." The ROADMAP's "portable partner-intact signal"
  phrasing is the delivered-representation reading only (C16 scope), never a concept identification.
- **R22 (held-out ≠ sampling stability; dyadic reuse):** C26's cross-gene result is a within-sample
  discovery/held-out split. B3's effect-EXISTENCE claim requires **multiple independently-seeded source↔target
  pairings** (not one D/H split) **and** two-way **gene-dyadic** inference, because each gene is reused as a
  source for many targets and a target for many sources (`dyadic_ci`, `run_a9b_payload_semantics_composition.py`).
- **R7/R21 (forced identities validate nothing):** the within-example self-injection ceiling (source=target)
  and the full-U-component restore are **partly plumbing anchors** — labelled FORCED where they are; the
  genuine test is the *cross-example, matched-source* transplant and its **dose-ordering** against the three
  specificity nulls, none of which is architecture-forced.
- **Evidence ladder:** B3 reaches **transferable OV/mediation → generalization** (behaviour-level, coupling).
  It does **NOT** reach biology, nor a payload-semantics claim (R22 — payload semantic identity is
  unresolved; B3 injects the donor-side `U_info` component, not a strength label).

---

## 0. INTENT (north star — re-read at every gate)

*Is the donor-side `U_info` state a transferable, interventionally-general causal variable, or an
example-local mediator?* Concretely: if `U_info` is derived ON DISCOVERY GENES and its clean ("partner
intact") component from a **matched SOURCE** example is injected at the donor of a **gene-DISJOINT,
covariate-MATCHED TARGET** example whose partner has been destroyed, does the target's native donor margin
move back toward its clean value in a **dose-ordered** way, **above** three specificity nulls, at a
recovery approaching the within-example ceiling — and does that survive **multi-draw** resampling and
**gene-dyadic** inference?

Model-native observable: the target's **donor coupling margin** `M = logit(G) − max(logit A,C,T)` at the
masked donor GT (`circuit.CouplingTask.margin`), read under injection at the donor token of a source-derived
`U_info` component, on the target's **corrupt** run (partner AG→CC at `[a−2,a−1]`, `CORRUPT_ALT="CC"`). This
is the cross-input INTERCHANGE the causal-abstraction view demands: a genuine causal variable supports
interventions ACROSS inputs, not only within a matched clean/corrupt pair.

---

## 1. ESTIMANDS

All quantities are per-target normalized recoveries on the coupling-margin scale. For a target example `i`
run corrupt, define three injected-margin anchors at the donor token `d_i`:

- **`M_empty(i)`** = corrupt margin with **no** injection (`dr = 0`; lower anchor).
- **`M_self(i)`** = inject `i`'s OWN clean `U_info` component (source = target; the within-example CEILING;
  this is the C16 within-example patch — a **plumbing/positive anchor**, partly forced).
- **`M_xfer(i, j)`** = inject the `U_info` component of a **matched cross-gene SOURCE** `j` (gene(`j`) ≠
  gene(`i`), `j` in the discovery pool, matched to `i` per §5).

The injected content is `Π_U[ r_src ]` — the discovery-`U_info` projection (`subspace.project`, `P = UᵀU`)
of the source's **clean final donor residual**, delivered to the target donor head output through the
target's **held-clean address** exactly as in C16/A8 (`payload.head_output` / `inject_capture` /
`patch_heads`, cross-example is already supported — `patch_heads` performs only a shape check, source
content need not share the example).

**Primary estimand — transfer recovery (ratio of means, R20-safe, dyadic):**
> `R_xfer = ⟨M_xfer − M_empty⟩ / ⟨M_self − M_empty⟩` (ratio of means over reliable targets, `gboot_ratio`;
> never mean-of-ratios). **Units:** dimensionless recovery fraction (0 = no transfer, 1 = full within-example
> ceiling). **Sign:** positive (destroying the partner drops the margin; restoring the `U_info` component
> should raise it). **Reference point:** the target's own self/empty anchors.

**Primary estimand — DOSE ordering (the non-forced core):**
> Inject `α · Π_U[r_src]` at doses `α ∈ {0, ¼, ½, 1, 1½}` (α=1 = native component magnitude). The **dose
> slope** `β_dose = d M_xfer / dα` (per-target OLS slope, gene-clustered) must be **CI-low > 0** and
> **monotone** (`M_xfer(α)` non-decreasing across the grid). Dose-ordering, not a single point, is the
> transfer signature (R21: a single point could be a norm artifact).

**Specificity contrasts (each a paired within-target difference; the effect must EXCEED these):**
- `Δ_rand  = M_xfer − M_rand`  — vs a **norm-matched random rank-8 subspace** (C16 null; ≥400 draws).
- `Δ_scr   = M_xfer − M_scr`   — vs a **scrambled source** (source residual coordinate-permuted before Π_U).
- `Δ_pos   = M_xfer − M_pos`   — vs **non-target-position** injection (same content at donor±k / a random
  non-donor token; positional specificity).

All three CI-lows (gene-dyadic) must exceed 0 by ≥ SESOI for a POSITIVE transfer call.

---

## 2. METHOD (reused validated machinery + one net-new cohort adapter, both piloted)

**Reused / validated (exact paths):**
- **`U_info` builder:** `frozen_uinfo(...)` in `scripts/splicebert/run_a9_payload_semantics.py` →
  `gfmcd.splicing.subspace.top_k_subspace` (rank-8 SVD of clean−corrupt final-donor residuals), computed
  **ON DISCOVERY GENES ONLY** (C16/C26 protocol). `subspace.project` / `payload.projector` (`P=UᵀU`) for Π_U.
- **Cross-example injection:** `gfmcd.splicing.payload.{value_weight_head, deliver, head_output,
  inject_margin_donor}` + `run_a9_payload_semantics.py::inject_capture` + `circuits.hf_bert.patch_heads`
  (source content need not share the example — verified: `patch_heads` has only a shape check). The
  target's clean **address** `A⁺` and base `A⁺·V⁻` are captured as in A8/A9 (`caprun`), so the address is
  held-clean and only the value payload varies (the A8 reconstruct-inject, cross-example).
- **Coupling / margins / corruption / health:** `circuit.CouplingTask` (`.margin`, `.pair`, `.margins`,
  `.mlm_health`; corrupt = partner AG→CC at `next_acceptor_pos−2`, `CORRUPT_ALT`). Example loader
  `circuit.load_examples` (`min_effect=None` for existence; conditional set drawn from a defined pool).
- **Inference:** `circuit.gboot` / `circuit.gboot_ratio` (gene-clustered ratio-of-means CI);
  `run_a9b_payload_semantics_composition.py::dyadic_ci` (two-way gene Poisson-multiplier bootstrap — the
  R22 reuse fix). `subspace.random_subspace` (norm-matched random-direction null).
- **Matched cohort scaffold:** `results/splicebert_head_context_atlas/atlas_context_matched_v7.*` — a
  gene-hash-disjoint discovery/validation split (validation side **in-repo**: 2,247 pairs / 4,019 packets /
  1,486 genes; categories `single_pair` 237, `donor_cluster` 179, `acceptor_cluster` 158, `cassette_exon`
  569, `multiple_independent_pairs` 380, `complex_event` 724; per-packet covariates `target_strength`,
  `partner_strength`, `target_partner_distance`, `gc_target_local50`, `gc_window`, `target_in_cds`,
  `distance_mod3`, `target_role` …). Discovery side gitignored but **regenerable** via
  `scripts/splicebert_head_context_atlas/build_atlas_matched_cohorts.py` (propensity-caliper matching).

**Net-new machinery (piloted in `--smoke`, must pass before freeze):**
- **Cohort adapter `atlas_donor_packets → CouplingExample` pairs.** The atlas packets are MLM masked-base
  packets (`target_role='donor'`), NOT `CouplingTask` short-intron pairs. The adapter maps a donor-role
  packet to a `CouplingExample` (donor pos + `next_acceptor_pos` + short-intron eligibility gate from
  `circuit.eligible`), then builds **gene-disjoint source↔target matched pairs** on the coupling-native
  covariates (§5). **Fallback (frozen path if the adapter loses too many packets):** derive the cohort
  directly from `load_examples` (val split), partition genes by the same gene-hash used by the atlas, and
  caliper-match on covariates computed by `gfmcd.splicing` scorers + `CouplingTask` — using the atlas ONLY
  for the disjoint-split definition and the context-class tags. The frozen path is chosen at pilot by
  whichever yields ≥ n_min balanced pairs (§7); both keep every injection on validated primitives.
- **Four-control driver** wiring `M_self / M_empty / M_xfer / M_rand / M_scr / M_pos` over the dose grid as
  one batched `interchange`-style loop (a driver over existing primitives, not a new mechanism).

**Modest first version (de-risks the paper most):** primary context class = `single_pair` (cleanest,
n=237 val pairs); optional **one** additional class (`cassette_exon` or `donor_cluster`) reported as a
context-generality check, NOT pooled into the primary estimand.

**EXCLUDED:** no new learned direction (B3 uses only the frozen `U_info`); no refit of `U_info` on target
genes (frozen, projected-onto only — the C16/A9 discipline).

---

## 3. MEMORIALS (on the measurement path; each CAN fail)

| # | Memorial | Passes iff | Note |
|---|---|---|---|
| M1 | value-linearity (post-LN exactness) | `‖(V⁺−V⁻) − W_V(r⁺−r⁻)‖_max < 1e-6` at the injection window | exact arithmetic anchor (A9 `value_linearity`=2.7e-6); **failable** (wrong `W_V` slice / axis) |
| M2 | empty == unpatched corrupt | `M_empty` reproduces the plain corrupt margin to **< 1e-6** | diagonal plumbing anchor; **failable** (dr≠0 leak) |
| M3 | self-ceiling recovers | within-example `R_self = ⟨M_self−M_empty⟩/⟨M_clean−M_empty⟩` ≈ C16 (≳0.8) | **partly FORCED (R21)** — validates the injection path, not transfer; the genuine test is R_xfer/β_dose |
| M4 | cross-example content actually differs | for xfer pairs, injected `Π_U[r_src] ≠ Π_U[r_tgt]` (‖Δ‖ CI-excludes 0) | else "cross-example" is secretly self; **failable** |
| M5 | dose construction | injected magnitude at α scales exactly `α·‖Π_U[r_src]‖` (element-wise) | construction anchor; **failable** (off-by-one dose) |
| M6 | monotone dose-response | `M_xfer(α)` non-decreasing on the grid; `β_dose` slope defined | the real transfer test (not forced) |
| M7 | gene-disjointness | `gene(src) ∩ gene(tgt) = ∅` AND U_info-train genes ∩ target genes = ∅ | **failable** (leakage) — halts run |
| M8 | health gate (every arm) | each interpreted arm keeps ≥0.5 MLM headroom over chance (R13, `mlm_health`, MASKED probes) | below-floor arm dropped |
| M9 | determinism | `PYTHONHASHSEED=0`; `sorted(set())`; cross-seed cohort identity | reproducibility (IC1 A2) |

---

## 4. CALIBRATION (sensitivity + specificity; hard preflight gates)

**Planted-positive (each travels the real channel):**
- **PP-U (delivery sensitivity):** the C16/C17 planted positive — patch the clean-U component into corrupt
  (within-example) recovers ≈0.9 vs a random rank-8 subspace ≤0.05. Calibrates the `U_info` readout + the
  injection path (M3).
- **PP-xfer (transfer sensitivity, synthetic):** construct a synthetic source whose donor `U_info`
  component is a KNOWN scalar multiple `c` of the target's clean component; injecting it must move
  `M_xfer` by the predicted amount, monotone in `c` (the A9 `synth=(Vp,u,beta)` override path already
  injects a pure population direction — reuse it). Confirms the four-control driver can DETECT a genuine
  cross-example transfer at a planted magnitude.

**Null / specificity (must NOT fire above target FPR):**
- **N-rand:** norm-matched random rank-8 subspace → `Δ_rand > 0`, i.e. random `R ≈ 0` (≥400 draws;
  `random_subspace`). Specificity of the direction.
- **N-scr:** scrambled source (coordinate-permuted residual before Π_U) → `Δ_scr > 0`, scrambled `R ≈ 0`.
  Specificity of the SOURCE CONTENT (not just the U-subspace geometry).
- **N-pos:** non-target-position injection → `Δ_pos > 0`, off-donor `R ≈ 0`. Positional specificity.
- **N-permute-pairing (dyadic FPR):** permute the source↔target matching (random source assignment within
  the matched pool) → the permuted `R_xfer` distribution must have **≤5% FPR** for the observed effect
  (the R22 sampling-stability calibration, at the dyad level).

**Confound-specificity:** a positive transfer counts only if `R_xfer` exceeds ALL THREE nulls
(`Δ_rand, Δ_scr, Δ_pos` CI-low > SESOI) **and** the matched pool passes common support (§5). An effect that
matches a norm-matched random direction is a norm artifact, not transduction.

---

## 5. IDENTIFIABILITY

- **The core confound: is transfer a genuine causal-variable interchange, or a generic "big donor push"?**
  Separated by the three specificity nulls (norm-matched random, scrambled-source, non-target-position),
  which are architecturally NOT forced. If `R_xfer` is statistically indistinguishable from `R_rand`
  (CIs overlap) → **NOT identifiable as content-specific transfer → report as a norm/positional effect,
  not transduction** (the honest non-identifiable outcome).
- **Matching vs definitional entanglement.** Source and target are matched on {donor site-type, native
  donor margin/strength (`target_strength`), partner distance (`target_partner_distance`), coding status
  (`target_in_cds`), local GC/composition (`gc_target_local50`, `gc_window`), and injected-component norm
  `‖Π_U[r_src]‖`}. If matched genes are TOO similar (matched on margin ⇒ `U_info` components near-identical),
  transfer is trivial; if too dissimilar, no common support. **Predefined overlap check:** standardized mean
  difference |SMD| < 0.10 on every match covariate AND ≥ n_min pairs with `‖Π_U[r_src]‖` within ±15% of
  `‖Π_U[r_tgt]‖` (norm-matched) — else the contrast is under-supported and **reported non-identifiable**,
  never forced (cf. A9b V4 non-identifiable honesty). M4 additionally requires the injected content to
  actually DIFFER from the target's own component, so a "successful transfer" cannot be secret self-injection.
- **`U_info` is donor-side, not semantic (R20).** The transferable object is the delivered representation.
  A positive result is "the donor-side state supports cross-input interchange," NOT "we identified a
  partner-intact concept." Payload semantic identity stays UNRESOLVED (R22).
- **Reuse ⇒ dyadic (R22).** Each gene is source-for-many and target-for-many; effect-existence uses
  two-way gene-dyadic bootstrap (`dyadic_ci`) + multiple independently-seeded pairings, never a single
  within-sample D/H split or permutation-on-correlation (which under-covers with reuse).
- **Post-LN linearity** underwrites the exact reconstruct-inject (M1); the address is held clean (A8), so
  only the value payload varies — the isolate the injection targets.

---

## 6. DECISION RULE (frozen; result → conclusion)

**SESOI (recovery scale, discovery-fixed):** `SESOI_R = 0.20` recovery (matches P3.2's `R_C≥0.2` recovery
bar and the IC1/A9 scale). A NULL/equivalence call requires **CI half-width < SESOI_R** (else
"inconclusive"). Dose: `β_dose` positive iff its gene-dyadic CI-low > 0 AND the grid is monotone.

**Family control:** the POSITIVE family = {`R_xfer` > SESOI, `β_dose` > 0, `Δ_rand` > SESOI, `Δ_scr` >
SESOI, `Δ_pos` > SESOI} composed with a **max-T** gene/dyad bootstrap (shared genes ⇒ max-T valid); the
EQUIVALENCE family = TOST at family-adjusted α (adjusted CI ⊂ ±SESOI_R). Every arm-vs-arm test is the
**paired within-target difference** CI, not non-overlap of two arm CIs.

**Endpoints (either closes the generalization-gap question):**
- **A (transfer established):** *"The donor-side `U_info` state is a transferable causal variable: a
  discovery-derived `U_info` component injected from a matched cross-gene source produces dose-ordered,
  content-specific recovery of the target's native donor margin (`R_xfer` ≥ SESOI, `β_dose` > 0), above the
  random-direction, scrambled-source, and non-target-position nulls, stable across seeded pairings and
  gene-dyadic inference."* — reached iff the full POSITIVE family passes, health-gated, common-support
  passing, multi-draw stable. Reaches **transferable OV/mediation → generalization** (behaviour/coupling),
  NOT biology, NOT payload semantics.
- **B (null / instrument-limited):** *"`U_info` does not demonstrably transfer across gene-disjoint matched
  targets at this detection floor"* — reached iff `R_xfer` ⊂ ±SESOI_R as a POWERED equivalence (CI
  half-width < SESOI_R), reported at the licensed MDE (Track 1 exit: an honest null at a stated floor).
- **C (non-identifiable):** matched pool fails common support (§5 |SMD| or norm bracket), `R_xfer` ≈ `R_rand`
  (content not separable from a norm push), or M3/M4 fail ⇒ *"cross-example transduction of `U_info` is not
  identifiable on this cohort"* — honestly scoped, not forced.

Mixed/partial patterns, any inconclusive family member, or an under-powered CI → **Endpoint B or C** per
which gate missed. No positive transfer claim survives a failed specificity null or a failed dyadic FPR.

---

## 7. POWER / MDE & HELD-OUT

- **Feasibility contrast with P3.4 (the MPS anchor).** P3.4 stalled because its equivalence bar was
  **0.02 nats** — unreachable (C8 raw-mean gene-clustered CI half-width 0.0198 at n=384). B3's SESOI is on
  the **recovery RATIO** (`SESOI_R=0.20`), a far coarser bar, and the underlying coupling signal is large
  (`C_I−C_D ≈ 0.68`; C16 cross-gene sufficiency **+0.875** vs random ≤0.05 — a ~0.87 gap). The transfer
  effect, if present, is big; a powered equivalence (half-width < 0.20) and a positive dose-slope are both
  reachable at MPS budget. **This is why B3 is feasible where P3.4 was not.**
- **Compute budget.** Per matched target: 6 arms × 5 doses ≈ 30 batched corrupt forwards (nulls collapse to
  ≥400 random draws once, amortized). At ~200–400 targets × a few matched sources each ≈ 10⁴–10⁵ forward
  passes — the A9 interchange (n=240 over ≤400 pairs) ran in this regime on MPS. Size the transplant N to
  the budget; if a contrast is under-powered it is reported **inconclusive**, never null.
- **Dyadic power.** The two-way gene-dyadic bootstrap widens CIs vs one-way clustering (reuse); target ≥
  n_min ≈ 150–200 matched pairs over ≥ ~100 genes so the dyadic half-width on `R_xfer` clears SESOI_R. The
  MDE is reported at pilot from the discovery split; **every NULL/equivalence call must be a powered
  equivalence** (IC1/R22 lesson).
- **Held-out + multi-draw.** Discovery genes fix `U_info`, calipers, dose grid, SESOI, norm bracket, health
  floor, and the context class. Confirmatory on a **disjoint, untouched** target-gene split. Because
  held-out ≠ sampling stability (R22), effect-EXISTENCE additionally requires **≥3 independently-seeded
  source↔target pairings** (demonstrably disjoint gene assignments) — the effect must persist across draws,
  not merely on one split. Gene-dyadic CIs throughout. Sealed test (chr8/9) **NOT opened**.

## 8. GATES

- **Global PASS = AND(** M1–M9, PP-U, PP-xfer, N-rand, N-scr, N-pos, N-permute-pairing, confound-specificity,
  common-support (§5) **).** Run STOPS on failure; below-floor arms (M8) dropped; a failed identifiability
  gate (common support / M4) **demotes to Endpoint C**, it does not fake a positive.
- **`--smoke`:** the net-new plumbing pilots on a tiny cohort, **val sealed** — (a) the cohort adapter
  produces ≥ n_min balanced gene-disjoint pairs (or the fallback path is selected); (b) M1 value-linearity,
  M2 empty-anchor, M5 dose-construction, M4 content-differs, M7 disjointness, M9 determinism; (c) PP-xfer
  synthetic transfer detected; (d) the three nulls read ≈0 on the pilot. Must pass before freeze. `--smoke`
  computes memorials/arithmetic only; no confirmatory estimand is read.

## 9. SCOPE

- **Reaches:** **transferable OV/mediation → generalization** (behaviour-level, donor coupling). Establishes
  (or licenses a floored null on) whether the validated donor-side `U_info` state supports cross-input
  interchange across gene-disjoint matched targets — the causal-abstraction generalization test (Geiger
  et al. 2023/2025; Zhang & Nanda, Heimersheim & Nanda patching best-practice). **This is what connects the
  audited Head–Context Atlas matched cohorts (HCA11/HCA19) to the donor circuit.**
- **Does NOT establish:** payload **semantic identity** (R22 — unresolved; B3 injects the donor-side
  `U_info` component, not a strength/partner label); a **partner-identity concept** (R20 — `U_info` is the
  delivered representation, not identity); **binding** (R23); **biology** (spliceosome biochemistry — a
  Biology-level source is out of scope); a claim about the second ≈10.63 carrier (C11) or the period-3
  carrier's origin (P3.4, deferred). A positive B3 is "general intron-definition competence at the model's
  donor readout," a MODEL mechanism, not a per-gene lookup and not biochemistry.
- **Provenance:** checkpoint sha256 `2ad91428…0149d`; `maxentpy 0.0.4` tables sha256 `8b6d7b46…b7c00` (if
  strength covariates are recomputed); atlas cohort inputs by sha (manifest); seeds + `PYTHONHASHSEED` to
  the manifest; results write a manifest beside output; never overwrite the scientific record.
- **Amendments:** pre-run only, logged `Amendment (date, pre-run, no data seen)`; once the primary is
  inspected a fix is *outcome-aware verification*, not confirmatory (IC1 A2).

---

## Closure disposition & must-fixes (2026-07-22 design red-team)

**Disposition: FIXABLE_THEN_RUNNABLE (survivor=NO).** Red-team synthesis:
[`../redteam/REDTEAM_SYNTHESIS_B3.md`](../redteam/REDTEAM_SYNTHESIS_B3.md) (5 independent adversarial lenses, run
before any confirmatory compute; formalize-audit-verify step 2). Two convergent fatals — an object/space
contradiction in the injected primary object (Π_U of the final donor residual routed "through the L3H1 head-output
path" mixes distinct objects at different layers) and a U_info backdoor (U_info IS the clean−corrupt axis, so a
generic clean-donor offset rides through all four controls) — are both fixable. This is a pre-run R7/R15/R23-class
catch, NOT a P3.4-style non-identifiable wall: the cross-input interchange is identifiable and MPS-feasible after the
bounded v2 redesign below.

**survivor = NO — not executed this session.** Confirmatory feasibility is unestablished until a discovery pilot
measures the actual cross-example numerator and its dyadic HW; the primary object must be re-specified and a
same-source-corrupt control added before any freeze, and this session is documentation finalization only.
**Operational note:** the SpliceBERT.1024nt weights and the coupling/atlas data are absent from this worktree, so any
model run must be launched from the main checkout.

**Must-fixes (verbatim, from the synthesis):**

1. FATAL-1 (object/space): re-specify the primary injected object — as written it is a physical shape/space contradiction. Recommended: transplant the acceptor payload cross-example (A9 generalized, dr=r3cl_src-r3co), DROP Π_U from the injection path, DELETE the 'validated U_info / lower-risk-than-B2' framing (C24: payload NOT concentrated in U_info, L3H1 converts). The alternative (direct final-residual U-patch) is architecture-forced by margin=cls(final residual).
2. FATAL-2 (backdoor): add a same-source-CORRUPT content control. Make the transfer estimand clean-source > corrupt-source (differ only in partner-intactness), NOT clean-source > empty; add a margin-mismatched stratum.
3. Add a discovery-split PILOT measuring the actual cross-example numerator <M_xfer-M_empty> and its per-gene dyadic variance BEFORE freeze; re-derive §7 power from the piloted numerator, not R_self=+0.875 (the ceiling) or rank-1 overlap 0.970 (study uses rank-8, cross-gene overlap only 0.63). Pre-declare contrasts with projected family-adjusted dyadic HW>SESOI_R inconclusive.
4. Re-scope the confirmable endpoint to a one-sided POSITIVE dose-slope (beta_dose>0, monotone); demote Endpoint-B equivalence to reachable ONLY if the pilot shows family-adjusted dyadic HW(R_xfer)<0.20.
5. Add 'two-way source-gene x target-gene Poisson-multiplier dyadic bootstrap for a ratio-of-means R_xfer' to net_new and FPR-calibrate it; it is NOT reused (gboot_ratio is single-gene one-way; dyadic_ci is hard-wired to a projection slope).
6. Add a planted-positive traveling B3's ACTUAL M_xfer channel (real projection + real deliver/head_output/inject_capture); promote the self-ceiling-through-the-real-driver to a hard --smoke gate.
7. Renorm N-scr to ||Π_U[r_src]|| before injection; add on-path memorial ||inj_scr||==||Π_U[r_src]||<1e-6; norm-match the N-pos delivered edit.
8. Add absolute donor position (+PE band/phase) to match/bracket (R15/R20 relocation confound); add a position-mismatched-but-content-matched control.
9. Re-size cohort on POST-short-intron-gate counts (620 eligible / single_pair 153, not pre-gate 237); make --smoke cohort gate quantitative+blocking; pre-declare donor_cluster/cassette_exon promotion if single_pair short.

---

## Amendment (2026-07-23, PRE-RUN) — confirmatory control redesign of the as-implemented v2 driver

**Scope & status.** This amendment attaches to the design-of-record above (v1, NOT FROZEN; the runnable
driver `scripts/splicebert/run_b3_cross_example_transduction.py` is the v2 that applied the must-fixes:
transplanted acceptor payload, dropped Π_U, `M_xfer = clean-source − corrupt-source` [must-fix #2],
margin-mismatched stratum, two-way gene-dyadic bootstrap [#5]). The **first model execution** of that v2
driver (val n=260, deterministic `SEED=0`) returned **INCONCLUSIVE**: strong primary
(`R_xfer(α=1)=0.912`, dyadic CI [0.757,1.112], dose-monotone 0.534→0.912→1.178 β=0.271, decoy collapses
R≈−0.06) but three confirm-blocking controls were mis-specified. This amendment freezes their redesign
**before the re-run**. Full executor spec: `docs/B3_CONFIRMATORY_CONTROL_REDESIGN_2026-07-23.md`.
Adversarially reviewed (design → skeptic pass per control; all repairs folded in).

### A0. Mandatory disclosure (honesty; concealing this would itself be the forking-paths violation)

The mis-specified-control confirmatory numbers **were seen** before this redesign was written:
**scramble 0.978** (no collapse), **MF2 margin-mismatched 1.094 ≥ matched 0.912** (CI [−0.164,1.510]),
**health 0.560** vs an absolute 0.8 floor. Consequently, per the **IC1-A2 / R22** discipline, the eventual
val re-run is **VERIFICATION-LEVEL, not fresh confirmation** — a control fix applied after the primary was
inspected is outcome-aware verification. `result["evidential_tier"]` is hard-set to `"verification"`. A
clean *confirmatory* C-claim of any polarity requires running the frozen redesign untouched on the **sealed
chr8/9 cohort (NOT opened here)**. On this deterministic val cohort the paired-MF2 point estimate is fixed
at `Δ_MF2 ≈ 0.912 − 1.094 = −0.182 < 0`, so **POSITIVE is unreachable**; the expected verdict is
**Endpoint C** (transfer position- but not content-specific) or INCONCLUSIVE.

### A1. COMPLIANCE RESTORATION (no new choice — code was brought back to the frozen design intent)

Three redesign items restore clauses the design-of-record already mandated; the shipped v2 driver deviated:

| Redesign item | Design-of-record clause it restores | Shipped-driver deviation |
|---|---|---|
| MF2 = **paired within-target difference** `Δ_MF2` | §6 Family control: *"Every arm-vs-arm test is the paired within-target difference CI, **not non-overlap of two arm CIs**"* | `mm_ok = prim_CI_low > mm_CI_high` was exactly the forbidden two-arm-CI non-overlap test |
| MF2 unfavorable-at-power → **Endpoint C** verdict | §6 Endpoint C: *"R_xfer ≈ R_rand → content not separable from a norm/margin push → not identifiable"* | driver folded refutation into a plain gate-fail → INCONCLUSIVE, discarding the §4/§5 against-content signal |
| Scramble control **norm-matched** | must-fix #7 (norm-match the scramble control) | done: the feature-coordinate permutation preserves per-row norm & per-position energy exactly (the v2 analogue of #7, since #1 dropped Π_U) |
| Health probes **MASKED, per interpreted arm** | M8: *"each interpreted arm keeps ≥0.5 headroom over chance (R13, mlm_health, MASKED probes); below-floor arm dropped"* | driver measured **baseline** (no injection); redesign measures under the injection the arm is interpreted through |

These are code-to-design fixes and need no new sign-off.

### A2. GENUINE PRE-RUN AMENDMENTS (owner-signed 2026-07-23)

**A2.1 — M8 numeric floor (clarifies an ambiguous clause; pins the exact formula).** M8's *"≥0.5 headroom
over chance"* read as an absolute floor (0.75) is unsatisfiable at the model's empirical ~0.56 masked
baseline (the shipped `>0.8` matched neither reading — it was simply wrong). Frozen reading: **retain ≥50%
of the model's baseline headroom over chance**, measured **under max-dose (α=1.5) injection** on the target
`co_ids` substrate:
> `chance = 0.25`; `acc_base` = uninjected top-1 masked non-site accuracy measured **in-run** (not hardcoded);
> `floor = 0.25 + 0.5·(acc_base − 0.25)` (≈0.405 at acc_base=0.56); an interpreted arm **passes** iff
> `acc_inj ≥ floor`, else it is **dropped** (a dropped primary/control blocks POSITIVE).
Disclosure: the ~0.56 baseline was seen. This pins a formula for an ambiguous memorial; it is neither
"compliance" nor "loosening" (floor 0.405 tolerates ~a 27% relative-accuracy loss — a lenient, disclosed bar).

**A2.2 — MF2 mismatched-stratum power fallback (D-4, new pre-committed rule).** The shipped mismatched
stratum picks the **maximally** margin-mismatched source (`argmax|Δm_cl|`; extreme, high-variance), which can
leave the paired `Δ_MF2` half-width ≥ SESOI_R so the against-content-specificity branch never fires. Frozen
rule:
> **Default:** paired `Δ_MF2` on the argmax mismatched stratum. **Trigger (measured at the discovery/train
> `--pilot`):** if the argmax-paired `hw(Δ_MF2) ≥ SESOI_R = 0.20`, the confirmatory switches the mismatched
> selection to a **fixed margin-gap band** — per target, among strand/coding/|Δpdist|≤200 candidates with gap
> `d_j = |m_cl(j) − m_cl(i)|`, take the candidate nearest the **median of the [q40, q80] band** of that
> target's candidate-gap distribution (a clearly-mismatched middle band that excludes both the near-matched
> tail <q40 and the variance-inflating extreme >q80). The switch is triggered **solely** by the argmax-paired
> `hw` exceeding SESOI_R — not by which selection yields a preferred sign (anti-forking). The band's own `hw`
> is reported for audit. Consistent in spirit with must-fix #8 (a content-matched but clearly-mismatched
> control); the specific band/trigger is new.

### A3. Frozen decision rule (replaces the shipped verdict formula 436–442)

```
mf2_confirmed = Δ_MF2 dyadic CI-low > 0                       # §6 confound-DIRECTION test (size bar R_xfer>SESOI_R enforced separately)
mf2_refuted   = (Δ_MF2 CI-high < 0)  OR  (CI straddles 0 AND hw < SESOI_R)   # decisive, or powered-equivalence-at-0
health_ok     = primary-arm@α=1.5 healthy AND every interpreted control arm healthy   # A2.1
POSITIVE_transfer  ⟺ R_xfer CI-low>SESOI_R ∧ β>0 ∧ monotone ∧ scramble&decoy collapse ∧ mf2_confirmed ∧ perm-FPR∋0 ∧ health_ok
ENDPOINT_C_not_content_specific  ⟺ mf2_refuted        # higher precedence than NULL/INCONCLUSIVE
NULL_no_transfer   ⟺ R_xfer CI-high < SESOI_R
else INCONCLUSIVE_underpowered
```
Dyadic clustering for `Δ_MF2` is the **conservative three-way** (both matched and mismatched source genes,
`w[ti]·w[si]·w[si2]`) — it only widens the CI, guarding against over-claiming Endpoint C (anti-forking).

### A4. §8 GATES addendum — health positive-control (R-h2), new `--smoke` gate

Add to the §8 `--smoke` gate set (makes the health gate demonstrably failable, per R7/R21):
> **N-health-sens (planted-destructive):** inject a known-destructive head output `od` (large-norm random
> vector at the donor) and assert `acc_inj < floor` (the health gate FIRES). Without it, a single-token L3H1
> clamp barely perturbs distant probes and the gate risks being an unfireable pass — the exact pathology it
> exists to fix. A health PASS is reported as "no *detectable* damage at distant probes," not a strong
> "injection did not break the model" certification (disclosed low failure-power under a localized edit).

### A5. Guardrails (binding)

Freeze scramble+health+MF2 as a SET; ONE re-run; no per-control iteration after any part of the re-run is
seen (next step on failure is the sealed cohort, never another val tweak). Results immutable; corrections
appended. `v1` retained intact as design-of-record; this amendment appended at the foot per repo convention.

### A6. Health gate DOWNGRADED to a reported diagnostic (owner 2026-07-23, pre-run, smoke-surfaced)

The A4/R-h2 positive control (a planted-destructive injection must FIRE the health gate) was run at
`--smoke` and did **not** fire: a 50×-norm off-manifold clamp at the single donor-token L3H1 output left
distant non-site masked accuracy unchanged (`acc_destructive 0.540` vs `acc_base 0.538`, floor 0.394).
This is a **verified architectural property** — a local single-token injection does not propagate to
distant probes through layers 4–5 — not a code fault (memorials `value_linearity` 2.6e-6, `recon` 0.0;
calibration passes). The distant-probe health gate is therefore **structurally unfireable by a local
injection**, so making it a hard POSITIVE-blocker would be an unfireable pass (the R7/R21 pathology it
exists to avoid).

**Owner decision (2026-07-23, pre-run — `--smoke` reads only memorials/plumbing, no confirmatory estimand):
DOWNGRADE health to a REPORTED DIAGNOSTIC.** It is still measured under injection per arm and reported
(`acc_base`, `floor`, per-arm `acc`/`ok`), but is **removed from the POSITIVE conjunction** and R-h2 is
**not** a `SMOKE_PASS` gate. Interpretation (R-h3): distant-probe health ≈ baseline under the payload
injection confirms **no GLOBAL forward-pass breakage** — appropriate, since a local injection cannot
globally break the model — reported as weak evidence, never a strong "injection did not break the model"
certification. The confound health guarded against (a globally-broken forward pass yielding an artifactual
margin) does not arise for a local injection, so the downgrade does not weaken the POSITIVE family. This
supersedes A2.1's "below-floor arm blocks POSITIVE" and A4's hard R-h2 gate.

### A7. MF2 power extension (owner 2026-07-23, pre-run, verification-level): n 260 → 512

The n=260 confirmatory returned the MF2 paired band `Δ_MF2 = +0.028`, CI [−0.215, 0.187], **hw 0.2008 —
0.0008 OVER `SESOI_R=0.20`**, so the powered-equivalence Endpoint-C branch did not fire (verdict
`INCONCLUSIVE_underpowered`). Cam authorized a **POWER EXTENSION** — explicitly **not** a post-hoc SESOI or
estimator tweak: re-run the confirmatory at a **larger pre-committed n** with the **identical** estimand
(`Δ_MF2`), `SESOI_R`, three-way dyadic estimator, and fixed `[q40,q80]` band. **Pre-committed `n = 512`**
(the naive `N_req` for a target hw≈0.16 is ~410; 512 buffers dyadic-clustering variance inflation, which
scales worse than 1/√n). This resolves an **underpowered equivalence** — the estimand, control, and bar are
unchanged, only n grows — so it is a legitimate power extension, and it **narrowly supersedes A5's
"one re-run" clause** *for this owner-authorized power extension only* (A5's intent — no *control* tweaking
after seeing the result — is preserved: nothing about the control changes).

Evidential status: **still VERIFICATION-level** (val was already inspected at n=260; the `Δ≈0` direction is
known, so this is outcome-aware). The n=260 result stays **immutable**; the n=512 result supersedes it for
the MF2 verdict only. **Stopping rule (binding, anti-chasing):** if n=512 still shows `hw ≥ SESOI_R`, the
honest outcome is "reported at achieved precision" — **NOT** a further power-up. The sealed chr8/9 cohort
remains the only path to a clean *confirmatory* (non-verification) call, and is not opened.
