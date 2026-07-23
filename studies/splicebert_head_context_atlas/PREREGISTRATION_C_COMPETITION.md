# PREREGISTRATION — Estimand C: competitor-conditioned component recruitment

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

> FROZEN 2026-07-18 after the 6-lens design red-team (`REDTEAM_SYNTHESIS_C_6lens.md`) and the Stage-0
> census + Stage-1 behavioral pilot. Amendments allowed only pre-run, logged inline as
> `Amendment (date, pre-run, no data seen)`. Discovery = autosomes ∉ {chr8,9,11,16,19}; **chr8/9 SEALED**.

## 0. INTENT
Identify which SpliceBERT components are causally recruited **because a splice site sits in a competitive
choice set** (alt-5′SS donor competition / alt-3′SS acceptor competition), controlling for strength,
distance, composition, and packet geometry — and whether recruitment acts through QK selection, OV
writing, or both. NOT "head *h* is an acceptor-cluster neuron"; NOT a re-derivation of the closed carrier
account.

## 1. Question and scope
- **Biological computation:** competitor-dependent splice-site reconstruction (intron-mate; §ONTOLOGY).
- **Checkpoint:** `SpliceBERT.1024nt` sha256 `2ad91428…0149d`, eager for attention (HC2 only).
- **Population:** GENCODE v50 comprehensive (`83fba3e9…13a453`), protein_coding, hg38; competition
  resolvable within a 1024-nt window (median intron ≈3.8 kb ⇒ short-intron sub-population — an explicit
  scope limit, census v1).
- **Model-native behavior:** `donor_margin`/`acceptor_margin`.
- **Rival explanations to defeat:** generic local perturbation (not competition), competitor STRENGTH
  (R22), target–competitor DISTANCE, sequence composition, splice-graph complexity, gene reuse,
  collider-induced strength↔distance dependence, C33 distribution masquerading as a per-head null.

## 2. Behavioral gate (THREE parts; all outcome-blind on discovery, sealed untouched)
Primary metric `M_comp = margin(t | competitor present) − margin(t | competitor destroyed w/ REAL bases)`,
target + shared partner intact. Direction: sign-heterogeneous ⇒ the gate is on **|M_comp|** and on
specificity, not on the signed mean.

| Gate | Rule | Status (pilot v1, n=400/370 genes) |
|---|---|---|
| **B — positive control** | partner-coupling `B>0.05`, gene-clustered CI excludes 0 | **PASS** (+0.175 [0.124,0.229]) |
| **|M_comp| existence** | `\|M_comp\|` material (CI excl 0 ∧ point ≥ SESOI) | **PASS** (+0.111 [0.082,0.142]) |
| **competition-specificity** | `\|M_comp\|` materially exceeds a **real, strength+distance-matched, non-competitor matched-decoy-destroy** (paired, gene-clustered CI of the difference excludes 0) | **UNRESOLVED — REQUIRED before any confirmatory/Colab run** (crude 2-bp sham gave `M_comp−sham` inconclusive [−0.067,+0.036]) |

**Minimum effect to proceed / stop rule:** if the specificity gate is bounded-null or inconclusive against
the REAL matched decoy, declare **INSTRUMENT-LIMITED / no competition-specific behavior to attribute**
(coheres with IC1/IC2) and do NOT spend Algoverse credits. This is a terminal, publishable result.

## 3. Instrument preflight (frozen thresholds; gates BEFORE any primary)
- **Positive controls:** (a) B partner-coupling material; (b) a **SLOT-analog** competition-adjacent effect
  (IC2 τ_D≈0.53 analog: the target's margin depends on which competent site occupies the shared slot) —
  so a competition null is specific, not a dead assay.
- **Negative controls → equivalence within ±SESOI:** intronless packets, no-site packets, partner-out-of-
  window packets, matched-decoy-destroy.
- **Sensitivity:** planted competition-head detectable (§Calibration). **Resolution:** adjacent-head
  discrimination. **Health:** `mlm_health` at non-site positions preserved under intervention.
- **Anomalous-control stop:** any failed positive/negative control ⇒ freeze, diagnose the instrument.

## 4. Candidate and causal tests
- **Localization (discovery only):** `head_attribution_scan` (Taylor) ranks all 96 heads per example —
  candidate generation, not necessity.
- **Intervention:** **scrub-to-corrupt** — patch each head's output to its competitor-DESTROYED value
  (on-distribution, condition-aware; `head_patch_scan`/`patch_heads`; reuses A6 machinery). Mean-ablation
  is discovery-robustness only (it injects a between-condition mean shift; MF7).
- **Primary estimand (re-scoped, MF3):** a **group / importance-ranked-set faithfulness-size** statistic
  for `M_comp`, able to detect a DISTRIBUTED signal. **A per-head null is pre-registered as EXPECTED under
  C33-distribution and is NOT evidence of absence.**
- **IUT controls (both required):** (1) matched **decoy-destroy** — same target, destroy a real strength+
  distance-matched non-competitor; (2) matched **single-site** — an isolated matched target, matched on
  SELECTION variables (competitor-annotation propensity, in-window probability) + confounds (MF5).
- **QK/OV (HC2, R23-safe):** C35-style A-vs-V **main effects** with importance-matched interaction nulls;
  A×V reported, never interpreted as binding. **HC2 is descriptive `route_shift` unless a QK-main
  sensitivity floor is first demonstrated on the gate-B coupling contrast** (MF8); re-derive which head
  reads at the competitor (do NOT import L3H1).

## 5. Confirmatory hypotheses (frozen)
- **HC1 (primary, existence):** ∃ a component (group) whose competition effect passes the IUT (floor ∧ >
  decoy-destroy ∧ > single-site), stable across pooled seeded chr8/9 draws, common-support+health OK. Test:
  **max-T Westfall–Young FWER omnibus across the 96-head family → BH-FDR to enumerate → per-component IUT.**
- **HC2 (secondary):** competition-conditioned QK route redistribution (sham-adjusted) with OV stability by
  an **equivalence bound** — only if the MF8 sensitivity floor passes; else descriptive.
- **HC3 (secondary):** ONE global recruitment-burden statistic + ONE ordered slope over
  `cluster_size`/`local_degree`, confound-adjusted, CI-excludes-0. `{HC2,HC3}` = separate BH family, gated
  on HC1. Discovery-only: H1-general, H3, H5, H6, H7, per-head dose curves.

## 6. Statistics
- **Unit:** gene (homology cluster). Gene-clustered CIs; **two-way gene-dyadic bootstrap** where a
  contrast reuses sender/receiver genes (matched single-site arm); one-way within-gene otherwise (draw the
  reuse graph before freezing).
- **Power gate (hard, pre-compute):** projected MDE (real discovery SD(M_comp)≈0.31 → `E_ih` SD ≥ that) at
  the realized sealed n, under 96-head max-T, must be < SESOI-C or declare INSTRUMENT-LIMITED at design
  time. (Pilot v1: per-head MDE ~0.024–0.065 straddles SESOI ⇒ the group estimand is the licensable route.)
- **SESOI-C:** calibrated WITHIN the C intervention/units (candidate = a pre-registered fraction of the
  behavioral `M_comp` swing), frozen by Cam after the specificity pilot, BEFORE the confirmatory run. The
  matched-null envelope is DETECTABILITY only, never the SESOI.
- **Seeded draws:** pooled (draw-aware bootstrap) with an across-draw **stability-SD trigger**
  (`SD > 0.5·SESOI ⇒ heterogeneity branch`), NOT "significant in all K"; report clusters with ≥2 candidates.
- **Multiplicity:** exact add-one permutation p; max-T FWER for HC1 existence; BH q=0.05 to enumerate and
  within the secondary family. **CI classes:** material / small-but-detectable / bounded-null / inconclusive.
- **Recovery = ratio-of-means; corruption uses REAL bases, never [MASK].**

## 7. Memorials (can-fail, on the measurement path)
`destroy_is_real_bases` (destroyed window is real bases, motif-only, length preserved); `margin_reproduces`
(MLM head reproduces the cached margin < 1e-4); **add-one identity** for scrub-to-corrupt (patch a head to
its own cached output ⇒ unablated margin < 1e-4 — on-path for HC1); `decoy_strength_matched` (|score(decoy)
− score(competitor)| ≤ frozen caliper) and `decoy_distance_matched` (target-anchored). NOT memorials
(flagged forced): the A×V identity `(A_I−A_D)(V_I−V_D)` (R23); any embed-clamp identity (R21).

## 8. Calibration (sensitivity + specificity)
- **Planted-positive:** inject a head whose value write depends on a **readable motif at the competitor
  window** (not the condition label), sized so the through-model `M_comp` swing = SESOI-C; require detection
  AND that it fires on decoy-destroy (tests competitor-vs-decoy discrimination).
- **Permuted-null:** permute competitor-STATUS across matched (target, destroyed-real-motif) pairs, holding
  perturbation TYPE fixed; require max-T ≤ 0.05 FPR (threshold split-calibrated to the permutation null).
- **Confound-specificity:** plant a target-margin change driven purely by competitor STRENGTH (destroy a
  strong non-competitor) and require the adjusted `E_ih` to stay NULL; plant a genuine competition head at
  the observed collinearity and require recovery. Predeclare the max collinearity at which both are jointly
  satisfiable; STOP → INSTRUMENT-LIMITED if the observed partial-R²/VIF exceeds it.

## 9. Replication and interpretation
- **Sealed:** chr8/9, opened once, not revised after unblinding. Confirmatory launches only if the
  pre-compute power gate AND behavioral gate (incl. specificity) AND SLOT positive control ALL pass.
- **Strongest claim per outcome:**
  - specificity + HC1 pass → "component (group) *g* causally contributes to competitor-dependent
    reconstruction across several splice contexts, controlling for strength/distance/composition/geometry,
    stable across seeded draws on gene-disjoint confirmation" (scope: intron-mate, ≤1 kb).
  - specificity fail → "SpliceBERT's splice-site reconstruction shows a competition *magnitude* effect not
    distinguishable from generic local perturbation — INSTRUMENT-LIMITED, coheres with IC1/IC2."
- **First experiment that would falsify the preferred interpretation:** a real strength-matched
  decoy-destroy that matches |M_comp| (kills competition-specificity), or a group ablation that leaves
  `M_comp` intact (kills necessity).
- **Compute:** Colab CUDA / Algoverse; all 96 heads exact; per-tuple immutable cache; resumable,
  outcome-silent; float32 + TF32/AMP off; no attention capture except L3H1/HC2; the pre-compute gate is a
  hard STOP baked into the runner.
