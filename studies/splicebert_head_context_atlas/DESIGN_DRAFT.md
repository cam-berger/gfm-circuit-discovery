# DESIGN DRAFT — SpliceBERT Head–Context Atlas (estimand C primary)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

> Status: **DRAFT for pre-compute adversarial red-team** (formalize-audit-verify step 1–2).
> Nothing here is frozen until the red-team must-fixes are applied. No model outcome has been read.
> This document is the input the red-team lenses attack. Attack it, do not defend it.

---

## 0. INTENT (north star)

**One sentence:** Identify which SpliceBERT attention heads are *causally recruited because a splice
site sits in a competitive choice set* (a donor choosing among acceptors, or vice-versa), after
controlling for site strength, target–partner distance, sequence composition, and packet geometry —
and whether that recruitment acts through QK selection, OV writing, or both.

**Model-native observable that would show it:** a head whose exact-ablation effect on the target's
masked-nucleotide reconstruction margin is larger when a *real competitor* is present than when a
*matched non-competitor* is present, and larger than for a *matched isolated single site*.

**What this is NOT:** it is not "head h is an acceptor-cluster neuron"; not a re-derivation of the
closed carrier account (C8/C19/C32/C34/C35); not an origin study.

**Relationship to prior work (must be differentiated, not duplicated):** the closed donor circuit
(C1–C35) established address (L3H1/L2H15), payload (L0→MLP2), U_info, and the distributed core (C33).
The IC arc tested *conditional splice competitiveness* at the **behavioral** level: **IC1** found distal
intron arrangement is NOT a meaningful partner-specific driver (ΔC_specific CI **includes 0**, sub-SESOI);
**IC2** (acceptor-competition, PARKED) found the transport DiD intrinsically noisy (SD 0.774, ±0.05
equivalence unreachable at ~1985 genes). **This atlas moves competition from behavior to head
attribution** — which is only well-posed if the *behavioral* competition effect it attributes is itself
non-null (see §5, §6 behavioral gate). The IC1/IC2 precedent is a live threat to identifiability.

---

## 1. ESTIMAND

Three estimands; **C is the sole primary confirmatory estimand**. A is a descriptive census; B is a
positive-control gate.

Let `margin(target)` = `CouplingScorer.donor_margin` (mask the G of a donor GT; `logit(G) − max(A,C,T)`)
or `acceptor_margin` analog. Units = nats. Reference point = the unperturbed packet.

### C (PRIMARY) — competitor-conditioned selection
For a target site `t` with an annotated partner `p` and an annotated same-type competitor `c` (a second
donor sharing acceptor `p`, or a second acceptor sharing donor `p`):

```
M_comp(t | c)  =  margin(t | c present)  −  margin(t | c destroyed with REAL bases, p and t intact)
```
`destroy` = substitute the competitor's core motif to non-motif REAL bases (reuse
`coupling.destroy_donor/destroy_acceptor`); never `[MASK]`.

**Per-head effect** for head `h` on example `i`:
```
E_ih  =  M_comp(head h ablated)  −  M_comp(reference, no ablation)
```
Head ablation = `hf_bert.head_ablation_scan` (mode frozen; see §2). Sign convention: `E_ih > 0` means
ablating `h` *reduces* the competition-dependence of the target (i.e. `h` supplied competition
information). Screening pre-rank = `attribution.head_attribution_scan` (Taylor); **exact
`head_ablation_scan` decides** (attribution ranks, intervention decides).

### A (CENSUS, descriptive — NOT a hypothesis test)
Every head measured on every eligible example for both the general reconstruction margin
`M_rec(t) = margin(t | t masked)` and `E_ih` above. Purpose: an unbiased all-head prevalence table that
corrects the "frequency-in-a-pruned-DAG ⇒ head is common" fallacy. Reported as distributions/ranks with
gene-clustered uncertainty; **no confirmatory claim rides on A.**

### B (POSITIVE CONTROL gate — NOT a hypothesis test)
On clean single donor–acceptor pairs, recover the established coupling behavior
`B = margin(donor | acceptor intact) − margin(donor | acceptor destroyed) > 0.05` nats (gene-clustered
CI excludes 0), and recover the L3H1 A-vs-V signature (restoring intact attention rescues coupling;
restoring values ≈ 0). Failure ⇒ **INSTRUMENT-LIMITED**; no C-null is interpretable.

---

## 2. METHOD

- **Model:** `SpliceBERT.1024nt`, post-LN BERT-MLM (6L×16H, d=512, P=1026 absolute learned positions),
  checkpoint sha256 `2ad91428…0149d`. `eager=True` (SDPA returns zero attentions). Env `gfm-thrust1`.
- **Intervention (C, HC1):** exact head ablation, **mode = `mean`** (resample/zero reported as robustness
  in discovery only). Screening = Taylor attribution (one fwd + one backward, ranks all 96 heads cheaply).
- **QK/OV separation (HC2, R23-safe):** C35-style A-vs-V. QK/address main effect = patch the head's
  *attention pattern* across the competitor-present/destroyed contrast holding values; OV main effect =
  patch its *values* holding attention. Reported as **separate main effects** with importance-matched
  nulls. The A×V interaction is **architecture-forced** (`O=A·V`); it is reported with an
  importance-matched (E-band) interaction null and **never interpreted as "binding" (R23)**. `route_shift`
  (H2) = a descriptive attention-source-distribution comparison (single-site vs competition), not binding.
- **Two matched controls in the IUT (both required):**
  1. **matched decoy-destroy** — same target `t` in the same cluster, but destroy a distance/phase/
     composition/strength-matched **non-competitor decoy** (`circuit.matched_decoy_donor/acceptor`,
     `address.phase_matched_decoy`) instead of the real competitor `c`. Isolates *competitor-specificity*.
  2. **matched single-site** — a genuinely isolated target (site with its partner, no cluster), matched on
     MaxEnt strength / target–partner distance / canonicality / CDS-vs-noncoding / GC / crop position.
     Isolates *competition-context vs generic reconstruction*.
- **Compute doctrine (frozen):** local MPS for census/matching/discovery/preflight/benchmark; **Colab
  CUDA (Algoverse credits) for the confirmatory run — ALL 96 heads exact** (completeness, not a
  shortlist). Per-`(target,competitor,decoy,condition,head)` immutable tuple cache: the 4 seeded draws
  re-index one table, no 4× forwards. Resumable, chunked, outcome-silent, float32 + TF32/AMP off, no
  attention-matrix capture except L3H1/HC2.

---

## 3. MEMORIALS (arithmetic/mechanical, must be able to fail, on the measurement path)

- **`value_linearity`** (reuse the A8/A10 anchor): the reconstructed per-head contribution sums to the
  measured residual to < 1e-4. On-path (exercises the ablation machinery). Can fail.
- **`destroy_is_real_bases`**: the destroyed competitor window contains no `[MASK]`; equals the source
  sequence except at the declared motif positions; length unchanged. Can fail.
- **`margin_reproduces`**: the MLM head reproduces `margin(target)` from the cached residual to < 1e-4.
- **`decoy_distance_matched`**: |dist(target,decoy) − dist(target,competitor)| ≤ frozen tolerance for
  every IUT example. Can fail (guards against a distance confound sneaking in via the decoy).
- **NOT a memorial (flagged forced):** the A×V interaction identity `Δ = (A_I−A_D)(V_I−V_D)` — reported,
  never sold as validation (R23/R21 discipline).

---

## 4. CALIBRATION (sensitivity + specificity, hard preflight gates)

- **Planted-positive (sensitivity):** inject a synthetic head-competition signal that travels the *same
  channel* as the real claim — e.g. a head whose value write is forced to depend on competitor presence —
  and require the pipeline's `E_ih` to detect it above the frozen floor. If a planted competition-head is
  missed, the classifier is underpowered ⇒ STOP.
- **Permuted-null (specificity):** shuffle the competitor/decoy labels within a cluster and require the
  max-T existence decision to fire ≤ 0.05 FPR. Threshold calibrated to the permutation null itself
  (split calibrate/test).
- **Confound-specificity:** plant a target-margin change driven *purely by distance* (or GC) and require
  the strength/distance/GC-adjusted `E_ih` to stay NULL; plant a genuine competition head at the observed
  collinearity and require it recovered.

---

## 5. IDENTIFIABILITY & common support

- **Behavioral-existence precondition (the IC1/IC2 threat):** head attribution of `M_comp` is well-posed
  only where `M_comp` itself is meaningfully non-null. Predefine a **behavioral competition gate**: on the
  unselected cohort, `M_comp` (competitor present vs destroyed) has a gene-clustered CI excluding 0 and a
  point ≥ the behavioral floor. If `M_comp` is bounded-null (whole CI inside ±SESOI) ⇒ report
  **INSTRUMENT-LIMITED / no behavior to attribute**, do not attribute heads to a non-effect.
- **Confound set:** MaxEnt `score5`/`score3` (strength), target–partner distance, competitor distance,
  GC/pyrimidine, low-complexity, packet length, crop position, CDS-vs-noncoding, canonicality. Adjust via
  the per-head model `E_ih = α_h + Σ_k β_hk T_ik + f_h(strength,distance,GC,length) + u_gene + ε`. Prefer
  matching (IUT controls) + cross-fitted adjustment over linear residualization.
- **Common support:** computed on the **unselected** cohort BEFORE outcomes; report overlap of the
  confound distributions between competition and matched-single-site cohorts; flag `cluster_size` /
  distance strata with no overlap as non-estimable.
- **Definitional entanglement check:** is `E_ih` separable from the behavioral `M_comp`, or is a head's
  effect definitionally a re-expression of the behavioral competition effect? State the honest composite
  outcome if entangled.

---

## 6. DECISION RULE (frozen before data)

- **B (gate):** pass ⇒ proceed; fail ⇒ INSTRUMENT-LIMITED.
- **Behavioral competition gate (§5):** pass ⇒ C attributable; fail ⇒ INSTRUMENT-LIMITED.
- **HC1 (primary, existential):** (1) global **max-T permutation omnibus across all 96 heads** — does any
  competition-specific head exist? If it fails ⇒ null (no competition head). (2) If it passes, **BH-FDR
  q=0.05** across 96 heads enumerates *which*. (3) Each enumerated head must pass an **intersection-union
  test**: `E_ih` exceeds the within-C meaningful floor AND exceeds its matched decoy-destroy effect AND
  exceeds its matched single-site effect — **all three**, and be **stable in sign+floor across all 4
  frozen seeded chr8/9 draws**, and pass common-support+health. CI classes: material / small-but-detectable
  / bounded-null / inconclusive.
- **HC2 (secondary, L3H1 pre-frozen):** positive sham-adjusted QK route redistribution AND OV stability by
  an **equivalence bound** (TOST inside ±SESOI, not mere non-significance); A-vs-V main effects separate;
  A×V reported vs importance-matched null, not interpreted.
- **HC3 (secondary):** one global recruitment-burden statistic (effective # heads with `E_ih` > floor) and
  one ordered slope over `cluster_size`/`local_degree`, adjusted for confounds; CI-excludes-0.
- **Secondary family {HC2,HC3}:** BH within the secondary family; mechanistic interpretation **gated on
  HC1 passing.**
- **SESOI-C (meaningful floor):** a within-C quantity in `donor_margin` nats — candidate = a pre-registered
  fraction of the total behavioral competition swing `M_comp` — frozen by Cam AFTER the preflight
  calibration, BEFORE the confirmatory run. The null envelope is **detectability only**, not the SESOI.

---

## 7. HELD-OUT & sampling stability

- **Discovery = autosomes ∉ {chr8, chr9}** (screening, matching, shortlisting, all `E_ih` inspection).
- **Sealed confirmation = chr8/9, untouched** until the confirmatory run.
- Unit of analysis = **gene** (homology cluster). Gene-clustered CIs; **two-way gene-dyadic bootstrap**
  wherever a pair reuses sender/receiver genes (R22 discipline).
- **K = 4 frozen independently-seeded draws** (seeds 0,1,2,17) over the sealed pool: each draw
  independently selects, per cluster, which competitor and which matched decoy to use; the underlying
  per-tuple `E` is cached once and re-indexed. HC1 head must pass in **all 4** (seeded draws, NOT
  replications; R22).

---

## 8. GATES & SCOPE

- **Global PASS = AND(memorials §3, calibration §4, B gate, behavioral competition gate §5).** Run STOPS
  on any failure. `--smoke` = memorials + arithmetic only, sealed data untouched, tiny fixtures.
- **Determinism:** `sorted(set(...))` at every hashable-iteration site; `PYTHONHASHSEED=0`; cross-seed
  determinism test asserting identical cohorts across {0,1,2,17}.
- **Provenance:** `ExperimentManifest` beside every output; refuse overwrite; pin commit, checkpoint sha,
  GTF sha, Python/PyTorch/CUDA, GPU type, seeds, precision.
- **Scope ceiling:** the strongest licensed claim is "head h causally contributes to competitor-dependent
  reconstruction across several splice contexts, controlling for strength/distance/composition/geometry,
  stable across seeded draws on gene-disjoint confirmation." No "acceptor-cluster neuron"; no carrier-origin
  claim; no binding claim from A×V.

---

## 9. OPEN QUESTIONS FOR THE RED-TEAM (attack these first)

1. **Is `M_comp` behaviorally non-null at all?** IC1/IC2 suggest competition effects are small/noisy. If
   the behavioral gate will almost always fail, the whole atlas is INSTRUMENT-LIMITED before it starts. Is
   the gate correctly specified? Is there a cohort where competition provably moves the margin?
2. **Is `E_ih` (head effect on a difference-of-differences) identifiable, or is it a high-variance contrast
   like IC2's β_transport (SD 0.774)?** Propagate the variance: `M_comp` is already a difference; `E_ih` is
   a difference of `M_comp`. Is the confirmatory run powered for the within-C SESOI at the sealed-cohort n?
3. **Does the matched decoy-destroy truly isolate competition** from "any real-base edit at that distance",
   given the decoy is also a real motif? Could destroying a real competitor vs a real decoy differ only by
   the competitor's *strength*, not its *competition role* (R22 — MaxEnt as stratifier)?
4. **Max-T omnibus across 96 heads:** is the permutation exchangeable unit correct (gene? cluster?), and
   does head-to-head correlation (redundant heads, C33 distributed core) inflate/deflate the max-T null?
5. **A-vs-V for HC2:** is the "OV stable / QK shifts" decomposition itself architecture-forced in the
   competition contrast the way `Δ_bind` was (R23)? Is the importance-matched interaction null the right
   null?
6. **Cohort definition:** does defining clusters from GENCODE `attested` isoform co-annotation conflate
   alternative splicing (mutually exclusive donors) with independent introns? Does `partner_in_window`
   at 1024 nt bias which competitors are visible?
7. **Selection-on-collider:** does selecting competition examples that *have* a visible competitor within
   the window (and matched single-sites) induce Berkson decoupling of strength from distance?
