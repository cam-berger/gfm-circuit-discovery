# Tracing plan — partner-selection circuit in frozen SpliceBERT

> ⛔ **SUPERSEDED / VOID.** This document belongs to the **retracted** v1/v2 partner-selection
> design (R2/R3) or to the pre-sealed-test protocol. It is kept for provenance only.
> **The document of record is `studies/splicebert_partner_selection/FINDINGS.md`;**
> the frozen protocol actually executed is `PREREGISTRATION_SEALED.md`.
> Retractions: `CLAIMS.md` §R1–R12.

**Status:** FROZEN 2026-07-11 (refined after the local smoke). Hashed in `FREEZE.md`
together with the discovery subset (D/V\_c/O) and the component-set size (5). Trace
target: the five **additive-frozen** checkpoints only. Test split (chr19–chr22) stays
sealed until the component set is frozen. Instrument (G1) and AtP screen are built
and validated; only the GPU run harness (G5) remains before execution.

This plan follows the registered intervention sequence in `METHOD.md` (steps 1–8)
and the thresholds in `PREREGISTRATION.md`. It records the method for each step,
the mech-interp literature it rests on, the tool that implements it (and the gaps
that don't yet exist), the metric, and the pre-registered pass rule.

## 0. Claim ladder — what each phase can and cannot license

The behavioral gate established that **partner information is present in the frozen
representations and is pretraining-derived** (probe-decodable). Tracing does not
automatically upgrade that to "a circuit the model uses for its own MLM objective."
We therefore separate three claims and never conflate them:

- **C1 (localization):** a compact set of heads carries the partner information the
  readout uses. — steps 1–3.
- **C2 (mediation of a partner-specific corruption):** clean activations from that
  set restore behavior after a *partner-relevant* corruption. — steps 4–5.
- **C3 (partner-specificity, not site-strength):** the set tracks the *identity of
  the correct partner*, tested with partner-swap chimeras, and is stronger on
  conflicts than matched ordinary exons. — specificity + swap tests.

C1–C3 together = "a localized, pretraining-derived, partner-specific circuit that
the trained readout reads." The strictly stronger claim "the model uses this in its
native MLM computation" is **out of the registered scope**; an MLM-native readout is
listed as an optional exploratory extension only.

## 1. Threats to validity and how the design controls them

| Threat | Why it bites here | Control (literature) |
|---|---|---|
| Zero-ablation puts the model off-distribution | spurious large effects / sign flips | require consensus across **zero / mean / resample** nulls (scaffold `ablation_modes`); prefer denoising patching (Heimersheim & Nanda 2024) |
| Backup / self-repair heads | single-head ablation understates importance, flips signs | promote on **cross-null consensus**, then test the **grouped** set, and use denoising recovery, not single-head deletion (Wang et al. 2022 IOI; McGrath et al. 2023 hydra effect) |
| Probe ≠ model usage | our score is the trained readout's margin | keep C1/C2/C3 vs the MLM claim separate (above); anchor "partner-specific" on corruptions/swaps, not on the readout alone |
| Selecting heads that overfit the discovery examples | 96 heads, small effect sizes | **discovery/confirmation split** of the conflict set; matched-random null; replication across 5 seeds |
| "Any large corruption recovers" | endpoint masking is generic | mediation only counts on a **partner-relevant** corruption and must beat a **context-only** corruption; `min-gap` guard already refuses ineffective corruptions |
| Circuit is generic site-finding, not partner-matching | strong-motif shortcut | **partner-swap chimeras** (`swap_site_neighborhood`, `boundary_pair_factorial`) + **specificity** vs matched ordinary exons |

Methodological basis for the interventions themselves: causal mediation for
transformers (Vig et al. 2020), causal tracing (Meng et al. 2022, ROME), activation
& path patching (Wang et al. 2022; Goldowsky-Dill et al. 2023), automated circuit
discovery / attribution patching as scalable screens (Conmy et al. 2023; Syed et al.
2023), attention knockout for edge-level tests (Geva et al. 2023), and causal
scrubbing as the consistency ideal we approximate (Chan et al. 2022).

## 2. Discovery data plan (fixed before any intervention)

- **Discovery subset D:** N=512 val conflicts, stratified by gene and exon-length
  bin, sampled with a fixed seed. Heads are selected on D only.
- **Confirmation subset V\_c:** a disjoint N=512 val conflicts. All necessity /
  mediation / specificity *pass rules* are evaluated on V\_c, not D — so selection
  cannot inflate the confirmatory numbers.
- **Matched-ordinary set O:** N=512 non-conflict val exons matched to V\_c on gene,
  exon length, and candidate count (for the specificity test). *Builder does not
  exist yet — see gap G3.*
- Grouping unit for any CI remains **gene\_id** (as in the behavioral gate).
- All three are drawn from validation chromosomes; the test split is untouched.

## 3. Procedure (registered steps 1–8)

> **Refinement (2026-07-11), from the local smoke.** The necessity-selected heads
> (whose ablation hurts the clean margin) and the mediation heads (whose clean
> outputs restore a partner-masked run) **diverge** — as they should (necessity ≠
> sufficiency). So the candidate circuit is defined by their **convergence**, not by
> forcing the ablation set through the patch, and mediation is run for **both**
> acceptor- and donor-mask. An **attribution-patching (AtP) fast screen** — validated
> locally against exact patching (Spearman 0.876; top-8 overlap 8/8) — pre-ranks all
> 96 heads in one forward+backward; exact ablation/patching then confirms the gates.

### Step 0 — AtP fast screen  (tool: `circuits/attribution.py` ✓, validated)
For necessity and for each mediation corruption, rank all 96 heads by linear
attribution on D. Shortlist the top `max_candidate_heads_after_screen` (16) per
axis for exact evaluation. Approximation only — never a gate.

### Step 1 — single-head causal screen  (tool: `trace_heads.py` ✓)
Score = mean true-pair margin (higher better). Exactly ablate the shortlisted heads
(and, budget permitting, all 96) under zero/mean/resample on D; effect = baseline − ablated.

### Step 2 — cross-null consensus  (tool: `trace_heads.py` ✓)
Necessity set N = heads **positive under ≥2 of 3 nulls**, ranked by **median** effect.

### Step 2b — mediation ranking, both masks  (tool: `patch_conflicts.py` ✓)
For `acceptor-mask` and `donor-mask` separately, exact single-head denoising recovery
on D. Mediation set M = heads with positive recovery on **both** partner masks.

### Step 2c — convergent candidate set  (analysis)
**Candidate circuit C = N ∩ M** (necessary *and* a partner-mediator), capped at
`default_component_set_size` (5) by combined rank. If N ∩ M is empty, report
"necessity without mediation" (a registered failure interpretation) — do not force a set.

### Step 3 — grouped necessity vs matched-random null  (tool: `trace_heads.py` ✓)
Ablate **C** jointly (mean null) on **V\_c**; compare to 1000 size-matched random sets.
**PASS rule (prereg):** observed grouped effect > 95th percentile of the null (add-one p ≤ 0.05).

### Step 4 — mediation of C, partner vs context  (tool: `patch_conflicts.py` ✓)
Denoising-patch **C** from clean into the corrupt run on **V\_c**, for acceptor- and
donor-mask (guard: refuse if gap < `min-gap`).
**PASS rule (prereg):** C recovery\_fraction ≥ 0.30 on ≥1 partner mask, and it must
**exceed** the recovery on a **context-only** corruption (`site-windows-only`).

### Step 5 — token-localized patching  (tool: `patch_conflicts.py --token-patch endpoints` ✓)
Repeat step 4 restricting the patch to the site-endpoint windows. Recovery ≈ the
all-token patch ⇒ the partner signal is carried at the endpoints. Evidence for
*where*, not a separate gate.

### Specificity — C on conflicts vs matched ordinary  (tool: `trace_heads.py` on O ✓)
Grouped ablation effect of C on **V\_c** must **exceed** its effect on the matched
ordinary set **O** (built in `discovery/ordinary.jsonl`). Registered GO requirement.

### Step 6 — edge / path decomposition, QK vs OV  (tool: **GAP — see G2**)
Registered but **not yet implemented**. Node patching (steps 4–5) shows heads
*matter*; to claim a *path* (e.g. an acceptor-token query attending to the donor
key) we need attention-knockout / path patching (Geva et al. 2023; Goldowsky-Dill
et al. 2023). Until built, step-6 claims are **exploratory**, not confirmatory.

### Step 7 — replication across seeds  (tool: loop steps 1–5 over the 5 additive-frozen checkpoints)
**PASS rule (prereg):** ≥ 3/5 seeds pass steps 3 and 4 in the same direction.
Also report component-identity overlap (do the same layer/head positions recur?)
and computation-level consistency even where identities differ.

### Step 8 — one-time held-out test  (only after freeze)
Freeze the component set + scores. Build the **test** conflict subset with the same
registered procedure, run necessity + mediation once, report, and stop. No head
reselection after opening test.

## 4. Pre-registered decision rules and failure interpretations

- **GO to test unblinding** requires: step-3 necessity PASS, step-4 mediation PASS
  (partner > context), specificity PASS (V\_c effect > O effect), replication ≥ 3/5.
- **Necessity without mediation** → components are involved but the causal path is
  incomplete; report as such, do not claim a circuit.
- **Mediation without partner-specificity** (swap/specificity fail) → a
  site-strength mechanism, not partner-matching.
- **Fails on test** → confirmatory claim not supported; the val result stands only
  as exploratory.

## 5. Build-first gaps — status

- **G1 — instrument tests.** ✅ DONE. `tests/test_circuits_hf_bert.py`, 9/9 pass
  (round-trip patch identity, hook cleanup, mask logic, random-set exclusion).
  Correction found while building: patching per-head attention context does NOT
  recover clean from corrupt (residual/MLP path unpatched), so the invariant is the
  clean→clean round-trip identity, not recovery≈1.
- **AtP fast screen.** ✅ DONE + validated. `circuits/attribution.py`,
  `tests/test_attribution.py`; Spearman 0.876 vs exact, top-8 overlap 8/8.
- **G3 / G4 — splits.** ✅ DONE. `scripts/splicebert/build_discovery_splits.py`
  → gene-disjoint D/V\_c (313/309 genes) + matched O; `split_manifest.json` hashed.
- **G2 — path/edge tooling** (attention knockout at QK / path patching). Still open.
  Optional for C1–C3; required only for confirmatory *path* claims (step 6 stays
  exploratory until built).
- **G5 — GPU harness.** In progress: a single `run_tracing.py` driver (AtP →
  exact confirm → grouped null → mediation both masks → specificity → 5 seeds),
  runnable on MPS (smoke) and CUDA (Colab L4/A10 real run). Scale-to-fit; gates
  unchanged. AtP + top-16 shortlist keeps the exact 1000-set null cheap, so an
  L4/A10 suffices.

## 6. Deliverables

`circuits/head_screen/…` (effects, consensus, random-null), `circuits/patching/…`
(mediation reports), a `TRACING_RESULTS.md` filling the causal/mediation/replication
sections of `RESULTS.md`, and an updated `return_to_planner.json`. Component set +
scores hashed into `FREEZE.md` before step 8.
