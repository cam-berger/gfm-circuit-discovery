# PREREGISTRATION D14 — What carries the TSS polarity: arrangement or composition?

> **Tier: T1 — frozen protocol.** Frozen 2026-07-29 before any confirmation-split scoring.

> **Status: FROZEN on 2026-07-29, before any confirmation-split scoring of the two native arms.**
> Supersedes the intent of D13 (`PREREGISTRATION_D13_COMPOSITION_VS_POSITION.md`), which is
> **retired unrun** — see §9.

## 0. Intent (the north star)

D12 confirmed a TSS polarity: masking the downstream tiles `{25, 27}` costs more supervised
promoter margin than masking the upstream tiles `{21, 22, 23}` (`E2 = +0.3262`,
95% CI `[+0.1110, +0.5564]`, Holm `p = 0.0096`, n = 149).

**The question this study exists to answer:** *what property of the sequence carries that
polarity — the arrangement of the bases (motif syntax) or their composition (GC/CpG content) —
and is the answer an artifact of the `[MASK]` token?*

This matters because a composition answer makes the phenomenon a causally-demonstrated
rediscovery of CpG-island promoter architecture, whereas an arrangement answer would make it a
syntactic computation worth tracing to a circuit. The study must be able to return either.

## 1. Why the D13 design is retired

D13 tried to separate composition from position by matching pairs of loci across the genome on
GC (caliper 0.05) and CpG (caliper 0.03). Three fatal problems, all confirmed:

1. **Its planted positive was confounded with its result.** It compared naturally unmatched
   against matched pairs, but the size of that gap *is* the quantity D13 measures. The control
   fails exactly when the hypothesis is true. No sample size repairs this.
2. **It was underpowered for its own effect.** MDE 0.241 against a plausible gap of 0.06–0.10.
3. **Its "position" arm is architecturally unavailable.** DNABERT-2 has no position embeddings
   and its ALiBi bias is `-m_h·|i-j|` (`bert_layers.py:394`, `torch.abs`), so the model cannot
   represent "downstream of the TSS" as a positional fact at all. The dichotomy had an empty arm.

D14 replaces cross-locus matching with **within-locus interventions that hold composition exactly
fixed by construction**, which is both exact and far better powered.

## 2. Architectural facts (deductive, no compute; established Phase 0)

| fact | source |
|---|---|
| no position embeddings | `bert_layers.py`, `# no position embeddings! ALiBi` |
| ALiBi bias `-m_h·|i-j|`, **symmetric** | `bert_layers.py:394` |
| classifier reads `hidden_states[:, 0]` = `[CLS]` at window index 0 = TSS-relative **−249** | `bert_layers.py:501` |

Consequence: because ALiBi is monotone decreasing in `|i-j|` and `[CLS]` sits at index 0, the
positional term **uniformly favours upstream in every head at every layer**. Measured token
separation downstream-minus-upstream is **8.45 tokens**, giving a per-head upstream headwind of
`m_h × 8.45`: from 5.98 logits (H8) down to 0.033 (H7). Only the shallow band H5–H7
(headwind ≤ 0.13) can support downstream dominance through direct `[CLS]` attention without a
large content term.

**This is reported as architecture, not as evidence.** It constrains Track B (deferred, §8); it
is not an endpoint here.

## 3. Interventions

All three are applied **per 10-nt tile**, at exactly D12's E2 tiles, so the endpoint is
structurally identical to the one D12 confirmed. Enumeration of dinucleotide-preserving reorders
is exponential in window width (w=22 → 878,399 candidates in 26 s; w=24 times out), so per-tile
application is also the only tractable choice.

| arm | operation | what it destroys | what it holds fixed |
|---|---|---|---|
| **A** | `mask` | content **and** manifold (inserts `[MASK]`) | nothing |
| **B** | `dinucleotide_reorder` | arrangement | mono- **and** linear-dinucleotide counts **exactly** (so GC *and* CpG counts are exactly preserved); real bases |
| **C** | `gc_depletion` (dose 3) | GC/CpG content | purine/pyrimidine class; real bases, no `[MASK]` |

Arm B is the exact composition control D13 tried to approximate. Arm C exists to separate
"composition" from "`[MASK]` is off-manifold for a classifier fine-tuned only on clean sequence."

## 4. Estimands

Per locus, for arm `X`: `E2_X = mean(target_specific_loss over available {25,27}) −
mean(over available {21,22,23})`, requiring ≥1 tile on each side (D12 v2 completeness rule).
`target_specific_loss = (margin(target_clean) − margin(target_edited)) −
(margin(control_clean) − margin(control_edited))`.

| id | estimand | what it actually tests |
|---|---|---|
| **P3** (primary) | `E2_reorder` against zero | **the arrangement contribution itself** |
| **P1** (primary) | paired `E2_mask − E2_reorder` | whether the polarity is **exhausted by** arrangement |
| **P2** | paired `E2_gc_depletion − E2_reorder` | does composition carry it, using real bases? |
| **S1** (secondary) | paired `E2_mask − E2_gc_depletion` | is a `[MASK]` artifact required? |
| **S2** (descriptive) | all of the above stratified by TATA / Inr call | biological structure |

**Estimand↔intent note (caught by the §2 red-team, recorded rather than silently fixed).** An
earlier draft labelled P1 "does arrangement carry the polarity?" That is wrong.
`E2_reorder` *is* the arrangement contribution, so **P3** is the direct test; `mask − reorder`
measures the **non-arrangement remainder** (composition + any `[MASK]` artifact). Both are primary,
and the intended conclusion needs both: P1 excluding 0 **and** P3 near 0.

**Analysis set: the three-arm intersection** (loci carrying all of A, B, C).

This is not merely tidiness — it neutralises the main identifiability threat. Arms B and C both
change BPE segmentation, and both must pass the same retokenization-geometry filter, which retains
only 68–78% of tiles. A null in arm B could therefore mean "arrangement does not matter" *or*
"the filter selected tiles where nothing matters." Restricting to loci where **both** B and C were
scored settles it internally: on development those same 90 loci give C = **+0.344** and
B = **+0.017**. Identical loci, identical filter, opposite answers — so the filter does not
suppress effects, and arm B's null is about arrangement.

Inference: bootstrap over unique locus blocks, 8000 draws, seed `20260729`, α = 0.05, Holm across
`{P1, P2, P3}`. **SESOI = 0.15** logit margin (unchanged from D13).

**Bootstrap-clustering disclosure:** in this cohort `homology_cluster` is **1:1 with loci**
(192 loci, 192 clusters, max 1 per cluster in both splits). The "cluster" bootstrap is therefore a
plain bootstrap. Flagged as **architecture-forced**: it provides no clustering robustness here and
must not be cited as if it did.

## 5. Decision rule — frozen before the confirmatory run

- **Arrangement is NOT the carrier** iff P1's CI excludes 0 **and** P3's CI includes 0. The
  strength of the second clause is reported as the **achieved upper bound** = P3's CI upper limit
  (projected ≈ 0.31), **not** as equivalence at the SESOI — see the power caveat in §7.
- **Composition SUPPORTED as the carrier** iff P2's CI excludes 0.
- **`[MASK]` artifact REQUIRED** iff S1's CI excludes 0 in the positive direction.
- **Uninformative** if P1's CI includes 0 — the study then licenses nothing, and in particular
  does *not* license "arrangement matters."

Any endpoint whose CI includes both 0 and its development point estimate is reported as
**underpowered**, not as a null.

**Explicitly not claimable:** "arrangement contributes nothing." With n ≈ 90–113 and sd 1.166 the
design cannot resolve an arrangement contribution below ≈ 0.31, which exceeds the 0.15 SESOI.
Establishing arrangement ≈ 0 at the SESOI would need ≈ 474 loci.

## 6. Memorials — on the measurement path, able to fail

| id | memorial | can fail? |
|---|---|---|
| M1 | `dinucleotide_rule`: for every arm-B edit, `Counter(region)` **and** `linear_dinucleotide_counts(region)` are preserved exactly | yes — implemented at `promoter_native_source.py:248` |
| M2 | `GC_depletion_rule`: arm-C edits reduce G+C by exactly the declared dose | yes |
| M3 | `non_identity`: Hamming > 0 for every edit in both arms | yes |
| M4 | equal Hamming dose between target and pseudo-TSS control | yes |
| M5 | batch invariance: rescoring under a different batch partition reproduces every margin to < 1e-4 | yes |
| M6 | clean-behaviour anchor: `mean(margin(target_clean) − margin(control_clean)) > 0` | yes |

**Positive control — exact replay (this is the one D13 lacked).** D12 stored its per-locus,
per-tile confirmation mask losses (`data/processed/dnabert2_atlas/promoter_domain_v1/
confirmation.raw.jsonl`, 979 rows, tiles 21/22/23/25/26/27/28/29). Arm A is therefore **not
re-estimated**; it is **replayed**. Gate: rescoring arm A through this study's pipeline on a
32-locus subset must reproduce every stored `target_specific_loss` to `< 1e-4`.

This replaces an earlier CI-containment gate ("reproduced mean must fall inside D12's CI"), which
the §2 red-team rejected: arms B and C survive a retokenization-geometry filter, so arm A on the
qualifying subset is a *selected* subset of D12's loci and its mean legitimately differs (on
development: +0.5752 on the 90-locus intersection vs +0.4045 on all 160). A CI gate would have
failed for a reason that is not instrument failure. Exact replay has no such confound.

**Specificity control:** arm B applied to a far-upstream inert tile (tile 5, TSS-relative
−199:−189) must give `|effect| < 0.15`.

**Dose-confound control (identifiability).** The three arms perturb different numbers of bases, so
a larger P1 could in principle be a larger perturbation rather than a different *kind* of
perturbation. Mean Hamming per arm is reported alongside every endpoint. The defence is empirical
and already visible on development: arm C changes **3** nt and yields **+0.34**, while arm B
changes **4.76** nt and yields **+0.02**. Effect is **not monotone in dose**, so P1 is not a dose
artifact. If this ordering reverses on confirmation, P1 must be reported as dose-confounded.

## 7. Power — stated before the run, honestly

From Phase 0 (development split, n = 192 loci → 113 with arm B, 120 with arm C, 90 with all three):

| endpoint | role | sd | projected n | MDE @80% | development estimate | verdict |
|---|---|---:|---:|---:|---:|---|
| P1 | primary | 1.5055 | ~113 | **0.397** | +0.5303 | **powered** |
| P3 | primary | 1.1658 | ~113 | 0.307 | +0.0429 | **bounds arrangement at ≈0.31, above the 0.15 SESOI** |
| P2 | — | 1.5185 | ~90 | 0.448 | +0.3271 | **underpowered — declared** |
| S1 | secondary | 1.9341 | ~90 | 0.571 | +0.2315 | **badly underpowered — declared** |

P3 is primary but is an **upper-bounding** endpoint, not an equivalence test: it can show the
arrangement contribution is small, it cannot show it is below the SESOI (that needs ≈474 loci).

**Variance reduction was attempted and rejected.** Averaging 4 independent reorder replicates per
tile cut sd by 16% (1.1658 → 0.9753) but collapsed supply from 113 to 24 loci, making MDE *worse*
(0.307 → 0.557). Recorded so it is not retried.

**P1 is the primary and is adequately powered. P2 and S1 are reported at achieved precision and
cannot support a null.** Powering P2 to MDE 0.25 would need ~289 three-arm loci (~615 raw), i.e. a
new cohort build; that is deliberately **not** attempted here rather than risking a defective
cohort, and is recorded as the extension if P2 lands ambiguous.

## 8. Scope — what this study does not claim

- **Track B (mechanism: which heads, relay vs direct) is deferred.** §2 constrains it but no
  attention or ablation endpoint is preregistered here.
- "Arrangement" means **arrangement modulo exact dinucleotide content**. A dinucleotide-preserving
  shuffle retains CpG count exactly and much local structure; a motif can partially survive it.
- Arm C is **directional** (GC only decreases) at a fixed dose of 3; it shows GC content matters
  asymmetrically, it does not isolate GC from other correlates of G/C bases.
- Nothing here speaks to endogenous promoter function in cells.

## 9. Seal status — stated precisely

The D8 confirmation split (192 loci) was opened by D12 for the **`mask`** endpoints E1/E1b/E2/E3/E4.
It has **never been scored under `dinucleotide_reorder` or `gc_depletion`**. Therefore:

- arms **B and C are sealed** on these loci;
- arm **A is already known** there (`E2_mask = +0.3262`), which is precisely why it serves as the
  positive control rather than as an endpoint;
- the paired estimands P1, P2, P3 and S1 have **never been estimated on these loci**.

This is a **partial seal** and is labelled as such. It is weaker than a fresh gene-disjoint cohort
and stronger than reusing a fully-opened endpoint. Phase-0 development results are **exploratory**
throughout and are recorded in `results/.../promoter_arrangement_composition_phase0_v1.json`.
