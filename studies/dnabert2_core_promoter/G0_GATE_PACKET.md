# G0 gate packet — computation lock

> **Tier: T2 — provenance.** Evidence for a gate decision. **The gate is not signed.**
> Prepared 2026-07-26. No confirmatory model output has been computed; the confirmation split does not
> yet exist.

---

## 1. Frozen question

> Does frozen DNABERT-2's native MLM use **TATA identity** and **native TATA-to-TSS geometry** when
> reconstructing a masked TSS-centred token span in human TATA-associated promoters — beyond
> target-token frequency, local composition, AT-richness, BPE segmentation, generic promoter identity,
> and arbitrary editing?

**Primary direction.** Context = the TATA-associated upstream span. Target = the BPE tokens overlapping
TSS positions −3 … +3. Readout = length-normalised true-token **pseudo**-log-likelihood at the
simultaneously masked target tokens. The reverse direction is secondary and is never pooled with it.

**Name.** *TATA→TSS assembly*, not *TATA→Initiator assembly*. Inr is a **stratum**, not an inclusion
requirement (strong-Inr and TATA signals are partly alternative human promoter architectures).

**Claim boundary.** A statement about a **computation implemented by a frozen checkpoint**. Not TFIID,
not Pol II, not transcription rate, not chromatin, not physical PIC assembly. "Assembly" never appears in
a licensed claim without its operational gloss (*identity* **and** *TSS-relative position*).

**Excluded from the primary claim:** broad/CpG promoters, TATA-less focused promoters, DPE/DPR promoters.

---

## 2. Evidence table (as supplied, with the local consequence actually implemented)

| Biological evidence | Consequence implemented here |
|---|---|
| Core promoter ≈ −40…+40; combinations, not a universal motif | window −96…+64 contains it with flanking context; claim restricted to local core-promoter organisation |
| Focused promoters ↔ a precisely positioned initiation complex | **BLOCKER B1 below** — asserted but not yet measurable from EPD |
| Functional TATA placement concentrated ≈ −32…−29 by motif start | coordinates frozen to one convention and **verified** (§3); geometry tested, not just presence |
| Focused human Inr ≈ −3…+3, `CA(+1)` central | target = tokens overlapping −3…+3; exact nt coverage reported per example |
| Element combinations and distances affect transcription; ~10-bp structure | spacing treated as a **response curve**; helical interpretation reserved for a later study |
| CAGE / 5′-GRO-seq anchor TSSs experimentally | EPDnew primary; FANTOM5 as annotation cross-check, never a model output |

Sources as supplied: Vo ngoc 2017 (PMC5287114); Ponjavic 2006 (PMC1779604); Weingarten-Gabbay 2019
(PMC6360817); EPD/EPDnew; FANTOM5 (PMC4529748); DNABERT-2 (arXiv:2306.15006).

---

## 3. Coordinate lock — **decoded and verified, not asserted**

Rival **R6** (annotation offset / mixed convention) is the cheapest way to destroy this study, so it was
closed first.

**Convention.** TSS = `+1`, no zero. Transcription-strand orientation. From the EPDnew BED:

```
+ strand :  0-based TSS = thickStart = start + 49
- strand :  0-based TSS = end - 50        (thickEnd = 1-based TSS)
```

**Verification (run):** for 3,000 randomly sampled promoters, the 60-nt window extracted from hg38 and
transcription-oriented equals the EPD `SE` line **3,000/3,000**, with the `SE` case layout exactly
49 lowercase (−49…−1) + 11 uppercase (+1…+11) in **3,000/3,000**, and `thickStart`/`thickEnd` consistent
with the rule in **1,554/1,554** (+) and **1,446/1,446** (−).

**TATA anchor.** **Motif START**, JASPAR **MA0108.3** (TBP, 7 nt, consensus `TATAAAA`), relative-score
threshold **0.85**. Functional interval **−33 … −27**.

**Anchor memorial (run, passes).** The motif-start histogram on EPD promoter sequence is sharply peaked
where the literature says it should be — which is what makes the convention *verified* rather than
assumed:

| motif start | −33 | −32 | −31 | −30 | −29 | −28 | −27 |
|---|---|---|---|---|---|---|---|
| n | 328 | 635 | **770** | **866** | 556 | 101 | 41 |

Mode −30/−31; 89% of mass in −33…−29. **Cohort: 3,297 / 29,598 = 11.1%**, consistent with published
TATA-promoter prevalence.

---

## 4. Rivals R0–R9 — each with an assigned control

| # | Rival | Control |
|---|---|---|
| **R0** | target BPE words are common and easy | **arithmetically cancels.** Target token ids are invariant across arms (M2), so target identity cannot contribute to any paired contrast. B0 remains descriptive |
| **R1** | local mono-/dinucleotide composition | all arms are **permutations of one another** — mononucleotide composition identical by construction (M4); dinucleotide distance minimised and reported |
| **R2** | AT-richness, not TATA identity | same as R1: `W` and `F2` are permutations, so AT content is *identical*; only arrangement varies |
| **R3** | any upstream edit / any retokenisation | `B(d)` arm is itself an edit; plus `C2` distant-window control; plus the filler-vs-filler sham |
| **R4** | additive position-specific PWM rule, no integration | baseline **B3** (no score×distance term) must not reproduce both primaries |
| **R5** | generic promoter / CpG state | CpG and promoter-class strata; `A2` TATA-like non-promoter decoy |
| **R6** | annotation offset / mixed convention | **closed in §3** — one convention, verified 3,000/3,000, anchor histogram passes |
| **R7** | duplicates, paralogs, promoter families | gene ∪ ≥90%-identity clustering before splitting; clustered bootstrap |
| **R8** | synthetic edit is OOD | edits are permutations of native letters; sequence/tokenisation plausibility and generic masked-DNA health recorded per arm |
| **R9** | human-reference result reflects pretraining exposure | primary estimand contrasts **two synthetic arms** (memorisation-free); `Psi_content_A − Psi_content` reports the memorisation component; cross-species arm deferred |

**Additionally identified locally (not in the supplied rival list):**

| # | Rival | Control |
|---|---|---|
| **R10** | **BPE segmentation shifts ALiBi geometry between arms.** Measured on real promoters: total token count differs in **31–39%** of naive TATA-vs-filler pairs, inter-slot token boundaries in **45–49%** | **M2b** — fillers are *selected under* a token-geometry-invariance constraint. Designed out, not modelled away. Feasible for **98.3–100%** of promoters |
| **R11** | "value decays with distance" is generic, not promoter-specific geometry | geometry primary anchored on a slot **closer** to the target than native (−23), where generic decay and promoter grammar predict **opposite** signs |

---

## 5. Instrument facts established (run, this branch)

| Fact | Value |
|---|---|
| MLM head | **complete and trained** — `cls.predictions.decoder [4096,768]`; `missing=[]`, `unexpected=[]` |
| Learned position table | **absent** (ALiBi). `config.position_embedding_type: "absolute"` is misleading |
| Attention path | PyTorch reference (Triton absent) — probabilities materialised, hookable |
| Load path | vendored @ pinned revision + `transformers==4.44.2`; fails on 5.13 and on 4.44 via AutoModel |
| nt/token | 4.66 (recorded) / 4.77 (uniform-random re-measure) |
| Retokenisation reach | median 2 nt, p90 7, max 16 past an 8-nt edit |
| Target invariance @ ≥20 nt buffer | 0.0–0.2% failure on real promoters; 3.2% at slot −23 |
| Matched-filler availability | 98.3–100% of promoters, median 32–43 candidates |

---

## 6. G0 checklist, and the two blockers

| Requirement | State |
|---|---|
| Computation statement, direction, target span, readout frozen | **yes** |
| "TATA→TSS assembly" is the primary name | **yes** |
| Inr is a stratum, not an inclusion requirement | **yes** |
| Broad/CpG and TATA-less architectures excluded from the primary claim | **yes** |
| R0–R9 each assigned a control | **yes** (+ R10, R11 added locally) |
| Claim boundary separates model computation from transcriptional function | **yes** |

### BLOCKER B1 — "focused" is asserted but unmeasured

EPDnew `.dat` and `.fps` carry **no TSS shape/focus score**. The frozen question currently says
"focused" without a criterion. Two admissible resolutions, gate-owner's choice:

* **(a)** acquire FANTOM5 CAGE peaks (verified reachable) and freeze a shape index as an inclusion rule; or
* **(b)** delete "focused" from the cohort definition **and** from every licensed claim, scoping the
  study to *TATA-associated EPDnew promoters*.

It may not be asserted unmeasured. This is a **G1** blocker, not a G0 failure — but it changes the
frozen claim text, so it is raised here.

### BLOCKER B2 — window not yet frozen

−96…+64 is a *candidate*. It must survive the window-length sensitivity preflight (a 160-nt input is
~34 tokens, potentially short-OOD for this checkpoint) before it is frozen.

---

## 7. Recommendation

**`PASS` on G0, conditional on the gate owner resolving B1** (the "focused" scope), with B2 discharged
during Phase-1 preflight.

The Phase-0 requirements are met, the coordinate convention is *verified* rather than assumed, every
rival has a control, and two confounds not present in the source plan (**R10**, **R11**) were found and
designed out before any compute. What is **not** established is whether the behaviour exists or is
detectable — that is Phase 3, and the two live risks (statistical power; whether `ell` responds to
*anything* 20+ nt away) carry explicit stop rules in the preregistration.

**Not run:** any confirmatory model output; cohort construction; baselines; the confirmation split.
**Strongest claim licensed today:** none. This packet licenses a gate decision, nothing else.
