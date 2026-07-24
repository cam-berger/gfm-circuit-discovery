# PREREGISTRATION — P3.4: what general computation created the period-3 carrier? (v1)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: NOT FROZEN — the 4-lens design red-team (2026-07-18) found this design NOT freezable as written.**
See `REDTEAM_SYNTHESIS_P3_4.md`. Three lenses returned `fatal`: **(1)** origin↔usage intent drift — a
frozen-checkpoint ΔL measures what the carrier is *used for*, not the training *origin* the intent claims;
**(2)** E4a identifies codon **composition**, not reading-frame (code-verified: `s − φ` is a per-example
constant, so slot cannot vary at fixed absolute phase, and the carrier is ~constant at fixed φ); **(3)** the
0.02-nat powered-equivalence bar is **unreachable at MPS budget** (verified vs C8: raw-mean CI half-width
0.0198 at n=384; every estimand here is a higher-variance contrast at ≤140 units), so the maintained-hypothesis
(equivalence) readings cannot be confirmed. Data **is** buildable (no engineering blocker; `codon_slot`
verified on real frame-2 sequence) — the blocker is **scientific**. This v1 is retained as the **design of
record**; it must **NOT be run** without the re-scope/redesign in the synthesis **and** a Cam decision on
direction (reframe-narrow-usage / training-time-origin / defer-and-transfer).

Modernizes the old `PREREGISTRATION_PERIOD3_COMPUTATION.md` (thread 2a) — **do not run that verbatim** (Cam):
it compared **raw ΔL** across contexts, which conflates the carrier's *role* with per-context baseline
difficulty and with the absolute-position ⊥ codon-slot confound. This v1 reframed every test to a
**carrier-deletion × context INTERACTION** with mandatory controls — but the red-team showed the interaction
framing, while necessary, is not sufficient to make the origin question identifiable on a frozen checkpoint.

**Separately gated (Cam):** P3.4 is an **origin / interpretation** study at the P→behaviour(MLM) level. It
asks *why SpliceBERT learned the carrier*, not *what it does in the coupling circuit*. **No P3.4 outcome can
redefine the sealed causal splice results** (C8, C19, C34, C35, …); a coding-origin finding would explain the
carrier's existence, not change its measured causal role. Conversely a "positional-coordinate" finding does
not weaken the coding-MLM benefit (C8).

Provenance: SpliceBERT.1024nt sha256 `2ad91428…0149d`, post-LN BERT-MLM 6L×16H, hidden 512, **absolute**
learned position embeddings P (1026×512). Env `gfm-thrust1`; compute **MPS**. Deterministic cohorts
(`PYTHONHASHSEED=0`, `sorted(set())`).

Name under test (the maintained hypothesis, C8/C19/C32):
> **A learned period-3 positional carrier** — a rank-2 cyclic coordinate read as relative-offset classes,
> **not** a codon-frame computation. P3.4 is precisely the test that could force "codon-frame" back in (if the
> biological reading-frame estimand E4a fires) or confirm the positional-coordinate reading.

---

## 0. INTENT (north star — re-read at every gate)

*What general MLM computation caused SpliceBERT to learn the period-3 positional carrier?* Candidate roles
(the prompt's truth table): **within-codon completion** (a translation/coding computation), a **coding-state
scaffold**, **coding/noncoding segmentation**, a **general modulo-3 routing substrate**, **biological
reading-frame use**, or a **pure absolute-positional artifact**. The carrier's MLM benefit is coding-enriched
(C8: deletion costs coding MLM 0.159 vs intronic 0.042 nats) — P3.4 asks *why*: does the carrier **compute**
codons, or is it a positional coordinate that is merely **more useful** in coding regions?

Model-native observable: the **carrier benefit** `ΔL = L_masked(carrier-deleted) − L_masked(intact)` (the
true-base NLL increase at masked positions when the period-3 clock is removed), measured under masking,
context, codon-slot, and translation manipulations — always as an **interaction/contrast**, never raw ΔL.

**This does not touch the sealed test (chr8/9); held-out on a disjoint val-gene split.**

---

## 1. ESTIMANDS (all are contrasts / interactions — never raw ΔL across unmatched contexts)

Carrier conditions on P: **I** (intact), **D** (`make(BAND)`, BAND=`range(339,346)`; the exact C8 deletion).
Per masked position `x`: `ΔL(x) = L_D(x) − L_I(x) = −logP_D(true) + logP_I(true) ≥ 0` typically (deletion
hurts). Gene-clustered bootstrap CIs (`gboot`); every contrast is a **paired within-position/within-gene
difference**, not a difference of two arm means.

### E1 — Within-codon COMPLETION (primary discriminator)
At a coding **target** base with a well-defined codon (slot s, mates at the other two codon positions), three
masking conditions, **all else identical**:
- **A (single):** mask only the target — codon-mates **visible**.
- **B (whole-codon):** mask the target **and its two codon-mates** — mates **hidden** (3 masked).
- **C (burden control):** mask the target **and two period-3-aligned same-slot bases at ±3/±6** (same codon
  slot, *different* codons) — codon-mates **visible**, **3 masked** (matches B's mask burden).

`E1 = ΔL(C) − ΔL(B)` — both mask 3 bases; **B hides the codon-mates, C hides distal same-slot bases**.
- **E1 CI-low > SESOI → within-codon completion** (hiding the *mates specifically* kills the carrier benefit).
- **E1 ⊂ ±SESOI (equivalence) → NOT completion** (a phase-conditioned prior / scaffold — hiding the mates
  does not matter). Raw `ΔL(A)−ΔL(B)` reported descriptively (burden-confounded; not the estimand).

### E2 — Coding-state SCAFFOLD (CDS vs matched noncoding), single-mask
`E2 = ΔL(CDS interior) − ΔL(matched noncoding)`, where noncoding (intron/UTR) is **matched per-target** on:
absolute position (±, so carrier phase matched), local GC, local 3-mer composition, **and baseline difficulty**
`L_I` (intact-model NLL bin). This is the **carrier-deletion × coding-state interaction at matched
difficulty** — the fix for the old prereg's raw CDS-vs-intron ΔL (which conflated coding-ness with difficulty).
- **E2 CI-low > SESOI → coding-state scaffold**; **equivalence → benefit not coding-specific** (→ E-general).

### E3 — Coding/noncoding SEGMENTATION (boundary vs interior)
`E3 = ΔL(CDS boundary ±k) − ΔL(CDS interior)`, difficulty/composition-matched.
- **E3 CI-low > SESOI → segmentation** (benefit peaks at coding boundaries).

### E4 — Biological reading-FRAME vs ABSOLUTE phase (the identifiability crux)
The carrier is **absolute-position-indexed**, so biological codon slot `s` correlates with absolute position
`φ = pos mod 3`. Two dissociating estimands:
- **E4a (slot at fixed absolute phase):** a 3×3 design, ΔL by (biological codon slot `s∈{0,1,2}`) × (absolute
  phase `φ = pos mod 3`). Estimand = the **slot main effect controlling for φ** (mixed model; slot contrast
  within each φ, requiring all 9 cells populated — see §5). **CI-excludes-0 slot effect at fixed φ →
  biological reading-frame use.** Flat across `s` at fixed `φ` → not frame.
- **E4b (translation leakage):** whole-sequence content shift by **+1/+2/+3** with **natural flanks** (the
  content moves relative to the absolute position index). `+3` preserves absolute carrier phase (`3≡0 mod 3`)
  **and** biological frame; `+1/+2` **rotate the absolute carrier phase** but keep the content/codon structure
  intact. `E4b = ΔL(+1,+2 avg) − ΔL(+3)` on the same masked content.
  - **E4b CI-excludes-0 → the benefit is absolute-carrier-phase-locked** (rotating phase disrupts it).
  - **E4b ⊂ ±SESOI → phase-invariant** (content/frame-driven, not absolute-phase).
- **Composite:** slot-dependence at fixed φ (E4a≠0) **and** phase-invariance (E4b≈0) → **biological reading
  frame**. φ-locked (E4a≈0 across s, E4b≠0) → **absolute positional coordinate, not frame** (the maintained
  hypothesis, C19/C32).

### General-substrate readout (not a separate manipulation)
"General modulo-3 routing substrate" = **E2 equivalence** (benefit present in matched noncoding too) **with a
non-zero carrier benefit overall** (M-sanity) and **E1 equivalence** (no completion). Reported as the residual
interpretation when E1–E4 are all null/equivalence except a broad main effect.

---

## 2. METHOD (reused machinery + new, piloted plumbing)

Reused: carrier deletion + construction anchor (`run_carrier_surgery::make`, BAND 339–346); MLM true-base NLL
+ health (`circuit.CouplingTask.mlm_health`, R13-hardened masked-probe); coding/annotation frame
(`gfmcd.splicing.phase`, extended to per-base codon slot from the containing CDS); gene-clustered CIs
(`gboot`). Adapter `eager=True` not required (no attentions read here) but kept for parity.

**New plumbing (piloted in `--smoke`, must pass before freeze):**
- **Per-base codon-slot + CDS/UTR/intron/noncoding annotation** from the GTF CDS features (start-phase aware);
  boundary tagging (CDS-start/CDS-end ±k). **Validated** (M-annot below): translate a known CDS → start codon
  at slot 0; slot cycles 0,1,2 in transcription order; strand-correct.
- **Multi-base masking** at chosen offset sets (single / codon-triple / burden-triple), true-base NLL summed
  at the **target only** (B and C hide extra bases but the readout position is the same target — comparability).
- **Per-target matched-control sampler** (E2/E3): deterministic caliper matching on (abs-pos band, GC bin,
  3-mer profile, `L_I` difficulty bin), `sorted(set())` + pinned hashseed.
- **Whole-sequence translation with natural flanks** (E4b): shift the analysis window content by +1/+2/+3 with
  genuine genomic flanks (not padding), re-annotate, re-mask the same content bases.

**EXCLUDED (R14):** no ±1/±2 **head-roll** or attention phase-shift assay (stride-3 manufactures phase
effects). Note E4b shifts **sequence content vs the position index**, deleting/keeping the carrier by
construction — it is *not* a comb-roll on attention; the carrier is DELETED (bins zeroed), never rolled.

---

## 3. MEMORIALS (on the measurement path; each CAN fail)

| # | Memorial | Passes iff | Note |
|---|---|---|---|
| M-sanity | carrier benefit reproduces C8 direction | `ΔL(coding, single)` gene-CI-low > `ΔL(intron)` and > 0 | failable; halts if wrong-signed |
| M-del | deletion is the exact C8 clock | `make(BAND)` zeroes rfft bins 339–345 of (P−μ); element-wise reconstruct anchor <1e-6 | construction anchor |
| M-annot | codon-slot annotation correct | on ≥50 known CDS: start-codon base = slot 0; slots cycle 0/1/2 in tx order; strand-correct; 100% | failable (frame/strand bug) |
| M-mask | mask machinery + burden parity | INTACT `L(whole-codon) > L(single)` (more masking ⇒ more loss); **B and C mask exactly 3 bases each**; readout is the shared target | construction + failable |
| M-balance | E4a absolute-phase balance achieved | after matching, max \|cell-count imbalance\| across the 9 (s×φ) cells < 20%; every cell ≥ n_min | failable ⇒ E4a **non-identifiable**, reported as such |
| M-match | E2/E3 common support | matched pairs achieve \|Δabs-pos\|≤W, \|ΔGC\|≤c, \|Δdifficulty-bin\|≤1; ≥ n_min pairs | failable ⇒ E2/E3 non-identifiable |
| M-health | global MLM health under D | deleted-model masked-probe health ≥ 0.5 headroom over chance (R13); C8-level, gated not assumed | below-floor → halt |
| M-det | determinism | PYTHONHASHSEED pinned; `sorted(set())`; cross-seed cohort identity | reproducibility (IC1 A2) |

---

## 4. CALIBRATION (sensitivity + specificity; hard preflight gates)

**Planted-positive (travels the real channel):**
- **PP-benefit:** coding single-mask ΔL > 0 (M-sanity) — the carrier benefit exists.
- **PP-completion (E1 sensitivity):** an **artificial codon-completion oracle** — construct positions where
  the true base is *deterministically* the XOR/def of its two mates (synthetic control sequences); masking the
  mates (B) must collapse a completion signal an intact model would have. Confirms E1 can DETECT completion.
  *(If the synthetic oracle is infeasible on real sequence, substitute a positive control where mates are
  perfectly predictive by annotation; documented at pilot.)*
- **PP-slot (E4a sensitivity):** inject a known slot-dependent signal (e.g., stratify by a base-identity proxy
  that varies with slot) → E4a must recover it at fixed φ.

**Null / specificity (must not fire above target FPR):**
- **N-permute-slot:** permute the codon-slot labels within φ → E4a slot effect ≈ 0 (specificity of E4a).
- **N-intron-floor:** deep-intron bases far from any CDS show small ΔL (the C8 0.042 floor), and E2's noncoding
  arm is *not* inflated by matching artifacts.
- **N-burden:** on a NON-coding triple (three adjacent intronic bases with no codon structure), `ΔL(triple) −
  ΔL(single)` reflects **only** mask burden → E1-style contrast between two *non*-codon triples ≈ 0 (the
  completion estimand must be null where there are no codons to complete).

**Confound-specificity:** E2/E3 count as coding-specific only **at matched difficulty + composition**
(M-match); E4a counts as frame only **at balanced absolute phase** (M-balance). Unmatched → non-identifiable,
never "coding-specific"/"frame".

---

## 5. IDENTIFIABILITY

- **Absolute phase ⊥ codon slot (the crux).** `s` correlates with `φ = pos mod 3`; separation needs genes
  whose CDS frames differ so a given `φ` hosts all slots. **Predefined overlap check (M-balance):** all 9
  (s×φ) cells populated ≥ n_min with <20% imbalance. If any cell is under-supported → **E4a non-identifiable
  → reported honestly, not forced** (cf. A9b V4).
- **Coding-ness ⊥ composition.** CDS differs from introns in GC/codon bias; matching on GC/3-mer/difficulty
  (M-match) may not fully de-confound (coding-ness is partly *definitional* over composition). If common
  support fails → E2 reported as a **composite** coding+composition effect, not a pure coding-state claim.
- **Mask burden (E1).** B hides 3 bases incl. the mates; C hides 3 incl. distal same-slot bases. E1 = C−B
  isolates *mate-hiding* from *burden*. Residual risk: the ±3/±6 distal bases in C are themselves
  period-3-aligned and could carry a completion-like signal → **E1 is a conservative lower bound on
  completion** (if distal same-slot bases help, C is depressed toward B, shrinking E1). Documented; a second
  burden control (distal *off*-slot ±1/±2 triple) reported as an upper-bound bracket.
- **Held-out ≠ sampling stability (R22).** Any bin/caliper/threshold set on a discovery gene split; confirmed
  on a **disjoint untouched** val-gene split; gene-clustered CIs; effect-existence via CI-excludes-SESOI.
- **Post-LN linearity** not required here (loss-level readout, not reconstruct-inject).

---

## 6. DECISION RULE (frozen; result → conclusion)

**Per-estimand SESOI (own scale, discovery-fixed):** SESOI_ΔL ≈ 0.02 nats (a fraction of the C8 0.159 coding
benefit; ≈ the 0.042 intronic floor scale). Each estimand: **positive** iff the family-adjusted CI-low >
SESOI; **equivalence/null** iff the adjusted CI ⊂ ±SESOI (CI half-width < SESOI); else **inconclusive**.

**Family control:** E1, E2, E3, E4a, E4b are **5 pre-registered contrasts** → max-T (positive) + TOST
(equivalence) at family-adjusted α; E4a's within-φ slot test is one test (not three rolled).

**Truth table → most-defensible reading (report ALL five; the composite is the conclusion):**

| Pattern | Reading |
|---|---|
| **E1 positive** (C−B > SESOI) | **within-codon completion** (a coding/translation computation) |
| E1 equiv, **E2 positive** | **coding-state scaffold** (carrier marks/serves coding regions) |
| **E3 positive** | **coding/noncoding segmentation** (benefit peaks at boundaries) |
| E1 equiv, **E2 equivalence** (benefit in matched noncoding too) | **general modulo-3 routing substrate** |
| **E4a positive** (slot effect at fixed φ) **and E4b equiv** | **biological reading-frame use** |
| E4a equiv **and E4b positive** (only absolute phase matters) | **absolute positional coordinate, not frame** |

**Endpoints:** **A (resolved):** *"SpliceBERT's period-3 carrier arose as {completion / coding scaffold /
segmentation / general positional coordinate}, and {does / does not} implement a biological reading frame."*
— reached iff the five contrasts give a coherent, health-/identifiability-gated pattern. **B (unresolved):**
mixed/inconclusive/non-identifiable (e.g., M-balance or M-match fails) → *"origin unresolved on this
evidence; {E4a non-identifiable / composition not separable}"* — a legitimate, honestly-scoped stop.

**Maintained-hypothesis note:** C8 predicts E2 positive; C19 (coupling phase-independent) predicts E4a ≈ 0.
A **surprising** E1-positive or E4a-positive would force reintroducing "codon-frame" into the carrier's name —
so those endpoints are pre-committed to be reported prominently, not explained away.

---

## 7. POWER / MDE & HELD-OUT

- **Power:** per-estimand MDE from a discovery pilot; every NULL/equivalence must be a **powered** equivalence
  (CI half-width < SESOI), never a wide CI (IC1/R22). E4a (3×3 within-φ slot) and E1 (second difference) are
  the most n-hungry → sized first; underpowered contrasts reported **inconclusive**, never null.
- **Held-out:** discovery split (subset of val genes) fixes SESOI, bins, calipers, boundary `k`, difficulty
  bins, matched pools; confirmatory on a **disjoint, untouched** val-gene split; untouched-gene confirmation
  after any selection. **Sealed test (chr8/9) NOT opened.** Deterministic cohort; gene-clustered CIs.

## 8. GATES
- **Global PASS = AND(** M-sanity, M-del, M-annot, M-mask, M-balance (for E4a) / M-match (for E2/E3),
  M-health, M-det, PP-benefit, PP-completion, PP-slot, N-permute-slot, N-intron-floor, N-burden,
  confound-specificity **).** Run STOPS on failure; a failed identifiability memorial (M-balance/M-match)
  **demotes only its estimand to non-identifiable**, not the whole run.
- **`--smoke`:** annotation validation (M-annot), mask-burden parity (M-mask), the construction anchor
  (M-del), determinism (M-det), and the matched/balanced samplers producing ≥ n_min cells — tiny cohort,
  **val sealed**. The "pilot the plumbing and freeze" step; must pass before freeze.

## 9. SCOPE
- **Reaches:** P→behaviour(MLM) characterization of the carrier's **general computational origin** (why it was
  learned), via carrier-deletion × context interactions.
- **Does NOT establish / cannot redefine:** the carrier's **causal role in donor coupling** (C34/C35 — sealed;
  a P3.4 coding-origin result does not change them); a **biological mechanism** (Biology level needs an
  independent source); anything about the **second** ≈10.63 multispecies carrier (C11, different feature).
- **Separately gated (Cam):** kept out of the causal-splice claim chain; reported as an origin/interpretation
  study with its own endpoints.
- **Provenance:** checkpoint sha; seeds + PYTHONHASHSEED to the manifest; results write a manifest beside
  output. **Amendments:** pre-run only, logged `Amendment (date, pre-run, no data seen)`; once the primary is
  inspected a fix is *outcome-aware verification*, not confirmatory (IC1 A2).
