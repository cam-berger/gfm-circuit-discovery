# P3.4 CARRIER ORIGIN — 4-lens design red-team synthesis (2026-07-18)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Reviews:** `../preregistrations/PREREGISTRATION_P3_4_CARRIER_ORIGIN.md` v1 (frozen-candidate). Four independent adversarial
lenses, run BEFORE any confirmatory compute (per Cam: "modernize and red-team the old preregistration; do not
run it verbatim"). **Outcome: the design is NOT freezable as written.** Three lenses returned `fatal`; the
core issues are identifiability + intent-drift + a power wall, all fixable only by **re-scope/redesign, not
footnotes**. This is an R23-class pre-run catch: the cycle prevented an expensive run whose modal outcome
would have been "inconclusive" on its primary question.

| Lens | Verdict | Headline |
|---|---|---|
| Estimand ↔ intent | fatal (fixable by relabel) | ΔL on a frozen checkpoint = **usage, not origin** |
| Identifiability | fatal | **E4a identifies codon composition, not reading-frame** (code-verified); E2/E3 collider on `L_I`; origin-vs-usage |
| Statistics | fatal (rescuable by re-scope) | **0.02-nat equivalence unreachable at MPS budget** (verified vs C8); E4a is a between-gene-frame-class comparison |
| Calibration / feasibility | fixable (NO data blocker) | **Data IS buildable; `codon_slot` verified on real frame-2 sequence (609/609)** — but calibration is holed: E1's planted-positive is likely DEAD on a frozen model |

---

## The three convergent FATALs (independently verified against code + data)

### FATAL-1 — Origin vs usage: the observable cannot see the question (estimand + identifiability lenses, unanimous)
- **Intent (§0, §9, Endpoint A):** *why did SpliceBERT LEARN the carrier* — a training-history/etiology claim
  ("arose as", "why it was learned", "computational origin").
- **Estimand (§1):** `ΔL = L_D − L_I` on the **converged, frozen** checkpoint = the carrier band's **marginal
  MLM contribution at the trained weights** = what the carrier is **currently used for**.
- **Why fatal:** a frozen-model ablation identifies current function, never formation pressure. A feature
  induced early by pressure X can be load-bearing for Y at convergence (recruitment/repurposing/superposition);
  deletion reveals Y. §9 explicitly promotes a "P→behaviour(MLM) characterization" into an "origin" claim — a
  ladder jump (Usage → Training-history) the project forbids. The design's own framing gives the tell: "does
  the carrier **compute** codons vs is it a positional coordinate merely **more useful** in coding" — *both are
  usage statements*.
- **Fix (relabel, not redesign):** rename origin → **current computational role** everywhere; Endpoint A
  "arose as" → "currently functions as." Genuine origin needs **training-time** evidence (checkpoint sweep:
  does the carrier appear before/after coding-MLM competence?; retrain with the period-3 bins penalized;
  cross-model) — none available on the sealed frozen checkpoint. **The measurements survive the relabel; the
  ambitious "origin" endpoint does not.**

### FATAL-2 — E4a cannot identify biological reading frame (identifiability + statistics lenses; CODE-VERIFIED)
- **Verified in `src/gfmcd/splicing/phase.py`:** `codon_slot` (L65–68) returns `s = (g − cds_start_0 −
  frame) % 3` with `g = window_start + seq_pos`; absolute phase `φ = (seq_pos + OFF) % 3`. Therefore
  **`s − φ = (window_start − cds_start_0 − frame − OFF) % 3` — a PER-EXAMPLE CONSTANT** (the `seq_pos`
  cancels). Within one example you cannot vary codon slot `s` at fixed absolute phase `φ`.
- **Consequence:** the 3×3 (s×φ) design is populated ONLY by pooling across examples of different frame-register
  `c`. At fixed φ the deleted absolute carrier is ~constant, so **any slot-dependence of ΔL at fixed φ cannot
  be a carrier effect** — it is slot-correlated **content** (codon/wobble composition, GC3). So "E4a positive"
  identifies **codon-position composition**, NOT the carrier encoding a reading frame. `M-balance` (position
  counts per cell) gates the **wrong unit**; the real n is genes-per-frame-class (~13–27). This is a
  definitional confound of exactly the R23/R15 type (the answer is forced by the coordinate system).
- **Fix:** demote E4a to corroborative; make **E4b the primary frame test** (within-content; the ΔL=L_D−L_I
  differencing cancels non-carrier absolute-PE changes → genuinely clean) — but note E4b is (a) a
  second-difference (power-hungry, FATAL-3) and (b) plausibly ≈0 by training-time phase randomization (C19),
  so it is clean but likely uninformative. Reading frame may simply be **not cleanly testable** here.

### FATAL-3 — the 0.02-nat powered-equivalence bar is unreachable at MPS budget (statistics lens; VERIFIED vs C8)
- **Verified in `results/phase-lattice/carrier_surgery.json`:** coding ΔL gene-clustered CI
  `[0.15551, 0.19506]` → **half-width 0.0198 nats at n=384 exons**; intron HW 0.0100. So even a **raw mean**
  needs ~380 exons to reach CI-HW < SESOI(0.02); every P3.4 estimand is a higher-variance **contrast** at the
  ≤140-unit MPS budget.
- **Projected exons for CI-HW < 0.02** (HW ∝ 1/√n, SD-inflation f for contrasts): raw mean ~380; E2/E3 (matched
  paired diff) ~540–1200; **E1, E4b (second differences) ~850–2350**; E4a ~2000+. At n≈140 even the raw mean
  has HW≈0.033 > SESOI. **Nothing clears the bar.**
- **Consequence:** three of six truth-table rows are **equivalence-gated** — "general modulo-3 substrate" needs
  E1≈0 AND E2≈0; "positional coordinate, not frame" needs E4a≈0. These are the **maintained-hypothesis**
  readings (C8/C19/C32/C34/C35), and they are **unreachable** at budget → the modal outcome of a correctly-run
  study is **"inconclusive" on its primary question**. Cannot be bought back by masking more positions/gene
  (between-gene σ_g≈0.2 is the irreducible floor; only more **genes** help, at forward-pass cost on MPS).
- **Fix:** re-scope to a **one-sided search for surprising POSITIVES** (E1+/E2+/E3+ are the name-changing
  outcomes and CAN fire if large), with a **discovery-pilot MDE gate** (pre-declare a contrast inconclusive
  *before* spending compute if projected HW > SESOI); or commit to a materially larger run (≥500 exons E2/E3,
  ≥850 DiDs). Equivalence-based confirmation of the maintained hypothesis is off the table at this budget.

---

## Additional majors (fixable)
- **E1 completion vs routing (estimand lens) + burden control not information-matched (identifiability lens).**
  A pure positional router that points attention at same-frame neighbors ALSO gives E1 = ΔL(C)−ΔL(B) > 0 (B
  hides the mates it routes to). So E1-positive maps onto BOTH "codon computation" AND the project's own
  routing account — not separated. And adjacent (B) vs distal ±3/±6 (C) bases differ in local information
  independent of codons → E1 is a biased bracket. `PP-completion` (synthetic XOR oracle) is off-manifold /
  unspecified → a null E1 is uninterpretable (insensitive vs no-completion). Fix: information-matched burden
  control (distal same-lag OFF-slot), pre-register both brackets, make E1 positive-only unless PP fires at
  confirm-n.
- **E2 = "coding-state computation" is not separable from "a positional coordinate more useful where coding
  has more mod-3 structure"** (estimand lens). Coding DNA has sharper slot-conditioned base distributions by
  definition; a pure absolute-position coordinate applying position-conditioned priors saves more loss in
  coding with NO coding computation. So E2-positive is confounded; E2-equivalence is near-unreachable (SESOI
  0.02 < intronic floor 0.042; raw gap 0.117). E2 is structurally biased to fire and mislabel.
- **E2/E3 estimator + clustering (statistics lens).** Replace bin-matching on `L_I` with **within-gene,
  covariate-adjusted DML** (the A9b estimator this project already validated; a gene's 1024nt window holds
  both its CDS exon and intron/UTR flanks → within-gene pairs make one-way `gboot` exact and remove the
  collider). Cross-gene controls need two-way (dyadic) clustering (R22).
- **Family control under-specified (statistics lens):** state α (FWER 0.05), bootstrap max-T over shared
  gene-resamples (uses contrast correlation), apply the family adjustment to TOST too (widens CIs → worsens
  FATAL-3). Discovery split must be EXTRA genes, not a carve-out of the scarce confirm pool.
- **Firewall asymmetry on "the name" (estimand lens):** the measurement firewall is real (MLM-ΔL vs coupling,
  disjoint data, sealed test untouched), but §6 lets P3.4 "force reintroducing codon-frame into the carrier's
  NAME," and that name is interpretive glue in C32/C34/C35/CLAUDE.md. State the firewall **symmetrically**: a
  frame-associated MLM result COEXISTS with C35's routing measurement (frame-for-MLM and routing-for-coupling
  are not contradictory) and must NOT retro-edit the causal claims' wording.

## Minors
- M-del risks a forced FFT round-trip identity (R21) — anchor to a *consequence* (comb amplitude collapses to
  ~3.4%, per `run_carrier_surgery`). M-mask is largely forced. Single SESOI across incommensurable scales
  (raw / first-diff / second-diff). E4b sideband slip + edge asymmetry; report +1 vs +2 separately before
  pooling. `window_codon_slot` drops multi-frame-ambiguous positions → mild selection; report drop rate.
  §5 wording bug ("distal ±1/±2" — ±1/±2 are adjacent).

## Feasibility & calibration (lens 4) — no data blocker, but the calibration is holed
- **BUILDABLE: yes.** The GTF exposes CDS + frame + UTR + start/stop codons (71,926 CDS, 6,956 start_codon,
  20,556 UTR); per-base codon-slot machinery already exists and is unit-tested (`phase.py` `codon_slot` L65–68,
  `window_codon_slot`, `load_cds_intervals`; `tests/test_phase.py`). The reviewer VERIFIED `codon_slot` on real
  **nonzero-frame** sequence (chr1 two-CDS ORF, f0+f2): ATG start, 0 internal stops, **609/609 slot match**.
  E4b natural-flank translation is a few lines (`dataset.py` fetch + window coords). **So the P3.4 blocker is
  SCIENTIFIC (identifiability + power), not engineering — no amount of plumbing fixes it.**
- **The primary discriminator's calibration is dead (concurs across 3 lenses).** `PP-completion` (a synthetic
  XOR/annotation completion oracle) does not travel a real channel — a **frozen** model never learned
  `base=f(mates)`, and real coding has no completion oracle (wobble pos-3 is not determined by pos 1–2). So an
  E1 equivalence is uninterpretable (insensitive vs no-completion). E4b has **no** calibration; E2/E3 have no
  planted-positive.
- **New confounds this lens added:** (M4) **E1 mask-token PROXIMITY** — B masks mates at ±1/±2 (adjacent), C
  masks distal ±3/±6, so B's `[MASK]`s sit closer to the readout; a carrier×proximity interaction survives the
  ΔL differencing and contaminates E1 (needs a proximity-matched control). (M6) the reused `load_examples`
  frame is **coupling-eligible** exons (short-intron, GAP 60–150) — a biased subpopulation that narrows any
  *origin/generality* claim; the MLM readout needs no coupling geometry, so sample arbitrary coding windows
  instead. (M3) `M-annot`'s "slots cycle 0/1/2" clause is vacuous (any constant offset cycles); anchor to the
  real start_codon features + end-to-end ORF integrity (no internal stop) instead. (M5) the `--smoke` gate does
  NOT dry-run the planted-positives, so a dead PP would only surface at confirmatory time.
- **Minor:** `acc_win∈[100,300]` randomization (not "genes with different frames") is the actual decorrelator
  that populates E4a's cells — but it only helps *populate*, not *de-confound* (FATAL-2 stands); UTR-vs-intron
  labeling needs a `load_exon_intervals` sibling (trivial); N-intron-floor is descriptive, not a hard gate.

---

## Independent verification (my checks, not the reviewers')
- **FATAL-2 algebra:** confirmed `codon_slot` = `(g − cds_start_0 − frame) % 3` (phase.py L68); `s − φ` cancels
  `seq_pos` ⇒ per-example constant. E4a slot-at-fixed-φ is between-example only. ✓
- **FATAL-3 power anchor:** confirmed `exon_damage_ci=[0.15551,0.19506]` (HW 0.0198) at n=384 in
  `carrier_surgery.json`. ✓
- **FATAL-1 / F3 lesson:** confirmed circuit.py module docstring (L17–21) already states conditioning on the
  outcome (min_effect) is "legitimate for a *ratio* and illegitimate" for a mean — exactly the E2/E3-on-`L_I`
  collider. ✓

---

## RECOMMENDATION (pending feasibility lens; to Cam)
The design red-team did its job: **do not freeze/run P3.4 as designed.** As posed it cannot answer the origin
question (frozen-model ΔL = usage), its cleanest-looking frame test (E4a) is a definitional composition
confound, and the maintained-hypothesis (equivalence) readings are unreachable at MPS budget. Options, in
descending order of my current preference:

- **(C) Record the null-of-method + proceed to transfer.** The carrier is already well-bounded at the causal
  level (C8 dependence, C19 phase-independence, C32 relative address, C34 distributed locus, C35 attention
  routing). "Why it was learned" is a **training-dynamics** question outside a frozen-checkpoint interp study;
  log that as an explicit, honestly-scoped deferral (a valuable methodological result), and move to the
  depth-first spine's endpoint — **Caduceus transfer**. Cheapest, most honest.
- **(A) Reframe to a narrow "current functional role" study, one-sided positives only.** Relabel origin→usage;
  drop E4a-as-frame and all equivalence claims; run E1+/E2+/E3+ as *surprising-positive* searches with
  discovery-MDE gates and within-gene DML. Runnable, but LIMITED: it can only find a surprising coding
  computation if one exists; given C8/C19 the modal result is "no surprising positive," which weakly
  corroborates the positional-coordinate reading without establishing origin. A modest datum, not a resolution.
- **(B) Pivot P3.4 to a training-time design** (checkpoint sweep / carrier-penalized retrain / cross-model) —
  the only thing that can actually answer origin, but a substantially larger project, likely infeasible on MPS.

I recommend **(C)**, optionally with a single cheap check from (A) if you want one datum on the carrier's
current coding-usage. I will NOT freeze or run anything without your call.
