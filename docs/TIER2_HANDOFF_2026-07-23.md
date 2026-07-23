# Tier-2 handoff — publication-readiness remediation

> **Tier: T2 — provenance/handoff.** Author: origin-evidence session (frame df5fc482), 2026-07-23.
> Branch: `tier1-handoff` (rename/rebranch as the owner prefers before fan-out).
> Controlling spec: [`docs/REVIEW_2026-07-22.md`](REVIEW_2026-07-22.md) (the peer review this integrates).
> Execution plan of record: [`docs/superpowers/plans/2026-07-22-publication-readiness.md`](superpowers/plans/2026-07-22-publication-readiness.md).
> **This handoff supersedes that plan's status claims** — much of Phases A–B is already done; the state
> table in §2 is the authoritative "what's left." Read this, then the review, then §6/§7 of the review
> before touching methodology or biology docs.

## 0. Why this handoff is framed as formalization, not science

The scientific record is complete and frozen. Tier-2 is the work of making the repository *legible and
credible to a peer reviewer* — organization, controlled nomenclature, methodological formalization against
the field's literature, figures, and integrity gates — **without altering one digit of the record**
(Global Constraint 1). The success criterion is not "new findings"; it is "a reviewer or a second lab can
navigate every claim to its controlling artifact, and the methods inherit the credibility of the formal
frameworks they instantiate." That is a formalization gap, and the review says so explicitly: *"It is not
a science gap; it is a formalization gap."*

Two mathematical-fundamentals threads run through the whole task and must be honored everywhere:

1. **Activation patching speaks only to the distribution it is run on.** This is the load-bearing
   limitation the entire methodology must state plainly (it is *the* reason G1/Tier-1 exists). Noising
   (necessity) and denoising (sufficiency) are **non-symmetric** operations — a fact from Heimersheim &
   Nanda (arXiv:2404.15255) and Zhang & Nanda (ICLR 2024) that the Methods must assert, because the repo
   already separates the two and gets credit only if it says so.
2. **Faithfulness is graded, not binary, and is a curve, not a point.** The evidence ladder is a
   per-granularity interchange-intervention accuracy (the causal-abstraction reading, Geiger et al.
   arXiv:2301.04709 / JMLR 2025), and the size-swept faithfulness result is an MIB-style curve (Mueller
   et al. arXiv:2504.13151), not a single-threshold number. Every methodological claim must be written at
   the granularity it was measured — the same discipline the CLAIMS ledger already enforces column by column.

If a proposed change would alter scientific meaning, it does **not** get applied — it is appended to
`docs/REVIEW_OPEN_QUESTIONS.md` for the owner (Global Constraint 7).

## 0.5 Methods spine for transfer — the atlas-first descent (READ THIS FIRST)

> **Why this section exists.** A new model inheriting this project must understand the *shape* of the
> method before the remediation tasks make sense. The organizing principle is **top-down: start at the
> whole-component atlas, descend into intervention only where the atlas licenses it.** This is not a new
> workflow — it is the spine that the existing `docs/METHODOLOGY.md` seven stages, formal grounding, and
> eight core principles already instantiate. State it explicitly; do not invent a parallel structure.

**The descent has five levels. Each is a different epistemic commitment, and evidence never auto-promotes
between them (core principle 6).**

1. **Level 0 — Atlas (wide, shallow, observational).** Census the full component space — all 102
   heads+MLPs × the six sequence-architecture contexts (single-pair / donor-cluster / acceptor-cluster /
   multi-pair / cassette / complex) × both target roles (donor, acceptor). Observational tools only
   (attention patterns, residual/QK decomposition, position-table Fourier, attribution rankings). **No
   causal claim is made here.** Output: the component×context map, partitioned into *general-use
   candidates* (role holds across contexts — L2H15, L3H1, L1H12, L2MLP) vs *context-associated signals*
   (fire in one architecture). This is `METHODOLOGY.md` stage 4, and it is the ENTRY POINT, not a
   mid-pipeline step. (Atlas study: HCA11–HCA18.)

2. **Level 1 — Shortlist (frozen, precision-gated).** From the atlas, freeze a candidate shortlist that
   clears a precision gate on a discovery split, and seal a validation cohort. Signals that cannot reach
   the gate are **closed as under-powered leads — a first-class outcome, not a failure** (HCA23 closed all
   20 context-associated signals this way). The freeze is what converts "hunting until something confirms"
   into "a pre-committed list tested once." This is the structural answer to the reviewer's sharpest
   objection ("you found the circuit you went looking for"): the map is committed **before** intervention.
   Core principle 3 — *attribution ranks; intervention decides* — the atlas is the ranking; the descent is
   the deciding.

3. **Level 2 — Causal validation (interchange interventions, per shortlisted component).** Only now
   descend into the per-circuit seven-stage workflow: necessity → specificity → mediation → direction,
   each an **interchange-intervention accuracy at increasing granularity** (the causal-abstraction reading,
   Geiger et al. JMLR 26, 2025). Necessity, mediation, direction, and specificity are **different claims**
   (core principle 4); each is written at the level it was measured, the same column-by-column discipline
   the CLAIMS ledger enforces. Faithfulness is graded and size-swept, not a single threshold (MIB curve;
   the "no sparse knee" result → *structured causal core within an irreducibly distributed computation*).
   (Atlas study: HCA19–HCA20 validate the four general-use candidates as material causal relations.)

4. **Level 3 — Carrier/representation layer (the GFM-specific descent).** Where a validated component
   *reads* a representational structure rather than computing it, descend further into that structure. The
   two positional carriers (period-3, ≈10.63) live in the position-embedding matrix **P**, and reader
   heads read them out — so the atlas has **two object classes, not one**: `component` entries
   (heads/MLPs) and `carrier` entries (structures in P), with a *reads* relation between them.
   Characterize the carrier (Fourier of P), test the reader relation (post-softmax attention), and prove
   use by band-deletion/transfer (never "surgery" in new prose). This layer is what makes a *genomic*-FM
   atlas structurally richer than a text-LM head atlas.

5. **Level 4 — Biological bridge (stop at the correct level; core principle 8).** Each validated
   component/carrier carries a bridge status, at its honest evidence level: **(a) clear bridge** (L3H1 →
   splice-site pairing / intron definition), **(b) candidate origin** (period-3 → codon 3-periodicity;
   corpus-attributed C39, dynamics unproven), **(c) refuted obvious bridge, unknown true correlate**
   (≈10.63 ≠ nucleosomal — a *powered refutation*, not a null). Presenting all three at their correct,
   distinct level is the discipline that separates this from the attention-visualization literature. **Do
   not force a biological story onto every component** (review §7).

**The GFM-specific axes (what makes this an atlas *for GFMs*, not a generic head taxonomy).** Two axes
fall out of this project's own results and should be named explicitly wherever the method is written up:
- **Tokenizer-class gates what a component can read.** C40 established this as a mathematical constraint,
  not an empirical accident: a codon-scale periodic code is representable under single-nucleotide
  tokenization and **Nyquist-forbidden** under BPE (3 nt → 0.64 token-steps < the 2-step period). So every
  atlas entry is conditional on tokenizer class; a null in a k-mer/BPE model is a tokenizer artifact, not
  evidence about the model. No text-LM head atlas has this axis.
- **Role vocabulary is molecular, not syntactic.** Entries are *splice-site reader*, *positional-carrier
  reader*, *codon-phase reader*, *boundary detector* — not induction/previous-token/name-mover. This is a
  controlled vocabulary (like `NOMENCLATURE.md`) for component function.

**Scope boundary the transfer must preserve (do not overclaim).** The atlas is validated on **n=1 model**
(SpliceBERT). The tokenizer-class axis is populated by a cross-model *refutation* (carriers ABSENT in
RNA-FM/RNA-BERT, C40), not by validated cross-model role *correspondence*. So the licensed framing is: **"a
GFM head-atlas formalism, worked example on SpliceBERT, with the tokenization-representability axis
demonstrated by cross-model absence."** A claim of a *general* GFM atlas needs ≥2 models with validated,
corresponding roles — that is a future experiment, not a documentation task, and must not be asserted in
the methods write-up.

**What the new model should do with this spine.** When formalizing the methods (the B2 fidelity pass,
§5.B), present `METHODOLOGY.md` in this top-down order — atlas → shortlist → intervention → carrier →
bridge — with the seven stages as the Level-2 detail and the eight core principles as the invariants that
hold at every level. The review already asked for the seven-stage workflow + tool-to-question map to be
**the paper's methods hero figure** (review §6.6): render it as the descent, with each result family's
cells filled/open per evidence level. That figure *is* the methodological contribution.

## 1. Hard constraints for every task (from the plan's Global Constraints — non-negotiable)

1. **Immutable record.** Never change a numeric value, claim text, evidence level, C/R-number, statistical
   result, or verdict. No re-interpretation.
2. **READ-ONLY:** `CLAIMS.md`, `CORRECTIONS.md`, `DECISIONS_A0.md`, `RESULTS_VERIFICATION.md`, all
   `preregistrations/*.md` — unless a task explicitly grants write.
3. **Never delete/overwrite a result artifact or `.manifest.json`.** Relocations use `git mv` only.
4. **Never run experiments / model inference / any GPU/MPS job.** Tier-2 produces no new results.
5. **Verification gate:** `uv run --extra splicebert --extra dev pytest -q` must stay green. Current
   baseline on this branch: **305 passed, 35 skipped, 0 failed** (re-confirm before you start; the plan's
   older baseline of 261 predates this session's additions).
6. **Git discipline:** never `git add -A`; stage only files you own (see the ownership map §4). **Never
   push. Never touch `master` or another worktree/branch.**
7. **Additive-by-default.** Meaning-altering change → `REVIEW_OPEN_QUESTIONS.md`, not applied.
8. **No unverified citations.** Any external reference must be confirmed to exist and to support the exact
   statement it is attached to. Unverified → omit.
9. **Owner-only (out of scope for the fleet):** branch reconciliation, worktree pruning, pushing, running G1.

## 2. Authoritative state — what is DONE vs REMAINING (audited 2026-07-23)

The execution plan lists Phases A–F as open checkboxes. **That is stale.** Measured state:

| Phase / task | Plan says | Actual state (audited) | Remaining work |
|---|---|---|---|
| **A1** result classification | `[ ]` | **DONE** — `results/MANIFEST.csv/.json` classifies all 770 tracked files (`status` populated: 527 canonical / 99 development / 132 superseded / 12 infrastructure) | none — this is the input C1 consumes |
| **A2** nomenclature audit | `[ ]` | **DONE (tooling)** — `gfm-repo-hygiene` retired-term scanner runs clean on live prose; the recurring hits are all retirement-documentation (correct) | spot-verify after C1 rewrites |
| **A3** citation verification | `[ ]` | **DONE** — `docs/REFERENCES.md` cites all review anchors (Geiger 2301.04709, Mueller 2504.13151, Zhang&Nanda ICLR24, Heimersheim 2404.15255, Conmy, Syed, Chan) | F3 re-verification only |
| **A4** reference-integrity survey | `[ ]` | **partial** — manifest gives the artifact→row map; the 323 in-prose path references are not yet mapped | build the reference map C1 needs (§5.C) |
| **B1** MANIFEST + guard | `[ ]` | **DONE** — `scripts/verify_manifest.py` + `tests/test_manifest.py` (33 tests) enforce coverage; fixed this session | keep in sync as C1 moves files |
| **B2** methodology/ladder/refs | `[ ]` | **DONE (populated, T1)** — `METHODOLOGY.md` (261 ln), `EVIDENCE_LADDER.md` (98), `REFERENCES.md` (118) exist and cite the frameworks | review §6 fidelity pass (§5.B) — verify each ladder level is written as an interchange-intervention accuracy |
| **B3** nomenclature doc | `[ ]` | **DONE (populated, T1)** — `NOMENCLATURE.md` (64 ln) exists | verify the 10.63-vs-10.298 same-sentence qualifier + two-checkpoint disclosure are present |
| **B4** biological bridges | `[ ]` | **DONE (populated, T1)** — `BIOLOGICAL_BRIDGES.md` (243 ln) exists | review §7 fidelity pass — the three bridges at distinct evidence levels (§5.D) |
| **B5** G1 prereg | `[ ]` | **DONE** — `PREREGISTRATION_G1_*.md` drafted (392 ln, NOT FROZEN) | none (Tier-1 handoff owns G1) |
| **C1** reorg (git mv) | `[ ]` | **NOT DONE** — **229 files** classified development/superseded are still at top level in `phase-lattice/` and `splicebert_coupling/` | **the single biggest remaining task** (§5.A) |
| **D1** figures | `[ ]` | **partial** — `figstyle.py`, `fig_evidence_ladder`, `fig_carrier_heatmap` exist | graphviz-from-edge-JSON schematic; confirm ladder-hero + 2D heatmap complete (§5.E) |
| **E1** tiering | `[ ]` | **partial** — ~90/160 stamped; **70 docs unstamped repo-wide** (48 under `studies/`, 10 `figures/`, 7 `docs/`, 3 top-level, 1 `data/`, 1 `results/`) | stamp all 70, not just the top-level ones (§5.F) |
| **F1–F4** adversarial verify | `[ ]` | **NOT DONE** — the final integrity gates | run after C1/D1/E1 (§5.G) |

**Net:** Phases A and B are effectively complete. The real remaining work is **C1 (reorg, high blast
radius), D1 (one figure), E1 (11 stamps), and F1–F4 (the integrity gates)** — plus the A4 reference map
that C1 depends on. This is a much smaller, better-scoped job than the plan's all-open checkboxes imply,
and its center of gravity is C1.

## 3. Execution architecture (from the plan; still correct)

Six sequential phases; within a phase, agents fan out over **disjoint file ownership** — the only safe
parallelism when many agents edit markdown. Read-only classification precedes every mutating phase, so
structural moves are driven by the reviewed manifest, not filename guessing. Because A/B are done, the
critical path is now: **A4 (reference map) → C1 (reorg) → D1 (figures, parallel) → E1 (tiering) → F1–F4
(verify)**. C1 must be a **single agent, sequential** — it has the highest blast radius (229 `git mv`s +
repo-wide reference rewrites).

## 4. File ownership map (conflict prevention — one owner per path per phase)

| Owner | Paths |
|---|---|
| A4 refmap | read-only survey → `docs/REFERENCE_MAP.json` (new) |
| C1 reorg | `results/**` (moves), repo-wide reference rewrites, `results/MANIFEST.*` regen |
| D1 figures | `figures/**`, `scripts/splicebert/make_fig*.py`, `scripts/splicebert/figstyle.py` |
| E1 tiering | header lines only, all `*.md` |
| F1–F4 | read-only verification + `docs/REVIEW_OPEN_QUESTIONS.md` appends |

## 5. Task briefs

### 5.A — C1 reorg (THE critical task; single agent, sequential)

**Goal.** Relocate the 229 tracked files classified `development`/`superseded` in the manifest into
`development/` and `superseded/` subdirectories under their result family, rewrite every reference to
them, regenerate the manifest, and prove the suite stays green.

**Method (mechanical, driven by the manifest — no filename guessing).**
1. Read `results/MANIFEST.csv`. For each row with `status ∈ {development, superseded}` whose `artifact`
   is not already under a `development/`/`superseded/` subdir, compute the target path
   (`results/<family>/<status>/<basename>`).
2. `git mv` the artifact **and its sibling `.manifest.json` together** (never split a result from its
   manifest). Canonical files (`status=canonical`, or referenced by `CLAIMS.md`/a figure/a memorial test)
   **never move** — Global Constraint and the A1 rule.
3. Rewrite every in-prose reference to a moved path, using the A4 reference map (§5.C) as the complete
   target list. This is the step that breaks silently — a moved artifact with a stale `CLAIMS.md` or
   `RESULTS_INDEX.md` pointer fails F2.
4. Regenerate `MANIFEST.csv/.json` (the `artifact` paths change) and re-run `verify_manifest.py` +
   `test_manifest.py` — coverage must stay exact.
5. **Prove green:** `uv run --extra splicebert --extra dev pytest -q` → 0 failed. **If the suite cannot be
   made green, `git reset --hard` all moves and report** rather than shipping a broken tree (plan C1 rule).

**Traps.** (a) Memorial/tripwire tests reference specific artifact paths — a move that orphans a memorial
fails the suite; the manifest's `status=canonical` guard is what protects against moving those. (b)
`RESULTS_INDEX.md` files are `infrastructure`, stay in place, but their *contents* list artifacts that
move — rewrite the listings. (c) do not move a file whose `status` you had to guess; if the manifest row
is ambiguous, leave it and flag it.

### 5.B — B2 methodology fidelity pass (docs exist; verify against review §6)

The docs are populated. The remaining work is a **fidelity audit** against review §6, not authoring.
Verify — and only append where missing, never weaken (Constraint 7):
- **Ladder = interchange interventions.** Each ladder level (P / QK / Attention / OV / Behavior / Biology)
  should be defined as an interchange-intervention accuracy at increasing granularity, citing Geiger et al.
  (causal abstraction, arXiv:2301.04709 / JMLR 26, 2025) as the formal backbone. This is *the* change the
  review says most directly answers "the formalization is not publishable yet."
- **MIB framing.** The `a12_faithfulness_size` no-knee result is framed as a size-swept curve in MIB terms
  (Mueller et al. arXiv:2504.13151; CPR / CMD size-swept scalars), cited as *evaluation vocabulary*, not a
  method reproduced. The contribution is orthogonal — you show there is *no* sparse knee.
- **Causal scrubbing positioning.** C27 scrub-to-corrupt is causal-scrubbing-adjacent (Chan et al.,
  Redwood, AI Alignment Forum Dec 2022 — **not peer-reviewed, cite as such**). State where you follow it
  (scrub to corrupt distribution) and where you deviate (graded recovery, not a binary
  hypothesis-consistency verdict).
- **Patching best-practices compliance made legible.** Assert the three things the repo already does:
  (a) in-distribution symmetric-token-style corruption (real non-partner/other-donor AG swaps, not
  Gaussian noise); (b) continuous logit-margin metric; (c) necessity/sufficiency separated as
  non-symmetric (Zhang & Nanda ICLR 2024; Heimersheim & Nanda arXiv:2404.15255).
- **ACDC/EAP cross-check.** Note automated edge attribution (Conmy et al. NeurIPS 2023; Syed/Rager/Conmy
  EAP 2023) as a *method-agnostic cross-check* that would blunt "you found the circuit you went looking
  for." Frame as recommended future validation — **do not run it** (Constraint 4).

### 5.C — A4 reference map (prerequisite for C1)

Survey the ~323 in-prose references to `results/` paths across ~180 files; emit `docs/REFERENCE_MAP.json`
= `{artifact_path: [referencing_file:line, …]}`. Read-only. This is what makes C1's rewrite complete
rather than best-effort. Without it, C1 will miss references and fail F2.

### 5.D — B4 biological-bridges fidelity pass (doc exists; verify against review §7)

Verify the three bridges are each stopped at their **correct, distinct** evidence level — the review is
explicit that presenting all three at their honest level is a *strength*, and forcing a biological story
onto every feature is the failure mode to avoid:
- **L3H1 ↔ splice-site pairing / intron definition** — the *defensible* bridge; short-intron (60–150 nt)
  scope is the intron-definition regime. Currently under-stated; strengthen by connecting
  partner-specificity + distance dependence to the intron-definition literature. (Do **not** invent a
  coupling-vs-U1/U2 result — that would be new science; frame as a testable bridge.)
- **period-3 ↔ codon 3-periodicity** — candidate origin, now corpus-attributed (C39/C40, this session).
  State: "candidate origin = codon 3-periodicity; present-day role = relative-positional routing
  coordinate (phase-independent, C19); origin corpus-attributed (C39), dynamics unproven." **Do not claim
  coding enrichment** — the matched CDS-minus-non-CDS-exon contrast was inconclusive (C36).
- **≈10.63 ↔ nucleosome/helical — REFUTED, and that is the result.** Real nucleosomal ≈10.298 nt (z=+249)
  vs model carrier ≈10.63 nt (±0.005), 0.33 nt apart = a *powered refutation*, not a null. Correct
  statement: "the obvious biological bridge is excluded; the true correlate is unknown and we make no
  claim." This session's human-genome corpus null (`corpus_spectrum.json`) and the full human-only
  spectrum (`human_only_full_spectrum.json`, artifact 7bf703d7) corroborate — fold them in if useful.

### 5.E — D1 figures (partial; finish + one new)

`figstyle.py`, `fig_evidence_ladder`, `fig_carrier_heatmap` already exist. Remaining per review §8:
- **Directed circuit schematic from edge JSON via graphviz/dot** (`a10_generator_edges`, `donor_heads`) so
  topology is data-derived and reproducible, not hand-placed.
- Confirm the **evidence-ladder hero figure** (filled/open cells per result family) and the **2D heatmap
  companion** to the 3D carrier surface are complete and pass their recompute-checks.
- **Keep** fig1 "two carriers" and fig4-panelC (wavelength zoom with the refutation) — the two most
  reviewer-persuasive panels; do not restyle them away.
- Use `figstyle.py` as the shared module; all existing recompute-checks must still pass. For any new
  deliverable figure, load the `figure-style` skill and `apply_figure_style()` first.

### 5.F — E1 tiering (partial; **70 docs unstamped repo-wide**)

Stamp every unstamped live doc with **T1 canonical / T2 provenance / T3 historical**. Header-line edits
only. **Scope is repo-wide, NOT top-level** — an earlier draft of this handoff under-counted by scanning
only `maxdepth 2`; the audited full count (excluding `.venv/.git/.claude` and the generated
`.pytest_cache/README.md`) is **70 unstamped**, distributed:

| Directory | Unstamped | Typical tier |
|---|---:|---|
| `studies/**` | 48 | mostly T2 (prereg/decision/study READMEs) / T3 (superseded) — judge per doc |
| `figures/**` | 10 | T2 (CAPTION/provenance docs) |
| `docs/**` | 7 | T1 for live methods/synthesis (e.g. `CARRIER_ORIGIN_SYNTHESIS_2026-07-23.md`), T2 for dated execution/handoff docs (`CLAIM_TRIAGE_2026-07-22`, `HANDOFF_B2_B3_EXECUTION`, `OPEN_TRACKS_EXPANDED_2026-07-22`, `REVIEW_2026-07-22`, both `*_2026-07-23` execution docs) |
| top-level | 3 | `AGENTS.md`, `CLAUDE.md`, `RESULTS_VERIFICATION.md` — judge (RESULTS_VERIFICATION is T2 provenance) |
| `data/**` | 1 | T2 |
| `results/**` | 1 | T2/infrastructure |

Regenerate the list at execution time — this session added docs — with a recursive scan
(`find . -name '*.md'` excluding `.venv/.git/.claude`, skip `.pytest_cache/README.md`) checking the first
~6 header lines for a `Tier:` stamp. **E1 is complete only when that repo-wide scan returns zero
unstamped**, not when the top-level docs are done.

### 5.G — F1–F4 adversarial verification (run last)

- **F1 scientific-record integrity:** diff every changed line across the whole Tier-2 branch; assert **no
  numeric value, claim, evidence level, or verdict changed.** This is the gate that protects Constraint 1.
  Any diff that touches a digit inside `CLAIMS/CORRECTIONS/DECISIONS/prereg` is a failure.
- **F2 link & path integrity:** every relative link and every `results/` path reference resolves
  post-C1. Use the A4 map as the checklist.
- **F3 citation integrity:** independently re-verify each `REFERENCES.md` citation exists and supports the
  exact attached statement; flag unsupported → `REVIEW_OPEN_QUESTIONS.md`.
- **F4 test/CI integrity:** full suite green **and** the manifest guard actually fails when a file is
  removed from the manifest (test the test).

## 6. Literature grounding (verified — the anchors the methodology must cite correctly)

These are already in `REFERENCES.md`; F3 must re-verify. Each is cited for a specific, bounded purpose:

| Work | Identifier | Cited FOR | Cited-as caveat |
|---|---|---|---|
| Geiger et al., Causal Abstraction | arXiv:2301.04709; **JMLR 26(83):1–64, 2025** | formal backbone: ladder levels = graded interchange-intervention faithfulness | scope VERIFIED 2026-07-23 — the paper explicitly formalizes *graded faithfulness* and unifies activation/path patching, causal mediation, causal scrubbing, circuit analysis under causal abstraction; adopt its notation |
| Mueller et al., MIB | arXiv:2504.13151; **ICML 2025 (PMLR v267:45069–45108)** | evaluation vocabulary for the size-swept faithfulness curve (CPR/CMD) | VERIFIED 2026-07-23 — cite as ICML 2025, not a bare preprint. Orthogonal contribution — you show *no* sparse knee; not a method reproduced |
| Zhang & Nanda | ICLR 2024 | localization is sensitive to corruption method + metric | you are compliant (in-dist corruption, margin metric); make it legible |
| Heimersheim & Nanda | arXiv:2404.15255 (2024) | noising (necessity) ≠ denoising (sufficiency), non-symmetric | you separate them; say so |
| Conmy et al., ACDC | NeurIPS 2023 | automated circuit discovery as a method-agnostic cross-check | recommended, NOT run (Constraint 4) |
| Syed, Rager & Conmy, EAP | 2023 | cheaper edge attribution alternative to ACDC | recommended, NOT run |
| Chan et al., Causal Scrubbing | Redwood, AI Alignment Forum, Dec 2022 | positioning for C27 scrub-to-corrupt | **not peer-reviewed — cite as such** |

**Mathematical fundamentals the docs must state (not soften):**
- Activation patching is distribution-bounded → the reason Tier-1/G1 exists; the Methods must say the
  intra-example causal evidence does not by itself establish cross-example transfer.
- Faithfulness is graded and size-dependent (a curve); the "no sparse knee" result means the necessary
  circuit is a *structured causal core within an irreducibly distributed computation* — the standing
  phrase to apply at first use in README/MECHANISM (review §2/§4).
- The MLM unembed `logits = cls(hidden_states[-1])` is bit-exact → why final-residual interventions are
  exact and why the non-target-position control is architecture-forced to 0 (relevant to figures and to
  how the methods describe the intervention locus).

## 7. What NOT to do

- **Do not alter the scientific record** — no digit, claim, evidence level, verdict, C/R-number. Meaning-
  altering proposals go to `REVIEW_OPEN_QUESTIONS.md`.
- **Do not run any experiment, inference, ACDC/EAP pass, or GPU/MPS job.** Tier-2 is formalization only.
- **Do not `git mv` a canonical artifact** (referenced by CLAIMS/figure/memorial) — only development/
  superseded files move, and always with their manifest sibling.
- **Do not push, touch `master`, or touch another worktree/branch.** A parallel session may share the repo.
- **Do not add an unverified citation** — F3 will catch it; omit if unconfirmed.
- **Do not "surgery"** in any new prose — content-safety tripped on it this session; use "band-deletion".
- **Do not treat the plan's open checkboxes as the work list** — §2 here is the audited state; A/B are done.

## 8. Deliverables

1. `docs/REFERENCE_MAP.json` (A4) — the complete path-reference index C1 consumes.
2. Reorganized `results/` tree (C1): 229 dev/superseded files under `development/`/`superseded/`, all
   references rewritten, MANIFEST regenerated, suite green.
3. Methodology/biology fidelity passes (B2/B4): appended framework grounding where missing; a REVIEW_OPEN_
   QUESTIONS entry for anything meaning-altering.
4. Graphviz circuit schematic + confirmed ladder-hero and 2D-heatmap figures (D1).
5. 11 remaining docs tier-stamped (E1).
6. F1–F4 verification report: record-integrity diff clean, links resolve, citations verified, suite green
   + manifest-guard self-test.

## 9. Provenance / baselines

- Branch baseline suite: **305 passed, 35 skipped, 0 failed** (re-confirm at start).
- Manifest: 770 rows (527 canonical / 99 development / 132 superseded / 12 infrastructure).
- Controlling spec: `docs/REVIEW_2026-07-22.md`; execution plan: the publication-readiness plan.
- Standing phrase (review §2/§4): "structured causal core within an irreducibly distributed computation."
- Nomenclature: `≈10.63 carrier` (not 10.67/10.58 DFT-bin labels); `10.298` = nucleosomal (DNA-measured);
  `10.2` = human-only proposal (closed at attention); "splice-site strength" withdrawn (R22).
