# Publication-Readiness Remediation Plan

> **For agentic workers:** this plan is executed by a `Workflow` fleet. Steps use checkbox (`- [ ]`) syntax.
> Spec of record: [`docs/REVIEW_2026-07-22.md`](../../REVIEW_2026-07-22.md).

**Goal:** Integrate every actionable critique from the 2026-07-22 peer review into the repository, raising
it to publication-grade organization, nomenclature, methodological formalization, and figure systems —
without altering one digit of the scientific record.

**Architecture:** Six sequential phases. Within a phase, agents fan out over **disjoint file ownership**
(the only safe parallelism when many agents edit markdown). Read-only classification precedes every
mutating phase, so structural moves are driven by a reviewed manifest rather than filename guessing.

**Tech Stack:** Python 3 / `uv`, pytest (307 tests), matplotlib + graphviz, markdown.

---

## Global Constraints

These apply to **every** task. Violation fails the task.

1. **The scientific record is immutable.** Never change a recorded numeric value, claim text, evidence
   level, C-number, R-number, statistical result, or verdict. No re-interpretation, no "tidying" of
   conclusions.
2. **`CLAIMS.md`, `CORRECTIONS.md`, `DECISIONS_A0.md`, `RESULTS_VERIFICATION.md`, and all
   `preregistrations/*.md` are READ-ONLY** unless a task explicitly grants write access.
3. **Never delete or overwrite a result artifact or `.manifest.json`.** Moves must use `git mv`.
4. **Never run experiments, model inference, or any GPU/MPS job.** No new scientific results.
5. **Verification gate:** `uv run --extra splicebert --extra dev pytest -q` must report
   **≥261 passed, 0 failed**. Baseline captured 2026-07-22: `261 passed, 46 skipped` in ~102 s.
6. **Git discipline:** never `git add -A`; stage only files you own. **Never push. Never touch `master`
   or any other worktree or branch.** A parallel session may share this repository.
7. **Additive-by-default.** If a change would alter scientific meaning, do **not** apply it — append it to
   `docs/REVIEW_OPEN_QUESTIONS.md` for the owner.
8. **No unverified citations.** Any external reference must be confirmed to exist and to support the exact
   statement it is attached to. Unverified → omit.
9. **Out of scope (owner-only):** branch reconciliation (F4), worktree pruning (F5), pushing, running G1.

---

## File Ownership Map

Conflicts are prevented by ownership, not by convention. Each path has exactly one owner per phase.

| Owner | Paths |
|---|---|
| B1 manifest | `results/MANIFEST.csv`, `results/MANIFEST.json`, `results/**/RESULTS_INDEX.md`, `scripts/verify_manifest.py`, `tests/test_manifest.py` |
| B2 methodology | `docs/METHODOLOGY.md`, `docs/EVIDENCE_LADDER.md`, `docs/REFERENCES.md` |
| B3 nomenclature | `docs/NOMENCLATURE.md`, `README.md`, `MECHANISM.md` |
| B4 biology | `docs/BIOLOGICAL_BRIDGES.md` |
| B5 prereg | `studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_G1_CROSS_EXAMPLE_TRANSDUCTION.md` |
| C1 reorg | `results/**` (moves), + reference rewrites repo-wide |
| D1 figures | `figures/**`, `scripts/splicebert/make_fig*.py` |
| E1 tiering | header lines only, all `*.md` |

---

## Phase A — Classify (read-only, parallel)

- [ ] **A1 Result classification.** Inspect all 726 tracked files under `results/`. For each, determine
      `claim_id`, `status` ∈ {canonical, development, superseded, infrastructure}, `evidence_level`,
      controlling `script`, sibling `manifest`, consuming `figure`. Cross-reference `CLAIMS.md`,
      `REPRODUCE.md`, `fig_report_data.json`, and memorial tests. **Any file referenced by `CLAIMS.md`, the
      figure pipeline, or a memorial test is `canonical` and must never be moved.** Return structured rows.
- [ ] **A2 Nomenclature audit.** Locate every occurrence of `10.67`, `10.58`, `10.2`, `10.298`, `10.63`,
      "splice strength"/"splice-site strength", and bare "circuit" at first use. Classify each occurrence as
      correct / needs-qualifier / retired-term-in-live-prose / retirement-documentation (leave alone).
- [ ] **A3 Citation verification.** For each §6 reference (faithfulness-curve/MIB, causal abstraction,
      causal scrubbing, activation-patching best practices, ACDC/EAP), verify via web search that the work
      exists, and capture authors/venue/year/identifier plus one verified sentence of what it establishes.
      Mark anything unconfirmed as `OMIT`.
- [ ] **A4 Reference-integrity survey.** Map all 323 `results/` path references across 180 files, so the
      Phase-C rewrite has a complete target list.

## Phase B — Author (parallel, disjoint ownership)

- [ ] **B1** Write `results/MANIFEST.{csv,json}` from A1 + `scripts/verify_manifest.py` + a pytest guard
      that fails if a tracked result file is missing from the manifest or points at a nonexistent path.
- [ ] **B2** Extend `docs/METHODOLOGY.md` and `docs/EVIDENCE_LADDER.md` per review §6; create
      `docs/REFERENCES.md`. Only A3-verified citations. Additive — do not weaken existing rules.
- [ ] **B3** Create `docs/NOMENCLATURE.md`; apply the standing phrase "structured causal core within an
      irreducibly distributed computation" at first use in `README.md`/`MECHANISM.md`; add the
      10.63-vs-10.298 same-sentence qualifier and the two-checkpoint disclosure.
- [ ] **B4** Create `docs/BIOLOGICAL_BRIDGES.md` per review §7 — three bridges at their correct, distinct
      evidence levels.
- [ ] **B5** Draft the G1 preregistration (design only, clearly marked **NOT FROZEN, NOT RUN**), following
      the house prereg format with falsifiable on-path memorials and a frozen decision rule.

## Phase C — Reorg (single agent, sequential, highest blast radius)

- [ ] **C1** Using the B1 manifest, `git mv` non-canonical artifacts into `development/`, rewrite every
      affected reference, regenerate the manifest, and prove `uv run --extra splicebert --extra dev pytest -q`
      still reports 0 failed. **If the suite cannot be made green, revert all moves** and report instead.

## Phase D — Figures

- [ ] **D1** Add a shared style module; regenerate the circuit schematic from edge JSON via graphviz; add
      the evidence-ladder hero figure and a 2D heatmap companion to the 3D carrier surface. Existing
      recompute-checks must still pass.

## Phase E — Tiering (last; touches every doc)

- [ ] **E1** Stamp every markdown doc with **T1 canonical / T2 provenance / T3 historical**.

## Phase F — Adversarial verification

- [ ] **F1** Scientific-record integrity: diff every changed line; assert no numeric/claim/verdict changed.
- [ ] **F2** Link & path integrity: every relative link and result path resolves.
- [ ] **F3** Citation integrity: independently re-verify each citation; flag any unsupported attachment.
- [ ] **F4** Test/CI integrity: full suite green; manifest guard actually fails when it should.

---

## Self-Review

- **Spec coverage:** review §2.1→B3, §2.2→B1/C1, §2.3→B3, §2.4→B3, §4 F1→C1, F2→B1, F3→B3, F6→E1,
  §5→B5, §6→B2, §7→B4, §8→D1. F4/F5 intentionally excluded as owner-only.
- **Type consistency:** the manifest column set is fixed in F2 of the review and reused verbatim by B1,
  C1, and F4.
- **No placeholders:** every task names exact paths, an owner, and a pass/fail gate.
