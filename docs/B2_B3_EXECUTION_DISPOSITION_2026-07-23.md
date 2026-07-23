# B2/B3 execution disposition (2026-07-23)

**Tier:** T2 (decision / provenance). Records the first model execution of the B2/B3 drivers, the
control-layer defects it surfaced, and the four owner decisions taken in response. Companion to the
execution handback (`return_to_planner.json`, `B2_B3_execution_report.md`, `proposed_control_fixes.patch`,
four ladder JSONs) produced on branch `claude/…/b2-b3-execution-handoff` and handed back **without
writing to the record**. The main checkout was restored pristine; this doc is the first thing committed.

## What ran

The full smoke→pilot→confirm ladder was executed from the main checkout (SpliceBERT.1024nt 6L×16H
hidden-512; torch 2.12.1 / transformers 4.44.2 / MPS). Plumbing and mechanics are sound (value-linearity
2.6e-6; reconstruction fidelity exact; dose response clean; calibration planted-positive recovers). The
smoke gate did its job: **both drivers failed on first contact with the model**, exposing six
control-layer mis-specifications.

| Stage | Result |
|---|---|
| Pre-flight | GO (env, model, both data primitives, cost) |
| B3 smoke (shipped) | FAIL → 2 correctness bugs fixed → PASS |
| B3 pilot (discovery) | Strong: R_xfer 0.931, half-width 0.146, N_req 214 |
| B3 confirm (val, n=260) | **INCONCLUSIVE** — large, dose-dependent, position-specific effect; 3 blocking gates mis-specified |
| B2 smoke | **FAIL (halts)** — wrong-direction steering (−0.538); labels don't generalize (0.182) |

## B3 confirmatory numbers (val, n=260)

- Primary transfer R_xfer(α=1) = **0.912**, dyadic CI [0.757, 1.112]; dose-monotone 0.53 → 0.91 → 1.18
  (β = 0.271); **decoy position collapses** (R ≈ −0.06, CI excludes 0). Permuted FPR includes 0.
- **Scramble did NOT collapse (0.978)** — but the shipped scramble was mis-specified (see below).
- **MF2 margin-mismatched ≥ matched (1.094 vs 0.912)**, CI [−0.164, 1.510] — underpowered and
  point-unfavorable.
- **Health 0.560** against a 0.8 threshold — the threshold sits above the model's own baseline.

Reading: the effect is real and large, but it is **not yet shown to be payload-content-specific** (weak
scramble) nor **free of a margin-gap confound** (MF2). Genuine INCONCLUSIVE — cannot honestly be called
POSITIVE (controls don't cleanly pass) or NULL (effect is large). The executor correctly refused to
iteratively "fix" the exact controls blocking a positive and re-run — that is a garden-of-forking-paths
hazard reserved for a pre-committed redesign.

## The six control defects

The six are: two B3 smoke-blocking correctness bugs (fixed), three B3 confirm-blocking control defects
(scramble / health / MF2), and one B2 substantive premise failure. All are B3 except #6.

**B3 smoke-blocking correctness bugs (outcome-independent) — ADOPTED (decision 1), applied to B3:**
1. **B3 empty-injection memorial** asserted `inject(A⁺·V⁻) == m_co`, impossible under MF1: delivery uses
   the CLEAN address A⁺, but the natural corrupt margin is produced by the CORRUPT address A⁻
   (|A⁺−A⁻| ≈ 0.945; corruption reallocates L3H1 attention, per C35). Replaced by a reconstruction-fidelity
   memorial verifying `inject(A⁻·V⁻) == m_co` (exact, 0.0000). *(R-b3fix1, applied)*
2. **B3 "permuted" FPR** used a second matched cohort (54 % identical sources), which preserves the
   clean−corrupt contrast and can never null a difference estimand. Replaced by a Rademacher sign-flip
   null of the per-pair contrast. *(R-b3fix2, applied)*

**B3 confirm-blocking control defects (the three gates that block a POSITIVE):**
3. **B3 scramble** permuted the 4 **window positions** (adjacent acceptor tokens ≈ identical, so the
   payload was barely perturbed) **and never fired** (its `scramble_rng` argument was never passed) →
   R_scr 0.978, no collapse. **REDESIGNED now (decision 2)** to a hidden-feature-coordinate permutation
   with a fixed per-source permutation applied identically to all four window rows. Verified: per-row norm
   and per-position energy are preserved exactly (injection geometry matched), content is destroyed,
   permutation is deterministic per source gene. A content-specific transfer must collapse under it; a
   "needs *some* acceptor-like state here" effect survives — the same NULL alternative MF2 probes, so the
   two controls agree by construction.
4. **B3 health** (`task.mlm_health`) measures *baseline* non-site masked accuracy with no injection; the
   model baseline ≈ 0.56 sits below the 0.8 threshold, so the gate can never pass and never measures
   injection damage. **STILL OPEN** — redesign to measure under max-dose injection and threshold relative
   to baseline (see below).
5. **B3 MF2 margin-mismatched stratum**: point estimate 1.094 (> matched 0.912), CI [−0.164, 1.510] —
   underpowered *and* unfavorable, so `matched_exceeds_mismatched` cannot hold. **STILL OPEN** — power +
   one-sided-interpretation redesign (see below).

**B2 substantive premise failure (not a control bug) — B2 RETIRED (decision 3):**
6. The naive `(r_pref − r_less)` steering direction lowers B_less (−0.538) and model-native preference
   labels do not generalize (held-out 0.182). This is the prereg's **own pre-declared FATAL-1** (position/
   distance geometry, R20). Steering on a frozen checkpoint is withdrawn as a route to C30. C30 remains OPEN.

**Separately (not part of the six):** the executor's `proposed_control_fixes.patch` also carried
**R-b2fix1** (B2 M-label was computed but never gated) and **R-b2fix2** (B2 had no under-injection
health/OOD gate). These are B2 gate-wiring gaps, made moot by the retirement, and were **NOT applied**.

## Decisions taken

1. **Adopt the correctness fixes.** R-b3fix1/2 applied to `run_b3_cross_example_transduction.py`; the
   B2-side fixes (R-b2fix1/2) are moot under the retirement and were not applied.
2. **Pre-committed scramble redesign** — feature-coordinate permutation, applied to B3 (above). This is the
   only one of B3's three blocking controls changed at this stage; health and MF2 are left for the
   pre-committed confirmatory redesign so the eventual re-run is confirmatory, not exploratory.
3. **Retire the B2 steering approach.** Script banner added; kept unrun as design-of-record. C30 stays OPEN.
4. **New branch** — this work lives on `b3-confirmatory-redesign` (cut from `p3-closure-confirmatory` at
   `6ebc0af`; the closure branch is unchanged). Future B3 confirmatory-redesign work lands here.

## Still open for the B3 confirmatory redesign (pre-commit BEFORE re-running)

Scramble is done. Two blocking controls remain to be specified and prereg-amended as a set, **able to
return NULL**, before a single confirmatory re-run:
- **Health**: measure top-1 masked accuracy at non-site positions **under the max-dose injection**, and
  threshold **relative to the model's own baseline** (~0.56), not an absolute 0.8.
- **MF2 margin-mismatched stratum**: power it adequately and fix the one-sided interpretation; its current
  point estimate favors the mismatched source, which — if it holds at higher n — is evidence **against**
  content-specificity and must be reportable as such, not powered away.

## Hygiene flags (owner)

- The **frozen B2/B3 preregs are not tracked on `p3-closure-confirmatory`** (last touched by `2baf31e`).
  Reconcile before any B2/B3 result is committed here.
- B3 script quirk: pilot and confirm share the default `--out`; confirm needs `--force`. Give the pilot a
  `_pilot` suffix in the redesign.
- Nothing was pushed; the execution left the record pristine. These doc + code edits are the first writes.
