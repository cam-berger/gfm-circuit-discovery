# Handoff — execute B2/B3 (payload steering + cross-example transduction)

**Tier:** T2 (execution handoff / provenance). Date 2026-07-23. Branch: `p3-closure-confirmatory`.
Companion to `CLAIM_TRIAGE_2026-07-22.md`, `OPEN_TRACKS_EXPANDED_2026-07-22.md`, and the B2/B3
preregistrations under `studies/splicebert_partner_selection/preregistrations/`.

## State at handoff (verified)

- Two runnable drivers exist and byte-compile clean against `src/gfmcd`:
  - `scripts/splicebert/run_b3_cross_example_transduction.py`  (red-team v2, all 5 must-fixes wired)
  - `scripts/splicebert/run_b2_payload_steering.py`            (existence→specificity→semantics firewall)
- **Design-verified only. NEITHER HAS BEEN RUN AGAINST THE MODEL.** The smoke gate is the first real
  test — treat any first-run traceback as expected shakeout, not as a result.
- The main checkout has the checkpoint + genome + GTF, so **smoke/pilot/confirm all run here** — no
  worktree symlink needed.

## The one rule: never skip the gate ladder

For **each** script, run in this order and DO NOT advance until the prior rung passes:

1. `--smoke`   — mechanics memorials + calibration on tiny-n. Must print `SMOKE_PASS`. No verdict.
2. `--pilot`   — measures the real effect numerator + its dyadic half-width on the DISCOVERY split,
                 back-computes `N_req_for_SESOI`. Re-size `--n` from this; never from a ceiling stat.
3. (default)   — confirmatory run on `val`. **chr8/chr9 stay sealed — do not pass them as --split.**

Results are immutable: a confirmatory JSON refuses to overwrite without `--force`. Every run also
writes a `.manifest.json` (git commit, input hashes, argv) — commit both.

## B3 first (do this one before B2)

Highest impact: cross-example transfer is the top rejection risk for the donor-circuit paper
(activation patching only licenses claims on its own prompt distribution until the payload state is
shown to transplant across genes).

```
python scripts/splicebert/run_b3_cross_example_transduction.py --smoke
python scripts/splicebert/run_b3_cross_example_transduction.py --pilot --n 400
python scripts/splicebert/run_b3_cross_example_transduction.py --split val --n <N_from_pilot>
```

A `POSITIVE_transfer` verdict requires ALL of (enforced in code, not just documented):
primary dyadic CI-low > SESOI_R(0.20) · positive monotone dose slope · scramble-source collapses ·
decoy-position collapses · **matched primary CI-low exceeds the margin-mismatched stratum CI-high**
(MF2 confound gate) · permuted-pairing FPR includes 0 · MLM health > 0.8.
Otherwise → `NULL_no_transfer` (bounded scope limit, still publishable) or `INCONCLUSIVE_underpowered`
(raise n per the pilot). All three outcomes are honest closures.

## B2 second

```
python scripts/splicebert/run_b2_payload_steering.py --smoke
python scripts/splicebert/run_b2_payload_steering.py --pilot --n 400
python scripts/splicebert/run_b2_payload_steering.py --split val --n <N_from_pilot>
python scripts/splicebert/run_b2_payload_steering.py --split val --n <N> --semantics   # only if core passed
```

Firewall (do not collapse it): EXISTENCE (causal choice direction) → SPECIFICITY (partner-specific,
survives random-direction + decoy-position controls) → SEMANTICS (C30 target, **pre-declared possibly
non-identifiable on a frozen checkpoint**). The `--semantics` arm self-reports `NOT_RUN_core_failed`
if existence/specificity did not pass, and even on a pass carries the R20 caveat in the JSON: a nonzero
MaxEnt correlation does NOT license a MaxEnt label. Likely honest result: existence+specificity
positive, semantics non-identifiable — which is the correct answer, not a failure.

## After the runs — close the loop

1. Commit each result JSON + its manifest; add one C-claim per confirmed result to
   `studies/splicebert_partner_selection/CLAIMS.md` at the evidence level the artifact verdict licenses
   (mirror the C37 discipline: do not stamp "established" past what the artifact's own `verdict` says).
2. Record the disposition in `DECISIONS.md` (append-only).
3. Run the `gfm-repo-hygiene` audit (`audit_repo`) on the tree — confirms each new result carries a
   manifest and no retired terms leaked in.
4. Hold the push for the repo owner (house convention).

## Guardrails (house conventions, do not violate)

- Scoped `git add` only — never `git add -A` on this shared worktree (5 other live `claude/*` branches).
- Results are immutable; corrections are appended, never deleted.
- A frozen checkpoint cannot answer training-origin questions — the carrier's *general* OV content
  (MLM readout) and its biological origin stay out of scope (deferred to the transfer milestone).
