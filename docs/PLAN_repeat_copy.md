# Execution Plan — Repeat-Copy Track

**For:** the repeat-copy agent · **Owner sign-off:** Cam Berger
**Read first:** `.archive/branches-codex-repeat-induction/codex-repeat-induction/FINDING.md`, `MECHANISM_CHOICE.md`, `IMPLEMENTATION.md`; and `RESULTS.md` §8/§9 for the ceiling discipline this track must match.
**Environment:** `uv sync --extra lab --extra dev`, then run commands via `uv run <cmd>`. Confirm `uv run pytest -q` passing (incl. `tests/test_repeat_copy.py`) before touching a checkpoint.

## Objective

FINDING.md establishes a **prefix-keyed genomic repeat-copy (induction-family) circuit** learned under a copy-position-weighted autoregressive objective (default 8× weight on copy positions, `--copy-weight 8.0` in `scripts/train_repeat_copy.py`); whether the same mechanism emerges under an unweighted objective remains untested — the shortcut-proof positive that mirrors the reading-frame negative. That is the strongest single claim in the program (a circuit appearing under this weighted objective because no shortcut exists), so it must be defended to the same standard the reading-frame circuit was. Two gaps stand between "found" and "paper-ready":

1. **The ceiling audit is incomplete.** FINDING.md reports only the **base-rate** ceiling (0.2516) — the weakest of the four baselines the branch registered. Shortcut-proofness is currently argued *by construction*. The reading-frame track learned **twice** that construction-based ceilings are lower bounds (cells B, C were both beaten). The registered stop-condition must actually be run.
2. **The trace hasn't had the project's paranoia applied.** Five heads each cost +0.31–0.72 loss when ablated *alone* — anomalous for a distributed circuit and unexplained. No IG-QK, no inert-head check.

**House rules (non-negotiable, from D24/D27):** attribution ranks, only loss-based intervention decides; every ablation reports Δloss with a control; watch for saturated metrics.

## Execution Status — 2026-07-09

- **Phase A: complete.** Seed 0 and seed 99 both grok the repeat-copy task; outputs:
  `.archive/branches-codex-repeat-induction/codex-repeat-induction/fig_repeat_grok.png` and
  `.archive/branches-codex-repeat-induction/codex-repeat-induction/fig_data/repeat_grok.csv`.
- **Phase B: complete.** Ceiling gate did **not** fire. RF-limited CNN denied source access stayed at
  0.2473 vs GFMNano 0.9608; outputs:
  `.archive/branches-codex-repeat-induction/codex-repeat-induction/COPY_CEILING.md`,
  `fig_copy_ceiling.png`, and raw CSVs in `fig_data/`.
- **Phase C: partial-to-strong.** QK-IG source-edge check, prefix-scramble control, and
  ablate-one-vs-subset are complete; outputs:
  `.archive/branches-codex-repeat-induction/codex-repeat-induction/COPY_TRACE.md`,
  `COPY_QK_IG.md`, `fig_copy_heads.png`, and raw CSVs in `fig_data/`.
- **Phase D: complete for exact natural repeats.** The synthetic-trained circuit partially transfers
  to exact hg38 repeats on chr17-19. Seed 0/99 distal-following under cue conflict: 0.7820/0.7830;
  local continuation: 0.1310/0.1330. Source-prefix scrambling collapses distal-following to
  0.2330/0.2280. Outputs: `.archive/branches-codex-repeat-induction/codex-repeat-induction/NATURAL_REPEATS.md`,
  `fig_natural_repeats.png`, and raw CSVs.

---

## Phase A — Figure from existing history (no model loading; do first)

Input: `checkpoints/repeat_copy_s0_history.json` and `repeat_copy_s99_history.json` (schema `{step, train, loss, copy_acc, all_acc}`). Load `figure-style`, `apply_figure_style()`.

1. **`fig_repeat_grok.png` — the copy-circuit grokking curve.** `copy_acc` vs `step`, both seeds (s0, s99) to show reproducibility. Draw the base-rate ceiling (0.2516) as a floor line. Annotate the climb (step 3000 → 8000, 0.34 → 0.96). Save PNG + tidy CSV.

## Phase B — Close the ceiling audit (the registered gate)

MECHANISM_CHOICE.md registered four baselines and a **stop-condition**: *demote the branch if a bounded-window CNN reaches the demanded copy accuracy without access to the first occurrence.* Run all four on the held-out `repeat_copy` val set and report as a single table.

2. **Unigram / base-rate** — already have it (0.2516), include for completeness.
3. **Local naive-Bayes** over a bounded window (no access to the first occurrence).
4. **Receptive-field-limited CNN** — the decisive one. Train a CNN whose receptive field is *shorter than the prefix-to-source distance*, denied the first occurrence, and measure its copy accuracy. **If it approaches the model's 0.9608, the branch is demoted — say so plainly.** If it stays far below, shortcut-proofness is quantified, not assumed.
5. **Transformer-with-copy-access** (upper reference) — a model given the source explicitly, to bracket the achievable ceiling.
6. **Deliverable:** `fig_copy_ceiling.png` (bar chart, four baselines + model + base-rate floor) and a `COPY_CEILING.md` note stating whether the stop-condition fired. This is the item that makes "shortcut-proof" a measurement.

## Phase C — Harden the trace to project standard

Target: `checkpoints/repeat_copy_s0_step8000.pt`. Use/extend `scripts/trace_repeat_copy.py`, `scripts/verdict_repeat_copy.py`, and `scripts/qk_ig_nano.py`.

7. **IG-QK on the copy heads.** Confirm L1H0/1/5/6/7 actually **key on the matching prefix** (not a positional or base-rate artefact). Report the fraction of key attribution landing on the prefix-match token vs a control position — the direct analogue of the reading-frame L1H3 marker-token check.
8. **Inert-head paranoia (D27).** For each nominated head, loss-based Δ with a control. All five already show large loss deltas, so no L1H2-style trap is expected — but confirm it explicitly and confirm attention-only ranking would not have misled here.
9. **Resolve the redundancy puzzle.** Five heads each costing +0.31–0.72 loss *alone* is odd: pure redundancy predicts near-zero single-head cost. Run **ablate-one vs ablate-subset vs ablate-all**:
   - if subset cost ≈ sum of singles → genuinely distributed, independent contributions;
   - if subset cost ≪ sum → non-independent / partially redundant (the singles overcount).
   Report the interaction. This is a real mechanistic finding either way and it's currently unexplained.
10. **Deliverable:** update `FINDING.md` (or a `COPY_TRACE.md` companion) with the IG-QK prefix-keying result, the redundancy verdict, and a head-level `fig_copy_heads.png` (per-head attention-to-source + ablation Δloss).

## Phase D — External validity (do if time; strengthens the biology claim)

11. **Natural genomic repeats.** The synthetic task proves the *capacity*. Test whether the same heads fire on **real** repeated elements in hg38 (tandem repeats / segmental duplications) — does the induction circuit generalise from synthetic random units to biological repeats, or is it corpus-specific? A cue-conflict on a natural repeat is the cleanest bridge to biology this track has.

---

## Deliverables

| file | phase | blocking? |
|---|---|---|
| `fig_repeat_grok.png` + csv | A | no — do now |
| `fig_copy_ceiling.png` + `COPY_CEILING.md` | B | the registered gate — highest priority |
| `fig_copy_heads.png` | C | with the trace |
| updated `FINDING.md` / `COPY_TRACE.md` | C | IG-QK + redundancy verdict |
| natural-repeat generalisation note | D | optional, strong if it lands |

## What I need from the owner
- Confirm the `uv` environment runs here (or a compute target for training the CNN baseline and loading `.pt`).
- The RF-limited CNN (Phase B step 4) needs a short train — confirm compute is available; it is the one item that could *demote* the branch, so it should run before the claim goes in the proposal.
- Whether Phase D (real genomic repeats) is in scope for this pass or deferred.
