> # ✅ ARCHIVED — EXECUTED
> **[ARCHIVED]** · This plan has been carried out. Phases A–C are complete; Phase D (three seeds) is
> still open and is tracked in `docs/PLAN.md`. Numbers quoted here are from the time of writing and
> several have since been retracted (D34/D35 — the "relay" was largely a probe artifact).
> **Read instead:** `docs/RELAY_TRACE.md` for the result, `docs/RESULTS.md` for the numbers.
> **Note:** the environment line below (`conda activate gfm-thrust1`) refers to a retired environment.

---

# Execution Plan — Reading-Frame Track

**For:** the reading-frame agent · **Owner sign-off:** Cam Berger
**Read first:** `RESULTS.md` (§4, §5, §7, §8, §9 — the completed 2×2), `REC.md`, `docs/DECISIONS.md` (D24–D32), `docs/PREREGISTRATION.md` (the frozen verdict).
**Environment:** `conda activate gfm-thrust1` then `pip install -e ".[dev]"`. Confirm `pytest -q` → 102 passing before touching a checkpoint.

## Objective

The 2×2 is **settled** and produced a result bigger than the binary it was designed to test: a **bag → relay → counter spectrum**. The single most valuable unclaimed result in the whole program is now to **trace the relay** — bag and counter were hand-buildable in advance, the relay was not, and right now it is a label between two mechanisms rather than a mechanism. This plan (a) turns the existing raw data into the figures the argument needs, and (b) characterises the relay mechanistically.

**House rules (non-negotiable, from D24/D27):**
- Attribution *ranks*; only **loss-based** intervention decides. Never conclude from an accuracy metric that saturates.
- Every ablation carries a paired **decoy-Met control** and reports Δloss, not ΔaaAcc alone.
- A saturated measurement reports nothing where the effect is largest — watch for it.

---

## Phase A — Figures from existing histories (no model loading; do first, same day)

All inputs are the `*_history.json` files already in `checkpoints/` (schema: list of `{step, train, val, aa_acc}`). Load `figure-style` and call `apply_figure_style()` before plotting; save each as 300-dpi PNG **and** the underlying tidy CSV.

1. **[x] `fig_lambda_grok.png` — the λ-sweep grokking curves.** `aa_acc` vs `step`, seven arms (`nano_uniform`, `nano_lam0.25/0.5/0.75/1/1.5/2`) on one panel. Mark the guessing ceiling (0.3768) as a floor line and annotate the grok step per arm (3250 → 3000 → 1500 → ~750 for the anchored arms). This is the headline "looks like grokking + shortcut scaffolds discovery" slide.
2. **[x] `fig_cellB_cutoff.png` — the D31 cautionary panel.** `cellB_s0_history.json` (stopped at 8k, printed "no circuit") vs `cellB_s1_history.json` (24k, groks ~step 18–20k to 0.9735), same cell. Caption: why we do not trust an accuracy verdict read mid-grok.
3. **[x] Deliverable check:** both PNG + SVG + `results/fig_data/*.csv`. No checkpoint loaded.

**Phase A execution notes (2026-07-09).**
- `figure-style` did not exist; created `src/gfmcd/figstyle.py::apply_figure_style()`. Palette validated with
  the dataviz validator (`#2a78d6,#e34948` and `#2a78d6,#eb6834`): all six checks PASS, worst adjacent CVD
  ΔE 74.6 and 96.7 against a ≥12 target. **Colour encodes the mechanism, never the arm index** — that is the
  point of `fig_lambda_grok`.
- Added the **sense-codon control** as an eighth (dashed) arm. It is anchored, and it is the run that
  falsified D29's stop-depletion story; leaving it out would hide a retraction.
- Measured grok steps: anchored 3250 → 3000 → 1500 → **750** as λ rises; anchor-free arms all ~1000.
- `pytest` is **109** passing, not 102 (tests added for cells B/C/D and the real-CDS amino-acid labels).
- `cell A` has no `cellA_s*` checkpoint — it is `nano_uniform_step8400.pt` (verified against the same
  verdict script; `verdict_2x2.py` now maps `pool: uniform → cell A`).
- **Unowned artifacts:** `repeat_copy_s0` / `repeat_copy_s99` checkpoints and histories exist in
  `checkpoints/` and were not produced by this track. Left untouched, excluded from all figures.

## Phase B — Continuous signatures from the verdict scripts (needs `gfm-thrust1` + checkpoints)

The per-window arrays behind the RESULTS.md tables live only in script stdout. Modify each script (or add a `--dump <path.npz>` flag) to persist its per-window arrays, then plot. Do **not** re-derive numbers by hand — dump and read.

4. **`fig_2x2_signature.png` — the central new result.** Run `scripts/verdict_2x2.py` on `cellA_s*_step8000.pt`, `cellB_s1_step24000.pt` (the groked seed), `cellC_s0_step8000.pt`, `cellD_s0_step8000.pt`. Scatter: **x = far-field ATG-ablation Δloss** (128–255 nt), **y = anchor share under cue conflict**. One point per cell + the real-DNA μ=0 bag as a fifth reference point. The three clusters (bag / relay / counter) *are* the finding — promote this from a table to the figure the proposal leans on.
5. **`fig_cue_conflict_decay.png`.** `scripts/frameshift_conflict.py`, dump anchor-share and content-share per distance bin (extend beyond the 3 bins in RESULTS.md §4 to a continuous curve). Overlay the trained-with-anchors vs trained-without-anchors control — the ~32 nt crossover is the measured pooling width. Run on `nano_uniform_step12000.pt` (forcing model: 1.0 anchor at range) and the real-DNA checkpoint (content at range) so both regimes appear.
6. **`fig_layer_probe.png`.** `scripts/probe_frame.py`, phase-decodability per layer. Real model flat ~0.79 from L0 vs forcing model climbing 0.41 → 1.00 across depth. "Pooled at layer 0" vs "computed across depth" in one figure.

## Phase C — Trace the relay (the high-value measurement)

This is the reason the track exists. Target: **`cellC_s0_step8000.pt`** and **`cellD_s0_step8000.pt`** (the two clean relay cells; C is detector-easy, D detector-hard — contrast them).

7. **Localise.** Run `scripts/trace_uniform.py` and `scripts/qk_ig_nano.py` against cellC/D. Which heads carry the anchor→far-field signal? Compare to the forcing-task counter's L1H3 marker head — is the relay using the *same* head differently, or different components?
8. **Test the relay hypothesis: periodic local re-sync seeded by the anchor.** Concretely — does ablating the anchor cost a **fixed** far-field loss (flat, +0.35/+0.15 as measured) because the frame is re-established at intervals rather than counted continuously? Design an intervention that distinguishes *global count from anchor* vs *local re-sync chain*: e.g. cue-conflict at graded distances and look for a **sawtooth** (re-sync resets) vs a **monotone decay** (single count). Register the prediction in `PREREGISTRATION.md` **before** running.
9. **Apply the inert-head paranoia (D27).** For every head you nominate, report loss-based Δ with a decoy-Met control; confirm no L1H2-style causally-inert head is being credited by attention or IG.
10. **Deliverable:** an operational definition of "relay" (what intervention separates it from bag and from counter), the carrying components, and a `RELAY_TRACE.md` written in the same status-tagged style as `FINDING.md`.

## Phase D — Robustness (needs compute; do if seed budget allows)

11. **≥3 seeds on the relay cells** so `fig_2x2_signature.png` carries error ellipses, not single points. Retrain `cellC`/`cellD` at seeds 1, 2 via `scripts/train_cell.py`; the relay cluster is far more convincing with spread shown.
12. **LayerNorm faithfulness (shared Gate-0 hole).** Coordinate with whoever owns Gate 0 — every tracer error rate came from LN-free planted models and the relay trace inherits this. If unclaimed, plant a circuit in an LN-equipped tiny transformer and re-run the D12/D20 recovery tests before the relay claim is called settled.

---

## Deliverables

| file | phase | blocking? |
|---|---|---|
| `fig_lambda_grok.png` + csv | A | no — do now |
| `fig_cellB_cutoff.png` + csv | A | no — do now |
| `fig_2x2_signature.png` + csv | B | needs env + checkpoints |
| `fig_cue_conflict_decay.png` + csv | B | needs env |
| `fig_layer_probe.png` + csv | B | needs env |
| `RELAY_TRACE.md` + trace figures | C | the headline deliverable |
| updated `PREREGISTRATION.md` (relay prediction) | C | before running Phase C step 8 |

## What I need from the owner
- Confirm `gfm-thrust1` runs here (or a compute target for the `.pt` loads).
- Seed budget: 1 seed gets the figures; **3 seeds** gets defensible error bars on the relay scatter for the PI.
- Priority call: figures for tomorrow's deck (Phase A+B) vs the relay trace (Phase C) — A+B is hours, C is the science.
