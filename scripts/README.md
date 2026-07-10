# Scripts index

Flat on purpose. Several scripts do `sys.path.insert(0, "scripts")` and import each other
(`from train_cell import pool`, `from phase_c_relay import per_position`), so subdirectories would
break the import graph. **Run everything from the repo root** (file paths are centralized in
`gfmcd.paths`, but the flat cross-script import graph still assumes a repo-root CWD), in the uv
environment: `uv run python scripts/<name>.py`.

Two tracks share this directory. They do not share files.

---

## Reading-frame track

### Data and ceilings — what the model *could* do, measured before it is asked to

| script | what it establishes |
|---|---|
| `validate_genome.py` | Biology as external ground truth: 0 in-frame stops in 117,481 CDS windows vs 4.19% / 5.54% in the shifted frames. |
| `anchor_ceiling.py` | Can the start codon be found from sequence at all? A learned detector reaches **0.9215** frame accuracy. Bounds any counter. |
| `pooled_oracle.py` | The **anchor-free** ceiling: codon-level naive Bayes, no start codon. Reproduces D30's **0.9509** at 240 nt. Also emits the per-position oracle that test T2 partitions on. |

### Training

| script | produces |
|---|---|
| `train_nano.py` | The μ = 0 arm: GFM-nano on real DNA, LM objective (`--causal` AR, else MLM). |
| `train_uniform.py` | The forcing task: uniform codons, amino-acid readout. `--lam` walks codon usage toward real; `--sense-only` is the registered stop-depletion control. |
| `train_cell.py` | The necessity 2×2. `--cell A\|B\|C\|D`. Also exports `pool()`, the shared data loader every measurement script imports. |
| `sweep_lambda.sh`, `sweep_rest.sh`, `run_arms.sh`, `run_mixed.sh`, `run_next.sh`, `run_after.sh` | Batch drivers. Logs land in `logs/`. |

### Measurement — attribution ranks, only intervention decides

| script | test | reads at |
|---|---|---|
| `probe_frame.py` | Is phase decodable, layer by layer? Four controls incl. untrained init and `index % 3`. | probe |
| `anchor_test.py` | Start-codon ablation, with the decoy-Met control. | output |
| `frameshift_conflict.py` | The cue-conflict intervention. | probe |
| `verify_uniform.py` | Four-assay confirmation that the forcing task installed the circuit. | mixed |
| `trace_uniform.py` | Head-level trace of the counter (L0H0 → L1H3, plus the inert L1H2). | intervention |
| `qk_ig_nano.py` | Integrated-gradient QK attribution. The naive pattern-gradient is `3.7e-15`; this recovers the marker edge. | attribution |
| `verdict_lambda.py`, `eval_lambda.py` | The λ dose–response. Seven arms, all `aa_acc = 1.0000`, four anchored and three not. | mixed |
| `verdict_2x2.py` | The frozen 2×2 verdict. **Its anchor share is an L2 probe — see D35.** | probe |
| `phase_b_measure.py` | Peak anchor dependence (max over depth) and the layer probe. | probe |
| `phase_c_relay.py` | **T1** (ablation Δ by distance) and **T2** (split by the pooled oracle). | output |
| `phase_c_t1_null.py` | T1's AR(1) surrogate null. Kills T1's periodicity: p = 0.14 / 0.36. | — |
| `phase_c_t3.py` | **T3** anchor relocation. The control that clears the ablation column of OOD. | output |
| `phase_c_conflict_layers.py` | **The retraction.** Cue conflict at every layer *and* at the model's output. | both |

### Figures

`fig_phase_a.py` → `fig_lambda_grok`, `fig_cellB_cutoff` · `fig_phase_b.py` → `fig_layer_probe`
`fig_phase_c.py` → `fig_conflict_depth`, `fig_2x2_corrected`, `fig_decay_output`

All read `results/reading-frame/fig_data/*.json` and write `results/reading-frame/figures/`.
**Never re-derive a number in a plotting script.** Figures are gitignored; the dumps are tracked.

---

## Repeat-copy track

`train_repeat_copy.py`, `repeat_copy_ceiling.py`, `repeat_copy_natural.py`, `repeat_copy_qk_ig.py`,
`repeat_copy_trace_controls.py`, `repeat_copy_loss_curves.py`, `trace_repeat_copy.py`,
`verdict_repeat_copy.py`, `plot_repeat_copy_grokking.py`

Owned by the repeat-copy track. Plan: `PLAN_repeat_copy.md`. Findings and figures now write to
`results/repeat-copy/` (the `--out-dir` default, centralized via `gfmcd.paths`). The historical
branch dump is archived at `.archive/branches-codex-repeat-induction/codex-repeat-induction/`.

---

## The two house rules

1. **A saturated measurement reports nothing exactly where the effect is largest.** Saturated softmax,
   element-wise variance on a sparse target, accuracy pinned at 1.0, accuracy read mid-grok. Four
   instances, all in `RESULTS.md` §10.
2. **A probe at a fixed layer measures whichever variable that layer happens to carry.** It cost us
   both axes of the 2×2 (D34, D35). Where a behavioural readout exists, use it.
