# Repeat-Copy Trace Controls

Checkpoint: `checkpoints/repeat_copy_s0_step8000.pt`. Base copy loss: 0.0930.

## Head Ablations

| Condition | Heads | Copy Loss | Delta |
|---|---:|---:|---:|
| ablate L1H7 | 1 | 0.7553 | +0.6623 |
| ablate L1H0 | 1 | 0.8274 | +0.7343 |
| ablate L1H5 | 1 | 0.8233 | +0.7302 |
| ablate L1H1 | 1 | 0.4308 | +0.3377 |
| ablate L1H6 | 1 | 0.5413 | +0.4482 |
| ablate top3 | 3 | 1.3960 | +1.3030 |
| ablate nominated | 5 | 1.4157 | +1.3226 |

Sum of single-head deltas: 2.9128.
Top3 subset delta: 1.3030. Nominated-head subset delta: 1.3226.
Subset cost is far below the sum of singles if the heads are partially redundant/non-independent.

## Prefix-Scramble Control

| Condition | Distal | Local | Neither |
|---|---:|---:|---:|
| single conflict | 0.9648 | 0.0117 | 0.0234 |
| source prefix scrambled | 0.2617 | 0.2344 | 0.5039 |

Scrambling the source prefix while preserving the local second-copy prefix tests whether the distal source is selected by a prefix match.

Raw data: `branches/codex-repeat-induction/fig_data/copy_head_ablation.csv`, `branches/codex-repeat-induction/fig_data/copy_prefix_control.csv`, `branches/codex-repeat-induction/fig_data/copy_prefix_attention.csv`. Figure: `branches/codex-repeat-induction/fig_copy_heads.png`.
