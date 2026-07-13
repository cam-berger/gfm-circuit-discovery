# Repeat-Copy QK-IG Edge Check

Checkpoint: `checkpoints/repeat_copy_s0_step8000.pt`. Layer: L1. IG steps: 16.

| Head | Source Edge Share | Prev-Key Share | Control Share | Attention To Source |
|---|---:|---:|---:|---:|
| L1H5 | 0.3504 | 0.0249 | 0.0037 | 0.6826 |
| L1H7 | 0.3465 | 0.0231 | 0.0062 | 0.7320 |
| L1H0 | 0.3401 | 0.0223 | 0.0034 | 0.6750 |
| L1H1 | 0.3350 | 0.0235 | 0.0050 | 0.6746 |
| L1H6 | 0.3264 | 0.0260 | 0.0036 | 0.6436 |
| L1H4 | 0.1544 | 0.0914 | 0.0061 | 0.4114 |
| L1H2 | 0.0424 | 0.2029 | 0.0037 | 0.0709 |
| L1H3 | 0.0411 | 0.0510 | 0.0059 | 0.0568 |

Raw data: `branches/codex-repeat-induction/fig_data/copy_qk_ig.csv`.
