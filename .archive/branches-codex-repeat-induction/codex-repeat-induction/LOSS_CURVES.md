# Repeat-Copy Train/Validation Loss Curves

Comparable CE recomputed on fixed train/validation subsets of 6000 sequences per seed.
Training-history `train` is not used for this comparison because it is a weighted minibatch loss.

| Seed | Train All CE | Val All CE | Train Copy CE | Val Copy CE | Val Copy Acc |
|---|---:|---:|---:|---:|---:|
| s0 | 1.3161 | 1.3165 | 0.0909 | 0.0947 | 0.9611 |
| s99 | 1.3175 | 1.3170 | 0.0818 | 0.0799 | 0.9698 |

Raw data: `branches/codex-repeat-induction/fig_data/repeat_loss_curves.csv`.
Figure: `branches/codex-repeat-induction/fig_repeat_loss_vs_copy.png`.
