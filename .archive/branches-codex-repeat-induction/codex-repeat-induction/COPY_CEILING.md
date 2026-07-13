# Repeat-Copy Ceiling Audit

Local crop length: 33 nt. The source occurrence is outside this crop by construction.

| Baseline | Sees Source? | Copy Accuracy | Notes |
|---|---:|---:|---|
| base-rate | no | 0.2477 | best constant base on copy positions |
| local naive Bayes | no | 0.2484 | position-specific NB over 33 nt local crop |
| RF-limited CNN | no | 0.2473 | CNN over 33 nt local crop; source denied |
| source oracle | yes | 1.0000 | upper reference given copied source base |
| GFMNano copy circuit | yes | 0.9608 | trained transformer, seed 0 checkpoint |

Trained `GFMNano` copy accuracy: 0.9608.

**Stop condition:** not fired. The RF-limited CNN remains far below the trained model without source access.

Raw data: `branches/codex-repeat-induction/fig_data/copy_ceiling.csv`. Figure: `branches/codex-repeat-induction/fig_copy_ceiling.png`.
