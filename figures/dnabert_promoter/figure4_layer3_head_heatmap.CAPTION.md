# Figure 4 — per-head ablation heatmap (SUPERSEDED — encoder Layer 3)

**Status: superseded. Controls no claim. Do not quote.** See `DBPR7`.

Zero-ablation effect per attention head across layers 0–3, read at an encoder **Layer-3** probe
(`all_layer_probes/layer_3_probe.joblib`, sha256 `2337441e…`) — **not** the study's canonical Layer-2 probe
(`selected_layer_2_probe.joblib`, sha256 `bd0d9e14…`). The canonical head scan behind `DBP5` is a different
artifact: `results/dnabert_promoter/causal/head_ablation/validation_head_scan_256_layers_0_1_2_summary.json`.

Retained because results in this repository are immutable, not because it is current. Two independent
reasons not to read a mechanism off it: the endpoint is the superseded one, and the underlying ranking
statistic is `mean_absolute_logit_change`, which is direction-blind (`DBPR3`).

Controlling artifact:
`results/dnabert_promoter/causal/layer3_head_ablation/validation_layer3_head_256_summary.json`.
