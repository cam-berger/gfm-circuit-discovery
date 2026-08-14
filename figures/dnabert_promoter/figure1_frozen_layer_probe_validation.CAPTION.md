# Figure 1 — frozen layer-probe trajectory

**Status: canonical.** Controls `DBP3`, `DBP4`.

Balanced accuracy of a masked-mean linear probe fitted independently at each of DNABERT-2's 12 encoder
layer outputs, frozen encoder throughout. Layer 2 is the argmax on validation and is the endpoint used by
every canonical assay in this study.

**Read the flatness, not the peak.** Validation balanced accuracy is `0.8050` at layer 2, `0.8031` at
layer 3, and `0.8014` at layers 1 and 4 — a `0.0019` spread on a single seed (42). The curve does not
separate the early layers, so the figure supports "promoter information is linearly recoverable early and
does not improve with depth", not "layer 2 is the promoter layer" (`DBPR2`).

Test metrics annotated on the figure (balanced accuracy `0.8102`, ROC-AUC `0.8919`) are descriptive: the
test split took no part in selection. They are measured on the leakage-controlled splits — the raw cohort's
104 cross-split duplicate groups and 64 label-conflicting groups are removed upstream, with disjointness
asserted (`DBP2`). Near-duplicates are not screened.

Controlling artifact: `results/dnabert_promoter/frozen/layer_probes/layer_probe_results.json`.
