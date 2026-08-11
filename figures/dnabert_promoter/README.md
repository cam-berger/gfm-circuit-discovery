# `figures/dnabert_promoter/` — frozen DNABERT-2 promoter figures

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Built by `scripts/dnabert_promoter/plot_frozen_figures.py`. Each figure has a `.CAPTION.md` beside it
carrying the narrative and the controlling artifact, per the repository figure convention — the artwork
itself states no claim.

**Only figures 1–3 depict the canonical Layer-2 assay.** Figures 4–8 were built from the superseded
encoder-Layer-3 replicate (`results/dnabert_promoter/causal/layer3_*`) and do **not** show the numbers in
[`CLAIMS.md`](../../studies/dnabert_promoter/CLAIMS.md); see `DBPR7`. The mismatch is not cosmetic:
`figure8_test_donor_matrix` plots a `PRIMARY_SIX_HEADS` recovery fraction of `0.258` at Layer 3, while the
confirmed Layer-2 donor matrix behind `DBP8` reports `0.879` for `L0 MLP + L0H8`. They are different
experiments with similar names.

| Figure | Status | Depicts | Claim |
|---|---|---|---|
| `figure1_frozen_layer_probe_validation` | canonical | Per-layer probe trajectory; the thin margin that makes Layer 2 a working choice. | DBP3, DBP4 |
| `figure2_validation_corruption_trajectory` (+`_drops`) | canonical | Validation corruption damage by layer. | DBP5 |
| `figure3_test_corruption_trajectory` (+`_drops`) | canonical | Test corruption damage by layer. | DBP5 |
| `figure4_layer3_head_heatmap` | **superseded** | Per-head ablation heatmap at **Layer 3**. | — |
| `figure5_validation_grouped_ablation` (+`_fractions`) | **superseded** | Grouped-head ablation, **Layer 3**, validation. | — |
| `figure6_test_grouped_ablation` (+`_fractions`) | **superseded** | Grouped-head ablation, **Layer 3**, test. | — |
| `figure7_validation_donor_matrix` | **superseded** | `PRIMARY_SIX_HEADS` transplant at **Layer 3**, validation. | — |
| `figure8_test_donor_matrix` | **superseded** | `PRIMARY_SIX_HEADS` transplant at **Layer 3**, test. | — |

## Not yet drawn

The study's three load-bearing Layer-2 results have **no published figure**: the candidate-circuit
necessity/delivery accounting (`DBP6`), the standalone faithfulness curve that refutes compactness
(`DBP7`), and the Layer-2 donor matrix that bounds transfer to class level (`DBP8`). Drawing the
faithfulness curve matters most — it is the study's central negative result and currently exists only as
a CSV.
