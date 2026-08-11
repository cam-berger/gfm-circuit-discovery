# `figures/dnabert_promoter/` — frozen DNABERT-2 promoter figures

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Built by `scripts/dnabert_promoter/plot_frozen_figures.py`. Each figure has a `.CAPTION.md` beside it
carrying the narrative and the controlling artifact, per the repository figure convention — the artwork
itself states no claim.

A caption is a tracked Markdown document, so it needs a `**Tier: T…**` banner within its first eight lines
or `tests/test_repository_hygiene.py` fails. Captions for canonical figures are **T2** (they support a
claim, they are not the claim); captions for the superseded Layer-3 figures are **T3**.

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
| `figure9_faithfulness_versus_circuit_size` | canonical | Standalone faithfulness vs circuit size, both splits. No knee; both jumps are MLP additions. | DBP7 |
| `figure10_candidate_circuit_accounting` | canonical | Necessity and delivery for the five-node circuit. Conditional, non-additive. | DBP6 |
| `figure11_validation_component_transplant_matrix` | canonical | Layer-2 transplant, four source conditions, validation. | DBP8 |
| `figure12_test_component_transplant_matrix` | canonical | Layer-2 transplant, test confirmation. The figure 8 is mistaken for. | DBP8 |

## Regenerating

```bash
uv run --extra dnabert python scripts/dnabert_promoter/plot_frozen_figures.py
```

Writes all twelve figures here. Matplotlib stamps a creation date into PDF metadata, so a rerun produces a
non-empty diff for byte-identical plots; commit only the figures whose underlying artifact actually changed.
