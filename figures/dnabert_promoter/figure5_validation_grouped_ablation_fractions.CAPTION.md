# Figure 5 — grouped-head ablation, validation (SUPERSEDED — encoder Layer 3)

> **Tier: T3 — historical.** Superseded Layer-3 replicate. Never cite as current.

**Status: superseded. Controls no claim. Do not quote.** See `DBPR7`.

Collective ablation of head groups at an encoder **Layer-3** readout against the Layer-3 probe. The
canonical grouped ablation behind `DBP5` is
`results/dnabert_promoter/causal/grouped_head_ablation/validation_grouped_heads_256_summary.json`, at
Layer 2.

The `_fractions` panel expresses each group's effect as a fraction of the total corruption effect. Those
fractions are **conditional and not additive** — they do not decompose the behaviour into shares, and
summing them is meaningless (`DBP6`, repo convention 5).

Controlling artifact:
`results/dnabert_promoter/causal/layer3_grouped_head_ablation/validation_layer3_grouped_256_summary.json`.
