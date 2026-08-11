# Figure 8 — activation-transplant matrix, test (SUPERSEDED — encoder Layer 3)

> **Tier: T3 — historical.** Superseded Layer-3 replicate. Never cite as current.

**Status: superseded. Controls no claim. Do not quote.** See `DBPR7`.

**The most misreadable figure in this set.** It shows a `PRIMARY_SIX_HEADS` transplant at encoder
**Layer 3** recovering fraction `0.258` on the test cohort. The study's *confirmed* test donor matrix —
the one behind `DBP8`, at Layer 2 — reports `0.739` for `L0 MLP` and `0.879` for `L0 MLP + L0H8`, in
`results/dnabert_promoter/frozen_claim_tests/components/donor_matrix/test_component_donor_matrix_confirmation_128_summary.json`.
A reader who pairs this figure with the registry's "Test recovery fractions" table is looking at two
unrelated experiments.

What the canonical result actually supports is narrower than any of these numbers suggest: transplanted
activations carry **class-level, not example-specific**, promoter evidence, because matched sources do not
beat unrelated same-class sources (`0.036`, p `0.252`). See `DBP8` and `DBPR5`.

Controlling artifact:
`results/dnabert_promoter/causal/layer3_primary_head_donor_matrix/test_layer3_primary_donor_matrix_128_summary.json`.
