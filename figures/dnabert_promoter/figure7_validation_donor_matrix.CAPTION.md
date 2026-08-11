# Figure 7 — activation-transplant matrix, validation (SUPERSEDED — encoder Layer 3)

> **Tier: T3 — historical.** Superseded Layer-3 replicate. Never cite as current.

**Status: superseded. Controls no claim. Do not quote.** See `DBPR7`.

**This figure does not depict the validation donor matrix reported in `CLAIMS.md`.** It plots a single
`PRIMARY_SIX_HEADS` condition at encoder **Layer 3**, with matched recovery fraction `0.234`. The
canonical validation donor matrix behind `DBP8` is a different experiment at Layer 2 with four component
conditions — `L0H8` `0.005`, `L0 MLP` `0.718`, `L0 MLP + L0H8` `0.848`, full early group `0.839` — in
`results/dnabert_promoter/frozen_claim_tests/components/donor_matrix/validation_component_donor_matrix_128_summary.json`.

Similar name, different assay, different endpoint, different numbers. Read the claim, not the figure.

"Donor" here means the **source example whose activations are transplanted**, not a splice donor site —
the repo-wide sense of the word. See `docs/NOMENCLATURE.md`.

Controlling artifact:
`results/dnabert_promoter/causal/layer3_primary_head_donor_matrix/validation_layer3_primary_donor_matrix_128_summary.json`.
