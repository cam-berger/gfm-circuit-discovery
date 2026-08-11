# Figure 11 — Layer-2 activation transplant, validation

> **Tier: T2 — provenance.** Supporting record for the validation transplant matrix (`DBP8`).

**Status: canonical.** Controls `DBP8`.

Mean recovery fraction when clean activations from a **source example** are transplanted into a corrupt
run, across four source conditions and four component sets, at the canonical encoder Layer-2 readout
(n=128 validation).

The comparison that carries the claim is **matched source versus unrelated same class**. If the
transplanted state carried example-specific promoter evidence, the matched bar would stand above its
same-class neighbour. It does not: `0.848` versus `0.819` for `L0 MLP + L0H8`, `0.718` versus `0.683` for
`L0 MLP`. `L0H8` alone recovers essentially nothing (`0.005`). What *is* large is the class contrast —
opposite-class sources drive recovery negative, pushing the readout the other way.

So the transplant moves **a class label's worth of promoter evidence, not a sequence-specific payload**.
"Transferable" and "sequence-specific delivery" are withdrawn on this evidence (`DBPR5`).

"Donor" in the artifact names means this **source example**, not a splice donor site — the sense the word
carries everywhere else in this repository. See `docs/NOMENCLATURE.md`.

Controlling artifact:
`results/dnabert_promoter/frozen_claim_tests/components/donor_matrix/validation_component_donor_matrix_128_summary.json`.
