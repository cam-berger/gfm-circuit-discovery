# Figure 12 — Layer-2 activation transplant, test confirmation

**Status: canonical.** Controls `DBP8`.

Figure 11's assay on the untouched test cohort, restricted to the two component sets carried into
confirmation (`L0 MLP`, `L0 MLP + L0H8`), n=128.

The null replicates, which is the point. Matched source `0.879` versus unrelated same-class `0.843` for
`L0 MLP + L0H8` — a difference of `0.036`, raw p `0.252`; and `0.739` versus `0.686` for `L0 MLP` —
`0.053`, raw p `0.091`. Neither reaches significance. The class contrast is overwhelming by comparison
(`0.995` and `0.861`, raw p underflowing to `0.0` in the artifact).

**This is the figure that Figure 8 is mistaken for.** Figure 8 is named "test donor matrix" but plots a
`PRIMARY_SIX_HEADS` condition at encoder Layer 3 recovering `0.258` — a different assay, at a superseded
endpoint, against a different probe (`DBPR7`). This one shows the confirmed Layer-2 result behind `DBP8`.

Controlling artifact:
`results/dnabert_promoter/frozen_claim_tests/components/donor_matrix/test_component_donor_matrix_confirmation_128_summary.json`.
