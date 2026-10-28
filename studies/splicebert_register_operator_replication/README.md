# Crossed-register verification with the paper's primary operator

This study ran an outcome-aware verification of C43's crossed-register contrast with the paper's primary
operator, CC/max, on a subset of C43: 342 motif-clean blocks (2,394 distinct source genes) at 120 nt. AC/AT and
linear margins were prespecified bridges.

The run measured Ψ_reg = +0.430 [0.385, 0.476] logit units. Its MDE80 of 0.064708 missed the frozen ≤0.05
sensitivity target; the other nine gates passed. The recorded classification is `INSTRUMENT_UNRESOLVED`,
with the positive estimate retained (C46). No cutoff was relaxed. The v1 calibration failure is preserved
(C45).

This is completed verification, not a fresh biological replication or a mechanism claim. The
retrospective artifact registration changed storage and navigation only; C43/C44 and their sealed
authorization paths are unchanged. Read the [conformance record](../../docs/PREREG_CONFORMANCE.md) before
citing historical gates.

## Files

- [`RESULTS.md`](RESULTS.md): concise results, all gates, and limits.
- [Registered summary](../../results/splicebert_register_operator_replication/canonical/summary_v2.json),
  [scalar block
  table](../../results/splicebert_register_operator_replication/canonical/block_psi_v2.json), and
  [artifact inventory](../../results/splicebert_register_operator_replication/RESULTS_INDEX.md).
- [`CLAIMS.md`](CLAIMS.md) (C45/C46), [`DECISIONS.md`](DECISIONS.md), and
  [`RESULTS_DETAIL.md`](RESULTS_DETAIL.md) (full execution account, preserved v1 report).
- [`PREREGISTRATION.md`](PREREGISTRATION.md), the [A1 amendment](AMENDMENT_A1.md), and the [v2
  freeze](protocol_freeze_v2.json).
