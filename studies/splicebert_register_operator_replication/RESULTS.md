# CC crossed-register verification

AG→CC/max-margin coupling gave **Ψ_reg=+0.430 [0.385, 0.476] logit units** on
342 motif-clean blocks (2,394 source genes) at a single 120-nt span. The measured
contrast is positive, but **MDE80=0.064708 exceeds the frozen 0.05 sensitivity target**.
The protocol classification remains **`INSTRUMENT_UNRESOLVED`**. All other nine
gates pass; none is unrun. The cutoff has not been changed.

This is outcome-aware operator verification on a subset of the original C43 cohort,
not an independent biological replication. Ψ_reg is the phase-matched minus
mismatched coupling contrast; it is distinct from the carrier-surgery Ψ_C.

[Compact result](../../results/splicebert_register_operator_replication/canonical/summary_v2.json) ·
[Block table](../../results/splicebert_register_operator_replication/canonical/block_psi_v2.json) ·
[Artifact inventory](../../results/splicebert_register_operator_replication/RESULTS_INDEX.md) ·
[Unabridged account and preserved v1 report](RESULTS_DETAIL.md) · [Claim ledger](CLAIMS.md)

## Operator and readout comparisons

All six prespecified arms use the same selected cohort and corrected uncertainty.

| Corruption / margin | Ψ_reg, 95% CI | MDE80 |
|---|---|---:|
| cc / max | +0.430 [0.385, 0.476] | 0.064708 |
| cc / linear | +0.414 [0.372, 0.457] | 0.060498 |
| ac / max | +0.408 [0.366, 0.452] | 0.061086 |
| ac / linear | +0.393 [0.353, 0.433] | 0.056978 |
| at / max | +0.406 [0.364, 0.450] | 0.061859 |
| at / linear | +0.391 [0.351, 0.432] | 0.057726 |

## Frozen gates

| Gate | Outcome |
|---|---|
| Construction and prerequisites | PASS |
| Native-source positive control | PASS |
| Crossed-lattice capability | PASS |
| Bridge-site negative control | PASS |
| Resolution (MDE80 ≤ 0.05) | **FAIL** |
| AC/AT robustness | PASS |
| Protected MLM health | PASS |
| Site health | PASS |
| Numerics and repeated forwards | PASS |
| Independent audit | PASS |

## Controls and limits

Matched, first-mismatch and second-mismatch means are +0.376, −0.068 and −0.039.
All five simultaneous directional lower bounds are positive (minimum +0.289).
The target exceeds all 99 balanced pseudo-phase controls (maximum +0.047619;
exact add-one p=0.01, the resolution floor of this control test). Both mismatch
intervals extend outside ±0.05, so “only when matched” is unsupported.

The paired CC−AC/max contrast is +0.021467 [0.012611, 0.030323] with the
prespecified four-family simultaneous interval. Stop excess remains a confound:
descriptive Spearman ρ=−0.136205; the assay does not isolate phase from stop
composition. No carrier, mediation, or other-distance conclusion follows.

## Preserved history and registration

V1 stopped before model scoring when directional noncoverage was 83/1,000
against a maximum of 71. The separately authorized A1 correction passed fresh
synthetic calibration and historical fidelity before the v2 target run.
[C45/C46](CLAIMS.md) and [R50–R52](../splicebert_partner_selection/CORRECTIONS.md#r50)
retain both outcomes. Original claim rows, result bytes and frozen protocols are unchanged.

This is retrospective **artifact registration**, not a new preregistration.
The compact export preserves the original 0.05 rule and unresolved classification.
No model inference or new threshold analysis was run for this consolidation.

```bash
python scripts/splicebert_register_operator_replication/export_compact_result.py \
  --out-dir results/splicebert_register_operator_replication/canonical --verify
```
