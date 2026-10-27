# Crossed-register evidence inventory

[Concise result](../../studies/splicebert_register_operator_replication/RESULTS.md) ·
[Authoritative C45/C46 ledger](../../studies/splicebert_register_operator_replication/CLAIMS.md)

| Role | Artifact | Recorded outcome |
|---|---|---|
| Compact report | [canonical/summary_v2.json](canonical/summary_v2.json) | Ψ_reg=0.429719; MDE80=0.064708; `INSTRUMENT_UNRESOLVED` |
| Scalar reanalysis | [canonical/block_psi_v2.json](canonical/block_psi_v2.json) | 342 blocks × six prespecified arms |
| Original final record | [result_v2.json](result_v2.json) | All endpoints, ten gates; resolution alone fails |
| Independent inference audit | [analysis_v2.audit.json](analysis_v2.audit.json) | PASS; 21,346 numeric comparisons |
| Raw measurement audit | [measurement_v2.audit.json](measurement_v2.audit.json) | PASS; 2,128,816 checks |
| Producer analysis | [analysis_v2.json](analysis_v2.json) | Preserved pre-audit status; final verdict is in result_v2 |
| Preserved v1 stop | [result_v1.json](result_v1.json), [analysis_v1.audit.json](analysis_v1.audit.json) | Calibration failure before model scoring |
| A1 calibration | [statistical_calibration_v2.json](statistical_calibration_v2.json), [audit](statistical_calibration_v2.audit.json) | PASS before CC |
| Historical fidelity | [historical_fidelity_v2.audit.json](historical_fidelity_v2.audit.json) | PASS before CC |

Each result has its original sibling manifest; compact exports have new derivation
manifests. The global registry records scientific outputs, audits, attempts,
calibration and smoke separately. `canonical` means artifact of record, not a
positive scientific result. The failed v1 and unresolved v2 remain first-class records.

## Runtime data

Raw `cells_v2.json.gz`, `native_v2.json.gz`, `historical_fidelity_v2.json.gz`,
`*.records.jsonl.gz`, and the planning construction payloads stay local and ignored.
Their original manifests preserve exact checksums, sizes, commands and provenance.
No raw payload was deleted. These streams are not shipped with a clean checkout;
the public compact package permits scalar reanalysis, not a fresh forward replay
or independent replay of every raw-data gate. The existing final/audit artifacts
remain available byte-for-byte, including the per-block endpoint table.

The immutable execution artifacts stay at their frozen paths. Only new compact
exports use `canonical/`; relocating old files would break their frozen consumers.
Registration is retrospective bookkeeping and does not alter the 0.05 threshold.
