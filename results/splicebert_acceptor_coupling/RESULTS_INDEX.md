# Masked-acceptor coupling — results index

This directory records the reciprocal endpoint: reconstruct the masked acceptor `G` while
intervening on its paired donor. The public record uses descriptive evidence IDs; historical
experiment and claim labels remain aliases because frozen files and hashes cannot be renamed.

## Current evidence map

| Evidence ID | Legacy alias | Standing | Active controlling record |
|---|---|---|---|
| `acceptor-coupling/behavior/validation-gate` | `AC1`, `R1`/`R1b` | bounded verification gate | `r1_reverse_coupling.json`; `r1b_reverse_coupling.json` |
| `acceptor-coupling/operator/donor-base-screen` | `AC2`, `R2` | bounded post-outcome screen | `r2_donor_base_decomposition.json` |
| `acceptor-coupling/selection/cc-stratum-retracted` | `AC3` | retracted | R1/R1b selection memorials |
| `acceptor-coupling/components/exploratory-paired-screen` | `AC4`, `R3` | instrument-unresolved candidate screen | `r3_paired_component_scan.json` |
| `acceptor-coupling/components/l1h12-mean-ablation` | `AC5`, `R3c v2.1` | supported under the declared intervention | `attestations/paired_donor_component_confirmation_v1.json` |
| `acceptor-coupling/components/l4-head-noising` | `AC6`, `R3c v2.1` | L4H11/H9/H7 supported under the declared intervention | same attestation |
| `acceptor-coupling/components/route-hypotheses-inconclusive` | `AC7`, `R3c v2.1` | L3H1 reuse and a distinct route remain inconclusive | same attestation |
| `acceptor-coupling/components/l2h15-health-ineligible` | `AC8`, `R3c v2.1` | health failed; no component finding | same attestation |
| `acceptor-coupling/instrument/frozen-handoff-schema-mismatch` | `AC9`, `R1c`/`R5` | instrument closed before the target; no behavioral result | `attestations/frozen_handoff_schema_mismatch_v1.json` |
| `acceptor-coupling/natural-gc/behavior-modifier` | `AC10`, `M6` | bounded verification evidence for a natural-GC behavior modifier | `attestations/natural_gc_validation_v1.json` |
| `acceptor-coupling/natural-gc/l3h1-attention-modifier` | `AC11`, `M6` | bounded verification evidence for an L3H1 attention modifier | same attestation |
| `acceptor-coupling/natural-gc/qk-score-diagnostic` | `M4d` | development-only selection/QK diagnostic; no claim | `attestations/development/natural_gc_qk_score_diagnostic_v1.json` |

The component confirmation used 500 held-out validation rows from 304 genes on chr11/16/19.
It licenses component-level intervention results only. It does not establish address, payload,
write, delivery, mediation, sufficiency, a complete circuit, or a biological mechanism.

The frozen reverse-behavior handoff terminated at a producer/consumer schema mismatch before
any R5 target, receipt, or null family was scored. It is an instrument-closure record, not a
positive, null, or bounded result about masked-acceptor behavior. The later natural-GC validation
is a separate, small verification record: it bounds a behavior-level and an L3H1-attention
modifier on 67 pairs from 56 treatment genes. The M4d score decomposition remains a
development-only candidate generator because its adaptive selection, multiplicity, dependence,
intervention, provenance, and persistence limits preclude promotion.

## Archived raw packages

The exact raw packages are intentionally absent from the active review surface and remain
unchanged under the annotated tag
`archive/acceptor-inverse-coupling-2026-08-17`, which peels to commit
`16a8bd0b2e716c229488cdf89be76ac97b174eac`. The compact attestation records every archived
path, byte size, SHA-256 digest, execution identity, and selected source-backed fact. Where
execution was dirty or otherwise not exactly recoverable, the attestation says so rather than
substituting a later source snapshot. The R3c independent audit and the
[exact M6 protocol](../../studies/splicebert_acceptor_coupling/preregistrations/PREREGISTRATION_M6_VALIDATION_CONFIRMATION.md)
remain active for direct review.

Verify all bindings without loading a model or genome:

```bash
uv run python scripts/verify_archived_evidence.py \
  results/splicebert_acceptor_coupling/attestations/paired_donor_component_confirmation_v1.json \
  results/splicebert_acceptor_coupling/attestations/frozen_handoff_schema_mismatch_v1.json \
  results/splicebert_acceptor_coupling/attestations/natural_gc_validation_v1.json \
  results/splicebert_acceptor_coupling/attestations/development/natural_gc_qk_score_diagnostic_v1.json
```

To inspect a raw file without restoring it into the working tree:

```bash
git show archive/acceptor-inverse-coupling-2026-08-17:results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.json
```

The claim wording is controlled by
[`studies/splicebert_acceptor_coupling/CLAIMS.md`](../../studies/splicebert_acceptor_coupling/CLAIMS.md),
not by filenames or artifact self-labels.
