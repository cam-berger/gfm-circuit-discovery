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

The component confirmation used 500 held-out validation rows from 304 genes on chr11/16/19.
It licenses component-level intervention results only. It does not establish address, payload,
write, delivery, mediation, sufficiency, a complete circuit, or a biological mechanism.

## Archived raw packages

The exact R3c preflight and target packages are too large for ordinary review and remain
unchanged under the annotated tag
`archive/acceptor-inverse-coupling-2026-08-17`, which peels to commit
`16a8bd0b2e716c229488cdf89be76ac97b174eac`. The compact attestation records every archived
path, byte size, SHA-256 digest, clean execution commit, and selected source-backed fact. The
3 KB independent audit and frozen protocol remain active for direct review.

Verify all bindings without loading a model or genome:

```bash
uv run python scripts/verify_archived_evidence.py \
  results/splicebert_acceptor_coupling/attestations/paired_donor_component_confirmation_v1.json
```

To inspect a raw file without restoring it into the working tree:

```bash
git show archive/acceptor-inverse-coupling-2026-08-17:results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.json
```

The claim wording is controlled by
[`studies/splicebert_acceptor_coupling/CLAIMS.md`](../../studies/splicebert_acceptor_coupling/CLAIMS.md),
not by filenames or artifact self-labels.
