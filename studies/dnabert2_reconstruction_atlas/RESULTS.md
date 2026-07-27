# DNABERT-2 Reconstruction Atlas results index

> **Tier: T1 — canonical for result status.**

## Scientific results

**None.** The target atlas has not run.

| Run | Question | Status | Artifact | Manifest | Claim IDs |
|---|---|---|---|---|---|
| — | — | not run | — | — | — |

## Infrastructure verification

The test suite verifies coordinate schemas, BPE cover selection, masking, scoring normalization,
alternative eligibility, empirical-family separation, context invariants, gate failure behavior,
add-one inference, matched-size causal nulls, adapter unpadding maps, and immutable writes.
Infrastructure tests license only the IF rows in `CLAIMS.md`.

### Phase-A bounded collection qualification

`results/dnabert2_reconstruction_atlas/collection_qualification_v1.json` records 5,120 target
examples and 2,560 baseline-only examples with zero cluster overlap. Under the frozen
one-control-per-family supply design, splice acceptor (501/503 development/confirmation
clusters), splice donor (451/445), start codon (354/350), and stop codon (419/393) enter
preflight calibration. Kozak (5/6) is quarantined before model outcomes. The rejected
three-control supply audit is preserved as a bound comparison in the same artifact.

This is a collection/instrument result only. It licenses IF6 and no biological behavior claim.

### Preflight calibration v1 — instrument unresolved

`results/dnabert2_reconstruction_atlas/preflight_calibration_v1.json` used 64 development
clusters per supply-qualified class and no empirical target-control contrast. Geometry, matching
balance, split invariants, finite scores, deterministic replay, and batch invariance passed.

The positive control did not qualify. Four-copy effects were +0.0051 acceptor, −0.0043 donor,
+0.0089 start codon, and +0.0088 stop codon nats per covered nucleotide; only stop codon was
strictly dose-monotone. The v1 plant was randomly located anywhere downstream after a ≥32-nt
buffer, so it mixed sensitivity with arbitrary-distance context use. Status is
`INSTRUMENT_UNRESOLVED`; interpretation is frozen while placement is diagnosed. This licenses
IF7 and no behavior claim.

Bulk runtime collections and per-site shards will remain ignored. Every settled compact result must
be immutable, indexed here, and accompanied by a manifest containing checkpoint, data inputs,
parameters, seeds, code revision, and command.
