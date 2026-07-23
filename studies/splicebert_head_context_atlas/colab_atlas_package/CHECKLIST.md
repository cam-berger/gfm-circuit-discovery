# Readiness checklist — wider atlas screen

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

## Already green locally

- [x] Annotation census and corrected donor/acceptor sequence enrichment
- [x] Ten role-valid matched contrasts; numeric SMD ≤0.10; non-focal tags exact
- [x] Independent cohort audit: 0 failures, no split/sequence leakage
- [x] Four-packet model preflight: 96 heads + 6 MLPs × A/B, masked health, 0 failures
- [x] Bulk code path reproduces preflight: 2,448 comparisons, max absolute difference 0.0
- [x] Shard validator and gzip merge exercised end-to-end
- [x] 25 tests pass

## Colab gates before bulk target inference

- [ ] Bundle hash verification passes
- [ ] CUDA runtime active; Torch was not reinstalled
- [ ] Cross-device four-packet smoke passes at absolute tolerance 0.002
- [ ] 32-packet benchmark reports GPU model, peak memory, seconds/packet, projected hours/storage
- [ ] 512-packet masked-health panel completes and aggregate gate passes
- [ ] Cameron/authorized agent reviews benchmark and sets `LAUNCH_TARGET = True`

## After target inference

- [ ] 11,807 target packet shards present
- [ ] 512 health shards present
- [ ] All shards hash-match their phase manifests
- [ ] Every shard has 102 unique components and finite A/B fields
- [ ] Validation audit status `PASS`
- [ ] Merge produces 1,204,314 target rows, 52,224 health rows, and 6,487 matched pairs

## Stop conditions

- Any coordinate, identity, determinism, component-grid, hash, or masked-health gate fails.
- CUDA is unavailable or the checkpoint/bundle hash differs.
- Benchmark cost materially exceeds the available budget.
- Never substitute validation or sealed packet files into the discovery runner.
