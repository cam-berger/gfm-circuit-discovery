# BG1/BG2 pre-execution amendment — 2026-08-11

> **Status: binding before target execution.** This amendment was written before any
> canonical BG1 preflight, BG1 validation, or BG2 target result was produced. It does
> not edit or replace the frozen
> `PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md`; that file remains the immutable
> design record, and this document records the narrow execution-time departures.

## A1. Two-way inference for the BG2 cross-gene contrast

The BG2 `true - same-transition cross-gene` endpoint joins a target gene to a
cross-gene payload source, and both target and source genes can recur across dyads.
The target-only paired gene bootstrap frozen in §6.2 therefore does not represent
the dependence created by source reuse. Before target execution, that interval is
replaced by the repository's two-way target-by-source gene multiplier bootstrap:

- one observation per target gene after the preregistered background average;
- cluster endpoints are the target gene and the recorded cross-gene source gene;
- additive independent mean-zero Gaussian multipliers are drawn for the union of
  endpoint genes, with same-gene endpoints handled as one cluster;
- 2,000 draws and the already frozen BG2 seed offsets are retained;
- the two-sided 95% basic interval uses `alpha=0.05`; the one-sided 95% lower bound
  uses the corresponding `alpha=0.10` lower endpoint.

The BG2 route decision still requires this one-sided lower bound to exceed zero.
The within-target signed-coordinate max-family test and every other BG2 endpoint
remain unchanged. The independent auditor must reconstruct the target/source dyads
from public rows and implement the multiplier calculation independently of runner
decision code.

## A2. Smoke-pass semantics

`--smoke` is a non-authorizing instrument/mechanics exercise. Its intentionally
small cohorts and 40-world calibration cannot adjudicate validation MDE, planted
power, or calibrated false-positive-rate gates. A smoke artifact therefore records
the raw outcomes of these five checks as non-authorizing diagnostics:

1. target resolution;
2. planted sensitivity;
3. endpoint-family false-positive rate;
4. joint-claim composite-null false-positive rate; and
5. route resolution.

For smoke only, top-level `PASS` means that the remaining instrument/mechanical
checks passed. `target_authorization_PASS` is always false for smoke, failed raw
diagnostics remain visible, and both target runners continue to reject every smoke
artifact. In a canonical non-smoke preflight all original checks remain active,
top-level `PASS` still requires every frozen gate, and no threshold, cohort size,
seed, resampling count, or authorization rule changes.

## A3. Implementation repair

The BG2 runner's frozen supply drift check referenced two undefined local aliases.
The aliases are fixed at the already declared minima (96 route genes total and 24
per strand), and a runner-contract test now exercises the check before model load.
This repairs executability without changing cohort selection or a scientific gate.

## A4. Deliberately unresolved item

The frozen protocol names an `INSTRUMENT_UNRESOLVED` result when an invariant fails,
whereas the current runners fail closed before writing a result artifact. That
provenance improvement is not part of this amendment: no partially populated target
artifact format has been specified or validated. A failed invariant still halts
execution and licenses no result, but the absence of a failure artifact remains an
open implementation issue recorded in `DECISIONS_A0.md`.
