# SpliceBERT study record

This directory is the authoritative record for the SpliceBERT worked example. Its historical name is retained because result manifests and frozen protocols refer to it; the live study is broader than partner selection.

The current research question is: **how does SpliceBERT implement genomic computations, and which parts of that discovery process transfer to another genomic foundation model?** SpliceBERT supplies three cases:

1. a donor–acceptor coupling mechanism that reaches causal behavior;
2. a learned period-3 positional clock that supports that mechanism;
3. a separate multispecies ≈10.63 positional carrier established at post-softmax attention, with downstream use and origin still open.

The proposed human-only ≈10.2 chromatin circuit is closed at the attention gate. That negative result is part of the method, not a side experiment.

## Authoritative records

| Record | Purpose |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | Current scientific narrative and limits |
| [`CLAIMS.md`](CLAIMS.md) | Compact status ledger and permanent retractions |
| [`METHODS.md`](METHODS.md) | Study-specific discovery, preflight, intervention, and stopping rules |
| [`DECISIONS.md`](DECISIONS.md) | Chronology of pivots and why interpretations changed |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Canonical commands, artifacts, manifests, and reproduction paths |
| [`PREREGISTRATION_SEALED.md`](PREREGISTRATION_SEALED.md) | Frozen pre-unblinding protocol; retained unchanged |

These seven files replace the previous collection of overlapping reports, plans, gate notes, migrations, and post-hoc mechanism narratives. Removed documents and retracted artifacts remain available in Git history; they are not active alternatives to this record.

## Where to go next

- For the transferable method, start at [`../../docs/METHODOLOGY.md`](../../docs/METHODOLOGY.md) and [`../../docs/EVIDENCE_LADDER.md`](../../docs/EVIDENCE_LADDER.md).
- For a concise walk-through, read [`../splicebert/README.md`](../splicebert/README.md).
- For machine-readable evidence, use [`../../results/splicebert_coupling/`](../../results/splicebert_coupling/) and the [`../../results/phase-lattice/RESULTS_INDEX.md`](../../results/phase-lattice/RESULTS_INDEX.md).
- For executable workflows, use [`../../scripts/README.md`](../../scripts/README.md).

## Reading rule

The strongest supported evidence level controls the wording. A position-table feature is not automatically an attention mechanism; an attention mechanism is not automatically an OV or behavioral mechanism; a model mechanism is not automatically a biological mechanism. When prose conflicts with an indexed artifact or the claim ledger, the artifact and ledger control.
