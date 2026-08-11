# Decision log — SpliceBERT positional boundary coherence

> **Tier: T2 — provenance.** Prospective study; no post-outcome decision has been made.

The frozen choices belong in [`PREREGISTRATION.md`](PREREGISTRATION.md). Add a dated entry here only after
data or diagnostics force a decision, amendment, stop, or scope change. Each entry must name the evidence
available at the time, alternatives considered, affected files, and effect on claim scope. This log never
silently amends the preregistration or claim ledger.

## 2026-08-11 — close the pre-target identity and conformance chain

**Evidence available.** No successor model forward or target outcome had been produced. Model-free
construction had begun at revision `470d68ff` using local files whose independently measured digests match
the identities recorded in Amendment A4. A pre-execution chain audit found that the freezer had no input for
the protocol-required independent conformance audit; production runners compared the checkpoint directory
rather than its weight bytes; reference/annotation/MaxEnt records were pinned but not compared to frozen
values at execution freeze; device and batch were accepted at runtime but absent from the seal; sharded
scoring would write its shard and then fail while building a manifest that assumed compact assembled
evidence; and the final audit would write JSON and then fail while reading an absent `target_mode` field for
its manifest. Before its first artifact completed, the model-free candidate producer was also observed to
perform a full chromosome-feature scan for every eligible edge, an avoidable quadratic overlap lookup. The
first full census then stopped without an artifact when alternative transcript edges materialized duplicated
role/source loci that violated the frozen uniqueness invariant.

**Alternatives considered.** Repeating the in-progress census solely to add identical hash comparisons to
the generic builder was rejected because it changes no candidate and supplies no independent evidence.
Leaving the overlap scan brute-force was rejected because the same exact source-order result can be obtained
by an interval range index without changing eligibility or ordering.
Choosing one representative from duplicated source loci was rejected because Section 4.1 says duplicate
loci are excluded and representative selection would add an unfrozen tie-break. Retaining every duplicate
was rejected because it violates global source uniqueness. Excluding all occurrences with a named reason
implements the existing frozen rule and remains independently reconstructable.
Treating the existing preflight auditor as protocol conformance was rejected because it audits recorded
measurements and never reads the full protocol-to-code surface. Requiring the model directory to live at one
worktree-relative path was rejected because directory location is not checkpoint identity. Leaving device,
batch, or writer failures as operator conventions was rejected because the frozen protocol requires a
machine-checked, replayable execution package.

**Decision and affected files.** Adopt Amendment A4. Add
`audit_protocol_code_conformance.py` and `pbc_frozen_inputs.py`; require and hash-bind the passing audit and
manifest in the Phase-II freezer and target seal; compare official weight and biological/config inputs by
exact digest; bind and recheck device, batch, and dtype; and repair the two guaranteed post-write crashes.
The scientific endpoints, thresholds, and frozen eligibility rules remain unchanged. Producer and auditor
gain exact order-preserving interval indexing and the already-required all-occurrence duplicate-source-locus
exclusion; the stopped census emitted no artifact or model output.
Focused regression tests memorialize failure on substituted bytes,
missing/failing conformance, runtime drift, shard-manifest generation, and final-audit manifest generation.

**Effect on claim scope.** None. This decision can only stop unauthorized execution or make already-declared
writers complete. It creates no evidence about `Ψ`, P3/P10 necessity, register relocation, mediation, or
biology, and it does not license a claim.
