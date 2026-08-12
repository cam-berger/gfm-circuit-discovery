# Decision log — SpliceBERT positional boundary coherence

> **Tier: T2 — provenance.** Model-free construction decisions are recorded below; no successor model
> outcome exists.

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

## 2026-08-11 — record the audited Phase-I supply stop before selection

**Evidence available.** The model-free production candidate census and its independent reconstruction had
completed and passed. Across the permitted chromosomes after the frozen freshness and annotation filters,
the unique normalized-gene counts for donor phases 0/1/2 were 1,908/1,400/809 and for acceptor phases 0/1/2
were 2,000/1,496/854. The frozen roles require 1,104 complete blocks, each consuming one globally unique gene
from every role/phase slot. Thus donor phase 2 has a necessary deficit of 295 genes and acceptor phase 2 a
necessary deficit of 250. No successor checkpoint had been loaded and no model outcome existed.

**Alternatives considered.** Letting the optimizer run until it raised without an artifact was rejected
because it would consume substantial work yet leave the terminal gate unauditable. Writing the best partial
cohort was rejected because partial selected blocks could be mistaken for a development result and cannot
satisfy the frozen role sizes. Reducing a role, relaxing freshness/global disjointness, or opening sealed
`chr8`/`chr9` was rejected because each changes the frozen design and none is authorized by low capacity.

**Decision and affected files.** Adopt Amendment A5. Before matching, the producer records exact unique-gene
capacity for all seven source slots and donor, acceptor, and global unions. A proved failure emits only a
compact immutable capacity-stop artifact plus manifest. The independent cohort auditor separately rederives
the certificate, binds the candidate/audit/manifest/blacklist/variance inputs, verifies that no block or cell
was selected, and returns `AUDIT_PASS_INSTRUMENT_UNRESOLVED` for an exact stop or `AUDIT_FAIL` for a mismatch.
The ordinary successful path is unchanged. Because the candidate census is intentionally ignored at 2.34 GB,
both records bind its byte and semantic identities, name `candidate_census_v1.manifest.json` as the
regeneration receipt, and state that direct clean-checkout re-execution requires regeneration.

**Effect on claim scope.** Phase I-A is instrument-unresolved on supply and the execution chain stops before
model access. This licenses no positive, null, inverted, behavioral, mechanistic, or biological conclusion;
it proves only that the frozen 1,104-block globally gene-disjoint cohort cannot be built from the eligible
census.
