# SpliceBERT decision record

This is a consolidated chronology of scientific pivots. It records why the active tree changed; it is not a second claim ledger.

## 2026-07-10 to 2026-07-11 — initial partner-selection framing

The project began as a frozen-representation forced-choice task intended to distinguish true splice partners from decoys. Early performance appeared to support the task.

Decision: **withdraw the framing and implementation.** The true site occupied a fixed position, a geometry-only oracle could solve the task, and site detection was sufficient. A v2 using annotated decoys still reduced to choosing the stronger major site over weaker minor sites. A “random backbone” control was also a no-op.

Consequence: forced-choice partner-selection code and reports are historical only. The active question became whether the model's native score at one splice site depends on its paired site.

## 2026-07-12 — pivot to native cross-site coupling

The replacement design masked a donor nucleotide, corrupted the paired acceptor with real bases, and compared motif- and distance-matched controls. This produced a native behavioral quantity that could be tested without a trained probe.

Decision: use unconditional coupling for behavioral magnitude and reserve effect-conditioned examples for recovery fractions. This replaced the inflated 1.359 held-out mean with the unconditional 0.973 validation estimate and later 0.872 sealed estimate.

Decision: include both strands after a transcript-order bug was found. The corrected eligible population roughly doubled and the selected heads remained L3H1 and L2H15.

## 2026-07-12 — localization became a subcircuit claim

L3H1 ranked first in the exhaustive head scan, but held-out factorials showed it also had a partner-independent role. L2H15 was a second causal contributor missed by denoising alone.

Decision: describe `C = {L3H1, L2H15}` as a **dominant subcircuit**, not a single dedicated head and not a share of summed causal mass. Marginal effects overlap in serial and redundant systems.

Decision: use masked probes for model health. The earlier unmasked metric measured input copying and could not fail.

Decision: enumerate or de-duplicate finite random-null universes and use exact add-one p-values. The previous duplicated null overstated significance.

## 2026-07-12 — mechanism interventions were narrowed

Key/value patching supported a conjunctive acceptor read. Whole-output restoration closed the L2H15 → L3H1 edge; position profiling located L3H1's output at the donor neighborhood; downstream patching showed a distributed readout chain.

Decision: retract key/value **swap** evidence because swapping complete pairs is an algebraic identity.

Decision: retract the later **content-addressed relocation** interpretation because moving vectors that already contain absolute position information makes the result architecturally forced. The sealed numerical result remains provenance, but it does not license a pointer or lookup claim.

Decision: report completeness as inconclusive. Large complement ablations damage general language-model behavior, and size-matched random sets are similarly destructive.

## 2026-07-12 — sealed replication

Four preflight blockers were fixed before unblinding: strand eligibility, algebraically invariant binding intervention, partner leakage into a non-partner pool, and an unmatched/distance-confounded specificity comparator. The circuit was reselected on corrected data and did not change.

Decision: freeze code hashes, circuit, controls, and 13 pass/fail gates in [`PREREGISTRATION_SEALED.md`](PREREGISTRATION_SEALED.md), then open chr8/chr9 once. All 13 gates passed.

Decision: preserve the preregistration unchanged. Post-unblinding corrections belong in this record and [`CLAIMS.md`](CLAIMS.md).

## 2026-07-12 to 2026-07-13 — period-3 interpretation corrected

An initial “42-head codon-phase lattice” was built on phase purity. Audit showed the statistic was maximized by single-offset heads and therefore measured sharpness/locality.

Decision: retract the broad lattice. Require repeated comb structure, deletion, protected controls, behavior, and restoration.

The replacement result established a rank-2 period-3 position code: deletion collapsed attention phase mass and donor–acceptor coupling, while matched single-head restorations identified L2H15 and L3H1 as mediators.

Decision: treat the period-3 clock as part of the donor–acceptor mechanistic story, not a detached collection of phase plots.

## 2026-07-13 — second-carrier hypotheses disciplined

A separate ≈10.63 positional carrier prompted α-helical, DNA-helical, nucleosomal, and generic-ruler hypotheses. Several early assays were either unpowered or built on unverified controls.

Decision: an anomalous control freezes interpretation. A band may be called energy-matched or low-leak only after those properties are measured; it may not be called inert by assertion.

Decision: calculate entropy headroom, coherence length, wavelength resolution, and expected detection floor before training or scanning. The synthetic/source exploration taught useful design lessons but is not an active result family.

Decision: use landmark-anchored linear matched filters for weak biological periodicity. Unanchored power is quadratic in effect depth and was calibrated to be blind at the relevant scale.

A powered dyad assay found a strong ≈10.298 nucleosomal signal, including frozen-phase replication and offset specificity. It did not match the multispecies ≈10.630 carrier.

Decision: close the nucleosomal-source interpretation for the multispecies carrier and leave its origin open. Preserve the positive biological finding as a distinct result.

## 2026-07-13 — the evidence-level firewall held

QK-logit scans suggested human-only ≈10.2 readers and multispecies ≈10.63 readers. Earlier post-softmax estimators produced impossible or unbounded quantities.

Decision: stop and build a gated post-softmax instrument with known period-3 positive/negative controls, injection recovery, per-head permutation nulls, bounded coherence, band-mass eligibility, deletion audits, reciprocal bands, and content-shuffle checks.

The validated instrument found eight multispecies Level-A heads and no corrected human-only Level-A head. Exact add-one correction removed a raw zero-hit artifact; band-mass measurement showed that most preregistered human-only readers were never eligible.

Decision: fire the preregistered stopping rule. Close the human-only chromatin-circuit proposal at QK, do not run species-matched rescue experiments, and advance only the multispecies carrier to potential OV/behavior work.

## 2026-07-14 — active-tree consolidation

The repository contained overlapping reports, superseded scripts, unindexed result JSON, VOID families, and older figures that made historical branches look active.

Decision: make the active tree an executable record of the current claims:

- keep seven authoritative study documents;
- keep only scripts that generate indexed discovery, validation, or reproduction artifacts;
- remove unindexed/VOID result families from the active tree;
- preserve frozen preregistration and permanent retractions;
- rely on Git history for obsolete experiment implementations.

This is a scope correction, not deletion of scientific provenance. The authoritative record keeps the conclusions, limitations, failures, and reasons for each pivot while removing competing entry points.
