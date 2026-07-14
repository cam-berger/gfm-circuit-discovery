# SpliceBERT: worked example

SpliceBERT is the first full case study for the repository's mechanistic-discovery methodology. It is used to demonstrate how a project moves from model behavior, through candidate localization and causal intervention, to a scoped circuit claim—and how it records failed interpretations without erasing them.

## Live findings

### 1. Donor–acceptor information-routing circuit

SpliceBERT exhibits a donor–acceptor coupling behavior. QK/attention analysis, ablation, rescue, and path decomposition support a directed circuit in which donor information is routed to an acceptor-addressing pathway. This is the main worked example of the complete causal workflow.

### 2. Learned period-3 positional clock

The learned position table contains a period-3 rotary code. Multiple heads read it, deleting it collapses donor–acceptor coupling, and targeted restoration mediates part of the lost effect. The clock is not a separate biological story; it is a positional mechanism supporting the donor–acceptor circuit.

### 3. Separate ≈10.63 positional carrier

A distinct carrier is present in the multi-species position table and is read through post-softmax attention by eight heads under controlled tests. Its carrier plane is separate from the period-3 clock. No downstream OV/mediation or behavioral function has been demonstrated, and its biological origin is unknown.

A proposed human-only ≈10.2 chromatin circuit failed at the attention gate. That null is part of the methodological record: the same instrument detects the multi-species circuit, so the negative is not explained by a blind assay.

## How the case maps to the method

| Method stage | SpliceBERT example |
|---|---|
| Biological computation | Relating donor and acceptor information across sequence context |
| Model behavior | Native donor–acceptor coupling under paired sequence interventions |
| Candidate localization | Attention/QK, position spectra, representation geometry, attribution |
| Necessity | Head, path, and positional-carrier ablations |
| Mediation | Activation rescue, carrier restoration, and path decomposition |
| Specificity | Reciprocal/low-leak bands, content shuffle, random matched head sets, protected metrics |
| Replication and correction | Sealed tests, checkpoint contrasts, exact permutation corrections, explicit retractions |

## Evidence and provenance

- [Current figure and evidence summary](../../figures/splicebert/README.md)
- [Evidence ladder with exact conventions](../../figures/splicebert/EVIDENCE.md)
- [Concise findings record](../splicebert_partner_selection/FINDINGS.md)
- [Detailed claim and retraction ledger](../splicebert_partner_selection/CLAIMS.md)
- [Methods preflight audit](../splicebert_partner_selection/METHODS_AUDIT.md)
- [Donor–acceptor results](../../results/splicebert_coupling/)
- [Position/carrier results index](../../results/phase-lattice/RESULTS_INDEX.md)
- [Active scripts](../../scripts/splicebert/)

The directory `studies/splicebert_partner_selection/` retains the detailed evolving record for provenance. Some filenames reflect earlier hypotheses; the claim ledger and retractions, not the historical name, determine current status.

## Transfer lesson

The reusable result is the approach: identify a native behavior, qualify the instrument, use observational tools to choose interventions, require necessity and mediation for circuit claims, and keep biological interpretation behind its own evidence gate. A second GFM should not be expected to reproduce these components; it should be subjected to the same discipline.
