# SpliceBERT: worked example

SpliceBERT is the first full case study for the repository's mechanistic-discovery methodology. It shows
how to move from a native behavior through candidate localization, causal intervention, mediation,
faithfulness testing, correction, and scoped interpretation.

Read the repository-level [`../../MECHANISM.md`](../../MECHANISM.md) for the current one-file mechanism.

## Live findings

### Donor–acceptor computation

Corrupting a donor's paired acceptor changes the native masked-donor margin, while matched non-partner
and other-donor controls have much smaller effects. The A0→A12 program supports a structured causal core:

- a mostly relative positional address routes through L2H15/MLP2 into L3H1 attention;
- an acceptor-local payload is generated through L0H7/H12/H10→MLP0→MLP1→MLP2;
- L3H1 combines address and payload, writes a low-rank donor-side `U_info` object, and downstream heads
  transport it to the MLM readout.

The structured core is not a minimal circuit. Clean core outputs are sufficient when injected, but the
computation has a gradual no-knee faithfulness curve and needs about 72/103 nodes for 90% recovery.

### Period-3 routing carrier

The learned position table contains a causal rank-2 period-3 code. Deletion collapses donor coupling;
matched restorations recover it. C34/C35 further show that its contribution is distributed across the
donor–acceptor span and, within the tested carrier intervention, acts primarily through L3H1's attention
pattern rather than carrier-dependent changes in L3H1's values. Its training origin remains open.

### Separate ≈10.63 carrier

A distinct multispecies positional carrier reaches controlled post-softmax attention in eight heads. It
is not the period-3 code. Its OV content, behavioral use, and biological origin remain unknown.

The proposed human-only ≈10.2 chromatin-reader circuit failed at the attention gate. A separate
nucleosomal signal near 10.298 nt is real but is distinguishable from the model's ≈10.63 carrier.

### Head–Context Atlas

The Atlas extends discovery across single-pair, clustered, multi-pair, cassette/complex, and matched
control contexts. Four general-use components validate at group level. Context-associated signals are
preserved as hypotheses but did not meet the frozen precision gate. This broadens the component map; it
does not yet establish cross-example state transduction.

## Main limitation

The payload is acceptor-local and is not explained by preregistered bulk composition, but its semantic
identity remains unresolved. The former splice-strength label was withdrawn after sample-instability and
two-way dyadic inference (R22).

## Evidence and provenance

- [Current mechanism and scope](../../MECHANISM.md)
- [Compact live claims](../splicebert_partner_selection/CLAIMS.md)
- [Full corrections and retractions](../splicebert_partner_selection/CORRECTIONS.md)
- [Preregistration status index](../splicebert_partner_selection/preregistrations/README.md)
- [Detailed reproduction map](../splicebert_partner_selection/REPRODUCIBILITY.md)
- [Head–Context Atlas](../splicebert_head_context_atlas/README.md)
- [Position/carrier results index](../../results/phase-lattice/RESULTS_INDEX.md)
- [Repository reproduction entry point](../../REPRODUCE.md)

## Transfer lesson

The reusable result is the method: qualify a native behavior, prove the instrument can see it, use
observational tools only to choose interventions, distinguish QK/routing/OV/behavior evidence, require
matched controls, and preserve every correction. A second GFM should not be expected to reproduce the
same heads; it should be subjected to the same evidential discipline.
