# SpliceBERT: worked example

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

SpliceBERT is the first full case study for the repository's mechanistic-discovery methodology. It shows
how to move from a native behavior through candidate localization, causal intervention, mediation,
faithfulness testing, correction, and scoped interpretation.

Read the repository-level [`../../MECHANISM.md`](../../MECHANISM.md) for the current one-file base-model
mechanism and the
[GTEx fine-tuning-shift findings](../splicebert_gtex_finetune_shift/FINDINGS.md) for the supervised
checkpoint family.

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
pattern rather than carrier-dependent changes in L3H1's values. Cross-model contrasts attribute its
corpus origin to multi-species, codon-linked pretraining; the training dynamics remain open.

### Separate ≈10.63 carrier

A distinct multispecies positional carrier reaches controlled post-softmax attention in eight heads. It
is not the period-3 code. Its donor-coupling OV/behavioral arm is bounded rather than positive, and its
general function remains open. Cross-model and sibling-checkpoint contrasts attribute its origin to
multi-species pretraining; they do not supply a biological function.

The proposed human-only ≈10.2 chromatin-reader circuit failed at the attention gate. A separate
≈10.298-nt nucleosomal signal measured in mouse ES-cell chromatin (Voong 2016, mm9) is real, but it is
0.33 nt from the model's ≈10.63 carrier and constitutes a powered refutation of that biological reading.

### Head–Context Atlas

The Atlas extends discovery across single-pair, clustered, multi-pair, cassette/complex, and matched
control contexts. Four general-use components validate at group level. Context-associated signals are
preserved as hypotheses but did not meet the frozen precision gate. This broadens the component map; it
does not yet establish cross-example state transduction.

### GTEx fine-tuning shift

The released GTEx classifiers retain paired-site coupling even though the inherited masked-LM readout
becomes broadly incompatible. Several layer-5 attention routes carry nonlocal sequence-order effects into
the final MLP, where a mesoscale native-unit population synthesizes a compact, checkpoint-rotated
splice-classification write.

The inherited period-3 and ≈10.63 carriers remain functional for local donor/acceptor evidence but are
separable from the paired-site coupling route. Sparse decompositions preserve the downstream MLP
difference but have not faithfully represented the upstream base→fine-tuned state shift, so feature
semantics and steering remain closed.

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
- [GTEx fine-tuning-shift findings](../splicebert_gtex_finetune_shift/FINDINGS.md)
- [GTEx live claims](../splicebert_gtex_finetune_shift/CLAIMS.md)
- [Position/carrier results index](../../results/phase-lattice/RESULTS_INDEX.md)
- [Repository reproduction entry point](../../REPRODUCE.md)

## Transfer lesson

The reusable result is the method: qualify a native behavior, prove the instrument can see it, use
observational tools only to choose interventions, distinguish QK/routing/OV/behavior evidence, require
matched controls, and preserve every correction. A second GFM should not be expected to reproduce the
same heads; it should be subjected to the same evidential discipline.
