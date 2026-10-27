# Research studies

SpliceBERT donor–acceptor coupling is the main worked example. The other studies test transfer,
context dependence, or the limits of proposed assays. This index separates recorded model evidence,
prospective designs, and model-free stopping results.

Read the [repository overview](../README.md) and [evidence audit](../RESULTS.md) first; the audit records
the evidence and manuscript state at its stated date.
Within each study, `CLAIMS.md` gives the licensed statement and its limits; the linked artifacts,
manifests, audits, and corrections determine what can be reproduced or cited.

## Studies with model evidence

| Study | Scope and evidence boundary | Claim ledger |
|---|---|---|
| [SpliceBERT partner selection](splicebert_partner_selection/README.md) | Main masked-donor mechanism and positional-carrier record. A structured causal core within distributed computation; payload biological meaning remains unresolved. | [Claims](splicebert_partner_selection/CLAIMS.md) |
| [SpliceBERT acceptor coupling](splicebert_acceptor_coupling/README.md) | Reciprocal masked-acceptor behavior and bounded component evidence. The later mechanism handoff stopped before target science. | [Claims](splicebert_acceptor_coupling/CLAIMS.md) |
| [DNABERT-2 promoter](dnabert_promoter/README.md) | Single-seed, frozen-encoder comparison on a different task and tokenizer; no frozen preregistration or replication of the SpliceBERT mechanism. | [Claims](dnabert_promoter/CLAIMS.md) |
| [SpliceBERT Head–Context Atlas](splicebert_head_context_atlas/README.md) | Component reuse across sequence contexts, with discovery and internal validation. Does not establish donor-mechanism transfer across contexts. | [Claims](splicebert_head_context_atlas/CLAIMS.md) |
| [SpliceBERT GTEx fine-tuning shift](splicebert_gtex_finetune_shift/README.md) | Comparison of original-MLM and native-classifier readouts across published fine-tuning seeds, with component and representation follow-ups. | [Claims](splicebert_gtex_finetune_shift/CLAIMS.md) |
| [SpliceBERT CC/max crossed-register verification](splicebert_register_operator_replication/README.md) | Registered v2 artifacts: measured CC/max Ψ=+0.430, but frozen resolution failed; all other nine gates pass. Outcome-aware 342-block, 120-nt verification; no confirmed preference claim. | [Draft claims and retained v1 failure](splicebert_register_operator_replication/CLAIMS.md) |

Sibling-checkpoint and cross-model positional-carrier comparisons belong to the donor study's ledger;
their artifacts are indexed under [origin-siblings](../results/origin-siblings/RESULTS_INDEX.md) and
[origin-crossmodel](../results/origin-crossmodel/README.md).

## Prospective and stopped designs

| Study or record | Status | Controlling record |
|---|---|---|
| [SpliceBERT acceptor context](splicebert_acceptor_context/README.md) | Draft protocol; no registered model result or live claim. | [Draft design](splicebert_acceptor_context/DRAFT_PREREGISTRATION.md) |
| [SpliceBERT positional boundary coherence](splicebert_positional_boundary_coherence/README.md) | Model-free capacity failure; no complete cohort or model outcome. | [Claims](splicebert_positional_boundary_coherence/CLAIMS.md) |
| [SpliceBERT positional-proxy modifiability](splicebert_positional_proxy_modifiability/README.md) | Model-free capacity result; the frozen model protocol was never executed. | [Claims](splicebert_positional_proxy_modifiability/CLAIMS.md) |
| [Exon-edge compatibility archive](splicebert_exon_edge_compatibility/README.md) | Preserved protocols and archive pointer for donor-ledger C43/C44; no executable study here. | [Donor claims](splicebert_partner_selection/CLAIMS.md) and [corrections](splicebert_partner_selection/CORRECTIONS.md) |

A construction failure or failed instrument is a research result about the assay. It does not supply a
positive or negative model endpoint. Retained protocols and runners do not authorize reopening a
closed execution path.

## Starting a study

Use the [real-GFM study template](TEMPLATE_REAL_GFM_STUDY/README.md), together with the
[methodology](../docs/METHODOLOGY.md), [evidence ladder](../docs/EVIDENCE_LADDER.md), and
[transfer guide](../docs/TRANSFER_GUIDE.md). Define a biological computation and model-native behavior,
then establish assay capability before interpreting internal components.

Each study owns its protocols, claim ledger, decisions, result provenance, and script directory. The
[script index](../scripts/README.md) locates implementations; the
[result index](../results/RESULTS_INDEX.md) explains the shared machine-readable manifest.
