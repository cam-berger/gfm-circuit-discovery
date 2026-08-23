# Mechanistic Discovery in Genomic Foundation Models

This repository studies what genomic foundation models compute, using causal interventions rather than
attention maps or probes alone. SpliceBERT is the main case study. Reciprocal masked-acceptor experiments
and DNABERT-2 promoter experiments are smaller tests of how well the method transfers.

The current checkout contains the research record, code, and figures, but not the live manuscript source.
[`RESULTS.md`](RESULTS.md) contains the current evidence audit and a fact-check of the working paper draft.

## Main result

On 400 sealed examples from chromosomes 8 and 9, corrupting the paired acceptor changes SpliceBERT's
masked-donor logit margin by `+0.872 [0.771, 0.969]`. Matched non-partner controls are near zero.

Causal interventions resolve this behavior into four roles:

- **Address:** L3H1 selects the paired acceptor mainly from positional information, with L2H15 strengthening
  the attention-side signal.
- **Payload:** L3H1 reads causally relevant values from a four-token acceptor window. Earlier heads and MLPs
  construct those values, but their biological meaning is still unknown.
- **Write:** L3H1 combines address and payload into a donor-local state captured well by a tested rank-8
  subspace.
- **Delivery:** later heads, including L4H0, L4H2, and L5H11, carry that state toward the masked-token output
  through their value-output paths.

These components form a structured, top-ranked causal core inside a broader distributed computation. They
are not a complete, minimal, or autonomous circuit. The held-out static-ranking curve has no sparse knee,
but it does not prove that no smaller alternative circuit exists.

The learned period-3 positional direction and the boundary-register preference seen in fixed-span chimeras
are separate findings. No current experiment connects them, identifies reading frame, or establishes a
cellular splicing mechanism.

The reciprocal SpliceBERT experiments show paired-donor dependence on validation examples and several
component effects, but not a reverse address → payload → write → delivery mechanism or symmetry. The
DNABERT-2 study finds a different, MLP-heavy conditional causal core; it is a bounded comparison, not a
replication of the SpliceBERT circuit.

## Evidence and claims

The claim ledgers and immutable result artifacts control the scientific record. Narrative summaries do not
override them.

- [`RESULTS.md`](RESULTS.md) — audited numbers, data splits, manuscript fact-check, and claim coverage
- [`MECHANISM.md`](MECHANISM.md) — current mechanism and its limits
- [Donor claim ledger](studies/splicebert_partner_selection/CLAIMS.md) — live SpliceBERT claims
- [Correction ledger](studies/splicebert_partner_selection/CORRECTIONS.md) — retractions and corrections
- [Preregistration conformance](docs/PREREG_CONFORMANCE.md) — where historical runs did and did not match
  their frozen protocols
- [Methodology](docs/METHODOLOGY.md) and [evidence ladder](docs/EVIDENCE_LADDER.md) — standards for new studies

Attention, attribution, probes, and spectra nominate candidates. Intervention establishes causal use. A
causal model mechanism is not automatically a biological mechanism.

## Study map

| Study | Status |
|---|---|
| [SpliceBERT partner selection](studies/splicebert_partner_selection/README.md) | Main donor-mechanism record |
| [SpliceBERT acceptor coupling](studies/splicebert_acceptor_coupling/README.md) | Bounded reciprocal behavior and component evidence; mechanism handoff stopped before chr8/9 |
| [DNABERT-2 promoter](studies/dnabert_promoter/README.md) | Bounded frozen-model comparison; notebook-only fine-tuned extension archived |
| [Head–Context Atlas](studies/splicebert_head_context_atlas/README.md) | Component-reuse discovery and internal validation |
| [GTEx fine-tuning shift](studies/splicebert_gtex_finetune_shift/README.md) | Fine-tuned readout and positional-carrier follow-up |
| [Acceptor context](studies/splicebert_acceptor_context/README.md) | Prospective protocol; no result |
| [Positional boundary coherence](studies/splicebert_positional_boundary_coherence/README.md) | Model-free capacity stop |
| [Positional proxy modifiability](studies/splicebert_positional_proxy_modifiability/README.md) | Model-free capacity result; frozen model protocol never run |
| [Exon-edge compatibility](studies/splicebert_exon_edge_compatibility/README.md) | Archive record for live claims C43 and C44 |

Legacy `acceptor-interpretability/`, `ntv3-interpretability/`, and top-level `dnabert/` results were
exploratory or superseded and have been removed from the active tree. They remain recoverable at
`archive/unsupported-exploratory-2026-08-22`. New studies should start from
[`studies/TEMPLATE_REAL_GFM_STUDY/`](studies/TEMPLATE_REAL_GFM_STUDY/README.md).

## Repository layout

- `src/gfmcd/` — reusable model adapters, circuit tools, and SpliceBERT assays
- `scripts/` — experiment, audit, and figure entry points
- `results/` — immutable results, manifests, and audits
- `figures/` — generated figures and provenance
- `tests/` — analysis checks and memorial tests
- `docs/` — methodology, evidence rules, and study guidance

## Reproducing the repository

```bash
uv sync --extra splicebert --extra biology --extra dev
uv run --extra splicebert --extra biology --extra dev pytest
```

The model checkpoint, hg38 FASTA, and GENCODE annotation are not committed. See
[`REPRODUCE.md`](REPRODUCE.md) for setup, data requirements, and the claim-to-artifact map.

Repository code is available under the [MIT License](LICENSE). External model and data terms are listed in
[`docs/THIRD_PARTY_ASSETS.md`](docs/THIRD_PARTY_ASSETS.md).
