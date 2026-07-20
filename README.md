# Mechanistic Discovery in Genomic Foundation Models

This repository develops a disciplined way to discover **how genomic foundation models (GFMs)
implement biological computations**. The objective is not to catalogue salient activations; it is to
move from a model-native behavior to a causally supported, explicitly bounded mechanism.

> **Repository state (2026-07-20):** the current scientific record is on
> `donor-graph-address-payload`, which is **substantially ahead of `master`** and has been pushed for review.
> Until that branch is merged, the default GitHub view is not the current project.

**SpliceBERT is the first worked example, not the boundary of the project.** The transfer target is a
second real GFM; Caduceus is the prepared but deferred candidate study.

## Current mechanism, in one sentence

> A structured, top-ranked causal core binds a predominantly relative positional address to an
> acceptor-local payload, then delivers the resulting information to the donor readout within an
> irreducibly distributed computation.

Here, **circuit** means a causally supported collection of components responsible for a scoped behavior.
It does not imply that the collection is the smallest possible circuit. In fact, SpliceBERT's 90%
faithfulness requires about 72 of 103 measured nodes; the named circuit is the structured core of that
distributed computation, not a minimal graph.

The key limitation is equally important: the payload is acceptor-local and is not explained by the
preregistered bulk-composition variables, but its **semantic identity remains unresolved**. The earlier
“splice-site strength” label was withdrawn after it failed independently seeded draws and two-way
gene-dyadic inference (R22).

## Current scientific record

| Result | Evidence reached | Licensed interpretation | First missing bridge |
|---|---|---|---|
| Native donor–acceptor coupling | Behavior + specificity | A masked donor score depends on its own paired acceptor under canonical short-intron controls. | Wider biological and sequence-context transduction. |
| Donor graph A0→A12 | QK → attention → OV/mediation → behavior | An acceptor-local generator feeds L3H1, which binds payload to a mostly relative address and delivers a low-rank donor-side signal through a structured core. | Payload semantics; cross-context causal transfer. |
| Period-3 carrier C8/C34/C35 | Position → routing → mediation → behavior | A distributed relative-positional routing coordinate shapes L3H1's attention across the donor–acceptor span. | Training origin and upstream effects on payload construction. |
| Multispecies ≈10.63 carrier | Position → QK → attention | Eight heads read a distinct positional carrier after softmax. | OV, behavior, and biological origin are open. |
| Human-only ≈10.2 proposal | Position/QK; fails at attention | The proposed chromatin-reader circuit is closed at the attention gate. | No higher claim is licensed. |
| Head–Context Atlas | Matched discovery → group validation → conditional mechanism tests | Four general-use component candidates validate; context-specific signals remain exploratory because they missed the frozen precision gate. | Better-powered context cohorts and cross-example transduction. |

Evidence levels are not interchangeable. A feature in a position table is not automatically used by
attention; attention is not automatically causal; a causal model mechanism is not automatically a
biological mechanism.

## Method

Every study follows the same broad sequence:

1. Define the biological computation and a model-native behavior.
2. Preflight the assay with positive, negative, sensitivity, resolution, and invariant gates.
3. Use probes, attention, attribution, spectra, or subspaces only to generate candidates.
4. Test necessity with matched-size or structured null interventions.
5. Test mediation, direction, and information using patching, rescue, transplant, QK/OV decomposition,
   or edge/path interventions.
6. Replicate across appropriate examples, draws, splits, contexts, checkpoints, or models and stop at
   the strongest interpretation licensed by the evidence.

Attribution ranks; intervention decides. See the [methodology](docs/METHODOLOGY.md),
[evidence ladder](docs/EVIDENCE_LADDER.md), and [mechanism account](MECHANISM.md).

## Study map

| Study | Status | Purpose |
|---|---|---|
| [`studies/splicebert/`](studies/splicebert/README.md) | worked-example overview | Concise entry point to the SpliceBERT evidence ladder. |
| [`studies/splicebert_partner_selection/`](studies/splicebert_partner_selection/README.md) | completed core record; open limitations | Donor graph, period-3 carrier, compact claims, corrections, preregistrations, and reproduction map. |
| [`studies/splicebert_head_context_atlas/`](studies/splicebert_head_context_atlas/README.md) | discovery and internal validation complete | Multi-context head/MLP reuse atlas; context-specific follow-ups remain exploratory. |
| [`studies/splicebert_segment_attribution/`](studies/splicebert_segment_attribution/README.md) | exploratory instrument | Joint-mask attribution DAGs used as candidate generators, not causal claims. |
| [`studies/caduceus_partner_transfer/`](studies/caduceus_partner_transfer/README.md) | design hardened; deferred, not run | Planned cross-architecture behavioral transfer to an RC-equivariant bidirectional Mamba MLM. |
| [`studies/codon_phase_lattice/`](studies/codon_phase_lattice/README.md) | historical exploratory protocol | Superseded phase-shift framing retained for provenance; not an active result family. |

The reusable study template is under
[`studies/TEMPLATE_REAL_GFM_STUDY/`](studies/TEMPLATE_REAL_GFM_STUDY/README.md).

## Repository map

- `docs/` — transferable method, evidence rules, transfer guide, roadmap, and cleanup record.
- `scripts/splicebert/` — active SpliceBERT experiment, audit, and figure scripts.
- `scripts/splicebert_head_context_atlas/` — Atlas census, screening, mechanism, merge, and audit workflows.
- `src/gfmcd/adapters/` — model access and hook interfaces.
- `src/gfmcd/circuits/` — reusable circuit-analysis primitives.
- `src/gfmcd/splicing/` — SpliceBERT-specific assays, competition logic, and cohort utilities.
- `results/splicebert_coupling/`, `results/phase-lattice/`, and
  `results/splicebert_head_context_atlas/` — immutable results, compact matrices, manifests, and audits.
- `figures/splicebert/` — generated figures and declared provenance.

Large Atlas shards and reproducible runtime intermediates are intentionally excluded from Git; compact
results, manifests, audits, code, tests, and Colab notebooks are version-controlled.

Historical GFM-nano, reading-frame, repeat-copy, and retracted v1 partner-selection implementations are
preserved in Git history rather than restored to the active tree.

## Start here

1. Read the [methodology](docs/METHODOLOGY.md) and [evidence ladder](docs/EVIDENCE_LADDER.md).
2. Read the one-file [current mechanism and limitations](MECHANISM.md).
3. Use the [live claim ledger](studies/splicebert_partner_selection/CLAIMS.md); consult
   [corrections](studies/splicebert_partner_selection/CORRECTIONS.md) before quoting a result.
4. Review the [Head–Context Atlas](studies/splicebert_head_context_atlas/README.md) for the current
   cross-context branch.
5. Read the [transfer guide](docs/TRANSFER_GUIDE.md) and
   [Caduceus package](studies/caduceus_partner_transfer/README.md) for the stated mission endpoint.
6. Follow [`REPRODUCE.md`](REPRODUCE.md) for the canonical environment and claim→script→artifact map.

For development:

```bash
uv sync --extra splicebert --extra dev
uv run pytest
```

Run experiments from the repository root. Result-producing scripts must write an immutable result and a
manifest; no new run may silently overwrite the scientific record.
