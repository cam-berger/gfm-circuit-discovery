# Mechanistic Discovery in Genomic Foundation Models

This repository develops a disciplined way to discover **how genomic foundation models (GFMs) implement biological computations**. The goal is not to catalogue interesting activations. It is to move from a model behavior to a causally supported mechanism, with explicit controls and a clear stopping point for interpretation.

**SpliceBERT is the first worked example, not the boundary of the project.** Its donor–acceptor circuit, learned period-3 clock, and unexplained ≈10.63 positional carrier are used to develop and stress-test a methodology that can be transferred to another GFM.

## Current scientific record

| Finding | Evidence reached | Current interpretation |
|---|---|---|
| Donor–acceptor information routing | QK → attention → OV/mediation → behavior | A directed causal circuit: donor information is routed to an acceptor-addressing pathway. |
| Learned period-3 positional clock | Position table → QK → attention → OV/mediation → behavior | A causal positional code that supports the donor–acceptor circuit in coding sequence. |
| Multi-species ≈10.63 carrier | Position table → QK → attention | A real positional attention circuit. Its downstream use and biological origin are unknown. |
| Human-only ≈10.2 candidate | Position table; proposed QK readers fail at attention | The proposed chromatin circuit is closed at the attention gate. |

The evidence levels are intentionally separate. A feature in a position table is not automatically used by attention; an attention pattern is not automatically a causal computation; a causal model mechanism is not automatically a biological mechanism.

## The approach

Every study follows the same broad sequence:

1. **Define the biological computation.** State what information must be integrated, where it comes from, and what model-native observable would demonstrate the behavior.
2. **Validate the assay.** Establish positive and negative controls, sensitivity, resolution, and failure conditions before interpreting a null or a hit.
3. **Localize candidate mechanisms.** Use representations, probes, spectra, attention, attribution, and subspace analyses to generate candidates.
4. **Test necessity.** Ablate the candidate while protecting matched controls and measuring model-native behavior.
5. **Test mediation and direction.** Patch, rescue, or transplant activations; decompose QK and OV roles; and intervene on edges or paths.
6. **Replicate and interpret conservatively.** Repeat across seeds, splits, checkpoints, strands, or models. Stop at the strongest interpretation actually licensed by the evidence.

Attribution is a ranking tool; intervention decides. See [the methodology](docs/METHODOLOGY.md) and [the evidence ladder](docs/EVIDENCE_LADDER.md).

## Repository map

- `docs/` — the transferable research method, evidence rules, transfer guide, and roadmap.
- `studies/splicebert/` — concise entry point to the SpliceBERT worked example.
- `studies/splicebert_partner_selection/` — detailed claim, decision, preregistration, and retraction record retained for provenance.
- `scripts/splicebert/` — active experiment, audit, and figure scripts.
- `src/gfmcd/adapters/` — model access and hook interfaces.
- `src/gfmcd/circuits/` — reusable circuit-analysis primitives.
- `src/gfmcd/splicing/` — SpliceBERT-specific assays and data logic.
- `results/splicebert_coupling/` and `results/phase-lattice/` — current machine-readable results and manifests.
- `figures/splicebert/` — generated figures, captions, provenance, and the current evidence summary.
- `studies/TEMPLATE_REAL_GFM_STUDY/` — starting point for the next model.

Historical GFM-nano, reading-frame, repeat-copy, and retracted v1 partner-selection implementations are preserved in Git history rather than the active research tree.

## Start here

1. Read [Methodology](docs/METHODOLOGY.md).
2. Read [Evidence ladder](docs/EVIDENCE_LADDER.md).
3. Walk through [SpliceBERT as a worked example](studies/splicebert/README.md).
4. Use the [transfer guide](docs/TRANSFER_GUIDE.md) and [study template](studies/TEMPLATE_REAL_GFM_STUDY/README.md) for a new GFM.

For development:

```bash
uv sync --extra splicebert --extra dev
uv run pytest
```

Run experiments from the repository root. Result-producing scripts must write a manifest beside their output and must not overwrite the scientific record silently.
