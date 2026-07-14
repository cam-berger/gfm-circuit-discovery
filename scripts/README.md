# Active experiment scripts

All active research scripts are under `scripts/splicebert/`. Root-level GFM-nano, reading-frame, repeat-copy, and v1 partner-selection drivers were historical experiments and are no longer part of the active tree.

The SpliceBERT scripts fall into six broad families:

1. behavioral assays and biological counterfactuals;
2. representation, position-table, carrier, and subspace analyses;
3. attention/QK/OV tracing and candidate screens;
4. ablation, patching, rescue, transplant, and mediation experiments;
5. preflight, power, calibration, provenance, and audit utilities;
6. figure-data and figure generation.

Use filenames and imports to locate a workflow rather than assuming a single monolithic pipeline:

```bash
rg --files scripts/splicebert
rg "coupling|carrier|postsoftmax|mediation" scripts/splicebert
```

## Execution contract

- Run from the repository root in the project environment.
- Read `docs/METHODOLOGY.md`, `docs/EVIDENCE_LADDER.md`, and the relevant study preregistration first.
- Run preflight and controls before the target experiment.
- Write machine-readable output under the relevant `results/` directory.
- Write a manifest beside every result with checkpoint, data, parameters, seed, code revision, and command.
- Never hand-edit a result to match a narrative; supersede it and update the claim ledger.
- Generate figures from recorded artifacts and retain provenance.

The detailed current record is indexed from `studies/splicebert/README.md` and `figures/splicebert/README.md`.
