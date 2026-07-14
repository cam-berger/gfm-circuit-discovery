# Active experiment scripts

All active research scripts are under `scripts/splicebert/`. They support the current
donor–acceptor circuit, period-3 clock, and ≈10.63 carrier record.

The scripts fall into six broad families:

1. behavioral assays and biological counterfactuals;
2. representation, position-table, carrier, and subspace analyses;
3. attention/QK/OV tracing and candidate screens;
4. ablation, patching, rescue, transplant, and mediation experiments;
5. preflight, power, calibration, provenance, and audit utilities;
6. publication-data and figure generation.

Use filenames and imports to locate a workflow:

```bash
rg --files scripts/splicebert
rg "coupling|carrier|postsoftmax|mediation" scripts/splicebert
```

## Execution contract

- Run from the repository root in the project environment.
- Read `docs/METHODOLOGY.md`, `docs/EVIDENCE_LADDER.md`, and the relevant preregistration first.
- Run preflight and controls before the target experiment.
- Write machine-readable output under the relevant `results/` directory.
- Record checkpoint, data, parameters, seed, code revision, and command in a manifest or in the
  result's self-contained provenance block.
- Never hand-edit a result to match a narrative; supersede it and update the claim ledger.
- Generate publication figures from recorded artifacts.

## Retractions

Do not keep a runnable tombstone for a failed design in the active script directory. Preserve the
reason for failure in the claim ledger and memorial tests, and use Git history when the old
implementation is needed for audit.

The detailed current record is indexed from `studies/splicebert/README.md`,
`figures/splicebert/README.md`, and `results/phase-lattice/RESULTS_INDEX.md`.
