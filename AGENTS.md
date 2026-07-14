# Repository instructions for research agents

This is a method-first mechanistic-interpretability repository. SpliceBERT is the current worked example; new work should strengthen the transferable methodology or apply it to a real GFM.

## Read before acting

1. `README.md`
2. `docs/METHODOLOGY.md`
3. `docs/EVIDENCE_LADDER.md`
4. `studies/splicebert/README.md`
5. For a detailed claim, `studies/splicebert_partner_selection/CLAIMS.md` and the relevant result manifest.

## Non-negotiable research rules

- Start from a biological computation and a model-native behavior, not from an attractive visualization.
- Run the assay preflight before the target experiment. Positive control, negative control, sensitivity, resolution, and invariants must pass with frozen thresholds.
- Treat probes, attention, attribution, and spectral peaks as candidate generators. They do not establish necessity or mediation.
- Keep position/representation, QK/logit, post-softmax attention, OV/mediation, behavior, and biological interpretation as separate evidence levels.
- Compare interventions with matched-size random or structured nulls. Do not pool null families of different intervention sizes.
- Use exact add-one permutation p-values and correct across the full scanned family when applicable.
- Preserve failed hypotheses and retractions in the claim ledger. Never rewrite a result file to make the narrative cleaner.
- If a control is anomalous, freeze interpretation and diagnose the instrument before proceeding.

## Change discipline

- Active SpliceBERT code belongs under `scripts/splicebert/`, `src/gfmcd/splicing/`, or the reusable adapter/circuit packages.
- A new GFM gets an adapter under `src/gfmcd/adapters/` and a study directory created from `studies/TEMPLATE_REAL_GFM_STUDY/`.
- Every result-producing run writes immutable machine-readable output plus a manifest containing checkpoint, data, parameters, seed, code revision, and command.
- Figures must be generated from recorded artifacts and declare their sources.
- Update the study claim ledger when evidence changes status. A summary document may point to the ledger but may not supersede it silently.
- Do not restore GFM-nano, reading-frame, repeat-copy, or v1 partner-selection code to the active tree. Use Git history when historical context is needed.

## Before finishing

- Check links and imports affected by the change.
- Run the narrowest relevant tests, then `uv run pytest` when feasible.
- State what was run, what was not run, and the strongest claim the new evidence licenses.
