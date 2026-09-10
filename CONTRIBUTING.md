# Contributing

Changes should improve the transferable method, its implementation, or the evidence for a scoped model
computation. Begin with the [methodology](docs/METHODOLOGY.md), [evidence ladder](docs/EVIDENCE_LADDER.md),
and the relevant study's claim and correction ledgers. The [study index](studies/README.md) locates them.

## Code and documentation

Use the environment in [REPRODUCE.md](REPRODUCE.md). Keep reusable model interfaces under
`src/gfmcd/adapters/`, shared intervention tools under `src/gfmcd/circuits/`, and experiment entry points
under the owning `scripts/<study>/` directory. The [script index](scripts/README.md) maps existing runners
to their studies.

For a new GFM, add an adapter and create a study from
[TEMPLATE_REAL_GFM_STUDY](studies/TEMPLATE_REAL_GFM_STUDY/README.md). Specify the biological computation,
model-native endpoint, controls, and evidence boundaries before inspecting candidate components.
Read the [transfer guide](docs/TRANSFER_GUIDE.md) for adapter and assay requirements.

Prefer focused changes with a clear reason. Check affected imports and documentation links. Add or
update tests when behavior changes or a scientific defect needs a regression guard; presentation-only
edits can use the existing documentation and provenance checks.

## Scientific records

- Run assay preflight before a target experiment. Freeze thresholds, matched nulls, multiplicity
  corrections, sensitivity, and stopping rules; stop interpretation if a control is anomalous.
- Write each new result to a new path with a manifest recording checkpoint, data, parameters, seed,
  code revision, and command. Preserve existing results, including failed hypotheses and retractions.
- Keep frozen protocols and hash-bound inputs byte-identical. Disclose deviations in the study's
  decision or correction ledger; fix implementation defects in successor runners where provenance
  requires the original implementation to remain fixed.
- Update the claim ledger when evidence changes status. Distinguish model behavior, representation,
  routing, mediation, and biological interpretation. Consult
  [preregistration conformance](docs/PREREG_CONFORMANCE.md) before describing a historical gate as enforced.
- Commit compact results, manifests, and audits. Keep model weights, fitted dictionaries, activation
  captures, and other regenerable bulk intermediates outside Git.
- Generate figures from recorded sources and retain their hashed provenance sidecars. A source change
  requires reviewing and regenerating the affected figure; see the [figure index](figures/README.md).

Use the study's `CLAIMS.md`, `CORRECTIONS.md`, and `DECISIONS*.md` for durable findings and decisions.
Temporary review notes and agent handoffs do not belong in the research tree. Archived programs remain
in Git history; their retained claims and successor studies are identified by the live ledgers.

## Validation

From the repository root, run the narrowest relevant tests first. For documentation, registry, or
provenance changes:

```bash
python3 scripts/verify_manifest.py
uv run --locked --extra splicebert --extra biology --extra dev pytest \
  tests/test_repository_hygiene.py tests/test_manifest.py \
  tests/test_ledger_consistency.py tests/test_figure_provenance_freshness.py
```

Then run the full suite when feasible:

```bash
uv run --locked --extra splicebert --extra biology --extra dev pytest
```

In the change description, explain what changed and why, list checks run and material skips, and state
whether the evidence supports a new claim. Repository integrity checks and regression tests do not
replace a model experiment or establish a biological mechanism.
