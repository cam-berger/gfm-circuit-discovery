# Repository instructions for research agents

This is a method-first mechanistic-interpretability repository. SpliceBERT is the current worked example; new work should strengthen the transferable methodology or apply it to a real GFM.

## Read before acting

1. `README.md`
2. `docs/METHODOLOGY.md`
3. `docs/EVIDENCE_LADDER.md`
4. `MECHANISM.md`
5. For a detailed claim, `studies/splicebert_partner_selection/CLAIMS.md` and the relevant result manifest.
6. Before citing any preregistered gate as enforced, `docs/PREREG_CONFORMANCE.md`.

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

- Active SpliceBERT code belongs under `scripts/splicebert/`, `src/gfmcd/splicing/`, or the reusable adapter/circuit packages. Each study owns its own `scripts/<study>/` tree; do not add a second study's runners to `scripts/splicebert/`.
- A new GFM gets an adapter under `src/gfmcd/adapters/` and a study directory created from `studies/TEMPLATE_REAL_GFM_STUDY/`.
- Every result-producing run writes immutable machine-readable output plus a manifest containing checkpoint, data, parameters, seed, code revision, and command.
- Figures must be generated from recorded artifacts and declare their sources.
- New interval, sign-flip, and Holm code uses `gfmcd.inference` (`GroupedSample`, `holm`), naming its weighting. Do not write another bootstrap, and do not clamp a ratio's denominator; pinned and protocol-cited copies stay as they are.
- Update the study claim ledger when evidence changes status. A summary document may point to the ledger but may not supersede it silently.
- Do not restore GFM-nano, reading-frame, repeat-copy, or v1 partner-selection code to the active tree. Nor the study programs removed on 2026-07-31 (DNABERT-2 reconstruction atlas, Caduceus transfer, codon phase lattice, segment attribution). Nor the exon-edge compatibility program archived on 2026-08-04 (`codex/intron-exon-phase` at `fd26cc1b`) — its runners, `src/gfmcd/splicing/phase_*`/`p3_direct_edge_*` modules, tests, and cohorts stay archived; only C43/C44, their twelve artifacts, and their two frozen protocols were carried, and that arc's own `CLAIMS.md`/`RESULTS.md` prose is superseded by R45 and must not be quoted from history. R49 later added C43's raw cell and native-source rows and a model-free reproduction script under `scripts/splicebert_exon_edge_compatibility/`. That script *transcribes* the archived arithmetic and restores no archived module. `fd26cc1b` exists only in the owner's local clone and was never pushed. Use Git history when historical context is needed. The sanctioned way to work on that question is the successor study, `studies/splicebert_positional_boundary_coherence/` with its own `scripts/` tree and `src/gfmcd/splicing/pbc_*` modules — it inherits blacklists, geometry, and sign-free variance for sizing, never the archived code and never C43/C44 signs or magnitudes. Importing from the archived program into it is forbidden, and its failed C44 authorization path may not be reopened or relaxed.
- Do not commit fitted dictionaries, activation captures, or other large binaries. The compact result JSON, its manifest, and its audit are what make a claim checkable; a `.pt` or `.npz` is not evidence and `.gitignore` already excludes them.
- Do not write inter-agent handoff, triage, or dated review documents into the tree. Durable outputs go to the claim ledger, `CORRECTIONS.md`, the relevant `DECISIONS*.md`, or a test that prevents the defect from recurring. Four dated documents remain under `docs/` and are **not** exceptions to this rule — each is cited by executable code or by a frozen protocol (`verify_manifest.py` and `tests/test_manifest.py` cite `REVIEW_2026-07-22.md` §4 as the manifest specification; frozen preregistrations cite the two B2/B3 notes; `CORRECTIONS.md` and `results/origin-siblings/RESULTS_INDEX.md` cite the carrier-origin synthesis). They are pinned, not retained by preference.

## Before finishing

- Check links and imports affected by the change.
- Run the narrowest relevant tests, then `uv run pytest` when feasible.
- State what was run, what was not run, and the strongest claim the new evidence licenses.
