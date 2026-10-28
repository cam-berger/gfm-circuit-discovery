# Contributing

A change should improve the transferable method, its implementation, or the evidence for a scoped model
computation. Start with the [methodology](METHODOLOGY.md), the
[evidence ladder](METHODOLOGY.md#evidence-ladder), and the claim and correction ledgers of the study you
are changing; the [study index](../studies/README.md) locates them.

## Code and documentation

Use the environment in [REPRODUCE.md](REPRODUCE.md). Reusable model interfaces go under
`src/gfmcd/adapters/`, shared intervention tools under `src/gfmcd/circuits/`, and experiment entry points
under the owning `scripts/<study>/` directory. The [script index](../scripts/README.md) maps existing
runners to their studies.

For a new GFM, add an adapter and create a study from
[TEMPLATE_REAL_GFM_STUDY](../studies/TEMPLATE_REAL_GFM_STUDY/README.md). Specify the biological
computation, model-native endpoint, controls, and evidence boundaries before inspecting candidate
components. The [transfer guide](TRANSFER_GUIDE.md) gives the adapter and assay requirements.

Keep changes focused and state the reason. Check affected imports and documentation links. Add or update
tests when behavior changes or a scientific defect needs a regression guard; presentation-only edits can
rely on the existing documentation and provenance checks.

## Scientific records

Run assay preflight before a target experiment, with thresholds, matched nulls, multiplicity
corrections, sensitivity, and stopping rules frozen; stop interpretation if a control is anomalous.

Write each new result to a new path, with a manifest recording checkpoint, data, parameters, seed, code
revision, and command, and preserve existing results, including failed hypotheses and retractions. Keep
frozen protocols and hash-bound inputs byte-identical and disclose deviations in the study's decision or
correction ledger. Where provenance requires the original implementation to stay fixed, fix its defects
in a successor runner.

Update the claim ledger when evidence changes status, keeping model behavior, representation, routing,
mediation, and biological interpretation distinct. Consult
[preregistration conformance](PREREG_CONFORMANCE.md) before describing a historical gate as enforced.

Commit compact results, manifests, and audits. Keep model weights, fitted dictionaries, activation
captures, and other regenerable bulk intermediates outside Git. Generate figures from recorded sources
and keep their hashed provenance sidecars; a source change requires reviewing and regenerating the
affected figure (see the [figure index](../figures/README.md)).

Durable findings and decisions go in the study's `CLAIMS.md`, `CORRECTIONS.md`, and `DECISIONS*.md`.
Temporary review notes and agent handoffs do not belong in the research tree. Archived programs remain
in Git history, and the live ledgers identify their retained claims and successor studies.

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

In the change description, say what changed and why, list the checks run and any material skips, and
state whether the evidence supports a new claim. Repository integrity checks and regression tests do not
replace a model experiment or establish a biological mechanism.

## Auditing a branch

Audit a research branch in layers, from the cheapest objective checks to the most expensive scientific
judgments, and do not reorganize evidence until the branch has passed the preceding layer.

### 0. Establish the true baseline

1. Fetch the remote merge target.
2. Identify the remote default branch and the exact merge base.
3. Record ahead/behind counts.
4. Prove whether the merge target is an ancestor of the branch.
5. Inspect every worktree before changing, stashing, or deleting anything.

Never infer scope from a diff against a stale local branch. If the primary checkout contains live work,
leave it untouched and use a separate clean worktree for the target comparison.

### 1. Measure scope

Summarize the merge-base diff by top-level directory, file type, text versus binary files, insertions
and deletions, and largest artifacts. Review result JSON, executable code, tests, and scientific prose
separately: a line count dominated by immutable machine output is not equivalent to the same count in
code or claims.

### 2. Run objective invariants on both trees

Run the same checks on the branch and on a clean checkout of the merge target:

- `scripts/verify_manifest.py`
- local Markdown link and anchor integrity
- the full test suite, including `test_manifest.py`, `test_ledger_consistency.py`,
  `test_hash_pin_ledger.py`, and `test_overwrite_guard.py`
- the repository-hygiene guards: internal link and anchor resolution, hash-pinned protocol documents, and
  canonical-claim coverage (the document tier system was retired on 2026-08-14; see R47)

Compare the failures. A branch defect is a failure introduced relative to the target, not every
historical gap visible in the repository.

### 3. Audit the claim-to-artifact contract

For every live claim, check that:

1. its controlling artifact exists;
2. its evidence level and polarity are legal;
3. its manifest claim identifier resolves to a defining ledger row;
4. every canonical result row carries a claim identifier;
5. every canonical artifact is reachable from the claim and index files.

Do not invent an execution manifest after the fact. If a legacy runner did not capture checkpoint,
command, seed, and revision provenance, disclose the gap in a decision or audit record.

### 4. Audit scientific prose

Only after the mechanical record is sound:

1. reproduce headline numbers from controlling artifacts;
2. check sample counts, intervals, signs, units, and thresholds;
3. weaken wording that exceeds its measured bound;
4. make retractions repair the operative sentence, while preserving the superseded wording nearby;
5. update top-level summaries through the newest claim;
6. keep representation, routing, mediation, behavior, and biological interpretation distinct.

Corrections belong in a dated audit or correction block; immutable results and frozen protocol text stay
byte-identical.

### 5. Improve navigation

Only after correctness, add concise findings and tiered entry points: a start-here page for findings and
current scope, the claim and decision ledgers for detail, and the preregistrations, result indexes,
manifests, and run artifacts for provenance. Prefer links and indexes to moving immutable artifacts.

### Decision rules

1. Immutability outranks tidiness.
2. A move that breaks a reference is worse than no move.
3. Disclose provenance that cannot be reconstructed without fabrication.
4. Verify every audit finding against disk before acting on it.
5. Correct in place while preserving the superseded record.
6. Inspect before deleting, resetting, stashing, or cleaning.
7. Wait for the complete audit synthesis before building on partial output.

Dated applications of this procedure are not kept as documents. Their durable outputs are the
corrections they produced (`studies/splicebert_partner_selection/CORRECTIONS.md`) and the guards that
keep those defects from recurring (`tests/test_repository_hygiene.py`, `tests/test_ledger_consistency.py`,
`tests/test_manifest.py`).
