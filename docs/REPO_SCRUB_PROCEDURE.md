# Repository scrub procedure

This procedure audits a research branch from the cheapest objective checks to the most expensive
scientific judgments. Do not reorganize evidence until the branch has passed the preceding layer.

## 0. Establish the true baseline

1. Fetch the remote merge target.
2. identify the remote default branch and exact merge base;
3. record ahead/behind counts;
4. prove whether the merge target is an ancestor of the branch;
5. inspect every worktree before changing, stashing, or deleting anything.

Never infer scope from a diff against a stale local branch. If the primary checkout contains live work,
leave it untouched and use a separate clean worktree for the target comparison.

## 1. Measure scope

Summarize the merge-base diff by:

- top-level directory;
- file type;
- text versus binary files;
- insertions and deletions;
- largest artifacts.

Treat result JSON, executable code, tests, and scientific prose as separate review populations. A large
line count dominated by immutable machine output is not equivalent to the same line count in code or
claims.

## 2. Run objective invariants on both trees

Run the same checks on the branch and a clean checkout of the merge target:

- `scripts/verify_manifest.py`;
- local Markdown link and anchor integrity;
- the full test suite;
- `test_manifest.py`;
- `test_ledger_consistency.py`;
- `test_hash_pin_ledger.py`;
- `test_overwrite_guard.py`;
- repository-hygiene guards: internal link/anchor resolution, hash-pinned protocol documents, and
  canonical-claim coverage. (The document tier system was retired on 2026-08-14; see R47.)

Compare failures. A branch defect is a failure introduced relative to the target, not every historical
gap visible in the repository.

## 3. Audit the claim-to-artifact contract

For every live claim:

1. its controlling artifact must exist;
2. its evidence level and polarity must be legal;
3. its manifest claim identifier must resolve to a defining ledger row;
4. every canonical result row must carry a claim identifier;
5. every canonical artifact must be reachable from the claim/index surfaces.

Do not invent an execution manifest after the fact. If a legacy runner did not capture checkpoint,
command, seed, and revision provenance, disclose the gap in a decision or audit record.

## 4. Audit scientific prose

Only after the mechanical record is sound:

1. reproduce headline numbers from controlling artifacts;
2. check sample counts, intervals, signs, units, and thresholds;
3. weaken wording that exceeds its measured bound;
4. make retractions repair the operative sentence, while preserving the superseded wording nearby;
5. update top-level summaries through the newest claim;
6. keep representation, routing, mediation, behavior, and biological interpretation distinct.

Corrections belong in a dated audit or correction block. Immutable results and frozen protocol text stay
byte-identical.

## 5. Improve navigation

Add concise findings and tiered entry points only after correctness:

- **start here:** findings and current scope;
- **detail:** claim and decision ledgers;
- **provenance:** preregistrations, result indexes, manifests, and run artifacts.

Prefer links and indexes over moving immutable artifacts. A tidier directory is not worth broken hashes or
rewritten provenance.

## Decision rules

1. **Immutability outranks tidiness.**
2. **A move that breaks a reference is worse than no move.**
3. **Disclose provenance that cannot honestly be reconstructed.**
4. **Verify every audit finding against disk before acting on it.**
5. **Correct in place while preserving the superseded record.**
6. **Inspect before deleting, resetting, stashing, or cleaning.**
7. **Wait for the complete audit synthesis before building on partial output.**

Individual dated applications of this procedure are not retained as documents; the durable outputs are
the corrections they produced (`studies/splicebert_partner_selection/CORRECTIONS.md`) and the guards
that keep them from recurring (`tests/test_repository_hygiene.py`, `tests/test_ledger_consistency.py`,
`tests/test_manifest.py`).
