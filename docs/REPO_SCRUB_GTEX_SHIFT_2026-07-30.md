# Repository scrub: GTEx fine-tuning-shift branch

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

This audit applies [`REPO_SCRUB_PROCEDURE.md`](REPO_SCRUB_PROCEDURE.md) to
`codex/splicebert-gtex-representational-shift`.

## Baseline and safety

- Remote merge target: `origin/master` at `754454a`.
- Audited branch: `f756da6` before scrub edits.
- Merge base: exactly `754454a`.
- Ahead/behind: 100/0; `origin/master` is an ancestor, so the target can fast-forward to the branch.
- The primary `master` checkout contained unrelated staged, unstaged, and untracked work. It was not
  stashed, reset, cleaned, or edited. The audit used the branch's separate clean worktree plus a temporary,
  detached clean checkout of `origin/master`.

## Scope

The merge-base diff contains 253 files: 241 text files and 12 binary artifacts, with 3,991,674 insertions
and 13 deletions.

| Kind | Files / lines | Share of inserted text |
|---|---:|---:|
| `results/` | 119 files; 3,936,092 lines | 98.61% |
| JSON | 54 files; 3,928,158 lines | 98.41% |
| Python | 101 files; 50,971 lines | 1.28% |
| Markdown | 32 files; 4,677 lines | 0.12% |

The branch is therefore mostly immutable machine output. Cleanup targets navigation, guards, and claims;
it does not hand-edit or relocate the result payloads.

## Objective checks

| Check | Clean `origin/master` | GTEx branch before scrub | Difference |
|---|---:|---:|---|
| Manifest verifier | PASS, 808 rows | PASS, 924 rows | no introduced failure |
| Manifest/ledger/hash-pin/overwrite guard set | 55 passed | 55 passed | no introduced failure |
| Local Markdown targets and anchors | 259 checked, 0 failures | 277 checked, 0 failures | no introduced failure |
| Full suite with `splicebert`, `genome`, and `dev` extras | 329 passed, 46 skipped | 513 passed, 45 skipped | no branch regression |

The first baseline full-suite attempt lacked optional genome/SpliceBERT dependencies and is not treated as
a code failure. After installing the repository-declared extras, the clean target passed.

## Claim and provenance contract

- All 43 artifact-like references in the GTEx claim ledger resolve; 41 are unique.
- Every canonical repository-manifest row has a non-empty claim identifier.
- Every GTEx result-manifest claim identifier resolves to GFS1–GFS34.
- The repository manifest has no missing, duplicate, dangling, illegal-status, illegal-polarity, or
  CSV/JSON parity failures.

One branch-specific provenance exception is retained: the legacy seed-2020 smoke stopped inside the old
runner, which wrote no result JSON or execution manifest. Its console outcome was transcribed to
`seed-2020/smoke_sealed_val_v1.failed.json`, and the decision ledger already discloses that limitation.
Creating a modern execution manifest now would falsely claim contemporaneous checkpoint/revision capture.

## Tier exception

Nine J-SAE preregistration or implementation-addendum documents lack inline tier banners. All nine are
hash-pinned as protocol inputs by immutable result manifests. Adding a banner would change the protocol
bytes after execution. They remain unmodified and are listed as verified frozen exceptions in
`tests/test_repository_hygiene.py`; any new untiered document or any change to their pinned bytes fails the
guard.

## Scientific and navigation findings

- The GFS1–GFS34 controlling numbers were checked against their named artifacts; no numerical value
  changed. One live unit label was corrected: native classifier margins are raw logit differences, not
  cross-entropy nats.
- The root and worked-example navigation stopped at the early `L5H6` result. They are updated to point to
  the full context-head, final-MLP, positional-carrier, and sparse-instrument record.
- The 34-row ledger now has a six-finding entry point in
  [`../studies/splicebert_gtex_finetune_shift/FINDINGS.md`](../studies/splicebert_gtex_finetune_shift/FINDINGS.md).
- GFS identifiers and GTEx controlling-artifact references are now part of the ledger-consistency guard.
- Local Markdown target/anchor integrity, tier exceptions, canonical claim-ID coverage, and the legacy
  provenance disclosure are now memorial tests.

## Deliberately unchanged

- No result artifact or preregistration was moved.
- No immutable result, manifest, model archive, or frozen protocol was rewritten.
- No missing execution provenance was fabricated.
- No unrelated work in the primary checkout was touched.

## Final verification

After the cleanup, the focused hygiene/manifest/ledger/hash-pin/overwrite set passes `61/61`, the
repository manifest verifier passes all 924 rows, and the full branch suite passes `519` tests with `45`
data- or environment-gated skips.
