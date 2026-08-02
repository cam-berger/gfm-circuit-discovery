---
name: gfm-repo-discipline
description: Build rules and a defect checker for the gfm-circuit-discovery repository — what may be added, what must never be, and how to change claim ledgers, results, tests, and runners without silently damaging the scientific record. Use when adding or editing anything in this repo (results, claims, corrections, docs, figures, runners, tests), when cleaning up or reorganizing it, when deleting files, before committing, or when asked whether something belongs in the tree.
---

# Repository discipline for gfm-circuit-discovery

> **Tier: T1 — canonical.** Current, authoritative. Read this.

`AGENTS.md` states the research and change rules — read it first; this skill does not repeat it.
What follows is the part that is not obvious: **the failure modes that a green test suite does not
catch**, each drawn from a real defect that survived one here.

## The governing fact

**Every guard in this repo checks paths, IDs, links, and tier banners. None checks meaning.** A
negation can invert, an evidence level can be upgraded, a memorial can be dead for months, a seed can
be unreproducible, and a claim's artifact can be missing — with `pytest` fully green throughout. So:

> A green suite means "nothing structural broke." It never means "the record is still true."

Before claiming a change is safe, run the checker below **and** re-read what you actually changed.

## Run the checker

```bash
python3 .claude/skills/gfm-repo-discipline/scripts/check_repo.py
```

Stdlib only; no project env needed. `FAIL` = fix before committing. `REVIEW` = advisory, judge it.
`--list` shows the checks, `--only a,b` selects, `--json` for machine output, `--repo PATH` for another
checkout. Each check exists because a real defect got through; see [FAILURE_MODES.md](FAILURE_MODES.md)
for the incident behind each one.

## The five rules that matter most

1. **Never reword a claim row to make it shorter.** Restructure mechanically and assert byte-identity;
   a compression pass here inverted a negation and upgraded an evidence level with every test passing.
   Grouping, headings, and indexes are safe. Prose is not.
2. **Report every gate, not the ones that passed.** A row that names its passing calibration and omits
   two failures is worse than no row. If gates failed, the arc licenses nothing — say so.
3. **`[a, b]` means a confidence interval and nothing else.** Spread across seeds or draws is written
   "range" or "SD". And never let a pilot number appear in a confirmatory summary.
4. **Deleting is two edits, not one.** Removing an artifact and removing the memorial that guards it
   both need a correction entry. "Zero importers" is not grounds to delete a module — frozen
   preregistrations cite tooling *as evidence* and cannot be edited to keep up.
5. **A claim must resolve to a committed artifact.** If it cannot, either commit the compact artifact
   plus its manifest, or give a retrievable URL and sha256 in the row. An unverifiable claim in a
   ledger a reviewer is told to trust is a defect, not a weak claim.

## Never add these

Binaries (`.pt`, `.npz`, `.npy`, `.safetensors`) · dated agent-session documents (handoffs, triage
notes, run summaries, review memos) · a result without a sibling `*.manifest.json` · a runner whose
`--out` defaults to a registered artifact · a second study's runners under another study's `scripts/`
tree · retired vocabulary in live prose (`10.67`, `10.58`, a `±0.005` carrier precision, "splice-site
strength") · any edit to a frozen preregistration, including additive ones — several are hash-pinned
and an inserted line breaks the packages that verify them.

## Before you commit

- [ ] `check_repo.py` reports no `FAIL`
- [ ] `uv run --extra splicebert --extra biology --extra dev pytest` (a bare `uv run pytest` drops the
      extras and reports a *failing* suite, not a smaller one)
- [ ] `uv run python scripts/verify_manifest.py`
- [ ] If a claim's wording, number, status word, or scope changed → a numbered `R` entry in
      `CORRECTIONS.md`, and grep the ledgers for other rows that *describe* that claim
- [ ] If you deleted anything → does a frozen prereg, test, or manifest still name it?
- [ ] State what you ran, what you did not, and the strongest claim the evidence licenses

See [FAILURE_MODES.md](FAILURE_MODES.md) for the full ruleset with the incident behind each rule.
