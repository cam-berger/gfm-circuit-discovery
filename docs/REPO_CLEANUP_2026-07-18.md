# Repo cleanup & consolidation plan — 2026-07-18

> **Tier: T3 — historical.** Superseded or point-in-time; retained for the record. Not current guidance.

**Status: Phases 1–4 complete; Phase 5 pending merge (updated 2026-07-20).** Phases 1–3 pruned and
archived old branches and removed `colab_results/`. Phase 4 is complete: preregistrations and red-team
records are indexed in subdirectories; the live claim ledger is compact; the full historical ledger is
preserved as `CORRECTIONS.md`; and repository-level `MECHANISM.md`/`REPRODUCE.md` provide the front door.
**Phase 5** remains user-owned: `donor-graph-address-payload` is pushed and 43 commits ahead of `master`;
an MR is pending.

## The one-sentence diagnosis

The repo is **not** chaotic — it is *one crown-jewel branch surrounded by fully-merged dead wood*, plus a
doc sprawl in a single study dir, plus one older research thread (Track-2) that never landed. "Behind/ahead"
reduces to a single fact: **the entire donor-circuit story lives on `donor-graph-address-payload` and has
never been merged to `master`** (master is ~30 commits behind; 10 of those aren't pushed to origin yet).

## Guardrails (apply to every phase)

- **Concurrency:** a live parallel session is actively working `studies/caduceus_partner_transfer/`
  (untracked `colab_package/` WIP, files written 12:35 today; branch tip committed 26 min ago). I will
  **never `git add -A`**, will stage only my own files by explicit path, will **not touch `colab_package/`
  or the caduceus dir**, and will not edit shared ledgers (`CLAIMS.md`, `DECISIONS_A0.md`, `CLAUDE.md`)
  while that session is active. (See memory: `concurrent-sessions-shared-repo`.)
- **Hold pushes** (your call, Q3). All destructive *remote* ops are deferred to a push-go; local-only until then.
- **Never silently overwrite the scientific record** (project rule 7): docs get *moved to `archive/`*, not
  deleted — except regenerable cruft (logs, planner-state JSON), which is removable.
- **Verify-before-move:** every file marked "verify" below is read and confirmed superseded at execution
  time; if it turns out load-bearing, it stays.

---

## Phase 1 — Local branch pruning  (SAFE, zero-loss, local-only)

All 5 are **fully merged into `master`** (verified `git branch --merged master`); `master` is an ancestor of
the current HEAD, so their content is preserved. Safe `-d` (refuses if not actually merged — a built-in check).

```sh
git branch -d splicebert-acceptor-address
git branch -d splicebert-coupling
git branch -d splicebert-donor-graph
git branch -d splicebert-v2-partner-binding
git branch -d track2-rigor-hardening
```

**Result:** local branch list drops from 8 → 3 (`donor-graph-address-payload`, `master`, + the Track-2 orphan
handled in Phase 2). No files touched, no push, invisible to the parallel session.

## Phase 2 — Track-2 orphan: archive-tag, then delete  (your Q2 choice)

`cleanup/scientific-claims-and-splicebert` = 12 commits, **never merged, never pushed** — the older Track-2
"repeat-copy / counter-vs-lookup" thread (not the donor story). Tag preserves all 12 commits permanently.

```sh
git tag archive/track2-repeat-copy cleanup/scientific-claims-and-splicebert
git branch -D cleanup/scientific-claims-and-splicebert     # -D required: intentionally unmerged
```

**⚠️ Durability caveat:** those 12 commits exist **only in this clone** (never pushed anywhere). After the
branch is deleted, the *only* copy is the local tag `archive/track2-repeat-copy`. To make the archive durable
it must eventually be pushed:

```sh
git push origin archive/track2-repeat-copy      # additive, zero impact on others — see recommendation
```

**Recommendation:** pushing a single archive *tag* is the safest possible push (purely additive; touches no
branch; invisible to the parallel session). I suggest allowing this one push for durability even under the
hold-pushes cadence. If you'd rather stay strictly local, we create the tag now and accept single-clone
fragility until the next push-go.

## Phase 3 — Remote branch cleanup  (DEFERRED — needs push-go)

All 7 remote branches below are **verified fully contained in `master`** (`0` commits not in master), so
deletion is zero-loss. Ready for you to run, or for me to run on a push-go:

```sh
git push origin --delete cleanup/canonical-splicebert-record
git push origin --delete cleanup/method-first-realignment
git push origin --delete splicebert-partner-selection
git push origin --delete splicebert-coupling
git push origin --delete splicebert-donor-graph
git push origin --delete splicebert-v2-partner-binding
git push origin --delete track2-rigor-hardening
git push origin archive/track2-repeat-copy       # Phase-2 durability (if approved)
git fetch --prune                                 # clean up local tracking refs afterward
```

**Result:** remote branch list drops to `master` + `donor-graph-address-payload` (+ the archive tag).

## Phase 4 — Doc / file condensation  (**COMPLETE 2026-07-20**)

High collision risk with the live session, so this runs *only* once the transfer thread parks and you approve.
The big target is `studies/splicebert_partner_selection/` (**41 markdown files**).

**Proposed target structure** (moves, not deletes):

```
studies/splicebert_partner_selection/
  CLAIMS.md  DECISIONS_A0.md  DONOR_CIRCUIT_ACCOUNT.md  FINDINGS.md      # canonical narrative — stay at top
  METHODS.md  README.md  RUNBOOK.md  REPRODUCIBILITY.md
  circuit.svg  carrier-wave-interaction-paper.png
  preregistrations/          # ~30 PREREGISTRATION_*.md move here + a README index w/ status
  redteam/                   # REDTEAM_SYNTHESIS_P3_4.md, _P3_4lens.md, P3_NIGHT_RUN_SUMMARY.md
  archive/                   # superseded docs (see "verify" list)
```

| item | disposition | note |
|---|---|---|
| `CLAIMS.md`, `DECISIONS_A0.md`, `DONOR_CIRCUIT_ACCOUNT.md`, `FINDINGS.md`, `METHODS.md`, `README.md`, `RUNBOOK.md`, `REPRODUCIBILITY.md` | **keep top-level** | canonical narrative + ledgers |
| ~30 `PREREGISTRATION_*.md` (A0–A12, donor/acceptor families, PERIOD3, SEALED, IC2A, IC1, IC2, P3, P3_4) | → `preregistrations/` | add `preregistrations/README.md` index with per-prereg status (complete / parked / deferred) |
| `REDTEAM_SYNTHESIS_P3_4.md`, `REDTEAM_SYNTHESIS_P3_4lens.md`, `P3_NIGHT_RUN_SUMMARY.md` | → `redteam/` | distinct records, both keepers |
| `DECISIONS.md`, `DECISIONS_ACCEPTOR_ADDRESS.md` | **adjudicated** | concise `DECISIONS.md` remains active; detailed acceptor-address log moved to `archive/`. |
| `RESULTS_TEMPLATE.md` | → `studies/TEMPLATE_REAL_GFM_STUDY/` | it's a template, wrong home |
| `return_to_planner.json` | **removed** | stale Track-2 behavioral-gate planner state; superseded by the donor graph and current ledgers. |

**Elsewhere at root:**

| item | disposition | note |
|---|---|---|
| `figures/splicebert/*.md` (11) | **keep** | legitimate figure captions + EVIDENCE.md — *not* sprawl |
| `docs/*.md` (EVIDENCE_LADDER, METHODOLOGY, README, ROADMAP, TRANSFER_GUIDE) | **keep** | canonical method docs (ROADMAP may need a content refresh) |
| `handoff/` (deck_outline.txt, verification_state.json) | **verify** → `archive/` or remove | stale handoff snapshot |
| `logs/` (7 tracked) | **verify** → remove from git | logs generally shouldn't be tracked |
| `colab_results/` (**104 tracked**) | **verify — likely KEEP** | Track-2 lineage BUT probably the foundational sealed-behavior evidence (additive-vs-interaction frozen). Do **not** archive without confirming it's not cited by CLAIMS/results. If keepable-but-heavy, add an index rather than move. |
| untracked smoke outputs (`results/**/ic2_*_smoke.json`, `results/splicebert_partner_selection/`, `results/reading-frame/`) | **leave** | may be the parallel session's; not mine to touch |

## Phase 5 — Land the science on master  (STRATEGIC — separate decision)

The real fix for "master is 30 behind." `master` is an ancestor of `donor-graph-address-payload`, so this can
be a **clean fast-forward** — but it's entangled with (a) 10 unpushed commits of mixed authorship (mine + the
parallel session's IC2/Caduceus) and (b) the hold-pushes cadence. **Not part of the cleanup; flagged for a
deliberate call** once the parallel thread parks and you're ready to push. Options when you are: fast-forward
`master` locally then push, or open a PR from `donor-graph-address-payload`.

---

## Execution order & approvals

1. **Phase 1** (local branch prune) — ready on your "go".
2. **Phase 2** (tag + delete orphan) — ready on your "go"; + decide the archive-tag push (recommend allow).
3. **Phase 3** (remote prune) — held for push-go.
4. **Phase 4** (doc condensation) — complete.
5. **Phase 5** (land on master) — branch pushed; user will create and merge the MR.

**What stays untouched throughout:** `colab_package/` (live WIP), the caduceus dir, shared ledgers while the
parallel session is active, and anything I didn't author without verifying it first.
