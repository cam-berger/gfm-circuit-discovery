# Failure modes — the incident behind each rule

> **Tier: T2 — provenance.** Supporting record: the incidents behind the rules in [`SKILL.md`](SKILL.md).

Every entry is a defect that actually occurred in this repository and survived a passing test suite.
`R*` names the correction entry in
[`studies/splicebert_partner_selection/CORRECTIONS.md`](../../../studies/splicebert_partner_selection/CORRECTIONS.md).
Read the rule; the incident is there so you can tell when the rule applies to a case it does not name.

## A. Editing the scientific record

**A1 — Claim prose is not compressible.** A 10-agent pass restructured three ledgers with an explicit
brief to preserve every number, hedge, and boundary. Three independent verifiers each found drift:
HCA11's *"this still does **not** imply that any head is general"* became *"**implies no head** is
general"*; C14 was upgraded from a distinctness statement into a refutation verdict with numbers
imported that were not in the row; roughly twenty qualifications were lost, including the measurement
evidence that made C12's "not evaluable as scoped" checkable rather than merely asserted. **All tests
passed the whole time.**
→ Restructure mechanically. Parse rows, re-emit verbatim, refuse to write unless the multiset of row
lines is unchanged, then diff against `git show HEAD:<file>` to confirm. Headings and index tables are
free. If a row genuinely must change, do it as one deliberate hand edit under an `R` number.

**A2 — Selective gate reporting (my own error).** A new `ED1` row reported the calibration gate that
passed (power 1.00) and said nothing about the two that failed — `planted_confound_null` at a 0.285
false-positive rate against a 0.1 target. It also called two `forced: true` memorials "PASS"; an
architecture-guaranteed memorial cannot fail and is not evidence (R7/R21).
→ Enumerate every gate. A failed gate means no interpretation in *either* direction — not a clean null.

**A3 — `[a, b]` that is not a CI (R40).** C41 rendered across-draw min/max in the same bracket notation
every other row uses for a confidence interval. The per-draw dyadic CIs were ~4× wider. The same row
quoted `β = 0.27`, the *pilot* slope, where the confirmatory across-draw mean was 0.2425.
→ Brackets are CIs. Spread is "range" or "SD". Check which split produced every number before it
enters a row.

**A4 — Correcting a claim is not correcting the ledger (R32).** R32 fixed C41's tier word at five sites
and missed a sixth: a clause inside **C38**, a row that *described* C41. The ledger contradicted itself
for six days.
→ After any status-word correction, grep every ledger for rows mentioning that claim ID.

**A5 — An identifier with no defining row.** Four canonical results carried `claim_id: ED1` with no row
anywhere. `test_ledger_consistency.py` never noticed: its pattern matches only `C*`/`HCA*`/`GFS*`.
→ When you invent an ID, define it or register it in `CLAIM_ID_EXCEPTIONS` with a reason.

**A6 — Unverifiable claims (R42).** HCA5 and HCA8 cited artifacts never committed on any branch. HCA20
and HCA22 still cite a bare Drive folder and a repository whose URL appears nowhere here. HCA21 does it
correctly: real URLs plus a bundle sha256.
→ Commit the compact artifact, or give a retrievable URL and hash in the row. Removing a claim is
legitimate only when the row is preserved verbatim in `CORRECTIONS.md` and every citation is repointed
in the same change.

## B. Deleting

**B1 — An artifact and its memorial are two edits (R39).** Four memorial files were deleted whose
artifacts *and* runners had been removed in an earlier commit that was itself never logged. Every test
in them had been silently skipping since.
→ Both deletions need a correction entry. A permanently-skipping test is worse than no test: it
occupies a slot in the coverage count.

**B2 — "Zero importers" is not sufficient (R39).** `src/gfmcd/circuits/attribution.py` had no importers
and a permanently-skipped test, so it was deleted — but the **frozen** B4 preregistration cites it by
path as evidence this repository has ACDC-family primitives, and B4 ran. Restored.
→ Before deleting a module, grep the frozen preregistrations and red-team syntheses for its path. They
cite tooling as evidence and cannot be edited to keep up.

**B3 — A conjunctive skip gate rots silently (R41).** `test_chromatin_refutation_memorials.py` gated on
`GRID.exists() and WAVE.exists() and REAL.exists()`. One artifact was deleted, so **all twelve**
tripwires defending live claim C14 skipped — including the seven needing only the two files that were
present the whole time.
→ Gate per-artifact. Mark genuinely unrunnable tests individually, with the reason, so the gap is
visible instead of hidden inside a broader skip. (`dead_gates` catches this.)

**B4 — Deleting a doc can orphan the only record of a live defect.** A dated sourcing audit was removed
as session exhaust; its §6 was the sole record of the two C41 inaccuracies that later became R40.
→ Before deleting a review document, read it for findings that were never applied.

## C. Runners and reproducibility

**C1 — A canonical producer that cannot run (R41).** `make_fig_phase_data.py`, the canonical producer
of C8's figure data, did an unguarded module-level `from make_fig_wave_fit import build`. That module
had been deleted two cleanups earlier. Regenerating the artifact died at import. No test runs the
figure producers. (`broken_sibling_imports` catches this.)

**C2 — `hash()` is not a seed (R43).** C40's published `n_sig` was drawn from
`np.random.default_rng(hash((h, T)) % 2**31)` where `h` is a head-label string. CPython randomizes
`hash()` of a `str` per process unless `PYTHONHASHSEED` is set — the runner did not set it and the
manifest did not record it, yet both declared `"seed": 0`. Four more instances survive in IC1 pilots
(development-status, so the checker reports them as `REVIEW`).
→ Use `int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)` — the `shash` idiom already in this repo.

**C3 — A checker that silently fails to parse (R43).** `test_overwrite_guard.py` resolved only bare
string-literal `--out` defaults, so it was blind to `default=str(RESULTS_DIR / "name.json")` — the
idiom the two newest studies use. `analyze_b5_context_signal_power.py` therefore defaulted to
HCA23's canonical controlling artifact with no exists-check, no guard, and no inventory entry, in the
one test that exists to catch exactly that. Widening the resolver raised the inventory 56 → 63; none of
the seven was new, all seven were invisible.
→ Test a source-walking checker against the idioms actually in the tree, not the one the first author
used. A guard that under-reports is worse than none, because its inventory reads as complete.

**C4 — One-directional verification.** `verify_manifest.py` checks that every MANIFEST path resolves.
It never checks that an artifact named in a `CLAIMS.md` row *has* a MANIFEST row — which is why HCA20's
missing artifacts passed every check for months. (`claim_artifacts` is the reverse check.)

**C5 — Machine-specific paths.** 21 tracked files hardcode `/Users/cberg/...`. Eight are not
CLI-overridable, including `run_g1_cross_example_transduction.py`, the **C41** runner, which hardcodes
the repository checkout path itself and so will not run from a clone or a worktree. Disclosed in
`REPRODUCE.md` §2.2; not rewritten, because R35 records the standing decision not to change a pinned
runner's defaults.

**C6 — 14 canonical runners have no `argparse`.** `--help` does not print usage; it starts a real run
against compiled-in paths, and some write to registered artifacts. Read the source before invoking any
runner you have not run before.

## D. What accumulates

The 2026-07-31 cleanup removed 484 files and ~887 MB. What it removed, in order of volume:

1. **Force-added binaries** — 847 MB of `.pt`/`.npz` past a `.gitignore` that already excluded them.
   No claim row cited one; the compact JSON twin plus manifest is what makes a claim checkable.
2. **Study programs with no live claim** — five directories, kept in Git history.
3. **Dated agent-session documents** — 17 handoffs, triage notes, execution reports, and an overnight
   run summary addressed to the owner in second person.
4. **A dead code island** — a supervised-probe module retired two corrections earlier, plus its tests.

Four dated documents remain under `docs/` and are **not** exceptions to the rule: each is cited by
executable code or a frozen protocol (`verify_manifest.py` and `tests/test_manifest.py` cite
`REVIEW_2026-07-22.md` §4 as the manifest specification). They are pinned, not preferred.
