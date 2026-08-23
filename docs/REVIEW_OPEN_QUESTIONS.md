# Review open questions — owner decisions

Items surfaced by the 2026-07-22 remediation program that could not be applied automatically, because
applying them would choose between two recorded numbers, two recorded wordings, or would alter the
meaning of a recorded statement. Nothing here has been changed in the repository. Each entry names the
exact location and the decision required.

---

## Post-audit reverts (2026-07-22) — what the remediation program did NOT keep

A four-lens adversarial audit ran after the remediation fleet and returned **3 BLOCKERs and 6 MAJORs**.
Each was independently re-verified and then reverted. Recorded here so the reverts are part of the
record rather than an invisible correction.

**BLOCKER 1 — frozen records were stamped.** The doc-tiering pass inserted a tier banner into **35
read-only files**, including 27 preregistrations whose own text reads `**Status: FROZEN.**`, plus three
`CLAIMS.md`, `CORRECTIONS.md`, `DECISIONS_A0.md`, and `RESULTS_VERIFICATION.md`. A frozen preregistration
must not be edited after freezing, even additively. **All 35 reverted; `git diff` against those paths is
now empty.** *Root cause: the tiering task's own assignment rules listed these files as T2, contradicting
the global read-only constraint. The task design was at fault, not the executing agent.*
**Decision required:** whether live ledgers (`CLAIMS.md`) should carry a tier banner deliberately.
Frozen preregistrations should not.

**BLOCKER 2 & 3 — figures and provenance of record overwritten.** See §D1 below. Reverted and
re-verified byte-identical to `HEAD`.

**MAJOR — the F1 physical reorganization is reverted in full.** 197 artifacts had been `git mv`-ed into
`development/` subtrees. This was reverted because the moved artifacts are named **by path** inside
frozen preregistrations and read-only ledgers that **cannot be amended** — so the move made the immutable
record under-resolve. Collateral damage confirmed before reverting: **13 runner `--out` defaults** had
been repointed, meaning a protocol governed by a frozen preregistration would write to a path that
protocol does not name; and **185 dangling `results/` path strings** were left across 53 files.

All 197 moves are undone, the 13 runners restored, and the `development/` directories removed.
**Review fix F1 is therefore NOT implemented, deliberately.** Its navigational benefit is delivered
instead by `results/MANIFEST.csv` (fix F2), which classifies every artifact **without moving anything**.
~~**Decision required:** whether to pursue F1 at all. The recommendation from this program is **no** — the
manifest already resolves the opacity F1 was meant to fix, at none of the cost.~~

> **RESOLVED (2026-07-22) — owner decision: F1 is DROPPED.** The physical `canonical/`+`development/`
> reorganization will not be pursued. The `results/MANIFEST.csv` classification (fix F2) delivers the
> navigational benefit F1 targeted without moving any artifact away from the path its frozen
> preregistration names. No further action; F1 is closed, not deferred.

**MAJOR — CI does not actually enforce the new manifest guard (pre-existing defect, not a regression).**
`.github/workflows/tests.yml` installs with `uv sync --locked --extra splicebert --extra dev`, then runs
`uv run pytest` — which re-syncs to the **default** dependency set first, uninstalling the extras.
~~**Decision required:** change the final step to `uv run --extra splicebert --extra dev pytest`. Until
then the F2 guard is not CI-enforced.~~ Note this also means the documented-green baseline
(`261 passed, 46 skipped`) is only reproducible with the extras flags.

> **RESOLVED (2026-07-22) — FIXED.** The `Run full test suite` step in `.github/workflows/tests.yml` now
> runs `uv run --extra splicebert --extra dev pytest`, so CI installs the extras when it runs and the
> manifest guard (`tests/test_manifest.py`) plus the data-gated suite are actually enforced. Verified
> locally: `294 passed, 46 skipped`, manifest guard 33/33.

---

## B3 — nomenclature (2026-07-22)

The superseded phase captions and stale carrier-circuit assets were removed from the active tree;
Git history preserves them. One live nomenclature question remains:

1. **`data/NUCLEOSOME_MAP.md:36` calls the nucleosomal signal "10.67-nt".** Two defects in one phrase:
   the label is retired, and the nucleosomal signal is ≈10.298, not any 10.6x value. Rewording is a
   provenance-document change that touches how a measurement is described. **Decision required:** owner
   ruling on a wavelength-neutral rewording. Path is not owned by B3.

---

## B4 — the DNA-helical range that backs up R30 (raised 2026-08-10)

**Surfaced by, not caused by, the positional-boundary-coherence branch.** Its
`HYPOTHESIS_REGISTER.md` (lines 181–186, sources at 580–584) records a source conflict against canonical
prose and correctly declines to act on it. It is logged here because a T2 register of an unexecuted study
is the wrong place for a challenge to a live canonical sentence.

**The conflict.** `docs/NOMENCLATURE.md` records **10.2–10.5 bp** as "the generic DNA-helical periodicity
range from the literature". The register cites **Rhodes and Klug, *Nature* 292:378–380 (1981)**, whose
random-sequence DNA estimate is **10.6 ± 0.1 bp**. If that estimate belongs in the recorded range, then
**10.630 does not lie outside it**.

**Why it matters, precisely.** R30's refutation of the nucleosomal reading of the ≈10.63 carrier has two
independent legs, and only the second is affected:

1. **Primary — unaffected.** The 0.33-nt gap to the ≈10.298 **mouse ES-cell** dyad measurement is ~7× the
   measured spreads on either side (0.045 nt each). This is the powered refutation and it does not use the
   literature range at all.
2. **Fallback — affected.** The sentence that the refutation "does not rest on cross-species transfer at
   0.33 nt precision" *because* 10.630 falls outside the 10.2–10.5 bp range. That leg is what Rhodes and
   Klug would remove. It appears in `docs/NOMENCLATURE.md` (the ≈10.298 row and standing phrase SP2) and
   in `docs/ROADMAP.md` open problem 6.

**No claim changes today** — C14/C15/R30 rest on leg 1, and the ≈10.63 carrier's corpus attribution
(C39/C40) is independent of both. What is at risk is a fallback argument stated more strongly than its
source may support, on the project's most externally contested claim.

**Decision required:** an owner ruling on the recorded range — confirm 10.2–10.5 with its provenance,
widen it to include the random-sequence estimate, or split the row into sequence-dependent and
random-sequence referents. Then either restate or drop the fallback leg in the two locations above. This
is a choice between two recorded numbers, so it is not applied here. Path is not owned by B3.

---

## C1 — results/ reorganization (2026-07-22)

Source: review fix F1, executed from `results/MANIFEST.csv` and the Phase-A `safe_to_move`
classification. 197 files (`status` ∈ {development, superseded}, `safe_to_move=true`) were `git mv`-ed
into `results/<subdir>/development/`. The following were deliberately **not** applied.

1. **No `canonical/` subdirectory was created.** The review wording is "split into `canonical/` and
   `development/`". Only the `development/` half was created; canonical artifacts stay at their existing
   paths. Moving ~499 canonical files would have invalidated the frozen preregistrations, the
   `REPRODUCE.md` claim→artifact table, the figure pipeline, and the memorial tests — all of which name
   canonical artifacts by their current full path. Leaving them in place kept the blast radius to 13
   reference rewrites instead of several hundred. **Decision required only if** the owner wants the
   symmetric two-directory layout; that is a much larger, separately-gated change.

2. **Provenance paths recorded *inside* moved result artifacts were not rewritten.** 182 path strings
   inside `results/**` JSON record where a run read and wrote its files; 180 of them are a moved
   `.manifest.json` naming its own moved artifact, and 2 are in
   `results/phase-lattice/development/p3_phase_surgery_context_difference_development_n33_20260721.manifest.json`
   naming moved inputs. These are run-time records, not live references. Editing them would overwrite a
   `.manifest.json` (forbidden) and would falsify the recorded statement of where the run actually wrote.
   **Consequence:** a moved artifact's manifest now records its pre-move path. **Decision required:**
   whether the manifests should carry an appended (not substituted) relocation note.

3. **Read-only ledger and preregistration documents name moved artifacts by bare basename.**
   `CLAUDE.md`, `RESULTS_VERIFICATION.md`, `studies/splicebert_partner_selection/{CLAIMS,CORRECTIONS,
   DECISIONS_A0}.md`, `handoff/verification_state.json` and eleven `preregistrations/*.md` mention names
   such as `localization.json`, `ic1_pilot.json`, `preflight_a4_address_geometry.json` and
   `ic2_calibration.json` without a directory. Nothing is broken — none of these files contains a
   directory-qualified path to a moved artifact — but the names now resolve one directory deeper. These
   paths are read-only to this program. **Decision required:** whether to requalify them.

4. **Twelve scripts would, on re-run, write a derived sibling to the pre-move location.** Where a
   script's `--out` default is a canonical artifact that stayed at the root, but a file it derives via
   `out.parent / "…"` was classified development and moved, the two now disagree. Affected:
   `run_a{2,2b,3,5,6,7,8,10,11,12}_*.py` and `run_a9b_payload_semantics_composition.py`, each writing
   `preflight_<exp>.json`; and `run_coupling.py`, writing `per_example.json`. The 13 scripts whose own
   `--out` artifact moved were rewritten and remain self-consistent. **Decision required:** whether the
   preflight artifacts belong beside their canonical result (i.e. should not have moved) or whether the
   scripts should target `development/` explicitly. Changing a preflight's destination affects where a
   future confirmatory run records its gate, so it was not applied automatically.

---

## D1 — figure system (2026-07-22; resolved 2026-08-23)

The former multi-suite figure system and its palette/DPI questions were retired when the active figure
tree was reduced to the four assets actually included by the workshop manuscript. Git history preserves
the prior files and this review's original context. The active PNGs now have repository-relative, hashed
source sidecars, and no styling decision remains open for deleted supplemental figures.
