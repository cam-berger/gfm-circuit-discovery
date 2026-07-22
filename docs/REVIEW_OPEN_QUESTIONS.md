# Review open questions — owner decisions

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

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
**Decision required:** whether to pursue F1 at all. The recommendation from this program is **no** — the
manifest already resolves the opacity F1 was meant to fix, at none of the cost.

**MAJOR — CI does not actually enforce the new manifest guard (pre-existing defect, not a regression).**
`.github/workflows/tests.yml` installs with `uv sync --locked --extra splicebert --extra dev`, then runs
`uv run pytest` — which re-syncs to the **default** dependency set first, uninstalling the extras.
Verified unchanged by this program (`git diff` on `.github/` is empty). **Decision required:** change the
final step to `uv run --extra splicebert --extra dev pytest`. Until then the F2 guard is not CI-enforced.
Note this also means the documented-green baseline (`261 passed, 46 skipped`) is only reproducible with
the extras flags.

---

## B3 — nomenclature (2026-07-22)

Source: the read-only nomenclature audit (task A2) plus review §2.1/§2.3/§2.4 and fix F3. Applied
changes were confined to `docs/NOMENCLATURE.md`, `README.md`, and `MECHANISM.md`. The following were
escalated instead.

1. **`figures/splicebert/fig_two_carriers.CAPTION.md` uses the retired "10.67" label in live prose.**
   Line 69 asserts `**The carrier is ~10.67, and "10.69" is a Fourier grid, not a measurement.**`, line
   58 carries an evidence-status row `| SpliceBERT selectively amplifies a **sharp** ~10.67 tone |
   **established** |`, and line 65 says `at 10.67 it is *not even a local maximum*`. This directly
   contradicts the retirement recorded at `results/phase-lattice/RESULTS_INDEX.md:102`. The file also
   cites the superseded D22 numbering, is absent from the publication table in
   `figures/splicebert/README.md`, and carries no historical/superseded stamp. **Decision required:**
   stamp the file T3-historical (phase E1) versus correct the caption text. Correcting the text would
   restate a superseded caption's conclusions and is therefore not an automated edit. Owner of the path
   is D1/E1, not B3.

2. **`data/NUCLEOSOME_MAP.md:36` calls the nucleosomal signal "10.67-nt".** Two defects in one phrase:
   the label is retired, and the nucleosomal signal is ≈10.298, not any 10.6x value. Rewording is a
   provenance-document change that touches how a measurement is described. **Decision required:** owner
   ruling on a wavelength-neutral rewording. Path is not owned by B3.

3. **`figures/splicebert/fig4_carrier_circuit.CAPTION.md:64` labels the human-only component ≈10.17
   while every other document says ≈10.2.** Both are recorded values (≈10.17 is the sub-bin estimate of
   the component that other documents name ≈10.2). `docs/NOMENCLATURE.md` records the term as the
   "human-only ≈10.2 component" because that is the ledger wording (C13). **Decision required:** which
   label is canonical in prose. Do not change either recorded number without that ruling.

4. **`figures/splicebert/EVIDENCE.md:49` says the human-only chromatin-circuit claim is "closed at
   QK"; `CLAIMS.md` C13 and four other documents say "closed at the attention gate".** These are
   different rungs of the evidence ladder, so this is not a free wording harmonisation. **Decision
   required:** confirm the intended rung, then harmonise. Path is not owned by B3.

5. **`studies/splicebert_partner_selection/CLAIMS.md:33` (C14) states the 10.298-vs-≈10.63 distinctness
   without the "0.33 nt apart, powered refutation" qualifier that review §2.3 requires in the same
   sentence.** `CLAIMS.md` is read-only and the ledger's compactness is deliberate. **Resolution taken:**
   the qualifier now lives in `docs/NOMENCLATURE.md` under the `10.298` entry, with `RESULTS_VERIFICATION.md`
   and `fig4_carrier_circuit.CAPTION.md` as the authoritative detail, rather than being inserted into the
   ledger. **Decision required only if** the owner would prefer the qualifier in the ledger itself.

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
   `results/phase-lattice/p3_phase_surgery_context_difference_development_n33_20260721.manifest.json`
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

## D1 — figure system (2026-07-22)

Source: review §8.

> **CORRECTION (2026-07-22, post-audit).** This section originally asserted that "the regenerated figures
> are byte-identical to the ones they replace." **That statement was false.** Measurement showed
> `fig1_coupling.pdf`, `fig2_mechanism.pdf`, `fig3_circuit.pdf` and
> `fig5_p3_phase_surgery_confirmation.pdf` all differed from their committed bytes, and the four
> `*.provenance.json` sidecars had been overwritten — replacing the original generation environment
> (`git_commit: null`, original project path) with this worktree's commit and an ephemeral
> `.claude/worktrees/` path, destroying the provenance record.
>
> **All four figures, their provenance sidecars, their captions, and every modified `make_fig*.py` have
> been reverted to their committed state and re-verified byte-identical to `HEAD`.** The style-module
> refactor is therefore **not applied** to the figures of record. What remains from D1 is purely
> additive and new: `figures/splicebert/figstyle.py`, plus three new figures
> (`fig_carrier_heatmap`, `fig_evidence_ladder`, `fig_generator_graph`) and their generators.
>
> **Decision required:** applying the style refactor to the four existing figures regenerates committed
> figures of record and rewrites their provenance. That is an owner decision, not an automated one.

Three further things could not be settled without a styling decision by the owner.

1. **Three palettes are in use, and unifying them would restyle published panels.** The CVD-validated
   SVG palette (`scripts/splicebert/figstyle.py`) uses `#0B6FB8` for the period-3 blue; the Okabe-Ito
   matplotlib tokens (`src/gfmcd/figures/style.py`) use `#0072B2` for "signal"; the confirmation figure
   uses `#3569a8`. All three are now named in one module (`figures/splicebert/figstyle.py`) instead of
   three, so the disagreement is visible rather than hidden, but none of the values was changed.
   **Decision required:** whether to re-colour `fig1_coupling`, `fig2_mechanism`, `fig3_circuit` and
   `fig5_p3_phase_surgery_confirmation` onto the validated palette. That is a restyle, not a refactor,
   and it changes committed figure files.

2. **DPI is not uniform.** The project rcParams save rasters at 300 dpi; the confirmation figure was
   authored at 240 and is still saved at 240 (`DPI_CONFIRMATION_PANEL`) so its PNG is unchanged.
   **Decision required:** whether to standardize on 300.

3. **The validated palette still physically lives under `scripts/splicebert/`.** `figures/splicebert/
   figstyle.py` re-exports it rather than copying it, so there is exactly one definition of the
   computed colours; the cost is that the study's style module depends on a file in the script tree.
   Moving `scripts/splicebert/figstyle.py` into `figures/splicebert/` and collapsing the re-export
   would be tidier, but that path was outside this task's ownership. **Decision required:** whether to
   move it.

Also recorded, not a decision: **Graphviz is not installed in this environment.**
`make_fig_generator_graph.py` writes the data-derived DOT source unconditionally and renders PDF/PNG
only when `dot` is on `PATH`. It reports the missing dependency and stops rather than substituting a
hand-placed drawing, per review §8.
