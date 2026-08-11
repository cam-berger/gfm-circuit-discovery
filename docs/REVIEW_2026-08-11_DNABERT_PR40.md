# Review 2026-08-11 — DNABERT-2 promoter arc (PR #40) against repository fundamentals

> **Tier: T2 — provenance.** The integration review that gave the DNABERT-2 arc its ledgers. Not a claim
> record: claims are in [`studies/dnabert_promoter/CLAIMS.md`](../studies/dnabert_promoter/CLAIMS.md) and
> withdrawals in [`CORRECTIONS.md`](../studies/dnabert_promoter/CORRECTIONS.md).

**Subject.** PR #40, `padmaja/dnabert-clean-pipeline` → `master`: 185 files, +128,263 lines. Frozen
DNABERT-2 promoter analysis — adapter, dataset and perturbation modules, 35 runners, ~109 result
artifacts, 24 figures, 3 test modules.

**Verdict.** The *science* is largely sound and its numbers are accurate: every metric quoted in the arc's
prose reconciles to its committed artifact to all printed digits, the pinned probe hash matches, identity
interventions are exactly zero throughout, and the validation→test discipline is real. What failed review
was **what the numbers were said to mean**, and the arc's integration into the repository record: it
arrived with five 0-byte ledger files, no `MANIFEST` rows, and prose that promoted a null into a
"confirmed compact" mechanism. Twenty-two findings: 5 blockers and 11 major issues fixed in this pass, and
6 recorded as gated work that needs a re-run or an owner decision.

The arc's most valuable result was one it had argued *against* itself. Its own faithfulness curve refutes
the compactness its prose claimed, and its own restoration verdict refutes the transferability its
"confirmed interpretation" asserted. Both are now the ledgered claims (`DBP7`, `DBP8`).

## Blockers — fixed

| # | Finding | Evidence | Resolution |
|---|---|---|---|
| B1 | **The repository's own guard test failed on the branch.** `tests/test_manifest.py` had 2 failing tests: 132 tracked result files had no `results/MANIFEST.csv` row. This is what made the PR `mergeable_state: unstable`. | `test_repository_manifest_is_consistent`, `test_repository_manifest_indexes_every_tracked_result_file` | 109 rows added to `MANIFEST.csv`/`.json` with `status`, `evidence_level`, `polarity`, producing script, sibling manifest where one exists, and figure links. `verify_manifest.py` clean; **40/40 tests pass**. |
| B2 | **No claims ledger.** `studies/dnabert_promoter_mechanism/{CLAIMS,PREREGISTRATION,DECISIONS,README,RESULTS}.md` were all **0 bytes**, while 277 lines of substantive prose sat in a root-level `dnabert_readme.md`. The study had live claims and no point of contact — invariant 3. | `wc -c` = 0 on all five | Study consolidated into `studies/dnabert_promoter/`. `CLAIMS.md` written with `DBP1`–`DBP10`, `CORRECTIONS.md` with `DBPR1`–`DBPR8`, `DECISIONS.md` with D1–D11 plus a departures table. Root readme became the study `README.md`. Empty placeholders removed. |
| B3 | **Undisclosed split contamination.** The arc's *own* `leakage_report.json` records 259 exact-duplicate groups, **104 spanning splits**, and 64 groups where one 70-nt sequence carries both labels (20 cross-split). No de-duplication was applied and **leakage is mentioned nowhere in the prose**, while `0.810` / `0.892` were presented as clean held-out performance. | `results/dnabert_promoter/shared/leakage_report.json` | `DBP2` records the defect as a first-class bounding claim; `DBP3`/`DBP4` restated as contaminated-split upper bounds. `DBPR1`. De-duplicated re-run is the gating next step in the roadmap. |
| B4 | **Two readout endpoints intermingled.** 42 artifacts under `causal/layer3_*` set `SELECTED_LAYER = 3`, read `hidden_states[3]`, and score with `all_layer_probes/layer_3_probe.joblib` (sha256 `2337441e…`) — not the canonical `selected_layer_2_probe.joblib` (`bd0d9e14…`). They sat flat beside the Layer-2 causal results with nothing marking the difference, and no prose cited them. | `selected_layer: 3` in 20 summaries; `grep SELECTED_LAYER scripts/dnabert_promoter/*.py` | Classified `superseded` (37 `MANIFEST` rows under `claim_id` `DBPR7`, registered as a non-claim ID in `results/RESULTS_INDEX.md`). Hazard documented at the top of the family index, in `DECISIONS.md` D6, and in the "Deliberately not claimed" section of the ledger. Retained, not deleted — results are immutable. |
| B5 | **A null promoted to a confirmed mechanism.** "The compact confirmed component set is `Layer-0 MLP + L0H8`" — while the PR's own faithfulness curve gives that set standalone faithfulness `0.194`, no knee, and `1.000` only at size 39 with all three early MLPs. Conditional delivery (`0.882`) had been read as faithfulness. | `frozen_claim_tests/faithfulness_curve/*_curve.csv` | `DBP7` written as a bound/refutation with `polarity: bounded`; `DBPR4` retires *compact*, *sparse*, and bare *the circuit* for this study. Registry prose corrected in place. Nomenclature entry added. |

## Major — fixed

| # | Finding | Resolution |
|---|---|---|
| M1 | **The registry contradicted itself.** "No sequence-specific delivery claim is currently licensed" and "not donor-specific" (restoration verdict) versus "carries a **transferable** promoter-class representation" / "`L0H8` substantially improves **delivery**" (confirmed interpretation) — in the same file. Matched-vs-same-class is null (`0.036`, p `0.252`; `0.053`, p `0.091`). | The restoration verdict was right and is annotated as vindicated. `DBP8` states class-level transfer; `DBPR5` withdraws "transferable", "delivery", "sequence-sensitive". |
| M2 | **An unsupported p-value, quoted twice.** "Holm-adjusted p = 2.8539e-22" for both class contrasts; the artifact records `p_value: 0.0` for both, and `2.8539e-22` appears nowhere. Holm was applied across only 2 of the 6 committed comparisons without saying so. | Registry corrected to the artifact's own values with the adjustment family stated. `DBPR5`. |
| M3 | **Metrics with no controlling artifact.** An eight-metric fine-tuning table plus "pipeline implemented; full runs ongoing" — **no DNABERT fine-tuning code, config, checkpoint, or result exists in the repository.** | Withdrawn from live prose (`DBPR6`); recorded under "Deliberately not claimed" and deferred in the roadmap. |
| M4 | **Stale status lines contradicting shipped artifacts.** Block 5 "not run as a controlled assay" (a controlled TATA assay is described later in the same file); block 8 "not run against the corrected endpoint" (circuit-accounting and faithfulness-curve artifacts ship at `selected_layer: 2`). Block 6 was accurate. | Both corrected in place with what actually ran and what did not; block 6 left alone. |
| M5 | **"Preregistered" with no preregistration.** The confirmatory donor-matrix section claimed "preregistered component conditions"; no protocol document exists anywhere in the arc. | Wording withdrawn (`DBPR8`); a "Preregistration status" section added to the ledger; `docs/PREREG_CONFORMANCE.md` records the family as having no protocol to audit, distinguishing it from BG1/BG2's frozen-but-unrun case. |
| M6 | **Figures 4–8 plot the superseded suite.** 10 of 24 figure files are built from `causal/layer3_*`. `figure8_test_donor_matrix` shows `PRIMARY_SIX_HEADS` recovery `0.258` at Layer 3, while the registry's "Test recovery fractions" table beside it reports `0.879` for `L0 MLP + L0H8` at Layer 2 — different experiments, similar names. | Each figure has a `.CAPTION.md` naming its controlling artifact and status; figures 4–8 are marked superseded and `figure8`'s caption states the mismatch explicitly. `figures/dnabert_promoter/README.md` tabulates it. |
| M7 | **Figures committed against `.gitignore`.** 24 files force-added under `results/dnabert_promoter/figures/frozen/`, which `.gitignore:73` (`results/**/figures/`) excludes precisely so published figures live in `figures/<family>/` with captions. None had a caption or provenance sibling. | Moved to `figures/dnabert_promoter/` with 12 `.CAPTION.md` files and a family README. `plot_frozen_figures.py` `OUTDIR` repointed so reruns land in the canonical location. |
| M8 | **No doc tiers.** All three new markdown files lacked a `T1`/`T2`/`T3` tag — invariant 7. | Tier tags added to the registry (T2), both result-tree docs (T2), and every new document. |
| M9 | **Not registered anywhere.** No family `RESULTS_INDEX.md`; untouched: `results/RESULTS_INDEX.md`, `README.md` study map, `CLAUDE.md`, `docs/ROADMAP.md`, `docs/PREREG_CONFORMANCE.md`, `docs/NOMENCLATURE.md`. | All updated. New `results/dnabert_promoter/RESULTS_INDEX.md`; study-map row; `CLAUDE.md` mission, ledger table, study count, corrections pointer, and a second-model entry in the environment section; roadmap transfer milestone; four nomenclature entries. |
| M10 | **No dependency declaration.** `pyproject.toml` was untouched, so the second model's requirements were undeclared and implicitly borrowed from the `splicebert` extra (where `scikit-learn` lives); `joblib`, which persists every probe artifact, was only a transitive dependency. | A `dnabert` extra added — `transformers`, `scikit-learn`, `joblib`, `pandas`, `matplotlib` — and the study README's run block updated to `uv run --extra dnabert`. |
| M11 | **`donor` collides with the repo's controlled vocabulary.** Repo-wide "donor" is a splice donor site; the arc uses it for the activation-source example ("donor matrix", "same-class donor", `negative_class_donors`). | `docs/NOMENCLATURE.md` entry marking it AMBIGUOUS with "activation source" preferred; `DECISIONS.md` D11; the restoration verdict's "not donor-specific" rewritten as "not source-specific". |

## Recorded as gated work, not fixed here

These need a re-run or a decision, and each now blocks promotion of the rows it affects.

| # | Finding | Where it is tracked |
|---|---|---|
| G1 | **33 of 35 runners emit no `*.manifest.json`** — invariant 1. Only 5 of ~109 artifacts have a sibling record. Back-filling would mean asserting provenance nobody recorded, so the `manifest` column is truthfully blank instead (`results/RESULTS_INDEX.md` defines it as the record "where one exists"). A helper already exists at `src/gfmcd/splicing/io.py`. | Family index "Known provenance gap"; `DECISIONS.md` departures table; roadmap gate 3; `CLAUDE.md` code-layout note. |
| G2 | **Single seed (42) throughout**, against convention 4's requirement of multiple seeded draws for effect-existence — with a `0.0019` endpoint margin resting on it. | `DBP4`, `DBPR2`; roadmap gate 2. |
| G3 | **No calibration pair** — no planted-positive sensitivity, no permuted-label FPR — for the head ranking, group selection, or donor matrix. | `DECISIONS.md` departures table; `PREREG_CONFORMANCE.md`; keeps `DBP5` a candidate. |
| G4 | **Direction-blind ranking statistic.** Heads are ranked by `mean_absolute_logit_change`, which counts movement either way; `L0H8`'s signed effect is damaging in only `0.539` of examples. | `DBP5`, `DBPR3`. |
| G5 | **Underpowered TATA strata** (cells of n=4 and n=5, uncorrected) quoted alongside the n=71 result. The arc did flag these as unstable — partial credit. | `DBP10` names the n=48 `TATAAA` cohort as the only one worth powering. |
| G6 | **No figure for the three load-bearing Layer-2 results** (`DBP6`, `DBP7`, `DBP8`). The faithfulness curve — the study's central negative result — exists only as a CSV. | `figures/dnabert_promoter/README.md` "Not yet drawn"; roadmap. |

## Checked and clean

Recorded so a later reviewer does not re-litigate them: **no** absolute, Colab, or personal paths and no
`os.chdir` in the added code; **no** secrets or credentials; reverse-complement leakage genuinely zero on
all three counters; the pinned probe sha256 matches the committed file and is byte-identical to
`all_layer_probes/layer_2_probe.joblib`, so the same-artifact rule was really enforced; commit `f38a345`
("Preserve canonical frozen DNABERT Layer 2 probe") exists as cited; identity interventions exactly zero
in every causal family; the adapter's 12-hidden-state contract is asserted rather than assumed, so there
is **no off-by-one** in the layer indexing despite the `layer3_`/`Layer-2` naming clash; and no retired
SpliceBERT term (`10.67`, `10.58`, "splice-site strength", the human-chromatin phrasing) leaked into the
new prose.

## One pre-existing inconsistency corrected in passing

`CLAUDE.md` described "the three `CLAIMS.md` ledgers that carry rows" and "Four study directories … three
carry the record". Both were already stale before this PR: `studies/splicebert_acceptor_coupling/CLAIMS.md`
is a live T1 ledger with four `AC*` rows and appears in neither count, and seven study directories exist.
The counts are now accurate (five ledgers carrying 45 `C*` / 34 `GFS*` / 21 `HCA*` / 10 `DBP*` / 4 `AC*`).
**Still open:** the `README.md` study map omits `splicebert_acceptor_coupling`. That is outside this
review's subject and is left for an owner ruling rather than fixed silently.
