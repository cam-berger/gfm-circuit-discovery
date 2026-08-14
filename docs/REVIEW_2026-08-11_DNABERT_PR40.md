# Review 2026-08-11 — DNABERT-2 promoter arc (PR #40) against repository fundamentals

> The integration review that gave the DNABERT-2 arc its ledgers. Not a claim
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
"confirmed compact" mechanism.

**Two of this review's own findings did not survive verification and are withdrawn** — both cases of reading
a committed diagnostic file as an unaddressed defect without checking whether a later stage acted on it.
They are recorded in full in `DBPR1` and `DBPR3` and in the "Refuted" section below, because a review that
quietly drops its own bad findings is not auditable either. Final tally: **4 blockers and 13 major issues
fixed**, **3 items recorded as gated work**, **2 findings refuted**.

The arc's most valuable result was one it had argued *against* itself. Its own faithfulness curve refutes
the compactness its prose claimed, and its own restoration verdict refutes the transferability its
"confirmed interpretation" asserted. Both are now the ledgered claims (`DBP7`, `DBP8`).

## Blockers — fixed

| # | Finding | Evidence | Resolution |
|---|---|---|---|
| B1 | **The repository's own guard test failed on the branch.** `tests/test_manifest.py` had 2 failing tests: 132 tracked result files had no `results/MANIFEST.csv` row. This is what made the PR `mergeable_state: unstable`. | `test_repository_manifest_is_consistent`, `test_repository_manifest_indexes_every_tracked_result_file` | 109 rows added to `MANIFEST.csv`/`.json` with `status`, `evidence_level`, `polarity`, producing script, sibling manifest where one exists, and figure links. `verify_manifest.py` clean; **40/40 tests pass**. |
| B2 | **No claims ledger.** `studies/dnabert_promoter_mechanism/{CLAIMS,PREREGISTRATION,DECISIONS,README,RESULTS}.md` were all **0 bytes**, while 277 lines of substantive prose sat in a root-level `dnabert_readme.md`. The study had live claims and no point of contact — invariant 3. | `wc -c` = 0 on all five | Study consolidated into `studies/dnabert_promoter/`. `CLAIMS.md` written with `DBP1`–`DBP10`, `CORRECTIONS.md` with `DBPR1`–`DBPR8`, `DECISIONS.md` with D1–D11 plus a departures table. Root readme became the study `README.md`. Empty placeholders removed. |
| B3 | **Two readout endpoints intermingled.** 42 artifacts under `causal/layer3_*` set `SELECTED_LAYER = 3`, read `hidden_states[3]`, and score with `all_layer_probes/layer_3_probe.joblib` (sha256 `2337441e…`) — not the canonical `selected_layer_2_probe.joblib` (`bd0d9e14…`). They sat flat beside the Layer-2 causal results with nothing marking the difference, and no prose cited them. | `selected_layer: 3` in 20 summaries; `grep SELECTED_LAYER scripts/dnabert_promoter/*.py` | Classified `superseded` (37 `MANIFEST` rows under `claim_id` `DBPR7`, registered as a non-claim ID in `results/RESULTS_INDEX.md`). Hazard documented at the top of the family index, in `DECISIONS.md` D6, and in the "Deliberately not claimed" section of the ledger. Retained, not deleted — results are immutable. |
| B4 | **A null promoted to a confirmed mechanism.** "The compact confirmed component set is `Layer-0 MLP + L0H8`" — while the PR's own faithfulness curve gives that set standalone faithfulness `0.194`, no knee, and `1.000` only at size 39 with all three early MLPs. Conditional delivery (`0.882`) had been read as faithfulness. | `frozen_claim_tests/faithfulness_curve/*_curve.csv` | `DBP7` written as a bound/refutation with `polarity: bounded`; `DBPR4` retires *compact*, *sparse*, and bare *the circuit* for this study. Registry prose corrected in place. Nomenclature entry added. |

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
| M11 | **33 of 35 runners emitted no `*.manifest.json`** — invariant 1, and the largest provenance gap in the repository. Only 5 of ~109 artifacts had a sibling record, and the 5 that existed were config echoes carrying no `command`, `git_commit`, `created_at_utc`, seed, or input hashes. | New `src/gfmcd/dnabert/manifest.py` wrapping the repository's own `ExperimentManifest`, wired into all 15 canonical runners; 11 unit tests in `tests/test_dnabert_manifest.py`. Records hashed inputs/outputs, the probe sha256 that defines the readout, the split provenance, seed, command, and git state. Historical artifacts keep a blank `manifest` column rather than a back-filled record. |
| M12 | **No figure for the three load-bearing Layer-2 results.** `DBP6`, `DBP7`, `DBP8` had none, while 10 of the 24 committed figures depicted the superseded suite. The faithfulness curve — the study's central negative result — existed only as a CSV. | Figures **9–12** added and generated from the committed artifacts: the faithfulness curve with both MLP jumps annotated, the necessity/delivery accounting with the non-additivity marked, and the Layer-2 transplant matrix on both splits. Captions written; `MANIFEST` figure links updated. The originals were regenerated byte-for-byte identically and restored, so the diff carries only new files. |
| M13 | **`donor` collides with the repo's controlled vocabulary.** Repo-wide "donor" is a splice donor site; the arc uses it for the activation-source example ("donor matrix", "same-class donor", `negative_class_donors`). | `docs/NOMENCLATURE.md` entry marking it AMBIGUOUS with "activation source" preferred; `DECISIONS.md` D11; the restoration verdict's "not donor-specific" rewritten as "not source-specific". |

## Recorded as gated work, not fixed here

These need a re-run or a decision, and each now blocks promotion of the rows it affects.

| # | Finding | Where it is tracked |
|---|---|---|
| G1 | **Single seed (42) throughout**, against convention 4's requirement of multiple seeded draws for effect-existence — with a `0.0019` endpoint margin resting on it. | `DBP4`, `DBPR2`; roadmap gate 2. |
| G2 | **No calibration pair** — no planted-positive sensitivity, no permuted-label FPR — for the head ranking, group selection, or donor matrix. | `DECISIONS.md` departures table; `PREREG_CONFORMANCE.md`; keeps `DBP5` a candidate. |
| G3 | **Underpowered TATA strata** (cells of n=4 and n=5, uncorrected) quoted alongside the n=71 result. The arc did flag these as unstable — partial credit. | `DBP10` names the n=48 `TATAAA` cohort as the only one worth powering. |

## Refuted — two of this review's own findings

Both are recorded in the study's `CORRECTIONS.md` as well, so a reader who arrives at either row from the
ledger finds the withdrawal rather than the claim.

**R1 — "the split is contaminated and bounds every metric" (was `DBPR1`; originally blocker B3).** The
first pass found `shared/leakage_report.json` — 259 exact-duplicate groups, 104 spanning splits, 64
label-conflicting — noted that no prose mentioned it, and concluded every metric was a contaminated upper
bound. Wrong: that file is the **diagnosis**, written over the raw cohort, and `clean_promoter_cohort.py`
is the **treatment**. It removes all conflicting-label hashes globally, de-duplicates within split, drops
cross-split repeats under `train > validation > test` precedence, and *asserts* pairwise disjointness of
the retained hash sets. `extract_frozen_representations.py` reads its output directory, so every number in
the study is computed on cleaned splits. Independently confirmable from committed artifacts: raw
`47,356 / 5,920 / 5,920` against the probe run's own row counts `47,149 / 5,856 / 5,865` — 326 rows removed
— and none of the 5,865 committed test-prediction hashes appears in the train-duplicated or
label-conflicting sets. The corrected row, `DBP2`, is a *positive* data-hygiene claim whose real residual is
narrow: exact and reverse-complement identity are screened, near-duplicates are not, and the cleaner's own
report is uncommitted.

**R2 — "the head ranking statistic is direction-blind" (was gated item G4).** The first pass reported that
heads were ranked by `mean_absolute_logit_change` and that `L0H8`'s effect was damaging in only `0.539` of
examples. Both halves were the reviewer's own artefact. `run_head_ablation_scan.py` sorts ascending by
**`promoter_mean_damage`**, a signed quantity where negative means damaging, so the most
promoter-supporting head sorts first; `run_frozen_faithfulness_curve.py` uses the same signed field as its
primary key with the absolute one only as a tie-break, and says so in a comment. The `0.539` was
`change_from_clean.fraction_positive` over **both classes**; the promoter-specific field is
`promoter_fraction_damaged` = `0.914`. The signed ranking is the stronger result, and `DBP5` is now written
from it.

The common failure was reading a committed diagnostic as an unaddressed defect. Both would have been caught
by one more question — *does anything downstream consume this?* — which is now the first check in
[`REPO_SCRUB_PROCEDURE.md`](REPO_SCRUB_PROCEDURE.md) terms: trace the consumer before calling a file a
finding.

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
