# DNABERT-2 promoter study — corrections

> History and retractions. Never cite as the current claim; the current
> claim is in [`CLAIMS.md`](CLAIMS.md).

`DBPR*` numbers are this study's corrections. Historical rows remain here even when an unsupported
artifact family leaves the active tree; archived files are recoverable from the Git tag named by the
later correction. Every entry records what was said, what the artifact actually supported, and what may
be said instead.

The file now carries `DBPR1`–`DBPR14` in five blocks. The **eight entries in the table
immediately below** were opened on **2026-08-11**, during the integration review that gave this arc its
first claim ledger; they correct the arc's own prose — the root-level `dnabert_readme.md` and
`FROZEN_EXPERIMENT_REGISTRY.md` as they arrived — against the artifacts committed alongside it. Three
later blocks follow: `DBPR9`–`DBPR10` (2026-08-13, the fine-tuned extension), `DBPR11`
(2026-08-14, the post-merge remediation) and the active-tree archive disposition (`DBPR12`), then
`DBPR13`–`DBPR14` (2026-08-24, frozen-side historical artifact recovery — unrelated to the fine-tuned
archive). None of these touches a `C*`, `HCA*` or `GFS*` claim.

| ID | Retired / bounded statement | What the artifact supports | Status |
|---|---|---|---|
| **DBPR1** | *(reviewer error — see "Findings raised against this arc, and refuted" below.)* This row originally asserted that no de-duplication had been applied and bounded every metric in the study as contaminated. | De-duplication **was** applied, by an explicit policy with an enforced disjointness assertion, removing 326 rows. | **WITHDRAWN in full.** The accurate statement is `DBP2`. No metric in this study is bounded by exact-duplicate contamination. |
| **DBPR2** | "The current frozen-model endpoint is the **Layer-2 masked-mean probe**"; "Layer 2 is the strongest tested frozen endpoint" used as a settled fact, and "the best frozen readout moved from Layer 5 to Layer 2" presented as a finding. | Validation balanced accuracy is `0.8050` (L2), `0.8031` (L3), `0.8014` (L1 and L4) — a `0.0019` margin on **one seed**. Repo convention 4 requires multiple seeded draws for effect-existence. | **BOUNDED** — Layer 2 is a *working choice with a stated margin*, not a localization result. Say "the strongest tested endpoint, by `0.0019` on a single seed, among layers that are not separated". |
| **DBPR3** | Head ablation described as testing "necessity", with `L0H8` as the "primary head candidate". | The ranking is properly signed and directed (see the note below), but necessity needs more than a large ablation effect at a linear readout: there is no calibration pair, one seed, and validation only. | **BOUNDED** — licensed as a candidate ranking (DBP5). "Necessary" requires the calibration controls and a multi-seed replicate, neither of which has been run. |
| **DBPR4** | "The compact confirmed component set is `Layer-0 MLP + L0H8`"; "The compact Layer-0 MLP + L0H8 set performs at least as well as the full early group." | The arc's own faithfulness curve gives the five-node set standalone faithfulness `0.194` (test), rising to `1.000` only at size **39** once all three early MLPs are included, with **no knee**. Conditional delivery (`0.882`) was read as if it were faithfulness. | **WITHDRAWN** — the words *compact*, *sparse*, and bare *the circuit* are retired for this study. Use "a structured causal core within a distributed, MLP-dominated computation" (DBP7). Mirrors **C33** in the SpliceBERT arc. |
| **DBPR5** | "The Layer-0 MLP carries a **transferable** promoter-class representation"; "`L0H8` substantially improves **delivery**"; "`L0H8` contributes **sequence-sensitive** information"; and Holm-adjusted p `= 2.8539e-22` quoted for two different comparisons. | Matched-versus-same-class — the contrast that would show example specificity — is null on both components (`0.036`, p `0.252`; `0.053`, p `0.091`). Only the class contrast is large (`0.861`, `0.995`). The quoted `2.8539e-22` **appears in no artifact**: both raw p-values underflowed to `0.0`, and Holm was applied across only two of the six committed comparisons. | **WITHDRAWN** for "transferable", "delivery", and "sequence-sensitive"; the p-value is **unsupported** and must not be requoted. Licensed statement: class-level transfer, no example specificity (DBP8). |
| **DBPR6** | An eight-metric fine-tuning table (validation balanced accuracy `0.5278`, ROC-AUC `0.7758`, PR-AUC `0.7346`, F1 `0.6222`; test `0.5108`, `0.6912`, `0.7076`, `0.6292`), plus "the self-fine-tuning pipeline is operational" and a study-status row reading "pipeline implemented; full runs ongoing". | **No DNABERT fine-tuning script, config, checkpoint, or result artifact exists in this repository.** There is nothing to control these numbers. | **WITHDRAWN** — removed from live prose. Restoring them requires the pipeline and its artifacts to be committed first, then a new `DBP` row. |
| **DBPR7** | Forty-two artifacts under `results/dnabert_promoter/causal/layer3_*` shipped flat alongside the Layer-2 causal results, with no marker that they use a different readout, while the study's prose named a single Layer-2 endpoint throughout. | Those runners set `SELECTED_LAYER = 3`, read `hidden_states[3]` — encoder **Layer 3** — and score with `all_layer_probes/layer_3_probe.joblib` (sha256 `2337441e…`), **not** the canonical `selected_layer_2_probe.joblib` (sha256 `bd0d9e14…`). Their summaries record `selected_layer: 3`. No prose cites them. | **RECLASSIFIED** as `superseded` in `results/MANIFEST.csv`. They control no claim and must never be quoted for a Layer-2 result. Retained rather than deleted, per the immutability rule. |
| **DBPR8** | "preregistered component conditions" (frozen registry, test-confirmation section); the section heading "Confirmed validation result". | No preregistration document was ever committed for this arc. The validation→test discipline *was* honoured in substance — untouched test cohorts, one reused probe artifact — but no protocol was frozen in advance. | **BOUNDED** — "preregistered" is withdrawn as a description. Say "validation-selected, test-confirmed, not preregistered". Recorded as unaudited in `docs/PREREG_CONFORMANCE.md`. |

## Findings raised against this arc, and refuted

Recorded because the reviewer was wrong and the arc was right, which is worth as much as a correction.
Both errors were the same shape: a committed diagnostic file was read as an unaddressed defect without
first checking whether a later stage acted on it.

**Withdrawn: "the split is contaminated and bounds every metric" (was `DBPR1`).** The first pass of this
review found `shared/leakage_report.json` — 259 exact-duplicate groups, 104 spanning splits, 64
label-conflicting — observed that no prose mentioned it, and concluded that `0.810` / `0.892` were
contaminated upper bounds. That was wrong. `leakage_report.json` is the **diagnosis**, written by
`build_promoter_cohort.py` over the *raw* cohort; `clean_promoter_cohort.py` is the **treatment**, and
`extract_frozen_representations.py` reads its output directory (`shared/clean/`), so every probe and
intervention in the study runs on cleaned splits. The cleaner removes all conflicting-label hashes
globally, de-duplicates within split, drops cross-split repeats under `train > validation > test`
precedence, and then *asserts* pairwise disjointness of the retained hash sets — it would crash rather
than emit a leaky split.

The arithmetic confirms it independently from committed artifacts: raw `47,356 / 5,920 / 5,920` against
the probe run's own `train_rows / validation_rows / test_rows` of `47,149 / 5,856 / 5,865` — 326 rows
removed — and not one of the 5,865 committed test-prediction hashes appears in either the
train-duplicated or the label-conflicting hash set. The correct claim is `DBP2`, which is a *positive*
data-hygiene result with a narrower residual: exact and reverse-complement identity are screened,
near-duplicates are not, and the cleaner's own report is not committed.

**Withdrawn: "the head ranking statistic is direction-blind."** The first pass of this review reported that
`run_head_ablation_scan.py` ranked heads by `mean_absolute_logit_change`, a statistic that counts movement
in either direction, and that `L0H8`'s signed effect was damaging in only `0.539` of examples. Both halves
were wrong, and the error was the reviewer's own: the runner sorts ascending by
**`promoter_mean_damage`**, a signed quantity in which negative means damaging, so the most
promoter-supporting head comes first. `run_frozen_faithfulness_curve.py` uses the same signed field as its
primary key with the absolute one only as a tie-break, and carries the comment "*ablating the node lowered
the promoter readout more. Therefore, causally damaging nodes must come first*". The `0.539` figure was
`change_from_clean.fraction_positive`, computed over **both classes**; the promoter-specific field is
`promoter_fraction_damaged`, which is `0.914` for `L0H8`.

The arc's direction handling is correct throughout, and the signed ranking is a stronger result than the
absolute one: it separates six early damaging heads from a median near zero *and* identifies `L2H7` as an
anti-head whose ablation raises promoter evidence — structure the absolute statistic destroys. `DBP5` is
written from the signed field. Nothing else in this file depended on the withdrawn reading.

## Two things the arc got right, recorded so they are not "corrected" later

Both were checked against the artifacts during this review and hold:

1. **The canonical probe hash is correct.** `FROZEN_EXPERIMENT_REGISTRY.md` pins
   `selected_layer_2_probe.joblib` at sha256 `bd0d9e14e1935e3ebd8e2ce638ccd5a721feb9dc9a0895447d9b62080ce98b3b`,
   and the committed file hashes to exactly that. It is also byte-identical to
   `all_layer_probes/layer_2_probe.joblib`, so the "same artifact for validation and test" rule was
   genuinely enforced.
2. **The headline metrics match their artifacts exactly.** Test balanced accuracy `0.8102122021467204`,
   ROC-AUC `0.8919269597801556`, PR-AUC `0.8918734234801855`, and every donor-matrix recovery fraction
   and mean difference quoted in the registry, reconcile to the committed JSON to all printed digits.
   The arc's numbers are accurate; what needed correcting was what they were said to *mean*.

## Internal contradictions resolved in this pass

The frozen registry contained statements that contradicted each other in the same file. They are
resolved in favour of the artifacts, and the registry has been edited to match:

- "No sequence-specific delivery claim is currently licensed" and "not donor-specific" (restoration
  verdict) versus "carries a **transferable** promoter-class representation" and "`L0H8` substantially
  improves **delivery**" (confirmed interpretation). **The restoration verdict was right**; see DBPR5.
- Experiment-block status lines claiming work was "not run as a controlled assay" (block 5, biological
  specificity) and "not run against the corrected endpoint" (blocks 6 and 8, low-rank mediation and
  circuit structure) while the arc *shipped* a controlled TATA perturbation and both circuit-accounting
  and faithfulness-curve artifacts at `selected_layer: 2`. Block 5 and block 8 status lines were stale
  and have been corrected; block 6 was accurate — low-rank mediation genuinely was not run.

## Fine-tuned extension corrections opened 2026-08-13

These rows do not rewrite DBPR6. DBPR6 remains the historical record of what was unsupported at the
2026-08-11 integration review. The later artifact package is a new evidentiary event.

| ID | Retired / bounded statement | What the artifact supports | Status |
|---|---|---|---|
| **DBPR9** | The live study text inherited DBPR6's historical statement that “no DNABERT fine-tuning code, checkpoint, or result artifact exists in this repository,” even after a later fine-tuned analysis package was committed. | A separately verified analysis of `magicslabnu/DNABERT-2-finetuned-prom_300_all` now exists, including checkpoint-loading validation, held-out predictive evaluation, layer probing, targeted ablation, activation restoration, activation-source swapping, and TATA perturbation. The old seed-42 self-fine-tuning smoke table remains unsupported and withdrawn. | **SUPERSEDED AS A CURRENT-STATUS STATEMENT, NOT REWRITTEN HISTORICALLY.** DBPR6 remains true about the evidence state on 2026-08-11. Current fine-tuned claims are DBP11–DBP16. |
| **DBPR10** | “Fine-tuning substantially improves promoter prediction and relocates the strongest promoter representation from Layer 2 to Layer 10,” and equivalent prose that treats the frozen and fine-tuned results as a controlled before/after experiment whose only changed variable is fine-tuning. | The two packages differ in more than model weights: frozen uses GUE `prom_core_all` at 70 nt, leakage-cleaned splits, and a fixed masked-mean probe; fine-tuned uses GUE `prom_300_all` at 300 nt, the checkpoint classifier for primary behaviour, and a separate layer-probe sweep. The frozen Layer-2 choice is itself thin (`0.0019` validation-BA lead, one seed). The within-state fine-tuned results are real; the cross-state numerical differences are descriptive. | **BOUNDED.** Licensed wording: “The analyzed fine-tuned state shows late-layer linear accessibility together with strong early Layer-0 causal dependence; compared descriptively with the frozen study, the representational profiles differ.” Do not attribute the difference uniquely to fine-tuning until a matched assay is run (DBP17). |

## Post-merge remediation opened 2026-08-14

Opened after PR #44 had already merged to `master`. `DBPR11` is one entry covering four defects the
merge introduced into the record itself, not into the science: the fine-tuned assays and their numbers
are unaffected. Nothing in `DBPR6` or `DBPR9` is rewritten — `DBPR6`'s condition for restoring
self-fine-tuning numbers (commit the pipeline and its artifacts first) is still unmet, and this entry
records that the condition was briefly bypassed by an artifact rather than by prose.

| ID | Retired / bounded statement | What the artifact supports | Status |
|---|---|---|---|
| **DBPR11** | Four items entered the tree with PR #44. **(a)** `results/dnabert_promoter/finetuned/performance_comparison/frozen_self_hf_performance.csv` carried a row `Self-finetuned DNABERT (mean seeds 42/43/44),0.831,0.662,0.909,0.912,0.831` filed `canonical` against `DBP17`; the values are hand-typed literals from the source notebook with no checkpoint, pipeline, or upstream artifact anywhere in the repository, and its companion frozen row rounded `DBP3` to `0.81`/`0.892`. **(b)** `results/dnabert_promoter/RESULTS_INDEX.md`, a T1 canonical document, was committed truncated at line 81 inside an unterminated code fence, deleting its provenance-gap and figures sections. **(c)** `layer_localization/selected_endpoint.json` shipped an `interpretation` string asserting representational relocation to Layer 10, and a derived `layer_shift: 8` field. **(d)** Nine `DBP14` rows in `results/MANIFEST.csv` carried `evidence_level: Causal`, a value that is not a rung on the repository's ladder and that `DBP15`'s own rows do not use. | **(a)** No artifact supports any self-fine-tuning metric; `DBPR6` is unchanged and its restoration condition remains unmet. The two artifact-backed rows are `DBP3` (`results/dnabert_promoter/frozen/layer_probes/selected_layer_2_test_statistics.json`) and `DBP12` (`results/dnabert_promoter/finetuned/test/metrics.csv`). **(b)** The deleted sections are recoverable at `a42197aa^1`. **(c)** `layer_selection_band.json` shows the Layer-10 argmax leads its runner-up by `0.000998` balanced accuracy = `0.33` standard errors; layers 7–11 lie within one SE and 5–11 within two, and the test argmax is Layer 8. The relocation reading is bounded by `DBPR10`, and the within-state selection is not separated either. **(d)** `DBP14` and `DBP15` are the same class of intervention; `DBP15` files as `OV/mediation`. | **CORRECTED.** **(a)** The unsourced row was **removed**; the file now carries only the two artifact-backed rows at full precision, with `gue_task`, `sequence_length_nt`, `readout`, `n_test` and `source_artifact` columns making the non-comparability explicit, and was **renamed** `frozen_self_hf_performance.csv` → `frozen_hf_performance.csv` because the old name named a row that no longer exists. **(b)** Restored and extended. **(c)** `interpretation` and `layer_shift` **removed** from the artifact; interpretation lives in `DBP13`, and the artifact now points at `layer_selection_band.json` and sets `selection_is_not_separated`. **(d)** Remapped to `OV/mediation` in `MANIFEST.csv` and `MANIFEST.json`. `DBP13` is restated to report a band rather than a point; the seeded probe repeat that would settle it is still owed and is gated on a GPU run. |

## Active-tree archive disposition

| ID | Retired / bounded statement | What the record supports | Status |
|---|---|---|---|
| **DBPR12** | DBP11–DBP17 were presented as live study claims even though 52 of their artifacts were produced only by a notebook, lacked producing runners and sibling manifests, and were not reproducible from a clean checkout. The cross-state comparison also used unmatched tasks, lengths, cohorts, and readouts. | The historical measurements remain available for provenance, but they do not meet this repository's active-evidence standard. No fine-tuned result is used by the paper's supported DNABERT-2 comparison, which is DBP6–DBP8. | **ARCHIVED / REMOVED FROM THE LIVE LEDGER.** The artifacts, notebook, figures, and former DBP11–DBP17 wording are recoverable at `archive/unsupported-exploratory-2026-08-22`. Any successor is a new, preregistered, multi-seed, matched-task study with producing scripts and manifests. |

## Historical artifact recovery, 2026-08-24

This block is unrelated to the fine-tuned archive disposition above: it covers only the frozen
(`DBP1`–`DBP10`) study's raw-artifact recovery. Nothing here touches `DBP11`–`DBP17` or reopens the
`DBPR12` archival.

The frozen arc's ten canonical per-example datasets — the raw rows behind `DBP5`'s head/branch/grouped-head
ablation scans and their `causal/localization_cohorts/` cohorts, `DBP6`'s circuit-accounting per-example
rows, `DBP7`'s component-faithfulness and faithfulness-curve per-example rows, and `DBP10`'s TATA
perturbation per-example row and cohort — had a working tree that carried only their committed summary
JSONs, not the underlying per-example CSVs the summaries are aggregated from. These ten raw datasets were
recovered byte-for-byte from a preserved 2026-08-04/2026-08-05 JupyterHub bundle (`C:\Algoverse\algoverse\tar`
at recovery time), verified against the bundle by SHA-256, and confirmed to reproduce their already-committed
summary JSONs exactly, field by field, using each experiment's own aggregation logic read out of its current
runner script (mean/std/median/min/max and the delivery/necessity/faithfulness/promoter-damage fraction
definitions) — 100% of checks matched across all ten datasets.

Ten standard sibling recovery manifests (`*.manifest.json`, the repository's `ExperimentManifest` schema)
were created for these ten datasets, since the runners' `write_run_manifest` wiring postdates the runs that
produced them (`DECISIONS.md`'s “Known departures from repository convention” table, convention 7; see also
`D14`). Each recovery manifest records,
distinctly: the reported 2026-08-04/2026-08-05 recovery window; the recovered artifact's and the committed
summary's SHA-256 hashes; the raw-to-summary reproduction result; the recovered *and* the current runner
script hashes where the historical script version was itself recoverable (three of the ten runners; the
other seven were not in the bundle); the model/revision/tokenizer-revision/selected-layer/probe/cohort
provenance; and an explicit `unknown` for the original command, seed, and execution environment — none of
which were fabricated. The manifests' own `git_commit`/`environment` fields, and their `created_at_utc`,
record only the *recovery-verification* state (2026-08-24), never asserted as the state that produced the
original result.

`DBP2`'s previously-uncommitted residual provenance gap (noted in its own row above: “the cleaning stage's
own `shared/clean/cleaning_report.json` and the cleaned split hashes are not committed”) is closed:
`clean_promoter_cohort.py` was re-run (non-GPU, no model loaded) against the raw splits after their row
counts and record hashes were confirmed to match the previously-committed values exactly, producing
`shared/clean/cleaning_report.json`. Since that report carries no content hashes for the cleaned splits
themselves, a compact `shared/clean/split_manifest.json` was created alongside it, reusing
`gfmcd.dnabert.dataset.records_sha256` — the same hash function `shared/split_manifest.json` already uses
for the raw splits — rather than introducing a second, incompatible hashing scheme. `results/MANIFEST.csv`
and `.json` were extended with rows for all recovered and created files (`DBP2`/`DBP5`/`DBP6`/`DBP7`/`DBP9`/
`DBP10`), and now record `split_leakage_control: leakage_controlled` for the ten recovery manifests instead
of the previously-unrecorded value.

No DNABERT model was loaded and no experiment was rerun at any point in this recovery. Every summary number
and every claim's evidentiary status is exactly as committed before the recovery began, with two exceptions,
each its own correction below: `DBP10`'s TATA headline magnitudes, and the `DBP8` tag on two artifacts its
own claim text never cited.

| ID | Retired / bounded statement | What the artifact supports | Status |
|---|---|---|---|
| **DBPR13** | `CLAIMS.md` (`DBP10`) and `FROZEN_EXPERIMENT_REGISTRY.md` quoted mean motif damage `-0.302`, matched-control damage `-0.376`, mean excess damage `0.074`, and one-sided paired p `0.2334` for the n=71 controlled TATA perturbation. | The recovered, committed `validation_tata_perturbation_summary.json` (per-example rows reproduce it exactly) gives mean motif damage `-0.288970`, matched-control damage `-0.324068`, mean excess damage `0.035098`, one-sided paired p `0.311089`, at the same n=71 and fraction-more-damaging `0.521`. The retired figures coincide to six digits with the `all` bucket of the separate `..._exploratory_strata.json`, suggesting they were transcribed from that file rather than the canonical per-example summary; that file's own numbers (including its positions-2-4 and `TATAAA`-subgroup figures) were not independently re-derived in this pass and are unchanged. | **CORRECTED** in `CLAIMS.md` and `FROZEN_EXPERIMENT_REGISTRY.md`. The conclusion is unchanged — still a causal null at the tested power, still not significant, no TATA-specific mechanistic claim licensed — only the quoted magnitudes change. |
| **DBPR14** | `results/MANIFEST.csv`/`.json` tagged `causal/activation_restoration/validation_negative_class_donors_128_offset_1_summary.json` and its `_statistics.md` as `DBP8`, `canonical`, `OV/mediation`, `bounded`. | `DBP8`'s own `CLAIMS.md` row cites only the donor-matrix confirmation-package artifacts (`frozen_claim_tests/components/donor_matrix/*_confirmation_128_{statistics,summary}.json`, validation and test) as its evidence; neither `validation_negative_class_donors_128_*` file is named there. | **RECLASSIFIED** in `results/MANIFEST.csv`/`.json` to `status=development`, blank `claim_id`, blank `evidence_level`, `polarity=na`. Files retained, not deleted (mirrors `DBPR7`). `DBP8`'s claim text is unchanged and remains independently supported by the donor-matrix confirmation artifacts. |
