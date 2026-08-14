# DNABERT-2 promoter study — corrections

> **Tier: T2 — provenance.** History and retractions. Never cite as the current claim; the current
> claim is in [`CLAIMS.md`](CLAIMS.md).

`DBPR*` numbers are this study's corrections. Nothing here is ever deleted or edited away: a row that
turns out wrong gets a later row, not a rewrite. Every entry records what was said, what the artifact
actually supports, and what may be said instead.

All eight entries below were opened on **2026-08-11**, during the integration review that gave this
arc its first claim ledger. They correct the arc's own prose — the root-level `dnabert_readme.md` and
`FROZEN_EXPERIMENT_REGISTRY.md` as they arrived — against the artifacts committed alongside it. None
of them touches a `C*`, `HCA*` or `GFS*` claim.

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
