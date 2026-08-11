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
| **DBPR1** | Held-out balanced accuracy `0.810` and ROC-AUC `0.892` presented as clean generalization, with no mention of duplication anywhere in the arc's prose. | The arc's own `shared/leakage_report.json` records 259 exact-duplicate groups, **104 spanning splits**, and 64 groups where one sequence carries both labels (20 of them cross-split). No de-duplication was applied. | **BOUNDED** — quote these metrics only as contaminated-split upper bounds (DBP2, DBP3). A de-duplicated re-run is required before any generalization language. |
| **DBPR2** | "The current frozen-model endpoint is the **Layer-2 masked-mean probe**"; "Layer 2 is the strongest tested frozen endpoint" used as a settled fact, and "the best frozen readout moved from Layer 5 to Layer 2" presented as a finding. | Validation balanced accuracy is `0.8050` (L2), `0.8031` (L3), `0.8014` (L1 and L4) — a `0.0019` margin on **one seed**. Repo convention 4 requires multiple seeded draws for effect-existence. | **BOUNDED** — Layer 2 is a *working choice with a stated margin*, not a localization result. Say "the strongest tested endpoint, by `0.0019` on a single seed, among layers that are not separated". |
| **DBPR3** | Head ablation described as testing "necessity", with `L0H8` as the "primary head candidate". | The ranking statistic is `mean_absolute_logit_change`, which is **direction-blind**; `L0H8`'s signed effect is damaging in only `0.539` of examples. | **BOUNDED** — licensed as "most intervention-sensitive head tested" (DBP5). "Necessary" requires a signed, controlled contrast that has not been run. |
| **DBPR4** | "The compact confirmed component set is `Layer-0 MLP + L0H8`"; "The compact Layer-0 MLP + L0H8 set performs at least as well as the full early group." | The arc's own faithfulness curve gives the five-node set standalone faithfulness `0.194` (test), rising to `1.000` only at size **39** once all three early MLPs are included, with **no knee**. Conditional delivery (`0.882`) was read as if it were faithfulness. | **WITHDRAWN** — the words *compact*, *sparse*, and bare *the circuit* are retired for this study. Use "a structured causal core within a distributed, MLP-dominated computation" (DBP7). Mirrors **C33** in the SpliceBERT arc. |
| **DBPR5** | "The Layer-0 MLP carries a **transferable** promoter-class representation"; "`L0H8` substantially improves **delivery**"; "`L0H8` contributes **sequence-sensitive** information"; and Holm-adjusted p `= 2.8539e-22` quoted for two different comparisons. | Matched-versus-same-class — the contrast that would show example specificity — is null on both components (`0.036`, p `0.252`; `0.053`, p `0.091`). Only the class contrast is large (`0.861`, `0.995`). The quoted `2.8539e-22` **appears in no artifact**: both raw p-values underflowed to `0.0`, and Holm was applied across only two of the six committed comparisons. | **WITHDRAWN** for "transferable", "delivery", and "sequence-sensitive"; the p-value is **unsupported** and must not be requoted. Licensed statement: class-level transfer, no example specificity (DBP8). |
| **DBPR6** | An eight-metric fine-tuning table (validation balanced accuracy `0.5278`, ROC-AUC `0.7758`, PR-AUC `0.7346`, F1 `0.6222`; test `0.5108`, `0.6912`, `0.7076`, `0.6292`), plus "the self-fine-tuning pipeline is operational" and a study-status row reading "pipeline implemented; full runs ongoing". | **No DNABERT fine-tuning script, config, checkpoint, or result artifact exists in this repository.** There is nothing to control these numbers. | **WITHDRAWN** — removed from live prose. Restoring them requires the pipeline and its artifacts to be committed first, then a new `DBP` row. |
| **DBPR7** | Forty-two artifacts under `results/dnabert_promoter/causal/layer3_*` shipped flat alongside the Layer-2 causal results, with no marker that they use a different readout, while the study's prose named a single Layer-2 endpoint throughout. | Those runners set `SELECTED_LAYER = 3`, read `hidden_states[3]` — encoder **Layer 3** — and score with `all_layer_probes/layer_3_probe.joblib` (sha256 `2337441e…`), **not** the canonical `selected_layer_2_probe.joblib` (sha256 `bd0d9e14…`). Their summaries record `selected_layer: 3`. No prose cites them. | **RECLASSIFIED** as `superseded` in `results/MANIFEST.csv`. They control no claim and must never be quoted for a Layer-2 result. Retained rather than deleted, per the immutability rule. |
| **DBPR8** | "preregistered component conditions" (frozen registry, test-confirmation section); the section heading "Confirmed validation result". | No preregistration document was ever committed for this arc. The validation→test discipline *was* honoured in substance — untouched test cohorts, one reused probe artifact — but no protocol was frozen in advance. | **BOUNDED** — "preregistered" is withdrawn as a description. Say "validation-selected, test-confirmed, not preregistered". Recorded as unaudited in `docs/PREREG_CONFORMANCE.md`. |

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
