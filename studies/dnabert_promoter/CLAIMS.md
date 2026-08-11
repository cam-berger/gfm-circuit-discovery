# DNABERT-2 promoter claim ledger

> **Tier: T1 — canonical.** Current, authoritative. Read this.

**How to read this ledger.** Each row names one claim, the evidence level it actually reached, the
artifact that controls it, and the scope boundary beyond which it may not be quoted. Rows carry the
claims; the headings assert nothing. The `DBP*` prefix is reserved for this study and is the fourth
ledger family in the repository alongside `C*` (donor circuit), `HCA*` (Head–Context Atlas) and
`GFS*` (GTEx fine-tuning shift).

This ledger was written on 2026-08-11 during the integration review of the frozen DNABERT-2 promoter
arc, which arrived with an empty ledger file. Every row was derived from the committed artifacts, not
from the arc's own prose; where the prose claimed more than the artifacts support, the row is written
at the licensed level and the gap is recorded in [`CORRECTIONS.md`](CORRECTIONS.md) under a `DBPR`
number. **Read `CORRECTIONS.md` before quoting anything from this study**: five of the arc's
headline sentences are bounded or withdrawn there.

| # | Section | Claims |
|---|---|---|
| 1 | [Instrument, data, and what the split does not guarantee](#1-instrument-data-and-what-the-split-does-not-guarantee) | DBP1, DBP2 |
| 2 | [The frozen representation](#2-the-frozen-representation) | DBP3, DBP4 |
| 3 | [Component necessity and delivery](#3-component-necessity-and-delivery) | DBP5, DBP6 |
| 4 | [Faithfulness — the compactness bound](#4-faithfulness-the-compactness-bound) | DBP7 |
| 5 | [Activation transplant — class-level, not example-specific](#5-activation-transplant-class-level-not-example-specific) | DBP8 |
| 6 | [The biological bridge, stopped at its gate](#6-the-biological-bridge-stopped-at-its-gate) | DBP9, DBP10 |

## 1. Instrument, data, and what the split does not guarantee

| ID | Current licensed claim | Evidence level | Controlling artifact | Scope / next bridge |
|---|---|---|---|---|
| DBP1 | The frozen instrument is pinned and its interventions are exact. The assay uses `quietflamingo/dnabert2-no-flashattention` at revision `813031b2bf86d9e960a027e2734c908009f31601`, read through the repository adapter, which returns exactly the 12 encoder-layer outputs and **no embedding output** — so `hidden_states[i]` *is* encoder layer `i`. Identity interventions are exactly zero in every committed causal family (`identity_max_logit_difference = 0.0`; self-clean and self-corrupt maximum absolute differences `0.0`). | established: instrument | `results/dnabert_promoter/frozen_claim_tests/registry/frozen_config.json`; `results/dnabert_promoter/provenance/checkpoint_equivalence.md`; identity fields in `results/dnabert_promoter/causal/head_ablation/validation_head_scan_256_layers_0_1_2_summary.json` and `results/dnabert_promoter/frozen_claim_tests/circuit_accounting/*_summary.json` | Instrument exactness does not license any mechanism. The 12-not-13 hidden-state contract is asserted in `src/gfmcd/adapters/dnabert.py` and is the reason layer indices in this study need no embedding offset — do not port that assumption to another adapter. |
| DBP2 | **The GUE `prom_core_all` split as used here is not duplicate-clean, and this bounds every metric in the study.** The arc's own leakage scan records 259 exact-duplicate sequence groups, of which **104 span more than one split**, and 64 groups in which an identical 70-nt sequence carries **both** labels — 20 of those 64 also span splits (13 train↔validation, 6 train↔test, 1 validation↔test). Reverse-complement overlap is genuinely zero (0 pairs on all three counters). No de-duplication was applied before probing. | established: data defect, bounding | `results/dnabert_promoter/shared/leakage_report.json`; split hashes in `results/dnabert_promoter/shared/split_manifest.json` and `dataset_sha256.json` | Held-out numbers in DBP3/DBP4 are therefore **upper bounds contaminated by cross-split repetition**, not clean generalization estimates. The bridge is a de-duplicated re-run under the same frozen protocol; until then no DBP row may be quoted as evidence about generalization. Recorded as **DBPR1**. |

## 2. The frozen representation

| ID | Current licensed claim | Evidence level | Controlling artifact | Scope / next bridge |
|---|---|---|---|---|
| DBP3 | Frozen DNABERT-2 carries promoter-discriminative information that a masked-mean linear probe recovers without any fine-tuning. At the selected endpoint the probe reaches balanced accuracy `0.8102`, ROC-AUC `0.8919` and PR-AUC `0.8919` on the held-out test split, against a majority baseline of `0.500`. | Representation | `results/dnabert_promoter/frozen/layer_probes/all_layer_probe_results.json`; `selected_layer_2_test_statistics.json`; probe `selected_layer_2_probe.joblib`, sha256 `bd0d9e14e1935e3ebd8e2ce638ccd5a721feb9dc9a0895447d9b62080ce98b3b` | Bounded by DBP2 — these are contaminated-split numbers. Single seed (42), one dataset, one 70-nt length, one pooling rule. A probe result is a representation claim only; it names no component and no computation. |
| DBP4 | Encoder **Layer 2** is the *strongest tested* frozen endpoint, selected on validation with the test split used descriptively only. Its lead is **thin**: validation balanced accuracy `0.8050` at layer 2 versus `0.8031` at layer 3, `0.8014` at layers 1 and 4 — a margin of `0.0019` on a single seed. The frozen probe beats the strongest tested sequence baseline (k-mer 3–6: test balanced accuracy `0.7748`, ROC-AUC `0.8563`) by `0.035` balanced accuracy, and beats GC-composition (`0.6758`) by more. | Representation, selection reported with its margin | `results/dnabert_promoter/frozen/layer_probes/all_layer_probe_results.json` (`ranked_layers`, `test_usage`); `results/dnabert_promoter/frozen/baselines/sequence_baselines.json` | **"Layer 2 is the promoter endpoint" is not licensed** — a `0.0019` single-seed margin does not separate layers 1–4. Repo convention 4 requires multiple seeded draws for effect-existence; this selection has one seed, so layer identity is a *working choice*, not a finding. The k-mer margin is likewise single-split with no interval. Recorded as **DBPR2**. |

## 3. Component necessity and delivery

| ID | Current licensed claim | Evidence level | Controlling artifact | Scope / next bridge |
|---|---|---|---|---|
| DBP5 | Zeroing single attention heads at the Layer-2 readout changes the promoter margin very unevenly across heads, with `L0H8` the largest by a factor of two: mean absolute logit change `2.303` versus `1.076` (`L0H10`), `1.036` (`L0H3`), `0.952` (`L2H7`), `0.747` (`L2H9`) across all 36 heads of layers 0–2 on 256 validation examples. | Attention / necessity candidate | `results/dnabert_promoter/causal/head_ablation/validation_head_scan_256_layers_0_1_2_summary.json`; grouped and branch variants in `causal/grouped_head_ablation/` and `causal/branch_ablation/` | **The ranking statistic is direction-blind.** `mean_absolute_logit_change` counts movement either way, and `L0H8`'s signed effect is positive in only `0.539` of examples. This licenses "L0H8 is the most intervention-sensitive head tested", not "L0H8 is necessary for promoter detection". Single seed, validation only. Recorded as **DBPR3**. |
| DBP6 | A five-node candidate set — heads `L0H8`, `L1H8`, `L2H9`, `L2H3` plus `MLP0` — carries a majority of the corruption effect **conditionally**, and replicates from validation to an untouched test cohort. Knocking it out destroys necessity fraction mean `0.630` (test; `0.651` validation); patching it clean into a corrupt run delivers fraction mean `0.882` (test; `0.839` validation). Both are positive in `128/128` examples on both splits. | OV / mediation, conditional | `results/dnabert_promoter/frozen_claim_tests/circuit_accounting/test_candidate_circuit_confirmation_128_summary.json`; `validation_candidate_circuit_128_summary.json` | **These are conditional causal effects and must never be added or read as a percentage decomposition** — necessity `0.630` + delivery `0.882` `= 1.51`, and delivery maxima exceed `1.0` (`1.45` test, `1.36` validation), because residual and LayerNorm paths remain nonlinear. Conditional delivery is *not* standalone faithfulness; for that see DBP7. |

## 4. Faithfulness — the compactness bound

| ID | Current licensed claim | Evidence level | Controlling artifact | Scope / next bridge |
|---|---|---|---|---|
| DBP7 | **No compact promoter circuit is established; the held-out faithfulness curve has no knee, and what it does show is MLP-dominated.** Run standalone, the five-node candidate set of DBP6 reaches mean faithfulness `0.194` (test; `0.200` validation) — under a fifth of full-model behaviour. Growing the head set eightfold barely moves it (`0.194` at size 5 with 4 heads → `0.244` at size 15 with 14 heads). The two large jumps are both MLP additions: adding `MLP1` at size 16 takes faithfulness to `0.487`, and only adding `MLP2` at size 39 reaches `1.000`. | Faithfulness bound / **null for compactness** | `results/dnabert_promoter/frozen_claim_tests/faithfulness_curve/test_faithfulness_curve_confirmation_128_curve.csv`; `validation_faithfulness_curve_refined_128_curve.csv` | This is the row that forbids the words **compact**, **the circuit**, and **sparse** for this study. It reproduces, in a second GFM, the shape the SpliceBERT arc reached at **C33**: a structured causal core inside an irreducibly distributed computation. Minimality is unclaimed and currently refuted at the sizes tested. The arc's own prose asserted a "compact confirmed component set"; that is withdrawn under **DBPR4**. |

## 5. Activation transplant — class-level, not example-specific

| ID | Current licensed claim | Evidence level | Controlling artifact | Scope / next bridge |
|---|---|---|---|---|
| DBP8 | Transplanting clean early activations into a corrupt run recovers promoter evidence, but the recovered information is **class-level, not example-specific**. On the untouched test cohort (n=128), `MLP0 + L0H8` recovers matched-source fraction `0.879` and unrelated *same-class* source `0.843` — a difference of `0.036` that is **not significant** (raw p `0.252`); `MLP0` alone gives `0.739` matched versus `0.686` same-class, difference `0.053`, also not significant (raw p `0.091`). What *is* overwhelming is the class contrast: same-class minus opposite-class is `0.861` (`MLP0`) and `0.995` (`MLP0 + L0H8`), with raw p underflowing to `0.0`. `L0H8` alone recovers essentially nothing (`0.005` matched, validation). | OV / mediation, **bounded** — class-level transfer only | `results/dnabert_promoter/frozen_claim_tests/components/donor_matrix/test_component_donor_matrix_confirmation_128_statistics.json`; `..._summary.json`; validation twins in the same directory | **"Transferable promoter representation" and any sequence-specific delivery claim are not licensed** — matched-versus-same-class is the contrast that would show example specificity, and it is null on both components. An opposite-class source pushing the readout the other way shows the transplant moves a *class label's worth* of evidence. Holm adjustment was applied across only the two matched-versus-same-class tests, not the six comparisons present; the study's quoted `2.8539e-22` appears in no artifact (both underflowed to `0.0`). Recorded as **DBPR5**. |

## 6. The biological bridge, stopped at its gate

| ID | Current licensed claim | Evidence level | Controlling artifact | Scope / next bridge |
|---|---|---|---|---|
| DBP9 | TATA-like motifs are **positionally** enriched in promoters in this dataset, and not otherwise. `TATA[AT][AT]` shows no promoter enrichment across the full 70-nt window; restricted to motif starts at positions 0–7 it is modestly enriched — validation odds ratio `1.3646` (p `0.0272`), test odds ratio `1.6233` (p `0.00104`). | Biology — observational association | `results/dnabert_promoter/frozen_claim_tests/biological_features/tata_like_enrichment.json`; `tata_like_eligibility_summary.json` | The 0–7 window is **dataset-aligned, not TSS-aligned**: the genomic coordinate relative to the transcription start site was never verified, so this must not be described as a core-promoter position. An enrichment in the data is a property of the data, not of the model. |
| DBP10 | **The frozen Layer-2 readout is not shown to depend causally on TATA-like motifs.** In the controlled perturbation (n=71 promoters with exactly one early-window motif, two-base disruption, matched non-motif control with identical T→G / A→C substitutions, identity control exact), mean motif damage is `-0.302` against matched-control damage `-0.376`, giving mean excess damage `0.074` with the motif more damaging in only `0.521` of pairs and a one-sided paired p of `0.2334`. Exploratory stratification found no coherent subgroup; starts at positions 2–4 gave excess `0.0049`, p `0.4699`. | Biology — **causal null at the tested power** | `results/dnabert_promoter/frozen_claim_tests/biological_features/tata_perturbations/probe_results/validation_tata_perturbation_summary.json`; strata in `..._exploratory_strata.json`; cohort in `validation_tata_perturbation_cohort.json` | **No TATA-specific mechanistic claim is licensed.** A `0.521` fraction is a coin flip. The stratification is exploratory and uncorrected, with cells as small as n=4 and n=5 whose per-variant means (e.g. `-0.549` at n=4) must not be quoted as effects; the `TATAAA` subgroup (n=48, excess `0.1445`) is the only one worth powering up. This is a null at n=71, not evidence of absence — the bridge is a powered, held-out, multi-seed cohort with motif-specific component localization. |

## Deliberately not claimed

These are stated so a reader does not go looking for a row that should exist.

- **Supervised fine-tuning.** The arc's prose reported a seed-42 fine-tuning smoke with eight metrics.
  **No DNABERT fine-tuning code, checkpoint, or artifact exists anywhere in this repository**, so
  there is no row and the numbers are withdrawn — see **DBPR6**. The frozen-versus-fine-tuned
  comparison is a future arc, tracked in [`docs/ROADMAP.md`](../../docs/ROADMAP.md).
- **The Layer-3 replicate.** Forty-two artifacts under `results/dnabert_promoter/causal/layer3_*`
  were run at encoder **Layer 3** against a *different* probe than every claim above. They control
  nothing and are classified `superseded` in `results/MANIFEST.csv`. See **DBPR7** and the
  [family index](../../results/dnabert_promoter/RESULTS_INDEX.md).
- **Token-span localization.** Blocked by tokenizer/nucleotide alignment; no artifact, no claim.
- **Low-rank mediation and pathway tracing.** Not run against this endpoint. Hypotheses only.
- **Cross-model transfer.** Nothing here licenses a statement about SpliceBERT, and no `C*`, `HCA*`
  or `GFS*` row is affected by this study. Method transfer is the point; result transfer is not
  claimed. See [`docs/TRANSFER_GUIDE.md`](../../docs/TRANSFER_GUIDE.md).

## Preregistration status

**This study has no frozen preregistration.** The arc's prose described "preregistered component
conditions", but no protocol document was ever committed, so the confirmatory donor-matrix conditions
were fixed by an undocumented decision rather than a frozen memorial. This is a departure from repo
convention 1 (prereg → smoke → confirmatory → REFLECTION → claim) and is recorded as **DBPR8**. The
validation→test structure was genuinely honoured — the test cohorts are untouched, and the same probe
artifact was reused — but "preregistered" is withdrawn as a description of it. No `DBP` row may be
cited as a preregistered result, and `docs/PREREG_CONFORMANCE.md` records this family as unaudited.
