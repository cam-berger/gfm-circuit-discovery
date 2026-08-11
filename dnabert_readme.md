# DNABERT-2 Promoter Mechanistic Discovery

## Current mechanism

> A frozen DNABERT-2 encoder contains a promoter-discriminative representation that is most strongly
> recoverable at an early Layer-2 endpoint under the current masked-mean readout. Causal interventions
> indicate that promoter information is not uniformly distributed across the encoder, but the compactness,
> semantic identity, and transferability of the responsible mechanism remain under active investigation.

Here, **mechanism** means a causally supported account of how model components contribute to the scoped
promoter-classification behavior. A predictive probe, attention pattern, or motif association is not by
itself a mechanism. Candidate components must survive intervention-based tests before they license a
causal interpretation.

The current frozen-model endpoint is the **Layer-2 masked-mean probe**. This endpoint replaced an earlier
Layer-5 result after the frozen DNABERT pipeline was corrected and rerun cleanly. On the held-out test
split, the corrected frozen representation reached approximately **0.810 balanced accuracy** and
**0.892 ROC-AUC**, improving over the earlier pipeline result of approximately 0.735 balanced accuracy
and 0.811 ROC-AUC.

The key limitation is equally important: the current experiments support the existence of
promoter-relevant information and provide causal evidence for selected components, but they do **not**
yet establish a complete biological mechanism for promoter recognition. In particular, the semantic
identity of the causal signal, the degree to which it is motif-specific, and the extent to which the
same mechanism persists after supervised fine-tuning remain open.

## Scientific record

| Result | Evidence reached | Licensed interpretation | First missing bridge |
| --- | --- | --- | --- |
| Frozen promoter representation | Layer-wise probing → held-out validation/test evaluation | DNABERT-2 contains promoter-discriminative information that is linearly recoverable from frozen hidden states under the tested masked-mean readout. | Generalization across promoter datasets, sequence lengths, and alternative readouts. |
| Layer-2 endpoint | Corrected layer sweep → held-out test confirmation | Layer 2 is the strongest tested frozen endpoint for the current promoter assay and replaces the superseded Layer-5 endpoint. | Whether Layer 2 remains optimal under alternative datasets, seeds, pooling rules, or model checkpoints. |
| Frozen DNABERT vs k-mer baseline | Matched predictive evaluation | The corrected frozen DNABERT probe outperforms the strongest tested k-mer baseline on the current benchmark. | Broader simple/classical baseline suite and cross-dataset confirmation. |
| Localization cohorts | Predefined positive/negative cohorts → causal score-change analysis | Promoter-relevant effects can be measured on balanced held-out cohorts using the fixed Layer-2 endpoint. | Stronger localization specificity and replication across independent cohorts. |
| Head ablation | Component intervention → behavior change | Individual attention heads can be tested for necessity with respect to the Layer-2 promoter readout. | Stable, replicated head-level effects that survive matched null controls. |
| Grouped head ablation | Structured multi-component intervention | Groups of heads can be tested for collective necessity and redundancy. | Evidence that any group forms a reproducible compact causal core rather than reflecting distributed damage. |
| Activation restoration | Corruption → restoration → recovery | Restoration experiments test whether clean internal states can causally recover promoter evidence after corruption. | Path-specific mediation and replicated recovery across perturbation families. |
| Branch ablation | Structured pathway intervention | Candidate branches can be tested for whether they carry promoter-relevant information to the Layer-2 endpoint. | A closed source→route→readout account with direction and mediation established. |
| Component donor matrix | Candidate component transplantation | Early components, including `L0H8`, `L0_MLP`, their combination, and a broader early group, have been explicitly tested as candidate donors in the frozen model. | Fully quantified replication, null comparison, and a closed mechanistic interpretation. |
| TATA-like motif association | Motif-stratified observational analysis | TATA-like sequence contexts provide a biologically meaningful stratification for testing promoter behavior. | Strong evidence that the model causally relies on the motif rather than correlated sequence context. |
| TATA perturbation | Matched motif perturbation → paired Layer-2 score comparison | The current perturbation assay directly tests whether altering TATA-like motifs changes promoter evidence beyond matched controls. Exploratory validation results are heterogeneous and do not yet license a strong motif-specific mechanism claim. | Larger powered cohorts, replication on held-out data, and motif-specific causal localization. |
| Self-fine-tuned DNABERT-2 | Supervised fine-tuning pipeline → smoke evaluation | A self-fine-tuned DNABERT-2 promoter classifier has been implemented and trained end-to-end, enabling direct frozen-vs-fine-tuned mechanistic comparison. | Full multi-seed training, corrected reload/evaluation path, and repetition of the frozen causal assay suite. |

Evidence levels are not interchangeable. A strong probe does not establish necessity; necessity does not
establish mediation; a causal model mechanism does not automatically establish a biological mechanism.

**Two model states are tracked in this study.** The primary frozen experiments use the pretrained
DNABERT-2 encoder accessed through the repository adapter and a fixed downstream Layer-2 masked-mean
probe. The second branch self-fine-tunes the same DNABERT-2 family on promoter classification and is
intended for a controlled before/after mechanistic comparison. A result from the frozen encoder should
not be quoted as a result about the fine-tuned encoder, and vice versa.

## Current frozen-model findings

The corrected frozen pipeline changed the main predictive conclusion relative to the earlier run.

- The best frozen readout moved from **Layer 5 to Layer 2**.
- Held-out test balanced accuracy improved from approximately **0.735 to 0.810**.
- Held-out test ROC-AUC improved from approximately **0.811 to 0.892**.
- The corrected frozen DNABERT representation now outperforms the strongest tested k-mer baseline.
- Because the endpoint changed, causal analyses tied to the older Layer-5 endpoint must not be treated as
  controlling evidence for the current Layer-2 claim.
- The current causal package therefore fixes Layer 2 as the readout endpoint and reruns localization,
  ablation, restoration, branch, donor-matrix, and motif-perturbation assays against that endpoint.

These results license a representation-level claim: **pretrained DNABERT-2 contains promoter-relevant
information before task-specific fine-tuning.** They do not yet license the stronger statement that the
model has learned a compact or biologically faithful promoter-recognition circuit.

## TATA-like motif analysis

The frozen-model motif analysis tests whether promoter-relevant behavior is specifically associated with
canonical or TATA-like sequence patterns rather than with arbitrary local corruption.

The current exploratory validation stratification includes:

- exact `TATAAA`;
- `TATA[AT][AT]` TATA-like motifs;
- broader `TATA` matches;
- motif start-position cohorts;
- matched corruption comparisons.

Selected exploratory validation statistics from the current assay include:

| Cohort | n | Mean excess effect | Fraction > 0 | Paired-test interpretation |
| --- | ---: | ---: | ---: | --- |
| All tested TATA-like cases | 71 | 0.0741 | 0.521 | Not statistically conclusive |
| `TATAAA` | 48 | 0.1445 | 0.5625 | Suggestive but not statistically conclusive |
| `TATATA` | 14 | 0.0265 | — | Not statistically conclusive |
| `TATAAT` | 4 | -0.5485 | — | Too small for a stable interpretation |
| `TATATT` | 5 | 0.0287 | — | Too small for a stable interpretation |

The `TATAAA` subgroup shows the strongest positive exploratory effect among the listed motifs, but the
current evidence does not justify the claim that DNABERT-2 uses a TATA-specific causal mechanism.
Replication, larger cohorts, held-out confirmation, and localization of the effect to model components
are still required.

## Fine-tuning branch

A separate self-fine-tuning branch asks:

> What changes when the same DNABERT-2 encoder is optimized directly for promoter classification?

The implementation uses the DNABERT adapter, masked-mean pooling, a two-class classification head, dropout 0.1, cross-entropy loss, and full encoder fine-tuning.

The initial seed-42 smoke run verified that the training and checkpoint pipeline works. Because it used a deliberately small configuration, these metrics are developmental rather than final:

| Split      | Balanced accuracy | ROC-AUC | PR-AUC |     F1 |
| ---------- | ----------------: | ------: | -----: | -----: |
| Validation |            0.5278 |  0.7758 | 0.7346 | 0.6222 |
| Test       |            0.5108 |  0.6912 | 0.7076 | 0.6292 |

The final comparison will test whether supervised fine-tuning preserves, strengthens, concentrates, or rewires the promoter-relevant mechanism observed in the frozen encoder.

## Method

The DNABERT promoter study follows the same evidence progression used throughout this repository:

1. Define promoter classification as the scoped behavior and lock the dataset split.
2. Identify and freeze the strongest hidden-state readout; the current frozen endpoint is Layer 2 with masked-mean pooling.
3. Compare against simple sequence baselines.
4. Generate candidate layers, heads, MLPs, and sequence regions.
5. Test necessity using controlled ablations.
6. Test mediation and information flow using restoration, branch, and donor interventions.
7. Test biological hypotheses, including TATA-like motif dependence, separately from model-level causal claims.
8. Replicate across held-out examples, seeds, and frozen/fine-tuned model states.

A predictive representation is not automatically causal, and a causal model mechanism is not automatically a biological mechanism.

## Study status

| Study                                  | Status                                      | Purpose                                           |
| -------------------------------------- | ------------------------------------------- | ------------------------------------------------- |
| Frozen layer probing                   | **complete, corrected**                     | Locks Layer 2 as the current frozen endpoint.     |
| Sequence baselines                     | **complete**                                | Compares frozen DNABERT against k-mer baselines.  |
| Causal localization and ablation       | **implemented**                             | Tests component necessity and pathway dependence. |
| Activation restoration and donor tests | **implemented**                             | Tests mediation and recoverable information flow. |
| TATA perturbation                      | **exploratory validation complete**         | Tests motif-specific promoter dependence.         |
| Self-fine-tuning                       | **pipeline implemented; full runs ongoing** | Enables frozen-vs-fine-tuned comparison.          |
| Fine-tuned causal replication          | **pending**                                 | Repeats the frozen assay suite after fine-tuning. |

## Evidence boundaries

The current evidence supports the following distinction:

* **Predictive:** a representation supports promoter classification.
* **Necessary:** intervening on a component reduces the behavior.
* **Mediating:** restoring or transplanting information recovers the behavior.
* **Mechanistic:** components can be linked into a causally supported information-flow account.
* **Biological:** the mechanism's information has been identified with a specific biological feature.

The DNABERT study currently has strong predictive evidence and an expanding causal-intervention record. Biological-semantic claims remain conservative.

## Current claim status

The following claims are currently supported:

1. Frozen DNABERT-2 contains promoter-discriminative information.
2. Under the corrected pipeline, Layer 2 is the strongest tested frozen readout.
3. The Layer-2 probe reaches approximately **0.810 balanced accuracy** and **0.892 ROC-AUC** on the held-out test split.
4. The corrected frozen representation outperforms the strongest tested k-mer baseline.
5. Causal assays have been implemented for localization, ablation, restoration, pathway dependence, donor tests, and motif sensitivity.
6. TATA perturbation remains exploratory and does not yet establish a TATA-specific mechanism.
7. The self-fine-tuning pipeline is operational, but the final fine-tuned mechanistic comparison is still in progress.

The current evidence does **not** yet support claims that DNABERT-2 has a fully mapped promoter-recognition circuit, that a specific component uniquely implements promoter detection, or that the model has learned a biologically validated TATA-box mechanism.

## What remains to do

The main remaining work is to:

1. complete the planned full fine-tuning runs across seeds;
2. lock the final fine-tuned checkpoints;
3. repeat the frozen causal assay suite on the fine-tuned model;
4. compare layer localization, causal effects, restoration, donor tests, and motif dependence before versus after fine-tuning;
5. update the final claims and manuscript only after this matched comparison is complete.

## DNABERT repository structure

```text
gfm-circuit-discovery/
├── configs/dnabert_promoter/               # experiment configuration
├── src/gfmcd/
│   ├── adapters/dnabert.py                 # DNABERT-2 adapter
│   └── dnabert/                            # dataset and perturbation utilities
├── scripts/dnabert_promoter/               # probing, causal, faithfulness, and motif experiments
├── studies/
│   ├── dnabert_promoter/
│   └── dnabert_promoter_mechanism/         # preregistration, decisions, claims, results
├── results/dnabert_promoter/
│   ├── shared/                             # splits, hashes, leakage checks
│   ├── frozen/                             # layer probes and baselines
│   ├── behavioral/                         # corruption analyses
│   ├── causal/                             # ablation, restoration, localization
│   ├── frozen_claim_tests/                 # faithfulness, donor, biological tests
│   ├── figures/
│   └── provenance/
├── scripts/crossmodel/                     # DNABERT cross-model assays
├── results/origin-crossmodel/
└── tests/                                  # adapter, dataset, perturbation tests
```

The pipeline follows:

**configuration → cohort construction → frozen probing → behavioral corruption → causal intervention → faithfulness and biological tests → claim-level results.**



## Start here

For the DNABERT study:

1. Read this README for the current evidence boundary.
2. Confirm the dataset split hashes before running any experiment.
3. Reproduce the corrected frozen layer sweep and verify Layer 2 as the readout endpoint.
4. Reproduce the frozen held-out evaluation and k-mer baseline comparison.
5. Run the Layer-2 causal assay suite.
6. Review the donor-matrix and TATA perturbation artifacts separately; neither should be promoted beyond
   its current evidence level.
7. Run the full self-fine-tuning seeds.
8. Lock the final fine-tuned checkpoint set.
9. Repeat the same causal assays without changing the assay definitions.
10. Compare frozen and fine-tuned mechanisms using matched metrics, cohorts, and interventions.
11. Only then update the manuscript and project-level claim ledger.

## Reproducibility rules

Run experiments from the repository root and keep the following invariants fixed within a claim:

- model checkpoint;
- tokenizer;
- dataset version;
- train/validation/test split;
- split hashes;
- sequence length;
- pooling/readout definition;
- intervention definition;
- random seed;
- evaluation metric;
- result-directory provenance.

Result-producing scripts should write immutable outputs together with sufficient metadata to reconstruct
the run. New experiments must not silently overwrite an existing scientific result.

When an endpoint, dataset, tokenizer, or implementation bug changes, all downstream causal results tied
to the superseded configuration must be treated as historical until rerun under the corrected setup.

## Development

Typical environment checks should confirm:

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

For GPU runs, verify the intended CUDA device before launching a long experiment.

The current self-fine-tuning environment has been run with an NVIDIA A100-class GPU. Exact environment
versions should be recorded with each final result package rather than inferred from the README.

## Interpretation principle

The purpose of this DNABERT extension is not simply to show that DNABERT can classify promoters.

The scientific question is whether the project's mechanistic-interpretability methodology can recover a
causally supported representation and information-flow account in a second genomic foundation model,
and whether supervised promoter fine-tuning preserves or rewires that account.

The target contribution is therefore:

> **Predictive representation → causal localization → information-flow test → biological hypothesis →
> frozen/fine-tuned comparison → cross-model comparison.**

Each arrow requires its own evidence. No later interpretation is licensed merely because an earlier
stage succeeds.
