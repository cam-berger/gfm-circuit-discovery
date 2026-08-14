# Fine-Tuned DNABERT Promoter Mechanistic Analysis

This directory contains the fine-tuned DNABERT experiments used to study how task-specific promoter fine-tuning changes the internal organization of promoter information and the causal role of early transformer components.

The main model analyzed here is:

```text
magicslabnu/DNABERT-2-finetuned-prom_300_all
```

The model is evaluated on the GUE promoter-classification task:

```text
GUE/prom_300_all
```

The goal of this analysis is not only to measure predictive performance, but to determine how promoter information is represented and causally implemented after task-specific fine-tuning.

---

## Research Question

The central question is:

> What happens to the promoter mechanism learned by pretrained DNABERT after task-specific promoter fine-tuning?

The fine-tuned experiments were designed to distinguish between several possibilities:

- the pretrained mechanism is preserved;
- the mechanism is strengthened;
- promoter information relocates to later layers;
- the mechanism becomes more distributed;
- the original mechanism is replaced;
- predictive performance improves without a corresponding mechanistic change.

The frozen and fine-tuned models are analyzed separately first. Their results are compared only after the fine-tuned experimental package is complete.

---

# Model

## Fine-Tuned Checkpoint

```text
magicslabnu/DNABERT-2-finetuned-prom_300_all
```

Checkpoint revision:

```text
7f2a5f5b8e7e6fffb2dcb396a77d1bc78b7f5f12
```

The checkpoint contains a two-class sequence-classification head:

```text
classifier.weight: (2, 768)
classifier.bias:   (2,)
```

The classifier head was verified to exactly match the checkpoint weights.

## Compatible DNABERT Implementation

The original DNABERT-2 dynamic implementation was not usable directly in the current Colab environment because of incompatibilities with modern Transformers meta-device initialization and the original Triton attention kernel.

The model was therefore instantiated using:

```text
quietflamingo/dnabert2-no-flashattention
```

revision:

```text
813031b2bf86d9e960a027e2734c908009f31601
```

The fine-tuned checkpoint state dictionary loaded into this implementation with:

```text
Missing keys:    0
Unexpected keys: 0
```

The classifier weights and biases matched the checkpoint exactly.

The implementation falls back to PyTorch attention rather than Triton/FlashAttention.

---

# Dataset

Dataset:

```text
leannmlindsey/GUE
```

Dataset revision:

```text
240df51a2cd2c0fb1d88efffaf86e8d1e477f932
```

Task:

```text
GUE/prom_300_all
```

Split sizes:

| Split | Examples |
|---|---:|
| Train | 47,356 |
| Validation | 5,920 |
| Test | 5,920 |

All sequences are exactly 300 nt long.

Class counts:

### Train

```text
label 0: 23,684
label 1: 23,672
```

### Validation

```text
label 0: 2,974
label 1: 2,946
```

### Test

```text
label 0: 2,940
label 1: 2,980
```

The label mapping was empirically validated:

```text
0 = non-promoter
1 = promoter
```

Promoter probability is therefore:

```python
softmax(logits)[:, 1]
```

---

# Dataset Provenance and Leakage Check

Small exact-sequence overlap exists between the official GUE splits:

```text
Train ∩ Dev  = 39
Train ∩ Test = 19
Dev ∩ Test   = 1
```

Only 19 official test examples overlap exactly with the training split.

A strict unseen sensitivity analysis excluding test sequences that occur in train or dev leaves 5,900 test examples.

Performance on the strict unseen split is nearly identical to the official test split, indicating that the reported fine-tuned performance is not materially explained by exact sequence duplication.

---

# Predictive Performance

## Full Official Test Split

| Metric | Value |
|---|---:|
| Accuracy | 0.935135 |
| Balanced Accuracy | **0.935205** |
| MCC | **0.870487** |
| ROC-AUC | **0.984785** |
| PR-AUC | **0.984351** |
| F1 | 0.934871 |
| Precision | 0.945130 |
| Recall | 0.924832 |
| Specificity | 0.945578 |

Confusion matrix:

```text
[[2780, 160],
 [ 224, 2756]]
```

Rows are true labels and columns are predicted labels.

## Strict Unseen Test Split

After removing exact test sequences occurring in train or dev:

```text
n = 5,900
```

| Metric | Value |
|---|---:|
| Accuracy | 0.935085 |
| Balanced Accuracy | 0.935136 |
| MCC | 0.870391 |
| ROC-AUC | 0.984781 |
| PR-AUC | 0.984279 |
| F1 | 0.934675 |

---

# Layer-Wise Promoter Localization

The fine-tuned model was analyzed using masked-mean pooled hidden representations from all 12 transformer layers.

For each layer:

1. hidden representations were extracted from the fine-tuned transformer;
2. non-padding token activations were reconstructed per example;
3. masked-mean pooled representations were computed;
4. a fixed linear probe was trained independently for each layer.

Probe:

```python
StandardScaler()
LogisticRegression(
    solver="liblinear",
    max_iter=3000,
    random_state=42
)
```

The layer was selected using validation balanced accuracy with validation ROC-AUC as a tie-breaker.

## Layer-Wise Results

| Layer | Dev BA | Test BA | Test MCC | Test ROC-AUC |
|---:|---:|---:|---:|---:|
| 0 | 0.902214 | 0.895268 | 0.790532 | 0.957253 |
| 1 | 0.917771 | 0.906679 | 0.813668 | 0.966726 |
| 2 | 0.918309 | 0.908709 | 0.817713 | 0.969769 |
| 3 | 0.929778 | 0.920522 | 0.841432 | 0.975097 |
| 4 | 0.928935 | 0.928098 | 0.856823 | 0.975826 |
| 5 | 0.939072 | 0.928477 | 0.857239 | 0.978842 |
| 6 | 0.938717 | 0.932710 | 0.865643 | 0.978078 |
| 7 | 0.941921 | 0.932882 | 0.865967 | 0.979318 |
| 8 | 0.940221 | **0.935252** | **0.870681** | 0.980092 |
| 9 | 0.940382 | 0.934946 | 0.869933 | 0.983177 |
| 10 | **0.942919** | 0.932418 | 0.864861 | **0.983282** |
| 11 | 0.941742 | 0.931726 | 0.863528 | 0.983004 |

Selected endpoint:

```text
Layer 10
```

Selected-layer metrics:

```text
Dev BA:      0.942919
Dev MCC:     0.885828
Dev ROC-AUC: 0.985708

Test BA:      0.932418
Test MCC:     0.864861
Test ROC-AUC: 0.983282
```

This is correlational evidence about linear accessibility of promoter information. It does not by itself identify the causal mechanism.

---

# Targeted Early-Component Ablation

A balanced causal cohort was constructed from the canonical test split:

```text
256 examples
128 non-promoters
128 promoters
```

The cohort was sampled deterministically with seed 42.

The following Layer-0 components were targeted:

```text
L0H8
L0_MLP
L0H8 + L0_MLP
```

## L0H8 Intervention

DNABERT hidden size: 768  
Number of attention heads: 12  
Per-head dimension: 64

Head 8 occupies:

```text
[512:576]
```

in the concatenated attention-head tensor immediately before the attention output projection.

`L0H8` was ablated by zeroing this slice before `attention.output.dense`.

## L0_MLP Intervention

The Layer-0 MLP branch was ablated at:

```text
layer0.mlp.wo output
```

before residual addition and LayerNorm.

## Ablation Results

| Condition | Balanced Accuracy | MCC | ROC-AUC |
|---|---:|---:|---:|
| Baseline | **0.957031** | 0.914090 | 0.991272 |
| L0H8 ablated | **0.511719** | 0.108893 | 0.685669 |
| L0_MLP ablated | **0.816406** | 0.671858 | 0.900757 |
| L0H8 + L0_MLP ablated | **0.503906** | 0.036298 | 0.665222 |

Balanced-accuracy changes:

```text
L0H8:        -0.445312
L0_MLP:      -0.140625
L0H8 + MLP:  -0.453125
```

The combined ablation reduces performance to approximately chance.

---

# Class-Conditional Ablation Effects

For `L0H8`:

```text
Non-promoters:
Mean ΔP(promoter): +0.819336
Predicted promoter after ablation: 97.7%

Promoters:
Mean ΔP(promoter): +0.003715
Predicted promoter after ablation: 100%
```

For `L0_MLP`:

```text
Non-promoter mean ΔP(promoter): +0.346258
Promoter mean ΔP(promoter):     +0.002347
```

For the combined early ablation:

```text
Non-promoter mean ΔP(promoter): +0.840735
```

This indicates that the early pathway is especially important for suppressing inappropriate promoter predictions on non-promoter sequences.

---

# Activation Restoration

Activation restoration was performed on the same locked 256-example causal cohort.

Procedure:

1. cache clean `L0H8` and `L0_MLP` activations;
2. ablate both components;
3. restore only `L0H8`;
4. restore only `L0_MLP`;
5. restore both.

Restoring both components reproduces the baseline logits exactly:

```text
Max baseline vs restore-both logit difference:
0.000000000000
```

## Restoration Results

| Condition | Balanced Accuracy | MCC | ROC-AUC |
|---|---:|---:|---:|
| Baseline | **0.957031** | 0.914090 | 0.991272 |
| Both ablated | **0.503906** | 0.036298 | 0.665222 |
| Restore L0H8 | **0.816406** | 0.671858 | 0.900757 |
| Restore L0_MLP | **0.925781** | 0.853675 | 0.979858 |
| Restore both | **0.957031** | 0.914090 | 0.991272 |

Recovery fractions:

```text
Restore L0H8:    0.553
Restore L0_MLP:  0.880
Restore both:    1.000
```

---

# Component Donor Matrix

A donor-swapping experiment was used to test whether early component activations carry class-specific information.

Same-class offsets:

```text
1, 3, 5, 7
```

Opposite-class offsets:

```text
0, 3, 5, 7
```

Total donor/recipient pairs:

```text
2,048
```

## Nucleotide-Aligned Donor Swapping

Because DNABERT tokenization is variable-length, donor and recipient activations were aligned using tokenizer nucleotide offset mappings rather than raw token index.

For each recipient token:

1. identify its nucleotide interval;
2. find donor tokens overlapping that interval;
3. compute an overlap-weighted average of donor activations;
4. insert the aligned donor activation into the recipient computation.

Special tokens are mapped separately:

```text
CLS → CLS
SEP → SEP
```

## Donor Compatibility Results

| Pair Type | Component | Mean Abs ΔP | Median Abs ΔP | Prediction Flip Fraction |
|---|---|---:|---:|---:|
| Same class | L0H8 | 0.092880 | 0.005874 | 0.088867 |
| Same class | L0_MLP | 0.090191 | 0.006461 | 0.072266 |
| Same class | Both | 0.114631 | 0.013116 | 0.098633 |
| Opposite class | L0H8 | 0.175941 | 0.033544 | 0.167969 |
| Opposite class | L0_MLP | **0.773006** | **0.952331** | **0.804688** |
| Opposite class | Both | **0.815616** | **0.967384** | **0.850586** |

## Statistical Separation

| Component | Opposite / Same Mean Disruption Ratio | P(opposite > same) |
|---|---:|---:|
| L0H8 | **1.89×** | 0.644 |
| L0_MLP | **8.57×** | 0.933 |
| Both | **7.12×** | 0.943 |

This provides causal evidence that the early components carry class-specific promoter information rather than only generic activation magnitude.

---

# TATA Motif Perturbation

Motif family:

```text
TATA[AT][AT]
```

Canonical motif:

```text
TATAAA
```

Among 2,980 test promoters, 354 contain at least one TATA-like motif.

Locked target rule:

```text
earliest motif occurrence per promoter
```

Locked motif counts:

| Motif | n |
|---|---:|
| TATAAA | 150 |
| TATAAT | 63 |
| TATATA | 69 |
| TATATT | 72 |

Baseline recall on the 354 eligible promoters:

```text
0.844633
```

Mean promoter probability:

```text
0.822061
```

## TATA Perturbation Design

For each eligible promoter:

- motif perturbation: replace selected 6-nt motif with `CGCGCG`;
- control perturbation: replace a mirrored 6-nt control window with `CGCGCG`.

Primary effect:

```text
motif_damage   = baseline_probability - motif_perturbed_probability
control_damage = baseline_probability - control_perturbed_probability

excess_motif_damage =
    motif_damage - control_damage
```

## Overall TATA Results

```text
n = 354
Mean motif damage: +0.025684
Mean control damage: +0.047558
Mean excess motif damage: -0.021874
Median excess motif damage: -0.000151
Fraction motif more damaging than control: 0.483051
Wilcoxon p = 0.162760
Bootstrap 95% CI = [-0.047613, +0.003288]
```

Therefore, the fine-tuned model shows no clear positive TATA-specific dependence under this perturbation/control design.

## Canonical TATAAA

```text
n = 150
Mean excess motif damage: -0.030441
Median excess motif damage: -0.004788
Fraction positive: 0.373333
Wilcoxon p = 0.001647
Bootstrap mean-effect CI = [-0.072282, +0.011993]
```

The Wilcoxon test detects a directional distributional difference, but the mean-effect bootstrap interval crosses zero and the effect direction is negative. This should not be interpreted as positive TATA dependence.

---

# Final Fine-Tuned Results Summary

| Experiment | Result |
|---|---|
| Full test performance | BA = **0.935205**, MCC = **0.870487**, ROC-AUC = **0.984785** |
| Layer localization | Best validation layer = **L10** |
| L0H8 ablation | BA **0.957031 → 0.511719** |
| L0_MLP ablation | BA **0.957031 → 0.816406** |
| Combined early ablation | BA **0.957031 → 0.503906** |
| Activation restoration | Restore both → **exact baseline recovery** |
| Donor compatibility | Opposite/same disruption = **1.89× H8, 8.57× MLP, 7.12× combined** |
| TATA perturbation | Mean excess = **−0.021874**, p = **0.162760** |

---

# Interpretation of the Fine-Tuned Model

The fine-tuned model exhibits a notable separation between representational localization and causal dependence.

Promoter information is most linearly accessible at Layer 10.

However, early Layer-0 components remain critically important to the model's actual computation.

In particular:

- `L0H8` is necessary for maintaining correct promoter/non-promoter discrimination;
- `L0_MLP` carries strongly class-specific information;
- ablating both components reduces classification performance to chance;
- restoring both components exactly reconstructs baseline outputs;
- opposite-class donor activations strongly disrupt the computation;
- the fine-tuned model does not show clear positive TATA-specific dependence under the paired perturbation design.

This suggests that task-specific fine-tuning can make promoter information increasingly linearly readable in later layers while preserving strong causal dependence on an early class-specific pathway.

---

# Figures

Final fine-tuned figures are saved under:

```text
results/final_finetuned_summary/figures/
```

## Figure 1 — Layer Localization

```text
figure1_layer_localization.png
figure1_layer_localization.pdf
```

## Figure 2 — Targeted Ablation

```text
figure2_targeted_ablation.png
figure2_targeted_ablation.pdf
```

## Figure 3 — Activation Restoration

```text
figure3_activation_restoration.png
figure3_activation_restoration.pdf
```

## Figure 4 — Donor Compatibility

```text
figure4_donor_compatibility.png
figure4_donor_compatibility.pdf
```

## Figure 5 — TATA Perturbation

```text
figure5_tata_perturbation.png
figure5_tata_perturbation.pdf
```

---

# Key Output Artifacts

## Full Test Evaluation

```text
results/hf_finetuned_test/
```

## Layer Localization

```text
results/hf_finetuned_layer_localization/
```

## Targeted Ablation

```text
results/hf_finetuned_causal/targeted_ablation/
```

## Activation Restoration

```text
results/hf_finetuned_causal/activation_restoration/
```

## Component Donor Matrix

```text
results/hf_finetuned_causal/component_donor_matrix/
```

## TATA Perturbation

```text
results/hf_finetuned_tata/
```

## Consolidated Fine-Tuned Summary

```text
results/final_finetuned_summary/
```

---

# Evidence Types

### Predictive evidence

- full test performance;
- strict unseen evaluation.

### Correlational / representational evidence

- layer-wise linear probing.

### Causal evidence

- targeted ablation;
- activation restoration;
- component donor swapping.

### Biological sequence-perturbation evidence

- TATA motif perturbation.

This distinction is important when translating results into manuscript claims.

---

# Main Fine-Tuned Takeaway

The fine-tuned DNABERT promoter classifier achieves strong predictive performance and makes promoter information increasingly linearly accessible in late transformer layers.

At the same time, the model remains strongly dependent on a compact early pathway involving:

```text
L0H8
L0_MLP
```

Ablation, restoration, and donor-swapping experiments provide causal evidence that these components carry information required for promoter discrimination.

The resulting picture is therefore not simply that fine-tuning moves the mechanism from Layer 0 to Layer 10.

Instead:

> Fine-tuning makes promoter information highly readable in late layers while early Layer-0 components remain causally necessary for the computation.

The next analysis should compare these results directly with frozen DNABERT to determine how task-specific fine-tuning transforms the pretrained promoter mechanism.
