# Frozen vs Fine-Tuned DNABERT-2: Mechanistic Comparison

## Overview

This document compares the **frozen DNABERT-2 promoter analysis** with the **fine-tuned DNABERT-2 promoter mechanistic analysis**.

The goal of the comparison is not only to ask whether fine-tuning improves promoter classification, but to determine how task-specific optimization changes the internal organization of promoter-relevant information.

The comparison is organized around five questions:

1. Does fine-tuning improve predictive performance?
2. Does the strongest promoter representation move to a different layer?
3. Are early components still causally necessary after fine-tuning?
4. Do early components still carry class-specific information?
5. Does fine-tuning increase, preserve, or reduce TATA-specific biological dependence?

The overall result is:

> **Fine-tuning substantially improves promoter prediction and relocates the strongest linearly accessible promoter representation from early Layer 2 to late Layer 10, while preserving strong causal dependence on an early Layer-0 pathway.**

This is more consistent with **representational relocation and downstream reorganization** than with wholesale replacement of the pretrained promoter mechanism.

---

# 1. Models Compared

## Frozen DNABERT-2

The frozen analysis uses the pretrained DNABERT-2 encoder with a fixed downstream masked-mean linear probe.

The corrected frozen endpoint is:

```text
Layer 2
```

The earlier Layer-5 endpoint was superseded after the frozen pipeline was corrected and rerun.

The frozen model therefore represents the promoter information available **before task-specific supervised fine-tuning**.

## Fine-Tuned DNABERT-2

The completed fine-tuned analysis uses:

```text
magicslabnu/DNABERT-2-finetuned-prom_300_all
```

on:

```text
GUE/prom_300_all
```

with the compatible no-FlashAttention implementation:

```text
quietflamingo/dnabert2-no-flashattention
```

The fine-tuned checkpoint is evaluated independently before being compared with the frozen model.

---

# 2. Evidence Boundaries

The comparison distinguishes several forms of evidence.

### Predictive evidence

Measures whether the model or representation supports promoter classification.

Examples:

- balanced accuracy;
- ROC-AUC;
- MCC;
- PR-AUC.

### Correlational / representational evidence

Measures where promoter information is linearly recoverable.

Example:

- layer-wise linear probing.

### Causal evidence

Measures whether a model component is necessary or whether restoring/transplanting an activation changes behavior.

Examples:

- targeted ablation;
- activation restoration;
- donor swapping.

### Biological sequence-perturbation evidence

Measures sensitivity to biologically motivated sequence changes.

Example:

- TATA motif perturbation.

A strong linear probe does not by itself establish a causal circuit, and a causal model mechanism does not automatically establish a biologically faithful promoter mechanism.

---

# 3. Predictive Performance

The corrected frozen Layer-2 probe reaches approximately:

```text
Balanced Accuracy: 0.810
ROC-AUC:            0.892
```

The fine-tuned classifier reaches:

```text
Balanced Accuracy: 0.935205
MCC:               0.870487
ROC-AUC:           0.984785
PR-AUC:            0.984351
F1:                0.934871
```

## Comparison

| Model state | Balanced Accuracy | ROC-AUC |
|---|---:|---:|
| Frozen DNABERT-2, Layer-2 probe | ~0.810 | ~0.892 |
| Fine-tuned DNABERT-2 | **0.935** | **0.985** |
| Difference | **+0.125** | **+0.093** |

Fine-tuning therefore produces a large predictive improvement.

The fine-tuned result also survives a strict unseen sensitivity analysis in which exact test sequences appearing in train or development splits are removed:

```text
Strict unseen test n = 5,900
Balanced Accuracy = 0.935136
ROC-AUC = 0.984781
```

The predictive improvement is therefore not materially explained by the small number of exact cross-split duplicates.

## Interpretation

The frozen model already contains substantial promoter-discriminative information before supervised task training.

Fine-tuning does not create promoter information entirely from scratch. Instead, supervised optimization substantially increases how accurately that information can be transformed into the promoter classification decision.

---

# 4. Layer-Wise Representational Localization

The most striking representational difference is the shift in the strongest layer-wise readout.

## Frozen model

The corrected frozen endpoint is:

```text
Layer 2
```

with approximately:

```text
Test Balanced Accuracy: 0.810
Test ROC-AUC:            0.892
```

## Fine-tuned model

The fine-tuned layer sweep uses the same probe family across all 12 transformer layers.

The endpoint is selected by:

```text
validation balanced accuracy
```

with validation ROC-AUC as a tie-breaker.

The selected fine-tuned endpoint is:

```text
Layer 10
```

with:

```text
Dev Balanced Accuracy:  0.942919
Dev MCC:                0.885828
Dev ROC-AUC:            0.985708

Test Balanced Accuracy: 0.932418
Test MCC:               0.864861
Test ROC-AUC:           0.983282
```

The fine-tuned layer profile is:

| Layer | Dev BA | Test BA | Test ROC-AUC |
|---:|---:|---:|---:|
| 0 | 0.902214 | 0.895268 | 0.957253 |
| 1 | 0.917771 | 0.906679 | 0.966726 |
| 2 | 0.918309 | 0.908709 | 0.969769 |
| 3 | 0.929778 | 0.920522 | 0.975097 |
| 4 | 0.928935 | 0.928098 | 0.975826 |
| 5 | 0.939072 | 0.928477 | 0.978842 |
| 6 | 0.938717 | 0.932710 | 0.978078 |
| 7 | 0.941921 | 0.932882 | 0.979318 |
| 8 | 0.940221 | 0.935252 | 0.980092 |
| 9 | 0.940382 | 0.934946 | 0.983177 |
| 10 | **0.942919** | 0.932418 | **0.983282** |
| 11 | 0.941742 | 0.931726 | 0.983004 |

## Main localization result

```text
Frozen strongest endpoint:     Layer 2
Fine-tuned selected endpoint:  Layer 10
Representational shift:        +8 layers
```

This provides strong evidence for:

> **representational relocation toward later layers after promoter fine-tuning.**

However, the probe result does not establish that the causal mechanism itself moved from L2 to L10.

The layer sweep measures where promoter information is most easily decoded, not necessarily where that information is generated or causally required.

---

# 5. Early Causal Components After Fine-Tuning

The frozen study already identified early components such as:

```text
L0H8
L0_MLP
L0H8 + L0_MLP
FULL_EARLY_GROUP
```

as candidate components in causal and donor-style analyses.

The fine-tuned experiments directly test whether two of these key early components remain necessary after supervised optimization.

A locked balanced causal cohort contains:

```text
256 examples
128 promoters
128 non-promoters
```

## Fine-tuned targeted ablation results

| Condition | Balanced Accuracy | MCC | ROC-AUC |
|---|---:|---:|---:|
| Baseline | **0.957031** | 0.914090 | 0.991272 |
| L0H8 ablated | **0.511719** | 0.108893 | 0.685669 |
| L0_MLP ablated | **0.816406** | 0.671858 | 0.900757 |
| L0H8 + L0_MLP ablated | **0.503906** | 0.036298 | 0.665222 |

Balanced-accuracy changes:

```text
L0H8:       -0.445312
L0_MLP:     -0.140625
Combined:   -0.453125
```

The combined ablation reduces performance to approximately chance.

## Interpretation

The layer-wise probe moves from L2 to L10, but the model still depends critically on an early Layer-0 computation.

Therefore, the fine-tuned model does not support the simple interpretation:

```text
fine-tuning moves the promoter circuit from early layers to L10
```

Instead, the evidence is more consistent with:

```text
early causal computation
        ↓
continued downstream processing
        ↓
late promoter-explicit representation
        ↓
classification head
```

---

# 6. Class-Conditional Role of the Early Pathway

The fine-tuned `L0H8` ablation is strongly asymmetric across classes.

## Non-promoters

```text
Mean ΔP(promoter) after L0H8 ablation: +0.819336
Predicted promoter after ablation:     97.7%
```

## Promoters

```text
Mean ΔP(promoter) after L0H8 ablation: +0.003715
Predicted promoter after ablation:     100%
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

## Interpretation

The early pathway is particularly important for suppressing inappropriate promoter predictions on non-promoter sequences.

This suggests that the mechanism does not merely add positive promoter evidence.

At least part of the early computation appears to contribute to **class discrimination**, especially the exclusion of non-promoter sequences.

---

# 7. Activation Restoration

Ablation establishes necessity, but restoration provides stronger causal evidence that the targeted activations contain information required for the behavior.

The fine-tuned restoration experiment proceeds by:

1. caching clean `L0H8` and `L0_MLP` activations;
2. ablating both components;
3. restoring only `L0H8`;
4. restoring only `L0_MLP`;
5. restoring both.

## Results

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

Restoring both components reconstructs the baseline logits exactly:

```text
Maximum baseline vs restore-both logit difference:
0.000000000000
```

## Interpretation

The early components are not merely associated with correct promoter classification.

Their clean activations are causally sufficient to recover the corresponding computation after targeted corruption.

The restoration results also indicate interaction between the head and MLP branches.

Although independent `L0H8` ablation produces the larger performance collapse, restoring the MLP branch in the jointly ablated model recovers more of the baseline behavior than restoring H8 alone.

The early mechanism should therefore not be reduced to a single isolated attention head.

---

# 8. Donor Compatibility and Class-Specific Information

The fine-tuned donor matrix tests whether early activations carry class-specific information.

Donor conditions use:

```text
Same-class offsets:     1, 3, 5, 7
Opposite-class offsets: 0, 3, 5, 7
```

Total donor/recipient pairs:

```text
2,048
```

Because DNABERT uses variable-length tokenization, donor activations are aligned to recipients through tokenizer nucleotide offset mappings rather than raw token indices.

## Collapsed donor results

| Pair type | Component | Mean Abs ΔP | Median Abs ΔP | Prediction Flip Fraction |
|---|---|---:|---:|---:|
| Same class | L0H8 | 0.092880 | 0.005874 | 0.088867 |
| Same class | L0_MLP | 0.090191 | 0.006461 | 0.072266 |
| Same class | Both | 0.114631 | 0.013116 | 0.098633 |
| Opposite class | L0H8 | 0.175941 | 0.033544 | 0.167969 |
| Opposite class | L0_MLP | **0.773006** | **0.952331** | **0.804688** |
| Opposite class | Both | **0.815616** | **0.967384** | **0.850586** |

## Opposite-class versus same-class disruption

| Component | Opposite / Same Mean Disruption | P(opposite > same) |
|---|---:|---:|
| L0H8 | **1.89×** | 0.644 |
| L0_MLP | **8.57×** | 0.933 |
| Both | **7.12×** | 0.943 |

## Interpretation

Same-class donor activations are generally tolerated.

Opposite-class activations are substantially more disruptive, especially for the MLP branch and combined pathway.

This supports the claim that the early pathway contains **class-specific promoter information**, rather than merely providing generic activation magnitude or numerical support.

The MLP branch appears particularly strongly class-conditioned after fine-tuning.

---

# 9. TATA-Like Biological Perturbation

The frozen and fine-tuned TATA analyses should be compared cautiously because the assays are not perfectly matched.

## Frozen exploratory result

The frozen README reports the following exploratory validation statistics:

| Frozen cohort | n | Mean excess effect | Fraction > 0 |
|---|---:|---:|---:|
| All tested TATA-like cases | 71 | **+0.0741** | 0.521 |
| TATAAA | 48 | **+0.1445** | 0.5625 |
| TATATA | 14 | +0.0265 | — |
| TATAAT | 4 | -0.5485 | — |
| TATATT | 5 | +0.0287 | — |

The frozen analysis explicitly treats these results as exploratory and not sufficient for a strong motif-specific biological mechanism claim.

## Fine-tuned result

The completed fine-tuned analysis uses:

```text
354 TATA-positive promoters
```

with a paired motif-versus-control perturbation.

Overall:

```text
Mean motif damage:              +0.025684
Mean control damage:            +0.047558
Mean excess motif damage:       -0.021874
Median excess motif damage:     -0.000151
Fraction motif > control:        0.483051
Wilcoxon p:                      0.162760
Bootstrap 95% CI:               [-0.047613, +0.003288]
```

Canonical `TATAAA`:

```text
n:                              150
Mean excess motif damage:       -0.030441
Median excess motif damage:     -0.004788
Fraction positive:               0.373333
Wilcoxon p:                      0.001647
Bootstrap mean-effect CI:       [-0.072282, +0.011993]
```

## Comparison

| Model state | TATA-like effect |
|---|---|
| Frozen | Weak positive exploratory signal |
| Fine-tuned | No clear positive TATA-specific dependence |

The frozen all-TATA estimate is approximately:

```text
+0.074
```

whereas the fine-tuned estimate is:

```text
-0.022
```

For canonical `TATAAA`:

```text
Frozen exploratory mean excess:    +0.145
Fine-tuned mean excess:             -0.030
```

## Interpretation boundary

This should **not** be presented as proof that fine-tuning causes DNABERT to forget the TATA box.

The frozen and fine-tuned TATA analyses differ in cohort size and perturbation design, so the difference is not a perfectly matched before/after experiment.

The appropriate conclusion is:

> The frozen analysis contained a weak positive TATA-associated signal, whereas the completed fine-tuned assay does not show positive TATA-specific dependence.

This raises the possibility that fine-tuning emphasizes broader discriminative sequence features rather than increasing reliance on a canonical TATA motif.

---

# 10. Summary Table

| Question | Frozen DNABERT-2 | Fine-Tuned DNABERT-2 | Interpretation |
|---|---|---|---|
| Predictive performance | BA ~0.810, AUC ~0.892 | BA **0.935**, AUC **0.985** | Fine-tuning strongly improves promoter discrimination |
| Strongest linear readout | **L2** | **L10** | Promoter representation relocates later |
| Early causal components | Candidate early components supported by frozen causal assays | `L0H8` and `L0_MLP` remain strongly necessary | Early computation is not eliminated |
| Combined early ablation | Frozen causal package supports early pathway testing | BA **0.957 → 0.504** | Fine-tuned classifier remains highly early-pathway dependent |
| Restoration | Implemented in frozen causal package | Restore both gives exact baseline recovery | Early activations causally mediate behavior |
| Donor compatibility | Early donor components tested | Opposite-class swaps strongly disruptive | Fine-tuned early pathway carries class-specific information |
| TATA dependence | Weak positive exploratory effect | No clear positive effect | Biological motif dependence is not strengthened |

---

# 11. What Changed After Fine-Tuning?

The evidence supports several changes.

## 11.1 Predictive strengthening

Fine-tuning increases balanced accuracy by approximately:

```text
+0.125
```

and ROC-AUC by approximately:

```text
+0.093
```

relative to the corrected frozen Layer-2 probe.

## 11.2 Representational relocation

The strongest linear readout changes from:

```text
L2 → L10
```

This is the clearest evidence that supervised optimization reorganizes promoter information across the network.

## 11.3 Greater downstream task alignment

The fine-tuned layer profile becomes increasingly promoter-readable through the middle and late transformer layers.

This suggests that supervised training progressively transforms early information into a more explicit task-aligned representation.

## 11.4 Persistent early causal dependence

Despite the late-layer readout shift, `L0H8` and `L0_MLP` remain critical.

Therefore, fine-tuning does not appear to discard the early computation and replace it entirely with a new late-layer mechanism.

## 11.5 Strong class-specific early states

The donor matrix shows especially strong class conditioning in the early MLP branch.

Opposite-class `L0_MLP` transplantation is approximately:

```text
8.57×
```

more disruptive than same-class transplantation.

This suggests that fine-tuning may sharpen or reorganize the semantic content carried by the early pathway.

## 11.6 No evidence of stronger TATA-specific reliance

The fine-tuned perturbation assay does not show a positive TATA-specific causal effect.

Thus the predictive improvement cannot currently be explained as stronger reliance on the canonical TATA signal.

---

# 12. Preserved, Strengthened, Relocated, Redistributed, or Replaced?

The comparison was designed to classify the effect of fine-tuning into these possibilities.

## Preserved?

### Evidence

- early components implicated in the frozen analysis remain relevant targets after fine-tuning;
- `L0H8` and `L0_MLP` remain strongly causally necessary;
- restoration of their clean activations recovers the fine-tuned computation.

### Conclusion

**Substantial evidence for preservation of an early causal pathway.**

However, the exact frozen and fine-tuned component effects are not yet perfectly matched quantitatively on the same cohort and intervention protocol.

Therefore, the strongest safe statement is:

> an early causal pathway is preserved qualitatively.

---

## Strengthened?

Fine-tuning strongly increases predictive performance, but the current frozen README does not provide fully matched quantitative ablation and restoration effect sizes for the same cohort.

Therefore:

> **strengthening of the exact same causal pathway is not yet established.**

Predictive strengthening is clear.

Mechanistic strengthening requires a stricter matched frozen/fine-tuned comparison.

---

## Relocated?

The strongest linearly accessible representation changes from:

```text
Frozen L2
→
Fine-tuned L10
```

Therefore:

> **strong evidence for representational relocation.**

---

## Redistributed?

The fine-tuned representation becomes highly readable across several late layers rather than only at one isolated endpoint.

The continued dependence on early components plus increasing late-layer readability is consistent with downstream redistribution or propagation of promoter information.

Therefore:

> **moderate-to-strong evidence for downstream redistribution.**

---

## Replaced?

If the pretrained pathway were completely replaced, early components would not be expected to remain critically necessary in the fine-tuned model.

Instead:

```text
L0H8 ablation       → BA 0.512
L0H8 + MLP ablation → BA 0.504
```

and restoring both early components returns the model exactly to baseline.

Therefore:

> **the evidence argues against complete mechanism replacement.**

---

# 13. Best Current Mechanistic Description

The most defensible interpretation is:

> **Fine-tuning preserves a causally important early promoter pathway while reorganizing how its information is propagated and represented downstream.**

Another concise formulation is:

> **The mechanism is preserved upstream but reorganized downstream.**

The fine-tuned system is therefore better described as:

```text
Sequence
   ↓
early promoter-relevant computation
   ↓
L0H8 + L0_MLP
   │
   │ causally necessary
   ↓
progressive supervised transformation
   ↓
middle / late layers
   ↓
L10 promoter-explicit representation
   ↓
classification head
```

rather than:

```text
old frozen mechanism disappears
   ↓
new independent L10 mechanism
```

---

# 14. Main Scientific Finding

The central result of the frozen-versus-fine-tuned comparison is:

> **Task-specific fine-tuning shifts where promoter information is linearly explicit without eliminating the early components required to compute that information.**

This reveals an important distinction between:

```text
where information is readable
```

and:

```text
where information is causally generated or carried
```

In frozen DNABERT-2, promoter information is most strongly recoverable from an early Layer-2 endpoint.

After supervised fine-tuning, promoter information becomes most strongly validation-recoverable at Layer 10.

Yet the fine-tuned classifier still critically depends on Layer-0 components.

This means that mechanistic localization based only on probing would have produced an incomplete story.

---

# 15. Paper-Ready Claim

A conservative manuscript claim is:

> **Task-specific promoter fine-tuning substantially increases DNABERT-2 predictive performance and shifts the strongest linearly recoverable promoter representation from an early frozen Layer-2 endpoint to late Layer 10. Despite this representational relocation, targeted ablation, activation restoration, and donor interventions demonstrate continued causal dependence on an early Layer-0 pathway involving L0H8 and L0_MLP. These results suggest that fine-tuning reorganizes and propagates promoter information through the network rather than simply replacing the pretrained early computation.**

Short version:

> **Fine-tuning relocates promoter representations without eliminating the early causal pathway.**

Biological qualifier:

> **The completed fine-tuned TATA perturbation assay does not provide evidence for increased TATA-specific causal reliance.**

---

# 16. Claims That Should Not Yet Be Made

The current comparison does **not** justify the following statements:

### Do not claim:

```text
"The exact frozen circuit is unchanged after fine-tuning."
```

The current experiments support qualitative preservation of an early causal pathway, but not perfect quantitative identity of the full mechanism.

### Do not claim:

```text
"Fine-tuning moves the causal circuit from Layer 2 to Layer 10."
```

Layer probing measures representational accessibility, not causal implementation.

### Do not claim:

```text
"Fine-tuning strengthens L0H8 by X amount."
```

A fully matched frozen/fine-tuned intervention comparison is needed for quantitative strengthening claims.

### Do not claim:

```text
"Fine-tuning removes TATA biology."
```

The TATA assays are not perfectly matched across model states.

### Do not claim:

```text
"L0H8 alone is the complete promoter mechanism."
```

The restoration and donor results indicate important interaction with the MLP branch.

---

# 17. Current Classification

| Mechanistic possibility | Current evidence |
|---|---|
| Predictive strengthening | **Strongly supported** |
| Representational relocation | **Strongly supported** |
| Early causal-pathway preservation | **Strong qualitative support** |
| Downstream redistribution | **Supported** |
| Exact mechanistic strengthening | **Not yet established** |
| Complete mechanism replacement | **Evidence argues against** |
| Increased TATA-specific dependence | **Not supported** |

---

# 18. Figures for the Comparison

The completed fine-tuned figures can be used directly when constructing the final frozen-versus-fine-tuned presentation.

## Fine-tuned figures

```text
results/final_finetuned_summary/figures/
```

### Figure 1

```text
figure1_layer_localization.png
figure1_layer_localization.pdf
```

Shows late-layer promoter localization after fine-tuning.

### Figure 2

```text
figure2_targeted_ablation.png
figure2_targeted_ablation.pdf
```

Shows causal necessity of `L0H8` and `L0_MLP`.

### Figure 3

```text
figure3_activation_restoration.png
figure3_activation_restoration.pdf
```

Shows recovery of the early computation.

### Figure 4

```text
figure4_donor_compatibility.png
figure4_donor_compatibility.pdf
```

Shows strong same-class versus opposite-class donor compatibility.

### Figure 5

```text
figure5_tata_perturbation.png
figure5_tata_perturbation.pdf
```

Shows the absence of a clear positive TATA-specific effect after fine-tuning.

A future comparison figure should directly overlay the frozen and fine-tuned layer-localization curves once the frozen per-layer table is loaded into the same plotting environment.

---

# 19. Final Takeaway

The frozen model already contains promoter-discriminative information before task-specific training.

Fine-tuning then:

1. substantially improves promoter classification;
2. moves the strongest linearly decodable representation from early Layer 2 to late Layer 10;
3. preserves strong dependence on an early Layer-0 pathway;
4. produces highly class-specific early activations;
5. propagates promoter information into increasingly task-explicit late representations;
6. does not produce evidence for stronger TATA-specific causal reliance.

The resulting picture is not one of simple replacement.

Instead:

> **Fine-tuning appears to preserve an early promoter-relevant computation while reorganizing and amplifying how that information is represented downstream.**

This distinction between **causal computation** and **representational accessibility** is the central mechanistic result of the frozen-versus-fine-tuned DNABERT-2 comparison.
