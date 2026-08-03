# Notebook 6 — GC-Controlled Interventions

## Main result

Controlled GC-altering substitutions causally moved pooled
Layer-1 MLP activations along a held-out promoter-associated
direction.

The five-fold out-of-fold ROC-AUC of the Layer-1 promoter
direction was **0.756**.

For promoter source sequences:

- 4-mutation GC-up versus neutral:
  **+0.227**
- 8-mutation GC-up versus neutral:
  **+0.492**
- 4-mutation GC-down versus neutral:
  **-0.190**
- 8-mutation GC-down versus neutral:
  **-0.332**

The cross-validated internal Layer-1 MLP change correlated
with the downstream Layer-5 diagnostic-probe change at
Spearman rho = **0.387**.

## Interpretation

GC content contributes causally to a promoter-associated
representation present in the Layer-1 MLP output. Increasing
GC shifts this representation toward the promoter direction,
whereas decreasing GC shifts it away. The directional
separation becomes larger at eight mutations than at four.

## Claim boundary

These results concern a fixed diagnostic promoter probe over
a pretrained DNABERT-2 backbone. They do not establish a
native DNABERT-2 promoter prediction mechanism or complete
causal mediation through the Layer-1 MLP.