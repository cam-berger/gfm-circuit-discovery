# Notebook 7 — Layer-1 MLP Low-Rank Subspace

## Main result

The GC-increase versus GC-decrease change in DNABERT-2's
Layer-1 MLP output contains a dominant low-dimensional
direction.

In the pooled uncentered SVD:

- PC1 explained **0.571**
  of total intervention-vector energy.
- The participation ratio was
  **2.99**.
- PC1 aligned with the Layer-1 promoter direction at
  **0.894**.

## Held-out generalization

When the subspace was learned without each held-out
biological source:

- PC1 captured **0.446**
  of intervention-vector energy.
- PC1 retained **0.982**
  of the mean promoter-direction shift.
- Four dimensions captured
  **0.591**.
- Eight dimensions captured
  **0.641**.
- Sixteen dimensions captured
  **0.690**.

## Downstream association

Held-out prediction of the Layer-5 diagnostic-probe effect:

- promoter-direction scalar:
  **R² = 0.549**
- one SVD coordinate:
  **R² = 0.554**
- four coordinates:
  **R² = 0.603**
- eight coordinates:
  **R² = 0.612**
- sixteen coordinates:
  **R² = 0.616**
- sixty-four coordinates:
  **R² = 0.623**

Most predictable downstream variation is captured by the
first few dimensions, with diminishing improvement at higher
rank.

## Interpretation

DNABERT-2's Layer-1 MLP contains a dominant shared
GC-responsive direction closely aligned with promoter
representation. Sequence-specific residual responses remain
more distributed.

## Claim boundary

This analysis establishes low-dimensional structure,
held-out generalization, and downstream association. It does
not establish that this subspace causally mediates the full
Layer-5 probe effect.