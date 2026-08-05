# DNABERT-2 Frozen Component-Level Causal Validation

## Intervention

Combined restoration of:

- Layer 0 MLP
- L0H8
- L1H8
- L2H9
- L2H3

## Cohort

- 128 selected promoter recipients
- 128 non-promoter donors
- Readout: frozen Layer 2 linear probe

## Main Results

### Matched Promoter Restoration

- Mean: 5.7558
- Median: 5.6693
- 95% bootstrap CI: [5.5396, 5.9703]
- Positive fraction: 1.000

### Mismatched Promoter Restoration

- Mean: 5.6107
- Median: 5.7103
- 95% bootstrap CI: [5.2656, 5.9439]
- Positive fraction: 0.992

### Non-Promoter Restoration

- Mean: -0.9076
- Median: -1.0225
- 95% bootstrap CI: [-1.1396, -0.6808]
- Positive fraction: 0.234

## Class-Specific Contrasts

### Matched Promoter Minus Non-Promoter

- Mean: 6.6634
- Median: 6.8095
- 95% bootstrap CI: [6.3156, 7.0088]
- Positive fraction: 1.000

### Mismatched Promoter Minus Non-Promoter

- Mean: 6.5183
- Median: 6.6675
- 95% bootstrap CI: [6.1734, 6.8543]
- Positive fraction: 1.000

### Matched Minus Mismatched Promoter

- Mean: 0.1451
- Median: 0.0999
- 95% bootstrap CI: [-0.2500, 0.5514]
- Positive fraction: 0.516

## Statistical Tests

### Matched Promoter vs Non-Promoter

- Wilcoxon one-sided p-value: 4.7565e-23

### Mismatched Promoter vs Non-Promoter

- Wilcoxon one-sided p-value: 4.7565e-23

### Matched vs Mismatched Promoter

- Wilcoxon two-sided p-value: 0.6497

### Promoter vs Non-Promoter Sign Test

- Promoter donor wins: 128/128
- One-sided binomial p-value: 2.9387e-39

## Sanity Control

- Maximum absolute self-corrupt change: 0.0

## Interpretation

The combined Layer 0 MLP and four-head intervention carries strong promoter-class-aligned causal information.

Promoter-derived activations restore promoter evidence, while non-promoter-derived activations suppress it on average. The matched promoter donor does not significantly outperform a mismatched promoter donor, indicating that the restored representation is class-specific rather than strongly sequence-specific.

## Status

The frozen DNABERT component-level causal localization and restoration analysis is complete for the current Layer 2 endpoint and selected localization cohort.
