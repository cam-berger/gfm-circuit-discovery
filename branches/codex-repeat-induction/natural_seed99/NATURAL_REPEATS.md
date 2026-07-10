# Natural hg38 Repeat Generalisation

Checkpoint: `checkpoints/repeat_copy_s99_step8000.pt`. Chromosomes: chr17, chr18, chr19. Windows: 1000.
Repeat criterion: exact `13`-mer, source distance 48-160 nt, at least 3 bases and max base fraction <= 0.7.

| Metric | Value |
|---|---:|
| base-rate ceiling | 0.3430 |
| intact copy accuracy | 0.8700 |
| conflict follows distal edit | 0.7830 |
| conflict follows local continuation | 0.1330 |
| conflict neither | 0.0840 |
| prefix-scrambled follows distal edit | 0.2280 |
| prefix-scrambled follows local continuation | 0.3540 |

## Source Attention

| Head | Conflict | Prefix Scrambled |
|---|---:|---:|
| L1H7 | 0.3984 | 0.0001 |
| L1H0 | 0.5100 | 0.0000 |
| L1H5 | 0.4409 | 0.0000 |
| L1H1 | 0.4983 | 0.0000 |
| L1H6 | 0.0267 | 0.0069 |

## Example

Metadata: `{'chrom': 'chr19', 'window_start': 55110275, 'source_continuation': 12, 'prediction_position': 75, 'distance': 63, 'repeat_kmer': 'GCATCAAGGTTGG', 'local_target': 'G'}`.

```text
source repeat       GCATCAAGGTTGG
second repeat       GCATCAAGGTTGG
source conflicted   GCATCAAGGTTGT
local target        G
distal target       T
model intact pred   G
model conflict pred T
```

Raw data: `branches/codex-repeat-induction/natural_seed99/fig_data/natural_repeat_summary.csv`, `branches/codex-repeat-induction/natural_seed99/fig_data/natural_repeat_examples.csv`, `branches/codex-repeat-induction/natural_seed99/fig_data/natural_repeat_attention.csv`. Figure: `branches/codex-repeat-induction/natural_seed99/fig_natural_repeats.png`.
