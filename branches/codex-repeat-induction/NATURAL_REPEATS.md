# Natural hg38 Repeat Generalisation

Checkpoint: `checkpoints/repeat_copy_s0_step8000.pt`. Chromosomes: chr17, chr18, chr19. Windows: 1000.
Repeat criterion: exact `13`-mer, source distance 48-160 nt, at least 3 bases and max base fraction <= 0.7.

| Metric | Value |
|---|---:|
| base-rate ceiling | 0.3500 |
| intact copy accuracy | 0.8780 |
| conflict follows distal edit | 0.7820 |
| conflict follows local continuation | 0.1310 |
| conflict neither | 0.0870 |
| prefix-scrambled follows distal edit | 0.2330 |
| prefix-scrambled follows local continuation | 0.3350 |

## Source Attention

| Head | Conflict | Prefix Scrambled |
|---|---:|---:|
| L1H7 | 0.5621 | 0.0001 |
| L1H0 | 0.5058 | 0.0006 |
| L1H5 | 0.4703 | 0.0013 |
| L1H1 | 0.5347 | 0.0001 |
| L1H6 | 0.4826 | 0.0004 |

## Example

Metadata: `{'chrom': 'chr18', 'window_start': 79035890, 'source_continuation': 12, 'prediction_position': 85, 'distance': 73, 'repeat_kmer': 'GCATCCTGGTTCT', 'local_target': 'T'}`.

```text
source repeat       GCATCCTGGTTCT
second repeat       GCATCCTGGTTCT
source conflicted   GCATCCTGGTTCA
local target        T
distal target       A
model intact pred   T
model conflict pred A
```

Raw data: `branches/codex-repeat-induction/fig_data/natural_repeat_summary.csv`, `branches/codex-repeat-induction/fig_data/natural_repeat_examples.csv`, `branches/codex-repeat-induction/fig_data/natural_repeat_attention.csv`. Figure: `branches/codex-repeat-induction/fig_natural_repeats.png`.

## Seed 99 Replication

Same assay, checkpoint `checkpoints/repeat_copy_s99_step8000.pt`, independently sampled natural
repeat windows (`seed=9191`):

| Metric | Value |
|---|---:|
| base-rate ceiling | 0.3430 |
| intact copy accuracy | 0.8700 |
| conflict follows distal edit | 0.7830 |
| conflict follows local continuation | 0.1330 |
| conflict neither | 0.0840 |
| prefix-scrambled follows distal edit | 0.2280 |
| prefix-scrambled follows local continuation | 0.3540 |

Seed 99 example:

```text
source repeat       GCATCAAGGTTGG
second repeat       GCATCAAGGTTGG
source conflicted   GCATCAAGGTTGT
local target        G
distal target       T
model intact pred   G
model conflict pred T
```

Raw seed-99 data and figure live under
`branches/codex-repeat-induction/natural_seed99/`.
