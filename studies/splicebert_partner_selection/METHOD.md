# Method

The full methodology is maintained in the repository root document `SPLICEBERT_CIRCUIT_DISCOVERY_IMPLEMENTATION.md`. This file records the study-specific choices.

## Population

GENCODE basic protein-coding transcripts on the GRCh38 primary assembly. Internal exons only. Sequence is transcript-oriented. Canonical AG acceptors and GT/GC donors are the default primary cohort.

## Candidate task

For each sequence window, enumerate motif-compatible acceptors and donors and every pair within the registered exon-length interval. Rank the annotated pair over decoys.

## Split

Default chromosome split:

- validation: chr17–chr18
- test: chr19–chr22
- train: remaining included autosomes
- excluded: chrX, chrY, chrM/MT

A gene-hash split is available as a sensitivity analysis but may not replace the frozen primary split after results are viewed.

## Baselines

- local acceptor PWM + local donor PWM + empirical length prior;
- additive frozen SpliceBERT readout;
- additive frozen randomized-backbone readout;
- context-restricted sequence controls;
- optional receptive-field-limited CNN, if added before preregistration.

## Readout conditions

See the table in the root implementation document. `additive-frozen` is the primary trace target.

## Causal score

Mean true-pair logit margin, true pair minus strongest decoy. Higher is better. Loss-based results may be added but signs must be recorded directly rather than converted informally.

## Discovery set

A fixed validation subset of natural local conflicts, chosen before component intervention. Selection may stratify by gene, strand, exon length, and candidate count but not by head effects.

## Intervention sequence

1. single-head zero/mean/resample screen;
2. cross-null consensus;
3. grouped necessity and random-set null;
4. clean-to-corrupt patching;
5. token-localized patching;
6. QK/OV and edge/path decomposition;
7. replication across readout seeds;
8. one-time held-out test.
