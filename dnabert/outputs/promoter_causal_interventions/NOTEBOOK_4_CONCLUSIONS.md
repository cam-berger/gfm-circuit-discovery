# Notebook 4: Promoter Causal Interventions

## Experimental target

All interventions were evaluated using the fixed Layer-5
diagnostic promoter probe from Notebook 1. This probe is not
a native DNABERT-2 promoter classifier.

The primary causal cohort contained 189 true promoter
sequences for which exact dinucleotide-preserving corruption
reduced the clean probe logit by at least one unit.

The cohort had:

- mean clean logit: 7.393
- mean corrupted logit: 1.024
- mean clean-corrupt gap: 6.369

## Tokenization alignment

Clean and corrupted sequences did not share DNABERT-2 BPE
boundaries:

- only 59/189 pairs had equal token counts
- 0/189 pairs had identical token-span patterns

Therefore, clean activations were projected into corrupted
token positions using overlap-weighted absolute nucleotide
span alignment.

Self-patching controls reproduced unpatched logits to within
approximately 3e-5, confirming correct implementation.

## L1H9

Clean L1H9 patching did not rescue promoter evidence:

- mean rescue: -2.384
- 95% CI: [-3.139, -1.596]
- positive rescue: 31.2%

Matched clean donors did not reliably outperform unrelated
clean promoter donors.

Thus, L1H9 is strongly sensitivity-associated in clean-run
mean replacement but is not independently sufficient to
restore promoter evidence under global dinucleotide
corruption.

## Layer-1 attention

Patching the full Layer-1 attention branch also failed to
produce average rescue:

- mean rescue: -0.806
- 95% CI: [-1.363, -0.248]
- positive rescue: 38.6%

This indicates that coordinated Layer-1 attention output alone
is not sufficient in the globally corrupted residual stream.

## Layer-1 MLP

Patching matched clean Layer-1 MLP outputs produced strong
over-restoration:

- mean rescue: 10.810
- 95% CI: [9.602, 12.009]
- positive rescue: 89.9%
- median recovery fraction: 1.731

However, unrelated clean promoter donors produced a similar
effect:

- mean rescue: 9.524
- matched-minus-unrelated difference: 1.285
- 95% CI: [-0.728, 3.253]

Therefore, most of the Layer-1 MLP effect is promoter-class
specific rather than matched-sequence specific.

## Donor-strength dependence

Among unrelated promoter donors, donor probe strength
predicted rescue magnitude:

- Spearman rho: 0.441
- Pearson r: 0.486

Mean rescue increased across donor-strength quartiles from
3.37 in the lowest quartile to 18.41 in the highest quartile.

## Non-promoter control

Non-promoter Layer-1 MLP donors produced no reliable average
rescue:

- mean rescue: 0.290
- median rescue: -0.191
- 95% CI: [-1.127, 1.697]

Both matched and unrelated promoter donors strongly exceeded
non-promoter donors:

- matched minus non-promoter: 10.520
  95% CI [8.880, 12.193]

- unrelated promoter minus non-promoter: 9.235
  95% CI [7.416, 11.037]

## Supported mechanistic model

The promoter-associated Layer-5 diagnostic representation is
built early in DNABERT-2.

Layer-1 attention components, especially L1H9 and L1H2, are
strongly necessary-like under clean-run mean replacement, but
their clean outputs do not independently restore the signal in
a globally corrupted run.

Layer-1 MLP output contains a transferable promoter-associated
representation. Promoter-derived MLP activations strongly
increase downstream promoter evidence, whereas non-promoter
MLP activations have no reliable average effect.

The Layer-1 MLP signal is predominantly class-level rather
than sequence-specific.

## Interpretation boundary

These findings support representation delivery to a fixed
diagnostic probe. They do not establish:

- a complete faithful circuit,
- native promoter classification by DNABERT-2,
- the biological motif encoded by the Layer-1 MLP,
- or exact token-level causal correspondence because BPE
  alignment required nucleotide-span projection.

## Next experiment

Characterize what biological sequence feature controls the
Layer-1 MLP promoter direction.

Priority analyses:

1. correlate MLP rescue with GC content, CpG density, TATA-box
   presence, and promoter subclasses;
2. test controlled motif insertion and deletion;
3. identify low-rank directions in Layer-1 MLP output;
4. test whether projecting only the promoter-associated
   direction reproduces the donor-class effect;
5. separate generic promoter evidence from sequence-specific
   information.