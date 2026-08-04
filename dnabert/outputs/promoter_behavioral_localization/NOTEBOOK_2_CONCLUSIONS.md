# Notebook 2: Promoter Behavioral Localization

## Experimental target

A logistic diagnostic probe trained on mean-pooled Layer-5
representations from baseline, non-fine-tuned DNABERT-2.

This is a representation-level readout, not a native DNABERT-2
promoter-classification output.

## Main findings

1. Clean test performance reproduced the original pilot:
   balanced accuracy = 0.735 and ROC-AUC = 0.811.

2. Exact mononucleotide-preserving shuffling caused a mean promoter-logit
   drop of 4.706. Eighty-one percent of promoter sequences showed a
   positive mean drop across repeated shuffles.

3. Because exact A/C/G/T counts were preserved, the full-shuffle effect
   cannot be explained by GC content or mononucleotide composition alone.

4. Exact dinucleotide-preserving shuffling caused a smaller but meaningful
   mean logit drop of 1.829. Therefore, dinucleotide statistics preserve
   substantial promoter-associated information, but do not explain all of it.

5. The paired difference between mononucleotide- and dinucleotide-preserving
   shuffle effects was 2.878 logits, with a bootstrap 95% confidence interval
   of [2.547, 3.215] and a one-sided Wilcoxon p-value of approximately
   7.18e-42.

6. The dinucleotide-preserving effect was approximately 38.9% of the
   mononucleotide-preserving shuffle effect. This residual is consistent
   with sensitivity to longer motifs or higher-order sequence organization.

7. Circular shifts produced a much smaller mean logit drop of 0.475.
   Thus, promoter-associated evidence depended more strongly on retained
   local sequence structure than on the original alignment within the
   70-nucleotide input.

8. Shuffling individual 10-nucleotide windows produced positive effects
   across all regions, with heavily overlapping bootstrap confidence
   intervals. No single 10-nucleotide region dominated under this test.

## Supported conclusion

Baseline DNABERT-2 Layer-5 representations contain promoter-associated
information that depends strongly on nucleotide order. Much of this
dependence is captured by dinucleotide statistics, while a significant
residual depends on longer-range motifs or higher-order sequence structure.
The effect appears distributed across the input rather than sharply
localized to one 10-nucleotide region.

## Limitations

- The measured output is a trained linear diagnostic probe, not a native
  promoter classifier.
- Corruption sensitivity does not identify specific biological motifs.
- Circular shifting is not a pure absolute-position intervention because
  it changes sequence boundaries and tokenizer segmentation.
- Regional shuffling may disrupt multiple overlapping motifs and therefore
  cannot by itself establish causal biological localization.
- Component-level and activation-level interventions are needed before
  making mechanistic circuit claims.

## Next stage

Use these clean-versus-corrupt conditions as behavioral targets for
component localization across DNABERT-2 layers, attention heads, and MLPs.