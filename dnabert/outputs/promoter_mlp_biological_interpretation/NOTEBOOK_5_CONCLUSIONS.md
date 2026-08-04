# Notebook 5: Biological Interpretation of the Layer-1 MLP Signal

## Objective

Determine which sequence properties are associated with the
transferable promoter-class representation found in the
Layer-1 MLP of baseline DNABERT-2.

## Dataset

The analysis used:

- 189 probe-recognized promoter sequences
- 189 unrelated-promoter donor interventions
- 189 sampled non-promoter sequences

The promoter cohort was selected because:

1. the fixed Layer-5 probe classified the sequence as promoter-like;
2. exact dinucleotide-preserving corruption reduced the probe
   logit by at least one unit.

Therefore, the cohort is not a representative sample of all
promoter architectures.

## Promoter versus non-promoter sequence features

The selected promoters were substantially more GC-rich:

- promoter mean GC fraction: 0.644
- non-promoter mean GC fraction: 0.543

Promoters also had:

- lower nucleotide entropy;
- higher GpC density;
- modestly higher CpG density;
- greater enrichment for GC-box-like motifs;
- fewer TATA-like sequences.

GC-box-like motifs occurred in:

- 23.8% of promoters
- 8.5% of non-promoters

The odds ratio was approximately 3.38.

These results describe the selected probe-recognized cohort and
should not be generalized to all promoters.

## Features associated with MLP transfer

Among unrelated promoter donors, stronger Layer-1 MLP rescue was
associated with:

- higher GC fraction;
- higher GpC density;
- higher CpG density;
- lower sequence entropy;
- greater distance from a TATA-like sequence;
- closer similarity to a GC-box-like motif.

GC fraction was the strongest individual predictor:

- raw Spearman rho: 0.550
- adjusted Spearman rho: 0.452

The adjusted analysis accounted for:

- donor Layer-5 probe logit;
- recipient corrupted logit;
- recipient clean-corrupt gap.

## Cross-validated predictive value

A baseline model using donor probe strength and recipient
difficulty achieved an out-of-fold R2 of approximately 0.382.

Adding biological sequence features increased out-of-fold R2 to
approximately 0.472.

Repeated cross-validation showed that GC content alone was the
most reliable addition:

- mean incremental R2: 0.106
- median incremental R2: 0.112
- fraction of folds improved: 0.920
- fraction of repeated splits with positive mean improvement: 1.0

Adding the full biological feature panel produced a smaller
mean improvement:

- mean incremental R2: 0.072

Features beyond GC reduced generalization:

- mean incremental R2 beyond GC: -0.033
- fraction of folds improved: 0.128
- fraction of repeated splits positive: 0.0

## Supported interpretation

GC content is the dominant simple sequence property associated
with the transferable Layer-1 MLP promoter representation.

Higher-GC promoter donors transfer stronger promoter-associated
evidence into corrupted recipients, even after accounting for
the donor's final Layer-5 probe score and recipient difficulty.

CpG density, GpC density, sequence entropy, and GC-box similarity
are correlated with the transfer effect, but they do not provide
reliable predictive value beyond total GC content.

## Interpretation boundary

The current evidence does not establish that:

- the Layer-1 MLP explicitly calculates GC percentage;
- GC content is the only feature encoded;
- the representation is specific to biological promoter function;
- GC content causally generates the Layer-1 MLP signal;
- the result generalizes to all promoter subclasses.

GC may be:

1. the directly encoded variable;
2. a proxy for GC-rich promoter architecture;
3. a proxy for CpG-island-like sequence structure;
4. correlated with a more complex learned sequence feature.

## Next experiment

Notebook 6 will causally manipulate GC content while controlling
sequence length and, where possible, dinucleotide or motif
structure.

The primary test is:

Does deliberately increasing or decreasing GC content produce a
corresponding change in:

1. Layer-1 MLP activation;
2. Layer-5 promoter probe evidence;
3. transferable MLP rescue strength?

Controlled interventions are required before making a mechanistic
claim about GC encoding.