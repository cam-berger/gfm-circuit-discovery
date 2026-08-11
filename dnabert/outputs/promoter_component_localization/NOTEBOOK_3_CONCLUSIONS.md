# Notebook 3: Promoter Component Localization

## Experimental target

A fixed linear diagnostic probe applied to mean-pooled Layer-5
representations from baseline, non-fine-tuned DNABERT-2.

The principal cohort contained 291 true promoter sequences that had
positive clean Layer-5 probe logits. The remaining 109 true promoters
were retained as a separate clean probe-failure cohort.

## Layerwise localization

Promoter information was linearly accessible throughout the network,
but Layer 5 produced the strongest test performance:

- Layer-5 balanced accuracy: 0.735
- Layer-5 ROC-AUC: 0.811

Layer 8 was the strongest secondary peak. Performance declined somewhat
in the final layers.

Layer 5 also showed the largest probe-logit loss under
dinucleotide-preserving corruption, making it the strongest joint
candidate for linear accessibility and higher-order sequence sensitivity.

## Branch-level localization

Zero ablation caused severe distribution shifts and produced several
sign reversals. Therefore, branch ranking was based primarily on
reference-mean replacement.

The strongest mean-replacement effects were:

- Layer 1 attention: 7.731 mean-logit drop
- Layer 1 MLP: 7.561
- Layer 2 attention: 5.183
- Layer 4 attention: 3.364

These effects indicate that early Layer-1 computation is especially
important for constructing the promoter-associated Layer-5 representation.

## Attention-head localization

The strongest individual heads were:

- L1H9: mean drop 6.646,
  bootstrap 95% CI [5.970, 7.316],
  positive for 88.0% of promoters

- L1H2: mean drop 5.899,
  bootstrap 95% CI [4.961, 6.892],
  positive for 79.7% of promoters

Secondary candidates included:

- L1H11
- L2H9
- L2H1
- L2H11

No single Layer-4 head reproduced the full Layer-4 attention-branch
effect, suggesting a more distributed or interaction-dependent
contribution at that layer.

## Head interactions

Joint replacement of L1H9 and L1H2 produced a mean-logit drop of 8.118,
compared with an additive expectation of 12.545.

The interaction term was:

- mean: -4.427
- bootstrap 95% CI: [-5.453, -3.332]

This supports strong redundancy or saturation between L1H9 and L1H2.

Adding L2H9 to the Layer-1 pair produced an additional mean drop of 1.077,
with bootstrap 95% CI [0.693, 1.451]. Thus, L2H9 contributes additional
but partly overlapping information.

## Supported working model

The promoter-associated representation is not created by a single late
component. It is built early in the network, with Layer-1 attention and
MLP computation playing dominant roles.

Within attention, L1H9 and L1H2 form the strongest sparse candidate set.
They carry overlapping promoter-relevant information. L2H9 provides an
additional cross-layer contribution. Layer 4 appears more distributed
across heads.

## Interpretation boundary

These experiments establish component sensitivity of a trained
diagnostic readout. They do not yet show:

- what nucleotide motifs or biological variables the heads encode,
- that the components form a complete faithful circuit,
- that the same components control a native promoter classifier,
- or that mean replacement is perfectly on-distribution.

Attribution and ablation rank candidate components. Activation patching,
clean-to-corrupt rescue, and biological stimulus tests are required for
strong causal and mechanistic claims.

## Next stage

Use clean and corrupted sequence pairs to perform activation patching:

1. run the corrupted sequence,
2. inject clean activations from candidate components,
3. measure restoration of the Layer-5 promoter-probe logit,
4. test L1H9, L1H2, L2H9, Layer-1 MLP, and combinations,
5. compare necessity with rescue and delivery.

This will be Notebook 4: Promoter Causal Interventions.