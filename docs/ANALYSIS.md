The clean publication story is:

**The original SpliceBERT paper showed correlation; your experiment identifies the causal mechanism behind part of that correlation.** Their Figure 3 established that same-intron donor–acceptor pairs receive elevated attention, particularly in layers 3–5. Your result goes one step further: the partner-dependent contribution to the model’s actual donor prediction is highly concentrated in **L3H1**, and that head behaves like a **signed partner-consistency signal**.

When the partner acceptor is intact, L3H1 adds (+0.910) to donor confidence. When the partner is corrupted, its presence instead lowers donor confidence by (0.200). Therefore, the head does not merely detect an acceptor and always contribute positively. Its contribution changes according to whether the donor has a valid partner context. The clean interpretation is:

> **L3H1 supplies evidence for or against a donor depending on the state of its same-intron acceptor partner.**

I would avoid calling the (-0.200) term a “local role.” It is measured when the partner is corrupted, but that does not mean it represents local donor recognition. It is better described as the head’s **partner-absent contribution** or **negative partner-consistency signal**. Also, because these are the two ends of the same intron, call this **same-intron boundary coupling**, not exon definition.

## Recommended figure structure

Do not simply show attention before and after ablation. Attention is the baseline observation, but your main contribution is causal. Use a three-panel figure that visually progresses from phenomenon to localization to computation.

### Panel A: Causal analogue of the SpliceBERT figure

Use the same pair categories as the original paper:

* Same-intron donor–acceptor
* Same-exon donor–acceptor
* Unpaired donor–acceptor
* Donor–donor
* Acceptor–acceptor
* Distance-matched random

But put **causal coupling strength** on the y-axis rather than raw attention. For each category, show:

* SpliceBERT intact
* L3H1 ablated

The expected visual message should be that same-intron coupling is strong in the intact model and selectively collapses after L3H1 ablation, while the control relationships change much less. This creates a direct bridge to Figure 3 while showing what the original analysis could not: which association is functionally used by the model.

### Panel B: Mechanistic localization across all heads

Show the held-out 96-head scan as a ranked dot or bar plot. Highlight L3H1 and include its confidence interval.

The reviewer should immediately see:

* L3H1 recovers (0.808) of the effect.
* It ranks first of 96 heads.
* It is approximately 4.3 times stronger than the next-best head.
* Its held-out recovery is nearly identical to its discovery-set recovery.

This panel establishes that the mechanism is localized, reproducible and not a product of cherry-picking.

### Panel C: The signed computation

Plot the (2\times2) factorial as an interaction plot:

* x-axis: partner intact versus partner corrupted
* y-axis: donor logit margin
* one line: L3H1 intact
* one line: L3H1 ablated

Annotate the two head contributions:

* Partner intact: (+0.910)
* Partner corrupted: (-0.200)
* Difference-in-differences: (+1.110,[1.018,1.205])

This is the most mechanistically important panel. The lines show that the head’s effect changes sign depending on partner status. A reviewer can understand the computation without needing to understand activation patching in detail.

The low-rank subspace result can be a fourth panel or supplementary figure. Include it in the main text only if feature geometry is important to the broader paper. Otherwise, the three panels above already tell a complete short story.

## What you can claim

The strongest defensible claim is not merely that L3H1 “attends to the partner.” The original paper already showed elevated donor–acceptor attention. Your contribution is:

> **The same-intron donor–acceptor association is causally used in the model’s native splice-site prediction, is predominantly mediated by one attention head, and is represented as signed evidence whose effect depends on whether a viable partner context is present.**

That gives you three distinct mechanistic properties:

1. **Causal:** removing the head removes most of the partner effect.
2. **Localized:** one head dominates the full 96-head scan.
3. **Conditional and signed:** the head supports donors with an intact partner and opposes them when that partner signal is absent.

## Publication-ready interpretation

Previous analysis of SpliceBERT showed that attention weights are elevated between donor and acceptor sites belonging to the same intron, particularly in intermediate transformer layers. We find that this statistical association corresponds to a localized causal mechanism. On 400 gene-disjoint held-out exons, layer 3 head 1 recovered 80.8% of the partner-dependent effect on native donor prediction, ranking first among all 96 attention heads and exceeding the next-best head by more than fourfold.

Factorial ablation further showed that the head implements a signed, partner-conditioned signal rather than generic donor recognition. With the partner acceptor intact, L3H1 increased the donor margin by 0.910; when the partner was corrupted, the head instead reduced the margin by 0.200. This produced a partner-dependent interaction of 1.110 [1.018, 1.205]. Thus, L3H1 appears to evaluate same-intron boundary consistency, supplying positive evidence when a compatible partner is present and negative evidence when that partner signal is absent. These results move beyond attention correlation by identifying a specific causal component and characterizing the computation it contributes to splice-site prediction.

