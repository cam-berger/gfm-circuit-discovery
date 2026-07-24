# Biological-foundation-model interpretability: the 2026 evidence for behavior-first circuit discovery

> **Tier: T2 — provenance.** Literature verification pass, 2026-07-24 (frame 0a2ba5cc). Companion to
> [`METHOD_CHARACTERIZATION_2026-07-24.md`](METHOD_CHARACTERIZATION_2026-07-24.md). Every DOI below resolves
> to a primary source read for this note; none is entered in a claim ledger by this document, which is the
> owner's step under the immutable-record discipline.

## Bottom line

The strongest recent case for anchoring circuit discovery on a causally qualified behavior before screening
components comes from a 2026 systematic audit of single-cell foundation models, which shows that attention
structure aligned with ground-truth biology can be entirely dispensable to the model's actual predictions.
That result is the empirical spine of the behavior-first ordering our pipeline already encodes, and three
further sources place it in the wider methodological arc.

## The load-bearing result: alignment is not importance

The central exhibit is [Kendiukhov 2026](https://doi.org/10.48550/arXiv.2602.17532), a systematic
interpretability audit of scGPT and Geneformer built from thirty-seven analyses and 153 statistical tests
across four cell types and two perturbation modalities. Its finding is unusually clean for a negative result.
Attention patterns do encode layer-organised biology, with protein–protein interactions in early layers and
transcriptional regulation in late layers, yet that structure "provides no incremental value for perturbation
prediction." Trivial gene-level baselines beat both attention and correlation edges (AUROC 0.81–0.88 versus
0.70), pairwise edge scores add zero predictive contribution, and, in the sentence that matters most for our
method, "causal ablation of regulatory heads produces no degradation." The effect generalises from K562 to
RPE1 cells, with gene-level dominance described as universal even where the attention-correlation relationship
is context-dependent. The paper is a single-author arXiv preprint, not yet peer-reviewed, and should be read
as such; its design is nonetheless exactly the necessity-against-a-baseline test that separates a mechanism
from a correlate, and it reports the honest negative.

Read against our evidence ladder, this is the failure mode the ladder exists to catch. The heads most aligned
with annotated regulation sit at the representation and attention columns yet fail the necessity intervention,
so they never earn promotion. Our donor–acceptor result runs the identical test with the opposite outcome:
ablating `{L3H1, L2H15}` removes most of the coupling while masked-language-model health is preserved, which
is what a licensed necessity claim looks like when the null could have failed and did not.

## The methodological arc around it

The move the audit relies on, deciding importance by intervention rather than by attention weight, is the
throughline of [Kadem & Zheng 2026](https://doi.org/10.48550/arXiv.2601.04398), a survey that traces how
attention-head intervention emerged as a key method for causal interpretability and frames the shift from
visualization to intervention as a move from observing correlations to causally validating hypotheses. It also
restates the specialisation-versus-redundancy tension that makes single-head stories fragile: many heads are
individually removable, and the interpretable ones are not always the load-bearing ones. That is precisely why
our method treats the atlas as a ranking instrument and reserves interpretation for post-intervention.

On the genomic side specifically, [Consens et al. 2025](https://doi.org/10.1101/2025.06.26.661544) offer a
framework for reading attention in genomic transformer models toward biological insight. It is the natural
point of contrast for our approach, because it works the attention-to-biology direction that Kendiukhov shows
must be validated causally rather than trusted. The ambition these audits are measured against is set by
[Rood, Hupalowska & Regev 2024](https://doi.org/10.1016/j.cell.2024.07.035), whose perturbation cell-and-tissue
atlas articulates the goal of a causal foundation model of cell biology. Against that standard, "attention
encodes co-expression, not unique regulatory signal" reads as a genuine gap rather than a technicality.

## What this licenses for the DNABERT2 port

None of these four sources is a SpliceBERT result and none belongs in our claim ledgers; they are external
evidence about method. What they jointly support is a single operational decision. The first DNABERT2
deliverable should be a Stage-2 behavioral qualification whose design can return a negative, meaning a
defensible native or zero-shot behavior, a paired biological counterfactual, and a simple-baseline horse-race,
followed by a Stage-3 preflight, with the head atlas deferred until that gate is passed. Kendiukhov is the
cautionary exhibit for why that ordering is not optional; the rest place it in the field's own trajectory.
