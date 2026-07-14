# SpliceBERT findings of record

This is the current scientific narrative. It replaces earlier reports that mixed live findings, plans, and retracted interpretations. Exact artifact paths and reproduction commands are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md); claim status and corrections are in [`CLAIMS.md`](CLAIMS.md).

## Status at a glance

| Finding | Strongest evidence | Interpretation |
|---|---|---|
| Donor–acceptor coupling | behavior, necessity, mediation, sealed replication | A dominant causal subcircuit couples short-intron donor and acceptor information. |
| Period-3 positional clock | position → QK → attention → mediation → behavior | A learned rotary positional code supports coding prediction and donor–acceptor coupling. |
| Multispecies ≈10.63 carrier | position → QK → post-softmax attention | A real positional attention circuit with unknown OV role, behavior, and origin. |
| Human-only ≈10.2 candidate | position and QK; fails attention | The proposed chromatin circuit is closed at the attention gate. |
| Nucleosomal ≈10.298 signal | sequence-side biological assay | A strong biological signal exists but does not explain the multispecies ≈10.63 carrier. |

## 1. Donor–acceptor coupling is real and partner-specific

The behavior is measured with the model's native masked-language-model output. The G of a donor `GT` is masked and scored by its logit margin. Corrupting the paired acceptor `AG` to `CC` changes that donor score.

On the sealed chr8/chr9 split (n = 400):

- unconditional donor coupling is **+0.872 [0.771, 0.969]**, median 0.638;
- corrupting a motif-matched non-partner `AG` costs **+0.007**;
- scoring the same corruption at a distance-matched other donor costs **+0.005**;
- all 13 pre-registered gates pass.

This licenses a partner-specific non-local coupling claim within the tested context. It does not establish literal splice-partner selection, transcript-level splicing, or generality to typical long introns.

The study is limited to canonical, internal, short introns whose donor and acceptor fit the model context (60–150 nt in the sealed protocol), both strands, and one SpliceBERT checkpoint.

## 2. The dominant causal subcircuit

The frozen circuit is `C = {L3H1, L2H15}`.

On the sealed split:

- ablating L3H1 removes **78.7%** of the coupling;
- ablating L2H15 removes **47.7%**;
- ablating C removes **81.6%** while retaining **99.2%** of masked-probe MLM health;
- C exceeds 400 distinct matched random two-head sets, exact add-one **p = 0.0025**.

The interventions show that L3H1 reads acceptor information and writes its causal contribution at the donor and ±5-token neighborhood. The acceptor region contributes almost none of the head's output effect. Downstream, no single readout head or block carries the result: recovery accumulates across the remaining transformer stack. The readout is a distributed chain.

L2H15's contribution routes through L3H1 when the full L3H1 output is restored. L1H12 is a real upstream contributor whose effect also routes through L3H1, but it is not a clean complementary “query” component. The attractive parallel-feeder story is unsupported.

C is therefore the **dominant subcircuit**, not a demonstrated complete circuit. Large complement ablations damage the language model broadly, and size-matched nulls show that the available completeness test has no power.

## 3. The acceptor read is conjunctive

Activation interventions separate the acceptor's key and value contributions to L3H1:

| Intervention | Sealed recovery |
|---|---:|
| key only | 0.032 |
| value only | 0.101 |
| key + value | 0.663 |
| K×V interaction | **+0.531 [0.477, 0.583]** |

The key establishes where attention is routed; the value supplies relevant content; the joint effect is strongly non-additive. Circuit denoising restores **79% [74%, 83%]** of the lost coupling.

Earlier key/value relocation experiments do **not** establish a content-addressed or position-independent lookup. In a BERT with absolute position information already embedded in key/value vectors, moving a complete key/value pair to another slot preserves what the attention mechanism sees. The relocation result was architecturally forced and is permanently retracted.

## 4. A learned period-3 positional clock supports the circuit

SpliceBERT's position table contains a rank-2 period-3 rotary code. Relevant heads read this code in their attention logits and post-softmax attention.

Deleting the period-3 band:

- collapses phase-selective attention mass from **0.786 to 0.345**, near chance 1/3;
- causes coding-exon masked prediction loss of **0.159 nats** versus **0.042** in introns;
- collapses donor–acceptor coupling from **0.946 to 0.027**;
- leaves only **1.1%** of fitted period-3 amplitude.

An energy-matched but functionally distinct positional band preserves the clock and the coupling, so the result is not “deleting any position-table energy hurts.”

Restoring L2H15 inside the clock-deleted model recovers **0.363** of the lost coupling. Restoring L3H1 recovers **0.348**. Each exceeds the maximum **0.042** among 24 size-matched one-head nulls; exact add-one **p = 0.040**. The six-head restoration is not independently diagnostic because six-head nulls can also be large, and the 16-head restoration lacks a size-matched null.

The clock appears in stage-1 multispecies pretraining and is much weaker in the human-only checkpoint: enrichment **477.7×** versus **6.3×**, with random initialization near 1×. Species diversity is confounded with training-data volume, so the checkpoints do not identify the cause.

This finding supersedes the old “42-head phase lattice.” That metric rewarded single-offset attention and measured sharpness, not periodicity. The valid period-3 result comes from explicit comb structure, deletion, protected controls, behavioral loss, and matched restoration.

## 5. The multispecies ≈10.63 carrier reaches attention, but no further claim

A separate rank-2 positional carrier occupies a plane orthogonal to the period-3 clock. Sub-bin analysis places it at approximately **10.630 nt**; “10.67” is a useful bin label, not the final estimate.

The post-softmax assay passed frozen positive, negative, injection, probability-invariant, leakage, and deletion controls. With exact add-one permutation p-values and family correction, eight heads reach Level A:

`L2H12, L3H3, L3H5, L3H8, L3H9, L3H11, L3H13, L4H13`.

Carrier deletion reduces the relevant attention coherence to roughly **40–44%**; reciprocal and low-leak control bands retain **96–100%**. Content shuffling retains **95–97%**, as a positional circuit should.

The licensed claim stops here:

- the feature exists in the position table;
- QK logits read it;
- it survives into post-softmax attention;
- its OV content, behavioral function, and biological origin are unknown.

## 6. The human-only ≈10.2 chromatin-circuit proposal is closed

The human-only checkpoint contains a weaker ≈10.2 position-table feature and candidate QK readers. The same validated post-softmax assay that detects eight multispecies readers detects **zero** human-only Level-A heads after exact add-one correction.

Most preregistered candidates did not have enough attention mass in the tested offset band to enter the scan. The primary L1H3 placed only **0.19%** of its attention there, below the 2% eligibility floor. The best apparent post-hoc hit disappeared when zero-hit permutation results were corrected from raw `0/400` to the exact add-one floor `1/401`.

The stopping rule therefore fires: no species-matched OV or behavioral rescue experiment is justified for this proposal. The final status is position established; QK supportive; post-softmax attention not established; OV and behavior not pursued.

## 7. The ≈10.298 nucleosomal signal is real and distinct

A chemical-cleavage nucleosome map provides a strong positive biological result. A dyad-anchored, frozen-phase matched filter finds a WW/SS periodic signal at approximately **10.298 nt**, with held-out signed **z = +249**, phase agreement across chromosome splits within **0.4°**, depth **4.87%**, and WW/SS counterphase **181.5°**. Offset controls show that it is confined to the nucleosome core.

The signal remains ≈10.296 in transcribed regions. It is therefore distinct from the multispecies ≈10.630 carrier. This is not an underpowered null: the assay finds a large biological signal at the wrong wavelength for the proposed explanation.

Splice-junction GG/CC phasing also replicates locally, but its ~51-nt support cannot resolve 10.298 from 10.630; the matched filters are about 98% correlated at that window length. Long introns supply enough resolution and a powered null, while long exons lack sample size. These results do not attribute the carrier.

## 8. What remains open

The next scientific target is not another rescue of the human-only chromatin story. It is the multispecies ≈10.63 attention carrier:

1. determine whether its OV output carries a stable, interpretable signal;
2. test a model-native behavior with a qualified intervention;
3. identify its origin only after computational use is established;
4. apply the same evidence ladder to a second real GFM.

The broader contribution is the disciplined progression from behavior to localization, necessity, mediation, replication, and bounded interpretation. SpliceBERT is the worked example; transfer to another GFM is the intended test of the method.

## Limits that travel with every claim

- one model family and a small set of checkpoints;
- short-intron context only for donor–acceptor coupling;
- intervention effects are model computations, not direct biological mechanisms;
- spectral coincidence is hypothesis generation, not provenance;
- observational importance does not imply causal relevance;
- nulls are reportable only when sensitivity, specificity, resolution, and power are demonstrated.
