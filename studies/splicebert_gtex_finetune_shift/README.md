# SpliceBERT GTEx fine-tuning shift

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

This study asks what happens to the established SpliceBERT donor–acceptor coupling computation when the
same `SpliceBERT.1024nt` encoder is fine-tuned for three-class splice-site prediction on SpliceAI's GTEx
dataset. It is a matched-checkpoint transfer study, not a new model or a claim about biological splicing.

## Study identity

- **Base checkpoint:** `SpliceBERT.1024nt`, SHA-256 `2ad91428…0149d`.
- **Fine-tuned checkpoints:** five published seeds (2020–2024) from
  [`chenkenbio/SpliceBERT-analysis`](https://github.com/chenkenbio/SpliceBERT-analysis), trained with
  learning rate `1e-4` and R-Drop weight `10`.
- **Biological computation:** masked-donor dependence on its paired acceptor in canonical short introns.
- **Primary model-native metric:** the original masked-LM donor margin and its clean-minus-corrupt
  coupling effect.
- **Comparison cohort:** the existing gene-split validation cohort; the sealed chr8/9 test is not reopened.
- **Mechanism transfer test:** the frozen base-model components and interventions are applied unchanged.

## Why a derived checkpoint is necessary

The published fine-tunes contain the full `bert.*` encoder and a new three-class `classifier.*` head, but
no masked-LM head. To keep the original coupling readout family fixed, the analysis constructs a derived
masked-LM checkpoint by transplanting all 101 fine-tuned encoder tensors into the original
`BertForMaskedLM`. The base MLM transform and output bias are retained, while the decoder remains tied to
the transplanted nucleotide embeddings as in standard BERT.

This hybrid is an analysis instrument, not an author-published model. A result on it licenses a claim
about what remains visible through the original MLM readout after encoder fine-tuning. It does not by
itself establish what the native three-class classifier uses.

## Stage gates

1. **Checkpoint/instrument preflight:** hashes, exact encoder key coverage, shapes, finiteness,
   architecture, tokenizer, tied decoder, and forward mechanics.
2. **Behavior transfer:** run the unchanged validation behavior and specificity arms on all five seeds.
3. **Frozen-circuit transfer:** apply the original `{L3H1, L2H15}` necessity, K/V mediation, binding, and
   denoise arms without reselecting heads.
4. **Representation shift:** compare the donor-side `U_info`, address/payload, and carrier assays only
   after the corresponding behavior and instrument gates pass.
5. **Replication:** treat all five seeds as the checkpoint family; no best-seed reporting.

The preregistration is [`PREREGISTRATION.md`](PREREGISTRATION.md), live status is in
[`CLAIMS.md`](CLAIMS.md), and run provenance is indexed in [`RESULTS.md`](RESULTS.md).

The follow-up native-head arm is frozen separately in
[`PREREGISTRATION_NATIVE_CLASSIFIER.md`](PREREGISTRATION_NATIVE_CLASSIFIER.md). It tests whether the
released three-class splice-site classifier can read paired-acceptor dependence from the fine-tuned
encoder, then permits a layerwise representation bridge only if that behavior passes its controls.

## Current result

The fixed-readout behavior gate is closed. The base checkpoint replayed the recorded validation values
(`C=0.958`), while the five GTEx-fine-tuned encoders produced coupling means from `−0.029` to `+0.068`.
The five-seed mean paired shift is `−0.932 [−1.038,−0.835]`, negative in 5/5 seeds, with add-one/BH
`q=0.0001`. Partner specificity fell by the same scale.

This is a large shift in what remains readable through the original MLM head, but it is not coupling-
specific: masked-MLM health also fell from `0.523` to `0.209–0.309`. The result therefore supports broad
MLM-readout incompatibility after supervised fine-tuning, not erasure of splice-site coupling from the
fine-tuned encoder and not a claim about the native three-class classifier. The old circuit is not traced
because the required behavior/denominator gate failed.

The released native classifier resolves the erasure ambiguity. After a preserved v1 coordinate failure,
the corrected assay passed all 51 controls. Destroying the paired acceptor lowers native donor evidence by
`0.646 [0.563,0.733]` nats across the five-seed mean, positive in every seed; a second substitution
replicates it, non-partner AGs are near-null, and the partner-specific contrast against a different donor
is `0.317 [0.237,0.402]`.

Under the same fine-tuned classifier vector, paired-acceptor coupling is only `0.022` nats after encoder
layer 4 and becomes `0.646` after the final layer, a `0.624 [0.547,0.704]` projected increment. Base/fine
donor-state CKA simultaneously drops from `0.731` to `0.363`. Thus the evidence supports re-expression in
a co-adapted final-layer splice-classification representation, not erasure. This layer localization is
descriptive and does not yet establish causal necessity or continuity with the base circuit.

The native models were trained on 900-nt examples, while this unchanged coupling assay uses the base
study's 1,024-nt windows. The architecture supports the full length and the native preflight is healthy,
but claims remain scoped to this validation instrument rather than the upstream benchmark distribution.
