# Implementation addendum v1: six-view sparse final-MLP instrument

> **Tier: T2 — provenance.** Frozen after the biological-feature discovery artifact
> and before exporting any sparse-training sequence state, capturing any activation
> for this instrument, fitting any dictionary, or applying any native classifier
> head to a reconstruction. This addendum fixes numerical implementation choices
> left open by
> `PREREGISTRATION_NATIVE_BIOLOGICAL_FEATURES_V1.md`; it does not change any
> biological endpoint or confirmation threshold.

## Evidence boundary

The shared/tuned/private blocks are an engineering parameterization, not evidence
that fine-tuning acquired a biological feature. The base view is structurally unable
to use the tuned-only block, so membership in that block may never itself be called
an acquired detector. An acquired, amplified, rewritten, or readout-only label
requires the preregistered post-hoc comparisons of detector activation, writer
geometry, and classifier mediation against the base model and all five tuned seeds.

The sparse model is a candidate-generation and patching instrument. Passing its
preflight licenses feature assignment experiments; it does not license biological
semantics, RBP identity, or a causal mechanism.

## Strict input firewall

The producer is fixed to
`scripts/splicebert/export_gtex_sparse_transcoder_sequence_package.py`.
It reconstructs the 128 discovery examples from the basic GENCODE dataset by exact
recorded example ID, re-audits both comprehensive-annotation CDS windows, and emits
only clean, target-shuffle, sham-shuffle, two maximum synonymous recodes, and every
single-codon synonymous state.

The package has a strict root and row allowlist. Every changed row must reference a
clean row with the same example, gene, endpoint, target coordinate, and sequence
length; its sequence hash, changed positions, and Hamming dose must replay exactly.
The package is bound to the preflight, discovery artifact, basic and comprehensive
GTFs, hg38 FASTA, and producer hashes. Activation capture re-hashes every bound input.

No classifier label, weight, logit, margin, effect, descriptor coefficient, candidate
rank, or external assay outcome enters sparse training.

## Activation capture

- Views, in sequential load order: base MLM, then tuned seeds 2020–2024.
- Site: layer-5 target-token input to `BertIntermediate` and exact layer-5 MLP write.
- Architecture invariant: 6 layers, 16 heads, hidden width 512, MLP width 2,048.
- Maximum equality error for the captured MLP input: `2e-5`.
- Maximum dense-write replay error: `5e-4`.
- Checkpoints are released between views; concurrent model-scale processes are
  checked before the run.

## Dictionary split and normalization

- Gene-level split namespace:
  `gtex-native-biological-features-transcoder-train-v1`.
- Validation genes: SHA256 bucket `<20`; all other discovery genes train.
- Validation gene IDs plus the hash of every validation state ID are persisted.
- Behavioral preflight may use only this held-out validation subset.
- Inputs are standardized independently per view from training genes only.
- Standard-deviation floor: `1e-4`.

## Frozen model family

- Feature widths: `{128,256,512,1024}`.
- Active features per example: `{8,16,32}`.
- Initializations: `{0,1,2}`.
- Initialization seed:
  `1,000,000 + 10,000*width + 100*active + initialization`.
- Capacity: one-half shared, one-quarter tuned-only, one-quarter private.
- Tuned-view activity: one-half shared, one-quarter tuned-only, one-quarter private.
- Base-view activity: one-half shared, one-half private; tuned-only is unavailable.
- Decoder columns are unit normalized.
- Training uses blockwise BatchTopK. Evaluation uses fixed-L0 per-row TopK. Stored
  full-training BatchTopK thresholds are diagnostics and are not used for inference.

The fixed allocation is reported as a hypothesis-bearing engineering scaffold.
Independent-per-view, no-view-difference, and no-seed-consistency fits are required
label-free comparisons. Post-hoc feature matching of the independent fits is required
before any cross-seed convergence claim.

## Optimization

- Adam learning rate: `1e-3`.
- Batch size: `256`, with half ordinary states and the remainder complete
  clean/counterfactual pairs where available.
- Maximum epochs: `60`; patience: `8`.
- Minimum validation improvement: `1e-6`.
- Gradient-norm clipping: `1.0`.
- Loss weights:
  reconstruction `1.0`, within-view paired difference `1.0`,
  base/fine view difference `1.0`, activation L1 `1e-3`,
  tuned-seed consistency `0.1`.
- Model checkpoint is the lowest validation write-delta error before patience.

The complete run has 36 primary fits and 36 loss-ablation fits. Fits run
sequentially.

## Representation preflight

Every view in every initialization must explain at least `0.95` of held-out
write-delta energy.

Cross-seed pairing is permuted in 20 deterministic draws. The permutation gate
requires:

- at least 95% of permuted consistency losses exceed the paired loss;
- mean permuted loss exceeds paired loss by at least `0.05`;
- mean permuted/paired loss ratio is at least `1.10`.

Per-view reconstruction is unchanged by this diagnostic because only the
cross-seed pairing is permuted.

Feature-family stability is evaluated in all five tuned views and all three
initialization pairs. Each comparison requires at least `0.25` reciprocal matches
and mean reciprocal combined score at least `0.50`; combined score is the equal
weight mean of nonnegative activation correlation and nonnegative writer cosine.
Because seed consistency is a fitted objective, this is optimizer stability, not
independent biological convergence.

The planted detector/writer simulation must recover minimum activation correlation
`0.80` and writer cosine `0.70`.

Required comparators are PCA/SVD, Haar-random subspaces, matched sparse random
dictionaries, the three loss ablations, the complete native MLP ceiling, and the
recorded GFS20 exact native-unit curves. GFS20 is explicitly historical and is not
re-estimated on this dictionary split.

## Frozen native-head behavioral preflight

Dictionary weights are immutable before any classifier parameter is loaded.
For each representation-passing setting, all three fits, all five tuned checkpoints,
and both endpoints must pass on held-out dictionary-validation genes:

- mean absolute native-versus-reconstructed margin error `<=0.05`;
- native target-shuffle damage `>=0.05`;
- reconstructed damage is positive;
- reconstructed/native damage ratio `>=0.90`.

Among complete settings, choose minimum mean validation write-delta error, then
feature width, then active count, then lexical setting ID. Choose the primary
initialization by minimum validation write-delta error, then initialization number.

Any failed gate is preserved as `INSTRUMENT_UNRESOLVED`. Failure leaves GFS20 and the
native-unit mechanism claims unchanged and stops sparse-feature interpretation.
