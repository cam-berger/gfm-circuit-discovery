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
a co-adapted final-layer splice-classification representation, not erasure.

The causal follow-up identifies one reproducible component of that representation. `L5H6`, selected on
train and tested unchanged across five validation checkpoints, causes `0.081 [0.065,0.100]` damage and
`0.142 [0.114,0.173]` recovery, exceeds frozen same-size controls, and preserves 99.3% of local donor
sensitivity. The final MLP write carries nearly the entire coupling shift (`0.640/0.643` bidirectionally);
clamping it clean reduces `L5H6` damage to `0.0011`, placing the head upstream of a strong MLP-dependent
transformation.

Mechanistically, `L5H6` is value-leaning but formally mixed, and its causal effect is partner-specific.
It is not a direct terminal-`AG` reader: clean K+V restored only at the paired acceptor recovers `0.0095`,
while full-sequence restoration recovers `0.136`, and no four-token region passes the unique-source gate.
The evidence now defines the phenomenon as a **seed-reproducible, partner-specific distributed value
signal transformed by the final MLP into splice-classification evidence**. Continuity with the base
masked-LM circuit and the positive biological identity of that distributed signal remain open.

A classifier-native context decomposition now extends this beyond coupling. Exact
directed-dinucleotide-preserving shuffles identify two nonlocal order dependencies: `[-80,−48)` relative
to the donor contributes `0.808 [0.647,0.971]` donor-margin units, and `[+16,+48)` relative to the
acceptor contributes `1.054 [0.855,1.266]` acceptor-margin units. Both exceed frozen equal-width shams,
are positive in all five checkpoint seeds, and replicate with an independently generated shuffle. These
are target-relative sequence regions, not yet named biological modules.

Their causal depth profiles are nearly symmetric. Clean information remains recoverable at the source
through layer 3, transfers sharply toward the classified token across the last two layers, and is fully
integrated only after layer 5. The final MLP carries `99.7–99.9%` of both effects. The fixed coupling head
`L5H6` does not transfer: it is null for donor context and smaller than all frozen control heads for the
acceptor context. The broader mechanistic definition is therefore **multiple distributed, late-integrated
sequence signals converging on a shared final-MLP splice-classification transformation**. `L5H6` is one
specialized coupling carrier, not a universal context pathway.

The complete layer-5 census resolves the upstream attention routes. All 16 heads were scanned on a
prior-gene-excluded training partition before any fresh confirmation outcome was opened. On a new
gene-disjoint 200-example partition across all five checkpoints, `L5H14` and `L5H8` confirm for both
context effects, `L5H12` confirms as a donor-selected candidate, and `L5H11` confirms as an
acceptor-selected candidate.
The discovery signal is broad: 8/16 donor heads and 6/16 acceptor heads cleared the bidirectional
eligibility rule before the frozen top-three cap; the remaining eligible heads were not confirmatory
candidates and remain exploratory.
All six endpoint assignments pass bidirectional effect, five-seed, multiplicity, and frozen equal-size
control gates. Individual head recovery accounts for `5.8–14.9%` of the corresponding behavior by a
ratio of gene means, so the result supports several routes into the MLP rather than a
single universal head. The opposite-endpoint measurements of `L5H11` and `L5H12` were descriptive
controls and are also positive, so endpoint specialization is not licensed.

Joint intervention and head-to-MLP path patching resolve the next level. The donor and acceptor triples
each carry `28.6–34.4%` of the corresponding behavior and exceed three matched-size control sets. They
are mildly complementary rather than redundant, but fail the frozen 80%-completeness rule. Replaying only
the MLP-write change induced by each frozen triple reproduces `98.0–100.0%` of its total effect; clamping
away that MLP change leaves at most `2.1%` through the residual bypass. This establishes a replicated
many-head→final-MLP causal route for both context signals.

The raw-coordinate low-rank hypothesis does not generalize cleanly. A task direction and SVD basis fit on
seed 2020 have substantial causal effect in that checkpoint but little magnitude in seeds 2021–2024.
The task direction fails the five-seed gate, and no tested rank through 64 reaches half of the full-MLP
effect. The mechanism is therefore shared at the functional level—late heads feeding a dominant final
MLP transformation—while its exact raw vector coordinates remain seed-specific.

An exploratory per-seed alignment assay resolves that boundary. All five checkpoints
put more than 99.6% of discovery MLP-delta energy in four dimensions, and even the
seed-specific rank-1 directions carry about 64–72% of the family causal effect on the
gene-disjoint evaluation partition. Orthogonal maps fit only on discovery activations
make cross-seed rank-4 subspaces recover about 99% of the full MLP effect; they also
transport the actual example-conditioned write at about 95–101%. Raw transport fails
recovery, while aligned transport exceeds dimension- or norm-matched controls and is
positive for every ordered checkpoint pair. This supports checkpoint-specific
rotations of a shared compact causal code, not a universal raw direction. Because the
assay reused the already-opened GFS18 partitions, it remains exploratory pending a new
analysis-sealed replication. A preserved summary-code error and its rank-4 correction
audit do not alter any intervention array or statistic.

The exact 2,048-unit MLP census then resolves how that compact output code is
implemented. Frozen seed-specific top-64 native-unit groups carry about 64–81% of the
full MLP effect, depending on endpoint and direction; 256 units cross the 80%
completeness threshold for both effects. The code is therefore neither a handful of
privileged neurons nor diffuse across nearly the entire MLP. It is a mesoscale,
overcomplete population—hundreds of native units synthesizing a one-to-four-dimensional
write.

Head→unit patching sharpens the directed story. The complete 2,048-unit route replays
the prior head→MLP effect within `2.1×10⁻⁶`. The acceptor triple reaches 80% of that
path with 64 head-responsive units, while the donor triple requires 256. Those
head-responsive top-64 populations overlap the whole-context top-64 populations by
84–86%, and every controlled group exceeds output-norm-matched nulls. Descriptively,
the singleton heads converge on much of the same population rather than feeding three
disjoint downstream modules. Unit identities remain checkpoint- and endpoint-specific,
and the reused GFS18 partitions keep this result exploratory.

One 16-nt acceptor half has no alternative exact Eulerian ordering for one sequence. The correction audit
quarantines that half-resolution contrast; it does not affect either full 32-nt region, sham, independent
draw, depth trace, component bridge, or head census.

The native models were trained on 900-nt examples, while this unchanged coupling assay uses the base
study's 1,024-nt windows. The architecture supports the full length and the native preflight is healthy,
but claims remain scoped to this validation instrument rather than the upstream benchmark distribution.
