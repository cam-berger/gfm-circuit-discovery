# SpliceBERT joint-segment attribution graphs

## Status

Exploratory instrument development. The exported graphs are measured candidate generators, not confirmatory evidence and not causal circuit claims.

## Why the MLM adaptation differs

The [Circuit Tracing attribution-graph method](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) explains an autoregressive target token with a local replacement model whose MLPs are replaced by cross-layer-transcoder features. SpliceBERT is bidirectional and predicts masked nucleotides. Its graph target is therefore a declared joint masking condition:

```text
x^(S)_i = [MASK]  if i ∈ S=[s,e)
          x_i     otherwise

M_native(S,q) = log p(x_q | x^(S)),  q ∈ S
H(S,q)        = -Σ_b p(b | x^(S)) log p(b | x^(S))
```

All positions in `S` are masked in one forward. The position-level predictions are simultaneous marginals under that shared context. Their product or summed log-probability is not a true joint sequence likelihood. A segment summary, when used, must be a length-normalized mean and named accordingly.

The native-base target asks whether and how the model reconstructs the known genome. A predicted-base target asks why the model prefers its own argmax. The first exporter uses native-base log-probability as primary and reports the predicted base separately.

## What v1 measures

`scripts/splicebert/export_segment_attribution_bundle.py` exports:

- one real, deterministic, outcome-unfiltered GENCODE validation sequence;
- joint masks of the donor `GT` and its intron-mate acceptor `AG`;
- one focused graph for each nucleotide within each segment;
- all 96 measured attention rows and total QK logits at each focus;
- component-level activation×local-Jacobian screening scores for attention-head context writers, fixed-pattern per-key OV routes, whole-MLP outputs, and input embeddings;
- explicit per-head omitted-source remainder nodes;
- immutable JSON output and manifest with checkpoint, code revision, selection rule, device, command, and correctness gates.

The graph adaptively keeps a small frontier of source positions at each layer. It is a hypothesis generator. It does not claim that its retained paths are additive, complete, feature-level, or causal.

## Relationship to the methods paper

The paper makes direct feature-feature attribution well-defined by using a cross-layer transcoder and freezing attention patterns and normalization denominators in a local replacement model. We have not trained or fidelity-gated a SpliceBERT transcoder. Accordingly:

- v1 nodes are model components, not sparse semantic features;
- attention source edges explain frozen-pattern OV transport, not the QK computation that produced the attention pattern;
- whole-MLP nodes remain polysemantic;
- activation×Jacobian edges are local screening estimates;
- exact interventions in the original model are required before an edge is called causal.

A future feature-level version should train a SpliceBERT cross-layer transcoder, preserve reconstruction-error nodes, and preflight replacement fidelity, graph completeness, and perturbation faithfulness before interpretation.

## Correctness gates

The exporter fails rather than writes a bundle unless:

1. applying the native MLM head to the cached final residual reproduces vocabulary logits;
2. the sum of per-key attention-value routes reconstructs each captured head context;
3. eager attention returns nonzero post-softmax tensors;
4. the tokenizer preserves one-token-per-nucleotide coordinates;
5. every graph records the exact joint mask span, focused output, native target, predicted segment, and attribution boundary.

The visualizer independently validates graph acyclicity, segment bounds, coordinate alignment, metric completeness, and that every displayed node reaches the output root.

## Next validation steps

1. Exact-zero or matched-reference patch the highest-ranked head, MLP, and per-key OV routes in the original model.
2. Compare screening rank with exact effects on planted and known-real positive controls.
3. Add influence-preserving node/edge pruning and report retained influence plus signed remainder rather than display-threshold mass.
4. Add a separate QK graph for important OV routes; never infer route selection from fixed attention.
5. Replicate graph motifs across disjoint genes and mask spans before naming a reusable mechanism.

## Run

```bash
PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false \
uv run python \
  scripts/splicebert/export_segment_attribution_bundle.py \
  --output results/splicebert_visualizer/segment_attribution_bundle_v2.json
```

Install the canonical environment first with `uv sync --extra splicebert --extra dev`; see
[`../../REPRODUCE.md`](../../REPRODUCE.md).

Load the resulting JSON through the visualizer's bundle control. Do not commit a new result as a claim artifact unless its manifest and status remain attached.
