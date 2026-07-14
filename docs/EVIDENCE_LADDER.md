# Evidence ladder

The ladder prevents a familiar failure mode: a striking observation gradually acquires causal or biological language without the experiment needed for that promotion.

## General ladder

| Level | Question | Typical evidence | Licensed statement |
|---|---|---|---|
| Behavior | Does the model perform the computation? | Native-output counterfactuals and baselines | The model exhibits a scoped behavior. |
| Representation | Is relevant information encoded? | Probes, geometry, spectra, position or residual features | Information is decodable or structurally present. |
| Selection / QK | Does a component read or select it before normalization? | Logit decomposition, QK bilinear analysis | A component scores the candidate relation. |
| Routing / attention | Does the signal survive competition? | Post-softmax attention and controlled deletion | The component routes attention using the signal. |
| Output / OV | What is written downstream? | OV analysis, output projection, downstream feature change | The component writes relevant information. |
| Mediation | Does the component carry the causal effect? | Patching, rescue, transplant, dose response | The component mediates a measured fraction of the effect. |
| Directed circuit | Do connected components implement the computation? | Edge/path interventions plus necessity and mediation | A scoped directed causal pathway implements the behavior. |
| Biological interpretation | Why does this mechanism exist and what does it mean biologically? | Training-data source tests, external assays, cross-model/species evidence | A biological origin or function is supported. |

A study may enter the ladder at different observational levels, but it may not skip the causal bridge when making a causal claim.

## SpliceBERT implementation of the ladder

For positional mechanisms, the project records five compact columns:

- **P** — present in the learned position table.
- **QK** — read in pre-softmax attention logits.
- **Attention** — survives softmax and controlled competition.
- **OV / mediation** — written downstream or causally mediates the effect.
- **Behavior** — changes a model-native behavior under intervention.

| Component | P | QK | Attention | OV / mediation | Behavior | Status |
|---|:--:|:--:|:--:|:--:|:--:|---|
| Period-3 clock | ✓ | ✓ | ✓ | ✓ | ✓ | Causal support for donor–acceptor routing |
| Donor–acceptor computation | n/a | ✓ | ✓ | ✓ | ✓ | Directed information-routing circuit |
| Multi-species ≈10.63 carrier | ✓ | ✓ | ✓ | ? | ? | Real positional attention circuit; purpose unknown |
| Human-only ≈10.2 candidate | ✓ | provisional | failed | — | null | Proposed chromatin circuit closed at attention |

The authoritative values and statistical conventions remain in [`figures/splicebert/EVIDENCE.md`](../figures/splicebert/EVIDENCE.md).

## Promotion requirements

- **Representation → use:** show that a model component reads or routes the feature under an appropriate control.
- **Use → necessity:** remove the feature or component and measure the declared behavior.
- **Necessity → mediation:** restore or patch the candidate and recover the effect relative to matched nulls.
- **Mediation → directed circuit:** isolate the path and rule out simpler or redundant alternatives.
- **Model circuit → biological interpretation:** connect the mechanism to training-data provenance or biological function with an independently capable assay.

## Required controls

1. Positive and negative assay controls.
2. Injection/recovery or another sensitivity calibration.
3. Resolution and detection-floor statement.
4. Matched-size random or structured null interventions.
5. Specificity controls that protect reciprocal features, content, position, or unrelated behavior.
6. Checkpoint, coordinate, tokenization, metric, and sample-count invariants.
7. Multiplicity correction across the actual scanned family.
8. A frozen stopping rule for anomalous controls.

If a control fails, the status is **instrument unresolved**, not positive or negative.
