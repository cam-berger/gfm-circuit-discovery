# Real-GFM Circuit Discovery Protocol

This is the repository-wide protocol for moving from controlled toy models to pretrained genomic foundation models. The SpliceBERT partner-selection study is the reference implementation.

## The seven questions, in order

1. **Does the model perform the behavior?** Use the model's native output where possible; otherwise use the smallest registered readout.
2. **Does it beat explicit shortcuts?** Name and fit the shortcut class. Never infer its ceiling from intuition.
3. **Does it follow the relational cue under conflict?** Ordinary in-distribution accuracy is insufficient when local and relational rules agree.
4. **Which components are necessary?** Use multiple intervention nulls and matched random sets.
5. **Which activations mediate the behavior?** Patch clean activations into aligned corrupt inputs with a verified behavior gap.
6. **How does information flow?** Resolve routing, writes, and intermediate transformations with edge/path interventions and QK/OV or architecture-appropriate analysis.
7. **Does the mechanism predict biology outside the model?** Freeze the circuit metric before independent validation.

## Non-negotiable controls

- Planted positive and negative mechanism controls.
- Untrained or randomized model control.
- Registered local/additive baseline.
- Natural cue-conflict set.
- Matched ordinary examples to detect global degradation.
- At least two ablation nulls.
- Matched random component sets.
- Multiple seeds.
- Grouped confidence intervals by gene/transcript/experiment.
- Explicit test-set unblinding.

## Claim discipline

Use the claim ladder in `SPLICEBERT_CIRCUIT_DISCOVERY_IMPLEMENTATION.md`. Attribution, attention maps, probes, and SAE features are discovery aids. None is causal evidence on its own.

## Model-with-readout rule

When a pretrained model lacks a native task output, the readout must be treated as part of the experimental system.

- A constrained additive readout can expose information already present in contextual representations.
- A flexible interaction readout can manufacture the relation.
- Fine-tuning can install new computation in the backbone.

Report these conditions separately and trace only the condition matching the intended claim.

## Counterfactual design rule

A counterfactual is valid only when:

- the intended cue changes;
- protected nuisance features are measured and matched;
- model token coordinates remain aligned for patching, or a registered alignment map is supplied;
- the clean–corrupt behavior gap is nonzero;
- the edit recipe is stored per example;
- negative and unintended effects are reported.

## Result registry

Every settled result should be generated from a run manifest containing commit hash, dirty status, config, input hashes, checkpoint hash, seeds, command, output hashes, and claim status. Markdown summaries are views of these artifacts, not alternative sources of truth.
