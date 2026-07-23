# Transfer guide: applying the method to another GFM

> **Tier: T1 — canonical.** Current, authoritative. Read this.

SpliceBERT supplies a worked example and reusable tools. The next model must re-earn every scientific claim; architecture similarity is not evidence transfer.

## What transfers

- the biological-computation-first workflow;
- the assay preflight and evidence ladder;
- adapter, hook, ablation, patching, and provenance interfaces;
- claim, decision, preregistration, result, and retraction discipline;
- families of controls and matched-null logic.

## What does not transfer automatically

- the donor–acceptor circuit components or layer/head identities;
- a period-3 implementation;
- the ≈10.63 carrier or any proposed biological meaning;
- thresholds, detection floors, or null distributions;
- conclusions across tokenizers, position schemes, species, objectives, or checkpoints.

## Choosing the next GFM

Prefer a model with:

1. a biologically meaningful native output or a defensible zero-shot behavioral metric;
2. access to activations, attention logits/weights, and output projections;
3. a stable public checkpoint and reconstructable preprocessing;
4. enough architectural contrast with SpliceBERT to test method portability;
5. a computation that admits paired biological counterfactuals and negative controls;
6. feasible compute for screening, ablation, patching, and replication.

Do not select a model only because its internals are convenient. Select a computation the model demonstrably performs.

## Adapter contract

A model adapter should expose, without study-specific assumptions:

| Capability | Requirement |
|---|---|
| Identity | Checkpoint, revision, tokenizer, objective, context length, and position scheme |
| Inputs | Deterministic tokenization and mapping between biological and token coordinates |
| Outputs | Native logits, loss, embeddings, or task scores used by the behavioral assay |
| Activations | Named residual, attention, Q/K/V, and output hook points where available |
| Interventions | Ablation, activation replacement, patching, and projection/subspace editing |
| Invariants | Shape, checksum, device, precision, and no-mutation checks |
| Provenance | Complete run manifest including code revision, data, parameters, seed, and command |

Architecture-specific code belongs in `src/gfmcd/adapters/`. Biological assay logic belongs in the study package, not the adapter.

## Port sequence

### Phase 0 — freeze the question

Create a study from `studies/TEMPLATE_REAL_GFM_STUDY/`. Define the computation, native behavior, rivals, scope, controls, and stopping rules.

### Phase 1 — behavioral qualification

Show that the target checkpoint performs the computation. Record baseline strength, heterogeneity, and failure cases. Stop if behavior is absent or wholly explained by a simpler baseline.

### Phase 2 — instrument qualification

Implement the adapter and preflight. Verify hooks and interventions with injection/recovery tests, positive and negative controls, and protected metrics.

### Phase 3 — exploratory localization

Screen layers, positions, heads, features, spectra, or subspaces. Treat the output as a candidate set. Freeze a confirmatory plan before using the same data to make a definitive claim.

### Phase 4 — causal circuit tests

Run necessity, mediation, information-specificity, and path tests. Compare every intervention with appropriate nulls and report collateral effects.

### Phase 5 — sealed replication and synthesis

Replicate on held-out data or an untouched checkpoint/model. Compare the discovered algorithm with SpliceBERT at the level of computational motifs, not matching head numbers.

## Cross-model comparison axes

Compare models on:

- the behavior implemented;
- information representation;
- addressing or routing strategy;
- redundancy and path structure;
- causal concentration versus distribution;
- dependence on positional, sequence, or contextual features;
- robustness across biological contexts;
- evidence level reached and first unresolved bridge.

The program succeeds if the methodology yields an honest circuit, an honest null, or a clearly diagnosed instrument limit in another GFM. Reproducing the SpliceBERT finding is not required.
