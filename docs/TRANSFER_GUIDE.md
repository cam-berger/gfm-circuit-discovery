# Applying the method to another GFM

SpliceBERT supplies a worked example and reusable tools. A new model must establish every scientific
claim again, because a similar architecture does not carry evidence across. The
[DNABERT-2 promoter study](../studies/dnabert_promoter/README.md) is the first such port.

## What transfers

- the workflow that starts from a biological computation
- the assay preflight and the evidence ladder
- the adapter, hook, ablation, patching and provenance interfaces
- claim, decision, preregistration, result and retraction discipline
- families of controls and matched-null logic

## What does not transfer automatically

- the donor–acceptor circuit components and their layer and head identities
- a period-3 implementation
- the ≈10.63 carrier or any proposed biological meaning
- thresholds, detection floors and null distributions
- conclusions across tokenizers, position schemes, species, objectives or checkpoints

## Choosing the next model

Choose a computation the model demonstrably performs, not a model whose internals are merely convenient.
Prefer a model with:

1. a biologically meaningful native output or a defensible zero-shot behavioral metric;
2. access to activations, attention logits and weights, and output projections;
3. a stable public checkpoint and reconstructable preprocessing;
4. enough architectural contrast with SpliceBERT to test whether the method ports;
5. a computation that admits paired biological counterfactuals and negative controls;
6. feasible compute for screening, ablation, patching and replication.

## Adapter contract

An adapter exposes these capabilities without study-specific assumptions:

| Capability | Requirement |
|---|---|
| Identity | Checkpoint, revision, tokenizer, objective, context length, and position scheme |
| Inputs | Deterministic tokenization and mapping between biological and token coordinates |
| Outputs | Native logits, loss, embeddings, or task scores used by the behavioral assay |
| Activations | Named residual, attention, Q/K/V, and output hook points where available |
| Interventions | Ablation, activation replacement, patching, and projection/subspace editing |
| Invariants | Shape, checksum, device, precision, and no-mutation checks |
| Provenance | Complete run manifest including code revision, data, parameters, seed, and command |

Architecture-specific code goes in `src/gfmcd/adapters/`; biological assay logic goes in the study
package, not the adapter.

## Port sequence

For the general method, see [the seven-stage workflow](METHODOLOGY.md#the-seven-stage-workflow).

0. Freeze the question. Create the study from `studies/TEMPLATE_REAL_GFM_STUDY/` and define the
   computation, native behavior, rivals, scope, controls and stopping rules.
1. Qualify the behavior. Show that the checkpoint performs the computation, and record baseline strength,
   heterogeneity and failure cases. Stop if the behavior is absent or wholly explained by a simpler
   baseline.
2. Qualify the instrument. Build the adapter and preflight, then verify hooks and interventions with
   injection/recovery tests, positive and negative controls, and protected metrics.
3. Localize (exploratory). Screen layers, positions, heads, features, spectra or subspaces for a candidate
   set. Freeze a confirmatory plan before using the same data for a definitive claim.
4. Test causally. Run necessity, mediation, information-specificity and path tests, each against
   appropriate nulls, and report collateral effects.
5. Replicate sealed, then synthesize. Replicate on held-out data or an untouched checkpoint or model, and
   compare the discovered algorithm with SpliceBERT's by computational motif, not by head number.

## Comparing models

Compare models on the behavior implemented; information representation; addressing or routing strategy;
redundancy and path structure; how concentrated or distributed the causal effect is; dependence on
positional, sequence or contextual features; robustness across biological contexts; and the evidence
level reached and the first unresolved bridge.

A port succeeds if it yields a circuit, a null result or a clearly diagnosed instrument limit, each
reported at the evidence level it reached. It does not have to reproduce the SpliceBERT finding.
