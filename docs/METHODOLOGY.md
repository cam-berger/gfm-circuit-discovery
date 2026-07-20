# Methodology: causal mechanistic discovery in GFMs

## Objective

The unit of discovery is not a neuron, head, motif, or attention map. It is a **model computation**: a reproducible behavior, the internal information and components that produce it, and a causal account of how that information moves through the model.

In this repository, a **circuit** is a collection of components shown to be responsible for a scoped
behavior under declared interventions. The default claim is sufficiency/necessity at the tested
granularity—not that the collection is the unique or smallest possible circuit. Minimality requires its
own identifiable size/faithfulness experiment.

The methodology adapts the discipline of LLM mechanistic interpretability to genomic models while respecting biological structure: strand, coordinate system, sequence context, species, genomic region, and the difference between a model mechanism and a biological mechanism.

## The seven-stage workflow

### 1. Specify the biological computation

Write a minimal computational statement before inspecting internals:

- What information must be detected or integrated?
- At which positions or genomic units is it available?
- Which model-native output should change if the information is used?
- Which biological counterfactual changes one factor while preserving plausible context?

The output of this stage is a behavioral assay and a set of rival explanations, not a circuit hypothesis.

### 2. Establish model behavior

Demonstrate the behavior using native logits, losses, embeddings, or task scores. Prefer paired counterfactuals and within-example contrasts over aggregate correlations. Stratify by relevant biological variables and confirm that the effect survives simple baselines.

If the model does not reliably perform the computation, circuit tracing stops. A circuit cannot explain a behavior the model does not have.

### 3. Preflight the instrument

Before interpreting the target experiment, freeze:

- a positive control that the assay must detect;
- a negative control that it must reject;
- sensitivity, resolution, and injection/recovery checks;
- checkpoint, tokenization, coordinate, metric, and sample-count invariants;
- decision thresholds and multiple-testing family;
- a stopping rule for anomalous controls.

A null is informative only when the instrument could have found the expected effect. A positive is persuasive only when specificity controls can fail.

### 4. Generate and localize candidates

Use observational tools to narrow the search:

- residual-stream probes and representation geometry;
- position-table decomposition, Fourier analysis, and low-rank subspaces;
- attention patterns and QK score decomposition;
- gradients, integrated gradients, activation or logit attribution;
- layer, token, head, feature, and context screens.

These tools answer **where to intervene next**. They do not, by themselves, establish that a component is necessary, sufficient, or part of the claimed path.

### 5. Test necessity and specificity

Intervene on the candidate with ablation, projection removal, activation replacement, position-band deletion, or a biologically matched counterfactual. Measure the change in the predeclared model behavior.

Every intervention needs controls appropriate to its size and geometry: random heads, reciprocal or low-leak bands, shuffled content, matched positions, and protected metrics. Record collateral damage as well as the target effect.

Necessity licenses: **this component contributes to this measured behavior under this intervention**. It does not yet identify the information carried or the directed path.

### 6. Test mediation, information, and direction

Build the causal account with complementary interventions:

- activation patching or path patching between paired examples;
- rescue after ablation;
- subspace transplant or dose–response restoration;
- QK decomposition to identify what selects a source or destination;
- OV decomposition to identify what is written downstream;
- edge or path interventions to distinguish direct from redundant routes.

A strong circuit claim combines necessity, mediation, a directed path, and information specificity. Rescue should be scored against matched-size null interventions.

### 7. Replicate, delimit, and interpret

Replicate the relevant step across held-out examples and, where appropriate, independently seeded draws,
checkpoints, strands, species, genomic contexts, or models. A within-sample held-out split tests
generalization conditional on that sample; it is not a substitute for sampling-stability analysis.

When the interpretation claims a transferable information-bearing state, add **cross-example or
cross-context transduction**: derive the state/direction on a source example or context and inject it into
a gene-disjoint target under matched controls. Intra-example clean/corrupt patching establishes mediation
on that prompt; it does not by itself establish interventional generality.

Then state the strongest licensed interpretation and the first missing bridge. Model use, training-data origin, and biological function are separate questions. A real model circuit may have an unknown biological meaning.

## Tool-to-question map

| Question | Main tools | Output | What it does not establish |
|---|---|---|---|
| Does the model perform the computation? | Native logits/losses, paired biological counterfactuals, baselines | Behavioral effect | Internal mechanism |
| Where might the information live? | Probes, spectra, subspaces, attention, attribution | Candidate components and locations | Causality |
| Is a component required? | Ablation, projection removal, band deletion | Necessity and collateral damage | Directed mediation |
| What information is carried? | Patching, counterfactual swaps, QK/OV decomposition | Information specificity | Complete path |
| Does it mediate the behavior? | Rescue, transplant, dose response, path patching | Mediation estimate | Biological origin or function |
| Is the pathway specific? | Matched nulls, protected channels, rival-path interventions | Comparative specificity | Generality across models |
| Does the interpretation generalize? | Sealed replication, checkpoint/model contrasts, external data | Scope and boundary conditions | More than the tested domain |

## Research operations

Each study maintains:

- a preregistration for confirmatory gates;
- a claim ledger with status, evidence, scope, and next test;
- a decision log for changes made after seeing data;
- immutable result artifacts and manifests;
- explicit retractions and supersessions;
- generated figures tied to recorded artifacts.

Exploratory work is welcome, but it must be labeled. The discipline is not “never explore”; it is “never let exploration silently become confirmation.”

## Core principles

1. Start with biology, then verify model behavior.
2. Preflight before interpretation.
3. Attribution ranks; intervention decides.
4. Necessity, mediation, direction, and specificity are different claims.
5. Controls must be capable of failure.
6. Evidence never auto-promotes between levels.
7. Preserve negative results and retractions as part of the method.
8. Stop at the first missing causal bridge.
