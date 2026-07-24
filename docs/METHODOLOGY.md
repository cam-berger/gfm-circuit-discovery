# Methodology: causal mechanistic discovery in GFMs

> **Tier: T1 — canonical.** Current, authoritative. Read this.

## Objective

The unit of discovery is not a neuron, head, motif, or attention map. It is a **model computation**: a reproducible behavior, the internal information and components that produce it, and a causal account of how that information moves through the model.

In this repository, a **circuit** is a collection of components shown to be responsible for a scoped
behavior under declared interventions. The default claim is sufficiency/necessity at the tested
granularity—not that the collection is the unique or smallest possible circuit. Minimality requires its
own identifiable size/faithfulness experiment.

The methodology adapts the discipline of LLM mechanistic interpretability to genomic models while respecting biological structure: strand, coordinate system, sequence context, species, genomic region, and the difference between a model mechanism and a biological mechanism.

## Formal grounding

This section is additive. It names, in the vocabulary the mechanistic-interpretability literature already
uses, what the workflow below does; it introduces no new rule, threshold, or claim, and changes no recorded
result. Full bibliographic entries, with their verification status, are in
[`docs/REFERENCES.md`](REFERENCES.md).

### A circuit as a causal abstraction

Causal abstraction (Geiger et al., 2025) treats an interpretability hypothesis as the claim that a
high-level causal model is an abstraction of the low-level network: an alignment maps network variables onto
the variables of the hypothesized algorithm, and the hypothesis is tested by intervening on the network and
asking whether the result matches the corresponding intervention on the high-level model. That framework
unifies a broad family of interpretability methods — activation and path patching, causal mediation
analysis, causal scrubbing, causal tracing, circuit analysis, and steering — as instances of a single
question.

Of that family, this repository performs activation and path patching, causal mediation analysis, and
circuit analysis. It does **not** perform steering, and it does not perform causal scrubbing proper — see
"Relation to causal scrubbing" below for what the A6/C27 scrub-to-corrupt test does and does not share
with it. The framework is cited here for the vocabulary it supplies, not as a description of methods run.

The repository's definition of "circuit" (above) restates in that language as:

- the **components** named in a circuit are the low-level variables of a proposed alignment;
- the **scoped behavior** is the high-level variable the abstraction is required to reproduce;
- **sufficiency** is recovery of the high-level outcome when the aligned variables are set to their source
  values (injection or interchange intervention);
- **necessity** is loss of the high-level outcome when the aligned variables are set to counterfactual
  values (ablation or scrub);
- **faithfulness is graded, not binary.** Causal abstraction supplies the precise notion: the degree to
  which the abstraction's predictions match the network's across an intervention distribution. A circuit
  therefore never stands alone; it carries a faithfulness value, at a stated node granularity, under a
  stated set of interventions.

This is exactly why the default claim here is sufficiency/necessity at the tested granularity and not
minimality. Under graded faithfulness, sufficiency-when-injected and near-total necessity are fully
compatible with a low faithfulness value, because they are different quantities over different intervention
sets. In this project they came apart, and that is a recorded result rather than an anomaly: C27 reports the
delivery spine as necessary and sufficient-when-injected while `F(C) = 0.005`.

### The faithfulness–size curve

A single faithfulness number under-determines a circuit claim, because faithfulness depends on how large the
circuit was allowed to be. The field-standard object is the whole faithfulness-vs-size curve, summarized by
area-based metrics: MIB defines **CPR** (Circuit Performance Ratio) as the area under that curve and **CMD**
(Circuit–Model Distance) as the area between the curve and `f = 1`, both by the trapezoidal rule over ten
circuit sizes `k ∈ {.001, .002, .005, .01, .02, .05, .1, .2, .5, 1}` (Mueller et al., 2025).

**This project's A12 result already is that object, and should be reported as such.**
`results/splicebert_coupling/a12_faithfulness_size.json` records held-out faithfulness with gene-bootstrap
CIs at 17 circuit sizes over a 103-node graph: 90% of the effect requires about **72 of 103** nodes (grid-resolution — the sampled sizes bracket the 90% crossing to (64, 72]; 50% → 34,
80% → 56, 95% → 80), and the curve is gradual, with **no knee** (C33).

The no-knee shape is the headline, not a caveat. It is a positive empirical statement about the model, not a
report of a search that failed to find something. A knee would have said that a sparse subgraph carries the
behavior; its absence, measured with tight CIs on a held-out split, says that **no sparse subgraph does at
this granularity** — while an importance ranking computed without reference to the hand-built circuit places
the identified mechanism at the top. The conclusion "a structured causal core within an irreducibly
distributed computation" is only available *because* the whole curve was measured. Reporting a sparse graph
plus one faithfulness number, the common practice, would have concealed it.

Two reporting rules follow.

1. **Report the curve, not a point.** When comparing to other work, give an area summary in the
   CPR/CMD sense together with the size grid and the node granularity, since both change the number. This
   project's grid is node counts on a 103-node graph, not MIB's fractional-of-model grid, so values are not
   directly comparable to MIB leaderboard numbers without recomputation on their grid.
2. **Name the ranking procedure.** A12's shipped runner computes a static backward-removal rank, whereas the
   preregistration specified an iterative greedy search; the static rank is an upper bound on the minimal
   faithful set at each faithfulness level, and a true lazy-greedy re-ranking could shave nodes.
   That caveat is recorded with the claim and must travel with any area summary derived from the curve.

### Relation to causal scrubbing

Causal scrubbing (Chan et al., 2022 — a blog-format technical report, not peer-reviewed) reads an
interpretability hypothesis as a claim about which internal activations may be resampled without changing
behavior, and tests it by performing exactly those behavior-preserving resampling ablations and measuring how
much behavior survives. The stage 5 and stage 6 interventions below are scrubbing-adjacent. Where they follow
it, and where they do not, is stated here so that no reader has to infer it.

**Follows:**

- the faithfulness measure `F(C)` is a resample-ablation survival measure: the complement of the hypothesized
  node set is replaced, and what survives is reported as a fraction of the clean/corrupt effect;
- the hypothesis is fixed before the ablation, as a preregistered node set with a frozen decision rule;
- the extremes of the curve are anchored (empty set at the corrupt end, full set at the clean end).

**Deviates:**

- **Scrub-to-corrupt, not resample over a hypothesis-consistent distribution.** The complement is set to its
  value on the matched counterfactual sequence rather than resampled across inputs the hypothesis deems
  equivalent. This keeps the intervention biological and paired, but it exercises a narrower intervention set
  than full causal scrubbing, and therefore does not certify invariance to arbitrary hypothesis-consistent
  resampling.
- **No treeified rewrite.** Interventions are at node granularity on the computational graph, not on a
  path-expanded treeified graph, so redundancy across parallel paths is resolved only in aggregate — by the
  shape of the faithfulness curve rather than per path.
- **Scrubbing is never load-bearing alone.** It is the most lenient of the tools used here: a hypothesis can
  pass a scrub for reasons other than being correct. Every claim that rests on a scrub is paired with an
  interchange or injection test and a held-out gene split.
- **The two exact whole-cover anchors are architecture-forced (R21).** They validate embedding plumbing, not
  node-cover completeness. In scrubbing terms they are normalization anchors, not tests, and `N(C) = 1.0` is a
  forced identity. The genuine on-path memorial is `value_linearity`.

### Conformance with activation-patching best practice

Zhang & Nanda (2024) show that the methodological details of activation patching — above all the evaluation
metric and the corruption method — change which components a study localizes, and argue for making these
choices deliberately; Heimersheim & Nanda (2024) give practitioner guidance on what a given patching setup
licenses as circuit evidence. Both axes are fixed in advance in this project's preregistrations:

- **Corruption is in-distribution and minimal.** The counterfactual is a length-preserving one-base edit to
  the canonical splice dinucleotide of a real genomic sequence (`src/gfmcd/splicing/perturb.py`). Clean and
  corrupt inputs are matched real sequence differing at a single base, so patching is defined symmetrically
  in both directions rather than against a zero, mean, or out-of-distribution baseline.
- **The metric is continuous.** Effects are measured as the native masked-LM true-pair margin, with NLL
  alongside (`src/gfmcd/splicing/metrics.py`), not as thresholded accuracy.
- **Necessity and sufficiency are separate tests.** Ablation-direction and injection-direction results are
  preregistered and reported separately, never combined into a single importance score and never expressed as
  additive fractions (core principles 4 and 6).

The claim here is conformance on the axes those works identify as decision-critical, plus preregistration of
each choice. It is not a claim that these settings are identical to defaults recommended for language tasks.

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
- layer, token, head, feature, and context screens;
- an **automated edge-attribution cross-check (ACDC/EAP-style)** — **recommended, and not yet performed in
  this project.** ACDC automates subgraph identification by iteratively pruning edges of the computational
  graph (Conmy et al., 2023); edge attribution patching approximates the same edge scores linearly in two
  forward passes and one backward pass, at far lower cost (Syed et al., 2024). Running one on the donor graph
  would test whether an automated search, with no knowledge of the hand-built circuit, recovers the same
  top-ranked core, and would blunt the standard objection that a study finds the circuit it went looking for.
  A12's static backward-removal ranking is a related but distinct check: node-level rather than edge-level,
  and not an automated discovery procedure. **No ACDC or EAP run exists in this repository, and no claim in
  the ledger rests on one.**

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
