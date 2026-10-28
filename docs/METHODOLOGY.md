# Methodology

The unit of discovery is a model computation: a reproducible behavior, the internal information and
components that produce it, and a causal account of how that information moves through the model. A
neuron, head, motif or attention map is not a finding on its own. The method adapts language-model
mechanistic interpretability to genomic models, respecting strand, coordinate system, sequence context,
species, genomic region, and the difference between a model mechanism and a biological one. Cited works
are in [`docs/REFERENCES.md`](REFERENCES.md).

## What counts as a circuit

A circuit is a collection of components shown to be responsible for a scoped behavior under declared
interventions. We claim sufficiency and necessity at the tested granularity, not uniqueness or minimality,
which needs its own identifiable size/faithfulness experiment. The SpliceBERT donor mechanism is a
structured causal core within distributed computation under the tested interventions, not established as
minimal or autonomous.

In causal-abstraction terms (Geiger et al., 2025), the components are network variables aligned with a
high-level causal model, and the scoped behavior is the high-level variable the abstraction must reproduce
under matching interventions. Sufficiency is recovery of that outcome when aligned variables take source
values (injection or interchange); necessity is its loss under counterfactual values (ablation or scrub).
Faithfulness is graded agreement between the two models across an intervention distribution, at a stated
node granularity and intervention set, so sufficiency and near-total necessity can coexist with low
faithfulness. They do here: C27 restores `D = 0.9864` when clean outputs are injected into the nine non-embedding
spine nodes, while isolating the full hand-built set preserves only `F(C) = 0.005`.
The two estimands use different component sets.

Of the methods causal abstraction unifies, we perform activation and path patching, causal mediation
analysis and circuit analysis. We do not perform causal scrubbing proper
([below](#relation-to-causal-scrubbing)) or automated circuit search, both listed under [Method
coverage](ROADMAP.md#method-coverage), and no claim rests on behavior-directed steering. The one
preregistered site-choice steering study (B2) was design-red-teamed, halted at its own smoke gate on first
contact with the model, and retired before any confirmatory run (B2 disposition in
[`CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md)). G1's additive, dose-graded injection of
a derived direction into the final donor residual is a sufficiency and specificity test, negative for
example-specificity (C41, verification tier; R28/R32).

### The faithfulness–size curve

Faithfulness depends on how large the circuit may be, so the standard object is the
faithfulness-versus-size curve. MIB summarizes it by CPR (Circuit Performance Ratio, the area under the
curve) and CMD (Circuit–Model Distance, the area between the curve and `f = 1`), both by the trapezoidal
rule over ten sizes `k ∈ {.001, .002, .005, .01, .02, .05, .1, .2, .5, 1}` (Mueller et al., 2025).

A12 measured this curve (`results/splicebert_coupling/a12_faithfulness_size.json`): held-out faithfulness
with row-weighted gene-bootstrap CIs at 17 sizes on a 103-node graph. The preregistered interpolation reaches 90% at
64.2 nodes; the first sampled point above 90% is 72 of 103, so the crossing lies in (64, 72]. The 50%, 80%
and 95% levels fall at 34, 56 and 80 nodes, and there is no knee (C33). That is a positive finding: a knee
would have supported a sparse subgraph under this ranking, and its absence shows that the tested
static-ranking construction is distributed at this granularity, while a ranking computed without reference to the hand-built circuit puts
the identified mechanism at the top.

Report the curve as a CPR/CMD-style area with its size grid and node granularity; ours, node counts on a
103-node graph, needs recomputation on MIB's fraction-of-model grid before comparison with MIB leaderboard
numbers. Name the ranking procedure too. A12's shipped runner computes a static backward-removal rank
where the preregistration specified an iterative greedy search; the static rank is an upper bound on the
minimal faithful set at each faithfulness level, a true lazy-greedy re-ranking could shave nodes, and this
caveat travels with any area summary from the curve.

## The seven-stage workflow

### 1. Specify the biological computation

Before looking inside the model, state what information must be detected or integrated, at which positions
or genomic units it is available, which model-native output should change if it is used, and which
biological counterfactual changes one factor while preserving plausible context. The output is a
behavioral assay and a set of rival explanations, not a circuit hypothesis.

### 2. Establish model behavior

Show the behavior on native logits, losses, embeddings or task scores, with paired counterfactuals and
within-example contrasts rather than aggregate correlations, stratified by the relevant biological
variables and shown to survive simple baselines. If the model does not reliably perform the computation,
stop: a circuit cannot explain a behavior the model does not have.

### 3. Preflight the instrument

Before interpreting the target experiment, freeze:

1. a positive control the assay must detect and a negative control it must reject;
2. injection/recovery or another sensitivity calibration, and a statement of resolution and detection floor;
3. matched-size random or structured null interventions;
4. specificity controls that protect reciprocal features, content, position, or unrelated behavior;
5. checkpoint, tokenization, coordinate, metric, and sample-count invariants;
6. decision thresholds, with multiplicity correction across the actual scanned family;
7. a stopping rule for anomalous controls.

A null is informative only if the instrument could have found the expected effect, and a positive is
persuasive only if the specificity controls could have failed. If a control fails, the status is
instrument unresolved, neither positive nor negative.

### 4. Generate and localize candidates

Residual-stream probes, representation geometry, position-table and Fourier decomposition, low-rank
subspaces, attention and QK score decomposition, gradient and attribution methods, and layer, token, head,
feature and context screens narrow the search. They say where to intervene next; by themselves they do not
establish that a component is necessary, sufficient, or part of the claimed path. An automated
edge-attribution cross-check (ACDC, Conmy et al., 2023; edge attribution patching, Syed et al., 2024) is
recommended and has not been run, and no claim rests on one.

### 5. Test necessity and specificity

Intervene by ablation, projection removal, activation replacement, position-band deletion, or a
biologically matched counterfactual, and measure the change in the predeclared behavior. Match controls to
the intervention's size and geometry (random heads, reciprocal or low-leak bands, shuffled content,
matched positions, protected metrics), and record collateral damage as well as the target effect.
Necessity shows that this component contributes to this measured behavior under this intervention, not yet
the information carried or the directed path.

### 6. Test mediation, information, and direction

Build the causal account from complementary interventions: activation or path patching between paired
examples, rescue after ablation, subspace transplant or dose–response restoration, QK decomposition (what
selects a source or destination), OV decomposition (what is written downstream), and edge or path
interventions (direct versus redundant routes). Score rescue against matched-size null interventions. A
strong circuit claim combines necessity, mediation, a directed path, and information specificity, and
still says nothing about biological origin or function, or about other models.

### 7. Replicate, delimit, and interpret

Replicate the relevant step on held-out examples and, where appropriate, on independently seeded draws,
checkpoints, strands, species, genomic contexts, or models; replication sets a claim's scope, and nothing
beyond the tested domain. A within-sample held-out split tests generalization conditional on that sample,
not sampling stability. A claimed transferable information-bearing state needs cross-example or
cross-context transduction into a gene-disjoint target under matched controls, because clean/corrupt
patching within one example establishes mediation on that prompt, not interventional generality. Then
state the strongest interpretation the evidence supports and the first missing bridge. Model use,
training-data origin and biological function are separate questions, and a real model circuit may have an
unknown biological meaning.

## Patching and scrubbing

### Conformance with activation-patching best practice

Zhang & Nanda (2024) show that the evaluation metric and the corruption method change which components
activation patching localizes, and Heimersheim & Nanda (2024) give practitioner guidance on what a given
patching setup supports as circuit evidence. Our preregistrations fix both in advance; we claim
conformance on these axes, not that our settings match the defaults recommended for language tasks.

- The corruption is a minimal, in-distribution, length-preserving substitution of the partner acceptor's
  canonical `AG` at `[a−2, a)` by the real bases `CC` (`CORRUPT_ALT`, `src/gfmcd/splicing/circuit.py:66`,
  applied by `CouplingTask.pair`; canonicity enforced at `src/gfmcd/splicing/coupling.py:347`; alternatives
  and the `destroy_acceptor` / `destroy_donor` helpers at `src/gfmcd/splicing/coupling.py:55-56,119-126`).
  No alternative can create a canonical motif, `[MASK]` is never used because it is off-manifold, and clean
  and corrupt inputs otherwise match, so patching is symmetric, with no zero, mean, or out-of-distribution
  baseline.
- The metric is the continuous native masked-donor margin `logit(G) − max(logit A, C, T)` at the masked
  donor `G`, in raw MLM logit units, not nats (`CouplingTask.margin`,
  `src/gfmcd/splicing/circuit.py:131-136`; frozen equivalent `margin_hard`,
  `src/gfmcd/splicing/unembed.py:19-22`). With no trained readout head or label set, nothing can leak.
  Thresholded accuracy enters only the non-site health control (`CouplingTask.mlm_health`), and masked-base
  cross-entropy only an OOD/health diagnostic or the band-deletion experiment's damage metric in nats (C39).
- Ablation-direction and injection-direction results are preregistered and reported separately, never
  combined into one importance score or expressed as additive fractions.

### Relation to causal scrubbing

Causal scrubbing (Chan et al., 2022, a blog-format technical report, not peer-reviewed) tests a hypothesis
by resampling the internal activations it says do not matter and measuring how much behavior survives. Our
stage 5 and 6 interventions, including the A6/C27 scrub-to-corrupt test, share its survival measure
(`F(C)`: the complement of the hypothesized node set is replaced, and the surviving fraction of the
clean/corrupt effect is reported), fix the hypothesis before the ablation as a preregistered node set and
decision rule, and anchor the curve's ends (empty set at the corrupt end, full set at the clean end).

They deviate in four ways. The complement takes its matched-counterfactual value instead of being
resampled over hypothesis-equivalent inputs, which keeps the intervention biological and paired but does
not certify invariance to arbitrary hypothesis-consistent resampling. Interventions are at node
granularity, not on a path-expanded treeified graph, so redundancy across parallel paths is resolved only
in aggregate, by the curve's shape. A scrub, our most lenient tool, never supports a claim alone, since a
hypothesis can pass one for reasons other than being correct; each is paired with an interchange or
injection test and a held-out gene split. The two exact whole-cover anchors are architecture-forced
normalization anchors, not tests (R21): they validate embedding plumbing rather than node-cover
completeness, and `N(C) = 1.0` is a forced identity. The real on-path memorial (a check on the measurement
path that can fail) is `value_linearity`.

## Evidence ladder

The ladder keeps a striking observation from gradually acquiring causal or biological language without the
experiment needed for that promotion.

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

A study may enter the ladder at different observational levels, but it may not skip the causal bridge when
making a causal claim. Each step up needs its own experiment. From representation to use, show that a
component reads or routes the feature under an appropriate control; to necessity, remove it and measure
the declared behavior; to mediation, restore or patch it and recover the effect relative to matched nulls;
to a directed circuit, isolate the path and rule out simpler or redundant alternatives; to biological
interpretation, connect the mechanism to training-data provenance or biological function with an
independently capable assay.

### The SpliceBERT columns

For positional mechanisms in SpliceBERT we record five columns: P (present in the learned position table),
QK (read in pre-softmax attention logits), Attention (survives softmax and controlled competition), OV /
mediation (written downstream, or causally mediates the effect) and Behavior (changes a model-native
behavior under intervention). In causal-abstraction terms, P and QK intervene on nothing: P finds a
candidate variable, with no alignment asserted, and QK a candidate alignment that is still a
decomposition. Attention intervenes but reads an internal variable, the routing pattern. Only OV /
mediation reads the behavior variable, after an interchange or injection, so only there can a graded
faithfulness number be computed; Behavior is that variable itself, under a high-level counterfactual.

| Component | P | QK | Attention | OV / mediation | Behavior | Status |
|---|:--:|:--:|:--:|:--:|:--:|---|
| Period-3 positional carrier | ✓ | ✓ | ✓ | ✓ | ✓ | Causal support for donor–acceptor routing |
| Donor–acceptor computation | n/a | ✓ | ✓ | ✓ | ✓ | Directed information-routing circuit |
| Multi-species ≈10.63 carrier | ✓ | ✓ | ✓ | null (donor coupling) | bounded (−0.033) | Real positional attention circuit, not part of the donor circuit. Both interventions ran (C37→C12). General OV content is a bounded null by diagnostic (C12); biological meaning remains open; the `U_info` write test was never computed (R31). |
| Human-only ≈10.2 candidate | ✓ | provisional | failed | — | null | Proposed chromatin circuit closed at attention |

Values and statistical conventions are in the
[`CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md) ledger and controlling artifacts. A ✓
means the defining intervention ran and the component survived it; ? that it has not run, so there is no
evidence either way; — that the level is not applicable or was not entered; failed or null that it ran and
rejected the hypothesis at that level; bounded that it ran and returned a measured boundary instead of a
verdict (a point estimate below the smallest effect size of interest, SESOI, or an equivalence test whose
half-width exceeds its margin). A null or a bound is a result, not an absence, and neither is a ✓.

The table must never be read as collapsing these states. The human-only ≈10.2 candidate was rejected at
attention by its defining intervention; the ≈10.63 carrier's OV and Behavior interventions both ran on
2026-07-23 (C37, folded into C12), so it no longer illustrates ?. Nor are the columns a faithfulness
measurement: the donor–acceptor computation has ✓ in every applicable column, yet low faithfulness for the sparse circuit
(C27) and no sparse knee (C33). Reading a full row of ✓ as a sparse faithful circuit is the inference the
ladder exists to block.

## From model mechanism to biology

Background biology supplies hypotheses and analogies, not evidence about what SpliceBERT implements. Claim
IDs below refer to
[`studies/splicebert_partner_selection/CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md),
which holds the measured values; corrections are in
[`studies/splicebert_partner_selection/CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md).
The donor mechanism and period-3 carrier are measured on `SpliceBERT.1024nt`, the ≈10.63 carrier most
directly on `SpliceBERT.510nt` against its human-only stage-1 sibling, and results from the two
checkpoints are not interchangeable.

### Donor–acceptor coupling and intron definition

Crossed so far: model behavior plus a directed causal core. In canonical internal 60–150 nt introns the
masked-donor margin depends partner- and donor-specifically on the paired acceptor
([C1/C2](../studies/splicebert_partner_selection/CLAIMS.md#1-the-behaviour-and-its-specificity)), through
a predominantly positional address and an acceptor-local payload that meet at L3H1 (C3–C5, C16–C18, C23,
C26, C31); cross-example transfer is gene-generic for the C16/C26 final donor residual under additive injection
(C41, verification tier, not confirmatory; R28/R32), and position-specific with content specificity
unresolved for the acceptor payload (C38, verification tier, inconclusive). This has the functional shape of intron
definition, a coherent analogy to test, not evidence that the model implements the biological regime. ED1,
the preregistered test of that stronger reading, failed its planted-confound instrument gate and returned
`GATES_FAILED -- no interpretation`, which a post-run sensitivity bound cannot relabel
([ED1](../studies/splicebert_partner_selection/CLAIMS.md#12-a-preregistered-arc-whose-gates-failed)), so
what may be said is:

> SpliceBERT computes partner-specific cross-intron communication whose functional geometry resembles
> intron definition; ED1 did not pass the instrument gates required to claim architecture-conditioned
> intron/exon-definition behavior.

Crossing further needs a redesigned ED1 with a valid magnitude-sensitive confound gate, then independent
biological validation. Spliceosome biochemistry, splicing in cells, long-intron generality, and the
semantic identity of the acceptor-local payload are not established.

### Period-3 carrier and coding-boundary register preference

Crossed so far: two separate model-internal results, with the causal bridge between them unresolved. The
learned rank-2 period-3 direction is a distributed relative-positional routing coordinate that causally
supports donor–acceptor coupling (deletion changes coupling from 0.946 to 0.026; C8, R53). It acts
across the donor–acceptor span through a substantial relative-offset signal with absolute/edge modulation
(C32; frozen decision MIXED), mediated primarily by L3H1's attention pattern within the tested
intervention ([C34, C35](../studies/splicebert_partner_selection/CLAIMS.md#7-the-period-3-carrier-and-its-causal-use)).
Nucleotide-invariant surgery shows phase-aligned use of the direction in the tested CDS-versus-upstream-
intron contrast (`Ψ_C = +0.141 [0.110, 0.172]`, above all 19 same-geometry controls; C36).
Separately, in one fixed 120-nt chimeric crossed lattice, coupling is stronger for GENCODE-annotated
boundary-register matches than mismatches
([C43](../studies/splicebert_partner_selection/CLAIMS.md#13-coding-boundary-phase-in-partner-selection-consolidated-from-the-archived-exon-edge-arc)),
not established in native sequence or shown to occur only for matches, with stop-load and codon-position
composition still live alternatives. C44's test of period-3 mediation failed four frozen authorization
gates before the target cohort was exposed and supports neither a positive, a null, nor an inverted
mechanism conclusion, so the paper treats the pair as motivating a positional-scaffold/content-gating
hypothesis, a discriminating hypothesis rather than a mechanism claim:

> Period-3 geometry may provide positional coupling gain, while still-unidentified boundary content
> determines where that gain is engaged.

Crossing needs register manipulations matched for span, motifs, edit dose, stop load and composition,
crossed with a capable period-3 phase-specific intervention read out both internally and at the native
margin, then external reporter validation; until then neither reading-frame sensing nor a cellular
splicing mechanism is established. Coding-specific readings are unsupported (C36's matched
CDS-minus-non-CDS-exon contrast, n=52 pairs, is inconclusive; C19 finds no detectable dependence on intron
length modulo 3 or annotated splice phase, descriptively and not as equivalence), and the carrier's comparative association with multi-species
pretraining (C39/C40) does not isolate species diversity from corpus size, composition, sampling or
optimization history. Codon periodicity is a plausible corpus correlate, but why training built the
coordinate needs checkpoint trajectories or controlled corpus or objective ablations.

### The ≈10.63 carrier and nucleosomal periodicity

Crossed so far: controlled post-softmax attention in eight heads
([C11](../studies/splicebert_partner_selection/CLAIMS.md#9-the-separate-1063-carrier)) and a powered
refutation of the proposed nucleosomal bridge. For the donor circuit its OV value-write is null and its
coupling effect small and suppressive (C37, recorded under C12); whether it writes onto the donor readout
`U_info` was never tested and is not evaluable as scoped on this checkpoint (R31/R36). The human-only
≈10.2 reader proposal fails at the controlled attention gate (C13), and local splice-window phasing cannot
resolve the two wavelengths, a resolution limit rather than a biological null (C15). What may be said is
that the ≈10.63 positional attention carrier (10.630 nt) is not the ≈10.298 nucleosomal periodicity
measured in mouse ES-cell chromatin (Voong 2016, mm9), 0.33 nt away, roughly seven times the largest
uncertainty on either side, a powered refutation
([C14](../studies/splicebert_partner_selection/CLAIMS.md#10-the-chromatin-comparator-and-what-it-refutes)).
Its biological correlate and general downstream function remain unknown; crossing would need a candidate
correlate tied to the carrier by an independently capable assay.

### 3′-boundary organisation and U2AF1-centred site recognition

Crossed so far: nothing. The development instrument is unresolved, and the bridge is a motivation recorded
before testing. BG1 borrows only the biological −3 and +1 positions from published U2AF1 work ([Ilagan
2015](https://doi.org/10.1101/gr.181016.114), [Yoshida 2020](https://doi.org/10.1038/s41467-020-18559-6)),
freezing T-over-A and G-over-A contrasts on a synthetic panel that holds the acceptor `AG` and every
positioned `AG`/`GT`/`GC` fixed, and BG2 asks whether the same edited-window value changes travel the
established L3H1 route. With a synthetic panel, a masked-donor logit readout and a frozen model that has
no notion of binding, complex assembly or catalysis, the design can never identify a protein. A pass would
have located where the acceptor-local payload is sensitive and whether that sensitivity uses the known
route: at the model level, a functional and positional parallel to U2AF1-centred 3′-site recognition; it
does not imply that SpliceBERT implements U2AF1, binds RNA, or performs spliceosomal biochemistry. The
wording that would have required BG1 and BG2 both to pass and clear independent audit is frozen verbatim
in the protocol (§7). Instead the train-split preflight failed four frozen capability/resolution families,
BG1 validation and BG2 stayed sealed, and what may be said is empty:

> No result in this repository relates SpliceBERT to U2AF1. A frozen protocol exists that would test a
> positional parallel; its development instrument failed before either target opened.

[`docs/NOMENCLATURE.md`](NOMENCLATURE.md) gives the denial required with every use of the protein name
(standing phrase 3) and the constructions forbidden outright (its `U2AF1` row).

## Research operations

Each study keeps a preregistration for confirmatory gates, a claim ledger (status, evidence, scope, next
test), a decision log for changes made after seeing data, immutable result artifacts and manifests,
explicit retractions and supersessions, and figures generated from recorded artifacts. Exploration is
welcome if labeled, but must never silently become confirmation. The core principles:

1. Start with biology, then verify model behavior.
2. Preflight before interpretation.
3. Attribution ranks; intervention decides.
4. Necessity, mediation, direction, and specificity are different claims.
5. Controls must be capable of failure.
6. Evidence never auto-promotes between levels.
7. Preserve negative results and retractions as part of the method.
8. Stop at the first missing causal bridge.
