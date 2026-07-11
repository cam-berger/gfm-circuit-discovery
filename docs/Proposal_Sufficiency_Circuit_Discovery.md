# Discovering Relational Circuits in Genomic Foundation Models

**A causal program for finding and verifying the mechanisms a GFM uses to read DNA: a tracer validated against hand-built ground truth, two circuits traced end-to-end on real sequence, a biology-guided path to the rest, and a shortcut-gap hypothesis about which computations a model is likely to mechanize and which it may fake**

Owner: Cam Berger. Full evidence: `RESULTS.md` (every measured number, with status), `REC.md` (the argument), `docs/DECISIONS.md`. The tracer, the planted-twin ground truth, and the cue-conflict causal harness are shared across all rungs.

> **State at a glance (for the PI briefing).** Using a tracer validated against hand-built ground truth and a cue-conflict protocol that tells a real mechanism apart from a lookup shortcut, **two circuits are traced end-to-end on a genomic model**: an anchor-relative reading-frame circuit (a phase-maintenance mechanism) that locates the start codon and maintains reading-frame phase relative to it — whether it counts sequentially or looks up an anchor/position residue is unresolved (`docs/studies/reading_frame/COUNTER_VS_LOOKUP.md`) — and an in-context repeat-copy (induction) circuit that finds an earlier occurrence and copies it forward. The reading-frame trace surfaced **two mechanisms nobody hand-designed:** a *relay* (the start codon is causally load-bearing at range but not counted from) and an *anchor-free scaffold* (a circuit that computes the frame to its 0.945 ceiling by layer 2, then discards it at layer 4). The copy circuit is a distributed, partially-redundant set of five prefix-matching heads learned under a copy-position-weighted autoregressive objective (default 8× weight on copy positions); whether the same mechanism emerges under an unweighted objective remains untested. Organizing which circuits appear is the **shortcut-gap hypothesis:** the probability that a model learns a relational mechanism should increase with the objective advantage of that mechanism over a pre-specified registered shortcut class (`shortcut_gap = loss(best registered shortcut) − loss(best relational solution)`), while also depending on example frequency, architecture, capacity, regularization, initialization, and training budget. These two circuits sit on opposite sides of the predicted gap, and a **provisional, reproduction-blocked** necessity 2×2 on real DNA is consistent with it (cells C/D require real hg38 absent from a clean checkout, only one seed was run where three were pre-registered, and the mechanism must be read at the model's output; historically the anchored circuit held in three of four conditions and only the natural real-DNA cell dropped to a relay). What remains is (i) component localisation on the relay/scaffold, (ii) the biological circuits (splice-site pairing, NMD, CTCF orientation), (iii) one Gate-0 hole (LayerNorm), (iv) the objective-threshold number μ\*. See §4 for per-item status; a companion one-page explainer covers the methods for non-specialists.

![The reading-frame necessity 2×2, scored at the model's own output. Each point is one training condition; the y-axis is how much the far field follows the start codon under a cue conflict, the x-axis is how much destroying the anchor costs. (Provisional; reproduction blocked — one seed where three were pre-registered, cells C/D need real hg38 absent from a clean checkout.) Historically, the anchor-relative circuit (blue) held in three of four cells; only the natural real-DNA cell (D) dropped below the 0.50 threshold into a relay, and the plain language-model objective (μ=0, red) ignored the anchor entirely.]({{artifact:art_a118d2ca-9885-4ac8-bda5-1018f2e1f43e}})

---

## 1. How this problem is tackled today, and where current methods fall short


Interpretability of genomic foundation models (GFMs) has converged on **feature cataloguing:** train an SAE or scan attention heads, then label directions that correlate with annotations (Evo 2 SAE features; motif-aligned heads). This is presence-level: it labels *what a model represents*, not *how it computes*, and it presumes the underlying circuits are there to be catalogued.

This work discovers and verifies the circuits themselves, the attention heads and their interactions that carry a computation, with a tracer validated against circuits constructed by hand and a cue-conflict protocol that separates a genuine mechanism from a lookup shortcut. Pointing that instrument at real DNA models forces a prior question the field skipped: often there is **no circuit** to trace, because a pooled shortcut already suffices. So discovery has two halves, establishing which circuits *exist* before tracing them, and then tracing the ones that do, and both need an instrument that in-distribution attribution cannot provide.

Two method-level gaps explain why this was missed. First, **in-distribution attribution cannot separate a pooled statistic from a relational circuit:** shortcut and intended decision rules are provably indistinguishable on i.i.d. inputs (Geirhos et al. 2020), and every method in the standard roster (attribution graphs, transcoders, edge patching, JSAE, crosscoders) operates in-distribution. Start-codon *ablation* alone cannot detect the reading-frame circuit: destroying the anchor leaves the pooled route intact and the model falls back on it. Only a **cue conflict:** perturb the sequence so the anchor and the local statistics disagree, separates them. Second, the field reads the behavioural benchmarks (Mechanistic Invariance Test; DART-Eval; evolutionarily-implausible-sequence probes) as black-box *symptoms*. These are behavioural audits that **motivate** a mechanistic hypothesis — they do not by themselves prove it: the hypothesis is that these models are pooling, because pooling suffices, which this program then tests causally.

## 2. Novelty statement


This program asks two questions about circuits in genomic foundation models, in order:

- **What circuits form?:** Which relational mechanisms a GFM builds, discovered and traced causally on real DNA, using biology to supply the candidate computations (reading frame, in-context repeat copy, splice-site pairing / exon definition, CTCF orientation). Two are already traced end-to-end, an anchor-relative reading-frame circuit (a phase-maintenance mechanism; sequential counting vs anchor/position-residue lookup unresolved, `docs/studies/reading_frame/COUNTER_VS_LOOKUP.md`) and an induction-family repeat-copy circuit, and the trace has surfaced mechanisms nobody hand-designed (a relay, and an anchor-free scaffold that builds a pooled estimate then discards it).
- **How and when do they form?:** What predicts, *before training*, whether a given computation becomes a circuit at all. The working hypothesis (the *shortcut-gap hypothesis*): the probability a mechanism forms should increase with the objective advantage of the relational solution over a pre-specified registered pooled shortcut class, `shortcut_gap = loss(best registered shortcut) − loss(best relational solution)`, while also depending on example frequency, architecture, capacity, regularization, initialization, and training budget; this ranks genomic computations from shortcut-trivial to shortcut-proof and turns "is this circuit likely" into a quantity estimable in hours.

Both rest on instruments that are themselves contributions: **a tracer with a measured error rate**, validated against two hand-weighted transformers with identical behaviour and opposite mechanism (a positive control cannot falsify; twins can), and **a cue-conflict causal protocol** that separates a pooled statistic from a relational circuit where in-distribution attribution provably cannot.

## 3. Research question and hypotheses


**Primary questions:** (1) *What* relational circuits does a genomic foundation model build, and what do they compute, traced causally on real DNA, with biology supplying the candidate mechanisms? (2) *How and when* do they form, and does the shortcut gap between a pre-specified registered pooled shortcut class and the relational solution — measured before training — predict how likely a circuit is, ranking genomic computations into those worth tracing and those not?

- **H1 (shortcut-gap hypothesis).** The probability that a model learns a relational mechanism should increase with the objective advantage of that mechanism over a pre-specified registered shortcut class, while also depending on example frequency, architecture, capacity, regularization, initialization, and training budget; the measured quantity is `shortcut_gap = loss(best registered shortcut) − loss(best relational solution)`. This is a hypothesis to test, not a settled iff law, and the transition need not be a knife edge. A **provisional, reproduction-blocked** necessity 2×2 (shortcut absent/present × anchor-detector easy/hard) probes *what* displaces the anchor-relative circuit, read off a 2-D causal signature (far-field ATG-ablation cost × cue-conflict anchor share, both scored at the model's output). Historically, and pending reproduction, the anchor-relative circuit held in three of the four 2×2 cells (anchor share 0.71–1.00), and the model abandoned it only when the anchor was unreliable **and** a pooled shortcut was available — the fourth cell, natural real DNA, which built a **relay** (anchor share 0.23: the start codon is causally load-bearing but not counted from). The two factors appeared to interact: a hard-to-find anchor alone dropped the anchor share only to 0.88, a pooled shortcut alone only to 0.71, which is the condition of real genomic self-supervision. A **bag** (anchor unused, anchor share 0.09) appeared only under the plain language-model objective (μ=0), outside the 2×2. These 2×2 numbers are historical: cells C/D require real hg38 absent from a clean checkout, only one seed was run where three were pre-registered, and the mechanism must be read at the model output.
- **H2 (objective-as-demand).** Circuit completeness rises with the objective **only through the demand side** of the inequality, at fixed data. Varying the objective alone measures half the phenomenon and would credit the objective for an effect controlled by codon statistics.
- **H3 (attribution ranks, intervention decides).** In a GFM, a component's attribution rank does not determine its causal role, so no circuit claim can be settled by attribution alone, only by loss-based intervention. The sharp case is a naturally-occurring, causally-inert marker head: L1H2 carries 406× attention specificity yet ablation Δloss +0.000002, against the essential L1H3's +8.36.
- **H4 (in-context repeat-copy circuit under a copy-position-weighted objective).** An in-context repeat-copy task grows a genuine induction-family circuit, learned under a copy-position-weighted autoregressive objective (default 8× weight on copy positions); whether the same mechanism emerges under an unweighted objective remains untested. Copy accuracy 0.9608 (replicated 0.9662 at a second seed) against a base-rate ceiling of 0.2516, and under cue conflict the model follows the distal edited source at 0.9635 vs local continuation 0.0130. The ceiling is measured, not assumed: three no-source estimators, base-rate, local naive Bayes, and a receptive-field-limited CNN denied the source occurrence, all sit at chance (0.2473–0.2484), while a source oracle reaches 1.0000. A generalization control shows the circuit is a rule, not a lookup: comparable cross-entropy on fixed train/validation subsets stays matched throughout (train/val all-token CE 1.316/1.317; copy-position CE 0.091/0.095), and the grok event is localized to copy-position CE, it collapses as copy accuracy rises, while all-token CE moves only modestly because most background DNA is unpredictable. This is the positive counterpart to the reading-frame negative: where a shortcut is provably unavailable, the circuit appears under the weighted objective, and it generalizes.
- **H5 (the next circuits, named before they are traced).** The shortcut-gap hypothesis picks its own discovery targets: wherever a real regulatory decision turns on a relation that no bounded-window pooled statistic captures, the hypothesis predicts a GFM builds a circuit for it. We predict, and will test with the same cue-conflict protocol, relational circuits for **exon definition** (summing donor and acceptor splice-site strengths across an intron), the **NMD 50–55 nt rule** (distance from a premature stop to the last exon–exon junction), **CTCF convergent orientation** (orientation-dependent long-range contact), and **RNA secondary structure** (a base-pairing / Dyck-matching circuit, the theoretically shortcut-proof limit). Ordering these by shortcut-proofness is the hypothesis's own prediction of which appear; each has a biology-supplied cue conflict, so each is falsifiable. A reverse-complement-equivariant architecture (Caduceus) is a valuable later architecture comparison: it tests whether an orientation-sensitive relational computation is represented in an RC-consistent form. RC equivariance does not imply loss of relative orientation (equivariance ≠ invariance); the experiment should test representation and causal use of relative orientation, not assume blindness.

The in-context repeat-copy circuit (H4) is learned under a copy-position-weighted autoregressive objective (default 8× weight on copy positions; whether it emerges under an unweighted objective remains untested) and clears its measured ceiling by a wide margin:

![Repeat-copy accuracy over training, two seeds. Copy accuracy groks from the 0.252 base-rate floor to ~0.96, sharply crossing the 0.75 verdict threshold around step 3500–4000, and reproduces across seeds.]({{artifact:art_7c7be2bb-ce09-45db-b75c-b0337ec71a16}})

![The copy-circuit ceiling audit. Three estimators with no access to the source occurrence, base rate, local naive Bayes, and a receptive-field-limited CNN, all sit at chance (~0.25), while a source oracle and the trained model reach ~0.96–1.00. The stop-condition did not fire: the task is shortcut-proof by measurement.]({{artifact:art_79fb34fa-7155-4f82-8973-5ed37b876e35}})

![Per-head trace of the copy circuit. Five L1 heads carry the distal copy, each with a large attention-to-source and a large loss cost when ablated; the circuit is distributed and partially redundant rather than resting on a single head.]({{artifact:art_a5c06edd-885a-4936-b902-7c14fbbf0322}})

## 4. Task breakdown (input / output / status)


| Phase / Task | Description | Input | Output | Status |
|---|---|---|---|---|
| 0 | **Tracer ground truth (Gate 0).** Planted compositional + heuristic twins; Tests A–D | Hand-weighted models | OV+QK recall 1.0; conservation 8e-22; MCC 0.945; two blind spots | **Done** |
| 1.1 | **Reading-frame worked example.** Pooled ceiling; cue-conflict on real hg38 GFM-nano | hg38 + GENCODE CDS | Ceiling 0.9509 @240 nt; model at 0.7936, no anchored circuit | **Done** |
| 1.2 | **Forcing task.** Make the frame necessary; trace the resulting circuit | Uniform-codon CDS | Circuit @~3000 steps; L1H3 marker head; inert L1H2 look-alike | **Done** |
| 1.3 | **The sweep.** Move the ceiling at fixed objective (λ = 0→2, seven arms) | λ-biased CDS | Anchor-relative circuit at λ≤0.75, anchor-free at λ≥1; the shortcut-gap hypothesis | **Done** |
| 2.1 | **Necessity 2×2.** Cross shortcut-absent/present × detector-easy/hard | A/B/C/D cells | *Provisional (reproduction blocked):* historically the anchor-relative circuit held in 3 of 4 cells; displaced to a relay only when anchor unreliable **and** shortcut available (cell D); two factors interact; ATG causally load-bearing everywhere. Cells C/D need real hg38 (absent from clean checkout); one seed where three pre-registered; mechanism read at model output | **Provisional** |
| 2.1b | **Second circuit, in-context repeat copy.** Shortcut-proof Rung 3 | Synthetic repeat corpus | Copy circuit learned under a copy-position-weighted objective (8× copy-position weight; unweighted-objective emergence untested) — 0.96 vs 0.25 base rate, 2 seeds; ceiling stop-condition did not fire (RF-CNN at chance); QK-IG confirms prefix-keying; 5 distributed L1 heads, partially redundant | **Done** |
| 2.2 | **Move the demand.** μ mixture `(1−μ)·[predict nt] + μ·[predict aa]`, entropy-normalised terms | Real hg38 CDS | **μ\*** with LM-loss price, ≥3 seeds as interval, matched-LM-loss comparison | Next |
| 2.3 | **Mechanism trace.** Operational definition of counter/relay/bag; the anchor-free scaffold | 2×2 checkpoints | Three mechanisms defined by the intervention pair; cell C carries a 0.945 pooled estimate at L2 that L4 overrules; component localisation (T4) still open | **Done** (T4 next) |
| 2.4 | **Transplantation.** Hand a real-DNA model the circuit; fine-tune | Shared backbone | Survives vs erodes; registered upstream-ATG misfire prediction | Next |
| 3.1 | **Confirm on a Tier-3 task.** Exon definition / NMD junction rule under a biology-supplied cue conflict | GENCODE, SpliceBERT | Relational circuit present, sign-correct routing | Future |
| 3.2 | **The shortcut-proof case.** CTCF convergent orientation (future biological target); Caduceus as an RC-consistent representation test | Loop maps, Caduceus | Orientation circuit; test whether relative orientation is represented and causally used in an RC-equivariant model (equivariance ≠ invariance; no blindness assumed) | Future |

## 5. Datasets

The datasets fall into three roles: real sequence where a circuit is discovered, synthetic sequence where the pooled ceiling is moved under control, and the planted twins where the tracer's error rate is measured.


| Name | Role | Notes |
|---|---|---|
| hg38 + GENCODE CDS | Reading-frame ceiling, forcing/λ/μ arms, exon-definition | Random-offset windows; frame labels validated against biology (0 in-frame stops / 117,481) |
| Synthetic uniform / sense-61 / λ-biased CDS | Move the pooled ceiling at fixed objective | Ceilings measured per arm; sense-61 (zero codon bias, zero stops) isolates the anchor from stop-codon structure |
| Splice-site / exon annotations (GENCODE; SpliceBERT in-distribution) | Tier-3 confirmation (exon definition, NMD rule) | Contamination-firewalled: mechanistic claims only, no held-out chromosome for pretrained anchors |
| CTCF loop / orientation maps | Tier-4 shortcut-proof case; biology-supplied cue conflict | Occupancy-vs-loop dissociation on orientation flip |
| Planted twins | Gate 0; the measured tracer error rate | The only place the edge set is known by construction |

## 6. Models / baselines


- **Model, planted:** compositional + heuristic twins (hand-weighted). The ground truth.
- **Model, ours:** GFM-nano (6L × d256 × 8h, 4.80M, single-nt, LayerNorm) — a controlled genomic transformer trained from scratch, not a pretrained genomic foundation model (that term is reserved for SpliceBERT and similar). Carries every rung; the λ and μ arms.
- **Model, anchor:** SpliceBERT (single-nt, 6L, pre-mRNA). Tier-3 splice tasks in-distribution.
- **Model, published:** Caduceus (RC-equivariant). A later architecture comparison testing whether an orientation-sensitive computation is represented in an RC-consistent form (equivariance ≠ invariance; the test targets representation and causal use of relative orientation, not assumed blindness).
- **The decisive baseline, the pooled-statistic ceiling:** a bounded-window estimator (k-mer bag / local naive Bayes / RF-limited CNN). A circuit claim must beat this; if the model only matches it, the model is pooling.
- **External behavioural anchor:** the benchmark rankings (Mechanistic Invariance Test, DART-Eval), our mechanistic account should *explain* them, not restate them.

## 7. Methods and toolbox

The program runs on one shared instrument set, validated once on ground truth and reused at every rung. Identifiers for the methods below are held in `REC.md`'s reference list.

- **The tracer (replacement model, attribution graph, causal patching).** Each MLP is re-expressed as a sparse transcoder so that a dense, all-to-all layer becomes a sum of interpretable feature contributions, which turns the model into an attribution graph whose edges can be read and cut. Because genomic models are not supported by existing circuit-tracing libraries, the tracer is a from-scratch implementation; its faithfulness gap (the reconstruction error introduced by the replacement) is measured, not assumed.
- **Planted-twin ground truth.** Two hand-weighted transformers with identical behaviour and opposite mechanism (a compositional counter and a bag-of-motifs heuristic, |Δacc| = 0.0034) give the one setting where the true edge set is known by construction, so the tracer's recall and false-discovery rate are measurable rather than argued. A positive control cannot falsify a tracer; behaviourally-identical twins can.
- **The cue-conflict protocol.** In-distribution attribution cannot separate a pooled statistic from a relational circuit, because on i.i.d. inputs the shortcut and the intended rule agree on every example. The protocol perturbs the sequence so the two disagree (relocate the start codon, edit a distal repeat, flip an orientation) and reads which cue the model follows at its own output. This is the load-bearing methodological move: it succeeds where ablation fails, because destroying one route leaves the redundant route intact.
- **Integrated gradients (EAP-IG and QK-IG).** A naive pattern gradient vanishes on a confident head (max |dL/dscore| ~ 1e-15), so edge importance is computed by integrating along a path from a baseline, applied to both value-flow (EAP-IG) and attention-key (QK-IG) edges. QK-IG is what confirms a head keys on the feature it appears to attend to.
- **Loss-based intervention as the arbiter.** A component's attribution rank does not settle its causal role; only the change in loss when it is ablated does. Every ablation is scored on loss (not only accuracy, which saturates) and carries a matched control (decoy-Met for the start codon) and, where possible, a second opposite-direction intervention as confirmation.
- **The pooled-statistic ceiling estimators.** The decisive baseline for every task is the best bounded-window pooled statistic: a k-mer bag, a local naive Bayes, and a receptive-field-limited CNN denied the long-range cue. These are measured, not assumed, and re-measured per dataset, because a ceiling that holds on synthetic codons need not bound the shortcut on real DNA.
- **Linear probes for locating a computation across depth.** Layer-wise probes show where in the residual stream a quantity (frame phase, anchor identity) becomes decodable, which distinguishes a value computed across depth from one pooled at the input. Probes locate; they do not adjudicate causality; that is the intervention's job.

## 8. Metrics and success thresholds


- **The shortcut-gap hypothesis (H1):** per task, `shortcut_gap = loss(best registered shortcut) − loss(best relational solution)`. A larger positive gap predicts a higher probability the circuit forms (not a deterministic iff); verdict by cue conflict, not accuracy. On the 2×2: far-field cue-conflict anchor share > 0.50 **and** start-codon ablation ΔaaAcc > 0.10 beyond a decoy-Met control, frozen before any cell ran.
- **Tracer gate (H0, passed):** OV+QK edge recall > 0.9 vs planted twins; conservation < 1e-21; dictionary MCC > 0.9 where features are identifiable.
- **Mechanism separation:** the tracer must find a circuit in the compositional twin and *fail* in the behaviourally-identical heuristic twin (|Δacc| = 0.0034).
- **Objective threshold (2.2):** μ\* reported as an interval over ≥3 seeds, with the LM-loss cost, arms compared at **matched LM loss** (grokking makes the stopping point a choice).
- **H5 (later comparison):** test whether relative orientation is represented and causally used in an RC-equivariant model on the CTCF-orientation cue conflict (equivariance ≠ invariance; the test targets representation and causal use, not assumed blindness).

## 9. Ablations


- **Move the ceiling** (λ sweep, done) vs **move the demand** (μ mixture), the two axes of the inequality, separately.
- **Ceiling estimator class:** naive Bayes vs RF-limited CNN. A transformer with 256 nt of context may beat our codon-level estimator, moving tasks *down* the hierarchy; every §-4/§-listing ceiling is re-measured before it is trusted.
- **Cue conflict vs ablation:** the same target scored both ways, to show ablation is blind to a redundant circuit (D24).
- **Attribution method:** attention / naive gradient / integrated gradient / ablation-on-accuracy / ablation-on-loss, on the inert-head case (only the last decides).
- **LayerNorm:** planted circuit in an LN-free vs LN-equipped tiny transformer; re-run Tests A and C. Closes the last Gate-0 hole and **gates the trace/transplant claims**.
- **Background composition, tokenization resolution:** to rule out composition and position-blurring confounds.

## 10. Expected contributions


**What circuits form (discovery and analysis).**

- **Two relational circuits traced end-to-end on a GFM, causally.** An anchor-relative reading-frame circuit (a phase-maintenance mechanism; sequential counting vs anchor/position-residue lookup unresolved, `docs/studies/reading_frame/COUNTER_VS_LOOKUP.md`) surfaced on the forcing task, and an in-context repeat-copy **induction-family** circuit learned under a copy-position-weighted autoregressive objective (8× copy-position weight; unweighted-objective emergence untested). The copy circuit is traced to full standard: QK integrated gradients confirm the five load-bearing L1 heads key on the matching prefix (source-edge share ~0.35 vs ~0.005 on a control position), a prefix-scramble control collapses distal selection from 0.96 to chance, and single- vs subset-ablation shows the heads are partially redundant (Σ single-head Δloss = 2.91, joint five-head Δ = 1.32), a genuine distributed circuit, not one head.
- **Two mechanisms nobody hand-designed.** The trace surfaced a **relay** (the anchor is causally load-bearing at range but not counted from) and an **anchor-free scaffold** (a circuit that builds the full pooled estimate to its 0.945 ceiling by layer 2 without the start codon, then overrules it at layer 4, carrying a 0.945-accurate answer it discards). Characterising these at the component level is the highest-value trace in hand.
- **A path to the first *biological* relational circuit.** The same discovery protocol, pointed at splice-site pairing / exon definition (donor–acceptor strength summation), the NMD junction rule, and CTCF convergent orientation, where biology supplies the cue conflict and composition can be held fixed while the label moves.
- **A validated tracer with a measured error rate** and a cue-conflict causal protocol, plus the reusable "attribution ranks, intervention decides" result on a naturally-occurring inert head, the instruments the discoveries rest on.

**How and when they form (the shortcut-gap hypothesis).**

- **A measurable, testable hypothesis** (the shortcut-gap hypothesis) about whether a genomic model will build a relational circuit, usable before training, and a mechanistic explanation of the behavioural positional-logic failures the field has only observed black-box. It ranks genomic computations shortcut-trivial → shortcut-proof.
- **Formation appears to be governed by an interaction, not a single knob.** In the provisional (reproduction-blocked) 2×2, the anchor-relative circuit was displaced only when the anchor was unreliable **and** a pooled shortcut was available, neither factor alone sufficing (anchor share dropped to 0.88 and 0.71 respectively, but to 0.23 with both). The reading-frame circuit (displaced as a shortcut becomes available) and the repeat-copy circuit (learned under a copy-position-weighted objective) illustrate the hypothesis in both directions.
- **μ\***, the smallest objective demand that installs a circuit, with its LM-loss price, the actionable number the objective literature has not produced.
- **A falsifiable test** on a published architecture: whether an RC-equivariant model (Caduceus) represents and causally uses relative orientation on the CTCF-orientation cue conflict (equivariance ≠ invariance; the test does not assume blindness).

## 11. Literature review


*Identifiers are held in `REC.md`'s reference list and cited from there verbatim; this table names works descriptively and states only the relevance/weakness, to avoid propagating hand-copied citation identifiers.*

| Work | Relevance | Weakness this proposal targets |
|---|---|---|
| Geirhos et al., *Shortcut Learning* (2020) | Shortcut vs intended rules indistinguishable on i.i.d. data; needs OOD interventions | Not yet applied to GFM interpretability, which is entirely in-distribution |
| Shah et al., *Pitfalls of Simplicity Bias* (2020) | SGD latches onto the simplest predictor even when less predictive, supplies the "only if" | Not connected to whether genomic circuits form |
| Mechanistic Invariance Test (2026) | Behavioural: gLMs solve positional tasks via composition, not position | Black-box; reports failure, not the pooling mechanism behind it |
| Evolutionarily-implausible-sequence probes | Accuracy collapses off-distribution → distributional prior, not computation | Behavioural; read here as mechanistic evidence for pooling |
| DART-Eval; foundation-model-vs-CNN benchmark | gLMs do not consistently beat simple ab initio baselines — a behavioural audit that motivates the mechanistic question | Behavioural motivation, not mechanistic proof; this program supplies the causal test |
| Goodfire × Arc, *Interpreting Evo 2* | SAE features incl. motifs, exon/intron | Presence-level; presumes circuits exist; no cue-conflict causal test |
| Exon-definition splice-site pairing (MPRA/genomics) | Inclusion tracks the *sum* of both splice-site strengths, a Tier-3 relation | Not used to test whether a model represents the relation |
| CTCF convergent-orientation looping | Orientation carries loop information independent of binding, a bench cue conflict | Not used as an interpretability ground truth |
| CREME / Enformer in-silico perturbation | The one architecture with a genuine distance-dependent relational circuit | Supervised, demand-heavy, the shortcut-gap hypothesis in the wild, not yet named as such |
| Group tracer + planted twins (internal, this work) | The causal instrument and its measured error rate | Established on ground truth; reused, not assumed |

## 12. Gap in the literature

The works above each frame an open problem this program takes up directly. Collecting them states the gap in one place.

- **Shortcut vs intended rule is undecidable in-distribution, and no one has applied the fix to GFMs.** The shortcut-learning and simplicity-bias literature establishes that the two are indistinguishable on i.i.d. data and that the remedy is an out-of-distribution intervention, but genomic-model interpretability remains almost entirely in-distribution (SAE feature catalogues, attention-head labelling). The open problem is a causal, off-distribution test of whether a labelled feature carries a computation. The cue-conflict protocol is that test.
- **The behavioural benchmarks report a symptom without its mechanism.** Positional-invariance probes, evolutionary-implausibility probes, and foundation-model-vs-CNN comparisons all show that genomic models often fail to use exact position and do not reliably beat simple baselines, but they read these as black-box scores. The open problem they leave is the mechanistic account: *why* the models behave this way. The shortcut-gap hypothesis proposes an answer to test: the models are pooling, because pooling suffices.
- **Feature cataloguing presumes the circuits it does not verify.** Work interpreting large genomic models (for example SAE features in Evo 2) is presence-level: it labels what a model represents and assumes a circuit underlies each feature, with no cue-conflict causal test and no check that a circuit exists at all. The open problem is existence before tracing, which this work makes the first question rather than an assumption.
- **The objective literature never produced the number.** Discussions of how the training objective shapes what a model learns have not delivered the smallest objective demand that installs a given circuit, at fixed data, with the language-modelling loss it costs. The open problem is a quantitative demand threshold. This program reports μ\* as an interval with its loss price.
- **No prior work ranks genomic computations by whether a mechanism is avoidable.** Biology supplies many candidate relations (splice-site pairing, the NMD junction rule, CTCF orientation, RNA base-pairing), but the literature does not order them by whether a bounded-window statistic can fake them, which is what predicts, before training, which ones a model must mechanize. That ranking is this program's organizing prediction.

## 13. Limitations and open questions


- **The ceiling estimator is a lower bound on the shortcut.** A naive-Bayes or codon-level ceiling underestimates what a higher-capacity bounded-window estimator can reach: cell B reaches 0.9735 against a 0.926 detector ceiling, cell C reaches 0.9997 against a 0.9509 pooled ceiling. A higher-capacity estimator moves tasks down the hierarchy, this cuts *against* the hypothesis, making circuits rarer and any positive finding stronger, so every ceiling is re-measured with a model-class-matched (RF-limited CNN) estimator before it is trusted.
- **A probe read at a fixed layer measures whichever variable that layer carries.** The cue-conflict anchor share must be scored at the model's own output, not off a linear probe at an intermediate layer: on real DNA the anchor does not enter the residual stream until layer 4, so a probe read at layer 2 reports a cell as anchor-free when its output is anchored. Both 2×2 axes are therefore read at the output, and every mechanism call carries a paired decoy-Met control and a second, opposite-direction intervention (anchor relocation) as confirmation.
- **Nothing transfers from the synthetic λ arms to real DNA without re-measurement.** Naive-Bayes independence closes λ = 1's ceiling to 1.0000 while real CDS is 0.9509, because real codons correlate, a ceiling measured on synthetic codons does not bound the shortcut on real DNA.
- **The hypothesis is a conjecture on n = 8 arms, one task, one architecture.** The provisional 2×2 probes it on real DNA (reproduction blocked); it must be reproduced and also confirmed on a Tier-3 task (where the prediction is a *circuit*) before being treated as anything firmer than a hypothesis.
- **The forcing task is synthetic and a reviewer will say so.** Its job is not realism but to prove the tracer finds a real learned circuit when one exists, so the hg38 negative is a fact about DNA, not about our instruments. The hg38 result is the finding; the synthetic model licenses it; the planted models prove the instruments work.
- **LayerNorm is unmeasured.** Every tracer error-rate number came from LN-free planted models; GFM-nano has LayerNorm. One day of work closes it, and it gates the causal claims.
- **μ is not a clean fraction of demand** until both loss terms are entropy-normalised (`ln 21` vs `ln 4`); a negative at small μ is undertraining until amino-acid accuracy is checked against the 0.3768 guessing ceiling.
- **A partial graft is meaningless without representational alignment**, and the adapter's capacity then confounds any positive transplant result.
- Causal patching on hybrid/SSM architectures (Caduceus, Evo 2) is less standardised than on transformers and needs its own sanity checks.

## 14. References

*This list is reproduced as provided; identifiers are not independently verified here and should be checked against `REC.md`'s reference list before external use.*

| Category | Work | Role in this program |
|---|---|---|
| Circuit-tracing tools | Ameisen, Lindsey et al., *Circuit Tracing: Revealing Computational Graphs in Language Models* (Anthropic, transformer-circuits.pub, 2025) | The main method template: cross-layer transcoders + attribution graphs |
| Circuit-tracing tools | Dunefsky et al., *Transcoders Find Interpretable LLM Feature Circuits* | Core technical basis for transcoders as circuit tools |
| Circuit-tracing tools | Marks et al., *Sparse Feature Circuits* | Fallback if cross-layer transcoders are too hard, or if SAE-node circuits are more feasible (the pre-registered SSM fallback) |
| Circuit-tracing tools | Syed, Rager, Conmy, *Attribution Patching Outperforms Automated Circuit Discovery* | The efficient edge-screening method (EAP / EAP-IG): the cheap screen before exact patching |
| Circuit-tracing tools | Goldowsky-Dill et al., *Localizing Model Behavior with Path Patching* | Testing pathways rather than components: the real-model edge intervention behind H4 |
| Circuit-tracing tools | Farnik et al., *Jacobian Sparse Autoencoders* (arXiv 2502.18147) | Most relevant to the real-composition vs bag-of-motifs discriminator (H3) |
| How a model's concepts form internally | Li et al., *Othello-GPT / Emergent World Representations* (with Nanda et al. linear/causal follow-up) | Conceptual precedent for a hidden state learned from next-token prediction: the template for "an objective installs a latent structure" |
| How a model's concepts form internally | Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability* | Precedent for "training creates a mechanism over time," and the recipe for planting a known-feature step |
| Genomic FMs: testbeds and feature-level foil | DNABERT-2 | First transformer-GFM testbed; the model in the 2025 G4 benchmark (Rung 3 anchor) |
| Genomic FMs: testbeds and feature-level foil | Nucleotide Transformer | Strong transformer-family GFM baseline |
| Genomic FMs: testbeds and feature-level foil | Goodfire / Arc, *Interpreting Evo 2* | The direct feature-level GFM interpretability foil ("proto-circuits") |

---

*Full evidence, decision log, and the reference list with verified identifiers: `RESULTS.md`, `REC.md`, and `docs/DECISIONS.md`. The highest-value biological targets are **exon definition**, the **NMD junction rule**, and **CTCF orientation:** where composition can be held fixed while the label moves, and biology has already run the cue-conflict experiment.*

---

## Claim-status table

This table records the correct current status of each headline claim, including claims that have been weakened or withdrawn. It is authoritative where the prose above is looser.

| Claim | Correct current status |
|---|---|
| Planted twins discriminate compositional and shortcut mechanisms | Settled within the planted architecture |
| Frozen-attention tracer misses QK selection edges | Settled |
| Integrated-gradient QK method recovers the planted marker edge | Settled within planted models |
| LayerNorm faithfulness is calibrated | Not yet; required |
| Real-DNA next-token GFM-nano uses an anchored frame circuit | Current evidence says no at long range for the tested condition |
| Forced amino-acid model learns an anchor-relative modulo-3 circuit | Supported; sequential counting unresolved |
| Inert marker head demonstrates attention is not causality | Supported for the analyzed checkpoint |
| Repeat-copy circuit emerges without forcing | **False — measured.** Circuit absent at copy weight 1 (0/3 seeds) and 2; forms only at weight ≥ 4 (`scripts/eval_repeat_copy_sweep.py`). Forced, not unbidden. |
| Repeat-copy circuit exists under weighted objective | Supported on the reported checkpoints; artifacts must be restored/pinned |
| Sufficiency law is an iff law | Unsupported; replace with shortcut-gap hypothesis |
| Reading-frame 2×2 is settled | Provisional/reproduction blocked |
| RC equivariance makes Caduceus orientation blind | Incorrect |
| SpliceBERT partner-aware circuit exists | Open research question |