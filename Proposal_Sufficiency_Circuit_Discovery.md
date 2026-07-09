# When Does a Genomic Model Build a Circuit at All?

**A mechanistic test of the hypothesis that a foundation model builds a relational circuit *if and only if* no statistic pooled over its context window is a sufficient statistic for the target — and a shortcut-ranked program of circuit discovery that follows from it**

Supersedes both *Proposal D2 (Full Circuit Discovery in GFMs)* and *Proposal D1 (Regulatory Grammar Inside GFMs)*. Owner: Cam Berger. Full evidence: `RESULTS.md` (every measured number, with status), `REC.md` (the argument), `docs/DECISIONS.md` (D1–D32). Shares the tracer, the planted-twin ground truth, and the cue-conflict causal harness across all rungs.

> **State at a glance (for the PI briefing).** The tracer is built and validated against hand-wired ground truth. Two learned circuits have been traced end-to-end — a reading-frame counter and an in-context repeat-copy circuit — on *opposite* sides of the criterion. The necessity 2×2 on real DNA is complete and did something better than confirm the binary: it resolved a **three-mechanism spectrum** (bag → relay → counter). What remains is (i) one Gate-0 hole (LayerNorm), (ii) the objective-threshold number μ\*, and (iii) the climb to biological Tier-3/4 tasks. See §4 for per-item status; a companion one-page explainer covers the methods for non-specialists.

---

## 1. How this problem is tackled today, and where current methods fall short

Interpretability of genomic foundation models (GFMs) has converged on **feature cataloguing** — train an SAE or scan attention heads, then label directions that correlate with annotations (Evo 2 SAE features; motif-aligned heads). At the frontier it has begun to ask *how* a computation is performed, and the prevailing framing (including our own earlier proposals) **presumes the circuits are there** and treats *tracing* as the open problem.

Our results invert that framing. We built a tracer, validated it against circuits we constructed by hand, pointed it at real DNA models — and, using an out-of-distribution cue-conflict protocol rather than in-distribution attribution, found **no circuit** where one was expected. **Existence, not tracing, is the open problem, and the modal answer is *no circuit*.**

Two method-level gaps explain why this was missed. First, **in-distribution attribution cannot separate a pooled statistic from a relational circuit** — shortcut and intended decision rules are provably indistinguishable on i.i.d. inputs (Geirhos et al. 2020), and every method in the standard roster (attribution graphs, transcoders, edge patching, JSAE, crosscoders) operates in-distribution. Our own start-codon *ablation* could not detect the reading-frame circuit: destroying the anchor leaves the pooled route intact and the model falls back on it. Only a **cue conflict** — perturb the sequence so the anchor and the local statistics disagree — separates them (D24). Second, the field reads the behavioural benchmarks (Mechanistic Invariance Test; DART-Eval; evolutionarily-implausible-sequence probes) as black-box *symptoms*. They are converging evidence for a **mechanistic** claim: these models are pooling, because pooling suffices.

## 2. Novelty statement

What is unexamined is **when a genomic model builds a relational mechanism at all**, and whether that is predictable *before training*. The novel contributions:

1. **A predictive criterion for circuit existence.** A mechanism is unavoidable exactly when no bounded-window pooled statistic is a *sufficient statistic* for the target — equivalently, `circuit forms ⟺ accuracy demanded by the objective > ceiling of the best pooled statistic`. The ceiling is estimable in hours (k-mer bag / local naive Bayes / receptive-field-limited CNN); if it meets the demand, there is nothing to trace.
2. **A tracer with a measured error rate**, validated against two hand-weighted transformers with identical behaviour and opposite mechanism (a positive control cannot falsify; twins can), carrying the cue-conflict protocol that in-distribution attribution provably lacks.
3. **A shortcut-availability hierarchy** — the first ranking of genomic computations by whether they admit a mechanism, from shortcut-trivial (reading frame, CpG) to shortcut-proof (CTCF orientation, base pairing) — which relocates the previous flagship (G-quadruplex) from "hardest rung" to a mid-tier task with a pooled proxy.
4. **The objective question, correctly posed** — demand vs ceiling, moved on both axes — yielding μ\*, the smallest objective weight that installs a circuit, reported with the language-modelling loss it costs.

## 3. Research question and hypotheses

**Primary question:** Is the existence of a relational circuit in a GFM predicted by the sufficiency of a bounded-window pooled statistic for the target — and does that criterion rank genomic computations, before training, into those worth tracing and those not?

- **H1 (sufficiency law).** *Supported, and refined into a spectrum.* A circuit forms iff the objective's demanded accuracy exceeds the pooled ceiling; the circuit appears when the pooled route is *imperfect*, not merely *weak* — a 96.5%-accurate shortcut does not displace a counter when the task demands 100%, so the transition is a knife edge that looks like grokking. The completed 2×2 sharpened "circuit / no circuit" into **three mechanisms** read off a 2-D causal signature (far-field ATG-ablation cost × cue-conflict anchor share): a **bag** (real DNA, low demand), a **relay** (real DNA, high demand — the anchor is causally load-bearing at range but the frame is re-established locally rather than globally counted), and an **anchored counter** (shortcut absent, high demand). Increasing demand moves bag → relay → counter; shortcut availability decides where it stops.
- **H2 (objective-as-demand, corrected).** Circuit completeness rises with the objective **only through the demand side** of the inequality, at fixed data. Varying the objective alone (the previous H2) measures half the phenomenon and would credit the objective for an effect controlled by codon statistics.
- **H3 (attribution ranks; intervention decides).** *Established.* Every attribution method mis-ranks a naturally-occurring, causally-inert marker head (L1H2: 406× attention specificity, ablation Δloss +0.000002, vs the essential L1H3's +8.36); only loss-based intervention decides.
- **H4 (shortcut-proof circuits form unbidden).** *Supported by a second circuit.* An in-context repeat-copy task — shortcut-proof by construction (random held-out units, prefix longer than any local window) — grows a genuine induction-family circuit with **no** forcing objective: base-rate ceiling 0.2516, copy accuracy 0.9608, and under cue conflict the model follows the distal edited source at 0.9635 vs local continuation 0.0130. This is the positive counterpart to the reading-frame negative: where a shortcut is truly unavailable, the circuit appears without coercion.
- **H5 (RC-equivariance is blindness).** For a reverse-complement-equivariant architecture (Caduceus), a motif and its RC are identified by construction, so orientation lies outside the hypothesis class — the model is structurally blind to the Tier-4 CTCF-orientation computation. A falsifiable prediction about a published model, one afternoon to test.
- ~~JSAE composition-sparsity~~ demoted (D16: measures in-degree, not composition). ~~Developmental / phase-change~~ cut (D11).

## 4. Task breakdown (input / output / status)

| Phase / Task | Description | Input | Output | Status |
|---|---|---|---|---|
| 0 | **Tracer ground truth (Gate 0).** Planted compositional + heuristic twins; Tests A–D | Hand-weighted models | OV+QK recall 1.0; conservation 8e-22; MCC 0.945; two blind spots | **Done** |
| 1.1 | **Reading-frame worked example.** Pooled ceiling; cue-conflict on real hg38 GFM-nano | hg38 + GENCODE CDS | Ceiling 0.9509 @240 nt; model at 0.7936, no anchored circuit | **Done** |
| 1.2 | **Forcing task.** Make the frame necessary; trace the resulting circuit | Uniform-codon CDS | Circuit @~3000 steps; L1H3 marker head; inert L1H2 look-alike | **Done** |
| 1.3 | **The sweep.** Move the ceiling at fixed objective (λ = 0→2, seven arms) | λ-biased CDS | Anchored counter at λ≤0.75, anchor-free at λ≥1; the criterion | **Done** |
| 2.1 | **Necessity 2×2.** Cross shortcut-absent/present × detector-easy/hard | A/B/C/D cells | **Three mechanisms** (bag→relay→counter); ATG load-bearing in all cells; D30 counter-prediction half-falsified (relay, not counter) | **Done** |
| 2.1b | **Second circuit — in-context repeat copy.** Shortcut-proof Rung 3 | Synthetic repeat corpus | Copy circuit forms unbidden; cue conflict 0.96 distal; distributed L1 copy heads | **Done** |
| 2.2 | **Move the demand.** μ mixture `(1−μ)·[predict nt] + μ·[predict aa]`, entropy-normalised terms | Real hg38 CDS | **μ\*** with LM-loss price, ≥3 seeds as interval, matched-LM-loss comparison | Next |
| 2.3 | **Trace the relay.** How a relay differs mechanistically from bag and counter | 2×2 checkpoints | Local re-sync vs anchor seed; operational definition of "relay" | Next (high value) |
| 2.4 | **Transplantation.** Hand a real-DNA model the circuit; fine-tune | Shared backbone | Survives vs erodes; registered upstream-ATG misfire prediction | Next |
| 3.1 | **Confirm on a Tier-3 task.** Exon definition / NMD junction rule under a biology-supplied cue conflict | GENCODE, SpliceBERT | Relational circuit present, sign-correct routing | Future |
| 3.2 | **The shortcut-proof case.** CTCF convergent orientation; the RC-equivariance blindness test | Loop maps, Caduceus | Orientation circuit; RC-model failure prediction (one afternoon) | Future |

## 5. Datasets

| Name | Role | Notes |
|---|---|---|
| hg38 + GENCODE CDS | Reading-frame ceiling, forcing/λ/μ arms, exon-definition | Random-offset windows; frame labels validated against biology (0 in-frame stops / 117,481) |
| Synthetic uniform / sense-61 / λ-biased CDS | Move the pooled ceiling at fixed objective | Ceilings measured; sense-61 control falsified the stop-depletion story (D30) |
| Splice-site / exon annotations (GENCODE; SpliceBERT in-distribution) | Tier-3 confirmation (exon definition, NMD rule) | Contamination-firewalled: mechanistic claims only, no held-out chromosome for pretrained anchors |
| CTCF loop / orientation maps | Tier-4 shortcut-proof case; biology-supplied cue conflict | Occupancy-vs-loop dissociation on orientation flip |
| Planted twins | Gate 0; the measured tracer error rate | The only place the edge set is known by construction |

## 6. Models / baselines

- **Model — planted:** compositional + heuristic twins (hand-weighted). The ground truth.
- **Model — ours:** GFM-nano (6L × d256 × 8h, 4.80M, single-nt, LayerNorm). Carries every rung; the λ and μ arms.
- **Model — anchor:** SpliceBERT (single-nt, 6L, pre-mRNA). Tier-3 splice tasks in-distribution.
- **Model — published:** Caduceus (RC-equivariant). The H4 blindness prediction.
- **The decisive baseline — the pooled-statistic ceiling:** a bounded-window estimator (k-mer bag / local naive Bayes / RF-limited CNN). A circuit claim must beat this; if the model only matches it, the model is pooling.
- **External behavioural anchor:** the benchmark rankings (Mechanistic Invariance Test, DART-Eval) — our mechanistic account should *explain* them, not restate them.

## 7. Metrics and success thresholds

- **The criterion (H1):** per task, `demanded accuracy − pooled ceiling`. Circuit predicted iff positive; verdict by cue conflict, not accuracy. On the 2×2: far-field cue-conflict anchor share > 0.50 **and** start-codon ablation ΔaaAcc > 0.10 beyond a decoy-Met control — frozen before any cell ran.
- **Tracer gate (H0, passed):** OV+QK edge recall > 0.9 vs planted twins; conservation < 1e-21; dictionary MCC > 0.9 where features are identifiable.
- **Mechanism separation:** the tracer must find a circuit in the compositional twin and *fail* in the behaviourally-identical heuristic twin (|Δacc| = 0.0034).
- **Objective threshold (2.2):** μ\* reported as an interval over ≥3 seeds, with the LM-loss cost, arms compared at **matched LM loss** (grokking makes the stopping point a choice).
- **H4 (one afternoon):** an RC-equivariant model scores below a non-equivariant one on the CTCF-orientation cue conflict, by a pre-registered margin.

## 8. Ablations

- **Move the ceiling** (λ sweep, done) vs **move the demand** (μ mixture) — the two axes of the inequality, separately.
- **Ceiling estimator class** — naive Bayes vs RF-limited CNN. A transformer with 256 nt of context may beat our codon-level estimator, moving tasks *down* the hierarchy; every §-4/§-listing ceiling is re-measured before it is trusted.
- **Cue conflict vs ablation** — the same target scored both ways, to show ablation is blind to a redundant circuit (D24).
- **Attribution method** — attention / naive gradient / integrated gradient / ablation-on-accuracy / ablation-on-loss, on the inert-head case (only the last decides).
- **LayerNorm** — planted circuit in an LN-free vs LN-equipped tiny transformer; re-run Tests A and C. Closes the last Gate-0 hole and **gates the trace/transplant claims**.
- **Background composition, tokenization resolution** — to rule out composition and position-blurring confounds.

## 9. Expected contributions

- A **predictive, measurable criterion** for whether a genomic model will build a relational circuit — usable before training, and a mechanistic explanation of the behavioural positional-logic failures the field has only observed black-box.
- **A three-mechanism spectrum, not a binary.** The novel empirical object is the **relay** — a mechanism that is neither a bag nor a global counter, appearing on real DNA under high demand, where the anchor is causally load-bearing everywhere yet loses to local content under conflict. Bag and counter were both hand-buildable in advance; the relay was not, and how it works in the weights is the highest-value trace left in the program.
- **Two circuits on opposite sides of the line.** Reading frame (avoidable → relay/bag on real DNA) and in-context repeat copy (shortcut-proof → clean circuit, formed unbidden) together demonstrate the criterion in both directions — a stronger claim than either alone.
- A **shortcut-availability hierarchy** of genomic computations (shortcut-trivial → shortcut-proof), reclassifying reading frame as the wrong proof-of-method rung and G-quadruplex as a mid-tier task with a pooled proxy.
- A **validated tracer with a measured error rate** and a cue-conflict causal protocol, plus the reusable "attribution ranks, intervention decides" result on a naturally-occurring inert head.
- **μ\***, the smallest objective demand that installs a circuit, with its LM-loss price — the actionable number the objective literature never produced.
- A **falsifiable, cheap prediction** about a published architecture (RC-equivariance is blindness to orientation).

## 10. Literature review

*Identifiers are held in `REC.md`'s reference list and cited from there verbatim; this table names works descriptively and states only the relevance/weakness, to avoid propagating hand-copied citation identifiers.*

| Work | Relevance | Weakness this proposal targets |
|---|---|---|
| Geirhos et al., *Shortcut Learning* (2020) | Shortcut vs intended rules indistinguishable on i.i.d. data; needs OOD interventions | Not yet applied to GFM interpretability, which is entirely in-distribution |
| Shah et al., *Pitfalls of Simplicity Bias* (2020) | SGD latches onto the simplest predictor even when less predictive — supplies the "only if" | Not connected to whether genomic circuits form |
| Mechanistic Invariance Test (2026) | Behavioural: gLMs solve positional tasks via composition, not position | Black-box; reports failure, not the pooling mechanism behind it |
| Evolutionarily-implausible-sequence probes | Accuracy collapses off-distribution → distributional prior, not computation | Behavioural; read here as mechanistic evidence for pooling |
| DART-Eval; foundation-model-vs-CNN benchmark | gLMs do not consistently beat simple ab initio baselines | Treated as benchmark noise, not as evidence circuits are absent |
| Goodfire × Arc, *Interpreting Evo 2* | SAE features incl. motifs, exon/intron | Presence-level; presumes circuits exist; no cue-conflict causal test |
| Exon-definition splice-site pairing (MPRA/genomics) | Inclusion tracks the *sum* of both splice-site strengths — a Tier-3 relation | Not used to test whether a model represents the relation |
| CTCF convergent-orientation looping | Orientation carries loop information independent of binding — a bench cue conflict | Not used as an interpretability ground truth |
| CREME / Enformer in-silico perturbation | The one architecture with a genuine distance-dependent relational circuit | Supervised, demand-heavy — the criterion in the wild, not yet named as such |
| Group tracer + planted twins (internal, this work) | The causal instrument and its measured error rate | Established on ground truth; reused, not assumed |

## 11. Limitations and open questions

- **The ceiling estimator is a lower bound on the shortcut — now confirmed twice.** Both cells whose ceiling we predicted from were beaten: cell B reached 0.9735 vs a 0.926 detector ceiling, cell C reached 0.9997 vs a 0.9509 pooled ceiling. A higher-capacity estimator moves tasks down the hierarchy — this cuts *against* us, making circuits rarer and the finding stronger — but every ceiling must be re-measured with a model-class-matched (RF-limited CNN) estimator before it is trusted.
- **The binary verdict is lossy.** The pre-registered "circuit / no circuit" verdict prints NO CIRCUIT for the relay cells (B, C, D); the result is the 2-D signature, which we pre-registered to report regardless. The verdict machinery should be replaced by the (ATG-ablation × anchor-share) plane.
- **Nothing transfers from the synthetic λ arms to real DNA without re-measurement** (D30). Naive-Bayes independence closes λ = 1's ceiling to 1.0000 while real CDS is 0.9509, because real codons correlate. We made exactly this error, for an hour.
- **The law is a conjecture on n = 8 arms, one task, one architecture.** The 2×2 tests it on real DNA; it must also be confirmed on a Tier-3 task (where the prediction is a *circuit*) before being stated as a law.
- **The forcing task is synthetic and a reviewer will say so.** Its job is not realism but to prove the tracer finds a real learned circuit when one exists — so the hg38 negative is a fact about DNA, not about our instruments. The hg38 result is the finding; the synthetic model licenses it; the planted models prove the instruments work.
- **LayerNorm is unmeasured.** Every tracer error-rate number came from LN-free planted models; GFM-nano has LayerNorm. One day of work closes it, and it gates the causal claims.
- **μ is not a clean fraction of demand** until both loss terms are entropy-normalised (`ln 21` vs `ln 4`); a negative at small μ is undertraining until amino-acid accuracy is checked against the 0.3768 guessing ceiling.
- **A partial graft is meaningless without representational alignment**, and the adapter's capacity then confounds any positive transplant result.
- Causal patching on hybrid/SSM architectures (Caduceus, Evo 2) is less standardised than on transformers and needs its own sanity checks.

---

*Full evidence, decision log, and the reference list with verified identifiers: `RESULTS.md`, `REC.md`, and `docs/DECISIONS.md` (D1–D32). Reading frame was the wrong rung; the right ones are **exon definition**, the **NMD junction rule**, and **CTCF orientation** — where composition can be held fixed while the label moves, and biology has already run the cue-conflict experiment for us.*