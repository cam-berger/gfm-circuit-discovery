# When Does a Genomic Model Build a Circuit? A Sufficiency Criterion for Mechanism, and a Shortcut-Ranked Program of Circuit Discovery

### Research Proposal — v3 (2026-07-09)

> **This supersedes `Proposal_D2_Full_Circuit_Discovery_GFMs_v2.md`.** That proposal asked *does a
> training objective install a circuit* and ordered its work by biological complexity. Both premises
> were wrong, and we have the data to say why. The reframing is recorded in `REC.md` (the critique of
> D2) and `docs/DECISIONS.md` (D1–D30); this document is the forward-looking program that replaces D2.
>
> **The thesis in one line:** a model builds a relational circuit **if and only if no statistic pooled
> over its context window is a sufficient statistic for the target** — and that line is measurable, per
> task, *before you train anything*.
>
> **What is already done** (102 tests passing): the tracer is built and validated against hand-wired
> ground truth (Gate 0 passed); the reading-frame case is fully worked, negative on real DNA and
> positive under a forcing objective, with the mechanism traced; the criterion is supported on an
> eight-arm sweep. This proposal's job is to (a) state the criterion as the contribution, (b) rank
> fifteen genomic computations by it, and (c) confirm it on shortcut-proof tasks D2 never named.

---

## Part A: Research Proposal

### Core idea

Genomic-foundation-model interpretability has catalogued *what* is represented and, at the frontier,
begun to ask *how the computation is performed*. Proposal D2 sat here: it presumed the circuits exist
(Anthropic named Evo 2 features "proto-circuits") and treated tracing as the open problem.

Our result inverts that. **Existence is the open problem, and the modal answer is *no circuit*.** We
built a validated tracer, pointed it at real DNA models, and — using an out-of-distribution
cue-conflict protocol, not in-distribution attribution — found that the reading-frame representation is
a pooled statistic, not a mechanism. Then we found the governing law, and it is not about objectives:

```
circuit forms   ⟺   accuracy demanded by the objective   >   ceiling of the best pooled statistic
```

This is a claim with a number for an answer, measurable before training, and it reorganises the entire
program: **from "which objectives install circuits" to "which computations over DNA are shortcut-proof."**

### The criterion, and why it is the right one

> **A mechanism is unavoidable exactly when no statistic pooled over the model's context window is a
> sufficient statistic for the target.**

Three legs, two from the literature and one from our data:

1. **Sufficiency (the "if").** If a bounded-window statistic `T(x)` satisfies `P(y|x) = P(y|T(x))` on
   the training distribution, a model that computes `T` is already optimal; nothing in the loss rewards
   a relational circuit. The only available reward is the residual `1 − ceiling(T)`.
2. **Simplicity bias (the "only if").** SGD latches onto the simplest predictive feature and stays
   invariant to more complex ones, *even when the simple feature is less predictive* (Shah et al.,
   NeurIPS 2020); cross-entropy "encourages DNNs to stop learning once a simple predictor is found"
   (Geirhos et al., Nat. Mach. Intell. 2020). Which data-consistent solution is learned is set by
   inductive bias, not by the data alone.
3. **Our sweep (the sharpening).** The circuit appears when the pooled route is *imperfect*, not merely
   *weak*. A 96.5%-accurate shortcut does **not** displace an anchored counter when the task demands
   100%. Sufficiency is a knife edge — which is why the transition is sharp and looks like grokking.

**The constructive test.** A task is shortcut-proof iff there exist sequence pairs with identical
bounded-window statistics but different labels. That is literally a **cue conflict**. So the same
instrument that adjudicates a trained model's mechanism also certifies, in advance, whether the task
admits one. And the ceiling is estimable before training with a k-mer bag, a local naive Bayes, or a
receptive-field-limited CNN: if it meets what the objective demands, there is nothing to trace.

### Research questions

- **RQ0 (tracer validation — done, Gate 0).** On planted models with known edge sets, does the tracer
  recover the circuit (OV + QK recall) and separate two behaviourally-identical, mechanistically-opposite
  twins? *Answered: recall 1.0; conservation to 8e-22; MCC 0.945.*
- **RQ1 (the criterion).** Does "circuit forms ⟺ demand > pooled ceiling" hold when we move **both**
  sides — the ceiling (data structure) and the demand (objective) — at fixed architecture?
- **RQ2 (the hierarchy).** Ranked by pooled-statistic sufficiency rather than biological complexity,
  which genomic computations are shortcut-trivial and which are shortcut-proof, and do the measured
  ceilings match the predictions?
- **RQ3 (confirmation on shortcut-proof tasks).** On Tier 3–4 tasks where composition can be held fixed
  while the label moves (exon definition, NMD junction rule, CTCF orientation), does a relational circuit
  exist and route causal influence sign-correctly under a biology-supplied cue conflict?

### Hypotheses

- **H0 (tracer gate).** *Passed.* Twins separated; two documented blind spots (QK, feature-naming).
- **H1 (sufficiency law).** A circuit forms iff the objective's demanded accuracy exceeds the pooled
  ceiling. Prediction on the running 2×2: cells C and D (real CDS, ceiling 0.9509 — imperfect) build the
  circuit; if they do not, the criterion is wrong and we say so.
- **H2 (objective-as-demand, corrected).** Completeness rises with the objective **only through the
  demand side of the inequality**, at fixed data. Varying the objective alone (as D2 proposed) measures
  half the phenomenon and would credit the objective for an effect controlled by codon statistics.
- **H4 (attribution ranks; intervention decides).** *Established on a real learned circuit.* Every
  attribution method mis-ranks a naturally-occurring causally-inert marker head; only loss-based
  intervention decides.
- ~~H3 (JSAE composition-sparsity)~~ **demoted** (D16: Jacobian sparsity measures in-degree, so it
  separates composition from a bag but not from a lone shortcut). ~~H5 (developmental)~~ **cut** (D11).

### The deliverable: fifteen genomic computations ranked by shortcut availability

Tier is set by whether a bounded-window pooled statistic suffices, **not** by biological complexity.
✓ = ceiling measured by us; the rest are predictions with the biological reason stated, each cheap to
confirm.

- **Tier 0 — shortcut trivial; do not trace.** CpG island (GC%/CpG o/e); codon usage; isolated
  splice-site motif (a PWM scan *is* the intended solution); **reading frame in real CDS ✓ (ceiling
  0.9509 @ 240 nt)**. *This is where D2 placed its proof-of-method.*
- **Tier 1 — shortcut mostly available.** uORF/Kozak start selection ✓ (a learned detector reaches
  0.9215 frame accuracy from ±12 nt + ORF integrity); branch-point + polypyrimidine spacing; polyA
  signal + downstream GU element.
- **Tier 2 — relational but a strong pooled proxy competes; measure first.** G-quadruplex loop grammar
  (proxy: G-density — which is exactly D2's own null); TF-site spacing grammar; enhancer–promoter
  helical phasing. **D2's flagship G4 is Tier 2, not the flagship.**
- **Tier 3 — shortcut-resistant; two anchors + a spacing constraint.** Exon definition (inclusion
  tracks the *sum* of both splice-site strengths, r = 0.82, vs 0.48/0.66 singly); NMD 50–55 nt rule
  (label depends on stop-to-junction distance — a biology-supplied cue conflict); programmed ribosomal
  frameshifting.
- **Tier 4 — shortcut-proof; the label is carried by a relation.** CTCF convergent-orientation loops
  (~65% convergent vs ~1% divergent; re-inserting an oppositely-oriented site restores occupancy but
  not the loop — orientation carries loop information independent of binding); RNA secondary structure /
  base pairing (which base pairs with which — a relation between two positions, composition-invariant).

**A falsifiable prediction about a published model.** For a reverse-complement-equivariant architecture
(Caduceus), a motif and its RC are identified by construction, so orientation lies outside the
hypothesis class. **RC-equivariance, sold as a helpful inductive bias, is blindness to the Tier-4 CTCF
computation.** One afternoon to test.

### Novelty

1. **A predictive criterion for circuit existence** — sufficiency of a pooled statistic — measurable
   before training, replacing D2's "assume the circuit is there."
2. **A tracer with a measured error rate**, validated against hand-built twins, with the cue-conflict
   protocol that in-distribution attribution provably lacks (Geirhos et al. 2020).
3. **The shortcut-availability hierarchy** — the first ranking of genomic computations by whether they
   admit a mechanism at all, and the reframing of G4 (D2's flagship) as Tier 2.
4. **The objective question, correctly posed** — demand vs ceiling, moved on both axes, yielding μ\*,
   the smallest objective weight that installs a circuit, reported with the LM loss it costs.
5. **Confirmation on cue conflicts biology already ran at the bench** — CTCF orientation, NMD junction
   distance, exon spacing — rather than on synthetic toys.

### Models, data, methods

| Category | Item | Role |
|---|---|---|
| Model — planted | compositional + heuristic twins (hand-weighted) | Gate 0; measured tracer error rate |
| Model — ours | GFM-nano (6L × d256 × 8h, 4.80M, single-nt) | carries every rung; the λ and μ arms |
| Model — anchor | SpliceBERT (single-nt, 6L, pre-mRNA) | Tier-3 splice tasks in-distribution; contamination-firewalled |
| Model — published | Caduceus (RC-equivariant) | the RC-blindness prediction (Tier-4 CTCF) |
| Data | hg38 + GENCODE; synthetic uniform/sense/λ-biased CDS | ceilings measured; forcing tasks |
| Method — dictionaries | one `Dictionary` ABC → SAE / Transcoder / CLT | transcoder first; crosscoders **cut** (Minder et al. 2025) |
| Method — causal | pooled-ceiling estimator → cue-conflict → edge/node patch | the criterion, then the trace |

### Staged plan — a protocol, not a ladder

1. **Measure the ceiling before training.** For each candidate task, estimate the best bounded-window
   pooled accuracy (k-mer bag / local naive Bayes / RF-limited CNN). Hours, not weeks. If it meets the
   objective's demand, stop — no circuit, no paper.
2. **Rank, and pick from Tier 3–4.** Not Tier 0, where D2 began.
3. **Trace only with a cue-conflict protocol.** Construct matched-composition, conflicting-relation
   pairs; biology often supplies them (CTCF orientation, NMD junction distance, exon spacing).
4. **The objective question.** Move both sides of the inequality. Ceiling: the λ sweep (done, seven
   arms). Demand: the μ mixture `loss = (1−μ)·[predict nucleotide] + μ·[predict amino acid]`, both terms
   **entropy-normalised** (`ln 21` vs `ln 4`). Deliverable: **μ\***, reported with its LM-loss price,
   over ≥3 seeds as an interval, arms compared at **matched LM loss** (grokking makes the stop a choice).
5. **Transplantation.** Hand a real-DNA model the circuit for free and fine-tune: does it survive or
   erode? Transplant is direct — forcing-task and real-DNA arms share one backbone. *Registered
   prediction:* the synthetic detector learned "first ATG = start," but 35.1% of real anchored windows
   carry an upstream ATG, so it should misfire on exactly those.

**Running now — the necessity 2×2** (μ = 1 throughout, frozen in `PREREGISTRATION.md`): [A] uniform
(circuit@3250), [B] uniform + decoys + Kozak matched to d = 0.9215, [C] real CDS filtered, [D] real CDS
natural. Verdict frozen and causal: far-field cue-conflict anchor share > 0.50 **and** start-codon
ablation ΔaaAcc > 0.10 beyond a decoy-Met control — never the training accuracy, which saturates and has
fooled us once.

### Risks and where we expect to be wrong

1. **The ceiling estimator is a lower bound on the shortcut.** A 256-nt transformer may beat our
   codon-level naive Bayes, moving tasks *down* the hierarchy. Re-measure every §-4 ceiling with an
   RF-limited CNN. This cuts against us — it makes circuits rarer and the finding stronger.
2. **Nothing transfers from synthetic λ arms to real DNA without re-measurement** (D30). Naive-Bayes
   independence closes λ = 1's ceiling to 1.0000 while real CDS is 0.9509, because real codons correlate.
3. **The law is a conjecture on n = 8 arms, one task, one architecture.** The 2×2 tests it on real DNA;
   it must also be tested on a Tier-3 task (where the prediction is a *circuit*) before being stated as law.
4. **LayerNorm is unmeasured.** Every tracer error-rate number came from LN-free planted models; GFM-nano
   has LayerNorm. Plant a circuit in an LN-equipped tiny transformer and re-run Tests A and C — a day, and
   it closes the last Gate-0 hole. **This gates the trace and transplant claims.**
5. **Pretrained-anchor contamination.** No held-out chromosome exists for SpliceBERT; mechanistic claims
   fine, generalisation claims impossible. Enforced in `config.assert_holdout_valid`.
6. **A partial graft is meaningless without representational alignment**, and the adapter's capacity then
   confounds any positive transplant result.

### The claim

> **A model builds a relational circuit exactly when no statistic pooled over its context window is
> sufficient for the target.** Real genomic self-supervision sits on the wrong side of that line for
> reading frame — the frame is nearly a sufficient statistic of local codon composition, worth 0.4% of
> the loss, and the model correctly declines to build machinery for it. The line is measurable before
> training, it ranks genomic computations from shortcut-trivial to shortcut-proof, and crossing it is a
> design choice available to anyone training a genomic foundation model.

**Reading frame was the wrong rung. The right ones are exon definition, the NMD junction rule, and CTCF
orientation** — where composition can be held fixed while the label moves, and biology has already run
the cue-conflict experiment for us.

---

*Full evidence and reference list: `REC.md` and `docs/DECISIONS.md` (D1–D30). This proposal cites the
same external literature — Geirhos et al. 2020; Shah et al. 2020; Nullsettes (2506.10271); Mechanistic
Invariance Test (2604.06549); DART-Eval; exon-definition PMC3203598; CTCF PII S1097276515007625; CREME
PMC10349992.*