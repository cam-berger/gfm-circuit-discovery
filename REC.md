# Recommendation: a critique of Proposal D2, and the program that replaces it

**Cam Berger — 2026-07-09** · v2 (v1 archived at `docs/archive/REC_v1_superseded.md`; §0 says what changed)
Repo: `~/Research/projects/gfm-circuit-discovery` · 102 tests passing
Evidence: `docs/DECISIONS.md` (D1–D30) · `docs/EXPERIMENTS.md` (plain-language) · `docs/PREREGISTRATION.md`

---

## Summary

Proposal D2 asked: *does a training objective install a circuit?* It presumed circuits are there to be
found, and that the job was to build a tracer good enough to see them.

We built the tracer, validated it against circuits we constructed by hand, pointed it at real DNA
models — and found no circuit. Then we found the reason, and the reason is not about objectives:

> **A model builds a relational circuit if and only if no statistic pooled over its context window is a
> *sufficient statistic* for the target. Not "if the shortcut is weak" — if the shortcut is *imperfect*.**

Reading frame in real coding sequence fails that test. We measured its anchor-free ceiling directly: a
codon-level pooled estimator recovers the frame at **0.9509** with no start codon at all. Under a
language-modelling objective the residual is worth **0.4% of the loss**, and no circuit appears. Under an
objective that demands exactness, the same architecture builds an ATG-anchored modular counter — abruptly,
after 2,750 steps of nothing.

Same weights, same data, same tracer: **the mechanism is selected by the gap between what the objective
demands and what a pooled statistic can deliver.**

This reframes the question from *"which objectives install circuits"* to **"which computations over DNA are
shortcut-proof"** — and that question has a measurable answer, per task, *before you train anything*. §4
ranks fifteen genomic computations against it.

---

## 0. What changed from v1, and what I got wrong

v1 (this morning) claimed the finding was *"the circuit is worth 0.4% of the loss, so the model declines to
build it."* That is true of the μ = 0 objective, and it is the shallow version. Three things happened since,
two of them my errors:

1. **Scoring a sweep on accuracy hid the result.** Seven λ arms all reached `aa_acc = 1.0000`. Only the
   causal readouts revealed that four built an anchored counter and three built an anchor-free estimator
   (D29). Reported on accuracy, I would have been wrong about four of seven arms. *These are the planted
   twins from Rung 0 — identical behaviour, opposite mechanism — arising spontaneously.*
2. **I explained that split with a story, and the control falsified it.** I claimed stop-codon depletion
   caused the switch. A registered control — uniform over the 61 *sense* codons, zero codon bias, zero
   in-frame stops — came out **anchored** (D30). Stop depletion is not the cause.
3. **The real criterion is sufficiency**, and it is measurable (§2). It also retro-explains two further
   claims I raised and retracted within an hour (D28): that genomic landmarks are unidentifiable from
   sequence (a *learned* detector reaches 0.9215 frame accuracy; my Kozak heuristic reached 0.707), and
   that stop-codon absence alone fixes the frame (0.8242, not 1.0).

Recorded because each was plausible enough to have shipped.

---

## 1. What Proposal D2 assumed, and why each assumption fails

### 1.1 It assumed the circuits are there

The proposal's framing — Anthropic "explicitly named Evo 2 features *proto-circuits*"; the gap is at the
"circuit level" — presumes existence and treats tracing as the open problem. Our result says **existence is
the open problem, and the modal answer is *no circuit*.** D2 lists "no circuit found" as **Risk #4**, with no
theory predicting when to expect it. It is not a risk. It is the default.

The literature already said so, and D2 cites it as *behavioural* work rather than as evidence about
mechanism. Across 12 state-of-the-art gLMs, accuracy collapses as sequences become evolutionarily
implausible — the models pattern-match a distributional prior rather than compute
([Nullsettes, arXiv 2506.10271](https://arxiv.org/abs/2506.10271)). Five architectures (HyenaDNA, Evo2-1B,
NT-500M, GROVER, Caduceus) solve a positional-regulation task **entirely via AT-content correlation**
(r = 0.78–0.96; compositional effects exceed positional by ~46×), while a ~100-parameter position-aware PWM
solves it exactly — and Evo2-1B and Caduceus score an UP-element *higher at the biologically wrong position*
([Mechanistic Invariance Test, arXiv 2604.06549](https://arxiv.org/pdf/2604.06549)). DNA language models do
not consistently beat simple *ab initio* baselines
([DART-Eval, NeurIPS 2024](https://proceedings.neurips.cc//paper_files/paper/2024/hash/71998bfc3217ffe1cca1ee084dfadadd-Abstract-Datasets_and_Benchmarks_Track.html);
[Nat. Commun. 2025](https://www.nature.com/articles/s41467-025-65823-8), where a task-specific CNN beats
every foundation model most of the time).

D2 reads these as *"benchmarks establish that GLMs fail positional/regulatory logic… they treat the model as
a black box."* The correct reading: they are converging evidence for a mechanistic claim. **These models are
pooling, and they are pooling because pooling suffices.**

### 1.2 Its ladder is ordered by the wrong axis

The ladder — reading frame → splice sites → G-quadruplex — is ordered by *biological relational complexity*.
That is not the axis that governs whether a circuit exists. The governing axis is **shortcut availability**,
and on it D2's ladder is close to inverted at rung 1.

Reading frame, chosen as the "proof-of-method… where the correct answer is known in advance," is precisely
the rung where a pooled statistic nearly suffices (ceiling **0.9509**). It is the *worst* proof-of-method
target on real DNA. We spent a day discovering that empirically.

### 1.3 H2 mis-locates the cause

*"Circuit completeness scales with training objective"* treats the objective as the cause. It is one of two
factors; the other is the data's shortcut structure:

```
circuit forms   ⟺   accuracy demanded by the objective   >   ceiling of the best pooled statistic
```

Our λ sweep is that equation run seven times. At λ ≥ 1 the pooled ceiling reaches 1.0000 and the circuit
vanishes **even though the objective never changes** (μ = 1 throughout). Vary the objective at fixed data and
you see half the story. **H2 varies only that half, and would have reported "objective installs circuit" for
a phenomenon controlled by codon statistics.**

### 1.4 It has no out-of-distribution test, and therefore cannot answer its own question

The deepest flaw. Shortcut and intended decision rules are **provably indistinguishable on i.i.d. data**;
separating them requires interventions that break the spurious correlation
([Geirhos et al., *Nature Machine Intelligence* 2020](https://www.nature.com/articles/s42256-020-00257-z)).
Every method in D2's table — attribution graphs, transcoders, edge patching, JSAE, crosscoders — operates on
in-distribution inputs.

We hit this directly. Our own start-codon **ablation could not detect the circuit**: destroying the anchor
leaves the pooled route intact, and the model simply falls back on it. Only a **cue conflict** — insert one
nucleotide after the ATG so anchor and local statistics disagree by exactly one codon — separates them
(D24). *That objection was Cam's; my "VERDICT: bag of motifs" was unearned when I printed it.*

D2's Novelty #1 is a tracer. **A tracer without a cue-conflict protocol cannot answer D2's own question.**

### 1.5 Four technical defects, found by ground truth

Each surfaced against hand-built circuits, before any real model was touched. Each would have produced
entirely plausible figures.

| # | defect | evidence |
|---|---|---|
| D12 | Frozen-attention attribution is **structurally blind to QK edges** | scored the marker-finding edge at `0.000`; ablating it cost 0.617 accuracy |
| D20 | Naive pattern-gradient attribution **vanishes on a confident head** | `max abs(dL/dscore) = 3.7e-15` — the gradient calls the edge irrelevant *because* the head is certain |
| D17 | A transcoder reconstructs a circuit perfectly and **recovers none of its concepts** | EV 1.0000; the decoder *bias* absorbed one concept; phase from features 0.662 vs 1.000 from neurons |
| D16 | JSAE Jacobian sparsity measures **in-degree, not composition** | separates composition from a bag (+0.0637) but not from a lone shortcut (+0.0063) |

Three belong to one family: **a saturated measurement reports nothing exactly where the effect is largest.**
Saturated softmax (D20); element-wise variance on a sparse target (D18); accuracy pinned at 1.0000 (D27). It
is what interpretability looks like without ground truth.

---

## 2. The criterion, and why it is the right one

> **A mechanism is unavoidable exactly when no statistic pooled over the model's context window is a
> sufficient statistic for the target.**

Three legs: two from the literature, one from our data.

**Sufficiency.** If a bounded-window statistic `T(x)` satisfies `P(y | x) = P(y | T(x))` on the training
distribution, a model computing `T` is *already optimal*. Nothing in the loss rewards the relational circuit.
The only reward available is the residual `1 − ceiling(T)`.

**Simplicity bias supplies the "only if."** SGD-trained networks latch onto the simplest predictive feature
and remain invariant to more complex ones — *even when the simple feature is less predictive*
([Shah et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/6cfe0e6127fa25df2a0ef2ae1067d915-Paper.pdf)).
Cross-entropy "encourages DNNs to stop learning once a simple predictor is found"
([Geirhos et al. 2020](https://www.nature.com/articles/s42256-020-00257-z)). **Which of many data-consistent
solutions is learned is set by inductive bias, not by the data alone.**

**Our sweep supplies the "if," and sharpens it.** The circuit appears when the pooled route is *imperfect*,
not merely *weak*:

```
data          I(nt;phase)   anchor-free ceiling            mechanism the
                            ctx=60  ctx=120  ctx=240       model built
uniform-64      0.00003     0.000    0.000    0.000        ANCHORED COUNTER
sense-61        0.00162     0.505    0.814    0.965        ANCHORED COUNTER
λ = 0.50        0.00393     0.410    0.508    0.533        ANCHORED COUNTER
λ = 0.75        0.00854     0.575    0.815    0.935        ANCHORED COUNTER
λ = 1.00        0.01471     0.821    0.976    1.0000       anchor-free
λ = 2.00        0.05778     0.995    1.000    1.0000       anchor-free
```

A 96.5%-accurate shortcut does **not** displace a counter when the task demands 100%. Sufficiency is a knife
edge — which is why the transition is sharp, and why it looks like grokking.

**A constructive test falls out.** A task is shortcut-proof iff there exist sequence pairs with identical
bounded-window statistics but different labels. **That is literally a cue conflict.** So the same instrument
that adjudicates a trained model's mechanism also certifies, in advance, whether the task admits one.

**And the ceiling is measurable before training.** Estimate it with a k-mer bag, a local naive Bayes, or a
receptive-field-limited CNN. If it meets what the objective demands, do not build a tracer. There is nothing
to trace.

---

## 3. What we established

**A tracer with a measured error rate.** Validated against two hand-weighted transformers with identical
behaviour (|Δacc| = 0.0034) and opposite mechanism — a positive control alone cannot falsify; twins can. OV +
QK edge recall against the planted circuit = **1.0**; attribution conservation exact to `8e-22`; dictionary
recovery MCC 0.945 where features are identifiable. Two documented blind spots.

**Real-DNA GFM-nano has no anchored frame circuit**, on either data regime, established by **cue conflict**
(D24), not ablation — which cannot see it. The model sits at the anchor-free ceiling: 0.7936 achieved against
a pooled ceiling of 0.82–0.95, stopping exactly where an anchor would be required.

**Make the frame necessary and the circuit appears**, abruptly, verifying on every causal assay: phase probe
1.0000 at L2 (untrained control 0.347, flat); start-codon ablation 1.0000 → 0.3552 (*below* the 0.3768
guessing ceiling); cue conflict 1.0000 anchor / 0.0000 content at 250 nt.

**We traced it.** L0 heads assemble the local trigram — ablating `L0H0` halves `L1H3`'s attention to the
marker while every other L0 head leaves it at 83–111% (K-composition). `L1H3` puts 0.80 of its attention on
the **T** of the start codon, and attends there **230× more** than to a decoy ATG. The MLPs count.

**And the trace produced the project's most useful methodological result.** The model grew a *second* marker
head, `L1H2`, attending the true start with **higher** specificity (406× vs 230×) and **causally inert** —
ablating it moves the loss by `2e-6` against `L1H3`'s `8.36`, a four-million-fold difference.

```
method                      verdict on L1H2 (inert) vs L1H3 (essential)
attention pattern           L1H2 looks BETTER
naive score gradient        ranks L1H2 ABOVE L1H3 — backwards
integrated gradient (D20)   ranks correctly, still credits the inert head 16.5%
ablation on accuracy        right by luck; the metric saturates at 1.0000
ablation on loss            8.36 vs 0.000002 — unambiguous
```

**Attribution ranks; only intervention decides.** And `L1H2` is exactly
`planted.build_bag_with_unread_marker_head` — a synthetic control we invented and worried was contrived. **It
occurs naturally.**

**Precedent for our method.** Deliberately blocking a shortcut in the training data to force the intended
mechanism is established — Stylized-ImageNet forcing shape over texture
([Geirhos et al. 2020](https://www.nature.com/articles/s42256-020-00257-z)). Our uniform-codon ablation is
the genomics analogue, and as far as we can find, the first in this domain.

---

## 4. The hierarchy: fifteen genomic computations, ranked by shortcut availability

This is the deliverable D2 should have had. **Tier is set by whether a bounded-window pooled statistic is
sufficient**, not by biological complexity. Ceilings marked ✓ are measured by us; the rest are predictions
with the biological reason stated, each cheap to measure.

### Tier 0 — shortcut trivially available. Do not trace these.
*The label is a function of local composition.*

| task | why the shortcut exists | pooled ceiling |
|---|---|---|
| **CpG island** | definitionally a composition statistic (GC%, CpG o/e) | 1.0 by construction |
| **codon usage / tRNA adaptation** | a composition statistic | 1.0 by construction |
| **splice-site motif detection (isolated)** | a PWM over ~9–23 nt; a local scan *is* the intended solution | ≈ PWM ceiling |
| **reading frame in real CDS** ✓ | codon usage + stop depletion make frame ≈ a sufficient statistic of local composition | **0.9509 @ 240 nt** (ours) |

**Reading frame sits here, and it is where D2 put its proof-of-method.**

### Tier 1 — shortcut mostly available; the relational residual is small.

| task | relational content | why a shortcut nearly suffices |
|---|---|---|
| **uORF / Kozak-dependent start selection** ✓ | pick the true ATG among decoys | a learned detector reaches **0.9215 frame accuracy** (exact 0.8535) from ±12 nt context + ORF integrity (ours) |
| **branch point + polypyrimidine tract** | spacing to the 3′ss | the spacing distribution is broad; a pyrimidine-content scan captures much of it |
| **polyadenylation signal + downstream GU-rich element** | fixed spacing between two elements | both elements are locally detectable; spacing adds modest information |

### Tier 2 — relational, but a strong pooled proxy competes. Measure before tracing.

| task | the relation | the proxy that competes |
|---|---|---|
| **G-quadruplex loop-length grammar** | G-runs × loop windows × strand | **G-density** — the exact null D2 names, and it *is* a composition statistic |
| **TF binding-site cooperativity / spacing grammar** | motif pairs at constrained spacing | motif presence alone explains much of the variance |
| **enhancer–promoter helical phasing (~10.5 bp)** | periodic distance between two elements | composition of both elements; phase is a residual |

**D2's flagship (G4) is Tier 2, not Tier 4.** Its stated null — a G-density counter — *is* the pooled
statistic, and D2 never proposes to measure that null's ceiling before committing to trace against it.

### Tier 3 — shortcut-resistant. Two anchors plus a spacing constraint.
*Composition can be held fixed while the label changes.*

| task | why pooling fails | quantitative evidence |
|---|---|---|
| **Exon definition** (5′ss + 3′ss paired *across the exon*) | recognition needs **both** sites, balanced (3′ ~51%, 5′ ~49%), and is favoured only at short inter-site spacing | inclusion correlates with the **sum** of splice-site strengths (r = 0.82) far better than either alone (3′ r = 0.48, 5′ r = 0.66) — [PMC3203598](https://pmc.ncbi.nlm.nih.gov/articles/PMC3203598/) |
| **NMD 50–55 nt rule** | the label depends on the **distance** from a stop codon to the last exon–exon junction | identical composition, stop at 40 nt vs 60 nt → opposite label. A cue conflict supplied by biology |
| **Programmed ribosomal frameshifting** | slippery heptamer **plus** a downstream structure at fixed spacing | neither element alone predicts; the relation does |

### Tier 4 — shortcut-proof. The label is carried by a *relation*; composition is (near-)uninformative.

| task | why no pooled statistic can work | quantitative evidence |
|---|---|---|
| **CTCF convergent-orientation loops** | loops form between **convergently** oriented sites: ~65% convergent vs ~1% divergent. Re-inserting an oppositely oriented motif **restores CTCF/cohesin occupancy but fails to re-establish the loop** — orientation carries loop information *independent of binding* | [Cell Reports / S1097276515007625](https://www.sciencedirect.com/science/article/pii/S1097276515007625) |
| **RNA secondary structure / base pairing** | the label is *which base pairs with which* — a relation between two positions, not a property of either | pairing partners are arbitrarily distant; composition is invariant to who pairs with whom |

**CTCF is the sharpest case in genomics.** Occupancy identical, orientation flipped, label changes — a cue
conflict *built into the biology*, already run at the bench.

And note the twist. For a **reverse-complement-equivariant** architecture (Caduceus), a motif and its reverse
complement are *identified by construction*, so orientation lies outside the hypothesis class.
**RC-equivariance, sold as a helpful inductive bias, is a blindness to a Tier-4 computation.** That is a
testable, falsifiable prediction about a published model, and it costs one afternoon.

### The one positive result in the literature

Enformer shows a genuine TSS-anchored, distance-dependent relation: an enhancer tile's predicted effect
decays with distance from the TSS, and CREs interact non-additively (super- and sub-additively)
([CREME, PMC10349992](https://pmc.ncbi.nlm.nih.gov/articles/PMC10349992/)). Yet sequence models capture
promoter determinants while largely **failing distal enhancers**
([arXiv 2602.01230](https://arxiv.org/pdf/2602.01230)). The one architecture trained on a supervised,
demand-heavy objective is the one with a relational circuit. **That is the criterion, in the wild.**

---

## 5. The reframed program

Replace D2's ladder with a protocol.

**Step 1 — Measure the ceiling before training anything.** For each candidate task, estimate the best accuracy
of a bounded-window pooled statistic (k-mer bag, local naive Bayes, receptive-field-limited CNN). Hours, not
weeks. *If the ceiling meets what the objective demands, there is no circuit and no paper.*

**Step 2 — Rank, and pick from Tier 3–4.** Not from Tier 0, which is where D2 began.

**Step 3 — Trace only with a cue-conflict protocol.** In-distribution attribution cannot distinguish a pooled
statistic from a relational circuit ([Geirhos et al. 2020](https://www.nature.com/articles/s42256-020-00257-z)).
Construct sequence pairs with matched composition and conflicting relations. Biology often supplies them:
CTCF orientation, NMD junction distance, exon spacing.

**Step 4 — The objective question, correctly posed.** Not *"does objective X install circuit C"* but:

```
circuit forms   ⟺   accuracy demanded   >   ceiling(pooled statistic)
```

Test it by moving **both** sides. We have moved the ceiling (λ sweep, seven arms). Moving the demand is the μ
mixture: `loss = (1−μ)·[predict nucleotide] + μ·[predict amino acid]`, both terms entropy-normalised.
**μ\*** — the smallest weight that installs the circuit — is the deliverable, reported with the LM loss it
costs, over ≥3 seeds as an interval.

**Step 5 — Transplantation.** Hand a real-DNA model the circuit for free and fine-tune: does it survive or
erode? Today we can say *the objective does not build the circuit*. If it erodes, we can say **it will not
even keep it** — a claim about the loss landscape rather than about optimisation difficulty. Transplant is
direct by construction: the forcing-task model and the real-DNA arms share one backbone.

*Registered prediction:* our synthetic task guarantees the start codon is the **first** ATG, but **35.1%** of
real anchored windows carry an upstream ATG. The transplanted detector should misfire on exactly those.

### Running now

The **necessity 2×2**, μ = 1 throughout, frozen in `PREREGISTRATION.md` before any cell ran:

```
                    detector EASY                    detector HARD
shortcut ABSENT     [A] uniform      circuit@3250    [B] uniform + decoys + Kozak (matched to d = 0.9215)
shortcut PRESENT    [C] real CDS, filtered           [D] real CDS, natural
```

Verdict (frozen): far-field cue-conflict anchor share > 0.50 **and** start-codon ablation ΔaaAcc > 0.10 beyond
a decoy-Met control. Both causal, both validated on the planted twins, **neither the training accuracy** —
which saturates, and has fooled us once.

**Registered prediction (D30):** real CDS has a pooled ceiling of **0.9509** — imperfect, adjacent to the
sense-codon control's 0.965, which produced a counter. **Cells C and D should build the circuit.** If they do
not, the sufficiency criterion is wrong and I will say so.

---

## 6. Where I expect to be wrong

**The ceiling estimator is a lower bound on the shortcut.** Ours is a codon-level naive Bayes. A transformer
with 256 nt of context and no independence assumption may do better, moving tasks *down* the hierarchy. Every
ceiling in §4 should be re-measured with a receptive-field-limited CNN before it is trusted. This cuts against
us: it makes circuits *rarer*, and the finding stronger.

**λ = 1 synthetic is a bad model of real CDS** (D30). Identical codon usage; naive-Bayes independence closes
its ceiling to 1.0000 while real DNA's is 0.9509, because real codons are correlated. **Nothing may be
transferred from the λ arms to real DNA without re-measurement.** I did exactly that, for an hour.

**"Circuit forms ⟺ demand > ceiling" is a conjecture on n = 8 arms, one task, one architecture.** The 2×2
tests it on real DNA. It should also be tested on a Tier-3 task, where the prediction is a circuit, before it
is stated as a law.

**μ is not a clean fraction of demand** until both loss terms are entropy-normalised (`ln 21 = 3.04` vs
`ln 4 = 1.39`). A negative at small μ is undertraining until amino-acid accuracy is checked against the
0.3768 guessing ceiling. Grokking makes *when you stop* a scientific choice: compare arms at **matched LM
loss**, never at matched steps.

**The forcing task is synthetic and a reviewer will say so.** Its job is not realism. Its job is to prove the
tracer finds a real learned circuit when one exists, so that the hg38 negative is a fact about DNA and not
about our instruments. **The hg38 result is the finding; the synthetic model is the control that licenses it;
the planted models are what prove the instruments work at all.**

**LayerNorm remains unmeasured.** Every number the tracer knows about its own error rate came from LN-free
planted models. GFM-nano has LayerNorm; the replacement model must freeze it. Plant a circuit in an
LN-equipped tiny transformer and re-run Tests A and C. A day's work, and it closes the last hole in Gate 0.

**A partial graft is meaningless without alignment.** Inserting `L1H3` alone into a separately-trained model
fails: the two residual streams have unrelated bases. It needs a learned alignment, and the adapter's capacity
then confounds the result.

---

## 7. The claim

> **A model builds a relational circuit exactly when no statistic pooled over its context window is
> sufficient for the target.** Genomic self-supervision on real DNA sits on the wrong side of that line for
> reading frame: the frame is nearly a sufficient statistic of local codon composition, worth 0.4% of the
> loss, and the model correctly declines to build machinery for it. The line is measurable *before training*,
> it ranks genomic computations from shortcut-trivial to shortcut-proof, and crossing it is a design choice
> available to anyone training a genomic foundation model.

The instruments that establish this were validated against circuits we built by hand; they have a measured
error rate; and every one of them was corrected by a defect that ground truth exposed and that
plausible-looking results would have hidden.

**Reading frame was the wrong rung.** The right ones are **exon definition**, the **NMD junction rule**, and
**CTCF orientation** — where composition can be held fixed while the label moves, and biology has already run
the cue-conflict experiment for us.

---

### References

- Geirhos et al., *Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence 2020 — https://www.nature.com/articles/s42256-020-00257-z
- Shah et al., *The Pitfalls of Simplicity Bias in Neural Networks*, NeurIPS 2020 — https://proceedings.neurips.cc/paper/2020/file/6cfe0e6127fa25df2a0ef2ae1067d915-Paper.pdf
- *Nullsettes: evolutionarily implausible sequences*, arXiv 2506.10271 — https://arxiv.org/abs/2506.10271
- *The Mechanistic Invariance Test*, arXiv 2604.06549 — https://arxiv.org/pdf/2604.06549
- *DART-Eval*, NeurIPS 2024 D&B — https://proceedings.neurips.cc//paper_files/paper/2024/hash/71998bfc3217ffe1cca1ee084dfadadd-Abstract-Datasets_and_Benchmarks_Track.html
- Benchmark of DNA foundation models vs task-specific CNN, Nat. Commun. 2025 — https://www.nature.com/articles/s41467-025-65823-8
- Exon definition and splice-site pairing — https://pmc.ncbi.nlm.nih.gov/articles/PMC3203598/
- CTCF convergent-orientation looping — https://www.sciencedirect.com/science/article/pii/S1097276515007625
- CREME in-silico perturbation of Enformer — https://pmc.ncbi.nlm.nih.gov/articles/PMC10349992/
- Sequence models capture promoters, fail distal enhancers — https://arxiv.org/pdf/2602.01230
