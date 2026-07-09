# Recommendation: what this project is now, and what to do next

**Cam Berger — 2026-07-09**
Repo: `~/Research/projects/gfm-circuit-discovery` · 16 commits · 92 tests passing
Full evidence: `docs/DECISIONS.md` (D1–D27), `docs/EXPERIMENTS.md` (plain-language), `docs/PLAN.md`

---

## Summary

The proposal set out to trace circuits in genomic foundation models and to test whether a training
objective installs one. We built the tracer, validated it against ground truth we constructed
ourselves, pointed it at real DNA models — and found **no circuit**. Then we determined *why*, and
the answer is not a failure of our instruments or of optimisation:

> **A reading-frame circuit is worth 0.4% of the loss to a genomic language model. It never builds
> one. When we make the circuit necessary rather than merely useful, it builds one abruptly and
> perfectly.**

That result is stronger than the one the proposal was designed to produce, and it reframes the
central question from *"does an objective install a circuit"* to **"how much must an objective demand
a latent variable before the model builds machinery for it"** — a question with a number for an
answer.

I recommend we **restructure the paper around that number**, and I lay out below why the evidence
compels it, what remains to be done, and where I think we are most likely to be wrong.

---

## 1. What we built, and why you should believe it

The proposal's Novelty #1 was "first circuit-tracing methodology for DNA-sequence GFMs." A tracer
that has never been checked against a known answer is a picture-generating machine, not an
instrument. So before touching real DNA we built two transformers **by hand**, setting every weight
ourselves:

| | mechanism | behaviour | start-marker ablation |
|---|---|---|---|
| compositional twin | attends to the marker, computes `(i − start − 1) mod 3` | 0.586 | 1.000 → 0.383 |
| bag-of-motifs twin | five local motif detectors, independent votes, no frame variable | 0.583 | 0.583 → 0.583 |

Behaviourally indistinguishable (|Δ| = 0.0034), mechanistically opposite. A positive control alone
cannot falsify; twins can.

**This was worth every hour it cost.** Four defects surfaced *before* any real model was touched,
each of which would have produced entirely plausible results:

1. **Frozen-attention attribution is structurally blind to QK edges.** It scored the marker-finding
   edge at exactly `0.000` while ablating that token cost 0.617 accuracy. Every rung of the ladder —
   find the ATG, find the donor GT, find the G-run — has a "decide where to look" step. (D12)
2. **Naive pattern-gradient attribution vanishes on a confident head.** `max |dL/dscore| = 3.7e-15`.
   The gradient reports the edge as irrelevant *because* it is so relevant the head is certain. On
   SpliceBERT this would have yielded an empty graph and looked like an honest negative. Fixed by
   integrating the gradient from a content-free baseline — the attention-pattern analogue of EAP-IG,
   which the proposal listed without saying why it was needed. (D20)
3. **A transcoder reconstructs a circuit perfectly (EV = 1.0000) and recovers none of its concepts.**
   The decoder *bias* absorbed one concept outright; another "feature" was a signed difference vector.
   Phase recoverable from features: 0.662, versus 1.000 from the raw neurons. Trust the **shape** of a
   circuit diagram; do not trust the **names** of its nodes. (D17)
4. **JSAE Jacobian sparsity measures in-degree, not composition.** It separates composition from a bag
   of weak features (separation +0.0637) but not from a lone strong shortcut (+0.0063) — a model that
   cheats with one trick is *maximally* sparse. H3 survives only in restricted form. (D16)

Three of these belong to one failure class that recurred all project long: **a saturated measurement
reports nothing exactly where the effect is largest.** Saturated softmax (D20), element-wise variance
that flatters any fit on a sparse target (D18), accuracy pinned at 1.0000 (D27). I do not think this
is coincidence. It is what interpretability looks like without ground truth.

**Gate 0 outcome:** OV + QK edge recall against the planted circuit = **1.0**. Attribution
conservation exact to `8e-22`. Dictionary recovery MCC 0.945 where features are identifiable. Two
documented blind spots. The tracer has a measured error rate. Nothing downstream would have been
reported had it failed.

---

## 2. The negative result, and why it is the finding

We trained GFM-nano (6L × d=256 × 8 heads, 4.80M params, single-nucleotide, LayerNorm) on hg38 +
GENCODE v50. Frame labels never entered any training objective. Windows were cut at **randomised
offsets** — mutation-tested, because CDS-anchored windows let `index % 3` predict phase at 1.000 and
the model would have read the answer off its position embedding while the loss curve looked perfect
(D21). Labels validated against biology, not against our own code: **zero in-frame stop codons in
117,481**, against 4.19% and 5.54% in the two shifted frames.

Phase is strongly linearly decodable — 0.7936, against a nucleotide baseline of 0.4105 and an
**untrained** model at 0.4094. For a while that looked like the proof-of-method.

It was not. Two facts killed it.

**The signal appears entirely at layer 0 and is flat thereafter** (L0 0.7936 → L5 0.7914). Nothing is
built. That is the signature of a single pooling operation, not a computation.

**And ablation cannot detect a redundant circuit.** Destroying the ATG removes one source of phase
information and leaves the other — downstream codon statistics, worth 0.79 alone — untouched. A model
that genuinely counts can lose nothing, because it falls back on local evidence. *(This objection was
Cam's, and it was correct. My "VERDICT: bag of motifs" was not earned at the time I printed it.)*

The fix is not to remove information but to make the cues **contradict**. Insert one nucleotide after
the ATG: counting from the anchor now says one phase, the local statistics say a phase exactly one
off. The model must pick a side.

```
                       trained WITH anchors     trained WITHOUT anchors (control)
distance from ATG      anchor    content        anchor    content
4–15 nt                0.5944    0.1524         0.5999    0.1553
32–63 nt               0.2423    0.5424         0.2490    0.5313
128–255 nt             0.0626    0.8354         0.0569    0.8406
```

Both follow **local content** at range. The control — which never saw a start codon and *cannot* be
anchored — produces an identical curve, which proves the near-field "anchor following" is pooling
across the insertion boundary, not counting. The ~32 nt crossover measures the pooling width, matching
an independent ±15 nt naive-Bayes estimate (0.6918).

Then the mechanism, in one calculation:

```
H(nucleotide)             = 1.3714 nats
H(nucleotide | phase)     = 1.3473 nats
gain from PERFECT phase   = 0.0241 nats
```

The model already infers phase at 0.79 from local statistics. **An anchored counter buys at most
0.005 nats — 0.40% of a 1.27-nat loss**, and that is an upper bound.

**The model is not failing. It is correctly declining to build machinery the objective does not pay
for.** Longer training cannot help; there is almost nothing to grok toward. This is the genomic
instance of Nikankin's arithmetic-as-heuristics result, and it supplies a mechanistic explanation for
why the behavioural benchmarks the proposal cites (Mechanistic Invariance Test, DART-Eval, Genomic
Touchstone) keep finding that genomic LMs fail positional and regulatory logic. They are not failing
to learn. They are declining to.

---

## 3. The positive result: make it necessary, and it appears

If the negative is real, the same architecture with the same tracer should produce a circuit the
moment the objective demands one. That is the control the negative result requires, and it is what
makes the negative a fact about DNA rather than about our instruments.

**A structural constraint we could not evade (D25):** *no self-supervised objective can force an
anchor.* Any phase-dependent structure in the sequence repeats every three bases and is therefore
locally detectable; remove the local cue and knowing the phase stops helping next-nucleotide
prediction, so the incentive vanishes with the cue. Usefulness and guessability come from the same
statistical structure.

So the forcing task must be **supervised on a readout that requires the frame** — D7's logic:

```
sequence   5' region with NO ATG | ATG | codons drawn uniformly from all 64
           I(nucleotide ; phase) = 0.000001 nats   (real hg38 CDS: 0.0241)
label      the amino acid of the codon containing each position — never the frame
           guessing the alignment caps accuracy at 0.3768; knowing the frame gives 1.0
```

**It sat at the guessing ceiling for 2,750 steps and transitioned in ~500.** *(Cam predicted a
grokking-style delay; on the task where the circuit is worth building, that is exactly what happened.)*

```
step 2750   aa_acc 0.4006   frameless
step 3000   aa_acc 0.4426
step 3250   aa_acc 0.9790
step 3750   aa_acc 0.9987
```

Every assay that found nothing in hg38 flips, at the final checkpoint:

```
phase probe      L0 0.4116 → L1 0.9138 → L2 1.0000     (untrained control 0.347 at every layer)
amino-acid accuracy                        1.0000
  START CODON ablated                      0.3552      below the 0.3768 guessing ceiling
  downstream Met ablated (control)         0.9871      identical perturbation, no effect
cue conflict, 128–255 nt                   1.0000 anchor / 0.0000 content     (hg38: 0.06 / 0.84)
```

The representation is **built across layers**, the start codon is causally load-bearing while an
identical perturbation elsewhere is not, and the model propagates a frameshift at 250 nt exactly as
biology does.

---

## 4. What the trace found, including one thing I did not expect

We traced it (D27). It is the circuit we planted in Rung 0, learned rather than hand-set:

- **L0 heads assemble the local trigram.** Ablating `L0H0` halves `L1H3`'s attention to the marker
  (42.4% retained); every other L0 head leaves it at 83–111%. That is K-composition.
- **`L1H3` is the marker head.** It puts 0.80 of its attention on the **T** of the start codon — the
  middle base, where a ±1 window sees all three — and attends there **230× more** than to a decoy ATG.
- **The MLPs count.** Phase builds L0 → L2.

And the model grew a **second** marker head that nobody reads:

```
head    attn→start   →decoy    ratio      ablation: Δloss     accuracy
L1H2      0.6656     0.0016    406.5×      +0.000002          1.0000
L1H3      0.8344     0.0036    230.1×      +8.363416          0.3875
```

`L1H2` attends to the true start codon with **higher specificity than the head that does the work**,
and is causally inert — a four-million-fold difference in effect. Attention inspection picks the wrong
head. The naive gradient ranks them **backwards**. Our integrated-gradient instrument ranks them
correctly and *still* credits the inert head 16.5%.

**Attribution ranks; only intervention decides.** That is the proposal's H4 position (D8), now
demonstrated on a real learned circuit instead of argued. And note what `L1H2` is: precisely
`planted.build_bag_with_unread_marker_head`, which we invented as a paranoid synthetic control and
worried was contrived. **It occurs naturally.**

*(I nearly reported `L1H2` inert off an accuracy metric saturated at 1.0000, which physically could
not have detected an effect. Third instance of the same failure class.)*

---

## 5. Where the λ sweep went wrong, and what replaces it

I proposed sweeping codon-usage bias λ from uniform to real, to find the threshold where the shortcut
kills the circuit. Three arms in, the data contradicted the design:

```
arm        shortcut (% of real hg38)   grokked at step
uniform             0%                      3250
λ = 0.25            4%                      3000
λ = 0.50           16%                      1500
λ = 0.75           35%                      ~750
```

**A stronger shortcut makes the circuit appear *faster*, not less often.** The reason is a flaw in the
design, not in the model: at every λ, predicting the amino acid still requires the frame *exactly*.
Codon bias supplies partial phase information, which smooths the loss landscape and gets the model off
the plateau sooner — but it never removes the *need* for the anchor. **I built a difficulty dial and
called it a necessity dial.**

This is worth keeping as a secondary result — *a partial shortcut scaffolds discovery of the exact
mechanism rather than suppressing it* is counterintuitive and clean — but it is not the measurement we
need.

### The objective-mixture dial

```
loss = (1 − μ) · [ predict the nucleotide ]  +  μ · [ predict the amino acid of the containing codon ]
```

The first term is exactly what every genomic LM is trained on: the frame is *useful* and locally
guessable. The second cannot be satisfied without the frame: no local shortcut, no partial credit.
**μ is a dial on how much of the training signal comes from a term that is unsatisfiable without the
latent variable.**

**Why this framework, specifically:**

1. **It separates necessity from difficulty**, which λ provably cannot. Codon bias makes the frame more
   useful and more guessable in lockstep, because both come from the same statistical structure (D25).
   The mixture varies necessity while the shortcut stays fixed at whatever the data gives.
2. **Both endpoints are already measured.** μ = 0 is real self-supervision → bag (§2). μ = 1 is the
   forcing task → circuit (§3). A threshold **must** exist between them. This is a measurement of a
   quantity guaranteed to be there, not a fishing expedition.
3. **Both terms are real objectives.** μ = 0 is literally DNABERT's loss; μ = 1 is translation. The
   mixture is a training recipe someone could adopt. By contrast λ = 0 is DNA that does not exist.
4. **It is non-circular.** The model is never told the frame; it is told the protein — a function of
   the sequence *given* the frame. Othello-GPT's structure, and the D7 constraint we imposed on the
   phyloP arm.
5. **It generalises to real auxiliary objectives.** phyloP conservation is period-3 in coding regions;
   ribosome profiling depends on frame; protein-level properties require it. If μ\* is small, all of
   these become plausible interventions.

**The deliverable is a number.** μ\*, the smallest weight on a frame-demanding objective that installs
the circuit — reported alongside **the price**: the language-modelling loss sacrificed at that μ. The
sentence we want to be able to write is:

> *"Adding a frame-demanding auxiliary loss at weight μ\* costs X nats of language-modelling
> performance and installs a causally-verified reading-frame circuit."*

Nothing in the original proposal produced an actionable number. This does.

**Readouts are the circuit, never the accuracy** — cue-conflict far-field anchor share, start-codon
ablation Δ(phase probe), and marker-head specificity. All three were validated on the planted twins
before they were ever pointed at a trained model. That is what Rung 0 was for.

---

## 6. Recommended path

**Run the μ sweep on real hg38 CDS, not synthetic sequence.** The amino-acid label is real there — it
is just translation, straight from GENCODE — and μ = 0 becomes *literally* the objective the field
trains on. More work, strictly more convincing, and it removes the "toy model" objection from the
primary result. μ spaced logarithmically; the interesting region is almost certainly small μ.

**Then circuit transplantation (Phase 3).** Transplant is direct by construction — the forcing-task
model and the real-DNA arms share one backbone; only the head differs. And it comes with a prediction
we can make before running it: our synthetic task guarantees the start codon is the *first* ATG, but
**35.1% of real anchored windows contain an upstream ATG** (5′ UTRs carry upstream ORFs; real
initiation weighs Kozak context). The transplanted detector should misfire on exactly those windows.

Of the three transplant experiments, **3c is the one I most want**: hand a real-DNA model the circuit
for free, fine-tune, and watch whether it survives or erodes. Today we can say *the objective does not
build the circuit*. If it erodes, we can say **the objective will not even keep it** — a claim about
the loss landscape rather than about optimisation difficulty. Bag-as-default versus bag-as-attractor.

**Still owed, and small:** the LayerNorm faithfulness gap. Everything the tracer knows about its own
error rate was learned on LN-free planted models. GFM-nano has LayerNorm; the replacement model must
freeze it; that approximation is currently **unmeasured**. Plant a circuit in an LN-equipped tiny
transformer and re-run Tests A and C. A day's work, and it closes the last hole in Gate 0.

**Deprioritise:** Rungs 2 and 3 (splice sites, G-quadruplex), SSMs, crosscoders (cut — Minder et al.
2025 show naive-L1 crosscoders fabricate exactly the model-unique structure H2 would claim), and JSAE
(demoted to a recorded caveat per D16). None of them are on the path to the number.

---

## 7. Where I expect to be wrong

**The forcing task is synthetic, and a reviewer will say so.** Its job is not to be realistic. Its job
is to establish that the tracer finds a real learned circuit when one exists, so that the hg38 negative
is a fact about DNA and not about our instruments. Running the μ sweep on real CDS removes this
objection from the primary claim. State the division of labour explicitly: **the hg38 result is the
finding; the synthetic model is the control that licenses it; the planted models are what prove the
instruments work at all.**

**The 0.4% figure is a marginal estimate.** `H(nt) − H(nt | phase)` ignores context, and the model
already has 256 nt of it. The true conditional gain is probably *smaller*, which strengthens the
argument — but I have not measured it directly, and I should.

**μ is not a clean fraction of "demand."** The amino-acid loss starts at `ln(21) = 3.04` nats, the
nucleotide loss at `ln(4) = 1.39`. Normalise each term by its entropy at initialisation and report raw
gradient norms, or μ means nothing.

**"No circuit at small μ" may be undertraining.** The check is free: if amino-acid accuracy sits at the
0.3768 guessing ceiling, the frame genuinely was not built. And because grokking makes *when you stop*
a scientific choice, compare arms at **matched language-modelling loss**, not matched steps, and always
qualify a negative as "within N steps" rather than "never."

**A partial graft is meaningless without alignment.** Inserting `L1H3` alone into a separately-trained
model fails: the two residual streams have unrelated bases. It needs a learned alignment, and then the
adapter's capacity confounds the result. Say so before a reviewer does.

---

## 8. The claim

The proposal asked whether a training objective installs a circuit. We can now say something more
useful, and defend every clause of it:

> **A circuit appears when the objective makes it necessary, not when it is merely useful.**
> Real genomic self-supervision falls on the wrong side of that line — a reading-frame circuit is
> worth 0.4% of its loss, and it builds a bag of motifs instead. The boundary is measurable, and
> crossing it is a design choice available to anyone training a genomic foundation model.

The instruments that establish this were validated against circuits we built by hand, they have a
measured error rate, and each of them was corrected by a defect that ground truth exposed and that
plausible-looking results would have hidden.
