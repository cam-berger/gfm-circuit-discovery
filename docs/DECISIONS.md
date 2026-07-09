# Design decisions

Each entry records a decision, the evidence that forced it, and what was given up. Written
during the design grilling on 2026-07-08, before any experiment was run.

---

## D1 — Rung 1 cannot run on DNABERT-2 or NT-v2

**Evidence.** Loaded all three tokenizers:

```
DNABERT-2 (BPE):    ATG | GCCATT | GTAA | TGGG | CC | GCTGAAA | GGGTG | CCCGA | TA | G
NT-v2 (6-mer):      ATGGCC | ATTGTA | ATGGGC | CGCTGA | AAGGGT | GCCCGA | T | A | G
SpliceBERT (1-nt):  A T G G C C A T T G T A A T G G G C C G C T G A A A G G G T G ...
```

Three fatal consequences: (a) both are MLM over 4096-unit vocabularies, so the per-*nucleotide*
readout H0 needs does not exist; (b) NT-v2's non-overlapping 6-mers each span exactly two codons,
so frame is constant by construction — the tokenizer hands the model the state variable it was
supposed to compute; (c) under BPE, inserting one nucleotide re-segments everything downstream, so
a frameshift is not a minimal pair.

DNABERT-1's *overlapping* 6-mers do preserve per-nt indexing, but a masked token is recoverable
from its neighbours (k-mer leakage — the reason DNABERT-2 moved to BPE), so its masked readout
cannot carry a causal claim. GPN is single-nt but a ConvNet: no attention.

**Decision.** Rung 1 requires single-nt tokenisation and a phase-dependent per-nucleotide readout.
It does *not* require autoregression — a single-nt MLM satisfies both, and at 1-nt tokenisation a
frameshift indel *is* a clean minimal pair.

**Given up.** The proposal's stated Phase 0.1 substrate.

---

## D2 — A trained model is a known *task*, not a known *edge set*

Rung 1 was called "the case where the correct answer is known in advance." Knowing the task has a
period-3 closed form does not mean the model implements one — Nikankin et al. showed arithmetic,
which certainly has a closed form, is implemented as a bag of heuristics. A GFM-nano trained on real
CDS can drive loss down with trigram statistics and never build a frame counter. Rung 1 would then
fail without telling us whether the *tracer* or the *model* was at fault. A gate that can fail two
ways is not a gate.

**Decision.** Add **Rung 0**: hand-weighted planted models where the edge set is known by
construction. See D3.

---

## D3 — A positive control cannot falsify; twins can

If the tracer reports a compact circuit on the planted model, that is consistent with "the tracer
works" *and* with "the tracer reports compact circuits on everything."

**Decision.** Two planted twins with **identical behaviour** and **different mechanism**:

| | mechanism | marker ablation |
|---|---|---|
| compositional | attends to start marker, computes `(i - start - 1) % 3` | 1.000 → 0.383 |
| heuristic | per-nucleotide lookup table, no frame variable | 0.583 → 0.583 |

Matched by gating the compositional twin's readout on a fixed position mask: solving
`p·1 + (1-p)/3 = bayes_unigram_accuracy()` gives equal accuracy with untouched computation.
Measured: 0.586 vs 0.583, |Δ| = 0.0034.

**Bonus.** H3 proposes JSAE Jacobian sparsity as a composition-vs-heuristic discriminator but never
validates it against ground truth. Behaviourally identical twins give it a measured ROC curve. To
our knowledge this has not been done for JSAE.

**Given up.** The discriminator is calibrated against failure modes we imagined; a real bag-of-motifs
model may be heuristic in unplanted ways.

---

## D4 — Validation must be factored, or the test inherits the ambiguity it was meant to remove

`recall(traced_edges, PLANTED_EDGES)` is not well-typed: traced edges live in *feature* space, planted
edges in *weight* space, and there is no identity map. A single number confounds dictionary-recovery
error with attribution error.

```
Test A   dictionary := identity (monosemantic neurons, no superposition) -> isolates attribution
Test B   plant directions IN superposition, train dictionary             -> isolates recovery (MCC)
Test C   A ∘ B end-to-end                                                -> the reported number
Test D   JSAE sparsity ROC on the twins                                  -> calibrates H3
```

Also: with absolute position embeddings, `phase = pos % 3` is readable straight off `pos_embed`. A
planted circuit anchored to position would be a lookup table. The frame must be anchored to a
**content-determined** start codon, forcing attention + modular arithmetic.

---

## D5 — Build the tracer from scratch

`circuit-tracer` builds on TransformerLens's `HookedTransformer`, which supports only GPT-style
decoder-only models; the `nnsight` backend is experimental and slower. **None of our three model
families are supported**: planted models are custom, GFM-nano is custom, SpliceBERT is a
bidirectional encoder.

**Decision.** Plain PyTorch forward hooks behind a pluggable `Dictionary` interface. SAE, Transcoder,
CLT and JSAE are subclasses of one abstraction — the method table's five papers are one class
hierarchy. Ship **Transcoder first**; CLT is a strict generalisation added once the harness is proven.

**Given up.** Neuronpedia-compatible outputs; battle-tested graph pruning. Mitigated by D3's tests,
which give the from-scratch implementation a measured error rate a borrowed library would not.

---

## D6 — H2 was ill-posed three ways; crosscoders are cut

1. **Readouts don't commute.** AR reads next-nt logits, MLM masked-nt logits, seq→function has no
   nucleotide readout at all. Edge-patch effects would be logit-diffs in different logit spaces; the
   predicted ladder could be produced entirely by readout scale.
   → **Common frozen linear eval head**, identical across arms, backbones frozen.

2. **Completeness ≈ accuracy.** The supervised arm is better at the task, so of course more of the
   computation is present. As specified, H2 cannot distinguish "objective installs C" from "objective
   raises accuracy, and completeness tracks accuracy."
   → **Performance-matched arms**, compared at equal eval-head accuracy.

3. **Crosscoders fabricate the effect.** Minder et al. (NeurIPS 2025, arXiv 2504.02922) show naive-L1
   crosscoders produce *Complete Shrinkage* (falsely model-specific latents) and *Latent Decoupling*.
   H2's claim is precisely "arm A has structure arm B lacks."
   → **Crosscoders cut.** Completeness is a within-model normalised ratio
   `captured attribution / total attribution to eval head` ∈ [0,1], unitless, needing no shared basis.
   Where feature correspondence *is* wanted, use **stimulus-defined identity**: match features by
   tuning curve over designed sequence panels, as a neuroscientist matches receptive fields. Basis-free,
   artifact-free, and native to a domain where any stimulus can be synthesised.

---

## D7 — The supervised arm must not be supervised on the eval task

If seq→function predicts splice sites and the Rung-2 eval head predicts splice sites, "supervised
installs the circuit" is the training objective restated.

**Decision.** Supervise on **phyloP conservation**. In coding regions conservation is period-3 by
construction (wobble positions accumulate synonymous substitutions — the signal PhyloCSF exploits).
The model is never told about frame; frame is the latent variable that makes conservation predictable.
This is the Othello-GPT structure, and it opens up the proposal's own ref 22.

```
arm 0  AR             p(nt_i | nt_<i)         frame useful, never mentioned
arm 1  MLM            p(nt_i | context)       frame useful, never mentioned
arm 2  MLM + phyloP   + λ·MSE(phyloP_i)       frame instrumentally rewarded
arm 3  phyloP only    seq → phyloP            frame is the ONLY route in CDS
arm 3′ phyloP-scrambled (control)             period-3 destroyed, mean conservation preserved
```

Arm 3′ ablates the *objective*, not the model, pre-empting "phyloP is a frame label in disguise."

Two pipeline requirements: **window offsets randomised** (ATG-aligned windows recreate the
`pos % 3` lookup degeneracy), and **frame labels never enter any backbone's training**.

---

## D8 — H4's inequality is backwards

`|eff(node u)| ≥ |eff(any single edge from u)|` always, since ablating a node removes all of its
outgoing edges. "Edge-patch effect > node-only control" predicts the part exceeding the whole.

**Decision.** H4 is about specificity and sign:

```
routing_share(u→m) = |eff_real(edge)| / |eff_real(node u)|
H4a  routing_share(traced) >> routing_share(matched random edges out of u)
H4b  sign(eff) == sign(biophysical prediction)
H4c  |eff_repl - eff_real| = faithfulness gap, reported not assumed
```

---

## D9 — GFM-nano carries the ladder; pretrained models are anchors

SpliceBERT is trained on pre-mRNA: sense strand, transcribed regions. G4's defining axis is
**cross-strand**. Under the original plan Rung 3 would change model *and* biology simultaneously,
so a Rung-3 null would be uninterpretable — destroying exactly the failure-localisation property
Novelty #4 claims.

Note the tokenisation objection that kills DNABERT-2 at Rung 1 **does not apply at Rung 3**: G4
recognition is a window-level readout, so BPE is harmless.

```
                 Rung 1 frame   Rung 2 splice   Rung 3 G4
GFM-nano-CDS        primary          —              —
GFM-nano-DNA        primary       primary        primary     <- carries the ladder
SpliceBERT          anchor        anchor           ✗          (sense strand only)
DNABERT-2             ✗              ✗           anchor       (window readout, BPE harmless)
```

Enforced by `config.assert_cross_strand_capable` and `config.assert_per_nucleotide_readout`.

---

## D10 — Nulls, inference, and the contamination firewall

The proposal's nulls are all model-internal (attention-matched, node-only, random-feature). They
compare one subgraph to another; they cannot say whether the tracer finds structure in sequences that
*have* none. Genomics supplies the null LLM interpretability cannot construct.

A test caught a real error here: a plain composition-preserving shuffle **assembled a `GGGG` run from
a max-run-3 sequence**. A "G-density-matched" decoy containing a G4 is a positive, not a null. The
null is defined by what it lacks and enforced by rejection sampling against the loop-length grammar
on both strands.

**Contamination.** SpliceBERT and DNABERT-2 were pretrained on the whole human genome. **There is no
held-out chromosome for a pretrained anchor.** Fine for mechanistic claims, fatal for generalisation
claims. `config.assert_holdout_valid` raises rather than warns.

**Inference.** Holm across millions of CLT edges destroys power. Five pre-registered confirmatory
tests under Holm; every per-edge claim exploratory under FDR. See `PREREGISTRATION.md`.

---

## D11 — H5 is cut

"Circuit forms at a training phase-change" is pretraining-dynamics emergence work, excluded by the
constraint that this project remain independent of the parallel motif-emergence project. H5 was the
one hypothesis that made this project *dependent* on the project it must be independent from.

---

## D12 — Frozen-attention attribution is structurally blind to QK edges (Test A)

Freezing the attention pattern is what makes the replacement model linear, and it is exactly why
attribution is exact here: incoming edges reconstruct every node's pre-activation to `< 1e-21`.
Recall and precision against the planted OV subgraph are both **1.0**.

But a frozen pattern means the QK circuit — the computation deciding *where* to look — has no edges
at all. Measured on the compositional twin:

```
attribution(embed.tok -> L0.attn.z)  = 0.000000   exactly zero
causal effect of ablating that token = 1.000 -> 0.383 accuracy
```

The tracer assigns zero weight to the single edge without which the circuit does not work. Any
edge-level claim built on frozen-attention attribution inherits this blind spot, and every real
rung of this ladder has a QK component (finding the ATG, finding the donor GT, finding the G-run).

**Decision.** Report OV recall and QK recall separately, never as one number. QK edges need a
separate instrument (pattern-gradient attribution); scoped as work for Phase 1.

## D13 — The planted circuit sat exactly on the ReLU kink

The first planting used MLP bias `-1.0`, which places the one-input-matches case at pre-activation
*exactly 0*. Frozen-gate and real-ReLU interventions then coincide for every patch, and the measured
faithfulness gap was identically `0.000` — the linearisation looked perfect for a reason that
generalises to nothing.

Bias `-1.5` with the output weight doubled computes the identical function while placing the inactive
case strictly below the kink. The gap appears:

```
head dim 0:  replacement=2.0000  real=1.0000  gap=1.0000
head dim 1:  replacement=2.0000  real=1.0000  gap=1.0000
head dim 2:  replacement=1.3333  real=0.6667  gap=0.6667
```

Linear attribution **overstates the true causal effect by exactly 2×**, from a single gate flip. The
faithfulness gap is not a small correction, and Risk #1 is not rhetorical.

Routing share is 1.000 for all three head dims — the traced edge carries the node's entire influence —
and edge effect never exceeds node effect, as D8 requires.

## D14 — The planted model's own activations are not identifiable (Test B)

Test B first used the planted model's residual as its activation source. The SAE reconstructed it
(EV > 0.95) and recovered nothing (MCC 0.099).

Cause: `phase_feature` is a deterministic function of `pos_phase` and `start_phase`, so the residual
code takes only `3 × 3 × 5 = 45` distinct values across every position of every sequence. Dictionary
learning is not identifiable on 45 points; the SAE learns co-occurring combinations rather than atoms.

**Decision.** Test B isolates dictionary recovery on independently-sampled k-sparse supports (64 atoms
in 32 dimensions, k=3, coherence 0.57), where recovery is possible in principle so a low MCC indicts
the dictionary. Result: `|cos| min 0.998`, **MCC 0.945**.

**Consequence for the real experiments, and it is not small.** Any transcoder trained on a circuit
whose features are deterministically coupled will produce features that are combinations, not causes.
Reading frame is *defined* as a deterministic function of position and start codon. Transcoder features
on the frame circuit must therefore be checked for identifiability before any of them is described as
"the phase feature." `tests/test_b_dictionary.py::test_planted_code_is_not_identifiable` pins this.

## D15 — Do not renormalise a TopK decoder

`train_sae` originally unit-normed decoder rows after every optimiser step. That is correct for L1
SAEs, where it stops the sparsity penalty from being gamed by shrinking features and inflating codes.
A TopK SAE carries no L1 term, so the renormalisation only fights the optimiser.

```
with per-step renorm:     EV 0.939   |cos| min 0.393   MCC 0.890
without:                  EV 0.966   |cos| min 0.998   MCC 0.945
```

---

## D16 — Jacobian sparsity measures in-degree, not composition (Test D)

H3's discriminator was first run against a "heuristic" twin that was a unigram lookup: one neuron
reading one input. It scored AUC 1.0 with a separation of **0.0070** — because composition has
in-degree 2 and a lone shortcut has in-degree 1. The test passed and meant nothing.

A bag of motifs must actually be a bag. The heuristic twin is now five local nucleotide detectors
at offsets -2..+2 casting independent weak votes, no frame variable anywhere. Rebuilt:

```
model           accuracy   Jacobian in-degree   Hoyer sparsity
compositional     1.0000          2.00              0.9882
bag               0.7872         14.91              0.9245
shortcut          0.5793          3.00              0.9818

AUC(comp > bag)      = 1.0000    separation = +0.0637
AUC(comp > shortcut) = 1.0000    separation = +0.0063     <- 10x smaller
```

**H3 survives, in restricted form.** Jacobian sparsity licenses "this model does not use many weak
features." It does **not** license "this model composes." A model that cheats with a single strong
shortcut is near-maximally sparse and is indistinguishable from a genuine composition circuit.

**Decision.** The pre-registered C3 must state the alternative it tests against. Rung 3's G-density
null is a *bag* (many G positions summed) so the discriminator applies. Any single-motif shortcut —
e.g. "there is a GGGG here" — is invisible to it and needs a separate control.

## D17 — Perfect reconstruction, meaningless features (Test C)

The compositional twin's MLP emits exactly three distinct outputs. A TopK transcoder reconstructs
them with **EV = 1.0000** and recovers them as concepts not at all:

```
b_dec[6:9] = [0.999, 0.006, -0.006]     the decoder BIAS is the phase-0 output
W_dec[7]   = [-0.990, -0.007, 0.999]    a DIFFERENCE feature: "switch phase 0 -> phase 2"

live features           6      (for 3 concepts)
phase recovered from features   0.6617    (from neurons: 1.0000)
```

Phase 0 is not a feature. It is what the transcoder emits when nothing fires. Feature 7 is a signed
difference vector. Reading its top-activating examples would show phase-2 sites and invite the name
"the phase-2 feature," while phase 0 stayed invisible.

Because Test A showed attribution is exact and Test B showed dictionary recovery works when features
*are* identifiable, the entire end-to-end shortfall is attributable to the dictionary. The factoring
did the job it was built for.

**Structure survives what nodes do not.** Abstract edge recall is 1.0; precision is 0.6 — the learned
encoder reads the token dimensions and the constant, which the planted circuit never touches.

**Consequence.** Report circuit *shape* from transcoder graphs; do not name a feature after its
top-activating examples without an identifiability check and an inspection of `b_dec`. A concept
absorbed into the decoder bias is a concept the circuit diagram will silently omit.

## D18 — `explained_variance` was flattering itself

The first implementation used element-wise `.var()`. Against a 36-dimensional target whose dimensions
are mostly zero, that reads near 1.0 almost regardless of fit. Replaced with `1 - FVU`. On this data
both give 1.0000, but only by luck; the metric is now correct by construction.

---

## D19 — Scope: Rungs 0–1, hypotheses H0/H1/H2/H4. H3 leaves the critical path

Per Cam, 2026-07-08. H3 (JSAE computation-sparsity) was a speculative addition and is demoted to an
exploratory result; D16's finding stands as a recorded caveat rather than a gate. Rungs 2 (splice)
and 3 (G-quadruplex) become future work.

The paper is now: **a validated circuit tracer, and a controlled test of whether a training objective
installs the reading-frame circuit.** Rung 0 supplies the tracer's measured error rate; Rung 1 supplies
the phenomenon; H2's four objective arms supply the causal claim; H4 supplies the causal test.

Cut from the critical path: JSAE, `dicts/base.py::JSAE`, G4 nulls, DNABERT-2 anchor, cross-strand
substrate. `nulls/sequence.py::g_density_matched` is retained -- it costs nothing and Rung 3 may return.

## D20 — Naive pattern-gradient attribution vanishes on a confident head (Test A2)

Reading frame is anchored by a QK edge: locating the ATG *is* the decide-where-to-look step. Test A
showed frozen-attention attribution scores that edge at exactly 0.000 (D12), so Rung 1 cannot be traced
without an instrument that sees through the attention pattern.

The obvious instrument -- differentiate the readout with respect to the attention scores -- **fails, and
fails silently**:

```
max |dL/dscore| over all heads and positions = 3.7e-15
```

The marker head has learned its job. Its score on the marker key is ~36 and ~0 elsewhere, so the softmax
is saturated and the gradient is annihilated. Gradient attribution reports the edge as irrelevant
*because* it is so relevant that the head is certain. Every sharp motif-finding head in a real GFM --
ATG, donor GT, G-run -- has exactly this shape. A tracer built on plain pattern-gradients would find
nothing and call it a negative result.

**Decision.** Integrate the gradient from a content-free baseline (the head's positional bias alone) to
the real scores -- the attention-pattern analogue of EAP-IG, which the proposal's method table already
listed without saying why it was needed. Then decompose each score linearly into the input dimensions
that produced it, key side and query side separately.

Validated on the planted twins, where both answers are known:

```
key side    embed.tok   -> L0.attn.pattern         0.003803    (W_K reads only the M dimension)
query side  embed.const -> L0.attn.pattern.query   0.019017    (W_Q reads only the constant)
every other residual group                         0.000000

share of key attribution sitting on the marker token          1.0000
marker ablation                            0.003803 -> 0.000000
bag-of-motifs twin, max QK edge                               0.0        falsification control
```

A head driven purely by positional bias has zero content contribution, so it receives zero QK
attribution by construction. **Combined OV + QK edge recall against `PLANTED_EDGES` is now 1.0.**


---

## D21 — Frame labels validated against biology, not against our own code (Phase 1)

`data/genome.py` computes codon phase from GENCODE's `frame` field and the strand. Both are easy to
get subtly wrong: on the minus strand the first codon base is `end - phase`, not `start + phase`, and
codon position advances as the genomic coordinate *decreases*. An off-by-one there would produce
labels that are internally consistent, pass every unit test, and be wrong.

So the check is external. If the arithmetic is right, in-frame codons in a coding exon must almost
never be stop codons; the two shifted frames should hit stops at the rate base composition predicts.
GENCODE excludes the terminal stop codon from CDS features, so the in-frame rate should be *exactly*
zero. Measured on 2,000 windows across 624,533 train CDS exons:

```
in-frame (our labels)    0.0000 stop codons    n = 117,481
shifted +1               0.0419               n = 116,854
shifted +2               0.0554               n = 116,342
```

Reproduce with `python scripts/validate_genome.py`.

**And the offset randomisation is mutation-tested**, because the guard matters more than the code it
guards. Anchoring windows at CDS starts makes `index % 3` predict phase at **1.000**; the randomised
sampler scores **0.3446** on real hg38 against a chance floor of 0.3333. A model with absolute
position embeddings would have read phase straight off `pos_embed`, and the loss curve would have
looked fine.


---

## D22 — The data made the planted circuit impossible to learn (Phase 1)

The first Rung 1 probe was a strong positive: codon phase is linearly decodable from GFM-nano's
residual stream at **0.7936**, against a nucleotide baseline of 0.4105, an `index % 3` baseline of
0.3401, a label-shuffled floor of 0.3351, and — the control that matters — an **untrained** model at
random init scoring 0.4094. Absolute position embeddings give the model nothing, which is what the
window randomisation of D21 was for.

Then two things about the *shape* of that result:

**It appears entirely at layer 0 and is flat thereafter** (L0 0.7936 → L5 0.7914). Nothing is built
after the first block.

**Our windows contain no marker to find.** They are 256 nt drawn at random offsets *inside* coding
exons. Internal coding exons have no ATG at all. So the planted circuit's computation — find the
start codon, compute `(i − i_ATG) mod 3` — is not available, and the model can only be inferring
phase from codon-usage statistics. A purely local naive-Bayes predictor confirms how far that goes:

```
context pooled           Bayes accuracy
±0 nt  (1 position)          0.4104
±2 nt  (5 positions)         0.5114
±7 nt  (15 positions)        0.6182
±15 nt (31 positions)        0.6918
```

The model's 0.7936 beats a 31-position independent-position bag, but that is a naive-Bayes lower
bound and the model has 256 nt of context. **0.794 is not evidence for a circuit rather than a bag.**
Rung 1 as specified was studying the wrong object.

**Decision (Cam, 2026-07-08).** Fix the data, and make the fix the experiment. `data/genome.py` now
builds two regimes:

```
anchored   the window contains the translation start codon (72,668 start exons in train)
internal   the window is drawn inside a coding exon and provably has no ATG (551,865 exons)
```

Train on a 50/50 mixture, so one model has both regimes available and the marker-ablation assay from
Rung 0 transfers directly. The prediction is pre-registerable and sharp: **an anchored counter where
an anchor exists, a bag of motifs where none does** — same weights, same objective, only the
availability of an anchor changing.

If it holds, the tracer will have distinguished composition from bag-of-motifs on a real genomic model,
with ground truth supplied by the *data design* rather than by a planted weight matrix. That is a
strictly stronger claim than either regime alone, and it is what the Rung 0 twins were rehearsing.

Offset randomisation is preserved in the anchored sampler: the ATG sits at a uniformly random position
in the window (measured 0–163), so `index % 3` still predicts phase at 0.3494, i.e. chance.
`scripts/anchor_test.py` runs the assay.


---

## D23 — The anchor assay needed two fixes before it could say anything

The first version of `scripts/anchor_test.py` compared "ablate the ATG" against "ablate a codon 30 nt
downstream" and reported a single accuracy drop. Two confounds made that number uninterpretable.

**The downstream control is vacuous for a causal model.** The AR arm cannot attend to the future, so
ablating a codon downstream of a scored position has *no possible* effect on it. The measured control
drop was exactly `0.0000` in the near bins — arithmetic, not specificity.

**`CTC` is not informativeness-matched to `ATG`.** A start codon carries far more phase information
than an average codon, so a bag of motifs weights it heavily and is genuinely hurt by its removal.
Deleting an arbitrary codon and finding a smaller effect proves nothing about anchoring.

**And the ablated codon's own positions were being scored** — deleting the token the probe reads is
not evidence about the model's representation.

Both fixed:

1. **The discriminating signal is distance, not magnitude.** An anchored counter loses accuracy at
   *every* downstream position; a bag recovers once you are past its pooling width. Drops are now
   reported per distance bin, measured from each ablation's own site, excluding that site.
2. **The control is a decoy ATG.** Anchored windows contain ~2.5 other ATG trinucleotides, ~0.8 of
   them in frame. Ablating an in-frame decoy is the same perturbation, on an equally informative
   motif, at the wrong location.

Validated on the AR **internal-only** model, which never saw an anchored window in training and must
therefore be a bag:

```
distance from ablation   true ATG drop     decoy ATG drop
3-15 nt                       +0.2092           +0.0616
16-31 nt                      +0.0985           +0.0387
32-63 nt                      +0.0625           +0.0262
64-255 nt                     +0.0085           +0.0106
```

Both decay to nothing by 64 nt. The true start codon is a *stronger local cue* than a decoy — three
times stronger nearby, plausibly its Kozak context — but it is not an anchor. **VERDICT: bag of
motifs**, exactly as this model's training data predicts. The negative arm of the assay works.

The mixed-pool models are the test. Pre-registered prediction (D22): if training on windows that
contain a real start codon installs an anchored counter, the true-ATG drop will persist at ≥64 nt
while the decoy drop stays near zero.

*Also: the MLM internal-only arm was stopped at its step-4000 checkpoint to free the GPU for the
mixed arms. It is a comparison arm for H2, not evidence for this question.*


---

## D24 — Ablation cannot detect a redundant circuit. Cue conflict can.

`anchor_test.py` ablates the start codon and asks whether phase accuracy falls. **It cannot answer
the question, and the "VERDICT: bag of motifs" it printed was not earned.** Deleting the ATG removes
one source of phase information and leaves the other — downstream codon statistics, worth 0.79 on
their own — completely intact. A model that genuinely counts from the ATG can lose nothing when the
ATG is destroyed, because it falls back on local evidence. Redundancy makes ablation blind. (Cam's
observation: *"maybe it learns to correct itself instead of falling apart."*)

Do not remove information. Make the cues **contradict** each other. Insert one nucleotide immediately
after the ATG; everything downstream shifts by one, so at every downstream position:

```
anchor says    phase = (q - atg) % 3        counting codons from the start
content says   phase = (q - 1 - atg) % 3    the codon statistics moved with the sequence
```

They differ by exactly one. The probe must pick a side.

```
                       MIXED (trained w/ anchors)   INTERNAL-ONLY (never saw one)
distance from ATG      anchor    content            anchor    content
4-15 nt                0.5944    0.1524             0.5999    0.1553
16-31 nt               0.4175    0.3152             0.4263    0.3081
32-63 nt               0.2423    0.5424             0.2490    0.5313
64-127 nt              0.1119    0.7562             0.1113    0.7550
128-255 nt             0.0626    0.8354             0.0569    0.8406
```

**The internal-only model is the control that makes this airtight.** It never saw an anchored window
and cannot be anchored, yet its curve is identical. So the near-field "anchor following" is not
counting — positions just past the insertion pool context from upstream of it, where the sequence is
unshifted and the local evidence agrees with the anchor. It is pooling across the boundary.

Both models follow **local content** at long range. They are statistically indistinguishable. The
crossover at ~32 nt measures the **pooling width** directly, matching the ±15 nt naive-Bayes estimate
(31 positions → 0.6918).

**GFM-nano does not build an anchored frame counter, on either data regime.** And there is a reason:

```
H(nucleotide)            = 1.3714 nats
H(nucleotide | phase)    = 1.3473 nats
gain from PERFECT phase  = 0.0241 nats
```

The model already infers phase at 0.79 from local statistics, so an anchored counter buys at most
~0.005 nats — **0.40% of a 1.27-nat loss**, and that is an upper bound. Longer training will not
conjure it; there is almost nothing to grok toward. The bag is not a failure of optimisation. **It is
the correct solution to the objective we set.**

This is the genomic instance of Nikankin's arithmetic result, and it reframes H2: the question is no
longer "does an objective install a circuit" but "what must an objective demand before a circuit
becomes worth building."

## D25 — No self-supervised objective can force an anchor

Any phase-dependent structure in the sequence is locally detectable: it repeats every three bases, so
a sufficiently wide bag finds it. Make the sequence phase-symmetric so that no local cue exists, and
knowing the phase stops helping next-nucleotide prediction — the incentive disappears with the cue.
Self-supervision cannot escape this.

So the forcing experiment must be **supervised on a readout that requires the frame**, which is D7's
logic in miniature:

```
sequence   random 5' region containing no ATG | ATG | codons drawn uniformly from the 61 sense codons
           -> local nucleotide statistics carry essentially no phase information
label      the amino acid of the codon containing each position -- never the frame itself
           -> emitting it requires knowing which triplet is in frame
```

Without the frame, accuracy is capped near 0.33 (guess the alignment). With it, 1.0. The model is
never told the frame; it is told the protein. The anchored counter becomes the *only* solution rather
than a 0.4% improvement — the learned analogue of Rung 0's planted compositional twin.


---

## D26 — The forcing task installs the circuit, and it groks

Uniform-codon synthetic CDS (D25): i.i.d. uniform sequence, `I(nucleotide; phase) = 0.000001` nats
against 0.0241 in real hg38 CDS. Label is the amino acid of the codon containing each position. The
model is never told the frame. Guessing the alignment caps accuracy at **0.3768**; knowing the frame
gives 1.0. Bidirectional by necessity — the amino acid at *i* depends on nucleotides at *i+1*, *i+2*.

**It sat on the frameless ceiling for 2,750 steps and then transitioned in ~500.**

```
step 2750   aa_acc 0.4006    frameless   (ceiling 0.3768)
step 3000   aa_acc 0.4426
step 3250   aa_acc 0.9790
step 3750   aa_acc 0.9987
```

There is no gradual path: every intermediate accuracy would require *partial* knowledge of the frame,
which this task does not offer. (Cam's grokking intuition, vindicated — on the task where the circuit
is worth building. On real DNA it was worth 0.4% of the loss and never appeared.)

**All four assays flip**, against the same battery that found nothing in hg38:

```
phase probe             real hg38 model      forcing-task model
  resid.embed               0.4061               0.3472
  L0.resid_post             0.7936  flat         0.4049
  L1.resid_post             0.7928               0.8768   builds
  L2.resid_post             0.7923               0.9907
  L5.resid_post             0.7914               0.9892

amino-acid accuracy
  intact                                         0.9875
  START CODON ablated                            0.3611   below the 0.3768 ceiling
  downstream Met ablated (control)               0.9749   identical perturbation, no effect

cue conflict (follows anchor / follows content)
  4-15 nt                                        0.9908 / 0.0061
  64-127 nt                                      0.9909 / 0.0065
  128-255 nt                                     0.9914 / 0.0061   no decay at any distance
```

Real-DNA model, same bin, followed **content** at 0.8354.

Three independent signatures agree. The representation is built **across layers** rather than read off
in one step — the hallmark of composition, versus the flat L0 profile of a single pooling operation.
The start codon is causally load-bearing while an identical perturbation to an ordinary Met codon is
not. And under cue conflict the model propagates a frameshift exactly as biology does.

**This is the learned analogue of Rung 0's planted compositional twin**, in a real transformer with
LayerNorm, and it is the substrate on which the tracer should now be run: we know a marker-finding QK
edge and a modular counter must exist, and we know the tracer's error rate on planted ground truth.

The scientific claim sharpens accordingly. Not "does a training objective install a circuit" but:
**a circuit appears exactly when the objective makes it necessary, and not when it is merely useful.**
Real genomic self-supervision falls on the wrong side of that line.
