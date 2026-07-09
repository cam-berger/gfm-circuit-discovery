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

At the final checkpoint (step 12,000) the circuit is exact, not approximate:

```
phase probe        L0 0.4116 -> L1 0.9138 -> L2 1.0000   (untrained control 0.347 at every layer)
amino-acid accuracy                          1.0000
  start codon ablated                        0.3552   below the 0.3768 guessing ceiling
  downstream Met ablated (control)           0.9871
cue conflict (anchor / content), 128-255 nt  1.0000 / 0.0000
```

**This is the learned analogue of Rung 0's planted compositional twin**, in a real transformer with
LayerNorm, and it is the substrate on which the tracer should now be run: we know a marker-finding QK
edge and a modular counter must exist, and we know the tracer's error rate on planted ground truth.

The scientific claim sharpens accordingly. Not "does a training objective install a circuit" but:
**a circuit appears exactly when the objective makes it necessary, and not when it is merely useful.**
Real genomic self-supervision falls on the wrong side of that line.


---

## D27 — The traced circuit, and a naturally-occurring unread marker head

Tracing the forcing-task model (`scripts/trace_uniform.py`, `scripts/qk_ig_nano.py`). The circuit is
the one Rung 0 planted, learned rather than hand-set:

```
L0 heads          assemble the local trigram; ablating L0H0 halves L1H3's attention to the marker
                  (42.4% retained; every other L0 head leaves it at 83-111%)  -> K-composition
L1H3              attends 0.80 of its mass to the T of the start codon -- the middle base, where a
                  +-1 window sees all three -- and 230x more to the true start than to a decoy ATG
L1-L2 MLPs        phase probe: L0 0.4116 -> L1 0.9138 -> L2 1.0000 (untrained control 0.347 flat)
```

**A measurement error, corrected.** The first survey measured attention to the *A* of the ATG and
concluded that attention magnitude is uncorrelated with causal importance. It is not: the marker head
attends to the *T*. Measured on the right column, the top-attention heads are the causal ones.

**But the sharper failure is real, and it is worse.** The model contains two heads that attend almost
exclusively to the true start codon:

```
head    attn->start   ->decoy    ratio     ablation: loss     accuracy
L1H2       0.6656     0.0016    406.5x     +0.000002          1.0000
L1H3       0.8344     0.0036    230.1x     +8.363416          0.3875
```

**L1H2 is causally inert** — its effect on the loss is four million times smaller than L1H3's — while
looking like the *better* marker head under attention inspection. It is a live, exquisitely-tuned
marker head whose output nothing downstream reads: precisely
`planted.build_bag_with_unread_marker_head`, which we invented as a synthetic control in D20 and
worried was contrived. **It occurs naturally in a trained transformer.**

Every attribution method mis-handles it to some degree:

| method | verdict |
|---|---|
| attention pattern | L1H2 looks *better* (406x vs 230x) |
| naive score gradient | ranks L1H2 **above** L1H3 (1881 vs 905) — backwards |
| integrated gradient (D20) | ranks correctly (83.3% vs 16.5%) but still credits the inert head 16.5% |
| ablation on **accuracy** | correct here only by luck: accuracy is saturated at 1.0000 |
| ablation on **loss** | 8.36 vs 0.000002 — unambiguous |

**Attribution ranks; only intervention decides.** This is D8/H4's position — attribution as a cheap
screen, causal effect as the claim — demonstrated on a real learned circuit rather than argued. And it
is a third instance of the same failure class that has recurred all through this project: *a saturated
measurement reports nothing where the effect is largest.* Saturated softmax (D20), saturated
element-wise variance (D18), saturated accuracy (here).

Still open on this model: the counter itself (is `(i - i_ATG) mod 3` computed in Fourier features, as
in Nanda's grokking result?), and the **LayerNorm faithfulness gap** — every number the tracer knows
about its own error rate came from LN-free planted models.


---

## D28 — Two claims raised and retracted in one hour, and the ceiling table that replaces them

Designing the μ sweep exposed that the λ sweep and my first μ proposal shared a category error, and
that two intermediate claims I made along the way were wrong. Recording all of it, because the
retracted claims were each plausible enough to have shipped.

### The λ sweep measures difficulty, not necessity

**Every λ arm is μ = 1** — a pure amino-acid readout. At every λ, predicting the amino acid still
requires the frame *exactly*; codon bias supplies partial phase information, which smooths the loss
landscape and gets the model off the plateau sooner but never removes the *need* for the anchor.

```
arm        shortcut (% of real)   grokked at step
uniform             0%                3250
λ = 0.25            4%                3000
λ = 0.50           16%                1500
λ = 0.75           35%                ~750
λ = 1.00           63%                ~1250 (aa 0.9374)
```

A stronger shortcut makes the circuit appear **faster**, never less often. I built a difficulty dial
and called it a necessity dial. Kept as a secondary result — *a partial shortcut scaffolds discovery of
the exact mechanism rather than suppressing it* — and reinterpreted as the **shortcut row of a 2×2 at
μ = 1**, which is what it actually is.

### The uniform-codon μ sweep would repeat the error

On uniform-codon sequence the nucleotide term has almost nothing to predict:

```
data                       H(nt)   H(nt|prev6)   reducible
uniform-codon synthetic   1.3863     1.3631        0.023     (mostly 6-mer estimator bias)
λ = 1 (real codon usage)  1.3839     1.3397        0.044
real hg38 CDS             1.3707     1.2026        0.168
```

`ln 4 = 1.3863`. So on uniform codons **μ = 0 is the void, not the bag** — the LM term is already at its
optimum for any model emitting a uniform distribution, and exerts no pull toward a competing solution.
μ\* there would measure gradient magnitude under zero competition. The bag only competes on real CDS.

### RETRACTED: "genomic landmarks are not identifiable from sequence"

A hand-written Kozak rule (purine at −3, G at +4) locates the true start in only 70.7% of anchored
windows, from which I computed that a circuit could reach 0.8034 against a bag at ~0.80 and therefore
buys **+0.0034** — concluding that the anchor is unfindable and no μ installs the circuit on real DNA.

**A learned detector demolishes this.** Given ±12 nt of one-hot context plus ORF-integrity features
(an authentic start has no in-frame stop downstream), gradient boosting reaches:

```
exact start accuracy   0.8535
FRAME accuracy (mod 3) 0.9215    <- the only one that matters: an in-frame decoy is harmless
```

My rule was a strawman. The circuit ceiling is ~0.925, and it buys **+0.125**, not +0.003.

### RETRACTED: "so real DNA's frame is anchor-free and the circuit is redundant"

The follow-on guess — that stop-codon absence alone determines the frame, making the anchor pointless —
is also wrong, though closer:

```
frame accuracy, anchor-free stop-codon scan
   30 nt context   0.4492
   90 nt           0.6533
  240 nt           0.8242     (34% of windows still tie)
```

### The table that replaces both

```
frame accuracy on real hg38 CDS
  bag, ±15 nt naive Bayes                      0.6419
  anchor-free stop-codon scan, 240 nt          0.8242
  the trained model actually achieves          0.7936
  learned anchor detector (ATG + Kozak + ORF)  0.9215
```

**The trained model sits essentially at the anchor-free ceiling.** It is not lazy relative to what is
locally available — it has extracted nearly all of the distributed frame signal the genetic code
provides, and it stops exactly where an anchor would be required. The anchor is worth a further ~+0.10,
and the model does not take it.

That is a better statement of this project's finding than "the circuit is worth 0.4% of the loss,"
which remains true of the **μ = 0** objective and is the reason no circuit forms there. The two
statements are the same fact measured under two objectives, and the 2×2 (see `PREREGISTRATION.md`
addendum) separates them.


---

## D29 — Same behaviour, opposite mechanism: the λ arms split into a counter and a scan

Every λ arm reaches `aa_acc = 1.0000`. **That number is the training accuracy, and it does not
distinguish the mechanisms.** Running the causal verdict on them (cue conflict + start-codon ablation,
both validated on the planted twins) splits the sweep in two:

```
arm       aa acc   ATG-abl dAA   anchor   content   mechanism
uniform   1.0000      0.6320     1.0000    0.0000   ANCHORED COUNTER
lam0.25   1.0000      0.5951     0.5835    0.2579   ANCHORED COUNTER
lam0.5    1.0000      0.5455     0.6048    0.2023   ANCHORED COUNTER
lam0.75   0.9999      0.2836     0.7025    0.2640   ANCHORED COUNTER
lam1      1.0000      0.0153     0.0027    0.9924   anchor-free
lam1.5    0.9988      0.0127     0.0000    0.9997   anchor-free
lam2      1.0000      0.0124     0.0000    1.0000   anchor-free
```

**These are the planted twins, learned, on a sweep.** Behaviourally identical, mechanistically opposite.
Had we scored the sweep on accuracy — as I did for an hour — we would have reported "the circuit forms
at every λ, faster each time" and been wrong about four of the seven arms.

### The controlling variable is not codon bias. It is whether an anchor-free route exists.

```
lambda   P(in-frame stop)   in-frame stops per 85-codon window   mechanism
  0.75       0.0117                    1.0                        anchored counter
  1.00       0.00001                   0.0                        anchor-free
```

At λ ≤ 0.75 the true frame still contains stop codons, so "the stop-free frame" is unreliable and the
model **must** anchor. At λ ≥ 1 — real human codon usage, and GENCODE's CDS excludes the terminal stop
— in-frame stops vanish, the true frame becomes the *unique* stop-free frame across 85 codons, and the
model recovers it by scanning. No anchor needed.

### This unifies the whole project

Real DNA has no in-frame stops. That is why hg38 GFM-nano followed *content* in the cue conflict (D24),
why it sits at the anchor-free ceiling of 0.79–0.82 (D28), and why the anchored circuit never appeared.

**It is not that the objective fails to pay for the circuit. The genetic code already supplies a cheaper
route to the same answer, and the model takes it — even under an objective that demands the frame
exactly (μ = 1).** The "0.4% of the loss" figure (D24) is the μ = 0 shadow of this deeper fact.

### Refinement, not retraction

*"A partial shortcut scaffolds discovery of the exact mechanism"* survives, but only inside the anchored
regime: among λ ≤ 0.75 the grok step falls 3250 → 3000 → 1500 → ~750. The apparent speed-up at λ ≥ 1 was
the model finding a **different, easier mechanism**, not finding the counter faster.

### The confound, and the control that kills it

λ moves stop depletion and overall codon bias *together*. Uniform over the **61 sense codons** separates
them: no codon bias, no in-frame stops.

```
                    P(in-frame stop)    I(nt;phase)
uniform, 64 codons      0.04688          0.000001
61 sense codons         0.00000          0.00157      <- 15x less bias than lam=1, stops fully depleted
lam = 1 (real usage)    0.00001          0.01516
real hg38 CDS           ~0               0.02410
```

**Prediction, registered before the run:** if the mechanism switch is caused by stop depletion, the
sense-codon model is **anchor-free**. If it is caused by codon bias, it is an **anchored counter**.


---

## D30 — RETRACTED (D29's stop-scan story). The criterion is SUFFICIENCY, not stop depletion.

D29 explained the λ mechanism switch as *"at λ ≥ 1 the true frame is the unique stop-free frame, so
the model scans."* The control it registered falsifies it.

**Uniform over the 61 sense codons: zero codon bias, zero in-frame stops.** Registered prediction: if
stop depletion causes the switch → anchor-free. Result:

```
sense-codon control, step 4000
  aa accuracy            0.9994
  ATG-ablation ΔAA       0.3370
  cue conflict: anchor   0.8205    content 0.1486
  -> ANCHORED COUNTER
```

Stop depletion alone does **not** cause the switch. The stop-scan route exists in the sense-codon data
(out-of-frame triplets still hit stops at 3/64) and the model declines it.

### What actually governs the choice

Measure the **anchor-free ceiling**: the best phase accuracy obtainable with no start codon, using
full codon-level evidence pooled over a context window.

```
data        I(nt;phase)   ctx=60  ctx=120  ctx=240   mechanism the model chose
uniform       0.00003     0.000    0.000    0.000    ANCHORED
sense         0.00162     0.505    0.814    0.965    ANCHORED
lam0.5        0.00393     0.410    0.508    0.533    ANCHORED
lam0.75       0.00854     0.575    0.815    0.935    ANCHORED
lam1          0.01471     0.821    0.976    1.0000   anchor-free
lam2          0.05778     0.995    1.000    1.0000   anchor-free
```

**The model builds the circuit if and only if the anchor-free route cannot reach the ceiling.** Not
"if the shortcut is weak" — *if the shortcut is imperfect*. At λ = 0.75 and in the sense control the
pooled estimator reaches 0.935 and 0.965 and the counter is built anyway, because the task demands
1.0 and a 96.5%-accurate frame caps amino-acid accuracy at 96.5%. At λ ≥ 1 the pooled estimator is
**exactly sufficient**, and the circuit vanishes.

### The criterion, stated generally

> **A mechanism is unavoidable exactly when no statistic pooled over the model's context window is a
> sufficient statistic for the target.**

This is the question worth asking of genomics, and reading frame in real CDS *fails* the test under
μ = 0: the frame is nearly a sufficient statistic of local codon composition, and the residual 0.4%
of loss does not pay for a circuit (D24).

### It also predicts cells C and D, before they run

```
REAL hg38 CDS, anchor-free ceiling (codon-level naive Bayes, no start codon)
  ctx  60 nt   0.7542
  ctx 120 nt   0.9216
  ctx 240 nt   0.9509      <- IMPERFECT, and adjacent to the sense control's 0.965
```

**Registered prediction: at μ = 1, cells C and D build an ANCHORED COUNTER.** Real DNA's anchor-free
route caps at 0.95, so an objective demanding exactness must anchor.

And a caveat this exposes: **synthetic λ = 1 is a bad model of real CDS.** Same codon usage, but the
naive-Bayes independence assumption makes its anchor-free route *perfect* (1.0000) where real DNA's is
not (0.9509). Real codons are correlated; the shortcut does not close. Any claim transferred from the
λ arms to real DNA must be re-measured, not assumed.

### What survives from D29

The mechanism split itself — same behaviour (aa_acc 1.0000 everywhere), opposite mechanism, revealed
only by causal readouts — stands, and remains the strongest single result in the project. Only its
*explanation* was wrong.


---

## D31 — Cell B's verdict is void: the run was cut off mid-transition

`cellB_s0_step8000` returns **NO CIRCUIT** under the frozen rule (anchor share 0.3569 < 0.50), and the
verdict is not interpretable, because the model never converged.

```
cell B trajectory   (guessing ceiling 0.3768; achievable ceiling ~0.926)
  step 1000   aa 0.3914     plateau
  step 5000   aa 0.4093
  step 7000   aa 0.4282
  step 8000   aa 0.6294     transition begins -- and the budget ends
```

It sat on the guessing plateau for 7,000 steps and started climbing exactly as the run ended.
Meanwhile the causal signal is already there:

```
start codon ablated       ΔaaAcc = +0.2356
decoy Met ablated (ctl)   ΔaaAcc = +0.0078      -> 30x specificity
cue conflict              anchor 0.3569 / content 0.3263 / neither 0.317
```

A 30× specific anchor with a three-way-split cue conflict is what a **half-built** circuit looks like,
not an absent one.

Our own pre-registration anticipated this: *"a negative is undertraining until amino-acid accuracy is
checked against the guessing ceiling."* Cell B fails that check — 0.61 against an achievable 0.926.

**Action:** cell B re-run at 24,000 steps (3× budget), seed 1. Cell B is the hardest cell by
construction (no shortcut *and* a matched-hard detector), so a longer transition is expected, not
anomalous. The 8k verdict is withdrawn, not reported.

**Note for every other cell:** grokking makes the stopping point a scientific choice. Arms must be
compared at matched achieved accuracy relative to their own ceilings, never at matched steps.


---

## D32 — A third mechanism: the relay. The pre-registered binary verdict cannot express it.

Cells C and D (real hg38 CDS, μ = 1) both print **NO CIRCUIT** under the frozen rule. The two causal
assays **disagree**, and that disagreement is the finding.

```
                        cell C (filtered)      cell D (natural)
aa accuracy, intact          0.9997                 0.9914
start codon ablated          0.6320  Δ +0.3678      0.8202  Δ +0.1711
decoy Met ablated (ctl)      0.9847  Δ +0.0148      0.9761  Δ +0.0218   -> 25x, 8x specificity
cue conflict (>=64 nt)       anchor 0.0396          anchor 0.1084
                             content 0.9131         content 0.8001
```

**The ablation effect does not decay with distance.** By our own D24 logic that is the counter signature:

```
cell C, aa accuracy by distance from the start codon
  3-15 nt      0.9991 -> 0.6494   +0.3497
  64-127 nt    0.9998 -> 0.6365   +0.3633
  128-255 nt   1.0000 -> 0.6532   +0.3468     flat
```

And moving the cue-conflict insertion away from the anchor (in case it was corrupting the anchor's own
neighbourhood — it was, slightly) does not rescue the anchor:

```
cell C   insert at ATG+3    anchor 0.0393  content 0.9140
         insert at ATG+30   anchor 0.0560  content 0.8936
         insert at ATG+60   anchor 0.0931  content 0.8661
         insert at ATG+90   anchor 0.1548  content 0.7937
```

### The relay

Absolute phase is **set by the anchor**; the register is then **propagated by local codon structure**.

- Kill the anchor → the register locks to the wrong phase → every position is wrong, at every distance.
- Insert one nucleotide mid-sequence → the register re-locks to the local structure downstream → content.

Both observations, one mechanism. **Neither planted twin is a relay**, so the binary verdict — anchor
share > 0.50 AND ablation > 0.10 — cannot express it, and prints NO CIRCUIT for a model whose start
codon is causally load-bearing at 255 nt.

### Three mechanisms, separated by the pair of assays

| regime | ATG ablation, far field | cue conflict, far field | mechanism |
|---|---|---|---|
| real DNA, μ = 0 (D23) | **decays** to +0.0085 | content 0.84 | bag of motifs |
| real DNA, μ = 1 (C, D) | **flat** +0.35 / +0.17 | content 0.91 / 0.80 | **relay** |
| uniform codons, μ = 1 | collapse *below* guessing | **anchor 1.0000** | anchored counter |

**Increasing demand moves the mechanism bag → relay → counter; shortcut availability decides where it
stops.** The assays are a 2-D signature, not a binary, and must be reported as such.

### Status of the D30 prediction

**Half vindicated, half falsified.** Real DNA at μ = 1 *does* build an anchor-dependent long-range
mechanism — the flat ablation proves it, and the μ = 0 bag proves the contrast. It is **not** the
anchored counter I predicted. Recorded as a failed prediction, not relabelled.

### Two corrections this forces

1. **Our pooled-ceiling estimate was too low.** Cell C reaches `aa 0.9997`, above the 0.9509 naive-Bayes
   ceiling I predicted from. A transformer's content pathway beats naive Bayes because real codons are
   correlated. `RESULTS.md` §11 flagged this as a caveat this morning; I predicted from it anyway.
   **Ceilings must be measured with a model-class-matched estimator** — e.g. train the same architecture
   on ATG-ablated windows and read off its accuracy.
2. **The cue-conflict insertion site is part of the assay's definition.** Inserting at `ATG+3` corrupts
   the anchor's own neighbourhood. The effect is small here (anchor 0.039 → 0.155 at ATG+90) but the
   assay should specify the offset, and report the sweep.

### An observation to test, not to assert

A relay is what the ribosome does: it initiates at the start codon and maintains frame by **stepping**,
not by counting. If that holds, the model learned the biologically correct mechanism and our assay
called it a bag.


---

## D33 — The completed necessity 2×2. The anchor is load-bearing in all four cells.

All four on the same footing (`scripts/verdict_2x2.py`, μ = 1, 900 val windows). Cell B at 24k steps
(it grokked at 18k–20k, confirming D31's withdrawal); cell A is `nano_uniform_step8400`.

| cell | shortcut | detector | aa acc | ATG-abl effect (vs decoy ctl) | anchor | content | mechanism |
|---|---|---|---|---|---|---|---|
| **A** | absent | easy | 1.0000 | **+0.6094** (0.0129) | **1.0000** | 0.0000 | anchored counter |
| **B** | absent | hard | 0.9605 | **+0.5794** (0.0135) | 0.3891 | 0.2905 | mixed / partial relay |
| **C** | present | easy | 0.9997 | **+0.3529** (0.0148) | 0.0396 | 0.9131 | relay |
| **D** | present | hard | 0.9914 | **+0.1493** (0.0218) | 0.1084 | 0.8001 | relay |

### Main effects

```
                        anchor share      ATG-ablation effect
A -> B  (detector hard)   1.0000 -> 0.3891    0.6094 -> 0.5794     detector barely touches causal weight
A -> C  (shortcut on)     1.0000 -> 0.0396    0.6094 -> 0.3529
A -> D  (both)            1.0000 -> 0.1084    0.6094 -> 0.1493
```

**The start codon is causally load-bearing in every cell** — ablation effect 0.149–0.609 against a
decoy-Met control of 0.013–0.022 (8×–47× specificity). What changes across the 2×2 is not *whether* the
anchor is used but **how the frame is propagated from it**:

```
counter (A)      count (i - i_ATG) mod 3 from the anchor      anchor 1.00
mixed   (B)      partially counted, partially re-locked       anchor 0.39 / content 0.29
relay   (C, D)   anchor initialises; local structure carries  anchor 0.04-0.11 / content 0.80-0.91
```

**Detector difficulty barely reduces the anchor's causal weight (0.609 → 0.579) while collapsing
counting (1.00 → 0.39).** The shortcut reduces both. A hard-to-find anchor is still *used*; it just
stops being *counted from*.

### What the frozen verdict got wrong, and what it got right

The pre-registered binary — anchor share > 0.50 **and** ablation > 0.10 — labels only cell A as
CIRCUIT FORMED, and prints NO CIRCUIT for three cells in which the anchor is causally necessary at
255 nt. **The binary was underspecified because neither planted twin is a relay** (D32).

It got the important thing right: it is **not** the training accuracy. All four cells score 0.96–1.00
and are mechanistically distinct. Accuracy would have reported four identical successes.

We pre-registered *"continuous values are reported regardless of which side of the threshold they land
on; the threshold only labels the cell."* That clause is doing the work. **The 2-D signature
(ablation profile, conflict share) is the result; the binary is a lossy summary of it.**

### Cell B confirms D31

```
step 18000  aa 0.5571      still on the plateau
step 20000  aa 0.9610      transition
step 24000  aa 0.9735      (predicted ceiling 0.926 -- exceeded, see below)
```

Cut off at 8k, it read as NO CIRCUIT with a half-built mechanism. Given 3× the budget it grokked.
**Cell B is the hardest cell by construction and needs ~2.5× the steps of cell A.**

### Both ceilings we predicted from were lower bounds

Cell B reaches 0.9735 against a Bayes-optimal-Kozak-detector ceiling of 0.926; cell C reaches 0.9997
against a naive-Bayes pooled ceiling of 0.9509. **Both of our ceiling estimators are lower bounds on
what the model class can do**, and we predicted from both. Ceilings must be measured with a
model-class-matched estimator — train the same architecture with the anchor removed and read off its
accuracy.

---

## D34 — The relay is *anchor-corrected pooling*: it pools to the anchor-free ceiling by L2 and spends the anchor at L4

Phase B asked for a 2-D signature scatter. Producing one exposed a measurement error and, in fixing it,
localised the relay by depth.

### The measurement error: a probe at a fixed layer measures whichever variable that layer carries

The first attempt read far-field anchor dependence off a probe at `L2.resid_post` for every model. It
disagreed with the accuracy-based numbers of D33:

```
model                   probe drop @ L2      aa-acc drop (D33)
A · uniform, easy            0.6459               0.6094
C · real CDS, easy           0.0037               0.3529
```

Cell C's L2 probe is at 0.9515, which is the pooled anchor-free ceiling (0.9509) to within noise.
Ablating the ATG cannot move a representation that never contained it. **Fixed-layer probing silently
asks a different question of every model.** The fix — `signatures.peak_anchor_dependence()` — takes the
**max over depth** of (far-field probe drop under ATG ablation) minus (the same for the decoy-Met
control), and reports the layer at which it peaks.

### Per-layer ablation

```
cell C (relay)          intact    ATG-abl    drop      decoy-Met ctl
  L0–L3               0.83–0.95    ~same    +0.000…+0.008    ~0.000
  L4                    0.9999     0.6178   +0.3822         -0.0001
cell A (counter)
  L1                    0.8823     0.3451   +0.5372         -0.0016
```

### The corrected signature

```
model                      peak anchor dep   at layer   anchor share   mechanism
A · uniform, easy               0.6467          L2         1.0000      counter
B · uniform, hard               0.5767          L4         0.3879      mixed
C · real CDS, easy              0.3913          L4         0.0375      relay
D · real CDS, natural           0.1480          L4         0.1014      relay
μ=0 · real CDS, LM              0.0025        (none)       0.0882      bag
```

### The layer probe

```
model                          embed     L0      L1      L2      L3      L4      L5
A · uniform (counter)         0.3437  0.3876  0.8853  0.9998  1.0000  0.9998  0.9998
B · uniform, hard (mixed)     0.3201  0.3495  0.3550  0.3967  0.6007  0.7743  0.7955
C · real CDS (relay)          0.4071  0.8225  0.8852  0.9515  0.9497  0.9997  0.9411
D · real CDS (relay)          0.4021  0.8894  0.9177  0.9584  0.9811  0.9927  0.9700
μ=0 · real CDS (bag)          0.4019  0.8172  0.8172  0.8195  0.8196  0.8178  0.8174
untrained (random init)       0.3428  0.3433  0.3455  0.3458  0.3435  0.3451  0.3432
```

Read the rows against the anchor-free ceiling of **0.9509**:

- **bag (μ=0)** — pools to 0.8172 at L0 and never moves again. Five further layers buy 0.0002. It is
  not even at the pooled ceiling; the LM objective does not pay for the last 0.13.
- **counter (A)** — starts at chance (uniform codons carry no local phase cue, by construction) and
  *builds*: L0 0.388 → L1 0.885 → L2 1.000. There is nothing to pool. The anchor is the only route,
  and it enters at L1/L2.
- **relay (C, D)** — pools like the bag through L0–L2, arrives at 0.9515 ≈ the anchor-free ceiling, sits
  there through L3, then **jumps to 0.9997 at L4**. Ablating the ATG costs 0.382 of *that* layer.

### The name is wrong; the mechanism is not a relay

D32 named it "relay" from a behavioural signature: the anchor is causally load-bearing at 255 nt, yet
under cue conflict the model follows local content. The depth profile says what it actually does:

> **Pool the anchor-free evidence to its ceiling, then spend the anchor to buy the residual.**

It is not propagating frame *from* the anchor (that is the counter). It is computing frame *without* the
anchor, hitting the imperfection that pooling leaves — the ~5% of windows the stop-free-frame scan
cannot resolve — and using the anchor as a **correction term** on exactly those. Hence the cue-conflict
result: on the 95% of windows local content already decides, moving the anchor changes nothing, so
"content share" ≈ 0.91. And hence the ablation result: on the residual, the anchor is all there is.

**Anchor-corrected pooling** is the accurate name. "Relay" is retained in the code and figures as the
enum value to keep the 2×2 comparable, and flagged here as a misnomer.

### Why this matters for the sufficiency criterion

D30 said a circuit forms iff no pooled statistic is sufficient, and that *imperfection*, not weakness,
is the threshold. D34 shows **where the imperfection is paid for**: not by rebuilding the computation,
but by bolting a late correction onto a pooled estimate. The model does the cheap thing for 95% of the
distribution and the expensive thing for 5% — and it does them in *different layers*. A tracer that
averages over the data distribution will see 95% bag and miss the circuit entirely.

**This predicts the Phase C hypothesis.** If L4's correction is a per-position re-sync to the anchor,
anchor dependence should be flat in distance (the correction is a lookup). If it is a global count,
dependence should decay. And the correction should be **concentrated on the windows the pooled scan
gets wrong** — that is a per-window prediction, and it is falsifiable.

### Status

`SETTLED` (measurement), one seed per cell. Error bars are Phase D. The claim "cell C's L4 is where the
anchor enters" rests on an ablation whose decoy-Met control is −0.0001, i.e. 3800× specificity.
