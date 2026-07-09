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
