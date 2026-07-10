# Foundations of circuit discovery

An introduction to the mathematics and literature of mechanistic interpretability, written for a
first-year CS/ML graduate student who knows linear algebra and basic deep learning but has never
opened a model to see how it computes. It is the background you need to understand what this repo does
and why. For the step-by-step procedure, read `CIRCUIT_DISCOVERY_HOW_TO.md`; for the genomics, read
`BIOLOGY_genomic_computations.md`. Every named method here maps to an entry in `VERIFIED_REFERENCES.md`.

---

## 1 · What is a circuit?

A **circuit** is a subgraph of a network's computation that implements a human-understandable
algorithm. The transformer-circuits program (Elhage et al. 2021; Olah et al. 2020, *Zoom In*) makes
this precise by choosing the right coordinate system: the **residual stream**.

A decoder-only transformer maintains, at each token position $t$, a vector $x_t \in \mathbb{R}^{d}$.
Every layer *reads* from this vector and *adds* its output back:

$$
x^{(\ell+1)}_t = x^{(\ell)}_t + \text{Attn}^{(\ell)}(x^{(\ell)})_t + \text{MLP}^{(\ell)}(x^{(\ell)}_t).
$$

Because every contribution is **added**, the residual stream is a shared communication channel: layer 5
can read something layer 1 wrote, untouched, if no intervening layer overwrote it. This linearity is
what makes circuits tractable — you can ask "which component wrote the information this later component
reads?" and get a well-posed answer.

**Attention as a bilinear form.** A single attention head is parameterised by four matrices
$W_Q, W_K, W_V, W_O$. It computes, from query position $p$ and key position $q$, a score

$$
s_{p,q} = \frac{(W_Q x_p)^\top (W_K x_q)}{\sqrt{d_\text{head}}}
        = \frac{x_p^\top W_Q^\top W_K\, x_q}{\sqrt{d_\text{head}}},
$$

which is a **bilinear form** on the residual stream with interaction matrix $W_{QK} = W_Q^\top W_K$.
The scores are turned into a distribution and used to mix values:

$$
\alpha_{p,\cdot} = \mathrm{softmax}_q(s_{p,\cdot}), \qquad
\text{head}(x)_p = W_O \sum_q \alpha_{p,q}\, W_V x_q .
$$

This exposes the program's two independent questions, the **QK/OV decomposition** (Elhage et al. 2021):

- **QK — "where to attend."** $W_{QK}$ alone determines the attention pattern $\alpha$. It is the
  circuit's routing decision: which earlier position does this position look at?
- **OV — "what to move."** $W_{OV} = W_O W_V$ determines what is copied from the attended position into
  the residual stream, independent of where attention landed.

A head is therefore two separable pieces: a *decision* about where to look and an *action* about what
to bring back. Much of interpretability is recovering these two pieces for the heads that matter and
discarding the ones that do not. MLPs, the other component, act position-wise and are where most
non-linear feature computation happens; we return to them in §2.

---

## 2 · Features, superposition, and dictionary learning

If circuits are the algorithm, **features** are the variables the algorithm operates on. The naive hope
was that a single neuron equals a single feature. It does not. Networks pack far more features than they
have neurons by placing them in **superposition** — as directions in activation space that are not
axis-aligned and overlap (Elhage et al. 2021; Bricken et al. 2023). A neuron then fires for several
unrelated features (it is *polysemantic*), and reading the model one neuron at a time fails.

**Sparse autoencoders (SAEs)** attempt to undo superposition by learning an overcomplete dictionary of
feature directions (Cunningham et al. 2023; Bricken et al. 2023). Given an activation $x \in
\mathbb{R}^{d}$, an SAE learns an encoder into a much larger sparse code $f \in \mathbb{R}^{m}$
($m \gg d$) and a decoder $W_\text{dec}$ that reconstructs $x$:

$$
f = \mathrm{ReLU}(W_\text{enc}(x - b) ),\qquad \hat x = W_\text{dec} f + b,
$$

trained on a reconstruction-plus-sparsity objective

$$
\mathcal{L}(x) = \underbrace{\lVert x - \hat x \rVert_2^2}_{\text{reconstruction}}
              + \lambda \underbrace{\lVert f \rVert_1}_{\text{sparsity}} .
$$

The $L_1$ penalty pushes the code to explain each activation with a few active dictionary atoms; the
columns of $W_\text{dec}$ are the candidate **monosemantic** features. A known weakness is that the
$L_1$ term shrinks the very activations it selects, biasing feature magnitudes; **gated SAEs**
(Rajamanoharan et al. 2024) separate the *which-features-are-on* decision from the *how-much* estimate
to reduce this bias. Dictionary methods matter here because a circuit stated in terms of interpretable
features is far more falsifiable than one stated in raw neuron indices — and this repo uses a learned
dictionary to separate a compositional feature from a lone shortcut.

---

## 3 · Causal interpretability: patching, ablation, and the saturation trap

Correlation between a component and an output is cheap and misleading. Causal methods intervene on the
computation and measure what changes. The core technique is **activation patching**, also called causal
mediation analysis (Vig et al. 2020; Meng et al. 2022, *ROME*; Heimersheim & Nanda 2024).

Define two inputs that differ in the property of interest: a **clean** run $x^{\text{clean}}$ and a
**corrupted** run $x^{\text{corr}}$. Run both, caching every internal activation. Now re-run the clean
input but **overwrite** one component's activation $a$ with its value from the corrupted run:

$$
\text{patch}: \quad a \leftarrow a^{\text{corr}} \ \text{while all else stays clean.}
$$

The **patching metric** is the change this induces in a chosen readout $M$ — a logit difference between
two answer tokens, or, as this repo insists, the change in **loss**:

$$
\Delta M = M\big(\text{clean run with } a \text{ patched}\big) - M\big(\text{clean run}\big).
$$

A large $|\Delta M|$ means the component carries information the output depends on. **Ablation** is the
special case where $a$ is replaced not with a corrupted value but with zero (zero-ablation) or a
resampled/mean value (a cleaner, on-distribution baseline).

**Why loss, not accuracy.** Once a model solves a task, accuracy is pinned at 1.0 and cannot report the
size of an effect: you can remove a component and watch loss jump by 8 nats while accuracy does not
move at all, because the remaining margin is enormous. A saturated metric is blind precisely where the
effect is largest. This repo treats a saturated readout as a bug and ships a guard against it.

**Why gradients fail on a confident head — and the Integrated Gradients fix.** The cheap way to rank
components is to read the gradient of the metric with respect to an activation. But consider a head that
has *learned* to find a marker: its attention score on the marker key is large and elsewhere near zero,
so its softmax is **saturated**. In the saturated regime $\partial\, \mathrm{softmax}/\partial s
\approx 0$, so $\partial M/\partial s \approx 0$: the gradient reports the edge is irrelevant *because*
the head is so certain. Measured on a planted model in this repo, naive pattern-gradient attribution is
identically zero on every edge of exactly the marker-finding head that ablation shows is essential.

Integrated Gradients (Sundararajan et al. 2017) repairs this by integrating the gradient along a path
from a content-free baseline $b$ to the real input $x$:

$$
\text{IG}_i(x) = (x_i - b_i)\int_{0}^{1}
\frac{\partial M\big(b + \alpha (x - b)\big)}{\partial x_i}\, d\alpha .
$$

Integrating traverses the unsaturated part of the curve, so credit is assigned even when the endpoint
gradient has vanished. Applying this idea to the **attention scores** — integrating $\partial M/\partial
s$ from a content-free baseline pattern (the head's positional bias alone) to the real pattern — is the
EAP-IG style of edge attribution, and it is how this repo attributes the QK ("where to look") circuit
that plain gradients delete. The reading-frame circuit is anchored by exactly such an edge: locating
the ATG *is* the decide-where-to-look step, so it cannot be traced without this instrument.

---

## 4 · Automated circuit discovery, and why attribution only ranks

Doing patching by hand over every component and edge does not scale. Two lines of work automate it.

**ACDC** (Conmy et al. 2023) searches the computational graph directly: starting from the full graph, it
prunes edges whose removal (by patching) barely changes the metric, leaving a minimal subgraph that
still performs the task. It is faithful but expensive — each candidate edge costs a forward pass.

**Attribution patching / EAP** (Syed et al. 2023) replaces the per-edge forward passes with a
first-order Taylor estimate of each edge's patching effect from a single backward pass, making
whole-graph attribution cheap. This is the same approximation that Integrated Gradients corrects for
saturation, which is why the integrated variant (EAP-IG) is preferred where attention patterns saturate.

The operational lesson, which this repo states as a rule, is:

> **Attribution ranks; intervention decides.**

Attribution methods — attention magnitude, gradients, attribution patching, even ACDC's cheap passes —
produce a *ranked list of suspects*. They are approximate and can be confidently wrong. Only an actual
intervention (ablate the component, measure $\Delta$loss) *decides* whether a suspect is load-bearing.
In this repo's reading-frame model, the head that attends the start codon **most** (L5H2) is causally
**inert** — ablating it changes the loss by essentially zero — while the essential head (L1H3) ranks
lower in attention but moves the loss by roughly $+8.65$ when removed. If you stop at attribution, you
name the wrong circuit.

**The redundancy blind spot.** Intervention has its own failure mode. If a computation is carried
**redundantly** by several components, removing any one of them changes almost nothing, and single-edge
ablation will call every member inert. In this repo's copy circuit the source-following heads are
partially redundant: the sum of the individual single-head ablation effects is roughly $6.1$ while
ablating them jointly costs only about $1.4$. No single head is necessary, so ablation *understates* the
circuit. Detecting a redundant circuit requires a test that does not rely on removal — which is the role
of cue conflict (see the HOW_TO, Pillar 5).

---

## 5 · Induction heads: the canonical circuit

The best-understood circuit in language models is the **induction head** (Olsson et al. 2022), the
mechanism behind much of in-context learning. It completes the pattern $[A][B]\dots[A] \to [B]$: having
seen $A$ followed by $B$ earlier, when $A$ recurs the model predicts $B$. It is implemented by two heads
across two layers:

1. a **previous-token head** in an earlier layer writes, at each position, information about the token
   that preceded it; then
2. an **induction head** in a later layer uses a QK circuit that matches the current token against those
   "previous-token" annotations, attends **back to the position after the earlier occurrence of the
   current token**, and its OV circuit copies that token forward.

This two-layer motif — previous-token setup below, match-and-copy above — is exactly the structure this
repo finds in its **repeat-copy** circuit: an L0 bank of setup heads (high $\Delta$loss, essentially no
source-attention) feeding L1 heads that attend back to the copy source (source-attention $\approx
0.65$–$0.73$) and move it to the output. The genomic copy circuit is an induction head in a DNA model.

---

## 6 · Grokking: circuit formation as a phase transition

Circuits do not fade in smoothly with the loss. **Grokking** (Power et al. 2022) is the phenomenon where
a model trains to near-perfect *training* performance while *generalisation* stays at chance for a long
time, then abruptly generalises much later. Nanda et al. (2023) opened up a grokking model and showed
the transition corresponds to a specific algorithmic circuit crystallising, and defined **progress
measures** — quantities computed from the weights that rise smoothly as the circuit forms, even while
accuracy is flat.

This matters for discovery because the *step at which a circuit appears* is itself evidence that a
discrete algorithm exists to be found, rather than a smooth interpolation. In this repo the reading-frame
anchor-relative modulo-3 circuit (whether it is a sequential counter or an anchor/position-residue lookup is
unresolved — see `docs/studies/reading_frame/COUNTER_VS_LOOKUP.md`) groks abruptly (around training step
3000–3250: accuracy sits near the shortcut floor, then jumps), while the copy circuit forms more gradually (accuracy crossing 0.5 near step 3500 and 0.9 near
step 4750). Watching a mechanism-sensitive metric jump at a training step — while accuracy is
uninformative — is the temporal signature of a circuit and the reason the study notebooks plot the
grokking curve before anything else.

---

## 7 · Shortcut learning and the shortcut-gap hypothesis

The reason a circuit might *not* form is **shortcut learning** (Geirhos et al. 2020): models prefer the
cheapest decision rule that fits the training distribution, and that rule is often a spurious statistic
rather than the intended algorithm. A model that can hit the target accuracy with a pooled statistic has
no gradient pressure to build the relational circuit.

This repo turns that observation into a sharp, testable criterion — the **shortcut-gap hypothesis**. Let the
target be $Y$ and let $S(x)$ be the best statistic in a **pre-specified (registered) family pooled over the
context window** (a bag-of-tokens count, a codon-composition estimate). Write the accuracy ceiling of that pooled statistic
as $\mathrm{acc}^\star_{\text{pool}} = \max_{g} \Pr[\,g(S(x)) = Y\,]$. Then:

$$
\Pr[\textbf{a relational circuit forms}] \ \text{rises as} \ \big(\text{accuracy demanded by the objective} \;-\; \mathrm{acc}^\star_{\text{pool}}\big) \ \text{grows}
$$

— a hypothesis, not a settled law: circuit formation also depends on example frequency, architecture,
capacity, regularization, initialization, and training budget. The threshold is **imperfection, not
weakness**: a shortcut worth 96.5% does not displace the relational (anchor-relative modulo-3) circuit when
the task demands 100%. Two experimental consequences follow directly. First, the measured pooled ceiling
$\mathrm{acc}^\star_{\text{pool}}$ is the *control* every circuit claim must beat — not a random baseline,
not a guess. Second, you can *dial the law*: hold the task fixed and vary how much frame information the
local content leaks — measured as the mutual information $I(\text{nt};\text{phase})$ between a
nucleotide and the reading frame — and watch the mechanism slide from an anchor-relative modulo-3 circuit
(weak shortcut) to a pooled bag (strong shortcut), even though accuracy stays near 1.0 across the whole sweep.
Accuracy tells you nothing; only the mechanism moves. This is the conceptual spine of the whole program.

---

## 8 · A worked example, by hand: induction head vs bag-of-tokens

Take a toy task: sequences of tokens with one repeated block, and the model must predict the token after
the second occurrence of a cue token $A$. Two hypotheses:

- **Induction circuit** — the model looks back to the token that followed $A$ last time and copies it.
- **Bag-of-tokens shortcut** — the model predicts the token most frequent in the window so far, ignoring
  position.

On clean data both can score well, so accuracy cannot separate them. Do a **cue-conflict patch**
instead. Construct a corrupted sequence identical to the clean one except that the token following the
*first* occurrence of $A$ is changed from $B$ to $B'$. Now the two hypotheses make opposite predictions
at the query:

| | predicts at the query after 2nd $A$ |
|---|---|
| induction (copies from source) | $B'$ (follows the edited source) |
| bag-of-tokens (window statistics) | $B$ (still the frequent token) |

Run the model and read the probability the metric assigns to $B'$ versus $B$. Define the metric as the
logit difference $M = \text{logit}(B') - \text{logit}(B)$. An induction circuit gives $M \gg 0$; a bag
gives $M \ll 0$. This single measurement decides the mechanism **without removing anything**, which is
why it survives redundancy (§4).

To localise the circuit, patch further. Cache the corrupted run; then, on the clean run, patch just the
attention pattern of a candidate induction head at the query position and measure $\Delta M$. If that one
head's pattern carries the effect, $\Delta M$ will swing most of the way from the clean to the corrupted
value. To confirm the two-layer motif, ablate a candidate previous-token head in the earlier layer and
check that the induction head's source-attention *collapses* — a two-hop dependency: the upper head can
only match on annotations the lower head wrote. That is the induction head of Olsson et al. (2022),
reconstructed from three measurements: a conflict metric, an attention patch, and an upstream ablation.
It is the same sequence of moves the repo runs on the genomic copy circuit.

---

## 9 · Glossary

- **Residual stream** — the per-position vector every layer reads from and adds to; the shared channel
  that makes circuits analysable.
- **QK circuit** — the query-key interaction $W_Q^\top W_K$ that sets *where* a head attends.
- **OV circuit** — the output-value map $W_O W_V$ that sets *what* a head moves from the attended
  position.
- **Feature / superposition** — an interpretable direction in activation space; superposition is packing
  more features than dimensions, causing polysemantic neurons.
- **SAE (sparse autoencoder)** — a learned overcomplete dictionary trained with reconstruction + $L_1$
  sparsity to recover monosemantic features.
- **Activation patching** — overwriting one component's activation with its value from another run and
  measuring the change in a metric.
- **Ablation** — patching a component to zero (zero-ablation) or to a resampled/mean baseline; measures
  necessity.
- **Attribution vs intervention** — attribution ranks suspects cheaply and approximately; intervention
  decides which are load-bearing.
- **Induction head** — the two-layer previous-token → match-and-copy circuit behind in-context copying.
- **Grokking** — delayed generalisation; a circuit forming as an abrupt phase transition after training
  accuracy has saturated.
- **Shortcut / shortcut-gap hypothesis** — a pooled statistic that suffices for the target; a circuit is
  increasingly likely to form as the demanded accuracy exceeds the best registered pooled shortcut's ceiling.

---

## References

Full identifiers and verification status are in `VERIFIED_REFERENCES.md`. Titles confirmed by an
OpenAlex lookup at build time except the two marked `[not machine-verified]` (Anthropic
transformer-circuits.pub articles not indexed in OpenAlex — cite by their canonical URLs and confirm
before formal publication).

- **Bricken et al. (2023)**. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Anthropic, transformer-circuits.pub. `[not machine-verified]`
- **Conmy et al. (2023)**. *Towards Automated Circuit Discovery for Mechanistic Interpretability*. NeurIPS / arXiv. https://doi.org/10.48550/arxiv.2304.14997
- **Cunningham et al. (2023)**. *Sparse Autoencoders Find Highly Interpretable Features in Language Models*. arXiv. https://doi.org/10.48550/arxiv.2309.08600
- **Elhage et al. (2021)**. *A Mathematical Framework for Transformer Circuits*. Anthropic, transformer-circuits.pub. `[not machine-verified]`
- **Geirhos et al. (2020)**. *Shortcut learning in deep neural networks*. Nature Machine Intelligence. https://doi.org/10.1038/s42256-020-00257-z
- **Hanna et al. (2023)**. *How does GPT-2 compute greater-than?*. NeurIPS / arXiv. https://doi.org/10.48550/arxiv.2305.00586
- **Heimersheim & Nanda (2024)**. *How to use and interpret activation patching*. arXiv. https://doi.org/10.48550/arxiv.2404.15255
- **Meng et al. (2022)**. *Locating and Editing Factual Associations in GPT (ROME)*. NeurIPS / arXiv. https://doi.org/10.48550/arxiv.2202.05262
- **Nanda et al. (2023)**. *Progress measures for grokking via mechanistic interpretability*. ICLR / arXiv. https://doi.org/10.48550/arxiv.2301.05217
- **Olah et al. (2020)**. *Zoom In: An Introduction to Circuits*. Distill. https://doi.org/10.23915/distill.00024.001
- **Olsson et al. (2022)**. *In-context Learning and Induction Heads*. Anthropic / arXiv. https://doi.org/10.48550/arxiv.2209.11895
- **Power et al. (2022)**. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*. arXiv. https://doi.org/10.48550/arxiv.2201.02177
- **Rajamanoharan et al. (2024)**. *Improving Dictionary Learning with Gated Sparse Autoencoders*. arXiv. https://doi.org/10.48550/arxiv.2404.16014
- **Sundararajan et al. (2017)**. *Axiomatic Attribution for Deep Networks (Integrated Gradients)*. ICML. https://doi.org/10.48550/arxiv.1703.01365
- **Syed et al. (2023/2024)**. *Attribution Patching Outperforms Automated Circuit Discovery*. BlackboxNLP / arXiv. https://doi.org/10.18653/v1/2024.blackboxnlp-1.25
- **Vig et al. (2020)**. *Investigating Gender Bias in Language Models Using Causal Mediation Analysis*. NeurIPS.
