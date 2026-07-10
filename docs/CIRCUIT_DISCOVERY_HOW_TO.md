# How to discover a circuit in a genomic foundation model

A step-by-step, implementable guide. Given a **genomic computation** (e.g. reading frame), a **model**
(a trained GFM checkpoint), a **dataset** (sequences with known labels), and a **task**, this document
takes you from "I suspect there is a circuit here" to "here is the circuit, wired across depth, and
here is how I know it is real and not a shortcut." It is written so that a computer-science student who
has never touched interpretability or genomics can follow it end to end using the tools in this repo,
and then apply the same recipe to a circuit of their own — in a GFM or beyond.

For the mathematical and literature background, read `docs/FOUNDATIONS_circuit_discovery.md` first.
For the biology of the computations themselves, read `docs/BIOLOGY_genomic_computations.md`. This
document is the *procedure*; those two are the *why*.

---

## 0 · The one idea you must hold onto

A model is far more likely to build a **circuit** — a reusable, relational algorithm — when it has no
cheaper way to hit the accuracy the task demands. If some statistic in a pre-specified pooled family is a
*sufficient statistic* for the target, the model tends to compute that statistic (a **shortcut**) and
usually does not build the algorithm. This is the **shortcut-gap hypothesis**, and it is the spine of
everything below:

> A relational circuit becomes increasingly likely to form as the accuracy demanded by the objective
> exceeds the ceiling of the best statistic in a pre-specified pooled family — while also depending on
> example frequency, architecture, capacity, regularization, initialization, and training budget.

Two consequences drive the whole method:

1. **You must know the shortcut before you look for the circuit.** The measured ceiling of the best
   pooled statistic is your control (Pillar 4). Without it, you cannot tell a circuit from a lookup.
2. **The threshold is imperfection, not weakness.** A 96.5% shortcut does not displace the relational
   (anchor-relative modulo-3) circuit when the task demands 100%. Do not assume a strong shortcut means "no circuit"; it means "no circuit
   *unless* the last few points of accuracy are demanded."

Everything else is instrumentation for measuring, honestly, which of the two the model actually did.

---

## 1 · The five pillars (the protocol in one table)

The notebook `notebooks/TEMPLATE_study_a_circuit.ipynb` runs this protocol top to bottom. Each pillar
is a question and the specific trap it guards against.

| # | Pillar | The question | The trap it guards against |
|---|--------|--------------|-----------------------------|
| 1 | **Discovery** | Which components look responsible? | Attention *ranks*; it does not *decide*. The head that attends the anchor most may be causally inert. |
| 2 | **Testing** | Is this a mechanism or a shortcut? | A task a pooled statistic already solves needs no circuit — measure the demand vs the ceiling. |
| 3 | **Ablation** | What breaks when you remove it? | Accuracy saturates and lies. Read Δ**loss**, not Δaccuracy. |
| 4 | **Baselines** | What could a bounded window do? | The measured pooled ceiling is the control — not a guess, not a random baseline. |
| 5 | **Conflict** | Which cue wins when they disagree? | Ablation cannot see a *redundant* circuit. Only cue conflict can. |

Then a sixth step, reverse engineering (§8), reconstructs *how* the circuit is wired across layers.

---

## 2 · Set up: what you need before you start

You need four things, and you should be able to name each one precisely before writing any code.

- **A computation.** State it as an input→output map with a *hard part*. "Reading frame: given a 256-nt
  window with one ATG, output the amino acid at each position; the hard part is that the amino acid
  depends on the reading frame, which is the offset mod 3 from the ATG."
- **A model.** A trained checkpoint. In this repo: `lab.list_checkpoints()` shows the families on disk.
  The two worked examples are `nano_uniform` (reading-frame anchor-relative modulo-3 circuit, amino-acid
  head; a controlled transformer trained from scratch, not a pretrained genomic foundation model) and
  `repeat_copy_s0` (induction/copy circuit, causal LM head).
- **A dataset.** Sequences with the labels the task needs, plus — critically — the ability to make
  *counterfactual* versions (move the anchor, scramble the source). The repo's loaders
  (`lab.load_uniform_windows`, `lab.load_repeat_windows`, `lab.load_cell_windows`) return these.
- **A task metric that does not saturate.** Prefer per-position **cross-entropy loss**. Accuracy is
  fine for reporting but useless as an arbiter once it hits 1.0 (see Pillar 3).

Environment: the repo is `uv`-managed. `uv sync --extra lab --extra dev`, then work in the `gfmcd-uv`
kernel. Every method used below lives in `gfmcd.lab` (`import gfmcd.lab as lab; from gfmcd.lab import
viz`).

---

## 3 · Load the model and state the problem (template §0)

```python
import gfmcd.lab as lab
from gfmcd.lab import viz

m = lab.load_checkpoint("nano_uniform")   # family stem -> latest step, or pass step=1200
print(m)   # tells you: aa-head vs causal, layers x heads, device
```

Write the **problem statement** down before anything else: input tokens, output space, the hard part,
the plausible shortcut, and the circuit you hypothesize. The template prints a version of this for the
loaded model. Naming the shortcut now is not optional — it is Pillar 4's control.

Then look at **training dynamics**. A circuit does not fade in with the loss; it *groks* — accuracy
sits near the shortcut floor for thousands of steps while loss drifts down, then jumps. The step of
that jump is when the mechanism snapped into place.

```python
history = lab.load_history("nano_uniform")
trans = lab.grok_transition(history, "aa_acc")
viz.plot_grokking(history, "aa_acc", "val", transition=trans)
```

In this repo the reading-frame anchor-relative modulo-3 circuit groks abruptly around step 3000–3250; the copy circuit forms
more gradually (accuracy ≥ 0.5 by ~3500, ≥ 0.9 by ~4750). Seeing the transition is your first evidence
that there is a discrete algorithm to find, not a smooth interpolation.

---

## 4 · Pillar 1 — Discovery: rank the suspects (do not trust the ranking)

Survey attention from downstream positions to the circuit's **anchor** (the start codon for the frame
phase circuit; the copy source for induction). This *ranks* candidate heads.

```python
from gfmcd.signatures import start_index
tok, aa, phase = lab.load_uniform_windows(n=256, seed=555)
start = start_index(phase)
attn = lab.attention_survey(m, tok, start)   # [n_layers, n_heads]
viz.plot_attention_survey(attn)
```

**The single most important warning in this repo:** the ranking is a list of *suspects*, not a verdict.
On `nano_uniform`, the head that attends the start codon *most* is L5H2 — and ablating it changes the
loss by essentially zero. The essential head, L1H3, ranks lower in attention. Attribution (attention,
gradients, patching scores) **ranks**; intervention **decides**. Never end at Pillar 1.

A second, deeper reason attention alone misleads: a head that has *learned* to find a marker is
confident, so its softmax is saturated and gradient-based attribution through the pattern reads
identically zero — "irrelevant because it is so relevant." (This is why the repo's QK attribution
integrates the gradient along a path, EAP-IG style; see FOUNDATIONS §3.) Take the ranking, move on.

---

## 5 · Pillars 2 & 4 — Testing and Baselines: is there anything to explain?

These two are one measurement: compare the accuracy the task **demands** to the ceiling of the best
**pooled statistic**. If the model does not beat the ceiling, there is no circuit to find — it *is* the
shortcut.

```python
report = lab.ceiling_report(m, tok, phase, demanded=1.0)
print(report.pooled_oracle_acc, report.model_phase_acc, report.circuit_predicted)
viz.plot_ceiling(report)
```

For reading frame, the "best pooled statistic" is a codon-composition naive-Bayes oracle that never
sees the start codon (`lab.pooled_ceiling`, implemented in `oracle.py`). On real human CDS it reaches
~0.95 at a 240-nt window; the uniform-codon task drives it down near 1/3. The model achieves ~1.0
phase accuracy. Because the *demanded* accuracy (1.0) exceeds the *ceiling* (< 1.0), a circuit is
predicted — and only then is the rest of the analysis meaningful.

This is where the shortcut-gap hypothesis becomes an experiment you can run rather than a slogan. If you want
to *see* the law operate, sweep the shortcut strength (reading-frame family): each λ arm trains on data
with a different `I(nt; phase)` — the mutual information between local content and the frame. Accuracy
stays ~1.0 across the sweep, so accuracy tells you nothing; only the *mechanism* moves, sliding from an
anchor-relative modulo-3 circuit to a pooled bag as the shortcut strengthens.

```python
sweep = lab.lambda_sweep(["nano_sense", "nano_lam0.5", "nano_lam1", "nano_lam2"])
viz.plot_lambda_sweep(sweep)
```

---

## 6 · Pillar 3 — Ablation: what is actually load-bearing?

Now intervene. Zero each head in turn and measure the change in **loss** (not accuracy).

```python
table = lab.head_ablation_table(m, tok, aa, target_pos=start)   # sorted by Δloss
viz.plot_head_effects(table)     # twin axis: Δloss vs attention — they disagree, and that is the point
```

On `nano_uniform` this returns L1H3 with Δloss ≈ +8 to +9 while L1H2 — which attends the start codon
with *higher* specificity — sits at Δloss ≈ +0.000001. That contrast is the whole lesson: **the plot
puts attention and Δloss on the same axes so you can see them disagree.**

Three hard rules for this pillar:

1. **Read loss, not accuracy.** Where the effect is largest, accuracy is often pinned at 1.0 and
   reports nothing. The repo ships a `saturation_guard` that warns when any metric you rely on is
   within tolerance of its ceiling or floor. Heed it.
2. **Zero-ablation is off-manifold.** Zeroing a head pushes the residual stream to a place the model
   never sees in training, so some of the "effect" is distribution shock, not the head's computation.
   Cross-check with **resample ablation** (`lab.resample_heads`), which replaces the head's output with
   its value on a shuffled batch — an on-manifold baseline. A large zero-vs-resample gap flags that you
   over-credited the head.
3. **Ablation cannot see a redundant circuit.** If five heads each do the same job, removing any one
   changes almost nothing, and single-head ablation will call them all inert. This is not hypothetical:

```python
red = lab.redundancy_test(m, tok, labels, heads)   # or copy_redundancy for the copy family
print(red.sum_of_singles, red.subset_delta, red.interpretation)
```

For the copy circuit, the sum of single-head ablation effects (~6.1) vastly exceeds the joint effect
(~1.4) — the five source heads are partially redundant. No single head is necessary, so ablation alone
*understates* the circuit. Which is exactly why you need Pillar 5.

---

## 7 · Pillar 5 — Conflict: the decisive test

Put the two cues in **conflict** and measure which one the model follows at range. This is the only
test that detects a redundant circuit, because it does not rely on removing anything — it asks the
intact model to choose.

- **Reading frame:** insert one nucleotide just after the ATG. Now the anchor says "frame X" and the
  local codon content says "frame X+1". A **counter** follows the anchor at every distance; a **bag**
  crosses over to the content frame beyond ~30 nt; a **relay** starts from the anchor but drifts to
  local structure.
- **Repeat copy:** edit one copied token so the distal source and the locally-plausible continuation
  disagree. A copy circuit follows the distal source; a local n-gram model follows the continuation.

```python
# reading frame
sig = lab.mechanism_signature(m, tok, phase)
print(sig.anchor_share, sig.verdict)      # anchor_share -> 1 = counter, -> 0 = bag
viz.plot_mechanism_map(sig)

# repeat copy
cm = lab.copy_metrics(m, batch)
print(cm["conflict_follows_distal"], cm["conflict_follows_local"])
```

In this repo the uniform-codon model shows anchor share ≈ 1.0 (a clean anchor-relative modulo-3 mechanism); the copy model follows
the distal source ~0.96 of the time versus ~0.00 for the local continuation. For the reading-frame
family the repo places the single verdict in two richer contexts: the λ sweep (§5) and the **2×2 grid**
(`lab.verdict_2x2` + `viz.plot_2x2_grid`), which crosses shortcut presence with detector difficulty to
show *which conditions force the circuit*. The 2×2 applies a verdict that was **frozen before the cells
were run** (anchor share > 0.50 *and* start-codon ablation beats a decoy-Met control by > 0.10) — a
preregistration discipline that keeps you from reading the threshold off the data.

**A caution the repo paid for:** never read a mechanism verdict off a fixed layer. The anchor enters
the residual stream at different depths in different data regimes (L2 for uniform codons, L4 for real
CDS), and reading both 2×2 axes at a fixed L2 probe once produced retracted claims. Use the output-level
readout and `peak_anchor_dependence`, not a hard-coded layer. (See `docs/RESULTS.md §10` and `TODO.md`
item 2.1.)

---

## 8 · Reverse-engineer the structure (template §6)

The five pillars establish *that* a circuit exists, which parts are essential, and *what* mechanism it
implements. Reverse engineering answers **how it is wired across depth**. The method is the same for
any circuit:

1. **Where does the answer become available?** Probe each layer's residual stream for the target.

   ```python
   decode = lab.phase_decodability_by_layer(m, tok, phase)   # linear-probe acc per layer
   ```

   The layer where the target becomes decodable is where the circuit finished computing it. Below that
   is construction; above it is propagation. On `nano_uniform` the frame goes from ~0.41 decodable at
   L0 to ~0.91 at L1 to ~1.0 at L2 — so the phase (anchor-relative modulo-3) is built in layers 0–1 and propagated thereafter.

2. **Which heads move it there?** Cross the ablation table (Pillar 3) with the attention survey
   (Pillar 1). An essential head that writes *into* the layer where the answer appears is a
   construction head; one *below* is setup; heads above mostly propagate.

3. **What is each head's job?** Setup heads have high Δloss but do **not** attend the anchor (they build
   local features — e.g. the L0 bank on `nano_uniform`, or the previous-token heads in the copy
   circuit). The anchor/induction head attends the anchor *and* has high Δloss (L1H3 for the phase circuit;
   the L1 source-following heads for copy). This distinction — "reads composition" vs "locates the
   anchor" — is the heart of the mechanism.

4. **Draw it, then falsify it.** The flow diagram (`viz.plot_circuit_flow`) is a hypothesis, and every
   arrow is an ablation you can run:

   - *Reading frame* — if L1H3 "sets the phase", ablating it must collapse decodability at **L2**, not
     L1. Re-run step 1 inside `with lab.ablate_heads(m.model, (1,3)):` and check.
   - *Repeat copy* — the induction motif is two-hop: L0 previous-token heads → L1 induction heads.
     Ablating an L0 head should blunt the L1 heads' *source-attention* (a two-hop dependency), while
     ablating an L1 head removes the copy directly. Test the hop.

This falsification loop is what the sandbox (`notebooks/sandbox_reverse_engineering.ipynb`) is for:
read a number, form a wiring guess, break the wire, confirm the number moved where the guess predicted.

---

## 9 · The reverse-engineering loop, distilled

```
state the computation, the shortcut, and the hypothesized circuit
        │
        ▼
grok curve ─ is there a discrete transition to find?          (template §0)
        │
        ▼
attention survey ─ rank suspects (do NOT trust the ranking)    (pillar 1)
        │
        ▼
demanded vs pooled ceiling ─ is there anything to explain?     (pillars 2 & 4)
        │  (if the model doesn't beat the ceiling: it IS the shortcut, stop)
        ▼
ablation Δloss ─ what is load-bearing? check redundancy        (pillar 3)
        │
        ▼
cue conflict ─ which cue wins? (sees redundant circuits)       (pillar 5)
        │
        ▼
decodability-by-layer + flow diagram ─ how is it wired?        (reverse engineering §8)
        │
        ▼
falsify each arrow in the sandbox ─ ablate, confirm the hop    (sandbox)
```

---

## 10 · Applying this to a new circuit (in a GFM or beyond)

The recipe is model- and domain-agnostic. To port it:

1. **Name the computation** as an input→output map with a hard part, and **name the plausible
   shortcut** — the best statistic poolable over the context. If you cannot name a shortcut, you cannot
   run Pillar 4, and you will not be able to tell a circuit from a lookup.
2. **Build the counterfactual generator.** Every pillar past Discovery needs paired sequences: move the
   anchor, scramble the source, plant a decoy. This is the single most important piece of engineering.
3. **Wrap your model in the lab API.** `LoadedModel` needs a `run_with_cache` returning per-layer
   residuals and attention patterns, and a per-head ablation knob (`head_mask`). If your model exposes
   those, the assays and viz work unchanged. If not, implement those two hooks and everything else
   follows.
4. **Freeze your verdict criteria before you look.** Decide the thresholds that would make you say
   "circuit" vs "shortcut", write them down, then run. The 2×2's frozen verdict is the template.
5. **Report loss deltas and measured ceilings, not accuracies and guesses.** Saturation is the most
   common way circuit claims go wrong; the measured pooled ceiling is the most defensible control.

The genomic computations this repo has not yet done — splice-site / exon definition, the NMD 50–55 nt
rule, CTCF motif orientation, RNA secondary structure — are each a new instance of exactly this loop.
`docs/BIOLOGY_genomic_computations.md` describes the biology and, for each, what the shortcut and the
circuit would look like. Pick one, name its shortcut, build its counterfactuals, and run the five
pillars.

---

## 11 · Common failure modes (learned the hard way in this repo)

- **Stopping at attention.** The top-attention head is often inert. Always intervene.
- **Reading accuracy where it saturates.** Use loss; watch for the saturation guard.
- **Single-head ablation on a redundant circuit.** Reports "inert" for every member. Use cue conflict.
- **Reading a mechanism off a fixed layer.** The anchor enters at different depths in different
  regimes; a fixed-layer probe produced retracted claims here.
- **No measured ceiling.** Without the pooled-statistic control you cannot distinguish circuit from
  shortcut, and a reviewer will (correctly) not believe you.
- **Zero-ablation treated as ground truth.** Cross-check with resample ablation; a large gap means you
  over-credited the head.

---

*This procedure is implemented end-to-end in `notebooks/TEMPLATE_study_a_circuit.ipynb` (worked twice:
`study_reading_frame_counter.ipynb`, `study_repeat_copy.ipynb`) and the interactive layer under
`src/gfmcd/lab/`. See `docs/LAB.md` for the API reference, `docs/FOUNDATIONS_circuit_discovery.md` for
the math and literature, and `docs/BIOLOGY_genomic_computations.md` for the genomics.*
