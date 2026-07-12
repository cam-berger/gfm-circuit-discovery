# Circuit Discovery in Genomic Foundation Models

Tracing how a training objective installs — or fails to install — a latent representation of genome
structure inside a genomic foundation model, and reverse-engineering the circuit that results.

The organizing hypothesis, tested rather than asserted, is the **shortcut-gap hypothesis**:

> Relational mechanisms become more likely when they provide enough objective improvement over a
> registered class of local or pooled shortcuts to overcome their optimization and complexity cost.
> Shortcut insufficiency creates pressure for a relation; it does not guarantee that gradient descent
> finds one, and shortcut availability does not prove that no relational mechanism exists.

**Status — hypothesis, not settled law.** This is the organizing hypothesis, strongly illustrated by
the λ-sweep and the 2×2 — but not established as a universal law: each condition is a **single seed**,
and the pooled ceiling is measured by a **lower-bound** estimator (codon naive-Bayes; the stronger
receptive-field-limited-CNN rival exists only for the repeat-copy track, not reading frame). Advancing
it to settled needs ≥3-seed replication and the stronger pooled rival on the reading-frame arms. See
`docs/CLAIMS.md` (C0). The original stronger “if and only if” sufficiency-law wording is retained only
in historical records where its status is marked; it is not treated as established.

Everything in the repo is instrumentation for measuring, honestly, which of the two — circuit or
shortcut — a trained model actually did.

Built from scratch: none of our model families (planted, GFM-nano, SpliceBERT) are supported by
TransformerLens / `circuit-tracer`, which target GPT-style decoder-only models. See
`docs/DECISIONS.md#d5`.

---

## New here? Read in this order

1. **`SPLICEBERT_CIRCUIT_DISCOVERY_IMPLEMENTATION.md`** — the paper-sized real-GFM plan, exact
   run order, claim ladder, and implemented SpliceBERT scaffold.
2. **`docs/CIRCUIT_DISCOVERY_HOW_TO.md`** — the step-by-step procedure for discovering and
   reverse-engineering a circuit. The practical method entry point.
3. **`docs/FOUNDATIONS_circuit_discovery.md`** — the math and literature behind the methods (residual
   stream, QK/OV, patching, integrated gradients, induction heads, grokking, the sufficiency law).
4. **`docs/BIOLOGY_genomic_computations.md`** — the genomics for an ML audience: what reading frame,
   repeats, splicing, etc. actually are, and what the shortcut vs. the circuit looks like for each.
5. **`docs/LAB.md`** — the `gfmcd.lab` API reference and the notebook workflow.
6. Then open **`notebooks/study_reading_frame_counter.ipynb`** to see the whole protocol run on a real
   model.

For the research record rather than the how-to, see `docs/EXPERIMENTS.md` (plain-language account),
`docs/RESULTS.md` (every measured number, with status), and `docs/REC.md` (the argument).

---

## Environment setup

The repo is **`uv`-managed** (it was previously conda `gfm-thrust1`; that env is retired — see
`docs/TODO.md` 3.1). `uv.lock` pins the full dependency set.

```bash
# 1. install uv (once), if you don't have it
pip install uv                      # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. create the environment and install the project + the interactive/dev extras
uv sync --extra lab --extra dev     # torch, numpy, scipy + matplotlib/jupyter/nbformat + pytest

# 3. run the tests to confirm the install
uv run pytest -q                    # 133 tests; Rung 0 is ground truth, not a smoke test
```

`uv run <cmd>` runs a command inside the environment; alternatively activate it with
`source .venv/uv/bin/activate`. The dependency **extras** are:

| Extra | Pulls in | Needed for |
|-------|----------|-----------|
| (base) | `torch`, `numpy`, `scipy` | the library and models |
| `lab` | matplotlib, jupyter, nbformat, ipykernel, pyfaidx | `gfmcd.lab`, the notebooks |
| `models` | transformers, datasets | general pretrained-model comparisons |
| `splicebert` | transformers, pyfaidx | the SpliceBERT partner-selection implementation |
| `genome` | pyfaidx | building windows from real hg38/GENCODE |
| `dev` | pytest, pytest-xdist | the test suite |

**Notebooks.** The Jupyter kernel is registered as `gfmcd-uv`. Open any notebook and select that
kernel. To regenerate the notebooks from source and execute an instance headlessly:

```bash
uv run python scripts/build_notebooks.py                       # writes the template + sandbox
uv run python scripts/execute_notebook.py \
    notebooks/TEMPLATE_study_a_circuit.ipynb \
    --output study_reading_frame_counter.ipynb \
    --param CHECKPOINT=nano_uniform --param N_WINDOWS=256      # executes a worked instance
```

`execute_notebook.py` runs cells in-process (capturing stdout and matplotlib figures) because the
sandbox these were built in blocks the socket a live Jupyter kernel needs; on a normal machine you can
just run the notebook in Jupyter directly.

**Data note.** Model checkpoints and cached windows are gitignored. The synthetic tasks (reading-frame
uniform/hard-detector cells A/B, repeat-copy) regenerate on demand. The real-hg38 tasks (2×2 cells
C/D) need `hg38.fa` + GENCODE, which are not in the repo — those cells skip cleanly until the data is
fetched (`docs/TODO.md` 1.1).

---

## Repository map

```
README.md                       you are here
pyproject.toml  uv.lock         dependencies; uv.lock pins the full set

src/gfmcd/                      the library
  config.py                     firewalls (see below) — raises, does not warn
  signatures.py                 the 2-D mechanism signature; anchor/probe primitives
  oracle.py                     the anchor-free pooled oracle: the best-frame-without-the-anchor ceiling
  figstyle.py                   figure style; colour encodes MECHANISM, never arm index
  data/
    uniform_cds.py              synthetic reading-frame windows (cells A/B, the λ sweep)
    repeat_copy.py              synthetic repeat-copy / induction batches
    genome.py                   real hg38 / GENCODE windows (needs the genome extra + data)
    synthetic.py                shared synthetic-data helpers
  models/
    gfm_nano.py                 GFM-nano: 6L x 8h, d_model 256, {A,C,G,T,N}; the trained subject
    planted.py                  planted twins: known edge set BY CONSTRUCTION (Rung 0)
    tiny.py                     the LN-free TinyTransformer the tracer targets
  tracer/
    graph.py  patch.py  qk.py  jacobian.py   frozen-attention + QK-IG attribution (planted model)
  nulls/                        surrogate/null-sequence generators
  dicts/                        sparse-dictionary (SAE-style) code
  lab/                          >>> the interactive reverse-engineering layer (see docs/LAB.md) <<<
  adapters/                     stable adapters for pretrained genomic models
  circuits/                     BERT head intervention/patching and QK/OV tools
  splicing/                     schemas, GENCODE builder, controls, readouts, metrics
  provenance.py                 hashes and run manifests
    load.py                     one-call checkpoint / window / history loading
    intervene.py                self-restoring interventions (ablate / resample / mean)
    assays.py                   the five-pillar measurements + reverse-engineering probes
    viz.py                      inline plots on figstyle, with a saturation guard
    __init__.py                 the public API

notebooks/
  TEMPLATE_study_a_circuit.ipynb        the parameterized protocol: grok -> 5 pillars -> reverse-engineer
  study_reading_frame_counter.ipynb     worked instance: nano_uniform (executed)
  study_repeat_copy.ipynb               worked instance: repeat_copy_s0 (executed)
  sandbox_reverse_engineering.ipynb     freeform poking surface: 6 self-contained recipes

scripts/
  build_notebooks.py            regenerates the notebooks from source (nbformat)
  execute_notebook.py           in-process, sandbox-safe notebook executor
  README.md                     which of the ~30 measurement scripts produces which number
  splicebert/                  real-GFM data, behavior, trace, and patch pipeline
  ... (one script per committed measurement)

tests/                          the test suite (133 tests); Rung 0 is ground truth, not a smoke test
  test_lab.py                   smoke tests for the gfmcd.lab API (18 tests)
  ...                           per-module tests for data, models, tracer, oracle, signatures

docs/
  CIRCUIT_DISCOVERY_HOW_TO.md   step-by-step: discover a circuit, then reverse-engineer it
  FOUNDATIONS_circuit_discovery.md   math + literature intro (for new grad students)
  BIOLOGY_genomic_computations.md    the genomic computations, for an ML audience
  LAB.md                        the gfmcd.lab API reference + notebook workflow
  TODO.md                       reproducibility backlog (blockers / correctness / hygiene)
  EXPERIMENTS.md                plain-language account of every experiment and what it showed
  RESULTS.md                    every measured number, with status: SETTLED/PENDING/WITHDRAWN/RETRACTED
  REC.md                        the argument: critique of Proposal D2, and the sufficiency criterion
  Proposal_Sufficiency_*.md     the PI-facing writeup
  RELAY_TRACE.md                Phase C: what separates a counter from a relay from a bag
  DECISIONS.md                  D1-D35, why the design is what it is, with the numbers
  PREREGISTRATION.md            frozen predictions, registered before each measurement
  PLAN.md  PLAN_reading_frame.md  PLAN_repeat_copy.md   phases and gates, across both tracks
  explainers/                   standalone HTML for non-specialists
  archive/                      superseded prose, kept not deleted

results/reading-frame/
  fig_data/                     the measured numbers -- TRACKED, so a retraction leaves a diff
  fig_data/retracted/           dumps whose measurement was withdrawn
  figures/                      gitignored; regenerate with scripts/fig_phase_*.py
branches/codex-repeat-induction/   track 2's findings and figures

data/cache/                     window caches, gitignored: the filename is the recipe
checkpoints/                    trained model checkpoints, gitignored (families listed below)
logs/archive/                   completed run logs
```

Two tracks share this repo and share no files: **reading frame** (an anchored period-3 counter) and
**repeat-copy / induction** (a prefix-keyed copy circuit). Each is a worked instance of the same
five-pillar protocol.

---

## The `gfmcd.lab` layer in one screen

The interactive layer turns circuit study into one-line calls (full reference in `docs/LAB.md`):

```python
import gfmcd.lab as lab
from gfmcd.lab import viz

m = lab.load_checkpoint("nano_uniform")          # family stem -> latest step, or step=1200
tok, aa, phase = lab.load_uniform_windows(n=256) # counterfactual-ready windows

# Pillar 1 discovery — attention RANKS heads (it does not decide)
viz.plot_attention_survey(lab.attention_survey(m, tok, start))
# Pillar 3 ablation — Δloss DECIDES; the top-attention head is often inert
viz.plot_head_effects(lab.head_ablation_table(m, tok, aa, target_pos=start))
# Pillars 2 & 4 — is there anything to explain? demanded accuracy vs the pooled ceiling
print(lab.ceiling_report(m, tok, phase, demanded=1.0).circuit_predicted)
# Pillar 5 conflict — the only test that sees a redundant circuit
print(lab.mechanism_signature(m, tok, phase).verdict)     # counter / relay / bag

with lab.ablate_heads(m.model, (1, 3)):           # interventions self-restore, always
    ...
```

The one rule the tooling enforces for you: every intervention is a context manager that restores on
exit, including on exception — you can never leave a head silently zeroed.

---

## The ladder

```
Rung 0  planted twins    edge set known BY CONSTRUCTION   -> tracer validation   [gate]
Rung 1  reading frame    period-3, anchored to an ATG     -> proof of method
Rung 2  splice pairing   GENCODE + pretrained SpliceBERT -> first real-GFM test
Rung 3  G-quadruplex     cross-strand, loop-window        -> flagship
H2      objective arms   AR < MLM < MLM+phyloP < phyloP   -> the payload
```

Rung 0 exists because a trained model gives you a known *task*, not a known *edge set*. Two planted
twins — one compositional, one a lookup table — have identical behaviour and different mechanism:

```
                    accuracy    after marker ablation
compositional         0.586            0.383
heuristic             0.583            0.583
                    |Δ| 0.0034
```

The tracer must find a circuit in one and fail in the other. A positive control alone cannot falsify a
tracer that reports compact circuits on everything.

---

## Checkpoint families

Gitignored, but the trained families the notebooks and scripts expect on disk (each carries `_step<N>`
suffixes; `lab.load_checkpoint` picks the latest step by default):

```
nano_uniform          reading-frame counter (amino-acid head)      -- the worked Rung-1 model
nano_sense, nano_lam0.25/0.5/0.75/1/1.5/2   the λ shortcut-strength sweep
cellB_s0, cellB_s1, cellC_s0, cellD_s0      the 2x2 necessity cells
nano_ar_*, nano_mlm_*                       objective-arm comparisons (H2)
repeat_copy_s0, repeat_copy_s99             the induction / copy circuit (causal head)
```

---

## Firewalls

`config.py` raises rather than warns, so an invalid comparison stops the run instead of quietly
producing a number:

- `assert_holdout_valid("splicebert")` → pretrained on the whole human genome; no chromosome is held
  out. Mechanistic claims fine, generalisation claims impossible.
- `assert_per_nucleotide_readout("dnabert2")` → BPE; a frameshift is not a minimal pair.
- `assert_cross_strand_capable("splicebert")` → pre-mRNA, sense strand only; cannot represent G4's
  cross-strand axis.
