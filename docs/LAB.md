# `gfmcd.lab` — the interactive reverse-engineering layer

`gfmcd.lab` is a thin, importable methods layer over the repo's existing tracer / signatures / oracle
code. It exists so that circuit study is a sequence of one-line calls in a notebook — load a
checkpoint, run an assay, intervene safely, plot the result — instead of editing and re-running CLI
scripts. Nothing in `lab` re-implements the science; it wraps `signatures.py`, `oracle.py`,
`data/uniform_cds.py`, `data/repeat_copy.py`, and `models/gfm_nano.py` so their numbers match the
committed results exactly.

```python
import gfmcd.lab as lab
from gfmcd.lab import viz          # viz is a separate submodule so matplotlib is optional
```

---

## Where things live

```
src/gfmcd/lab/
  load.py        loading checkpoints, windows, and training histories
  intervene.py   safe, self-restoring interventions (ablation, resampling)
  assays.py      the five-pillar measurements + reverse-engineering probes
  viz.py         inline plotting on the repo's figure style, with a saturation guard
  __init__.py    the public API (everything below is re-exported from `lab`)

notebooks/
  TEMPLATE_study_a_circuit.ipynb     the parameterized five-pillar + reverse-engineering protocol
  study_reading_frame_counter.ipynb  worked instance: nano_uniform (executed)
  study_repeat_copy.ipynb            worked instance: repeat_copy_s0 (executed)
  sandbox_reverse_engineering.ipynb  freeform poking surface

docs/
  CIRCUIT_DISCOVERY_HOW_TO.md        the step-by-step procedure (start here)
  FOUNDATIONS_circuit_discovery.md   math + literature background
  BIOLOGY_genomic_computations.md    the genomic computations for an ML audience
  LAB.md                             this file
  TODO.md                            reproducibility backlog

scripts/
  build_notebooks.py     regenerates the notebooks from source (nbformat)
  execute_notebook.py    in-process notebook executor (no ZMQ kernel needed; sandbox-safe)
```

The notebooks are **generated**, not hand-edited: change `scripts/build_notebooks.py`, run
`python scripts/build_notebooks.py`, then execute with `python scripts/execute_notebook.py …`. This
keeps the template and its two worked instances in lockstep.

---

## Loading (`load.py`)

| Call | Returns | Notes |
|------|---------|-------|
| `load_checkpoint(name, step=None, device=None)` | `LoadedModel` | `name` is a family stem (→ latest step) or a path; `step=` pins a step |
| `list_checkpoints(families=True)` | `list[str]` | families on disk |
| `resolve_checkpoint(name, step=None)` | `Path` | raises `FileNotFoundError` listing known families |
| `load_uniform_windows(n, window, seed)` | `(tok, aa, phase)` | reading-frame windows (synthetic) |
| `load_repeat_windows(n, window, seed)` | `RepeatCopyBatch` | copy batch; use `.tokens`, `.source_index`, `.copy_mask` |
| `load_cell_windows(cell, n, seed)` | `(tok, aa, phase)` | 2×2 cell A/B/C/D via the training pool; C/D need cached hg38 windows |
| `load_history(stem)` | `list[dict]` | training curve rows from `<stem>_history.json` |
| `grok_transition(history, acc_key)` | `dict` | transition step + 0.5/0.9 crossings + final acc |

`LoadedModel` carries `model, cfg, device, step, pool, path, meta` and the convenience properties
`is_aa_head` (amino-acid head, 21-way), `is_causal`, `name`. `meta` holds any extra checkpoint fields
(`lam`, `shortcut_nats`, `cell`, `seed`) used by the λ sweep and 2×2.

---

## Intervening (`intervene.py`)

All interventions are **context managers that restore on exit, including on exception** — you can never
leave a head silently zeroed. This is the single safety guarantee the layer makes.

| Call | Effect |
|------|--------|
| `ablate_heads(model, heads)` | zero-ablate one or more `(layer, head)` heads |
| `ablate_all_but(model, keep, layer=None)` | zero every head except `keep` |
| `resample_heads(model, heads, source_tokens)` | replace head output with its value on a shuffled batch (on-manifold baseline) |
| `mean_heads(model, heads, ref_tokens)` | replace head output with its per-position mean |

```python
base, _ = lab.aa_loss_acc(m, tok, aa)
with lab.ablate_heads(m.model, (1, 3)):
    ablated, _ = lab.aa_loss_acc(m, tok, aa)
print("Δloss", ablated - base)     # head_mask is restored here, guaranteed
```

Use `resample_heads` to cross-check a zero-ablation: a large gap between the two means part of the
"effect" was distribution shock, not the head's computation.

---

## Assays — the five pillars (`assays.py`)

| Pillar | Call | Returns |
|--------|------|---------|
| shared | `aa_loss_acc(lm, tok, aa)` | `(loss, acc)` — the non-saturating arbiter |
| 1 discovery | `attention_survey(lm, tok, target_pos, min_dist=10)` | `[n_layers, n_heads]` attention to the anchor from downstream queries |
| 3 ablation | `head_ablation_table(lm, tok, labels, target_pos=None)` | `list[HeadEffect]` sorted by Δloss |
| 3 ablation | `redundancy_test(lm, tok, labels, heads)` | `RedundancyResult` (single vs joint) |
| 2/4 testing | `pooled_ceiling(lm, tok, phase, ctx=120)` | float — the anchor-free pooled-statistic accuracy |
| 2/4 testing | `ceiling_report(lm, tok, phase, demanded=1.0)` | `CeilingReport` (ceiling vs model vs demand → `circuit_predicted`) |
| 5 conflict | `cue_conflict_curve(lm, tok, phase, bins=None)` | anchor/content share by distance |
| 5 conflict | `mechanism_signature(lm, tok, phase)` | `MechanismSignature` (anchor share, verdict counter/relay/bag) |
| 5 conflict | `start_codon_ablation(lm, tok, phase)` | dict with the decoy-Met control |

**Reading-frame sweep / 2×2:**

| Call | Returns |
|------|---------|
| `lambda_sweep(checkpoints, n, seed)` | `list[LambdaPoint]` — anchor share vs `shortcut_nats`, sorted |
| `verdict_2x2(lm, tok, aa, phase, cell=None)` | `Verdict2x2` — the frozen verdict (anchor share > 0.50 AND ablation beats decoy by > 0.10) |

**Copy family (causal):**

| Call | Returns |
|------|---------|
| `copy_metrics(lm, batch, seed=0)` | dict: copy accuracy, base-rate ceiling, conflict follows-distal/local |
| `copy_source_attention(lm, batch)` | `[L, H]` mean attention from copy positions to their source |
| `copy_head_ablation(lm, batch, heads=None)` | `list[HeadEffect]` by Δcopy-loss |
| `copy_redundancy(lm, batch, heads)` | `RedundancyResult` |

**Reverse engineering:**

| Call | Returns |
|------|---------|
| `phase_decodability_by_layer(lm, tok, phase)` | `list[float]` — linear-probe accuracy for the frame at each layer's residual stream |

> `ig_head_ranking` deliberately raises `NotImplementedError`: the QK integrated-gradient tracer targets
> the planted `TinyTransformer`, not `GFMNano`. Ranking is done by attention + ablation here.

---

## Plotting (`viz.py`)

`viz` imports the repo's `apply_figure_style()` at load, so plots match the committed figures. Colour
encodes the **mechanism**, never an arm index (BLUE = counter/anchored, RED = bag/anchor-free,
VIOLET = relay, ORANGE = mixed/truncated).

| Call | Panel |
|------|-------|
| `saturation_guard(value, name, ceiling=1.0, tol=1e-3)` | warns (and returns False) if a metric sits at its ceiling/floor — the D20/D27/D31 failure class |
| `plot_attention_survey(attn)` | head × layer attention heatmap |
| `plot_head_effects(effects, top=10)` | twin-axis Δloss vs attention — shows they disagree |
| `plot_redundancy(result)` | single vs joint ablation |
| `plot_ceiling(report)` | pooled ceiling vs model vs demand |
| `plot_cue_conflict(curve)` | anchor/content share by distance |
| `plot_mechanism_map(signature)` | the 2-D mechanism plane with the verdict |
| `plot_grokking(history, acc_key, loss_key, transition=None, base_floor=None)` | loss + accuracy vs step, transition marked |
| `plot_lambda_sweep(points)` | anchor share vs shortcut information |
| `plot_2x2_grid(verdicts)` | the shortcut × detector grid, coloured by verdict |
| `plot_circuit_flow(stages, probe_by_layer=None)` | layer-by-layer information-flow diagram for reverse engineering |

The `saturation_guard` is not decoration: several past errors in this repo came from reading a metric
that was pinned at 1.0. When a conclusion rests on a quantity, guard it.

---

## The notebook workflow

1. **Study a known circuit** → open `study_reading_frame_counter.ipynb` or `study_repeat_copy.ipynb`.
   These are the template executed on the two worked systems.
2. **Study a new circuit** → copy `TEMPLATE_study_a_circuit.ipynb`, change the parameters cell
   (`CHECKPOINT`, `N_WINDOWS`, `SEED`, `DEMANDED`), run top to bottom. The template branches on the
   model family, so it handles both amino-acid and causal checkpoints unchanged.
3. **Poke by hand** → open `sandbox_reverse_engineering.ipynb`. Six self-contained recipes: view a
   head's attention, ablate-and-diff live, resample-ablate, run one cue-conflict example, compare two
   checkpoints across grokking, and the copy circuit. Copy a cell, change a head, re-run.

To regenerate the notebooks after editing the builder:

```bash
python scripts/build_notebooks.py                                  # writes template + sandbox
python scripts/execute_notebook.py notebooks/TEMPLATE_study_a_circuit.ipynb \
    --output study_reading_frame_counter.ipynb \
    --param CHECKPOINT=nano_uniform --param N_WINDOWS=256          # executes an instance
```

`execute_notebook.py` runs the cells in-process (capturing stdout and matplotlib figures into the
notebook) because the sandbox used to build these notebooks blocks the local socket a Jupyter kernel
needs. On a normal machine you can instead open the notebook directly with the `gfmcd-uv` kernel.

---

## Environment

The repo is `uv`-managed. `uv sync --extra lab --extra dev` builds the environment (`torch`, plus the
`lab` extra: matplotlib / jupyter / nbformat / ipykernel / pyfaidx). The Jupyter kernel is registered
as `gfmcd-uv`. See `docs/TODO.md` for the reproducibility backlog, and note item 1.1: the real-hg38
2×2 cells (C, D) need reference data that is not in the repo, so those cells skip until it is fetched.
