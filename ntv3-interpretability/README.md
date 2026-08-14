# NTv3 8M Promoter Mechanistic Discovery

A port of the DNABERT-2 promoter-mechanism notebooks (`dnabert/01_...` through
`09_...`) to [Nucleotide Transformer v3](https://huggingface.co/spaces/InstaDeepAI/ntv3)
8M (`InstaDeepAI/NTv3_8M_pre`), as Python scripts rather than notebooks. This
replaces an earlier NTv2 port (`ntv2-interpretability/`, removed 2026-08-12) --
NTv3 is architecturally unrelated to NTv2, so the port started fresh rather
than search-replacing.

## Setup

The model is **gated** on Hugging Face:

1. Visit https://huggingface.co/InstaDeepAI/NTv3_8M_pre and accept the license.
2. Get a read token from https://huggingface.co/settings/tokens.
3. Make it available to the scripts, either via env var:
   ```
   export HF_TOKEN=hf_...        # bash
   $env:HF_TOKEN = "hf_..."      # PowerShell
   ```
   or by running `huggingface-cli login` once.

Unlike the old NTv2 port, this one does **not** need an isolated environment
-- it was verified against `transformers==5.14.1` / `torch==2.13` (CPU) in
the environment already used elsewhere in this repo. Dependencies:
`torch`, `transformers`, `pandas`, `numpy`, `scikit-learn`, `scipy`,
`matplotlib`, `joblib`.

Run scripts from this directory (`ntv3-interpretability/`):

```
python experiments/phase01_representation_baseline.py
```

## Architecture, verified against the real checkpoint

NTv3 is a **U-Net**, not a flat transformer encoder: a 7-stage convolutional
downsampling tower feeds a 2-layer Transformer bottleneck, which a matched
7-stage deconvolutional tower upsamples back to single-nucleotide resolution.
Every architectural fact this study depends on -- module paths, tokenizer
behavior, `hidden_states` layout -- was verified by loading the real gated
checkpoint and its remote-code source, not assumed from the model card. See
`ntv3/model.py`'s module docstring for the full list and how each fact was
checked, and `ntv3/config.py` for the input-padding tradeoff this forces.

The practically important consequence: at the 128nt pad length every phase
in this study uses, the Transformer bottleneck is a **single token**
(`128 / 2**7 == 1`). Pooled-representation probing (phase01) and black-box
behavioral corruption assays (phase02) don't care -- they never touch
attention structure. But attention-head-level component localization would
be close to vacuous at this resolution: there's nothing for one bottleneck
token to attend to but itself. Phase03 onward therefore reframes
"component" toward the conv/deconv towers and skip connections, which
retain per-nucleotide resolution throughout, plus the bottleneck's per-token
FFN (still meaningful without attention) -- see `ntv3/patching.py`'s module
docstring for the full mechanism.

**Readout depth matters for causal work, not just accuracy.** `hidden_states`
are produced in forward-pass order (conv1..conv7, xformer1-2,
deconv1..deconv7); a component can only causally affect a readout that runs
*after* it. Phase01 picks its probe depth purely by accuracy (phase02 is
fine reusing it, since input corruption affects every depth), but phase03+
specifically use the *final* depth (deconv7) so every component tested is
upstream of the readout -- phase03's own development run caught this the
hard way (an early-depth readout produced an exact 0.0000 effect for every
downstream component) before the fix. See `phase01`'s
`ntv3_all_depth_probes.joblib` and `phase03`'s module docstring.

## Phases

| Phase | Notebook origin | Status |
| --- | --- | --- |
| 01 representation baseline | `01_Representation_baseline.ipynb` | done |
| 02 behavioral localization | `02_promoter_behavioral_localization.ipynb` | done |
| 03 component localization | `03_promoter_component_localization.ipynb` | done -- reframed around conv/deconv/skip, not attention heads |
| 04 causal interventions | `04_promoter_causal_interventions.ipynb` | done -- reconstruct-inject, reframed to phase03's components |
| 05 component feature characterization | `05_promoter_mlp_feature_characterization.ipynb` | done -- targets `deconv7_main`, phase04's clearest transferable-signal carrier |
| 06 GC-controlled interventions | `06_gc_controlled_interventions.ipynb` | done |
| 07 low-rank subspace analysis | `07_low_rank_subspace_analysis.ipynb` | done |
| 08 subspace causal interventions | `08_promoter_mlp_subspace_causal_interventions.ipynb` | done |
| 09 pathway mediation validation | `09_promoter_pathway_mediation_validation.ipynb` | done -- see phase09's module docstring for a mediation-methodology caveat |

## Headline findings so far

Each of these was produced by actually running the corresponding script
against the real gated checkpoint, not simulated -- see each phase's
`results/phaseNN_results.json` and `results/figures/` for the full numbers.

- **Phase01**: linear probe accuracy across all 16 U-Net depths peaks right
  at the narrowest point (`conv7`/`xformer1`), beating k-mer baselines,
  chance on shuffled labels.
- **Phase02**: mono-nucleotide shuffling collapses promoter confidence
  (-0.29); dinucleotide-preserving shuffling barely touches it (-0.015) --
  the signal is almost entirely dinucleotide-composition-driven, not
  higher-order structure.
- **Phase03/04**: `deconv7_main` (the final deconv stage) is both necessary
  and, when transplanted from a clean promoter into a corrupted one,
  sufficient to substantially restore promoter confidence -- and an
  *unrelated* clean promoter donor rescues about as well as the sequence's
  own true donor, while a non-promoter donor doesn't rescue at all. That's
  a transferable, class-level signal, not sequence-specific memorization.
- **Phase05/06**: GC content is the strongest simple biological correlate of
  that transfer signal (after the donor's own probe score). A controlled,
  length-preserving GC shift moves the readout causally, with a clear
  dose-response separated from a matched neutral (non-GC) edit control.
  A rank-1 "promoter direction" in `deconv7_main`'s activation space,
  estimated on one half of the data, predicts both the achieved GC shift
  and the score change on the *other*, held-out half (rho = 0.83 / 0.61).
- **Phase07/08**: an unsupervised SVD of clean-vs-corrupted delta vectors is
  less compact and less predictive than that supervised direction (effective
  rank ~35 of 80), but the top singular directions are still causally real:
  removing the top-1 direction from clean promoters drops confidence more
  than any of 20 random norm-matched directions (100th percentile).
  Sufficiency (injecting the direction into non-promoters) is present but
  far weaker and not clearly direction-specific (~60th percentile vs. random).
- **Phase09**: pushing `conv2_main` (phase03's most necessary component)
  toward the promoter direction on non-promoter sequences propagates all the
  way to the final readout, rising again through the late deconv stages
  before settling. A single downstream component blocks the majority of
  that propagated effect when restored to its clean value -- but see
  phase09's module docstring for why "redundant vs. necessary" can't be
  fully disentangled with this study's hook mechanics on a serial chain.

Results land in `results/phaseNN_results.json` (+ cached arrays), figures in
`results/figures/phaseNN_*.png`. Every script prints and records the sample
sizes and settings it actually used -- see `ntv3/config.py`'s module
docstring for the CPU-only-inference tradeoffs this documents.

## Report

`NTv3_Promoter_Interpretability_Report.pdf` is a consolidated PDF of every
phase -- objective, method, key numbers, and every figure -- built by
`build_report.py` directly from `results/phaseNN_results.json` and
`results/figures/`, so it can't drift out of sync with the actual data.
Rebuild it after rerunning any phase with:

```
python build_report.py
```

## Layout

```
ntv3/            shared package: config, data loading, model adapter, metrics,
                  stats, plotting -- all model-specific code lives in model.py
experiments/      one script per phase, run directly with `python experiments/phaseNN_*.py`
data/             cached GUE prom_core_all CSVs
results/          phaseNN_results.json, cached representation arrays, figures/
```
