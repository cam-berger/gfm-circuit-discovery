# Freeze record — behavioral gate (GO)

> ⛔ **SUPERSEDED / VOID.** This document belongs to the **retracted** v1/v2 partner-selection
> design (R2/R3) or to the pre-sealed-test protocol. It is kept for provenance only.
> **The document of record is `studies/splicebert_partner_selection/FINDINGS.md`;**
> the frozen protocol actually executed is `PREREGISTRATION_SEALED.md`.
> Retractions: `CLAIMS.md` §R1–R12.

Frozen 2026-07-11, at the behavioral gate and before any circuit tracing or test
unblinding. Hashes are sha256. The chromosome-held-out test split was **not**
opened; only its provenance is noted, not its contents.

## Frozen artifacts

| Artifact | sha256 |
|---|---|
| `studies/.../PREREGISTRATION.md` | `cada58d3f9e05e7fc2ee67e4eeeaa2cc289f1ee7bfa59d373ec373e5f2dab2a5` |
| `studies/.../METHOD.md` | `d3d0de3bbe0a00d3bddedd6d47d463284d10a2d8e7d18f7852069084829b5512` |
| `configs/splicebert_partner_selection/base.json` | `c17c555c7cc5a01c63766cc4bb8ee7778dea43898aa2037f53779eeacd6a703f` |
| `scripts/splicebert/train_pair_readout.py` (control fix) | `aecab6916012a74fa20362ae95ccd36eb2e31c071b752b14d59428041257fc8e` |
| `colab_results/gate_bootstrap.json` (formal gate) | `8cf7bc820b7602692218c634bf6183bd9af3f94135f686c4154d7c7f3beff0a3` |
| `colab_results/summary.csv` (25-run matrix) | `8046d151285fbc89ffdb72dd11019bbbabb4c88004751d05224e7cc378dc7f6f` |

## Data provenance

- Dataset build v1 is deterministic: local `val.jsonl` is byte-identical to the
  Colab-built copy on Drive (844,041,632 bytes).
  `val.jsonl` sha256 `a10f01488e761d7aa40f23d77ff99b6cd92b108c93dd0600a4b1e90bbb92d071`.
- Conflict discovery subset: `val_local_conflicts.jsonl` (10,034 examples; gate
  eval used the registered first 4,000). Fixed before any component intervention.
- Test split (chr19–chr22): **sealed**, opened only once after components are frozen.

## Environment (local re-eval / bootstrap)

python 3.13.12 · torch 2.12.0 · transformers 5.12.1 · numpy 2.4.4 · device MPS.
Training matrix produced on Google Colab (CUDA). Backbone weights: official
SpliceBERT.1024nt.

## Tracing freeze — 2026-07-11 (before first real intervention)

Frozen after the local smoke and instrument validation; before the GPU run and
before any test unblinding. Component-set size = **5** (`default_component_set_size`).
Candidate circuit C = N ∩ M (necessity ∩ mediation), per the refined plan.

| Artifact | sha256 |
|---|---|
| `studies/.../TRACING_PLAN.md` (refined, FROZEN) | `180567185f508d06851dd7da2a2174f291d7e53dd080d3c06ce1bc4211c47abc` |
| `src/gfmcd/circuits/hf_bert.py` (intervention primitives) | `a5d5fbc7bb7d14a0dd3d5461bb56bc572d9de5f70ae6d29a7e6e3cba4196dc49` |
| `src/gfmcd/circuits/attribution.py` (AtP screen) | `620f3291bb690003789d724c2d57a35d33486fbabb87dc302029fed8dc07f5e5` |
| `scripts/splicebert/build_discovery_splits.py` | `aeb264ce57bba2807c2e29397ec7f68e66ff03ebc873740e4ade22b423a2e9ba` |
| `results/.../circuits/discovery/split_manifest.json` (D/V_c/O) | `97f7b7f78b175a1bb846d70b0322fff3de93414dd24df6ca31b139b57ee6d43f` |
| `tests/test_circuits_hf_bert.py` | `5da4c7f33ce6e1936bf44f45fb5f4a8d6aa6f474dfa16e38a941be03e9d69f0f` |
| `tests/test_attribution.py` | `30f8433e93748f86065ba975b0c772d6deaaf84f598ffe4070f44223f8cccea4` |

Discovery split hashes (D/V_c/O jsonl) are recorded inside `split_manifest.json`.
Instrument verification at freeze time: 10/10 circuit tests pass; AtP Spearman 0.876
vs exact patching (top-8 overlap 8/8).

Still not frozen: the **selected component set C itself** (produced by the run) and
step-8 test scores — hashed after the run, before the one-time test unblinding.
