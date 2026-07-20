# Caduceus-PS partner-coupling transfer

**Status: DESIGN HARDENED, COLAB PACKAGE LOCALLY VALIDATED, NOT FROZEN, NOT RUN. Deferred while the
SpliceBERT context/transduction program is prioritized.**

This is the prepared second-model study: test whether SpliceBERT's partner-specific donor↔acceptor
coupling **behavior** appears in Caduceus-PS
(`kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16`), a single-nucleotide,
RC-equivariant, bidirectional Mamba MLM. Behavioral transfer is only the first milestone; a positive
result would not establish mechanism transfer.

## Current package

- [`PREREGISTRATION_CADUCEUS_T1_BEHAVIORAL_TRANSFER.md`](PREREGISTRATION_CADUCEUS_T1_BEHAVIORAL_TRANSFER.md)
  — v2 frozen-candidate with the five-lens red-team fixes integrated.
- [`REDTEAM_SYNTHESIS_5lens.md`](REDTEAM_SYNTHESIS_5lens.md) — identifiability, statistics, calibration,
  estimand, and state-space instrument review (fixable ×5, fatal ×0).
- [`colab_package/`](colab_package/README.md) — CUDA runner, deterministic cohort, coordinate-correct
  scorer, mock planted-positive tests, and end-to-end decision pipeline.
- `PREREGISTRATION_CADUCEUS_T1_BEHAVIORAL_TRANSFER.DRAFT.md` — superseded pre-red-team draft, retained
  only for provenance.

## Resolved design hazards

1. A distance-resolved long-range transmission control distinguishes a true behavioral null from a dead
   reverse-SSM information channel.
2. Coordinate handling is model-native: no prefix token, trailing `[SEP]`, fixed-length/no-padding model
   inputs, and dynamically resolved nucleotide IDs.
3. The meaningfulness floor is applied to partner-specific effect size and is derived from a genuine
   distance-matched permuted-partner null across multiple seeded draws.
4. Joint power is evaluated at a preregistered anticipated effect, not the observed confirmatory effect.
5. The inherited partner-strength confound is explicit: distance-matched primary, MaxEnt-score-caliper
   secondary with a predeclared non-identifiable branch.

## Remaining gates before GPU deployment

1. Confirm or override the documented defaults: permuted-null floor, distance-matched primary plus
   score-caliper secondary, Colab CUDA, and no Caduceus-PH companion.
2. Pin a compatible Colab CUDA/Torch/`mamba-ssm`/`causal-conv1d` environment.
3. Freeze the v2 protocol and package hashes before any real-model output is read.
4. Run smoke gates first; stop on coordinate, MLM-health, local-anchor, strand-balance, or long-range
   transmission failure.

No CPU/MPS fallback exists for the checkpoint's hard-imported Mamba/Triton kernels. The actual model run
has not occurred, and no transfer result may be inferred from the locally passing mock pipeline.
