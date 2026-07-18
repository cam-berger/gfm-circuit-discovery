# Caduceus-PS partner-coupling TRANSFER — DEFERRED (future study)

**Status: DESIGN HARDENED, NOT FROZEN, NOT RUN. Deferred future work.** Per the 2026-07-17 spine
decision (**depth-first**): finish the SpliceBERT donor–acceptor mechanism (P3B) and *then* treat
cross-model transfer as a **separate follow-on study/paper**, not unfinished work inside the SpliceBERT
account. Resume this package once SpliceBERT is closed after P3B.

## What this is

The first transfer experiment: does the sealed partner-specific donor↔acceptor coupling **behaviour**
replicate in **Caduceus-PS** (`kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16`) — a
single-nucleotide, RC-equivariant, **bidirectional Mamba (state-space) MLM**, maximally unlike SpliceBERT's
post-LN attention BERT? TRANSFER_GUIDE Phase 0/1 (behavioral qualification) + the minimal Phase-2 adapter.
Behaviour is only the FIRST milestone — method portability needs Phases 3–4 (SSM-appropriate localization +
intervention), which for a BiMamba must be redesigned (no L3H1/attention/period-3 analog).

## Contents

- `PREREGISTRATION_CADUCEUS_T1_BEHAVIORAL_TRANSFER.DRAFT.md` — the formalized draft (INTENT, scale-free
  estimand ρ=C̄/A_donor, memorials, decision rule, held-out).
- `REDTEAM_SYNTHESIS_5lens.md` — the 5-lens adversarial design red-team (identifiability, statistics,
  estimand↔intent, calibration, instrument/SSM). **Verdict: fixable ×5, fatal ×0.** Contains the MASTER
  20-fix list that must be integrated before this is frozen.

## Why the hardening matters (headline red-team results — verified against Caduceus repo code)

1. **Long-range control is mandatory** (4-lens unanimous). Downstream→upstream flow rides ONLY the reverse
   Mamba pass; the local `A_donor` plant passes even if that channel is dead ⇒ an "architecture null" is
   unidentified against an "instrument-range limit." **Fix:** a distance-resolved transmission-curve control,
   wired to a new INSTRUMENT-LIMIT decision branch.
2. **Three silent-killer plumbing bugs:** Caduceus has NO prefix token (trailing `[SEP]`, `prefix_tokens=0`,
   not SpliceBERT's `p+1`); `padding_side='left'` (corrupts positions in mixed-length batches);
   A/C/G/T ids after 7 reserved specials (resolve dynamically). Each = silent wrong readout.
3. **No CPU/MPS fallback** — `mamba-ssm`/Triton kernels are CUDA-only; **Colab CUDA is primary compute**,
   gated by a feasibility preflight. (My "7.7M params → CPU trivial" was wrong.)
4. **RC-equivariance demotion CONFIRMED** (3 lenses): forward↔RC coupling is architecture-forced for PS
   (exact param-tying in `RCPSLMHead`) ⇒ a PLUMBING memorial, not strand-specificity. An independent
   strand control would need Caduceus-PH.

## Open forks when resumed (from the synthesis, "Genuine forks for Cam")

τ/meaningfulness floor (within-model permuted-null vs fraction-of-ρ_SpliceBERT); acceptor-strength confound
(distance-matched primary + score3-caliper secondary); confirm Colab compute; PH companion (lean drop).

Provenance: 5-lens red-team 2026-07-17; checkpoint facts verified via HF + arXiv:2403.03234 + repo code.
