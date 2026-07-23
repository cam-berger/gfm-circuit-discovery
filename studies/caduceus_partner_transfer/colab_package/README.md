# Caduceus-PS T1 — behavioural-transfer Colab package

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Runs the **frozen prereg v2** (`../PREREGISTRATION_CADUCEUS_T1_BEHAVIORAL_TRANSFER.md`): does SpliceBERT's
partner-specific donor↔acceptor coupling **behaviour** transfer to **Caduceus-PS**, a BiMamba state-space MLM
(maximally unlike SpliceBERT)?  **Phase-1 behavioural qualification only** — a positive result licenses
Phase-3/4 mechanism work, it does **not** prove mechanism/method portability.

## Why a Colab package (not local)
Caduceus-PS's HF modeling code unconditionally imports `mamba_ssm` (+ Triton RMSNorm, causal-conv1d) — **CUDA
only, no CPU/MPS fallback** (probe-confirmed 2026-07-18). The Mac loads its config + tokenizer but not the model.

## Discipline: don't waste GPU time/money
Everything that does **not** need the model is built and **validated locally** first, so the GPU run is one
clean, smoke-gated pass:
- **Coordinate correctness** (the #1 silent-failure risk) — Caduceus has `prefix_tokens=0`, a **trailing**
  `[SEP]`, and `padding_side='left'`; a naive SpliceBERT port (`p+1`, right-pad) reads the neighbouring base.
  `caduceus_scorer.py` fixes this (readout at `p`; **fixed 1024 nt, no padding** — right-padding still corrupts
  an SSM's reverse pass) and **asserts it at init**. Validated locally against the **real tokenizer** + a
  planted **mock** model (`test_local_dryrun.py`: margin@donor ≫ margin@±1, A_donor, coupling, decay,
  determinism, padding-safety — all pass).
- **Full decision pipeline** (estimands → BCa gene-cluster CIs → Fieller ratio bootstrap → joint power → §6
  IUT decision) — validated end-to-end against the mock (`test_pipeline_dryrun.py`: planted-positive →
  QUALIFIES; permuted-partner control → does NOT qualify).
- **Cohort + controls** — built from the real genome locally (`build_cohort.py`): 6,940 examples, **1,072
  confirmatory genes**, distance-matched same-side non-partner control (B-3 exclusion, common-support/Berkson
  retention reported), other-`gt` donor control, long-range transmission constructs. GPU only does forwards.

## Workflow
1. **Local:** `python build_cohort.py` → `cohort_constructs.jsonl` (+ `cohort_meta.json`). Optionally re-run
   the two dry-run tests to reconfirm plumbing.
2. **Colab:** open `caduceus_t1_colab.ipynb` on a **GPU** runtime, upload the four `.py` modules +
   `cohort_constructs.jsonl`, run top-to-bottom. The **smoke gate** hard-stops on any memorial failure
   (coordinate roundtrip, `mlm_health>0`, `A_donor>0`) before the confirmatory pass.
3. **Result:** `results_t1.json` → `VERDICT ∈ {QUALIFIES-FOR-PHASE-3, DOES-NOT-QUALIFY (honest architecture
   null), INSTRUMENT-RANGE-LIMIT, UNDER-POWERED}`. Report per prereg §6.

## Files
| file | role | needs GPU |
|---|---|---|
| `caduceus_scorer.py` | coordinate-correct masked-LM readout (E2/E3/E6) | run only |
| `caduceus_t1.py` | estimands + stats (Fieller, BCa, joint power) + §6 decision | no |
| `build_cohort.py` | local cohort/construct/control builder (real genome) | no |
| `run_t1.py` | orchestrator: load → τ(discovery) → decision(confirm) | run only |
| `mock_caduceus.py` | planted-positive oracle for local validation | no |
| `test_local_dryrun.py` / `test_pipeline_dryrun.py` | local validation | no |
| `caduceus_t1_colab.ipynb` | the GPU notebook | yes |

## Post-verification corrections (2-lens adversarial pass, both fixed + re-validated)
The verification caught **two verdict-flipping blockers before any GPU spend** — now fixed:
- **τ was toothless** (it negated the observed effect ⇒ collapsed to 0 exactly when partner-specificity was
  significant ⇒ false QUALIFIES). Now a **genuine permuted-partner null**: two non-partner AGs per example
  (`nonpartner2_cc`), `null = C_nonpartner1 − C_nonpartner2` (no real partner identity), τ = its upper CI,
  **max over seeds {0,1,2,17}** (R22 sampling-stability). `build_cohort` emits the 2nd non-partner (2,795 loci).
- **Joint power was post-hoc** (at the observed confirm effect ⇒ a true null always read UNDER-POWERED ⇒ the
  honest architecture-null was unreachable). Now computed at a **pre-registered anticipated effect**
  `ρ_anticipated·A_donor` (default 0.15) on the realized gene-n SEs, over all conjunction arms.
- Majors: `both_strands_present ∈ [0.3,0.7]` memorial added (cohort share 0.503); long-range certification now
  requires **visibility** (near block large) AND **non-negligible** mid-range (≥10% of near) AND CI-excludes-0;
  QUALIFIES now requires long-range certification (§6f); the coordinate/`mlm`/`A_donor`/strand preflight is a
  **hard STOP inside `run_t1.main()`** (not just the notebook), so a standalone run also refuses to waste GPU.
Both local dry-runs pass on the corrected code (`test_local_dryrun`: coordinate + behaviour; `test_pipeline_dryrun`:
planted-positive → QUALIFIES with joint_power 1.0 + long-range certified, permuted-partner → not qualified).

## Not yet done (held per Cam)
- The actual GPU run is **not deployed** — this package is for review first.
- Open forks in the prereg header (τ philosophy, strength-confound arm, PH drop) use the red-team's recommended
  defaults; confirm before deploying.
- `mamba-ssm`/`causal-conv1d` versions in cell 1 may need pinning to the Colab CUDA/torch combo at deploy time.
