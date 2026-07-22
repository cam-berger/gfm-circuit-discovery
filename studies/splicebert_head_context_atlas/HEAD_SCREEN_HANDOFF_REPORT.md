# Head-screen handoff report (one page)

> **Tier: T3 — historical.** Superseded or point-in-time; retained for the record. Not current guidance.

**Bottom line:** the all-96-head A/B/C/D exact-effect screen is **built, locally tested, and smoke-verified**;
the full run is **not** launched (reserved for your GO after the Colab benchmark). No sealed run, no credit
spend, no visualizer. C is labeled **competitor-site perturbation response** (the specificity pilot was null).

| Piece | Status | Evidence |
|---|---|---|
| Stage -1 pre-flight (patching + cost) | **GO** | source-to-source no-op \|diff\|=0; 27 ms/head-forward (MPS) |
| Runner `run_all_head_screen.py` (scrub-to-corrupt, 96 heads, 12 gates, resumable shards) | **built + smoke** | 4-example smoke: 0 gate failures, 96 head records/condition |
| `validate` / `merge` / `benchmark` | **built + run** | validate ok=true; merge → matrix + C33 group summary; benchmark 11.6 s/ex |
| Tests | **9 passed** | cross-hashseed determinism + vectorized==scalar agreement + gate/metric math |
| Colab package (notebook/req/config/checklist) | **built** | orchestration-only notebook (17 cells, valid) |
| Full discovery run | **NOT-YET** | gated on your GO after the Colab-CUDA benchmark projection |

**Benchmark (MPS proxy):** 310 examples/hr → **~15.7 MPS-hours** for the 4,864-gene discovery cohort, **~315 MB**.
The Colab-CUDA benchmark cell supplies the real GPU-hour projection (expect far less than 550 credits).

**Documentation corrections applied (as you directed):**
- The C specificity result is **discovery-stage, instrument-limited** — *not* an equivalence null. A
  publication-grade equivalence would need a powered test on a disjoint/sealed cohort.
- IC1/IC2 agreement is **qualitative coherence**, not independent replication.
- All topology tags are **annotation-defined alternatives**, not tissue-active competition.
- B/C behavioral effects are **recorded diagnostics**, not launch gates; the exploratory screen proceeds
  regardless. chr8/9 remain **sealed** (hard-excluded; `--unlock-sealed` needs a separate confirm config).
  The visualizer stays **deferred** until the head-effect matrix passes validation.

**Analysis boundary (frozen into the output):** exploratory head-effect matrix only. Permitted — a head has
an exact A/B/C/D effect on discovery examples, varying descriptively across topology; screening attribution
agrees/disagrees with exact intervention. NOT permitted — competition-specificity, biological-function
naming, sealed generalization, "single-head null ⇒ distributed absence" (C33 preserved via the group
summary), additive head fractions.

**Remaining scientific limitations:** short-intron/≤1 kb cohort bias (~90% of competition truncated by the
window); single canonical destroy variant for the head cache; per-head collateral health is a cheap proxy
(masked health at condition level); C is a competitor-site perturbation response of undetermined specificity.

**Blockers for you:** (1) fill the notebook clone URL/token + Drive data paths; (2) run the Colab
smoke+benchmark and review the projection; (3) GO before `--launch-full`.

## Git + proposed commit (not committed — awaiting your cadence)
New/untracked: `src/gfmcd/splicing/{competition,competition_seq,specificity}.py`,
`scripts/splicebert_head_context_atlas/`, `studies/splicebert_head_context_atlas/`,
`results/splicebert_head_context_atlas/`, `tests/test_competition.py`, `tests/test_head_context_screen.py`.

Proposed message:
```
Head-Context Atlas: intron-mate competition cohort, behavioral null, all-head screen package

- 6-lens design red-team -> fix-in-place (intron-mate cohort, scrub-to-corrupt, group estimand)
- census + behavioral + specificity pilots: competition is a competitor-site perturbation
  response, NOT competition-specific (discovery-stage instrument-limited; coheres w/ IC1/IC2)
- all-96-head A/B/C/D exact-effect screen: runner + validate + merge + benchmark + Colab package,
  12 mechanical gates, resumable shards; full run gated on benchmark + GO. No sealed run.
```
