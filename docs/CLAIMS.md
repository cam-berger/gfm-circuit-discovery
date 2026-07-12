# Claims registry — reading-frame + repeat-copy tracks

One row per empirical claim. **A passing unit test verifies software, never a scientific claim** — it
can advance a claim to `unit-tested`, never to `supported`; only a fresh, manifest-stamped artifact
does that. Downgrades and retractions are permanent rows. Companion: `RESULTS.md` (numbers),
`DECISIONS.md` (rationale), `PREREGISTRATION.md` (frozen predictions).

Status: `open` · `artifact-supported` · `pending-fresh-run` · `unit-tested` · `withdrawn` · `retracted`.

## Scientific claims

| ID | Claim | Status | Evidence / what would advance it |
|----|-------|--------|----------------------------------|
| C0 | Sufficiency law: a model builds a relational circuit **iff** no pooled-over-context statistic is sufficient for the target (threshold = imperfection). | **open** (hypothesis) | Illustrated by §7 λ-sweep + §9 2×2. Blocked from `supported` by single-seed conditions (see F2) and a lower-bound pooled ceiling (F5). Advances on ≥3-seed replication **and** the RF-limited-CNN pooled rival on the reading-frame arms. |
| C1 | 2×2 mechanism verdicts (A/B/C = anchored counter, D = relay); far-field anchor-share + ablation specificity 3.4×–30×. | **artifact-supported** | `results/reading-frame/fig_data/signature_corrected.json`, `phase_c_t3.json` (recompute exact: 0.5891/0.0193=30.5×, 0.1126/0.0335=3.36×). Per-cell numbers are single-seed (see F2). |
| C4 | λ-sweep: 7 arms, aa→1.0; anchored below the ceiling, anchor-free at/above it. | **pending-fresh-run** | Was printed only; `verdict_lambda.py` now persists `fig_data/verdict_lambda.json` (F1). Advances once that artifact is regenerated from the checkpoints. |
| C5 | Pooled anchor-free ceilings by context window (§8 table). | **open** | No saved artifact; add a dumping script mirroring F1. |
| C6 | Repeat-copy: prefix-keyed copy circuit; distal ~0.964 / local ~0.013; base-rate ceiling ~0.25. | **artifact-supported** | `branches/codex-repeat-induction/…/copy_ceiling.csv`, `copy_prefix_control.csv`, `copy_qk_ig.csv`, `gate_bootstrap.json`. Two seeds (s0/s99). |
| C7 | Tracer error rates: OV recall/precision = 1.0; QK blind-spot costs 0.617; IG recovers the marker edge. | **artifact-supported** | `src/gfmcd/tracer/qk.py`, `planted.py`; gauge-correct QK-product attribution with a live-but-unread-head control. LN-free (disclosed limitation). |

## Retractions / withdrawals (permanent)

| ID | Claim | Date | Reason |
|----|-------|------|--------|
| R-T1 | Reading-frame periodicity detectable by ΔBIC. | (T1) | **withdrawn** — ΔBIC threshold assumed no autocorrelation; an AR(1)-surrogate null's q95 (17–26) sits above the observed ΔBIC (11). `phase_c_t1_null.json`. |
| R-D32/D35 | Mechanism labels read from an L2 probe (D34 on x, D35 on y). | (D32/D35) | **retracted** — a fixed-layer probe reported the wrong mechanism for the cells that produced a new claim (anchor enters at L4 on real CDS, not L2). Corrected by output-level readout + `peak_anchor_dependence`; `signature.json` (wrong) kept beside `signature_corrected.json` (right). |

## Open / pending-fresh-run (RRR audit findings)

| ID | Item | Status | Exact next step |
|----|------|--------|-----------------|
| F2 | Universal iff rests on n=1 seed per λ-arm / 2×2 cell. | **pending-fresh-run** | Re-run `scripts/verdict_lambda.py` and the `scripts/phase_c_*` 2×2 scripts with ≥3 seeds each; report verdict stability per arm/cell. Only cell B (s0/s1) and repeat-copy (s0/s99) currently replicate. |
| F5 | "Only if" half untestable with a lower-bound (naive-Bayes) ceiling. | **pending-fresh-run** | Build/run an RF-limited-CNN pooled ceiling on the reading-frame λ arms + real CDS (mirror repeat-copy's `copy_ceiling`); report the frontier vs the naive-Bayes oracle in `oracle.py`. |
| F6 | Split for cell C/D aa-accuracy (0.99) unverified — possible in-sample. | **open (verify)** | Confirm the `phase_c` window loader draws C/D readout positions from held-out chromosomes (VAL/TEST per `genome.py:41`), not TRAIN; if in-sample, mark the 0.99 accuracy figures accordingly. |
