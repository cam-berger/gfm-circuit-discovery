# RESULTS — repeat_copy

Rolls up into the global `docs/RESULTS.md`. Status ∈ {EXPLORATORY, PROVISIONAL, REPLICATED, SETTLED,
WITHDRAWN, RETRACTED, BLOCKED}.

## The objective threshold — the circuit is forced, not unbidden

**REPLICATED (weight 1, 3 seeds); PROVISIONAL (weight 2/4, 1 seed each).**
`scripts/train_repeat_copy.py` → `scripts/eval_repeat_copy_sweep.py`; artifact
`results/repeat-copy/objective_sweep.json`. Verdict is causal (cue-conflict distal-follow), not
accuracy. Base-rate copy ceiling ≈ 0.25.

| copy weight | seeds | copy acc | distal-follow | local-follow | verdict |
|---|---|---|---|---|---|
| 1 | 0,1,2 | 0.313 / 0.311 / 0.315 | 0.320 / 0.318 / 0.319 | ~0.23 | **NO CIRCUIT** (0/3) |
| 2 | 0 | 0.345 | 0.348 | 0.218 | NO CIRCUIT |
| 4 | 0 | 0.953 | 0.942 | 0.016 | **FORMED** |
| 8 | 0 (legacy `repeat_copy_s0`) | 0.962 | 0.956 | 0.014 | FORMED |

**Reading.** The copy circuit does **not** form under an unweighted (weight-1) or lightly-weighted
(weight-2) objective — all three weight-1 seeds sit near the base-rate floor and fail the cue-conflict
verdict. It forms only once copy positions are upweighted to **≥ 4**. The 8× default is therefore a
load-bearing forcing signal, and the earlier "circuit appears unbidden" framing is **withdrawn** and
replaced by this measurement. The transition between weight 2 and 4 is itself consistent with the
shortcut-gap hypothesis: the circuit appears when the objective's demand on the copy positions crosses
a threshold.

**Open:** the exact threshold (a finer grid between 2 and 4), and multi-seed replication at weights 2/4,
would tighten the dose-response. The circuit's trace (five prefix-matching heads) is unaffected — it was
always measured on the weight-8 model.
