# BASELINES — <study name>

**Registered shortcut class (3).** The specific pooled family the circuit must beat. Not "all pooled
statistics" — an enumerated list (local motif score, additive score, distance-only, k-mer logistic,
bag/DeepSets, receptive-field-limited CNN, explicit-interaction model, …).

**Shortcut ceilings (4).** Measure the best baseline's accuracy **before tracing anything**. If it
meets what the objective demands, there is no circuit to trace — say so and stop.

```
shortcut_gap = loss(best registered shortcut) - loss(best relational solution)
```

Report each baseline's ordinary test performance, cue-conflict performance, calibration, and
performance as matching strictness increases.
