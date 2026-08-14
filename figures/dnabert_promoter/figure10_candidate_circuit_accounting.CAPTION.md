# Figure 10 — candidate-circuit necessity and delivery

**Status: canonical.** Controls `DBP6`.

Two conditional interventions on the five-node candidate circuit (`L0H8`, `L1H8`, `L2H9`, `L2H3`,
`MLP0`), both normalized by the same total corruption effect, at the encoder Layer-2 readout.
**Necessity** patches corrupt activations into a clean run; **delivery** patches clean activations into a
corrupt run. Validation and untouched test, n=128 each. Error bars are the standard deviation of the
per-example distribution, **not** a standard error.

Necessity mean `0.630` (test) / `0.651` (validation); delivery mean `0.882` / `0.839`. Both are positive
in 128/128 examples on both splits, and the effect replicates from validation to test.

**These bars must not be added.** They sum past 1.0 (`0.630 + 0.882 = 1.51`) and delivery maxima exceed
the grey unit line (`1.45` test, `1.36` validation), because residual and LayerNorm paths remain nonlinear
under patching. They are conditional causal effects sharing a denominator, not shares of a decomposition —
the repository forbids reporting them as additive fractions.

For what the same five nodes do *standing alone*, see Figure 9. That is the measurement that bears on
compactness, and it is far lower.

Controlling artifacts:
`results/dnabert_promoter/frozen_claim_tests/circuit_accounting/validation_candidate_circuit_128_summary.json`,
`test_candidate_circuit_confirmation_128_summary.json`.
