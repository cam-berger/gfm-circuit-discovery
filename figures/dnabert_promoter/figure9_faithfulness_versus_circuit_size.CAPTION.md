# Figure 9 — standalone faithfulness versus circuit size

**Status: canonical.** Controls `DBP7`. The study's central negative result.

Mean standalone faithfulness of the candidate circuit as nodes are added in ranked order, with every
node outside the circuit ablated. Encoder Layer 2 readout, validation (n=128) and untouched test (n=128).

**There is no knee.** The five-node candidate set — `L0H8`, `L1H8`, `L2H9`, `L2H3`, `MLP0` — reproduces
`0.194` of full-model behaviour on test (`0.200` validation), marked by the dotted baseline. Growing the
head set eightfold barely moves it: size 15 with 14 heads reaches only `0.244`. The two jumps are both
**MLP additions** — `+MLP1` at size 16 (`0.244 → 0.487`) and `+MLP2` at size 39 (`0.819 → 1.000`) — which
is why the computation is described as MLP-dominated rather than head-carried.

Read together with Figure 10: the same five nodes deliver `0.882` of the corruption effect *conditionally*
under patching, and `0.194` standing alone. Both numbers are correct; only the second speaks to
compactness. Conflating them is what produced the withdrawn "compact confirmed component set" language
(`DBPR4`).

This reproduces, in a second model, the shape the SpliceBERT donor arc reached at **C33**. The agreement
is in the shape of the limit, not in any shared number.

Controlling artifacts:
`results/dnabert_promoter/frozen_claim_tests/faithfulness_curve/validation_faithfulness_curve_refined_128_curve.csv`,
`test_faithfulness_curve_confirmation_128_curve.csv`.
