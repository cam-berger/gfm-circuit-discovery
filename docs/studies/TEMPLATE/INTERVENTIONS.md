# INTERVENTIONS — <study name>

**Scalar causal metric (6).** One number the whole study turns on (e.g. clean−corrupt relational
score). Loss-based where accuracy saturates.

**Intervention targets (7).** Heads / MLP features / residual directions nominated by attribution —
which *ranks*, never decides.

**Necessity (8).** Ablate the nominated set; report Δ vs (a) matched random sets, (b) same-size
high-attention-but-low-causal sets, (c) irrelevant-position edits. Every nominated component reported
with a paired control, including inert ones (attribution ranks; intervention decides).

**Sufficiency / mediation (9).** Patch clean→corrupt through the path:
`recovery = (patched − corrupt) / (clean − corrupt)`, with a bootstrap CI and matched-random-path
comparison.

**Specificity.** The circuit responds to cognate relations more than motif-matched decoys and
matched-distance random pairs.

House rules: a saturated measurement reports nothing where the effect is largest; a fixed-layer probe
measures whatever that layer carries — use a behavioral output readout where one exists.
