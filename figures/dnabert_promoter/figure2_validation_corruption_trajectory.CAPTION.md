# Figure 2 — validation corruption trajectory

**Status: canonical.** Controls `DBP5`.

Promoter-margin damage from mononucleotide-shuffle corruption, traced layer by layer through the frozen
encoder on the validation cohort, read at the Layer-2 probe. The companion `_drops` panel shows the
per-layer change rather than the cumulative level.

Establishes that corruption damage is measurable and graded across depth — the behavioural handle that the
component interventions are normalized against. It localizes nothing on its own: a trajectory over layers
is not a statement about which heads or MLPs carry the effect.

Controlling artifact:
`results/dnabert_promoter/behavioral/all_layer_corruption/validation_full_trajectory.csv`.
