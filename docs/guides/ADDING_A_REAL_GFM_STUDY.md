# Adding a Real-GFM Circuit Study

1. Copy `studies/TEMPLATE_REAL_GFM_STUDY` to `studies/<study_id>`.
2. Define one behavior and one biological relation. Avoid a list of unrelated targets.
3. Write the strongest local/additive shortcut class before model experiments.
4. Specify natural and controlled cue conflicts.
5. Implement an adapter under `src/gfmcd/adapters/` that validates token alignment, layer access, and model metadata.
6. Put domain-specific schemas, datasets, baselines, perturbations, and metrics in a dedicated source package.
7. Register zero-shot, frozen-readout, random-control, and fine-tuned conditions.
8. Add tests for coordinates, strand/orientation, token mapping, and intervention hooks.
9. Freeze validation conflict examples before circuit tracing.
10. Promote claims only through behavior, necessity, mediation, path, replication, and external validation.

The study README must include a stop rule for each possible negative outcome. A missing circuit is a result when the assay and shortcuts are well specified.
