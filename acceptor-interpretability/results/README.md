# results/

Empty by design as of 2026-08-22.

Every result file and figure previously in this directory was produced by a
pipeline that seeded corruption from `hash(seq)`, which CPython salts per
process. The committed JSON, the committed figures, and the study write-up were
three different runs, none of them regenerable, and they disagreed with each
other on published conclusions. They were deleted rather than reprinted; they
remain in git history at `ba01eb6` if the withdrawn values are ever needed for
comparison.

Regenerate with the repaired pipeline:

```bash
python experiments/phase0_pilot.py
python experiments/phase1_control_fix.py
python experiments/phase2_absolute_floor.py
python experiments/phase3_real_coordinates.py
python experiments/phase4_head_content_control.py
python experiments/phase5_intervention_ladder.py
```

Each runner writes `<name>_results.json`, its per-sample CSV, and a sibling
`<name>.manifest.json` recording the cohort content hash, the pinned checkpoint
revision, the seed, and the environment. Two runs of the same phase on the same
config must now produce byte-identical corruptions; if their manifests' cohort
hashes agree and their numbers do not, that is a real finding, not noise.

Figures are regenerated from these files by `visualization/visualization_*.py`.

See the audit notice in [`../README.md`](../README.md) before quoting anything
produced here.
