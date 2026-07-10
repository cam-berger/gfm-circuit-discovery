# DATA_CARD — <study name>

- **Source + version.** Reference build, release, retrieval date; pin in `data/references/manifest.json`.
- **Construction.** Generator script + seed; window definition; filters (e.g. exclude ambiguous bases).
- **Splits.** Group-by-gene (or equivalent) to stop related items crossing supervised train/test. A
  chromosome split is supervised-eval hygiene, **not** a pretraining holdout for a pretrained model.
- **Matching diagnostics.** For decoys/cue conflicts: what local cues are matched, and the measured
  residual. If matching fails, the set is not a cue conflict.
- **Provenance.** Cache `.meta.json` sidecars record generator, seed, source checksums, shapes, git.
