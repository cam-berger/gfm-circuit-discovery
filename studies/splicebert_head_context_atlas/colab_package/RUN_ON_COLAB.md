# Run the all-head screen on Colab — 5 steps

This zip is self-contained except the two large public data files (GTF + hg38), which a script fetches.

## What's inside
```
gfm_head_screen/
  RUN_ON_COLAB.md                 <- this file
  repo/                           <- the code + bundled model + reference results
    src/gfmcd/…                   <- the package (imported via sys.path)
    scripts/splicebert_head_context_atlas/{run_all_head_screen,validate_head_screen,merge_head_screen,benchmark_colab}.py
    tests/{test_competition,test_head_context_screen}.py
    data/models/…/SpliceBERT.1024nt/   <- BUNDLED 75 MB checkpoint (sha 2ad91428…)
    studies/splicebert_head_context_atlas/…   <- docs + colab_package (notebook, configs)
    results/splicebert_head_context_atlas/*.json  <- census + pilots (reference)
```

## Steps
1. **Upload** `gfm_head_screen_colab.zip` to your Drive at `MyDrive/gfm_head_screen_colab.zip`.
2. Open `repo/studies/splicebert_head_context_atlas/colab_package/run_head_screen_colab.ipynb` in Colab
   (**Runtime → GPU**). Run cells 1–4: mount, unpack, install, **fetch GTF + hg38** (~3 GB; cached to Drive
   so later sessions skip it), verify hashes.
3. Run cell 5 (**pytest + 1-example smoke**, all 96 heads, all 12 mechanical gates) and cell 6
   (**benchmark**, 12 examples → GPU type, peak mem, examples/hr, **projected full-run GPU-hours + storage**).
4. **Review the benchmark projection.** Only then set `LAUNCH_FULL = True` in cell 7 and run the full
   discovery screen (shards write to Drive → resumable across session timeouts).
5. Cell 8: **validate + merge** → `head_effect_matrix.jsonl` + `head_effect_summary.json` (per-head effects
   + the C33 group/faithfulness summary) in your Drive shards folder.

## Guarantees baked in
- chr8/9 are **hard-excluded** (a discovery run cannot touch the sealed set).
- The runner **refuses to write** if any of the 12 mechanical gates fail.
- **Resumable**: a re-run skips completed shards; the manifest is rebuilt from disk.
- **Deterministic**: `PYTHONHASHSEED=0` + sorted cohort; float32, TF32/AMP off.
- C is labeled **competitor-site perturbation response** (not competition-specific).

## Boundary
Exploratory head-effect matrix only. Do not read it as competition-specificity, biological function, or
sealed-chromosome generalization; a single-head null is not evidence against a distributed mechanism (C33 —
see the group summary). No visualizer yet.
