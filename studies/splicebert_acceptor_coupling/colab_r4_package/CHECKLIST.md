# Readiness checklist — R4 paired edge-level ACDC

## Green locally before packaging

- [x] Kernel exact on the real checkpoint at S=1024: `kernel_exact_full` 3.05e-5,
      `kernel_exact_empty` 1.53e-5, `per_head_input_identity` 2.86e-6
- [x] Edge counts pinned: 4,636 head-granular / 12,988 Q/K/V
- [x] 77 library tests pass (graph, kernel, Algorithm 1, device policy, reverse-coupling memorials)
- [x] `--smoke` preflight PASS on R3's cohort
- [x] Reference panel committed (`results/splicebert_acceptor_coupling/r4_reference_smoke.json`)

## Colab gates before the target sweep

- [ ] Bundle sha256 matches its sidecar
- [ ] CUDA runtime active; torch was **not** reinstalled
- [ ] `BUNDLE_MANIFEST.json` seal check present and chr8/9 absent
- [ ] Library tests pass on the GPU runtime
- [ ] Cross-device panel runs and `tau_above_device_floor` **passes** — this sets the τ grid.
      If the measured floor leaves no informative range, **STOP and report an instrument limit**;
      do not lower the multiple.
- [ ] `decision_determinism` passes on the first 200 decisions
- [ ] `equivalence_check` passes, or `--mode sequential` is used
- [ ] Benchmark reports GPU model, peak memory, s/condition and **projected hours per τ per arm**
- [ ] Owner reads the benchmark and sets `LAUNCH_TARGET = True`

## Stop conditions

- Any memorial fails, or `preflight_PASS` is false.
- The measured device floor forces a τ grid with no informative range.
- Projected cost materially exceeds budget → cut the τ grid or discovery `n` **equally on both
  arms**. Never give the acceptor arm less: unequal treatment makes the paired partition
  uninterpretable.
- Never substitute a chr8/9 cohort into the runner.

## After the sweep

- [ ] Both directions and both idioms present for every τ
- [ ] Partition reported **per idiom, never pooled** (D9)
- [ ] Artifact carries `is_partial_sweep: false`
- [ ] Result + manifest copied back into `results/splicebert_acceptor_coupling/`
