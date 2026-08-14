# Colab package — R4 paired edge-level ACDC

Runs [`PREREGISTRATION_R4_EDGE_LEVEL_CAUSAL_CORE.md`](../preregistrations/PREREGISTRATION_R4_EDGE_LEVEL_CAUSAL_CORE.md)
on a Colab GPU. `AC4` partitions the 96 heads; R4 asks the same question over the 4,636
head-granular edges, on R3's cohort, both directions in one process.

## Why this needs a GPU

The edge kernel measures **15.3× a plain forward** on the local reference device (144 ms/example
at S=1024). That is intrinsic, not a defect: edge granularity requires a separate residual per
destination, so the LayerNorm recursion runs over `[n_dst, B, S, 512]` tensors ~528 times per
forward against 12. At that speed both arms' τ sweeps are ~120 h local-equivalent.

## What is bundled

All of `src/gfmcd/**.py`, the R4 and R3 runners, the six relevant test modules, the committed
cross-device reference panel, the study's claim ledger and protocol, and the 75 MB checkpoint.

**No GTF and no FASTA** — Colab needs neither the 3.2 GB genome nor the annotation.

**The seal is enforced by absence.** `build_bundle.py` refuses to write a bundle containing any
chr8 or chr9 record, so a remote runtime cannot spend the seal by accident.

## Order

0. Mount Drive → verify bundle sha256 → assert CUDA → install deps **without replacing torch**
1. Library tests on the GPU runtime
2. **Cross-device floor → this sets the τ grid.** `min(τ) = 3 × measured floor`
3. Calibration: determinism, mode equivalence, node-capped run
4. Benchmark: GPU, peak memory, projected hours per τ per arm
5. **You set `LAUNCH_TARGET = True`** → resumable target sweep
6. Validate → copy result **and manifest** back into `results/splicebert_acceptor_coupling/`

## The two things that must not be traded away

**τ is measured, not chosen.** The Atlas package gates cross-device agreement at 2e-3 and A3
accepts `(1e-6, 1e-4]`; an a-priori grid starting at 3e-4 would sit *below* that floor and yield
a circuit that differs by accelerator with nothing disclosing it. If the measured floor leaves no
informative range, **stop and report an instrument limit** rather than lowering the multiple.

**Cut both arms equally.** If cost exceeds budget, trim the τ grid or the cohort — the same
amount on each direction. Giving the acceptor arm less makes the paired partition uninterpretable,
which is the one thing R4 exists to produce.

## Build

```bash
uv run python studies/splicebert_acceptor_coupling/colab_r4_package/build_bundle.py
```

Writes `dist/gfm_r4_edge_acdc_colab_v1.zip` plus a `.sha256.json` sidecar, and refuses to
overwrite either.
