# Repeat-Induction Branch Implementation

Runnable artifacts added by this branch:

| Artifact | Purpose |
|---|---|
| `src/gfmcd/data/repeat_copy.py` | Synthetic genomic repeat-copy dataset, AR targets, cue-conflict edits, copy-position metrics |
| `tests/test_repeat_copy.py` | Guards against local/base-rate/position shortcuts and verifies conflict construction |
| `scripts/train_repeat_copy.py` | Trains causal `GFMNano` with copy-position weighting |
| `scripts/verdict_repeat_copy.py` | Reports base-rate ceiling, intact copy accuracy, and distal-vs-local cue conflict |
| `scripts/trace_repeat_copy.py` | Screens heads by attention to copied source positions and loss-based ablation |

Smoke commands:

```bash
PYTHONPATH=src python scripts/train_repeat_copy.py --steps 1 --train-seqs 128 --val-seqs 64 --batch 16 --seed 99
PYTHONPATH=src python scripts/verdict_repeat_copy.py --checkpoint checkpoints/repeat_copy_s99_step1.pt --seqs 64 --batch 16
PYTHONPATH=src python scripts/trace_repeat_copy.py --checkpoint checkpoints/repeat_copy_s99_step1.pt --seqs 16 --batch 8 --top 3
```

Verification run:

```bash
pytest -q tests/test_repeat_copy.py
python -m py_compile src/gfmcd/data/repeat_copy.py scripts/train_repeat_copy.py scripts/verdict_repeat_copy.py scripts/trace_repeat_copy.py
```

The one-step smoke checkpoint is expected to return `NO COPY CIRCUIT`; it only verifies that the
pipeline is wired correctly.

## Full Run, Seed 0

Command:

```bash
PYTHONPATH=src python scripts/train_repeat_copy.py --steps 8000 --train-seqs 120000 --val-seqs 6000 --batch 64 --window 256 --copy-weight 8 --seed 0
```

Runtime: 32.7 minutes on MPS.

Learning curve:

| Step | Copy Accuracy | All-Token Accuracy |
|---:|---:|---:|
| 1 | 0.2449 | 0.2277 |
| 3000 | 0.3392 | 0.2575 |
| 3250 | 0.3753 | 0.2598 |
| 3500 | 0.5339 | 0.2694 |
| 3750 | 0.7363 | 0.2822 |
| 4000 | 0.8259 | 0.2895 |
| 8000 | 0.9611 | 0.3030 |

Main verdict on `checkpoints/repeat_copy_s0_step8000.pt`:

```text
base-rate copy ceiling                        0.2516
copy accuracy, intact                         0.9608

cue conflict on copy positions (single)
  follows DISTAL edited copy                  0.9635
  follows LOCAL/original continuation         0.0130
  neither                                     0.0235

VERDICT: COPY CIRCUIT FORMED
```

Trace screen:

```text
base copy loss: 0.1001

top attention-to-source heads:
L1H7  attn->src 0.7427  d_copy_loss +0.6753
L1H0  attn->src 0.6831  d_copy_loss +0.7177
L1H5  attn->src 0.6755  d_copy_loss +0.7017
L1H1  attn->src 0.6682  d_copy_loss +0.3108
L1H6  attn->src 0.6475  d_copy_loss +0.4557
```

Important assay note: the default verdict now uses a **single-edit conflict**, editing one source
continuation per sequence and scoring only the matching prediction. The earlier all-edit conflict
edits many positions inside the first occurrence at once, which corrupts the source prefix used as
the key for later copy positions. It remains useful as a stress test, but not as the main causal
verdict.
