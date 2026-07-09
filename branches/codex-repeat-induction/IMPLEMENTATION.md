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
