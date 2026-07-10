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

## Full Run, Seed 99

Command:

```bash
PYTHONPATH=src python scripts/train_repeat_copy.py --steps 8000 --train-seqs 120000 --val-seqs 6000 --batch 64 --window 256 --copy-weight 8 --seed 99
```

Runtime: 37.4 minutes on MPS. Final copy accuracy: 0.9698.

Main verdict on `checkpoints/repeat_copy_s99_step8000.pt`:

```text
base-rate copy ceiling                        0.2516
copy accuracy, intact                         0.9662

cue conflict on copy positions (single)
  follows DISTAL edited copy                  0.9640
  follows LOCAL/original continuation         0.0140
  neither                                     0.0220

VERDICT: COPY CIRCUIT FORMED
```

Phase A outputs:

```text
branches/codex-repeat-induction/fig_repeat_grok.png
branches/codex-repeat-induction/fig_data/repeat_grok.csv
```

Train-vs-validation CE diagnostic:

```bash
PYTHONPATH=src python scripts/repeat_copy_loss_curves.py --seeds 0,99 --subset-seqs 6000 --batch 64
```

This recomputes comparable unweighted CE on fixed train/validation subsets at the saved checkpoints.
The `train` field in the training history is a weighted minibatch loss, so it is not used for this
comparison.

Final checkpoint:

```text
seed   train all CE   val all CE   train copy CE   val copy CE   val copy acc
s0       1.3161        1.3165        0.0909         0.0947        0.9611
s99      1.3175        1.3170        0.0818         0.0799        0.9698
```

Outputs:

```text
branches/codex-repeat-induction/LOSS_CURVES.md
branches/codex-repeat-induction/fig_repeat_loss_vs_copy.png
branches/codex-repeat-induction/fig_data/repeat_loss_curves.csv
```

## Phase B Ceiling Audit

Command:

```bash
PYTHONPATH=src python scripts/repeat_copy_ceiling.py --device cpu --train-seqs 20000 --val-seqs 6000 --cnn-epochs 8 --batch 1024 --crop 33 --model-acc 0.9608
```

Result:

```text
base-rate                         0.2477
local naive Bayes, 33 nt crop      0.2484
RF-limited CNN, 33 nt crop         0.2473
source oracle                      1.0000
GFMNano copy circuit               0.9608
```

The registered stop-condition did not fire. The RF-limited CNN remained at chance while denied source
access.

Outputs:

```text
branches/codex-repeat-induction/COPY_CEILING.md
branches/codex-repeat-induction/fig_copy_ceiling.png
branches/codex-repeat-induction/fig_data/copy_ceiling.csv
branches/codex-repeat-induction/fig_data/copy_ceiling_cnn_history.csv
```

## Phase C Trace Controls

Commands:

```bash
PYTHONPATH=src python scripts/repeat_copy_trace_controls.py --checkpoint checkpoints/repeat_copy_s0_step8000.pt --seqs 1024 --batch 64
PYTHONPATH=src python scripts/repeat_copy_qk_ig.py --checkpoint checkpoints/repeat_copy_s0_step8000.pt --seqs 64 --layer 1 --steps 16
```

Results:

```text
single conflict             distal 0.9648   local 0.0117
source prefix scrambled     distal 0.2617   local 0.2344
```

Scrambling the source prefix collapses distal-following, supporting prefix-keying.

Head ablation:

```text
base copy loss       0.0930
ablate L1H7          +0.6623
ablate L1H0          +0.7343
ablate L1H5          +0.7302
ablate L1H1          +0.3377
ablate L1H6          +0.4482
ablate top3          +1.3030
ablate nominated     +1.3226
sum singles          +2.9128
```

The five heads are causal, but not independent: subset cost is far below the sum of single-head
costs.

QK-IG:

```text
head   source-edge share   control-edge share
L1H5        0.3504              0.0037
L1H7        0.3465              0.0062
L1H0        0.3401              0.0034
L1H1        0.3350              0.0050
L1H6        0.3264              0.0036
```

Outputs:

```text
branches/codex-repeat-induction/COPY_TRACE.md
branches/codex-repeat-induction/COPY_QK_IG.md
branches/codex-repeat-induction/fig_copy_heads.png
branches/codex-repeat-induction/fig_data/copy_head_ablation.csv
branches/codex-repeat-induction/fig_data/copy_prefix_control.csv
branches/codex-repeat-induction/fig_data/copy_prefix_attention.csv
branches/codex-repeat-induction/fig_data/copy_qk_ig.csv
```

## Phase D Natural hg38 Repeats

Command, seed 0 checkpoint:

```bash
PYTHONPATH=src python scripts/repeat_copy_natural.py --checkpoint checkpoints/repeat_copy_s0_step8000.pt --seqs 1000 --chroms chr17,chr18,chr19 --seed 9090 --batch 64
```

Command, seed 99 checkpoint:

```bash
PYTHONPATH=src python scripts/repeat_copy_natural.py --checkpoint checkpoints/repeat_copy_s99_step8000.pt --seqs 1000 --chroms chr17,chr18,chr19 --seed 9191 --batch 64 --out-dir branches/codex-repeat-induction/natural_seed99
```

The sampler uses exact naturally occurring 13-mers (`prefix=12` plus continuation), source distance
48-160 nt, at least three distinct bases, and max base fraction <= 0.7. It reads `hg38.fa.fai`
directly, so it does not depend on `pyfaidx`.

Results:

```text
                    seed 0    seed 99
base-rate            0.3500    0.3430
intact copy acc      0.8780    0.8700
distal conflict      0.7820    0.7830
local conflict       0.1310    0.1330
prefix-scrambled
  distal             0.2330    0.2280
```

The synthetic-trained circuit partially transfers to exact natural repeats, and the effect remains
prefix-keyed: source-prefix scrambling collapses distal-following.

Outputs:

```text
branches/codex-repeat-induction/NATURAL_REPEATS.md
branches/codex-repeat-induction/fig_natural_repeats.png
branches/codex-repeat-induction/fig_data/natural_repeat_summary.csv
branches/codex-repeat-induction/fig_data/natural_repeat_examples.csv
branches/codex-repeat-induction/fig_data/natural_repeat_attention.csv
branches/codex-repeat-induction/natural_seed99/
```
