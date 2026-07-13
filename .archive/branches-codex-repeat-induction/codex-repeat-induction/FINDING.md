# Finding: Prefix-Keyed Genomic Repeat Copying

## One Finding

`GFMNano` learns a **prefix-keyed genomic repeat-copy circuit** when the objective demands a
shortcut-proof copy computation. The learned behavior is induction-family, but the finding is not
merely "a classical induction head exists." The circuit copies a base from a previous occurrence of a
random DNA unit after matching a multi-base prefix in a four-letter alphabet, survives a cue conflict
where local continuation and distal source disagree, and is carried by several load-bearing L1 heads.

Evidence from `checkpoints/repeat_copy_s0_step8000.pt`:

```text
base-rate copy ceiling                        0.2516
copy accuracy, intact                         0.9608

cue conflict on copy positions (single)
  follows DISTAL edited copy                  0.9635
  follows LOCAL/original continuation         0.0130
  neither                                     0.0235

VERDICT: COPY CIRCUIT FORMED
```

Replication: `checkpoints/repeat_copy_s99_step8000.pt` reaches copy accuracy 0.9662 and follows the
distal edited copy at 0.9640 vs local continuation 0.0140.

Ceiling audit:

```text
base-rate                         0.2477
local naive Bayes, 33 nt crop      0.2484
RF-limited CNN, 33 nt crop         0.2473
source oracle                      1.0000
GFMNano copy circuit               0.9608
```

The registered stop-condition did not fire: the strongest no-source baseline remained at chance.

Training dynamics:

```text
step 3000   copy 0.3392
step 3250   copy 0.3753
step 3500   copy 0.5339
step 3750   copy 0.7363
step 4000   copy 0.8259
step 8000   copy 0.9611
```

Trace screen (initial single-head screen; the controlled ablation run of record is in
`COPY_TRACE.md`, which uses a fixed eval batch — base loss 0.0930 — and adds the subset-ablation
and prefix-scramble controls. Both runs agree on the ranking and on all headline claims; cite
`COPY_TRACE.md` for the exact per-head deltas):

```text
base copy loss: 0.1001

L1H7  attn->source 0.7427   ablation delta +0.6753
L1H0  attn->source 0.6831   ablation delta +0.7177
L1H5  attn->source 0.6755   ablation delta +0.7017
L1H1  attn->source 0.6682   ablation delta +0.3108
L1H6  attn->source 0.6475   ablation delta +0.4557
```

QK-IG edge check, L1:

```text
head   source-edge share   control-edge share
L1H5        0.3504              0.0037
L1H7        0.3465              0.0062
L1H0        0.3401              0.0034
L1H1        0.3350              0.0050
L1H6        0.3264              0.0036
```

Prefix-scramble control:

```text
single conflict             distal 0.9648   local 0.0117
source prefix scrambled     distal 0.2617   local 0.2344
```

Scrambling the source prefix collapses distal-following, so the source continuation is selected by a
matched-prefix relation rather than a fixed position or base-rate artifact.

Natural hg38 repeat transfer:

```text
                    seed 0    seed 99
intact copy acc      0.8780    0.8700
distal conflict      0.7820    0.7830
local conflict       0.1310    0.1330
prefix-scrambled
  distal             0.2330    0.2280
```

The synthetic-trained circuit partially transfers to exact natural repeats in chr17-19: it follows
the distal edited source far above the local continuation, and source-prefix scrambling collapses the
effect. This is a bridge-to-biology result, not yet a claim about all repeat classes.

## Specific Example

Checkpoint: `repeat_copy_s0_step8000.pt`, sampled row 97 from seed 4242.

The model sees two copies of a 20 nt unit. The second copy has revealed a 14 nt prefix. The task is to
predict the next base.

```text
first unit, intact       CCCTCATATTGCCTTCCCTT
second unit, local       CCCTCATATTGCCTTCCCTT
first unit, conflicted   CCCTCATATTGCCTACCCTT

prefix used as key       CCCTCATATTGCCT
second-copy prefix       CCCTCATATTGCCT

local/original target    T
distal edited target     A
model prediction         A

model probabilities:
  A 0.9976   C 0.0008   G 0.0008   T 0.0008
```

At the prediction position, these heads attend to the edited distal continuation base:

```text
L1H0  attention to source A = 0.9060
L1H5  attention to source A = 0.9100
L1H6  attention to source A = 0.8918
L1H7  attention to source A = 0.7613
L1H4  attention to source A = 0.6727
```

So the local second copy still says the continuation should be `T`, but the edited distal source says
it should be `A`. The model predicts `A` almost deterministically and the heads attend to that edited
source base.

## Why This Is Not Just The Classical Toy Claim

Classical induction is usually summarized as:

```text
... A B ... A -> predict B
```

This branch tests a stricter genomic version:

```text
... u[:j] u[j] ... u[:j] -> predict u[j]
```

The differences that matter:

1. The key is a **14 nt prefix** in the concrete example, not a single rich token identity.
2. The alphabet has only four bases, so local and base-rate shortcuts are unusually tempting; the
   measured base-rate ceiling is only 0.2516 and the conflict separates copy from local continuation.
3. The copied unit is random per sequence, so success cannot be a memorized motif dictionary.
4. The causal verdict edits the distal source while preserving the local second-copy context; the
   model follows the distal edit at 0.9635 and the local continuation at 0.0130.
5. The trace is distributed across several L1 source-copy heads, with loss-based ablation deltas and
   QK-IG concentration on source-copy edges, rather than a single named toy head selected by
   attention alone.
6. The same trained mechanism partially transfers to exact natural hg38 repeats under the same cue
   conflict; it is not confined to synthetic random strings.

The honest wording is: **this is an induction-family mechanism, instantiated as a genomic
prefix-matching copy circuit and validated by cue conflict plus intervention.**
