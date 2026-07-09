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

Training dynamics:

```text
step 3000   copy 0.3392
step 3250   copy 0.3753
step 3500   copy 0.5339
step 3750   copy 0.7363
step 4000   copy 0.8259
step 8000   copy 0.9611
```

Trace screen:

```text
base copy loss: 0.1001

L1H7  attn->source 0.7427   ablation delta +0.6753
L1H0  attn->source 0.6831   ablation delta +0.7177
L1H5  attn->source 0.6755   ablation delta +0.7017
L1H1  attn->source 0.6682   ablation delta +0.3108
L1H6  attn->source 0.6475   ablation delta +0.4557
```

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
5. The trace is distributed across several L1 source-copy heads, with loss-based ablation deltas,
   rather than a single named toy head selected by attention alone.

The honest wording is: **this is an induction-family mechanism, instantiated as a genomic
prefix-matching copy circuit and validated by cue conflict plus intervention.**
