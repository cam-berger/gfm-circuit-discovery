# Codex Branch: Mechanism Choice

## Decision

Choose **in-context repeat copying**, implemented as a genomic repeat-induction task.

This is the next mechanism I would branch on for circuit discovery in GFMs. It is not the final
biological flagship. It is the fastest clean bridge between the repository's current frame-counter
result and the higher-value Tier 3/4 biology tasks.

## Why This One

The proposal's central rule is:

```text
circuit forms iff objective demand > ceiling of the best bounded-window pooled statistic
```

The mechanism ladder adds one special case: in-context repeat copying is both shortcut-proof and easy
to trace. That combination is rare. Exon definition, NMD, and CTCF orientation are more biologically
important, but they require more data plumbing and harder tracing. Repeat induction gives a cheap
positive mechanism whose expected circuit is a single, named object: an induction-style QK edge.

So the branch goal is:

> Build and trace a repeat-copying circuit in GFM-nano, then use the exact same ceiling-audit and
> cue-conflict language as the proposal.

## Mechanism

Given a random DNA unit `u` and a later partial recurrence of that unit, the model must copy the next
base from the earlier occurrence:

```text
... u[0:k] u[k] ... spacer ... u[0:k] [MASK or next-token target]
                              local prefix says "same key"
                              distal copy says "the next base is u[k]"
```

For a nucleotide model, the alphabet is small, so the guardrail is essential: the unit and matching
prefix must be long enough that a local k-mer rule cannot memorize the continuation. The useful
computation is not "this local motif predicts A"; it is "find the previous matching prefix and copy
the following base."

## Initial Experimental Shape

Use `GFMNano` because the repo already has activation caching, QK attribution, and head ablations for
that architecture.

Start with a synthetic diagnostic corpus before touching natural repeats:

1. Sample random DNA units, length 16-32 nt.
2. Place two occurrences in a 256 nt window with a random spacer.
3. Train a causal next-token variant first, because MLM right-context can leak the answer unless we
   use block masking.
4. Score only positions in the second occurrence where the local prefix is insufficient to predict
   the next base.
5. Keep unit identities held out across train and validation, so success cannot be a memorized
   dictionary.

After the synthetic circuit is traced, repeat the assay on natural tandem/interspersed repeat windows
as an external-validity check, not as the first proof.

## Ceiling Audit

Before tracing, measure shortcut ceilings:

| Baseline | What It Tests | Expected Result |
|---|---|---|
| unigram / k-mer bag | composition shortcut | near chance |
| local naive Bayes over +/- w | bounded-window local rule | near chance when the earlier copy is outside `w` |
| RF-limited CNN | stronger bounded-window shortcut | should fail until its receptive field reaches the earlier copy |
| transformer with copy access | relational route | high accuracy |

The mechanism is worth tracing only if the bounded-window ceilings remain below the demanded copy
accuracy.

## Cue Conflict

Use a conflict that preserves the visible local prefix while changing the distal copied continuation.

```text
intact:
  first copy   ACGT...C
  second copy  ACGT...[target should be C]

conflict:
  first copy   ACGT...G
  second copy  ACGT...[target still locally wants C]
```

Verdict:

- **Copy circuit:** prediction follows the edited first copy.
- **Local/statistical shortcut:** prediction stays with the second-copy continuation or falls to base
  rates.

This mirrors the frame cue conflict: make the relational cue and local content disagree, then force
the model to pick a side.

## Trace Target

Expected circuit:

1. A QK head matches the current local prefix/key to the previous occurrence.
2. The OV path copies the base following the matched prefix.
3. Later MLP/readout components convert copied base information into the nucleotide prediction.

Primary causal checks:

- attention from second occurrence to the corresponding first occurrence,
- QK integrated gradients concentrated on the matching prefix,
- head ablation increases loss specifically on repeat-copy positions,
- conflict assay follows the distal edited copy,
- matched decoy repeats with same composition do not trigger the same route.

## Why Not The Other Rungs First

**NMD 50-55 nt rule** is my preferred biological follow-up. It has a clean distance-threshold relation
and a natural cue conflict, but it needs exon-junction-specific data construction and probably a
SpliceBERT or transcript-style model path that is not yet in this repo.

**CTCF orientation** is the sharpest flagship. It is also the best test of the RC-equivariance
blindness claim. I would not start there because the data/model stack and hybrid tracing are less
ready than GFM-nano's transformer path.

**Exon definition** is biologically central but trace-hard: two anchors, spacing, and splice-strength
balancing. It is better after the ceiling-audit machinery exists for a simpler relational task.

**G-quadruplex** stays deprioritized. The ladder and proposal agree that G-density is a strong pooled
proxy, so the ceiling audit may say there is no circuit to trace.

## Stop Conditions

Stop or demote this branch if any of these happen:

- a bounded-window CNN reaches the demanded copy accuracy without access to the first occurrence,
- cue conflict does not move predictions toward the edited first copy,
- the apparent induction head is not causal under loss-based ablation,
- performance depends on repeated fixed motifs rather than held-out random units,
- the synthetic result cannot be stated as a bridge/control and starts being mistaken for the main
  biological claim.

## Handoff

This branch should produce, in order:

1. `src/gfmcd/data/repeat_copy.py`
2. `tests/test_repeat_copy.py`
3. `scripts/train_repeat_copy.py`
4. `scripts/verdict_repeat_copy.py`
5. `scripts/trace_repeat_copy.py`

The first reportable artifact is not accuracy. It is the same causal verdict format used elsewhere
in the repo: ceiling audit, cue conflict, and loss-based intervention.
