"""Synthetic repeat-copy task for a clean induction-style mechanism.

The task is the genomic analogue of an induction head:

    ... u[:j] u[j] ... u[:j]  -> predict u[j]

where `u` is a random DNA unit. A bounded local statistic should not know the continuation of a
held-out random unit; the relational route is to find the previous matching prefix and copy the base
that followed it.

The model is trained with the ordinary causal next-token objective. The copy mask and source indices
exist only for evaluation, cue conflict, and tracing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A, C, G, T = range(4)
NT2ID = {"A": A, "C": C, "G": G, "T": T}
ID2NT = "ACGT"
IGNORE = -100


@dataclass(frozen=True)
class RepeatCopyBatch:
    """A batch plus evaluation metadata.

    `copy_mask` is aligned to AR input positions. If `copy_mask[r, p]` is true, the model sees
    tokens up to position `p` and should predict `tokens[r, p + 1]` by copying from
    `source_index[r, p]`, the matching position in the first occurrence.
    """

    tokens: np.ndarray
    copy_mask: np.ndarray
    source_index: np.ndarray
    first_start: np.ndarray
    second_start: np.ndarray
    unit_len: np.ndarray


def make_batch(
    n_seq: int,
    window: int = 256,
    unit_range: tuple[int, int] = (18, 32),
    gap_range: tuple[int, int] = (32, 96),
    min_prefix: int = 12,
    seed: int = 0,
) -> RepeatCopyBatch:
    """Return random DNA windows with a repeated unit and copy-evaluation positions.

    The repeated unit is sampled fresh per sequence. Positions are randomised so absolute position
    embeddings cannot encode a reusable answer.
    """

    lo_u, hi_u = unit_range
    lo_g, hi_g = gap_range
    if lo_u < 2 or hi_u < lo_u:
        raise ValueError("invalid unit_range")
    if min_prefix < 1 or min_prefix >= hi_u:
        raise ValueError("min_prefix must leave at least one copied continuation")

    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 4, size=(n_seq, window), dtype=np.int8)
    copy_mask = np.zeros((n_seq, window - 1), dtype=bool)
    source_index = np.full((n_seq, window - 1), IGNORE, dtype=np.int64)
    first_start = np.zeros(n_seq, dtype=np.int64)
    second_start = np.zeros(n_seq, dtype=np.int64)
    unit_len = np.zeros(n_seq, dtype=np.int64)

    for r in range(n_seq):
        length = int(rng.integers(lo_u, hi_u + 1))
        gap = int(rng.integers(lo_g, hi_g + 1))
        max_s1 = window - (2 * length + gap)
        if max_s1 < 0:
            raise ValueError("window too small for unit_range + gap_range")

        s1 = int(rng.integers(0, max_s1 + 1))
        s2 = s1 + length + gap
        unit = rng.integers(0, 4, size=length, dtype=np.int8)

        tokens[r, s1 : s1 + length] = unit
        tokens[r, s2 : s2 + length] = unit
        first_start[r] = s1
        second_start[r] = s2
        unit_len[r] = length

        for j in range(min_prefix, length):
            pred_pos = s2 + j - 1
            if pred_pos < window - 1:
                copy_mask[r, pred_pos] = True
                source_index[r, pred_pos] = s1 + j

    return RepeatCopyBatch(tokens, copy_mask, source_index, first_start, second_start, unit_len)


def ar_inputs_targets(batch: RepeatCopyBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inputs, next-token targets, and the copy mask, all aligned to AR positions."""

    return batch.tokens[:, :-1], batch.tokens[:, 1:].astype(np.int64), batch.copy_mask


def conflict_batch(batch: RepeatCopyBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a distal-copy cue conflict.

    The first occurrence's copied continuation bases are edited. The second occurrence is left
    unchanged, so local/original continuation and distal-copy continuation disagree.

    Returns `(tokens_conflict, distal_targets, local_targets)`, where targets are aligned to AR input
    positions and non-copy positions are `IGNORE`.
    """

    edited = batch.tokens.copy()
    distal = np.full(batch.copy_mask.shape, IGNORE, dtype=np.int64)
    local = np.full(batch.copy_mask.shape, IGNORE, dtype=np.int64)

    rows, pred_positions = np.nonzero(batch.copy_mask)
    for r, p in zip(rows, pred_positions):
        src = int(batch.source_index[r, p])
        new_base = int((edited[r, src] + 1) % 4)
        edited[r, src] = new_base
        distal[r, p] = new_base
        local[r, p] = int(batch.tokens[r, p + 1])

    return edited, distal, local


def copy_accuracy(logits: np.ndarray, targets: np.ndarray, copy_mask: np.ndarray) -> float:
    """Accuracy restricted to repeat-copy positions."""

    pred = logits.argmax(axis=-1)
    return float((pred[copy_mask] == targets[copy_mask]).mean())


def base_rate_ceiling(batch: RepeatCopyBatch) -> float:
    """Best constant-base predictor on copy positions."""

    _, targets, mask = ar_inputs_targets(batch)
    counts = np.bincount(targets[mask].astype(int), minlength=4)
    return float(counts.max() / counts.sum())
