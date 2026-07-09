"""Repeat-copy task: a shortcut-proof, easy-to-trace induction mechanism."""

import numpy as np
import pytest

from gfmcd.data.repeat_copy import (
    IGNORE,
    ar_inputs_targets,
    base_rate_ceiling,
    conflict_batch,
    make_batch,
    single_conflict_batch,
)


@pytest.fixture(scope="module")
def batch():
    return make_batch(n_seq=2000, window=256, seed=0)


def test_batch_shapes(batch):
    assert batch.tokens.shape == (2000, 256)
    assert batch.copy_mask.shape == (2000, 255)
    assert batch.source_index.shape == (2000, 255)
    assert batch.tokens.min() >= 0 and batch.tokens.max() < 4


def test_every_sequence_has_copy_supervision(batch):
    per_seq = batch.copy_mask.sum(axis=1)
    assert per_seq.min() >= 6
    assert per_seq.max() <= 20


def test_repeated_unit_is_actually_repeated(batch):
    for r in range(100):
        s1 = batch.first_start[r]
        s2 = batch.second_start[r]
        n = batch.unit_len[r]
        assert np.array_equal(batch.tokens[r, s1 : s1 + n], batch.tokens[r, s2 : s2 + n])


def test_copy_source_points_to_matching_continuation(batch):
    inputs, targets, mask = ar_inputs_targets(batch)
    assert inputs.shape == targets.shape == mask.shape == (2000, 255)

    rows, pred_positions = np.nonzero(mask[:200])
    for r, p in zip(rows, pred_positions):
        src = batch.source_index[r, p]
        assert src != IGNORE
        assert batch.tokens[r, src] == targets[r, p]
        # The prefix token immediately before the copied base matches too: the induction key.
        assert batch.tokens[r, src - 1] == batch.tokens[r, p]


def test_base_rate_ceiling_is_near_chance(batch):
    ceiling = base_rate_ceiling(batch)
    assert 0.24 < ceiling < 0.28


def test_conflict_edits_distal_copy_not_local_copy(batch):
    edited, distal, local = conflict_batch(batch)
    assert edited.shape == batch.tokens.shape
    assert distal.shape == local.shape == batch.copy_mask.shape

    rows, pred_positions = np.nonzero(batch.copy_mask[:100])
    for r, p in zip(rows, pred_positions):
        src = batch.source_index[r, p]
        assert edited[r, src] == distal[r, p]
        assert distal[r, p] != local[r, p]
        assert batch.tokens[r, p + 1] == local[r, p]
        assert edited[r, p + 1] == local[r, p], "the second/local occurrence must stay unchanged"

    assert (distal[~batch.copy_mask] == IGNORE).all()
    assert (local[~batch.copy_mask] == IGNORE).all()


def test_single_conflict_keeps_scored_key_intact(batch):
    edited, distal, local, score_mask = single_conflict_batch(batch, seed=1)
    assert score_mask.shape == batch.copy_mask.shape
    assert score_mask.sum() == len(batch.tokens)

    rows, pred_positions = np.nonzero(score_mask[:200])
    for r, p in zip(rows, pred_positions):
        src = int(batch.source_index[r, p])
        assert distal[r, p] != local[r, p]
        assert edited[r, src] == distal[r, p]
        assert edited[r, p + 1] == local[r, p]
        # The source prefix used as the key is unchanged.
        start = batch.first_start[r]
        assert np.array_equal(edited[r, start:src], batch.tokens[r, start:src])

    assert (distal[~score_mask] == IGNORE).all()
    assert (local[~score_mask] == IGNORE).all()


def test_position_index_does_not_identify_copy_target(batch):
    _, targets, mask = ar_inputs_targets(batch)
    pos = np.tile(np.arange(mask.shape[1]), (len(mask), 1))
    counts = np.zeros((mask.shape[1], 4), dtype=int)
    np.add.at(counts, (pos[mask], targets[mask]), 1)
    live = counts.sum(axis=1) > 10
    assert counts[live].max(axis=1).sum() / counts[live].sum() < 0.31
