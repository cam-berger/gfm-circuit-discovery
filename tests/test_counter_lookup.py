"""Counter-vs-lookup stimulus generators (TO_FIX.md §5.4).

Verifies the generators build correctly-structured stimuli: anchor at the intended position, frame
labels = (q - anchor) % 3, unique anchor (no upstream ATG), and the discriminating perturbations
(joint translation, graded indel) with the right competing frame hypotheses.
"""

from __future__ import annotations

import numpy as np

from gfmcd.data.counter_lookup import (
    ATG,
    IGNORE,
    anchored_window,
    graded_indel,
    joint_translation_pair,
    variable_anchor_batch,
)


def test_anchor_at_position_and_frame_labels():
    rng = np.random.default_rng(0)
    toks, frame = anchored_window(anchor_pos=20, length=120, rng=rng)
    assert np.array_equal(toks[20:23], ATG)
    # frame is (q - 20) % 3 downstream, IGNORE upstream
    assert (frame[:20] == IGNORE).all()
    q = np.arange(120)
    assert np.array_equal(frame[20:], (q[20:] - 20) % 3)


def test_no_upstream_atg_unique_anchor():
    rng = np.random.default_rng(1)
    for _ in range(50):
        toks, _ = anchored_window(anchor_pos=30, length=100, rng=rng)
        upstream = toks[:30]
        atg_hits = [i for i in range(len(upstream) - 2) if tuple(upstream[i : i + 3]) == tuple(ATG)]
        assert not atg_hits, "5' filler must contain no ATG"


def test_variable_anchor_batch_positions_and_heldout_split():
    positions = [10, 20, 40, 60]
    toks, frames, anchors = variable_anchor_batch(positions, n_per=3, length=128, seed=2)
    assert toks.shape == (12, 128) and frames.shape == (12, 128)
    assert set(anchors.tolist()) == set(positions)
    # each window's ATG sits at its recorded anchor
    for t, a in zip(toks, anchors):
        assert np.array_equal(t[a : a + 3], ATG)


def test_joint_translation_shares_frame_but_moves_absolute_index():
    out = joint_translation_pair(anchor_pos=15, shift=9, length=160, readout_offset=64, seed=3)
    assert np.array_equal(out["window_a"][15:18], ATG)
    assert np.array_equal(out["window_b"][24:27], ATG)  # 15 + 9
    # same relative offset -> same true frame; absolute readout index differs by the shift
    assert out["readout_b"] - out["readout_a"] == 9
    assert out["shared_frame"] == 64 % 3


def test_graded_indel_competing_frames():
    out = graded_indel(anchor_pos=12, length=256, k=1, readout_pos=200, seed=4)
    assert out["intact"].shape == out["edited"].shape  # length preserved (tail truncated)
    # anchor frame counts from the ATG after the indel; content frame is shifted right by k
    assert out["anchor_frame"] == (200 - 12) % 3
    assert out["content_frame"] == (200 - 12 - 1) % 3
    assert out["anchor_frame"] != out["content_frame"]  # k=1 => they disagree by one codon


def test_graded_indel_k3_frames_agree():
    # a full-codon (k=3) insertion keeps content in frame: anchor and content agree
    out = graded_indel(anchor_pos=12, length=256, k=3, readout_pos=200, seed=5)
    assert out["anchor_frame"] == out["content_frame"]
