"""Stimulus generators for the counter-vs-lookup discrimination (TO_FIX.md §5.4).

A trained anchor-relative model computes frame(q) as a function of (q - anchor). Two implementations
are behaviourally identical on unperturbed sequence: an accumulating *counter* and an anchor/position
*residue lookup*. These generators build the stimuli that pull them apart (preregistration:
`docs/studies/reading_frame/COUNTER_VS_LOOKUP.md`). Data only; the discrimination itself runs a trained
model on these (needs the forcing checkpoint — separate, compute-gated step).

Alphabet matches `uniform_cds`: A=0, C=1, G=2, T=3; the start codon ATG = [0, 3, 2]. Frame label at
position q is (q - anchor) % 3 downstream of the anchor, IGNORE (-1) upstream.
"""

from __future__ import annotations

import numpy as np

ATG = np.array([0, 3, 2], dtype=np.int8)
IGNORE = -1


def _filler_no_atg(length: int, rng: np.random.Generator) -> np.ndarray:
    """Random nucleotides containing no ATG, so the start codon is the unique anchor."""
    out = rng.integers(0, 4, size=max(length, 0)).astype(np.int8)
    i = 0
    while i + 2 < len(out):
        if out[i] == 0 and out[i + 1] == 3 and out[i + 2] == 2:
            out[i + 2] = 1  # break the ATG (G->C)
        i += 1
    return out


def anchored_window(anchor_pos: int, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A window with the ATG at `anchor_pos`, uniform codons after it, no upstream ATG.

    Returns (tokens[length], frame[length]) with frame = (q - anchor_pos) % 3 for q >= anchor_pos.
    """
    if not (0 <= anchor_pos <= length - 3):
        raise ValueError(f"anchor_pos {anchor_pos} does not fit a window of length {length}")
    tokens = np.empty(length, dtype=np.int8)
    tokens[:anchor_pos] = _filler_no_atg(anchor_pos, rng)
    tokens[anchor_pos : anchor_pos + 3] = ATG
    tail = length - (anchor_pos + 3)
    if tail > 0:
        tokens[anchor_pos + 3 :] = rng.integers(0, 4, size=tail).astype(np.int8)
    frame = np.full(length, IGNORE, dtype=np.int64)
    q = np.arange(length)
    down = q >= anchor_pos
    frame[down] = (q[down] - anchor_pos) % 3
    return tokens, frame


def variable_anchor_batch(
    positions: list[int], n_per: int, length: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test 1 (held-out anchor positions). Batch of windows with the anchor at each of `positions`.

    Returns (tokens[N, length], frame[N, length], anchor_pos[N]). Split `positions` into train/test to
    check whether the model generalises to anchor positions it never saw (a counter must; a lookup keyed
    to absolute position need not).
    """
    rng = np.random.default_rng(seed)
    toks, frames, anchors = [], [], []
    for p in positions:
        for _ in range(n_per):
            t, f = anchored_window(p, length, rng)
            toks.append(t)
            frames.append(f)
            anchors.append(p)
    return np.stack(toks), np.stack(frames), np.array(anchors, dtype=np.int64)


def joint_translation_pair(
    anchor_pos: int, shift: int, length: int, readout_offset: int, seed: int
) -> dict:
    """Test 2 (joint translation). Same window, anchor moved from `anchor_pos` to `anchor_pos+shift`.

    The readout is at a FIXED RELATIVE offset from the anchor in both, so the true frame is identical.
    A counter predicts the same frame in both windows; an absolute-position lookup, reading a different
    absolute index, does not. Returns both windows + the two absolute readout indices + the shared frame.
    """
    rng = np.random.default_rng(seed)
    ta, _ = anchored_window(anchor_pos, length, rng)
    tb, _ = anchored_window(anchor_pos + shift, length, np.random.default_rng(seed))  # same filler stream
    qa, qb = anchor_pos + readout_offset, anchor_pos + shift + readout_offset
    if qb >= length:
        raise ValueError("readout falls outside the shifted window; reduce shift/offset or grow length")
    return {
        "window_a": ta,
        "window_b": tb,
        "readout_a": qa,
        "readout_b": qb,
        "shared_frame": readout_offset % 3,  # counter: identical; absolute lookup: differs by shift%3
    }


def graded_indel(
    anchor_pos: int, length: int, k: int, readout_pos: int, seed: int
) -> dict:
    """Graded insertion between anchor and a fixed distal readout (the accumulation test).

    Inserts `k` nucleotides immediately after the ATG (truncating the tail to keep `length`). A counter
    re-derives (q - anchor) and preserves the anchor frame for every k; a distance-indexed lookup keyed
    to the pre-edit geometry misfires in a k-dependent way. Returns intact/edited tokens and the two
    competing frame hypotheses at the readout.
    """
    rng = np.random.default_rng(seed)
    tokens, _ = anchored_window(anchor_pos, length, rng)
    ins_at = anchor_pos + 3
    insert = rng.integers(0, 4, size=k).astype(np.int8)
    edited = np.concatenate([tokens[:ins_at], insert, tokens[ins_at:]])[:length]
    anchor_frame = (readout_pos - anchor_pos) % 3          # counting from the anchor, after the indel
    content_frame = (readout_pos - anchor_pos - k) % 3     # local content shifted right by k
    return {
        "intact": tokens,
        "edited": edited,
        "readout_pos": readout_pos,
        "k": k,
        "anchor_frame": int(anchor_frame),
        "content_frame": int(content_frame),
    }
