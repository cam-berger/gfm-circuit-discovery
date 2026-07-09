"""The pooled oracle is the reference the Phase C conditioning test (T2) partitions on.

If it were broken, T2's partition would be noise and the ratio would come out at 1.0 for the wrong
reason. These tests pin the two properties T2 actually relies on: it recovers the frame far above
chance on in-frame codon sequence, and it is genuinely anchor-free -- deleting the start codon does not
move it.
"""

from __future__ import annotations

import numpy as np

from gfmcd.data.uniform_cds import make_batch, real_codon_probs
from gfmcd.oracle import CTX, oracle_correct, pooled_frame_logits


def _probs():
    # Uniform codons carry no frame signal by construction, so score them under human usage.
    return real_codon_probs(cache="data/cache/codon_usage_hg38.npy")


def test_logits_shape_and_frame_argmax():
    tok, _, ph = make_batch(8, 256, seed=0)
    lg = pooled_frame_logits(tok, ctx=CTX, codon_probs=_probs())
    assert lg.shape == (8, 256, 3)
    assert np.isfinite(lg).all()


def test_recovers_frame_far_above_chance_on_biased_codons():
    from gfmcd.data.uniform_cds import blended_codon_probs

    p = blended_codon_probs(2.0)  # strong codon bias -> the pooled route should work
    tok, _, ph = make_batch(64, 256, seed=1, codon_probs=p)
    acc = oracle_correct(tok, ph, codon_probs=p)[ph >= 0].mean()
    assert acc > 0.8, acc


def test_is_anchor_free():
    """Deleting the ATG must not change the oracle. It pools codon usage; it never looks for a start."""
    from gfmcd.data.uniform_cds import blended_codon_probs

    p = blended_codon_probs(2.0)
    tok, _, ph = make_batch(32, 256, seed=2, codon_probs=p)
    start = np.argmax(ph == 0, axis=1)

    abl = tok.copy()
    for r, s in enumerate(start):
        abl[r, s : s + 3] = np.array([1, 3, 1], dtype=tok.dtype)  # CTC

    a = oracle_correct(tok, ph, codon_probs=p)
    b = oracle_correct(abl, ph, codon_probs=p)
    far = (ph >= 0) & (np.arange(tok.shape[1])[None, :] - start[:, None] >= 64)
    assert abs(a[far].mean() - b[far].mean()) < 0.01
