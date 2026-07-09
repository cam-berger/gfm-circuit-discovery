"""The anchor-free pooled oracle: the best frame estimate available without the start codon.

D30 measured this in aggregate (0.9509 on real hg38 CDS at 240 nt) in a script that was never
committed. Phase C needs it **per position**, because the registered test T2 partitions positions by
whether the pooled route already gets the frame right. So it is rebuilt here, and
`scripts/pooled_oracle.py` checks it reproduces the aggregate number before anything is built on it.

The estimator is codon-level naive Bayes. Given a frame residue r, the context's triplets at positions
congruent to r mod 3 are scored under human codon usage; the winning residue is the predicted frame.
It never sees the start codon, and it has no positional feature -- only sequence.
"""

from __future__ import annotations

import numpy as np

from gfmcd.data.uniform_cds import CODONS, codon_index, real_codon_probs

CTX = 120  # +-120 nt, i.e. the 240-nt pooling window of D30


def _codon_llr(codon_probs: np.ndarray) -> np.ndarray:
    """(64,) log p(codon | in-frame) - log p(codon | independent bases).

    A ratio, not a likelihood. Only 70% of a random 256-nt window is coding, so a bare log-likelihood
    lets a codon carrying no frame information (one whose usage matches its base composition) drag the
    score of whichever frame happens to tile more of the intron. Under the ratio such a codon scores 0.
    """
    base = np.zeros(4)
    np.add.at(base, CODONS.ravel(), np.repeat(codon_probs, 3))
    bg = base[CODONS[:, 0]] * base[CODONS[:, 1]] * base[CODONS[:, 2]]
    return np.log(codon_probs) - np.log(bg)


def _triplet_score(tok: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Score of the codon starting at each position. (n, w) -> (n, w-2). Zero where any base is N."""
    tri = np.stack([tok[:, :-2], tok[:, 1:-1], tok[:, 2:]], axis=-1).astype(np.int64)
    good = (tri < 4).all(-1)
    out = np.zeros(tri.shape[:2])
    out[good] = table[codon_index(tri[good])]
    return out


def pooled_frame_logits(tok: np.ndarray, ctx: int = CTX, codon_probs: np.ndarray | None = None,
                        coding: np.ndarray | None = None) -> np.ndarray:
    """(n, w, 3): pooled log-evidence for each frame residue, at every position.

    Evidence for residue r at position i is the summed codon log-ratio over the codons of frame r that
    start inside [i - ctx, i + ctx]. Computed with a cumulative sum rather than once per position; the
    naive loop is O(n * w * ctx) and this is O(n * w).

    `coding` (n, w) masks non-coding positions out of the pool. Supplying it hands the estimator the CDS
    boundaries, which the model does not get -- use it only to reproduce D30's pure-CDS ceiling.
    """
    probs = real_codon_probs() if codon_probs is None else codon_probs
    lp = _triplet_score(tok, _codon_llr(probs))   # (n, w-2), value at each codon START
    if coding is not None:
        whole = coding[:, :-2] & coding[:, 1:-1] & coding[:, 2:]
        lp = lp * whole
    n, wm2 = lp.shape
    w = wm2 + 2

    out = np.zeros((n, w, 3))
    pos = np.arange(w)
    for r in range(3):
        starts = np.arange(r, wm2, 3)      # codon starts of frame r
        cum = np.concatenate([np.zeros((n, 1)), np.cumsum(lp[:, starts], axis=1)], axis=1)
        # codons of frame r starting in [i-ctx, i+ctx] are the slice [lo, hi) of `starts`
        lo = np.searchsorted(starts, np.maximum(pos - ctx, 0), side="left")
        hi = np.searchsorted(starts, np.minimum(pos + ctx, wm2 - 1), side="right")
        out[:, :, r] = cum[:, hi] - cum[:, lo]
    return out


def pooled_frame_pred(tok: np.ndarray, ctx: int = CTX, codon_probs: np.ndarray | None = None,
                      coding: np.ndarray | None = None) -> np.ndarray:
    """(n, w) predicted frame residue (position mod 3 of the codon start) at every position."""
    return pooled_frame_logits(tok, ctx, codon_probs, coding).argmax(-1)


def oracle_correct(tok: np.ndarray, phase: np.ndarray, ctx: int = CTX,
                   codon_probs: np.ndarray | None = None, mask_noncoding: bool = False) -> np.ndarray:
    """(n, w) bool: does the anchor-free pooled route recover the true frame at this position?

    `phase[i] = (i - start) % 3`, so the true frame residue is `(i - phase[i]) % 3`.
    Positions outside the coding body are False.
    """
    coding = phase >= 0
    pred = pooled_frame_pred(tok, ctx, codon_probs, coding if mask_noncoding else None)
    pos = np.arange(tok.shape[1])[None, :]
    true_r = (pos - np.where(coding, phase, 0)) % 3
    return (pred == true_r) & coding
