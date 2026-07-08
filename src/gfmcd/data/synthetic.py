"""Rung 0's synthetic reading-frame task.

A sequence is a random 5' region, a unique start marker M, then a body whose
nucleotides are drawn from phase-conditional distributions. Two facts make this the
right substrate for planted twins:

1. Phase at position i is (i - start - 1) % 3, so recovering it requires locating M.
   Nothing local determines it. A compositional model must attend to the marker.
2. EMIT is phase-biased, so a per-nucleotide lookup table predicts phase well above
   chance without ever representing frame. A heuristic model can match behaviour.

The gap between "exactly right" and "Bayes-optimal-but-local" is what the twins exploit.
"""

from __future__ import annotations

import torch

A, C, G, T, M = 0, 1, 2, 3, 4
VOCAB = "ACGTM"

EMIT = torch.tensor(
    [
        [0.55, 0.15, 0.20, 0.10],
        [0.15, 0.50, 0.15, 0.20],
        [0.15, 0.15, 0.30, 0.40],
    ]
)

IGNORE = -100


def bayes_unigram_accuracy() -> float:
    """Accuracy ceiling for any model that sees only the nucleotide at position i."""
    joint = EMIT / 3.0
    return float(joint.max(dim=0).values.sum())


def posterior_phase_given_nt() -> torch.Tensor:
    joint = EMIT / 3.0
    return (joint / joint.sum(dim=0, keepdim=True)).T


def make_batch(
    n_seq: int,
    body_len: int = 90,
    utr_range: tuple[int, int] = (3, 12),
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    lo, hi = utr_range
    max_len = hi + 1 + body_len

    tokens = torch.zeros(n_seq, max_len, dtype=torch.long)
    labels = torch.full((n_seq, max_len), IGNORE, dtype=torch.long)

    for s in range(n_seq):
        utr = int(torch.randint(lo, hi + 1, (1,), generator=g))
        tokens[s, :utr] = torch.randint(0, 4, (utr,), generator=g)
        tokens[s, utr] = M

        n_body = max_len - utr - 1
        phases = (torch.arange(n_body) % 3).long()
        probs = EMIT[phases]
        tokens[s, utr + 1 :] = torch.multinomial(probs, 1, generator=g).squeeze(-1)
        labels[s, utr + 1 :] = phases

    return tokens, labels


def ablate_marker(tokens: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Replace M with an ordinary nucleotide, destroying the only frame anchor.

    A compositional model must collapse to chance. A local heuristic must not notice.
    """
    g = torch.Generator().manual_seed(seed)
    out = tokens.clone()
    mask = out == M
    out[mask] = torch.randint(0, 4, (int(mask.sum()),), generator=g)
    return out
