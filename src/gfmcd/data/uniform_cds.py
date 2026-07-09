"""Synthetic coding sequence where the frame is the ONLY route to the label.

D24 showed GFM-nano never builds an anchored frame counter on real DNA, and D25 explains why: real
codon usage is biased, so phase is inferable from a ~30 nt bag, and an anchored counter buys 0.4% of
the loss. No self-supervised objective can change that -- any phase-dependent structure repeats every
three bases and is therefore locally detectable.

So we remove the shortcut and supervise a readout that requires the frame:

    sequence   5' region with NO ATG | ATG | codons drawn uniformly from all 64 codons
    label      the amino acid (or stop) of the codon containing each position

All 64 codons, not the 61 sense codons: the three stops are all T-initial, so excluding them depletes
T at codon position 0 and leaves a residual phase cue. With all 64 the sequence is exactly i.i.d.
uniform over {A,C,G,T} and local statistics carry *zero* phase information -- verified in
`tests/test_uniform_cds.py`. Emitting the right amino acid requires knowing which triplet is in frame, and the only
event that fixes the frame is the start codon. Guessing the alignment caps accuracy near 1/3; knowing
it gives 1.0.

The model is never told the frame. It is told the protein. That is Othello-GPT's structure, and it
makes the anchored counter the only solution rather than a marginal one -- the learned analogue of
Rung 0's planted compositional twin.
"""

from __future__ import annotations

import numpy as np

from gfmcd.data.genome import IGNORE

BASES = "TCAG"
AA_TABLE = (
    "FFLLSSSSYY**CC*W"
    "LLLLPPPPHHQQRRRR"
    "IIIMTTTTNNKKSSRR"
    "VVVVAAAADDEEGGGG"
)
NT2ID = {"A": 0, "C": 1, "G": 2, "T": 3}
STOP = "*"

# Codon index in AA_TABLE is 16*i + 4*j + k over TCAG; we store codons as (a, b, c) in ACGT ids.
_CODONS: list[tuple[int, int, int]] = []
_AA: list[str] = []
for i, b1 in enumerate(BASES):
    for j, b2 in enumerate(BASES):
        for k, b3 in enumerate(BASES):
            _CODONS.append((NT2ID[b1], NT2ID[b2], NT2ID[b3]))
            _AA.append(AA_TABLE[16 * i + 4 * j + k])

CODONS = np.array(_CODONS, dtype=np.int8)  # [64, 3] -- all of them, stops included
AMINO_ACIDS = sorted(set(_AA))
AA2ID = {a: i for i, a in enumerate(AMINO_ACIDS)}
CODON_AA = np.array([AA2ID[a] for a in _AA], dtype=np.int64)  # [64]
N_AA = len(AMINO_ACIDS)  # 21: twenty amino acids plus stop

ATG = np.array([0, 3, 2], dtype=np.int8)


def _utr_without_atg(length: int, rng: np.random.Generator) -> np.ndarray:
    """A 5' region containing no ATG, so the start codon is the unique anchor."""
    while True:
        u = rng.integers(0, 4, size=length).astype(np.int8)
        if length < 3:
            return u
        hits = (u[:-2] == 0) & (u[1:-1] == 3) & (u[2:] == 2)
        if not hits.any():
            return u


def make_batch(
    n_seq: int,
    window: int = 256,
    utr_range: tuple[int, int] = (6, 40),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (tokens [n, w], amino-acid labels [n, w], phase labels [n, w]).

    Phase labels are for probing only and never enter any training objective.
    """
    rng = np.random.default_rng(seed)
    tokens = np.zeros((n_seq, window), dtype=np.int8)
    aa = np.full((n_seq, window), IGNORE, dtype=np.int64)
    phase = np.full((n_seq, window), IGNORE, dtype=np.int64)

    lo, hi = utr_range
    atg_codon = int(codon_index(ATG.astype(np.int64)))

    for s in range(n_seq):
        u = int(rng.integers(lo, hi + 1))
        tokens[s, :u] = _utr_without_atg(u, rng)

        n_cod = (window - u) // 3
        idx = rng.integers(0, len(CODONS), size=n_cod)
        idx[0] = atg_codon  # the start codon IS the first codon, not a prefix that gets overwritten
        body = CODONS[idx].reshape(-1)
        end = u + 3 * n_cod
        tokens[s, u:end] = body[: end - u]
        tokens[s, end:] = rng.integers(0, 4, size=window - end)

        pos = np.arange(u, end)
        phase[s, u:end] = (pos - u) % 3
        aa[s, u:end] = np.repeat(CODON_AA[idx], 3)[: end - u]

    return tokens, aa, phase


def codon_index(triplet: np.ndarray) -> np.ndarray:
    """Map (a, b, c) in ACGT ids to the row of CODONS, vectorised over a leading axis."""
    key = CODONS[:, 0] * 16 + CODONS[:, 1] * 4 + CODONS[:, 2]
    lookup = np.zeros(64, dtype=np.int64)
    lookup[key] = np.arange(64)
    return lookup[triplet[..., 0] * 16 + triplet[..., 1] * 4 + triplet[..., 2]]


def alignment_free_ceiling(seed: int = 0, n: int = 40000) -> float:
    """The best a model can do without knowing the frame.

    It sees the local sequence but not the alignment. Optimally it marginalises over the three
    possible alignments and predicts the most likely amino acid. Everything above this number is
    evidence that the model located the start codon and counted.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 64, size=(n, 3))
    stream = CODONS[idx].reshape(n, 9)
    true_aa = CODON_AA[idx[:, 1]]

    votes = np.zeros((n, N_AA))
    for shift in (-1, 0, 1):
        aa = CODON_AA[codon_index(stream[:, 3 + shift : 6 + shift])]
        votes[np.arange(n), aa] += 1
    return float((votes.argmax(1) == true_aa).mean())
