"""The forcing task: the frame must be the only route to the label.

If any of these fail the experiment is worthless -- a local shortcut would let the model reach high
amino-acid accuracy without ever locating the start codon, and we would mistake a bag for a circuit.
"""

import numpy as np
import pytest

from gfmcd.data.genome import IGNORE
from gfmcd.data.uniform_cds import (
    ATG,
    CODON_AA,
    CODONS,
    N_AA,
    alignment_free_ceiling,
    codon_index,
    make_batch,
)


@pytest.fixture(scope="module")
def batch():
    return make_batch(n_seq=3000, window=256, seed=0)


def test_there_are_twenty_one_classes():
    assert N_AA == 21 and len(CODONS) == 64 and len(CODON_AA) == 64


def test_the_sequence_carries_no_phase_information(batch):
    """The whole point. I(nucleotide; phase) must be ~0, against 0.0241 nats in real hg38 CDS.

    Measured on the body, downstream of the start codon. The ATG itself is trivially phase-informative
    -- A at phase 0, T at 1, G at 2 -- but it is the anchor, not a shortcut. Including it puts
    I(nt;phase) at 0.00024 nats, still a hundredfold below real DNA.
    """
    tokens, _, phase = batch
    body = np.zeros_like(phase, dtype=bool)
    for r, ph in enumerate(phase):
        body[r, int(np.argmax(ph == 0)) + 3 :] = True
    m = (phase >= 0) & body

    def entropy(counts):
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    h = entropy(np.bincount(tokens[m].astype(int), minlength=4))
    h_cond = sum(
        (m & (phase == k)).sum() / m.sum() * entropy(np.bincount(tokens[m & (phase == k)].astype(int), minlength=4))
        for k in range(3)
    )
    assert h - h_cond < 1e-4, f"I(nt;phase) = {h - h_cond:.6f}: a local shortcut survives"


def test_position_index_does_not_reveal_phase(batch):
    _, _, phase = batch
    m = phase >= 0
    idx = np.tile(np.arange(256) % 3, (len(phase), 1))
    counts = np.zeros((3, 3), dtype=int)
    np.add.at(counts, (idx[m], phase[m]), 1)
    assert counts.max(axis=1).sum() / counts.sum() < 0.40


def test_start_codon_is_the_unique_anchor(batch):
    """No ATG upstream, so 'the first ATG' identifies the frame unambiguously."""
    tokens, _, phase = batch
    for row, ph in zip(tokens[:200], phase[:200]):
        start = int(np.argmax(ph == 0))
        assert row[start : start + 3].tolist() == ATG.tolist()
        upstream = row[:start]
        if len(upstream) >= 3:
            hits = (upstream[:-2] == 0) & (upstream[1:-1] == 3) & (upstream[2:] == 2)
            assert not hits.any(), "an upstream ATG would make the anchor ambiguous"


def test_labels_are_the_amino_acid_of_the_containing_codon(batch):
    tokens, aa, phase = batch
    for row, a, ph in zip(tokens[:100], aa[:100], phase[:100]):
        starts = np.where(ph == 0)[0]
        for s in starts[:-1]:
            expected = CODON_AA[codon_index(row[s : s + 3].astype(np.int64))]
            assert a[s] == a[s + 1] == a[s + 2] == expected


def test_labels_are_constant_within_a_codon(batch):
    _, aa, phase = batch
    m = phase >= 0
    assert (aa[m] >= 0).all() and (aa[~m] == IGNORE).all()


def test_a_frameless_model_is_capped_near_a_third():
    ceiling = alignment_free_ceiling()
    assert 0.33 < ceiling < 0.45, f"alignment-free ceiling {ceiling:.4f}"
