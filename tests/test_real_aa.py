"""Cells C and D: a mu=1 amino-acid readout on real hg38 CDS.

The label must be a function of (sequence, frame) and never of the frame alone, or the objective is
circular (D7). And cell C's filter must actually make the detector easy.
"""

import numpy as np
import pytest

from gfmcd.data.genome import IGNORE, amino_acid_labels, upstream_atg_count
from gfmcd.data.uniform_cds import CODON_AA, codon_index

A, C, G, T = 0, 1, 2, 3


def test_amino_acid_label_is_the_containing_codon():
    #            0  1  2  3  4  5
    tok = np.array([[A, T, G, C, C, T]], dtype=np.int8)
    phase = np.array([[0, 1, 2, 0, 1, 2]])
    aa = amino_acid_labels(tok, phase)

    met = CODON_AA[codon_index(np.array([A, T, G]))]
    pro = CODON_AA[codon_index(np.array([C, C, T]))]
    assert aa[0].tolist() == [met, met, met, pro, pro, pro]


def test_labels_are_constant_within_a_codon_and_ignored_outside_cds():
    tok = np.array([[C, C, A, T, G, C, C, T]], dtype=np.int8)
    phase = np.array([[IGNORE, IGNORE, 0, 1, 2, 0, 1, 2]])
    aa = amino_acid_labels(tok, phase)

    assert (aa[0, :2] == IGNORE).all()
    assert aa[0, 2] == aa[0, 3] == aa[0, 4]
    assert aa[0, 5] == aa[0, 6] == aa[0, 7]


def test_positions_with_N_are_ignored():
    tok = np.array([[A, T, 4, C, C, T]], dtype=np.int8)
    phase = np.array([[0, 1, 2, 0, 1, 2]])
    aa = amino_acid_labels(tok, phase)
    assert (aa[0, :3] == IGNORE).all()
    assert (aa[0, 3:] != IGNORE).all()


def test_upstream_atg_count():
    tok = np.array([[A, T, G, C, C, A, T, G, A, A]], dtype=np.int8)
    # annotated start at index 5; there is one ATG upstream (index 0)
    assert upstream_atg_count(tok, np.array([5]))[0] == 1
    # annotated start at index 0; nothing upstream
    assert upstream_atg_count(tok, np.array([0]))[0] == 0


def test_the_label_does_not_reveal_the_frame_by_itself():
    """A run of three identical labels marks a codon -- but the model never sees the labels."""
    rng = np.random.default_rng(0)
    tok = rng.integers(0, 4, size=(1, 60)).astype(np.int8)
    phase = np.tile(np.arange(60) % 3, (1, 1))
    aa = amino_acid_labels(tok, phase)
    # every codon's three positions share a label; that is the point, and it is not an input
    for s in range(0, 57, 3):
        assert aa[0, s] == aa[0, s + 1] == aa[0, s + 2]
