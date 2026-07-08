import random

import pytest

from gfmcd.nulls.sequence import dinuc_counts, dinuc_shuffle, g_density_matched, has_g4_motif

SEQ = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGCTAGCATCGATCGGGATCCATTGGGTTAGGGCCAGGGTTAGGG"


@pytest.mark.parametrize("seed", range(8))
def test_dinuc_shuffle_preserves_dinucleotide_counts(seed):
    out = dinuc_shuffle(SEQ, random.Random(seed))
    assert len(out) == len(SEQ)
    assert out[0] == SEQ[0] and out[-1] == SEQ[-1]
    assert dinuc_counts(out) == dinuc_counts(SEQ)


def test_dinuc_shuffle_actually_shuffles():
    outs = {dinuc_shuffle(SEQ, random.Random(s)) for s in range(20)}
    assert len(outs) > 1, "shuffle is degenerate on this sequence"


def test_source_sequence_is_g4_positive():
    assert has_g4_motif(SEQ), "test fixture must contain a canonical G4 or the null is vacuous"


@pytest.mark.parametrize("seed", range(8))
def test_g_density_matched_is_composition_matched_and_g4_negative(seed):
    out = g_density_matched(SEQ, random.Random(seed))
    assert sorted(out) == sorted(SEQ), "G-density must be preserved exactly"
    assert not has_g4_motif(out), "a decoy containing a G4 is a positive, not a null"
