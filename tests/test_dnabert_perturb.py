from gfmcd.dnabert.perturb import (
    changed_fraction,
    dinucleotide_counts,
    dinucleotide_shuffle,
    hamming_distance,
    mononucleotide_shuffle,
    nucleotide_counts,
)


SEQUENCE = (
    "GCTAGCTCATCTTGCGGCTGGGCGGGGCCCAGGACT"
    "GCTGCTGCTGACCGCCTTGATAGGCTACACCGTG"
)


def test_mononucleotide_shuffle_preserves_counts():
    shuffled = mononucleotide_shuffle(
        SEQUENCE,
        seed=42,
    )

    assert len(shuffled) == len(SEQUENCE)
    assert nucleotide_counts(shuffled) == nucleotide_counts(SEQUENCE)
    assert shuffled != SEQUENCE


def test_mononucleotide_shuffle_is_deterministic():
    first = mononucleotide_shuffle(
        SEQUENCE,
        seed=123,
    )

    second = mononucleotide_shuffle(
        SEQUENCE,
        seed=123,
    )

    assert first == second


def test_dinucleotide_shuffle_preserves_exact_counts():
    shuffled = dinucleotide_shuffle(
        SEQUENCE,
        seed=42,
    )

    assert len(shuffled) == len(SEQUENCE)
    assert nucleotide_counts(shuffled) == nucleotide_counts(SEQUENCE)
    assert dinucleotide_counts(shuffled) == dinucleotide_counts(SEQUENCE)
    assert shuffled != SEQUENCE


def test_dinucleotide_shuffle_is_deterministic():
    first = dinucleotide_shuffle(
        SEQUENCE,
        seed=123,
    )

    second = dinucleotide_shuffle(
        SEQUENCE,
        seed=123,
    )

    assert first == second


def test_distance_helpers():
    assert hamming_distance("AAAA", "AATA") == 1
    assert changed_fraction("AAAA", "AATA") == 0.25
