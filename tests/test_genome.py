"""Rung 1's data pipeline, tested against a miniature genome whose answer is hand-computable.

The load-bearing assertion is `test_window_index_carries_no_phase_information`. If windows were
anchored to CDS starts, codon phase would equal `index % 3`, a model with absolute position
embeddings would read it off `pos_embed`, and Rung 1 would trace a lookup table while reporting a
circuit. Nothing in a loss curve would show it. So the corpus-level and within-window statements
are asserted separately: phase must be unpredictable from `index % 3` across windows, and exactly
predictable from it within a window once the anchor is known.
"""

import random

import numpy as np
import pytest

from gfmcd.data.genome import (
    CDS,
    IGNORE,
    Split,
    SplitLockedError,
    assert_split_allowed,
    encode,
    parse_gencode_cds,
    revcomp,
    sample_window,
    sample_windows,
)

CHROM = "chrT"
GENOME_LEN = 3000


class MiniFasta(dict):
    """Stands in for pyfaidx: chrom -> string, sliceable."""


@pytest.fixture(scope="module")
def fasta():
    rng = random.Random(0)
    return MiniFasta({CHROM: "".join(rng.choice("ACGT") for _ in range(GENOME_LEN))})


# CDS on the + strand, 0-based inclusive [300, 899], phase 1.
# First codon base is therefore 301.
PLUS = CDS(CHROM, 300, 899, "+", 1)

# CDS on the - strand, 0-based inclusive [1200, 1799], phase 2.
# Transcription runs right-to-left, so the first codon base is 1799 - 2 = 1797.
MINUS = CDS(CHROM, 1200, 1799, "-", 2)


def test_first_codon_base_respects_strand_and_phase():
    assert PLUS.first_codon_base == 301
    assert MINUS.first_codon_base == 1797


def test_codon_position_counts_along_transcription():
    assert [PLUS.codon_position(g) for g in (301, 302, 303, 304)] == [0, 1, 2, 0]
    # On the minus strand the codon advances as the coordinate decreases.
    assert [MINUS.codon_position(g) for g in (1797, 1796, 1795, 1794)] == [0, 1, 2, 0]


def test_revcomp_is_an_involution_and_complements():
    assert revcomp("ACGTN") == "NACGT"
    assert revcomp(revcomp("ACGGTTAN")) == "ACGGTTAN"


def test_encode_maps_unknown_bases_to_n():
    assert encode("ACGTNx").tolist() == [0, 1, 2, 3, 4, 4]


def test_labels_are_ignored_outside_the_cds(fasta):
    rng = random.Random(1)
    w = sample_window(PLUS, fasta, window=256, rng=rng, min_cds_in_window=90)
    assert w is not None

    inside = np.array([PLUS.contains(w.start + i) for i in range(256)])
    assert (w.labels[~inside] == IGNORE).all()
    assert (w.labels[inside] >= 0).all()


def test_minus_strand_window_is_reverse_complemented_with_labels(fasta):
    rng = random.Random(2)
    w = sample_window(MINUS, fasta, window=120, rng=rng, min_cds_in_window=90)
    assert w is not None

    genomic = str(fasta[CHROM][w.start : w.start + 120]).upper()
    assert w.tokens.tolist() == encode(revcomp(genomic)).tolist()

    # Position j of the returned window is genomic coordinate start + (L - 1 - j).
    for j in range(120):
        g = w.start + (120 - 1 - j)
        expected = MINUS.codon_position(g) if MINUS.contains(g) else IGNORE
        assert w.labels[j] == expected


def test_phase_advances_by_one_per_position_within_a_window(fasta):
    """Within a window the frame is a clean period-3 counter. That is the phenomenon."""
    rng = random.Random(3)
    w = sample_window(PLUS, fasta, window=256, rng=rng, min_cds_in_window=90)
    labelled = np.where(w.labels >= 0)[0]

    lab = w.labels[labelled]
    assert ((np.diff(lab) - 1) % 3 == 0).all()


def test_window_index_carries_no_phase_information(fasta):
    """The assertion Rung 1 depends on.

    Across the corpus, `index % 3` must be uninformative about codon phase. The best possible
    fixed map from `index % 3` to phase should score at chance.
    """
    tokens, labels = sample_windows([PLUS, MINUS], fasta, n=400, window=128, seed=7, min_cds_in_window=90)

    idx = np.tile(np.arange(labels.shape[1]) % 3, (labels.shape[0], 1))
    mask = labels >= 0
    counts = np.zeros((3, 3), dtype=int)
    np.add.at(counts, (idx[mask], labels[mask]), 1)

    best = counts.max(axis=1).sum() / counts.sum()
    assert best < 0.40, f"index % 3 predicts phase at {best:.3f}; windows are not offset-randomised"


def test_within_a_window_phase_is_a_fixed_offset_of_index(fasta):
    """The other half: given the window's anchor, phase IS index % 3 plus a constant."""
    tokens, labels = sample_windows([PLUS], fasta, n=50, window=128, seed=11, min_cds_in_window=90)

    offsets = set()
    for row in labels:
        live = np.where(row >= 0)[0]
        shift = (row[live] - live) % 3
        assert len(set(shift.tolist())) == 1, "phase is not a rigid shift of position within a window"
        offsets.add(int(shift[0]))

    assert len(offsets) == 3, f"window anchors do not cover all three residues mod 3: {offsets}"


def test_test_split_is_locked(fasta):
    assert_split_allowed(Split.TRAIN)
    assert_split_allowed(Split.VAL)
    with pytest.raises(SplitLockedError, match="pre-registration"):
        assert_split_allowed(Split.TEST)


def test_gtf_parsing_round_trips(tmp_path):
    gtf = tmp_path / "mini.gtf"
    gtf.write_text(
        "\n".join(
            [
                "##description: mini",
                f"{CHROM}\tHAVANA\tCDS\t301\t900\t.\t+\t1\tgene_id \"g1\";",
                f"{CHROM}\tHAVANA\tCDS\t1201\t1800\t.\t-\t2\tgene_id \"g2\";",
                f"{CHROM}\tHAVANA\texon\t1\t100\t.\t+\t.\tgene_id \"g3\";",
                f"{CHROM}\tHAVANA\tCDS\t1\t50\t.\t+\t0\tgene_id \"g4\";",
            ]
        )
        + "\n"
    )

    cds = parse_gencode_cds(gtf, min_len=150)
    assert len(cds) == 2, "must skip non-CDS features and CDS shorter than min_len"
    assert cds[0] == PLUS
    assert cds[1] == MINUS
