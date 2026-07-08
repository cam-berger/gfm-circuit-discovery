"""The firewall must fire on exactly the three mistakes the grilling surfaced."""

import pytest

from gfmcd.config import (
    ContaminationError,
    assert_cross_strand_capable,
    assert_holdout_valid,
    assert_per_nucleotide_readout,
)


def test_pretrained_anchors_have_no_valid_holdout():
    for name in ("splicebert", "dnabert2"):
        with pytest.raises(ContaminationError, match="no chromosome is held out"):
            assert_holdout_valid(name)


def test_our_own_models_have_valid_holdout():
    for name in ("gfm-nano-cds", "gfm-nano-dna", "planted-compositional"):
        assert_holdout_valid(name)


def test_dnabert2_rejects_per_nucleotide_readouts():
    with pytest.raises(ContaminationError, match="single-nucleotide resolution"):
        assert_per_nucleotide_readout("dnabert2")
    assert_per_nucleotide_readout("splicebert")


def test_splicebert_rejects_rung3():
    with pytest.raises(ContaminationError, match="reverse strand"):
        assert_cross_strand_capable("splicebert")
    assert_cross_strand_capable("gfm-nano-dna")
