"""M1 payload-identity primitives. No model or genome required."""

from __future__ import annotations

import pytest

from gfmcd.splicing.acceptor_payload import (
    LADDER_ARMS,
    REFERENCE_ALT,
    TREATMENT_ARM,
    TWIN_ARMS,
    allows_written_motif,
    arm_admissible,
    arm_dispositions,
)
from gfmcd.splicing.coupling import CouplingExample
from gfmcd.splicing.reverse_coupling import donor_alt_admissible

DONOR_POS = 21


def make_example(minus_one: str = "G", plus_three: str = "A", ex_id: str = "e0",
                 gene_id: str = "g0") -> CouplingExample:
    """A window whose donor GT sits at DONOR_POS.

    seq[DONOR_POS-1] is the last exon base; seq[DONOR_POS:DONOR_POS+2] is the GT;
    seq[DONOR_POS+2] is the +3 base. Those three positions drive every straddling register.
    """
    seq = "A" * (DONOR_POS - 1) + minus_one + "GT" + plus_three + "C" * 20
    assert seq[DONOR_POS:DONOR_POS + 2] == "GT"
    return CouplingExample(
        example_id=ex_id, sequence=seq, acceptor_pos=5, donor_pos=DONOR_POS,
        chrom="chr1", strand="+", gene_id=gene_id, split="train",
        exon_length=16, intron_length_down=100, n_isoforms_donor=1,
        n_isoforms_acceptor=1, gene_type="protein_coding", next_acceptor_pos=60,
    )


def test_frozen_arm_constants():
    assert REFERENCE_ALT == "AC"
    assert TREATMENT_ARM == "GC"
    assert TWIN_ARMS == ("GA", "GG")
    assert LADDER_ARMS == ("AC", "AT", "GA", "GG", "GC")
    assert allows_written_motif("GC") is True
    assert allows_written_motif("GG") is False


def test_gc_flags_only_the_written_register_on_a_clean_flank():
    x = make_example(minus_one="G", plus_three="A")
    assert arm_dispositions(x, "GC") == {"written": "GC"}
    assert arm_admissible(x, "GC", allow_written_motif=True) is True
    assert arm_admissible(x, "GC", allow_written_motif=False) is False


def test_gc_upstream_ag_still_disqualifies():
    """A -1 exon base of A makes upstream 'AG' -- a planted acceptor, not the treatment."""
    x = make_example(minus_one="A", plus_three="A")
    assert arm_dispositions(x, "GC")["upstream"] == "AG"
    assert arm_admissible(x, "GC", allow_written_motif=True) is False


def test_gg_downstream_gt_disqualifies():
    """alt='GG' with a +3 base of T plants a canonical GT one position downstream."""
    x = make_example(minus_one="G", plus_three="T")
    assert arm_dispositions(x, "GG")["downstream"] == "GT"
    assert arm_admissible(x, "GG", allow_written_motif=False) is False


def test_reference_arm_is_clean_on_a_clean_flank():
    x = make_example(minus_one="G", plus_three="A")
    assert arm_dispositions(x, REFERENCE_ALT) == {}
    assert arm_admissible(x, REFERENCE_ALT) is True


@pytest.mark.parametrize("minus_one", ["A", "C", "G", "T"])
@pytest.mark.parametrize("plus_three", ["A", "C", "G", "T"])
@pytest.mark.parametrize("alt", ["AC", "AT", "GA", "GG", "GC"])
def test_without_carveout_matches_the_existing_hygiene_rule(minus_one, plus_three, alt):
    """Regression guard: allow_written_motif=False must be exactly donor_alt_admissible."""
    x = make_example(minus_one=minus_one, plus_three=plus_three)
    assert arm_admissible(x, alt, allow_written_motif=False) == donor_alt_admissible(x, alt)


from gfmcd.splicing.acceptor_payload import admissible_fractions, common_admissible_rows


def test_common_rows_are_admissible_under_every_arm():
    exs = [
        make_example(minus_one="G", plus_three="A", ex_id="clean", gene_id="g1"),
        make_example(minus_one="A", plus_three="A", ex_id="bad_gc", gene_id="g2"),
        make_example(minus_one="G", plus_three="T", ex_id="bad_gg", gene_id="g3"),
    ]
    assert common_admissible_rows(exs, LADDER_ARMS) == [0]


def test_common_rows_empty_arms_returns_all():
    exs = [make_example(ex_id="a"), make_example(minus_one="A", ex_id="b")]
    assert common_admissible_rows(exs, ()) == [0, 1]


def test_admissible_fractions_are_per_arm_and_reported():
    exs = [
        make_example(minus_one="G", plus_three="A", ex_id="clean"),
        make_example(minus_one="A", plus_three="A", ex_id="bad_gc"),
    ]
    fracs = admissible_fractions(exs, ("AC", "GC"))
    assert fracs["AC"] == 1.0
    assert fracs["GC"] == 0.5


def test_admissible_fractions_empty_cohort_is_zero_not_a_crash():
    assert admissible_fractions([], ("AC",)) == {"AC": 0.0}
