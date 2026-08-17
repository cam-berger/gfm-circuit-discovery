"""Outcome-blind matching of a GT-AG comparator set to the GC-AG treatment arm.

After the ``GG`` operator filter the two arms remain grossly unbalanced on the +3 base --
GC is A94/G6, GT is A50/G50 -- so an unmatched contrast would measure flanking context rather
than donor class. The matcher uses sequence and annotation only; it never sees a margin.
"""

from __future__ import annotations

import pytest

from gfmcd.splicing.donor_class import (
    donor_motif,
    flank_key,
    match_comparators,
    matching_balance,
)
from gfmcd.splicing.coupling import CouplingExample

DONOR_POS = 40


def make_example(motif="GT", minus1="G", plus3="A", plus5="G", gap=100,
                 ex_id="e", gene="g"):
    """A window whose donor sits at DONOR_POS with the requested flank bases."""
    seq = ("A" * (DONOR_POS - 1) + minus1 + motif + plus3 + "T" + plus5
           + "C" * 400)
    assert seq[DONOR_POS:DONOR_POS + 2] == motif
    assert seq[DONOR_POS - 1] == minus1 and seq[DONOR_POS + 2] == plus3
    assert seq[DONOR_POS + 4] == plus5
    return CouplingExample(
        example_id=ex_id, sequence=seq, acceptor_pos=10, donor_pos=DONOR_POS,
        chrom="chr1", strand="+", gene_id=gene, split="train",
        exon_length=30, intron_length_down=gap, n_isoforms_donor=1,
        n_isoforms_acceptor=1, gene_type="protein_coding",
        next_acceptor_pos=DONOR_POS + gap,
    )


def test_donor_motif_reads_the_native_dinucleotide():
    assert donor_motif(make_example(motif="GC")) == "GC"
    assert donor_motif(make_example(motif="GT")) == "GT"


def test_flank_key_captures_the_three_confounding_bases():
    x = make_example(minus1="T", plus3="G", plus5="C")
    assert flank_key(x) == ("T", "G", "C")


def test_matches_only_within_an_identical_flank_stratum():
    gc = [make_example("GC", plus3="A", ex_id="t0", gene="ga")]
    pool = [
        make_example("GT", plus3="G", ex_id="wrong_stratum", gene="gb"),
        make_example("GT", plus3="A", ex_id="right_stratum", gene="gc"),
    ]
    m = match_comparators(gc, pool, k=1)
    assert [x.example_id for x in m.comparators] == ["right_stratum"]
    assert m.unmatched_treatment == []


def test_treatment_with_no_stratum_partner_is_dropped_and_counted():
    """A dropped treatment row must be reported, never silently retained."""
    gc = [
        make_example("GC", plus3="A", ex_id="has_partner", gene="ga"),
        make_example("GC", plus3="C", ex_id="no_partner", gene="gb"),
    ]
    pool = [make_example("GT", plus3="A", ex_id="p", gene="gc")]
    m = match_comparators(gc, pool, k=1)
    assert [x.example_id for x in m.treatment] == ["has_partner"]
    assert m.unmatched_treatment == ["no_partner"]


def test_within_stratum_it_picks_the_nearest_gap():
    gc = [make_example("GC", gap=100, ex_id="t", gene="ga")]
    pool = [
        make_example("GT", gap=140, ex_id="far", gene="gb"),
        make_example("GT", gap=103, ex_id="near", gene="gc"),
        make_example("GT", gap=120, ex_id="mid", gene="gd"),
    ]
    m = match_comparators(gc, pool, k=1)
    assert [x.example_id for x in m.comparators] == ["near"]


def test_k_to_one_matching_returns_k_comparators_per_treatment():
    gc = [make_example("GC", gap=100, ex_id="t", gene="ga")]
    pool = [make_example("GT", gap=100 + i, ex_id=f"p{i}", gene=f"g{i}")
            for i in range(5)]
    m = match_comparators(gc, pool, k=3)
    assert len(m.comparators) == 3


def test_a_comparator_is_never_reused_across_treatment_rows():
    """Reuse would understate variance by duplicating rows into the bootstrap."""
    gc = [make_example("GC", gap=100, ex_id="t0", gene="ga"),
          make_example("GC", gap=100, ex_id="t1", gene="gb")]
    pool = [make_example("GT", gap=100, ex_id="only", gene="gc")]
    m = match_comparators(gc, pool, k=1)
    ids = [x.example_id for x in m.comparators]
    assert ids == ["only"]
    assert len(set(ids)) == len(ids)
    assert m.unmatched_treatment == ["t1"]


def test_matching_is_deterministic_under_ties():
    gc = [make_example("GC", gap=100, ex_id="t", gene="ga")]
    pool = [make_example("GT", gap=100, ex_id="b", gene="gb"),
            make_example("GT", gap=100, ex_id="a", gene="gc")]
    first = [x.example_id for x in match_comparators(gc, pool, k=1).comparators]
    second = [x.example_id for x in match_comparators(gc, pool, k=1).comparators]
    assert first == second


def test_gap_tolerance_refuses_a_distant_partner():
    gc = [make_example("GC", gap=100, ex_id="t", gene="ga")]
    pool = [make_example("GT", gap=149, ex_id="too_far", gene="gb")]
    m = match_comparators(gc, pool, k=1, gap_tol=10)
    assert m.comparators == []
    assert m.unmatched_treatment == ["t"]


def test_balance_table_reports_realized_composition_not_assumed():
    gc = [make_example("GC", plus3="A", ex_id="t", gene="ga")]
    pool = [make_example("GT", plus3="A", ex_id="p", gene="gb")]
    m = match_comparators(gc, pool, k=1)
    bal = matching_balance(m)
    assert bal["treatment"]["+3"]["A"] == pytest.approx(1.0)
    assert bal["comparator"]["+3"]["A"] == pytest.approx(1.0)
    assert bal["n_treatment"] == 1 and bal["n_comparator"] == 1


def test_matcher_never_reads_a_margin():
    """Outcome blindness is structural: the signature accepts no outcome argument."""
    import inspect
    params = set(inspect.signature(match_comparators).parameters)
    assert not (params & {"margins", "y", "effects", "outcome", "coupling"})
