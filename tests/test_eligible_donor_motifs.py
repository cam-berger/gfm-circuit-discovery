"""``circuit.eligible`` gains a donor-motif parameter without moving any existing cohort.

Every study in this repository predating M1 is GT-only by construction: ``eligible`` hardcoded
``== "GT"``, so no prior cohort contains a ``GC-AG`` intron even though the dataset builder admits
one (``coupling.DONOR_MOTIFS == ("GT", "GC")``). M1's primary arm needs those rows. The parameter
therefore defaults to GT-only, and the first test below pins that default against a literal copy of
the historical predicate rather than trusting it.
"""

from __future__ import annotations

import pytest

from gfmcd.splicing.circuit import GAP_MAX, GAP_MIN, eligible
from gfmcd.splicing.coupling import CouplingExample


def make_example(motif: str, *, split: str = "train", gap: int = 100,
                 ex_id: str = "e", next_acceptor: bool = True) -> CouplingExample:
    donor_pos = 40
    seq = "A" * donor_pos + motif + "C" * 400
    return CouplingExample(
        example_id=ex_id, sequence=seq, acceptor_pos=10, donor_pos=donor_pos,
        chrom="chr1", strand="+", gene_id=f"g_{ex_id}", split=split,
        exon_length=30, intron_length_down=gap, n_isoforms_donor=1,
        n_isoforms_acceptor=1, gene_type="protein_coding",
        next_acceptor_pos=(donor_pos + gap) if next_acceptor else None,
    )


def historical_eligible(examples, split):
    """A literal copy of the pre-M1 implementation, kept as the equivalence oracle."""
    return [x for x in examples
            if x.split == split
            and x.next_acceptor_pos is not None
            and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
            and GAP_MIN <= (x.next_acceptor_pos - x.donor_pos) <= GAP_MAX]


@pytest.fixture
def mixed_cohort():
    return [
        make_example("GT", ex_id="gt_ok"),
        make_example("GC", ex_id="gc_ok"),
        make_example("AT", ex_id="at_reject"),
        make_example("GT", ex_id="gt_wrong_split", split="val"),
        make_example("GC", ex_id="gc_wrong_split", split="val"),
        make_example("GT", ex_id="gt_no_acceptor", next_acceptor=False),
        make_example("GC", ex_id="gc_no_acceptor", next_acceptor=False),
        make_example("GT", ex_id="gt_gap_short", gap=GAP_MIN - 1),
        make_example("GC", ex_id="gc_gap_long", gap=GAP_MAX + 1),
    ]


def test_default_is_byte_identical_to_the_historical_predicate(mixed_cohort):
    """The regression guard: no existing caller's cohort may move."""
    assert eligible(mixed_cohort, "train") == historical_eligible(mixed_cohort, "train")
    assert eligible(mixed_cohort, "val") == historical_eligible(mixed_cohort, "val")


def test_default_excludes_gc_exactly_as_before(mixed_cohort):
    got = {x.example_id for x in eligible(mixed_cohort, "train")}
    assert got == {"gt_ok"}


def test_gc_can_be_requested_explicitly(mixed_cohort):
    got = {x.example_id for x in eligible(mixed_cohort, "train", donor_motifs=("GT", "GC"))}
    assert got == {"gt_ok", "gc_ok"}


def test_gc_only_selects_the_treatment_arm(mixed_cohort):
    got = {x.example_id for x in eligible(mixed_cohort, "train", donor_motifs=("GC",))}
    assert got == {"gc_ok"}


def test_widening_the_motifs_does_not_relax_any_other_filter(mixed_cohort):
    """Split, acceptor presence and the gap window must still bind on the new arm."""
    got = {x.example_id for x in eligible(mixed_cohort, "train", donor_motifs=("GT", "GC"))}
    for rejected in ("gc_wrong_split", "gc_no_acceptor", "gc_gap_long",
                     "gt_wrong_split", "gt_no_acceptor", "gt_gap_short", "at_reject"):
        assert rejected not in got


def test_empty_motifs_is_refused_rather_than_silently_empty():
    with pytest.raises(ValueError):
        eligible([], "train", donor_motifs=())


def test_unknown_motif_is_refused():
    """A typo must fail loudly, not silently return nothing."""
    with pytest.raises(ValueError):
        eligible([], "train", donor_motifs=("GX",))
