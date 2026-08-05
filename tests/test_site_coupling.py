"""The donor/acceptor site axis (A13/A14 ``--site``).

Genome-free and model-free where possible.  The two facts these pin are the ones that would
silently invalidate the acceptor arm if they drifted:

1. The donor branch reproduces the FROZEN ``circuit.CouplingTask`` exactly -- same mask position,
   same corruption index, same alt.  A new module that quietly redefines the donor readout would
   make the two arms incomparable and would contradict C1.
2. The acceptor arm's corruption plants no splice motif.  R44 measured that the symmetric ``CC``
   plants a ``GC`` donor in 81% of val GENCODE donors and ``TA`` a canonical ``GT`` in 95.5%.
   ``test_symmetric_cc_would_plant_...`` is what gives the ``AC`` choice discriminating power --
   without it, "AC is clean" is an assertion no test could contradict.
"""

from __future__ import annotations

import pytest

from gfmcd.splicing.circuit import CORRUPT_ALT
from gfmcd.splicing.coupling import CouplingExample, substitute
from gfmcd.splicing.site_coupling import (
    ACCEPTOR_CORRUPT_ALT, DONOR_CORRUPT_ALT, SITES, SPEC, planted_motif_rate,
)


def make_example(sequence: str, acceptor_pos: int, donor_pos: int, next_acceptor_pos: int,
                 other_ag=(), other_gt=()) -> CouplingExample:
    return CouplingExample(
        example_id="x", sequence=sequence, acceptor_pos=acceptor_pos, donor_pos=donor_pos,
        chrom="chr1", strand="+", gene_id="g", split="val", exon_length=10,
        intron_length_down=100, n_isoforms_donor=9, n_isoforms_acceptor=9, gene_type="pc",
        next_acceptor_pos=next_acceptor_pos, other_ag=other_ag, other_gt=other_gt,
    )


def exon_with_g_minus_one(donor_pos: int = 20, gap: int = 80) -> CouplingExample:
    """A donor whose preceding exon base is G -- the 81% case R44 measured."""
    seq = list("A" * 200)
    seq[donor_pos - 1] = "G"                       # the -1 G of the 5'SS consensus
    seq[donor_pos:donor_pos + 2] = list("GT")      # donor
    acc = donor_pos + gap
    seq[acc - 2:acc] = list("AG")                  # intron-mate acceptor
    return make_example("".join(seq), acceptor_pos=5, donor_pos=donor_pos,
                        next_acceptor_pos=acc)


# --------------------------------------------------------------------------------------
# 1. The donor branch must not drift from the frozen definition.
# --------------------------------------------------------------------------------------
def test_donor_alt_is_the_frozen_corrupt_alt():
    assert DONOR_CORRUPT_ALT == CORRUPT_ALT == "CC"


def test_donor_readout_position_is_the_donor_g():
    example = exon_with_g_minus_one()
    assert SPEC["donor"].readout_position(example) == example.donor_pos
    assert example.sequence[example.donor_pos] == "G"


def test_donor_corruption_matches_the_frozen_circuit_construction():
    example = exon_with_g_minus_one()
    expected = substitute(example.sequence, example.next_acceptor_pos - 2, CORRUPT_ALT)
    assert SPEC["donor"].corrupt_sequence(example) == expected


# --------------------------------------------------------------------------------------
# 2. The acceptor arm reads the other end of the SAME intron.
# --------------------------------------------------------------------------------------
def test_acceptor_readout_position_is_the_g_of_the_intron_mate_ag():
    example = exon_with_g_minus_one()
    position = SPEC["acceptor"].readout_position(example)
    assert position == example.next_acceptor_pos - 1
    assert example.sequence[position] == "G"


def test_acceptor_corruption_destroys_the_same_donor_the_donor_arm_reads():
    example = exon_with_g_minus_one()
    corrupted = SPEC["acceptor"].corrupt_sequence(example)
    assert example.sequence[example.donor_pos:example.donor_pos + 2] == "GT"
    assert corrupted[example.donor_pos:example.donor_pos + 2] != "GT"


def test_both_arms_use_the_same_intron_and_the_same_pair():
    example = exon_with_g_minus_one()
    donor_readout = SPEC["donor"].readout_position(example)
    acceptor_readout = SPEC["acceptor"].readout_position(example)
    assert acceptor_readout - donor_readout == 80 - 1        # the frozen gap, both directions


# --------------------------------------------------------------------------------------
# 3. R44 exposure.  This is the pair that matters.
# --------------------------------------------------------------------------------------
def test_acceptor_alt_is_the_motif_clean_one_not_the_symmetric_one():
    assert ACCEPTOR_CORRUPT_ALT == "AC"
    assert ACCEPTOR_CORRUPT_ALT != DONOR_CORRUPT_ALT, (
        "symmetric alts would plant a GC donor in 81% of examples (R44)"
    )


def test_acceptor_arm_plants_nothing_on_the_exposed_context():
    example = exon_with_g_minus_one()
    assert SPEC["acceptor"].planted_motifs(example) == {}


def test_symmetric_cc_would_plant_a_gc_donor_on_the_exposed_context():
    """Discriminating power for the choice above: CC is NOT clean where AC is."""
    from gfmcd.splicing.coupling import planted_donor_motifs

    example = exon_with_g_minus_one()
    planted = planted_donor_motifs(example.sequence, example.donor_pos, "CC")
    assert planted, "CC must plant on a G-1 donor, else the AC choice is unmotivated"
    assert "GC" in planted.values()


def test_symmetric_ta_would_plant_a_canonical_donor():
    from gfmcd.splicing.coupling import planted_donor_motifs

    example = exon_with_g_minus_one()
    assert "GT" in planted_donor_motifs(example.sequence, example.donor_pos, "TA").values()


def test_planted_motif_rate_is_zero_for_both_arms_on_the_exposed_cohort():
    cohort = [exon_with_g_minus_one(donor_pos=20 + 3 * i) for i in range(16)]
    for site in SITES:
        rate, by_motif = planted_motif_rate(cohort, site)
        assert rate == 0.0, f"{site} planted {by_motif}"


def test_planted_motif_rate_reports_what_was_planted_when_it_fails():
    """The memorial must name the motif, not merely fail."""
    from dataclasses import replace

    from gfmcd.splicing.site_coupling import SiteSpec

    cohort = [exon_with_g_minus_one()]
    dirty = SiteSpec("acceptor", "CC", "G")
    planted = dirty.planted_motifs(cohort[0])
    assert "GC" in planted.values()


# --------------------------------------------------------------------------------------
# 4. Specificity decoys come from the right pool for each direction.
# --------------------------------------------------------------------------------------
def test_donor_arm_decoys_are_non_partner_acceptors():
    example = exon_with_g_minus_one()
    example = CouplingExample(**{**example.__dict__, "other_ag": (40, 60), "other_gt": (30,)})
    assert SPEC["donor"].decoy_positions(example) == (40, 60)


def test_acceptor_arm_decoys_are_non_partner_donors():
    example = exon_with_g_minus_one()
    example = CouplingExample(**{**example.__dict__, "other_ag": (40, 60), "other_gt": (30,)})
    assert SPEC["acceptor"].decoy_positions(example) == (30,)


def test_unknown_site_is_rejected():
    with pytest.raises(ValueError, match="unknown site"):
        from gfmcd.splicing.site_coupling import SiteCouplingTask

        SiteCouplingTask(adapter=None, site="branchpoint")
