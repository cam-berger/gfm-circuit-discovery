"""Which corruption alts CREATE a splice motif, exhaustively over every flanking context.

`ACCEPTOR_ALTS` / `DONOR_ALTS` guarantee only that the bases WRITTEN are not a motif. These
tests pin what they plant in the two dinucleotides that STRADDLE the edit -- the gap that
R44 records. They are written to FAIL if anyone reorders the alt tuples or "cleans up" the
clean-alt constants, because the study's unmasked-readout callers depend on the answer.

No model, no genome: all 16 flanking contexts are enumerated directly.
"""
from __future__ import annotations

import itertools

import pytest

from gfmcd.splicing.coupling import (
    ACCEPTOR_ALTS, DONOR_ALTS, MOTIF_CLEAN_ACCEPTOR_ALT, MOTIF_CLEAN_ALT,
    MOTIF_CLEAN_DONOR_ALT, destroy_acceptor_clean, destroy_donor_clean,
    planted_acceptor_motifs, planted_donor_motifs, planted_motifs,
)

BASES = "ACGT"
ALL_ALTS = [a + b for a in BASES for b in BASES]
PAD = "N" * 12


def _donor_seq(up: str, down: str) -> tuple[str, int]:
    """Sequence whose donor GT sits at index len(PAD)+1, with chosen -1 and +2 flanks."""
    seq = PAD + up + "GT" + down + PAD
    return seq, len(PAD) + 1


def _acceptor_seq(up: str, down: str) -> tuple[str, int]:
    """Sequence whose acceptor AG ends at index len(PAD)+2, with chosen -3 and +0 flanks."""
    seq = PAD + up + "AG" + down + PAD
    return seq, len(PAD) + 3


@pytest.mark.parametrize("up,down", list(itertools.product(BASES, BASES)))
def test_AC_plants_nothing_at_either_site_in_every_context(up, down):
    """The load-bearing property of MOTIF_CLEAN_ALT. If this fails, the constant is wrong."""
    seq, d = _donor_seq(up, down)
    assert planted_donor_motifs(seq, d, MOTIF_CLEAN_DONOR_ALT) == {}
    seq, a = _acceptor_seq(up, down)
    assert planted_acceptor_motifs(seq, a, MOTIF_CLEAN_ACCEPTOR_ALT) == {}


def test_AC_and_AT_are_the_only_universally_clean_alts():
    """Pins the derivation: X must be A, Y must be C or T, and XY != AG."""
    clean = []
    for alt in ALL_ALTS:
        if all(planted_motifs(PAD + up + "XX" + down + PAD, len(PAD) + 3, alt) == {}
               for up, down in itertools.product(BASES, BASES)):
            clean.append(alt)
    assert clean == ["AC", "AT"], clean
    assert MOTIF_CLEAN_ALT in clean


def test_every_shipped_donor_alt_except_AC_plants_a_motif_somewhere():
    """DONOR_ALTS is NOT four interchangeable options -- the R44 finding, pinned."""
    planted = {}
    for alt in DONOR_ALTS:
        hits = {}
        for up, down in itertools.product(BASES, BASES):
            seq, d = _donor_seq(up, down)
            for reg, motif in planted_donor_motifs(seq, d, alt).items():
                hits[f"{reg}:{motif}"] = hits.get(f"{reg}:{motif}", 0) + 1
        planted[alt] = hits
    # CC plants a GC donor one base upstream whenever the last exon base is G.
    assert "upstream:GC" in planted["CC"]
    # TA plants a CANONICAL GT upstream -- the worst case in the shipped tuple.
    assert "upstream:GT" in planted["TA"]
    # AA plants an acceptor downstream when the intronic +3 base is G.
    assert "downstream:AG" in planted["AA"]
    # AC is the only shipped member that plants nothing anywhere.
    assert planted["AC"] == {}


def test_every_shipped_acceptor_alt_plants_a_donor_and_CA_also_plants_an_acceptor():
    """Acceptor side is far less exposed, but it is NOT clean either."""
    planted = {}
    for alt in ACCEPTOR_ALTS:
        hits = {}
        for up, down in itertools.product(BASES, BASES):
            seq, a = _acceptor_seq(up, down)
            for reg, motif in planted_acceptor_motifs(seq, a, alt).items():
                hits[f"{reg}:{motif}"] = hits.get(f"{reg}:{motif}", 0) + 1
        planted[alt] = hits
    for alt in ACCEPTOR_ALTS:
        assert planted[alt], f"{alt} unexpectedly clean -- re-derive R44"
    assert "upstream:GC" in planted["CC"]
    assert "upstream:GT" in planted["TT"]
    # CA is the acceptor-side outlier: it rescues an AG one base downstream.
    assert "downstream:AG" in planted["CA"]


def test_clean_destroyers_actually_destroy_the_motif():
    """Cleanliness is worthless if the edit no longer destroys anything."""
    seq, d = _donor_seq("G", "A")
    assert seq[d:d + 2] == "GT"
    assert destroy_donor_clean(seq, d)[d:d + 2] == MOTIF_CLEAN_DONOR_ALT != "GT"
    seq, a = _acceptor_seq("C", "G")
    assert seq[a - 2:a] == "AG"
    assert destroy_acceptor_clean(seq, a)[a - 2:a] == MOTIF_CLEAN_ACCEPTOR_ALT != "AG"


def test_clean_destroyers_change_exactly_two_bases():
    """Length-preserving, locus-confined -- the operator class the preregs assume."""
    seq, d = _donor_seq("G", "A")
    out = destroy_donor_clean(seq, d)
    assert len(out) == len(seq)
    assert [i for i in range(len(seq)) if seq[i] != out[i]] == [d, d + 1]
