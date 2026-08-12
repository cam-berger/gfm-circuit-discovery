"""Tests for the model-free designed-register operator (``gfmcd.splicing.designed_register``).

Every expected value below is a literal constant, computed by hand (verified once, offline,
with plain Python string slicing that does not import this module) and pasted in -- never a
value re-derived by calling ``shift_module``/``seam_census`` and comparing the result to a
second expression built from the same rotation formula. That is deliberate: a test that
recomputes ``interior[shift:] + interior[:shift]`` to check ``shift_module`` would pass
against almost any internally-consistent-but-wrong implementation of the same formula.
"""

from __future__ import annotations

from collections import Counter

import pytest

from gfmcd.splicing.designed_register import (
    register_triplet,
    seam_census,
    shift_module,
)


# ---------------------------------------------------------------------------
# Fixture 1: a plain module for the basic structural invariants (1-4).
# flank5="GGCC" (4) + interior="ACTGACTGACTGACTGACTG" (20) + flank3="TTAA" (4) = 28
# ---------------------------------------------------------------------------
BASIC_MODULE = "GGCCACTGACTGACTGACTGACTGTTAA"
BASIC_PROTECT_5P = 4
BASIC_PROTECT_3P = 4
BASIC_INTERIOR_LEN = 20


def test_shift_module_preserves_length():
    shifted = shift_module(
        BASIC_MODULE, shift=7, protect_5p=BASIC_PROTECT_5P, protect_3p=BASIC_PROTECT_3P
    )
    assert len(shifted) == 28
    assert len(shifted) == len(BASIC_MODULE)


def test_shift_module_preserves_nucleotide_multiset():
    shifted = shift_module(
        BASIC_MODULE, shift=7, protect_5p=BASIC_PROTECT_5P, protect_3p=BASIC_PROTECT_3P
    )
    # A rotation permutes characters; it may never substitute one for another.
    assert Counter(shifted) == Counter(BASIC_MODULE)


def test_shift_module_preserves_both_protected_flanks_byte_identically():
    shifted = shift_module(
        BASIC_MODULE, shift=7, protect_5p=BASIC_PROTECT_5P, protect_3p=BASIC_PROTECT_3P
    )
    assert shifted[:BASIC_PROTECT_5P] == "GGCC"
    assert shifted[len(shifted) - BASIC_PROTECT_3P :] == "TTAA"


def test_shift_module_full_rotation_is_identity():
    shifted = shift_module(
        BASIC_MODULE,
        shift=BASIC_INTERIOR_LEN,
        protect_5p=BASIC_PROTECT_5P,
        protect_3p=BASIC_PROTECT_3P,
    )
    assert shifted == BASIC_MODULE


# ---------------------------------------------------------------------------
# Fixture 2: register semantics (invariant 5).
#
# The interior is the exact 3-periodic repeat "ACG"x4: by construction the letter at any
# interior position names that base's own codon position (A=0, C=1, G=2), regardless of
# where the rotation later moves it to. That makes the predicted output of a given shift a
# fact one can state as a literal string by hand, and the "register label" of any resulting
# base a lookup by letter identity -- independent of shift_module's own rotation formula.
# ---------------------------------------------------------------------------
REGISTER_FLANK_5P = "TT"
REGISTER_FLANK_3P = "TT"
REGISTER_INTERIOR = "ACGACGACGACG"
REGISTER_MODULE = REGISTER_FLANK_5P + REGISTER_INTERIOR + REGISTER_FLANK_3P
CODON_LABEL = {"A": 0, "C": 1, "G": 2}


def test_shift_module_multiple_of_three_preserves_every_interior_register():
    # shift=3, shift % 3 == 0: hand-computed via "ACG"x4 rotated by 3 is itself (period 3).
    shifted = shift_module(REGISTER_MODULE, shift=3, protect_5p=2, protect_3p=2)
    assert shifted == "TTACGACGACGACGTT"

    interior_out = shifted[2:14]
    labels = [CODON_LABEL[base] for base in interior_out]
    assert labels == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_shift_module_shift_mod_three_equals_one_shifts_register_by_predicted_amount():
    # shift=4, shift % 3 == 1: hand-computed rotation of "ACG"x4 by 4.
    shifted = shift_module(REGISTER_MODULE, shift=4, protect_5p=2, protect_3p=2)
    assert shifted == "TTCGACGACGACGATT"

    interior_out = shifted[2:14]
    labels = [CODON_LABEL[base] for base in interior_out]
    # Every base's original register label is now offset by +1 relative to its new
    # position's index mod 3 (a base with original label L now sits where label (L-1)%3
    # would "belong" -- equivalently, reading off labels in position order gives this
    # explicit, hand-typed sequence rather than a formula re-derivation).
    assert labels == [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]


def test_shift_module_shift_mod_three_equals_two_shifts_register_by_predicted_amount():
    # shift=5, shift % 3 == 2: hand-computed rotation of "ACG"x4 by 5.
    shifted = shift_module(REGISTER_MODULE, shift=5, protect_5p=2, protect_3p=2)
    assert shifted == "TTGACGACGACGACTT"

    interior_out = shifted[2:14]
    labels = [CODON_LABEL[base] for base in interior_out]
    assert labels == [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]


# ---------------------------------------------------------------------------
# Fixture 3: register_triplet(k) is edit-matched (invariant 6).
# flank5="GGCC" (4) + interior="ACTGACTGACTGACTGACTG" (20) + flank3="TTAA" (4) = 28
# k=3 -> shifts (9, 10, 11). Independently verified (plain-Python rotation, not this
# module): wrap positions are {15, 14, 13} and every arm's Hamming distance is 20.
# ---------------------------------------------------------------------------
TRIPLET_MODULE = BASIC_MODULE
TRIPLET_PROTECT_5P = BASIC_PROTECT_5P
TRIPLET_PROTECT_3P = BASIC_PROTECT_3P


def test_register_triplet_shifts_are_edit_matched():
    shifts = register_triplet(3)
    assert shifts == (9, 10, 11)

    wrap_positions = []
    hamming_distances = []
    for shift in shifts:
        shifted = shift_module(
            TRIPLET_MODULE,
            shift=shift,
            protect_5p=TRIPLET_PROTECT_5P,
            protect_3p=TRIPLET_PROTECT_3P,
        )
        census = seam_census(
            TRIPLET_MODULE,
            shifted,
            protect_5p=TRIPLET_PROTECT_5P,
            protect_3p=TRIPLET_PROTECT_3P,
            shift=shift,
        )
        wrap_positions.append(census["junctions"]["internal_wrap"]["position"])
        hamming_distances.append(census["hamming_distance"])

    # Hand-verified exact values, not re-derived here.
    assert wrap_positions == [15, 14, 13]
    assert hamming_distances == [20, 20, 20]

    # The property invariant 6 actually asks for: wrap points within 2 nt of each other,
    # and Hamming distances equal to within 2 -- this is what "edit-matched" means, and it
    # is what breaks if the three shifts do not share one k.
    assert max(wrap_positions) - min(wrap_positions) <= 2
    assert max(hamming_distances) - min(hamming_distances) <= 2
    # Not a degenerate (no-op) rotation -- a real edit happened on every arm.
    assert min(hamming_distances) > 0


# ---------------------------------------------------------------------------
# Fixture 4: seam_census detects a planted GT at the internal wrap point (invariant 7).
#
# interior = "TAACCCCCG": first base 'T', last base 'G'. A circular rotation always makes
# the interior's original LAST base and original FIRST base neighbors at the wrap -- so
# "...G" meeting "T..." plants a GT there for any non-trivial shift, independent of the
# exact shift chosen. Hand-verified (plain-Python rotation, not this module).
# ---------------------------------------------------------------------------
WRAP_MODULE = "CCCTAACCCCCGCCC"
WRAP_SHIFT = 4
WRAP_PROTECT_5P = 3
WRAP_PROTECT_3P = 3
WRAP_SHIFTED = "CCCCCCCGTAACCCC"


def test_shift_module_wrap_fixture_matches_hand_computation():
    shifted = shift_module(
        WRAP_MODULE, shift=WRAP_SHIFT, protect_5p=WRAP_PROTECT_5P, protect_3p=WRAP_PROTECT_3P
    )
    assert shifted == WRAP_SHIFTED


def test_seam_census_detects_planted_gt_at_internal_wrap():
    census = seam_census(
        WRAP_MODULE,
        WRAP_SHIFTED,
        protect_5p=WRAP_PROTECT_5P,
        protect_3p=WRAP_PROTECT_3P,
        shift=WRAP_SHIFT,
    )
    wrap = census["junctions"]["internal_wrap"]
    assert wrap["position"] == 8
    assert wrap["dinucleotide_before"] == "CC"
    assert wrap["dinucleotide_after"] == "GT"
    assert wrap["created_motif"] == "GT"
    assert wrap["context_6mer"] == "CCGTAA"
    assert census["hamming_distance"] == 8
    assert census["created_motif_count"] == 1


# ---------------------------------------------------------------------------
# Fixture 5: seam_census detects a planted motif at a PROTECTED-FLANK join, not only at
# the internal wrap (invariant 8).
#
# flank5="AA", interior="CAGTTT" (6), flank3="TT". shift=2 rotates the interior's 'G' (at
# interior index 2) into the position right after the 5' flank, so flank5[-1]='A' meets
# interior_out[0]='G': a fresh "AG" at the 5'-flank/interior join. Before the edit that
# join was "A"+"C" = "AC". Hand-verified (plain-Python rotation, not this module); the
# rotation also relocates a pre-existing "GT" pair to a coordinate that held "CA" before,
# which is a second, independent created motif elsewhere in the module -- exactly the
# "away from the edited seam" case R44 warns about.
# ---------------------------------------------------------------------------
FLANK_MODULE = "AACAGTTTTT"
FLANK_SHIFT = 2
FLANK_PROTECT_5P = 2
FLANK_PROTECT_3P = 2
FLANK_SHIFTED = "AAGTTTCATT"


def test_shift_module_flank_fixture_matches_hand_computation():
    shifted = shift_module(
        FLANK_MODULE,
        shift=FLANK_SHIFT,
        protect_5p=FLANK_PROTECT_5P,
        protect_3p=FLANK_PROTECT_3P,
    )
    assert shifted == FLANK_SHIFTED


def test_seam_census_detects_planted_motif_at_protected_flank_join():
    census = seam_census(
        FLANK_MODULE,
        FLANK_SHIFTED,
        protect_5p=FLANK_PROTECT_5P,
        protect_3p=FLANK_PROTECT_3P,
        shift=FLANK_SHIFT,
    )
    flank5_join = census["junctions"]["flank5_interior"]
    assert flank5_join["position"] == 2
    assert flank5_join["dinucleotide_before"] == "AC"
    assert flank5_join["dinucleotide_after"] == "AG"
    assert flank5_join["created_motif"] == "AG"
    assert flank5_join["context_6mer"] == "AAGTT"

    # Confirm this creation is genuinely at the protected-flank join, not a restatement of
    # a wrap-point creation: the internal wrap itself plants nothing here.
    assert census["junctions"]["internal_wrap"]["created_motif"] is None
    assert census["junctions"]["interior_flank3"]["created_motif"] is None

    assert census["hamming_distance"] == 5
    # Two created motifs total: the flank join above, plus one more elsewhere in the
    # module where the rotation relocated an intact "GT" pair to a new coordinate.
    assert census["created_motif_count"] == 2


# ---------------------------------------------------------------------------
# Invariant 9: k <= 0 raises.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_k", [0, -1, -5])
def test_register_triplet_rejects_non_positive_k(bad_k):
    with pytest.raises(ValueError):
        register_triplet(bad_k)
