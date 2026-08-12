"""Model-free gate battery for designed register-lattice variants.

Every fixture here is hand-built (never imported from ``designed_register.py`` or
``pbc_v2_capacity.py``, which are being written concurrently) so each test is self-contained and
its numbers are independently checkable. Fixture families recur:

* A small, purpose-built pair/triple of sequences for testing ONE gate in isolation (fastest to
  reason about, used for most clean/violation pairs).
* ``_build_clean_family()`` -- a single, larger (138nt) family that jointly satisfies all eight
  model-free gates at once, used for ``evaluate_family`` integration tests. It is built from a
  no-``T`` codon cycle (so no in-frame stop codon is possible in ANY reading frame) plus a
  register edit realised as "AAA" (a fixed, register-neutral 3nt spacer, kept so even the
  register-0 member has a genuine, non-degenerate edited seam) + ``r`` extra neutral T's at a
  seam placed 6nt from one end (so the frame-shifted tail is a small enough fraction of the
  sequence to keep composition/GC-position drift under the default tolerances). This fixture
  realises its register by INSERTION (variable length across the family), so every call into it
  passes ``scheme="insertion"`` explicitly -- it is deliberately not representative of the
  PRIMARY ``"rotation"`` scheme (the module's new default), which is exercised by its own
  ROTATION_* fixtures below.
* ``ROTATION_PARENT`` / ``ROTATION_PROTECT_5P`` / ``ROTATION_PROTECT_3P`` -- a length-preserving
  fixture for the rotation scheme, built from ``designed_register.shift_module`` /
  ``register_triplet`` directly (the real, primary operator), used by the ``gate_designed_register``
  reconstruction tests and the ``gate_seam`` rotation-tolerance tests below. Its 12nt interior
  ("TTAAACCCGGGT") was picked because it has NO coincidental same-base repeats at the small shift
  offsets (3..9) these tests use, so a genuine rotation triplet built from it has an EXACT,
  verifiable positional-Hamming edit distance (12) for every arm -- see ``gate_seam``'s docstring
  for why that invariant, not an approximate one, is what makes tolerance 0 the correct default for
  this scheme.
"""
from __future__ import annotations

import numpy as np
import pytest

from gfmcd.splicing.designed_register import register_triplet, shift_module
from gfmcd.splicing.designed_register_gates import (
    DEFAULT_TOLERANCES,
    REALISATION_SCHEMES,
    SEAM_TOLERANCE_DEFAULTS,
    evaluate_family,
    gate_composition,
    gate_created_motifs,
    gate_designed_register,
    gate_gc_and_codon_position,
    gate_period3,
    gate_seam,
    gate_similarity_to_authentic,
    gate_site_scores,
    gate_stop_load,
    numeric_encode,
    period3_component,
    realised_register,
    seam_span,
)

# ---------------------------------------------------------------------------------------------
# Shared fixture: a real (length-preserving) rotation triplet built from the primary operator
# ---------------------------------------------------------------------------------------------
# Picked by search so that positions differ at EVERY interior base for shifts 3..9 (verified in
# review; see the module docstring note above) -- this is what makes the rotation-scheme edit
# distance in gate_seam land on an exact, not merely approximate, invariant for these tests.
ROTATION_INTERIOR = "TTAAACCCGGGT"
ROTATION_FLANK_5P = "GGGG"
ROTATION_FLANK_3P = "CCCC"
ROTATION_PARENT = ROTATION_FLANK_5P + ROTATION_INTERIOR + ROTATION_FLANK_3P
ROTATION_PROTECT_5P = len(ROTATION_FLANK_5P)
ROTATION_PROTECT_3P = len(ROTATION_FLANK_3P)


def _build_rotation_triplet(k: int):
    """A genuine register-0/+1/-1-matched triplet built via ``register_triplet`` + ``shift_module``
    on ``ROTATION_PARENT``, plus the ``shifts``/``intended_registers`` a caller would declare for it.
    """
    shifts = register_triplet(k)
    variants = {
        f"r{i}": shift_module(
            ROTATION_PARENT, shift=s, protect_5p=ROTATION_PROTECT_5P, protect_3p=ROTATION_PROTECT_3P
        )
        for i, s in enumerate(shifts)
    }
    declared_shifts = {f"r{i}": s for i, s in enumerate(shifts)}
    intended_registers = {f"r{i}": realised_register(s) for i, s in enumerate(shifts)}
    return variants, declared_shifts, intended_registers

# ---------------------------------------------------------------------------------------------
# Shared fixture: one family that jointly satisfies all 8 model-free gates
# ---------------------------------------------------------------------------------------------
_CODON_CYCLE = ("GAC", "GAG", "CAC", "GAA", "CAG", "GAC", "AAC", "GAG")  # never contains "T"


def _build_clean_family():
    prefix = "".join(_CODON_CYCLE) * 5 + "GAC" + "GAG" + "CAC" + "GAA"
    tail = "CAG" + "GAC"
    parent = prefix + tail
    seam = len(prefix)
    variants = {f"r{r}": parent[:seam] + "AAA" + "T" * r + parent[seam:] for r in (0, 1, 2)}
    intended_registers = {"r0": 0, "r1": 1, "r2": 2}
    # A pure-insertion register realisation necessarily costs `3 + r` edit operations (Levenshtein)
    # relative to `parent`, so edit distance spans a range of 2 across the family -- see this
    # module's report on gate_seam vs gate_designed_register for why 0 is the wrong default here.
    tolerances = {**DEFAULT_TOLERANCES, "seam_edit_distance": 2.0}
    return parent, variants, intended_registers, tolerances


# ---------------------------------------------------------------------------------------------
# gate_designed_register -- scheme="insertion" (SECONDARY, operator-robustness panel)
# ---------------------------------------------------------------------------------------------
def test_gate_designed_register_clean_family_passes():
    parent = "ACGTACGTAC" * 5
    variants = {
        "same": parent,           # length +0 -> register 0
        "plus3": parent + "AAA",  # length +3 -> register 0
        "plus1": parent + "A",    # length +1 -> register 1
        "minus2": parent[:-2],    # length -2 -> register 1 (== -2 % 3)
    }
    intended = {"same": 0, "plus3": 0, "plus1": 1, "minus2": 1}
    result = gate_designed_register(
        variants, parent=parent, intended_registers=intended, scheme="insertion"
    )
    assert result["passed"]
    assert result["failing_members"] == []
    assert all(m["match"] for m in result["measured"].values())


def test_gate_designed_register_catches_mislabeled_register():
    parent = "ACGTACGTAC" * 5
    variants = {"same": parent, "plus1": parent + "A"}
    intended = {"same": 0, "plus1": 2}  # WRONG: plus1's realised register is 1, not 2
    result = gate_designed_register(
        variants, parent=parent, intended_registers=intended, scheme="insertion"
    )
    assert not result["passed"]
    assert result["failing_members"] == ["plus1"]
    assert result["measured"]["plus1"] == {"realised": 1, "intended": 2, "match": False}
    assert result["measured"]["same"]["match"]  # the untouched member is unaffected


def test_gate_designed_register_rejects_unknown_scheme():
    """Catches an implementation that silently falls back to one scheme instead of validating
    the caller's request -- an unrecognised ``scheme`` must raise, not be quietly reinterpreted."""
    with pytest.raises(ValueError):
        gate_designed_register(
            {"a": "ACGT"}, parent="ACGT", intended_registers={"a": 0}, scheme="bogus"
        )


# ---------------------------------------------------------------------------------------------
# gate_designed_register -- scheme="rotation" (PRIMARY, genome-anchored operator; default)
#
# The bug this replaces: against shift_module (strictly length-preserving), a register gate that
# infers "realised register" from a length delta is either vacuously true (every rotation has
# length delta 0) or vacuously false for every variant, regardless of what register was actually
# realised -- a gate that cannot fail is worse than no gate. Every test below is built so that a
# register-only check (ignoring reconstruction) or a reconstruction-only check (ignoring the
# declared register) would each individually miss at least one of these cases.
# ---------------------------------------------------------------------------------------------
def test_realised_register_is_shift_mod_three():
    """Catches an implementation that ties register to a length delta (the OLD, wrong contract)
    or that mishandles negative/zero shifts: since shift_module never changes length, register
    must come from ``shift % 3`` alone, for every integer shift including negatives and multiples
    of 3."""
    cases = {
        0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2, 6: 0,
        -1: 2, -2: 1, -3: 0, -4: 2,
        3 * 7: 0, 3 * 7 + 1: 1, 3 * 7 + 2: 2,
    }
    for shift, expected in cases.items():
        assert realised_register(shift) == expected, shift


def test_gate_designed_register_rotation_reconstruction_passes_for_correctly_built_variant():
    """Catches an implementation that always reports ``reconstructed: False`` (or never computes
    it) -- a genuinely correct rotation variant, declared with its true shift, must reconstruct
    exactly and the gate must pass."""
    shift = 4  # register_triplet(1)'s middle (+1) arm
    variant = shift_module(
        ROTATION_PARENT, shift=shift, protect_5p=ROTATION_PROTECT_5P, protect_3p=ROTATION_PROTECT_3P
    )
    result = gate_designed_register(
        {"v": variant},
        parent=ROTATION_PARENT,
        intended_registers={"v": realised_register(shift)},
        scheme="rotation",
        shifts={"v": shift},
        protect_5p=ROTATION_PROTECT_5P,
        protect_3p=ROTATION_PROTECT_3P,
    )
    assert result["passed"]
    assert result["measured"]["v"]["reconstructed"] is True
    assert result["measured"]["v"]["register_match"] is True


def test_gate_designed_register_rotation_reconstruction_fails_when_declared_shift_off_by_one():
    """Catches an implementation that trusts the caller-declared ``shift`` for register (``shift
    % 3``) without verifying the variant's actual bytes against ``shift_module(parent,
    shift=...)``. The variant here is REALLY built with shift=4; the declared shift is wrong by
    one (5), and -- to isolate reconstruction from the register check -- the (also wrong) intended
    register is set to match the WRONG declared shift's own register (5 % 3 == 2). A register-only
    check would therefore wrongly PASS (declared register matches intended); only the byte-level
    reconstruction check catches that this variant was never produced by shift=5."""
    true_shift = 4
    wrong_shift = true_shift + 1
    variant = shift_module(
        ROTATION_PARENT,
        shift=true_shift,
        protect_5p=ROTATION_PROTECT_5P,
        protect_3p=ROTATION_PROTECT_3P,
    )
    result = gate_designed_register(
        {"v": variant},
        parent=ROTATION_PARENT,
        intended_registers={"v": realised_register(wrong_shift)},
        scheme="rotation",
        shifts={"v": wrong_shift},
        protect_5p=ROTATION_PROTECT_5P,
        protect_3p=ROTATION_PROTECT_3P,
    )
    assert not result["passed"]
    assert "v" in result["failing_members"]
    assert result["measured"]["v"]["register_match"] is True  # a register-only check would MISS this
    assert result["measured"]["v"]["reconstructed"] is False
    assert "reason" in result["measured"]["v"]


def test_gate_designed_register_rotation_reconstruction_fails_on_single_base_mutation():
    """Catches an implementation that checks only the declared shift's register and never touches
    the variant's actual content -- one interior base is flipped after a correct rotation, the
    declared shift and intended register are both left correct (so register_match stays True),
    and only the reconstruction check can detect the corruption."""
    shift = 4
    variant = shift_module(
        ROTATION_PARENT, shift=shift, protect_5p=ROTATION_PROTECT_5P, protect_3p=ROTATION_PROTECT_3P
    )
    pos = ROTATION_PROTECT_5P + 2  # inside the interior, away from the protected flanks
    mutated_base = "A" if variant[pos] != "A" else "C"
    mutated = variant[:pos] + mutated_base + variant[pos + 1 :]
    assert mutated != variant  # sanity: the mutation is real

    result = gate_designed_register(
        {"v": mutated},
        parent=ROTATION_PARENT,
        intended_registers={"v": realised_register(shift)},
        scheme="rotation",
        shifts={"v": shift},
        protect_5p=ROTATION_PROTECT_5P,
        protect_3p=ROTATION_PROTECT_3P,
    )
    assert not result["passed"]
    assert result["measured"]["v"]["register_match"] is True  # register alone would MISS this
    assert result["measured"]["v"]["reconstructed"] is False


def test_gate_designed_register_rotation_catches_register_mismatch_despite_valid_reconstruction():
    """Catches an implementation that only checks reconstruction and forgets to compare against
    the caller's claimed (intended) register: the variant here is a perfectly genuine,
    byte-for-byte reconstructible rotation by shift=4 (register 1) -- but the study claims it
    realises register 2. A reconstruction-only check would wrongly PASS."""
    shift = 4
    variant = shift_module(
        ROTATION_PARENT, shift=shift, protect_5p=ROTATION_PROTECT_5P, protect_3p=ROTATION_PROTECT_3P
    )
    result = gate_designed_register(
        {"v": variant},
        parent=ROTATION_PARENT,
        intended_registers={"v": (realised_register(shift) + 1) % 3},  # WRONG on purpose
        scheme="rotation",
        shifts={"v": shift},
        protect_5p=ROTATION_PROTECT_5P,
        protect_3p=ROTATION_PROTECT_3P,
    )
    assert not result["passed"]
    assert result["measured"]["v"]["reconstructed"] is True  # reconstruction alone would MISS this
    assert result["measured"]["v"]["register_match"] is False


def test_gate_designed_register_rotation_rejects_length_change_with_clear_reason():
    """Catches an implementation that blindly calls shift_module/compares without first checking
    length -- a length-preserving-operator variant that has, in fact, changed length (corruption,
    or a truncation/extension bug upstream) can never have been produced by any declared shift, and
    must be rejected with a reason that says so, not silently mis-scored or crashed on."""
    shift = 4
    variant = shift_module(
        ROTATION_PARENT, shift=shift, protect_5p=ROTATION_PROTECT_5P, protect_3p=ROTATION_PROTECT_3P
    )
    corrupted = variant + "A"  # length +1 -- impossible under a length-preserving operator
    result = gate_designed_register(
        {"v": corrupted},
        parent=ROTATION_PARENT,
        intended_registers={"v": realised_register(shift)},
        scheme="rotation",
        shifts={"v": shift},
        protect_5p=ROTATION_PROTECT_5P,
        protect_3p=ROTATION_PROTECT_3P,
    )
    assert not result["passed"]
    assert "v" in result["failing_members"]
    reason = result["measured"]["v"]["reason"]
    assert "length" in reason.lower()


def test_gate_designed_register_rotation_requires_shifts_and_protected_flanks():
    """Catches an implementation that silently defaults shifts/protect_5p/protect_3p (e.g. to 0)
    instead of demanding them -- under scheme="rotation" these are not optional, because register
    is a property of the declared shift plus reconstruction, never inferable from the variants
    alone."""
    with pytest.raises(ValueError):
        gate_designed_register(
            {"v": ROTATION_PARENT}, parent=ROTATION_PARENT, intended_registers={"v": 0}
        )  # scheme defaults to "rotation" but shifts/protect_5p/protect_3p are missing


def test_gate_designed_register_rotation_clean_triplet_passes():
    """Integration check: a full register_triplet-built triplet, declared honestly, passes as a
    whole -- catches an implementation that only handles the single-variant case correctly."""
    variants, shifts, intended = _build_rotation_triplet(k=1)
    result = gate_designed_register(
        variants,
        parent=ROTATION_PARENT,
        intended_registers=intended,
        scheme="rotation",
        shifts=shifts,
        protect_5p=ROTATION_PROTECT_5P,
        protect_3p=ROTATION_PROTECT_3P,
    )
    assert result["passed"]
    assert result["failing_members"] == []
    assert all(m["reconstructed"] for m in result["measured"].values())


# ---------------------------------------------------------------------------------------------
# gate_stop_load
# ---------------------------------------------------------------------------------------------
def test_gate_stop_load_clean_family_passes():
    parent = "GAC" * 10  # zero in-frame stop codons
    variants = {"a": parent, "b": parent[:-3] + "GAC"}
    result = gate_stop_load(variants, parent=parent, tolerance=0)
    assert result["passed"]
    assert result["reference"] == 0


def test_gate_stop_load_catches_extra_stop_codon():
    parent = "GAC" * 10
    variant_with_stop = parent[:-3] + "TGA"  # exactly one new in-frame stop
    result = gate_stop_load({"a": parent, "b": variant_with_stop}, parent=parent, tolerance=0)
    assert not result["passed"]
    assert result["failing_members"] == ["b"]
    assert result["measured"]["b"]["count"] == 1
    assert result["measured"]["b"]["deviation"] == 1


def test_gate_stop_load_tolerance_boundary_is_inclusive():
    """A deviation exactly equal to the tolerance PASSES (documented ``<=`` convention)."""
    parent = "GAC" * 10
    variant_with_stop = parent[:-3] + "TGA"  # deviation from reference == 1
    variants = {"a": parent, "b": variant_with_stop}
    at_boundary = gate_stop_load(variants, parent=parent, tolerance=1)
    just_below = gate_stop_load(variants, parent=parent, tolerance=0)
    assert at_boundary["passed"]       # deviation == tolerance -> PASS
    assert not just_below["passed"]    # deviation  > tolerance -> FAIL


# ---------------------------------------------------------------------------------------------
# gate_composition
# ---------------------------------------------------------------------------------------------
def test_gate_composition_clean_family_passes():
    base = "ACGTGCATACGTGCATACGTGCATACGTGCAT"
    twin = base[:-1] + ("A" if base[-1] != "A" else "C")  # one trailing base swapped
    result = gate_composition({"a": base, "b": twin}, tolerance=0.05)
    assert result["passed"]


def test_gate_composition_catches_composition_drift():
    variants = {"a": "A" * 40 + "C" * 20, "b": "G" * 40 + "T" * 20}  # disjoint alphabets
    result = gate_composition(variants, tolerance=0.05)
    assert not result["passed"]
    assert result["measured"]["overall_max_range"] > 0.5


def test_composition_violation_does_not_trip_register_gate():
    """A composition-only violation must not also flip the (length-based) insertion-scheme
    register gate."""
    parent = "A" * 60
    variants = {"a": "A" * 40 + "C" * 20, "b": "G" * 40 + "T" * 20}  # same length as parent
    intended = {"a": 0, "b": 0}
    composition = gate_composition(variants, tolerance=0.05)
    register = gate_designed_register(
        variants, parent=parent, intended_registers=intended, scheme="insertion"
    )
    assert not composition["passed"]  # this IS the composition violation
    assert register["passed"]         # ... and register must stay clean


# ---------------------------------------------------------------------------------------------
# gate_gc_and_codon_position
# ---------------------------------------------------------------------------------------------
def test_gate_gc_and_codon_position_clean_family_passes():
    seq = "GACGACGACGACGACGACGACGACGACGAC"
    result = gate_gc_and_codon_position({"a": seq, "b": seq}, tolerance=0.05)
    assert result["passed"]


def test_gate_gc_and_codon_position_catches_codon_position_drift():
    # "b" is "a" cyclically re-registered by one base: what was codon position 1 (all G) is now
    # codon position 3, so GC1/GC2/GC3 swap almost entirely between the two variants.
    variants = {
        "a": "GACGACGACGACGACGACGACGACGACGAC",
        "b": "AGCAGCAGCAGCAGCAGCAGCAGCAGCAGC",
    }
    result = gate_gc_and_codon_position(variants, tolerance=0.05)
    assert not result["passed"]
    assert result["measured"]["overall_max_range"] > 0.5


# ---------------------------------------------------------------------------------------------
# numeric_encode / period3_component (the pure DFT primitives)
# ---------------------------------------------------------------------------------------------
def test_numeric_encode_is_the_gc_indicator():
    encoded = numeric_encode("gcAT")
    assert list(encoded) == [1.0, 1.0, 0.0, 0.0]  # lowercase normalised, G/C=1, A/T=0


def test_period3_component_distinguishes_amplitude_from_phase():
    """Required check: a same-length rotation preserves amplitude EXACTLY and moves phase by
    the exact expected amount -- the clearest possible demonstration that the two are measured,
    and can diverge, independently."""
    seq = "GAT" * 20  # length 60 (multiple of 3), strong period-3 GC-content bias

    def rotate(s: str, r: int) -> str:
        return s[r:] + s[:r]

    amp0, phase0 = period3_component(rotate(seq, 0))
    amp1, phase1 = period3_component(rotate(seq, 1))
    amp2, phase2 = period3_component(rotate(seq, 2))

    assert amp0 == pytest.approx(amp1, abs=1e-9)
    assert amp0 == pytest.approx(amp2, abs=1e-9)
    assert amp0 > 0.5  # a real, non-trivial signal -- not a degenerate near-zero amplitude

    # phases must differ, and by exactly the DFT shift for a 3-sample-period circular rotation
    assert phase0 != pytest.approx(phase1, abs=1e-6)
    assert phase0 != pytest.approx(phase2, abs=1e-6)
    assert np.degrees(phase1) % 360 == pytest.approx(120.0, abs=1e-6)
    assert np.degrees(phase2) % 360 == pytest.approx(240.0, abs=1e-6)


def test_gate_period3_clean_family_passes():
    parent = "GAT" * 20  # 60nt
    seam = 57  # only the last 3nt (one codon) are reframed by the edit
    variants = {f"r{r}": parent[:seam] + "A" * r + parent[seam:] for r in (0, 1, 2)}
    intended = {"r0": 0, "r1": 1, "r2": 2}
    result = gate_period3(
        variants,
        parent=parent,
        intended_registers=intended,
        amplitude_tolerance=DEFAULT_TOLERANCES["period3_amplitude"],
        phase_tolerance_deg=DEFAULT_TOLERANCES["period3_phase_deg"],
    )
    assert result["passed"]
    assert result["amplitude_passed"]
    assert all(m["phase_error_deg"] == pytest.approx(0.0, abs=1e-6) for m in result["measured"].values())


def test_gate_period3_catches_wrong_register_phase():
    # A 50/50 split (large reframed tail) makes the register-carried phase signal strong enough
    # that a mislabeled register produces a large, easily-detected phase error -- see this
    # module's joint-satisfiability note on why a REALISTIC small-seam family (as used in the
    # "clean family passes" test above) has much lower sensitivity to this specific corruption.
    parent = "GAT" * 20  # 60nt
    seam = 30
    variants = {f"r{r}": parent[:seam] + "A" * r + parent[seam:] for r in (0, 1, 2)}
    bad_intended = {"r0": 0, "r1": 2, "r2": 2}  # r1's true realised register is 1, not 2
    result = gate_period3(
        variants,
        parent=parent,
        intended_registers=bad_intended,
        amplitude_tolerance=DEFAULT_TOLERANCES["period3_amplitude"],
        phase_tolerance_deg=DEFAULT_TOLERANCES["period3_phase_deg"],
    )
    assert not result["passed"]
    assert "r1" in result["failing_members"]


# ---------------------------------------------------------------------------------------------
# gate_created_motifs
# ---------------------------------------------------------------------------------------------
def test_gate_created_motifs_clean_family_passes():
    parent = "ACACACACACACACAC"  # no GT/AG anywhere
    result = gate_created_motifs({"a": parent + "CC"}, parent=parent, tolerance=0.0)
    assert result["passed"]
    assert result["measured"]["a"]["count"] == 0


def test_gate_created_motifs_catches_new_dinucleotide():
    parent = "ACACACACACACACAC"
    result = gate_created_motifs({"a": parent + "GT"}, parent=parent, tolerance=0.0)
    assert not result["passed"]
    assert result["measured"]["a"]["count"] == 1
    assert result["measured"]["a"]["created"]["GT"] == [16]


# ---------------------------------------------------------------------------------------------
# gate_seam -- scheme-independent behaviour (seam position matching), exercised at the
# default scheme ("rotation"); both fixtures below happen to be equal-length so this is valid
# under either scheme's edit-distance metric (verified: both metrics give 4 for this fixture).
# ---------------------------------------------------------------------------------------------
def test_gate_seam_clean_family_passes():
    parent = "AAAACCCCGGGGTTTT"
    variant_a = parent[:4] + "GGAA" + parent[8:]
    variant_b = parent[:4] + "AAGG" + parent[8:]  # same window, all 4 bases differ either way
    result = gate_seam({"a": variant_a, "b": variant_b}, parent=parent, tolerance=0)
    assert result["passed"]
    assert result["seam_positions_match"]
    assert result["edit_distance_spread"] == 0
    assert result["measured"]["a"]["edit_distance"] == result["measured"]["b"]["edit_distance"] == 4


def test_gate_seam_catches_mismatched_seam_position():
    parent = "AAAACCCCGGGGTTTT"
    variant_a = parent[:4] + "GGAA" + parent[8:]        # edits positions [4, 8)
    variant_shifted = "GGAA" + parent[4:]                # edits positions [0, 4) instead
    result = gate_seam({"a": variant_a, "d": variant_shifted}, parent=parent, tolerance=0)
    assert not result["passed"]
    assert not result["seam_positions_match"]


def test_gate_seam_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        gate_seam({"a": "ACGT"}, parent="ACGT", scheme="bogus")


def test_gate_seam_tolerance_defaults_are_scheme_dependent():
    """Catches an implementation that hardcodes one tolerance regardless of scheme, or that
    forgets to expose the mapping for audit -- the two schemes are edit-matched by different
    invariants (see ``gate_seam``'s docstring) and must not share a default tolerance."""
    assert SEAM_TOLERANCE_DEFAULTS["rotation"] == 0.0
    assert SEAM_TOLERANCE_DEFAULTS["insertion"] == 2.0
    assert set(SEAM_TOLERANCE_DEFAULTS) == set(REALISATION_SCHEMES)


# ---------------------------------------------------------------------------------------------
# gate_seam -- scheme="rotation" tolerance (PRIMARY; default tolerance 0)
#
# The bug this replaces: gate_seam's edit-distance tolerance was calibrated (2) for the
# insertion scheme, where a 0/+1/+2 register triplet legitimately spans an edit-distance range of
# 2. Under a shift_module rotation, all three arms of a register_triplet-built family rotate the
# SAME interior span with three distinct nonzero shifts; for ROTATION_INTERIOR (chosen with no
# coincidental same-base repeats at these offsets) that gives an EXACT, equal positional-Hamming
# distance for every arm -- so tolerance 2 would have silently let a genuinely mismatched triplet
# (built with different k's, hence different displacement) through, and tolerance 0 is what
# actually enforces edit-matching for this scheme.
# ---------------------------------------------------------------------------------------------
def test_gate_seam_rotation_tolerance_zero_passes_for_correct_shared_k_triplet():
    """A real register_triplet(k=1)-built triplet (shifts 3, 4, 5, all sharing k=1) must pass
    gate_seam at the rotation-scheme default tolerance (0), with EXACTLY zero edit-distance
    spread. Catches an implementation that computes edit distance via general Levenshtein instead
    of positional Hamming for this scheme: Levenshtein's optimal alignment can exploit an indel
    shortcut whose cost differs across these three shifts (verified against ``_levenshtein``: 6,
    8, 10 for shifts 3, 4, 5 respectively -- NOT equal, spread 4), which would wrongly fail this
    genuinely edit-matched triplet at tolerance 0."""
    variants, _, _ = _build_rotation_triplet(k=1)
    result = gate_seam(variants, parent=ROTATION_PARENT, scheme="rotation", tolerance=0)
    assert result["passed"]
    assert result["seam_positions_match"]
    assert result["edit_distance_spread"] == 0
    assert all(m["edit_distance"] == 12 for m in result["measured"].values())


def test_gate_seam_rotation_tolerance_zero_fails_when_one_arm_uses_a_different_k():
    """Swap one arm of a k=1 triplet for a shift belonging to a DIFFERENT k (still register-valid
    on its own, but not edit-matched to its siblings): the displaced span's relationship to the
    shared seam differs, so the rotation-scheme (positional Hamming) edit distance genuinely
    differs across the family, and tolerance 0 must catch it. Catches an implementation that
    reports a tolerance-independent "passed" (e.g. one that only checks seam_positions_match and
    ignores edit_distance_spread entirely) -- this fixture is built so seam_positions_match stays
    True and ONLY the edit-distance check can catch the mismatch."""
    variants, _, _ = _build_rotation_triplet(k=1)
    mismatched_shift = register_triplet(3)[1]  # k=3's middle arm: shift=10, not edit-matched to k=1
    variants = dict(variants)
    variants["r1"] = shift_module(
        ROTATION_PARENT,
        shift=mismatched_shift,
        protect_5p=ROTATION_PROTECT_5P,
        protect_3p=ROTATION_PROTECT_3P,
    )
    result = gate_seam(variants, parent=ROTATION_PARENT, scheme="rotation", tolerance=0)
    assert result["seam_positions_match"]  # isolate: NOT a seam-position confound
    assert result["edit_distance_spread"] > 0
    assert not result["passed"]


def test_gate_seam_rotation_rejects_length_changed_variant_with_a_reason():
    """A rotation-scheme variant whose length differs from parent cannot be Hamming-compared at
    all; the gate must fail it with a stated reason rather than raising or silently comparing a
    truncated pair. Catches an implementation that lets a length mismatch crash the whole gate (or
    coerce lengths silently) instead of reporting it per-member."""
    variants, _, _ = _build_rotation_triplet(k=1)
    variants = dict(variants)
    variants["r0"] = variants["r0"] + "A"  # corrupt: one base too long
    result = gate_seam(variants, parent=ROTATION_PARENT, scheme="rotation", tolerance=0)
    assert not result["passed"]
    assert result["measured"]["r0"]["edit_distance"] is None
    assert "reason" in result["measured"]["r0"]
    assert "length" in result["measured"]["r0"]["reason"].lower()


# ---------------------------------------------------------------------------------------------
# gate_seam -- scheme="insertion" tolerance (SECONDARY; default tolerance 2)
# ---------------------------------------------------------------------------------------------
def test_gate_seam_insertion_tolerance_two_passes_for_correct_0_plus1_plus2_triplet():
    """An insertion-scheme 0/+1/+2 triplet (a fixed "AAA" register-neutral spacer -- so even the
    register-0 member has a genuine, non-degenerate edited seam, not a variant literally identical
    to ``parent`` -- plus ``r`` extra neutral T's, ``r`` in {0, 1, 2}) spans a genuine edit-distance
    range of 2. Catches an implementation that reuses the rotation-scheme default (0) for
    insertion, which would wrongly fail this legitimately edit-matched family."""
    parent = "GACGACGACGACGACGACGACGACGAC"  # 27nt, no stop codons, no GT/AG
    seam = 12
    variants = {f"r{r}": parent[:seam] + "AAA" + "T" * r + parent[seam:] for r in (0, 1, 2)}
    result = gate_seam(variants, parent=parent, scheme="insertion")  # tolerance omitted -> default 2
    assert result["passed"]
    assert result["edit_distance_spread"] == 2


def test_gate_seam_insertion_tolerance_two_fails_wider_spread():
    """A 4th, wildly mismatched insertion (10 extra bases instead of 0/1/2) pushes the spread past
    the insertion-scheme default tolerance -- confirms tolerance 2 is an actual ceiling, not a
    number that happens to always pass."""
    parent = "GACGACGACGACGACGACGACGACGAC"
    seam = 12
    variants = {f"r{r}": parent[:seam] + "AAA" + "T" * r + parent[seam:] for r in (0, 1, 2)}
    variants["outlier"] = parent[:seam] + "AAA" + "T" * 10 + parent[seam:]
    result = gate_seam(variants, parent=parent, scheme="insertion")
    assert not result["passed"]
    assert result["edit_distance_spread"] > 2


# ---------------------------------------------------------------------------------------------
# gate_similarity_to_authentic
# ---------------------------------------------------------------------------------------------
def test_gate_similarity_to_authentic_clean_family_passes():
    parent = "AAAACCCCGGGGTTTT"
    variant = parent[:-1] + ("A" if parent[-1] != "A" else "C")  # one base changed
    result = gate_similarity_to_authentic({"a": variant}, parent=parent, tolerance=0.5)
    assert result["passed"]


def test_gate_similarity_to_authentic_catches_unrelated_sequence():
    parent = "AAAACCCC"
    unrelated = "TTTTTTTTTTTTTTTT"  # shares nothing with parent
    result = gate_similarity_to_authentic({"a": unrelated}, parent=parent, tolerance=0.5)
    assert not result["passed"]
    assert result["measured"]["a"]["inherited_fraction"] == 0.0


def test_gate_similarity_to_authentic_tolerance_boundary_is_inclusive():
    """This is the one FLOOR-direction gate: a fraction exactly AT tolerance PASSES (``>=``)."""
    parent = "AAAACCCC"
    variant = "AAAAGGGG"  # common prefix "AAAA" (4/8 = 0.5), no common suffix
    span = seam_span(variant, parent)
    assert (span["prefix_len"] + span["suffix_len"]) / len(variant) == pytest.approx(0.5)

    at_boundary = gate_similarity_to_authentic({"a": variant}, parent=parent, tolerance=0.5)
    just_above = gate_similarity_to_authentic({"a": variant}, parent=parent, tolerance=0.51)
    assert at_boundary["passed"]     # fraction == tolerance -> PASS
    assert not just_above["passed"]  # fraction  < tolerance -> FAIL


# ---------------------------------------------------------------------------------------------
# gate_site_scores -- the optional maxentpy-dependent gate
# ---------------------------------------------------------------------------------------------
def test_gate_site_scores_reports_unavailable_when_maxentpy_missing():
    """maxentpy is not installed in the required test environment (splicebert/biology/dev
    extras only, no `maxent` extra) -- this exercises the real absence, not a simulated one."""
    parent, variants, _, _ = _build_clean_family()
    result = gate_site_scores(variants, parent=parent, tolerance=2.0)
    assert result["available"] is False
    assert result["passed"] is None  # NEVER True: an unrun gate is not a passed gate
    assert "reason" in result and result["reason"]
    assert result["measured"] == {}


# ---------------------------------------------------------------------------------------------
# evaluate_family -- the single family-level verdict
# ---------------------------------------------------------------------------------------------
def test_evaluate_family_clean_family_passes_all_evaluable_gates():
    parent, variants, intended, tolerances = _build_clean_family()
    result = evaluate_family(
        variants,
        parent=parent,
        intended_registers=intended,
        tolerances=tolerances,
        scheme="insertion",  # this fixture realises register by insertion, not rotation
        block_unevaluated_gates=False,  # isolate: are the 8 evaluable gates clean?
    )
    assert result["failing_gates"] == []
    assert result["gates_not_evaluated"] == ["site_scores"]  # maxentpy genuinely absent here
    assert result["family_passes"]
    for name, gate_result in result["gates"].items():
        if name == "site_scores":
            continue
        assert gate_result["passed"], f"{name} unexpectedly failed: {gate_result}"


def test_evaluate_family_blocks_by_default_when_site_scores_unevaluated():
    """Required: with maxentpy unavailable, family_passes is False and gates_not_evaluated names
    the site-score gate -- even though every gate that COULD run, passed."""
    parent, variants, intended, tolerances = _build_clean_family()
    result = evaluate_family(
        variants, parent=parent, intended_registers=intended, tolerances=tolerances,
        scheme="insertion",
    )
    assert result["block_unevaluated_gates"] is True  # the documented default
    assert result["failing_gates"] == []               # nothing actually FAILED
    assert result["gates_not_evaluated"] == ["site_scores"]
    assert result["family_passes"] is False             # ... but the family is still rejected
    assert result["gates"]["site_scores"]["available"] is False


def test_evaluate_family_one_member_failing_rejects_whole_family():
    parent, variants, intended, tolerances = _build_clean_family()
    corrupted = dict(intended)
    corrupted["r1"] = 0  # WRONG: r1's realised register is 1 (it has one extra T)
    result = evaluate_family(
        variants, parent=parent, intended_registers=corrupted, tolerances=tolerances,
        scheme="insertion", block_unevaluated_gates=False,
    )
    assert not result["family_passes"]
    assert "designed_register" in result["failing_gates"]
    # the whole family -- including the failing member -- is still present, nothing was dropped
    assert result["members"] == list(variants)
    assert set(result["gates"]["designed_register"]["measured"]) == set(variants)
    assert result["gates"]["designed_register"]["measured"]["r1"]["match"] is False


def test_evaluate_family_threads_rotation_scheme_to_register_and_seam_gates():
    """Integration check for the default scheme: evaluate_family's ``designed_register`` and
    ``seam`` sub-gates must receive ``scheme``/``shifts``/``protect_5p``/``protect_3p`` correctly.
    ``ROTATION_PARENT``'s triplet is not engineered to satisfy every gate in the battery (unlike
    ``_build_clean_family``, built for the insertion scheme), so this checks the two scheme-aware
    gates specifically rather than the whole-family verdict. Catches an implementation that adds
    the ``scheme`` parameter to ``evaluate_family`` but forgets to actually pass it (or the
    shift/flank arguments) down to one of the two gates that need it."""
    variants, shifts, intended = _build_rotation_triplet(k=1)
    result = evaluate_family(
        variants,
        parent=ROTATION_PARENT,
        intended_registers=intended,
        tolerances=DEFAULT_TOLERANCES,
        scheme="rotation",
        shifts=shifts,
        protect_5p=ROTATION_PROTECT_5P,
        protect_3p=ROTATION_PROTECT_3P,
        block_unevaluated_gates=False,
    )
    assert result["gates"]["designed_register"]["passed"]
    assert result["gates"]["seam"]["passed"]
    assert "designed_register" not in result["failing_gates"]
    assert "seam" not in result["failing_gates"]


def test_evaluate_family_never_filters_members():
    """There must be no way to ask this module for a passing-only subset of a family: the whole
    point of the gate battery is that the family is rejected as a unit."""
    import gfmcd.splicing.designed_register_gates as module

    forbidden_fragments = ("filter", "drop", "remove", "subset", "exclude", "prune", "keep_only")
    public_names = [name for name in dir(module) if not name.startswith("_")]
    offenders = [
        name for name in public_names if any(frag in name.lower() for frag in forbidden_fragments)
    ]
    assert offenders == []

    parent, variants, intended, tolerances = _build_clean_family()
    corrupted = dict(intended)
    corrupted["r1"] = 0
    result = evaluate_family(
        variants, parent=parent, intended_registers=corrupted, tolerances=tolerances,
        scheme="insertion", block_unevaluated_gates=False,
    )
    assert not result["family_passes"]
    # the return value carries only measurements and a verdict -- no filtered variant mapping
    assert "members" in result and result["members"] == list(variants)


# ---------------------------------------------------------------------------------------------
# gate_period3 scheme awareness -- regression memorial for a gate that failed correct work
# ---------------------------------------------------------------------------------------------
def _rotation_family():
    """A CORRECT shared-k rotation triplet: the thing the gate must not reject."""
    from gfmcd.splicing.designed_register import register_triplet, shift_module

    # The interior MUST carry real period-3 structure in the GC indicator that
    # period3_component reads, or every amplitude is ~0 and a relative spread divides noise by
    # noise. "GCA" gives GC = 1,1,0 repeating: period 3 exactly. "ACTG" does NOT -- it repeats
    # with period 4 and produces a zero period-3 component.
    parent = "GGCCTT" + ("GCA" * 24) + "TTAACC"
    shifts = register_triplet(4)          # (12, 13, 14) -- shared k, registers 0/1/2
    variants = {
        f"r{s % 3}": shift_module(parent, shift=s, protect_5p=6, protect_3p=6) for s in shifts
    }
    registers = {f"r{s % 3}": s % 3 for s in shifts}
    return parent, variants, registers, shifts


def test_gate_period3_does_not_falsely_fail_a_correct_rotation_family():
    """THE regression memorial.

    The insertion expected-phase model assumes everything downstream of the seam shifts
    uniformly by `register` positions. A rotation permutes the interior IN PLACE, so nothing
    downstream moves, and applying that model produced ~60-120 degrees of spurious error per
    unit of register against a 20 degree tolerance -- rejecting genuinely correct families.

    Fails if the phase check is ever re-gated under the rotation scheme.
    """
    parent, variants, registers, _ = _rotation_family()
    out = gate_period3(
        variants, parent=parent, intended_registers=registers,
        amplitude_tolerance=0.10, phase_tolerance_deg=20.0, scheme="rotation",
        protect_5p=6, protect_3p=6,
    )
    assert out["passed"] is True
    assert out["amplitude_measured_on"] == "interior"
    assert out["amplitude_relative_spread"] < 1e-12   # rotation invariance is EXACT
    assert out["phase_gated"] is False
    assert out["phase_not_gated_reason"]
    assert "gate_designed_register" in out["phase_not_gated_reason"]


def test_gate_period3_still_gates_amplitude_under_rotation():
    """Phase is ungated; amplitude is NOT. Without this, the fix would have silently turned a
    real gate into a no-op."""
    parent, variants, registers, _ = _rotation_family()
    broken = dict(variants)
    broken[sorted(broken)[0]] = "A" * len(parent)      # destroys the period-3 magnitude
    out = gate_period3(
        broken, parent=parent, intended_registers=registers,
        amplitude_tolerance=0.10, phase_tolerance_deg=20.0, scheme="rotation",
        protect_5p=6, protect_3p=6,
    )
    assert out["amplitude_passed"] is False
    assert out["passed"] is False


def test_gate_period3_rejects_an_unknown_scheme():
    parent, variants, registers, _ = _rotation_family()
    with pytest.raises(ValueError, match="unknown scheme"):
        gate_period3(
            variants, parent=parent, intended_registers=registers,
            amplitude_tolerance=0.10, phase_tolerance_deg=20.0, scheme="teleportation",
        )


def test_amplitude_region_is_recorded_and_interior_invariance_is_exact():
    """Rotation invariance of the period-3 magnitude holds EXACTLY on the interior, because a
    circular rotation only multiplies each DFT bin by a phase.

    It does NOT hold in general on the whole sequence: the protected flanks do not rotate, so
    the total is |X_flank + X_interior * w^q| and flank-interior interference makes it register
    dependent. How large that effect is depends on how much period-3 content the flanks carry,
    so it is fixture dependent and is deliberately NOT asserted here as an inequality. What is
    asserted is the invariant that always holds, plus the fact that the gate RECORDS which
    region it measured -- so a reader can tell whether the number they are looking at is the
    exact invariant or the fixture-dependent one.
    """
    parent, variants, registers, _ = _rotation_family()
    whole = gate_period3(
        variants, parent=parent, intended_registers=registers,
        amplitude_tolerance=0.10, phase_tolerance_deg=20.0, scheme="rotation",
    )
    interior = gate_period3(
        variants, parent=parent, intended_registers=registers,
        amplitude_tolerance=0.10, phase_tolerance_deg=20.0, scheme="rotation",
        protect_5p=6, protect_3p=6,
    )
    assert whole["amplitude_measured_on"] == "whole_sequence"
    assert whole["amplitude_region_note"]
    assert interior["amplitude_measured_on"] == "interior"
    assert interior["amplitude_region_note"] is None
    assert interior["amplitude_relative_spread"] < 1e-12
