"""Model-free gate battery for designed register-lattice variants.

Every fixture here is hand-built (never imported from ``designed_register.py`` or
``pbc_v2_capacity.py``, which are being written concurrently) so each test is self-contained and
its numbers are independently checkable. Two fixture families recur:

* A small, purpose-built pair/triple of sequences for testing ONE gate in isolation (fastest to
  reason about, used for most clean/violation pairs).
* ``_build_clean_family()`` -- a single, larger (138nt) family that jointly satisfies all eight
  model-free gates at once, used for ``evaluate_family`` integration tests. It is built from a
  no-``T`` codon cycle (so no in-frame stop codon is possible in ANY reading frame) plus a
  register edit realised as "AAA" (a fixed, register-neutral 3nt spacer, kept so even the
  register-0 member has a genuine, non-degenerate edited seam) + ``r`` extra neutral T's at a
  seam placed 6nt from one end (so the frame-shifted tail is a small enough fraction of the
  sequence to keep composition/GC-position drift under the default tolerances).
"""
from __future__ import annotations

import numpy as np
import pytest

from gfmcd.splicing.designed_register_gates import (
    DEFAULT_TOLERANCES,
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
    seam_span,
)

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
# gate_designed_register
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
    result = gate_designed_register(variants, parent=parent, intended_registers=intended)
    assert result["passed"]
    assert result["failing_members"] == []
    assert all(m["match"] for m in result["measured"].values())


def test_gate_designed_register_catches_mislabeled_register():
    parent = "ACGTACGTAC" * 5
    variants = {"same": parent, "plus1": parent + "A"}
    intended = {"same": 0, "plus1": 2}  # WRONG: plus1's realised register is 1, not 2
    result = gate_designed_register(variants, parent=parent, intended_registers=intended)
    assert not result["passed"]
    assert result["failing_members"] == ["plus1"]
    assert result["measured"]["plus1"] == {"realised": 1, "intended": 2, "match": False}
    assert result["measured"]["same"]["match"]  # the untouched member is unaffected


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
    """A composition-only violation must not also flip the (length-based) register gate."""
    parent = "A" * 60
    variants = {"a": "A" * 40 + "C" * 20, "b": "G" * 40 + "T" * 20}  # same length as parent
    intended = {"a": 0, "b": 0}
    composition = gate_composition(variants, tolerance=0.05)
    register = gate_designed_register(variants, parent=parent, intended_registers=intended)
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
# gate_seam
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
    result = evaluate_family(variants, parent=parent, intended_registers=intended, tolerances=tolerances)
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
        block_unevaluated_gates=False,
    )
    assert not result["family_passes"]
    assert "designed_register" in result["failing_gates"]
    # the whole family -- including the failing member -- is still present, nothing was dropped
    assert result["members"] == list(variants)
    assert set(result["gates"]["designed_register"]["measured"]) == set(variants)
    assert result["gates"]["designed_register"]["measured"]["r1"]["match"] is False


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
        block_unevaluated_gates=False,
    )
    assert not result["family_passes"]
    # the return value carries only measurements and a verdict -- no filtered variant mapping
    assert "members" in result and result["members"] == list(variants)
