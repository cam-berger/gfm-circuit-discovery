from __future__ import annotations

import pytest

from gfmcd.reconstruction_atlas.promoter_composition import (
    CoverDelta,
    cover_delta,
    cpg_density,
    gc_fraction,
)


def test_gc_fraction_counts_g_and_c_only() -> None:
    assert gc_fraction("GCGC") == pytest.approx(1.0)
    assert gc_fraction("ATAT") == pytest.approx(0.0)
    assert gc_fraction("GATC") == pytest.approx(0.5)


def test_gc_fraction_is_case_insensitive() -> None:
    assert gc_fraction("gcAT") == pytest.approx(0.5)


def test_gc_fraction_of_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        gc_fraction("")


def test_cpg_density_counts_only_cg_dinucleotides() -> None:
    # CGCG has CG at 0 and 2, GC at 1 -> 2 of 3 dinucleotides
    assert cpg_density("CGCG") == pytest.approx(2 / 3)
    # GCGC has CG at 1 only -> 1 of 3
    assert cpg_density("GCGC") == pytest.approx(1 / 3)
    assert cpg_density("AAAA") == pytest.approx(0.0)


def test_cpg_density_is_case_insensitive() -> None:
    assert cpg_density("cgcg") == pytest.approx(2 / 3)


def test_cpg_density_requires_at_least_two_bases() -> None:
    with pytest.raises(ValueError, match="two"):
        cpg_density("")
    with pytest.raises(ValueError, match="two"):
        cpg_density("C")


def test_cpg_density_with_two_bases_does_not_raise() -> None:
    assert cpg_density("CG") == pytest.approx(1.0)
    assert cpg_density("AT") == pytest.approx(0.0)


def test_cover_delta_is_target_minus_control() -> None:
    target = "AAAA" + "GCGC" + "AAAA"
    control = "AAAA" + "ATAT" + "AAAA"
    delta = cover_delta(
        target, control,
        target_start=4, target_end=8,
        control_start=4, control_end=8,
    )
    assert isinstance(delta, CoverDelta)
    assert delta.delta_gc == pytest.approx(1.0)
    assert delta.delta_cpg == pytest.approx(1 / 3)


def test_cover_delta_is_signed_not_absolute() -> None:
    target = "ATAT"
    control = "GCGC"
    delta = cover_delta(
        target, control,
        target_start=0, target_end=4,
        control_start=0, control_end=4,
    )
    assert delta.delta_gc == pytest.approx(-1.0)
    assert delta.delta_cpg == pytest.approx(-1 / 3)


def test_cover_delta_with_distinct_offsets() -> None:
    """Target and control offsets are distinct; swapping them gives wrong answer.

    This test catches bugs where offsets are swapped, reused across sequences,
    or sequences are swapped, because each distinct offset pair extracts
    sequences with very different compositions.
    """
    # target_seq: AAAA (GC=0, CpG=0) at [0:4], GCGC (GC=1.0, CpG=1/3) at [4:8]
    target_seq = "AAAA" + "GCGC"
    # control_seq: TTTT (GC=0, CpG=0) at [0:4], ATAT (GC=0, CpG=0) at [4:8]
    control_seq = "TTTT" + "ATAT"

    # Correct offsets: extract GCGC from target, TTTT from control
    delta = cover_delta(
        target_seq, control_seq,
        target_start=4, target_end=8,
        control_start=0, control_end=4,
    )
    assert delta.delta_gc == pytest.approx(1.0)
    assert delta.delta_cpg == pytest.approx(1 / 3)
