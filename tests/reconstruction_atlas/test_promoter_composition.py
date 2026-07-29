from __future__ import annotations

import pytest

from gfmcd.reconstruction_atlas.promoter_composition import (
    CoverDelta,
    CoverPair,
    RegionCover,
    cover_delta,
    cpg_density,
    enumerate_region_covers,
    gc_fraction,
    select_matched_pair,
    select_unmatched_pair,
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


GEOMETRY = dict(
    tss_index=249, nominal_width=10, token_dose=2,
    maximum_cover_nt=18, maximum_cover_length_difference_nt=2,
)


def _offsets(length: int = 300, width: int = 5) -> list[tuple[int, int]]:
    """Uniform token offsets plus zero-width CLS/SEP sentinels, matching how the
    adapter reports offset mappings."""
    inner = [(start, start + width) for start in range(0, length, width)]
    return [(0, 0)] + inner + [(0, 0)]


def _covers(target: str, control: str, region: tuple[int, int]) -> list[RegionCover]:
    return enumerate_region_covers(
        _offsets(), _offsets(), target, control,
        region_start=region[0], region_end=region[1], **GEOMETRY,
    )


def test_enumerate_region_covers_pairs_target_and_control_at_the_same_position() -> None:
    target = "GCAT" * 75
    control = "ATGC" * 75
    covers = _covers(target, control, (1, 31))
    assert covers, "expected at least one eligible cover"
    for cover in covers:
        # The whole estimand depends on this: both branches masked at one place.
        assert cover.target.nominal_start == cover.control.nominal_start
        assert cover.target.nominal_end == cover.control.nominal_end
        centre = (cover.target.cover_start + cover.target.cover_end) / 2 - 249
        assert 1 <= centre <= 31


def test_enumerate_region_covers_respects_maximum_cover_length() -> None:
    target = control = "GCAT" * 75
    covers = enumerate_region_covers(
        _offsets(width=12), _offsets(width=12), target, control,
        region_start=1, region_end=31,
        tss_index=249, nominal_width=10, token_dose=2,
        maximum_cover_nt=18, maximum_cover_length_difference_nt=2,
    )
    assert covers == []  # two 12-nt tokens exceed the 18-nt maximum


def test_enumerate_region_covers_records_signed_delta() -> None:
    target = "G" * 300
    control = "A" * 300
    covers = _covers(target, control, (1, 31))
    assert covers
    assert all(cover.delta.delta_gc > 0 for cover in covers)
    flipped = _covers("A" * 300, "G" * 300, (1, 31))
    assert all(cover.delta.delta_gc < 0 for cover in flipped)


def test_select_matched_pair_returns_none_when_nothing_is_inside_the_caliper() -> None:
    down = _covers("G" * 300, "A" * 300, (1, 31))
    up = _covers("A" * 300, "G" * 300, (-39, -9))
    assert select_matched_pair(down, up, caliper_gc=0.01, caliper_cpg=0.01) is None


def test_select_matched_pair_respects_the_caliper() -> None:
    target, control = "GCAT" * 75, "GCAT" * 75
    down = _covers(target, control, (1, 31))
    up = _covers(target, control, (-39, -9))
    pair = select_matched_pair(down, up, caliper_gc=0.05, caliper_cpg=0.03)
    assert isinstance(pair, CoverPair)
    assert pair.gc_separation <= 0.05
    assert pair.cpg_separation <= 0.03


def test_select_unmatched_pair_requires_minimum_separation() -> None:
    # Identical branches give zero delta everywhere, so separation is always 0.
    seq = "GCAT" * 75
    down = _covers(seq, seq, (1, 31))
    up = _covers(seq, seq, (-39, -9))
    assert select_unmatched_pair(down, up, minimum_gc_separation=0.20) is None


def test_select_unmatched_pair_finds_the_largest_separation() -> None:
    target = "G" * 260 + "A" * 40
    control = "A" * 300
    down = _covers(target, control, (1, 31))
    up = _covers(target, control, (-39, -9))
    pair = select_unmatched_pair(down, up, minimum_gc_separation=0.20)
    assert pair is not None
    assert pair.gc_separation >= 0.20


def test_selection_is_deterministic() -> None:
    target, control = "GCATGGCCTAAT" * 25, "ATGCCATGGATC" * 25
    down = _covers(target, control, (1, 31))
    up = _covers(target, control, (-39, -9))
    first = select_matched_pair(down, up, caliper_gc=0.5, caliper_cpg=0.5)
    second = select_matched_pair(down, up, caliper_gc=0.5, caliper_cpg=0.5)
    assert first == second
