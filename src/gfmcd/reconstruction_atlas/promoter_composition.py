"""Composition primitives for the D13 composition-vs-position assay.

Standard library only, so this imports cleanly in both the default and the
pinned model environments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gfmcd.reconstruction_atlas.promoter_source_localization import (
    TokenTile,
    paired_fixed_dose_tile,
)

__all__ = [
    "CoverDelta",
    "CoverPair",
    "RegionCover",
    "cover_delta",
    "cpg_density",
    "enumerate_region_covers",
    "gc_fraction",
    "select_matched_pair",
    "select_unmatched_pair",
]


def gc_fraction(seq: str) -> float:
    """GC content as a fraction of total bases.

    Args:
        seq: DNA sequence (case-insensitive).

    Returns:
        Fraction of bases that are G or C.

    Raises:
        ValueError: If sequence is empty.
    """
    if not seq:
        raise ValueError("cannot compute GC fraction of an empty sequence")
    upper = seq.upper()
    return sum(base in "GC" for base in upper) / len(upper)


def cpg_density(seq: str) -> float:
    """CpG dinucleotides per dinucleotide position.

    Counts the literal 5'-CG-3' dinucleotide only; GC is not a CpG.

    Args:
        seq: DNA sequence (case-insensitive).

    Returns:
        Fraction of dinucleotide positions that are CG.

    Raises:
        ValueError: If sequence has fewer than two bases.
    """
    if len(seq) < 2:
        raise ValueError("cannot compute CpG density from fewer than two bases")
    upper = seq.upper()
    return sum(
        upper[index : index + 2] == "CG" for index in range(len(upper) - 1)
    ) / (len(upper) - 1)


@dataclass(frozen=True)
class CoverDelta:
    """Composition difference for one cover pair.

    Attributes:
        delta_gc: Target minus control GC fraction.
        delta_cpg: Target minus control CpG density.
    """

    delta_gc: float
    delta_cpg: float


def cover_delta(
    target_seq: str,
    control_seq: str,
    *,
    target_start: int,
    target_end: int,
    control_start: int,
    control_end: int,
) -> CoverDelta:
    """Signed target-minus-control composition difference for one cover pair.

    The sign is load-bearing: the confound under test is that the target-minus-
    control differential is systematically larger downstream than upstream, so
    an absolute value here would destroy the very quantity being matched.

    Args:
        target_seq: Full target sequence.
        control_seq: Full control sequence.
        target_start: Start position (inclusive) in target_seq.
        target_end: End position (exclusive) in target_seq.
        control_start: Start position (inclusive) in control_seq.
        control_end: End position (exclusive) in control_seq.

    Returns:
        CoverDelta with signed composition differences.
    """
    target = target_seq[target_start:target_end]
    control = control_seq[control_start:control_end]
    return CoverDelta(
        delta_gc=gc_fraction(target) - gc_fraction(control),
        delta_cpg=cpg_density(target) - cpg_density(control),
    )


@dataclass(frozen=True)
class RegionCover:
    """One nominal position, with BOTH branches' covers and their signed delta."""

    target: TokenTile
    control: TokenTile
    delta: CoverDelta
    nominal_start: int
    nominal_end: int


@dataclass(frozen=True)
class CoverPair:
    """A downstream/upstream pair of covers proposed for a D13 comparison."""

    downstream: RegionCover
    upstream: RegionCover
    gc_separation: float
    cpg_separation: float


def enumerate_region_covers(
    target_offsets: Sequence[Sequence[int]],
    control_offsets: Sequence[Sequence[int]],
    target_seq: str,
    control_seq: str,
    *,
    region_start: int,
    region_end: int,
    tss_index: int,
    nominal_width: int,
    token_dose: int,
    maximum_cover_nt: int,
    maximum_cover_length_difference_nt: int,
) -> list[RegionCover]:
    """Every positionally-paired cover whose nominal window lies in the region.

    Region bounds are TSS-relative; offsets and nominal windows are
    window-absolute. Delegates the target/control pairing to the already-tested
    ``paired_fixed_dose_tile`` so both branches are always masked at the SAME
    nominal position for every candidate — the precondition for
    ``target_loss - control_loss`` to mean anything. Target and control covers
    are never enumerated separately and matched after the fact.

    Outcome-blind: reads sequence and token geometry only, never a model score.
    """
    covers: list[RegionCover] = []
    lowest = region_start + tss_index
    highest = region_end + tss_index - nominal_width
    for nominal_start in range(lowest, highest + 1):
        nominal_end = nominal_start + nominal_width
        paired = paired_fixed_dose_tile(
            target_offsets,
            control_offsets,
            nominal_start=nominal_start,
            nominal_end=nominal_end,
            token_dose=token_dose,
            maximum_cover_nt=maximum_cover_nt,
            maximum_cover_length_difference_nt=maximum_cover_length_difference_nt,
        )
        if paired is None:
            continue
        target_tile, control_tile = paired
        centre = (target_tile.cover_start + target_tile.cover_end) / 2 - tss_index
        if not region_start <= centre <= region_end:
            continue
        covers.append(
            RegionCover(
                target=target_tile,
                control=control_tile,
                delta=cover_delta(
                    target_seq,
                    control_seq,
                    target_start=target_tile.cover_start,
                    target_end=target_tile.cover_end,
                    control_start=control_tile.cover_start,
                    control_end=control_tile.cover_end,
                ),
                nominal_start=nominal_start,
                nominal_end=nominal_end,
            )
        )
    return covers


def _separations(
    downstream: Sequence[RegionCover], upstream: Sequence[RegionCover]
) -> list[tuple[RegionCover, RegionCover, float, float]]:
    """Every downstream/upstream combination with its absolute composition gap."""
    return [
        (
            down,
            up,
            abs(down.delta.delta_gc - up.delta.delta_gc),
            abs(down.delta.delta_cpg - up.delta.delta_cpg),
        )
        for down in downstream
        for up in upstream
    ]


def select_matched_pair(
    downstream: Sequence[RegionCover],
    upstream: Sequence[RegionCover],
    *,
    caliper_gc: float,
    caliper_cpg: float,
) -> CoverPair | None:
    """Tightest composition-matched downstream/upstream pair inside both calipers.

    Outcome-blind: ranks on sequence composition only. ``None`` when no
    combination clears both calipers — never widen the caliper to force a
    match; a caliper that nothing clears is itself a finding.
    """
    best: tuple[tuple[float, ...], CoverPair] | None = None
    for down, up, gc_sep, cpg_sep in _separations(downstream, upstream):
        if gc_sep > caliper_gc or cpg_sep > caliper_cpg:
            continue
        rank = (gc_sep, cpg_sep, down.nominal_start, up.nominal_start)
        candidate = CoverPair(down, up, gc_sep, cpg_sep)
        if best is None or rank < best[0]:
            best = (rank, candidate)
    return None if best is None else best[1]


def select_unmatched_pair(
    downstream: Sequence[RegionCover],
    upstream: Sequence[RegionCover],
    *,
    minimum_gc_separation: float,
) -> CoverPair | None:
    """Most strongly composition-SEPARATED pair — the planted-positive arm.

    Without this arm a null matched-arm result is uninterpretable, because it
    cannot be distinguished from an instrument that detects nothing at all.
    """
    best: tuple[tuple[float, ...], CoverPair] | None = None
    for down, up, gc_sep, cpg_sep in _separations(downstream, upstream):
        if gc_sep < minimum_gc_separation:
            continue
        rank = (-gc_sep, -cpg_sep, down.nominal_start, up.nominal_start)
        candidate = CoverPair(down, up, gc_sep, cpg_sep)
        if best is None or rank < best[0]:
            best = (rank, candidate)
    return None if best is None else best[1]
