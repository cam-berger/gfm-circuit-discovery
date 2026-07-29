"""Composition primitives for the D13 composition-vs-position assay.

Standard library only, so this imports cleanly in both the default and the
pinned model environments.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CoverDelta", "cover_delta", "cpg_density", "gc_fraction"]
# Task 3 appends: "CoverPair", "RegionCover", "enumerate_region_covers",
#                 "select_matched_pair", "select_unmatched_pair"


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
