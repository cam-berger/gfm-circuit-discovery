"""Context-removal controls for deciding whether a readout uses distal information."""

from __future__ import annotations

from typing import Iterable, Literal

from gfmcd.splicing.schema import SplicePairExample


def _paint(mask: list[bool], start: int, end: int) -> None:
    for i in range(max(0, start), min(len(mask), end)):
        mask[i] = True


def retain_site_windows(
    example: SplicePairExample,
    *,
    flank: int = 12,
    include_all_candidates: bool = True,
    fill: str = "N",
) -> str:
    """Keep local splice-site neighborhoods and replace all other bases.

    Candidate coordinates and sequence length remain fixed.  The control tests
    whether behavior survives when endpoint-local motifs remain but distal
    sequence content is unavailable.
    """
    if len(fill) != 1:
        raise ValueError("fill must be one character")
    positions: Iterable[int]
    if include_all_candidates:
        positions = [x.position for x in example.acceptors + example.donors]
    else:
        positions = [example.true_pair.acceptor_pos, example.true_pair.donor_pos]
    keep = [False] * len(example.sequence)
    for position in positions:
        _paint(keep, position - flank, position + flank)
    return "".join(base if kept else fill for base, kept in zip(example.sequence, keep))


def mask_true_partner_neighborhood(
    example: SplicePairExample,
    *,
    kind: Literal["acceptor", "donor"],
    flank: int = 12,
    fill: str = "N",
) -> str:
    """Remove one true endpoint neighborhood while preserving all coordinates."""
    position = (
        example.true_pair.acceptor_pos if kind == "acceptor" else example.true_pair.donor_pos
    )
    start, end = max(0, position - flank), min(len(example.sequence), position + flank)
    return example.sequence[:start] + fill * (end - start) + example.sequence[end:]


def retain_pair_endpoints_only(
    example: SplicePairExample,
    *,
    flank: int = 12,
    fill: str = "N",
) -> str:
    """Keep only the true acceptor and donor neighborhoods."""
    keep = [False] * len(example.sequence)
    _paint(keep, example.true_pair.acceptor_pos - flank, example.true_pair.acceptor_pos + flank)
    _paint(keep, example.true_pair.donor_pos - flank, example.true_pair.donor_pos + flank)
    return "".join(base if kept else fill for base, kept in zip(example.sequence, keep))
