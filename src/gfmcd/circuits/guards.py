"""Permanent safeguards for patching experiments.

WHY THIS EXISTS
---------------
A two-base indexing error nearly produced a completely wrong mechanistic conclusion.

The acceptor's AG motif occupies ``[a-2, a-1]``; the corruption destroys exactly those
two bases.  The K/V patch was applied at ``a`` -- a position the corruption never touched.
Result: key+value recovery of **0.037**, which reads as "the acceptor's key and value
barely matter, so the effect must enter somewhere else."  Patching the window that
actually changed gives **0.711**.  Same code, same data, opposite conclusion, two bases.

A patch that does not overlap the corruption cannot restore it.  That is not a subtle
scientific judgement -- it is an arithmetic precondition, so it is enforced, not reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


class PatchWindowError(AssertionError):
    """A patch window does not overlap the corruption it claims to repair."""


@dataclass
class PatchWindow:
    """A patch site, with the corruption it is meant to restore.

    Records the exact token indices so they land in provenance -- an off-by-two is
    invisible in a summary statistic but obvious in a recorded index.
    """

    name: str
    patch_positions: tuple[int, ...]
    corruption_positions: tuple[int, ...]
    require_overlap: bool = True
    is_control: bool = False
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.is_control or not self.require_overlap:
            # a control window is SUPPOSED to miss the corruption; assert that it does
            if self.is_control and (set(self.patch_positions) & set(self.corruption_positions)):
                raise PatchWindowError(
                    f"{self.name}: control window OVERLAPS the corruption at "
                    f"{sorted(set(self.patch_positions) & set(self.corruption_positions))} "
                    "-- it is not a control")
            return
        overlap = set(self.patch_positions) & set(self.corruption_positions)
        if not overlap:
            raise PatchWindowError(
                f"{self.name}: patch window {sorted(self.patch_positions)} does not overlap "
                f"the corruption at {sorted(self.corruption_positions)}. A patch that does "
                "not cover the corrupted tokens cannot restore them; any low recovery is an "
                "indexing artifact, not a scientific result. "
                "(This guard exists because patching the acceptor at `a` instead of the AG "
                "at `[a-2, a-1]` turned a recovery of 0.711 into 0.037.)")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "patch_positions": list(self.patch_positions),
            "corruption_positions": list(self.corruption_positions),
            "overlap": sorted(set(self.patch_positions) & set(self.corruption_positions)),
            "is_control": self.is_control,
            **self.meta,
        }


def acceptor_motif_positions(acceptor_pos: int, offset: int = 0) -> tuple[int, ...]:
    """The AG lives at [a-2, a-1] -- NOT at a (which is the first exon base)."""
    return (acceptor_pos - 2 + offset, acceptor_pos - 1 + offset)


def donor_motif_positions(donor_pos: int, offset: int = 0) -> tuple[int, ...]:
    """The GT lives at [d, d+1]."""
    return (donor_pos + offset, donor_pos + 1 + offset)


def adjacent_window_controls(patch_positions: Sequence[int], corruption: Sequence[int],
                             shifts: Iterable[int] = (-8, +8)) -> list[PatchWindow]:
    """Shifted copies of the patch window that MISS the corruption.

    If a shifted window recovers as much as the real one, the "recovery" is not about the
    corrupted tokens at all.  Run these automatically, not on request.
    """
    out = []
    for s in shifts:
        shifted = tuple(p + s for p in patch_positions)
        if set(shifted) & set(corruption):
            continue  # too close to be a control at this shift
        out.append(PatchWindow(name=f"adjacent-control{s:+d}", patch_positions=shifted,
                               corruption_positions=tuple(corruption), is_control=True))
    return out


def validate_all(windows: Iterable[PatchWindow]) -> list[dict]:
    """Validate every window and return the provenance record."""
    recs = []
    for w in windows:
        w.validate()
        recs.append(w.to_dict())
    return recs
