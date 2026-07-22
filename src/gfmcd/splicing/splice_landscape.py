"""Pinned local splice-landscape audits for sequence perturbations.

Only score windows that overlap the edited cassette: windows outside it are byte-identical and
cannot change.  MaxEntScan-compatible 23-nt acceptor and 9-nt donor scores are loaded lazily so the
core package remains importable when the optional ``maxentpy`` dependency is absent.  Publication
runners call :func:`maxent_scorers` up front and fail closed if either matrix is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable


ScoreFunction = Callable[[str], float]


@dataclass(frozen=True)
class LandscapeSummary:
    window_length: int
    start_indices: tuple[int, ...]
    scores: tuple[float, ...]
    max_score: float
    max_position: int
    n_strong: int


@dataclass(frozen=True)
class LandscapeChange:
    before: LandscapeSummary
    after: LandscapeSummary
    delta_max: float
    delta_n_strong: int
    new_competitive_site: bool


def overlapping_window_starts(
    sequence_length: int,
    edit_start: int,
    edit_end: int,
    *,
    window_length: int,
) -> tuple[int, ...]:
    """Starts of fixed-width windows whose intervals overlap an edited block."""

    if sequence_length <= 0 or window_length <= 0 or window_length > sequence_length:
        raise ValueError("invalid sequence/window length")
    if not (0 <= edit_start < edit_end <= sequence_length):
        raise ValueError("invalid edited block")
    lo = max(0, edit_start - window_length + 1)
    hi = min(sequence_length - window_length, edit_end - 1)
    return tuple(range(lo, hi + 1))


def summarize_landscape(
    sequence: str,
    edit_start: int,
    edit_end: int,
    *,
    window_length: int,
    scorer: ScoreFunction,
    strong_threshold: float = 6.0,
) -> LandscapeSummary:
    """Score every candidate splice window affected by an edit."""

    starts = overlapping_window_starts(
        len(sequence), edit_start, edit_end, window_length=window_length
    )
    scores = tuple(float(scorer(sequence[i:i + window_length].upper())) for i in starts)
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError("splice landscape scorer returned a non-finite value")
    best_index = max(range(len(scores)), key=lambda i: (scores[i], -starts[i]))
    return LandscapeSummary(
        window_length=window_length,
        start_indices=starts,
        scores=scores,
        max_score=scores[best_index],
        max_position=starts[best_index],
        n_strong=sum(score > strong_threshold for score in scores),
    )


def compare_landscapes(
    original: str,
    variant: str,
    edit_start: int,
    edit_end: int,
    *,
    window_length: int,
    scorer: ScoreFunction,
    strong_threshold: float = 6.0,
    tolerance: float = 1e-9,
) -> LandscapeChange:
    """Compare native and edited local landscapes with a conservative creation flag."""

    if len(original) != len(variant):
        raise ValueError("landscape comparison requires equal-length sequences")
    before = summarize_landscape(
        original, edit_start, edit_end, window_length=window_length,
        scorer=scorer, strong_threshold=strong_threshold,
    )
    after = summarize_landscape(
        variant, edit_start, edit_end, window_length=window_length,
        scorer=scorer, strong_threshold=strong_threshold,
    )
    new_competitive = after.max_score > max(before.max_score, strong_threshold) + tolerance
    return LandscapeChange(
        before=before,
        after=after,
        delta_max=after.max_score - before.max_score,
        delta_n_strong=after.n_strong - before.n_strong,
        new_competitive_site=new_competitive,
    )


@lru_cache(maxsize=1)
def maxent_scorers() -> tuple[ScoreFunction, ScoreFunction]:
    """Return pinned 3′ and 5′ MaxEnt scorers, failing closed when unavailable."""

    try:
        from maxentpy import maxent
        from maxentpy.maxent import load_matrix3, load_matrix5
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "P3 closure requires maxentpy with both 3' and 5' matrices; "
            "run in the pinned gfm-thrust1 environment"
        ) from exc
    matrix3 = load_matrix3()
    matrix5 = load_matrix5()

    def score3(window: str) -> float:
        if len(window) != 23 or set(window) - set("ACGT"):
            raise ValueError("3' MaxEnt windows must be 23 DNA bases")
        # MaxEntScan's 23-mer parameterization assumes the canonical acceptor is fixed at
        # positions 18:20.  Scoring arbitrary non-AG windows assigns a meaningless pseudo-score
        # and massively overcalls cryptic sites.
        if window[18:20] != "AG":
            return -100.0
        return float(maxent.score3(window, matrix=matrix3))

    def score5(window: str) -> float:
        if len(window) != 9 or set(window) - set("ACGT"):
            raise ValueError("5' MaxEnt windows must be 9 DNA bases")
        # The 9-mer model fixes the donor dinucleotide at positions 3:5.  GC donors are retained
        # because the study's canonical donor definition includes them as real 5' sites.
        if window[3:5] not in {"GT", "GC"}:
            return -100.0
        return float(maxent.score5(window, matrix=matrix5))

    return score3, score5


__all__ = [
    "LandscapeChange",
    "LandscapeSummary",
    "compare_landscapes",
    "maxent_scorers",
    "overlapping_window_starts",
    "summarize_landscape",
]
