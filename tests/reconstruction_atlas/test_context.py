from __future__ import annotations

from gfmcd.reconstruction_atlas.context import (
    ContextIntervention,
    apply_context_exchange,
)
from gfmcd.reconstruction_atlas.schema import Interval


def fixed_encoder(sequence: str):
    ids = [101] + [200 + ord(base) for base in sequence] + [102]
    offsets = [(0, 0)] + [(i, i + 1) for i in range(len(sequence))] + [(0, 0)]
    return ids, offsets


def test_context_exchange_preserves_target_and_geometry() -> None:
    intervention = ContextIntervention(
        "x",
        "upstream_context",
        Interval(0, 2),
        Interval(4, 6),
        Interval(2, 4),
        "donor",
    )
    changed, audit = apply_context_exchange(
        "AACCGG",
        "TTTTAA",
        intervention,
        encoder=fixed_encoder,
    )
    assert changed == "AACCGG"  # donor[4:6] is AA, an explicit sham
    assert audit.changed_nucleotides == 0
    assert not audit.eligible

    changed, audit = apply_context_exchange(
        "AACCGG",
        "TTTTCC",
        intervention,
        encoder=fixed_encoder,
    )
    assert changed == "CCCCGG"
    assert audit.target_unchanged
    assert audit.target_geometry_invariant
    assert audit.eligible


def test_context_exchange_cannot_overlap_target() -> None:
    intervention = ContextIntervention(
        "x",
        "immediate_flank",
        Interval(1, 3),
        Interval(0, 2),
        Interval(2, 4),
        "donor",
    )
    try:
        apply_context_exchange("AACCGG", "TTTTCC", intervention, encoder=fixed_encoder)
    except ValueError as exc:
        assert "cannot alter" in str(exc)
    else:
        raise AssertionError("target-overlapping context exchange was accepted")
