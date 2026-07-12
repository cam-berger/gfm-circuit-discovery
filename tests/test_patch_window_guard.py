"""Memorial test for the two-base patch-window bug.

Patching the acceptor at `a` instead of the AG at `[a-2, a-1]` turned a key+value
recovery of 0.711 into 0.037 -- the same code and data supporting the opposite
mechanistic conclusion.  The guard makes that class of error impossible to ship.
"""
from __future__ import annotations

import pytest

from gfmcd.circuits.guards import (
    PatchWindow, PatchWindowError, acceptor_motif_positions, adjacent_window_controls,
    donor_motif_positions, validate_all,
)


def test_the_actual_bug_is_now_rejected():
    """THE BUG: corruption at the AG [a-2, a-1], patch applied at `a`."""
    a = 300
    corruption = acceptor_motif_positions(a)          # (298, 299)
    with pytest.raises(PatchWindowError, match="does not overlap"):
        PatchWindow("key@acceptor", patch_positions=(a,), corruption_positions=corruption
                    ).validate()


def test_the_corrected_window_passes():
    a = 300
    corruption = acceptor_motif_positions(a)
    PatchWindow("key@acceptor", patch_positions=(a - 3, a - 2, a - 1, a),
                corruption_positions=corruption).validate()


def test_ag_is_not_at_the_acceptor_position():
    """The precondition the bug violated, stated once, explicitly."""
    assert acceptor_motif_positions(300) == (298, 299)
    assert 300 not in acceptor_motif_positions(300)
    assert donor_motif_positions(500) == (500, 501)


def test_a_control_window_must_actually_miss_the_corruption():
    a = 300
    corruption = acceptor_motif_positions(a)
    bad = PatchWindow("ctrl", patch_positions=(a - 2,), corruption_positions=corruption,
                      is_control=True)
    with pytest.raises(PatchWindowError, match="not a control"):
        bad.validate()


def test_adjacent_controls_are_generated_and_miss_the_corruption():
    a = 300
    corruption = acceptor_motif_positions(a)
    ctrls = adjacent_window_controls((a - 3, a - 2, a - 1, a), corruption)
    assert ctrls
    for c in ctrls:
        assert not (set(c.patch_positions) & set(corruption))
        c.validate()


def test_provenance_records_exact_token_indices():
    a = 300
    recs = validate_all([PatchWindow("key@acceptor", (a - 3, a - 2, a - 1, a),
                                     acceptor_motif_positions(a))])
    assert recs[0]["patch_positions"] == [297, 298, 299, 300]
    assert recs[0]["corruption_positions"] == [298, 299]
    assert recs[0]["overlap"] == [298, 299]      # an off-by-two is invisible in a mean,
                                                 # but obvious in a recorded index
