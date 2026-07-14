"""Tripwires for "what is a bin's phase?"  Answer: the carriers do not have one.

A bin of a [1026 x 512] table gives a complex VECTOR in C^512 -- a phase per embedding dimension.
Its contribution traces an ELLIPSE in the plane spanned by (Re F[k], Im F[k]) as position advances.
Both carriers are near-perfect CIRCLES, so every phase in [0, 2pi) is realised in SOME direction
and no single phase is a property of the bin.  The heads CHOOSE one, by choosing a direction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[1] / "results/phase-lattice/carrier_phase.json"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run_carrier_phase.py has not been run")


@pytest.fixture(scope="module")
def R():
    return json.loads(DATA.read_text())


def test_the_carriers_ARE_CIRCLES_and_therefore_HAVE_NO_PHASE(R):
    """*** Do not quote 'the phase of the period-3 bin'.  There isn't one. ***"""
    by = {b["bin"]: b for b in R["bins"]}
    for k, nm in ((342, "the codon clock"), (96, "the second carrier")):
        assert by[k]["circularity"] > 0.95, f"{nm} traces a circle -- a pure rotation"
        assert by[k]["phase_concentration"] < 0.10, (
            f"{nm}'s 512 dimensions do NOT agree on a phase -- they are spread over [0, 2pi)"
        )


def test_the_SLOW_modes_DO_have_a_phase_and_that_is_the_contrast(R):
    """The control.  If everything came out 'circular', the metric would be measuring nothing.
    Bin 1 (the global positional ramp) is a STANDING WAVE: near-zero circularity, and its
    dimensions agree on a phase.  So the metric can tell the two apart."""
    by = {b["bin"]: b for b in R["bins"]}
    assert by[1]["circularity"] < 0.3, "the global ramp is a line, not a circle"
    assert by[1]["phase_concentration"] > 0.8, "and its dimensions DO share a phase"


def test_a_heads_codon_position_IS_a_rotation_angle_of_its_QK_form(R):
    """The closed form.  R(k) = DC + Pc.cos(2pi k/3) + Qs.sin(2pi k/3), where Pc and Qs are the
    symmetric and antisymmetric parts of the head's QK bilinear form restricted to the clock's
    2-D plane.  phase = atan2(-Qs, Pc), computed from W_Q and W_K ALONE -- nothing fitted to R(k).

    It reproduces every comb head's measured phase to under a degree, where a whole codon
    position is 120 degrees.  The head's codon position is not read off the table; it is the
    rotation angle its own weights induce on the clock plane.
    """
    assert R["worst_phase_err_deg"] < 5.0
    for h in R["heads"]:
        assert h["phase_err_deg"] < 5.0, f"{h['head']} phase mispredicted"
        # and the prediction lands in the right codon position, which is the interpretable claim
        pk = (-h["phase_pred"] / (2 * np.pi / 3)) % 3
        assert round(pk) % 3 == h["codon_position"]


def test_the_clock_PRECESSES_and_that_is_what_the_sidebands_are(R):
    """The amplitude comes out 2-3x too small if (c, s) is fitted over a long window -- because the
    clock's DIRECTION ROTATES along the sequence.  That precession IS the sideband structure at
    341/343.  Shrink the window and the amplitude recovers; the phase does not move, because it is
    an angle BETWEEN W_Q and W_K on the plane and is invariant under a global rotation of it.

    If a future version of this fits (c, s) over a long window and reports the AMPLITUDE, this
    test is the objection.
    """
    long_r = np.mean([h["ratio_long"] for h in R["heads"]])
    short_r = np.mean([h["ratio_short"] for h in R["heads"]])
    assert long_r < 0.6, "a long fit window badly UNDERSTATES the amplitude"
    assert 0.85 < short_r < 1.25, "a short one recovers it"
    assert short_r > 1.8 * long_r, "and that gap is the precession"
