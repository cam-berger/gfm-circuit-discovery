"""Tripwires for the CALIBRATED corpus-spectrum estimator (the rebuild after R16).

R16 retracted a null because the instrument was never validated.  This estimator carries its own
power calibration, and these tests pin the two things that calibration taught -- one of which is
a design error the calibration caught in the estimator ITSELF, before it could produce another
false null.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[1] / "results/phase-lattice/corpus_spectrum.json"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run_corpus_spectrum.py has not been run")


@pytest.fixture(scope="module")
def R():
    return json.loads(DATA.read_text())


def test_the_preregistered_bin_is_EXACT(R):
    """512 / 48 = 10.6667 exactly -- the same Fourier grid SpliceBERT's own first-stage 512-row
    position table used.  No leakage, no window smearing, and no freedom to go edge-seeking."""
    assert R["L"] == 512 and R["bin0"] == 48
    assert R["period0"] == pytest.approx(10.6667, abs=1e-3)


def test_the_estimator_HAS_a_false_positive_rate_and_it_is_clean(R):
    """Depth 0: the estimator must NOT find a tone that is not there."""
    for ch, v in R["calibration"]["0.0"].items():
        assert v["p"] > 0.05, f"{ch} false-positives at depth 0"


def test_the_detection_floor_is_MEASURED_not_assumed(R):
    """It sees 10%.  It is marginal at 5%.  It is blind below that.  Say so, always."""
    assert R["calibration"]["0.1"]["WW"]["p"] < 0.05, "must detect a 10% tone"
    assert R["calibration"]["0.02"]["WW"]["p"] > 0.05, "and must NOT claim to detect 2%"


def test_SEGMENT_LENGTH_MUST_MATCH_THE_SIGNALS_COHERENCE_LENGTH(R):
    """*** The design error the calibration caught, before it could produce a false null. ***

    The first build used L = 4096, where one Fourier bin is +-0.03 nt wide.  The preregistered
    filter at 10.6667 was therefore BLIND to an injected tone at 10.45 -- and the calibration
    could not see even a 10% signal.  The helical periodicity is a LOCAL feature (nucleosome
    scale, ~150-300 nt); over 4 kb its phase decoheres and its power smears across bins.

    L = 512 makes the bin wide enough to hold a real, slightly-drifting signal.  If anyone
    lengthens the segment to 'get better resolution', this test is the objection.
    """
    assert R["L"] <= 1024, "a longer segment makes the filter NARROWER than the signal is wide"
    bin_w = R["L"] / (R["bin0"] - 0.5) - R["L"] / (R["bin0"] + 0.5)
    assert bin_w > 0.15, "the preregistered bin must be wide enough to hold a 10.3-10.9 tone"


def test_THE_UNANCHORED_ASSAY_IS_A_QUADRATIC_DETECTOR_AND_THEREFORE_A_DEAD_END(R):
    """*** Why Assay 1 can never answer this, and Assay 2 can. ***

    Pooling POWER makes the statistic grow as depth^2.  Pooling COMPLEX coefficients across
    landmarks makes it grow as depth.  So:

        unanchored:  SNR ~ depth^2 * sqrt(N)  ->  10% to 2% needs 25^2 = 625x more sequence
                                                  = ~75 GB.  NOT VIABLE.
        anchored:    SNR ~ depth   * sqrt(N)  ->  10% to 2% needs      25x more landmarks
                                                  = ~10^6 dyads.  ROUTINE.

    Anchoring is not a refinement of the unanchored assay.  It is the difference between a
    quadratic and a linear detector, and at a few percent that is the difference between
    impossible and easy.  Do not try to rescue Assay 1 with more sequence.
    """
    zs = {float(d): R["calibration"][d]["WW"]["Z"] - 1 for d in R["calibration"]}
    # the response is quadratic: quadrupling the depth from 5% to 10% raises Z-1 by ~4x, not ~2x
    assert zs[0.10] > 3 * zs[0.05] > 0, "Z-1 grows as depth^2, not depth"
    assert zs[0.02] < 0.05 * zs[0.10], "and 2% is therefore ~25x weaker again"


def test_no_10_67_signal_was_found_but_the_null_is_NOT_POWERED(R):
    """The honest state.  120 Mb, four channels, with and without repeats: nothing at 10.6667.
    And it does not matter, because the floor (5-10%) is above the effect size in question
    (a few percent).  **This is NOT evidence against the chromatin reading.**"""
    for lab in R["real"]:
        for ch, v in R["real"][lab].items():
            assert v["p"] > 0.01, f"{lab}/{ch} would be a real detection -- update the claim!"
    assert R["calibration"]["0.02"]["WW"]["p"] > 0.05, (
        "the floor is above 2%, so the null above is UNPOWERED and must never be quoted as a "
        "refutation.  R16 is the memorial for doing that once already."
    )
