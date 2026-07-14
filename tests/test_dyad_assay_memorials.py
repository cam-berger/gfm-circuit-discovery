"""Tripwires for the DYAD-ANCHORED assay (Assay 2) -- built and power-calibrated BEFORE any real
nucleosome map is acquired, because acquiring data for an underpowered assay is R16 one step
earlier.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[1] / "results/phase-lattice/dyad_assay.json"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run_dyad_assay.py has not been run")


@pytest.fixture(scope="module")
def R():
    return json.loads(DATA.read_text())


def test_ASSAY_1_IS_CLOSED_and_its_p_values_are_NOT_EVIDENCE():
    """*** The wording rule.  Never quote Assay 1's p >= 0.39 as evidence against chromatin. ***

    The unanchored power assay is a QUADRATIC detector (SNR ~ a^2 sqrt(N)).  At 120 Mb it can
    exclude only large, globally incoherent effects.  Its ONLY reportable result is its
    calibrated sensitivity limit: it sees 10%, is marginal at 5%, and is blind below that --
    while the effect in question is a few percent.
    """
    cs = Path(__file__).resolve().parents[1] / "results/phase-lattice/corpus_spectrum.json"
    if not cs.exists():
        pytest.skip("corpus_spectrum.json absent")
    C = json.loads(cs.read_text())
    assert C["calibration"]["0.02"]["WW"]["p"] > 0.05, (
        "Assay 1 cannot see 2%.  Its null is a SENSITIVITY LIMIT, never a refutation."
    )


def test_the_anchored_detector_is_LINEAR_which_is_the_whole_point(R):
    """Signal adds linearly across dyads, noise as sqrt(N).  So the signed projection z should
    grow ~linearly in depth -- not quadratically as Assay 1's power statistic did."""
    z = {float(d): R["splithalf"][d]["z"] for d in R["splithalf"]}
    # 1% -> 2% -> 5% should scale roughly 1 : 2 : 5, not 1 : 4 : 25
    r = z[0.05] / z[0.01]
    assert 3.0 < r < 8.0, f"z(5%)/z(1%) = {r:.1f}; a LINEAR detector gives ~5, a quadratic ~25"


def test_the_power_criteria_are_MET_at_2_percent(R):
    """Preregistered acceptance: FPR ~5%, >=80% detection at 2% modulation."""
    assert R["power"]["0.0_0.0"]["detect"] <= 0.15, "FPR must be near 5% (24 reps -> wide CI)"
    assert R["power"]["0.02_0.0"]["detect"] >= 0.80, "must reach 80% power at 2%"
    assert R["power"]["0.02_1.0"]["detect"] >= 0.80, "and still at 1 bp dyad jitter"


def test_the_frozen_phase_projection_has_NO_MAGNITUDE_BIAS(R):
    """|C| is positively biased -- estimating the phase and testing it on the same data turns
    that bias into another precise-looking result.  The split-half signed projection does not:
    under the null it is symmetric about zero."""
    assert abs(R["splithalf"]["0.0"]["z"]) < 2.0, "the null projection must sit at zero"
    assert R["splithalf"]["0.02"]["z"] > 5.0, "and 2% must be unmistakable on the HELD-OUT half"


def test_DYAD_PRECISION_IS_THE_BINDING_CONSTRAINT_NOT_DYAD_COUNT(R):
    """*** Why the chemical-cleavage map, and not MNase-seq. ***

    Gaussian dyad jitter sigma attenuates the amplitude by exp[-0.5*(2*pi*sigma/10.67)^2].
    MEASURED: 1.00 / 0.85 / 0.48 at sigma = 0 / 1 / 2 bp, against a prediction of
    1.00 / 0.84 / 0.50.  Confirmed to within 2%.

    Attenuation enters SNR LINEARLY but the dyad count only as sqrt(N) -- so 2 bp of jitter
    costs the same as throwing away FOUR FIFTHS of the dyads.  A nominal million badly-positioned
    dyads is an underpowered million.  Do not substitute MNase-seq for the chemical map to get a
    bigger N; its ~30x A/T cleavage bias is separately disqualifying when the OUTCOME is itself a
    weak nucleotide periodicity.
    """
    for s in ("0.0", "1.0", "2.0"):
        pred = np.exp(-0.5 * (2 * np.pi * float(s) / 10.6667) ** 2)
        assert R["jitter"][s]["measured"] == pytest.approx(pred, abs=0.05)
    # 2 bp jitter halves the amplitude -> costs 4x in effective N
    att = R["jitter"]["2.0"]["measured"]
    assert 1 / att**2 > 3.5, "2 bp of jitter costs ~4x the dyad count"


def test_a_million_dyads_is_far_MORE_than_enough(R):
    """~3,000 perfect dyads reach 80% power at 2%.  The Voong map has ~10^6.  So the assay has
    a large margin even after the realistic degradations that this simulation does NOT model:
    only a MINORITY of nucleosomal sequences carry the periodicity, and the phase may vary
    across dyads.  Both attenuate the coherent sum LINEARLY (SNR ~ f * sqrt(N))."""
    assert R["need_dyads_2pct"] < 50_000, "the requirement must be far below 10^6"
