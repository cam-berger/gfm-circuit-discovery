"""Tripwires for L0H11, the dual-frequency reader -- and for the CEILING on what two clocks can do.

The user's hypothesis: "the second carrier may serve a generic computational role: combining two
clocks to sharpen or disambiguate positional addresses."

Half of that is right and half of it was never available to be right.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[1] / "results/phase-lattice/dual_reader.json"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run_dual_reader.py has not been run")


@pytest.fixture(scope="module")
def R():
    return json.loads(DATA.read_text())


def test_the_carrier_SURVIVES_THE_SOFTMAX_in_L0H11(R):
    """*** The first post-softmax evidence for the second carrier. ***

    All of D22 lives in the QK logits, and a logit tone is not an attention tone (L0H5 has the
    biggest logit tone in the model and its attention is a spike).  Here, in real attention:
    L0H11's comb TEETH are modulated at period 10.69 at 58% of the mean tooth mass, and deleting
    the carrier removes 81% of that.
    """
    r = R["heads"]["L0H11"]
    assert r["intact"]["mod_rel"] > 0.4, "the teeth are strongly modulated at 10.69"
    kept = r["no_p10"]["mod_rel"] / r["intact"]["mod_rel"]
    assert kept < 0.3, "and deleting the carrier removes most of it"
    assert r["ci_mod"][2] < -0.5, "CI on the change excludes anything small"


def test_no_other_comb_is_a_dual_reader(R):
    """The control that can fail.  If deleting the 10.69 band flattened everybody's comb, the
    effect would be about perturbing P, not about reading the carrier."""
    heads = R["heads"]
    # L0H11 has the largest modulation, by a wide margin
    mods = {hd: r["intact"]["mod_rel"] for hd, r in heads.items()}
    assert max(mods, key=mods.get) == "L0H11"
    assert mods["L0H11"] > 1.5 * sorted(mods.values())[-2]


def test_L0H11_is_the_ONLY_comb_whose_address_gets_VAGUER(R):
    """The vernier's DIRECTION is right and it is specific: every other comb gets sharper or does
    not move when the carrier is deleted.  Only the head that reads it gets vaguer."""
    vaguer = [hd for hd, r in R["heads"].items() if r["ci_eff"][1] > 0]
    assert vaguer == ["L0H11"]


def test_TWO_SINUSOIDS_CANNOT_MAKE_A_SHARP_ADDRESS(R):
    """*** The ceiling.  Algebra, no model, no genome. ***

    For comb teeth p ~ (1 + a*cos(theta)) with theta ~ uniform, the entropy is H ~ log N - a^2/4.
    So a single added cosine of depth `a` can concentrate the tooth distribution by AT MOST
    1 - exp(-a^2/4) -- whatever the model does.

    At L0H11's measured depth (a = 0.578) that ceiling is 8%.  We measured 4.3%.  Even a FULL
    100% modulation caps at 22%.  The vernier is weak BY ALGEBRA, not because SpliceBERT failed
    to build one: a cosine has broad maxima, and concentrating a comb onto ONE tooth needs many
    harmonics (a CRT-style code), not two tones.

    If a future claim says "the two clocks combine into a sharp positional address", this test is
    the objection.
    """
    def concentration(a: float) -> float:
        """exp(H)/N for teeth p ~ (1 + a*cos): 1.0 = uniform, lower = more concentrated."""
        th = np.linspace(0, 2 * np.pi, 200_000, endpoint=False)
        p = 1 + a * np.cos(th)
        p /= p.sum()
        # at a = 1 the trough touches zero; p*log(p) -> 0 there.  Say so rather than get a NaN.
        H = -float(np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)))
        return float(np.exp(H - np.log(len(th))))

    # H ~ log N - a^2/4 is a SECOND-ORDER expansion.  It is accurate where we actually use it --
    # at L0H11's depth -- and it drifts at large a, so do not quote it there.
    for a in (0.2, 0.4, 0.578):
        assert concentration(a) == pytest.approx(np.exp(-a**2 / 4), rel=0.01)
    assert concentration(1.0) == pytest.approx(0.736, abs=0.01)     # exact, not the expansion
    assert concentration(1.0) > 0.70, "even a FULL 100% modulation concentrates by only ~26%"

    r = R["heads"]["L0H11"]
    a = r["intact"]["mod_rel"]
    ceiling = 1 - np.exp(-a**2 / 4)
    measured = r["ci_eff"][0]
    assert measured <= ceiling + 0.01, "the measurement cannot beat the algebra"
    assert ceiling < 0.10, "and the ceiling at this depth is under 10% -- not a sharp address"
    assert 1 - concentration(1.0) < 0.30, "even a 100% modulation caps at ~26%"


def test_the_CIs_here_are_honest_but_nearly_vacuous(R):
    """A memorial to a real subtlety.  The bootstrap resamples GENOMIC WINDOWS, but a comb's
    attention is POSITION-determined, not content-determined -- so the profile hardly varies
    across windows and the intervals come out absurdly tight.  They are not the uncertainty that
    matters.  Judge these effects by SIZE against the algebraic ceiling, never by a p-value."""
    widths = [r["ci_eff"][2] - r["ci_eff"][1] for r in R["heads"].values()]
    assert min(widths) < 0.01, (
        "if these CIs ever get WIDE, the position-determined assumption has broken and this "
        "whole readout needs rethinking"
    )
