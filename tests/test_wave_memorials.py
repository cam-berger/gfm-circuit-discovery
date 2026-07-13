"""Tripwires for the WAVE rendering of the period-3 readout.

Drawing a smooth cosine through a jagged comb is either (a) an exact statement of what the data
already are, or (b) a beautification that manufactures structure.  Which one it is, is decidable.
These tests decide it, with no genome and no model.

The load-bearing claims the figure makes, and the test that pins each:
  1. one cosine is not a choice -- it is all a period-3 integer signal CAN carry   -> (1), (2)
  2. the envelope removal is model-free, not a fitted whitewash                    -> (3), (4)
  3. the de-enveloping DOES NOT MANUFACTURE A WAVE from a smooth signal            -> (5)  <-- THE CONTROL
  4. the integer phase-lock is PARTLY SELECTED FOR and is not a discovery          -> (6)  <-- THE TRAP
"""
from __future__ import annotations

import numpy as np
import pytest

from gfmcd.splicing.wave import W, boxcar_amplitude, clock, curve, decompose

K = np.arange(-12, 13, dtype=float)          # the offsets the figure actually plots


# ----------------------------------------------------------------- 1. the Nyquist theorem
def test_a_period3_integer_signal_has_exactly_one_resolvable_harmonic():
    """The 2nd harmonic of period 3 is ABOVE Nyquist and aliases onto the fundamental.

    This is why the smooth curve is an interpolant and not a fit: at 3 samples/cycle there is no
    second harmonic to be missing.  cos(2*W*k) IS cos(W*k) for every integer k.
    """
    k = np.arange(-50, 51, dtype=float)
    assert np.allclose(np.cos(2 * W * k), np.cos(W * k), atol=1e-12)
    assert np.allclose(np.sin(2 * W * k), -np.sin(W * k), atol=1e-12)

    # ... and therefore 3 real samples per period carry exactly 3 real dof: DC + (amp, phase).
    y = np.array([2.46, 2.40, 5.15])                       # L2H15's actual triple
    dc = y.mean()
    z = np.exp(-1j * W * np.arange(3)) @ (y - dc) / 3      # the single non-DC Fourier coefficient
    recon = dc + 2 * np.abs(z) * np.cos(W * np.arange(3) + np.angle(z))
    assert np.allclose(recon, y, atol=1e-12), "a length-3 real signal is EXACTLY DC + one cosine"


def test_the_curve_through_a_period3_comb_is_exact_not_a_smoothing():
    """Zero residual.  Not 'small'.  Zero."""
    y = np.tile([2.46, 2.40, 5.15], 9)[: len(K)]           # an exactly period-3 comb
    fit = decompose(K, y, envdeg=0)                        # no envelope: pure DC + one cosine
    assert fit.max_resid < 1e-10, f"residual {fit.max_resid} -- the interpolant should be exact"
    assert np.allclose(curve(fit, K), y, atol=1e-10)


# ----------------------------------------------------------------- 2. the envelope is model-free
def test_a_3tap_boxcar_annihilates_period_3_exactly():
    """1 + 2*cos(2pi/3) == 0.  This is the whole reason the envelope can be removed without a model."""
    assert abs(1 + 2 * np.cos(W)) < 1e-12
    pure = 1.7 * np.cos(W * np.arange(-40, 41) + 0.9)
    assert np.abs(np.convolve(pure, np.ones(3) / 3, mode="valid")).max() < 1e-12


@pytest.mark.parametrize("envdeg", [1, 2, 3, 4, 6])
def test_the_envelope_model_does_not_steal_the_amplitude(envdeg):
    """OLS (assumes a polynomial envelope) and the boxcar (assumes NOTHING) must agree.

    If they diverge, the envelope is soaking up clock amplitude and the fitted A is fiction.
    """
    env = 5.0 + 0.03 * K - 0.004 * K**2 + 1e-4 * K**3
    y = env + 1.7 * np.cos(W * K + 0.9)
    ols = decompose(K, y, envdeg=envdeg)
    assert ols.A == pytest.approx(1.7, rel=0.02)
    assert boxcar_amplitude(K, y) == pytest.approx(ols.A, rel=0.03)


def test_the_fit_recovers_a_planted_amplitude_and_phase():
    for A_true, phi_true in [(2.1, 0.0), (0.8, 2.09), (1.5, -2.09), (3.0, 1.0)]:
        y = 4.0 - 0.01 * K**2 + A_true * np.cos(W * K + phi_true)
        fit = decompose(K, y)
        assert fit.A == pytest.approx(A_true, rel=1e-3)
        assert np.cos(fit.phi - phi_true) == pytest.approx(1.0, abs=1e-4)


# ----------------------------------------------------------------- 3. THE CONTROL
#: worst-case period-3 amplitude the estimators hallucinate from a signal that HAS NO CLOCK.
#: Measured below.  A cubic envelope cannot absorb an arbitrary smooth signal, and what it fails
#: to absorb leaks into the period-3 bin -- so the floor is not zero, and pretending it is would
#: be the fudge.  The figure's claim is not "the leak is zero", it is "the signal clears the leak
#: by more than an order of magnitude".
LEAK_FLOOR_OLS = 0.08
LEAK_FLOOR_BOX = 0.06


def test_de_enveloping_does_NOT_manufacture_a_wave_from_a_smooth_signal():
    """*** The control that licenses the de-enveloped panel. ***

    The figure plots (R(k) - envelope) and draws a cosine through it.  If that pipeline turned
    ANY signal into a wave, the panel would be a lie.  So: feed it signals with NO clock, measure
    what it hallucinates, and require the real signal to clear that floor by a wide margin.

    The last two confusers are ADVERSARIAL -- slow waves at period 6 and period 4, deliberately
    placed as close to period 3 as anything can be.  P's actual spectrum has nothing there (the
    next-strongest mode is period 10.7, panel A), so they are a worst case, not a realistic one.
    """
    clock_free = [
        4.0 - 0.02 * K**2,                                  # a parabola
        3.5 + 0.05 * K,                                     # a ramp
        3.0 + 0.6 * np.tanh(K / 5),                         # a sigmoid
        3.0 + 0.8 * np.cos(2 * np.pi * K / 17),             # a slow wave, wrong period
        3.0 + 0.8 * np.cos(2 * np.pi * K / 6),              # ADVERSARIAL: period 6
        3.0 + 0.8 * np.cos(2 * np.pi * K / 4),              # ADVERSARIAL: period 4
    ]
    for y in clock_free:
        assert decompose(K, y).A < LEAK_FLOOR_OLS, "the leak floor moved -- re-derive the margin"
        assert boxcar_amplitude(K, y) < LEAK_FLOOR_BOX

    # the real thing: SpliceBERT's L2H15, intact vs the period-3 band deleted from P.
    intact = np.array([2.46, 2.40, 5.15, 2.44, 2.40, 5.13, 2.40, 2.39, 5.11, 2.37, 2.38, 5.09,
                       2.34, 2.38, 5.08, 2.31, 2.39, 5.07, 2.28, 2.39, 5.06, 2.26, 2.39, 5.04,
                       2.23])
    killed = np.array([3.69, 3.69, 3.70, 3.68, 3.68, 3.68, 3.67, 3.66, 3.67, 3.65, 3.64, 3.65,
                       3.63, 3.63, 3.63, 3.61, 3.61, 3.62, 3.60, 3.60, 3.60, 3.58, 3.58, 3.58,
                       3.56])

    # 1. the intact head clears the hallucination floor by more than an order of magnitude
    assert decompose(K, intact).A > 10 * LEAK_FLOOR_OLS
    assert boxcar_amplitude(K, intact) > 10 * LEAK_FLOOR_BOX

    # 2. and once the clock is deleted, what is left is INDISTINGUISHABLE FROM NO CLOCK AT ALL --
    #    it falls back to the estimator's own noise floor.  That is the whole result, in one line.
    assert decompose(K, killed).A < LEAK_FLOOR_OLS
    assert boxcar_amplitude(K, killed) < LEAK_FLOOR_BOX
    assert decompose(K, killed).A / decompose(K, intact).A < 0.02


# ----------------------------------------------------------------- 4. THE TRAP
def test_the_integer_phase_lock_is_PARTLY_SELECTED_FOR_and_is_not_a_discovery():
    """*** Do not let anyone claim 'the heads' phases lock to integer offsets'. ***

    The six heads were chosen by a classifier that demands every attention tooth be congruent
    mod 3 (run_comb_shift.py:130).  A head whose wave peaked at a HALF-integer offset would split
    its mass across two ADJACENT offsets -- which lie in DIFFERENT residue classes -- and so could
    never have been called a comb.  Integer-locking is thus a property of the SURVIVORS, partly by
    construction.  It is a descriptor of a head.  It is not evidence of anything.  (cf. R14.)
    """
    offs = np.arange(-24, 25)

    def teeth_of(peak: float) -> list[int]:
        logits = 3.0 * np.cos(W * (offs - peak)) - 0.02 * offs**2   # a comb centred on `peak`
        a = np.exp(logits - logits.max())
        a /= a.sum()
        return [int(o) for o in offs[a >= 0.03]]                    # the classifier's 3% rule

    def congruent(teeth: list[int]) -> bool:
        return bool(teeth) and all(o % 3 == teeth[0] % 3 for o in teeth)

    assert congruent(teeth_of(0.0)), "an integer-phase head passes the comb filter"
    assert congruent(teeth_of(2.0))
    assert not congruent(teeth_of(1.5)), (
        "a HALF-INTEGER-phase head is REJECTED by the comb filter -- so the surviving heads are "
        "integer-locked partly BY CONSTRUCTION, and the lock must never be reported as a finding"
    )
    assert not congruent(teeth_of(0.5))


def test_the_222_codon_split_is_described_never_claimed():
    """The six heads happen to split 2/2/2 over the three codon positions.  Under uniform random
    assignment that has p = 90/729 = 0.123.  With six heads it is unremarkable.  If a future
    version of the figure asserts significance here, this test is the objection."""
    from math import comb as C
    p = C(6, 2) * C(4, 2) * C(2, 2) / 3**6
    assert p == pytest.approx(0.1235, abs=1e-3)
    assert p > 0.05, "a balanced 2-2-2 split over 6 heads is NOT significant -- describe, do not claim"


def test_clock_is_evaluable_on_a_continuous_grid():
    """The wave is a real function of a real argument -- that is what makes the curve drawable."""
    fit = decompose(K, 4.0 + 1.8 * np.cos(W * K + 2.1))
    fine = np.linspace(-12, 12, 24 * 100 + 1)      # step 0.01 -> the integers land exactly on it
    c = clock(fit, fine)
    assert c.shape == fine.shape
    assert np.allclose(c[::100], clock(fit, K), atol=1e-9)
