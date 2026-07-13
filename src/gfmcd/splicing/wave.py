"""The period-3 readout R(k), decomposed into a slow envelope and ONE cosine.

Why one cosine is not a modelling choice but a theorem
------------------------------------------------------
R(k) is sampled at integer offsets k, and the clock has period 3 -- three samples per cycle.
Nyquist for an integer-sampled signal is 0.5 cycles/sample; the clock sits at 1/3.  Its second
harmonic would sit at 2/3 cycles/sample, which is ABOVE Nyquist and aliases straight back onto
-1/3, i.e. onto the fundamental.  A period-3 integer sequence therefore has **exactly one
resolvable sinusoid** (plus DC): three real samples per period carry exactly three real degrees
of freedom, and DC + (amplitude, phase) uses all three.

So the smooth curve drawn through a period-3 comb is not a smoothing, a spline, or a fit with
residuals.  It is the *exact* trigonometric interpolant.  The jaggedness of the usual plot is an
artefact of joining the samples with straight lines -- nothing more.

What IS fitted
--------------
Only the envelope: R(k) also carries a slow drift (attention decays with |k|), which is not the
clock.  We fit `env` (a low-order polynomial) and the clock JOINTLY by OLS, because cos(2pi k/3)
is not exactly orthogonal to a linear trend over a finite window and a two-step fit would bias A.

Two independent estimators are provided and must agree:
  * `decompose`         -- joint OLS, assumes a polynomial envelope
  * `boxcar_amplitude`  -- MODEL-FREE.  A 3-tap moving average has an exact zero at period 3
                           (1 + 2cos(2pi/3) = 0), so it annihilates the clock identically and
                           leaves the envelope.  The high-pass residual is then the clock, at
                           gain 1, with no envelope model assumed anywhere.
If they disagree, the envelope model is stealing amplitude and the fit is not to be trusted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PERIOD = 3
W = 2 * np.pi / PERIOD


@dataclass(frozen=True)
class WaveFit:
    """R(k) ~= env(k) + A*cos(W*k + phi)."""

    A: float                     # clock amplitude, in the units of R (attention logits)
    phi: float                   # radians
    env_coef: tuple[float, ...]  # polynomial coefficients, ascending powers of k
    var_explained: float         # fraction of the DE-ENVELOPED variance taken by the one cosine
    max_resid: float
    peak_offset: float           # offset in [0,3) where the wave peaks == the head's codon phase

    @property
    def codon_position(self) -> int:
        return int(round(self.peak_offset)) % PERIOD

    @property
    def lock_error(self) -> float:
        """Distance of the peak from the nearest integer offset, in nucleotides.

        NOTE: near-zero lock error is PARTLY SELECTED FOR -- see the module note in
        tests/test_wave_memorials.py.  It is a descriptor of a head, never evidence.
        """
        return float(min(abs(self.peak_offset - r) for r in (0, 1, 2, 3)))


def _design(k: np.ndarray, envdeg: int) -> np.ndarray:
    return np.column_stack(
        [k ** p for p in range(envdeg + 1)] + [np.cos(W * k), np.sin(W * k)]
    )


def decompose(k: np.ndarray, y: np.ndarray, envdeg: int = 3) -> WaveFit:
    """Joint OLS of a polynomial envelope and the single resolvable period-3 cosine."""
    k = np.asarray(k, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(_design(k, envdeg), y, rcond=None)
    a, b = float(beta[-2]), float(beta[-1])
    A = float(np.hypot(a, b))
    phi = float(np.arctan2(-b, a))

    env = np.polyval(beta[: envdeg + 1][::-1], k)
    resid = y - env - A * np.cos(W * k + phi)
    deenv = y - env
    ve = float(1 - resid.var() / deenv.var()) if deenv.var() > 0 else 0.0

    return WaveFit(
        A=A, phi=phi, env_coef=tuple(float(c) for c in beta[: envdeg + 1]),
        var_explained=ve, max_resid=float(np.abs(resid).max()),
        peak_offset=float((-phi / W) % PERIOD),
    )


def clock(fit: WaveFit, k: np.ndarray) -> np.ndarray:
    """The clock alone, evaluable on a CONTINUOUS grid -- this is the wave."""
    return fit.A * np.cos(W * np.asarray(k, float) + fit.phi)


def envelope(fit: WaveFit, k: np.ndarray) -> np.ndarray:
    return np.polyval(np.asarray(fit.env_coef)[::-1], np.asarray(k, float))


def curve(fit: WaveFit, k: np.ndarray) -> np.ndarray:
    return envelope(fit, k) + clock(fit, k)


def boxcar_amplitude(k: np.ndarray, y: np.ndarray) -> float:
    """Model-free clock amplitude.  Assumes NOTHING about the envelope.

    A 3-tap boxcar annihilates period 3 exactly, so `y - boxcar(y)` is the clock (gain 1) plus a
    slow remainder; project that onto {cos, sin, 1, k} by least squares.
    """
    k = np.asarray(k, float)
    y = np.asarray(y, float)
    hp = (y - np.convolve(y, np.ones(3) / 3, mode="same"))[1:-1]   # drop the padded edges
    kk = k[1:-1]
    X = np.column_stack([np.cos(W * kk), np.sin(W * kk), np.ones_like(kk), kk])
    beta, *_ = np.linalg.lstsq(X, hp, rcond=None)
    return float(np.hypot(beta[0], beta[1]))
