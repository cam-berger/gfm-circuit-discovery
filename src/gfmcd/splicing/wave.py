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


# ---------------------------------------------------------------------------------------------
# TWO TONES.  L0H11 reads BOTH carriers -- the codon clock at period 3 AND the second carrier at
# ~10.67 -- so a single cosine explains only 80.8% of its readout, against >=98.4% for the other
# five combs.  Its complete wave function is:
#
#     R(k) = envelope(k) + A3*cos(2*pi*k/3 + phi3) + A2*cos(2*pi*k/T2 + phi2)
#
# and that takes it to 99.9%.  See D22/D23 and tests/test_second_carrier_memorials.py.
#
# NOTE ON T2.  The final 1024nt model's carrier sits on bin 96 of its 1026-row table, i.e.
# 1026/96 = 10.6875.  The STABLE cross-checkpoint object is ~10.6667 (bin 48 of the 512-row
# stage-1 table; carrier-plane overlap 0.997).  The 0.02 nt difference is the Fourier grid, not
# the model.  Do not quote either to two decimals as though it were exact.
PERIOD_2 = 1026 / 96


@dataclass(frozen=True)
class TwoToneFit:
    """R(k) ~= env(k) + A3*cos(2pi k/3 + phi3) + A2*cos(2pi k/T2 + phi2)."""

    A3: float
    phi3: float
    A2: float
    phi2: float
    T2: float
    env_coef: tuple[float, ...]
    var_explained: float          # of the DE-ENVELOPED variance, by BOTH cosines together
    var_explained_1tone: float    # ... by the period-3 cosine ALONE (the gap is the second tone)
    max_resid: float


def decompose2(k, y, T2: float = PERIOD_2, envdeg: int = 3) -> TwoToneFit:
    """Joint OLS of a polynomial envelope and BOTH resolvable tones.

    Fitted jointly, not sequentially: cos(2pi k/3) and cos(2pi k/T2) are not exactly orthogonal
    over a finite window, so a two-step fit would bias both amplitudes.
    """
    k = np.asarray(k, float)
    y = np.asarray(y, float)
    w2 = 2 * np.pi / T2
    cols = ([k ** p for p in range(envdeg + 1)]
            + [np.cos(W * k), np.sin(W * k), np.cos(w2 * k), np.sin(w2 * k)])
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)

    env = np.polyval(b[: envdeg + 1][::-1], k)
    resid = y - X @ b
    deenv = y - env
    ve = float(1 - resid.var() / deenv.var()) if deenv.var() > 0 else 0.0

    one = decompose(k, y, envdeg=envdeg)     # the same signal with ONE tone, for the contrast

    return TwoToneFit(
        A3=float(np.hypot(b[envdeg + 1], b[envdeg + 2])),
        phi3=float(np.arctan2(-b[envdeg + 2], b[envdeg + 1])),
        A2=float(np.hypot(b[envdeg + 3], b[envdeg + 4])),
        phi2=float(np.arctan2(-b[envdeg + 4], b[envdeg + 3])),
        T2=float(T2),
        env_coef=tuple(float(c) for c in b[: envdeg + 1]),
        var_explained=ve, var_explained_1tone=one.var_explained,
        max_resid=float(np.abs(resid).max()),
    )


def curve2(fit: TwoToneFit, k) -> np.ndarray:
    """The complete wave function, on a CONTINUOUS grid."""
    k = np.asarray(k, float)
    return (np.polyval(np.asarray(fit.env_coef)[::-1], k)
            + fit.A3 * np.cos(W * k + fit.phi3)
            + fit.A2 * np.cos(2 * np.pi / fit.T2 * k + fit.phi2))
