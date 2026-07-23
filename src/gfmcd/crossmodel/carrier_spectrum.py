"""Leakage-aware P-spectrum estimator, ported from the SpliceBERT carrier assays.

WHY A LEAKAGE-AWARE ESTIMATOR
-----------------------------
A carrier is a narrow-band concentration of power in the position table P at a
nucleotide-wavelength period.  When P's row count N is divisible by the period, that
period lands on an EXACT rfft bin (no leakage) and a plain FFT bin ratio is the right
statistic -- this is how SpliceBERT's final 1026-row table was read (period 3 = bin 342).

When N is NOT divisible by the period (RNA-BERT: 440 rows, 440/3 = 146.67), the period
falls BETWEEN bins.  A plain FFT then smears the tone across neighbouring bins (spectral
leakage) and understates it, while a naive "nearest bin" overstates the neighbour.  The
fix, taken verbatim in spirit from ``run_second_carrier.amp2``: fit, at a CONTINUOUS
frequency, a joint least-squares model

    P[:, d]  =  envelope(k)  +  A_d cos(w k)  +  B_d sin(w k)

where ``envelope`` is a low-degree polynomial in position k that absorbs the slow global
drift, and ``w = 2*pi/T`` is imposed.  The per-dimension amplitude is ``hypot(A_d, B_d)``;
the band power is ``sum_d (A_d^2 + B_d^2)``.  Because the frequency is continuous this
reads a between-bin tone without leakage, and because the envelope is fitted jointly the
statistic cannot re-find the drift as a fast oscillation.

NULL / NOISE FLOOR
------------------
The band power at the target period means nothing without the distribution of the SAME
statistic where no carrier is expected.  Two nulls are provided:
  * ``carrier_test`` builds an empirical null from band powers at CONTROL periods (a dense
    grid over a fast-period range, excluding the target and its neighbourhood).  Z = target
    band power / median(control); p = rank of the target among controls.
  * ``inject_calibration`` (in the runner) injects a KNOWN rank-structured tone at the
    target period on top of a phase-scrambled copy of P and reports the detection floor and
    the depth-0 false-positive rate -- a null is worthless until the floor is known.

Every statistic here is over the position table P only.  A concentration in P is a P-level
claim; it is NOT attention and NOT causal.  Those live in separate assays.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def get_position_table(model: Any) -> np.ndarray:
    """Return the learned absolute position-embedding weight (N, D) as float64.

    Resolves the module whose name ends in ``position_embeddings`` -- works for the
    HF BERT layout (``bert.embeddings.position_embeddings``) and the MultiMolecule
    layout (``<base>.embeddings.position_embeddings``) without hard-coding the base.
    """
    hits = [
        (name, mod)
        for name, mod in model.named_modules()
        if name.endswith("position_embeddings") and hasattr(mod, "weight")
    ]
    if not hits:
        raise RuntimeError("model exposes no *.position_embeddings module")
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous position_embeddings modules: {[h[0] for h in hits]}")
    return hits[0][1].weight.detach().float().cpu().numpy().astype(np.float64)


def _design(N: int, T: float, envdeg: int) -> np.ndarray:
    """Envelope polynomial (degree envdeg) + a single cos/sin pair at period T.

    Positions are centred and scaled to [-1, 1] so the polynomial columns are well
    conditioned regardless of N.
    """
    k = np.arange(N, dtype=np.float64)
    ks = (k - (N - 1) / 2.0) / ((N - 1) / 2.0)  # in [-1, 1]
    w = 2 * np.pi / T
    cols = [ks**p for p in range(envdeg + 1)]
    cols += [np.cos(w * k), np.sin(w * k)]
    return np.column_stack(cols)


def leakage_aware_band_power(
    P: np.ndarray, T: float, *, envdeg: int = 6, return_perdim: bool = False
):
    """Band power of a period-T tone in P, envelope-removed, leakage-free.

    Fits ``P[:, d] = envelope(k) + A_d cos(wk) + B_d sin(wk)`` jointly for every dim d in
    one least-squares solve, then sums ``A_d^2 + B_d^2`` over dims.  Works at any real T
    (between-bin included).  Returns the scalar band power, or ``(band_power, per_dim_amp)``.
    """
    P = np.asarray(P, dtype=np.float64)
    N, D = P.shape
    X = _design(N, T, envdeg)
    coef, *_ = np.linalg.lstsq(X, P, rcond=None)  # (n_basis, D)
    a = coef[envdeg + 1]   # cos coefficients, per dim
    b = coef[envdeg + 2]   # sin coefficients, per dim
    amp2 = a**2 + b**2
    band_power = float(amp2.sum())
    if return_perdim:
        return band_power, np.sqrt(amp2)
    return band_power


def periodogram(P: np.ndarray, periods: np.ndarray, *, envdeg: int = 6) -> np.ndarray:
    """Leakage-aware band power at each period in ``periods`` (envelope fitted jointly)."""
    return np.array([leakage_aware_band_power(P, float(T), envdeg=envdeg) for T in periods])


def carrier_test(
    P: np.ndarray,
    T0: float,
    *,
    control_lo: float = 4.0,
    control_hi: float = 40.0,
    exclude_frac: float = 0.06,
    n_control: int = 400,
    envdeg: int = 6,
) -> dict:
    """Test for a period-T0 carrier against an empirical null of control periods.

    The null is the band power at ``n_control`` periods spaced uniformly in FREQUENCY over
    [control_lo, control_hi] nt, excluding a +-``exclude_frac`` neighbourhood of T0 (and of
    period 3 and period 10.63 if they fall in range, so a real carrier does not contaminate
    the null).  Returns Z = P(T0)/median(null), a rank p-value, and the null summary.
    """
    N = P.shape[0]
    f0, f1 = 1.0 / control_hi, 1.0 / control_lo
    freqs = np.linspace(f0, f1, n_control)
    ctrl_T = 1.0 / freqs
    # exclude neighbourhoods of the target and of the two named carriers
    protect = [T0, 3.0, 1026 / 96]  # 1026/96 = 10.6875 ~ the =10.63 carrier period
    keep = np.ones(len(ctrl_T), bool)
    for tp in protect:
        keep &= np.abs(ctrl_T - tp) > exclude_frac * tp
    ctrl_T = ctrl_T[keep]
    null = periodogram(P, ctrl_T, envdeg=envdeg)
    b0 = leakage_aware_band_power(P, T0, envdeg=envdeg)
    med = float(np.median(null))
    Z = float(b0 / med) if med > 0 else float("inf")
    p = float((np.sum(null >= b0) + 1) / (len(null) + 1))
    return {
        "T0": float(T0),
        "band_power": float(b0),
        "Z_over_median_control": Z,
        "p_rank": p,
        "n_control": int(len(ctrl_T)),
        "control_median": med,
        "control_p95": float(np.percentile(null, 95)),
        "control_max": float(null.max()),
        "control_lo": control_lo,
        "control_hi": control_hi,
        "envdeg": envdeg,
    }


def exact_bin_band_ratio(P: np.ndarray, period: float, halfwidth: int = 3) -> dict:
    """Classic FFT band-vs-median ratio, for periods that land on (near) an exact bin.

    Reproduces the SpliceBERT reading convention (period 3 = bins 339-345 of the 1026-row
    table, reported as x-median).  ``bin = round(N/period)``; band = bin +- halfwidth.
    Only meaningful when N/period is close to an integer; the returned ``bin_is_exact``
    flag says whether it is.
    """
    P = np.asarray(P, dtype=np.float64)
    N, D = P.shape
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    power = (np.abs(F) ** 2).sum(1)          # per-bin power summed over dims
    bin_f = N / period
    b = int(round(bin_f))
    lo, hi = max(1, b - halfwidth), min(len(power) - 1, b + halfwidth)
    band = list(range(lo, hi + 1))
    band_power = float(power[band].sum())
    med = float(np.median(power[1:]))         # exclude DC
    return {
        "period": float(period),
        "bin_float": float(bin_f),
        "bin": b,
        "bin_is_exact": bool(abs(bin_f - round(bin_f)) < 1e-6),
        "band_bins": band,
        "band_power": band_power,
        "peak_bin_power": float(power[b]),
        "median_bin_power": med,
        "band_over_median": float(band_power / med) if med > 0 else float("inf"),
        "peak_over_median": float(power[b] / med) if med > 0 else float("inf"),
    }
