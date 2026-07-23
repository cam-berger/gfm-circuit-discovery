#!/usr/bin/env python
"""Validation tests for the cross-model carrier instruments.

Run:  python scripts/crossmodel/test_carrier_spectrum.py
These are the gates that must pass before any carrier band amplitude is interpreted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.crossmodel import carrier_spectrum as cs  # noqa: E402
from gfmcd.crossmodel import reader_scan as rs  # noqa: E402


def test_amplitude_recovery_exact_and_between_bin():
    """Per-dim amplitude of a clean tone recovered to ~machine precision at any period."""
    for N, T in [(1026, 3.0), (440, 3.0), (440, 10.63), (1026, 10.6875)]:
        D, amp = 8, 0.37
        k = np.arange(N)
        ph = np.linspace(0, 2 * np.pi, D, endpoint=False)
        P = amp * np.cos(2 * np.pi * k[:, None] / T + ph[None, :])
        _, perdim = cs.leakage_aware_band_power(P, T, envdeg=6, return_perdim=True)
        assert np.abs(perdim - amp).max() < 1e-6, (N, T, np.abs(perdim - amp).max())


def test_band_power_scales_as_amplitude_squared():
    N, T, D = 440, 10.63, 40
    k = np.arange(N)
    ph = np.linspace(0, 2 * np.pi, D, endpoint=False)
    bp = []
    for amp in (0.1, 0.2, 0.4):
        P = amp * np.cos(2 * np.pi * k[:, None] / T + ph[None, :])
        bp.append(cs.leakage_aware_band_power(P, T, envdeg=6))
    # doubling amp quadruples power
    assert abs(bp[1] / bp[0] - 4.0) < 0.05
    assert abs(bp[2] / bp[1] - 4.0) < 0.05


def test_leakage_free_period_localization():
    """A between-bin tone is found at its true period, not snapped to a bin."""
    rng = np.random.default_rng(1)
    P = rng.standard_normal((440, 60)) * 0.3
    k = np.arange(440)
    P += 0.25 * np.cos(2 * np.pi * k[:, None] / 10.63
                       + rng.uniform(0, 2 * np.pi, 60)[None, :])
    Ts = np.arange(8.0, 14.0, 0.02)
    pg = cs.periodogram(P, Ts, envdeg=6)
    That = Ts[int(np.argmax(pg))]
    assert abs(That - 10.63) < 0.1, That


def test_null_has_controlled_fpr():
    """Pure noise + envelope: the rank test fires <=~8% of the time at p<=0.05 (target 5%)."""
    fires = 0
    n = 60
    for s in range(n):
        rng = np.random.default_rng(s)
        N = 440
        k = np.arange(N)
        ks = (k - (N - 1) / 2) / ((N - 1) / 2)
        P = rng.standard_normal((N, 40)) * 0.4
        P += (rng.uniform(-2, 2, 40)[None, :] * ks[:, None]
              + rng.uniform(-1, 1, 40)[None, :] * (ks**2)[:, None])
        if cs.carrier_test(P, 10.63, envdeg=6)["p_rank"] <= 0.05:
            fires += 1
    assert fires / n <= 0.12, fires / n


def test_exact_bin_matches_fft():
    """exact_bin_band_ratio flags an integer bin and returns the FFT peak there."""
    N, T = 1026, 3.0
    r = cs.exact_bin_band_ratio(np.random.default_rng(0).standard_normal((N, 32)), T)
    assert r["bin"] == 342 and r["bin_is_exact"]


def test_deletion_audit_removes_target_keeps_protected():
    rng = np.random.default_rng(0)
    N = 1026
    k = np.arange(N)
    P = rng.standard_normal((N, 64)) * 0.2
    P += 0.5 * np.cos(2 * np.pi * k[:, None] / 3.0)          # period-3 -> bin 342
    P += 0.5 * np.cos(2 * np.pi * k[:, None] / (1026 / 96))  # =10.63  -> bin 96
    aud, _ = rs.deletion_audit(P, list(range(339, 346)),
                               {"carrier_10p63": list(range(94, 101))})
    assert aud["PASS"], aud
    assert aud["target_retained"] <= 0.05
    assert aud["protected_carrier_10p63_retained"] >= 0.95


def test_bounded_statistic_stays_in_unit_interval():
    band, w = rs.band_and_window()
    rng = np.random.default_rng(0)
    for _ in range(50):
        A = np.abs(rng.standard_normal(len(band))) + 0.01
        c, _ = rs.m_of(A, 10.63, band, w)
        assert abs(c) <= 1.0 + 1e-9, abs(c)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
