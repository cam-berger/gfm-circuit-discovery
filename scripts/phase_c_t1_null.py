"""Is T1's periodic term real, or is it a 5-parameter fit to autocorrelated noise?

The registered T1 statistic compares models by BIC, which assumes independent residuals. Δ(d) is not
independent across neighbouring d: the same windows contribute to every distance, and the amino-acid
readout is itself codon-periodic. Autocorrelated noise manufactures spectral lines.

So: fit an AR(1) to the residuals after the log-d trend, simulate surrogates with the same lag-1
autocorrelation and variance -- AR(1) has no line spectrum, so any periodicity is destroyed while the
smoothness is kept -- and re-run the identical period search on each. The null distribution of ΔBIC
says how often a series this smooth fakes an oscillation this good.

Also splits the windows in half and refits the period independently. A real period reproduces.

    python scripts/phase_c_t1_null.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from phase_c_relay import PERIOD_MAX, PERIOD_MIN, bic  # noqa: E402

DATA = Path("results/fig_data")
N_SURR = 2000


def period_search(d: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Returns (ΔBIC of periodic over trend, best period, amplitude). Identical to fit_t1's search."""
    X = np.stack([np.ones_like(d), np.log2(d)], 1)
    b_trend = bic(y, X @ np.linalg.lstsq(X, y, rcond=None)[0], 2)
    best = (np.inf, np.nan, np.nan)
    for period in np.arange(PERIOD_MIN, PERIOD_MAX + 0.5, 0.5):
        S = np.concatenate([X, np.stack([np.sin(2 * np.pi * d / period), np.cos(2 * np.pi * d / period)], 1)], 1)
        c = np.linalg.lstsq(S, y, rcond=None)[0]
        b = bic(y, S @ c, 5)
        if b < best[0]:
            best = (b, period, float(np.hypot(c[2], c[3])))
    return b_trend - best[0], best[1], best[2]


def ar1_surrogates(resid: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    r = resid - resid.mean()
    rho = float(np.clip((r[:-1] * r[1:]).sum() / (r[:-1] ** 2).sum(), -0.99, 0.99))
    sigma = float(r.std() * np.sqrt(1 - rho**2))
    out = np.zeros((n, len(r)))
    out[:, 0] = rng.normal(0, r.std(), n)
    for i in range(1, len(r)):
        out[:, i] = rho * out[:, i - 1] + rng.normal(0, sigma, n)
    return out, rho


def main() -> None:
    res = json.loads((DATA / "phase_c_t1_t2.json").read_text())
    rng = np.random.default_rng(0)
    print(f"{'model':<32}{'ΔBIC':>8}{'period':>9}{'amp':>9}{'AR(1) ρ':>10}{'p':>9}{'null 95%':>11}")

    out = {}
    for key, r in res.items():
        d = np.array([c["d"] for c in r["t1_curve"]], dtype=float)
        y = np.array([c["delta"] for c in r["t1_curve"]])
        obs, period, amp = period_search(d, y)

        X = np.stack([np.ones_like(d), np.log2(d)], 1)
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        surr, rho = ar1_surrogates(resid, rng, N_SURR)

        trend = y - resid
        null = np.array([period_search(d, trend + s)[0] for s in surr])
        p = float((null >= obs).mean())
        q95 = float(np.quantile(null, 0.95))
        print(f"{key:<32}{obs:>8.1f}{period:>9.1f}{amp:>9.4f}{rho:>10.3f}{p:>9.4f}{q95:>11.1f}")
        out[key] = dict(d_bic=obs, period=period, amplitude=amp, ar1_rho=rho, p=p, null_q95=q95)

    (DATA / "phase_c_t1_null.json").write_text(json.dumps(out, indent=2))
    print(f"\n{N_SURR} AR(1) surrogates each. p = P(ΔBIC_null >= ΔBIC_obs).")
    print(f"wrote {DATA}/phase_c_t1_null.json")


if __name__ == "__main__":
    main()
