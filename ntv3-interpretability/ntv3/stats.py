"""Bootstrap CIs, paired tests, and FDR correction shared across phases.
Model-agnostic -- shared verbatim with the (removed) NTv2 port."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    statistic=np.mean,
) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = statistic(sample)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean": float(statistic(values)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
        "frac_positive": float(np.mean(values > 0)),
    }


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, *, n_resamples: int = 2000, seed: int = 42
) -> dict:
    """Bootstrap CI on the paired mean difference a - b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal length")
    diff = a - b
    return bootstrap_ci(diff, n_resamples=n_resamples, seed=seed)


def wilcoxon_one_sided(a: np.ndarray, b: np.ndarray) -> dict:
    """One-sided paired Wilcoxon signed-rank test: is a > b?"""
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    stat, p = wilcoxon(a, b, alternative="greater")
    return {"statistic": float(stat), "p_value": float(p)}


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH-adjusted (FDR) q-values, same order as input."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out.tolist()


def spearman(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    rho, p = spearmanr(a, b)
    return {"rho": float(rho), "p_value": float(p)}


def pearson(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import pearsonr

    r, p = pearsonr(a, b)
    return {"r": float(r), "p_value": float(p)}
