import numpy as np


def bootstrap_ci(values, n_boot: int, alpha: float, seed: int):
    """Percentile bootstrap CI for the mean of `values`.

    Returns (mean, lo, hi). Falls back to (nan, nan, nan) on empty input.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_recovery_ci(patched, degraded, clean, n_boot: int, alpha: float, seed: int):
    """Percentile bootstrap CI for a recovery fraction:

        (mean(patched) - mean(degraded)) / (mean(clean) - mean(degraded))

    i.e. how much of the clean-vs-degraded gap a single patched component
    restores. Computed as a ratio of resampled *means* (not a per-sample
    ratio) to avoid blowing up on samples where clean and degraded are
    nearly equal - standard for recovery-fraction reporting in activation
    patching. Returns (point_estimate, lo, hi).
    """
    patched = np.asarray(patched, dtype=float)
    degraded = np.asarray(degraded, dtype=float)
    clean = np.asarray(clean, dtype=float)
    point = (patched.mean() - degraded.mean()) / (clean.mean() - degraded.mean())

    rng = np.random.default_rng(seed)
    n = len(patched)
    idx = rng.integers(0, n, size=(n_boot, n))
    num = patched[idx].mean(axis=1) - degraded[idx].mean(axis=1)
    den = clean[idx].mean(axis=1) - degraded[idx].mean(axis=1)
    ratio = num / den
    lo, hi = np.percentile(ratio, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
