"""Gene-grouped uncertainty helpers for paired splicing assays.

The bootstrap and sign-flip helpers are adapters over :mod:`gfmcd.inference` and return the dicts
they always did (row-weighted, sorted groups). New code should use ``gfmcd.inference`` directly.
"""

from __future__ import annotations

import numpy as np

from gfmcd.inference import GroupedSample


def grouped_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    """Gene-grouped percentile interval for the exon-weighted mean."""

    return GroupedSample(values, groups).mean(n_boot=n_boot, seed=seed).legacy_dict()


def ratio_of_means(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Ratio of exon-weighted means, keeping the denominator's sign.

    The byte-pinned ``circuit.ratio_of_means`` divides by ``max(sum, 1e-9)``, so a negative
    denominator sum returns about +/-1e9 times the numerator instead of the ratio (R55). This
    returns the arithmetic ratio and refuses only a zero denominator.
    """

    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    if numerator_array.shape != denominator_array.shape:
        raise ValueError("numerator and denominator must have matched shapes")
    keep = np.isfinite(numerator_array) & np.isfinite(denominator_array)
    if not keep.any():
        raise ValueError("no finite paired values")
    denominator_mean = float(denominator_array[keep].mean())
    if abs(denominator_mean) <= np.finfo(np.float64).eps:
        raise ValueError("ratio denominator mean is zero")
    return float(numerator_array[keep].mean()) / denominator_mean


def grouped_signflip_pvalue(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> float:
    """Two-sided gene-level sign flip with the exact add-one correction."""

    return GroupedSample(values, groups).signflip_p(n_perm=n_perm, seed=seed)


def grouped_bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    """Gene-grouped interval for a ratio of exon-weighted means.

    The numerator and denominator are resampled together by gene. This is a
    ratio of means, never a mean of per-example ratios. Sign-changing replicates are
    allowed here, as they always were; ``gfmcd.inference`` refuses them by default.
    """

    sample = GroupedSample(numerator, groups)
    return sample.ratio(denominator, n_boot=n_boot, seed=seed, strict_sign=False).legacy_dict()
