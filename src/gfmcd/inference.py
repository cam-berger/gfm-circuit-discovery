"""Gene-grouped inference: one module for bootstrap intervals, sign-flip tests and Holm.

The repository had about thirty bootstrap definitions. Two of them, ``circuit.gboot`` and
``grouped_stats.grouped_bootstrap``, are the same estimator and differ only in group order; others
silently use a different estimand (mean of gene means) under the same "gene bootstrap" label, or
clamp a ratio's denominator so that a negative sum returns about +/-1e9 times the numerator (R55).

This module names every choice at its interface:

* ``weighting="row"`` pools rows and resamples whole genes (the pinned copies' estimand);
  ``weighting="gene"`` averages gene means.
* ``order="sorted"`` makes results independent of row order; ``"first_seen"`` exists so tests can
  reproduce ``circuit.gboot`` exactly.
* A ratio keeps its denominator's sign. A replicate whose denominator mean has the opposite sign to
  the point estimate's passes through infinity, so ``strict_sign=True`` refuses it.
* ``n_boot``/``n_perm`` and ``seed`` are required: silent defaults are how draw counts drifted.

Resampling draws ``rng.integers(0, K, (chunk, K))``, the same stream the per-draw
``rng.choice(K, K)`` loops consume, and evaluates each replicate on the rows in the same order, so
results are bit-identical to the copies it replaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np

Weighting = Literal["row", "gene"]
Order = Literal["sorted", "first_seen"]

_CHUNK = 1024
_LEGACY_KEYS = ("estimate", "ci_low", "ci_high", "n", "n_groups", "n_boot")


@dataclass(frozen=True)
class Interval:
    """A point estimate with its gene-grouped percentile interval and how it was made."""

    estimate: float
    low: float
    high: float
    n: int
    n_groups: int
    n_dropped: int
    n_boot: int
    seed: int
    confidence: float
    method: str
    sign_changes: int = 0

    def as_dict(self) -> dict:
        """``grouped_stats``' keys first (``ci_low``/``ci_high``), then the rest."""
        out = {"estimate": self.estimate, "ci_low": self.low, "ci_high": self.high,
               "n": self.n, "n_groups": self.n_groups, "n_boot": self.n_boot}
        out.update({k: v for k, v in asdict(self).items()
                    if k not in ("estimate", "low", "high", "n", "n_groups", "n_boot")})
        return out

    def legacy_dict(self) -> dict:
        """Exactly the dict ``grouped_stats`` returned before this module existed."""
        return {k: self.as_dict()[k] for k in _LEGACY_KEYS}


def _quantile_levels(confidence: float) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    # Rounded so 0.95 gives exactly 0.025/0.975, the levels every pinned copy uses.
    low = round((1.0 - confidence) / 2.0, 12)
    return low, round(1.0 - low, 12)


class GroupedSample:
    """Rows with a group label each (a gene, usually), resampled by whole group.

    Non-finite rows are dropped and counted. Within a group, rows keep their input order.
    """

    def __init__(self, values: Sequence[float] | np.ndarray, groups: Sequence | np.ndarray, *,
                 weighting: Weighting = "row", order: Order = "sorted") -> None:
        if weighting not in ("row", "gene"):
            raise ValueError("weighting must be 'row' or 'gene'")
        if order not in ("sorted", "first_seen"):
            raise ValueError("order must be 'sorted' or 'first_seen'")
        self._values = np.asarray(values, dtype=np.float64)
        self._groups = np.asarray(groups, dtype=object)
        if self._values.ndim != 1 or self._values.shape != self._groups.shape:
            raise ValueError("values and groups must be matched one-dimensional arrays")
        self.weighting, self.order = weighting, order

    # -- estimands -----------------------------------------------------------------------------

    def mean(self, *, n_boot: int, seed: int, confidence: float = 0.95) -> Interval:
        keep = np.isfinite(self._values)
        if not keep.any():
            raise ValueError("no finite values")
        layout = _Layout(self._groups[keep], self.order)
        values = self._values[keep]
        if self.weighting == "row":
            estimate = float(values.mean())
            statistic = lambda rows, ids: values[rows].mean()  # noqa: E731
        else:
            gene_means = layout.group_means(values)
            estimate = float(gene_means.mean())
            statistic = lambda rows, ids: gene_means[ids].mean()  # noqa: E731
        draws = layout.bootstrap(statistic, n_boot=n_boot, seed=seed)
        low_q, high_q = _quantile_levels(confidence)
        return Interval(estimate, float(np.quantile(draws, low_q)), float(np.quantile(draws, high_q)),
                        int(values.size), layout.n_groups, int((~keep).sum()), int(n_boot), int(seed),
                        float(confidence), self._method("mean"))

    def ratio(self, denominator: Sequence[float] | np.ndarray, *, n_boot: int, seed: int,
              confidence: float = 0.95, strict_sign: bool = True) -> Interval:
        """Ratio of means; numerator and denominator are resampled together by group."""
        denominator = np.asarray(denominator, dtype=np.float64)
        if denominator.shape != self._values.shape:
            raise ValueError("numerator, denominator, and groups must have matched shapes")
        keep = np.isfinite(self._values) & np.isfinite(denominator)
        if not keep.any():
            raise ValueError("no finite paired values")
        layout = _Layout(self._groups[keep], self.order)
        num, den = self._values[keep], denominator[keep]
        if self.weighting == "row":
            num_point, den_point = float(num.mean()), float(den.mean())
            parts = lambda rows, ids: (num[rows].mean(), den[rows].mean())  # noqa: E731
        else:
            num_means, den_means = layout.group_means(num), layout.group_means(den)
            num_point, den_point = float(num_means.mean()), float(den_means.mean())
            parts = lambda rows, ids: (num_means[ids].mean(), den_means[ids].mean())  # noqa: E731
        eps = np.finfo(np.float64).eps
        if abs(den_point) <= eps:
            raise ValueError("ratio denominator mean is zero")
        sign_changes = 0

        def statistic(rows, ids):
            nonlocal sign_changes
            top, bottom = parts(rows, ids)
            bottom = float(bottom)
            if abs(bottom) <= eps:
                return np.nan
            sign_changes += int(np.sign(bottom) != np.sign(den_point))
            return float(top) / bottom

        draws = layout.bootstrap(statistic, n_boot=n_boot, seed=seed)
        if strict_sign and sign_changes:
            raise ValueError(
                f"{sign_changes}/{n_boot} bootstrap denominators change sign; the ratio passes "
                "through infinity and has no percentile interval (strict_sign=False to override)"
            )
        finite = draws[np.isfinite(draws)]
        if finite.size < 0.99 * n_boot:
            raise ValueError("too many zero bootstrap denominators")
        low_q, high_q = _quantile_levels(confidence)
        return Interval(num_point / den_point, float(np.quantile(finite, low_q)),
                        float(np.quantile(finite, high_q)), int(num.size), layout.n_groups,
                        int((~keep).sum()), int(n_boot), int(seed), float(confidence),
                        self._method("ratio of means"), sign_changes)

    def signflip_p(self, *, n_perm: int, seed: int) -> float:
        """Two-sided group-level sign flip of the mean, with the exact add-one correction."""
        keep = np.isfinite(self._values)
        if not keep.any():
            raise ValueError("no finite values")
        layout = _Layout(self._groups[keep], self.order)
        values = self._values[keep]
        if self.weighting == "gene":
            values, inverse = layout.group_means(values), np.arange(layout.n_groups)
        else:
            inverse = layout.inverse
        observed = abs(float(values.mean()))
        rng = np.random.default_rng(seed)
        signs = np.asarray([-1.0, 1.0])
        exceed = 0
        for start in range(0, n_perm, _CHUNK):
            flips = signs[rng.integers(0, 2, size=(min(_CHUNK, n_perm - start), layout.n_groups))]
            for row in flips:
                exceed += abs(float(np.mean(values * row[inverse]))) >= observed
        return float((exceed + 1) / (n_perm + 1))

    def _method(self, estimand: str) -> str:
        return (f"gene-grouped percentile bootstrap; {self.weighting}-weighted {estimand}; "
                f"{self.order} groups; PCG64")


class _Layout:
    """Group codes, row positions per group, and the per-replicate row gather."""

    def __init__(self, groups: np.ndarray, order: Order) -> None:
        if order == "sorted":
            labels, inverse = np.unique(groups, return_inverse=True)
        else:
            codes: dict = {}
            inverse = np.fromiter((codes.setdefault(g, len(codes)) for g in groups), dtype=np.int64,
                                  count=groups.size)
            labels = np.asarray(list(codes), dtype=object)
        self.labels, self.inverse = labels, np.asarray(inverse, dtype=np.int64)
        self.n_groups = int(labels.size)
        self._rows = np.argsort(self.inverse, kind="stable")  # rows grouped, input order kept
        self._counts = np.bincount(self.inverse, minlength=self.n_groups)
        self._starts = np.concatenate([[0], np.cumsum(self._counts)[:-1]])

    def group_means(self, values: np.ndarray) -> np.ndarray:
        return np.asarray([values[self._rows[s:s + c]].mean()
                           for s, c in zip(self._starts, self._counts)])

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Row indices for resampled groups, concatenated in draw order."""
        counts = self._counts[ids]
        offsets = np.repeat(self._starts[ids] - np.cumsum(counts) + counts, counts)
        return self._rows[offsets + np.arange(int(counts.sum()))]

    def bootstrap(self, statistic, *, n_boot: int, seed: int) -> np.ndarray:
        if n_boot < 1:
            raise ValueError("n_boot must be positive")
        rng = np.random.default_rng(seed)
        draws = np.empty(n_boot, dtype=np.float64)
        for start in range(0, n_boot, _CHUNK):
            block = rng.integers(0, self.n_groups, size=(min(_CHUNK, n_boot - start), self.n_groups))
            for offset, ids in enumerate(block):
                draws[start + offset] = statistic(self.gather(ids), ids)
        return draws


def holm(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values: stable sort, running maximum, capped at 1."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite one-dimensional array in [0,1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted
