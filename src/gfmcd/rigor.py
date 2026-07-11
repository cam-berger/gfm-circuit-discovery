"""Reusable statistical machinery for rigor refactors: nulls, calibration, uncertainty.

VENDORED 2026-07-11 from the research-rigor-refactor kit
(rigor_kit.py, source sha256 989576b0b7ce7509805d6fd37d10caa6b2d73b8cd188138205b832dfad3a2a1b),
unmodified. Requires real numpy + scipy (pinned in pyproject; scipy is used only for
scipy.stats.kstest). Do NOT substitute the /tmp scipy shim — a hand-rolled shim
masquerading as scipy would make calibration results irreproducible.

Import this into the target repo (copy it in, or vendor the functions you need) rather than
rewriting max-null calibration by hand for the tenth time.  Only numpy and scipy required.

The organizing rule: a statistic selected by a search is only interpretable against a null that
performs the same search.

Quick use:

    from rigor_kit import max_null_calibrate, audit_calibration, benjamini_hochberg

    # calibrate a max-over-candidates score
    result = max_null_calibrate(observed_scores, null_draw_fn, n_null=499)

    # verify the machinery on pure noise before trusting it on real data
    print(audit_calibration(score_fn, null_draw_fn, search_sizes=(4, 16, 48)))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import stats as _stats

FloatArray = np.ndarray


# ---------------------------------------------------------------------------
# p-values and multiplicity
# ---------------------------------------------------------------------------


def empirical_upper_p_value(observed: float, null_samples: Iterable[float]) -> float:
    """Finite-sample corrected upper-tail p-value: (1 + #{null >= obs}) / (n + 1).

    Never use #{null >= obs}/n — it can return exactly 0, which is a claim no finite null
    supports and which breaks downstream FDR.
    """

    null = np.asarray(list(null_samples), dtype=np.float64).reshape(-1)
    return float((1 + np.sum(null >= observed)) / (null.size + 1))


def benjamini_hochberg(p_values: Sequence[float]) -> FloatArray:
    """FDR-adjusted q-values (Benjamini-Hochberg)."""

    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * p.size / np.arange(1, p.size + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty_like(q_ranked)
    q[order] = np.clip(q_ranked, 0.0, 1.0)
    return q


def required_null_draws(n_tests: int, *, q: float = 0.1, expected_hits: int = 1) -> int:
    """Minimum null draws so a true positive can actually pass BH.

    With n draws the smallest possible p is 1/(n+1); BH needs p <= q*rank/m.  Too few draws makes
    discoveries mathematically impossible regardless of signal — a silent, common failure.
    """

    if n_tests < 1 or expected_hits < 1 or not 0 < q < 1:
        raise ValueError("bad arguments")
    return int(np.ceil(n_tests / (q * expected_hits)))


# ---------------------------------------------------------------------------
# Max-over-search calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibratedScores:
    """Raw and calibrated statistics kept side by side, never collapsed into one number."""

    raw: FloatArray
    z: FloatArray
    p: FloatArray
    q: FloatArray
    null_mean: float
    null_std: float
    search_size: int
    metadata: dict = field(default_factory=dict)

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.raw))


def max_null_calibrate(
    observed_scores: Sequence[float],
    null_draw_fn: Callable[[np.random.Generator], Sequence[float]],
    *,
    n_null: int = 499,
    random_seed: int = 0,
) -> CalibratedScores:
    """Calibrate scores selected by a max over a search against a same-search max null.

    ``null_draw_fn(rng)`` must return a full set of scores for ONE null trial — the same number
    of candidates the real search had, produced by the same scoring code with the hypothesized
    alignment destroyed (e.g. sign-permuted operands, shuffled labels within exchangeable
    groups).  Preserving nuisance structure (norms, spectra, batch composition) is what makes
    the null informative rather than trivially rejectable.

    Interpretation note: only the p-value of the best candidate is uniform on a true null;
    non-maximal candidates are conservative by construction.  Audit uniformity on maxima only.
    """

    observed = np.asarray(observed_scores, dtype=np.float64).reshape(-1)
    if observed.size == 0:
        raise ValueError("observed_scores cannot be empty")
    if n_null < 1:
        raise ValueError("n_null must be positive")
    rng = np.random.default_rng(random_seed)
    null_maxima = np.empty(n_null, dtype=np.float64)
    for trial in range(n_null):
        draw = np.asarray(list(null_draw_fn(rng)), dtype=np.float64).reshape(-1)
        if draw.size != observed.size:
            raise ValueError(
                f"null draw has {draw.size} scores but the real search had {observed.size}; "
                "the null must reproduce the same search size"
            )
        null_maxima[trial] = draw.max()
    mean = float(null_maxima.mean())
    std = float(null_maxima.std(ddof=1)) if n_null > 1 else 0.0
    p = np.array([empirical_upper_p_value(x, null_maxima) for x in observed])
    return CalibratedScores(
        raw=observed,
        z=(observed - mean) / max(std, 1e-12),
        p=p,
        q=benjamini_hochberg(p),
        null_mean=mean,
        null_std=std,
        search_size=int(observed.size),
        metadata={"n_null": n_null, "calibration": "same_search_max"},
    )


def pooled_bootstrap_max_null(
    pool: Sequence[float],
    *,
    search_size: int,
    n_bootstrap: int = 2000,
    random_seed: int = 0,
) -> FloatArray:
    """Cheap max-null: resample ``search_size`` scores from a pooled comparable distribution.

    Use when per-trial randomization is too expensive.  Pool only over comparable searches (same
    layer / batch / condition).  Real signal in the pool inflates the null maxima, so this errs
    conservative — the safe direction.
    """

    values = np.asarray(pool, dtype=np.float64).reshape(-1)
    if values.size == 0 or search_size < 1 or n_bootstrap < 1:
        raise ValueError("bad arguments")
    rng = np.random.default_rng(random_seed)
    return rng.choice(values, size=(n_bootstrap, search_size), replace=True).max(axis=1)


# ---------------------------------------------------------------------------
# Nulls that preserve nuisance structure
# ---------------------------------------------------------------------------


def signed_permute(matrix: np.ndarray, rng: np.random.Generator) -> FloatArray:
    """Signed row/column permutation: preserves Frobenius norm and singular values exactly.

    Caveat worth stating in any report that uses it: this also destroys global structure shared
    across items (dominant directions, batch effects).  If alignment through shared structure is
    a plausible confound, add a control that removes it before claiming a pair-specific effect.
    """

    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("expected a matrix")
    rp, cp = rng.permutation(m.shape[0]), rng.permutation(m.shape[1])
    rs = rng.choice(np.array([-1.0, 1.0]), size=m.shape[0])
    cs = rng.choice(np.array([-1.0, 1.0]), size=m.shape[1])
    return rs[:, None] * m[rp][:, cp] * cs[None, :]


def block_permute(labels: Sequence, blocks: Sequence, rng: np.random.Generator) -> np.ndarray:
    """Permute labels only within blocks — the correct null when data are grouped/correlated.

    An iid shuffle across correlated units (linked loci, repeated measures, time series) breaks
    the correlation structure and produces an anti-conservative null.
    """

    labels = np.asarray(labels)
    blocks = np.asarray(blocks)
    out = labels.copy()
    for block in np.unique(blocks):
        mask = blocks == block
        out[mask] = rng.permutation(labels[mask])
    return out


# ---------------------------------------------------------------------------
# Calibration audit — the exit criterion for the statistics
# ---------------------------------------------------------------------------


def uniformity_test(p_values: Sequence[float]) -> dict[str, float]:
    """KS test of p-values against Uniform(0,1)."""

    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    if p.size == 0:
        raise ValueError("p_values cannot be empty")
    result = _stats.kstest(p, "uniform")
    return {"ks_statistic": float(result.statistic), "ks_p_value": float(result.pvalue)}


def audit_calibration(
    score_fn: Callable[[np.random.Generator, int], Sequence[float]],
    null_draw_fn_factory: Callable[[np.random.Generator, int], Callable[[np.random.Generator], Sequence[float]]],
    *,
    search_sizes: Sequence[int] = (4, 16, 48),
    n_searches: int = 60,
    n_null: int = 199,
    alpha: float = 0.05,
    fdr_q: float = 0.1,
    random_seed: int = 0,
) -> list[dict]:
    """Run the REAL analysis path on pure-noise inputs; the scan must find nothing.

    ``score_fn(rng, search_size)`` produces one noise search's observed scores.
    ``null_draw_fn_factory(rng, search_size)`` returns that search's null draw function.

    Checks, per search size:
      - KS uniformity of the per-search MAXIMUM p-values (only these are uniform under a max null);
      - false-positive rate at ``alpha`` across all scores (conservatism is fine, excess is not);
      - discoveries at FDR ``fdr_q`` (should be ~0 on pure noise).

    Passing this is what licenses belief in the calibrated numbers on real data.
    """

    rows: list[dict] = []
    for size in search_sizes:
        rng = np.random.default_rng(random_seed + size)
        max_p: list[float] = []
        all_p: list[float] = []
        all_q: list[float] = []
        for _ in range(n_searches):
            observed = score_fn(rng, size)
            calibrated = max_null_calibrate(
                observed,
                null_draw_fn_factory(rng, size),
                n_null=n_null,
                random_seed=int(rng.integers(1 << 30)),
            )
            max_p.append(float(calibrated.p[calibrated.best_index]))
            all_p.extend(calibrated.p.tolist())
        all_q = benjamini_hochberg(all_p).tolist()
        ks = uniformity_test(max_p)
        rows.append(
            {
                "search_size": size,
                "n_searches": n_searches,
                "ks_statistic_maxima": ks["ks_statistic"],
                "ks_p_value_maxima": ks["ks_p_value"],
                "fpr_all_scores": float(np.mean(np.asarray(all_p) <= alpha)),
                "discoveries_at_fdr": float(np.mean(np.asarray(all_q) <= fdr_q)),
                "uniform_ok": bool(ks["ks_p_value"] >= 0.01),
                "fpr_ok": bool(np.mean(np.asarray(all_p) <= alpha) <= alpha * 1.5),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    random_seed: int = 0,
) -> tuple[float, float, float]:
    """(mean, low, high) percentile bootstrap over the units that actually vary."""

    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size == 0 or not 0 < confidence < 1:
        raise ValueError("bad arguments")
    rng = np.random.default_rng(random_seed)
    idx = rng.integers(0, samples.size, size=(n_bootstrap, samples.size))
    means = samples[idx].mean(axis=1)
    a = (1 - confidence) / 2
    return (
        float(samples.mean()),
        float(np.quantile(means, a)),
        float(np.quantile(means, 1 - a)),
    )


def recall_curve(
    ranking: Sequence[str],
    effect_mass: dict[str, float],
    ks: Sequence[int],
) -> dict[int, float]:
    """Fraction of total effect mass captured by the top-k of a ranking.

    The natural screening metric: pair it with random-ordering p95 and the cheap baselines, never
    report it alone.
    """

    total = sum(abs(v) for v in effect_mass.values())
    if total <= 0:
        return {k: 0.0 for k in ks}
    return {k: float(sum(abs(effect_mass[i]) for i in ranking[:k]) / total) for k in ks}


def random_ranking_baseline(
    items: Sequence[str],
    effect_mass: dict[str, float],
    ks: Sequence[int],
    *,
    n_draws: int = 200,
    random_seed: int = 0,
) -> dict[str, dict[int, float]]:
    """Mean and p95 recall curves for random orderings — the floor any ranking must clear."""

    rng = np.random.default_rng(random_seed)
    curves = np.array(
        [
            [recall_curve(list(rng.permutation(items)), effect_mass, ks)[k] for k in ks]
            for _ in range(n_draws)
        ]
    )
    return {
        "mean": {k: float(curves[:, i].mean()) for i, k in enumerate(ks)},
        "p95": {k: float(np.quantile(curves[:, i], 0.95)) for i, k in enumerate(ks)},
    }
