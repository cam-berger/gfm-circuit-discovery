"""Evaluation and causal-effect metrics for splice-pair selection."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable

import numpy as np
import torch

from gfmcd.splicing.readout import PairIndexBatch, true_pair_margin


@dataclass(frozen=True)
class PairMetrics:
    n: int
    top1_accuracy: float
    mean_reciprocal_rank: float
    mean_nll: float
    mean_margin: float
    positive_margin_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_pair_logits(logits: torch.Tensor, pairs: PairIndexBatch) -> PairMetrics:
    if logits.shape != pairs.mask.shape:
        raise ValueError("logits and mask shapes disagree")
    with torch.no_grad():
        masked = logits.masked_fill(~pairs.mask, torch.finfo(logits.dtype).min)
        order = torch.argsort(masked, dim=-1, descending=True)
        matches = order == pairs.labels[:, None]
        ranks = matches.float().argmax(-1) + 1
        log_probs = torch.log_softmax(masked, dim=-1)
        b = torch.arange(len(logits), device=logits.device)
        nll = -log_probs[b, pairs.labels]
        margin = true_pair_margin(masked, pairs)
        return PairMetrics(
            n=len(logits),
            top1_accuracy=float((ranks == 1).float().mean()),
            mean_reciprocal_rank=float((1.0 / ranks.float()).mean()),
            mean_nll=float(nll.mean()),
            mean_margin=float(margin.mean()),
            positive_margin_rate=float((margin > 0).float().mean()),
        )


def bootstrap_difference(
    a: Iterable[float],
    b: Iterable[float],
    *,
    paired: bool = True,
    n_boot: int = 5000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap mean(A-B), with paired resampling by default."""
    aa = np.asarray(list(a), dtype=np.float64)
    bb = np.asarray(list(b), dtype=np.float64)
    if paired and len(aa) != len(bb):
        raise ValueError("paired bootstrap requires equal lengths")
    if len(aa) == 0 or len(bb) == 0:
        raise ValueError("bootstrap samples cannot be empty")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    if paired:
        diff = aa - bb
        for i in range(n_boot):
            idx = rng.integers(0, len(diff), len(diff))
            stats[i] = diff[idx].mean()
        estimate = diff.mean()
    else:
        for i in range(n_boot):
            ia = rng.integers(0, len(aa), len(aa))
            ib = rng.integers(0, len(bb), len(bb))
            stats[i] = aa[ia].mean() - bb[ib].mean()
        estimate = aa.mean() - bb.mean()
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(stats, [alpha, 1 - alpha])
    return {
        "estimate": float(estimate),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": confidence,
        "n_boot": n_boot,
    }


def recovery_fraction(clean: float, corrupt: float, patched: float, *, eps: float = 1e-12) -> float:
    """Fraction of the clean-corrupt score gap restored by an intervention."""
    gap = clean - corrupt
    if abs(gap) <= eps:
        raise ValueError("clean and corrupt scores are indistinguishable")
    return (patched - corrupt) / gap


def matched_random_pvalue(observed: float, random_effects: Iterable[float], *, larger_is_stronger: bool = True) -> float:
    effects = np.asarray(list(random_effects), dtype=np.float64)
    if effects.size == 0:
        raise ValueError("random_effects cannot be empty")
    if larger_is_stronger:
        exceed = np.count_nonzero(effects >= observed)
    else:
        exceed = np.count_nonzero(effects <= observed)
    # Add-one correction prevents a misleading p=0 from finite random draws.
    return float((exceed + 1) / (len(effects) + 1))


def grouped_bootstrap(
    values: Iterable[float],
    groups: Iterable[str],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap by gene/transcript rather than treating exons as independent."""
    vals = np.asarray(list(values), dtype=np.float64)
    grps = np.asarray(list(groups), dtype=object)
    if len(vals) != len(grps) or len(vals) == 0:
        raise ValueError("values/groups must be non-empty and aligned")
    unique = np.unique(grps)
    by_group = {g: vals[grps == g] for g in unique}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        merged = np.concatenate([by_group[g] for g in sampled])
        boot[i] = statistic(merged)
    return {
        "estimate": float(statistic(vals)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_groups": int(len(unique)),
        "n": int(len(vals)),
    }
