"""Attention on the LOG-ODDS scale, and the softmax-curvature confound it removes.

``M4c`` measured how much of ``L3H1``'s donor attention MASS an early-head ablation destroys,
and found the minor class losing 20-38% against canonical's 0-9%.  That was read as evidence
that those heads carry class-specific content.  It does not establish it, because the two arms
start from different baseline masses (``GC`` 0.170, ``GT`` 0.467) and the softmax is nonlinear:
identical shifts in the underlying attention scores produce different fractional mass changes
depending on where on the curve each arm sits.

The fix is to work in the quantity the model actually computes.  Writing ``D`` for the donor's
key columns and ``V`` for every valid column, the donor's attention mass is

    p = sum_D exp(s) / sum_V exp(s) = sigmoid( LSE(s_D) - LSE(s_{V\\D}) ) = sigmoid(m)

so the score margin ``m`` is an EXACT reparameterisation of the mass -- no approximation, and
recoverable from ``p`` alone.  Margins are additive where masses are not, which is what makes
the two arms comparable: an ablation's effect is ``delta_m``, and the same ``delta_m`` on two
different baselines yields two different fractional mass drops for no reason but curvature.

That yields the ordered diagnostic in :func:`decompose_asymmetry`.  Give the comparator arm
the treatment arm's margin shift and re-read its fractional drop; the two resulting terms are
exactly additive in the log ratio.  This is one reference-arm path through an interaction,
not a unique causal decomposition: reversing the order can reallocate mass between the terms.

Everything here is pure arithmetic on arrays -- no model, no torch, no data access -- so the
decision rule is testable against hand-computed cases before any GPU time is spent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def logsumexp(values) -> float:
    """Stable ``log(sum(exp(values)))``; ``-inf`` for an all-masked set rather than ``nan``."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return float("-inf")
    peak = v.max()
    if not np.isfinite(peak):
        # Every entry is -inf (fully masked) or the scores are corrupt; either way the
        # max-shift below would produce nan.  Propagate the honest value.
        return float(peak)
    return float(peak + np.log(np.exp(v - peak).sum()))


def sigmoid(margin) -> np.ndarray:
    """Logistic function, evaluated without overflow in either tail.

    Written through ``exp(-|m|)``, which cannot overflow, so the minor class's large negative
    margins stay exact instead of saturating.  Branch-free so that scalars, 0-d arrays and
    arrays all behave identically.
    """
    m = np.asarray(margin, dtype=float)
    decay = np.exp(-np.abs(m))
    return np.where(m >= 0, 1.0 / (1.0 + decay), decay / (1.0 + decay))


def margin_from_scores(row, target_start: int, valid_len: int, *, width: int = 2):
    """Score margin of a target span against every other valid column of one attention row.

    ``row`` is one query's POST-MASK PRE-SOFTMAX scores.  Only the first ``valid_len`` columns
    are considered, so padding cannot leak into the comparison set.  Returns
    ``(margin, target_lse, other_lse)``; ``sigmoid(margin)`` equals the softmax mass on the
    target span exactly, which is the identity every consumer here relies on.
    """
    s = np.asarray(row, dtype=float).ravel()
    if valid_len <= 0 or valid_len > s.size:
        raise ValueError(f"valid_len {valid_len} outside 1..{s.size}")
    if width <= 0:
        raise ValueError("width must be positive")
    stop = target_start + width
    if target_start < 0 or stop > valid_len:
        raise ValueError(
            f"target span [{target_start},{stop}) is not inside the valid region "
            f"[0,{valid_len})"
        )
    s = s[:valid_len]
    target = logsumexp(s[target_start:stop])
    other = logsumexp(np.concatenate([s[:target_start], s[stop:]]))
    return target - other, target, other


def fractional_drop(baseline, ablated) -> float:
    """``1 - mean(ablated)/mean(baseline)``.

    Ratio of means, never a mean of per-row ratios: masses run close to zero on the minor
    class and a per-row ratio would be dominated by whichever row happened to be smallest.
    """
    b = float(np.asarray(baseline, dtype=float).mean())
    if not b > 0.0:
        raise ValueError("baseline mass must have a strictly positive mean")
    return 1.0 - float(np.asarray(ablated, dtype=float).mean()) / b


def implied_drop(baseline_margin, shift, baseline_mass=None) -> float:
    """Fractional mass drop an arm would show if its margins fell by ``shift``.

    ``baseline_mass`` defaults to ``sigmoid(baseline_margin)``.  Pass the measured masses
    instead to keep the counterfactual anchored to observation: the two agree to numerical
    precision whenever the margins were derived from those same masses, so a disagreement is
    a bookkeeping error worth surfacing rather than smoothing over.
    """
    m = np.asarray(baseline_margin, dtype=float)
    base = sigmoid(m) if baseline_mass is None else np.asarray(baseline_mass, dtype=float)
    return fractional_drop(base, sigmoid(m - np.asarray(shift, dtype=float)))


@dataclass(frozen=True)
class Asymmetry:
    """One ordered arithmetic split of the arms' attention-mass-drop asymmetry."""

    treatment_drop: float
    comparator_drop: float
    swapped_drop: float
    log_ratio: float
    curvature: float
    score_shift: float
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


#: Fraction of the log asymmetry that must sit in one component before it is named the
#: dominant cause.  Fixed here rather than chosen per probe, so it cannot be tuned to a result.
DOMINANCE = 0.60


def decompose_asymmetry(
    treatment_drop: float,
    comparator_drop: float,
    swapped_drop: float,
) -> Asymmetry:
    """Split ``log(F_treatment / F_comparator)`` along one comparator-anchored path.

    ``swapped_drop`` is the comparator arm re-evaluated under the TREATMENT arm's margin
    shifts.  It shares its baseline with the comparator and its shift with the treatment, so:

        log F_t - log F_swap   both carry the treatment's shift, baselines differ  -> CURVATURE
        log F_swap - log F_c   both carry the comparator's baseline, shifts differ -> SCORE SHIFT

    and the two sum to the whole log ratio identically.  The interaction allocation depends on
    this order/reference arm; it is a descriptive diagnostic, not a unique causal attribution.
    A drop that is zero or negative (an ablation that INCREASED donor mass) has no logarithm;
    those return ``UNDEFINED`` rather than a silently clipped number.
    """
    drops = (treatment_drop, comparator_drop, swapped_drop)
    if not all(np.isfinite(d) and d > 0 for d in drops):
        return Asymmetry(*drops, float("nan"), float("nan"), float("nan"), "UNDEFINED")

    log_ratio = float(np.log(treatment_drop) - np.log(comparator_drop))
    curvature = float(np.log(treatment_drop) - np.log(swapped_drop))
    score_shift = float(np.log(swapped_drop) - np.log(comparator_drop))

    if abs(log_ratio) < 1e-12:
        verdict = "NO_ASYMMETRY"
    elif score_shift / log_ratio >= DOMINANCE:
        verdict = "SCORE_SHIFT_DOMINATES"
    elif curvature / log_ratio >= DOMINANCE:
        verdict = "CURVATURE_DOMINATES"
    else:
        verdict = "MIXED"
    return Asymmetry(treatment_drop, comparator_drop, swapped_drop,
                     log_ratio, curvature, score_shift, verdict)
