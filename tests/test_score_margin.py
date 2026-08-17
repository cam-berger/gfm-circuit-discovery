"""The arithmetic M4d's verdict rests on, checked before any GPU time is spent.

The decisive property is that ``sigmoid(margin)`` reproduces the softmax mass EXACTLY.  If it
does not, the whole reparameterisation is wrong and every "curvature vs score shift" verdict
built on it is meaningless -- so that identity is tested against an independently computed
softmax rather than against another call into the same module.

The two decomposition cases are constructed so their answers are known in advance by
CONSTRUCTION, not by running the code: equal margin shifts must leave the score-shift
component at exactly zero, and equal baselines must leave the curvature component at exactly
zero.  Neither can pass by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from gfmcd.splicing.score_margin import (
    Asymmetry, decompose_asymmetry, fractional_drop, implied_drop, logsumexp,
    margin_from_scores, sigmoid,
)


def _softmax(v):
    v = np.asarray(v, dtype=float)
    e = np.exp(v - v.max())
    return e / e.sum()


# --------------------------------------------------------------------------- the identity


@pytest.mark.parametrize("seed", range(8))
def test_sigmoid_of_margin_equals_the_softmax_mass(seed):
    """sigmoid(LSE(D) - LSE(V\\D)) == sum of softmax probabilities over D, to machine precision."""
    rng = np.random.default_rng(seed)
    row = rng.normal(0.0, 4.0, size=96)
    valid, start = 71, 13
    margin, _, _ = margin_from_scores(row, start, valid, width=2)
    expected = _softmax(row[:valid])[start:start + 2].sum()
    assert abs(float(sigmoid(margin)) - float(expected)) < 1e-12


def test_padding_beyond_valid_len_cannot_change_the_margin():
    """A pad column with a huge score must not enter the comparison set."""
    rng = np.random.default_rng(0)
    row = rng.normal(0.0, 2.0, size=40)
    clean, _, _ = margin_from_scores(row, 5, 40, width=2)
    polluted = np.concatenate([row, np.full(16, 50.0)])
    got, _, _ = margin_from_scores(polluted, 5, 40, width=2)
    assert abs(clean - got) < 1e-12


def test_margin_decomposes_into_target_and_other_terms():
    rng = np.random.default_rng(3)
    row = rng.normal(0.0, 3.0, size=50)
    margin, target, other = margin_from_scores(row, 9, 50, width=2)
    assert abs(margin - (target - other)) < 1e-12
    assert abs(target - logsumexp(row[9:11])) < 1e-12


def test_margin_rejects_a_span_running_past_the_valid_region():
    row = np.zeros(20)
    with pytest.raises(ValueError):
        margin_from_scores(row, 19, 20, width=2)


def test_margin_rejects_a_valid_len_longer_than_the_row():
    with pytest.raises(ValueError):
        margin_from_scores(np.zeros(10), 0, 11, width=2)


# ------------------------------------------------------------------------------- stability


def test_logsumexp_does_not_overflow_on_large_scores():
    got = logsumexp([800.0, 800.0])
    assert np.isfinite(got)
    assert abs(got - (800.0 + np.log(2.0))) < 1e-9


def test_logsumexp_of_a_fully_masked_set_is_negative_infinity_not_nan():
    got = logsumexp([-np.inf, -np.inf])
    assert got == float("-inf")


def test_sigmoid_saturates_without_overflow():
    assert float(sigmoid(-800.0)) == pytest.approx(0.0, abs=1e-300)
    assert float(sigmoid(800.0)) == pytest.approx(1.0)
    assert np.allclose(sigmoid([0.0]), [0.5])


# -------------------------------------------------------------------------- ratio of means


def test_fractional_drop_is_a_ratio_of_means_not_a_mean_of_ratios():
    """Discriminating: the two estimators disagree here, so only one of them can pass."""
    baseline = np.array([1.0, 100.0])
    ablated = np.array([0.5, 99.0])
    ratio_of_means = 1.0 - (0.5 + 99.0) / (1.0 + 100.0)
    mean_of_ratios = 1.0 - np.mean([0.5 / 1.0, 99.0 / 100.0])
    assert abs(ratio_of_means - mean_of_ratios) > 0.2, "the case must separate the estimators"
    assert fractional_drop(baseline, ablated) == pytest.approx(ratio_of_means)


def test_fractional_drop_rejects_a_non_positive_baseline():
    with pytest.raises(ValueError):
        fractional_drop([0.0, 0.0], [0.0, 0.0])


# ------------------------------------------------------------------------- the adjudication


def test_equal_margin_shifts_put_the_whole_asymmetry_in_curvature():
    """Two arms, same shift, different baselines -> pure softmax curvature, by construction."""
    shift = 0.9
    m_treat = np.full(64, -1.6)                       # minor-class-like: mass ~0.17
    m_comp = np.full(64, -0.13)                       # canonical-like:   mass ~0.47
    f_treat = implied_drop(m_treat, shift)
    f_comp = implied_drop(m_comp, shift)
    f_swap = implied_drop(m_comp, shift)              # comparator baseline, treatment's shift

    out = decompose_asymmetry(f_treat, f_comp, f_swap)
    assert out.score_shift == pytest.approx(0.0, abs=1e-12)
    assert out.curvature == pytest.approx(out.log_ratio, abs=1e-12)
    assert out.verdict == "CURVATURE_DOMINATES"


def test_equal_baselines_put_the_whole_asymmetry_in_the_score_shift():
    """Two arms, same baseline, different shifts -> pure score shift, by construction."""
    base = np.full(64, -0.8)
    f_treat = implied_drop(base, 1.4)
    f_comp = implied_drop(base, 0.2)
    f_swap = implied_drop(base, 1.4)                  # comparator baseline, treatment's shift

    out = decompose_asymmetry(f_treat, f_comp, f_swap)
    assert out.curvature == pytest.approx(0.0, abs=1e-12)
    assert out.score_shift == pytest.approx(out.log_ratio, abs=1e-12)
    assert out.verdict == "SCORE_SHIFT_DOMINATES"


@pytest.mark.parametrize("seed", range(6))
def test_the_two_components_always_sum_to_the_log_ratio(seed):
    rng = np.random.default_rng(seed)
    f_treat, f_comp, f_swap = rng.uniform(0.01, 0.9, size=3)
    out = decompose_asymmetry(float(f_treat), float(f_comp), float(f_swap))
    assert out.curvature + out.score_shift == pytest.approx(out.log_ratio, abs=1e-12)


def test_a_non_positive_drop_is_undefined_rather_than_clipped():
    """An ablation that RAISED donor mass has no log ratio; it must not be silently floored."""
    out = decompose_asymmetry(0.3, -0.02, 0.1)
    assert out.verdict == "UNDEFINED"
    assert np.isnan(out.log_ratio)


def test_collapsed_constant_margin_toy_ratio_stays_below_two():
    """The old 1.556 value is only a constant-baseline/shared-shift toy benchmark.

    This deliberately collapses each arm to one constant margin.  It must not be described as
    a ceiling for heterogeneous row-level baselines or shifts in the M4d cohort.
    """
    m_treat = np.log(0.1702 / (1 - 0.1702))
    m_comp = np.log(0.4669 / (1 - 0.4669))
    ratios = []
    for shift in np.linspace(0.05, 4.0, 80):
        f_t = implied_drop(np.full(1, m_treat), shift)
        f_c = implied_drop(np.full(1, m_comp), shift)
        ratios.append(f_t / f_c)
    assert max(ratios) < 2.0, f"curvature-only ratio reached {max(ratios):.2f}"


def test_component_allocation_changes_when_the_reference_path_is_reversed():
    """The additive split is exact but not a unique causal decomposition."""

    treat_margin = np.array([-2.0, -0.7, 0.1])
    comp_margin = np.array([-0.4, 0.3, 1.2])
    treat_shift = np.array([1.1, 0.5, 0.2])
    comp_shift = np.array([0.2, 0.1, 0.05])
    f_treat = implied_drop(treat_margin, treat_shift)
    f_comp = implied_drop(comp_margin, comp_shift)
    f_comp_under_treat_shift = implied_drop(comp_margin, treat_shift)
    forward = decompose_asymmetry(f_treat, f_comp, f_comp_under_treat_shift)

    f_treat_under_comp_shift = implied_drop(treat_margin, comp_shift)
    reverse_curvature = np.log(f_treat_under_comp_shift) - np.log(f_comp)
    reverse_score_shift = np.log(f_treat) - np.log(f_treat_under_comp_shift)

    assert forward.curvature + forward.score_shift == pytest.approx(forward.log_ratio)
    assert reverse_curvature + reverse_score_shift == pytest.approx(forward.log_ratio)
    assert forward.curvature != pytest.approx(reverse_curvature)
    assert forward.score_shift != pytest.approx(reverse_score_shift)


def test_asymmetry_is_serialisable_for_the_artifact():
    out = decompose_asymmetry(0.3, 0.1, 0.12)
    assert isinstance(out, Asymmetry)
    assert set(out.to_dict()) == {
        "treatment_drop", "comparator_drop", "swapped_drop",
        "log_ratio", "curvature", "score_shift", "verdict",
    }
