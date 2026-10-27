"""R55: the byte-pinned ratio helpers clamp a non-positive denominator; grouped_stats does not.

`circuit.ratio_of_means` / `circuit.gboot_ratio` divide by `max(sum, 1e-9)`. They cannot be fixed
in place (protocol freezes pin `circuit.py`), so these tests pin both behaviours: parity wherever
the denominator is positive, which covers every recorded canonical caller, and the divergence
that new code must avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

from gfmcd.splicing import circuit, grouped_stats


def test_parity_with_the_pinned_helper_for_positive_denominators() -> None:
    rng = np.random.default_rng(0)
    numerator = rng.normal(0.4, 0.3, size=200)
    denominator = rng.normal(0.9, 0.2, size=200)
    assert grouped_stats.ratio_of_means(numerator, denominator) == pytest.approx(
        circuit.ratio_of_means(numerator.tolist(), denominator.tolist()), rel=1e-12
    )


def test_negative_denominator_keeps_its_sign() -> None:
    numerator, denominator = [-0.59, -0.59], [-1.0, -1.0]
    assert grouped_stats.ratio_of_means(numerator, denominator) == pytest.approx(0.59)
    # The pinned helper's clamp: documented here so a silent "fix" of a frozen file is visible.
    assert circuit.ratio_of_means(numerator, denominator) < -1e8


def test_zero_denominator_is_refused() -> None:
    with pytest.raises(ValueError, match="zero"):
        grouped_stats.ratio_of_means([1.0, -1.0], [1.0, -1.0])


def test_grouped_bootstrap_ratio_is_sign_correct() -> None:
    groups = np.repeat(np.arange(20), 3)
    denominator = np.full(groups.size, -1.0)
    numerator = 0.59 * denominator + np.random.default_rng(1).normal(0, 0.01, groups.size)
    out = grouped_stats.grouped_bootstrap_ratio(numerator, denominator, groups, n_boot=500, seed=0)
    assert out["estimate"] == pytest.approx(0.59, abs=0.01)
    assert 0.5 < out["ci_low"] <= out["ci_high"] < 0.7
    low, high = circuit.gboot_ratio(numerator, denominator, groups, n=500, seed=0)
    assert high < -1e6  # every replicate clamped
