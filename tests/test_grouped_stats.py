import numpy as np
import pytest

from gfmcd.splicing.grouped_stats import (
    grouped_bootstrap,
    grouped_bootstrap_ratio,
    grouped_signflip_pvalue,
)


def test_grouped_bootstrap_is_deterministic_and_group_aware():
    values = np.asarray([1.0, 3.0, 10.0])
    groups = np.asarray(["a", "a", "b"], dtype=object)
    first = grouped_bootstrap(values, groups, n_boot=200, seed=4)
    second = grouped_bootstrap(values, groups, n_boot=200, seed=4)
    assert first == second
    assert first["estimate"] == pytest.approx(14 / 3)
    assert first["n_groups"] == 2


def test_grouped_signflip_has_add_one_floor():
    groups = np.asarray([f"g{index}" for index in range(20)], dtype=object)
    pvalue = grouped_signflip_pvalue(
        np.ones(20),
        groups,
        n_perm=999,
        seed=2,
    )
    assert 1 / 1000 <= pvalue <= 0.01


def test_grouped_bootstrap_ratio_is_ratio_of_means_and_deterministic():
    numerator = np.asarray([2.0, 4.0, 3.0, 9.0])
    denominator = np.asarray([1.0, 2.0, 3.0, 3.0])
    groups = np.asarray(["a", "a", "b", "c"], dtype=object)
    first = grouped_bootstrap_ratio(
        numerator,
        denominator,
        groups,
        n_boot=200,
        seed=7,
    )
    second = grouped_bootstrap_ratio(
        numerator,
        denominator,
        groups,
        n_boot=200,
        seed=7,
    )
    assert first == second
    assert first["estimate"] == pytest.approx(
        np.mean(numerator) / np.mean(denominator)
    )
