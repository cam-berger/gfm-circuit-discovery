import numpy as np
import pytest

from gfmcd.splicing.positional_spectrum import (
    band_coefficient_matrix,
    band_component,
    compare_band,
    delete_band,
    exact_signflip_pvalue,
    parseval_relative_error,
    power_spectrum,
    replace_band,
    scan_windows,
    select_power_matched_controls,
    select_scan_candidates,
    subspace_comparison,
)


def _tone_table(n=66, d=8):
    positions = np.arange(n)[:, None]
    table = np.zeros((n, d), dtype=np.float64)
    table[:, 0] = np.cos(2 * np.pi * positions[:, 0] / 3)
    table[:, 1] = np.sin(2 * np.pi * positions[:, 0] / 3)
    table[:, 2] = 0.4 * np.cos(2 * np.pi * positions[:, 0] / 11)
    table[:, 3] = 0.4 * np.sin(2 * np.pi * positions[:, 0] / 11)
    return table


def test_fft_roundtrip_parseval_delete_and_replace():
    table = _tone_table()
    bins = (21, 22, 23)
    component = band_component(table, bins)
    deleted = delete_band(table, bins)
    assert parseval_relative_error(table) <= 1e-14
    assert np.max(np.abs(table - (deleted + component))) <= 1e-14
    assert np.max(np.abs(power_spectrum(deleted)[list(bins)])) <= 1e-20
    assert np.allclose(replace_band(deleted, table, bins), table, atol=1e-14)


def test_identity_comparison_and_orthogonal_invariance():
    rng = np.random.default_rng(4)
    table = rng.normal(size=(66, 8))
    comparison = compare_band(table, table, (19, 20, 21))
    assert comparison["normalized_power_ratio"] == pytest.approx(1.0)
    assert comparison["reconstruction_cosine"] == pytest.approx(1.0)
    assert comparison["subspace"]["symmetric_projection_overlap"] == pytest.approx(1.0)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert np.allclose(power_spectrum(table @ q), power_spectrum(table), rtol=1e-12)


def test_subspace_comparison_separates_orthogonal_rows():
    left = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    same = subspace_comparison(left, left)
    orthogonal = subspace_comparison(left, np.asarray([[0.0, 0.0, 1.0]]))
    assert same["symmetric_projection_overlap"] == pytest.approx(1.0)
    assert orthogonal["symmetric_projection_overlap"] == pytest.approx(0.0)


def test_scan_and_candidate_rule_find_consistent_gain():
    base = _tone_table(n=66)
    tuned = {}
    for seed in range(5):
        current = base.copy()
        current[:, 4] = (3.0 + 0.1 * seed) * np.cos(
            2 * np.pi * np.arange(66) / 6
        )
        tuned[seed] = current
    records = scan_windows(base + 1e-4, tuned, width=3, start_min=8, start_max=30)
    selected = select_scan_candidates(
        records,
        protected_bands=((21, 22, 23),),
        maximum_per_direction=1,
        minimum_fold=2,
    )
    assert selected
    assert selected[0]["direction"] == "gain"
    assert 11 in selected[0]["bins"]


def test_exact_signflip_has_add_one_resolution_floor():
    assert exact_signflip_pvalue(np.ones(5)) == pytest.approx(3 / 33)


def test_power_matched_controls_respect_protected_and_nonoverlap():
    rng = np.random.default_rng(9)
    table = rng.normal(size=(66, 8))
    targets = ({"name": "target", "bins": [10, 11, 12]},)
    controls = select_power_matched_controls(
        table,
        targets,
        protected_bands=([10, 11, 12],),
        n_controls=2,
        tolerance=1.0,
        start_min=3,
        start_max=28,
    )["target"]
    assert controls is not None
    assert len(controls) == 2
    assert set(controls[0]).isdisjoint(controls[1])
    assert set(controls[0]).isdisjoint({10, 11, 12})
    assert set(controls[1]).isdisjoint({10, 11, 12})


def test_band_coefficient_matrix_has_real_imag_rows():
    matrix = band_coefficient_matrix(_tone_table(), (21, 22, 23))
    assert matrix.shape == (6, 8)
