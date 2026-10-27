"""A1 arithmetic memorials, independent index oracle, and immutable v1 barrier."""
from dataclasses import replace
import importlib.util
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

from gfmcd.splicing import register_operator_statistics as v1
from gfmcd.splicing import register_operator_statistics_v2 as v2
from gfmcd.splicing.register_operator_measurement import reduce_score_payload
from gfmcd.splicing.register_operator_io import verify_freeze as verify_v1
from gfmcd.splicing.register_operator_io_v2 import verify_freeze as verify_v2

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("calibration_for_a1_memorials",
    ROOT / "scripts/splicebert_register_operator_replication/calibrate_statistics_v2.py")
cal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cal
SPEC.loader.exec_module(cal)


def oracle_transform(x, strata):
    x = np.asarray(x, dtype=float)
    result = x.copy()
    for label in set(strata):
        rows = np.flatnonzero(np.array(strata) == label)
        center = np.mean(x[rows], axis=0)
        result[rows] = center + (x[rows] - center) * np.sqrt(len(rows) / (len(rows) - 1))
    return result


def test_exact_unequal_stratum_covariance_and_no_global_factor():
    strata = ["a"] * 2 + ["b"] * 3
    x = np.array([[1, 2], [7, 5], [-4, 1], [3, -8], [9, 4]], dtype=float)
    y = v2.corrected_values(x, strata)
    np.testing.assert_allclose(y, oracle_transform(x, strata), atol=1e-14, rtol=0)
    samples = np.array([left + right for left in product(range(2), repeat=2)
                        for right in product(range(2, 5), repeat=3)])
    draws = y[samples].mean(axis=1)
    expected = (2 * np.cov(x[:2], rowvar=False, ddof=1)
                + 3 * np.cov(x[2:], rowvar=False, ddof=1)) / 25
    np.testing.assert_allclose(np.cov(draws, rowvar=False, ddof=0), expected,
                               atol=4e-14, rtol=0)
    uncorrected = np.cov(x[samples].mean(axis=1), rowvar=False, ddof=0)
    assert not np.allclose(uncorrected, expected)
    assert not np.allclose(uncorrected * 5 / 3, expected)
    for rows in (slice(0, 2), slice(2, 5)):
        np.testing.assert_allclose(y[rows].mean(axis=0), x[rows].mean(axis=0), atol=1e-14)


def test_affine_count_context_is_transformed_index_mean_with_same_pairs():
    ids = [f"b{i}" for i in range(12)]
    strata = ["a"] * 2 + ["b"] * 4 + ["c"] * 6
    context = v2.make_inference_context(ids, strata, draws=256)
    x = np.column_stack((np.sin(np.arange(12)), np.arange(12) ** 2 / 30))
    y = oracle_transform(x, strata)
    np.testing.assert_allclose(context.weights @ x, y[context.indices].mean(axis=1),
                               atol=2e-15, rtol=0)
    np.testing.assert_allclose(context.weights @ np.ones(12), 1, atol=5e-16, rtol=0)
    np.testing.assert_allclose(v2.corrected_values(x + 2, strata), y + 2, atol=1e-15)
    np.testing.assert_allclose(v2.corrected_values(x[:, 0] - x[:, 1], strata),
                               y[:, 0] - y[:, 1], atol=2e-15)
    assert np.any(context.weights < 0)
    original = v1.make_inference_context(ids, strata, draws=256)
    np.testing.assert_array_equal(context.indices, original.indices)
    np.testing.assert_array_equal(context.assignments, original.assignments)
    np.testing.assert_array_equal(context.coefficients, original.coefficients)
    for rows in (slice(0, 2), slice(2, 6), slice(6, 12)):
        np.testing.assert_allclose(context.weights[:, rows].sum(axis=1),
                                   (rows.stop - rows.start) / 12, atol=3e-16, rtol=0)


def fixture():
    ids = [f"b{i}" for i in range(12)]
    strata = ["a"] * 2 + ["b"] * 4 + ["c"] * 6
    z = np.sin(np.arange(12))
    latent = replace(cal.deterministic_latents(12, noise_scale=.06),
        native=.04 * z, bridge=.02 * z,
        health_delta=.03 * z[:, None, None] * np.ones((1, 1, 24)))
    raw, _ = cal.make_payload(ids, strata, latent, delta=.15, null=False)
    return reduce_score_payload(raw), v2.make_inference_context(ids, strata, draws=1024)


def test_all_production_scalar_paths_and_mde_use_corrected_samples():
    reduced, context = fixture()
    result = v2.analyze(reduced, context=context)
    original = v1.analyze(reduced, context=v1.make_inference_context(
        reduced["block_ids"], reduced["strata"], draws=1024))
    assert result["matching_means"] == original["matching_means"]
    assert result["block_endpoints"] == original["block_endpoints"]
    for name in ("control_means", "maximum_control", "exact_add_one_one_sided_p", "target_exceeds_all"):
        assert result["pseudo_phase"][name] == original["pseudo_phase"][name]
    pairs = [(result["instrument"]["native"][key], values) for key, values in reduced["native"].items()]
    pairs += [(result["instrument"]["sites"][key], values) for key, values in reduced["phase_blind"].items()]
    pairs += [(result["instrument"]["protected_logp_delta"], reduced["health"]["logp_delta"])]
    for arm, matrix in reduced["matrices"].items():
        diagonal = np.trace(matrix, axis1=1, axis2=2)
        psi = diagonal / 3 - (matrix.sum(axis=(1, 2)) - diagonal) / 6
        pairs.append((result["arms"][arm]["psi"], psi))
    for summary, x in pairs:
        draws = oracle_transform(x, reduced["strata"])[context.indices].mean(axis=1)
        np.testing.assert_allclose(summary["ci95"], np.quantile(draws, [.025, .975]), atol=2e-15, rtol=0)
        assert summary["mean"] == pytest.approx(np.mean(x), abs=1e-15)
    psi = v1.matching_endpoints(reduced["matrices"]["cc_max"])["psi"]
    centered = psi.copy()
    for label in set(reduced["strata"]):
        take = np.array(reduced["strata"]) == label
        centered[take] -= psi[take].mean()
    expected_mde = v1.MDE_FACTOR * np.sqrt(sum(
        np.sum(centered[np.array(reduced["strata"]) == label] ** 2)
        * reduced["strata"].count(label) / (reduced["strata"].count(label) - 1)
        for label in set(reduced["strata"]))) / 12
    assert result["primary_psi"]["analytic_empirical_mde80"] == pytest.approx(expected_mde, abs=1e-15)
    assert result["primary_psi"]["uncorrected_analytic_empirical_mde80"] == original["primary_psi"]["analytic_empirical_mde80"]
    assert result["primary_psi"]["mde80"] == pytest.approx(v1.MDE_FACTOR *
        oracle_transform(centered, reduced["strata"])[context.indices].mean(axis=1).std(ddof=1), abs=1e-15)
    # Pseudo assignments depend on block ID: form the paired endpoint before scaling.
    matrices = reduced["matrices"]["cc_max"]
    per_pseudo = np.sum(context.coefficients * matrices[None], axis=(2, 3))
    paired = psi - per_pseudo.mean(axis=0)
    expected = np.quantile(oracle_transform(paired, reduced["strata"])[context.indices].mean(axis=1), .05)
    assert result["pseudo_phase"]["target_minus_mean_control"]["lower95"] == pytest.approx(expected, abs=1e-15)
    assert result["instrument"]["native"]["cc_max"]["ci95"] != original["instrument"]["native"]["cc_max"]["ci95"]
    assert result["instrument"]["protected_logp_delta"]["ci95"] != original["instrument"]["protected_logp_delta"]["ci95"]
    with pytest.raises(ValueError, match="corrected context"):
        v2.analyze(reduced, context=v1.make_inference_context(reduced["block_ids"], reduced["strata"], draws=32))


@pytest.mark.parametrize("strata", [[], ["a"], ["a", "a", "b"]])
def test_singleton_or_empty_strata_rejected(strata):
    with pytest.raises(ValueError, match="at least two"):
        v2.residual_scales(strata)


def test_version_boundary_rejects_before_reading_other_fields(tmp_path):
    path = tmp_path / "freeze.json"
    path.write_text('{"status":"FROZEN_V2"}')
    with pytest.raises(ValueError, match="frozen package"):
        verify_v1(path, tmp_path)
    for status in ("FROZEN", "FROZEN_V2"):
        path.write_text('{"status":"' + status + '"}')
        with pytest.raises(ValueError, match="separately frozen"):
            verify_v2(path, tmp_path)


def test_every_existing_deterministic_memorial_on_corrected_raw_path(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh stream opened during mechanics")
    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    result = cal.run_smoke(draws=128)
    assert result["status"] == "SMOKE_PASS", result["failures"]
    assert not result["heldout_stream_opened"] and not result["development_stream_opened"]
