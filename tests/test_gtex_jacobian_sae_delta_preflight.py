import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO / "scripts/splicebert/run_gtex_jacobian_sae_delta_preflight.py"
)
SPEC = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def classifier_parameters():
    return {
        "layer_norm_weight": np.ones(4),
        "layer_norm_bias": np.zeros(4),
        "layer_norm_eps": 1e-12,
        "classifier_direction": np.asarray([1.0, -1.0, 0.0, 0.0]),
        "classifier_bias": 0.0,
    }


def test_decoder_difference_error_checks_all_feature_sum():
    codes = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    decoder = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    reconstructed = codes @ decoder + np.asarray([4.0, -3.0])
    error = RUNNER.decoder_difference_error(
        reconstructed,
        codes,
        decoder,
        np.asarray([[0, 1]]),
    )
    assert error == 0.0


def test_endpoint_gate_accepts_exact_decoded_deltas():
    residual = np.zeros((4, 4))
    native = np.asarray(
        [
            [2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, -2.0, 0.0, 0.0],
            [1.9, -1.9, 0.0, 0.0],
        ]
    )
    metrics = RUNNER.endpoint_metrics(
        residual,
        native,
        native + np.asarray([9.0, -2.0, 4.0, 1.0]),
        np.asarray([[0, 1]]),
        np.asarray([[2, 3]]),
        classifier_parameters(),
    )
    assert metrics["PASS"]
    assert metrics["target_shuffle"]["damage_ratio"] == 1.0
    assert metrics["target_shuffle"]["recovery_ratio"] == 1.0


def test_endpoint_gate_rejects_wrong_feature_delta():
    residual = np.zeros((4, 4))
    native = np.asarray(
        [
            [2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, -2.0, 0.0, 0.0],
            [1.9, -1.9, 0.0, 0.0],
        ]
    )
    reconstructed = native.copy()
    reconstructed[1] = reconstructed[0]
    metrics = RUNNER.endpoint_metrics(
        residual,
        native,
        reconstructed,
        np.asarray([[0, 1]]),
        np.asarray([[2, 3]]),
        classifier_parameters(),
    )
    assert not metrics["PASS"]
    assert not metrics["gates"]["damage_relative_error"]
    assert not metrics["gates"]["recovery_relative_error"]


def test_float64_zero_delta_identity_is_exact_for_large_float32_values():
    native = np.asarray([[100.0, -100.0]], dtype=np.float32).astype(np.float64)
    reconstruction = np.asarray([[63.125, -72.5]], dtype=np.float32).astype(
        np.float64
    )
    zero_delta = reconstruction - reconstruction
    assert np.max(np.abs(native + zero_delta - native)) == 0.0
