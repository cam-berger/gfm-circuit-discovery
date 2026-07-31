import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts/splicebert_gtex_finetune_shift/run_gtex_jacobian_sae_behavior_preflight.py"
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


def test_endpoint_gate_accepts_exact_reconstruction():
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
        native.copy(),
        np.asarray([[0, 1]]),
        np.asarray([[2, 3]]),
        classifier_parameters(),
    )
    assert metrics["PASS"]
    assert metrics["effect_recovery_ratio"] == 1.0


def test_endpoint_gate_rejects_target_specific_reconstruction_loss():
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
    assert not metrics["gates"]["effect_recovery_ratio"]
