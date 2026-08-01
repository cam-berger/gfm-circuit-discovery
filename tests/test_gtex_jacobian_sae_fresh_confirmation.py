import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts/splicebert_gtex_finetune_shift/run_gtex_jacobian_sae_fresh_delta_confirmation.py"
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


def test_two_block_shapley_is_complete_for_exact_feature_delta():
    residual = np.zeros((2, 4))
    native = np.asarray(
        [[2.0, -2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    result = RUNNER.two_block_shapley(
        residual,
        native,
        native.copy(),
        np.asarray([[0, 1]]),
        classifier_parameters(),
    )
    for direction in ("damage", "recovery"):
        assert result[direction]["error_shapley_effect"] == 0.0
        assert result[direction]["error_shapley_fraction"] == 0.0
        assert result[direction]["joint_dense_margin_max_abs_error"] == 0.0
        assert result[direction]["shapley_sum_max_abs_error"] == 0.0


def test_two_block_shapley_assigns_missing_delta_to_error():
    residual = np.zeros((2, 4))
    native = np.asarray(
        [[2.0, -2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    reconstructed = np.repeat(native[:1], 2, axis=0)
    result = RUNNER.two_block_shapley(
        residual,
        native,
        reconstructed,
        np.asarray([[0, 1]]),
        classifier_parameters(),
    )
    for direction in ("damage", "recovery"):
        assert result[direction]["feature_shapley_effect"] == 0.0
        assert result[direction]["error_shapley_fraction"] == 1.0
