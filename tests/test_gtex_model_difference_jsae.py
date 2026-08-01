from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from gfmcd.circuits.model_difference_jsae import (
    ModelDifferenceJSAEConfig,
    ModelDifferenceJacobianSAE,
)


REPO = Path(__file__).resolve().parents[1]


def _load_runner():
    path = REPO / "scripts/splicebert_gtex_finetune_shift/run_gtex_model_difference_jsae.py"
    spec = importlib.util.spec_from_file_location("model_difference_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _states() -> list[dict[str, str]]:
    rows = []
    variants = {
        "clean": "clean",
        "target_shuffle": "target_shuffle",
        "sham_shuffle": "sham_shuffle",
        "max_recode": "max_recode_draw0",
        "synonymous_single": "synonymous_single",
    }
    for endpoint in RUNNER.ENDPOINTS:
        for family, variant in variants.items():
            del family
            for gene in ("g0", "g1"):
                rows.append(
                    {
                        "endpoint": endpoint,
                        "variant_kind": variant,
                        "gene_id": gene,
                    }
                )
    return rows


def _weights(
    input_dim: int, output_dim: int, mlp_dim: int, seed: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "input_weight": torch.randn(
            mlp_dim, input_dim, generator=generator
        ),
        "input_bias": torch.randn(mlp_dim, generator=generator),
        "output_weight": torch.randn(
            output_dim, mlp_dim, generator=generator
        ),
        "output_bias": torch.randn(output_dim, generator=generator),
    }


def test_hierarchical_state_schedule_balances_all_cells() -> None:
    states = _states()
    schedule, report = RUNNER.hierarchical_state_schedule(
        states,
        np.arange(len(states), dtype=np.int64),
        draws=100,
        seed=7,
    )
    assert schedule.shape == (100,)
    assert report["PASS"]
    assert set(report["counts_by_stratum"].values()) == {10}


def test_hierarchical_fsee_is_zero_anchored() -> None:
    states = _states()
    indices = np.arange(len(states), dtype=np.int64)
    target = torch.ones(len(states), 3)
    assert RUNNER.hierarchical_fsee(
        target, torch.zeros_like(target), states, indices
    ) == 0.0
    assert RUNNER.hierarchical_fsee(
        target, target.clone(), states, indices
    ) == 1.0


def test_hierarchical_weights_equalize_cells_and_genes() -> None:
    states = _states()
    # Add many synonymous rows to one gene; its whole gene must retain the same
    # weight as the other gene inside the same cell.
    for _ in range(20):
        states.append(
            {
                "endpoint": "donor",
                "variant_kind": "synonymous_single",
                "gene_id": "g0",
            }
        )
    indices = np.arange(len(states), dtype=np.int64)
    weights = RUNNER.hierarchical_row_weights(states, indices)
    assert np.isclose(weights.sum(), 1.0)
    cells = {}
    genes = {}
    for index, row in enumerate(states):
        cell = (
            row["endpoint"],
            RUNNER.state_family(row["variant_kind"]),
        )
        cells[cell] = cells.get(cell, 0.0) + weights[index]
        genes[(cell, row["gene_id"])] = (
            genes.get((cell, row["gene_id"]), 0.0) + weights[index]
        )
    assert np.allclose(list(cells.values()), 0.1)
    assert np.allclose(list(genes.values()), 0.05)


def test_weighted_distribution_uses_hierarchical_weights() -> None:
    report = RUNNER.weighted_distribution(
        [0.0, 10.0, 10.0], np.asarray([0.8, 0.1, 0.1])
    )
    assert report["mean"] == 2.0
    assert report["median"] == 0.0


def test_exact_four_cell_targets_close_and_replay() -> None:
    base_weights = _weights(3, 2, 5, 11)
    tuned_weights = _weights(3, 2, 5, 13)
    base_inputs = torch.randn(9, 3)
    tuned_inputs = base_inputs + torch.randn(9, 3) / 5
    base_writes = RUNNER.run_mlp(
        base_inputs, base_weights, device=torch.device("cpu")
    )
    tuned_writes = RUNNER.run_mlp(
        tuned_inputs, tuned_weights, device=torch.device("cpu")
    )
    targets, report = RUNNER.build_difference_targets(
        base_inputs,
        tuned_inputs,
        base_writes,
        tuned_writes,
        base_weights,
        tuned_weights,
        device=torch.device("cpu"),
    )
    assert report["PASS"]
    assert report[
        "write_shapley_closure_maximum_absolute_error_float64"
    ] <= 1e-10
    assert torch.allclose(
        targets["state"] + targets["parameter"],
        targets["total"],
        atol=2e-6,
    )


def test_signed_feature_matching_is_sign_invariant() -> None:
    generator = np.random.default_rng(5)
    left = generator.normal(size=(200, 6))
    left[generator.random(size=left.shape) < 0.65] = 0
    permutation = np.asarray([3, 5, 0, 2, 1, 4])
    right = -left[:, permutation]
    decoder = generator.normal(size=(8, 6))
    right_decoder = -decoder[:, permutation]
    report = RUNNER.feature_match(
        left,
        right,
        decoder,
        right_decoder,
        np.full(len(left), 1 / len(left)),
    )
    assert report["PASS"]
    assert report["reciprocal_match_fraction"] == 1.0
    assert report["reciprocal_score_mean"] > 0.99


def test_signed_feature_matching_rejects_inconsistent_signs() -> None:
    generator = np.random.default_rng(6)
    left = generator.normal(size=(300, 4))
    decoder = np.eye(4)
    report = RUNNER.feature_match(
        left,
        left.copy(),
        decoder,
        -decoder,
        np.full(len(left), 1 / len(left)),
    )
    assert not report["PASS"]
    assert report["reciprocal_score_mean"] < 0.10


def test_primary_assay_has_no_auxiliary_repair_or_full_state_condition() -> None:
    assert RUNNER.CONDITIONS == {
        "shapley_deltaJ": 0.25,
        "shapley_no_deltaJ": 0.0,
    }
    source = (
        REPO / "scripts/splicebert_gtex_finetune_shift/run_gtex_model_difference_jsae.py"
    ).read_text()
    assert "auxiliary_dead_feature_loss" not in source
    assert "shared_private" not in source


def test_runtime_aligned_planted_controls_pass() -> None:
    report = RUNNER.aligned_planted_controls()
    assert report["PASS"]
    assert report["checks"]["known_edge_location"]
    assert report["checks"]["known_edge_sign"]
    assert report["dense_control_E90"] >= 12


def test_zero_mass_graph_has_zero_effective_edge_count() -> None:
    graph = torch.zeros(3, 4, 4)
    assert torch.equal(
        RUNNER.effective_edge_count(graph), torch.zeros(3)
    )


def test_finite_difference_routes_zero_block_to_absolute_gate() -> None:
    base = _weights(4, 3, 6, 17)
    model = ModelDifferenceJacobianSAE(
        ModelDifferenceJSAEConfig(
            input_dim=4,
            output_dim=3,
            mlp_dim=6,
            input_features=7,
            output_features=5,
            active_features=2,
        ),
        base_weights=base,
        tuned_weights={key: value.clone() for key, value in base.items()},
        input_scale=torch.tensor(1.0),
        output_scale=torch.tensor(1.0),
    )
    base_inputs = torch.randn(3, 4)
    input_delta = torch.zeros_like(base_inputs)
    writes = model.write_shapley(base_inputs, input_delta)
    targets = {
        "input": input_delta,
        "state": writes["state"],
        "parameter": writes["parameter"],
        "total": writes["total"],
    }
    report = RUNNER.finite_difference_control(
        model,
        base_inputs,
        targets,
        np.arange(3, dtype=np.int64),
        examples=3,
    )
    assert report["PASS"]
    assert all(
        not block["resolved"] and block["PASS"]
        for block in report["blocks"].values()
    )
