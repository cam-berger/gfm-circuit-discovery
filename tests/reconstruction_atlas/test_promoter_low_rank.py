from __future__ import annotations

import torch

from gfmcd.reconstruction_atlas.promoter_low_rank import (
    decode_float32_basis,
    encode_float32_basis,
    fit_uncentered_difference_basis,
    random_orthonormal_basis,
    select_low_rank_candidate,
)


def test_difference_basis_recovers_low_rank_span() -> None:
    target = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
        ]
    )
    control = torch.zeros_like(target)
    interleaved = torch.stack((target, control), dim=1).reshape(-1, 4)
    basis, diagnostics = fit_uncentered_difference_basis(interleaved, maximum_rank=3)
    projector = basis[:, :2] @ basis[:, :2].T
    assert torch.allclose(projector, torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0])))
    assert diagnostics["cumulative_difference_energy"][1] == 1.0
    assert diagnostics["singular_values"][2] == 0.0


def test_basis_json_encoding_round_trips_exact_float32_values() -> None:
    basis = random_orthonormal_basis(hidden_size=12, rank=3, seed=7)
    record = encode_float32_basis(basis)
    decoded = decode_float32_basis(record)
    assert torch.equal(decoded, basis)
    assert record["shape"] == [12, 3]


def test_random_basis_is_deterministic_and_orthonormal() -> None:
    first = random_orthonormal_basis(hidden_size=10, rank=4, seed=17)
    second = random_orthonormal_basis(hidden_size=10, rank=4, seed=17)
    different = random_orthonormal_basis(hidden_size=10, rank=4, seed=18)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert torch.allclose(first.T @ first, torch.eye(4), atol=1e-6)


def test_rank_selection_uses_smallest_rank_clearing_all_gates() -> None:
    full = {"class_margin_loss": 1.0}
    summaries = {
        1: {
            "class_margin_loss": 0.9,
            "class_margin_loss_CI95": [0.1, 1.2],
            "accuracy_loss": 0.01,
        },
        2: {
            "class_margin_loss": 0.85,
            "class_margin_loss_CI95": [0.1, 1.1],
            "accuracy_loss": 0.05,
        },
        4: {
            "class_margin_loss": 0.95,
            "class_margin_loss_CI95": [0.2, 1.2],
            "accuracy_loss": 0.07,
        },
    }
    selected, checks = select_low_rank_candidate(
        summaries,
        full_summary=full,
        maximum_rank=4,
        minimum_class_margin_loss=0.5,
        minimum_capture_fraction=0.8,
        minimum_accuracy_loss=0.02,
    )
    assert selected == 2
    assert not checks[1]["qualified_before_random_null"]
    assert checks[2]["qualified_before_random_null"]
