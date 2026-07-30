from __future__ import annotations

import torch

from gfmcd.circuits.model_difference_jsae import (
    ModelDifferenceJSAEConfig,
    ModelDifferenceJacobianSAE,
    SignedTopKSAE,
    zero_anchored_scalar_scale,
)


def _weights(
    input_dim: int, output_dim: int, mlp_dim: int, seed: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "input_weight": torch.randn(
            mlp_dim, input_dim, generator=generator
        )
        / input_dim**0.5,
        "input_bias": torch.randn(mlp_dim, generator=generator) / 10,
        "output_weight": torch.randn(
            output_dim, mlp_dim, generator=generator
        )
        / mlp_dim**0.5,
        "output_bias": torch.randn(output_dim, generator=generator) / 10,
    }


def _model(*, identical: bool = False) -> ModelDifferenceJacobianSAE:
    config = ModelDifferenceJSAEConfig(
        input_dim=4,
        output_dim=3,
        mlp_dim=6,
        input_features=7,
        output_features=5,
        active_features=2,
    )
    base = _weights(4, 3, 6, 3)
    tuned = (
        {key: value.clone() for key, value in base.items()}
        if identical
        else _weights(4, 3, 6, 5)
    )
    torch.manual_seed(11)
    return ModelDifferenceJacobianSAE(
        config,
        base_weights=base,
        tuned_weights=tuned,
        input_scale=torch.tensor(0.7),
        output_scale=torch.tensor(1.3),
    )


def test_signed_topk_is_signed_bias_free_and_zero_anchored() -> None:
    sae = SignedTopKSAE(3, 5, 2)
    with torch.no_grad():
        sae.encoder_weight.copy_(
            torch.tensor(
                [
                    [2.0, -3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            )
        )
    values = torch.tensor([[1.0, 0.0, 0.0]])
    _, activations, _ = sae(values)
    assert int(torch.count_nonzero(activations)) == 2
    assert float(activations[0, 0].detach()) == 2.0
    assert float(activations[0, 1].detach()) == -3.0
    zero_hat, zero_acts, _ = sae(torch.zeros_like(values))
    assert torch.equal(zero_acts, torch.zeros_like(zero_acts))
    assert torch.equal(zero_hat, torch.zeros_like(zero_hat))
    assert not hasattr(sae, "encoder_bias")
    assert not hasattr(sae, "decoder_bias")


def test_shared_output_support_is_common_and_signed() -> None:
    sae = SignedTopKSAE(3, 4, 2)
    left = torch.tensor([[1.0, -2.0, 0.5]])
    right = torch.tensor([[-1.0, 0.5, 2.0]])
    left_acts, right_acts, indices = sae.encode_shared_support(left, right)
    assert indices.shape == (1, 2)
    assert torch.equal(
        torch.nonzero(left_acts[0], as_tuple=False).flatten().sort().values,
        torch.nonzero(right_acts[0], as_tuple=False).flatten().sort().values,
    )


def test_write_and_jacobian_shapley_close() -> None:
    model = _model()
    base_input = torch.randn(3, 4)
    input_delta = torch.randn(3, 4) / 4
    writes = model.write_shapley(base_input, input_delta)
    assert torch.allclose(
        writes["state"] + writes["parameter"],
        writes["total"],
        atol=2e-6,
        rtol=1e-6,
    )
    encoded = model.encode_difference(
        input_delta, writes["state"], writes["parameter"]
    )
    graphs = model.active_jacobian_shapley(
        base_input,
        input_delta,
        encoded["input_indices"],
        encoded["output_indices"],
    )
    assert torch.allclose(
        graphs["state"] + graphs["parameter"],
        graphs["total"],
        atol=2e-6,
        rtol=1e-6,
    )
    native_operator = model.active_operator_shapley(
        base_input, input_delta, encoded["input_indices"]
    )
    decoded_operator = model.decoded_operator_shapley(
        graphs, encoded["output_indices"]
    )
    assert native_operator["total"].shape == (3, 3, 2)
    assert decoded_operator["total"].shape == (3, 3, 2)
    assert torch.allclose(
        native_operator["state"] + native_operator["parameter"],
        native_operator["total"],
        atol=2e-6,
        rtol=1e-6,
    )


def test_identical_model_and_state_is_exact_difference_negative() -> None:
    model = _model(identical=True)
    base_input = torch.randn(4, 4)
    input_delta = torch.zeros_like(base_input)
    writes = model.write_shapley(base_input, input_delta)
    for key in ("state", "parameter", "total", "interaction"):
        assert torch.equal(writes[key], torch.zeros_like(writes[key]))
    encoded = model.encode_difference(
        input_delta, writes["state"], writes["parameter"]
    )
    assert torch.equal(
        encoded["reconstructed_input"],
        torch.zeros_like(encoded["reconstructed_input"]),
    )
    assert torch.equal(
        encoded["reconstructed_total"],
        torch.zeros_like(encoded["reconstructed_total"]),
    )
    graphs = model.active_jacobian_shapley(
        base_input,
        input_delta,
        encoded["input_indices"],
        encoded["output_indices"],
    )
    assert torch.equal(
        graphs["total"], torch.zeros_like(graphs["total"])
    )
    operator = model.active_operator_shapley(
        base_input, input_delta, encoded["input_indices"]
    )
    assert torch.equal(
        operator["total"], torch.zeros_like(operator["total"])
    )


def test_each_analytic_cell_matches_autograd_and_finite_difference() -> None:
    model = _model()
    model.eval()
    base_input = torch.randn(1, 4)
    input_delta = torch.randn(1, 4) / 5
    writes = model.write_shapley(base_input, input_delta)
    encoded = model.encode_difference(
        input_delta, writes["state"], writes["parameter"]
    )
    input_indices = encoded["input_indices"]
    output_indices = encoded["output_indices"]
    cells = model.active_jacobian_cells(
        base_input, input_delta, input_indices, output_indices
    )
    directions = (
        model.input_sae.decoder()[input_indices[0]] * model.input_scale
    )
    for state, raw_input in (
        ("0", base_input),
        ("1", base_input + input_delta),
    ):
        for view, model_name in (("0", "base"), ("1", "tuned")):
            key = state + view

            def output_features(coefficients: torch.Tensor) -> torch.Tensor:
                perturbed = raw_input + coefficients @ directions
                output = model.mlp(perturbed, view=model_name)
                assert isinstance(output, torch.Tensor)
                normalized = output / model.output_scale
                features = normalized @ model.output_sae.encoder_weight
                return features[0, output_indices[0]]

            coefficients = torch.zeros(
                model.config.active_features, requires_grad=True
            )
            observed = torch.autograd.functional.jacobian(
                output_features, coefficients
            )
            assert torch.allclose(
                cells[key][0], observed, atol=2e-5, rtol=2e-5
            )
            dose = 1e-3
            finite = torch.stack(
                [
                    (
                        output_features(
                            torch.nn.functional.one_hot(
                                torch.tensor(index),
                                model.config.active_features,
                            ).float()
                            * dose
                        )
                        - output_features(
                            torch.zeros(model.config.active_features)
                        )
                    )
                    / dose
                    for index in range(model.config.active_features)
                ],
                dim=1,
            )
            assert torch.allclose(
                cells[key][0], finite, atol=2e-3, rtol=2e-3
            )


def test_only_sparse_dictionary_parameters_receive_gradients() -> None:
    model = _model()
    base_input = torch.randn(5, 4)
    input_delta = torch.randn(5, 4) / 5
    run = model(base_input, input_delta, include_jacobian=True)
    loss = (
        (run["reconstructed_input"] - run["normalized_input"])
        .square()
        .mean()
        + (run["reconstructed_total"] - run["normalized_total"])
        .square()
        .mean()
        + run["jacobian_total"].abs().mean()
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(not buffer.requires_grad for buffer in model.buffers())


def test_zero_anchored_scale_keeps_mean_shift() -> None:
    values = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    scale = zero_anchored_scalar_scale(values)
    assert float(scale) == 1.0
