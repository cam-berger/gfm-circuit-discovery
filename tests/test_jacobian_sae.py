import torch

from gfmcd.circuits.jacobian_sae import (
    JacobianSAEConfig,
    JacobianSAEPair,
    TopKSAE,
    auxiliary_dead_feature_loss,
    edge_mass_count,
    explained_variance,
    scalar_normalization,
)


def make_pair() -> JacobianSAEPair:
    torch.manual_seed(7)
    config = JacobianSAEConfig(
        input_dim=4,
        output_dim=3,
        mlp_dim=6,
        num_features=8,
        active_features=2,
    )
    values = torch.randn(20, 4)
    outputs = torch.randn(20, 3)
    input_center, input_scale = scalar_normalization(values)
    output_center, output_scale = scalar_normalization(outputs)
    return JacobianSAEPair(
        config,
        input_weight=torch.randn(6, 4) / 2,
        input_bias=torch.randn(6) / 5,
        output_weight=torch.randn(3, 6) / 2,
        output_bias=torch.randn(3) / 5,
        input_center=input_center,
        input_scale=input_scale,
        output_center=output_center,
        output_scale=output_scale,
    )


def test_topk_sae_enforces_l0_and_unit_decoder_rows():
    torch.manual_seed(1)
    sae = TopKSAE(5, 11, 3)
    reconstructed, activations, indices = sae(torch.randn(7, 5))
    assert reconstructed.shape == (7, 5)
    assert indices.shape == (7, 3)
    assert torch.all((activations > 0).sum(dim=1) <= 3)
    assert torch.allclose(
        sae.decoder().norm(dim=1),
        torch.ones(11),
        atol=1e-6,
    )


def test_pair_supports_asymmetric_input_and_output_capacities():
    torch.manual_seed(2)
    config = JacobianSAEConfig(
        input_dim=4,
        output_dim=3,
        mlp_dim=6,
        num_features=9,
        output_num_features=5,
        active_features=2,
    )
    pair = JacobianSAEPair(
        config,
        input_weight=torch.randn(6, 4),
        input_bias=torch.zeros(6),
        output_weight=torch.randn(3, 6),
        output_bias=torch.zeros(3),
        input_center=torch.zeros(4),
        input_scale=torch.ones(()),
        output_center=torch.zeros(3),
        output_scale=torch.ones(()),
    )
    run = pair(torch.randn(7, 4))
    assert run["input_activations"].shape == (7, 9)
    assert run["output_activations"].shape == (7, 5)
    assert run["active_jacobian"].shape == (7, 2, 2)


def test_analytic_active_jacobian_matches_autograd():
    pair = make_pair()
    raw_input = torch.randn(3, 4)
    run = pair(raw_input)
    analytic = run["active_jacobian"]

    observed = []
    for row in range(raw_input.shape[0]):
        input_indices = run["input_indices"][row]
        output_indices = run["output_indices"][row]
        decoder_directions = (
            pair.input_sae.decoder()[input_indices] * pair.input_scale
        )

        def selected_output_preactive(feature_delta):
            raw = raw_input[row : row + 1] + feature_delta @ decoder_directions
            output = pair.mlp(raw)
            normalized_output = pair.normalize_output(output)
            preactive = (
                normalized_output @ pair.output_sae.encoder_weight
                + pair.output_sae.encoder_bias
            )
            return preactive[0, output_indices]

        selected = torch.zeros(2, requires_grad=True)
        observed.append(
            torch.autograd.functional.jacobian(
                selected_output_preactive,
                selected,
            )
        )
    observed_tensor = torch.stack(observed)
    assert torch.allclose(analytic, observed_tensor, atol=2e-5, rtol=2e-4)


def test_loss_backpropagates_only_into_saes():
    pair = make_pair()
    result = pair.loss(torch.randn(5, 4), jacobian_coefficient=0.1)
    result["loss"].backward()
    assert pair.input_sae.encoder_weight.grad is not None
    assert pair.output_sae.encoder_weight.grad is not None
    assert pair.mlp_input_weight.grad is None
    assert pair.mlp_output_weight.grad is None


def test_auxiliary_loss_routes_gradients_to_dead_features():
    torch.manual_seed(4)
    sae = TopKSAE(5, 12, 2)
    values = torch.randn(7, 5)
    reconstruction, activations, _ = sae(values)
    dead = torch.zeros(12, dtype=torch.bool)
    dead[6:] = True
    loss = auxiliary_dead_feature_loss(
        sae,
        values,
        reconstruction,
        dead,
        auxiliary_features=4,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0
    assert torch.any(sae.encoder_weight.grad[:, dead] != 0)
    assert torch.all(activations[:, dead] == 0) or torch.any(
        sae.encoder_weight.grad[:, dead] != 0
    )


def test_metrics_report_fve_and_edge_concentration():
    target = torch.tensor([[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]])
    assert explained_variance(target, target) == 1.0
    jacobian = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    assert edge_mass_count(jacobian, fraction=0.9).tolist() == [1, 4]
