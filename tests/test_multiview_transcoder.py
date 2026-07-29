import pytest
import torch

from gfmcd.circuits.multiview_transcoder import (
    MultiViewSparseTranscoder,
    MultiViewTranscoderConfig,
    decoder_alignment_loss,
    feature_similarity_matrices,
    multiview_transcoder_loss,
    paired_difference_mse,
    standardized_consistency_loss,
    view_difference_mse,
)


def make_model() -> MultiViewSparseTranscoder:
    torch.manual_seed(2)
    return MultiViewSparseTranscoder(
        MultiViewTranscoderConfig(
            input_dim=3,
            output_dim=4,
            num_features=5,
            views=("base", "2020", "2021"),
            aligned_views=("2020", "2021"),
        )
    )


def test_decoder_columns_are_normalized_and_selected_writes_sum():
    model = make_model()
    inputs = torch.randn(6, 3)
    features, reconstruction = model("2020", inputs)
    decoder = model.decoder("2020")
    assert torch.allclose(
        torch.linalg.vector_norm(decoder, dim=0),
        torch.ones(5),
        atol=1e-6,
    )
    parts = [
        model.selected_feature_write("2020", features, [index])
        for index in range(5)
    ]
    assert torch.allclose(
        torch.stack(parts).sum(dim=0) + model.output_bias["2020"],
        reconstruction,
        atol=1e-6,
    )
    assert torch.equal(
        model.selected_feature_write("2020", features, []),
        torch.zeros(6, 4),
    )


def test_paired_difference_loss_recovers_ordered_deltas():
    target = torch.tensor([[3.0, 1.0], [1.0, 0.0], [5.0, 2.0], [2.0, 2.0]])
    pairs = torch.tensor([[0, 1], [2, 3]])
    assert paired_difference_mse(target, target, pairs).item() == pytest.approx(0.0)
    shifted = target + torch.tensor([7.0, -4.0])
    assert paired_difference_mse(shifted, target, pairs).item() == pytest.approx(0.0)


def test_consistency_is_zero_for_identical_paired_features():
    features = torch.rand(7, 5)
    assert standardized_consistency_loss(
        {"2020": features, "2021": features.clone()},
        ("2020", "2021"),
    ).item() == pytest.approx(0.0)
    changed = features.clone()
    changed[:, 0] += 1.0
    assert (
        standardized_consistency_loss(
            {"2020": features, "2021": changed},
            ("2020", "2021"),
        ).item()
        > 0.0
    )


def test_view_difference_loss_ignores_a_shared_reconstruction_error():
    targets = {
        "base": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "2020": torch.tensor([[2.0, 4.0], [2.0, 8.0]]),
    }
    shared_error = torch.tensor([[5.0, -3.0], [5.0, -3.0]])
    reconstructed = {
        view: target + shared_error for view, target in targets.items()
    }
    assert view_difference_mse(
        reconstructed,
        targets,
        [("base", "2020")],
    ).item() == pytest.approx(0.0)


def test_identity_rotations_give_zero_decoder_alignment_for_equal_writers():
    model = make_model()
    with torch.no_grad():
        model.decoder_raw["2021"].copy_(model.decoder_raw["2020"])
    rotations = {"2020": torch.eye(4), "2021": torch.eye(4)}
    assert decoder_alignment_loss(
        model,
        rotations,
        reference_view="2020",
    ).item() == pytest.approx(0.0, abs=1e-6)


def test_multiview_loss_excludes_base_from_seed_consistency():
    model = make_model()
    inputs = {view: torch.randn(8, 3) for view in model.config.views}
    targets = {view: torch.randn(8, 4) for view in model.config.views}
    result = multiview_transcoder_loss(
        model,
        inputs,
        targets,
        pair_indices=torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
        contrast_pairs=(("base", "2020"), ("base", "2021")),
        view_difference_weight=0.5,
        consistency_weight=0.2,
    )
    assert result["loss"].ndim == 0
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
    )
    changed = dict(result["features"])
    changed["base"] = changed["base"] + 1000.0
    expected = standardized_consistency_loss(
        result["features"], model.config.aligned_views
    )
    observed = standardized_consistency_loss(changed, model.config.aligned_views)
    assert torch.equal(expected, observed)


def test_feature_similarity_recovers_matching_permutation():
    left = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [4.0, 1.0, 1.0],
        ]
    )
    permutation = torch.tensor([2, 0, 1])
    right = left[:, permutation]
    left_decoder = torch.eye(3)
    right_decoder = left_decoder[:, permutation]
    scores = feature_similarity_matrices(
        left,
        right,
        left_decoder,
        right_decoder,
    )["combined"]
    inverse = torch.argsort(permutation)
    assert torch.equal(scores.argmax(dim=1), inverse)


def test_configuration_and_shape_failures_are_explicit():
    with pytest.raises(ValueError, match="aligned_views"):
        MultiViewTranscoderConfig(
            input_dim=3,
            output_dim=4,
            num_features=5,
            views=("base", "2020"),
            aligned_views=("missing",),
        )
    model = make_model()
    with pytest.raises(ValueError, match="inputs"):
        model.encode("2020", torch.randn(2, 4))
    with pytest.raises(ValueError, match="unique"):
        model.selected_feature_write("2020", torch.rand(2, 5), [1, 1])
