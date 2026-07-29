"""Paired multi-view sparse transcoders for checkpoint families.

The module is deliberately model-agnostic.  A *view* is one checkpoint's
representation of the same ordered examples.  Each view gets its own detector
and writer because independently trained checkpoints need not share native
coordinates, while a consistency loss can give the fine-tuned views a common
feature index.

The intended use is to approximate a frozen MLP map

``MLP input -> sparse features -> MLP write``

without using labels or a downstream classifier during fitting.  Causal
faithfulness remains a separate evaluation on the frozen native model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MultiViewTranscoderConfig:
    """Architecture shared by every checkpoint view."""

    input_dim: int
    output_dim: int
    num_features: int
    views: tuple[str, ...]
    aligned_views: tuple[str, ...]
    threshold_init: float = 0.1
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if min(self.input_dim, self.output_dim, self.num_features) < 1:
            raise ValueError("input, output, and feature dimensions must be positive")
        if not self.views or len(set(self.views)) != len(self.views):
            raise ValueError("views must be a non-empty unique sequence")
        if not self.aligned_views or len(set(self.aligned_views)) != len(
            self.aligned_views
        ):
            raise ValueError("aligned_views must be a non-empty unique sequence")
        if not set(self.aligned_views).issubset(self.views):
            raise ValueError("aligned_views must be drawn from views")
        if self.threshold_init <= 0.0 or self.eps <= 0.0:
            raise ValueError("threshold_init and eps must be positive")


def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64)
    return float(torch.log(torch.expm1(tensor)))


class MultiViewSparseTranscoder(nn.Module):
    """Checkpoint-specific detectors/writers with a shared sparse feature index.

    Decoder columns are normalized during every forward pass.  This removes the
    scale degeneracy in which encoder activations shrink while decoder weights
    grow, making activation sparsity and cross-view consistency comparable.
    """

    def __init__(self, config: MultiViewTranscoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleDict(
            {
                view: nn.Linear(config.input_dim, config.num_features)
                for view in config.views
            }
        )
        self.decoder_raw = nn.ParameterDict(
            {
                view: nn.Parameter(
                    torch.empty(config.output_dim, config.num_features)
                )
                for view in config.views
            }
        )
        self.output_bias = nn.ParameterDict(
            {
                view: nn.Parameter(torch.zeros(config.output_dim))
                for view in config.views
            }
        )
        initial_log_threshold = _inverse_softplus(config.threshold_init)
        self.log_threshold = nn.ParameterDict(
            {
                view: nn.Parameter(
                    torch.full((config.num_features,), initial_log_threshold)
                )
                for view in config.views
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use ordinary linear initialization and unit-scale writer seeds."""

        initial_log_threshold = _inverse_softplus(self.config.threshold_init)
        for view in self.config.views:
            self.encoders[view].reset_parameters()
            nn.init.normal_(
                self.decoder_raw[view],
                mean=0.0,
                std=self.config.output_dim**-0.5,
            )
            nn.init.zeros_(self.output_bias[view])
            nn.init.constant_(self.log_threshold[view], initial_log_threshold)

    def _validate_view(self, view: str) -> None:
        if view not in self.config.views:
            raise KeyError(f"unknown view {view!r}")

    def decoder(self, view: str) -> torch.Tensor:
        """Return the ``[output_dim, features]`` unit-column-norm writer."""

        self._validate_view(view)
        return F.normalize(
            self.decoder_raw[view],
            p=2.0,
            dim=0,
            eps=self.config.eps,
        )

    def thresholds(self, view: str) -> torch.Tensor:
        """Return positive learned activation thresholds for one view."""

        self._validate_view(view)
        return F.softplus(self.log_threshold[view])

    def encode(self, view: str, inputs: torch.Tensor) -> torch.Tensor:
        """Detect non-negative sparse features from one checkpoint's MLP input."""

        self._validate_view(view)
        if inputs.ndim != 2 or inputs.shape[1] != self.config.input_dim:
            raise ValueError("inputs must be [batch, input_dim]")
        preactivation = self.encoders[view](inputs)
        return F.relu(preactivation - self.thresholds(view))

    def decode(
        self,
        view: str,
        features: torch.Tensor,
        *,
        include_bias: bool = True,
    ) -> torch.Tensor:
        """Decode sparse features into one checkpoint's MLP-write coordinates."""

        self._validate_view(view)
        if features.ndim != 2 or features.shape[1] != self.config.num_features:
            raise ValueError("features must be [batch, num_features]")
        output = features @ self.decoder(view).T
        if include_bias:
            output = output + self.output_bias[view]
        return output

    def forward(
        self,
        view: str,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(features, reconstructed_write)`` for one view."""

        features = self.encode(view, inputs)
        return features, self.decode(view, features)

    def selected_feature_write(
        self,
        view: str,
        features: torch.Tensor,
        selected: Sequence[int],
    ) -> torch.Tensor:
        """Return only the additive write from selected feature coordinates."""

        if features.ndim != 2 or features.shape[1] != self.config.num_features:
            raise ValueError("features must be [batch, num_features]")
        indices = torch.as_tensor(
            tuple(int(index) for index in selected),
            dtype=torch.long,
            device=features.device,
        )
        if indices.ndim != 1 or indices.numel() != torch.unique(indices).numel():
            raise ValueError("selected feature indices must be unique")
        if indices.numel() == 0:
            return torch.zeros(
                (features.shape[0], self.config.output_dim),
                dtype=features.dtype,
                device=features.device,
            )
        if torch.any(indices < 0) or torch.any(indices >= self.config.num_features):
            raise ValueError("selected feature index outside dictionary")
        return features[:, indices] @ self.decoder(view)[:, indices].T


def paired_difference_mse(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    pair_indices: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Variance-normalized reconstruction loss on ordered paired differences."""

    if reconstructed.shape != target.shape or reconstructed.ndim != 2:
        raise ValueError("reconstructed and target must be same-shaped matrices")
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pair_indices must be [pairs, 2]")
    indices = pair_indices.to(device=target.device, dtype=torch.long)
    if indices.numel() == 0:
        return target.new_zeros(())
    if torch.any(indices < 0) or torch.any(indices >= target.shape[0]):
        raise ValueError("pair index outside batch")
    predicted_delta = reconstructed[indices[:, 0]] - reconstructed[indices[:, 1]]
    target_delta = target[indices[:, 0]] - target[indices[:, 1]]
    scale = target_delta.detach().square().mean().clamp_min(eps)
    return (predicted_delta - target_delta).square().mean() / scale


def standardized_consistency_loss(
    features: Mapping[str, torch.Tensor],
    views: Sequence[str],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize disagreement of paired feature activations across named views."""

    names = tuple(views)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("consistency requires at least two unique views")
    missing = set(names) - set(features)
    if missing:
        raise KeyError(f"missing feature views: {sorted(missing)}")
    tensors = [features[name] for name in names]
    if any(tensor.ndim != 2 or tensor.shape != tensors[0].shape for tensor in tensors):
        raise ValueError("paired feature matrices must have the same 2D shape")
    stacked = torch.stack(tensors, dim=0)
    consensus = stacked.mean(dim=0)
    scale = consensus.detach().square().mean().clamp_min(eps)
    return (stacked - consensus).square().mean() / scale


def view_difference_mse(
    reconstructed: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    contrast_pairs: Sequence[tuple[str, str]],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Prioritize checkpoint shifts on identical ordered examples.

    Each pair is ``(reference, changed)``.  Native coordinates must be directly
    comparable for this loss; otherwise callers should transform both outputs
    into a common, outcome-blind coordinate system before invoking it.
    """

    pairs = tuple((str(left), str(right)) for left, right in contrast_pairs)
    if not pairs or any(left == right for left, right in pairs):
        raise ValueError("contrast_pairs must contain distinct view pairs")
    required = {name for pair in pairs for name in pair}
    missing = required - (set(reconstructed) & set(targets))
    if missing:
        raise KeyError(f"missing contrast views: {sorted(missing)}")
    terms = []
    for reference, changed in pairs:
        reference_prediction = reconstructed[reference]
        changed_prediction = reconstructed[changed]
        reference_target = targets[reference]
        changed_target = targets[changed]
        if not (
            reference_prediction.shape
            == changed_prediction.shape
            == reference_target.shape
            == changed_target.shape
        ):
            raise ValueError("contrasted views must have matching matrix shapes")
        predicted_delta = changed_prediction - reference_prediction
        target_delta = changed_target - reference_target
        scale = target_delta.detach().square().mean().clamp_min(eps)
        terms.append((predicted_delta - target_delta).square().mean() / scale)
    return torch.stack(terms).mean()


def decoder_alignment_loss(
    model: MultiViewSparseTranscoder,
    rotations_to_reference: Mapping[str, torch.Tensor],
    *,
    reference_view: str,
) -> torch.Tensor:
    """Align same-index writers after supplied outcome-blind rotations.

    A rotation maps row vectors from a source view into the reference view:
    ``row_source @ rotation``.  Decoder columns therefore transform as
    ``rotation.T @ decoder_source``.
    """

    names = model.config.aligned_views
    if reference_view not in names:
        raise ValueError("reference_view must be one of aligned_views")
    missing = set(names) - set(rotations_to_reference)
    if missing:
        raise KeyError(f"missing rotations: {sorted(missing)}")
    reference = model.decoder(reference_view)
    losses = []
    for view in names:
        rotation = rotations_to_reference[view].to(
            device=reference.device,
            dtype=reference.dtype,
        )
        expected = (model.config.output_dim, model.config.output_dim)
        if rotation.shape != expected:
            raise ValueError(f"rotation for {view!r} must have shape {expected}")
        transported = F.normalize(
            rotation.T @ model.decoder(view),
            p=2.0,
            dim=0,
            eps=model.config.eps,
        )
        losses.append(1.0 - (transported * reference).sum(dim=0))
    return torch.stack(losses).mean()


def multiview_transcoder_loss(
    model: MultiViewSparseTranscoder,
    inputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    pair_indices: torch.Tensor | None = None,
    contrast_pairs: Sequence[tuple[str, str]] | None = None,
    reconstruction_weight: float = 1.0,
    difference_weight: float = 1.0,
    view_difference_weight: float = 0.0,
    sparsity_weight: float = 1e-3,
    consistency_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute label-free reconstruction, delta, and sparsity losses."""

    expected = set(model.config.views)
    if set(inputs) != expected or set(targets) != expected:
        raise ValueError("input and target mappings must contain every configured view")
    if min(
        reconstruction_weight,
        difference_weight,
        view_difference_weight,
        sparsity_weight,
        consistency_weight,
    ) < 0.0:
        raise ValueError("loss weights must be non-negative")

    features: dict[str, torch.Tensor] = {}
    reconstructed: dict[str, torch.Tensor] = {}
    reconstruction_terms = []
    difference_terms = []
    sparsity_terms = []
    for view in model.config.views:
        target = targets[view]
        feature_values, prediction = model(view, inputs[view])
        if target.shape != prediction.shape:
            raise ValueError("target must match reconstructed output shape")
        features[view] = feature_values
        reconstructed[view] = prediction
        centered = target.detach() - target.detach().mean(dim=0, keepdim=True)
        scale = centered.square().mean().clamp_min(model.config.eps)
        reconstruction_terms.append((prediction - target).square().mean() / scale)
        sparsity_terms.append(feature_values.abs().mean())
        if pair_indices is not None:
            difference_terms.append(
                paired_difference_mse(
                    prediction,
                    target,
                    pair_indices,
                    eps=model.config.eps,
                )
            )

    reconstruction = torch.stack(reconstruction_terms).mean()
    difference = (
        torch.stack(difference_terms).mean()
        if difference_terms
        else reconstruction.new_zeros(())
    )
    sparsity = torch.stack(sparsity_terms).mean()
    consistency = (
        standardized_consistency_loss(
            features,
            model.config.aligned_views,
            eps=model.config.eps,
        )
        if consistency_weight > 0.0
        else reconstruction.new_zeros(())
    )
    view_difference = (
        view_difference_mse(
            reconstructed,
            targets,
            contrast_pairs,
            eps=model.config.eps,
        )
        if view_difference_weight > 0.0 and contrast_pairs is not None
        else reconstruction.new_zeros(())
    )
    if view_difference_weight > 0.0 and contrast_pairs is None:
        raise ValueError("positive view_difference_weight requires contrast_pairs")
    total = (
        reconstruction_weight * reconstruction
        + difference_weight * difference
        + view_difference_weight * view_difference
        + sparsity_weight * sparsity
        + consistency_weight * consistency
    )
    return {
        "loss": total,
        "reconstruction": reconstruction,
        "paired_difference": difference,
        "view_difference": view_difference,
        "sparsity": sparsity,
        "consistency": consistency,
        "features": features,
        "reconstructed": reconstructed,
    }


def feature_similarity_matrices(
    left_features: torch.Tensor,
    right_features: torch.Tensor,
    left_decoder: torch.Tensor,
    right_decoder: torch.Tensor,
    *,
    activation_weight: float = 0.5,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compare independently indexed dictionaries by activation and writer geometry."""

    if (
        left_features.ndim != 2
        or right_features.ndim != 2
        or left_features.shape[0] != right_features.shape[0]
    ):
        raise ValueError("feature matrices must be 2D with paired rows")
    if (
        left_decoder.ndim != 2
        or right_decoder.ndim != 2
        or left_decoder.shape[0] != right_decoder.shape[0]
        or left_decoder.shape[1] != left_features.shape[1]
        or right_decoder.shape[1] != right_features.shape[1]
    ):
        raise ValueError("decoder columns must match feature dimensions")
    if not 0.0 <= activation_weight <= 1.0:
        raise ValueError("activation_weight must be in [0, 1]")

    left_centered = left_features - left_features.mean(dim=0, keepdim=True)
    right_centered = right_features - right_features.mean(dim=0, keepdim=True)
    activation_numerator = left_centered.T @ right_centered
    left_scale = left_centered.square().sum(dim=0).sqrt().clamp_min(eps)
    right_scale = right_centered.square().sum(dim=0).sqrt().clamp_min(eps)
    activation_correlation = activation_numerator / (
        left_scale[:, None] * right_scale[None, :]
    )

    left_directions = F.normalize(left_decoder, dim=0, eps=eps)
    right_directions = F.normalize(right_decoder, dim=0, eps=eps)
    decoder_cosine = left_directions.T @ right_directions
    combined = (
        activation_weight * activation_correlation.clamp(0.0, 1.0)
        + (1.0 - activation_weight) * decoder_cosine.clamp(0.0, 1.0)
    )
    return {
        "activation_correlation": activation_correlation,
        "decoder_cosine": decoder_cosine,
        "combined": combined,
    }


__all__ = [
    "MultiViewSparseTranscoder",
    "MultiViewTranscoderConfig",
    "decoder_alignment_loss",
    "feature_similarity_matrices",
    "multiview_transcoder_loss",
    "paired_difference_mse",
    "standardized_consistency_loss",
    "view_difference_mse",
]
