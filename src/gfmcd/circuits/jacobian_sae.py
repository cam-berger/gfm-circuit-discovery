"""Jacobian sparse autoencoders for frozen, non-gated MLPs.

The implementation follows the active-support construction of Farnik et al.
(2025): separate TopK SAEs encode an MLP's input and output, while a loss on the
feature-to-feature Jacobian encourages a sparse computational graph.  Only the
active ``k_out x k_in`` block is materialized.

This module is intentionally independent of TransformerLens and SAE Lens.  It
supports the exact two-linear-layer GELU MLP used by SpliceBERT and keeps
reconstruction, graph discovery, and downstream causal evaluation separate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class JacobianSAEConfig:
    """Architecture of a paired input/output TopK SAE."""

    input_dim: int
    output_dim: int
    mlp_dim: int
    num_features: int
    active_features: int
    eps: float = 1e-8

    def __post_init__(self) -> None:
        dimensions = (
            self.input_dim,
            self.output_dim,
            self.mlp_dim,
            self.num_features,
            self.active_features,
        )
        if min(dimensions) < 1:
            raise ValueError("all dimensions and active counts must be positive")
        if self.active_features > self.num_features:
            raise ValueError("active_features cannot exceed num_features")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


def _validate_center_scale(
    center: torch.Tensor,
    scale: torch.Tensor,
    dimension: int,
) -> None:
    if center.shape != (dimension,):
        raise ValueError("normalization center has the wrong shape")
    if scale.shape not in {(), (1,)}:
        raise ValueError("normalization scale must be scalar")
    if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
        raise ValueError("normalization statistics must be finite")
    if torch.any(scale <= 0):
        raise ValueError("normalization scale must be positive")


def scalar_normalization(
    values: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a per-coordinate center and a scalar RMS standard deviation.

    A scalar scale preserves directions and cosines while putting input and
    output reconstruction losses on comparable units.  Statistics must be fit
    on training genes only.
    """

    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("values must be [examples, dimensions] with >=2 rows")
    center = values.mean(dim=0)
    scale = torch.sqrt(torch.mean((values - center) ** 2)).clamp_min(eps)
    return center, scale


class TopKSAE(nn.Module):
    """Standard non-negative TopK sparse autoencoder with unit decoder rows."""

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        active_features: int,
        *,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if min(input_dim, num_features, active_features) < 1:
            raise ValueError("SAE dimensions and active count must be positive")
        if active_features > num_features:
            raise ValueError("active_features cannot exceed num_features")
        self.input_dim = int(input_dim)
        self.num_features = int(num_features)
        self.active_features = int(active_features)
        self.eps = float(eps)
        self.encoder_weight = nn.Parameter(torch.empty(input_dim, num_features))
        self.encoder_bias = nn.Parameter(torch.zeros(num_features))
        self.decoder_raw = nn.Parameter(torch.empty(num_features, input_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.decoder_raw.copy_(self.encoder_weight.T)
        nn.init.zeros_(self.encoder_bias)
        nn.init.zeros_(self.decoder_bias)

    def decoder(self) -> torch.Tensor:
        return F.normalize(self.decoder_raw, p=2.0, dim=1, eps=self.eps)

    def preactivations(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("SAE input must be [batch, input_dim]")
        return values @ self.encoder_weight + self.encoder_bias

    def encode(
        self,
        values: torch.Tensor,
        *,
        return_indices: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        scores = self.preactivations(values)
        selected, indices = torch.topk(
            scores,
            k=self.active_features,
            dim=-1,
            sorted=False,
        )
        selected = F.relu(selected)
        activations = torch.zeros_like(scores)
        activations.scatter_(1, indices, selected)
        if return_indices:
            return activations, indices
        return activations

    def decode(self, activations: torch.Tensor) -> torch.Tensor:
        if (
            activations.ndim != 2
            or activations.shape[1] != self.num_features
        ):
            raise ValueError("feature activations have the wrong shape")
        return activations @ self.decoder() + self.decoder_bias

    def forward(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        activations, indices = self.encode(values, return_indices=True)
        return self.decode(activations), activations, indices


def auxiliary_dead_feature_loss(
    sae: TopKSAE,
    values: torch.Tensor,
    reconstruction: torch.Tensor,
    dead_mask: torch.Tensor,
    *,
    auxiliary_features: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """AuxK reconstruction loss using only features that have stopped firing.

    This follows the dead-latent recovery objective in Gao et al. (2024).  The
    auxiliary reconstruction models the detached main residual; it is used only
    during training and never changes deployment-time TopK activity.
    """

    if values.shape != reconstruction.shape or values.ndim != 2:
        raise ValueError("values and reconstruction must be matching 2D tensors")
    if dead_mask.shape != (sae.num_features,) or dead_mask.dtype != torch.bool:
        raise ValueError("dead_mask must be a boolean feature mask")
    if auxiliary_features < 1:
        raise ValueError("auxiliary_features must be positive")
    dead_count = int(dead_mask.sum().detach().cpu())
    if dead_count == 0:
        return values.new_zeros(())
    count = min(auxiliary_features, dead_count)
    scores = sae.preactivations(values)
    masked_scores = scores.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
    selected, indices = torch.topk(masked_scores, k=count, dim=1, sorted=False)
    selected = F.relu(selected)
    auxiliary_activations = torch.zeros_like(scores)
    auxiliary_activations.scatter_(1, indices, selected)
    auxiliary_reconstruction = sae.decode(auxiliary_activations)
    target = values - reconstruction.detach() + sae.decoder_bias.detach()
    numerator = torch.mean((auxiliary_reconstruction - target) ** 2)
    denominator = torch.mean(
        (target - target.mean(dim=0, keepdim=True)) ** 2
    )
    return numerator / denominator.clamp_min(eps)


class JacobianSAEPair(nn.Module):
    """Paired TopK SAEs around a frozen two-layer GELU MLP."""

    def __init__(
        self,
        config: JacobianSAEConfig,
        *,
        input_weight: torch.Tensor,
        input_bias: torch.Tensor,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor,
        input_center: torch.Tensor,
        input_scale: torch.Tensor,
        output_center: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if input_weight.shape != (config.mlp_dim, config.input_dim):
            raise ValueError("MLP input weight has the wrong shape")
        if input_bias.shape != (config.mlp_dim,):
            raise ValueError("MLP input bias has the wrong shape")
        if output_weight.shape != (config.output_dim, config.mlp_dim):
            raise ValueError("MLP output weight has the wrong shape")
        if output_bias.shape != (config.output_dim,):
            raise ValueError("MLP output bias has the wrong shape")
        _validate_center_scale(input_center, input_scale, config.input_dim)
        _validate_center_scale(output_center, output_scale, config.output_dim)
        self.config = config
        self.input_sae = TopKSAE(
            config.input_dim,
            config.num_features,
            config.active_features,
            eps=config.eps,
        )
        self.output_sae = TopKSAE(
            config.output_dim,
            config.num_features,
            config.active_features,
            eps=config.eps,
        )
        self.register_buffer("mlp_input_weight", input_weight.detach().clone())
        self.register_buffer("mlp_input_bias", input_bias.detach().clone())
        self.register_buffer("mlp_output_weight", output_weight.detach().clone())
        self.register_buffer("mlp_output_bias", output_bias.detach().clone())
        self.register_buffer("input_center", input_center.detach().clone())
        self.register_buffer("input_scale", input_scale.detach().clone().reshape(()))
        self.register_buffer("output_center", output_center.detach().clone())
        self.register_buffer("output_scale", output_scale.detach().clone().reshape(()))

    def normalize_input(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.input_center) / self.input_scale

    def denormalize_input(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.input_scale + self.input_center

    def normalize_output(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.output_center) / self.output_scale

    def denormalize_output(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.output_scale + self.output_center

    @staticmethod
    def gelu_derivative(preactivation: torch.Tensor) -> torch.Tensor:
        """Derivative of PyTorch/Hugging Face exact GELU."""

        sqrt_two = math.sqrt(2.0)
        sqrt_two_pi = math.sqrt(2.0 * math.pi)
        return (
            0.5 * (1.0 + torch.erf(preactivation / sqrt_two))
            + preactivation
            * torch.exp(-0.5 * preactivation.square())
            / sqrt_two_pi
        )

    def mlp(
        self,
        raw_input: torch.Tensor,
        *,
        return_derivative: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if raw_input.ndim != 2 or raw_input.shape[1] != self.config.input_dim:
            raise ValueError("MLP input must be [batch, input_dim]")
        preactivation = F.linear(
            raw_input,
            self.mlp_input_weight,
            self.mlp_input_bias,
        )
        hidden = F.gelu(preactivation, approximate="none")
        output = F.linear(
            hidden,
            self.mlp_output_weight,
            self.mlp_output_bias,
        )
        if return_derivative:
            return output, self.gelu_derivative(preactivation)
        return output

    def active_jacobian(
        self,
        raw_input: torch.Tensor,
        input_indices: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return active output-feature by input-feature Jacobian blocks."""

        batch = raw_input.shape[0]
        k = self.config.active_features
        if input_indices.shape != (batch, k):
            raise ValueError("input_indices have the wrong shape")
        if output_indices.shape != (batch, k):
            raise ValueError("output_indices have the wrong shape")
        _, activation_derivative = self.mlp(
            raw_input,
            return_derivative=True,
        )
        input_decoder_raw = self.input_sae.decoder() * self.input_scale
        decoder_to_hidden = input_decoder_raw @ self.mlp_input_weight.T
        selected_decoder_to_hidden = decoder_to_hidden[input_indices]

        raw_to_output_features = (
            self.mlp_output_weight.T
            @ self.output_sae.encoder_weight
            / self.output_scale
        )
        selected_hidden_to_output = (
            raw_to_output_features.T[output_indices].transpose(1, 2)
        )
        # [batch, k_out, k_in]
        return torch.einsum(
            "bhi,bh,bho->boi",
            selected_decoder_to_hidden.transpose(1, 2),
            activation_derivative,
            selected_hidden_to_output,
        )

    def forward(
        self,
        raw_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        normalized_input = self.normalize_input(raw_input)
        reconstructed_input, input_acts, input_indices = self.input_sae(
            normalized_input
        )
        raw_output = self.mlp(raw_input)
        assert isinstance(raw_output, torch.Tensor)
        normalized_output = self.normalize_output(raw_output)
        reconstructed_output, output_acts, output_indices = self.output_sae(
            normalized_output
        )
        jacobian = self.active_jacobian(
            raw_input,
            input_indices,
            output_indices,
        )
        return {
            "normalized_input": normalized_input,
            "reconstructed_input": reconstructed_input,
            "input_activations": input_acts,
            "input_indices": input_indices,
            "raw_output": raw_output,
            "normalized_output": normalized_output,
            "reconstructed_output": reconstructed_output,
            "output_activations": output_acts,
            "output_indices": output_indices,
            "active_jacobian": jacobian,
        }

    def loss(
        self,
        raw_input: torch.Tensor,
        *,
        jacobian_coefficient: float,
        jacobian_metric: Literal["l1", "l1_over_rms"] = "l1_over_rms",
    ) -> dict[str, torch.Tensor]:
        if jacobian_coefficient < 0:
            raise ValueError("jacobian_coefficient must be nonnegative")
        run = self(raw_input)
        input_mse = F.mse_loss(
            run["reconstructed_input"],
            run["normalized_input"],
        )
        output_mse = F.mse_loss(
            run["reconstructed_output"],
            run["normalized_output"],
        )
        jacobian = run["active_jacobian"]
        if jacobian_metric == "l1":
            jacobian_penalty = jacobian.abs().mean()
        elif jacobian_metric == "l1_over_rms":
            rms = jacobian.square().mean(dim=(-2, -1), keepdim=True).sqrt()
            jacobian_penalty = (
                jacobian.abs() / rms.clamp_min(self.config.eps)
            ).mean()
        else:
            raise ValueError(f"unknown Jacobian metric {jacobian_metric!r}")
        total = (
            input_mse
            + output_mse
            + float(jacobian_coefficient) * jacobian_penalty
        )
        return {
            **run,
            "loss": total,
            "input_mse": input_mse,
            "output_mse": output_mse,
            "jacobian_penalty": jacobian_penalty,
        }


def explained_variance(
    target: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    """Return global held-out fraction of variance explained."""

    if target.shape != reconstructed.shape or target.ndim != 2:
        raise ValueError("target and reconstruction must be matching 2D tensors")
    residual = torch.sum((target - reconstructed) ** 2)
    centered = target - target.mean(dim=0, keepdim=True)
    denominator = torch.sum(centered**2)
    if float(denominator) <= eps:
        raise ValueError("target variance is zero")
    return float((1.0 - residual / denominator).detach().cpu())


def edge_mass_count(
    jacobian: torch.Tensor,
    *,
    fraction: float = 0.9,
) -> torch.Tensor:
    """Edges needed per example to explain a fraction of squared Jacobian mass."""

    if jacobian.ndim != 3 or not 0 < fraction <= 1:
        raise ValueError("jacobian must be 3D and fraction must lie in (0, 1]")
    mass = jacobian.square().flatten(start_dim=1)
    sorted_mass = torch.sort(mass, dim=1, descending=True).values
    cumulative = torch.cumsum(sorted_mass, dim=1)
    threshold = cumulative[:, -1:] * fraction
    count = torch.sum(cumulative < threshold, dim=1) + 1
    count = torch.where(
        cumulative[:, -1] > 0,
        count,
        torch.zeros_like(count),
    )
    return count


__all__ = [
    "JacobianSAEConfig",
    "JacobianSAEPair",
    "TopKSAE",
    "auxiliary_dead_feature_loss",
    "edge_mass_count",
    "explained_variance",
    "scalar_normalization",
]
