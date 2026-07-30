"""Sparse coordinates for the difference between two frozen GELU MLPs.

The module models only a matched base-to-tuned displacement.  It uses signed,
zero-bias TopK dictionaries and an exact state-versus-parameter Shapley
decomposition of both the finite MLP write and the local Jacobian change.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelDifferenceJSAEConfig:
    """Architecture of a base-to-tuned model-difference J-SAE."""

    input_dim: int
    output_dim: int
    mlp_dim: int
    input_features: int
    output_features: int
    active_features: int
    eps: float = 1e-8

    def __post_init__(self) -> None:
        values = (
            self.input_dim,
            self.output_dim,
            self.mlp_dim,
            self.input_features,
            self.output_features,
            self.active_features,
        )
        if min(values) < 1:
            raise ValueError("all dimensions and active counts must be positive")
        if self.active_features > min(
            self.input_features, self.output_features
        ):
            raise ValueError("active_features exceeds a dictionary width")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


class SignedTopKSAE(nn.Module):
    """Bias-free signed TopK linear autoencoder with unit decoder rows."""

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
        self.decoder_raw = nn.Parameter(torch.empty(num_features, input_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.decoder_raw.copy_(self.encoder_weight.T)

    def decoder(self) -> torch.Tensor:
        return F.normalize(self.decoder_raw, p=2.0, dim=1, eps=self.eps)

    def preactivations(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("SAE input must be [batch, input_dim]")
        return values @ self.encoder_weight

    def encode(
        self,
        values: torch.Tensor,
        *,
        return_indices: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        scores = self.preactivations(values)
        indices = torch.topk(
            scores.abs(),
            k=self.active_features,
            dim=-1,
            sorted=False,
        ).indices
        selected = torch.gather(scores, 1, indices)
        activations = torch.zeros_like(scores)
        activations.scatter_(1, indices, selected)
        if return_indices:
            return activations, indices
        return activations

    def encode_shared_support(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode two blocks on one support selected by their group magnitude."""

        if left.shape != right.shape:
            raise ValueError("shared-support inputs must have matching shapes")
        left_scores = self.preactivations(left)
        right_scores = self.preactivations(right)
        group = torch.sqrt(left_scores.square() + right_scores.square())
        indices = torch.topk(
            group,
            k=self.active_features,
            dim=-1,
            sorted=False,
        ).indices
        left_activations = torch.zeros_like(left_scores)
        right_activations = torch.zeros_like(right_scores)
        left_activations.scatter_(
            1, indices, torch.gather(left_scores, 1, indices)
        )
        right_activations.scatter_(
            1, indices, torch.gather(right_scores, 1, indices)
        )
        return left_activations, right_activations, indices

    def decode(self, activations: torch.Tensor) -> torch.Tensor:
        if (
            activations.ndim != 2
            or activations.shape[1] != self.num_features
        ):
            raise ValueError("feature activations have the wrong shape")
        return activations @ self.decoder()

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        activations, indices = self.encode(values, return_indices=True)
        return self.decode(activations), activations, indices


def zero_anchored_scalar_scale(
    values: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Return scalar RMS without removing the mean displacement."""

    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("values must be [examples, dimensions]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    return torch.sqrt(torch.mean(values.square())).clamp_min(eps)


class ModelDifferenceJacobianSAE(nn.Module):
    """Signed sparse dictionaries for a matched base-to-tuned MLP change."""

    _PARAMETER_NAMES = (
        "input_weight",
        "input_bias",
        "output_weight",
        "output_bias",
    )

    def __init__(
        self,
        config: ModelDifferenceJSAEConfig,
        *,
        base_weights: dict[str, torch.Tensor],
        tuned_weights: dict[str, torch.Tensor],
        input_scale: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self._validate_weights(base_weights)
        self._validate_weights(tuned_weights)
        for view, weights in (("base", base_weights), ("tuned", tuned_weights)):
            for name in self._PARAMETER_NAMES:
                self.register_buffer(
                    f"{view}_{name}", weights[name].detach().clone()
                )
        if input_scale.shape not in {(), (1,)}:
            raise ValueError("input_scale must be scalar")
        if output_scale.shape not in {(), (1,)}:
            raise ValueError("output_scale must be scalar")
        if (
            not torch.isfinite(input_scale).all()
            or not torch.isfinite(output_scale).all()
            or torch.any(input_scale <= 0)
            or torch.any(output_scale <= 0)
        ):
            raise ValueError("scales must be finite and positive")
        self.register_buffer("input_scale", input_scale.detach().clone().reshape(()))
        self.register_buffer(
            "output_scale", output_scale.detach().clone().reshape(())
        )
        self.input_sae = SignedTopKSAE(
            config.input_dim,
            config.input_features,
            config.active_features,
            eps=config.eps,
        )
        self.output_sae = SignedTopKSAE(
            config.output_dim,
            config.output_features,
            config.active_features,
            eps=config.eps,
        )

    def _validate_weights(self, weights: dict[str, torch.Tensor]) -> None:
        expected = {
            "input_weight": (self.config.mlp_dim, self.config.input_dim),
            "input_bias": (self.config.mlp_dim,),
            "output_weight": (self.config.output_dim, self.config.mlp_dim),
            "output_bias": (self.config.output_dim,),
        }
        if set(weights) != set(expected):
            raise ValueError("MLP weight mapping has the wrong keys")
        for name, shape in expected.items():
            if tuple(weights[name].shape) != shape:
                raise ValueError(f"{name} has the wrong shape")
            if not torch.isfinite(weights[name]).all():
                raise ValueError(f"{name} is not finite")

    @staticmethod
    def gelu_derivative(preactivation: torch.Tensor) -> torch.Tensor:
        sqrt_two = math.sqrt(2.0)
        sqrt_two_pi = math.sqrt(2.0 * math.pi)
        return (
            0.5 * (1.0 + torch.erf(preactivation / sqrt_two))
            + preactivation
            * torch.exp(-0.5 * preactivation.square())
            / sqrt_two_pi
        )

    def _weights(
        self, view: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if view not in {"base", "tuned"}:
            raise ValueError("view must be 'base' or 'tuned'")
        return tuple(
            getattr(self, f"{view}_{name}") for name in self._PARAMETER_NAMES
        )  # type: ignore[return-value]

    def mlp(
        self,
        raw_input: torch.Tensor,
        *,
        view: str,
        return_derivative: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            raw_input.ndim != 2
            or raw_input.shape[1] != self.config.input_dim
        ):
            raise ValueError("MLP input must be [batch, input_dim]")
        input_weight, input_bias, output_weight, output_bias = self._weights(view)
        preactivation = F.linear(raw_input, input_weight, input_bias)
        hidden = F.gelu(preactivation, approximate="none")
        output = F.linear(hidden, output_weight, output_bias)
        if return_derivative:
            return output, self.gelu_derivative(preactivation)
        return output

    def write_cells(
        self,
        base_input: torch.Tensor,
        input_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if base_input.shape != input_delta.shape:
            raise ValueError("base_input and input_delta must match")
        tuned_input = base_input + input_delta
        return {
            "00": self.mlp(base_input, view="base"),
            "10": self.mlp(tuned_input, view="base"),
            "01": self.mlp(base_input, view="tuned"),
            "11": self.mlp(tuned_input, view="tuned"),
        }  # type: ignore[return-value]

    @staticmethod
    def shapley_from_cells(
        cells: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if set(cells) != {"00", "10", "01", "11"}:
            raise ValueError("four factorial cells are required")
        state = 0.5 * (
            (cells["10"] - cells["00"]) + (cells["11"] - cells["01"])
        )
        parameter = 0.5 * (
            (cells["01"] - cells["00"]) + (cells["11"] - cells["10"])
        )
        return {
            "state": state,
            "parameter": parameter,
            "total": cells["11"] - cells["00"],
            "interaction": (
                cells["11"] - cells["10"] - cells["01"] + cells["00"]
            ),
            "state_tuned_path": cells["11"] - cells["01"],
            "parameter_tuned_path": cells["01"] - cells["00"],
            "state_base_path": cells["10"] - cells["00"],
            "parameter_base_path": cells["11"] - cells["10"],
        }

    def write_shapley(
        self,
        base_input: torch.Tensor,
        input_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.shapley_from_cells(
            self.write_cells(base_input, input_delta)
        )

    def normalize_input_delta(self, values: torch.Tensor) -> torch.Tensor:
        return values / self.input_scale

    def denormalize_input_delta(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.input_scale

    def normalize_output_delta(self, values: torch.Tensor) -> torch.Tensor:
        return values / self.output_scale

    def denormalize_output_delta(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.output_scale

    def encode_difference(
        self,
        input_delta: torch.Tensor,
        state_write: torch.Tensor,
        parameter_write: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        normalized_input = self.normalize_input_delta(input_delta)
        reconstructed_input, input_acts, input_indices = self.input_sae(
            normalized_input
        )
        normalized_state = self.normalize_output_delta(state_write)
        normalized_parameter = self.normalize_output_delta(parameter_write)
        state_acts, parameter_acts, output_indices = (
            self.output_sae.encode_shared_support(
                normalized_state, normalized_parameter
            )
        )
        reconstructed_state = self.output_sae.decode(state_acts)
        reconstructed_parameter = self.output_sae.decode(parameter_acts)
        return {
            "normalized_input": normalized_input,
            "reconstructed_input": reconstructed_input,
            "input_activations": input_acts,
            "input_indices": input_indices,
            "normalized_state": normalized_state,
            "normalized_parameter": normalized_parameter,
            "normalized_total": normalized_state + normalized_parameter,
            "reconstructed_state": reconstructed_state,
            "reconstructed_parameter": reconstructed_parameter,
            "reconstructed_total": (
                reconstructed_state + reconstructed_parameter
            ),
            "state_activations": state_acts,
            "parameter_activations": parameter_acts,
            "output_indices": output_indices,
        }

    def _projected_cell_jacobian(
        self,
        raw_input: torch.Tensor,
        *,
        view: str,
        input_indices: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch = raw_input.shape[0]
        k = self.config.active_features
        if input_indices.shape != (batch, k):
            raise ValueError("input_indices have the wrong shape")
        if output_indices.shape != (batch, k):
            raise ValueError("output_indices have the wrong shape")
        _, derivative = self.mlp(
            raw_input, view=view, return_derivative=True
        )
        input_weight, _, output_weight, _ = self._weights(view)
        raw_input_directions = self.input_sae.decoder() * self.input_scale
        decoder_to_hidden = raw_input_directions @ input_weight.T
        selected_decoder_to_hidden = decoder_to_hidden[input_indices]
        hidden_to_output_features = (
            output_weight.T
            @ self.output_sae.encoder_weight
            / self.output_scale
        )
        selected_hidden_to_output = (
            hidden_to_output_features.T[output_indices].transpose(1, 2)
        )
        return torch.einsum(
            "bhi,bh,bho->boi",
            selected_decoder_to_hidden.transpose(1, 2),
            derivative,
            selected_hidden_to_output,
        )

    def active_jacobian_cells(
        self,
        base_input: torch.Tensor,
        input_delta: torch.Tensor,
        input_indices: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if base_input.shape != input_delta.shape:
            raise ValueError("base_input and input_delta must match")
        tuned_input = base_input + input_delta
        return {
            "00": self._projected_cell_jacobian(
                base_input,
                view="base",
                input_indices=input_indices,
                output_indices=output_indices,
            ),
            "10": self._projected_cell_jacobian(
                tuned_input,
                view="base",
                input_indices=input_indices,
                output_indices=output_indices,
            ),
            "01": self._projected_cell_jacobian(
                base_input,
                view="tuned",
                input_indices=input_indices,
                output_indices=output_indices,
            ),
            "11": self._projected_cell_jacobian(
                tuned_input,
                view="tuned",
                input_indices=input_indices,
                output_indices=output_indices,
            ),
        }

    def _native_cell_operator(
        self,
        raw_input: torch.Tensor,
        *,
        view: str,
        input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized native output actions for active input directions."""

        batch = raw_input.shape[0]
        k = self.config.active_features
        if input_indices.shape != (batch, k):
            raise ValueError("input_indices have the wrong shape")
        _, derivative = self.mlp(
            raw_input, view=view, return_derivative=True
        )
        input_weight, _, output_weight, _ = self._weights(view)
        raw_input_directions = self.input_sae.decoder() * self.input_scale
        decoder_to_hidden = raw_input_directions @ input_weight.T
        selected = decoder_to_hidden[input_indices]
        # [batch, output_dim, k_in], in normalized output coordinates.
        return (
            torch.einsum(
                "bih,bh,dh->bdi",
                selected,
                derivative,
                output_weight,
            )
            / self.output_scale
        )

    def active_operator_cells(
        self,
        base_input: torch.Tensor,
        input_delta: torch.Tensor,
        input_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the four native operator actions on active input directions."""

        if base_input.shape != input_delta.shape:
            raise ValueError("base_input and input_delta must match")
        tuned_input = base_input + input_delta
        return {
            "00": self._native_cell_operator(
                base_input,
                view="base",
                input_indices=input_indices,
            ),
            "10": self._native_cell_operator(
                tuned_input,
                view="base",
                input_indices=input_indices,
            ),
            "01": self._native_cell_operator(
                base_input,
                view="tuned",
                input_indices=input_indices,
            ),
            "11": self._native_cell_operator(
                tuned_input,
                view="tuned",
                input_indices=input_indices,
            ),
        }

    def active_operator_shapley(
        self,
        base_input: torch.Tensor,
        input_delta: torch.Tensor,
        input_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.shapley_from_cells(
            self.active_operator_cells(
                base_input, input_delta, input_indices
            )
        )

    def decoded_operator_shapley(
        self,
        graph_shapley: dict[str, torch.Tensor],
        output_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode feature-graph actions into normalized native output space."""

        batch = output_indices.shape[0]
        if output_indices.shape != (
            batch,
            self.config.active_features,
        ):
            raise ValueError("output_indices have the wrong shape")
        selected_output_decoder = self.output_sae.decoder()[output_indices]
        result = {}
        for name in ("state", "parameter", "total"):
            graph = graph_shapley[name]
            if graph.shape != (
                batch,
                self.config.active_features,
                self.config.active_features,
            ):
                raise ValueError("graph block has the wrong shape")
            result[name] = torch.einsum(
                "boi,bod->bdi", graph, selected_output_decoder
            )
        return result

    def active_jacobian_shapley(
        self,
        base_input: torch.Tensor,
        input_delta: torch.Tensor,
        input_indices: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cells = self.active_jacobian_cells(
            base_input, input_delta, input_indices, output_indices
        )
        return self.shapley_from_cells(cells)

    def forward(
        self,
        base_input: torch.Tensor,
        input_delta: torch.Tensor,
        *,
        include_jacobian: bool = True,
    ) -> dict[str, torch.Tensor]:
        writes = self.write_shapley(base_input, input_delta)
        encoded = self.encode_difference(
            input_delta, writes["state"], writes["parameter"]
        )
        result = {**writes, **encoded}
        if include_jacobian:
            jacobian = self.active_jacobian_shapley(
                base_input,
                input_delta,
                encoded["input_indices"],
                encoded["output_indices"],
            )
            result.update(
                {
                    "jacobian_state": jacobian["state"],
                    "jacobian_parameter": jacobian["parameter"],
                    "jacobian_total": jacobian["total"],
                    "jacobian_interaction": jacobian["interaction"],
                }
            )
        return result


__all__ = [
    "ModelDifferenceJSAEConfig",
    "ModelDifferenceJacobianSAE",
    "SignedTopKSAE",
    "zero_anchored_scalar_scale",
]
