"""Circuit-facing access to the pinned DNABERT-2 bundle.

This wrapper contains architecture facts only. Site definitions, masking, biological
controls, and atlas statistics live in :mod:`gfmcd.reconstruction_atlas`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from .loader import Dnabert2Bundle


@dataclass(frozen=True)
class Dnabert2HookPoint:
    """A named, hookable component with its runtime coordinate convention."""

    name: str
    module: Any
    representation: str
    coordinate_system: str


class Dnabert2Adapter:
    """Stable token, output, and component conventions for DNABERT-2-117M."""

    def __init__(self, bundle: Dnabert2Bundle):
        self.bundle = bundle
        self.model = bundle.model
        self.tokenizer = bundle.tokenizer
        self.config = bundle.config
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration as exc:  # pragma: no cover - real models have parameters
            raise ValueError("DNABERT-2 model exposes no parameters") from exc
        if self.tokenizer.mask_token_id is None:
            raise ValueError("DNABERT-2 tokenizer has no mask token")
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("atlas requires a fast tokenizer with nucleotide offsets")
        self._layers = tuple(self.model.bert.encoder.layer)
        if len(self._layers) != int(self.config.num_hidden_layers):
            raise ValueError("encoder layer count disagrees with checkpoint config")

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    @property
    def num_heads(self) -> int:
        return int(self.config.num_attention_heads)

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def head_size(self) -> int:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden size is not divisible by head count")
        return self.hidden_size // self.num_heads

    @property
    def max_tokens(self) -> int:
        return int(self.config.max_position_embeddings)

    def identity(self) -> dict[str, Any]:
        return self.bundle.identity()

    def encode(self, sequence: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tokenizer(
            sequence,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        ids = [int(x) for x in encoded["input_ids"]]
        offsets = [tuple(int(y) for y in x) for x in encoded["offset_mapping"]]
        if len(ids) != len(offsets):
            raise ValueError("token IDs and offset mapping have different lengths")
        if len(ids) > self.max_tokens:
            raise ValueError(
                f"sequence produces {len(ids)} tokens, exceeding checkpoint limit {self.max_tokens}"
            )
        return ids, offsets

    @torch.no_grad()
    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have identical shape")
        raw = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )
        logits = raw.logits if hasattr(raw, "logits") else raw[0]
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise ValueError(
                "DNABERT-2 MLM did not return full [batch, token, vocabulary] logits; "
                "do not pass labels to atlas scoring"
            )
        return logits

    def hook_points(self) -> tuple[Dnabert2HookPoint, ...]:
        """Return hook points without pretending unpadded states are nucleotide indexed."""

        points: list[Dnabert2HookPoint] = [
            Dnabert2HookPoint(
                "embedding",
                self.model.bert.embeddings,
                "residual",
                "padded_batch_token",
            )
        ]
        for layer_index, layer in enumerate(self._layers):
            prefix = f"layer.{layer_index}"
            points.extend(
                (
                    Dnabert2HookPoint(
                        f"{prefix}.qkv",
                        layer.attention.self.Wqkv,
                        "concatenated_qkv",
                        "unpadded_valid_token",
                    ),
                    Dnabert2HookPoint(
                        f"{prefix}.attention_context",
                        layer.attention.self,
                        "merged_head_context",
                        "unpadded_valid_token",
                    ),
                    Dnabert2HookPoint(
                        f"{prefix}.attention_write",
                        layer.attention.output.dense,
                        "attention_output_before_residual",
                        "unpadded_valid_token",
                    ),
                    Dnabert2HookPoint(
                        f"{prefix}.attention_residual",
                        layer.attention.output,
                        "post_attention_residual",
                        "unpadded_valid_token",
                    ),
                    Dnabert2HookPoint(
                        f"{prefix}.mlp_hidden",
                        layer.mlp.gated_layers,
                        "glu_pre_activation",
                        "unpadded_valid_token",
                    ),
                    Dnabert2HookPoint(
                        f"{prefix}.mlp_write",
                        layer.mlp.wo,
                        "mlp_output_before_residual",
                        "unpadded_valid_token",
                    ),
                    Dnabert2HookPoint(
                        f"{prefix}.residual",
                        layer,
                        "post_layer_residual",
                        "unpadded_valid_token",
                    ),
                )
            )
        points.append(
            Dnabert2HookPoint(
                "mlm_transform",
                self.model.cls.predictions.transform,
                "mlm_readout_hidden",
                "padded_batch_token",
            )
        )
        return tuple(points)

    @staticmethod
    def unpadded_flat_indices(attention_mask: torch.Tensor) -> torch.Tensor:
        """Map MosaicBERT's unpadded row index to padded ``batch * tokens`` index."""

        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must be [batch, tokens]")
        return torch.nonzero(
            attention_mask.reshape(-1).bool(), as_tuple=False
        ).flatten()

    @staticmethod
    def unpadded_row(
        attention_mask: torch.Tensor,
        batch_index: int,
        token_index: int,
    ) -> int:
        """Return the unpadded row for a valid padded token coordinate."""

        batch, tokens = attention_mask.shape
        if not (0 <= batch_index < batch and 0 <= token_index < tokens):
            raise IndexError("padded token coordinate is outside the batch")
        if not bool(attention_mask[batch_index, token_index]):
            raise ValueError("padding tokens have no unpadded activation row")
        flat = batch_index * tokens + token_index
        indices = Dnabert2Adapter.unpadded_flat_indices(attention_mask)
        hit = torch.nonzero(indices == flat, as_tuple=False).flatten()
        if hit.numel() != 1:
            raise RuntimeError("failed to invert MosaicBERT unpadding map")
        return int(hit.item())

    def component_names(self) -> tuple[str, ...]:
        return tuple(point.name for point in self.hook_points())

    def resolve_hook_points(
        self, names: Iterable[str]
    ) -> tuple[Dnabert2HookPoint, ...]:
        by_name = {point.name: point for point in self.hook_points()}
        requested = tuple(names)
        missing = sorted(set(requested) - set(by_name))
        if missing:
            raise KeyError(f"unknown DNABERT-2 hook point(s): {missing}")
        return tuple(by_name[name] for name in requested)
