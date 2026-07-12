"""Shared protocols for pretrained genomic-model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TokenizedSequences:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prefix_tokens: int
    nucleotide_lengths: list[int]
    texts: list[str]

    def to(self, device: str | torch.device) -> "TokenizedSequences":
        return TokenizedSequences(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            prefix_tokens=self.prefix_tokens,
            nucleotide_lengths=self.nucleotide_lengths,
            texts=self.texts,
        )


@dataclass
class GenomicModelOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None
    attentions: tuple[torch.Tensor, ...] | None
    logits: torch.Tensor | None
    raw: Any
