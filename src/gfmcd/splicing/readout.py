"""Minimal additive and interaction readouts for frozen/fine-tuned SpliceBERT.

The key comparison is architectural: the additive head can score each splice
site independently (plus an exon-length feature); the bilinear head is the
smallest model that can express partner-specific interactions between the two
site representations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from gfmcd.splicing.schema import SplicePairExample


@dataclass
class PairIndexBatch:
    positions: torch.Tensor  # [batch, max_pairs, 2], token coordinates
    lengths: torch.Tensor  # [batch, max_pairs]
    mask: torch.Tensor  # bool [batch, max_pairs]
    labels: torch.Tensor  # [batch]

    def to(self, device: torch.device | str) -> "PairIndexBatch":
        return PairIndexBatch(
            positions=self.positions.to(device),
            lengths=self.lengths.to(device),
            mask=self.mask.to(device),
            labels=self.labels.to(device),
        )


def collate_pair_indices(
    examples: Sequence[SplicePairExample],
    *,
    token_offset: int = 1,
    device: torch.device | str | None = None,
) -> PairIndexBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    max_pairs = max(len(x.pairs) for x in examples)
    positions = torch.zeros((len(examples), max_pairs, 2), dtype=torch.long)
    lengths = torch.zeros((len(examples), max_pairs), dtype=torch.float32)
    mask = torch.zeros((len(examples), max_pairs), dtype=torch.bool)
    labels = torch.empty(len(examples), dtype=torch.long)
    for b, example in enumerate(examples):
        labels[b] = example.true_pair_index
        for p, pair in enumerate(example.pairs):
            positions[b, p, 0] = pair.acceptor_pos + token_offset
            positions[b, p, 1] = pair.donor_pos + token_offset
            lengths[b, p] = pair.exon_length
            mask[b, p] = True
    batch = PairIndexBatch(positions, lengths, mask, labels)
    return batch.to(device) if device is not None else batch


def _gather_sites(hidden: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather acceptor and donor hidden states for every candidate pair."""
    if hidden.ndim != 3 or positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("hidden must be [B,S,D] and positions [B,P,2]")
    batch, seq_len, width = hidden.shape
    if positions.shape[0] != batch:
        raise ValueError("batch dimensions disagree")
    if positions.min() < 0 or positions.max() >= seq_len:
        raise ValueError("pair position outside hidden-state sequence")
    b = torch.arange(batch, device=hidden.device)[:, None]
    acceptor = hidden[b, positions[..., 0]]
    donor = hidden[b, positions[..., 1]]
    if acceptor.shape[-1] != width:
        raise RuntimeError("site gather produced unexpected shape")
    return acceptor, donor


class AdditivePairReadout(nn.Module):
    """Independent acceptor + donor scores and a learned length prior.

    No term can depend jointly on both endpoint representations.  This is the
    representation-level analogue of the local PWM baseline.
    """

    def __init__(self, hidden_size: int, site_hidden: int = 128, length_hidden: int = 16):
        super().__init__()
        self.acceptor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, site_hidden),
            nn.GELU(),
            nn.Linear(site_hidden, 1),
        )
        self.donor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, site_hidden),
            nn.GELU(),
            nn.Linear(site_hidden, 1),
        )
        self.length = nn.Sequential(
            nn.Linear(2, length_hidden),
            nn.GELU(),
            nn.Linear(length_hidden, 1),
        )

    @staticmethod
    def _length_features(lengths: torch.Tensor) -> torch.Tensor:
        x = lengths.clamp_min(1.0)
        return torch.stack((torch.log(x), torch.sqrt(x)), dim=-1)

    def forward(self, hidden: torch.Tensor, pairs: PairIndexBatch) -> torch.Tensor:
        acceptor, donor = _gather_sites(hidden, pairs.positions)
        logits = (
            self.acceptor(acceptor).squeeze(-1)
            + self.donor(donor).squeeze(-1)
            + self.length(self._length_features(pairs.lengths)).squeeze(-1)
        )
        return logits.masked_fill(~pairs.mask, torch.finfo(logits.dtype).min)


class BilinearPairReadout(nn.Module):
    """Additive terms plus a low-rank acceptor×donor interaction."""

    def __init__(
        self,
        hidden_size: int,
        interaction_rank: int = 32,
        site_hidden: int = 128,
        length_hidden: int = 16,
    ):
        super().__init__()
        if interaction_rank < 1:
            raise ValueError("interaction_rank must be positive")
        self.additive = AdditivePairReadout(hidden_size, site_hidden, length_hidden)
        self.left_norm = nn.LayerNorm(hidden_size)
        self.right_norm = nn.LayerNorm(hidden_size)
        self.left = nn.Linear(hidden_size, interaction_rank, bias=False)
        self.right = nn.Linear(hidden_size, interaction_rank, bias=False)
        self.scale = interaction_rank**-0.5
        self.interaction_bias = nn.Parameter(torch.zeros(()))

    def interaction(self, hidden: torch.Tensor, pairs: PairIndexBatch) -> torch.Tensor:
        acceptor, donor = _gather_sites(hidden, pairs.positions)
        left = self.left(self.left_norm(acceptor))
        right = self.right(self.right_norm(donor))
        return (left * right).sum(-1) * self.scale + self.interaction_bias

    def forward(self, hidden: torch.Tensor, pairs: PairIndexBatch) -> torch.Tensor:
        additive = self.additive(hidden, pairs)
        interaction = self.interaction(hidden, pairs)
        logits = additive + interaction
        return logits.masked_fill(~pairs.mask, torch.finfo(logits.dtype).min)


def pair_cross_entropy(logits: torch.Tensor, pairs: PairIndexBatch, reduction: str = "mean") -> torch.Tensor:
    if logits.shape != pairs.mask.shape:
        raise ValueError("logits and pair mask shapes disagree")
    return F.cross_entropy(logits, pairs.labels, reduction=reduction)


def pair_accuracy(logits: torch.Tensor, pairs: PairIndexBatch) -> torch.Tensor:
    return (logits.argmax(-1) == pairs.labels).float().mean()


def true_pair_margin(logits: torch.Tensor, pairs: PairIndexBatch) -> torch.Tensor:
    """Per-example true logit minus the strongest decoy logit."""
    batch = logits.shape[0]
    b = torch.arange(batch, device=logits.device)
    true = logits[b, pairs.labels]
    decoy_logits = logits.clone()
    decoy_logits[b, pairs.labels] = torch.finfo(logits.dtype).min
    decoy = decoy_logits.max(-1).values
    # A one-pair example has no decoy and should have been rejected by the data
    # builder; guard it anyway to avoid silently returning +inf.
    if not torch.isfinite(decoy).all():
        raise ValueError("every example needs at least one decoy pair")
    return true - decoy


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
