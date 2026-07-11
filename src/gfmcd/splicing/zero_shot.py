"""Masked-language-model dependency scores for splice-site pairs.

The score asks whether predicting the acceptor motif becomes harder when the
candidate donor neighborhood is removed, and vice versa.  It uses the native
masked-LM head and therefore provides a zero-shot behavioral assay before any
splice-pair readout is trained.

This is a dependency measure, not causal circuit localization.  Multiple masks
and N-replacement are distribution shifts, so the study reports motif-only and
neighborhood-removal variants separately and requires matched decoy controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.splicing.schema import PairCandidate, SplicePairExample


@dataclass(frozen=True)
class NativeDependencyScore:
    acceptor_with_donor: float
    acceptor_without_donor: float
    donor_with_acceptor: float
    donor_without_acceptor: float

    @property
    def acceptor_dependency(self) -> float:
        return self.acceptor_with_donor - self.acceptor_without_donor

    @property
    def donor_dependency(self) -> float:
        return self.donor_with_acceptor - self.donor_without_acceptor

    @property
    def symmetric_dependency(self) -> float:
        return self.acceptor_dependency + self.donor_dependency

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update(
            acceptor_dependency=self.acceptor_dependency,
            donor_dependency=self.donor_dependency,
            symmetric_dependency=self.symmetric_dependency,
        )
        return out


def _single_nt_id(adapter: SpliceBertAdapter, base: str) -> int:
    batch = adapter.tokenize(base)
    return int(batch.input_ids[0, batch.prefix_tokens])


def _positions(center: int, flank: int, length: int) -> list[int]:
    return list(range(max(0, center - flank), min(length, center + flank)))


@torch.no_grad()
def motif_pll(
    adapter: SpliceBertAdapter,
    sequence: str,
    target_positions: Iterable[int],
    *,
    removed_positions: Iterable[int] = (),
    replacement_base: str = "N",
) -> float:
    """Sum log-probability of target bases under one-at-a-time MLM masking."""
    mask_id = getattr(adapter.tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("the tokenizer has no mask token; native MLM dependency is unavailable")
    replacement_id = _single_nt_id(adapter, replacement_base)
    batch = adapter.tokenize(sequence)
    target_positions = list(target_positions)
    removed_positions = set(removed_positions)
    total = 0.0
    for target_nt in target_positions:
        if target_nt < 0 or target_nt >= len(sequence):
            raise ValueError("target position outside sequence")
        ids = batch.input_ids.clone()
        for nt in removed_positions:
            if 0 <= nt < len(sequence) and nt != target_nt:
                ids[0, adapter.nucleotide_to_token(nt)] = replacement_id
        token_pos = adapter.nucleotide_to_token(target_nt)
        true_id = int(ids[0, token_pos])
        ids[0, token_pos] = int(mask_id)
        raw = adapter.model(
            input_ids=ids,
            attention_mask=batch.attention_mask,
            return_dict=True,
        )
        logits = getattr(raw, "logits", None)
        if logits is None:
            raise ValueError("loaded model has no MLM logits; use model_class='masked_lm'")
        total += float(torch.log_softmax(logits[0, token_pos], dim=-1)[true_id])
    return total


def pair_native_dependency(
    adapter: SpliceBertAdapter,
    example: SplicePairExample,
    pair: PairCandidate,
    *,
    partner_flank: int = 2,
) -> NativeDependencyScore:
    acc_targets = (pair.acceptor_pos - 2, pair.acceptor_pos - 1)
    don_targets = (pair.donor_pos, pair.donor_pos + 1)
    donor_removed = _positions(pair.donor_pos, partner_flank, len(example.sequence))
    acceptor_removed = _positions(pair.acceptor_pos, partner_flank, len(example.sequence))
    return NativeDependencyScore(
        acceptor_with_donor=motif_pll(adapter, example.sequence, acc_targets),
        acceptor_without_donor=motif_pll(
            adapter,
            example.sequence,
            acc_targets,
            removed_positions=donor_removed,
        ),
        donor_with_acceptor=motif_pll(adapter, example.sequence, don_targets),
        donor_without_acceptor=motif_pll(
            adapter,
            example.sequence,
            don_targets,
            removed_positions=acceptor_removed,
        ),
    )
