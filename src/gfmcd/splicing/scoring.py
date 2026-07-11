"""Behavioral scoring helpers that keep candidate coordinates fixed.

Counterfactual sequence edits may deliberately destroy a canonical motif.  The
candidate set is therefore inherited from the clean example rather than
re-enumerated after the edit; otherwise the intervention would change both the
input and the answer space.  This module makes that choice explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn

from gfmcd.adapters.splicebert import SpliceBertAdapter, normalize_sequence
from gfmcd.splicing.metrics import PairMetrics, evaluate_pair_logits
from gfmcd.splicing.readout import collate_pair_indices, pair_cross_entropy, true_pair_margin
from gfmcd.splicing.schema import SplicePairExample
from gfmcd.splicing.training import batches


@dataclass(frozen=True)
class ExampleScore:
    example_id: str
    gene_id: str
    transcript_id: str
    split: str
    true_pair_index: int
    predicted_pair_index: int
    true_margin: float
    nll: float
    reciprocal_rank: float
    candidate_count: int
    logits: tuple[float, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def forward_with_sequences(
    adapter: SpliceBertAdapter,
    readout: nn.Module,
    examples: Sequence[SplicePairExample],
    sequences: Sequence[str],
    *,
    representation_layer: int = -1,
    grad_backbone: bool = False,
) -> tuple[torch.Tensor, object]:
    if len(examples) != len(sequences):
        raise ValueError("examples and sequences must be aligned")
    normalized = [normalize_sequence(x) for x in sequences]
    for example, sequence in zip(examples, normalized):
        if len(sequence) != len(example.sequence):
            raise ValueError(
                f"counterfactual changed length for {example.example_id}: "
                f"{len(example.sequence)} -> {len(sequence)}"
            )
    tokenized = adapter.tokenize(normalized)
    outputs = adapter.forward(
        tokenized,
        output_hidden_states=True,
        output_attentions=False,
        grad=grad_backbone,
    )
    if outputs.hidden_states is None:
        raise RuntimeError("SpliceBERT did not return hidden states")
    hidden = outputs.hidden_states[representation_layer]
    pair_batch = collate_pair_indices(
        examples,
        token_offset=tokenized.prefix_tokens,
        device=hidden.device,
    )
    return readout(hidden, pair_batch), pair_batch


@torch.no_grad()
def evaluate_sequence_overrides(
    adapter: SpliceBertAdapter,
    readout: nn.Module,
    examples: Sequence[SplicePairExample],
    sequences: Sequence[str] | None = None,
    *,
    batch_size: int = 8,
    representation_layer: int = -1,
) -> tuple[float, PairMetrics, list[ExampleScore]]:
    if sequences is None:
        sequences = [x.sequence for x in examples]
    if len(examples) != len(sequences):
        raise ValueError("examples and sequences must be aligned")
    adapter.model.eval()
    readout.eval()

    total_n = 0
    loss_sum = 0.0
    acc_sum = 0.0
    mrr_sum = 0.0
    nll_sum = 0.0
    margin_sum = 0.0
    positive_sum = 0.0
    records: list[ExampleScore] = []

    for start in range(0, len(examples), batch_size):
        batch_examples = list(examples[start : start + batch_size])
        batch_sequences = list(sequences[start : start + batch_size])
        logits, pair_batch = forward_with_sequences(
            adapter,
            readout,
            batch_examples,
            batch_sequences,
            representation_layer=representation_layer,
            grad_backbone=False,
        )
        metric = evaluate_pair_logits(logits, pair_batch)
        losses = pair_cross_entropy(logits, pair_batch, reduction="none")
        margins = true_pair_margin(logits, pair_batch)
        masked = logits.masked_fill(~pair_batch.mask, torch.finfo(logits.dtype).min)
        order = torch.argsort(masked, dim=-1, descending=True)
        matches = order == pair_batch.labels[:, None]
        ranks = matches.float().argmax(-1) + 1
        predictions = masked.argmax(-1)

        n = len(batch_examples)
        total_n += n
        loss_sum += float(losses.sum())
        acc_sum += metric.top1_accuracy * n
        mrr_sum += metric.mean_reciprocal_rank * n
        nll_sum += metric.mean_nll * n
        margin_sum += metric.mean_margin * n
        positive_sum += metric.positive_margin_rate * n

        for i, example in enumerate(batch_examples):
            count = len(example.pairs)
            records.append(
                ExampleScore(
                    example_id=example.example_id,
                    gene_id=example.gene_id,
                    transcript_id=example.transcript_id,
                    split=example.split,
                    true_pair_index=example.true_pair_index,
                    predicted_pair_index=int(predictions[i]),
                    true_margin=float(margins[i]),
                    nll=float(losses[i]),
                    reciprocal_rank=float(1.0 / ranks[i]),
                    candidate_count=count,
                    logits=tuple(float(x) for x in masked[i, :count].cpu()),
                )
            )

    if total_n == 0:
        raise ValueError("cannot evaluate an empty example set")
    aggregate = PairMetrics(
        n=total_n,
        top1_accuracy=acc_sum / total_n,
        mean_reciprocal_rank=mrr_sum / total_n,
        mean_nll=nll_sum / total_n,
        mean_margin=margin_sum / total_n,
        positive_margin_rate=positive_sum / total_n,
    )
    return loss_sum / total_n, aggregate, records
