"""Native three-class splice-site readout for published GTEx fine-tunes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from gfmcd.splicing.coupling import CouplingExample

NO_SITE_LABEL = 0
DONOR_LABEL = 1
ACCEPTOR_LABEL = 2


@dataclass(frozen=True)
class ClassifierScores:
    """A class-vs-no-site margin and the corresponding softmax probability."""

    margin: np.ndarray
    probability: np.ndarray


class NativeClassifierTask:
    """Score annotated positions with the authors' unmasked three-class head."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str,
        batch_size: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.prefix_tokens = 1

    def encode(self, sequences: Sequence[str]) -> Any:
        encoded = self.tokenizer(
            [" ".join(sequence) for sequence in sequences],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        expected = max(len(sequence) for sequence in sequences) + 2
        if int(encoded["input_ids"].shape[1]) != expected:
            raise RuntimeError(
                "tokenizer is not one-token-per-nucleotide with [CLS]/[SEP]"
            )
        return encoded

    def logits(self, sequences: Sequence[str]) -> np.ndarray:
        """Return `[example, token, class]` logits on CPU."""

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                encoded = self.encode(sequences[start : start + self.batch_size])
                logits = self.model(**encoded).logits.float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("native classifier produced non-finite logits")
                chunks.append(logits.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def score(
        self,
        logits: np.ndarray,
        positions: Sequence[int],
        *,
        label: int,
    ) -> ClassifierScores:
        """Score `logit(label)-logit(no-site)` at nucleotide positions."""

        values = np.asarray(logits, dtype=np.float64)
        positions_array = np.asarray(positions, dtype=np.int64) + self.prefix_tokens
        rows = np.arange(len(positions_array))
        selected = values[rows, positions_array]
        margins = selected[:, label] - selected[:, NO_SITE_LABEL]
        stable = selected - selected.max(axis=1, keepdims=True)
        probabilities = np.exp(stable)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return ClassifierScores(
            margin=margins,
            probability=probabilities[:, label],
        )

    def final_projection_error(self, sequences: Sequence[str]) -> float:
        """Verify that the explicit final-state readout is the native model readout."""

        encoded = self.encode(sequences)
        with torch.no_grad():
            native = self.model(**encoded).logits.float()
            hidden = self.model.bert(**encoded, return_dict=True).last_hidden_state
            projected = self.model.classifier(self.model.dropout(hidden)).float()
        return float((native - projected).abs().max().detach().cpu())


def neutral_dinucleotide_position(example: CouplingExample) -> int | None:
    """Pick a deterministic downstream non-motif control near the partner distance."""

    donor = example.donor_pos
    partner_start = int(example.next_acceptor_pos) - 2
    excluded = {
        donor,
        partner_start,
        example.acceptor_pos - 2,
        *(position - 2 for position in example.other_ag),
        *example.other_gt,
        *example.real_don,
        *(position - 2 for position in example.real_acc),
    }
    candidates: list[int] = []
    for position in range(max(10, donor + 20), len(example.sequence) - 11):
        dinucleotide = example.sequence[position : position + 2]
        if len(dinucleotide) != 2 or any(base not in "ACGT" for base in dinucleotide):
            continue
        if dinucleotide in {"AG", "GT"}:
            continue
        if abs(position - partner_start) < 12:
            continue
        if any(abs(position - blocked) <= 5 for blocked in excluded):
            continue
        candidates.append(position)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda position: (abs(position - partner_start), position),
    )


def load_annotation_cohort(
    gtf: str,
    fasta: str,
    *,
    split: str,
    n: int,
    tag: str,
) -> list[CouplingExample]:
    """Reproduce the frozen cohort order without evaluating any model outcome."""

    from gfmcd.splicing.circuit import _dataset, eligible
    from gfmcd.splicing.coupling import CouplingConfig, stable_rng

    config = CouplingConfig()
    examples = eligible(_dataset(gtf, fasta), split)
    stable_rng(tag, config.seed).shuffle(examples)
    if len(examples) < n:
        raise RuntimeError(
            f"only {len(examples)} annotation-eligible {split} examples; asked for {n}"
        )
    return examples[:n]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered-kernel alignment across matched example representations."""

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.ndim != 2 or y_array.ndim != 2 or x_array.shape[0] != y_array.shape[0]:
        raise ValueError("linear CKA requires two 2D matrices with matched rows")
    x_centered = x_array - x_array.mean(axis=0, keepdims=True)
    y_centered = y_array - y_array.mean(axis=0, keepdims=True)
    cross = x_centered.T @ y_centered
    xx = x_centered.T @ x_centered
    yy = y_centered.T @ y_centered
    denominator = np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro")
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("linear CKA is undefined for a zero-variance representation")
    return float(np.linalg.norm(cross, ord="fro") ** 2 / denominator)
