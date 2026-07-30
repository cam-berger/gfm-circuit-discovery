"""Native masked-LM reconstruction metrics for variable-length BPE targets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .schema import Interval
from .tokenization import (
    MaskPlan,
    encode_mask_plan,
    fair_alternative_token_ids,
)


@dataclass(frozen=True)
class ScoringRequest:
    request_id: str
    sequence: str
    site: Interval
    alternatives: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructionScore:
    request_id: str
    true_token_log_probability: float
    true_token_log_probability_per_covered_nt: float
    mean_true_token_log_probability: float
    token_accuracy: float
    exact_token_reconstruction: bool
    mask_plan: MaskPlan
    alternative_margins_per_covered_nt: Mapping[str, float]
    eligible_alternative_count: int
    alternative_logsumexp_margin_per_covered_nt: float | None
    alternative_logmeanexp_margin_per_covered_nt: float | None
    ineligible_alternatives: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "true_token_log_probability": self.true_token_log_probability,
            "true_token_log_probability_per_covered_nt": (
                self.true_token_log_probability_per_covered_nt
            ),
            "mean_true_token_log_probability": self.mean_true_token_log_probability,
            "token_accuracy": self.token_accuracy,
            "exact_token_reconstruction": self.exact_token_reconstruction,
            "mask_plan": self.mask_plan.to_dict(),
            "alternative_margins_per_covered_nt": dict(
                self.alternative_margins_per_covered_nt
            ),
            "eligible_alternative_count": self.eligible_alternative_count,
            "alternative_logsumexp_margin_per_covered_nt": (
                self.alternative_logsumexp_margin_per_covered_nt
            ),
            "alternative_logmeanexp_margin_per_covered_nt": (
                self.alternative_logmeanexp_margin_per_covered_nt
            ),
            "ineligible_alternatives": list(self.ineligible_alternatives),
        }


def score_from_logits(
    request_id: str,
    plan: MaskPlan,
    logits: torch.Tensor,
    *,
    alternative_token_ids: Mapping[str, Sequence[int]] | None = None,
    ineligible_alternatives: Sequence[str] = (),
) -> ReconstructionScore:
    """Score labels at simultaneously masked token positions.

    The sum is a simultaneous-mask pseudo-log-likelihood, not a normalized joint
    probability over nucleotide strings. Its primary normalization is by the complete
    BPE token cover, including unavoidable overhang around the annotated site.
    """

    if logits.ndim != 2:
        raise ValueError("single-example logits must be [tokens, vocabulary]")
    indices = torch.tensor(plan.token_indices, dtype=torch.long, device=logits.device)
    true_ids = torch.tensor(plan.true_token_ids, dtype=torch.long, device=logits.device)
    if int(indices.max()) >= logits.shape[0]:
        raise ValueError("mask plan token index exceeds logits")
    selected = torch.log_softmax(logits.index_select(0, indices).float(), dim=-1)
    if int(true_ids.max()) >= selected.shape[-1]:
        raise ValueError("true token ID exceeds logits vocabulary")
    token_log_probabilities = selected.gather(1, true_ids[:, None]).squeeze(1)
    total = float(token_log_probabilities.sum().item())
    if not math.isfinite(total):
        raise ValueError("non-finite true-token log probability")
    predictions = selected.argmax(dim=-1)
    accuracy = float((predictions == true_ids).float().mean().item())
    margins: dict[str, float] = {}
    alternative_totals: list[float] = []
    for name, token_ids in (alternative_token_ids or {}).items():
        alt = torch.tensor(tuple(token_ids), dtype=torch.long, device=logits.device)
        if alt.shape != true_ids.shape:
            raise ValueError(f"alternative {name!r} has a different token count")
        if int(alt.max()) >= selected.shape[-1]:
            raise ValueError(f"alternative {name!r} token ID exceeds logits vocabulary")
        alternative_total = float(selected.gather(1, alt[:, None]).sum().item())
        alternative_totals.append(alternative_total)
        margins[name] = (total - alternative_total) / plan.covered_nucleotides
    if alternative_totals:
        maximum = max(alternative_totals)
        alternative_logsumexp = maximum + math.log(
            sum(math.exp(value - maximum) for value in alternative_totals)
        )
        logsumexp_margin = (total - alternative_logsumexp) / plan.covered_nucleotides
        logmeanexp_margin = (
            total - (alternative_logsumexp - math.log(len(alternative_totals)))
        ) / plan.covered_nucleotides
    else:
        logsumexp_margin = None
        logmeanexp_margin = None
    return ReconstructionScore(
        request_id=request_id,
        true_token_log_probability=total,
        true_token_log_probability_per_covered_nt=total / plan.covered_nucleotides,
        mean_true_token_log_probability=total / len(plan.token_indices),
        token_accuracy=accuracy,
        exact_token_reconstruction=bool(torch.equal(predictions, true_ids)),
        mask_plan=plan,
        alternative_margins_per_covered_nt=margins,
        eligible_alternative_count=len(alternative_totals),
        alternative_logsumexp_margin_per_covered_nt=logsumexp_margin,
        alternative_logmeanexp_margin_per_covered_nt=logmeanexp_margin,
        ineligible_alternatives=tuple(sorted(ineligible_alternatives)),
    )


class ReconstructionScorer:
    """Batch scorer over a DNABERT-2-style adapter."""

    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.tokenizer = adapter.tokenizer
        if self.tokenizer.mask_token_id is None or self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must expose mask and pad token IDs")

    def score(
        self,
        requests: Sequence[ScoringRequest],
        *,
        batch_size: int = 16,
    ) -> list[ReconstructionScore]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not requests:
            return []
        prepared: list[
            tuple[
                ScoringRequest,
                list[int],
                MaskPlan,
                dict[str, tuple[int, ...]],
                tuple[str, ...],
            ]
        ] = []
        for request in requests:
            input_ids, plan = encode_mask_plan(
                request.sequence,
                request.site,
                self.adapter.encode,
            )
            alternatives: dict[str, tuple[int, ...]] = {}
            ineligible: list[str] = []
            for name, sequence in sorted(request.alternatives.items()):
                ids = fair_alternative_token_ids(
                    request.sequence,
                    plan,
                    sequence,
                    self.adapter.encode,
                )
                if ids is None:
                    ineligible.append(name)
                else:
                    alternatives[name] = ids
            masked = plan.masked_input_ids(input_ids, self.tokenizer.mask_token_id)
            prepared.append((request, masked, plan, alternatives, tuple(ineligible)))

        scores: list[ReconstructionScore] = []
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start : start + batch_size]
            max_tokens = max(len(row[1]) for row in chunk)
            input_batch = torch.full(
                (len(chunk), max_tokens),
                int(self.tokenizer.pad_token_id),
                dtype=torch.long,
            )
            attention = torch.zeros((len(chunk), max_tokens), dtype=torch.long)
            for row_index, (_, masked, _, _, _) in enumerate(chunk):
                input_batch[row_index, : len(masked)] = torch.tensor(
                    masked, dtype=torch.long
                )
                attention[row_index, : len(masked)] = 1
            logits = self.adapter.forward_logits(input_batch, attention)
            for row_index, (request, _, plan, alternatives, ineligible) in enumerate(
                chunk
            ):
                scores.append(
                    score_from_logits(
                        request.request_id,
                        plan,
                        logits[row_index, : int(attention[row_index].sum().item())],
                        alternative_token_ids=alternatives,
                        ineligible_alternatives=ineligible,
                    )
                )
        return scores
