"""Fine-tunable DNABERT-2 promoter classifier."""

from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

from gfmcd.adapters.dnabert import DnaBertAdapter


DEFAULT_MODEL_NAME = "quietflamingo/dnabert2-no-flashattention"


class DnaBertPromoterClassifier(nn.Module):
    """DNABERT encoder with masked-mean binary classification head."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_labels: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        adapter = DnaBertAdapter.from_pretrained(
            model_name,
            device=None,
        )

        self.encoder = adapter.model
        self.tokenizer = adapter.tokenizer
        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout_probability = dropout

        hidden_size = int(self.encoder.config.hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=False,
        )

        hidden = outputs[0]

        if hidden.ndim != 3:
            raise ValueError(
                "Expected hidden states with shape "
                f"[batch, tokens, hidden], got {tuple(hidden.shape)}"
            )

        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (
                (hidden * mask).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1.0)
            )

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
