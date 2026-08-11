"""Adapter for DNABERT-2 checkpoints with MosaicBERT-style remote code.

DNABERT-2 differs from standard Hugging Face BERT in several important ways:

* It uses variable-length BPE tokens rather than one token per nucleotide.
* Token-to-nucleotide alignment must be recovered from offset mappings.
* Its model forward method returns tuples rather than ModelOutput objects.
* ``output_hidden_states=True`` is ignored by the custom implementation.
* All encoder layers are requested with ``output_all_encoded_layers=True``.
* Intermediate layers are returned in an unpadded ``[N_tokens, hidden]`` form.
* The checkpoint does not contain a trained pooler, so the pooler is disabled.

This adapter converts those behaviors into the repository's common
``GenomicModelOutput`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from gfmcd.adapters.base import GenomicModelOutput, TokenizedSequences


DEFAULT_MODEL = "quietflamingo/dnabert2-no-flashattention"


def normalize_sequence(sequence: str) -> str:
    """Normalize and validate a DNA sequence."""

    seq = sequence.upper().replace("U", "T").replace(" ", "")
    bad = set(seq) - set("ACGTN")

    if bad:
        raise ValueError(f"unsupported nucleotide symbols: {sorted(bad)}")

    if not seq:
        raise ValueError("sequence cannot be empty")

    return seq


@dataclass
class DnaTokenizedSequences(TokenizedSequences):
    """DNABERT tokenized batch with nucleotide-alignment information."""

    offset_mapping: torch.Tensor
    special_tokens_mask: torch.Tensor
    token_to_nucleotide_spans: list[list[tuple[int, int] | None]]

    def to(self, device: str | torch.device) -> "DnaTokenizedSequences":
        return DnaTokenizedSequences(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            prefix_tokens=self.prefix_tokens,
            nucleotide_lengths=self.nucleotide_lengths,
            texts=self.texts,
            offset_mapping=self.offset_mapping.to(device),
            special_tokens_mask=self.special_tokens_mask.to(device),
            token_to_nucleotide_spans=self.token_to_nucleotide_spans,
        )


@dataclass(frozen=True)
class DnaBertMetadata:
    source: str
    revision: str | None
    prefix_tokens: int
    max_nucleotides: int
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    parameter_count: int
    config_sha256: str
    uses_variable_length_tokens: bool
    pooler_disabled: bool


class DnaBertAdapter:
    """Expose DNABERT-2 through a stable circuit-facing API."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        source: str = "in-memory",
        revision: str | None = None,
        device: str | torch.device | None = None,
        max_nucleotides: int | None = None,
        disable_pooler: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.source = source
        self.revision = revision

        self.config = getattr(model, "config", None)
        if self.config is None:
            raise ValueError("DNABERT model must expose a Hugging Face-style config")

        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "DNABERT adapter requires a fast tokenizer for offset mappings"
            )

        if disable_pooler and hasattr(self.model, "pooler"):
            # The encoder checkpoint does not contain trained pooler weights.
            # Disabling it also avoids a bug in output_all_encoded_layers mode.
            self.model.pooler = None

        self.pooler_disabled = getattr(self.model, "pooler", None) is None

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        config_limit = getattr(self.config, "max_position_embeddings", None)

        plausible_limits = [
            int(value)
            for value in (tokenizer_limit, config_limit)
            if value is not None and 0 < int(value) < 1_000_000
        ]

        default_limit = min(plausible_limits) if plausible_limits else 512
        self.max_nucleotides = int(max_nucleotides or default_limit)

        self._layers = self._resolve_encoder_layers()
        self._attention_modules = self._resolve_attention_modules()
        self._mlp_modules = self._resolve_mlp_modules()
        self.prefix_tokens = self._infer_prefix_tokens()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path = DEFAULT_MODEL,
        *,
        tokenizer_name_or_path: str | Path | None = None,
        revision: str | None = None,
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
        max_nucleotides: int | None = None,
        **kwargs: Any,
    ) -> "DnaBertAdapter":
        """Load a DNABERT encoder using the known-compatible code path."""

        try:
            from transformers import AutoModel, AutoTokenizer
            from transformers.models.bert.configuration_bert import BertConfig
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "install DNABERT dependencies before loading the adapter"
            ) from exc

        model_path = str(model_name_or_path)
        tokenizer_path = str(tokenizer_name_or_path or model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

        # Explicit BertConfig loading avoids incompatible automatic/meta-device
        # initialization paths in some Transformers versions.
        config = BertConfig.from_pretrained(
            model_path,
            revision=revision,
            local_files_only=local_files_only,
        )

        model = AutoModel.from_pretrained(
            model_path,
            revision=revision,
            config=config,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            low_cpu_mem_usage=False,
            **kwargs,
        )

        return cls(
            model,
            tokenizer,
            source=model_path,
            revision=revision,
            device=device,
            max_nucleotides=max_nucleotides,
            disable_pooler=True,
        )

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.config, "hidden_size"))

    @property
    def num_heads(self) -> int:
        return int(getattr(self.config, "num_attention_heads"))

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden size is not divisible by number of heads")
        return self.hidden_size // self.num_heads

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    @property
    def encoder_layers(self) -> tuple[Any, ...]:
        return tuple(self._layers)

    @property
    def attention_self_modules(self) -> tuple[Any, ...]:
        """Return hookable unpadded self-attention modules.

        Their activations are not standard ``[batch, sequence, hidden]``
        tensors. DNABERT-specific intervention utilities must account for the
        unpadded token representation.
        """

        return tuple(self._attention_modules)

    @property
    def mlp_modules(self) -> tuple[Any, ...]:
        return tuple(self._mlp_modules)

    def _resolve_encoder_layers(self) -> list[Any]:
        encoder = getattr(self.model, "encoder", None)
        layers = getattr(encoder, "layer", None)

        if layers is None:
            raise AttributeError("expected DNABERT layers at model.encoder.layer")

        result = list(layers)

        if not result:
            raise ValueError("DNABERT encoder contains no layers")

        return result

    def _resolve_attention_modules(self) -> list[Any]:
        modules = []

        for index, layer in enumerate(self._layers):
            attention = getattr(layer, "attention", None)
            module = getattr(attention, "self", None)

            if module is None:
                raise AttributeError(
                    f"expected attention module at encoder layer {index}.attention.self"
                )

            if not hasattr(module, "register_forward_hook"):
                raise TypeError(f"layer {index} attention module is not hookable")

            modules.append(module)

        return modules

    def _resolve_mlp_modules(self) -> list[Any]:
        modules = []

        for index, layer in enumerate(self._layers):
            module = getattr(layer, "mlp", None)

            if module is None:
                raise AttributeError(
                    f"expected MLP module at encoder layer {index}.mlp"
                )

            if not hasattr(module, "register_forward_hook"):
                raise TypeError(f"layer {index} MLP module is not hookable")

            modules.append(module)

        return modules

    def _infer_prefix_tokens(self) -> int:
        encoded = self.tokenizer(
            "AC",
            add_special_tokens=True,
            return_special_tokens_mask=True,
            truncation=False,
        )

        mask = list(encoded["special_tokens_mask"])

        prefix = 0
        for is_special in mask:
            if is_special:
                prefix += 1
            else:
                break

        if prefix < 1:
            raise ValueError("expected a CLS-style prefix token")

        return prefix

    def tokenize(
        self,
        sequences: str | Iterable[str],
    ) -> DnaTokenizedSequences:
        """Tokenize DNA and retain token-to-nucleotide spans."""

        seqs = [sequences] if isinstance(sequences, str) else list(sequences)
        normalized = [normalize_sequence(sequence) for sequence in seqs]

        if not normalized:
            raise ValueError("no sequences")

        too_long = [
            len(sequence)
            for sequence in normalized
            if len(sequence) > self.max_nucleotides
        ]

        if too_long:
            raise ValueError(
                f"sequence length {max(too_long)} exceeds configured maximum "
                f"{self.max_nucleotides}; circuit examples must not be silently truncated"
            )

        encoded = self.tokenizer(
            normalized,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        offset_mapping = encoded["offset_mapping"]
        special_tokens_mask = encoded["special_tokens_mask"]
        attention_mask = encoded["attention_mask"]

        all_spans: list[list[tuple[int, int] | None]] = []

        for row, sequence in enumerate(normalized):
            row_spans: list[tuple[int, int] | None] = []

            for column in range(offset_mapping.shape[1]):
                attended = bool(attention_mask[row, column].item())
                special = bool(special_tokens_mask[row, column].item())
                start, end = offset_mapping[row, column].tolist()

                if not attended or special or end <= start:
                    row_spans.append(None)
                    continue

                if start < 0 or end > len(sequence):
                    raise ValueError(
                        f"token offset {(start, end)} lies outside sequence "
                        f"length {len(sequence)}"
                    )

                if sequence[start:end] == "":
                    raise ValueError("non-special token has an empty nucleotide span")

                row_spans.append((int(start), int(end)))

            covered = [False] * len(sequence)

            for span in row_spans:
                if span is None:
                    continue

                start, end = span
                for position in range(start, end):
                    covered[position] = True

            if not all(covered):
                missing = [index for index, value in enumerate(covered) if not value]
                raise ValueError(
                    f"token offsets do not cover nucleotide positions {missing}"
                )

            all_spans.append(row_spans)

        batch = DnaTokenizedSequences(
            input_ids=encoded["input_ids"],
            attention_mask=attention_mask,
            prefix_tokens=self.prefix_tokens,
            nucleotide_lengths=[len(sequence) for sequence in normalized],
            texts=normalized,
            offset_mapping=offset_mapping,
            special_tokens_mask=special_tokens_mask,
            token_to_nucleotide_spans=all_spans,
        )

        return batch.to(self.device)

    @staticmethod
    def token_indices_for_span(
        batch: DnaTokenizedSequences,
        batch_index: int,
        start: int,
        end: int,
    ) -> list[int]:
        """Return tokens overlapping nucleotide interval ``[start, end)``."""

        if not 0 <= batch_index < len(batch.texts):
            raise IndexError("batch index out of range")

        if start < 0 or end <= start:
            raise ValueError("expected a non-empty nucleotide interval")

        if end > batch.nucleotide_lengths[batch_index]:
            raise ValueError("nucleotide interval exceeds sequence length")

        indices = []

        for token_index, span in enumerate(
            batch.token_to_nucleotide_spans[batch_index]
        ):
            if span is None:
                continue

            token_start, token_end = span

            if token_start < end and token_end > start:
                indices.append(token_index)

        return indices

    @staticmethod
    def tokens_covering_nucleotide(
        batch: DnaTokenizedSequences,
        batch_index: int,
        position: int,
    ) -> list[int]:
        """Return all token indices whose span contains a nucleotide."""

        return DnaBertAdapter.token_indices_for_span(
            batch,
            batch_index,
            position,
            position + 1,
        )

    @staticmethod
    def biological_token_mask(
        batch: DnaTokenizedSequences,
    ) -> torch.Tensor:
        """Boolean mask containing only tokens aligned to nucleotides."""

        return (
            batch.attention_mask.bool()
            & ~batch.special_tokens_mask.bool()
            & (batch.offset_mapping[..., 1] > batch.offset_mapping[..., 0])
        )

    @staticmethod
    def _repad_layer(
        layer: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_size: int,
    ) -> torch.Tensor:
        """Restore an unpadded DNABERT layer to ``[batch, sequence, hidden]``."""

        if layer.ndim == 3:
            expected = (
                attention_mask.shape[0],
                attention_mask.shape[1],
                hidden_size,
            )

            if tuple(layer.shape) != expected:
                raise ValueError(
                    f"unexpected padded layer shape {tuple(layer.shape)}; "
                    f"expected {expected}"
                )

            return layer

        if layer.ndim != 2:
            raise ValueError(
                f"DNABERT layer must be rank 2 or 3, got {tuple(layer.shape)}"
            )

        total_tokens = int(attention_mask.sum().item())

        if tuple(layer.shape) != (total_tokens, hidden_size):
            raise ValueError(
                f"unexpected unpadded layer shape {tuple(layer.shape)}; "
                f"expected {(total_tokens, hidden_size)}"
            )

        batch_size, sequence_length = attention_mask.shape

        padded = layer.new_zeros(
            batch_size,
            sequence_length,
            hidden_size,
        )

        padded[attention_mask.bool()] = layer
        return padded

    def forward(
        self,
        batch: DnaTokenizedSequences,
        *,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        grad: bool = False,
    ) -> GenomicModelOutput:
        """Run DNABERT and normalize its custom tuple output."""

        if output_attentions:
            raise NotImplementedError(
                "DNABERT remote code does not expose standard attention tensors; "
                "capture attention internals with DNABERT-specific hooks instead"
            )

        context = torch.enable_grad() if grad else torch.no_grad()

        with context:
            if output_hidden_states:
                raw = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    output_all_encoded_layers=True,
                )

                if not isinstance(raw, tuple) or len(raw) != 2:
                    raise TypeError(
                        "expected DNABERT all-layer output to be a 2-item tuple"
                    )

                encoded_layers, _pooled_output = raw

                if not isinstance(encoded_layers, (list, tuple)):
                    raise TypeError(
                        "expected DNABERT encoded layers to be a list or tuple"
                    )

                hidden_states = tuple(
                    self._repad_layer(
                        layer,
                        batch.attention_mask,
                        self.hidden_size,
                    )
                    for layer in encoded_layers
                )

                if len(hidden_states) != self.num_layers:
                    raise ValueError(
                        f"expected {self.num_layers} encoded layers, "
                        f"received {len(hidden_states)}"
                    )

                last_hidden = hidden_states[-1]

            else:
                raw = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    output_all_encoded_layers=False,
                )

                if not isinstance(raw, tuple) or len(raw) != 2:
                    raise TypeError(
                        "expected DNABERT final-layer output to be a 2-item tuple"
                    )

                last_hidden, _pooled_output = raw

                if not torch.is_tensor(last_hidden):
                    raise TypeError("DNABERT final hidden state is not a tensor")

                hidden_states = None

        return GenomicModelOutput(
            last_hidden_state=last_hidden,
            hidden_states=hidden_states,
            attentions=None,
            logits=None,
            raw=raw,
        )

    def masked_mean(
        self,
        output: GenomicModelOutput,
        batch: DnaTokenizedSequences,
    ) -> torch.Tensor:
        """Mean-pool representations over biological tokens only."""

        mask = self.biological_token_mask(batch)
        weights = mask.to(output.last_hidden_state.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)

        return (output.last_hidden_state * weights).sum(dim=1) / denominator

    def metadata(self) -> DnaBertMetadata:
        if hasattr(self.config, "to_dict"):
            config_dict = self.config.to_dict()
        else:
            config_dict = vars(self.config)

        config_text = json.dumps(
            config_dict,
            sort_keys=True,
            default=str,
        )

        return DnaBertMetadata(
            source=self.source,
            revision=self.revision,
            prefix_tokens=self.prefix_tokens,
            max_nucleotides=self.max_nucleotides,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            parameter_count=sum(
                parameter.numel()
                for parameter in self.model.parameters()
            ),
            config_sha256=hashlib.sha256(
                config_text.encode("utf-8")
            ).hexdigest(),
            uses_variable_length_tokens=True,
            pooler_disabled=self.pooler_disabled,
        )
