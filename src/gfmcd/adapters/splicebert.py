"""Hugging Face adapter for the official or compatible SpliceBERT checkpoints.

The official SpliceBERT repository stores a standard BERT-style tokenizer and
model directory.  It expects whitespace-separated nucleotides.  Compatible
ports such as MultiMolecule accept raw RNA strings.  ``tokenizer_mode='auto'``
validates both forms and selects one that yields exactly one token per
nucleotide; it never silently accepts k-mer or unknown-token tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal

import torch

from gfmcd.adapters.base import GenomicModelOutput, TokenizedSequences

TokenizerMode = Literal["auto", "spaced", "raw"]


def normalize_sequence(sequence: str) -> str:
    seq = sequence.upper().replace("U", "T")
    bad = set(seq) - set("ACGTN")
    if bad:
        raise ValueError(f"unsupported nucleotide symbols: {sorted(bad)}")
    if not seq:
        raise ValueError("sequence cannot be empty")
    return seq


def _walk_attr(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _find_first(obj: Any, paths: Iterable[str]) -> Any:
    for path in paths:
        value = _walk_attr(obj, path)
        if value is not None:
            return value
    raise AttributeError(f"none of the expected model paths exist: {list(paths)}")


@dataclass
class SpliceBertMetadata:
    source: str
    tokenizer_mode: str
    prefix_tokens: int
    max_nucleotides: int
    hidden_size: int
    num_layers: int
    num_heads: int
    parameter_count: int
    config_sha256: str


class SpliceBertAdapter:
    """Thin adapter that exposes stable token and layer conventions."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        source: str = "in-memory",
        tokenizer_mode: TokenizerMode = "auto",
        device: str | torch.device | None = None,
        max_nucleotides: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.source = source
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        self.config = getattr(model, "config", None)
        if self.config is None:
            raise ValueError("SpliceBERT model must expose a Hugging Face-style config")
        max_positions = int(getattr(self.config, "max_position_embeddings", 1024))
        self.max_nucleotides = max_nucleotides or max_positions - 2
        self.tokenizer_mode = self._resolve_tokenizer_mode(tokenizer_mode)
        self.prefix_tokens = self._infer_prefix_tokens(self.tokenizer_mode)
        if self.prefix_tokens < 1:
            raise ValueError("expected a [CLS]-style prefix special token")
        # Resolve now so architecture incompatibility fails before a long run.
        self._layers = self._resolve_encoder_layers()
        self._attention_modules = self._resolve_attention_modules()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        tokenizer_name_or_path: str | Path | None = None,
        tokenizer_mode: TokenizerMode = "auto",
        device: str | torch.device | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_class: Literal["masked_lm", "encoder"] = "masked_lm",
        **kwargs: Any,
    ) -> "SpliceBertAdapter":
        try:
            from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("install model dependencies: uv sync --extra splicebert") from exc
        model_path = str(model_name_or_path)
        tok_path = str(tokenizer_name_or_path or model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        loader = AutoModelForMaskedLM if model_class == "masked_lm" else AutoModel
        model = loader.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            **kwargs,
        )
        return cls(
            model,
            tokenizer,
            source=model_path,
            tokenizer_mode=tokenizer_mode,
            device=device,
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
        return tuple(self._attention_modules)

    def _resolve_encoder_layers(self) -> list[Any]:
        layers = _find_first(
            self.model,
            (
                "bert.encoder.layer",
                "base_model.encoder.layer",
                "base_model.bert.encoder.layer",
                "splicebert.encoder.layer",
                "encoder.layer",
            ),
        )
        return list(layers)

    def _resolve_attention_modules(self) -> list[Any]:
        modules: list[Any] = []
        for i, layer in enumerate(self._layers):
            module = _find_first(layer, ("attention.self", "self_attn", "attention"))
            if not hasattr(module, "register_forward_hook"):
                raise TypeError(f"layer {i} attention module is not hookable")
            modules.append(module)
        return modules

    def _text(self, sequence: str, mode: str) -> str:
        seq = normalize_sequence(sequence)
        return " ".join(seq) if mode == "spaced" else seq

    def _token_ids(self, sequence: str, mode: str) -> list[int]:
        encoded = self.tokenizer(
            self._text(sequence, mode),
            add_special_tokens=True,
            truncation=False,
        )
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return list(ids)

    def _is_single_nt_mode(self, mode: str) -> bool:
        test = "ACGTN"
        ids = self._token_ids(test, mode)
        try:
            mask = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            nonspecial = [token for token, special in zip(ids, mask) if not special]
        except Exception:
            special_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
            nonspecial = ids[1 : len(ids) - max(1, special_count - 1)]
        if len(nonspecial) != len(test):
            return False
        unk = getattr(self.tokenizer, "unk_token_id", None)
        return unk is None or all(token != unk for token in nonspecial)

    def _resolve_tokenizer_mode(self, requested: TokenizerMode) -> str:
        if requested in {"spaced", "raw"}:
            if not self._is_single_nt_mode(requested):
                raise ValueError(f"tokenizer_mode={requested!r} is not one-token-per-nucleotide")
            return requested
        valid = [mode for mode in ("spaced", "raw") if self._is_single_nt_mode(mode)]
        if not valid:
            raise ValueError("tokenizer is not one-token-per-nucleotide in spaced or raw mode")
        # Prefer the official representation when both are accepted.
        return "spaced" if "spaced" in valid else valid[0]

    def _infer_prefix_tokens(self, mode: str) -> int:
        ids = self._token_ids("AC", mode)
        mask = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
        prefix = 0
        for special in mask:
            if special:
                prefix += 1
            else:
                break
        nonspecial = sum(1 for x in mask if not x)
        if nonspecial != 2:
            raise ValueError("failed to verify one-token-per-nucleotide mapping")
        return prefix

    def tokenize(self, sequences: str | Iterable[str]) -> TokenizedSequences:
        seqs = [sequences] if isinstance(sequences, str) else list(sequences)
        normalized = [normalize_sequence(x) for x in seqs]
        if not normalized:
            raise ValueError("no sequences")
        too_long = [len(x) for x in normalized if len(x) > self.max_nucleotides]
        if too_long:
            raise ValueError(
                f"sequence length {max(too_long)} exceeds max {self.max_nucleotides}; "
                "do not silently truncate circuit examples"
            )
        texts = [self._text(x, self.tokenizer_mode) for x in normalized]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        batch = TokenizedSequences(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            prefix_tokens=self.prefix_tokens,
            nucleotide_lengths=[len(x) for x in normalized],
            texts=texts,
        )
        # Validate every unpadded token count.  This catches a tokenizer that
        # unexpectedly changed behavior across inputs.
        special = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        counts = batch.attention_mask.sum(-1).tolist()
        for n, count in zip(batch.nucleotide_lengths, counts):
            if int(count) != n + special:
                raise ValueError(
                    f"tokenizer produced {count-special} non-special tokens for {n} nucleotides"
                )
        return batch.to(self.device)

    def nucleotide_to_token(self, position: int) -> int:
        if position < 0:
            raise ValueError("nucleotide position must be non-negative")
        return position + self.prefix_tokens

    def forward(
        self,
        batch: TokenizedSequences,
        *,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        grad: bool = False,
    ) -> GenomicModelOutput:
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            raw = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True,
            )
        hidden_states = getattr(raw, "hidden_states", None)
        last_hidden = getattr(raw, "last_hidden_state", None)
        if last_hidden is None and hidden_states:
            last_hidden = hidden_states[-1]
        if last_hidden is None:
            # MaskedLM outputs from BERT normally expose hidden_states only when
            # requested; fail rather than accidentally using vocabulary logits.
            raise AttributeError("model output contains no hidden states")
        return GenomicModelOutput(
            last_hidden_state=last_hidden,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            attentions=tuple(raw.attentions) if getattr(raw, "attentions", None) is not None else None,
            logits=getattr(raw, "logits", None),
            raw=raw,
        )

    def metadata(self) -> SpliceBertMetadata:
        if hasattr(self.config, "to_dict"):
            cfg = self.config.to_dict()
        else:
            cfg = vars(self.config)
        cfg_text = json.dumps(cfg, sort_keys=True, default=str)
        return SpliceBertMetadata(
            source=self.source,
            tokenizer_mode=self.tokenizer_mode,
            prefix_tokens=self.prefix_tokens,
            max_nucleotides=self.max_nucleotides,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            config_sha256=hashlib.sha256(cfg_text.encode()).hexdigest(),
        )
