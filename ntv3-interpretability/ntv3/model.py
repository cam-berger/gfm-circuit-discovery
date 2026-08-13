"""Adapter around Nucleotide Transformer v3 8M (InstaDeepAI, gated, remote code).

NTv3 is architecturally unrelated to NTv2 (the earlier ``ntv2-interpretability/``
port, removed): NTv2 was a flat 12-layer ESM-style encoder; NTv3 is a **U-Net**
-- a convolutional downsampling tower feeds a small Transformer bottleneck,
which a matched deconvolutional tower upsamples back to single-nucleotide
resolution. Every fact below was verified against the real checkpoint
(``InstaDeepAI/NTv3_8M_pre``, gated -- requires an authenticated HF session,
see ``README.md``), not assumed from the model card:

* **Tokenization is single-nucleotide, 1:1.** ``NTv3Tokenizer._tokenize``
  maps one input character to one token; unlike NTv2's greedy 6-mers or
  DNABERT-2's BPE, there is no offset-mapping reconstruction needed here.
* **No special tokens are added.** ``build_inputs_with_special_tokens``
  returns the input unchanged -- confirmed by tokenizing a 128nt probe and
  getting exactly 128 ``input_ids``, not 129 or 130. There is no leading
  ``<cls>`` or trailing ``<eos>``, unlike every other adapter in this repo.
* **The input length must be a multiple of ``2**num_downsamples``** (128 for
  this checkpoint: ``num_downsamples=7``). Shorter/misaligned sequences must
  be padded before tokenizing; see ``pad_to_multiple``.
* **``attention_mask`` is accepted but silently ignored.** ``NTv3PreTrained.
  forward`` takes an ``attention_mask`` kwarg but never forwards it into
  ``Core.forward``, which has no such parameter at all -- verified by reading
  the real ``modeling_ntv3_pretrained.py`` pulled from the gated repo. There
  is no way to mask padding out of the convolution towers' receptive field.
  This adapter therefore never relies on an attention mask: pooling instead
  tracks each sequence's real-nucleotide span explicitly (``content_span``)
  and only averages over that span at whatever resolution a given depth is.
* **``hidden_states`` has ``2*num_downsamples + num_layers`` entries** (16 for
  this checkpoint: 7 conv + 2 transformer + 7 deconv), at resolutions
  ``L, L/2, ..., L/2**num_downsamples`` (conv, descending), then
  ``L/2**num_downsamples`` repeated ``num_layers`` times (the Transformer
  bottleneck), then ``L/2**(num_downsamples-1), ..., L`` (deconv, ascending
  back to full nucleotide resolution) -- verified directly against a 128nt
  forward pass's returned tensor shapes, not derived from the source alone.
* **The Transformer bottleneck degenerates at short inputs.** At the 128nt
  pad length this study currently uses, the bottleneck resolution is
  ``128/2**7 == 1`` token: attention has nothing to attend to but itself.
  This adapter exposes bottleneck resolution via ``resolution_schedule`` so
  callers can detect this rather than silently getting a vacuous result.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch


def normalize_sequence(sequence: str) -> str:
    seq = sequence.upper().replace("U", "T").replace(" ", "")
    bad = set(seq) - set("ACGTN")
    if bad:
        raise ValueError(f"unsupported nucleotide symbols: {sorted(bad)}")
    if not seq:
        raise ValueError("sequence cannot be empty")
    return seq


def pad_to_multiple(seq: str, multiple: int, *, pad_char: str = "N") -> tuple[str, int, int]:
    """Center-pad ``seq`` to the next multiple of ``multiple``.

    Padding is split as evenly as possible between the two ends (extra
    character, if any, goes to the right) so the real window sits at a
    consistent relative position instead of hugging one edge of the conv
    tower's receptive field. Returns ``(padded_seq, left_pad, right_pad)``.
    """

    n = len(seq)
    target = -(-n // multiple) * multiple  # ceil to multiple
    total_pad = target - n
    left = total_pad // 2
    right = total_pad - left
    return pad_char * left + seq + pad_char * right, left, right


def resolution_schedule(padded_length: int, num_downsamples: int, num_layers: int) -> list[int]:
    """Sequence length at each ``hidden_states`` entry, in forward order:
    conv tower (descending), Transformer bottleneck (constant), deconv tower
    (ascending). Verified against a real forward pass in ``Ntv3Adapter``;
    see this module's docstring."""

    if padded_length % (2 ** num_downsamples) != 0:
        raise ValueError(
            f"padded_length {padded_length} is not a multiple of "
            f"2**num_downsamples ({2 ** num_downsamples})"
        )
    conv = [padded_length // (2 ** i) for i in range(num_downsamples)]
    bottleneck_res = padded_length // (2 ** num_downsamples)
    transformer = [bottleneck_res] * num_layers
    deconv = [padded_length // (2 ** i) for i in range(num_downsamples - 1, -1, -1)]
    return conv + transformer + deconv


def stage_labels(num_downsamples: int, num_layers: int) -> list[str]:
    """Human-readable label per ``hidden_states`` entry, same order as
    ``resolution_schedule``."""

    conv = [f"conv{i + 1}" for i in range(num_downsamples)]
    transformer = [f"xformer{j + 1}" for j in range(num_layers)]
    deconv = [f"deconv{k + 1}" for k in range(num_downsamples)]
    return conv + transformer + deconv


@dataclass
class Ntv3TokenizedSequences:
    input_ids: torch.Tensor
    texts: list[str]  # padded texts actually fed to the model
    nucleotide_lengths: list[int]  # true (unpadded) lengths
    content_spans: list[tuple[int, int]]  # [start, end) of real nucleotides in padded coords
    padded_length: int

    def to(self, device: str | torch.device) -> "Ntv3TokenizedSequences":
        return Ntv3TokenizedSequences(
            input_ids=self.input_ids.to(device),
            texts=self.texts,
            nucleotide_lengths=self.nucleotide_lengths,
            content_spans=self.content_spans,
            padded_length=self.padded_length,
        )


@dataclass
class Ntv3ModelOutput:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None
    attentions: tuple[torch.Tensor, ...] | None
    raw: Any


@dataclass(frozen=True)
class Ntv3Metadata:
    source: str
    hidden_size: int
    num_heads: int
    num_layers: int
    num_downsamples: int
    parameter_count: int
    config_sha256: str


DEFAULT_MODEL = "InstaDeepAI/NTv3_8M_pre"


class Ntv3Adapter:
    """Circuit-facing adapter for NTv3. Deliberately self-contained (does not
    depend on ``gfmcd.adapters``), mirroring the rest of this repo's
    model-specific study adapters."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        source: str = DEFAULT_MODEL,
        device: str | torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.source = source
        self.config = getattr(model, "config", None)
        if self.config is None:
            raise ValueError("NTv3 model must expose a Hugging Face-style config")

        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.num_downsamples = int(self.config.num_downsamples)
        self.num_transformer_layers = int(self.config.num_layers)
        self.seq_len_multiple = 2 ** self.num_downsamples
        self._verify_tokenizer_adds_no_special_tokens()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = DEFAULT_MODEL,
        *,
        device: str | torch.device | None = None,
    ) -> "Ntv3Adapter":
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        return cls(model, tokenizer, source=str(model_name_or_path), device=device)

    # -- architecture -----------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.config, "embed_dim"))

    @property
    def num_heads(self) -> int:
        return int(getattr(self.config, "attention_heads"))

    @property
    def conv_tower_blocks(self) -> tuple[Any, ...]:
        return tuple(self.model.core.conv_tower_blocks)

    @property
    def transformer_blocks(self) -> tuple[Any, ...]:
        return tuple(self.model.core.transformer_blocks)

    @property
    def deconv_tower_blocks(self) -> tuple[Any, ...]:
        return tuple(self.model.core.deconv_tower_blocks)

    def resolution_schedule(self, padded_length: int) -> list[int]:
        return resolution_schedule(padded_length, self.num_downsamples, self.num_transformer_layers)

    def stage_labels(self) -> list[str]:
        return stage_labels(self.num_downsamples, self.num_transformer_layers)

    # -- tokenization -------------------------------------------------------

    def tokenize(self, sequences: str | Iterable[str], *, pad_length: int | None = None) -> Ntv3TokenizedSequences:
        seqs = [sequences] if isinstance(sequences, str) else list(sequences)
        normalized = [normalize_sequence(s) for s in seqs]
        if not normalized:
            raise ValueError("no sequences")

        if pad_length is None:
            longest = max(len(s) for s in normalized)
            pad_length = -(-longest // self.seq_len_multiple) * self.seq_len_multiple
        elif pad_length % self.seq_len_multiple != 0:
            raise ValueError(
                f"pad_length {pad_length} is not a multiple of {self.seq_len_multiple} "
                f"(2**num_downsamples)"
            )

        padded_texts: list[str] = []
        content_spans: list[tuple[int, int]] = []
        for seq in normalized:
            if len(seq) > pad_length:
                raise ValueError(f"sequence of length {len(seq)} exceeds pad_length {pad_length}")
            total_pad = pad_length - len(seq)
            left = total_pad // 2
            padded_texts.append("N" * left + seq + "N" * (total_pad - left))
            content_spans.append((left, left + len(seq)))

        encoded = self.tokenizer(
            padded_texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        if input_ids.shape[1] != pad_length:
            raise ValueError(
                f"expected {pad_length} tokens after padding, tokenizer produced "
                f"{input_ids.shape[1]}; NTv3Tokenizer's no-special-token assumption "
                f"may no longer hold"
            )

        batch = Ntv3TokenizedSequences(
            input_ids=input_ids,
            texts=padded_texts,
            nucleotide_lengths=[len(s) for s in normalized],
            content_spans=content_spans,
            padded_length=pad_length,
        )
        return batch.to(self.device)

    def _verify_tokenizer_adds_no_special_tokens(self) -> None:
        probe_len = self.seq_len_multiple
        probe = self.tokenize("ACGT" * (probe_len // 4))
        if probe.input_ids.shape[1] != probe_len:
            raise ValueError(
                f"NTv3 tokenizer produced {probe.input_ids.shape[1]} tokens for a "
                f"{probe_len}nt probe with no padding needed; expected {probe_len} "
                "(no special tokens); special-token layout assumption is wrong"
            )

    # -- forward / pooling --------------------------------------------------

    def forward(
        self,
        batch: Ntv3TokenizedSequences,
        *,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        grad: bool = False,
    ) -> Ntv3ModelOutput:
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            raw = self.model(
                input_ids=batch.input_ids,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True,
            )
        hidden_states = getattr(raw, "hidden_states", None)
        attentions = getattr(raw, "attentions", None)
        return Ntv3ModelOutput(
            logits=raw.logits,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            attentions=tuple(attentions) if attentions is not None else None,
            raw=raw,
        )

    def content_span_at_depth(
        self, batch: Ntv3TokenizedSequences, batch_index: int, depth_resolution: int
    ) -> tuple[int, int]:
        """Map a sequence's real-nucleotide span from padded-input coordinates
        down to the compressed coordinates of a given ``hidden_states`` depth.
        Falls back to the full available range if the mapped span degenerates
        to empty (e.g. at the Transformer bottleneck, where resolution may be
        a single token covering the whole padded window)."""

        factor = batch.padded_length // depth_resolution
        start, end = batch.content_spans[batch_index]
        cs = start // factor
        ce = max(cs + 1, -(-end // factor))
        ce = min(ce, depth_resolution)
        cs = min(cs, ce - 1) if ce > 0 else 0
        if ce <= cs:
            return 0, depth_resolution
        return cs, ce

    def masked_mean_at_depth(self, hidden: torch.Tensor, batch: Ntv3TokenizedSequences) -> torch.Tensor:
        """Mean-pool a ``hidden_states`` tensor over each row's real-nucleotide
        span, mapped to that tensor's own resolution. ``hidden`` must be one
        entry from ``Ntv3ModelOutput.hidden_states`` (shape ``[B, L_depth, H]``)."""

        depth_resolution = hidden.shape[1]
        pooled = []
        for i in range(hidden.shape[0]):
            cs, ce = self.content_span_at_depth(batch, i, depth_resolution)
            pooled.append(hidden[i, cs:ce, :].mean(dim=0))
        return torch.stack(pooled, dim=0)

    def metadata(self) -> Ntv3Metadata:
        cfg = self.config.to_dict() if hasattr(self.config, "to_dict") else vars(self.config)
        cfg_text = json.dumps(cfg, sort_keys=True, default=str)
        return Ntv3Metadata(
            source=self.source,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_transformer_layers,
            num_downsamples=self.num_downsamples,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            config_sha256=hashlib.sha256(cfg_text.encode()).hexdigest(),
        )
