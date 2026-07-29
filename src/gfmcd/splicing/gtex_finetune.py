"""Validated encoder transplants for the published SpliceAI-GTEx fine-tunes.

The checkpoints published with ``chenkenbio/SpliceBERT-analysis`` are raw
``BertForTokenClassification`` state dictionaries.  They contain the complete
SpliceBERT encoder (``bert.*``) and a three-class splice-site head
(``classifier.*``), but no masked-LM head.

The coupling study is defined on the native masked-LM margin.  To hold that
readout family fixed while measuring the encoder shift, this module installs
the published encoder into the original ``BertForMaskedLM`` model:

* every encoder tensor must be present, finite, and shape-compatible;
* the only non-encoder tensors allowed in the publication are the documented
  three-class classifier weight and bias;
* the base MLM transform and output bias are retained;
* the decoder remains tied to the transplanted nucleotide embeddings, matching
  the original BERT architecture.

This is a derived analysis checkpoint, not a checkpoint released by the model
authors.  The returned report makes that boundary explicit and records enough
detail to audit the transplant.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import torch

from gfmcd.provenance import sha256_file

ENCODER_PREFIX = "bert."
CLASSIFIER_KEYS = frozenset({"classifier.weight", "classifier.bias"})
SEED_PATTERN = re.compile(r"(?:^|[._-])(20(?:20|21|22|23|24))(?:[._-]|$)")


class GtexCheckpointError(ValueError):
    """The published checkpoint cannot be transplanted without ambiguity."""


@dataclass(frozen=True)
class TensorShift:
    """Parameter-space displacement for one encoder tensor."""

    name: str
    shape: tuple[int, ...]
    numel: int
    base_l2: float
    tuned_l2: float
    delta_l2: float
    relative_delta_l2: float
    cosine: float
    max_abs_delta: float


@dataclass(frozen=True)
class GtexTransplantReport:
    """Auditable description of a base-MLM / fine-tuned-encoder transplant."""

    derived_checkpoint: bool
    source_task: str
    target_readout: str
    mlm_head_policy: str
    encoder_checkpoint: str
    encoder_checkpoint_sha256: str
    seed: int | None
    encoder_tensor_count: int
    encoder_parameter_count: int
    classifier_keys: tuple[str, ...]
    classifier_weight_shape: tuple[int, ...]
    classifier_bias_shape: tuple[int, ...]
    missing_encoder_keys: tuple[str, ...]
    extra_encoder_keys: tuple[str, ...]
    extra_nonencoder_keys: tuple[str, ...]
    tensor_shifts: tuple[TensorShift, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GtexNativeClassifierReport:
    """Auditable description of a strict native token-classifier load."""

    source_task: str
    checkpoint: str
    checkpoint_sha256: str
    seed: int | None
    tensor_count: int
    parameter_count: int
    class_labels: tuple[str, ...]
    missing_keys: tuple[str, ...]
    extra_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def checkpoint_seed(path: str | Path) -> int | None:
    """Extract the documented 2020--2024 training seed from a checkpoint name."""

    match = SEED_PATTERN.search(Path(path).name)
    return int(match.group(1)) if match else None


def _state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Normalize the published raw state dict without enabling pickle objects."""

    if not isinstance(payload, Mapping):
        raise GtexCheckpointError(
            f"expected a tensor mapping, found {type(payload).__name__}"
        )
    if "model" in payload and isinstance(payload["model"], Mapping):
        payload = payload["model"]
    state: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise GtexCheckpointError(
                "checkpoint must contain only string-to-tensor state-dict entries"
            )
        state[key] = value
    return state


def load_gtex_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a published checkpoint using PyTorch's tensor-only safe loader."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return _state_dict(payload)


def validate_gtex_token_classifier_state(
    model: Any,
    state: Mapping[str, torch.Tensor],
    *,
    checkpoint_path: str | Path,
) -> GtexNativeClassifierReport:
    """Require the published state to exactly cover a three-class token classifier."""

    expected_state = model.state_dict()
    expected = set(expected_state)
    observed = set(state)
    missing = tuple(sorted(expected - observed))
    extra = tuple(sorted(observed - expected))
    if missing or extra:
        raise GtexCheckpointError(
            "native classifier key mismatch: "
            f"missing={list(missing)}, extra={list(extra)}"
        )

    for name in sorted(expected):
        expected_tensor = expected_state[name]
        observed_tensor = state[name]
        if tuple(observed_tensor.shape) != tuple(expected_tensor.shape):
            raise GtexCheckpointError(
                f"{name} has shape {tuple(observed_tensor.shape)}, "
                f"expected {tuple(expected_tensor.shape)}"
            )
        if not torch.isfinite(observed_tensor).all():
            raise GtexCheckpointError(f"{name} contains non-finite values")

    checkpoint = Path(checkpoint_path)
    return GtexNativeClassifierReport(
        source_task="SpliceAI-GTEx three-class token classification",
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha256_file(checkpoint),
        seed=checkpoint_seed(checkpoint),
        tensor_count=len(expected),
        parameter_count=sum(tensor.numel() for tensor in expected_state.values()),
        class_labels=("no_splice_site", "donor", "acceptor"),
        missing_keys=missing,
        extra_keys=extra,
    )


def load_gtex_token_classifier(
    base_model_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    attn_implementation: str | None = None,
) -> tuple[Any, Any, GtexNativeClassifierReport]:
    """Load the authors' native three-class model with a strict tensor contract."""

    from transformers import (
        AutoConfig,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    base_path = str(Path(base_model_path).expanduser().resolve())
    config = AutoConfig.from_pretrained(base_path)
    config.num_labels = 3
    config.id2label = {0: "no_splice_site", 1: "donor", 2: "acceptor"}
    config.label2id = {value: key for key, value in config.id2label.items()}
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    model = AutoModelForTokenClassification.from_config(config)
    state = load_gtex_state_dict(checkpoint_path)
    report = validate_gtex_token_classifier_state(
        model,
        state,
        checkpoint_path=checkpoint_path,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    return model, tokenizer, report


def _tensor_shift(name: str, base: torch.Tensor, tuned: torch.Tensor) -> TensorShift:
    b = base.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    t = tuned.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    delta = t - b
    base_l2 = float(torch.linalg.vector_norm(b))
    tuned_l2 = float(torch.linalg.vector_norm(t))
    delta_l2 = float(torch.linalg.vector_norm(delta))
    denom = max(base_l2, torch.finfo(torch.float64).eps)
    cos_denom = max(base_l2 * tuned_l2, torch.finfo(torch.float64).eps)
    return TensorShift(
        name=name,
        shape=tuple(base.shape),
        numel=base.numel(),
        base_l2=base_l2,
        tuned_l2=tuned_l2,
        delta_l2=delta_l2,
        relative_delta_l2=delta_l2 / denom,
        cosine=float(torch.dot(b, t)) / cos_denom,
        max_abs_delta=float(delta.abs().max()) if delta.numel() else 0.0,
    )


def validate_gtex_encoder(
    base_mlm: Any,
    state: Mapping[str, torch.Tensor],
    *,
    checkpoint_path: str | Path,
) -> GtexTransplantReport:
    """Validate exact encoder coverage and summarize the parameter displacement."""

    if not hasattr(base_mlm, "bert"):
        raise GtexCheckpointError("base masked-LM model does not expose a .bert encoder")

    base_encoder = base_mlm.bert.state_dict()
    tuned_encoder = {
        key.removeprefix(ENCODER_PREFIX): value
        for key, value in state.items()
        if key.startswith(ENCODER_PREFIX)
    }
    expected = set(base_encoder)
    observed = set(tuned_encoder)
    missing = tuple(sorted(expected - observed))
    extra_encoder = tuple(sorted(observed - expected))
    nonencoder = set(state) - {ENCODER_PREFIX + key for key in observed}
    extra_nonencoder = tuple(sorted(nonencoder - CLASSIFIER_KEYS))

    if missing or extra_encoder or extra_nonencoder:
        raise GtexCheckpointError(
            "checkpoint key mismatch: "
            f"missing_encoder={list(missing)}, "
            f"extra_encoder={list(extra_encoder)}, "
            f"extra_nonencoder={list(extra_nonencoder)}"
        )
    if set(nonencoder) != set(CLASSIFIER_KEYS):
        raise GtexCheckpointError(
            "published checkpoint must contain exactly classifier.weight and "
            f"classifier.bias outside the encoder; found {sorted(nonencoder)}"
        )

    classifier_weight = state["classifier.weight"]
    classifier_bias = state["classifier.bias"]
    hidden_size = int(getattr(base_mlm.config, "hidden_size"))
    if tuple(classifier_weight.shape) != (3, hidden_size):
        raise GtexCheckpointError(
            f"classifier.weight has shape {tuple(classifier_weight.shape)}, "
            f"expected {(3, hidden_size)}"
        )
    if tuple(classifier_bias.shape) != (3,):
        raise GtexCheckpointError(
            f"classifier.bias has shape {tuple(classifier_bias.shape)}, expected (3,)"
        )

    shifts: list[TensorShift] = []
    for name in sorted(expected):
        base = base_encoder[name]
        tuned = tuned_encoder[name]
        if tuple(base.shape) != tuple(tuned.shape):
            raise GtexCheckpointError(
                f"{name} has shape {tuple(tuned.shape)}, expected {tuple(base.shape)}"
            )
        if not torch.isfinite(tuned).all():
            raise GtexCheckpointError(f"{name} contains non-finite values")
        shifts.append(_tensor_shift(name, base, tuned))

    checkpoint = Path(checkpoint_path)
    return GtexTransplantReport(
        derived_checkpoint=True,
        source_task="SpliceAI-GTEx three-class token classification",
        target_readout="original SpliceBERT masked-LM margin",
        mlm_head_policy=(
            "retain the base MLM transform and output bias; keep the decoder tied "
            "to the transplanted nucleotide embeddings"
        ),
        encoder_checkpoint=str(checkpoint.resolve()),
        encoder_checkpoint_sha256=sha256_file(checkpoint),
        seed=checkpoint_seed(checkpoint),
        encoder_tensor_count=len(expected),
        encoder_parameter_count=sum(base_encoder[name].numel() for name in expected),
        classifier_keys=tuple(sorted(CLASSIFIER_KEYS)),
        classifier_weight_shape=tuple(classifier_weight.shape),
        classifier_bias_shape=tuple(classifier_bias.shape),
        missing_encoder_keys=missing,
        extra_encoder_keys=extra_encoder,
        extra_nonencoder_keys=extra_nonencoder,
        tensor_shifts=tuple(shifts),
    )


def transplant_gtex_encoder(
    base_mlm: Any,
    checkpoint_path: str | Path,
) -> GtexTransplantReport:
    """Install a validated published encoder into an original SpliceBERT MLM."""

    state = load_gtex_state_dict(checkpoint_path)
    report = validate_gtex_encoder(
        base_mlm,
        state,
        checkpoint_path=checkpoint_path,
    )
    encoder_state = {
        key.removeprefix(ENCODER_PREFIX): value
        for key, value in state.items()
        if key.startswith(ENCODER_PREFIX)
    }
    base_mlm.bert.load_state_dict(encoder_state, strict=True)
    if hasattr(base_mlm, "tie_weights"):
        base_mlm.tie_weights()
    return report


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Stable tensor-content digest used by tests and preflight controls."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
