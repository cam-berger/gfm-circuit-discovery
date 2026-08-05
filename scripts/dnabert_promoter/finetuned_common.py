"""Shared utilities for fine-tuned DNABERT mechanistic experiments."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from gfmcd.dnabert.promoter_classifier import (
    DnaBertPromoterClassifier,
)


DEFAULT_SEED = 42

FINETUNED_ROOT = Path(
    "results/dnabert_promoter/finetuned"
)

SHARED_ROOT = Path(
    "results/dnabert_promoter/shared"
)


def seed_directory(seed: int = DEFAULT_SEED) -> Path:
    return FINETUNED_ROOT / f"seed_{seed}"


def best_checkpoint_directory(
    seed: int = DEFAULT_SEED,
) -> Path:
    return seed_directory(seed) / "best_checkpoint"


def load_manifest(
    seed: int = DEFAULT_SEED,
) -> dict:
    path = (
        best_checkpoint_directory(seed)
        / "model_manifest.json"
    )

    return json.loads(path.read_text())


def load_training_summary(
    seed: int = DEFAULT_SEED,
) -> dict:
    path = seed_directory(seed) / "summary.json"
    return json.loads(path.read_text())


def load_finetuned_classifier(
    seed: int = DEFAULT_SEED,
    device: str = "cuda:0",
) -> DnaBertPromoterClassifier:
    """Strictly load a saved fine-tuned classifier."""

    checkpoint_dir = best_checkpoint_directory(seed)
    manifest = load_manifest(seed)

    model = DnaBertPromoterClassifier(
        model_name=manifest["base_model"],
        num_labels=int(manifest["num_labels"]),
        dropout=float(manifest["dropout"]),
    )

    state = torch.load(
        checkpoint_dir / "model_state_dict.pt",
        map_location="cpu",
        weights_only=True,
    )

    load_result = model.load_state_dict(
        state,
        strict=True,
    )

    if load_result.missing_keys:
        raise RuntimeError(
            f"Missing keys: {load_result.missing_keys}"
        )

    if load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys: "
            f"{load_result.unexpected_keys}"
        )

    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def promoter_margin(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Return promoter logit minus non-promoter logit."""

    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError(
            "Expected logits with shape [batch, 2], "
            f"got {tuple(logits.shape)}"
        )

    return logits[:, 1] - logits[:, 0]


def tokenize_sequences(
    model: DnaBertPromoterClassifier,
    sequences: list[str],
    seed: int = DEFAULT_SEED,
) -> dict[str, torch.Tensor]:
    manifest = load_manifest(seed)

    return model.tokenizer(
        sequences,
        padding="max_length",
        truncation=True,
        max_length=int(manifest["max_length"]),
        return_tensors="pt",
    )
