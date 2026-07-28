"""Execution helpers shared by promoter head-atlas calibration and target runs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    Dnabert2PromoterBundle,
    Head,
    all_promoter_heads,
    bidirectional_patch_effects_for_batch,
    capture_cls_attention_bands,
    paired_head_attributions,
    patch_effect_for_batch,
    promoter_margin_tensor,
    tokenize_promoter_pairs,
)

Pair = tuple[str, str, str]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_gue_pairs(
    path: str | Path,
    *,
    count: int,
    offset: int,
    seed: int,
) -> list[Pair]:
    """Select deterministic balanced cross-class pairs from a GUE split."""

    if count < 1 or offset < 0:
        raise ValueError("GUE pair count must be positive and offset nonnegative")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_label = {
        label: sorted(
            (
                str(row["sequence"]).upper()
                for row in rows
                if int(row["label"]) == label
            ),
            key=lambda sequence: hashlib.sha256(
                f"{seed}|promoter-head-atlas|{label}|{sequence}".encode()
            ).hexdigest(),
        )
        for label in (0, 1)
    }
    end = offset + count
    if any(len(values) < end for values in by_label.values()):
        raise ValueError(
            "GUE split cannot supply the requested disjoint head-atlas pairs"
        )
    return [
        (
            f"GUE-head-pair-{offset + index:04d}",
            positive,
            negative,
        )
        for index, (positive, negative) in enumerate(
            zip(by_label[1][offset:end], by_label[0][offset:end])
        )
    ]


def within_class_gue_pairs(cross_class: Sequence[Pair]) -> list[Pair]:
    """Rotate positive and negative sequences to make balanced same-class patches."""

    if len(cross_class) < 2:
        raise ValueError("within-class control requires at least two GUE pairs")
    positive = [pair[1] for pair in cross_class]
    negative = [pair[2] for pair in cross_class]
    output = []
    for index in range(len(cross_class)):
        output.append(
            (
                f"GUE-positive-control-{index:04d}",
                positive[index],
                positive[(index + 1) % len(positive)],
            )
        )
        output.append(
            (
                f"GUE-negative-control-{index:04d}",
                negative[index],
                negative[(index + 1) % len(negative)],
            )
        )
    return output


def scan_promoter_heads(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    patch_batch_pairs: int,
    gradient_batch_pairs: int,
    include_attention: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run native, attribution, exact single-head patch, and source-attention scans."""

    import torch

    if patch_batch_pairs < 1 or gradient_batch_pairs < 1:
        raise ValueError("head-atlas batch sizes must be positive")
    heads = all_promoter_heads(bundle)
    pair_rows: dict[str, dict[str, Any]] = {}
    head_pair_rows: dict[tuple[str, Head], dict[str, Any]] = {}
    for pair_id, _, _ in pairs:
        pair_rows[pair_id] = {"pair_id": pair_id}
        for layer, head in heads:
            head_pair_rows[(pair_id, (layer, head))] = {
                "pair_id": pair_id,
                "layer": layer,
                "head": head,
            }

    for batch_index, chunk in enumerate(_chunks(pairs, gradient_batch_pairs), start=1):
        batch = tokenize_promoter_pairs(bundle, chunk)
        with torch.no_grad():
            margins = promoter_margin_tensor(bundle, batch).detach().cpu()
        attributions = paired_head_attributions(bundle, batch)
        attention = (
            capture_cls_attention_bands(bundle, batch) if include_attention else None
        )
        for pair_index, pair_id in enumerate(batch.pair_ids):
            target_margin = float(margins[2 * pair_index])
            control_margin = float(margins[2 * pair_index + 1])
            pair_rows[pair_id].update(
                {
                    "target_margin": target_margin,
                    "control_margin": control_margin,
                    "clean_enrichment": target_margin - control_margin,
                    "target_token_count": batch.token_counts[2 * pair_index],
                    "control_token_count": batch.token_counts[2 * pair_index + 1],
                }
            )
            for head in heads:
                row = head_pair_rows[(pair_id, head)]
                row["attribution_necessity"] = attributions[head]["necessity"][
                    pair_index
                ]
                row["attribution_sufficiency"] = attributions[head]["sufficiency"][
                    pair_index
                ]
                if attention is not None:
                    row["target_attention"] = {
                        band: values[2 * pair_index]
                        for band, values in attention[head].items()
                    }
                    row["control_attention"] = {
                        band: values[2 * pair_index + 1]
                        for band, values in attention[head].items()
                    }
        if progress:
            progress(f"gradient/attention batch {batch_index}")

    for batch_index, chunk in enumerate(_chunks(pairs, patch_batch_pairs), start=1):
        batch = tokenize_promoter_pairs(bundle, chunk)
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, batch).detach()
        for head_index, head in enumerate(heads, start=1):
            effects = bidirectional_patch_effects_for_batch(
                bundle, batch, head, native_margins=native
            )
            for pair_index, pair_id in enumerate(batch.pair_ids):
                row = head_pair_rows[(pair_id, head)]
                row["necessity"] = effects["necessity"][pair_index]
                row["sufficiency"] = effects["sufficiency"][pair_index]
            if progress and head_index % int(bundle.config.num_attention_heads) == 0:
                progress(
                    f"exact patch batch {batch_index}, "
                    f"layer {head[0] + 1}/{bundle.config.num_hidden_layers}"
                )
    missing = [
        key
        for key, row in head_pair_rows.items()
        if not {
            "attribution_necessity",
            "attribution_sufficiency",
            "necessity",
            "sufficiency",
        }.issubset(row)
    ]
    if missing:
        raise RuntimeError(f"incomplete head scan: {missing[:3]}")
    return {
        "pair_rows": [pair_rows[pair_id] for pair_id, _, _ in pairs],
        "head_pair_rows": [
            head_pair_rows[(pair_id, head)] for pair_id, _, _ in pairs for head in heads
        ],
    }


def evaluate_head_set(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    heads: Iterable[Head],
    *,
    direction: str,
    batch_pairs: int,
) -> list[float]:
    """Evaluate one multi-head intervention over all pairs."""

    effects: list[float] = []
    for chunk in _chunks(pairs, batch_pairs):
        batch = tokenize_promoter_pairs(bundle, chunk)
        effects.extend(
            patch_effect_for_batch(
                bundle,
                batch,
                heads,
                direction=direction,  # type: ignore[arg-type]
            )
        )
    return effects


def native_pair_rows(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    batch_pairs: int,
) -> list[dict[str, float | str | int]]:
    """Measure native pair margins without any internal intervention."""

    import torch

    output = []
    for chunk in _chunks(pairs, batch_pairs):
        batch = tokenize_promoter_pairs(bundle, chunk)
        with torch.no_grad():
            margins = promoter_margin_tensor(bundle, batch).detach().cpu()
        for index, pair_id in enumerate(batch.pair_ids):
            target = float(margins[2 * index])
            control = float(margins[2 * index + 1])
            output.append(
                {
                    "pair_id": pair_id,
                    "target_margin": target,
                    "control_margin": control,
                    "clean_enrichment": target - control,
                    "target_token_count": batch.token_counts[2 * index],
                    "control_token_count": batch.token_counts[2 * index + 1],
                }
            )
    return output


def _chunks(values: Sequence[Pair], size: int) -> list[Sequence[Pair]]:
    return [values[index : index + size] for index in range(0, len(values), size)]
