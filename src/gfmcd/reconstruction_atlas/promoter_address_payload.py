"""Execution helpers for promoter K/V and residual-payload decomposition."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from gfmcd.adapters.dnabert2 import (
    Dnabert2PromoterBundle,
    Head,
    QkvOperation,
    capture_cls_residual_writes,
    promoter_margin_tensor,
    qkv_operation_effects_for_batch,
    tokenize_promoter_pairs,
)

Pair = tuple[str, str, str]


def scan_qkv_operations(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    head_pairs: Mapping[str, Sequence[Head]],
    operations: Sequence[QkvOperation],
    *,
    source_token_count: int,
    seed: int,
    batch_pairs: int,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, float | str]]:
    """Run fixed-dose K/V surgeries for candidate and matched head pairs."""

    import torch

    if batch_pairs < 1 or not head_pairs or not operations:
        raise ValueError("QKV scan requires batches, head pairs, and operations")
    output: list[dict[str, float | str]] = []
    for batch_index, chunk in enumerate(_chunks(pairs, batch_pairs), start=1):
        batch = tokenize_promoter_pairs(bundle, chunk)
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, batch).detach()
        for pair_name, heads in head_pairs.items():
            for operation in operations:
                effects = qkv_operation_effects_for_batch(
                    bundle,
                    batch,
                    heads,
                    operation=operation,
                    source_token_count=source_token_count,
                    seed=seed,
                    native_margins=native,
                )
                for pair_index, pair_id in enumerate(batch.pair_ids):
                    output.append(
                        {
                            "pair_id": pair_id,
                            "head_pair": pair_name,
                            "operation": operation,
                            "target_loss": effects["target_loss"][pair_index],
                            "control_loss": effects["control_loss"][pair_index],
                            "target_specific_loss": effects["target_specific_loss"][
                                pair_index
                            ],
                        }
                    )
        if progress:
            progress(f"QKV-operation batch {batch_index}")
    return output


def capture_residual_payload_rows(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    layer: int,
    candidate_heads: Sequence[Head],
    batch_pairs: int,
    progress: Callable[[str], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Capture candidate-pair residual vectors and all same-layer head norms."""

    import torch

    all_heads = tuple(
        (layer, head) for head in range(int(bundle.config.num_attention_heads))
    )
    candidate = tuple(candidate_heads)
    if (
        len(candidate) != 2
        or len(set(candidate)) != 2
        or any(head[0] != layer for head in candidate)
    ):
        raise ValueError("residual payload candidate must be two heads in one layer")
    joint_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    for batch_index, chunk in enumerate(_chunks(pairs, batch_pairs), start=1):
        batch = tokenize_promoter_pairs(bundle, chunk)
        margins, writes = capture_cls_residual_writes(bundle, batch, all_heads)
        for pair_index, pair_id in enumerate(batch.pair_ids):
            target_row, control_row = 2 * pair_index, 2 * pair_index + 1
            target_joint = sum(writes[head][target_row] for head in candidate)
            control_joint = sum(writes[head][control_row] for head in candidate)
            delta = target_joint - control_joint
            joint_rows.append(
                {
                    "pair_id": pair_id,
                    "target_margin": float(margins[target_row]),
                    "control_margin": float(margins[control_row]),
                    "target_write_norm": float(torch.linalg.vector_norm(target_joint)),
                    "control_write_norm": float(
                        torch.linalg.vector_norm(control_joint)
                    ),
                    "target_control_cosine": _cosine(target_joint, control_joint),
                    "delta_norm": float(torch.linalg.vector_norm(delta)),
                    "delta_vector": [float(value) for value in delta.cpu()],
                }
            )
            for head in all_heads:
                target = writes[head][target_row]
                control = writes[head][control_row]
                head_rows.append(
                    {
                        "pair_id": pair_id,
                        "layer": head[0],
                        "head": head[1],
                        "target_write_norm": float(torch.linalg.vector_norm(target)),
                        "control_write_norm": float(torch.linalg.vector_norm(control)),
                        "target_control_cosine": _cosine(target, control),
                        "delta_norm": float(torch.linalg.vector_norm(target - control)),
                    }
                )
        if progress:
            progress(f"residual-payload batch {batch_index}")
    return {"joint_rows": joint_rows, "head_rows": head_rows}


def _chunks(values: Sequence[Pair], size: int) -> list[Sequence[Pair]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _cosine(left: Any, right: Any) -> float:
    import torch

    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0:
        return 0.0
    return float(torch.dot(left, right) / denominator)
