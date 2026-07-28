"""Accuracy-first causal atlas over promoter CLS head and MLP writes."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from typing import Any

from gfmcd.adapters.dnabert2 import (
    Dnabert2PromoterBundle,
    Head,
    head_label,
    patch_paired_cls_heads,
    patch_paired_cls_mlp_writes,
    patch_paired_final_cls_residual,
    promoter_margin_tensor,
    tokenize_promoter_pairs,
)
from gfmcd.reconstruction_atlas.promoter_assay import (
    binary_classification_metrics,
)
from gfmcd.reconstruction_atlas.statistics import add_one_permutation_pvalue

Pair = tuple[str, str, str]


def all_cls_nodes(
    *,
    layers: int,
    heads_per_layer: int,
) -> tuple[dict[str, Any], ...]:
    """Return the complete, deterministic CLS write-node universe."""

    if layers < 1 or heads_per_layer < 1:
        raise ValueError("node-universe dimensions must be positive")
    return (
        *(
            {
                "node": head_label((layer, head)),
                "node_type": "head",
                "layer": layer,
                "head": head,
            }
            for layer in range(layers)
            for head in range(heads_per_layer)
        ),
        *(
            {
                "node": f"L{layer}MLP",
                "node_type": "mlp",
                "layer": layer,
                "head": None,
            }
            for layer in range(layers)
        ),
    )


def paired_node_accuracy_rows(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    batch_pairs: int,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Score reciprocal target/control exchange for every individual CLS node."""

    import torch

    if not pairs or batch_pairs < 1:
        raise ValueError("node atlas requires pairs and a positive batch size")
    nodes = all_cls_nodes(
        layers=int(bundle.config.num_hidden_layers),
        heads_per_layer=int(bundle.config.num_attention_heads),
    )
    output = []
    for batch_index, start in enumerate(range(0, len(pairs), batch_pairs), start=1):
        chunk = pairs[start : start + batch_pairs]
        batch = tokenize_promoter_pairs(bundle, chunk)
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, batch).detach()
        for node_index, node in enumerate(nodes, start=1):
            with (
                _node_patch(bundle, batch, node, direction="bidirectional_swap"),
                torch.no_grad(),
            ):
                patched = promoter_margin_tensor(bundle, batch).detach()
            if not torch.isfinite(patched).all():
                raise RuntimeError("node exchange produced non-finite margins")
            for pair_index, pair_id in enumerate(batch.pair_ids):
                target_native = float(native[2 * pair_index])
                control_native = float(native[2 * pair_index + 1])
                target_patched = float(patched[2 * pair_index])
                control_patched = float(patched[2 * pair_index + 1])
                target_correctness_loss = int(target_native > 0) - int(
                    target_patched > 0
                )
                control_correctness_loss = int(control_native <= 0) - int(
                    control_patched <= 0
                )
                output.append(
                    {
                        "pair_id": pair_id,
                        **node,
                        "native_target_margin": target_native,
                        "native_control_margin": control_native,
                        "patched_target_margin": target_patched,
                        "patched_control_margin": control_patched,
                        "target_margin_loss": target_native - target_patched,
                        "control_margin_loss": control_patched - control_native,
                        "class_margin_loss": (
                            target_native
                            - target_patched
                            + control_patched
                            - control_native
                        )
                        / 2,
                        "target_correctness_loss": target_correctness_loss,
                        "control_correctness_loss": control_correctness_loss,
                        "accuracy_loss": (
                            target_correctness_loss + control_correctness_loss
                        )
                        / 2,
                    }
                )
            if progress is not None and (
                node_index % 24 == 0 or node_index == len(nodes)
            ):
                progress(f"batch {batch_index}: {node_index}/{len(nodes)} nodes")
    return output


def paired_node_set_rows(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    heads: Sequence[Head],
    mlp_layers: Sequence[int],
    direction: str,
    batch_pairs: int,
) -> list[dict[str, Any]]:
    """Score one joint head/MLP exchange or self-patch condition."""

    import torch

    if not pairs or batch_pairs < 1 or (not heads and not mlp_layers):
        raise ValueError("joint node condition requires nodes and pairs")
    output = []
    for start in range(0, len(pairs), batch_pairs):
        chunk = pairs[start : start + batch_pairs]
        batch = tokenize_promoter_pairs(bundle, chunk)
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, batch).detach()
        with ExitStack() as stack:
            if heads:
                stack.enter_context(
                    patch_paired_cls_heads(
                        bundle,
                        batch,
                        heads,
                        direction=direction,
                    )
                )
            if mlp_layers:
                stack.enter_context(
                    patch_paired_cls_mlp_writes(
                        bundle,
                        batch,
                        mlp_layers,
                        direction=direction,
                    )
                )
            with torch.no_grad():
                patched = promoter_margin_tensor(bundle, batch).detach()
        for pair_index, pair_id in enumerate(batch.pair_ids):
            output.append(
                {
                    "pair_id": pair_id,
                    "native_target_margin": float(native[2 * pair_index]),
                    "native_control_margin": float(native[2 * pair_index + 1]),
                    "patched_target_margin": float(patched[2 * pair_index]),
                    "patched_control_margin": float(patched[2 * pair_index + 1]),
                }
            )
    return output


def paired_final_cls_residual_rows(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    direction: str,
    batch_pairs: int,
) -> list[dict[str, Any]]:
    """Score a terminal paired CLS residual swap or self-patch control."""

    import torch

    if not pairs or batch_pairs < 1:
        raise ValueError("final-residual condition requires pairs")
    output = []
    for start in range(0, len(pairs), batch_pairs):
        chunk = pairs[start : start + batch_pairs]
        batch = tokenize_promoter_pairs(bundle, chunk)
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, batch).detach()
        with (
            patch_paired_final_cls_residual(bundle, direction=direction),
            torch.no_grad(),
        ):
            patched = promoter_margin_tensor(bundle, batch).detach()
        for pair_index, pair_id in enumerate(batch.pair_ids):
            output.append(
                {
                    "pair_id": pair_id,
                    "native_target_margin": float(native[2 * pair_index]),
                    "native_control_margin": float(native[2 * pair_index + 1]),
                    "patched_target_margin": float(patched[2 * pair_index]),
                    "patched_control_margin": float(patched[2 * pair_index + 1]),
                }
            )
    return output


def summarize_node_accuracy_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_draws: int,
    sign_permutations: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Summarize accuracy-first effects with full-family max-sign correction."""

    if not rows or bootstrap_draws < 1 or sign_permutations < 1:
        raise ValueError("node rows and inference draws must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["node"]), []).append(row)
    pair_order = sorted({str(row["pair_id"]) for row in rows})
    if any(
        {str(row["pair_id"]) for row in node_rows} != set(pair_order)
        for node_rows in grouped.values()
    ):
        raise ValueError("every node must contain the same pair family")
    summaries = {
        node: _node_point_summary(node_rows)
        for node, node_rows in sorted(grouped.items())
    }
    rng = random.Random(seed)
    bootstrap_values: dict[str, list[float]] = {node: [] for node in grouped}
    by_node_pair = {
        node: {str(row["pair_id"]): row for row in node_rows}
        for node, node_rows in grouped.items()
    }
    for _ in range(bootstrap_draws):
        sampled = [rng.choice(pair_order) for _ in pair_order]
        for node in grouped:
            bootstrap_values[node].append(
                statistics.mean(
                    float(by_node_pair[node][pair_id]["accuracy_loss"])
                    for pair_id in sampled
                )
            )
    for node, values in bootstrap_values.items():
        summaries[node]["accuracy_loss_CI95"] = _percentile_interval(values)

    node_values = {
        node: [
            float(by_node_pair[node][pair_id]["accuracy_loss"])
            for pair_id in pair_order
        ]
        for node in grouped
    }
    maxima = []
    for _ in range(sign_permutations):
        signs = [1.0 if rng.getrandbits(1) else -1.0 for _ in pair_order]
        maxima.append(
            max(
                statistics.mean(sign * value for sign, value in zip(signs, values))
                for values in node_values.values()
            )
        )
    for node, summary in summaries.items():
        summary["p_maxT_add_one"] = add_one_permutation_pvalue(
            float(summary["accuracy_loss"]),
            maxima,
            alternative="greater",
        )
    return summaries


def rank_node_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Rank accuracy first, then AUROC and continuous class-margin loss."""

    return sorted(
        summaries,
        key=lambda node: (
            -float(summaries[node]["accuracy_loss"]),
            -float(summaries[node]["auroc_loss"]),
            -float(summaries[node]["class_margin_loss"]),
            node,
        ),
    )


def _node_point_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    labels = [label for _row in rows for label in (1, 0)]
    native = [
        margin
        for row in rows
        for margin in (
            float(row["native_target_margin"]),
            float(row["native_control_margin"]),
        )
    ]
    patched = [
        margin
        for row in rows
        for margin in (
            float(row["patched_target_margin"]),
            float(row["patched_control_margin"]),
        )
    ]
    clean_metrics = binary_classification_metrics(labels, native)
    patched_metrics = binary_classification_metrics(labels, patched)
    return {
        "node": str(rows[0]["node"]),
        "node_type": str(rows[0]["node_type"]),
        "layer": int(rows[0]["layer"]),
        "head": (None if rows[0]["head"] is None else int(rows[0]["head"])),
        "n_pairs": len(rows),
        "accuracy_loss": float(clean_metrics["accuracy"])
        - float(patched_metrics["accuracy"]),
        "auroc_loss": float(clean_metrics["auroc"]) - float(patched_metrics["auroc"]),
        "class_margin_loss": statistics.mean(
            float(row["class_margin_loss"]) for row in rows
        ),
        "clean": clean_metrics,
        "patched": patched_metrics,
        "target_promoter_to_nonpromoter": sum(
            float(row["native_target_margin"]) > 0
            and float(row["patched_target_margin"]) <= 0
            for row in rows
        ),
        "target_nonpromoter_to_promoter": sum(
            float(row["native_target_margin"]) <= 0
            and float(row["patched_target_margin"]) > 0
            for row in rows
        ),
        "control_nonpromoter_to_promoter": sum(
            float(row["native_control_margin"]) <= 0
            and float(row["patched_control_margin"]) > 0
            for row in rows
        ),
        "control_promoter_to_nonpromoter": sum(
            float(row["native_control_margin"]) > 0
            and float(row["patched_control_margin"]) <= 0
            for row in rows
        ),
    }


def _percentile_interval(values: Sequence[float]) -> list[float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("percentile interval requires finite values")
    ordered = sorted(values)
    lo = max(0, math.floor(0.025 * len(ordered)))
    hi = min(len(ordered) - 1, math.ceil(0.975 * len(ordered)) - 1)
    return [ordered[lo], ordered[hi]]


def _node_patch(
    bundle: Dnabert2PromoterBundle,
    batch: Any,
    node: Mapping[str, Any],
    *,
    direction: str,
):
    if node["node_type"] == "head":
        return patch_paired_cls_heads(
            bundle,
            batch,
            [(int(node["layer"]), int(node["head"]))],
            direction=direction,
        )
    if node["node_type"] == "mlp":
        return patch_paired_cls_mlp_writes(
            bundle,
            batch,
            [int(node["layer"])],
            direction=direction,
        )
    raise ValueError(f"unknown node type: {node['node_type']}")
