"""Outcome-independent selection and inference helpers for the promoter head atlas."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from gfmcd.adapters.dnabert2.promoter_head_atlas import Head, head_label


def select_locus_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    maximum: int,
    seed: int,
    tag: str,
) -> list[dict[str, Any]]:
    """Select a deterministic, outcome-blind one-row-per-locus subset."""

    if maximum < 1:
        raise ValueError("maximum must be positive")
    eligible = [dict(row) for row in rows if row.get("split") == split]
    by_cluster: dict[str, dict[str, Any]] = {}
    for row in eligible:
        cluster = str(row["homology_cluster"])
        rank = _rank(seed, tag, str(row["example_id"]))
        previous = by_cluster.get(cluster)
        if previous is None or rank < previous["_head_atlas_rank"]:
            row["_head_atlas_rank"] = rank
            by_cluster[cluster] = row
    selected = sorted(
        by_cluster.values(),
        key=lambda row: (row["_head_atlas_rank"], str(row["example_id"])),
    )[:maximum]
    for row in selected:
        row.pop("_head_atlas_rank")
    if len(selected) < maximum:
        raise ValueError(
            f"split {split!r} supplied {len(selected)} unique loci, need {maximum}"
        )
    return selected


def select_head_candidates(
    head_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_count: int,
    maximum_per_layer: int,
) -> list[dict[str, Any]]:
    """Apply the frozen sign-consistent rank aggregation and layer cap."""

    if candidate_count < 1 or maximum_per_layer < 1:
        raise ValueError("candidate count and layer cap must be positive")
    required = (
        "necessity_mean",
        "sufficiency_mean",
        "attribution_necessity_mean",
        "attribution_sufficiency_mean",
    )
    rows = [dict(row) for row in head_rows]
    if not rows or any(
        key not in row or not math.isfinite(float(row[key]))
        for row in rows
        for key in required
    ):
        raise ValueError("head rows are empty, incomplete, or non-finite")
    ranks = {key: _descending_fractional_ranks(rows, key) for key in required}
    for index, row in enumerate(rows):
        row["sign_consistent"] = all(float(row[key]) > 0 for key in required)
        row["rank_score"] = mean(ranks[key][index] for key in required)
    eligible = sorted(
        (row for row in rows if row["sign_consistent"]),
        key=lambda row: (
            float(row["rank_score"]),
            int(row["layer"]),
            int(row["head"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    layer_counts: Counter[int] = Counter()
    for row in eligible:
        layer = int(row["layer"])
        if layer_counts[layer] >= maximum_per_layer:
            continue
        selected.append(row)
        layer_counts[layer] += 1
        if len(selected) == candidate_count:
            break
    return selected


def layer_matched_random_head_sets(
    candidates: Sequence[Head],
    *,
    num_layers: int,
    num_heads: int,
    draws: int,
    seed: int,
) -> list[tuple[Head, ...]]:
    """Return distinct candidate-excluding sets with the same per-layer profile."""

    selected = tuple(sorted(set(candidates)))
    if not selected or len(selected) != len(candidates):
        raise ValueError("candidate heads must be nonempty and unique")
    if draws < 1:
        raise ValueError("null draws must be positive")
    profile = Counter(layer for layer, _ in selected)
    selected_set = set(selected)
    choices: list[list[tuple[int, ...]]] = []
    layers = sorted(profile)
    for layer in layers:
        if not 0 <= layer < num_layers:
            raise ValueError(f"candidate layer outside model: {layer}")
        available = [
            head for head in range(num_heads) if (layer, head) not in selected_set
        ]
        count = profile[layer]
        if len(available) < count:
            raise ValueError(
                f"layer {layer} cannot supply a matched candidate-excluding null"
            )
        choices.append(list(itertools.combinations(available, count)))
    total = math.prod(len(layer_choices) for layer_choices in choices)
    if total <= draws:
        products = itertools.product(*choices)
        return [
            tuple(
                sorted(
                    (layer, head)
                    for layer, heads in zip(layers, product)
                    for head in heads
                )
            )
            for product in products
        ]
    rng = random.Random(seed)
    output: set[tuple[Head, ...]] = set()
    while len(output) < draws:
        sampled: list[Head] = []
        for layer, layer_choices in zip(layers, choices):
            heads = rng.choice(layer_choices)
            sampled.extend((layer, head) for head in heads)
        output.add(tuple(sorted(sampled)))
    return sorted(output)


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Spearman correlation with average ranks for ties."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError(
            "Spearman inputs must be aligned and contain at least two values"
        )
    if any(not math.isfinite(float(value)) for value in (*left, *right)):
        raise ValueError("Spearman inputs must be finite")
    x = _average_ranks([float(value) for value in left])
    y = _average_ranks([float(value) for value in right])
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else 0.0


def serialize_head_set(heads: Sequence[Head]) -> list[str]:
    return [head_label(head) for head in sorted(heads)]


def parse_head_label(label: str) -> Head:
    if not label.startswith("L") or "H" not in label:
        raise ValueError(f"invalid head label: {label}")
    layer, head = label[1:].split("H", 1)
    return int(layer), int(head)


def _descending_fractional_ranks(
    rows: Sequence[Mapping[str, Any]], key: str
) -> list[float]:
    values = [-float(row[key]) for row in rows]
    ranks = _average_ranks(values)
    denominator = max(1, len(rows) - 1)
    return [(rank - 1) / denominator for rank in ranks]


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _rank(seed: int, tag: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}|{tag}|{identity}".encode()).hexdigest()
