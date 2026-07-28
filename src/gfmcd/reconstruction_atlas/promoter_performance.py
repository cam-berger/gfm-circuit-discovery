"""Paired classification metrics for a nominated promoter source region."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from .promoter_assay import binary_classification_metrics

MARGIN_FIELDS = (
    "target_clean_margin",
    "target_masked_margin",
    "control_clean_margin",
    "control_masked_margin",
)


def summarize_masked_classification(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize clean/masked metrics and paired hard-label transitions."""

    if not rows or bootstrap_draws < 1:
        raise ValueError("performance rows and bootstrap draws must be non-empty")
    keys = [(str(row.get("split", "")), str(row.get("pair_id", ""))) for row in rows]
    if any(not split or not pair_id for split, pair_id in keys):
        raise ValueError("every performance row requires split and pair_id")
    if len(set(keys)) != len(keys):
        raise ValueError("performance rows must have unique split/pair identities")
    for row in rows:
        if any(
            field not in row or not math.isfinite(float(row[field]))
            for field in MARGIN_FIELDS
        ):
            raise ValueError("performance rows require finite clean/masked margins")

    point = _point_summary(rows)
    rng = random.Random(seed)
    draws = []
    for _ in range(bootstrap_draws):
        sampled = [rows[rng.randrange(len(rows))] for _ in rows]
        draws.append(_point_summary(sampled))
    intervals = {
        metric: _percentile_interval([float(draw["changes"][metric]) for draw in draws])
        for metric in ("accuracy", "auroc", "mcc")
    }
    clean_intervals = {
        metric: _percentile_interval([float(draw["clean"][metric]) for draw in draws])
        for metric in ("accuracy", "auroc", "mcc")
    }
    masked_intervals = {
        metric: _percentile_interval([float(draw["masked"][metric]) for draw in draws])
        for metric in ("accuracy", "auroc", "mcc")
    }
    target_flip_intervals = {
        "promoter_to_non_promoter_fraction_all_targets": _percentile_interval(
            [
                float(
                    draw["label_transitions"]["target"][
                        "promoter_to_non_promoter_fraction_all_targets"
                    ]
                )
                for draw in draws
            ]
        ),
        "promoter_to_non_promoter_fraction_clean_positive": _percentile_interval(
            [
                float(
                    draw["label_transitions"]["target"][
                        "promoter_to_non_promoter_fraction_clean_positive"
                    ]
                )
                for draw in draws
            ]
        ),
    }
    return {
        **point,
        "bootstrap": {
            "draws": bootstrap_draws,
            "seed": seed,
            "clean_ci95": clean_intervals,
            "masked_ci95": masked_intervals,
            "change_ci95": intervals,
            "target_flip_fraction_ci95": target_flip_intervals,
        },
    }


def _point_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [label for _ in rows for label in (1, 0)]
    clean_margins = [
        margin
        for row in rows
        for margin in (
            float(row["target_clean_margin"]),
            float(row["control_clean_margin"]),
        )
    ]
    masked_margins = [
        margin
        for row in rows
        for margin in (
            float(row["target_masked_margin"]),
            float(row["control_masked_margin"]),
        )
    ]
    clean = binary_classification_metrics(labels, clean_margins)
    masked = binary_classification_metrics(labels, masked_margins)
    target_clean_positive = sum(float(row["target_clean_margin"]) > 0 for row in rows)
    target_promoter_to_non = sum(
        float(row["target_clean_margin"]) > 0
        and float(row["target_masked_margin"]) <= 0
        for row in rows
    )
    target_non_to_promoter = sum(
        float(row["target_clean_margin"]) <= 0
        and float(row["target_masked_margin"]) > 0
        for row in rows
    )
    control_non_to_promoter = sum(
        float(row["control_clean_margin"]) <= 0
        and float(row["control_masked_margin"]) > 0
        for row in rows
    )
    control_promoter_to_non = sum(
        float(row["control_clean_margin"]) > 0
        and float(row["control_masked_margin"]) <= 0
        for row in rows
    )
    return {
        "n_pairs": len(rows),
        "n_sequences": 2 * len(rows),
        "clean": clean,
        "masked": masked,
        "changes": {
            "accuracy": float(masked["accuracy"]) - float(clean["accuracy"]),
            "auroc": float(masked["auroc"]) - float(clean["auroc"]),
            "mcc": float(masked["mcc"]) - float(clean["mcc"]),
        },
        "label_transitions": {
            "target": {
                "clean_predicted_promoter": target_clean_positive,
                "promoter_to_non_promoter": target_promoter_to_non,
                "non_promoter_to_promoter": target_non_to_promoter,
                "promoter_to_non_promoter_fraction_all_targets": (
                    target_promoter_to_non / len(rows)
                ),
                "promoter_to_non_promoter_fraction_clean_positive": (
                    target_promoter_to_non / target_clean_positive
                    if target_clean_positive
                    else 0.0
                ),
            },
            "control": {
                "clean_predicted_non_promoter": len(rows)
                - int(clean["confusion"]["fp"]),
                "non_promoter_to_promoter": control_non_to_promoter,
                "promoter_to_non_promoter": control_promoter_to_non,
            },
        },
    }


def _percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    lo = max(0, math.floor(0.025 * len(ordered)))
    hi = min(len(ordered) - 1, math.ceil(0.975 * len(ordered)) - 1)
    return [ordered[lo], ordered[hi]]
