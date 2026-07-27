"""Outcome-blind sequence controls and summaries for atlas preflight."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist, mean, stdev
from typing import Any

from .schema import Interval
from .statistics import cluster_bootstrap_interval
from .tokenization import encode_mask_plan, geometry_is_invariant


@dataclass(frozen=True)
class PreflightVariants:
    reference: str
    copy_dose_1: str
    copy_dose_2: str
    copy_dose_4: str
    irrelevant_exchange: str
    target: Interval
    copied_word: str
    copy_region: Interval
    irrelevant_region: Interval

    def sequences(self) -> Mapping[str, str]:
        return {
            "reference": self.reference,
            "copy_dose_1": self.copy_dose_1,
            "copy_dose_2": self.copy_dose_2,
            "copy_dose_4": self.copy_dose_4,
            "irrelevant_exchange": self.irrelevant_exchange,
        }


def build_preflight_variants(
    sequence: str,
    target: Interval,
    *,
    encoder: Any,
    seed: int,
    minimum_target_buffer_nt: int = 4,
    maximum_region_nt: int = 64,
) -> PreflightVariants | None:
    """Plant 1/2/4 exact scored-cover copies downstream of an unchanged target."""

    _, reference_plan = encode_mask_plan(sequence, target, encoder)
    copied_word = sequence[reference_plan.covered.start : reference_plan.covered.end]
    width = min(maximum_region_nt, 4 * len(copied_word))
    if width < 4 * len(copied_word):
        return None
    earliest = reference_plan.covered.end + minimum_target_buffer_nt
    latest = len(sequence) - width
    if earliest > latest:
        return None
    for start in range(earliest, latest + 1):
        copy_region = Interval(start, start + width)
        variants = {
            dose: _plant_copies(sequence, copy_region, copied_word, dose)
            for dose in (1, 2, 4)
        }
        if any(value == sequence for value in variants.values()):
            continue
        if not all(
            geometry_is_invariant(
                reference_plan,
                encode_mask_plan(value, target, encoder)[1],
            )
            for value in variants.values()
        ):
            continue
        irrelevant = _irrelevant_exchange(
            sequence,
            target,
            copy_region,
            encoder=encoder,
            seed=seed,
            minimum_target_buffer_nt=minimum_target_buffer_nt,
        )
        if irrelevant is None:
            continue
        irrelevant_sequence, irrelevant_region = irrelevant
        return PreflightVariants(
            reference=sequence,
            copy_dose_1=variants[1],
            copy_dose_2=variants[2],
            copy_dose_4=variants[4],
            irrelevant_exchange=irrelevant_sequence,
            target=target,
            copied_word=copied_word,
            copy_region=copy_region,
            irrelevant_region=irrelevant_region,
        )
    return None


def summarize_preflight_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_draws: int,
    seed: int,
    power: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Summarize positive-dose and irrelevant-exchange effects by site class."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["site_class"])].append(row)
    output: dict[str, Any] = {}
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(power)
    for site_class, class_rows in sorted(grouped.items()):
        clusters = [str(row["homology_cluster"]) for row in class_rows]
        effects = {
            dose: [
                float(row[f"copy_dose_{dose}"]) - float(row["reference"])
                for row in class_rows
            ]
            for dose in (1, 2, 4)
        }
        irrelevant = [
            float(row["irrelevant_exchange"]) - float(row["reference"])
            for row in class_rows
        ]
        dose_means = {str(dose): mean(values) for dose, values in effects.items()}
        positive_interval = cluster_bootstrap_interval(
            effects[4],
            clusters,
            draws=bootstrap_draws,
            seed=seed,
            alpha=alpha,
        )
        negative_interval = cluster_bootstrap_interval(
            irrelevant,
            clusters,
            draws=bootstrap_draws,
            seed=seed + 1,
            alpha=alpha,
        )
        paired_sd = stdev(effects[4]) if len(effects[4]) > 1 else math.inf
        mde = (z_alpha + z_power) * paired_sd / math.sqrt(len(effects[4]))
        output[site_class] = {
            "n_examples": len(class_rows),
            "n_clusters": len(set(clusters)),
            "copy_effect_mean_by_dose": dose_means,
            "copy_dose_4_ci": list(positive_interval),
            "copy_dose_4_ci_half_width": (positive_interval[1] - positive_interval[0])
            / 2,
            "copy_dose_4_paired_sd": paired_sd,
            "minimum_detectable_effect_nats_per_covered_nt": mde,
            "copy_means_monotonic": (
                0 < dose_means["1"] < dose_means["2"] < dose_means["4"]
            ),
            "irrelevant_effect_mean": mean(irrelevant),
            "irrelevant_effect_ci": list(negative_interval),
            "irrelevant_equivalence_half_width": max(
                abs(negative_interval[0]), abs(negative_interval[1])
            ),
        }
    return output


def _plant_copies(sequence: str, region: Interval, copied_word: str, dose: int) -> str:
    replacement = list(sequence[region.start : region.end])
    width = len(copied_word)
    for index in range(dose):
        start = index * width
        replacement[start : start + width] = copied_word
    return sequence[: region.start] + "".join(replacement) + sequence[region.end :]


def _irrelevant_exchange(
    sequence: str,
    target: Interval,
    excluded_region: Interval,
    *,
    encoder: Any,
    seed: int,
    minimum_target_buffer_nt: int,
) -> tuple[str, Interval] | None:
    _, reference_plan = encode_mask_plan(sequence, target, encoder)
    width = excluded_region.length
    earliest = target.end + minimum_target_buffer_nt
    latest = len(sequence) - 2 * width
    if earliest > latest:
        return None
    starts = sorted(
        range(earliest, latest + 1),
        key=lambda start: hashlib.sha256(
            f"{seed}|irrelevant|{sequence}|{target.start}|{start}".encode()
        ).hexdigest(),
    )
    for recipient_start in starts:
        recipient = Interval(recipient_start, recipient_start + width)
        donor = Interval(recipient.end, recipient.end + width)
        if recipient.overlaps(excluded_region) or donor.overlaps(excluded_region):
            continue
        replacement = sequence[donor.start : donor.end]
        changed = sequence[: recipient.start] + replacement + sequence[recipient.end :]
        if changed == sequence:
            continue
        comparison = encode_mask_plan(changed, target, encoder)[1]
        if geometry_is_invariant(reference_plan, comparison):
            return changed, recipient
    return None
