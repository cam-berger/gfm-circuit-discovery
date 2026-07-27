"""Deterministic construction and auditing of empirical control families."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .schema import (
    ControlFamily,
    Interval,
    MatchedControl,
    MatchedSet,
    SiteReference,
)
from .tokenization import MaskPlan


@dataclass(frozen=True)
class SiteFeatures:
    site: SiteReference
    sequence: str
    gc_fraction: float
    cpg_fraction: float
    dinucleotide_frequencies: Mapping[str, float]
    token_count: int
    covered_nucleotides: int
    geometry_signature: tuple[tuple[int, int], ...]
    overlaps_annotation: bool
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.sequence) != self.site.interval.length:
            raise ValueError("site feature sequence length disagrees with its interval")
        if not 0 <= self.gc_fraction <= 1 or not 0 <= self.cpg_fraction <= 1:
            raise ValueError("composition fractions must be in [0, 1]")
        if self.token_count < 1 or self.covered_nucleotides < self.site.interval.length:
            raise ValueError("invalid token geometry in site features")
        object.__setattr__(
            self, "dinucleotide_frequencies", dict(self.dinucleotide_frequencies)
        )
        object.__setattr__(self, "attributes", dict(self.attributes))


@dataclass(frozen=True)
class MatchingSpec:
    controls_per_family: int = 1
    gc_caliper: float = 0.05
    cpg_caliper: float = 0.05
    dinucleotide_l1_caliper: float = 0.50
    require_exact_covered_nucleotides: bool = True
    require_exact_token_count: bool = True
    seed: int = 20260727

    def __post_init__(self) -> None:
        if self.controls_per_family < 1:
            raise ValueError("controls_per_family must be positive")
        for name in ("gc_caliper", "cpg_caliper", "dinucleotide_l1_caliper"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")


def compute_site_features(
    site: SiteReference,
    sequence_window: str,
    mask_plan: MaskPlan,
    *,
    annotated_intervals: Iterable[Interval] = (),
    attributes: Mapping[str, Any] | None = None,
) -> SiteFeatures:
    interval = site.interval
    word = sequence_window[interval.start : interval.end].upper()
    if len(word) != interval.length:
        raise ValueError("site falls outside its sequence window")
    gc = sum(base in "GC" for base in word) / len(word)
    cpg = sum(
        word[index : index + 2] == "CG" for index in range(max(0, len(word) - 1))
    ) / max(1, len(word) - 1)
    overlaps = any(interval.overlaps(other) for other in annotated_intervals)
    return SiteFeatures(
        site=site,
        sequence=word,
        gc_fraction=gc,
        cpg_fraction=cpg,
        dinucleotide_frequencies=_dinucleotide_frequencies(word),
        token_count=len(mask_plan.token_indices),
        covered_nucleotides=mask_plan.covered_nucleotides,
        geometry_signature=mask_plan.geometry_signature,
        overlaps_annotation=overlaps,
        attributes=dict(attributes or {}),
    )


def build_matched_sets(
    targets: Sequence[SiteFeatures],
    candidates: Mapping[ControlFamily, Sequence[SiteFeatures]],
    spec: MatchingSpec,
) -> list[MatchedSet]:
    """Match every target independently within each empirical control family.

    Families are never pooled: each target receives the frozen number of controls from
    each family or matching fails. Candidate reuse across different targets is allowed,
    and must be handled by target-cluster inference rather than silently dropping targets.
    """

    empirical = (
        ControlFamily.SAME_MOTIF_UNANNOTATED,
        ControlFamily.COMPOSITION_TOKEN_MATCHED,
        ControlFamily.SAME_SEQUENCE_NON_SITE,
    )
    missing_pools = [family.value for family in empirical if family not in candidates]
    if missing_pools:
        raise ValueError(f"missing candidate control pools: {missing_pools}")
    output: list[MatchedSet] = []
    for target in targets:
        controls: list[MatchedControl] = []
        for family in empirical:
            eligible = [
                candidate
                for candidate in candidates[family]
                if _eligible(target, candidate, family, spec)
            ]
            if len(eligible) < spec.controls_per_family:
                raise ValueError(
                    f"{target.site.example_id}:{target.site.interval} has "
                    f"{len(eligible)} eligible {family.value} controls; "
                    f"requires {spec.controls_per_family}"
                )
            ranked = sorted(
                eligible,
                key=lambda candidate: (
                    _distance(target, candidate),
                    _stable_tiebreak(target, candidate, family, spec.seed),
                ),
            )
            for candidate in ranked[: spec.controls_per_family]:
                controls.append(
                    MatchedControl(
                        family=family,
                        site=candidate.site,
                        match_distance=_distance(target, candidate),
                        matched_features={
                            "target_gc": target.gc_fraction,
                            "control_gc": candidate.gc_fraction,
                            "target_cpg": target.cpg_fraction,
                            "control_cpg": candidate.cpg_fraction,
                            "target_token_count": target.token_count,
                            "control_token_count": candidate.token_count,
                            "target_covered_nucleotides": target.covered_nucleotides,
                            "control_covered_nucleotides": candidate.covered_nucleotides,
                            "same_motif": target.sequence == candidate.sequence,
                            "same_example": (
                                target.site.example_id == candidate.site.example_id
                            ),
                        },
                    )
                )
        output.append(MatchedSet(target=target.site, controls=tuple(controls)))
    return output


def matching_balance(
    targets_by_key: Mapping[tuple[str, int, int], SiteFeatures],
    controls_by_key: Mapping[tuple[str, int, int], SiteFeatures],
    matched_sets: Sequence[MatchedSet],
) -> dict[str, dict[str, float]]:
    """Report standardized mean differences separately for every family."""

    result: dict[str, dict[str, float]] = {}
    for family in (
        ControlFamily.SAME_MOTIF_UNANNOTATED,
        ControlFamily.COMPOSITION_TOKEN_MATCHED,
        ControlFamily.SAME_SEQUENCE_NON_SITE,
    ):
        target_values: dict[str, list[float]] = {
            "gc_fraction": [],
            "cpg_fraction": [],
            "token_count": [],
            "covered_nucleotides": [],
        }
        control_values = {name: [] for name in target_values}
        for matched in matched_sets:
            target = targets_by_key[_key(matched.target)]
            for control in matched.controls:
                if control.family is not family:
                    continue
                candidate = controls_by_key[_key(control.site)]
                for name, values in target_values.items():
                    values.append(float(getattr(target, name)))
                    control_values[name].append(float(getattr(candidate, name)))
        result[family.value] = {
            name: _standardized_mean_difference(
                target_values[name], control_values[name]
            )
            for name in target_values
        }
    return result


def _eligible(
    target: SiteFeatures,
    candidate: SiteFeatures,
    family: ControlFamily,
    spec: MatchingSpec,
) -> bool:
    if candidate.site == target.site:
        return False
    if candidate.site.site_class != target.site.site_class:
        return False
    if candidate.site.interval.length != target.site.interval.length:
        return False
    if spec.require_exact_token_count and candidate.token_count != target.token_count:
        return False
    if (
        spec.require_exact_covered_nucleotides
        and candidate.covered_nucleotides != target.covered_nucleotides
    ):
        return False
    if abs(candidate.gc_fraction - target.gc_fraction) > spec.gc_caliper:
        return False
    if abs(candidate.cpg_fraction - target.cpg_fraction) > spec.cpg_caliper:
        return False
    if _dinucleotide_l1(target, candidate) > spec.dinucleotide_l1_caliper:
        return False
    if family is ControlFamily.SAME_MOTIF_UNANNOTATED:
        return (
            candidate.sequence == target.sequence and not candidate.overlaps_annotation
        )
    if family is ControlFamily.COMPOSITION_TOKEN_MATCHED:
        return not candidate.overlaps_annotation
    if family is ControlFamily.SAME_SEQUENCE_NON_SITE:
        return (
            candidate.site.example_id == target.site.example_id
            and not candidate.site.interval.overlaps(target.site.interval)
            and not candidate.overlaps_annotation
        )
    return False


def _distance(target: SiteFeatures, candidate: SiteFeatures) -> float:
    return (
        abs(target.gc_fraction - candidate.gc_fraction)
        + abs(target.cpg_fraction - candidate.cpg_fraction)
        + _dinucleotide_l1(target, candidate)
        + 0.25 * abs(target.covered_nucleotides - candidate.covered_nucleotides)
    )


def _dinucleotide_l1(target: SiteFeatures, candidate: SiteFeatures) -> float:
    keys = set(target.dinucleotide_frequencies) | set(
        candidate.dinucleotide_frequencies
    )
    return sum(
        abs(
            target.dinucleotide_frequencies.get(key, 0.0)
            - candidate.dinucleotide_frequencies.get(key, 0.0)
        )
        for key in keys
    )


def _dinucleotide_frequencies(sequence: str) -> dict[str, float]:
    denominator = max(1, len(sequence) - 1)
    counts = {a + b: 0 for a in "ACGT" for b in "ACGT"}
    for index in range(len(sequence) - 1):
        word = sequence[index : index + 2]
        if word in counts:
            counts[word] += 1
    return {word: count / denominator for word, count in counts.items()}


def _stable_tiebreak(
    target: SiteFeatures,
    candidate: SiteFeatures,
    family: ControlFamily,
    seed: int,
) -> str:
    text = (
        f"{seed}|{family.value}|{target.site.example_id}|{target.site.interval.start}|"
        f"{candidate.site.example_id}|{candidate.site.interval.start}"
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _standardized_mean_difference(
    left: Sequence[float], right: Sequence[float]
) -> float:
    if len(left) != len(right) or not left:
        return math.nan
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    if len(left) == 1:
        return 0.0 if mean_left == mean_right else math.inf
    var_left = sum((value - mean_left) ** 2 for value in left) / (len(left) - 1)
    var_right = sum((value - mean_right) ** 2 for value in right) / (len(right) - 1)
    pooled = math.sqrt((var_left + var_right) / 2)
    if pooled == 0:
        return 0.0 if mean_left == mean_right else math.inf
    return (mean_left - mean_right) / pooled


def _key(site: SiteReference) -> tuple[str, int, int]:
    return (site.example_id, site.interval.start, site.interval.end)
