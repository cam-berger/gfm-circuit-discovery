"""Deterministic construction and auditing of empirical control families."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
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
    overlaps = bool(site.attributes.get("source_annotation_overlap")) or any(
        interval.overlaps(other) for other in annotated_intervals
    )
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
    output, failures = build_matchable_sets(targets, candidates, spec)
    if failures:
        failure = failures[0]
        raise ValueError(
            f"{failure['example_id']}:{failure['interval']} has "
            f"{failure['eligible']} eligible {failure['family']} controls; "
            f"requires {spec.controls_per_family}"
        )
    return output


def build_matchable_sets(
    targets: Sequence[SiteFeatures],
    candidates: Mapping[ControlFamily, Sequence[SiteFeatures]],
    spec: MatchingSpec,
) -> tuple[list[MatchedSet], list[dict[str, Any]]]:
    """Return complete matched sets plus outcome-blind supply failures.

    A site with inadequate controls is excluded before any model outcome exists.
    The caller must report the exclusions and may only admit site classes whose
    frozen cluster-supply gate still passes.
    """

    empirical = (
        ControlFamily.SAME_MOTIF_UNANNOTATED,
        ControlFamily.COMPOSITION_TOKEN_MATCHED,
        ControlFamily.SAME_SEQUENCE_NON_SITE,
    )
    missing_pools = [family.value for family in empirical if family not in candidates]
    if missing_pools:
        raise ValueError(f"missing candidate control pools: {missing_pools}")
    indexes = {
        family: _CandidateIndex(tuple(candidates[family]), spec) for family in empirical
    }
    output: list[MatchedSet] = []
    failures: list[dict[str, Any]] = []
    for target in targets:
        controls: list[MatchedControl] = []
        for family in empirical:
            eligible = [
                candidate
                for candidate in indexes[family].candidates_for(target, family)
                if _eligible(target, candidate, family, spec)
            ]
            if len(eligible) < spec.controls_per_family:
                failures.append(
                    {
                        "example_id": target.site.example_id,
                        "interval": (
                            target.site.interval.start,
                            target.site.interval.end,
                        ),
                        "site_class": target.site.site_class,
                        "family": family.value,
                        "eligible": len(eligible),
                    }
                )
                break
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
        else:
            output.append(MatchedSet(target=target.site, controls=tuple(controls)))
    return output, failures


def generate_candidate_sites(
    records: Sequence[Any],
    *,
    positions_per_example: int,
    seed: int,
    annotation_intervals_by_example: Mapping[str, Sequence[Interval]] | None = None,
) -> list[SiteReference]:
    """Build an outcome-blind candidate pool from the admitted sequence windows."""

    if positions_per_example < 1:
        raise ValueError("positions_per_example must be positive")
    source_annotations = annotation_intervals_by_example or {}
    targets = []
    motifs_by_class: dict[str, set[str]] = defaultdict(set)
    for record in records:
        target_id = record.metadata.get("target_element_id")
        if target_id is None:
            raise ValueError(f"{record.example_id} has no target_element_id")
        target = record.element(str(target_id))
        targets.append((record, target))
        motifs_by_class[target.site_class].add(
            record.sequence[target.interval.start : target.interval.end]
        )
    output: dict[tuple[str, int, int, str], SiteReference] = {}
    for record, target in targets:
        length = target.interval.length
        maximum_start = len(record.sequence) - length
        target_word = record.sequence[target.interval.start : target.interval.end]
        ranked = sorted(
            range(maximum_start + 1),
            key=lambda start: hashlib.sha256(
                f"{seed}|candidate-tie|{record.example_id}|"
                f"{target.site_class}|{length}|{start}".encode()
            ).hexdigest(),
        )
        ranked.sort(
            key=lambda start: _word_composition_distance(
                target_word,
                record.sequence[start : start + length],
            )
        )
        starts = set(ranked[:positions_per_example])
        motifs = motifs_by_class[target.site_class]
        starts.update(
            start
            for start in range(maximum_start + 1)
            if record.sequence[start : start + length] in motifs
        )
        for start in starts:
            if start == target.interval.start and start + length == target.interval.end:
                continue
            site = SiteReference(
                example_id=record.example_id,
                interval=Interval(start, start + length),
                site_class=target.site_class,
                attributes={
                    "source_annotation_overlap": any(
                        Interval(start, start + length).overlaps(interval)
                        for interval in source_annotations.get(record.example_id, ())
                    )
                },
            )
            output[
                (
                    site.example_id,
                    site.interval.start,
                    site.interval.end,
                    site.site_class,
                )
            ] = site
    return [output[key] for key in sorted(output)]


def _word_composition_distance(left: str, right: str) -> int:
    """Outcome-blind integer distance for local candidate prioritization."""

    left_dinucleotides = Counter(
        left[index : index + 2] for index in range(max(0, len(left) - 1))
    )
    right_dinucleotides = Counter(
        right[index : index + 2] for index in range(max(0, len(right) - 1))
    )
    words = set(left_dinucleotides) | set(right_dinucleotides)
    return (
        abs(sum(base in "GC" for base in left) - sum(base in "GC" for base in right))
        + abs(left.count("CG") - right.count("CG"))
        + sum(
            abs(left_dinucleotides[word] - right_dinucleotides[word]) for word in words
        )
    )


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
    target_split = target.attributes.get("split")
    candidate_split = candidate.attributes.get("split")
    if (
        target_split is not None
        and candidate_split is not None
        and target_split != candidate_split
    ):
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


class _CandidateIndex:
    def __init__(self, candidates: Sequence[SiteFeatures], spec: MatchingSpec) -> None:
        self.spec = spec
        self.by_base: dict[
            tuple[Any, ...], dict[tuple[float, float], list[SiteFeatures]]
        ] = defaultdict(lambda: defaultdict(list))
        self.by_motif: dict[tuple[Any, ...], list[SiteFeatures]] = defaultdict(list)
        self.by_example: dict[tuple[Any, ...], list[SiteFeatures]] = defaultdict(list)
        for candidate in candidates:
            base = _base_key(candidate, spec)
            self.by_base[base][(candidate.gc_fraction, candidate.cpg_fraction)].append(
                candidate
            )
            self.by_motif[(*base, candidate.sequence)].append(candidate)
            self.by_example[(*base, candidate.site.example_id)].append(candidate)

    def candidates_for(
        self, target: SiteFeatures, family: ControlFamily
    ) -> Sequence[SiteFeatures]:
        base = _base_key(target, self.spec)
        if family is ControlFamily.SAME_MOTIF_UNANNOTATED:
            return self.by_motif.get((*base, target.sequence), ())
        if family is ControlFamily.SAME_SEQUENCE_NON_SITE:
            return self.by_example.get((*base, target.site.example_id), ())
        cells = self.by_base.get(base, {})
        return [
            candidate
            for (gc, cpg), cell in cells.items()
            if abs(gc - target.gc_fraction) <= self.spec.gc_caliper
            and abs(cpg - target.cpg_fraction) <= self.spec.cpg_caliper
            for candidate in cell
        ]


def _base_key(feature: SiteFeatures, spec: MatchingSpec) -> tuple[Any, ...]:
    return (
        feature.site.site_class,
        feature.site.interval.length,
        feature.token_count if spec.require_exact_token_count else None,
        (
            feature.covered_nucleotides
            if spec.require_exact_covered_nucleotides
            else None
        ),
        feature.attributes.get("split"),
    )


def matching_supply_summary(
    targets: Sequence[SiteFeatures],
    matched_sets: Sequence[MatchedSet],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_counts = Counter(
        (x.attributes.get("split"), x.site.site_class) for x in targets
    )
    matched_by_key = {
        (item.target.example_id, item.target.interval.start, item.target.interval.end)
        for item in matched_sets
    }
    matched_counts = Counter(
        (x.attributes.get("split"), x.site.site_class)
        for x in targets
        if _key(x.site) in matched_by_key
    )
    failure_counts = Counter(
        (
            str(item.get("site_class")),
            str(item.get("family")),
        )
        for item in failures
    )
    return {
        "target_counts": {
            f"{split}|{site_class}": count
            for (split, site_class), count in sorted(target_counts.items())
        },
        "matched_counts": {
            f"{split}|{site_class}": count
            for (split, site_class), count in sorted(matched_counts.items())
        },
        "failure_counts": {
            f"{site_class}|{family}": count
            for (site_class, family), count in sorted(failure_counts.items())
        },
    }
