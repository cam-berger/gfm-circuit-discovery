"""Behavioral excess-reconstruction estimands."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .schema import ControlFamily, MatchedSet, SiteReference


@dataclass(frozen=True)
class SiteReconstruction:
    site: SiteReference
    score_per_covered_nt: float
    homology_cluster: str
    baseline_scores_per_nt: Mapping[str, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.score_per_covered_nt):
            raise ValueError("site score must be finite")
        if not self.homology_cluster:
            raise ValueError("homology cluster is required")
        if any(
            not math.isfinite(value) for value in self.baseline_scores_per_nt.values()
        ):
            raise ValueError("baseline scores must be finite")


@dataclass(frozen=True)
class ExcessReconstruction:
    target: SiteReference
    homology_cluster: str
    target_score_per_covered_nt: float
    empirical_advantage_by_family: Mapping[str, float]
    baseline_advantage: Mapping[str, float]
    control_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "homology_cluster": self.homology_cluster,
            "target_score_per_covered_nt": self.target_score_per_covered_nt,
            "empirical_advantage_by_family": dict(self.empirical_advantage_by_family),
            "baseline_advantage": dict(self.baseline_advantage),
            "control_counts": dict(self.control_counts),
        }


def compute_excess_reconstruction(
    matched_sets: Sequence[MatchedSet],
    scores: Mapping[tuple[str, int, int], SiteReconstruction],
) -> list[ExcessReconstruction]:
    output: list[ExcessReconstruction] = []
    for matched in matched_sets:
        target = scores[_key(matched.target)]
        grouped: dict[ControlFamily, list[float]] = defaultdict(list)
        for control in matched.controls:
            grouped[control.family].append(
                scores[_key(control.site)].score_per_covered_nt
            )
        required = {
            ControlFamily.SAME_MOTIF_UNANNOTATED,
            ControlFamily.COMPOSITION_TOKEN_MATCHED,
            ControlFamily.SAME_SEQUENCE_NON_SITE,
        }
        if set(grouped) != required:
            raise ValueError(
                "excess reconstruction requires all empirical families separately"
            )
        empirical = {
            family.value: target.score_per_covered_nt - mean(values)
            for family, values in grouped.items()
        }
        baseline = {
            name: target.score_per_covered_nt - value
            for name, value in target.baseline_scores_per_nt.items()
        }
        output.append(
            ExcessReconstruction(
                target=matched.target,
                homology_cluster=target.homology_cluster,
                target_score_per_covered_nt=target.score_per_covered_nt,
                empirical_advantage_by_family=empirical,
                baseline_advantage=baseline,
                control_counts={
                    family.value: len(values) for family, values in grouped.items()
                },
            )
        )
    return output


def summarize_by_site_class(
    rows: Sequence[ExcessReconstruction],
) -> dict[str, dict[str, Any]]:
    """Cluster-first summaries; no observation-level pseudo-replication."""

    by_class: dict[str, list[ExcessReconstruction]] = defaultdict(list)
    for row in rows:
        by_class[row.target.site_class].append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for site_class, class_rows in sorted(by_class.items()):
        family_names = sorted(
            {name for row in class_rows for name in row.empirical_advantage_by_family}
        )
        baseline_names = sorted(
            {name for row in class_rows for name in row.baseline_advantage}
        )
        cluster_empirical = {
            family: _cluster_means(
                class_rows,
                lambda row, family=family: row.empirical_advantage_by_family[family],
            )
            for family in family_names
        }
        cluster_baseline = {
            name: _cluster_means(
                class_rows,
                lambda row, name=name: row.baseline_advantage[name],
            )
            for name in baseline_names
        }
        summaries[site_class] = {
            "n_sites": len(class_rows),
            "n_homology_clusters": len({row.homology_cluster for row in class_rows}),
            "mean_target_score_per_covered_nt": mean(
                row.target_score_per_covered_nt for row in class_rows
            ),
            "mean_empirical_advantage_by_family": {
                family: mean(values.values())
                for family, values in cluster_empirical.items()
            },
            "mean_baseline_advantage": {
                name: mean(values.values()) for name, values in cluster_baseline.items()
            },
        }
    return summaries


def _cluster_means(
    rows: Sequence[ExcessReconstruction], getter: Any
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.homology_cluster].append(float(getter(row)))
    return {cluster: mean(values) for cluster, values in grouped.items()}


def _key(site: SiteReference) -> tuple[str, int, int]:
    return (site.example_id, site.interval.start, site.interval.end)
