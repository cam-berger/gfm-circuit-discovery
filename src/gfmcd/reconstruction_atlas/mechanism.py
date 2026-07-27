"""Phase gates and records that keep candidate evidence separate from causality."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceLevel(str, Enum):
    BEHAVIOR = "behavior"
    REPRESENTATION = "representation"
    QK = "qk_selection"
    ATTENTION = "post_softmax_attention"
    OV = "ov_output"
    MEDIATION = "mediation"
    DIRECTED_CIRCUIT = "directed_circuit"
    BIOLOGICAL = "biological_interpretation"


class InterventionType(str, Enum):
    ABLATION = "ablation"
    ACTIVATION_PATCH = "activation_patch"
    PATH_PATCH = "path_patch"
    RESCUE = "rescue"
    SUBSPACE = "subspace_intervention"


@dataclass(frozen=True)
class BehaviorQualification:
    site_class: str
    reconstruction_gate_passed: bool
    empirical_specificity_gate_passed: bool
    context_specificity_gate_passed: bool
    qualified_artifact: str

    @property
    def qualifies_for_internal_screen(self) -> bool:
        return (
            self.reconstruction_gate_passed
            and self.empirical_specificity_gate_passed
            and self.context_specificity_gate_passed
            and bool(self.qualified_artifact)
        )


@dataclass(frozen=True)
class ComponentCandidate:
    component_id: str
    component_type: str
    layer: int
    target_positions: tuple[int, ...]
    source_region_type: str
    nomination_methods: tuple[str, ...]
    discovery_artifact: str
    evidence_level: EvidenceLevel = EvidenceLevel.REPRESENTATION

    def __post_init__(self) -> None:
        if self.evidence_level not in {
            EvidenceLevel.REPRESENTATION,
            EvidenceLevel.QK,
            EvidenceLevel.ATTENTION,
            EvidenceLevel.OV,
        }:
            raise ValueError("candidate records cannot claim causal evidence")
        if not self.nomination_methods or not self.discovery_artifact:
            raise ValueError("candidate must cite its nomination method and artifact")


@dataclass(frozen=True)
class CausalInterventionSpec:
    intervention_id: str
    intervention_type: InterventionType
    candidate_components: tuple[str, ...]
    matched_null_sets: tuple[tuple[str, ...], ...]
    protected_metrics: tuple[str, ...]
    behavior_artifact: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        size = len(self.candidate_components)
        if size < 1:
            raise ValueError(
                "causal intervention needs at least one candidate component"
            )
        if not self.matched_null_sets:
            raise ValueError("causal intervention needs matched null sets")
        if any(len(null_set) != size for null_set in self.matched_null_sets):
            raise ValueError(
                "every null intervention must match candidate size exactly"
            )
        if not self.protected_metrics:
            raise ValueError("causal intervention must declare protected metrics")
        object.__setattr__(self, "parameters", dict(self.parameters))


def require_mechanism_qualification(qualification: BehaviorQualification) -> None:
    if not qualification.qualifies_for_internal_screen:
        raise RuntimeError(
            f"{qualification.site_class} has not passed behavior, empirical-specificity, "
            "and context-specificity gates; internal screening is prohibited"
        )
