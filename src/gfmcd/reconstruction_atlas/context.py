"""Target-preserving context exchanges and their token-geometry audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import Interval
from .tokenization import (
    MaskPlan,
    encode_mask_plan,
    geometry_is_invariant,
    normalize_dna,
)


@dataclass(frozen=True)
class ContextIntervention:
    intervention_id: str
    region_type: str
    recipient_region: Interval
    donor_region: Interval
    target: Interval
    donor_example_id: str


@dataclass(frozen=True)
class ContextInterventionAudit:
    intervention_id: str
    target_unchanged: bool
    sequence_length_unchanged: bool
    target_geometry_invariant: bool
    changed_nucleotides: int
    reference_plan: MaskPlan
    intervened_plan: MaskPlan

    @property
    def eligible(self) -> bool:
        return (
            self.target_unchanged
            and self.sequence_length_unchanged
            and self.target_geometry_invariant
            and self.changed_nucleotides > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "target_unchanged": self.target_unchanged,
            "sequence_length_unchanged": self.sequence_length_unchanged,
            "target_geometry_invariant": self.target_geometry_invariant,
            "changed_nucleotides": self.changed_nucleotides,
            "eligible": self.eligible,
            "reference_plan": self.reference_plan.to_dict(),
            "intervened_plan": self.intervened_plan.to_dict(),
        }


def apply_context_exchange(
    recipient_sequence: str,
    donor_sequence: str,
    intervention: ContextIntervention,
    *,
    encoder: Any,
) -> tuple[str, ContextInterventionAudit]:
    recipient = normalize_dna(recipient_sequence)
    donor = normalize_dna(donor_sequence)
    if intervention.recipient_region.length != intervention.donor_region.length:
        raise ValueError("context exchange must be length preserving")
    if intervention.recipient_region.end > len(recipient):
        raise ValueError("recipient region falls outside its sequence")
    if intervention.donor_region.end > len(donor):
        raise ValueError("donor region falls outside its sequence")
    if intervention.recipient_region.overlaps(intervention.target):
        raise ValueError("context exchange cannot alter the masked target")
    if intervention.target.end > len(recipient):
        raise ValueError("target falls outside recipient sequence")
    replacement = donor[intervention.donor_region.start : intervention.donor_region.end]
    intervened = (
        recipient[: intervention.recipient_region.start]
        + replacement
        + recipient[intervention.recipient_region.end :]
    )
    _, reference_plan = encode_mask_plan(recipient, intervention.target, encoder)
    _, intervened_plan = encode_mask_plan(intervened, intervention.target, encoder)
    target_unchanged = (
        recipient[intervention.target.start : intervention.target.end]
        == intervened[intervention.target.start : intervention.target.end]
    )
    audit = ContextInterventionAudit(
        intervention_id=intervention.intervention_id,
        target_unchanged=target_unchanged,
        sequence_length_unchanged=len(recipient) == len(intervened),
        target_geometry_invariant=geometry_is_invariant(
            reference_plan, intervened_plan
        ),
        changed_nucleotides=sum(a != b for a, b in zip(recipient, intervened)),
        reference_plan=reference_plan,
        intervened_plan=intervened_plan,
    )
    return intervened, audit
