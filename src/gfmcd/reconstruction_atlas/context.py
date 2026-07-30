"""Target-preserving context exchanges and their token-geometry audits."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from .schema import Interval
from .tokenization import (
    MaskPlan,
    encode_mask_plan,
    geometry_is_invariant,
    geometry_qualified_hamming_one_alternatives,
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


@dataclass(frozen=True)
class ContextWindowSpec:
    """Class-specific context widths around the immutable scored BPE cover."""

    upstream_nt: int
    downstream_nt: int
    irrelevant_gap_nt: int

    def __post_init__(self) -> None:
        if self.upstream_nt < 0 or self.downstream_nt < 0:
            raise ValueError("context widths cannot be negative")
        if self.upstream_nt + self.downstream_nt < 2:
            raise ValueError("context intervention must span at least two nucleotides")
        if self.irrelevant_gap_nt < 1:
            raise ValueError("irrelevant context gap must be positive")


@dataclass(frozen=True)
class TargetPreservingContextVariants:
    """A matched-dose structured/irrelevant pair with a shared margin family."""

    reference: str
    structured_shuffle: str
    irrelevant_shuffle: str
    target: Interval
    mask_plan: MaskPlan
    structured_spans: tuple[Interval, ...]
    irrelevant_spans: tuple[Interval, ...]
    structured_changed_nucleotides: int
    irrelevant_changed_nucleotides: int
    alternatives: dict[str, str]

    def __post_init__(self) -> None:
        if self.structured_changed_nucleotides <= 0:
            raise ValueError("structured context shuffle must change nucleotides")
        if self.structured_changed_nucleotides != self.irrelevant_changed_nucleotides:
            raise ValueError("structured and irrelevant corruption doses must match")
        if not self.alternatives:
            raise ValueError("context variants require a non-empty margin family")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": {
                "start": self.target.start,
                "end": self.target.end,
            },
            "mask_plan": self.mask_plan.to_dict(),
            "structured_spans": [
                {"start": span.start, "end": span.end} for span in self.structured_spans
            ],
            "irrelevant_spans": [
                {"start": span.start, "end": span.end} for span in self.irrelevant_spans
            ],
            "structured_changed_nucleotides": (self.structured_changed_nucleotides),
            "irrelevant_changed_nucleotides": self.irrelevant_changed_nucleotides,
            "alternative_names": sorted(self.alternatives),
            "eligible_alternative_count": len(self.alternatives),
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


def build_target_preserving_context_variants(
    sequence: str,
    target: Interval,
    spec: ContextWindowSpec,
    *,
    encoder: Any,
    seed: int,
    candidate_shuffles: int = 512,
) -> TargetPreservingContextVariants | None:
    """Build a class-specific context corruption and a dose-matched local null.

    Both interventions preserve sequence length, the complete scored BPE cover, and
    mononucleotide composition over the edited spans. The null uses equally sized
    spans displaced farther from the target on the same side(s). A pair is accepted
    only when the number of changed nucleotide positions is exactly equal and the
    same geometry-qualified Hamming-one alternative family can score all variants.
    """

    if candidate_shuffles < 1:
        raise ValueError("candidate_shuffles must be positive")
    reference = normalize_dna(sequence)
    _, plan = encode_mask_plan(reference, target, encoder)
    spans = _context_spans(plan, spec, len(reference), irrelevant=False)
    irrelevant_spans = _context_spans(plan, spec, len(reference), irrelevant=True)
    if spans is None or irrelevant_spans is None:
        return None
    if any(left.overlaps(right) for left in spans for right in irrelevant_spans):
        return None
    structured = _candidate_shuffles(
        reference,
        spans,
        seed=_derived_seed(seed, "structured"),
        candidates=candidate_shuffles,
    )
    irrelevant = _candidate_shuffles(
        reference,
        irrelevant_spans,
        seed=_derived_seed(seed, "irrelevant"),
        candidates=candidate_shuffles,
    )
    shared_doses = sorted(set(structured) & set(irrelevant), reverse=True)
    for dose in shared_doses:
        structured_sequence = structured[dose]
        irrelevant_sequence = irrelevant[dose]
        comparisons = [
            encode_mask_plan(value, target, encoder)[1]
            for value in (structured_sequence, irrelevant_sequence)
        ]
        if not all(geometry_is_invariant(plan, item) for item in comparisons):
            continue
        alternative_families = [
            geometry_qualified_hamming_one_alternatives(value, item, encoder)
            for value, item in (
                (reference, plan),
                (structured_sequence, comparisons[0]),
                (irrelevant_sequence, comparisons[1]),
            )
        ]
        common_names = set(alternative_families[0])
        for family in alternative_families[1:]:
            common_names &= set(family)
        alternatives = {
            name: alternative_families[0][name] for name in sorted(common_names)
        }
        if not alternatives:
            continue
        return TargetPreservingContextVariants(
            reference=reference,
            structured_shuffle=structured_sequence,
            irrelevant_shuffle=irrelevant_sequence,
            target=target,
            mask_plan=plan,
            structured_spans=spans,
            irrelevant_spans=irrelevant_spans,
            structured_changed_nucleotides=dose,
            irrelevant_changed_nucleotides=dose,
            alternatives=alternatives,
        )
    return None


def _context_spans(
    plan: MaskPlan,
    spec: ContextWindowSpec,
    sequence_length: int,
    *,
    irrelevant: bool,
) -> tuple[Interval, ...] | None:
    spans: list[Interval] = []
    gap = spec.irrelevant_gap_nt if irrelevant else 0
    if spec.upstream_nt:
        end = plan.covered.start - gap
        start = end - spec.upstream_nt
        if start < 0:
            return None
        spans.append(Interval(start, end))
    if spec.downstream_nt:
        start = plan.covered.end + gap
        end = start + spec.downstream_nt
        if end > sequence_length:
            return None
        spans.append(Interval(start, end))
    return tuple(spans)


def _candidate_shuffles(
    reference: str,
    spans: tuple[Interval, ...],
    *,
    seed: int,
    candidates: int,
) -> dict[int, str]:
    source = "".join(reference[span.start : span.end] for span in spans)
    if len(set(source)) < 2:
        return {}
    rng = random.Random(seed)
    output: dict[int, str] = {}
    letters = list(source)
    for _ in range(candidates):
        shuffled = letters.copy()
        rng.shuffle(shuffled)
        edited = list(reference)
        cursor = 0
        for span in spans:
            width = span.length
            edited[span.start : span.end] = shuffled[cursor : cursor + width]
            cursor += width
        sequence = "".join(edited)
        changed = sum(a != b for a, b in zip(reference, sequence))
        if changed == 0 or changed in output:
            continue
        output[changed] = sequence
    return output


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
