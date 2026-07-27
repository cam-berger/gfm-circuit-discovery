"""Validated, nucleotide-coordinate schemas for reconstruction-atlas collections."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

DNA_ALPHABET = frozenset("ACGTN")


class EvidenceType(str, Enum):
    EXPERIMENTAL = "experimentally_supported"
    CURATED = "curated"
    DERIVED = "derived_from_curated_annotation"
    COMPUTATIONAL = "computational_prediction"


class ControlFamily(str, Enum):
    SAME_MOTIF_UNANNOTATED = "same_motif_unannotated"
    COMPOSITION_TOKEN_MATCHED = "composition_token_matched"
    SAME_SEQUENCE_NON_SITE = "same_sequence_non_site"
    LOCAL_FREQUENCY_BASELINE = "local_frequency_baseline"


@dataclass(frozen=True, order=True)
class Interval:
    """Zero-based, half-open interval on the stored sequence."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("interval start must be non-negative")
        if self.end <= self.start:
            raise ValueError("interval must have positive length")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Interval) -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, other: Interval) -> bool:
        return self.start <= other.start and self.end >= other.end


@dataclass(frozen=True)
class Evidence:
    source_id: str
    source_version: str
    evidence_type: EvidenceType
    record_id: str

    def __post_init__(self) -> None:
        for name in ("source_id", "source_version", "record_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"evidence {name} cannot be blank")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Evidence:
        return cls(
            source_id=str(value["source_id"]),
            source_version=str(value["source_version"]),
            evidence_type=EvidenceType(value["evidence_type"]),
            record_id=str(value["record_id"]),
        )


@dataclass(frozen=True)
class BiologicalElement:
    element_id: str
    site_class: str
    interval: Interval
    evidence: Evidence
    sequence: str | None = None
    paired_element_ids: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.element_id.strip() or not self.site_class.strip():
            raise ValueError("element_id and site_class cannot be blank")
        if self.sequence is not None:
            normalized = self.sequence.upper()
            if set(normalized) - DNA_ALPHABET:
                raise ValueError(
                    f"element {self.element_id} contains unsupported bases"
                )
            if len(normalized) != self.interval.length:
                raise ValueError(
                    f"element {self.element_id} sequence length does not match its interval"
                )
            object.__setattr__(self, "sequence", normalized)
        object.__setattr__(self, "paired_element_ids", tuple(self.paired_element_ids))
        object.__setattr__(self, "attributes", dict(self.attributes))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BiologicalElement:
        return cls(
            element_id=str(value["element_id"]),
            site_class=str(value["site_class"]),
            interval=Interval(int(value["start"]), int(value["end"])),
            evidence=Evidence.from_dict(value["evidence"]),
            sequence=value.get("sequence"),
            paired_element_ids=tuple(
                str(x) for x in value.get("paired_element_ids", ())
            ),
            attributes=dict(value.get("attributes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        interval = value.pop("interval")
        value["start"] = interval["start"]
        value["end"] = interval["end"]
        value["evidence"]["evidence_type"] = self.evidence.evidence_type.value
        return value


@dataclass(frozen=True)
class ContextRegion:
    region_id: str
    region_type: str
    interval: Interval
    target_element_ids: tuple[str, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.region_id.strip() or not self.region_type.strip():
            raise ValueError("region_id and region_type cannot be blank")
        if not self.target_element_ids:
            raise ValueError("context region must name at least one target element")
        object.__setattr__(self, "target_element_ids", tuple(self.target_element_ids))
        object.__setattr__(self, "attributes", dict(self.attributes))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContextRegion:
        return cls(
            region_id=str(value["region_id"]),
            region_type=str(value["region_type"]),
            interval=Interval(int(value["start"]), int(value["end"])),
            target_element_ids=tuple(str(x) for x in value["target_element_ids"]),
            attributes=dict(value.get("attributes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        interval = value.pop("interval")
        value["start"] = interval["start"]
        value["end"] = interval["end"]
        return value


@dataclass(frozen=True)
class AnnotatedSequence:
    """One genomic window with transcription-aware nucleotide labels."""

    example_id: str
    sequence: str
    assembly: str
    chrom: str
    genomic_start: int
    genomic_end: int
    strand: str
    sequence_orientation: str
    split: str
    homology_cluster: str
    elements: tuple[BiologicalElement, ...]
    context_regions: tuple[ContextRegion, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sequence = self.sequence.upper()
        if not sequence or set(sequence) - DNA_ALPHABET:
            raise ValueError(
                f"example {self.example_id} has unsupported or empty sequence"
            )
        object.__setattr__(self, "sequence", sequence)
        if not self.example_id.strip():
            raise ValueError("example_id cannot be blank")
        if self.strand not in {"+", "-"}:
            raise ValueError("strand must be '+' or '-'")
        if self.sequence_orientation not in {"genomic_forward", "transcription_5_to_3"}:
            raise ValueError("unsupported sequence_orientation")
        if self.genomic_start < 0 or self.genomic_end <= self.genomic_start:
            raise ValueError("invalid genomic interval")
        if self.genomic_end - self.genomic_start != len(sequence):
            raise ValueError(
                "genomic interval length must equal stored sequence length"
            )
        if not self.split.strip() or not self.homology_cluster.strip():
            raise ValueError("split and homology_cluster are required")
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(self, "context_regions", tuple(self.context_regions))
        object.__setattr__(self, "metadata", dict(self.metadata))
        element_ids = [element.element_id for element in self.elements]
        if not element_ids:
            raise ValueError(
                "annotated sequence must contain at least one biological element"
            )
        if len(element_ids) != len(set(element_ids)):
            raise ValueError(f"example {self.example_id} has duplicate element IDs")
        region_ids = [region.region_id for region in self.context_regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError(
                f"example {self.example_id} has duplicate context-region IDs"
            )
        window = Interval(0, len(sequence))
        known = set(element_ids)
        for element in self.elements:
            if not window.contains(element.interval):
                raise ValueError(
                    f"element {element.element_id} falls outside its sequence"
                )
            observed = sequence[element.interval.start : element.interval.end]
            if element.sequence is not None and element.sequence != observed:
                raise ValueError(
                    f"element {element.element_id} sequence disagrees with the stored window"
                )
            unknown_pairs = set(element.paired_element_ids) - known
            if unknown_pairs:
                raise ValueError(
                    f"element {element.element_id} names absent paired elements: "
                    f"{sorted(unknown_pairs)}"
                )
        for region in self.context_regions:
            if not window.contains(region.interval):
                raise ValueError(
                    f"context region {region.region_id} falls outside its sequence"
                )
            unknown_targets = set(region.target_element_ids) - known
            if unknown_targets:
                raise ValueError(
                    f"context region {region.region_id} names absent targets: "
                    f"{sorted(unknown_targets)}"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AnnotatedSequence:
        return cls(
            example_id=str(value["example_id"]),
            sequence=str(value["sequence"]),
            assembly=str(value["assembly"]),
            chrom=str(value["chrom"]),
            genomic_start=int(value["genomic_start"]),
            genomic_end=int(value["genomic_end"]),
            strand=str(value["strand"]),
            sequence_orientation=str(value["sequence_orientation"]),
            split=str(value["split"]),
            homology_cluster=str(value["homology_cluster"]),
            elements=tuple(BiologicalElement.from_dict(x) for x in value["elements"]),
            context_regions=tuple(
                ContextRegion.from_dict(x) for x in value.get("context_regions", ())
            ),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "sequence": self.sequence,
            "assembly": self.assembly,
            "chrom": self.chrom,
            "genomic_start": self.genomic_start,
            "genomic_end": self.genomic_end,
            "strand": self.strand,
            "sequence_orientation": self.sequence_orientation,
            "split": self.split,
            "homology_cluster": self.homology_cluster,
            "elements": [element.to_dict() for element in self.elements],
            "context_regions": [region.to_dict() for region in self.context_regions],
            "metadata": dict(self.metadata),
        }

    def element(self, element_id: str) -> BiologicalElement:
        matches = [
            element for element in self.elements if element.element_id == element_id
        ]
        if len(matches) != 1:
            raise KeyError(
                f"example {self.example_id} has no unique element {element_id!r}"
            )
        return matches[0]


@dataclass(frozen=True)
class SiteReference:
    example_id: str
    interval: Interval
    site_class: str
    element_id: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SiteReference:
        return cls(
            example_id=str(value["example_id"]),
            interval=Interval(int(value["start"]), int(value["end"])),
            site_class=str(value["site_class"]),
            element_id=str(value["element_id"])
            if value.get("element_id") is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "start": self.interval.start,
            "end": self.interval.end,
            "site_class": self.site_class,
            "element_id": self.element_id,
        }


@dataclass(frozen=True)
class MatchedControl:
    family: ControlFamily
    site: SiteReference
    match_distance: float
    matched_features: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.match_distance < 0:
            raise ValueError("match distance cannot be negative")
        object.__setattr__(self, "matched_features", dict(self.matched_features))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MatchedControl:
        return cls(
            family=ControlFamily(value["family"]),
            site=SiteReference.from_dict(value["site"]),
            match_distance=float(value["match_distance"]),
            matched_features=dict(value.get("matched_features", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "site": self.site.to_dict(),
            "match_distance": self.match_distance,
            "matched_features": dict(self.matched_features),
        }


@dataclass(frozen=True)
class MatchedSet:
    target: SiteReference
    controls: tuple[MatchedControl, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "controls", tuple(self.controls))
        present = {control.family for control in self.controls}
        required = {
            ControlFamily.SAME_MOTIF_UNANNOTATED,
            ControlFamily.COMPOSITION_TOKEN_MATCHED,
            ControlFamily.SAME_SEQUENCE_NON_SITE,
        }
        missing = required - present
        if missing:
            raise ValueError(
                f"matched set lacks empirical control families: {sorted(missing)}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MatchedSet:
        return cls(
            target=SiteReference.from_dict(value["target"]),
            controls=tuple(MatchedControl.from_dict(x) for x in value["controls"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "controls": [control.to_dict() for control in self.controls],
        }


def load_collection(path: str | Path) -> list[AnnotatedSequence]:
    records = _load_jsonl(path, AnnotatedSequence.from_dict)
    ids = [record.example_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("collection contains duplicate example IDs")
    return records


def load_matched_sets(path: str | Path) -> list[MatchedSet]:
    return _load_jsonl(path, MatchedSet.from_dict)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return output


def _load_jsonl(path: str | Path, loader: Any) -> list[Any]:
    records: list[Any] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(loader(value))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path} contains no records")
    return records
