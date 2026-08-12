"""Dataset utilities for the DNABERT promoter experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from gfmcd.adapters.dnabert import normalize_sequence


DNA_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a normalized DNA sequence."""

    sequence = normalize_sequence(sequence)
    return sequence.translate(DNA_COMPLEMENT)[::-1]


def sequence_sha256(sequence: str) -> str:
    """Stable hash for one normalized sequence."""

    sequence = normalize_sequence(sequence)
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def gc_fraction(sequence: str) -> float:
    """Fraction of canonical nucleotides that are G or C."""

    sequence = normalize_sequence(sequence)
    canonical = [base for base in sequence if base in "ACGT"]

    if not canonical:
        return 0.0

    return sum(base in "GC" for base in canonical) / len(canonical)


@dataclass(frozen=True)
class PromoterRecord:
    split: str
    source_index: int
    sequence: str
    label: int
    length: int
    gc_fraction: float
    sequence_sha256: str
    reverse_complement_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_records(
    sequences: Iterable[str],
    labels: Iterable[int],
    *,
    split: str,
) -> list[PromoterRecord]:
    """Normalize one split and construct reproducible records."""

    sequence_list = list(sequences)
    label_list = list(labels)

    if len(sequence_list) != len(label_list):
        raise ValueError("sequence and label counts disagree")

    records = []

    for index, (sequence, label) in enumerate(
        zip(sequence_list, label_list)
    ):
        normalized = normalize_sequence(str(sequence))
        integer_label = int(label)

        if integer_label not in {0, 1}:
            raise ValueError(
                f"unexpected binary label at index {index}: {label!r}"
            )

        reverse = reverse_complement(normalized)

        records.append(
            PromoterRecord(
                split=split,
                source_index=index,
                sequence=normalized,
                label=integer_label,
                length=len(normalized),
                gc_fraction=gc_fraction(normalized),
                sequence_sha256=sequence_sha256(normalized),
                reverse_complement_sha256=sequence_sha256(reverse),
            )
        )

    return records


def split_summary(records: list[PromoterRecord]) -> dict:
    """Return stable descriptive statistics for one split."""

    if not records:
        raise ValueError("split cannot be empty")

    labels = Counter(record.label for record in records)
    lengths = [record.length for record in records]
    gc_values = [record.gc_fraction for record in records]

    return {
        "rows": len(records),
        "label_counts": {
            str(label): count
            for label, count in sorted(labels.items())
        },
        "minimum_length": min(lengths),
        "maximum_length": max(lengths),
        "unique_lengths": sorted(set(lengths)),
        "mean_gc_fraction": sum(gc_values) / len(gc_values),
        "minimum_gc_fraction": min(gc_values),
        "maximum_gc_fraction": max(gc_values),
        "unique_sequence_count": len(
            {record.sequence_sha256 for record in records}
        ),
    }


def exact_duplicate_report(
    records_by_split: dict[str, list[PromoterRecord]],
) -> dict:
    """Find exact sequence overlaps within and across splits."""

    hash_to_rows: dict[str, list[dict]] = {}

    for split, records in records_by_split.items():
        for record in records:
            hash_to_rows.setdefault(
                record.sequence_sha256,
                [],
            ).append(
                {
                    "split": split,
                    "source_index": record.source_index,
                    "label": record.label,
                }
            )

    duplicate_groups = [
        {
            "sequence_sha256": sequence_hash,
            "occurrences": occurrences,
        }
        for sequence_hash, occurrences in hash_to_rows.items()
        if len(occurrences) > 1
    ]

    cross_split_groups = [
        group
        for group in duplicate_groups
        if len(
            {
                occurrence["split"]
                for occurrence in group["occurrences"]
            }
        ) > 1
    ]

    conflicting_label_groups = [
        group
        for group in duplicate_groups
        if len(
            {
                occurrence["label"]
                for occurrence in group["occurrences"]
            }
        ) > 1
    ]

    return {
        "duplicate_group_count": len(duplicate_groups),
        "cross_split_duplicate_group_count": len(cross_split_groups),
        "conflicting_label_duplicate_group_count": len(
            conflicting_label_groups
        ),
        "duplicate_groups": duplicate_groups,
        "cross_split_duplicate_groups": cross_split_groups,
        "conflicting_label_duplicate_groups": conflicting_label_groups,
    }


def reverse_complement_overlap_report(
    records_by_split: dict[str, list[PromoterRecord]],
) -> dict:
    """Find sequences whose reverse complement occurs elsewhere."""

    exact_lookup: dict[str, list[dict]] = {}

    for split, records in records_by_split.items():
        for record in records:
            exact_lookup.setdefault(
                record.sequence_sha256,
                [],
            ).append(
                {
                    "split": split,
                    "source_index": record.source_index,
                    "label": record.label,
                }
            )

    overlaps = []

    for split, records in records_by_split.items():
        for record in records:
            matches = exact_lookup.get(
                record.reverse_complement_sha256,
                [],
            )

            for match in matches:
                same_row = (
                    match["split"] == split
                    and match["source_index"] == record.source_index
                )

                if same_row:
                    continue

                overlaps.append(
                    {
                        "source_split": split,
                        "source_index": record.source_index,
                        "source_label": record.label,
                        "matched_split": match["split"],
                        "matched_index": match["source_index"],
                        "matched_label": match["label"],
                    }
                )

    # A pair is discovered in both directions. Canonicalize it.
    canonical = {}

    for overlap in overlaps:
        left = (
            overlap["source_split"],
            overlap["source_index"],
        )
        right = (
            overlap["matched_split"],
            overlap["matched_index"],
        )

        key = tuple(sorted((left, right)))
        canonical[key] = overlap

    unique_overlaps = list(canonical.values())

    cross_split = [
        overlap
        for overlap in unique_overlaps
        if overlap["source_split"] != overlap["matched_split"]
    ]

    conflicting_labels = [
        overlap
        for overlap in unique_overlaps
        if overlap["source_label"] != overlap["matched_label"]
    ]

    return {
        "overlap_pair_count": len(unique_overlaps),
        "cross_split_overlap_pair_count": len(cross_split),
        "conflicting_label_overlap_pair_count": len(
            conflicting_labels
        ),
        "overlaps": unique_overlaps,
        "cross_split_overlaps": cross_split,
        "conflicting_label_overlaps": conflicting_labels,
    }


def records_sha256(records: list[PromoterRecord]) -> str:
    """Hash an ordered split for exact reproducibility."""

    payload = [
        {
            "source_index": record.source_index,
            "sequence": record.sequence,
            "label": record.label,
        }
        for record in records
    ]

    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
