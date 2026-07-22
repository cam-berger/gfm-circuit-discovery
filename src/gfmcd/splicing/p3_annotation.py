"""Outcome-blind annotation safety for the P3 closure study.

The confirmatory perturbation rotates sequence inside a fixed genomic cassette.  A cassette called
``CDS`` must have a single, unambiguous coding register across the comprehensive annotation; a
cassette called ``intron`` must not overlap any annotated exon; and a non-CDS exon control must be
fully exonic while avoiding every annotated CDS.  These checks happen before a model is loaded and
therefore cannot condition the cohort on SpliceBERT outcomes.

Coordinates in this module are 0-based, half-open genomic coordinates.  Model windows are
transcript-oriented; :func:`window_block_to_genomic` performs the strand-aware conversion.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gfmcd.splicing.gtf import _open_text, attr_first, parse_attributes
from gfmcd.splicing.phase import codon_slot


@dataclass(frozen=True, order=True)
class AnnotationFeature:
    chrom: str
    start: int
    end: int
    strand: str
    gene_id: str
    transcript_id: str
    feature: str
    frame: int | None = None


@dataclass(frozen=True)
class AnnotationGateResult:
    checks: Mapping[str, bool]
    metrics: Mapping[str, object]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.checks.items() if not value)


class _IntervalTable:
    """Small immutable interval index with prefix-max pruning."""

    def __init__(self, features: Iterable[AnnotationFeature]):
        self.features = tuple(sorted(features, key=lambda x: (x.start, x.end, x.gene_id,
                                                               x.transcript_id, x.feature)))
        self.starts = tuple(x.start for x in self.features)
        running = -1
        prefix = []
        for feature in self.features:
            running = max(running, feature.end)
            prefix.append(running)
        self.prefix_max_end = tuple(prefix)

    def overlapping(self, start: int, end: int) -> tuple[AnnotationFeature, ...]:
        if start < 0 or end <= start:
            raise ValueError("invalid query interval")
        hi = bisect_left(self.starts, end)
        found = []
        index = hi - 1
        while index >= 0 and self.prefix_max_end[index] > start:
            feature = self.features[index]
            if feature.end > start:
                found.append(feature)
            index -= 1
        return tuple(sorted(found))


class AnnotationSafetyIndex:
    """Comprehensive exon/CDS index used only for sequence-level eligibility."""

    def __init__(self, *, exons: Mapping[str, Iterable[AnnotationFeature]],
                 cds: Mapping[str, Iterable[AnnotationFeature]]):
        chromosomes = set(exons) | set(cds)
        self._exons = {chrom: _IntervalTable(exons.get(chrom, ())) for chrom in chromosomes}
        self._cds = {chrom: _IntervalTable(cds.get(chrom, ())) for chrom in chromosomes}

    def exons(self, chrom: str, start: int, end: int) -> tuple[AnnotationFeature, ...]:
        table = self._exons.get(chrom)
        return () if table is None else table.overlapping(start, end)

    def cds(self, chrom: str, start: int, end: int) -> tuple[AnnotationFeature, ...]:
        table = self._cds.get(chrom)
        return () if table is None else table.overlapping(start, end)


def load_annotation_safety_index(
    path: str | Path,
    *,
    allowed_chromosomes: set[str] | None = None,
) -> AnnotationSafetyIndex:
    """Load every exon and framed CDS from a comprehensive GTF."""

    exons: dict[str, list[AnnotationFeature]] = {}
    cds: dict[str, list[AnnotationFeature]] = {}
    with _open_text(path) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9:
                raise ValueError(f"{path}:{line_no}: expected 9 GTF columns")
            chrom, _source, feature, start, end, _score, strand, frame_text, attr_text = cols
            if feature not in {"exon", "CDS"}:
                continue
            if allowed_chromosomes is not None and chrom not in allowed_chromosomes:
                continue
            attrs = parse_attributes(attr_text)
            frame = int(frame_text) if feature == "CDS" and frame_text in {"0", "1", "2"} else None
            if feature == "CDS" and frame is None:
                continue
            record = AnnotationFeature(
                chrom=chrom,
                start=int(start) - 1,
                end=int(end),
                strand=strand,
                gene_id=attr_first(attrs, "gene_id"),
                transcript_id=attr_first(attrs, "transcript_id"),
                feature=feature,
                frame=frame,
            )
            target = exons if feature == "exon" else cds
            target.setdefault(chrom, []).append(record)
    return AnnotationSafetyIndex(exons=exons, cds=cds)


def window_block_to_genomic(
    start: int,
    end: int,
    *,
    window_start: int,
    window_end: int,
    strand: str,
) -> tuple[int, int]:
    """Convert a transcript-oriented window block to a genomic half-open interval."""

    if not (0 <= start < end <= window_end - window_start):
        raise ValueError("window block lies outside genomic window")
    if strand == "+":
        return window_start + start, window_start + end
    if strand == "-":
        return window_end - end, window_end - start
    raise ValueError("strand must be '+' or '-'")


def _has_boundary_near(
    features: Iterable[AnnotationFeature],
    start: int,
    end: int,
    clearance: int,
) -> bool:
    lo, hi = start - clearance, end + clearance
    return any(lo <= feature.start <= hi or lo <= feature.end <= hi for feature in features)


def validate_annotation_block(
    index: AnnotationSafetyIndex,
    *,
    chrom: str,
    start: int,
    end: int,
    strand: str,
    gene_id: str,
    context: str,
    splice_clearance: int = 8,
) -> AnnotationGateResult:
    """Validate a CDS, non-CDS-exon, or intron cassette against comprehensive annotation.

    CDS bases must be covered by the declared gene/strand with one codon slot at every position.
    Overlapping CDS from another gene/strand, partial CDS boundaries inside the block, or any exon
    boundary within ``splice_clearance`` invalidate the block.  Intronic blocks must be clear of
    *all* comprehensive exons after the same clearance expansion.
    """

    if context not in {"cds", "non_cds_exon", "upstream_intron"}:
        raise ValueError("context must be 'cds', 'non_cds_exon', or 'upstream_intron'")
    if start < 0 or end <= start or splice_clearance < 0:
        raise ValueError("invalid block or splice clearance")

    query_start = max(0, start - splice_clearance)
    query_end = end + splice_clearance
    nearby_exons = index.exons(chrom, query_start, query_end)
    nearby_cds = index.cds(chrom, start, end)
    boundary_free = not _has_boundary_near(nearby_exons, start, end, splice_clearance)

    if context == "upstream_intron":
        checks = {
            "no_annotated_exon_overlap_with_clearance": not nearby_exons,
            "no_annotated_cds_overlap": not nearby_cds,
            "no_annotated_splice_boundary_nearby": boundary_free,
        }
        return AnnotationGateResult(
            checks=checks,
            metrics={"n_nearby_exons": len(nearby_exons), "n_overlapping_cds": len(nearby_cds)},
        )

    if context == "non_cds_exon":
        overlapping_exons = index.exons(chrom, start, end)
        matching_exons = tuple(
            feature for feature in overlapping_exons
            if feature.gene_id == gene_id and feature.strand == strand
        )
        conflicting_exons = tuple(
            feature for feature in overlapping_exons
            if feature.gene_id != gene_id or feature.strand != strand
        )
        uncovered = tuple(
            position for position in range(start, end)
            if not any(feature.start <= position < feature.end for feature in matching_exons)
        )
        checks = {
            "exon_fully_covered": not uncovered,
            "no_conflicting_exon_overlap": not conflicting_exons,
            "no_annotated_cds_overlap": not nearby_cds,
            "no_annotated_splice_boundary_nearby": boundary_free,
        }
        return AnnotationGateResult(
            checks=checks,
            metrics={
                "n_matching_exons": len(matching_exons),
                "n_conflicting_exons": len(conflicting_exons),
                "n_overlapping_cds": len(nearby_cds),
                "uncovered_bases": uncovered,
            },
        )

    matching = tuple(
        feature for feature in nearby_cds
        if feature.gene_id == gene_id and feature.strand == strand
    )
    conflicting = tuple(
        feature for feature in nearby_cds
        if feature.gene_id != gene_id or feature.strand != strand
    )
    slots_by_position = []
    uncovered = []
    ambiguous = []
    for position in range(start, end):
        slots = {
            codon_slot(position, feature.start, feature.end, feature.strand, int(feature.frame))
            for feature in matching
            if feature.start <= position < feature.end
        }
        slots_by_position.append(tuple(sorted(slots)))
        if not slots:
            uncovered.append(position)
        elif len(slots) != 1:
            ambiguous.append(position)
    partial_boundaries = tuple(
        feature for feature in matching
        if start < feature.start < end or start < feature.end < end
    )
    checks = {
        "cds_fully_covered": not uncovered,
        "cds_register_unambiguous": not ambiguous,
        "no_conflicting_cds_overlap": not conflicting,
        "no_partial_cds_boundary_inside": not partial_boundaries,
        "no_annotated_splice_boundary_nearby": boundary_free,
    }
    unique_slots = tuple(sorted({slot[0] for slot in slots_by_position if len(slot) == 1}))
    return AnnotationGateResult(
        checks=checks,
        metrics={
            "n_matching_cds": len(matching),
            "n_conflicting_cds": len(conflicting),
            "uncovered_bases": tuple(uncovered),
            "ambiguous_bases": tuple(ambiguous),
            "codon_slots_present": unique_slots,
        },
    )


__all__ = [
    "AnnotationFeature",
    "AnnotationGateResult",
    "AnnotationSafetyIndex",
    "load_annotation_safety_index",
    "validate_annotation_block",
    "window_block_to_genomic",
]
