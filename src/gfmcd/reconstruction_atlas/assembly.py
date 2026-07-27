"""Assembly of strand-aware annotated windows from normalized genomic features."""

from __future__ import annotations

import gzip
import hashlib
import heapq
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, TextIO

from .schema import (
    AnnotatedSequence,
    BiologicalElement,
    ContextRegion,
    Evidence,
    EvidenceType,
    Interval,
)

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@dataclass(frozen=True)
class GenomicFeature:
    feature_id: str
    chrom: str
    start: int
    end: int
    strand: str
    site_class: str
    evidence: Evidence
    homology_cluster: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("invalid genomic feature interval")
        if self.strand not in {"+", "-"}:
            raise ValueError("feature strand must be '+' or '-'")
        if not self.feature_id or not self.chrom or not self.site_class:
            raise ValueError("feature identity fields cannot be blank")
        if not self.homology_cluster:
            raise ValueError("feature homology_cluster cannot be blank")
        object.__setattr__(self, "attributes", dict(self.attributes))


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1].upper()


def assemble_annotated_windows(
    features: Iterable[GenomicFeature],
    fasta: Any,
    *,
    annotation_features: Iterable[GenomicFeature] | None = None,
    assembly: str,
    upstream_nt: int,
    downstream_nt: int,
    split_seed: int,
    development_percent: int = 40,
    baseline_training_percent: int = 0,
    immediate_flank_nt: int = 25,
) -> list[AnnotatedSequence]:
    """Create one target-centred, transcription-oriented example per feature."""

    feature_list = sorted(
        features,
        key=lambda item: (item.chrom, item.start, item.end, item.feature_id),
    )
    if not feature_list:
        raise ValueError("no genomic features supplied")
    if min(upstream_nt, downstream_nt, immediate_flank_nt) < 0:
        raise ValueError("window sizes cannot be negative")
    _validate_split_percentages(development_percent, baseline_training_percent)
    annotation_list = (
        feature_list
        if annotation_features is None
        else sorted(
            annotation_features,
            key=lambda item: (item.chrom, item.start, item.end, item.feature_id),
        )
    )
    grouped: dict[tuple[str, str], list[GenomicFeature]] = {}
    for feature in annotation_list:
        grouped.setdefault((feature.chrom, feature.strand), []).append(feature)
    output: list[AnnotatedSequence] = []
    for target in feature_list:
        if target.strand == "+":
            window_start = target.start - upstream_nt
            window_end = target.end + downstream_nt
        else:
            window_start = target.start - downstream_nt
            window_end = target.end + upstream_nt
        if window_start < 0 or target.chrom not in fasta:
            continue
        raw = str(fasta[target.chrom][window_start:window_end]).upper()
        if len(raw) != window_end - window_start or set(raw) - set("ACGTN"):
            continue
        sequence = raw if target.strand == "+" else reverse_complement(raw)
        overlapping = [
            feature
            for feature in grouped[(target.chrom, target.strand)]
            if feature.start >= window_start and feature.end <= window_end
        ]
        overlapping_ids = {feature.feature_id for feature in overlapping}
        elements = tuple(
            _local_element(
                feature,
                sequence,
                window_start,
                window_end,
                target.strand,
                overlapping_ids,
            )
            for feature in overlapping
        )
        target_local_id = target.feature_id
        target_element = next(
            element for element in elements if element.element_id == target_local_id
        )
        context_regions = _default_context_regions(
            target_element,
            elements=elements,
            sequence_length=len(sequence),
            immediate_flank_nt=immediate_flank_nt,
        )
        split = _split_for_cluster(
            target.homology_cluster,
            seed=split_seed,
            development_percent=development_percent,
            baseline_training_percent=baseline_training_percent,
        )
        example_id = _example_id(target, window_start, window_end)
        output.append(
            AnnotatedSequence(
                example_id=example_id,
                sequence=sequence,
                assembly=assembly,
                chrom=target.chrom,
                genomic_start=window_start,
                genomic_end=window_end,
                strand=target.strand,
                sequence_orientation="transcription_5_to_3",
                split=split,
                homology_cluster=target.homology_cluster,
                elements=elements,
                context_regions=context_regions,
                metadata={
                    "target_element_id": target.feature_id,
                    "target_site_class": target.site_class,
                    "window_upstream_nt": upstream_nt,
                    "window_downstream_nt": downstream_nt,
                },
            )
        )
    if not output:
        raise ValueError("no complete annotated windows could be assembled")
    return output


def read_normalized_bed(
    path: str | Path,
    *,
    source_id: str,
    source_version: str,
    evidence_type: EvidenceType,
) -> list[GenomicFeature]:
    """Read the atlas BED contract.

    Required tab-separated columns:
    ``chrom start end feature_id score strand site_class homology_cluster``.
    Additional ``key=value`` columns become attributes.
    """

    features: list[GenomicFeature] = []
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                raise ValueError(f"{path}:{line_number}: expected at least 8 columns")
            chrom, start, end, feature_id, _score, strand, site_class, cluster = fields[
                :8
            ]
            attributes: dict[str, str] = {}
            for item in fields[8:]:
                if "=" not in item:
                    raise ValueError(
                        f"{path}:{line_number}: extra columns must be key=value"
                    )
                key, value = item.split("=", 1)
                attributes[key] = value
            features.append(
                GenomicFeature(
                    feature_id=feature_id,
                    chrom=chrom,
                    start=int(start),
                    end=int(end),
                    strand=strand,
                    site_class=site_class,
                    evidence=Evidence(
                        source_id=source_id,
                        source_version=source_version,
                        evidence_type=evidence_type,
                        record_id=feature_id,
                    ),
                    homology_cluster=cluster,
                    attributes=attributes,
                )
            )
    return features


def gencode_features_from_gtf(
    path: str | Path,
    *,
    source_version: str,
    include_gene_ids: frozenset[str] | set[str] | None = None,
) -> list[GenomicFeature]:
    """Derive splice dinucleotides, start/stop codons, and contiguous Kozak spans."""

    transcripts: dict[str, dict[str, Any]] = {}
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            chrom, _source, feature_type, start, end, _score, strand, _frame, attrs = (
                fields
            )
            if feature_type not in {"exon", "start_codon", "stop_codon"}:
                continue
            parsed = _parse_gtf_attributes(attrs)
            transcript_id = parsed.get("transcript_id")
            gene_id = parsed.get("gene_id")
            if transcript_id is None or gene_id is None:
                continue
            if include_gene_ids is not None and gene_id not in include_gene_ids:
                continue
            record = transcripts.setdefault(
                transcript_id,
                {
                    "chrom": chrom,
                    "strand": strand,
                    "gene_id": gene_id,
                    "gene_name": parsed.get("gene_name", gene_id),
                    "exon": [],
                    "start_codon": [],
                    "stop_codon": [],
                },
            )
            # GTF is 1-based inclusive; atlas is 0-based half-open.
            record[feature_type].append((int(start) - 1, int(end)))
    evidence_type = EvidenceType.DERIVED
    features: list[GenomicFeature] = []
    for transcript_id, record in sorted(transcripts.items()):
        chrom = record["chrom"]
        strand = record["strand"]
        cluster = record["gene_id"]
        common = {
            "transcript_id": transcript_id,
            "gene_id": record["gene_id"],
            "gene_name": record["gene_name"],
        }
        exons = sorted(record["exon"], reverse=strand == "-")
        for intron_index, (upstream, downstream) in enumerate(pairwise(exons), start=1):
            if strand == "+":
                intron_start, intron_end = upstream[1], downstream[0]
                donor = (intron_start, intron_start + 2)
                acceptor = (intron_end - 2, intron_end)
            else:
                intron_start, intron_end = downstream[1], upstream[0]
                donor = (intron_end - 2, intron_end)
                acceptor = (intron_start, intron_start + 2)
            if intron_end - intron_start < 4:
                continue
            donor_id = f"{transcript_id}:intron{intron_index}:donor"
            acceptor_id = f"{transcript_id}:intron{intron_index}:acceptor"
            features.extend(
                (
                    _derived_feature(
                        donor_id,
                        chrom,
                        donor,
                        strand,
                        "splice.donor",
                        source_version,
                        evidence_type,
                        cluster,
                        {**common, "paired_element_id": acceptor_id},
                    ),
                    _derived_feature(
                        acceptor_id,
                        chrom,
                        acceptor,
                        strand,
                        "splice.acceptor",
                        source_version,
                        evidence_type,
                        cluster,
                        {**common, "paired_element_id": donor_id},
                    ),
                )
            )
        for feature_type, site_class in (
            ("start_codon", "translation.start_codon"),
            ("stop_codon", "translation.stop_codon"),
        ):
            intervals = record[feature_type]
            if len(intervals) != 1 or intervals[0][1] - intervals[0][0] != 3:
                continue
            interval = intervals[0]
            feature_id = f"{transcript_id}:{feature_type}"
            features.append(
                _derived_feature(
                    feature_id,
                    chrom,
                    interval,
                    strand,
                    site_class,
                    source_version,
                    evidence_type,
                    cluster,
                    common,
                )
            )
            if feature_type == "start_codon":
                kozak = (
                    (interval[0] - 6, interval[0] + 4)
                    if strand == "+"
                    else (interval[1] - 4, interval[1] + 6)
                )
                if kozak[0] >= 0:
                    features.append(
                        _derived_feature(
                            f"{transcript_id}:kozak",
                            chrom,
                            kozak,
                            strand,
                            "translation.kozak",
                            source_version,
                            evidence_type,
                            cluster,
                            common,
                        )
                    )
    return features


def select_gencode_gene_ids(
    path: str | Path,
    *,
    seed: int,
    maximum_per_split: int,
    development_percent: int,
    baseline_training_percent: int,
    gene_types: frozenset[str] | set[str] = frozenset({"protein_coding"}),
    chromosomes: frozenset[str] | set[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Outcome-blind, bounded min-hash sampling from GENCODE ``gene`` rows.

    GENCODE v50 basic contains millions of transcript feature rows. Sampling gene
    clusters before transcript expansion keeps atlas construction bounded while
    preserving cluster-first assignment and a sealed confirmation split.
    """

    if maximum_per_split < 1:
        raise ValueError("maximum_per_split must be positive")
    _validate_split_percentages(development_percent, baseline_training_percent)
    allowed_types = frozenset(gene_types)
    if not allowed_types:
        raise ValueError("gene_types cannot be empty")
    allowed_chromosomes = None if chromosomes is None else frozenset(chromosomes)
    heaps: dict[str, list[tuple[int, str]]] = {
        "development": [],
        "baseline_training": [],
        "confirmation": [],
    }
    seen: set[str] = set()
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            chrom = fields[0]
            if allowed_chromosomes is not None and chrom not in allowed_chromosomes:
                continue
            parsed = _parse_gtf_attributes(fields[8])
            gene_id = parsed.get("gene_id")
            gene_type = parsed.get("gene_type")
            if gene_id is None or gene_id in seen or gene_type not in allowed_types:
                continue
            seen.add(gene_id)
            split = _split_for_cluster(
                gene_id,
                seed=seed,
                development_percent=development_percent,
                baseline_training_percent=baseline_training_percent,
            )
            priority = int(
                hashlib.sha256(f"{seed}|sample|{gene_id}".encode()).hexdigest(),
                16,
            )
            heap = heaps[split]
            item = (-priority, gene_id)
            if len(heap) < maximum_per_split:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    return {
        split: tuple(
            gene_id
            for _negative_priority, gene_id in sorted(
                heap,
                key=lambda item: (-item[0], item[1]),
            )
        )
        for split, heap in heaps.items()
    }


def sample_gencode_features(
    features: Iterable[GenomicFeature],
    *,
    seed: int,
    maximum_clusters_per_site_class: int,
    development_percent: int,
    baseline_training_percent: int,
) -> list[GenomicFeature]:
    """Select at most one target per gene × site class, then bound each split."""

    if maximum_clusters_per_site_class < 1:
        raise ValueError("maximum_clusters_per_site_class must be positive")
    _validate_split_percentages(development_percent, baseline_training_percent)
    per_cluster: dict[tuple[str, str, str], GenomicFeature] = {}
    for feature in features:
        split = _split_for_cluster(
            feature.homology_cluster,
            seed=seed,
            development_percent=development_percent,
            baseline_training_percent=baseline_training_percent,
        )
        key = (split, feature.site_class, feature.homology_cluster)
        previous = per_cluster.get(key)
        if previous is None or _feature_priority(feature, seed) < _feature_priority(
            previous, seed
        ):
            per_cluster[key] = feature
    grouped: dict[tuple[str, str], list[GenomicFeature]] = {}
    for (split, site_class, _cluster), feature in per_cluster.items():
        grouped.setdefault((split, site_class), []).append(feature)
    selected = [
        feature
        for key in sorted(grouped)
        for feature in sorted(
            grouped[key],
            key=lambda item: (_feature_priority(item, seed), item.feature_id),
        )[:maximum_clusters_per_site_class]
    ]
    return sorted(
        selected,
        key=lambda item: (item.chrom, item.start, item.end, item.feature_id),
    )


def gencode_annotation_intervals_for_windows(
    path: str | Path,
    records: Iterable[AnnotatedSequence],
) -> dict[str, tuple[Interval, ...]]:
    """Map conservative GENCODE element intervals into sampled sequence windows.

    Every exon boundary is treated as potentially splice-annotated; transcript-terminal
    boundaries therefore create conservative false exclusions rather than false
    ``unannotated`` controls. Start/stop codons and Kozak spans are mapped directly.
    """

    by_key: dict[tuple[str, str], tuple[list[int], list[AnnotatedSequence], int]] = {}
    grouped: dict[tuple[str, str], list[AnnotatedSequence]] = {}
    for record in records:
        grouped.setdefault((record.chrom, record.strand), []).append(record)
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda item: (item.genomic_start, item.example_id))
        by_key[key] = (
            [item.genomic_start for item in ordered],
            ordered,
            max(len(item.sequence) for item in ordered),
        )
    intervals: dict[str, set[Interval]] = defaultdict(set)
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in {
                "exon",
                "start_codon",
                "stop_codon",
            }:
                continue
            chrom, feature_type, strand = fields[0], fields[2], fields[6]
            if (chrom, strand) not in by_key:
                continue
            start, end = int(fields[3]) - 1, int(fields[4])
            genomic_intervals: list[tuple[int, int]]
            if feature_type == "exon":
                genomic_intervals = [(start - 2, start), (end, end + 2)]
            elif feature_type == "start_codon":
                genomic_intervals = [(start, end)]
                kozak = (start - 6, start + 4) if strand == "+" else (end - 4, end + 6)
                genomic_intervals.append(kozak)
            else:
                genomic_intervals = [(start, end)]
            for genomic_start, genomic_end in genomic_intervals:
                if genomic_start < 0 or genomic_end <= genomic_start:
                    continue
                starts, windows, maximum_length = by_key[(chrom, strand)]
                left = bisect_left(starts, genomic_start - maximum_length)
                right = bisect_right(starts, genomic_start)
                for record in windows[left:right]:
                    if not (
                        record.genomic_start <= genomic_start
                        and genomic_end <= record.genomic_end
                    ):
                        continue
                    if strand == "+":
                        local = Interval(
                            genomic_start - record.genomic_start,
                            genomic_end - record.genomic_start,
                        )
                    else:
                        local = Interval(
                            record.genomic_end - genomic_end,
                            record.genomic_end - genomic_start,
                        )
                    intervals[record.example_id].add(local)
    return {
        example_id: tuple(sorted(values)) for example_id, values in intervals.items()
    }


def _local_element(
    feature: GenomicFeature,
    sequence: str,
    window_start: int,
    window_end: int,
    strand: str,
    known_element_ids: set[str],
) -> BiologicalElement:
    if strand == "+":
        start = feature.start - window_start
        end = feature.end - window_start
    else:
        start = window_end - feature.end
        end = window_end - feature.start
    interval = Interval(start, end)
    paired = feature.attributes.get("paired_element_ids")
    if paired is None and feature.attributes.get("paired_element_id") is not None:
        paired = (feature.attributes["paired_element_id"],)
    elif isinstance(paired, str):
        paired = tuple(item for item in paired.split(",") if item)
    return BiologicalElement(
        element_id=feature.feature_id,
        site_class=feature.site_class,
        interval=interval,
        evidence=feature.evidence,
        sequence=sequence[start:end],
        paired_element_ids=tuple(
            element_id
            for element_id in (paired or ())
            if element_id in known_element_ids
        ),
        attributes=feature.attributes,
    )


def _default_context_regions(
    target: BiologicalElement,
    *,
    elements: tuple[BiologicalElement, ...],
    sequence_length: int,
    immediate_flank_nt: int,
) -> tuple[ContextRegion, ...]:
    regions: list[ContextRegion] = []
    definitions = (
        (
            "immediate_5p_flank",
            max(0, target.interval.start - immediate_flank_nt),
            target.interval.start,
        ),
        (
            "immediate_3p_flank",
            target.interval.end,
            min(sequence_length, target.interval.end + immediate_flank_nt),
        ),
        ("upstream_context", 0, target.interval.start),
        ("downstream_context", target.interval.end, sequence_length),
    )
    for region_type, start, end in definitions:
        if end <= start:
            continue
        regions.append(
            ContextRegion(
                region_id=f"{target.element_id}:{region_type}",
                region_type=region_type,
                interval=Interval(start, end),
                target_element_ids=(target.element_id,),
            )
        )
    by_id = {element.element_id: element for element in elements}
    for paired_id in target.paired_element_ids:
        paired = by_id[paired_id]
        regions.append(
            ContextRegion(
                region_id=f"{target.element_id}:paired:{paired_id}",
                region_type="paired_biological_element",
                interval=paired.interval,
                target_element_ids=(target.element_id,),
                attributes={"paired_element_id": paired_id},
            )
        )
    return tuple(regions)


def split_for_cluster(
    cluster: str,
    *,
    seed: int,
    development_percent: int,
    baseline_training_percent: int = 0,
) -> str:
    """Assign a cluster once across development, baseline, and confirmation."""

    _validate_split_percentages(development_percent, baseline_training_percent)
    digest = hashlib.sha256(f"{seed}|{cluster}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < development_percent:
        return "development"
    if bucket < development_percent + baseline_training_percent:
        return "baseline_training"
    return "confirmation"


def _split_for_cluster(
    cluster: str,
    *,
    seed: int,
    development_percent: int,
    baseline_training_percent: int = 0,
) -> str:
    return split_for_cluster(
        cluster,
        seed=seed,
        development_percent=development_percent,
        baseline_training_percent=baseline_training_percent,
    )


def _validate_split_percentages(
    development_percent: int, baseline_training_percent: int
) -> None:
    if not 0 < development_percent < 100:
        raise ValueError("development_percent must be between 0 and 100")
    if baseline_training_percent < 0:
        raise ValueError("baseline_training_percent cannot be negative")
    if development_percent + baseline_training_percent >= 100:
        raise ValueError(
            "development plus baseline-training percentages must be below 100"
        )


def _feature_priority(feature: GenomicFeature, seed: int) -> str:
    return hashlib.sha256(
        f"{seed}|feature|{feature.feature_id}|{feature.chrom}|"
        f"{feature.start}|{feature.end}".encode()
    ).hexdigest()


def _example_id(target: GenomicFeature, start: int, end: int) -> str:
    digest = hashlib.sha256(
        f"{target.feature_id}|{target.chrom}|{start}|{end}|{target.strand}".encode()
    ).hexdigest()[:16]
    return f"atlas:{digest}"


def _parse_gtf_attributes(text: str) -> dict[str, str]:
    return {
        key: value for key, value in re.findall(r'([A-Za-z0-9_]+) "([^"]*)";', text)
    }


def _derived_feature(
    feature_id: str,
    chrom: str,
    interval: tuple[int, int],
    strand: str,
    site_class: str,
    source_version: str,
    evidence_type: EvidenceType,
    cluster: str,
    attributes: Mapping[str, Any],
) -> GenomicFeature:
    return GenomicFeature(
        feature_id=feature_id,
        chrom=chrom,
        start=interval[0],
        end=interval[1],
        strand=strand,
        site_class=site_class,
        evidence=Evidence(
            source_id="GENCODE",
            source_version=source_version,
            evidence_type=evidence_type,
            record_id=feature_id,
        ),
        homology_cluster=cluster,
        attributes=attributes,
    )


def _open_text(path: str | Path) -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")
