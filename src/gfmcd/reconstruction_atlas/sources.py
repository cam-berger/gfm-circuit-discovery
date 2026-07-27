"""Pinned, outcome-blind source adapters for the multi-feature atlas."""

from __future__ import annotations

import csv
import gzip
import hashlib
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from .assembly import GenomicFeature, reverse_complement, split_for_cluster
from .schema import Evidence, EvidenceType

PRIMARY_CHROMS = frozenset([f"chr{index}" for index in range(1, 23)] + ["chrX", "chrY"])
POLYA_CANONICAL_HEXAMERS = (
    "AATAAA",
    "ATTAAA",
    "TATAAA",
    "AGTAAA",
    "AAGAAA",
    "AATATA",
    "AATACA",
    "CATAAA",
    "GATAAA",
    "AATGAA",
    "ACTAAA",
    "AATAGA",
)


def iter_epdnew_tss_cores(
    path: str | Path,
    *,
    source_version: str = "human-v006",
    core_width: int = 7,
) -> Iterator[GenomicFeature]:
    """Yield symmetric, TSS-centred promoter cores from EPDnew BED records."""

    if core_width < 1 or core_width % 2 == 0:
        raise ValueError("EPDnew core width must be a positive odd number")
    half = core_width // 2
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split()
            if len(fields) < 6:
                raise ValueError(f"{path}:{line_number}: expected BED6 or wider")
            chrom, start_text, end_text, promoter_id, _score, strand = fields[:6]
            start, end = int(start_text), int(end_text)
            if end - start != 60 or strand not in {"+", "-"}:
                continue
            # EPDnew human BED windows are -49..+10 in transcription direction.
            tss = start + 49 if strand == "+" else end - 50
            feature_id = f"EPDnew:{promoter_id}:tss-core"
            gene_symbol = promoter_id.rsplit("_", 1)[0]
            yield GenomicFeature(
                feature_id=feature_id,
                chrom=chrom,
                start=tss - half,
                end=tss + half + 1,
                strand=strand,
                site_class="promoter.epd_tss_core",
                evidence=_evidence(
                    "EPDnew",
                    source_version,
                    EvidenceType.EXPERIMENTAL,
                    promoter_id,
                ),
                homology_cluster=feature_id,
                attributes={
                    "source_record_id": promoter_id,
                    "gene_symbol": gene_symbol,
                    "source_interval_start": start,
                    "source_interval_end": end,
                    "coordinate_derivation": "EPDnew_-49_to_+10_window_centered_on_TSS",
                    "target_precision": f"{core_width}_nt_TSS_centered_core",
                },
            )


def iter_polyasite_elements(
    path: str | Path,
    *,
    source_version: str = "atlas-2.0-GRCh38.96",
    minimum_fraction_samples: float = 0.10,
    minimum_protocols: int = 2,
    allowed_annotations: frozenset[str] = frozenset({"TE"}),
) -> Iterator[GenomicFeature]:
    """Yield high-confidence PAS hexamers and representative cleavage nucleotides."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            signals = _polyasite_signals(str(row["repSite_signals"]))
            if (
                not signals
                or float(row["frac_samples"]) < minimum_fraction_samples
                or int(row["nr_prots"]) < minimum_protocols
                or str(row["annotation"]) not in allowed_annotations
                or str(row["gene_id"]) == "NA"
            ):
                continue
            strand = str(row["strand"])
            if strand not in {"+", "-"}:
                continue
            signal = min(
                (
                    item
                    for item in signals
                    if -60 <= item[1] <= -5 and item[0] in POLYA_CANONICAL_HEXAMERS
                ),
                key=lambda item: (
                    POLYA_CANONICAL_HEXAMERS.index(item[0]),
                    abs(item[1] + 20),
                    item[2],
                ),
                default=None,
            )
            if signal is None:
                continue
            chrom = _with_chr(str(row["chrom"]))
            record_id = str(row["name"])
            gene_id = str(row["gene_id"]).split(".", 1)[0]
            motif, relative_position, signal_position = signal
            signal_start = (
                signal_position - 1 if strand == "+" else signal_position - len(motif)
            )
            common = {
                "source_record_id": record_id,
                "gene_id": gene_id,
                "gene_name": str(row["gene_name"]),
                "source_interval_start": int(row["chromStart"]),
                "source_interval_end": int(row["chromEnd"]),
                "fraction_samples": float(row["frac_samples"]),
                "number_protocols": int(row["nr_prots"]),
                "polyasite_annotation": str(row["annotation"]),
            }
            signal_id = f"PolyASite:{record_id}:signal"
            cleavage_id = f"PolyASite:{record_id}:cleavage"
            yield GenomicFeature(
                feature_id=signal_id,
                chrom=chrom,
                start=signal_start,
                end=signal_start + len(motif),
                strand=strand,
                site_class="polya.signal",
                evidence=_evidence(
                    "PolyASite",
                    source_version,
                    EvidenceType.EXPERIMENTAL,
                    record_id,
                ),
                homology_cluster=gene_id,
                attributes={
                    **common,
                    "declared_motif": motif,
                    "relative_position": relative_position,
                    "paired_element_id": cleavage_id,
                    "target_precision": "reported_signal_hexamer",
                },
            )
            representative = int(row["rep"])
            yield GenomicFeature(
                feature_id=cleavage_id,
                chrom=chrom,
                start=representative - 1,
                end=representative,
                strand=strand,
                site_class="polya.cleavage_region",
                evidence=_evidence(
                    "PolyASite",
                    source_version,
                    EvidenceType.EXPERIMENTAL,
                    record_id,
                ),
                homology_cluster=gene_id,
                attributes={
                    **common,
                    "paired_element_id": signal_id,
                    "target_precision": "representative_cleavage_nucleotide",
                },
            )


def iter_encode_tf_peak_cores(
    path: str | Path,
    *,
    factors: Sequence[str] = ("CTCF", "REST", "CEBPB"),
    source_version: str = "UCSC-hg38-encRegTfbsClustered-2025-05",
    minimum_score: int = 500,
    minimum_source_count: int = 2,
    core_width: int = 7,
) -> Iterator[GenomicFeature]:
    """Yield fixed-width cores of multi-experiment ENCODE TF ChIP clusters."""

    allowed = frozenset(factors)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                raise ValueError(f"{path}:{line_number}: expected UCSC table columns")
            _bin, chrom, start_text, end_text, factor, score_text, count_text = fields[
                :7
            ]
            score, source_count = int(score_text), int(count_text)
            if (
                factor not in allowed
                or score < minimum_score
                or source_count < minimum_source_count
            ):
                continue
            start, end = int(start_text), int(end_text)
            core_start, core_end = _centered_interval(start, end, core_width)
            record_id = f"{factor}:{chrom}:{start}-{end}"
            yield GenomicFeature(
                feature_id=f"ENCODE-TF:{record_id}:core",
                chrom=chrom,
                start=core_start,
                end=core_end,
                strand="+",
                site_class=f"tfbs.{factor.lower()}_peak_core",
                evidence=_evidence(
                    "ENCODE-TF-ChIP-clusters",
                    source_version,
                    EvidenceType.EXPERIMENTAL,
                    record_id,
                ),
                homology_cluster=record_id,
                attributes={
                    "source_interval_start": start,
                    "source_interval_end": end,
                    "source_score": score,
                    "source_experiment_count": source_count,
                    "source_strand": ".",
                    "sequence_orientation": "genomic_forward",
                    "target_precision": f"{core_width}_nt_derived_peak_center",
                },
            )


def iter_encode_dels_cores(
    path: str | Path,
    *,
    source_version: str = "ENCODE-Registry-v4-GRCh38",
    core_width: int = 7,
) -> Iterator[GenomicFeature]:
    """Yield fixed-width cores from distal enhancer-like ENCODE cCREs."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                raise ValueError(f"{path}:{line_number}: expected ENCODE BED9+ class")
            chrom, start_text, end_text, accession = fields[:4]
            ccre_class = fields[9]
            if ccre_class != "dELS":
                continue
            start, end = int(start_text), int(end_text)
            core_start, core_end = _centered_interval(start, end, core_width)
            yield GenomicFeature(
                feature_id=f"ENCODE-cCRE:{accession}:core",
                chrom=chrom,
                start=core_start,
                end=core_end,
                strand="+",
                site_class="enhancer.encode_dels_core",
                evidence=_evidence(
                    "ENCODE-Registry-cCRE",
                    source_version,
                    EvidenceType.EXPERIMENTAL,
                    accession,
                ),
                homology_cluster=accession,
                attributes={
                    "source_interval_start": start,
                    "source_interval_end": end,
                    "ccre_class": ccre_class,
                    "source_strand": ".",
                    "sequence_orientation": "genomic_forward",
                    "target_precision": f"{core_width}_nt_derived_cCRE_center",
                },
            )


def iter_branchpoints(
    path: str | Path,
    *,
    source_version: str = "Mercer-2015-high-confidence-GRCh38-lifted",
) -> Iterator[GenomicFeature]:
    """Read exact, already lifted high-confidence branch nucleotides."""

    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                raise ValueError(f"{path}:{line_number}: expected BED6")
            chrom, start_text, end_text, record_id, _score, strand = fields[:6]
            start, end = int(start_text), int(end_text)
            if end - start != 1 or strand not in {"+", "-"}:
                continue
            yield _branchpoint_feature(
                (chrom, start_text, end_text, record_id, _score, strand),
                source_version=source_version,
            )


def derive_polypyrimidine_tracts(
    features: Iterable[GenomicFeature],
    fasta: Any,
    *,
    tract_width: int = 7,
    search_upstream_nt: int = 40,
    acceptor_buffer_nt: int = 3,
) -> list[GenomicFeature]:
    """Derive the nearest all-pyrimidine tract upstream of each splice acceptor."""

    output: list[GenomicFeature] = []
    for acceptor in features:
        if acceptor.site_class != "splice.acceptor":
            continue
        if acceptor.strand == "+":
            region_start = acceptor.start - search_upstream_nt
            region_end = acceptor.start - acceptor_buffer_nt
        else:
            region_start = acceptor.end + acceptor_buffer_nt
            region_end = acceptor.end + search_upstream_nt
        if region_start < 0 or region_end - region_start < tract_width:
            continue
        raw = str(fasta[acceptor.chrom][region_start:region_end]).upper()
        sequence = raw if acceptor.strand == "+" else reverse_complement(raw)
        starts = [
            index
            for index in range(len(sequence) - tract_width + 1)
            if set(sequence[index : index + tract_width]) <= {"C", "T"}
        ]
        if not starts:
            continue
        local_start = max(starts)
        if acceptor.strand == "+":
            start = region_start + local_start
            end = start + tract_width
        else:
            end = region_end - local_start
            start = end - tract_width
        feature_id = f"{acceptor.feature_id}:ppt"
        output.append(
            GenomicFeature(
                feature_id=feature_id,
                chrom=acceptor.chrom,
                start=start,
                end=end,
                strand=acceptor.strand,
                site_class="splice.polypyrimidine_tract",
                evidence=_evidence(
                    "GENCODE",
                    acceptor.evidence.source_version,
                    EvidenceType.DERIVED,
                    feature_id,
                ),
                homology_cluster=acceptor.homology_cluster,
                attributes={
                    **dict(acceptor.attributes),
                    "paired_element_id": acceptor.feature_id,
                    "derivation": (
                        f"nearest_all_C_or_T_{tract_width}mer_in_{search_upstream_nt}nt"
                        f"_acceptor_upstream_window_with_{acceptor_buffer_nt}nt_buffer"
                    ),
                    "target_precision": "derived_contiguous_polypyrimidine_tract",
                },
            )
        )
    return output


def sample_locus_blocked_features(
    features: Iterable[GenomicFeature],
    *,
    seed: int,
    maximum_clusters_per_site_class: int,
    development_percent: int,
    baseline_training_percent: int,
    window_flank_nt: int,
    locus_block_nt: int = 1_000_000,
    primary_chromosomes: frozenset[str] = PRIMARY_CHROMS,
) -> tuple[list[GenomicFeature], dict[str, Any]]:
    """Split all sources by shared genomic blocks, then sample one site per class/block."""

    per_cluster: dict[tuple[str, str, str], GenomicFeature] = {}
    counts: Counter[str] = Counter()
    for feature in features:
        counts["input_features"] += 1
        if feature.chrom not in primary_chromosomes:
            counts["excluded_non_primary_chromosome"] += 1
            continue
        midpoint = (feature.start + feature.end - 1) // 2
        block_index = midpoint // locus_block_nt
        block_start = block_index * locus_block_nt
        block_end = block_start + locus_block_nt
        if (
            feature.start - window_flank_nt < block_start
            or feature.end + window_flank_nt > block_end
        ):
            counts["excluded_cross_block_window"] += 1
            continue
        cluster = f"locus{locus_block_nt}:{feature.chrom}:{block_index}"
        blocked = replace(feature, homology_cluster=cluster)
        split = split_for_cluster(
            cluster,
            seed=seed,
            development_percent=development_percent,
            baseline_training_percent=baseline_training_percent,
        )
        key = (split, blocked.site_class, cluster)
        previous = per_cluster.get(key)
        if previous is None or _priority(blocked, seed) < _priority(previous, seed):
            per_cluster[key] = blocked
    grouped: dict[tuple[str, str], list[GenomicFeature]] = {}
    for (split, site_class, _cluster), feature in per_cluster.items():
        grouped.setdefault((split, site_class), []).append(feature)
    selected = [
        feature
        for key in sorted(grouped)
        for feature in sorted(
            grouped[key],
            key=lambda item: (_priority(item, seed), item.feature_id),
        )[:maximum_clusters_per_site_class]
    ]
    selected = sorted(
        selected,
        key=lambda item: (item.chrom, item.start, item.end, item.feature_id),
    )
    audit = {
        **dict(sorted(counts.items())),
        "eligible_locus_class_combinations": len(per_cluster),
        "selected_features": len(selected),
        "selected_by_split_and_class": {
            f"{split}|{site_class}": sum(
                split_for_cluster(
                    item.homology_cluster,
                    seed=seed,
                    development_percent=development_percent,
                    baseline_training_percent=baseline_training_percent,
                )
                == split
                and item.site_class == site_class
                for item in selected
            )
            for split, site_class in sorted(grouped)
        },
        "locus_block_nt": locus_block_nt,
        "window_flank_nt": window_flank_nt,
    }
    return selected, audit


def annotation_intervals_from_bed_for_windows(
    path: str | Path,
    records: Sequence[Any],
) -> dict[str, tuple[Any, ...]]:
    """Map a genome-coordinate BED into every overlapping atlas window."""

    from bisect import bisect_left, bisect_right
    from collections import defaultdict

    from .schema import Interval

    grouped: dict[str, list[Any]] = {}
    for record in records:
        grouped.setdefault(record.chrom, []).append(record)
    indexed = {}
    for chrom, values in grouped.items():
        ordered = sorted(values, key=lambda item: (item.genomic_start, item.example_id))
        indexed[chrom] = (
            [item.genomic_start for item in ordered],
            ordered,
            max(len(item.sequence) for item in ordered),
        )
    output: dict[str, set[Any]] = defaultdict(set)
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, start_text, end_text = fields[:3]
            if chrom not in indexed:
                continue
            start, end = int(start_text), int(end_text)
            if end <= start:
                continue
            starts, windows, maximum_length = indexed[chrom]
            left = bisect_left(starts, start - maximum_length)
            right = bisect_right(starts, start)
            for record in windows[left:right]:
                if not (record.genomic_start <= start and end <= record.genomic_end):
                    continue
                local = (
                    Interval(start - record.genomic_start, end - record.genomic_start)
                    if record.strand == "+"
                    else Interval(record.genomic_end - end, record.genomic_end - start)
                )
                output[record.example_id].add(local)
    return {
        example_id: tuple(sorted(intervals)) for example_id, intervals in output.items()
    }


def liftover_branchpoints_roundtrip(
    input_bed: str | Path,
    *,
    forward_chain: str | Path,
    reverse_chain: str | Path,
    liftover_binary: str | Path,
    fasta: Any,
) -> tuple[list[GenomicFeature], dict[str, Any]]:
    """Lift exact hg19 branch nucleotides and retain unique exact round trips."""

    with tempfile.TemporaryDirectory(prefix="dnabert2-atlas-liftover-") as directory:
        root = Path(directory)
        source_bed = root / "source.bed"
        with (
            _open_text(input_bed) as source,
            source_bed.open("w", encoding="utf-8") as destination,
        ):
            shutil.copyfileobj(source, destination)
        forward = root / "forward.bed"
        forward_unmapped = root / "forward.unmapped"
        reverse = root / "reverse.bed"
        reverse_unmapped = root / "reverse.unmapped"
        subprocess.run(
            [
                str(liftover_binary),
                str(source_bed),
                str(forward_chain),
                str(forward),
                str(forward_unmapped),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                str(liftover_binary),
                str(forward),
                str(reverse_chain),
                str(reverse),
                str(reverse_unmapped),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        original = _unique_bed_records(source_bed)
        lifted = _unique_bed_records(forward)
        roundtrip = _unique_bed_records(reverse)
    counts: Counter[str] = Counter()
    output: list[GenomicFeature] = []
    for record_id, original_rows in original.items():
        counts["input_records"] += 1
        if len(original_rows) != 1:
            counts["excluded_nonunique_input"] += 1
            continue
        forward_rows = lifted.get(record_id, ())
        reverse_rows = roundtrip.get(record_id, ())
        if len(forward_rows) != 1 or len(reverse_rows) != 1:
            counts["excluded_nonunique_or_unmapped"] += 1
            continue
        source = original_rows[0]
        mapped = forward_rows[0]
        returned = reverse_rows[0]
        if returned[:3] != source[:3] or returned[5] != source[5]:
            counts["excluded_inexact_roundtrip"] += 1
            continue
        chrom, start, end, _name, _score, strand = mapped
        if chrom not in PRIMARY_CHROMS or int(end) - int(start) != 1:
            counts["excluded_non_primary_or_non_unit"] += 1
            continue
        base = str(fasta[chrom][int(start) : int(end)]).upper()
        oriented = base if strand == "+" else reverse_complement(base)
        declared_base = record_id.rsplit("_", 1)[-1].upper()
        if declared_base not in {"A", "C", "G", "T"} or oriented != declared_base:
            counts["excluded_reference_base_mismatch"] += 1
            continue
        output.append(
            replace(
                _branchpoint_feature(
                    mapped,
                    source_version="Mercer-2015-high-confidence-GRCh38-lifted",
                ),
                attributes={
                    "target_precision": ("experimentally_supported_branch_nucleotide"),
                    "original_assembly": "hg19",
                    "liftover_policy": (
                        "unique_forward_exact_reverse_roundtrip_and_base_match"
                    ),
                    "branchpoint_nucleotide": declared_base,
                },
            )
        )
        counts["retained_exact_roundtrip_base_match"] += 1
    return output, dict(sorted(counts.items()))


def _branchpoint_feature(
    fields: Sequence[str],
    *,
    source_version: str,
) -> GenomicFeature:
    chrom, start_text, end_text, record_id, _score, strand = fields[:6]
    return GenomicFeature(
        feature_id=f"Mercer:{record_id}",
        chrom=chrom,
        start=int(start_text),
        end=int(end_text),
        strand=strand,
        site_class="splice.branchpoint",
        evidence=_evidence(
            "Mercer-lariat-seq",
            source_version,
            EvidenceType.EXPERIMENTAL,
            record_id,
        ),
        homology_cluster=record_id,
        attributes={
            "target_precision": "experimentally_supported_branch_nucleotide",
            "original_assembly": "hg19",
            "liftover_policy": "unique_forward_exact_reverse_roundtrip",
        },
    )


def _unique_bed_records(path: str | Path) -> dict[str, list[tuple[str, ...]]]:
    output: dict[str, list[tuple[str, ...]]] = {}
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = tuple(line.rstrip("\n").split("\t"))
            if len(fields) < 6:
                continue
            output.setdefault(fields[3], []).append(fields)
    return output


def _polyasite_signals(text: str) -> list[tuple[str, int, int]]:
    output = []
    if text == "NA":
        return output
    for item in text.split(";"):
        fields = item.split("@")
        if len(fields) != 3:
            continue
        motif, relative, genomic_position = fields
        output.append((motif.upper(), int(relative), int(genomic_position)))
    return output


def _centered_interval(start: int, end: int, width: int) -> tuple[int, int]:
    if width < 1 or width % 2 == 0 or end <= start:
        raise ValueError("centered interval requires an odd width and valid source")
    center = (start + end - 1) // 2
    half = width // 2
    return center - half, center + half + 1


def _with_chr(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def _evidence(
    source_id: str,
    source_version: str,
    evidence_type: EvidenceType,
    record_id: str,
) -> Evidence:
    return Evidence(
        source_id=source_id,
        source_version=source_version,
        evidence_type=evidence_type,
        record_id=record_id,
    )


def _priority(feature: GenomicFeature, seed: int) -> str:
    return hashlib.sha256(
        f"{seed}|feature|{feature.feature_id}|{feature.chrom}|"
        f"{feature.start}|{feature.end}".encode()
    ).hexdigest()


def _open_text(path: str | Path) -> TextIO:
    value = Path(path)
    if value.suffix == ".gz":
        return gzip.open(value, "rt", encoding="utf-8")
    return value.open(encoding="utf-8")
