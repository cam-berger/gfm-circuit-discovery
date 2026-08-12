"""Model-free biology inputs for positional-boundary-coherence Phase I.

This module is deliberately limited to low-level, outcome-blind operations:
indexed FASTA access, GTF feature parsing, transcript-oriented splice-edge
enumeration, and the frozen MaxEntScan equations.  It does not select cohort
blocks and it never imports a model adapter.

Coordinates are 0-based and half-open.  Sequence coordinates increase in the
transcript direction on both strands.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import math
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Protocol, Sequence, TextIO

from gfmcd.splicing.dataset import reverse_complement
from gfmcd.splicing.gtf import attr_first, parse_attributes
from gfmcd.splicing.phase import codon_slot, intron_phase


BASES = frozenset("ACGT")
CANDIDATE_SCHEMA = "gfmcd.positional_boundary_coherence.candidates.v1"
CANDIDATE_ALGORITHM = "pbc-real-census-v1"
DEFAULT_ALLOWED_CHROMS = frozenset(
    {*(f"chr{i}" for i in range(1, 8)), *(f"chr{i}" for i in range(10, 23)), "chrX"}
)
FORBIDDEN_CHROMS = frozenset({"chr8", "chr9", "chrY", "chrM", "MT"})


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "size": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
    }


def _open_text(path: str | Path) -> TextIO:
    source = Path(path)
    if source.suffix == ".gz":
        return gzip.open(source, "rt", encoding="utf-8")
    return source.open("rt", encoding="utf-8")


@dataclass(frozen=True)
class FaiRecord:
    name: str
    length: int
    offset: int
    line_bases: int
    line_width: int


class IndexedFasta:
    """Minimal read-only FASTA accessor whose exact ``.fai`` is an input.

    Requiring the caller-provided index avoids an implicit pyfaidx index build
    and makes the index itself part of the immutable biology binding.
    """

    def __init__(self, fasta: str | Path, fai: str | Path):
        self.fasta_path = Path(fasta).expanduser().resolve(strict=True)
        self.fai_path = Path(fai).expanduser().resolve(strict=True)
        records: dict[str, FaiRecord] = {}
        with self.fai_path.open("rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                columns = line.rstrip("\n").split("\t")
                if len(columns) < 5:
                    raise ValueError(f"{self.fai_path}:{line_no}: invalid FAI row")
                record = FaiRecord(
                    columns[0], *(int(value) for value in columns[1:5])
                )
                if (
                    record.length <= 0
                    or record.offset < 0
                    or record.line_bases <= 0
                    or record.line_width < record.line_bases
                    or record.name in records
                ):
                    raise ValueError(f"{self.fai_path}:{line_no}: invalid FAI fields")
                records[record.name] = record
        if not records:
            raise ValueError("FAI contains no sequences")
        self.records = records

    def _resolve(self, chrom: str) -> FaiRecord:
        if chrom in self.records:
            return self.records[chrom]
        alternate = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        if alternate in self.records:
            return self.records[alternate]
        raise KeyError(f"chromosome {chrom!r} is absent from the pinned FAI")

    def length(self, chrom: str) -> int:
        return self._resolve(chrom).length

    def fetch(self, chrom: str, start: int, end: int) -> str:
        record = self._resolve(chrom)
        if start < 0 or end < start or end > record.length:
            raise ValueError(
                f"FASTA interval {chrom}:{start}-{end} lies outside 0-{record.length}"
            )
        if start == end:
            return ""
        chunks: list[bytes] = []
        position = start
        with self.fasta_path.open("rb") as handle:
            while position < end:
                line_index, within = divmod(position, record.line_bases)
                take = min(end - position, record.line_bases - within)
                byte_offset = record.offset + line_index * record.line_width + within
                handle.seek(byte_offset)
                chunk = handle.read(take)
                if len(chunk) != take:
                    raise ValueError("FASTA ended before the FAI-declared interval")
                chunks.append(chunk)
                position += take
        sequence = b"".join(chunks).decode("ascii").upper().replace("U", "T")
        if len(sequence) != end - start:
            raise AssertionError("indexed FASTA extraction length drift")
        return sequence


class GenomeLike(Protocol):
    def fetch(self, chrom: str, start: int, end: int) -> str: ...

    def length(self, chrom: str) -> int: ...


def oriented_interval(boundary: int, rel_start: int, rel_end: int, strand: str) -> tuple[int, int]:
    """Map a transcript-oriented boundary interval to genomic coordinates."""

    if rel_end <= rel_start:
        raise ValueError("relative interval must be non-empty")
    if strand == "+":
        return boundary + rel_start, boundary + rel_end
    if strand == "-":
        return boundary - rel_end, boundary - rel_start
    raise ValueError("strand must be '+' or '-'")


def fetch_oriented(
    genome: GenomeLike,
    chrom: str,
    boundary: int,
    rel_start: int,
    rel_end: int,
    strand: str,
) -> tuple[str, tuple[int, int]]:
    start, end = oriented_interval(boundary, rel_start, rel_end, strand)
    if start < 0 or end > genome.length(chrom):
        raise ValueError("insufficient FASTA flank")
    sequence = genome.fetch(chrom, start, end)
    if strand == "-":
        sequence = reverse_complement(sequence)
    return sequence.upper(), (start, end)


def boundary_to_genomic(window_start: int, window_end: int, position: int, strand: str) -> int:
    """Map an oriented between-base boundary back to a genomic coordinate."""

    if strand == "+":
        return window_start + position
    if strand == "-":
        return window_end - position
    raise ValueError("strand must be '+' or '-'")


@dataclass(frozen=True)
class GtfFeature:
    chrom: str
    feature: str
    start: int
    end: int
    strand: str
    frame: int | None
    gene_id: str
    transcript_id: str
    gene_type: str
    transcript_type: str

    @property
    def length(self) -> int:
        return self.end - self.start


def iter_gtf_features(
    path: str | Path,
    *,
    allowed_chromosomes: set[str] | frozenset[str] | None = None,
) -> Iterator[GtfFeature]:
    """Yield the exon and framed-CDS rows needed by the census."""

    with _open_text(path) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                raise ValueError(f"{path}:{line_no}: expected nine GTF columns")
            chrom, _source, feature, start, end, _score, strand, frame_text, attrs_text = columns
            if feature not in {"exon", "CDS"}:
                continue
            if allowed_chromosomes is not None and chrom not in allowed_chromosomes:
                continue
            if strand not in {"+", "-"}:
                continue
            attrs = parse_attributes(attrs_text)
            transcript_type = attr_first(attrs, "transcript_type") or attr_first(
                attrs, "transcript_biotype"
            )
            gene_type = attr_first(attrs, "gene_type") or attr_first(attrs, "gene_biotype")
            if transcript_type != "protein_coding" and gene_type != "protein_coding":
                continue
            frame = int(frame_text) if feature == "CDS" and frame_text in {"0", "1", "2"} else None
            if feature == "CDS" and frame is None:
                continue
            gene_id = attr_first(attrs, "gene_id")
            transcript_id = attr_first(attrs, "transcript_id")
            if not gene_id or not transcript_id:
                continue
            yield GtfFeature(
                chrom=chrom,
                feature=feature,
                start=int(start) - 1,
                end=int(end),
                strand=strand,
                frame=frame,
                gene_id=gene_id,
                transcript_id=transcript_id,
                gene_type=gene_type,
                transcript_type=transcript_type,
            )


@dataclass(frozen=True)
class EdgeSupport:
    chrom: str
    donor: int
    acceptor: int
    strand: str
    gene_id: str
    transcript_id: str
    upstream_exon_start: int
    upstream_exon_end: int
    downstream_exon_start: int
    downstream_exon_end: int
    upstream_cds_start: int
    upstream_cds_end: int
    downstream_cds_start: int
    downstream_cds_end: int
    upstream_frame: int
    downstream_frame: int
    donor_phase: int
    acceptor_phase: int

    @property
    def phase(self) -> int:
        if self.donor_phase != self.acceptor_phase:
            raise ValueError("edge support has discordant endpoint phases")
        return self.donor_phase

    @property
    def intron_length(self) -> int:
        return abs(self.acceptor - self.donor)


def _edge_supports(features: Iterable[GtfFeature]) -> list[EdgeSupport]:
    """Derive independently framed adjacent-exon edges transcript by transcript."""

    by_transcript: dict[str, list[GtfFeature]] = defaultdict(list)
    for feature in features:
        by_transcript[feature.transcript_id].append(feature)
    supports: list[EdgeSupport] = []
    for transcript_id in sorted(by_transcript):
        rows = by_transcript[transcript_id]
        exons = sorted(
            (row for row in rows if row.feature == "exon"),
            key=lambda row: (row.start, row.end),
            reverse=rows[0].strand == "-",
        )
        cds = [row for row in rows if row.feature == "CDS"]
        if len(exons) < 2:
            continue
        for upstream_exon, downstream_exon in zip(exons[:-1], exons[1:]):
            if upstream_exon.strand == "+":
                donor, acceptor = upstream_exon.end, downstream_exon.start
                upstream_touch = [row for row in cds if row.end == donor and row.start >= upstream_exon.start]
                downstream_touch = [row for row in cds if row.start == acceptor and row.end <= downstream_exon.end]
            else:
                donor, acceptor = upstream_exon.start, downstream_exon.end
                upstream_touch = [row for row in cds if row.start == donor and row.end <= upstream_exon.end]
                downstream_touch = [row for row in cds if row.end == acceptor and row.start >= downstream_exon.start]
            if len(upstream_touch) != 1 or len(downstream_touch) != 1:
                continue
            up, down = upstream_touch[0], downstream_touch[0]
            donor_phase = intron_phase(up.start, up.end, int(up.frame))
            acceptor_phase = (3 - int(down.frame)) % 3
            supports.append(
                EdgeSupport(
                    chrom=upstream_exon.chrom,
                    donor=donor,
                    acceptor=acceptor,
                    strand=upstream_exon.strand,
                    gene_id=upstream_exon.gene_id,
                    transcript_id=transcript_id,
                    upstream_exon_start=upstream_exon.start,
                    upstream_exon_end=upstream_exon.end,
                    downstream_exon_start=downstream_exon.start,
                    downstream_exon_end=downstream_exon.end,
                    upstream_cds_start=up.start,
                    upstream_cds_end=up.end,
                    downstream_cds_start=down.start,
                    downstream_cds_end=down.end,
                    upstream_frame=int(up.frame),
                    downstream_frame=int(down.frame),
                    donor_phase=donor_phase,
                    acceptor_phase=acceptor_phase,
                )
            )
    return supports


@dataclass(frozen=True)
class EdgeEvidence:
    key: tuple[str, int, int, str, str]
    representative: EdgeSupport
    phase: int
    basic_support_ids: tuple[str, ...]
    comprehensive_support_ids: tuple[str, ...]
    comprehensive_features: tuple[GtfFeature, ...]
    checks: Mapping[str, bool]


def _overlaps(feature: GtfFeature, start: int, end: int) -> bool:
    return feature.start < end and feature.end > start


class _FeatureIntervalIndex:
    """Report exact interval overlaps while retaining source-file order.

    The census asks the same local-overlap question once per candidate edge.
    Scanning every chromosome feature for every edge is quadratic on the full
    GENCODE annotation.  This index sorts only lookup metadata; returned rows
    are restored to their original order so downstream exclusion records and
    candidate hashes are unchanged.
    """

    def __init__(self, features: Iterable[GtfFeature]):
        self._features = tuple(features)
        records = sorted(
            (feature.start, feature.end, index)
            for index, feature in enumerate(self._features)
        )
        self._records = tuple(records)
        self._starts = tuple(record[0] for record in records)
        size = 1
        while size < len(records):
            size *= 2
        self._tree_size = size
        max_ends = [-1] * (2 * size)
        for offset, (_start, end, _index) in enumerate(records):
            max_ends[size + offset] = end
        for node in range(size - 1, 0, -1):
            max_ends[node] = max(max_ends[2 * node], max_ends[2 * node + 1])
        self._max_ends = tuple(max_ends)

    def _collect(
        self,
        *,
        node: int,
        left: int,
        right: int,
        upper_record: int,
        query_start: int,
        found: set[int],
    ) -> None:
        if left >= upper_record or self._max_ends[node] <= query_start:
            return
        if right - left == 1:
            found.add(self._records[left][2])
            return
        middle = (left + right) // 2
        self._collect(
            node=2 * node,
            left=left,
            right=middle,
            upper_record=upper_record,
            query_start=query_start,
            found=found,
        )
        if middle < upper_record:
            self._collect(
                node=2 * node + 1,
                left=middle,
                right=right,
                upper_record=upper_record,
                query_start=query_start,
                found=found,
            )

    def overlaps(self, intervals: Iterable[tuple[int, int]]) -> tuple[GtfFeature, ...]:
        found: set[int] = set()
        for start, end in intervals:
            if end <= start:
                raise ValueError("query interval must be non-empty")
            upper_record = bisect_left(self._starts, end)
            if upper_record:
                self._collect(
                    node=1,
                    left=0,
                    right=self._tree_size,
                    upper_record=upper_record,
                    query_start=start,
                    found=found,
                )
        return tuple(self._features[index] for index in sorted(found))


def _local_intervals(edge: EdgeSupport) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        oriented_interval(edge.donor, -90, 36, edge.strand),
        oriented_interval(edge.acceptor, -50, 90, edge.strand),
    )


def enumerate_edge_evidence(
    basic_gtf: str | Path,
    comprehensive_gtf: str | Path,
    *,
    allowed_chromosomes: Iterable[str] = DEFAULT_ALLOWED_CHROMS,
) -> tuple[list[EdgeEvidence], list[dict[str, object]]]:
    """Enumerate annotation-safe coordinate-identical coding introns.

    Basic annotation defines eligibility.  Comprehensive annotation supplies
    an independent exact-edge phase check and a local overlap/register safety
    check across both measured source windows.
    """

    allowed = frozenset(allowed_chromosomes)
    basic_features = list(iter_gtf_features(basic_gtf, allowed_chromosomes=allowed))
    comprehensive_features = list(
        iter_gtf_features(comprehensive_gtf, allowed_chromosomes=allowed)
    )
    basic_supports = _edge_supports(basic_features)
    comprehensive_supports = _edge_supports(comprehensive_features)
    by_basic: dict[tuple[str, int, int, str, str], list[EdgeSupport]] = defaultdict(list)
    by_comp: dict[tuple[str, int, int, str, str], list[EdgeSupport]] = defaultdict(list)
    for support in basic_supports:
        root = support.gene_id.split(".", 1)[0]
        by_basic[(support.chrom, support.donor, support.acceptor, support.strand, root)].append(support)
    for support in comprehensive_supports:
        root = support.gene_id.split(".", 1)[0]
        by_comp[(support.chrom, support.donor, support.acceptor, support.strand, root)].append(support)

    # A coordinate-identical edge assigned to multiple normalized genes is not
    # a safe independent source (including pseudoautosomal-style duplicates).
    coordinate_genes: dict[tuple[str, int, int, str], set[str]] = defaultdict(set)
    for chrom, donor, acceptor, strand, root in by_basic:
        coordinate_genes[(chrom, donor, acceptor, strand)].add(root)

    comp_by_chrom: dict[str, list[GtfFeature]] = defaultdict(list)
    for feature in comprehensive_features:
        comp_by_chrom[feature.chrom].append(feature)
    comp_indexes = {
        chrom: _FeatureIntervalIndex(features)
        for chrom, features in comp_by_chrom.items()
    }

    # The supports and indexes now own every object needed below.  Releasing
    # the multi-million-row parser lists keeps the production census bounded.
    del basic_features, comprehensive_features, basic_supports, comprehensive_supports
    del comp_by_chrom

    retained: list[EdgeEvidence] = []
    exclusions: list[dict[str, object]] = []
    for key in sorted(by_basic):
        chrom, donor, acceptor, strand, root = key
        supports = by_basic[key]
        comp = by_comp.get(key, [])
        reasons: list[str] = []
        endpoint_pairs = {(row.donor_phase, row.acceptor_phase) for row in supports}
        basic_phases = {
            row.donor_phase for row in supports if row.donor_phase == row.acceptor_phase
        }
        comp_pairs = {(row.donor_phase, row.acceptor_phase) for row in comp}
        comp_phases = {row.donor_phase for row in comp if row.donor_phase == row.acceptor_phase}
        if any(left != right for left, right in endpoint_pairs):
            reasons.append("basic_endpoint_phase_disagreement")
        if len(basic_phases) != 1 or len(endpoint_pairs) != 1:
            reasons.append("basic_coordinate_transcript_phase_conflict")
        if not comp:
            reasons.append("comprehensive_exact_edge_absent")
        if any(left != right for left, right in comp_pairs):
            reasons.append("comprehensive_endpoint_phase_disagreement")
        if len(comp_phases) != 1 or len(comp_pairs) != 1:
            reasons.append("comprehensive_coordinate_transcript_phase_conflict")
        if basic_phases and comp_phases and basic_phases != comp_phases:
            reasons.append("basic_comprehensive_phase_conflict")
        if len(coordinate_genes[(chrom, donor, acceptor, strand)]) != 1:
            reasons.append("duplicate_or_pseudoautosomal_locus")

        representative = sorted(supports, key=lambda row: row.transcript_id)[0]
        intervals = _local_intervals(representative)
        index = comp_indexes.get(chrom)
        nearby = index.overlaps(intervals) if index is not None else ()
        if any(feature.strand != strand for feature in nearby):
            reasons.append("comprehensive_local_strand_conflict")
        if any(feature.gene_id.split(".", 1)[0] != root for feature in nearby):
            reasons.append("comprehensive_local_gene_conflict")

        # All same-gene/same-strand CDS annotations that cover a base must give
        # one codon slot.  This is stricter than exact-edge phase agreement and
        # catches local alternative-frame coding features.
        coding_conflict = False
        same_cds = [
            feature
            for feature in nearby
            if feature.feature == "CDS"
            and feature.strand == strand
            and feature.gene_id.split(".", 1)[0] == root
        ]
        for start, end in intervals:
            for position in range(start, end):
                slots = {
                    codon_slot(position, feature.start, feature.end, strand, int(feature.frame))
                    for feature in same_cds
                    if feature.start <= position < feature.end
                }
                if len(slots) > 1:
                    coding_conflict = True
                    break
            if coding_conflict:
                break
        if coding_conflict:
            reasons.append("comprehensive_local_coding_register_conflict")

        edge_label = f"{chrom}:{donor}>{acceptor}:{strand}:{root}"
        if reasons:
            exclusions.append(
                {
                    "edge_id": edge_label,
                    "gene_id": root,
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        phase = next(iter(basic_phases))
        checks = {
            "upstream_downstream_phase_equal": True,
            "basic_transcript_coordinate_agreement": True,
            "comprehensive_no_coding_conflict": True,
            "comprehensive_no_strand_conflict": True,
            "coding_offset_consistent_with_gtf_frame": True,
            "comprehensive_exact_edge_phase_agreement": True,
            "no_duplicate_or_pseudoautosomal_locus": True,
        }
        retained.append(
            EdgeEvidence(
                key=key,
                representative=representative,
                phase=int(phase),
                basic_support_ids=tuple(sorted(row.transcript_id for row in supports)),
                comprehensive_support_ids=tuple(sorted(row.transcript_id for row in comp)),
                comprehensive_features=nearby,
                checks=checks,
            )
        )
    return retained, sorted(exclusions, key=lambda row: str(row["edge_id"]))


class MaxEntTables:
    """Frozen MaxEntScan-compatible scorer using explicitly supplied tables."""

    BGD5 = {"A": 0.27, "C": 0.23, "G": 0.23, "T": 0.27}
    CONS1_5 = {"A": 0.004, "C": 0.0032, "G": 0.9896, "T": 0.0032}
    CONS2_5 = {"A": 0.0034, "C": 0.0039, "G": 0.0042, "T": 0.9884}
    BGD3 = {"A": 0.27, "C": 0.23, "G": 0.23, "T": 0.27}
    CONS1_3 = {"A": 0.9903, "C": 0.0032, "G": 0.0034, "T": 0.0030}
    CONS2_3 = {"A": 0.0027, "C": 0.0037, "G": 0.9905, "T": 0.0030}

    def __init__(self, donor_table: str | Path, acceptor_table: str | Path):
        self.donor_path = Path(donor_table).expanduser().resolve(strict=True)
        self.acceptor_path = Path(acceptor_table).expanduser().resolve(strict=True)
        donor: dict[str, float] = {}
        with self.donor_path.open("rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                key, value = line.split()
                if len(key) != 7 or set(key) - BASES:
                    raise ValueError(f"{self.donor_path}:{line_no}: invalid donor key")
                donor[key] = float(value)
        acceptor: dict[int, dict[int, float]] = defaultdict(dict)
        with self.acceptor_path.open("rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                group, key, value = line.split()
                acceptor[int(group)][int(key)] = float(value)
        if not donor or set(acceptor) != set(range(9)):
            raise ValueError("MaxEnt tables are incomplete")
        self.donor = donor
        self.acceptor = dict(acceptor)

    @staticmethod
    def _hashseq(sequence: str) -> int:
        digits = {"A": 0, "C": 1, "G": 2, "T": 3}
        return sum(digits[base] * 4 ** (len(sequence) - index - 1) for index, base in enumerate(sequence))

    def score5(self, sequence: str) -> float:
        sequence = sequence.upper()
        if len(sequence) != 9 or set(sequence) - BASES or sequence[3:5] != "GT":
            raise ValueError("donor MaxEnt requires a canonical-GT 9-mer")
        key = sequence[3:5]
        consensus = self.CONS1_5[key[0]] * self.CONS2_5[key[1]] / (
            self.BGD5[key[0]] * self.BGD5[key[1]]
        )
        return float(math.log2(consensus * self.donor[sequence[:3] + sequence[5:]]))

    def score3(self, sequence: str) -> float:
        sequence = sequence.upper()
        if len(sequence) != 23 or set(sequence) - BASES or sequence[18:20] != "AG":
            raise ValueError("acceptor MaxEnt requires a canonical-AG 23-mer")
        key = sequence[18:20]
        consensus = self.CONS1_3[key[0]] * self.CONS2_3[key[1]] / (
            self.BGD3[key[0]] * self.BGD3[key[1]]
        )
        rest = sequence[:18] + sequence[20:]
        pieces = (
            (0, rest[:7], 1),
            (1, rest[7:14], 1),
            (2, rest[14:], 1),
            (3, rest[4:11], 1),
            (4, rest[11:18], 1),
            (5, rest[4:7], -1),
            (6, rest[7:11], -1),
            (7, rest[11:14], -1),
            (8, rest[14:18], -1),
        )
        score = consensus
        for group, piece, exponent in pieces:
            value = self.acceptor[group][self._hashseq(piece)]
            score = score * value if exponent > 0 else score / value
        return float(math.log2(score))

    def pin(self) -> dict[str, object]:
        return {
            "implementation": file_record(Path(__file__).resolve()),
            "donor_table": file_record(self.donor_path),
            "acceptor_table": file_record(self.acceptor_path),
            "equations": "Yeo_Burge_MaxEntScan_9mer_23mer_local_v1",
        }


def annotation_payload(edge: EdgeEvidence) -> dict[str, object]:
    row = edge.representative
    return {
        "phase_convention": "intron_phase",
        "raw_gtf_frame": f"upstream={row.upstream_frame};downstream={row.downstream_frame}",
        "upstream_gtf_frame": int(row.upstream_frame),
        "downstream_gtf_frame": int(row.downstream_frame),
        "upstream_cds_end": int(row.donor),
        "downstream_cds_start": int(row.acceptor),
        "upstream_cds_end_phase": int(row.donor_phase),
        "downstream_cds_start_phase": int(row.acceptor_phase),
        "basic_phase": int(edge.phase),
        "comprehensive_phase": int(edge.phase),
        "basic_support_ids": list(edge.basic_support_ids),
        "comprehensive_support_ids": list(edge.comprehensive_support_ids),
        "checks": dict(sorted(edge.checks.items())),
    }


def native_window(
    edge: EdgeEvidence,
    genome: GenomeLike,
) -> tuple[str, int, int, tuple[int, int]]:
    """Return an authentic 1,022-nt source window with acceptor at 512."""

    row = edge.representative
    sequence, interval = fetch_oriented(
        genome, row.chrom, row.acceptor, -512, 510, row.strand
    )
    donor_pos = 512 - row.intron_length
    acceptor_pos = 512
    if not 90 <= donor_pos < acceptor_pos - 2:
        raise ValueError("native donor lacks the frozen health-window support")
    if sequence[donor_pos : donor_pos + 2] != "GT" or sequence[510:512] != "AG":
        raise ValueError("native source motifs do not reconstruct")
    return sequence, donor_pos, acceptor_pos, interval


def module_record(
    edge: EdgeEvidence,
    genome: GenomeLike,
    scorer: MaxEntTables,
    *,
    role: str,
    include_native_health_window: bool = True,
) -> dict[str, object]:
    """Materialize one authentic donor or acceptor source module."""

    if role not in {"donor", "acceptor"}:
        raise ValueError("module role must be donor or acceptor")
    row = edge.representative
    boundary = row.donor if role == "donor" else row.acceptor
    rel_start, rel_end = (-90, 36) if role == "donor" else (-50, 90)
    sequence, source_interval = fetch_oriented(
        genome, row.chrom, boundary, rel_start, rel_end, row.strand
    )
    boundary_index = -rel_start
    expected = "GT" if role == "donor" else "AG"
    motif = (
        sequence[boundary_index : boundary_index + 2]
        if role == "donor"
        else sequence[boundary_index - 2 : boundary_index]
    )
    if set(sequence) - BASES:
        raise ValueError("module contains a non-ACGT reference base")
    if motif != expected:
        raise ValueError(f"module lacks canonical {expected}")
    maxent_window = (
        sequence[boundary_index - 3 : boundary_index + 6]
        if role == "donor"
        else sequence[boundary_index - 20 : boundary_index + 3]
    )
    score = scorer.score5(maxent_window) if role == "donor" else scorer.score3(maxent_window)
    exon_length = (
        row.upstream_exon_end - row.upstream_exon_start
        if role == "donor"
        else row.downstream_exon_end - row.downstream_exon_start
    )
    root = row.gene_id.split(".", 1)[0]
    coordinate = f"{row.chrom}:{boundary}:{row.strand}"
    record: dict[str, object] = {
        "candidate_id": f"{role}:{coordinate}:{root}",
        "role": role,
        "gene_id": row.gene_id,
        "chrom": row.chrom,
        "strand": row.strand,
        "sequence": sequence,
        "source_start": int(source_interval[0]),
        "source_end": int(source_interval[1]),
        "source_exon_length": int(exon_length),
        "transcript_id": edge.basic_support_ids[0],
        "phase": int(edge.phase),
        "boundary_index": int(boundary_index),
        "coding_offset": int((boundary_index - edge.phase) % 3),
        "geometry": {
            "module_span": float(len(sequence)),
            "native_intron_length": float(row.intron_length),
        },
        "scores": {"maxent": float(score)},
        "annotation": annotation_payload(edge),
        "eligibility_checks": {
            **dict(edge.checks),
            "unambiguous_reference": True,
            "canonical_motif": True,
            "freshness_registry_clear": True,
        },
    }
    if include_native_health_window:
        native, native_donor, native_acceptor, native_interval = native_window(edge, genome)
        record.update(
            {
                "native_sequence": native,
                "native_sequence_sha256": hashlib.sha256(native.encode("ascii")).hexdigest(),
                "native_window_start": int(native_interval[0]),
                "native_window_end": int(native_interval[1]),
                "native_donor_pos": int(native_donor),
                "native_acceptor_pos": int(native_acceptor),
                "native_donor_genomic": int(row.donor),
                "native_acceptor_genomic": int(row.acceptor),
            }
        )
        record["eligibility_checks"]["native_health_window"] = True  # type: ignore[index]
    return record


def scaffold_record(
    edge: EdgeEvidence,
    genome: GenomeLike,
    scorer: MaxEntTables,
) -> dict[str, object]:
    """Materialize an authentic bridge scaffold and role-wise anchor vectors.

    The 1,022-nt scaffold is a contiguous hg38 window whose native acceptor AG
    supplies the frozen untouched bridge at boundary 437.  The crossed donor
    and acceptor modules later overwrite the target windows around d=392 and
    a=512, so the scaffold source intron is not required to be 120 nt.  Its own
    authentic donor/acceptor modules supply the two matching anchors.
    """

    from gfmcd.splicing.positional_boundary_coherence import (
        DEFAULT_ACCEPTOR_WINDOW,
        DEFAULT_DONOR_WINDOW,
        extract_module_covariates,
    )

    row = edge.representative
    bridge_pos = 437
    sequence, interval = fetch_oriented(
        genome, row.chrom, row.acceptor, -bridge_pos, 1022 - bridge_pos, row.strand
    )
    if len(sequence) != 1022 or set(sequence) - BASES:
        raise ValueError("scaffold lacks a complete unambiguous 1,022-nt hg38 window")
    # Amendment A1 freezes this exact untouched bridge operator.  It is not a
    # sequence-selected first hit within a bridge scan.
    if sequence[bridge_pos - 2 : bridge_pos] != "AG":
        raise ValueError("fixed scaffold bridge lacks AG at frozen boundary 437")
    donor = module_record(
        edge, genome, scorer, role="donor", include_native_health_window=False
    )
    acceptor = module_record(
        edge, genome, scorer, role="acceptor", include_native_health_window=False
    )
    root = row.gene_id.split(".", 1)[0]
    return {
        "candidate_id": f"scaffold:{row.chrom}:{row.donor}>{row.acceptor}:{row.strand}:{root}",
        "role": "scaffold",
        "gene_id": row.gene_id,
        "chrom": row.chrom,
        "strand": row.strand,
        "sequence": sequence,
        "source_start": int(interval[0]),
        "source_end": int(interval[1]),
        "source_exon_length": int(row.upstream_exon_end - row.upstream_exon_start),
        "transcript_id": edge.basic_support_ids[0],
        "donor_window": list(DEFAULT_DONOR_WINDOW),
        "acceptor_window": list(DEFAULT_ACCEPTOR_WINDOW),
        "donor_pos": 392,
        "acceptor_pos": 512,
        "native_sequence": sequence,
        "native_sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "native_donor_pos": int(bridge_pos - row.intron_length),
        "native_acceptor_pos": int(bridge_pos),
        "native_donor_genomic": int(row.donor),
        "native_acceptor_genomic": int(row.acceptor),
        "bridge_acceptor_pos": int(bridge_pos),
        "bridge_acceptor_genomic": int(
            boundary_to_genomic(interval[0], interval[1], bridge_pos, row.strand)
        ),
        "bridge_anchor_motif": "AG",
        "bridge_anchor_region": [DEFAULT_DONOR_WINDOW[1], DEFAULT_ACCEPTOR_WINDOW[0]],
        "donor_anchor_covariates": extract_module_covariates(donor),
        "acceptor_anchor_covariates": extract_module_covariates(acceptor),
        "geometry": {
            "bridge_length": float(DEFAULT_ACCEPTOR_WINDOW[0] - DEFAULT_DONOR_WINDOW[1]),
            "native_intron_length": float(row.intron_length),
        },
        "scores": {},
        "annotation": annotation_payload(edge),
        "eligibility_checks": {
            **dict(edge.checks),
            "unambiguous_reference": True,
            "canonical_motif": True,
            "exact_1022_geometry": True,
            "fixed_bridge_anchor": True,
            "freshness_registry_clear": True,
        },
    }


def exclusion_record(edge: EdgeEvidence, reason: str) -> dict[str, object]:
    row = edge.representative
    root = row.gene_id.split(".", 1)[0]
    return {
        "edge_id": f"{row.chrom}:{row.donor}>{row.acceptor}:{row.strand}:{root}",
        "gene_id": root,
        "reasons": [reason],
    }


__all__ = [
    "BASES",
    "CANDIDATE_ALGORITHM",
    "CANDIDATE_SCHEMA",
    "DEFAULT_ALLOWED_CHROMS",
    "EdgeEvidence",
    "EdgeSupport",
    "FaiRecord",
    "FORBIDDEN_CHROMS",
    "GenomeLike",
    "GtfFeature",
    "IndexedFasta",
    "MaxEntTables",
    "annotation_payload",
    "boundary_to_genomic",
    "enumerate_edge_evidence",
    "exclusion_record",
    "fetch_oriented",
    "file_record",
    "file_sha256",
    "iter_gtf_features",
    "module_record",
    "native_window",
    "oriented_interval",
    "scaffold_record",
]
