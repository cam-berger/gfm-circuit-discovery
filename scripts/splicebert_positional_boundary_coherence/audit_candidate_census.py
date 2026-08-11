#!/usr/bin/env python
"""Independently reconstruct the real Phase-I candidate census."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.pbc_biology import (  # noqa: E402
    CANDIDATE_ALGORITHM,
    CANDIDATE_SCHEMA,
    DEFAULT_ALLOWED_CHROMS,
    FORBIDDEN_CHROMS,
    file_record,
)
from gfmcd.splicing.pbc_freshness import (  # noqa: E402
    AUDIT_SCHEMA as FRESHNESS_AUDIT_SCHEMA,
    REGISTRY_SCHEMA as FRESHNESS_REGISTRY_SCHEMA,
    sha256_payload,
)


AUDIT_SCHEMA = "gfmcd.positional_boundary_coherence.candidate_census_audit.v1"
BASES = "ACGT"
STOPS = frozenset({"TAA", "TAG", "TGA"})
DONOR_WINDOW = (302, 428)
ACCEPTOR_WINDOW = (462, 602)
ROLE_WINDOWS = {
    "donor": {
        "core": (-3, 6),
        "exonic": (-90, 0),
        "proximal_intronic": (6, 36),
    },
    "acceptor": {
        "core": (-20, 3),
        "proximal_intronic": (-50, -20),
        "exonic": (0, 90),
    },
}


class _AuditIndexedFasta:
    """Auditor-owned FAI parser and random-access FASTA reader."""

    def __init__(self, fasta: str | Path, fai: str | Path):
        self.fasta_path = Path(fasta).expanduser().resolve(strict=True)
        self.records: dict[str, tuple[int, int, int, int]] = {}
        with Path(fai).expanduser().resolve(strict=True).open("rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    raise ValueError(f"audit FAI line {line_no} has fewer than five fields")
                name = fields[0]
                values = tuple(int(value) for value in fields[1:5])
                length, offset, line_bases, line_width = values
                if (
                    name in self.records
                    or length <= 0
                    or offset < 0
                    or line_bases <= 0
                    or line_width < line_bases
                ):
                    raise ValueError(f"audit FAI line {line_no} has invalid fields")
                self.records[name] = values
        if not self.records:
            raise ValueError("audit FAI contains no sequences")

    def _record(self, chrom: str) -> tuple[int, int, int, int]:
        if chrom in self.records:
            return self.records[chrom]
        alternate = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        if alternate in self.records:
            return self.records[alternate]
        raise KeyError(f"audit FASTA lacks chromosome {chrom!r}")

    def length(self, chrom: str) -> int:
        return self._record(chrom)[0]

    def fetch(self, chrom: str, start: int, end: int) -> str:
        length, offset, line_bases, line_width = self._record(chrom)
        if start < 0 or end < start or end > length:
            raise ValueError("insufficient FASTA flank")
        chunks: list[bytes] = []
        position = start
        with self.fasta_path.open("rb") as handle:
            while position < end:
                line_index, within = divmod(position, line_bases)
                take = min(end - position, line_bases - within)
                handle.seek(offset + line_index * line_width + within)
                chunk = handle.read(take)
                if len(chunk) != take:
                    raise ValueError("audit FASTA ended before its FAI interval")
                chunks.append(chunk)
                position += take
        sequence = b"".join(chunks).decode("ascii").upper().replace("U", "T")
        if len(sequence) != end - start:
            raise ValueError("audit FASTA extraction length drift")
        return sequence


class _AuditMaxEnt:
    """Auditor-owned parser and implementation of the frozen MaxEnt equations."""

    BGD5 = {"A": 0.27, "C": 0.23, "G": 0.23, "T": 0.27}
    CONS1_5 = {"A": 0.004, "C": 0.0032, "G": 0.9896, "T": 0.0032}
    CONS2_5 = {"A": 0.0034, "C": 0.0039, "G": 0.0042, "T": 0.9884}
    BGD3 = {"A": 0.27, "C": 0.23, "G": 0.23, "T": 0.27}
    CONS1_3 = {"A": 0.9903, "C": 0.0032, "G": 0.0034, "T": 0.0030}
    CONS2_3 = {"A": 0.0027, "C": 0.0037, "G": 0.9905, "T": 0.0030}

    def __init__(self, donor_table: str | Path, acceptor_table: str | Path):
        self.donor_path = Path(donor_table).expanduser().resolve(strict=True)
        self.acceptor_path = Path(acceptor_table).expanduser().resolve(strict=True)
        self.donor: dict[str, float] = {}
        with self.donor_path.open("rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                key, value = line.split()
                if len(key) != 7 or set(key) - set(BASES):
                    raise ValueError(f"audit donor table line {line_no} is invalid")
                self.donor[key] = float(value)
        acceptor: dict[int, dict[int, float]] = defaultdict(dict)
        with self.acceptor_path.open("rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                group, key, value = line.split()
                acceptor[int(group)][int(key)] = float(value)
        if not self.donor or set(acceptor) != set(range(9)):
            raise ValueError("audit MaxEnt tables are incomplete")
        self.acceptor = dict(acceptor)

    @staticmethod
    def _hashseq(sequence: str) -> int:
        digits = {"A": 0, "C": 1, "G": 2, "T": 3}
        return sum(
            digits[base] * 4 ** (len(sequence) - index - 1)
            for index, base in enumerate(sequence)
        )

    def score5(self, sequence: str) -> float:
        sequence = sequence.upper()
        if len(sequence) != 9 or set(sequence) - set(BASES) or sequence[3:5] != "GT":
            raise ValueError("donor MaxEnt requires a canonical-GT 9-mer")
        key = sequence[3:5]
        consensus = self.CONS1_5[key[0]] * self.CONS2_5[key[1]] / (
            self.BGD5[key[0]] * self.BGD5[key[1]]
        )
        return float(math.log2(consensus * self.donor[sequence[:3] + sequence[5:]]))

    def score3(self, sequence: str) -> float:
        sequence = sequence.upper()
        if len(sequence) != 23 or set(sequence) - set(BASES) or sequence[18:20] != "AG":
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
            "implementation": file_record(REPO / "src/gfmcd/splicing/pbc_biology.py"),
            "donor_table": file_record(self.donor_path),
            "acceptor_table": file_record(self.acceptor_path),
            "equations": "Yeo_Burge_MaxEntScan_9mer_23mer_local_v1",
        }


class _AuditEdge:
    def __init__(
        self,
        *,
        key: tuple[str, int, int, str, str],
        representative: Mapping[str, Any],
        phase: int,
        basic_support_ids: tuple[str, ...],
        comprehensive_support_ids: tuple[str, ...],
        checks: Mapping[str, bool],
    ) -> None:
        self.key = key
        self.representative = representative
        self.phase = phase
        self.basic_support_ids = basic_support_ids
        self.comprehensive_support_ids = comprehensive_support_ids
        self.checks = checks


def _audit_gtf_attributes(text: str) -> dict[str, str]:
    """Auditor-local GTF attribute parser (not the production parser)."""

    return {
        match.group(1): match.group(2)
        for match in re.finditer(r'(\S+)\s+"([^"]*)"\s*;', text)
    }


def _audit_open_gtf(path: str | Path):
    source = Path(path)
    return (
        gzip.open(source, "rt", encoding="utf-8")
        if source.suffix == ".gz"
        else source.open("rt", encoding="utf-8")
    )


def _audit_gtf_rows(
    path: str | Path, *, allowed_chromosomes: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    allowed = None if allowed_chromosomes is None else set(allowed_chromosomes)
    rows: list[dict[str, Any]] = []
    with _audit_open_gtf(path) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                raise ValueError(f"audit GTF line {line_no} has {len(columns)} columns")
            chrom, feature, strand = columns[0], columns[2], columns[6]
            if feature not in {"exon", "CDS"}:
                continue
            if allowed is not None and chrom not in allowed:
                continue
            if strand not in {"+", "-"}:
                continue
            attrs = _audit_gtf_attributes(columns[8])
            transcript_type = attrs.get("transcript_type") or attrs.get(
                "transcript_biotype", ""
            )
            gene_type = attrs.get("gene_type") or attrs.get("gene_biotype", "")
            if transcript_type != "protein_coding" and gene_type != "protein_coding":
                continue
            frame = (
                int(columns[7])
                if feature == "CDS" and columns[7] in {"0", "1", "2"}
                else None
            )
            if feature == "CDS" and frame is None:
                continue
            transcript = attrs.get("transcript_id")
            gene = attrs.get("gene_id")
            if not transcript or not gene:
                continue
            rows.append(
                {
                    "chrom": chrom,
                    "feature": feature,
                    "start": int(columns[3]) - 1,
                    "end": int(columns[4]),
                    "strand": strand,
                    "frame": frame,
                    "gene_id": gene,
                    "transcript_id": transcript,
                    "gene_type": gene_type,
                    "transcript_type": transcript_type,
                }
            )
    return rows


def _audit_edge_supports(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_transcript: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_transcript[str(row["transcript_id"])].append(row)
    supports: list[dict[str, Any]] = []
    for transcript in sorted(by_transcript):
        local = by_transcript[transcript]
        strand = str(local[0]["strand"])
        exons = sorted(
            (row for row in local if row["feature"] == "exon"),
            key=lambda row: (int(row["start"]), int(row["end"])),
            reverse=strand == "-",
        )
        cds = [row for row in local if row["feature"] == "CDS"]
        if len(exons) < 2:
            continue
        for upstream, downstream in zip(exons[:-1], exons[1:]):
            if strand == "+":
                donor, acceptor = int(upstream["end"]), int(downstream["start"])
                up_rows = [
                    row
                    for row in cds
                    if int(row["end"]) == donor
                    and int(row["start"]) >= int(upstream["start"])
                ]
                down_rows = [
                    row
                    for row in cds
                    if int(row["start"]) == acceptor
                    and int(row["end"]) <= int(downstream["end"])
                ]
            else:
                donor, acceptor = int(upstream["start"]), int(downstream["end"])
                up_rows = [
                    row
                    for row in cds
                    if int(row["start"]) == donor
                    and int(row["end"]) <= int(upstream["end"])
                ]
                down_rows = [
                    row
                    for row in cds
                    if int(row["end"]) == acceptor
                    and int(row["start"]) >= int(downstream["start"])
                ]
            if len(up_rows) != 1 or len(down_rows) != 1:
                continue
            up, down = up_rows[0], down_rows[0]
            donor_phase = (
                (3 - int(up["frame"])) + int(up["end"]) - int(up["start"])
            ) % 3
            acceptor_phase = (3 - int(down["frame"])) % 3
            supports.append(
                {
                    "chrom": str(upstream["chrom"]),
                    "donor": donor,
                    "acceptor": acceptor,
                    "strand": strand,
                    "gene_id": str(upstream["gene_id"]),
                    "transcript_id": transcript,
                    "upstream_exon_start": int(upstream["start"]),
                    "upstream_exon_end": int(upstream["end"]),
                    "downstream_exon_start": int(downstream["start"]),
                    "downstream_exon_end": int(downstream["end"]),
                    "upstream_cds_start": int(up["start"]),
                    "upstream_cds_end": int(up["end"]),
                    "downstream_cds_start": int(down["start"]),
                    "downstream_cds_end": int(down["end"]),
                    "upstream_frame": int(up["frame"]),
                    "downstream_frame": int(down["frame"]),
                    "donor_phase": donor_phase,
                    "acceptor_phase": acceptor_phase,
                    "intron_length": abs(acceptor - donor),
                }
            )
    return supports


def _audit_oriented_interval(
    boundary: int, relative_start: int, relative_end: int, strand: str
) -> tuple[int, int]:
    if strand == "+":
        return boundary + relative_start, boundary + relative_end
    if strand == "-":
        return boundary - relative_end, boundary - relative_start
    raise ValueError("audit interval has invalid strand")


def _audit_reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _audit_fetch_oriented(
    genome: _AuditIndexedFasta,
    chrom: str,
    boundary: int,
    relative_start: int,
    relative_end: int,
    strand: str,
) -> tuple[str, tuple[int, int]]:
    start, end = _audit_oriented_interval(
        boundary, relative_start, relative_end, strand
    )
    if start < 0 or end > genome.length(chrom):
        raise ValueError("insufficient FASTA flank")
    sequence = genome.fetch(chrom, start, end)
    if strand == "-":
        sequence = _audit_reverse_complement(sequence)
    return sequence.upper(), (start, end)


def _audit_codon_slot(
    position: int, start: int, end: int, strand: str, frame: int
) -> int:
    offset = position - start if strand == "+" else end - 1 - position
    return (offset - frame) % 3


def _audit_edge_id(edge: _AuditEdge) -> str:
    row = edge.representative
    root = str(row["gene_id"]).split(".", 1)[0]
    return f"{row['chrom']}:{row['donor']}>{row['acceptor']}:{row['strand']}:{root}"


def _audit_exclusion(edge: _AuditEdge, reason: str) -> dict[str, object]:
    root = str(edge.representative["gene_id"]).split(".", 1)[0]
    return {"edge_id": _audit_edge_id(edge), "gene_id": root, "reasons": [reason]}


class _AuditFeatureIntervalIndex:
    """Auditor-owned exact interval index preserving annotation row order."""

    def __init__(self, features: Iterable[Mapping[str, Any]]):
        self._features = tuple(features)
        records = sorted(
            (int(feature["start"]), int(feature["end"]), index)
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

    def overlaps(
        self, intervals: Iterable[tuple[int, int]]
    ) -> tuple[Mapping[str, Any], ...]:
        found: set[int] = set()
        for start, end in intervals:
            if end <= start:
                raise ValueError("audit query interval must be non-empty")
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


def _audit_enumerate_edges(
    basic_gtf: str | Path,
    comprehensive_gtf: str | Path,
    *,
    allowed_chromosomes: Iterable[str],
) -> tuple[list[_AuditEdge], list[dict[str, object]]]:
    allowed = frozenset(str(value) for value in allowed_chromosomes)
    basic_features = _audit_gtf_rows(basic_gtf, allowed_chromosomes=allowed)
    comprehensive_features = _audit_gtf_rows(
        comprehensive_gtf, allowed_chromosomes=allowed
    )
    basic_supports = _audit_edge_supports(basic_features)
    comprehensive_supports = _audit_edge_supports(comprehensive_features)
    by_basic: dict[tuple[str, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_comprehensive: dict[
        tuple[str, int, int, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for support in basic_supports:
        root = str(support["gene_id"]).split(".", 1)[0]
        key = (
            str(support["chrom"]),
            int(support["donor"]),
            int(support["acceptor"]),
            str(support["strand"]),
            root,
        )
        by_basic[key].append(support)
    for support in comprehensive_supports:
        root = str(support["gene_id"]).split(".", 1)[0]
        key = (
            str(support["chrom"]),
            int(support["donor"]),
            int(support["acceptor"]),
            str(support["strand"]),
            root,
        )
        by_comprehensive[key].append(support)
    coordinate_genes: dict[tuple[str, int, int, str], set[str]] = defaultdict(set)
    for chrom, donor, acceptor, strand, root in by_basic:
        coordinate_genes[(chrom, donor, acceptor, strand)].add(root)
    comprehensive_by_chrom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in comprehensive_features:
        comprehensive_by_chrom[str(feature["chrom"])].append(feature)
    comprehensive_indexes = {
        chrom: _AuditFeatureIntervalIndex(features)
        for chrom, features in comprehensive_by_chrom.items()
    }

    del basic_features, comprehensive_features, basic_supports, comprehensive_supports
    del comprehensive_by_chrom

    retained: list[_AuditEdge] = []
    exclusions: list[dict[str, object]] = []
    for key in sorted(by_basic):
        chrom, donor, acceptor, strand, root = key
        supports = by_basic[key]
        comprehensive = by_comprehensive.get(key, [])
        reasons: list[str] = []
        endpoint_pairs = {
            (int(row["donor_phase"]), int(row["acceptor_phase"]))
            for row in supports
        }
        basic_phases = {
            int(row["donor_phase"])
            for row in supports
            if row["donor_phase"] == row["acceptor_phase"]
        }
        comprehensive_pairs = {
            (int(row["donor_phase"]), int(row["acceptor_phase"]))
            for row in comprehensive
        }
        comprehensive_phases = {
            int(row["donor_phase"])
            for row in comprehensive
            if row["donor_phase"] == row["acceptor_phase"]
        }
        if any(left != right for left, right in endpoint_pairs):
            reasons.append("basic_endpoint_phase_disagreement")
        if len(basic_phases) != 1 or len(endpoint_pairs) != 1:
            reasons.append("basic_coordinate_transcript_phase_conflict")
        if not comprehensive:
            reasons.append("comprehensive_exact_edge_absent")
        if any(left != right for left, right in comprehensive_pairs):
            reasons.append("comprehensive_endpoint_phase_disagreement")
        if len(comprehensive_phases) != 1 or len(comprehensive_pairs) != 1:
            reasons.append("comprehensive_coordinate_transcript_phase_conflict")
        if basic_phases and comprehensive_phases and basic_phases != comprehensive_phases:
            reasons.append("basic_comprehensive_phase_conflict")
        if len(coordinate_genes[(chrom, donor, acceptor, strand)]) != 1:
            reasons.append("duplicate_or_pseudoautosomal_locus")

        representative = sorted(supports, key=lambda row: str(row["transcript_id"]))[0]
        intervals = (
            _audit_oriented_interval(donor, -90, 36, strand),
            _audit_oriented_interval(acceptor, -50, 90, strand),
        )
        index = comprehensive_indexes.get(chrom)
        nearby = index.overlaps(intervals) if index is not None else ()
        if any(str(feature["strand"]) != strand for feature in nearby):
            reasons.append("comprehensive_local_strand_conflict")
        if any(str(feature["gene_id"]).split(".", 1)[0] != root for feature in nearby):
            reasons.append("comprehensive_local_gene_conflict")
        same_cds = [
            feature
            for feature in nearby
            if feature["feature"] == "CDS"
            and feature["strand"] == strand
            and str(feature["gene_id"]).split(".", 1)[0] == root
        ]
        conflict = False
        for start, end in intervals:
            for position in range(start, end):
                slots = {
                    _audit_codon_slot(
                        position,
                        int(feature["start"]),
                        int(feature["end"]),
                        strand,
                        int(feature["frame"]),
                    )
                    for feature in same_cds
                    if int(feature["start"]) <= position < int(feature["end"])
                }
                if len(slots) > 1:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            reasons.append("comprehensive_local_coding_register_conflict")
        edge_id = f"{chrom}:{donor}>{acceptor}:{strand}:{root}"
        if reasons:
            exclusions.append(
                {
                    "edge_id": edge_id,
                    "gene_id": root,
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        phase = next(iter(basic_phases))
        retained.append(
            _AuditEdge(
                key=key,
                representative=representative,
                phase=int(phase),
                basic_support_ids=tuple(
                    sorted(str(row["transcript_id"]) for row in supports)
                ),
                comprehensive_support_ids=tuple(
                    sorted(str(row["transcript_id"]) for row in comprehensive)
                ),
                checks={
                    "upstream_downstream_phase_equal": True,
                    "basic_transcript_coordinate_agreement": True,
                    "comprehensive_no_coding_conflict": True,
                    "comprehensive_no_strand_conflict": True,
                    "coding_offset_consistent_with_gtf_frame": True,
                    "comprehensive_exact_edge_phase_agreement": True,
                    "no_duplicate_or_pseudoautosomal_locus": True,
                },
            )
        )
    return retained, sorted(exclusions, key=lambda row: str(row["edge_id"]))


def _audit_phase_index(path: str | Path) -> dict[tuple[str, int, int, str, str], dict[str, Any]]:
    """Independently derive endpoint phases from raw exon/CDS rows."""

    transcripts: dict[str, list[dict[str, Any]]] = {}
    with _audit_open_gtf(path) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                raise ValueError(f"audit GTF line {line_no} has {len(columns)} columns")
            if columns[2] not in {"exon", "CDS"}:
                continue
            attrs = _audit_gtf_attributes(columns[8])
            if (
                attrs.get("transcript_type") != "protein_coding"
                and attrs.get("gene_type") != "protein_coding"
            ):
                continue
            transcript = attrs.get("transcript_id")
            gene = attrs.get("gene_id")
            if not transcript or not gene:
                continue
            frame = int(columns[7]) if columns[2] == "CDS" and columns[7] in {"0", "1", "2"} else None
            if columns[2] == "CDS" and frame is None:
                continue
            transcripts.setdefault(transcript, []).append(
                {
                    "chrom": columns[0],
                    "feature": columns[2],
                    "start": int(columns[3]) - 1,
                    "end": int(columns[4]),
                    "strand": columns[6],
                    "frame": frame,
                    "gene": gene,
                }
            )
    evidence: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    for transcript in sorted(transcripts):
        rows = transcripts[transcript]
        strand = str(rows[0]["strand"])
        exons = sorted(
            (row for row in rows if row["feature"] == "exon"),
            key=lambda row: (row["start"], row["end"]),
            reverse=strand == "-",
        )
        cds = [row for row in rows if row["feature"] == "CDS"]
        for upstream, downstream in zip(exons[:-1], exons[1:]):
            if strand == "+":
                donor, acceptor = int(upstream["end"]), int(downstream["start"])
                up_rows = [row for row in cds if row["end"] == donor and row["start"] >= upstream["start"]]
                down_rows = [row for row in cds if row["start"] == acceptor and row["end"] <= downstream["end"]]
            else:
                donor, acceptor = int(upstream["start"]), int(downstream["end"])
                up_rows = [row for row in cds if row["start"] == donor and row["end"] <= upstream["end"]]
                down_rows = [row for row in cds if row["end"] == acceptor and row["start"] >= downstream["start"]]
            if len(up_rows) != 1 or len(down_rows) != 1:
                continue
            up, down = up_rows[0], down_rows[0]
            donor_phase = ((3 - int(up["frame"])) + int(up["end"]) - int(up["start"])) % 3
            acceptor_phase = (3 - int(down["frame"])) % 3
            root = str(up["gene"]).split(".", 1)[0]
            key = (str(up["chrom"]), donor, acceptor, strand, root)
            record = evidence.setdefault(key, {"pairs": set(), "support_ids": set()})
            record["pairs"].add((donor_phase, acceptor_phase))
            record["support_ids"].add(transcript)
    return evidence


def _audit_candidate_phases(
    candidates: Any, basic_gtf: str | Path, comprehensive_gtf: str | Path
) -> bool:
    if not isinstance(candidates, list) or not candidates:
        return False
    basic = _audit_phase_index(basic_gtf)
    comprehensive = _audit_phase_index(comprehensive_gtf)
    for row in candidates:
        annotation = row.get("annotation")
        if not isinstance(annotation, Mapping):
            # Every production scaffold also carries the edge-derived anchor
            # annotation even though it has no top-level phase label.
            return False
        try:
            root = str(row["gene_id"]).split(".", 1)[0]
            key = (
                str(row["chrom"]),
                int(row["native_donor_genomic"]),
                int(row["native_acceptor_genomic"]),
                str(row["strand"]),
                root,
            )
            phase = int(row.get("phase", annotation["basic_phase"]))
            basic_row, comp_row = basic[key], comprehensive[key]
        except (KeyError, TypeError, ValueError):
            return False
        if (
            basic_row["pairs"] != {(phase, phase)}
            or comp_row["pairs"] != {(phase, phase)}
            or sorted(basic_row["support_ids"]) != sorted(annotation["basic_support_ids"])
            or sorted(comp_row["support_ids"])
            != sorted(annotation["comprehensive_support_ids"])
            or int(annotation.get("upstream_cds_end_phase", -1)) != phase
            or int(annotation.get("downstream_cds_start_phase", -1)) != phase
        ):
            return False
    return True


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve(strict=True).open("rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _freshness_ids(registry: Mapping[str, Any], audit: Mapping[str, Any]) -> set[str]:
    if registry.get("schema") != FRESHNESS_REGISTRY_SCHEMA:
        raise ValueError("wrong freshness registry schema")
    if registry.get("stage") != "construction":
        raise ValueError("wrong freshness registry stage")
    if (
        audit.get("schema") != FRESHNESS_AUDIT_SCHEMA
        or audit.get("stage") != "construction"
        or audit.get("status") != "AUDIT_PASS"
        or audit.get("passed") is not True
        or audit.get("registry_content_sha256") != registry.get("content_sha256")
    ):
        raise ValueError("freshness registry lacks a matching passing audit")
    genes = registry.get("gene_ids")
    if not isinstance(genes, list) or genes != sorted(set(genes)):
        raise ValueError("freshness gene list is not sorted and unique")
    return {str(value) for value in genes}


def _audit_annotation(edge: _AuditEdge) -> dict[str, object]:
    row = edge.representative
    return {
        "phase_convention": "intron_phase",
        "raw_gtf_frame": (
            f"upstream={row['upstream_frame']};downstream={row['downstream_frame']}"
        ),
        "upstream_gtf_frame": int(row["upstream_frame"]),
        "downstream_gtf_frame": int(row["downstream_frame"]),
        "upstream_cds_end": int(row["donor"]),
        "downstream_cds_start": int(row["acceptor"]),
        "upstream_cds_end_phase": int(row["donor_phase"]),
        "downstream_cds_start_phase": int(row["acceptor_phase"]),
        "basic_phase": int(edge.phase),
        "comprehensive_phase": int(edge.phase),
        "basic_support_ids": list(edge.basic_support_ids),
        "comprehensive_support_ids": list(edge.comprehensive_support_ids),
        "checks": dict(sorted(edge.checks.items())),
    }


def _audit_native_window(
    edge: _AuditEdge, genome: _AuditIndexedFasta
) -> tuple[str, int, int, tuple[int, int]]:
    row = edge.representative
    sequence, interval = _audit_fetch_oriented(
        genome,
        str(row["chrom"]),
        int(row["acceptor"]),
        -512,
        510,
        str(row["strand"]),
    )
    donor_pos = 512 - int(row["intron_length"])
    if not 90 <= donor_pos < 510:
        raise ValueError("native donor lacks the frozen health-window support")
    if sequence[donor_pos : donor_pos + 2] != "GT" or sequence[510:512] != "AG":
        raise ValueError("native source motifs do not reconstruct")
    return sequence, donor_pos, 512, interval


def _audit_module(
    edge: _AuditEdge,
    genome: _AuditIndexedFasta,
    scorer: _AuditMaxEnt,
    *,
    role: str,
    include_native_health_window: bool = True,
) -> dict[str, object]:
    if role not in {"donor", "acceptor"}:
        raise ValueError("module role must be donor or acceptor")
    row = edge.representative
    boundary = int(row["donor"] if role == "donor" else row["acceptor"])
    relative_start, relative_end = (-90, 36) if role == "donor" else (-50, 90)
    sequence, interval = _audit_fetch_oriented(
        genome,
        str(row["chrom"]),
        boundary,
        relative_start,
        relative_end,
        str(row["strand"]),
    )
    boundary_index = -relative_start
    motif = (
        sequence[boundary_index : boundary_index + 2]
        if role == "donor"
        else sequence[boundary_index - 2 : boundary_index]
    )
    expected = "GT" if role == "donor" else "AG"
    if set(sequence) - set(BASES):
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
        int(row["upstream_exon_end"]) - int(row["upstream_exon_start"])
        if role == "donor"
        else int(row["downstream_exon_end"]) - int(row["downstream_exon_start"])
    )
    root = str(row["gene_id"]).split(".", 1)[0]
    coordinate = f"{row['chrom']}:{boundary}:{row['strand']}"
    record: dict[str, object] = {
        "candidate_id": f"{role}:{coordinate}:{root}",
        "role": role,
        "gene_id": str(row["gene_id"]),
        "chrom": str(row["chrom"]),
        "strand": str(row["strand"]),
        "sequence": sequence,
        "source_start": int(interval[0]),
        "source_end": int(interval[1]),
        "source_exon_length": exon_length,
        "transcript_id": edge.basic_support_ids[0],
        "phase": int(edge.phase),
        "boundary_index": int(boundary_index),
        "coding_offset": int((boundary_index - edge.phase) % 3),
        "geometry": {
            "module_span": float(len(sequence)),
            "native_intron_length": float(row["intron_length"]),
        },
        "scores": {"maxent": float(score)},
        "annotation": _audit_annotation(edge),
        "eligibility_checks": {
            **dict(edge.checks),
            "unambiguous_reference": True,
            "canonical_motif": True,
            "freshness_registry_clear": True,
        },
    }
    if include_native_health_window:
        native, native_donor, native_acceptor, native_interval = _audit_native_window(
            edge, genome
        )
        record.update(
            {
                "native_sequence": native,
                "native_sequence_sha256": hashlib.sha256(
                    native.encode("ascii")
                ).hexdigest(),
                "native_window_start": int(native_interval[0]),
                "native_window_end": int(native_interval[1]),
                "native_donor_pos": int(native_donor),
                "native_acceptor_pos": int(native_acceptor),
                "native_donor_genomic": int(row["donor"]),
                "native_acceptor_genomic": int(row["acceptor"]),
            }
        )
        checks = record["eligibility_checks"]
        assert isinstance(checks, dict)
        checks["native_health_window"] = True
    return record


def _audit_kmer_frequencies(sequence: str, k: int) -> dict[str, float]:
    kmers = tuple("".join(chars) for chars in itertools.product(BASES, repeat=k))
    denominator = max(1, len(sequence) - k + 1)
    counts = Counter(
        sequence[index : index + k] for index in range(len(sequence) - k + 1)
    )
    return {kmer: counts.get(kmer, 0) / denominator for kmer in kmers}


def _audit_stop_load(sequence: str, coding_offset: int) -> dict[str, float | int]:
    starts = range(int(coding_offset), max(int(coding_offset), len(sequence) - 2), 3)
    codons = [
        sequence[start : start + 3]
        for start in starts
        if start + 3 <= len(sequence)
    ]
    count = sum(codon in STOPS for codon in codons)
    return {
        "stop_count": int(count),
        "codon_count": int(len(codons)),
        "stop_rate": float(count / len(codons)) if codons else 0.0,
    }


def _audit_module_covariates(candidate: Mapping[str, Any]) -> dict[str, float]:
    role = str(candidate["role"])
    sequence = str(candidate["sequence"])
    boundary = int(candidate["boundary_index"])
    output: dict[str, float] = {
        "sequence.length": float(len(sequence)),
        "source.exon_length": float(candidate["source_exon_length"]),
        "sequence.gc": float(sum(base in "GC" for base in sequence) / len(sequence)),
        "sequence.cpg_count": float(
            sum(sequence[index : index + 2] == "CG" for index in range(len(sequence) - 1))
        ),
        "sequence.cpg": float(
            sum(sequence[index : index + 2] == "CG" for index in range(len(sequence) - 1))
            / max(1, len(sequence) - 1)
        ),
        "strand.plus": float(candidate["strand"] == "+"),
        "strand.minus": float(candidate["strand"] == "-"),
    }
    for chrom in sorted(DEFAULT_ALLOWED_CHROMS):
        output[f"chromosome.{chrom}"] = float(candidate["chrom"] == chrom)
    core_start, core_end = ROLE_WINDOWS[role]["core"]
    for offset in range(core_start, core_end):
        base_at_position = sequence[boundary + offset]
        label = f"p{offset}" if offset >= 0 else f"m{abs(offset)}"
        for base in BASES:
            output[f"core.{label}.{base}"] = float(base_at_position == base)
    for window_name in ("exonic", "proximal_intronic"):
        relative_start, relative_end = ROLE_WINDOWS[role][window_name]
        window = sequence[boundary + relative_start : boundary + relative_end]
        output[f"{window_name}.length"] = float(len(window))
        output[f"{window_name}.gc"] = float(
            sum(base in "GC" for base in window) / len(window)
        )
        cpg_count = sum(
            window[index : index + 2] == "CG" for index in range(len(window) - 1)
        )
        output[f"{window_name}.cpg_count"] = float(cpg_count)
        output[f"{window_name}.cpg_rate"] = float(
            cpg_count / max(1, len(window) - 1)
        )
        for k, family in ((1, "mono"), (2, "di"), (3, "tri")):
            for kmer, value in _audit_kmer_frequencies(window, k).items():
                output[f"{window_name}.{family}.{kmer}"] = float(value)
    relative_start, relative_end = ROLE_WINDOWS[role]["exonic"]
    coding_start, coding_end = boundary + relative_start, boundary + relative_end
    coding = sequence[coding_start:coding_end]
    coding_offset = (int(candidate["coding_offset"]) - coding_start) % 3
    output["coding.length"] = float(len(coding))
    slot_counts = {(slot, base): 0 for slot in range(3) for base in BASES}
    slot_totals = Counter()
    for index, base in enumerate(coding):
        slot = (index - coding_offset) % 3
        slot_counts[(slot, base)] += 1
        slot_totals[slot] += 1
    for slot in range(3):
        for base in BASES:
            output[f"coding.pos{slot}.{base}"] = float(
                slot_counts[(slot, base)] / max(1, slot_totals[slot])
            )
    tri_by_slot = {
        (slot, kmer): 0
        for slot in range(3)
        for kmer in ("".join(chars) for chars in itertools.product(BASES, repeat=3))
    }
    tri_totals = Counter()
    for index in range(max(0, len(coding) - 2)):
        slot = (index - coding_offset) % 3
        tri_by_slot[(slot, coding[index : index + 3])] += 1
        tri_totals[slot] += 1
    for (slot, kmer), count in sorted(tri_by_slot.items()):
        output[f"coding.tri_pos{slot}.{kmer}"] = float(
            count / max(1, tri_totals[slot])
        )
    output.update(
        {
            f"coding.{key}": float(value)
            for key, value in _audit_stop_load(coding, coding_offset).items()
        }
    )
    output.update(
        {
            f"geometry.{key}": float(value)
            for key, value in candidate["geometry"].items()
        }
    )
    output.update(
        {f"score.{key}": float(value) for key, value in candidate["scores"].items()}
    )
    return dict(sorted(output.items()))


def _audit_scaffold(
    edge: _AuditEdge, genome: _AuditIndexedFasta, scorer: _AuditMaxEnt
) -> dict[str, object]:
    row = edge.representative
    bridge_pos = 437
    sequence, interval = _audit_fetch_oriented(
        genome,
        str(row["chrom"]),
        int(row["acceptor"]),
        -bridge_pos,
        1022 - bridge_pos,
        str(row["strand"]),
    )
    if len(sequence) != 1022 or set(sequence) - set(BASES):
        raise ValueError("scaffold lacks a complete unambiguous 1,022-nt hg38 window")
    if sequence[bridge_pos - 2 : bridge_pos] != "AG":
        raise ValueError("fixed scaffold bridge lacks AG at frozen boundary 437")
    donor = _audit_module(
        edge, genome, scorer, role="donor", include_native_health_window=False
    )
    acceptor = _audit_module(
        edge, genome, scorer, role="acceptor", include_native_health_window=False
    )
    root = str(row["gene_id"]).split(".", 1)[0]
    bridge_genomic = (
        int(interval[0]) + bridge_pos
        if row["strand"] == "+"
        else int(interval[1]) - bridge_pos
    )
    return {
        "candidate_id": (
            f"scaffold:{row['chrom']}:{row['donor']}>{row['acceptor']}:"
            f"{row['strand']}:{root}"
        ),
        "role": "scaffold",
        "gene_id": str(row["gene_id"]),
        "chrom": str(row["chrom"]),
        "strand": str(row["strand"]),
        "sequence": sequence,
        "source_start": int(interval[0]),
        "source_end": int(interval[1]),
        "source_exon_length": int(row["upstream_exon_end"])
        - int(row["upstream_exon_start"]),
        "transcript_id": edge.basic_support_ids[0],
        "donor_window": list(DONOR_WINDOW),
        "acceptor_window": list(ACCEPTOR_WINDOW),
        "donor_pos": 392,
        "acceptor_pos": 512,
        "native_sequence": sequence,
        "native_sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "native_donor_pos": int(bridge_pos - int(row["intron_length"])),
        "native_acceptor_pos": bridge_pos,
        "native_donor_genomic": int(row["donor"]),
        "native_acceptor_genomic": int(row["acceptor"]),
        "bridge_acceptor_pos": bridge_pos,
        "bridge_acceptor_genomic": bridge_genomic,
        "bridge_anchor_motif": "AG",
        "bridge_anchor_region": [DONOR_WINDOW[1], ACCEPTOR_WINDOW[0]],
        "donor_anchor_covariates": _audit_module_covariates(donor),
        "acceptor_anchor_covariates": _audit_module_covariates(acceptor),
        "geometry": {
            "bridge_length": float(ACCEPTOR_WINDOW[0] - DONOR_WINDOW[1]),
            "native_intron_length": float(row["intron_length"]),
        },
        "scores": {},
        "annotation": _audit_annotation(edge),
        "eligibility_checks": {
            **dict(edge.checks),
            "unambiguous_reference": True,
            "canonical_motif": True,
            "exact_1022_geometry": True,
            "fixed_bridge_anchor": True,
            "freshness_registry_clear": True,
        },
    }


def independently_reconstruct(
    *,
    fasta_path: str | Path,
    fai_path: str | Path,
    basic_gtf_path: str | Path,
    comprehensive_gtf_path: str | Path,
    donor_table_path: str | Path,
    acceptor_table_path: str | Path,
    freshness_registry: Mapping[str, Any],
    freshness_audit: Mapping[str, Any],
    allowed_chromosomes: set[str] | frozenset[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int], dict[str, int]]:
    """Second materialization pass, separate from the producer entrypoint."""

    blacklist = _freshness_ids(freshness_registry, freshness_audit)
    genome = _AuditIndexedFasta(fasta_path, fai_path)
    scorer = _AuditMaxEnt(donor_table_path, acceptor_table_path)
    edges, exclusions = _audit_enumerate_edges(
        basic_gtf_path,
        comprehensive_gtf_path,
        allowed_chromosomes=allowed_chromosomes,
    )
    rebuilt: list[dict[str, object]] = []
    rebuilt_exclusions = list(exclusions)
    for edge in edges:
        root = str(edge.representative["gene_id"]).split(".", 1)[0]
        if root in blacklist:
            rebuilt_exclusions.append(_audit_exclusion(edge, "freshness_registry_gene"))
            continue
        try:
            donor = _audit_module(edge, genome, scorer, role="donor")
            acceptor = _audit_module(edge, genome, scorer, role="acceptor")
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            rebuilt_exclusions.append(
                _audit_exclusion(edge, f"module:{type(exc).__name__}:{exc}")
            )
        else:
            rebuilt.extend((donor, acceptor))
        try:
            rebuilt.append(_audit_scaffold(edge, genome, scorer))
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            rebuilt_exclusions.append(
                _audit_exclusion(edge, f"scaffold:{type(exc).__name__}:{exc}")
            )
    rebuilt.sort(key=lambda row: str(row["candidate_id"]))
    rebuilt_exclusions.sort(key=lambda row: (str(row["edge_id"]), tuple(row["reasons"])))
    counts = Counter(str(row["role"]) for row in rebuilt)
    for row in rebuilt:
        role = str(row["role"])
        if role in {"donor", "acceptor"}:
            counts[f"{role}.phase{int(row['phase'])}"] += 1
        counts[f"{role}.chrom.{row['chrom']}.strand.{row['strand']}"] += 1
    exclusion_counts = Counter(
        reason for row in rebuilt_exclusions for reason in row.get("reasons", [])
    )
    return (
        rebuilt,
        rebuilt_exclusions,
        dict(sorted(counts.items())),
        dict(sorted(exclusion_counts.items())),
    )


def audit_package(
    package: Mapping[str, Any],
    *,
    fasta_path: str | Path,
    fai_path: str | Path,
    basic_gtf_path: str | Path,
    comprehensive_gtf_path: str | Path,
    donor_table_path: str | Path,
    acceptor_table_path: str | Path,
    freshness_registry: Mapping[str, Any],
    freshness_audit: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = frozenset(str(value) for value in package.get("allowed_chromosomes", []))
    current_pins = {
        "fasta": file_record(fasta_path),
        "fai": file_record(fai_path),
        "basic_gtf": {
            **file_record(basic_gtf_path),
            "release": "GENCODE_v50",
            "coordinate_convention": "GTF_1_based_closed_parsed_as_0_based_half_open",
            "phase_convention": "GTF_frame_to_conventional_intron_phase",
        },
        "comprehensive_gtf": {
            **file_record(comprehensive_gtf_path),
            "release": "GENCODE_v50",
            "role": "local_coding_strand_and_phase_safety_audit",
        },
        "maxent": _AuditMaxEnt(donor_table_path, acceptor_table_path).pin(),
        "freshness_registry_sha256": sha256_payload(freshness_registry),
        "freshness_registry_content_sha256": freshness_registry.get("content_sha256"),
        "freshness_audit_sha256": sha256_payload(freshness_audit),
    }
    try:
        candidates, exclusions, counts, exclusion_counts = independently_reconstruct(
            fasta_path=fasta_path,
            fai_path=fai_path,
            basic_gtf_path=basic_gtf_path,
            comprehensive_gtf_path=comprehensive_gtf_path,
            donor_table_path=donor_table_path,
            acceptor_table_path=acceptor_table_path,
            freshness_registry=freshness_registry,
            freshness_audit=freshness_audit,
            allowed_chromosomes=allowed,
        )
        reconstruction_error = None
    except Exception as exc:
        candidates, exclusions, counts, exclusion_counts = [], [], {}, {}
        reconstruction_error = f"{type(exc).__name__}:{exc}"
    recorded_candidates = package.get("candidates")
    candidate_ids = (
        [str(row.get("candidate_id")) for row in recorded_candidates]
        if isinstance(recorded_candidates, list)
        else []
    )
    checks = {
        "schema": package.get("schema") == CANDIDATE_SCHEMA,
        "stage": package.get("stage") == "construction",
        "production_mode": package.get("candidate_mode") == "production",
        "algorithm_version": package.get("algorithm_version") == CANDIDATE_ALGORITHM,
        "outcome_firewall": package.get("model_outcomes_read") is False,
        "allowed_chromosomes": allowed == DEFAULT_ALLOWED_CHROMS,
        "forbidden_chromosomes": package.get("forbidden_chromosomes")
        == sorted(FORBIDDEN_CHROMS),
        "input_pins": package.get("input_pins") == current_pins,
        "freshness_count_and_hash": package.get("freshness")
        == {
            "normalized_count": len(freshness_registry.get("gene_ids", [])),
            "gene_ids_sha256": freshness_registry.get("gene_ids_sha256"),
            "audit_status": freshness_audit.get("status"),
        },
        "candidate_ids_sorted_unique": bool(candidate_ids)
        and candidate_ids == sorted(set(candidate_ids)),
        "biology_reconstruction": reconstruction_error is None
        and recorded_candidates == candidates,
        "independent_phase_reconstruction": _audit_candidate_phases(
            recorded_candidates, basic_gtf_path, comprehensive_gtf_path
        ),
        "exclusion_reconstruction": package.get("exclusions") == exclusions,
        "counts_reconstruction": package.get("counts") == counts,
        "exclusion_counts_reconstruction": package.get("exclusion_counts")
        == exclusion_counts,
        "content_hash": package.get("content_sha256")
        == sha256_payload({key: value for key, value in package.items() if key != "content_sha256"}),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if reconstruction_error is not None:
        failures.append(f"reconstruction_error:{reconstruction_error}")
    passed = not failures
    return {
        "schema": AUDIT_SCHEMA,
        "stage": "construction",
        "status": "AUDIT_PASS" if passed else "AUDIT_FAIL",
        "passed": passed,
        "target_execution_authorized": False,
        "checks": checks,
        "failures": failures,
        "candidate_census_sha256": sha256_payload(package),
        "candidate_content_sha256": package.get("content_sha256"),
        "reconstructed_candidates_sha256": sha256_payload(candidates),
        "reconstructed_exclusions_sha256": sha256_payload(exclusions),
        "counts": counts,
    }


def audit_artifact(
    *,
    candidate_path: str | Path,
    fasta_path: str | Path,
    fai_path: str | Path,
    basic_gtf_path: str | Path,
    comprehensive_gtf_path: str | Path,
    donor_table_path: str | Path,
    acceptor_table_path: str | Path,
    freshness_registry_path: str | Path,
    freshness_audit_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    candidate_file = Path(candidate_path).expanduser().resolve(strict=True)
    registry_file = Path(freshness_registry_path).expanduser().resolve(strict=True)
    freshness_audit_file = Path(freshness_audit_path).expanduser().resolve(strict=True)
    output = guard_output_path(Path(output_path).expanduser().resolve())
    if output.suffix != ".json":
        raise ValueError("candidate audit output must be JSON")
    manifest_path = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    audit = audit_package(
        _read_json(candidate_file),
        fasta_path=fasta_path,
        fai_path=fai_path,
        basic_gtf_path=basic_gtf_path,
        comprehensive_gtf_path=comprehensive_gtf_path,
        donor_table_path=donor_table_path,
        acceptor_table_path=acceptor_table_path,
        freshness_registry=_read_json(registry_file),
        freshness_audit=_read_json(freshness_audit_file),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert-positional-boundary-coherence:candidate-census-audit-v1",
        sys.argv,
        repo=REPO,
        status=str(audit["status"]),
        config={"stage": "construction"},
        notes=[
            "Independent biology reconstruction from pinned FASTA, FAI, both GTFs, and MaxEnt tables.",
            "This audit authorizes only candidate-census use; it never authorizes a model target.",
        ],
    )
    for path in (
        candidate_file,
        fasta_path,
        fai_path,
        basic_gtf_path,
        comprehensive_gtf_path,
        donor_table_path,
        acceptor_table_path,
        registry_file,
        freshness_audit_file,
        Path(__file__).resolve(),
        REPO / "src/gfmcd/splicing/pbc_biology.py",
    ):
        manifest.add_input(Path(path).expanduser().resolve(strict=True))
    manifest.add_output(output)
    manifest.write(manifest_path)
    return output, manifest_path, audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--fai", required=True)
    parser.add_argument("--basic-gtf", required=True)
    parser.add_argument("--comprehensive-gtf", required=True)
    parser.add_argument("--maxent-donor-table", required=True)
    parser.add_argument("--maxent-acceptor-table", required=True)
    parser.add_argument("--freshness-registry", required=True)
    parser.add_argument("--freshness-audit", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output, manifest, audit = audit_artifact(
        candidate_path=args.candidates,
        fasta_path=args.fasta,
        fai_path=args.fai,
        basic_gtf_path=args.basic_gtf,
        comprehensive_gtf_path=args.comprehensive_gtf,
        donor_table_path=args.maxent_donor_table,
        acceptor_table_path=args.maxent_acceptor_table,
        freshness_registry_path=args.freshness_registry,
        freshness_audit_path=args.freshness_audit,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output": str(output),
                "manifest": str(manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not audit["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
