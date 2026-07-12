"""Minimal GENCODE GTF parser used by the splice-pair dataset builder.

The parser deliberately keeps only the fields required by this study and does
not try to be a general GTF library.  It supports plain text and ``.gz`` files,
repeated attributes such as ``tag``, and both ``transcript_type`` and
``transcript_biotype`` naming conventions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import gzip
from pathlib import Path
from typing import Iterable, Iterator, TextIO


def _open_text(path: str | Path) -> TextIO:
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return p.open("r", encoding="utf-8")


def parse_attributes(text: str) -> dict[str, str | list[str]]:
    """Parse the ninth GTF column, preserving repeated keys as lists."""
    out: dict[str, str | list[str]] = {}
    for item in text.strip().strip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        if " " not in item:
            key, value = item, ""
        else:
            key, value = item.split(" ", 1)
            value = value.strip().strip('"')
        if key in out:
            existing = out[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                out[key] = [existing, value]
        else:
            out[key] = value
    return out


def attr_first(attrs: dict[str, str | list[str]], key: str, default: str = "") -> str:
    value = attrs.get(key, default)
    return value[0] if isinstance(value, list) else value


def attr_list(attrs: dict[str, str | list[str]], key: str) -> tuple[str, ...]:
    value = attrs.get(key)
    if value is None:
        return ()
    return tuple(value) if isinstance(value, list) else (value,)


@dataclass(frozen=True)
class ExonRecord:
    chrom: str
    start: int  # 0-based, inclusive
    end: int  # 0-based, exclusive
    strand: str
    gene_id: str
    transcript_id: str
    exon_number: int | None
    gene_type: str
    transcript_type: str
    tags: tuple[str, ...] = ()
    attributes: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TranscriptRecord:
    transcript_id: str
    gene_id: str
    chrom: str
    strand: str
    transcript_type: str
    tags: tuple[str, ...]
    exons: tuple[ExonRecord, ...]

    def ordered_exons(self) -> tuple[ExonRecord, ...]:
        ordered = sorted(self.exons, key=lambda e: (e.start, e.end))
        if self.strand == "-":
            ordered.reverse()
        return tuple(ordered)


def iter_exons(
    path: str | Path,
    *,
    transcript_types: Iterable[str] | None = ("protein_coding",),
    required_tags: Iterable[str] | None = None,
    allowed_chromosomes: set[str] | None = None,
) -> Iterator[ExonRecord]:
    allowed_types = set(transcript_types) if transcript_types is not None else None
    required = set(required_tags or ())
    with _open_text(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9:
                raise ValueError(f"{path}:{line_no}: expected 9 GTF columns, got {len(cols)}")
            chrom, _source, feature, start, end, _score, strand, _frame, attr_text = cols
            if feature != "exon":
                continue
            if allowed_chromosomes is not None and chrom not in allowed_chromosomes:
                continue
            attrs = parse_attributes(attr_text)
            transcript_type = (
                attr_first(attrs, "transcript_type")
                or attr_first(attrs, "transcript_biotype")
            )
            if allowed_types is not None and transcript_type not in allowed_types:
                continue
            tags = attr_list(attrs, "tag")
            if required and not required.issubset(tags):
                continue
            exon_number_text = attr_first(attrs, "exon_number")
            exon_number = int(exon_number_text) if exon_number_text.isdigit() else None
            yield ExonRecord(
                chrom=chrom,
                start=int(start) - 1,
                end=int(end),
                strand=strand,
                gene_id=attr_first(attrs, "gene_id"),
                transcript_id=attr_first(attrs, "transcript_id"),
                exon_number=exon_number,
                gene_type=attr_first(attrs, "gene_type") or attr_first(attrs, "gene_biotype"),
                transcript_type=transcript_type,
                tags=tags,
                attributes=attrs,
            )


def load_transcripts(
    path: str | Path,
    **kwargs,
) -> list[TranscriptRecord]:
    grouped: dict[str, list[ExonRecord]] = defaultdict(list)
    for exon in iter_exons(path, **kwargs):
        if not exon.transcript_id:
            continue
        grouped[exon.transcript_id].append(exon)

    transcripts: list[TranscriptRecord] = []
    for transcript_id, exons in grouped.items():
        first = exons[0]
        if any((e.chrom, e.strand, e.gene_id) != (first.chrom, first.strand, first.gene_id) for e in exons):
            raise ValueError(f"inconsistent transcript record: {transcript_id}")
        tags = tuple(sorted({tag for exon in exons for tag in exon.tags}))
        transcripts.append(
            TranscriptRecord(
                transcript_id=transcript_id,
                gene_id=first.gene_id,
                chrom=first.chrom,
                strand=first.strand,
                transcript_type=first.transcript_type,
                tags=tags,
                exons=tuple(exons),
            )
        )
    transcripts.sort(key=lambda t: (t.chrom, t.gene_id, t.transcript_id))
    return transcripts
