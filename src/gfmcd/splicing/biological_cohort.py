"""Outcome-blind cohort construction for biological interpretation assays.

The previous localization windows were target-relative.  This module is the
annotation firewall that must run before those windows may be called coding or
exonic.  It also excludes every gene that has appeared in an earlier GTEx
fine-tuning result, using only recorded files and annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable, Sequence

from gfmcd.provenance import sha256_file
from gfmcd.splicing.circuit import _dataset, eligible
from gfmcd.splicing.context_localization import RelativeWindow
from gfmcd.splicing.coupling import CouplingConfig, CouplingExample, stable_rng
from gfmcd.splicing.native_classifier import gene_hash_partition
from gfmcd.splicing.p3_annotation import (
    AnnotationSafetyIndex,
    validate_annotation_block,
    window_block_to_genomic,
)
from gfmcd.splicing.phase import window_codon_slot


GENE_PATTERN = re.compile(rb"ENSG[0-9]+(?:\.[0-9]+)?")
FORBIDDEN_CHROMOSOMES = frozenset(
    {"chr8", "chr9", "chr11", "chr16", "chr19", "chrX", "chrY", "chrM", "MT"}
)
NAMESPACE = "gtex-native-biological-features-v1"
DONOR_WINDOW = RelativeWindow(-80, -48)
ACCEPTOR_WINDOW = RelativeWindow(16, 48)


@dataclass(frozen=True)
class PriorGeneExclusion:
    gene_ids: tuple[str, ...]
    source_files: tuple[dict[str, object], ...]
    aggregate_sha256: str


@dataclass(frozen=True)
class CodingWindowAudit:
    endpoint: str
    relative_window: str
    absolute_start: int
    absolute_end: int
    genomic_start: int
    genomic_end: int
    codon_slots: tuple[int, ...]
    complete_codon_starts: tuple[int, ...]
    annotation_checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return (
            all(self.annotation_checks.values())
            and len(self.complete_codon_starts) == 10
            and set(self.codon_slots) == {0, 1, 2}
        )


def collect_prior_gene_exclusion(root: str | Path) -> PriorGeneExclusion:
    """Collect every Ensembl gene identifier appearing in an earlier JSON result."""

    directory = Path(root)
    genes: set[str] = set()
    sources: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*.json")):
        found: set[str] = set()
        with path.open("rb") as handle:
            for line in handle:
                found.update(match.decode("ascii") for match in GENE_PATTERN.findall(line))
        if found:
            genes.update(found)
            sources.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "n_gene_ids": len(found),
                }
            )
    payload = "\n".join(
        f"{row['path']}\t{row['sha256']}\t{row['n_gene_ids']}" for row in sources
    )
    aggregate = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return PriorGeneExclusion(
        gene_ids=tuple(sorted(genes)),
        source_files=tuple(sources),
        aggregate_sha256=aggregate,
    )


def _complete_starts_from_slots(start: int, end: int, slots: Sequence[int | None]) -> tuple[int, ...]:
    return tuple(
        start + offset
        for offset in range(0, len(slots) - 2)
        if tuple(slots[offset : offset + 3]) == (0, 1, 2)
        and start + offset + 3 <= end
    )


def audit_coding_window(
    example: CouplingExample,
    *,
    endpoint: str,
    window: RelativeWindow,
    anchor: int,
    annotation_index: AnnotationSafetyIndex,
    cds_intervals: dict[str, list[tuple]],
) -> CodingWindowAudit:
    """Require one comprehensive-annotation CDS register across a whole window."""

    start, end = window.absolute(anchor, len(example.sequence))
    metadata = example.metadata
    genomic_start, genomic_end = window_block_to_genomic(
        start,
        end,
        window_start=int(metadata["genomic_window_start"]),
        window_end=int(metadata["genomic_window_end"]),
        strand=example.strand,
    )
    gate = validate_annotation_block(
        annotation_index,
        chrom=example.chrom,
        start=genomic_start,
        end=genomic_end,
        strand=example.strand,
        gene_id=example.gene_id,
        context="cds",
        splice_clearance=0,
    )
    slots = tuple(
        window_codon_slot(example, position, cds_intervals)
        for position in range(start, end)
    )
    complete = _complete_starts_from_slots(start, end, slots)
    return CodingWindowAudit(
        endpoint=endpoint,
        relative_window=window.name,
        absolute_start=start,
        absolute_end=end,
        genomic_start=genomic_start,
        genomic_end=genomic_end,
        codon_slots=tuple(-1 if slot is None else int(slot) for slot in slots),
        complete_codon_starts=complete,
        annotation_checks={name: bool(value) for name, value in gate.checks.items()},
    )


def load_fresh_cds_cohort(
    basic_gtf: str,
    fasta: str,
    *,
    annotation_index: AnnotationSafetyIndex,
    cds_intervals: dict[str, list[tuple]],
    analysis_partition: str,
    n: int,
    tag: str,
    excluded_gene_ids: Iterable[str],
    additional_excluded_gene_ids: Iterable[str] = (),
    discovery_percent: int = 50,
) -> tuple[list[CouplingExample], dict[str, dict[str, CodingWindowAudit]], dict[str, int]]:
    """Filter annotation first, then deterministically select one example per gene."""

    if analysis_partition not in {"discovery", "confirmation"}:
        raise ValueError("analysis_partition must be discovery or confirmation")
    excluded = set(excluded_gene_ids) | set(additional_excluded_gene_ids)
    candidates = []
    audits: dict[str, dict[str, CodingWindowAudit]] = {}
    counts = {
        "eligible_train_before_exclusion": 0,
        "after_prior_and_chromosome_exclusion": 0,
        "after_partition": 0,
        "both_windows_cds": 0,
        "unique_genes_cds": 0,
    }
    for example in eligible(_dataset(basic_gtf, fasta), "train"):
        counts["eligible_train_before_exclusion"] += 1
        if example.gene_id in excluded or example.chrom in FORBIDDEN_CHROMOSOMES:
            continue
        counts["after_prior_and_chromosome_exclusion"] += 1
        if (
            gene_hash_partition(
                example.gene_id,
                namespace=NAMESPACE,
                discovery_percent=discovery_percent,
            )
            != analysis_partition
        ):
            continue
        counts["after_partition"] += 1
        donor = audit_coding_window(
            example,
            endpoint="donor",
            window=DONOR_WINDOW,
            anchor=int(example.donor_pos),
            annotation_index=annotation_index,
            cds_intervals=cds_intervals,
        )
        acceptor = audit_coding_window(
            example,
            endpoint="acceptor",
            window=ACCEPTOR_WINDOW,
            anchor=int(example.next_acceptor_pos),
            annotation_index=annotation_index,
            cds_intervals=cds_intervals,
        )
        if not donor.passed or not acceptor.passed:
            continue
        counts["both_windows_cds"] += 1
        audits[example.example_id] = {"donor": donor, "acceptor": acceptor}
        candidates.append(example)

    stable_rng(tag, CouplingConfig().seed).shuffle(candidates)
    selected = []
    seen_genes: set[str] = set()
    for example in candidates:
        if example.gene_id in seen_genes:
            continue
        seen_genes.add(example.gene_id)
        selected.append(example)
        if len(selected) == n:
            break
    counts["unique_genes_cds"] = len({example.gene_id for example in candidates})
    if len(selected) != n:
        raise RuntimeError(
            f"only {len(selected)} unique CDS-qualified {analysis_partition} genes; asked for {n}"
        )
    return selected, {example.example_id: audits[example.example_id] for example in selected}, counts


__all__ = [
    "ACCEPTOR_WINDOW",
    "CodingWindowAudit",
    "DONOR_WINDOW",
    "FORBIDDEN_CHROMOSOMES",
    "NAMESPACE",
    "PriorGeneExclusion",
    "audit_coding_window",
    "collect_prior_gene_exclusion",
    "load_fresh_cds_cohort",
]
