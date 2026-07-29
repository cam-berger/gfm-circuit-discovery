"""Outcome-blind construction and analysis helpers for promoter classifiers."""

from __future__ import annotations

import csv
import gzip
import math
import random
import re
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .assembly import reverse_complement

PRIMARY_CHROMS = frozenset([f"chr{index}" for index in range(1, 23)] + ["chrX", "chrY"])
_GTF_ATTRIBUTE = re.compile(r'([A-Za-z0-9_]+) "([^"]*)";')


@dataclass(frozen=True)
class GencodeTss:
    chrom: str
    tss: int
    strand: str
    gene_id: str
    transcript_id: str
    gene_name: str
    is_mane: bool = True


def iter_gencode_mane_tss(path: str | Path) -> Iterable[GencodeTss]:
    """Yield protein-coding MANE Select TSSs in zero-based GRCh38 coordinates."""

    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if (
                len(fields) < 9
                or fields[2] != "transcript"
                or fields[0] not in PRIMARY_CHROMS
                or fields[6] not in {"+", "-"}
                or 'tag "MANE_Select"' not in fields[8]
            ):
                continue
            attributes = dict(_GTF_ATTRIBUTE.findall(fields[8]))
            if attributes.get("gene_type") != "protein_coding":
                continue
            start = int(fields[3]) - 1
            end = int(fields[4])
            tss = start if fields[6] == "+" else end - 1
            yield GencodeTss(
                chrom=fields[0],
                tss=tss,
                strand=fields[6],
                gene_id=attributes["gene_id"],
                transcript_id=attributes["transcript_id"],
                gene_name=attributes.get("gene_name", ""),
                is_mane=True,
            )


def iter_gencode_protein_coding_tss(path: str | Path) -> list[GencodeTss]:
    """One protein-coding TSS per gene, preferring a non-MANE-Select transcript.

    MANE Select supply is exhausted under the block-disjointness admission rule
    used by this study's earlier cohorts (see
    ``studies/dnabert2_reconstruction_atlas/PREREGISTRATION_D13_COMPOSITION_VS_POSITION.md``),
    so this iterator deliberately prefers an alternate-promoter (non-MANE-Select)
    transcript for every gene that has one, falling back to the gene's MANE
    Select transcript only when it is that gene's sole protein-coding
    transcript. Selection within a preference tier is deterministic --
    ``Ensembl_canonical`` first, then the lexicographically smallest
    transcript ID -- so the result does not depend on GTF line order.

    Unlike :func:`iter_gencode_mane_tss` this must see every transcript of a
    gene before it can pick a winner, so it returns a materialized list
    rather than a lazy generator.
    """

    best: dict[str, tuple[tuple[int, int, str], GencodeTss]] = {}
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if (
                len(fields) < 9
                or fields[2] != "transcript"
                or fields[0] not in PRIMARY_CHROMS
                or fields[6] not in {"+", "-"}
            ):
                continue
            attributes = dict(_GTF_ATTRIBUTE.findall(fields[8]))
            if attributes.get("gene_type") != "protein_coding":
                continue
            is_mane = 'tag "MANE_Select"' in fields[8]
            is_canonical = 'tag "Ensembl_canonical"' in fields[8]
            start = int(fields[3]) - 1
            end = int(fields[4])
            tss = start if fields[6] == "+" else end - 1
            gene_id = attributes["gene_id"]
            transcript_id = attributes["transcript_id"]
            # Lower rank wins. MANE (1) always loses to non-MANE (0); within a
            # tier, a canonical tag (0) wins, then the smallest transcript ID.
            rank = (1 if is_mane else 0, 0 if is_canonical else 1, transcript_id)
            previous = best.get(gene_id)
            if previous is None or rank < previous[0]:
                best[gene_id] = (
                    rank,
                    GencodeTss(
                        chrom=fields[0],
                        tss=tss,
                        strand=fields[6],
                        gene_id=gene_id,
                        transcript_id=transcript_id,
                        gene_name=attributes.get("gene_name", ""),
                        is_mane=is_mane,
                    ),
                )
    return [feature for _, feature in best.values()]


def all_gencode_tss_by_chrom(path: str | Path) -> dict[str, list[int]]:
    """Read all basic-annotation transcript starts for conservative control exclusion."""

    output: dict[str, set[int]] = {}
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if (
                len(fields) < 9
                or fields[2] != "transcript"
                or fields[0] not in PRIMARY_CHROMS
                or fields[6] not in {"+", "-"}
            ):
                continue
            start = int(fields[3]) - 1
            end = int(fields[4])
            tss = start if fields[6] == "+" else end - 1
            output.setdefault(fields[0], set()).add(tss)
    return {chrom: sorted(values) for chrom, values in output.items()}


def has_nearby_tss(
    tss_by_chrom: Mapping[str, Sequence[int]],
    chrom: str,
    center: int,
    *,
    radius_nt: int,
) -> bool:
    values = tss_by_chrom.get(chrom, ())
    index = bisect_left(values, center - radius_nt)
    return index < len(values) and values[index] <= center + radius_nt


def oriented_tss_window(
    fasta: Any,
    *,
    chrom: str,
    tss: int,
    strand: str,
    upstream_nt: int,
    downstream_nt: int,
) -> str | None:
    """Extract an inclusive-TSS window in transcription orientation."""

    if strand not in {"+", "-"} or min(tss, upstream_nt, downstream_nt) < 0:
        raise ValueError("invalid TSS window request")
    if strand == "+":
        start, end = tss - upstream_nt, tss + downstream_nt + 1
    else:
        start, end = tss - downstream_nt, tss + upstream_nt + 1
    if start < 0 or chrom not in fasta:
        return None
    sequence = str(fasta[chrom][start:end]).upper()
    expected = upstream_nt + downstream_nt + 1
    if len(sequence) != expected or set(sequence) - set("ACGT"):
        return None
    return sequence if strand == "+" else reverse_complement(sequence)


def read_gue_sequences(paths: Sequence[str | Path]) -> dict[str, list[tuple[str, int]]]:
    """Index exact forward and reverse-complement GUE membership by sequence."""

    output: dict[str, list[tuple[str, int]]] = {}
    for path in paths:
        split = Path(path).stem
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sequence = str(row["sequence"]).upper()
                label = int(row["label"])
                output.setdefault(sequence, []).append((split, label))
                reverse = reverse_complement(sequence)
                if reverse != sequence:
                    output.setdefault(reverse, []).append((f"{split}_reverse", label))
    return output


def sequence_composition(sequence: str) -> dict[str, float]:
    length = len(sequence)
    return {
        "gc_fraction": sum(base in "GC" for base in sequence) / length,
        "cpg_fraction": (
            sum(sequence[index : index + 2] == "CG" for index in range(length - 1))
            / (length - 1)
            if length > 1
            else 0.0
        ),
    }


def context_indices(
    *,
    tss_index: int,
    target_width_nt: int,
    upstream_context_nt: int,
    downstream_context_nt: int,
) -> tuple[tuple[int, ...], tuple[int, int]]:
    """Return shuffled indices and the preserved odd-width TSS anchor."""

    if target_width_nt < 1 or target_width_nt % 2 == 0:
        raise ValueError("target_width_nt must be positive and odd")
    half = target_width_nt // 2
    anchor = (tss_index - half, tss_index + half + 1)
    indices = tuple(range(anchor[0] - upstream_context_nt, anchor[0])) + tuple(
        range(anchor[1], anchor[1] + downstream_context_nt)
    )
    if not indices or min(indices) < 0:
        raise ValueError("context window falls outside the classifier input")
    return indices, anchor


def _shuffle_candidates(
    sequence: str,
    indices: Sequence[int],
    *,
    seed: int,
    candidates: int,
) -> dict[int, str]:
    original = [sequence[index] for index in indices]
    output: dict[int, str] = {}
    rng = random.Random(seed)
    for _ in range(candidates):
        shuffled = original.copy()
        rng.shuffle(shuffled)
        changed = sum(left != right for left, right in zip(original, shuffled))
        if changed == 0 or changed in output:
            continue
        bases = list(sequence)
        for index, base in zip(indices, shuffled):
            bases[index] = base
        output[changed] = "".join(bases)
    return output


def matched_context_shuffles(
    target: str,
    control: str,
    *,
    indices: Sequence[int],
    seed: int,
    candidates: int = 512,
) -> tuple[str, str, int] | None:
    """Build target/control shuffles with exact equal Hamming dose."""

    if len(target) != len(control):
        raise ValueError("target and control classifier inputs must have equal length")
    target_candidates = _shuffle_candidates(
        target, indices, seed=seed ^ 0xA5A5A5A5, candidates=candidates
    )
    control_candidates = _shuffle_candidates(
        control, indices, seed=seed ^ 0x5A5A5A5A, candidates=candidates
    )
    common = set(target_candidates) & set(control_candidates)
    if not common:
        return None
    changed = max(common)
    return target_candidates[changed], control_candidates[changed], changed


def binary_classification_metrics(
    labels: Sequence[int], margins: Sequence[float]
) -> dict[str, Any]:
    """Binary metrics with label 1 defined as positive and margin > 0 as its call."""

    if len(labels) != len(margins) or not labels:
        raise ValueError("labels and margins must be non-empty and aligned")
    if set(labels) - {0, 1} or any(not math.isfinite(value) for value in margins):
        raise ValueError("binary metrics require 0/1 labels and finite margins")
    predictions = [int(value > 0) for value in margins]
    tp = sum(prediction == label == 1 for prediction, label in zip(predictions, labels))
    tn = sum(prediction == label == 0 for prediction, label in zip(predictions, labels))
    fp = sum(
        prediction == 1 and label == 0 for prediction, label in zip(predictions, labels)
    )
    fn = sum(
        prediction == 0 and label == 1 for prediction, label in zip(predictions, labels)
    )
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    positive = sum(labels)
    negative = len(labels) - positive
    order = sorted(range(len(margins)), key=lambda index: margins[index])
    ranks = [0.0] * len(margins)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and margins[order[end]] == margins[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    auroc = (
        sum(rank for rank, label in zip(ranks, labels) if label)
        - positive * (positive + 1) / 2
    ) / (positive * negative)
    return {
        "n": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "mcc": mcc,
        "auroc": auroc,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "mean_margin_positive": mean(
            value for value, label in zip(margins, labels) if label
        ),
        "mean_margin_negative": mean(
            value for value, label in zip(margins, labels) if not label
        ),
    }
