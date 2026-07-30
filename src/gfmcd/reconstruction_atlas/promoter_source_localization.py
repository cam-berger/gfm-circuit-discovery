"""Outcome-blind geometry and biological annotations for promoter localization."""

from __future__ import annotations

import gzip
import hashlib
import html
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenomicInterval:
    chrom: str
    start: int
    end: int
    name: str
    strand: str | None = None


class BinnedIntervalIndex:
    """Small deterministic interval index suitable for fixed genomic annotations."""

    def __init__(
        self, intervals: Iterable[GenomicInterval], *, bin_size: int = 10_000
    ) -> None:
        if bin_size < 1:
            raise ValueError("bin_size must be positive")
        self.bin_size = int(bin_size)
        bins: dict[tuple[str, int], list[GenomicInterval]] = {}
        for interval in intervals:
            if interval.start < 0 or interval.end <= interval.start:
                raise ValueError(f"invalid genomic interval: {interval}")
            first = interval.start // self.bin_size
            last = (interval.end - 1) // self.bin_size
            for bin_index in range(first, last + 1):
                bins.setdefault((interval.chrom, bin_index), []).append(interval)
        self._bins = {
            key: tuple(
                sorted(
                    {value for value in values},
                    key=lambda item: (
                        item.start,
                        item.end,
                        item.name,
                        item.strand or "",
                    ),
                )
            )
            for key, values in bins.items()
        }

    def overlapping(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        strand: str | None = None,
    ) -> tuple[GenomicInterval, ...]:
        if start < 0 or end <= start:
            raise ValueError("query interval must be nonempty and nonnegative")
        found: set[GenomicInterval] = set()
        for bin_index in range(start // self.bin_size, (end - 1) // self.bin_size + 1):
            for interval in self._bins.get((chrom, bin_index), ()):
                if (
                    interval.start < end
                    and start < interval.end
                    and (
                        strand is None
                        or interval.strand is None
                        or interval.strand == strand
                    )
                ):
                    found.add(interval)
        return tuple(
            sorted(
                found,
                key=lambda item: (
                    item.start,
                    item.end,
                    item.name,
                    item.strand or "",
                ),
            )
        )


@dataclass(frozen=True)
class EpdMotif:
    name: str
    family: str
    position_offsets: tuple[int, ...]
    weights: Mapping[str, tuple[float, ...]]
    preferred_position0_min: int
    preferred_position0_max: int
    cutoff: float
    source_sha256: str

    def __post_init__(self) -> None:
        length = len(self.position_offsets)
        if (
            length < 2
            or 0 not in self.position_offsets
            or set(self.weights) != set("ACGT")
            or any(len(values) != length for values in self.weights.values())
            or self.preferred_position0_min > self.preferred_position0_max
        ):
            raise ValueError(f"malformed EPD motif: {self.name}")

    @property
    def length(self) -> int:
        return len(self.position_offsets)

    @property
    def zero_index(self) -> int:
        return self.position_offsets.index(0)

    def score(self, sequence: str) -> float:
        sequence = sequence.upper()
        if len(sequence) != self.length or set(sequence) - set("ACGT"):
            raise ValueError(f"{self.name} requires {self.length} ACGT bases")
        return sum(self.weights[base][index] for index, base in enumerate(sequence))


@dataclass(frozen=True)
class MotifInstance:
    motif: str
    family: str
    start: int
    end: int
    position0_relative_tss: int
    score: float
    cutoff: float
    sequence: str


@dataclass(frozen=True)
class TokenTile:
    nominal_start: int
    nominal_end: int
    token_indices: tuple[int, ...]
    cover_start: int
    cover_end: int
    overlap_nt: int

    @property
    def cover_length(self) -> int:
        return self.cover_end - self.cover_start


def iter_fantom5_dominant_ctss(path: str | Path) -> Iterable[GenomicInterval]:
    """Yield the strand-specific dominant CTSS base from FANTOM robust peaks."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8 or fields[5] not in {"+", "-"}:
                continue
            start = int(fields[6])
            end = int(fields[7])
            if end <= start:
                continue
            yield GenomicInterval(
                chrom=fields[0],
                start=start,
                end=end,
                name=fields[3],
                strand=fields[5],
            )


def iter_encode_pls(path: str | Path) -> Iterable[GenomicInterval]:
    """Yield promoter-like cCRE intervals from the full ENCODE Registry BED."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 10 and fields[9] == "PLS":
                yield GenomicInterval(
                    chrom=fields[0],
                    start=int(fields[1]),
                    end=int(fields[2]),
                    name=fields[3],
                )


def iter_ucsc_cpg_islands(path: str | Path) -> Iterable[GenomicInterval]:
    """Yield UCSC hg38 CpG-island intervals from the database table dump."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            yield GenomicInterval(
                chrom=fields[1],
                start=int(fields[2]),
                end=int(fields[3]),
                name=fields[4] if len(fields) > 4 else "CpG_island",
            )


def parse_epd_motif(
    path: str | Path,
    *,
    name: str,
    family: str,
) -> EpdMotif:
    """Parse an archived EPD promoter-element log-weight matrix and cutoff."""

    text = Path(path).read_text(encoding="utf-8")
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.IGNORECASE | re.DOTALL)
    weights: dict[str, tuple[float, ...]] = {}
    position_values: list[int] = []
    for row in rows:
        cells = [
            html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()
            for cell in re.findall(
                r"<td[^>]*>(.*?)</td>",
                row,
                flags=re.IGNORECASE | re.DOTALL,
            )
        ]
        cells = [re.sub(r"\s+", " ", cell) for cell in cells if cell.strip()]
        if not cells:
            continue
        label = cells[0].upper().replace(" ", "")
        weight_match = re.fullmatch(r"W\(([ACGT])\)", label)
        if weight_match:
            weights[weight_match.group(1)] = tuple(float(value) for value in cells[1:])
            continue
        if label == "POSITION":
            position_values = [int(value) for value in cells[1:]]
        elif not position_values and all(
            re.fullmatch(r"[+-]?\d+", value) for value in cells
        ):
            position_values = [int(value) for value in cells]
    if set(weights) != set("ACGT"):
        raise ValueError(f"could not parse all weight rows from {path}")
    lengths = {len(values) for values in weights.values()}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent EPD matrix row lengths in {path}")
    length = next(iter(lengths))
    if len(position_values) < length:
        raise ValueError(f"could not parse EPD position row from {path}")
    positions = tuple(position_values[:length])
    preferred = re.search(
        r"Preferred region:.*?between\s+([+-]?\d+)\s+and\s+([+-]?\d+)",
        re.sub(r"<[^>]+>", " ", text),
        flags=re.IGNORECASE | re.DOTALL,
    )
    cutoff = re.search(
        r"cut-off value:\s*([+-]?\d+(?:\.\d+)?)",
        re.sub(r"<[^>]+>", " ", text),
        flags=re.IGNORECASE,
    )
    if preferred is None or cutoff is None:
        raise ValueError(f"could not parse EPD preferred region/cutoff from {path}")
    return EpdMotif(
        name=name,
        family=family,
        position_offsets=positions,
        weights=weights,
        preferred_position0_min=int(preferred.group(1)),
        preferred_position0_max=int(preferred.group(2)),
        cutoff=float(cutoff.group(1)),
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )


def scan_epd_motif(
    sequence: str, motif: EpdMotif, *, tss_index: int
) -> tuple[MotifInstance, ...]:
    """Call all above-cutoff motif instances in the published preferred region."""

    sequence = sequence.upper()
    instances = []
    for position0 in range(
        motif.preferred_position0_min, motif.preferred_position0_max + 1
    ):
        start = tss_index + position0 - motif.zero_index
        end = start + motif.length
        if start < 0 or end > len(sequence):
            continue
        subsequence = sequence[start:end]
        score = motif.score(subsequence)
        if score >= motif.cutoff:
            instances.append(
                MotifInstance(
                    motif=motif.name,
                    family=motif.family,
                    start=start,
                    end=end,
                    position0_relative_tss=position0,
                    score=score,
                    cutoff=motif.cutoff,
                    sequence=subsequence,
                )
            )
    return tuple(
        sorted(
            instances,
            key=lambda item: (-item.score, item.start, item.end, item.sequence),
        )
    )


def oriented_interval_to_genome(
    *,
    tss: int,
    strand: str,
    start: int,
    end: int,
    tss_index: int,
) -> tuple[int, int]:
    """Map a half-open transcription-oriented sequence interval to GRCh38."""

    if start < 0 or end <= start or strand not in {"+", "-"}:
        raise ValueError("invalid oriented sequence interval")
    if strand == "+":
        return tss - tss_index + start, tss - tss_index + end
    return (
        tss + tss_index - end + 1,
        tss + tss_index - start + 1,
    )


def fixed_dose_tile(
    offsets: Sequence[Sequence[int]],
    *,
    nominal_start: int,
    nominal_end: int,
    token_dose: int = 2,
) -> TokenTile | None:
    """Choose adjacent genomic BPE tokens maximizing overlap with one nominal tile."""

    if token_dose < 1 or nominal_start < 0 or nominal_end <= nominal_start:
        raise ValueError("invalid tile geometry")
    genomic = [
        (index, int(start), int(end))
        for index, (start, end) in enumerate(offsets)
        if int(end) > int(start)
    ]
    if len(genomic) < token_dose:
        return None
    candidates: list[tuple[tuple[float, ...], TokenTile]] = []
    tile_center = (nominal_start + nominal_end) / 2
    for begin in range(len(genomic) - token_dose + 1):
        group = genomic[begin : begin + token_dose]
        if any(
            group[index + 1][0] != group[index][0] + 1
            for index in range(token_dose - 1)
        ):
            continue
        cover_start = min(item[1] for item in group)
        cover_end = max(item[2] for item in group)
        overlap = max(0, min(cover_end, nominal_end) - max(cover_start, nominal_start))
        cover_center = (cover_start + cover_end) / 2
        tile = TokenTile(
            nominal_start=nominal_start,
            nominal_end=nominal_end,
            token_indices=tuple(item[0] for item in group),
            cover_start=cover_start,
            cover_end=cover_end,
            overlap_nt=overlap,
        )
        rank = (
            -float(overlap),
            abs(cover_center - tile_center),
            float(tile.cover_length),
            float(cover_start),
        )
        candidates.append((rank, tile))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def paired_fixed_dose_tile(
    target_offsets: Sequence[Sequence[int]],
    control_offsets: Sequence[Sequence[int]],
    *,
    nominal_start: int,
    nominal_end: int,
    token_dose: int,
    maximum_cover_nt: int,
    maximum_cover_length_difference_nt: int,
) -> tuple[TokenTile, TokenTile] | None:
    target = fixed_dose_tile(
        target_offsets,
        nominal_start=nominal_start,
        nominal_end=nominal_end,
        token_dose=token_dose,
    )
    control = fixed_dose_tile(
        control_offsets,
        nominal_start=nominal_start,
        nominal_end=nominal_end,
        token_dose=token_dose,
    )
    if (
        target is None
        or control is None
        or target.cover_length > maximum_cover_nt
        or control.cover_length > maximum_cover_nt
        or abs(target.cover_length - control.cover_length)
        > maximum_cover_length_difference_nt
    ):
        return None
    return target, control


def composition_preserving_paired_disruption(
    target: str,
    control: str,
    *,
    start: int,
    end: int,
    motif: EpdMotif,
    tss_index: int,
    seed: int,
    geometry_qualified_batch: Callable[[Sequence[tuple[str, str]]], Sequence[bool]],
    candidates: int = 4096,
    maximum_geometry_candidates: int = 1024,
) -> dict[str, Any] | None:
    """Find equal-Hamming local shuffles that remove a target motif call."""

    if (
        len(target) != len(control)
        or end - start != motif.length
        or target[start:end] == control[start:end] == ""
    ):
        raise ValueError("invalid paired motif-disruption request")
    original_target_calls = scan_epd_motif(target, motif, tss_index=tss_index)
    selected_calls = [
        call
        for call in original_target_calls
        if call.start == start and call.end == end
    ]
    if not selected_calls:
        return None
    if scan_epd_motif(control, motif, tss_index=tss_index):
        return None
    target_candidates = _local_shuffles(
        target, start=start, end=end, seed=seed ^ 0xA5A5, candidates=candidates
    )
    control_candidates = _local_shuffles(
        control, start=start, end=end, seed=seed ^ 0x5A5A, candidates=candidates
    )
    common_dose = sorted(set(target_candidates) & set(control_candidates), reverse=True)
    for dose in common_dose:
        ranked_target = sorted(
            target_candidates[dose],
            key=lambda value: (
                max(
                    (
                        call.score
                        for call in scan_epd_motif(value, motif, tss_index=tss_index)
                    ),
                    default=float("-inf"),
                ),
                value,
            ),
        )
        ranked_target = [
            value
            for value in ranked_target
            if not scan_epd_motif(value, motif, tss_index=tss_index)
        ]
        ranked_control = [
            value
            for value in sorted(control_candidates[dose])
            if not scan_epd_motif(value, motif, tss_index=tss_index)
        ]
        paired_candidates = [
            (target_disrupted, control_disrupted)
            for target_disrupted in ranked_target
            for control_disrupted in ranked_control
        ][:maximum_geometry_candidates]
        if not paired_candidates:
            continue
        geometry = list(geometry_qualified_batch(paired_candidates))
        if len(geometry) != len(paired_candidates):
            raise ValueError("geometry batch result does not align with candidates")
        for (target_disrupted, control_disrupted), qualified in zip(
            paired_candidates, geometry
        ):
            if qualified:
                return {
                    "target_sequence": target_disrupted,
                    "control_sequence": control_disrupted,
                    "changed_nucleotides_each": dose,
                    "start": start,
                    "end": end,
                    "target_original_subsequence": target[start:end],
                    "target_disrupted_subsequence": target_disrupted[start:end],
                    "control_original_subsequence": control[start:end],
                    "control_disrupted_subsequence": control_disrupted[start:end],
                    "target_original_score": selected_calls[0].score,
                    "target_disrupted_max_score": max(
                        (
                            call.score
                            for call in scan_epd_motif(
                                target_disrupted, motif, tss_index=tss_index
                            )
                        ),
                        default=None,
                    ),
                    "composition_preserved": (
                        Counter(target[start:end])
                        == Counter(target_disrupted[start:end])
                        and Counter(control[start:end])
                        == Counter(control_disrupted[start:end])
                    ),
                }
    return None


def _local_shuffles(
    sequence: str,
    *,
    start: int,
    end: int,
    seed: int,
    candidates: int,
) -> dict[int, set[str]]:
    original = list(sequence[start:end])
    rng = random.Random(seed)
    output: dict[int, set[str]] = {}
    for _ in range(candidates):
        shuffled = original.copy()
        rng.shuffle(shuffled)
        dose = sum(left != right for left, right in zip(original, shuffled))
        if dose < 1:
            continue
        value = sequence[:start] + "".join(shuffled) + sequence[end:]
        output.setdefault(dose, set()).add(value)
    return output
