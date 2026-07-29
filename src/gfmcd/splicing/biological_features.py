"""Outcome-blind sequence features and coding-preserving interventions.

This module contains no model scoring.  It turns independently supplied coding
annotations and RBNS-derived affinity matrices into auditable sequence
counterfactuals for the GTEx biological-feature study.

All sequence positions follow the repository convention: zero-based coordinates
in transcript orientation.  Synonymous variants are emitted only for complete
codons whose three bases have an unambiguous CDS slot in the supplied GTF.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from gfmcd.splicing.context_localization import RelativeWindow
from gfmcd.splicing.phase import window_codon_slot


_BASE_TO_INDEX = {base: index for index, base in enumerate("ACGT")}

# Standard nuclear genetic code in DNA alphabet.
_AA_BY_CODON = {
    codon: amino_acid
    for amino_acid, codons in {
        "F": ("TTT", "TTC"),
        "L": ("TTA", "TTG", "CTT", "CTC", "CTA", "CTG"),
        "I": ("ATT", "ATC", "ATA"),
        "M": ("ATG",),
        "V": ("GTT", "GTC", "GTA", "GTG"),
        "S": ("TCT", "TCC", "TCA", "TCG", "AGT", "AGC"),
        "P": ("CCT", "CCC", "CCA", "CCG"),
        "T": ("ACT", "ACC", "ACA", "ACG"),
        "A": ("GCT", "GCC", "GCA", "GCG"),
        "Y": ("TAT", "TAC"),
        "*": ("TAA", "TAG", "TGA"),
        "H": ("CAT", "CAC"),
        "Q": ("CAA", "CAG"),
        "N": ("AAT", "AAC"),
        "K": ("AAA", "AAG"),
        "D": ("GAT", "GAC"),
        "E": ("GAA", "GAG"),
        "C": ("TGT", "TGC"),
        "W": ("TGG",),
        "R": ("CGT", "CGC", "CGA", "CGG", "AGA", "AGG"),
        "G": ("GGT", "GGC", "GGA", "GGG"),
    }.items()
    for codon in codons
}
_SYNONYMS = {
    codon: tuple(
        sorted(
            candidate
            for candidate, candidate_amino_acid in _AA_BY_CODON.items()
            if candidate_amino_acid == amino_acid
        )
    )
    for codon, amino_acid in _AA_BY_CODON.items()
}


@dataclass(frozen=True)
class SynonymousVariant:
    """One single-codon amino-acid-preserving substitution."""

    sequence: str
    codon_start: int
    original_codon: str
    alternate_codon: str
    changed_positions: tuple[int, ...]

    @property
    def hamming(self) -> int:
        return len(self.changed_positions)

    @property
    def amino_acid(self) -> str:
        return _AA_BY_CODON[self.original_codon]


@dataclass(frozen=True)
class PsamComponent:
    """One RBPamp position-specific affinity matrix."""

    scale: float
    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != 4 or matrix.shape[0] < 1:
            raise ValueError("PSAM matrix must have shape [length, 4]")
        if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
            raise ValueError("PSAM entries must be finite and non-negative")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("PSAM scale must be finite and positive")
        object.__setattr__(self, "matrix", matrix)

    @property
    def width(self) -> int:
        return int(self.matrix.shape[0])


@dataclass(frozen=True)
class RbpPsam:
    """All independently fitted PSAM components for one RBP."""

    name: str
    components: tuple[PsamComponent, ...]
    source_path: str

    def __post_init__(self) -> None:
        if not self.name or not self.components:
            raise ValueError("RBP PSAM requires a name and at least one component")


def codon_amino_acid(codon: str) -> str:
    """Return the standard-code amino acid for an uppercase DNA codon."""

    try:
        return _AA_BY_CODON[codon]
    except KeyError as exc:
        raise ValueError(f"unsupported codon {codon!r}") from exc


def synonymous_codons(codon: str, *, include_original: bool = False) -> tuple[str, ...]:
    """Return deterministic standard-code synonyms."""

    try:
        values = _SYNONYMS[codon]
    except KeyError as exc:
        raise ValueError(f"unsupported codon {codon!r}") from exc
    if include_original:
        return values
    return tuple(value for value in values if value != codon)


def complete_codon_starts(
    example,
    window: RelativeWindow,
    cds_intervals: Mapping[str, Sequence[tuple]],
    *,
    anchor: int,
) -> tuple[int, ...]:
    """Return codon starts fully contained in ``window`` for one example."""

    start, end = window.absolute(anchor, len(example.sequence))
    slots = {
        position: window_codon_slot(example, position, cds_intervals)
        for position in range(start, end)
    }
    return tuple(
        position
        for position in range(start, end - 2)
        if (
            slots[position],
            slots[position + 1],
            slots[position + 2],
        )
        == (0, 1, 2)
    )


def synonymous_saturation_variants(
    example,
    window: RelativeWindow,
    cds_intervals: Mapping[str, Sequence[tuple]],
    *,
    anchor: int,
) -> tuple[SynonymousVariant, ...]:
    """Enumerate every single-codon synonymous variant in a coding window."""

    sequence = str(example.sequence)
    variants: list[SynonymousVariant] = []
    for start in complete_codon_starts(
        example,
        window,
        cds_intervals,
        anchor=anchor,
    ):
        original = sequence[start : start + 3]
        for alternate in synonymous_codons(original):
            changed = tuple(
                start + offset
                for offset, (left, right) in enumerate(zip(original, alternate))
                if left != right
            )
            variants.append(
                SynonymousVariant(
                    sequence=sequence[:start] + alternate + sequence[start + 3 :],
                    codon_start=start,
                    original_codon=original,
                    alternate_codon=alternate,
                    changed_positions=changed,
                )
            )
    return tuple(variants)


def _stable_tiebreak(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def maximal_synonymous_recode(
    example,
    window: RelativeWindow,
    cds_intervals: Mapping[str, Sequence[tuple]],
    *,
    anchor: int,
    draw: int = 0,
    forbid_new_splice_dinucleotides: bool = False,
) -> tuple[str, tuple[SynonymousVariant, ...]]:
    """Recode every mutable complete codon at maximum nucleotide Hamming dose."""

    sequence = str(example.sequence)
    region_start, region_end = window.absolute(anchor, len(sequence))
    clean_splice_positions = {
        position
        for position in range(region_start, region_end - 1)
        if sequence[position : position + 2] in {"AG", "GT"}
    }
    selected: list[SynonymousVariant] = []
    recoded = sequence
    for start in complete_codon_starts(
        example,
        window,
        cds_intervals,
        anchor=anchor,
    ):
        original = sequence[start : start + 3]
        alternatives = synonymous_codons(original)
        if not alternatives:
            continue
        admissible = []
        for alternate in alternatives:
            candidate = recoded[:start] + alternate + recoded[start + 3 :]
            candidate_splice_positions = {
                position
                for position in range(region_start, region_end - 1)
                if candidate[position : position + 2] in {"AG", "GT"}
            }
            if (
                forbid_new_splice_dinucleotides
                and not candidate_splice_positions.issubset(clean_splice_positions)
            ):
                continue
            admissible.append(alternate)
        if not admissible:
            continue
        max_hamming = max(
            sum(left != right for left, right in zip(original, alternate))
            for alternate in admissible
        )
        eligible = [
            alternate
            for alternate in admissible
            if sum(left != right for left, right in zip(original, alternate))
            == max_hamming
        ]
        alternate = min(
            eligible,
            key=lambda value: (
                _stable_tiebreak(
                    example.example_id,
                    window.name,
                    start,
                    draw,
                    value,
                ),
                value,
            ),
        )
        changed = tuple(
            start + offset
            for offset, (left, right) in enumerate(zip(original, alternate))
            if left != right
        )
        selected.append(
            SynonymousVariant(
                sequence="",
                codon_start=start,
                original_codon=original,
                alternate_codon=alternate,
                changed_positions=changed,
            )
        )
        recoded = recoded[:start] + alternate + recoded[start + 3 :]
    selected = [
        SynonymousVariant(
            sequence=recoded,
            codon_start=variant.codon_start,
            original_codon=variant.original_codon,
            alternate_codon=variant.alternate_codon,
            changed_positions=variant.changed_positions,
        )
        for variant in selected
    ]
    return recoded, tuple(selected)


_PSAM_HEADER = re.compile(r"^PSAM\s+A0=([0-9.eE+-]+)\s+n=(\d+)")


def parse_rbns_psam(path: str | Path) -> RbpPsam:
    """Parse the public RBPamp text format used by the SAM fixed motifs."""

    source = Path(path)
    components: list[PsamComponent] = []
    scale: float | None = None
    width: int | None = None
    rows: list[list[float]] = []

    def finish() -> None:
        nonlocal scale, width, rows
        if scale is None:
            return
        if width is None or len(rows) != width:
            raise ValueError(
                f"{source}: PSAM declared width {width} but contained {len(rows)} rows"
            )
        components.append(
            PsamComponent(scale=scale, matrix=np.asarray(rows, dtype=np.float64))
        )
        scale = None
        width = None
        rows = []

    for line in source.read_text().splitlines():
        match = _PSAM_HEADER.match(line.strip())
        if match:
            finish()
            scale = float(match.group(1))
            width = int(match.group(2))
            continue
        if scale is None or not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{source}: malformed PSAM row {line!r}")
        rows.append([float(value) for value in fields[:4]])
    finish()
    if not components:
        raise ValueError(f"{source}: no PSAM components found")
    return RbpPsam(
        name=source.stem,
        components=tuple(components),
        source_path=str(source.resolve()),
    )


def load_rbns_psam_directory(path: str | Path) -> dict[str, RbpPsam]:
    """Load a deterministic ``RBP name -> PSAM`` dictionary."""

    root = Path(path)
    models = {
        item.stem: parse_rbns_psam(item)
        for item in sorted(root.glob("*.txt"))
    }
    if not models:
        raise ValueError(f"no RBNS PSAM files found in {root}")
    return models


def psam_affinity_track(sequence: str, model: RbpPsam) -> np.ndarray:
    """Return summed component affinity at every valid sequence start."""

    if set(sequence) - set(_BASE_TO_INDEX):
        raise ValueError("PSAM scoring requires an uppercase A/C/G/T sequence")
    track = np.zeros(len(sequence), dtype=np.float64)
    for component in model.components:
        if component.width > len(sequence):
            continue
        for start in range(len(sequence) - component.width + 1):
            indices = [
                _BASE_TO_INDEX[base]
                for base in sequence[start : start + component.width]
            ]
            affinity = component.scale * float(
                np.prod(component.matrix[np.arange(component.width), indices])
            )
            track[start] += affinity
    return track


def psam_region_score(
    sequence: str,
    model: RbpPsam,
    *,
    start: int,
    end: int,
) -> float:
    """Sum affinity for components wholly contained in a half-open region."""

    if not 0 <= start < end <= len(sequence):
        raise ValueError("invalid PSAM scoring region")
    score = 0.0
    for component in model.components:
        for offset in range(start, end - component.width + 1):
            indices = [
                _BASE_TO_INDEX[base]
                for base in sequence[offset : offset + component.width]
            ]
            score += component.scale * float(
                np.prod(component.matrix[np.arange(component.width), indices])
            )
    return float(score)


def psam_score_vector(
    sequence: str,
    models: Mapping[str, RbpPsam],
    *,
    start: int,
    end: int,
) -> dict[str, float]:
    """Score a region under every named external RBP affinity model."""

    return {
        name: psam_region_score(
            sequence,
            model,
            start=start,
            end=end,
        )
        for name, model in sorted(models.items())
    }


def feature_delta(
    clean: Mapping[str, float],
    changed: Mapping[str, float],
) -> dict[str, float]:
    """Return ``clean - changed`` for an exactly matched feature family."""

    if set(clean) != set(changed):
        raise ValueError("feature mappings must have identical keys")
    return {name: float(clean[name] - changed[name]) for name in sorted(clean)}


def beam_synonymous_optimize(
    example,
    window: RelativeWindow,
    cds_intervals: Mapping[str, Sequence[tuple]],
    *,
    anchor: int,
    objective: Callable[[str], float],
    maximize: bool,
    max_changed_bases: int = 4,
    beam_size: int = 256,
) -> tuple[str, dict[str, object]]:
    """Optimize a sequence-only objective over synonymous codon choices.

    The beam and all tie-breaks are outcome-blind.  The clean sequence remains
    eligible, which makes a reported zero delivered dose an explicit inability
    to construct the requested counterfactual rather than a silent edit.
    """

    if max_changed_bases < 1 or beam_size < 2:
        raise ValueError("max_changed_bases and beam_size must be positive")
    clean = str(example.sequence)
    starts = complete_codon_starts(
        example,
        window,
        cds_intervals,
        anchor=anchor,
    )
    beam: dict[str, tuple[int, tuple[tuple[int, str, str], ...]]] = {
        clean: (0, ())
    }
    for start in starts:
        original = clean[start : start + 3]
        choices = synonymous_codons(original, include_original=True)
        expanded: dict[str, tuple[int, tuple[tuple[int, str, str], ...]]] = {}
        for sequence, (dose, changes) in beam.items():
            for alternate in choices:
                changed = sum(
                    left != right for left, right in zip(original, alternate)
                )
                next_dose = dose + changed
                if next_dose > max_changed_bases:
                    continue
                candidate = sequence[:start] + alternate + sequence[start + 3 :]
                candidate_changes = changes
                if alternate != original:
                    candidate_changes = changes + ((start, original, alternate),)
                previous = expanded.get(candidate)
                if previous is None or next_dose < previous[0]:
                    expanded[candidate] = (next_dose, candidate_changes)
        ranked = sorted(
            expanded.items(),
            key=lambda item: (
                -objective(item[0]) if maximize else objective(item[0]),
                item[1][0],
                _stable_tiebreak(example.example_id, window.name, item[0]),
            ),
        )
        beam = dict(ranked[:beam_size])
    if not beam:
        raise RuntimeError("synonymous beam search produced no candidates")
    winner, (dose, changes) = min(
        beam.items(),
        key=lambda item: (
            -objective(item[0]) if maximize else objective(item[0]),
            item[1][0],
            _stable_tiebreak(example.example_id, window.name, item[0]),
        ),
    )
    return winner, {
        "objective_clean": float(objective(clean)),
        "objective_changed": float(objective(winner)),
        "changed_bases": int(dose),
        "codon_changes": [
            {
                "codon_start": int(start),
                "original": original,
                "alternate": alternate,
            }
            for start, original, alternate in changes
        ],
        "beam_size": int(beam_size),
        "max_changed_bases": int(max_changed_bases),
    }


def vienna_mfe(sequence: str) -> float:
    """Return ViennaRNA minimum free energy, if the optional package is present."""

    try:
        import RNA  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional scientific dependency
        raise ImportError(
            "ViennaRNA is required for the preregistered structure arm"
        ) from exc
    _structure, energy = RNA.fold_compound(sequence.replace("T", "U")).mfe()
    return float(energy)


def vienna_unpaired_mean(
    sequence: str,
    *,
    start: int,
    end: int,
    unpaired_length: int = 7,
    window_size: int = 150,
    max_bp_span: int = 100,
) -> float:
    """Mean RNAplfold probability for unpaired segments beginning in a region.

    ViennaRNA returns a 1-based matrix where ``matrix[i][u]`` is the
    probability that the length-``u`` segment beginning at nucleotide ``i`` is
    unpaired.  Only starts whose entire segment lies in ``[start, end)`` enter
    the mean.
    """

    try:
        import RNA  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional scientific dependency
        raise ImportError(
            "ViennaRNA is required for the preregistered structure arm"
        ) from exc
    if (
        set(sequence) - set("ACGTU")
        or not 0 <= start < end <= len(sequence)
        or unpaired_length < 1
        or end - start < unpaired_length
        or window_size < unpaired_length
        or max_bp_span < unpaired_length
    ):
        raise ValueError("invalid sequence, region, or RNAplfold settings")
    probabilities = RNA.pfl_fold_up(
        sequence.replace("T", "U"),
        int(unpaired_length),
        int(window_size),
        int(max_bp_span),
    )
    values = [
        float(probabilities[position + 1][unpaired_length])
        for position in range(start, end - unpaired_length + 1)
    ]
    if not values or not np.isfinite(values).all():
        raise RuntimeError("RNAplfold returned no finite target-region probabilities")
    return float(np.mean(values))


__all__ = [
    "PsamComponent",
    "RbpPsam",
    "SynonymousVariant",
    "beam_synonymous_optimize",
    "codon_amino_acid",
    "complete_codon_starts",
    "feature_delta",
    "load_rbns_psam_directory",
    "maximal_synonymous_recode",
    "parse_rbns_psam",
    "psam_affinity_track",
    "psam_region_score",
    "psam_score_vector",
    "synonymous_codons",
    "synonymous_saturation_variants",
    "vienna_mfe",
]
