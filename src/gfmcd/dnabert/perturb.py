"""Sequence perturbations for DNABERT promoter experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
import random

from gfmcd.adapters.dnabert import normalize_sequence


DNA_ALPHABET = frozenset("ACGT")


def nucleotide_counts(sequence: str) -> Counter[str]:
    """Return nucleotide counts for a canonical DNA sequence."""

    sequence = normalize_sequence(sequence)
    return Counter(sequence)


def dinucleotide_counts(sequence: str) -> Counter[str]:
    """Return overlapping dinucleotide counts."""

    sequence = normalize_sequence(sequence)

    return Counter(
        sequence[index : index + 2]
        for index in range(len(sequence) - 1)
    )


def validate_canonical_dna(sequence: str) -> str:
    """Normalize and verify that a sequence contains only A/C/G/T."""

    sequence = normalize_sequence(sequence)

    invalid = sorted(set(sequence) - DNA_ALPHABET)

    if invalid:
        raise ValueError(
            f"sequence contains non-canonical bases: {invalid}"
        )

    if len(sequence) < 2:
        raise ValueError(
            "sequence must contain at least two nucleotides"
        )

    return sequence


def mononucleotide_shuffle(
    sequence: str,
    *,
    seed: int,
) -> str:
    """Shuffle nucleotide order while preserving exact base counts."""

    sequence = validate_canonical_dna(sequence)

    rng = random.Random(seed)
    characters = list(sequence)
    rng.shuffle(characters)

    shuffled = "".join(characters)

    if nucleotide_counts(shuffled) != nucleotide_counts(sequence):
        raise RuntimeError(
            "mononucleotide shuffle failed to preserve base counts"
        )

    return shuffled


def _randomized_eulerian_path(
    sequence: str,
    *,
    seed: int,
) -> str:
    """Construct a randomized Eulerian path over dinucleotide edges."""

    rng = random.Random(seed)

    adjacency: dict[str, list[str]] = defaultdict(list)

    for left, right in zip(sequence[:-1], sequence[1:]):
        adjacency[left].append(right)

    for neighbors in adjacency.values():
        rng.shuffle(neighbors)

    stack = [sequence[0]]
    path: list[str] = []

    while stack:
        node = stack[-1]

        if adjacency[node]:
            stack.append(adjacency[node].pop())
        else:
            path.append(stack.pop())

    path.reverse()

    if len(path) != len(sequence):
        raise RuntimeError(
            "Eulerian reconstruction produced incorrect length"
        )

    return "".join(path)


def dinucleotide_shuffle(
    sequence: str,
    *,
    seed: int,
    max_attempts: int = 100,
    require_change: bool = True,
) -> str:
    """Shuffle while preserving every overlapping dinucleotide count."""

    sequence = validate_canonical_dna(sequence)

    for attempt in range(max_attempts):
        shuffled = _randomized_eulerian_path(
            sequence,
            seed=seed + attempt,
        )

        if (
            dinucleotide_counts(shuffled)
            != dinucleotide_counts(sequence)
        ):
            raise RuntimeError(
                "dinucleotide shuffle failed exact preservation"
            )

        if (
            nucleotide_counts(shuffled)
            != nucleotide_counts(sequence)
        ):
            raise RuntimeError(
                "dinucleotide shuffle changed nucleotide counts"
            )

        if not require_change or shuffled != sequence:
            return shuffled

    raise RuntimeError(
        "could not generate a changed dinucleotide-preserving "
        f"shuffle after {max_attempts} attempts"
    )


def hamming_distance(left: str, right: str) -> int:
    """Return the number of differing nucleotide positions."""

    left = validate_canonical_dna(left)
    right = validate_canonical_dna(right)

    if len(left) != len(right):
        raise ValueError("sequences must have equal length")

    return sum(a != b for a, b in zip(left, right))


def changed_fraction(left: str, right: str) -> float:
    """Fraction of nucleotide positions that differ."""

    return hamming_distance(left, right) / len(left)
