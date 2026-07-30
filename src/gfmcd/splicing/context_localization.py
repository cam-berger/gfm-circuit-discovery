"""Outcome-blind sequence interventions for native classifier localization.

The public helpers in this module deliberately separate string-level intervention
mechanics from model scoring.  A region is perturbed by an exact directed-dinucleotide
shuffle, selected only for delivered Hamming distance.  Model outcomes never influence
which shuffled sequence is used.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable

from gfmcd.splicing.perturb import dinucleotide_shuffle


@dataclass(frozen=True, order=True)
class RelativeWindow:
    """A half-open nucleotide window relative to a biological anchor."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("relative window must have positive width")

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def name(self) -> str:
        return f"{self.start:+d}:{self.end:+d}"

    def absolute(self, anchor: int, sequence_length: int) -> tuple[int, int]:
        start = int(anchor) + self.start
        end = int(anchor) + self.end
        if start < 0 or end > sequence_length:
            raise ValueError(
                f"window {self.name} around anchor {anchor} exceeds "
                f"sequence length {sequence_length}"
            )
        return start, end

    def halves(self) -> tuple["RelativeWindow", "RelativeWindow"]:
        if self.width % 2:
            raise ValueError("only even-width windows have equal halves")
        middle = self.start + self.width // 2
        return (
            RelativeWindow(self.start, middle),
            RelativeWindow(middle, self.end),
        )


DONOR_SCAN_WINDOWS = tuple(
    RelativeWindow(start, start + 32)
    for start in range(-112, 176, 32)
)
ACCEPTOR_SCAN_WINDOWS = tuple(
    RelativeWindow(start, start + 32)
    for start in range(-176, 48, 32)
)
CENTRAL_SCAN_WINDOW = RelativeWindow(-16, 16)
PREFLIGHT_CORE_WINDOW = RelativeWindow(-8, 8)
DONOR_PREFLIGHT_SHAM = RelativeWindow(8, 24)
ACCEPTOR_PREFLIGHT_SHAM = RelativeWindow(-24, -8)


def directed_dinucleotide_counts(sequence: str) -> Counter[str]:
    """Count adjacent ordered nucleotide pairs."""

    return Counter(sequence[index : index + 2] for index in range(len(sequence) - 1))


def stable_shuffle_seed(*parts: object) -> int:
    """Return a process-independent 64-bit seed for a frozen intervention key."""

    key = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class ShuffleAudit:
    """String-level provenance and invariant checks for one intervention."""

    relative_window: str
    absolute_start: int
    absolute_end: int
    width: int
    draw: int
    candidates: int
    candidate_index: int
    hamming: int
    hamming_fraction: float
    length_preserved: bool
    outside_preserved: bool
    endpoints_preserved: bool
    mononucleotides_preserved: bool
    dinucleotides_preserved: bool

    @property
    def invariants_pass(self) -> bool:
        return all(
            (
                self.length_preserved,
                self.outside_preserved,
                self.endpoints_preserved,
                self.mononucleotides_preserved,
                self.dinucleotides_preserved,
            )
        )

    def to_dict(self) -> dict:
        return asdict(self)


def max_hamming_dinucleotide_shuffle(
    segment: str,
    *,
    key: Iterable[object],
    draw: int,
    candidates: int = 64,
) -> tuple[str, int]:
    """Choose an outcome-blind exact shuffle with maximal delivered dose.

    Candidate generation is deterministic for ``key`` and ``draw``. Lexical sequence
    order and then candidate index make ties deterministic.
    """

    if candidates < 1:
        raise ValueError("candidates must be positive")
    keyed = tuple(key)
    generated: list[tuple[int, str, int]] = []
    for candidate_index in range(candidates):
        seed = stable_shuffle_seed(*keyed, "draw", draw, "candidate", candidate_index)
        shuffled = dinucleotide_shuffle(segment, seed=seed)
        hamming = sum(left != right for left, right in zip(segment, shuffled))
        generated.append((hamming, shuffled, candidate_index))
    max_hamming = max(row[0] for row in generated)
    best = min(
        (row for row in generated if row[0] == max_hamming),
        key=lambda row: (row[1], row[2]),
    )
    return best[1], best[2]


def shuffle_relative_window(
    sequence: str,
    *,
    anchor: int,
    window: RelativeWindow,
    key: Iterable[object],
    draw: int = 0,
    candidates: int = 64,
) -> tuple[str, ShuffleAudit]:
    """Shuffle one relative window and return the edited sequence plus audit."""

    start, end = window.absolute(anchor, len(sequence))
    before = sequence[start:end]
    after, candidate_index = max_hamming_dinucleotide_shuffle(
        before,
        key=key,
        draw=draw,
        candidates=candidates,
    )
    shuffled = sequence[:start] + after + sequence[end:]
    hamming = sum(left != right for left, right in zip(before, after))
    audit = ShuffleAudit(
        relative_window=window.name,
        absolute_start=start,
        absolute_end=end,
        width=window.width,
        draw=int(draw),
        candidates=int(candidates),
        candidate_index=int(candidate_index),
        hamming=int(hamming),
        hamming_fraction=float(hamming / window.width),
        length_preserved=len(sequence) == len(shuffled),
        outside_preserved=(
            sequence[:start] == shuffled[:start]
            and sequence[end:] == shuffled[end:]
        ),
        endpoints_preserved=(
            before[0] == after[0]
            and before[-1] == after[-1]
        ),
        mononucleotides_preserved=Counter(before) == Counter(after),
        dinucleotides_preserved=(
            directed_dinucleotide_counts(before)
            == directed_dinucleotide_counts(after)
        ),
    )
    if not audit.invariants_pass:
        raise RuntimeError(f"shuffle invariant failed: {audit}")
    return shuffled, audit


def nucleotide_token_mask(
    *,
    anchors: Iterable[int],
    window: RelativeWindow,
    sequence_length: int,
    prefix_tokens: int,
    device,
):
    """Build a `[batch, token]` mask for a relative nucleotide window."""

    import torch

    anchors_list = [int(anchor) for anchor in anchors]
    token_length = sequence_length + prefix_tokens + 1
    mask = torch.zeros(
        (len(anchors_list), token_length),
        dtype=torch.bool,
        device=device,
    )
    for row, anchor in enumerate(anchors_list):
        start, end = window.absolute(anchor, sequence_length)
        mask[row, start + prefix_tokens : end + prefix_tokens] = True
    return mask


def target_token_mask(
    *,
    nucleotide_positions: Iterable[int],
    sequence_length: int,
    prefix_tokens: int,
    device,
):
    """Build a one-token mask at each classified nucleotide coordinate."""

    import torch

    positions = [int(position) for position in nucleotide_positions]
    token_length = sequence_length + prefix_tokens + 1
    mask = torch.zeros(
        (len(positions), token_length),
        dtype=torch.bool,
        device=device,
    )
    rows = torch.arange(len(positions), device=device)
    columns = torch.as_tensor(
        positions,
        dtype=torch.long,
        device=device,
    ) + prefix_tokens
    if bool(((columns < prefix_tokens) | (columns >= token_length - 1)).any()):
        raise ValueError("target nucleotide coordinate is out of bounds")
    mask[rows, columns] = True
    return mask


__all__ = [
    "ACCEPTOR_PREFLIGHT_SHAM",
    "ACCEPTOR_SCAN_WINDOWS",
    "CENTRAL_SCAN_WINDOW",
    "DONOR_PREFLIGHT_SHAM",
    "DONOR_SCAN_WINDOWS",
    "PREFLIGHT_CORE_WINDOW",
    "RelativeWindow",
    "ShuffleAudit",
    "directed_dinucleotide_counts",
    "max_hamming_dinucleotide_shuffle",
    "nucleotide_token_mask",
    "shuffle_relative_window",
    "stable_shuffle_seed",
    "target_token_mask",
]
