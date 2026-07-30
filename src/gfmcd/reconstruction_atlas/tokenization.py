"""Nucleotide-to-BPE alignment and geometry-qualified masking."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from .schema import DNA_ALPHABET, Interval

Encoder = Callable[[str], tuple[list[int], list[tuple[int, int]]]]


@dataclass(frozen=True)
class MaskPlan:
    """The minimal consecutive BPE tokens whose offsets cover a biological site."""

    site: Interval
    token_indices: tuple[int, ...]
    true_token_ids: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]
    covered: Interval
    sequence_length: int

    def __post_init__(self) -> None:
        if not self.token_indices:
            raise ValueError("mask plan contains no target tokens")
        if len(self.token_indices) != len(self.true_token_ids):
            raise ValueError("token indices and true IDs differ in length")
        if len(self.token_indices) != len(self.token_offsets):
            raise ValueError("token indices and offsets differ in length")
        if (
            tuple(range(self.token_indices[0], self.token_indices[-1] + 1))
            != self.token_indices
        ):
            raise ValueError("target BPE tokens must be consecutive")
        if not self.covered.contains(self.site):
            raise ValueError("token cover does not contain the biological site")
        if self.covered.end > self.sequence_length:
            raise ValueError("token cover falls outside the sequence")

    @property
    def covered_nucleotides(self) -> int:
        return self.covered.length

    @property
    def site_nucleotides(self) -> int:
        return self.site.length

    @property
    def overhang_nucleotides(self) -> int:
        return self.covered.length - self.site.length

    @property
    def geometry_signature(self) -> tuple[tuple[int, int], ...]:
        """Token offsets relative to the biological site, including BPE overhang."""

        return tuple(
            (start - self.site.start, end - self.site.start)
            for start, end in self.token_offsets
        )

    def masked_input_ids(
        self, input_ids: Sequence[int], mask_token_id: int
    ) -> list[int]:
        if len(input_ids) == 0:
            raise ValueError("input IDs cannot be empty")
        masked = [int(x) for x in input_ids]
        for index, true_id in zip(self.token_indices, self.true_token_ids):
            if masked[index] != true_id:
                raise ValueError(
                    "input IDs drifted from the tokenization used to build the mask"
                )
            masked[index] = int(mask_token_id)
        if any(masked[index] != int(mask_token_id) for index in self.token_indices):
            raise RuntimeError("failed to mask every target token")
        return masked

    def to_dict(self) -> dict[str, object]:
        return {
            "site_start": self.site.start,
            "site_end": self.site.end,
            "token_indices": list(self.token_indices),
            "true_token_ids": list(self.true_token_ids),
            "token_offsets": [list(offset) for offset in self.token_offsets],
            "covered_start": self.covered.start,
            "covered_end": self.covered.end,
            "covered_nucleotides": self.covered_nucleotides,
            "site_nucleotides": self.site_nucleotides,
            "overhang_nucleotides": self.overhang_nucleotides,
            "geometry_signature": [list(offset) for offset in self.geometry_signature],
        }


def normalize_dna(sequence: str) -> str:
    normalized = sequence.upper().replace("U", "T")
    bad = set(normalized) - DNA_ALPHABET
    if not normalized or bad:
        raise ValueError(
            f"unsupported or empty DNA sequence; invalid symbols={sorted(bad)}"
        )
    return normalized


def build_mask_plan(
    sequence: str,
    site: Interval,
    *,
    input_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
) -> MaskPlan:
    sequence = normalize_dna(sequence)
    if site.end > len(sequence):
        raise ValueError("site falls outside sequence")
    if len(input_ids) != len(offsets):
        raise ValueError("input IDs and offsets have different lengths")
    selected = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < site.end and end > site.start
    ]
    if not selected:
        raise ValueError("tokenizer produced no tokens overlapping the biological site")
    if selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError("site maps to non-consecutive BPE tokens")
    selected_offsets = tuple((int(offsets[i][0]), int(offsets[i][1])) for i in selected)
    for left, right in pairwise(selected_offsets):
        if left[1] != right[0]:
            raise ValueError("target BPE offsets contain a gap or overlap")
    covered = Interval(selected_offsets[0][0], selected_offsets[-1][1])
    plan = MaskPlan(
        site=site,
        token_indices=tuple(selected),
        true_token_ids=tuple(int(input_ids[i]) for i in selected),
        token_offsets=selected_offsets,
        covered=covered,
        sequence_length=len(sequence),
    )
    reconstructed = "".join(sequence[start:end] for start, end in selected_offsets)
    if reconstructed != sequence[covered.start : covered.end]:
        raise ValueError("token offsets do not form a faithful nucleotide cover")
    return plan


def encode_mask_plan(
    sequence: str, site: Interval, encoder: Encoder
) -> tuple[list[int], MaskPlan]:
    normalized = normalize_dna(sequence)
    input_ids, offsets = encoder(normalized)
    return input_ids, build_mask_plan(
        normalized,
        site,
        input_ids=input_ids,
        offsets=offsets,
    )


def geometry_is_invariant(reference: MaskPlan, comparison: MaskPlan) -> bool:
    return (
        reference.site == comparison.site
        and reference.token_indices == comparison.token_indices
        and reference.token_offsets == comparison.token_offsets
        and reference.covered == comparison.covered
        and reference.sequence_length == comparison.sequence_length
    )


def fair_alternative_token_ids(
    sequence: str,
    plan: MaskPlan,
    alternative_sequence: str,
    encoder: Encoder,
) -> tuple[int, ...] | None:
    """Return alternative labels only when BPE geometry is exactly comparable.

    The masked input must be identical for a margin to be meaningful. Therefore the
    alternative must have the same nucleotide length, the full offset map must remain
    unchanged, all token IDs outside the masked cover must remain unchanged, and the
    alternative cover must use the same token indices.
    """

    original = normalize_dna(sequence)
    alternative = normalize_dna(alternative_sequence)
    if len(alternative) != plan.site.length:
        return None
    edited = original[: plan.site.start] + alternative + original[plan.site.end :]
    original_ids, original_offsets = encoder(original)
    alt_ids, alt_offsets = encoder(edited)
    if len(original_ids) != len(alt_ids) or original_offsets != alt_offsets:
        return None
    original_plan = build_mask_plan(
        original,
        plan.site,
        input_ids=original_ids,
        offsets=original_offsets,
    )
    alt_plan = build_mask_plan(
        edited,
        plan.site,
        input_ids=alt_ids,
        offsets=alt_offsets,
    )
    if not geometry_is_invariant(original_plan, alt_plan):
        return None
    target_indices = set(plan.token_indices)
    if any(
        original_ids[index] != alt_ids[index]
        for index in range(len(original_ids))
        if index not in target_indices
    ):
        return None
    return tuple(int(alt_ids[index]) for index in plan.token_indices)


def geometry_qualified_hamming_one_alternatives(
    sequence: str,
    plan: MaskPlan,
    encoder: Encoder,
) -> Mapping[str, str]:
    """Enumerate every one-base target substitution with fair BPE labels.

    DNABERT-2 BPE segmentation is sequence dependent, so a nominal nucleotide
    alternative is retained only when :func:`fair_alternative_token_ids` proves that
    it uses the identical masked input and token geometry.
    """

    normalized = normalize_dna(sequence)
    target = normalized[plan.site.start : plan.site.end]
    alternatives: dict[str, str] = {}
    for position, observed in enumerate(target):
        if observed not in "ACGT":
            continue
        for replacement in "ACGT":
            if replacement == observed:
                continue
            candidate = target[:position] + replacement + target[position + 1 :]
            if (
                fair_alternative_token_ids(
                    normalized,
                    plan,
                    candidate,
                    encoder,
                )
                is not None
            ):
                alternatives[f"substitution:{position}:{replacement}"] = candidate
    return alternatives


def common_geometry_qualified_alternatives(
    sequences: Sequence[str],
    site: Interval,
    candidates: Mapping[str, str],
    encoder: Encoder,
) -> Mapping[str, str]:
    """Return candidate target strings that are BPE-fair in every sequence."""

    if not sequences:
        raise ValueError("at least one sequence is required")
    common = set(candidates)
    for sequence in sequences:
        _, plan = encode_mask_plan(sequence, site, encoder)
        common = {
            name
            for name in common
            if fair_alternative_token_ids(
                sequence,
                plan,
                candidates[name],
                encoder,
            )
            is not None
        }
        if not common:
            break
    return {name: candidates[name] for name in sorted(common)}
