"""Native-nucleotide counterfactuals for the promoter source region.

The D8 mask scan nominated the nominal interval +11:+21 relative to the TSS.
This module constructs two outcome-blind, length-preserving perturbation families:

* exact linear-dinucleotide-preserving reorders, which change arrangement without
  changing mono- or dinucleotide counts; and
* GC-depleting transitions (G→A and C→T), which preserve purine/pyrimidine class
  while changing local GC content by a fixed, paired dose.

Model scoring is deliberately absent. Tokenizer geometry is supplied by the caller
so that candidate construction and target outcomes remain separate.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NativeSourceEdit:
    """One native-base edit confined to a declared nucleotide interval."""

    operation: str
    start: int
    end: int
    original_subsequence: str
    edited_subsequence: str
    edited_sequence: str
    changed_nucleotides: int
    gc_count_change: int
    mononucleotide_preserved: bool
    dinucleotide_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def linear_dinucleotide_counts(sequence: str) -> Counter[tuple[str, str]]:
    """Return adjacent, non-circular dinucleotide counts."""

    sequence = sequence.upper()
    if len(sequence) < 2 or set(sequence) - set("ACGT"):
        raise ValueError("dinucleotide counting requires at least two ACGT bases")
    return Counter(itertools.pairwise(sequence))


def dinucleotide_preserving_reorders(
    sequence: str,
    *,
    start: int,
    end: int,
) -> tuple[NativeSourceEdit, ...]:
    """Enumerate non-identity Euler trails with the same dinucleotide multiset."""

    sequence, original = _validate_region(sequence, start=start, end=end)
    edges = linear_dinucleotide_counts(original)
    candidates: set[str] = set()

    def visit(path: list[str], remaining: int) -> None:
        if remaining == 0:
            candidate = "".join(path)
            if candidate != original:
                candidates.add(candidate)
            return
        source = path[-1]
        for edge in sorted(edges):
            if edge[0] != source or edges[edge] == 0:
                continue
            edges[edge] -= 1
            visit([*path, edge[1]], remaining - 1)
            edges[edge] += 1

    for first in "ACGT":
        visit([first], len(original) - 1)
    output = []
    for candidate in sorted(candidates):
        if Counter(candidate) != Counter(original):
            continue
        if linear_dinucleotide_counts(candidate) != linear_dinucleotide_counts(
            original
        ):
            raise RuntimeError("invalid dinucleotide-preserving candidate")
        output.append(
            _edit_record(
                sequence,
                candidate,
                operation="dinucleotide_reorder",
                start=start,
                end=end,
            )
        )
    return tuple(output)


def gc_depleting_transitions(
    sequence: str,
    *,
    start: int,
    end: int,
    dose: int,
) -> tuple[NativeSourceEdit, ...]:
    """Enumerate fixed-dose G→A/C→T edits within the declared interval."""

    sequence, original = _validate_region(sequence, start=start, end=end)
    if dose < 1:
        raise ValueError("GC-depletion dose must be positive")
    eligible = [index for index, base in enumerate(original) if base in {"G", "C"}]
    if len(eligible) < dose:
        return ()
    output = []
    for selected in itertools.combinations(eligible, dose):
        edited = list(original)
        for index in selected:
            edited[index] = "A" if edited[index] == "G" else "T"
        output.append(
            _edit_record(
                sequence,
                "".join(edited),
                operation="gc_depletion",
                start=start,
                end=end,
            )
        )
    return tuple(output)


def select_geometry_qualified_paired_edits(
    target: str,
    control: str,
    *,
    start: int,
    end: int,
    gc_depletion_dose: int,
    geometry_qualifier: Callable[[str, Sequence[NativeSourceEdit]], Sequence[bool]],
    paired_geometry_qualifier: Callable[
        [str, NativeSourceEdit, str, NativeSourceEdit], bool
    ]
    | None = None,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Select one equal-Hamming, geometry-qualified edit per operation and branch."""

    if len(target) != len(control):
        raise ValueError("paired native edits require equal-length sequences")
    candidates = {
        "dinucleotide_reorder": (
            dinucleotide_preserving_reorders(target, start=start, end=end),
            dinucleotide_preserving_reorders(control, start=start, end=end),
        ),
        "gc_depletion": (
            gc_depleting_transitions(
                target,
                start=start,
                end=end,
                dose=gc_depletion_dose,
            ),
            gc_depleting_transitions(
                control,
                start=start,
                end=end,
                dose=gc_depletion_dose,
            ),
        ),
    }
    selected: dict[str, dict[str, Any]] = {}
    for operation, (target_candidates, control_candidates) in candidates.items():
        qualified_target = _qualified_edits(
            target, target_candidates, geometry_qualifier
        )
        qualified_control = _qualified_edits(
            control, control_candidates, geometry_qualifier
        )
        common_doses = sorted(
            {edit.changed_nucleotides for edit in qualified_target}
            & {edit.changed_nucleotides for edit in qualified_control},
            reverse=True,
        )
        if not common_doses:
            continue
        pair_candidates = []
        for dose in common_doses:
            for target_edit in qualified_target:
                if target_edit.changed_nucleotides != dose:
                    continue
                for control_edit in qualified_control:
                    if control_edit.changed_nucleotides != dose:
                        continue
                    if paired_geometry_qualifier is not None and not (
                        paired_geometry_qualifier(
                            target,
                            target_edit,
                            control,
                            control_edit,
                        )
                    ):
                        continue
                    pair_candidates.append((target_edit, control_edit))
            if pair_candidates:
                break
        if not pair_candidates:
            continue
        target_edit, control_edit = _stable_pair_choice(
            pair_candidates,
            seed=seed,
            label=operation,
        )
        dose = target_edit.changed_nucleotides
        selected[operation] = {
            "target": target_edit.to_dict(),
            "control": control_edit.to_dict(),
            "changed_nucleotides_each": dose,
            "equal_hamming_dose": (
                target_edit.changed_nucleotides == control_edit.changed_nucleotides
            ),
        }
    return selected


def native_edit_invariants(
    original: str,
    edit: Mapping[str, Any],
    *,
    start: int,
    end: int,
) -> dict[str, bool]:
    """Audit sequence-level invariants for a serialized edit record."""

    edited = str(edit["edited_sequence"])
    operation = str(edit["operation"])
    original = original.upper()
    confined = original[:start] == edited[:start] and original[end:] == edited[end:]
    region_original = original[start:end]
    region_edited = edited[start:end]
    hamming = sum(left != right for left, right in zip(region_original, region_edited))
    return {
        "length_preserved": len(original) == len(edited),
        "ACGT_only": not (set(edited) - set("ACGT")),
        "edit_confined_to_region": confined,
        "declared_hamming_matches": hamming == int(edit["changed_nucleotides"]),
        "non_identity": hamming > 0,
        "dinucleotide_rule": (
            operation != "dinucleotide_reorder"
            or (
                Counter(region_original) == Counter(region_edited)
                and linear_dinucleotide_counts(region_original)
                == linear_dinucleotide_counts(region_edited)
            )
        ),
        "GC_depletion_rule": (
            operation != "gc_depletion"
            or (
                region_original.count("G")
                + region_original.count("C")
                - region_edited.count("G")
                - region_edited.count("C")
                == hamming
                and all(
                    (left, right) in {("G", "A"), ("C", "T")}
                    for left, right in zip(region_original, region_edited)
                    if left != right
                )
            )
        ),
    }


def local_retokenization_geometry(
    original_ids: Sequence[int],
    original_offsets: Sequence[Sequence[int]],
    edited_ids: Sequence[int],
    edited_offsets: Sequence[Sequence[int]],
    *,
    start: int,
    end: int,
    maximum_affected_cover_nt: int,
    maximum_region_overhang_each_side_nt: int,
) -> dict[str, Any]:
    """Describe the minimal changed BPE segment with exact outer anchors."""

    if (
        start < 0
        or end <= start
        or maximum_affected_cover_nt < 1
        or maximum_region_overhang_each_side_nt < 0
        or len(original_ids) != len(original_offsets)
        or len(edited_ids) != len(edited_offsets)
    ):
        raise ValueError("invalid local-retokenization geometry request")
    original_entries = [
        (int(token_id), (int(offset[0]), int(offset[1])))
        for token_id, offset in zip(original_ids, original_offsets)
    ]
    edited_entries = [
        (int(token_id), (int(offset[0]), int(offset[1])))
        for token_id, offset in zip(edited_ids, edited_offsets)
    ]
    prefix = 0
    while (
        prefix < len(original_entries)
        and prefix < len(edited_entries)
        and original_entries[prefix] == edited_entries[prefix]
    ):
        prefix += 1
    suffix = 0
    while (
        suffix < len(original_entries) - prefix
        and suffix < len(edited_entries) - prefix
        and original_entries[-1 - suffix] == edited_entries[-1 - suffix]
    ):
        suffix += 1
    original_affected = original_entries[prefix : len(original_entries) - suffix]
    edited_affected = edited_entries[prefix : len(edited_entries) - suffix]
    affected_offsets = [
        offset
        for _token_id, offset in (*original_affected, *edited_affected)
        if offset[1] > offset[0]
    ]
    cover_start = (
        min(offset[0] for offset in affected_offsets) if affected_offsets else None
    )
    cover_end = (
        max(offset[1] for offset in affected_offsets) if affected_offsets else None
    )
    cover_length = (
        int(cover_end) - int(cover_start)
        if cover_start is not None and cover_end is not None
        else 0
    )
    local = (
        cover_start is not None
        and cover_end is not None
        and int(cover_start) >= start - maximum_region_overhang_each_side_nt
        and int(cover_end) <= end + maximum_region_overhang_each_side_nt
        and cover_length <= maximum_affected_cover_nt
        and int(cover_start) < end
        and start < int(cover_end)
    )
    return {
        "qualified": bool(original_affected and edited_affected and local),
        "token_count": len(original_ids),
        "edited_token_count": len(edited_ids),
        "token_count_delta": len(edited_ids) - len(original_ids),
        "unchanged_prefix_token_entries": prefix,
        "unchanged_suffix_token_entries": suffix,
        "original_affected_token_count": len(original_affected),
        "edited_affected_token_count": len(edited_affected),
        "affected_cover_start": cover_start,
        "affected_cover_end": cover_end,
        "affected_cover_length": cover_length,
        "retokenization_is_local": local,
    }


def paired_retokenization_geometry_matches(
    target: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    maximum_affected_token_count_difference: int,
    maximum_affected_cover_difference_nt: int,
) -> bool:
    """Match local BPE dose between target and pseudo-TSS branches."""

    if (
        maximum_affected_token_count_difference < 0
        or maximum_affected_cover_difference_nt < 0
    ):
        raise ValueError("paired geometry calipers must be nonnegative")
    return (
        bool(target["qualified"])
        and bool(control["qualified"])
        and int(target["token_count_delta"]) == int(control["token_count_delta"])
        and abs(
            int(target["original_affected_token_count"])
            - int(control["original_affected_token_count"])
        )
        <= maximum_affected_token_count_difference
        and abs(
            int(target["edited_affected_token_count"])
            - int(control["edited_affected_token_count"])
        )
        <= maximum_affected_token_count_difference
        and abs(
            int(target["affected_cover_length"]) - int(control["affected_cover_length"])
        )
        <= maximum_affected_cover_difference_nt
    )


def geometry_qualified_native_edits(
    tokenizer: Any,
    target: str,
    control: str,
    *,
    start: int,
    end: int,
    gc_depletion_dose: int,
    policy: Mapping[str, Any],
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Construct and annotate paired edits under the local BPE policy."""

    cache: dict[tuple[str, str], dict[str, Any]] = {}

    def geometry_from_encoded(
        original_ids: Sequence[int],
        original_offsets: Sequence[Sequence[int]],
        edited_ids: Sequence[int],
        edited_offsets: Sequence[Sequence[int]],
    ) -> dict[str, Any]:
        return local_retokenization_geometry(
            original_ids,
            original_offsets,
            edited_ids,
            edited_offsets,
            start=start,
            end=end,
            maximum_affected_cover_nt=int(policy["maximum_affected_cover_nt"]),
            maximum_region_overhang_each_side_nt=int(
                policy["maximum_region_overhang_each_side_nt"]
            ),
        )

    def record(original: str, edited: str) -> dict[str, Any]:
        key = (original, edited)
        if key not in cache:
            encoded = tokenizer(
                [original, edited],
                padding=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            cache[key] = geometry_from_encoded(
                encoded["input_ids"][0],
                encoded["offset_mapping"][0],
                encoded["input_ids"][1],
                encoded["offset_mapping"][1],
            )
        return cache[key]

    def qualify(
        original: str,
        candidates: Sequence[NativeSourceEdit],
    ) -> list[bool]:
        if not candidates:
            return []
        encoded = tokenizer(
            [original, *(edit.edited_sequence for edit in candidates)],
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        original_ids = encoded["input_ids"][0]
        original_offsets = encoded["offset_mapping"][0]
        records = [
            geometry_from_encoded(
                original_ids,
                original_offsets,
                encoded["input_ids"][index],
                encoded["offset_mapping"][index],
            )
            for index in range(1, len(candidates) + 1)
        ]
        for edit, geometry in zip(candidates, records):
            cache[(original, edit.edited_sequence)] = geometry
        return [bool(geometry["qualified"]) for geometry in records]

    def paired(
        target_sequence: str,
        target_edit: NativeSourceEdit,
        control_sequence: str,
        control_edit: NativeSourceEdit,
    ) -> bool:
        return paired_retokenization_geometry_matches(
            record(target_sequence, target_edit.edited_sequence),
            record(control_sequence, control_edit.edited_sequence),
            maximum_affected_token_count_difference=int(
                policy["maximum_target_control_affected_token_count_difference"]
            ),
            maximum_affected_cover_difference_nt=int(
                policy["maximum_target_control_affected_cover_difference_nt"]
            ),
        )

    edits = select_geometry_qualified_paired_edits(
        target,
        control,
        start=start,
        end=end,
        gc_depletion_dose=gc_depletion_dose,
        geometry_qualifier=qualify,
        paired_geometry_qualifier=paired,
        seed=seed,
    )
    for operation in edits.values():
        for branch, original in (("target", target), ("control", control)):
            operation[branch]["token_geometry"] = record(
                original,
                str(operation[branch]["edited_sequence"]),
            )
    return edits


def geometry_qualified_native_edit_ensemble(
    tokenizer: Any,
    target: str,
    control: str,
    *,
    start: int,
    end: int,
    gc_depletion_dose: int,
    policy: Mapping[str, Any],
    seed: int,
    replicates: int,
    maximum_attempts: int = 32,
    operations: Sequence[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Select distinct paired edits so the locus mean is edit-choice robust."""

    if replicates < 1 or maximum_attempts < replicates:
        raise ValueError("invalid native-edit ensemble request")
    supported = {"dinucleotide_reorder", "gc_depletion"}
    required = set(operations) if operations is not None else supported
    if not required or not required <= supported:
        raise ValueError("unsupported or empty native-edit operation family")
    output: dict[str, list[dict[str, Any]]] = {operation: [] for operation in supported}
    signatures: dict[str, set[tuple[str, str]]] = {
        operation: set() for operation in supported
    }
    for attempt in range(maximum_attempts):
        edits = geometry_qualified_native_edits(
            tokenizer,
            target,
            control,
            start=start,
            end=end,
            gc_depletion_dose=gc_depletion_dose,
            policy=policy,
            seed=seed ^ _attempt_seed(attempt),
        )
        for operation, edit in edits.items():
            signature = (
                str(edit["target"]["edited_sequence"]),
                str(edit["control"]["edited_sequence"]),
            )
            if signature in signatures[operation]:
                continue
            signatures[operation].add(signature)
            output[operation].append(edit)
        if all(len(output[operation]) >= replicates for operation in required):
            return {
                operation: output[operation][:replicates]
                for operation in sorted(required)
            }
    return {
        operation: output[operation]
        for operation in sorted(required)
        if len(output[operation]) >= replicates
    }


def _validate_region(
    sequence: str,
    *,
    start: int,
    end: int,
) -> tuple[str, str]:
    sequence = sequence.upper()
    if start < 0 or end <= start or end > len(sequence) or set(sequence) - set("ACGT"):
        raise ValueError("invalid native source-edit region or sequence")
    return sequence, sequence[start:end]


def _edit_record(
    sequence: str,
    edited_subsequence: str,
    *,
    operation: str,
    start: int,
    end: int,
) -> NativeSourceEdit:
    original = sequence[start:end]
    edited_sequence = sequence[:start] + edited_subsequence + sequence[end:]
    changed = sum(left != right for left, right in zip(original, edited_subsequence))
    return NativeSourceEdit(
        operation=operation,
        start=start,
        end=end,
        original_subsequence=original,
        edited_subsequence=edited_subsequence,
        edited_sequence=edited_sequence,
        changed_nucleotides=changed,
        gc_count_change=(
            edited_subsequence.count("G")
            + edited_subsequence.count("C")
            - original.count("G")
            - original.count("C")
        ),
        mononucleotide_preserved=Counter(original) == Counter(edited_subsequence),
        dinucleotide_preserved=(
            linear_dinucleotide_counts(original)
            == linear_dinucleotide_counts(edited_subsequence)
        ),
    )


def _qualified_edits(
    original: str,
    candidates: Sequence[NativeSourceEdit],
    qualifier: Callable[[str, Sequence[NativeSourceEdit]], Sequence[bool]],
) -> list[NativeSourceEdit]:
    if not candidates:
        return []
    qualified = list(qualifier(original, candidates))
    if len(qualified) != len(candidates):
        raise ValueError("geometry qualifier did not align with edit candidates")
    return [edit for edit, accepted in zip(candidates, qualified) if bool(accepted)]


def _stable_pair_choice(
    candidates: Sequence[tuple[NativeSourceEdit, NativeSourceEdit]],
    *,
    seed: int,
    label: str,
) -> tuple[NativeSourceEdit, NativeSourceEdit]:
    if not candidates:
        raise ValueError("cannot choose from an empty paired edit family")
    return min(
        candidates,
        key=lambda edits: hashlib.sha256(
            (
                f"{seed}|{label}|{edits[0].original_subsequence}|"
                f"{edits[0].edited_subsequence}|"
                f"{edits[1].original_subsequence}|"
                f"{edits[1].edited_subsequence}"
            ).encode()
        ).hexdigest(),
    )


def _attempt_seed(attempt: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"native-edit-attempt|{attempt}".encode()).digest()[:8],
        "big",
    )
