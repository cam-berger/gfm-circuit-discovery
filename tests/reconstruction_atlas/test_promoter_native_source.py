from __future__ import annotations

from collections import Counter

from gfmcd.reconstruction_atlas import promoter_native_source
from gfmcd.reconstruction_atlas.promoter_native_source import (
    dinucleotide_preserving_reorders,
    gc_depleting_transitions,
    linear_dinucleotide_counts,
    local_retokenization_geometry,
    native_edit_invariants,
    paired_retokenization_geometry_matches,
    select_geometry_qualified_paired_edits,
)


def test_dinucleotide_reorders_preserve_linear_counts() -> None:
    sequence = "TTTAACGACATTT"
    edits = dinucleotide_preserving_reorders(sequence, start=3, end=10)
    assert edits
    for edit in edits:
        assert edit.edited_sequence[:3] == "TTT"
        assert edit.edited_sequence[10:] == "TTT"
        assert Counter(edit.original_subsequence) == Counter(edit.edited_subsequence)
        assert linear_dinucleotide_counts(
            edit.original_subsequence
        ) == linear_dinucleotide_counts(edit.edited_subsequence)
        assert edit.changed_nucleotides > 0


def test_gc_depletion_uses_purine_pyrimidine_preserving_transitions() -> None:
    edits = gc_depleting_transitions(
        "AAAAGCGCTTAA",
        start=4,
        end=10,
        dose=3,
    )
    assert len(edits) == 4
    for edit in edits:
        assert edit.changed_nucleotides == 3
        assert edit.gc_count_change == -3
        assert not edit.mononucleotide_preserved
        assert all(
            native_edit_invariants(
                "AAAAGCGCTTAA",
                edit.to_dict(),
                start=4,
                end=10,
            ).values()
        )


def test_paired_selection_requires_common_hamming_and_geometry() -> None:
    def geometry(_original: str, edits: list[object] | tuple[object, ...]):
        return [edit.edited_subsequence != "GCGCAT" for edit in edits]

    selected = select_geometry_qualified_paired_edits(
        "TTTAACGACATTT",
        "TTTACGACAGTTT",
        start=3,
        end=10,
        gc_depletion_dose=2,
        geometry_qualifier=geometry,
        paired_geometry_qualifier=lambda _ta, te, _ca, ce: (
            te.changed_nucleotides == ce.changed_nucleotides
        ),
        seed=11,
    )
    assert set(selected) == {"dinucleotide_reorder", "gc_depletion"}
    for operation in selected.values():
        assert operation["equal_hamming_dose"]
        assert (
            operation["target"]["changed_nucleotides"]
            == operation["control"]["changed_nucleotides"]
        )


def test_no_dinucleotide_reorder_for_homopolymer() -> None:
    assert not dinucleotide_preserving_reorders("TTTAAAAATTT", start=3, end=8)


def test_local_retokenization_geometry_matches_equal_dose() -> None:
    original_ids = [1, 10, 11, 12, 2]
    original_offsets = [(0, 0), (0, 4), (4, 8), (8, 12), (0, 0)]
    left = local_retokenization_geometry(
        original_ids,
        original_offsets,
        [1, 10, 20, 21, 2],
        [(0, 0), (0, 4), (4, 7), (7, 12), (0, 0)],
        start=5,
        end=10,
        maximum_affected_cover_nt=12,
        maximum_region_overhang_each_side_nt=4,
    )
    right = local_retokenization_geometry(
        original_ids,
        original_offsets,
        [1, 10, 30, 31, 2],
        [(0, 0), (0, 4), (4, 6), (6, 12), (0, 0)],
        start=5,
        end=10,
        maximum_affected_cover_nt=12,
        maximum_region_overhang_each_side_nt=4,
    )
    assert left["qualified"] and right["qualified"]
    assert paired_retokenization_geometry_matches(
        left,
        right,
        maximum_affected_token_count_difference=0,
        maximum_affected_cover_difference_nt=0,
    )


def test_native_edit_ensemble_can_freeze_one_operation(monkeypatch) -> None:
    def fake_edits(*_args, **_kwargs):
        return {
            operation: {
                "target": {"edited_sequence": f"target-{operation}"},
                "control": {"edited_sequence": f"control-{operation}"},
            }
            for operation in ("dinucleotide_reorder", "gc_depletion")
        }

    monkeypatch.setattr(
        promoter_native_source,
        "geometry_qualified_native_edits",
        fake_edits,
    )
    result = promoter_native_source.geometry_qualified_native_edit_ensemble(
        object(),
        "AACG",
        "ACGT",
        start=0,
        end=4,
        gc_depletion_dose=1,
        policy={},
        seed=1,
        replicates=1,
        operations=("dinucleotide_reorder",),
    )
    assert set(result) == {"dinucleotide_reorder"}
    assert len(result["dinucleotide_reorder"]) == 1
