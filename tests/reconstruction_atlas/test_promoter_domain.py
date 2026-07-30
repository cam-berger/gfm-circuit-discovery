from __future__ import annotations

import json
from pathlib import Path

import pytest

from gfmcd.reconstruction_atlas.promoter_domain import (
    TileRow,
    check_memorials,
    contrast_by_locus,
    load_clusters,
    load_tile_rows,
    paired_tile_difference,
    single_tile_by_locus,
)


def _row(pair: str, cluster: str, tile: int, loss: float, **kw) -> TileRow:
    return TileRow(
        pair_id=pair,
        cluster=cluster,
        tile_index=tile,
        target_specific_loss=loss,
        target_cover_length=kw.get("target_cover_length", 10),
        control_cover_length=kw.get("control_cover_length", 10),
        token_count=kw.get("token_count", 2),
    )


def test_contrast_require_all_drops_incomplete_locus() -> None:
    rows = [_row("p1", "c1", 25, 1.0), _row("p1", "c1", 28, 0.0)]
    values, clusters = contrast_by_locus(
        rows, inner_tiles=(25, 27), outer_tiles=(28, 29), require_all=True
    )
    assert values == []
    assert clusters == []


def test_contrast_require_all_computes_correct_value_when_complete() -> None:
    # Finding 3: the require_all=True *success* path was only ever checked for
    # "not empty" (via the drop test above). Pin the actual number too, so
    # v1-rule equivalence is verified the same way the relaxed rule is.
    rows = [
        _row("p1", "c1", 25, 1.0),
        _row("p1", "c1", 27, 3.0),
        _row("p1", "c1", 28, 0.5),
        _row("p1", "c1", 29, 1.5),
    ]
    values, clusters = contrast_by_locus(
        rows, inner_tiles=(25, 27), outer_tiles=(28, 29), require_all=True
    )
    # mean(1.0, 3.0) - mean(0.5, 1.5) = 2.0 - 1.0 = 1.0
    assert values == [pytest.approx(1.0)]
    assert clusters == ["c1"]


def test_contrast_relaxed_uses_available_tiles() -> None:
    # v2 rule: >=1 inner and >=1 outer; average over whichever are present.
    rows = [_row("p1", "c1", 25, 1.0), _row("p1", "c1", 28, 0.0)]
    values, clusters = contrast_by_locus(rows, inner_tiles=(25, 27), outer_tiles=(28, 29))
    assert values == [pytest.approx(1.0)]
    assert clusters == ["c1"]


def test_contrast_relaxed_still_drops_a_locus_missing_a_whole_side() -> None:
    rows = [_row("p1", "c1", 25, 1.0), _row("p1", "c1", 27, 2.0)]
    values, _ = contrast_by_locus(rows, inner_tiles=(25, 27), outer_tiles=(28, 29))
    assert values == []


def test_paired_tile_difference_needs_both_tiles() -> None:
    rows = [
        _row("p1", "c1", 27, 0.9),
        _row("p1", "c1", 28, 0.1),
        _row("p2", "c2", 27, 0.5),
    ]
    values, clusters = paired_tile_difference(rows, tile_a=27, tile_b=28)
    assert values == [pytest.approx(0.8)]
    assert clusters == ["c1"]


def test_contrast_is_mean_inner_minus_mean_outer() -> None:
    rows = [
        _row("p1", "c1", 25, 1.0),
        _row("p1", "c1", 27, 3.0),
        _row("p1", "c1", 28, 0.5),
        _row("p1", "c1", 29, 1.5),
    ]
    values, clusters = contrast_by_locus(rows, inner_tiles=(25, 27), outer_tiles=(28, 29))
    assert values == [pytest.approx(1.0)]
    assert clusters == ["c1"]


def test_single_tile_selects_only_that_tile() -> None:
    rows = [_row("p1", "c1", 22, -0.3), _row("p1", "c1", 25, 0.9)]
    values, clusters = single_tile_by_locus(rows, tile_index=22)
    assert values == [pytest.approx(-0.3)]
    assert clusters == ["c1"]


def test_memorial_M3_fails_on_unbalanced_cover() -> None:
    rows = [_row("p1", "c1", 25, 1.0, target_cover_length=10, control_cover_length=14)]
    result = check_memorials(rows, expected_tokens=2, max_cover_difference_nt=2)
    assert result["M3_cover_balance"] is False


def test_memorial_M4_fails_on_wrong_dose() -> None:
    rows = [_row("p1", "c1", 25, 1.0, token_count=3)]
    result = check_memorials(rows, expected_tokens=2, max_cover_difference_nt=2)
    assert result["M4_equal_dose"] is False


def test_memorials_pass_on_clean_rows() -> None:
    rows = [_row("p1", "c1", 25, 1.0), _row("p2", "c2", 27, 0.4)]
    result = check_memorials(rows, expected_tokens=2, max_cover_difference_nt=2)
    assert result == {"M3_cover_balance": True, "M4_equal_dose": True}


def test_memorial_thresholds_are_read_not_hardcoded() -> None:
    # Finding 1: every other memorial test uses expected_tokens=2,
    # max_cover_difference_nt=2, so an implementation that hardcoded 2 and 2
    # instead of reading the keyword arguments would still pass them. Use the
    # same row under two different threshold pairs and require the verdicts
    # to flip in the direction those thresholds imply.
    rows = [
        _row(
            "p1",
            "c1",
            25,
            1.0,
            target_cover_length=10,
            control_cover_length=14,  # cover diff = 4
            token_count=3,
        )
    ]
    strict = check_memorials(rows, expected_tokens=2, max_cover_difference_nt=2)
    assert strict == {"M3_cover_balance": False, "M4_equal_dose": False}

    relaxed = check_memorials(rows, expected_tokens=3, max_cover_difference_nt=5)
    assert relaxed == {"M3_cover_balance": True, "M4_equal_dose": True}


def test_check_memorials_raises_on_empty_rows() -> None:
    # Finding 4: all() is vacuously True on an empty sequence, so without an
    # explicit guard both memorials would silently report "passed" having
    # measured zero rows. This function gates a run that opens sealed data,
    # so it must raise instead of green-lighting on nothing.
    with pytest.raises(ValueError, match="zero rows"):
        check_memorials([], expected_tokens=2, max_cover_difference_nt=2)


def test_load_clusters_maps_example_id_to_homology_cluster(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.jsonl"
    cohort.write_text(
        json.dumps({"example_id": "p1", "homology_cluster": "locus1000000:chr1:112"})
        + "\n"
    )
    assert load_clusters(cohort) == {"p1": "locus1000000:chr1:112"}


def test_load_tile_rows_joins_clusters_and_filters_operation(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    # NOTE: real D8 tile_effect rows carry NO homology_cluster field. The cluster
    # must come from the cohort join, which is what this test pins down.
    records = [
        {
            "row_type": "tile_effect",
            "operation": "mask",
            "pair_id": "p1",
            "tile_index": 25,
            "target_specific_loss": 0.5,
            "target_cover_start": 250,
            "target_cover_end": 260,
            "control_cover_start": 250,
            "control_cover_end": 260,
            "target_token_indices": [40, 41],
        },
        {
            "row_type": "tile_effect",
            "operation": "candidate_value_swap",
            "pair_id": "p1",
            "tile_index": 25,
            "target_specific_loss": 0.001,
            "target_cover_start": 250,
            "target_cover_end": 260,
            "control_cover_start": 250,
            "control_cover_end": 260,
            "target_token_indices": [40, 41],
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    rows = load_tile_rows(path, clusters={"p1": "c1"}, operation="mask")
    assert len(rows) == 1
    assert rows[0].tile_index == 25
    assert rows[0].cluster == "c1"
    assert rows[0].token_count == 2


def test_load_tile_rows_reads_operation_argument_not_just_default(
    tmp_path: Path,
) -> None:
    # Finding 2: every other call site uses operation="mask", which is also
    # the default, so a hardcoded `!= "mask"` filter would pass unnoticed.
    # Request the *other* operation from the same two-row fixture and assert
    # it is the other row that comes back.
    path = tmp_path / "raw.jsonl"
    records = [
        {
            "row_type": "tile_effect",
            "operation": "mask",
            "pair_id": "p1",
            "tile_index": 25,
            "target_specific_loss": 0.5,
            "target_cover_start": 250,
            "target_cover_end": 260,
            "control_cover_start": 250,
            "control_cover_end": 260,
            "target_token_indices": [40, 41],
        },
        {
            "row_type": "tile_effect",
            "operation": "candidate_value_swap",
            "pair_id": "p1",
            "tile_index": 25,
            "target_specific_loss": 0.001,
            "target_cover_start": 250,
            "target_cover_end": 260,
            "control_cover_start": 250,
            "control_cover_end": 260,
            "target_token_indices": [40, 41],
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    rows = load_tile_rows(path, clusters={"p1": "c1"}, operation="candidate_value_swap")
    assert len(rows) == 1
    assert rows[0].target_specific_loss == pytest.approx(0.001)


def test_load_tile_rows_refuses_an_unmapped_pair(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps(
            {
                "row_type": "tile_effect",
                "operation": "mask",
                "pair_id": "missing",
                "tile_index": 25,
                "target_specific_loss": 0.5,
                "target_cover_start": 250,
                "target_cover_end": 260,
                "control_cover_start": 250,
                "control_cover_end": 260,
                "target_token_indices": [40, 41],
            }
        )
        + "\n"
    )
    with pytest.raises(KeyError, match="missing"):
        load_tile_rows(path, clusters={"p1": "c1"}, operation="mask")
