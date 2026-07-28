from __future__ import annotations

import torch

from gfmcd.adapters.dnabert2.promoter_head_atlas import (
    PROMOTER_SOURCE_BANDS,
    _token_band_allocations,
    paired_cls_patch_tensor,
    paired_cls_query_patch_tensor,
    paired_cls_write_patch_tensor,
    restore_cls_heads_from_reference,
)
from gfmcd.reconstruction_atlas.promoter_circuit import layer_pair_nulls
from gfmcd.reconstruction_atlas.promoter_head_analysis import (
    layer_matched_random_head_sets,
    parse_head_label,
    select_head_candidates,
    select_locus_rows,
    spearman_correlation,
)


def test_paired_cls_patch_changes_only_requested_target_head() -> None:
    context = torch.arange(12 * 8, dtype=torch.float32).view(12, 8)
    rows = torch.tensor([0, 3, 6, 9])
    patched = paired_cls_patch_tensor(
        context,
        cls_rows=rows,
        heads=[1],
        num_heads=2,
        direction="target_from_control",
    ).view(12, 2, 4)
    original = context.view(12, 2, 4)
    assert torch.equal(patched[0, 1], original[3, 1])
    assert torch.equal(patched[6, 1], original[9, 1])
    assert torch.equal(patched[0, 0], original[0, 0])
    assert torch.equal(patched[3], original[3])
    assert torch.equal(patched[9], original[9])


def test_paired_cls_self_patch_is_exact_noop() -> None:
    context = torch.randn(10, 12)
    patched = paired_cls_patch_tensor(
        context,
        cls_rows=torch.tensor([0, 5]),
        heads=[0, 1, 2],
        num_heads=3,
        direction="target_from_self",
    )
    assert torch.equal(patched, context)


def test_bidirectional_patch_uses_native_sources_in_both_directions() -> None:
    context = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    patched = paired_cls_patch_tensor(
        context,
        cls_rows=torch.tensor([0, 4]),
        heads=[0, 1],
        num_heads=2,
        direction="bidirectional_swap",
    )
    assert torch.equal(patched[0], context[4])
    assert torch.equal(patched[4], context[0])


def test_paired_cls_write_patch_changes_only_target_rows() -> None:
    writes = torch.arange(12 * 4, dtype=torch.float32).view(12, 4)
    rows = torch.tensor([0, 3, 6, 9])
    patched = paired_cls_write_patch_tensor(
        writes,
        cls_rows=rows,
        direction="target_from_control",
    )
    assert torch.equal(patched[0], writes[3])
    assert torch.equal(patched[6], writes[9])
    assert torch.equal(patched[3], writes[3])
    assert torch.equal(patched[9], writes[9])


def test_query_patch_changes_q_but_not_kv() -> None:
    qkv = torch.arange(8 * 24, dtype=torch.float32).view(8, 24)
    patched = paired_cls_query_patch_tensor(
        qkv,
        cls_rows=torch.tensor([0, 4]),
        heads=[1],
        num_heads=2,
        direction="target_from_control",
    ).view(8, 3, 2, 4)
    original = qkv.view(8, 3, 2, 4)
    assert torch.equal(patched[0, 0, 1], original[4, 0, 1])
    assert torch.equal(patched[0, 0, 0], original[0, 0, 0])
    assert torch.equal(patched[0, 1:], original[0, 1:])
    assert torch.equal(patched[4], original[4])


def test_native_cache_restores_only_requested_target_heads() -> None:
    context = torch.zeros((8, 8))
    reference = torch.arange(8 * 8, dtype=torch.float32).view(8, 8)
    restored = restore_cls_heads_from_reference(
        context,
        reference=reference,
        cls_rows=torch.tensor([0, 4]),
        heads=[1],
        num_heads=2,
    ).view(8, 2, 4)
    expected = reference.view(8, 2, 4)
    assert torch.equal(restored[0, 1], expected[0, 1])
    assert torch.equal(restored[0, 0], torch.zeros(4))
    assert torch.equal(restored[4], torch.zeros((2, 4)))


def test_attention_tokens_are_fractionally_allocated_across_tss_bands() -> None:
    allocations = _token_band_allocations(
        [(0, 0), (197, 201), (247, 252), (299, 300), (0, 300)],
        tss_index=249,
        bands=PROMOTER_SOURCE_BANDS,
    )
    assert torch.allclose(allocations.sum(dim=1), torch.ones(5))
    assert allocations[0, 0] == 1
    # [247,252) is exactly the five-nucleotide [-2,+3) anchor.
    assert allocations[2, 3] == 1
    # A whole-window synthetic token is distributed by nucleotide width.
    assert torch.isclose(allocations[4, 1], torch.tensor(199 / 300))
    assert torch.isclose(allocations[4, 3], torch.tensor(5 / 300))


def test_outcome_blind_locus_subset_is_unique_and_reproducible() -> None:
    rows = [
        {
            "example_id": f"e{index}",
            "split": "development",
            "homology_cluster": f"c{index // 2}",
        }
        for index in range(20)
    ]
    first = select_locus_rows(
        rows, split="development", maximum=5, seed=7, tag="head-atlas"
    )
    second = select_locus_rows(
        list(reversed(rows)),
        split="development",
        maximum=5,
        seed=7,
        tag="head-atlas",
    )
    assert [row["example_id"] for row in first] == [row["example_id"] for row in second]
    assert len({row["homology_cluster"] for row in first}) == 5


def test_candidate_selection_requires_four_way_sign_and_enforces_layer_cap() -> None:
    rows = []
    for layer in range(3):
        for head in range(4):
            value = 20 - (layer * 4 + head)
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "necessity_mean": value,
                    "sufficiency_mean": value,
                    "attribution_necessity_mean": value,
                    "attribution_sufficiency_mean": value,
                }
            )
    rows[1]["sufficiency_mean"] = -1
    selected = select_head_candidates(rows, candidate_count=4, maximum_per_layer=2)
    assert len(selected) == 4
    assert sum(int(row["layer"]) == 0 for row in selected) <= 2
    assert all(row["sign_consistent"] for row in selected)
    assert not any(int(row["layer"]) == 0 and int(row["head"]) == 1 for row in selected)


def test_layer_matched_nulls_are_distinct_and_exclude_candidates() -> None:
    candidates = [(0, 0), (0, 1), (2, 3)]
    nulls = layer_matched_random_head_sets(
        candidates,
        num_layers=3,
        num_heads=5,
        draws=20,
        seed=11,
    )
    # The layer-profile null is exhaustible here: C(3,2) * C(4,1) = 12.
    assert len(nulls) == len(set(nulls)) == 12
    for null in nulls:
        assert not set(null) & set(candidates)
        assert [layer for layer, _ in null].count(0) == 2
        assert [layer for layer, _ in null].count(2) == 1


def test_spearman_and_head_parser() -> None:
    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == 1
    assert parse_head_label("L11H7") == (11, 7)


def test_layer_pair_nulls_are_exact_and_candidate_excluding() -> None:
    nulls = layer_pair_nulls([(11, 2), (11, 5)], num_heads=12)
    assert len(nulls) == 45
    assert len(set(nulls)) == 45
    assert all({(11, 2), (11, 5)}.isdisjoint(pair) for pair in nulls)
