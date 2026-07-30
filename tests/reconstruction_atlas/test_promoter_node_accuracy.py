from __future__ import annotations

from gfmcd.reconstruction_atlas.promoter_node_accuracy import (
    all_cls_nodes,
    rank_node_summaries,
    summarize_node_accuracy_rows,
)


def test_all_cls_nodes_contains_heads_and_one_mlp_per_layer() -> None:
    nodes = all_cls_nodes(layers=2, heads_per_layer=3)
    assert len(nodes) == 8
    assert [node["node"] for node in nodes] == [
        "L0H0",
        "L0H1",
        "L0H2",
        "L1H0",
        "L1H1",
        "L1H2",
        "L0MLP",
        "L1MLP",
    ]


def test_accuracy_first_summary_ranks_more_disruptive_node_first() -> None:
    rows = []
    for node, node_type, failures in (
        ("L0H0", "head", {0, 1}),
        ("L0MLP", "mlp", {0}),
    ):
        for pair in range(4):
            target_patched = -1.0 if pair in failures else 1.0
            loss = int(target_patched <= 0)
            rows.append(
                {
                    "pair_id": f"pair-{pair}",
                    "node": node,
                    "node_type": node_type,
                    "layer": 0,
                    "head": 0 if node_type == "head" else None,
                    "native_target_margin": 1.0,
                    "native_control_margin": -1.0,
                    "patched_target_margin": target_patched,
                    "patched_control_margin": -1.0,
                    "class_margin_loss": (1.0 - target_patched) / 2,
                    "accuracy_loss": loss / 2,
                }
            )
    summaries = summarize_node_accuracy_rows(
        rows,
        bootstrap_draws=50,
        sign_permutations=99,
        seed=7,
    )
    assert summaries["L0H0"]["accuracy_loss"] == 0.25
    assert summaries["L0MLP"]["accuracy_loss"] == 0.125
    assert rank_node_summaries(summaries) == ["L0H0", "L0MLP"]
