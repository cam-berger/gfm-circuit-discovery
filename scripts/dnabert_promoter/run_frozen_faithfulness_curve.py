"""Measure frozen DNABERT faithfulness versus circuit size.

Nodes:
- 36 attention heads across encoder Layers 0--2
- 3 MLP outputs across encoder Layers 0--2

The initial five-node circuit is fixed:
- L0H8
- L1H8
- L2H3
- L2H9
- Layer-0 MLP

Remaining nodes are added using a fixed validation-derived ranking based on
single-node promoter mean ablation damage.

The curve is evaluated using head-and-MLP component complement patching.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from gfmcd.adapters.dnabert import DnaBertAdapter

from run_frozen_component_faithfulness import (
    OUTPUT_DIR as FAITHFULNESS_OUTPUT_DIR,
    PATCHED_LAYERS,
    capture_all_components_and_logit,
    finite_summary,
    load_promoters,
    run_component_patch,
)

from run_head_restoration_pilot import (
    DEFAULT_COHORT,
    MODEL_NAME,
    MODEL_REVISION,
    PROBE_PATH,
    SELECTED_LAYER,
    assert_adapter_tokenization_matches,
    get_corrupt_sequence,
    tokenize_with_offsets,
)
from gfmcd.dnabert.manifest import write_run_manifest


OUTPUT_DIR = Path(
    "results/dnabert_promoter/frozen_claim_tests/"
    "faithfulness_curve"
)

HEAD_RANKING_PATH = Path(
    "results/dnabert_promoter/causal/head_ablation/"
    "validation_head_scan_256_layers_0_1_2_summary.json"
)

MLP_RANKING_PATH = Path(
    "results/dnabert_promoter/causal/branch_ablation/"
    "validation_localization_256_corrected_summary.json"
)

BASE_HEADS = {
    (0, 8),
    (1, 8),
    (2, 3),
    (2, 9),
}

BASE_MLPS = {
    0,
}

DEFAULT_SIZES = (
    5,
    7,
    10,
    15,
    20,
    25,
    30,
    35,
    39,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        type=Path,
        default=DEFAULT_COHORT,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--output-name",
        default="validation_faithfulness_curve_128",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
    )

    parser.add_argument(
        "--minimum-total-effect",
        type=float,
        default=1e-8,
    )

    return parser.parse_args()


def load_ranked_nodes() -> list[dict[str, object]]:
    head_summary = json.loads(
        HEAD_RANKING_PATH.read_text()
    )

    mlp_summary = json.loads(
        MLP_RANKING_PATH.read_text()
    )

    nodes: list[dict[str, object]] = []

    for row in head_summary["heads"]:
        nodes.append(
            {
                "type": "head",
                "name": row["name"],
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "promoter_mean_damage": float(
                    row["promoter_mean_damage"]
                ),
                "mean_absolute_logit_change": float(
                    row["mean_absolute_logit_change"]
                ),
            }
        )

    for row in mlp_summary["interventions"]:
        if row["branch"] != "mlp":
            continue

        nodes.append(
            {
                "type": "mlp",
                "name": row["name"],
                "layer": int(row["layer"]),
                "head": None,
                "promoter_mean_damage": float(
                    row["promoter_mean_damage"]
                ),
                "mean_absolute_logit_change": float(
                    row["mean_absolute_logit_change"]
                ),
            }
        )

    if len(nodes) != 39:
        raise RuntimeError(
            f"Expected 39 nodes, found {len(nodes)}"
        )

    # More negative promoter_mean_damage means that
    # ablating the node lowered the promoter readout more.
    # Therefore, causally damaging nodes must come first.
    nodes.sort(
        key=lambda row: (
            row["promoter_mean_damage"],
            -row["mean_absolute_logit_change"],
        ),
    )

    return nodes


def build_growth_order(
    ranked_nodes: list[dict[str, object]],
) -> list[dict[str, object]]:
    base_nodes = []
    remaining_nodes = []

    for node in ranked_nodes:
        if node["type"] == "head":
            key = (
                int(node["layer"]),
                int(node["head"]),
            )
            is_base = key in BASE_HEADS
        else:
            is_base = (
                int(node["layer"])
                in BASE_MLPS
            )

        if is_base:
            base_nodes.append(node)
        else:
            remaining_nodes.append(node)

    if len(base_nodes) != 5:
        raise RuntimeError(
            f"Expected 5 base nodes, found "
            f"{len(base_nodes)}"
        )

    base_nodes.sort(
        key=lambda node: (
            str(node["type"]),
            int(node["layer"]),
            (
                int(node["head"])
                if node["head"] is not None
                else -1
            ),
        )
    )

    return base_nodes + remaining_nodes


def circuit_at_size(
    growth_order: list[dict[str, object]],
    size: int,
) -> tuple[
    frozenset[tuple[int, int]],
    frozenset[int],
    list[dict[str, object]],
]:
    selected = growth_order[:size]

    heads = frozenset(
        (
            int(node["layer"]),
            int(node["head"]),
        )
        for node in selected
        if node["type"] == "head"
    )

    mlps = frozenset(
        int(node["layer"])
        for node in selected
        if node["type"] == "mlp"
    )

    return heads, mlps, selected


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sizes = sorted(set(args.sizes))

    if sizes[0] < 5:
        raise ValueError(
            "Minimum circuit size is 5"
        )

    if sizes[-1] > 39:
        raise ValueError(
            "Maximum circuit size is 39"
        )

    ranked_nodes = load_ranked_nodes()
    growth_order = build_growth_order(
        ranked_nodes
    )

    rows = load_promoters(
        args.cohort,
        args.limit,
    )

    print("Rows:", len(rows))
    print("Circuit sizes:", sizes)
    print("Total nodes:", len(growth_order))
    print("Selected layer:", SELECTED_LAYER)

    print("\nGrowth order:")

    for index, node in enumerate(
        growth_order,
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"{node['name']:<24} "
            f"damage="
            f"{node['promoter_mean_damage']:+.6f}"
        )

    adapter = DnaBertAdapter.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=args.device,
    )

    adapter.model.eval()

    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)

    probe = joblib.load(PROBE_PATH)

    clean_sequences = [
        row["clean_sequence"]
        for row in rows
    ]

    corrupt_sequences = [
        get_corrupt_sequence(row)
        for row in rows
    ]

    clean_offsets = []
    corrupt_offsets = []

    print("\nPreparing offsets...")

    for index in range(len(rows)):
        clean_ids, clean_offset_row = (
            tokenize_with_offsets(
                adapter,
                clean_sequences[index],
            )
        )

        corrupt_ids, corrupt_offset_row = (
            tokenize_with_offsets(
                adapter,
                corrupt_sequences[index],
            )
        )

        assert_adapter_tokenization_matches(
            adapter,
            clean_sequences[index],
            clean_ids,
        )

        assert_adapter_tokenization_matches(
            adapter,
            corrupt_sequences[index],
            corrupt_ids,
        )

        clean_offsets.append(clean_offset_row)
        corrupt_offsets.append(corrupt_offset_row)

    clean_attention_outputs = []
    corrupt_attention_outputs = []
    clean_mlp_outputs = []
    corrupt_mlp_outputs = []
    clean_logits = []
    corrupt_logits = []

    print("Capturing clean and corrupt nodes...")

    for index in range(len(rows)):
        (
            clean_attention,
            clean_mlps,
            clean_logit,
        ) = capture_all_components_and_logit(
            adapter,
            probe,
            clean_sequences[index],
        )

        (
            corrupt_attention,
            corrupt_mlps,
            corrupt_logit,
        ) = capture_all_components_and_logit(
            adapter,
            probe,
            corrupt_sequences[index],
        )

        clean_attention_outputs.append(
            clean_attention
        )
        corrupt_attention_outputs.append(
            corrupt_attention
        )
        clean_mlp_outputs.append(clean_mlps)
        corrupt_mlp_outputs.append(corrupt_mlps)
        clean_logits.append(clean_logit)
        corrupt_logits.append(corrupt_logit)

        print(
            f"\rCaptured {index + 1}/{len(rows)}",
            end="",
            flush=True,
        )

    print()

    detail_rows = []
    curve_rows = []

    for size in sizes:
        circuit_heads, circuit_mlps, selected_nodes = (
            circuit_at_size(
                growth_order,
                size,
            )
        )

        print(
            f"\n===== Circuit size {size} ====="
        )
        print("Heads:", len(circuit_heads))
        print("MLPs:", len(circuit_mlps))

        # Temporarily replace the globals read by
        # run_component_patch's hook constructors.
        import run_frozen_component_faithfulness as faith_module

        old_heads = faith_module.CIRCUIT_HEADS
        old_mlps = faith_module.CIRCUIT_MLP_LAYERS

        faith_module.CIRCUIT_HEADS = circuit_heads
        faith_module.CIRCUIT_MLP_LAYERS = circuit_mlps

        faithfulness_values = []
        faithful_amounts = []

        try:
            for index, row in enumerate(rows):
                clean_logit = clean_logits[index]
                corrupt_logit = corrupt_logits[index]
                total_effect = (
                    clean_logit - corrupt_logit
                )

                faithful_logit = run_component_patch(
                    adapter,
                    probe,
                    clean_sequences[index],
                    clean_attention_outputs=(
                        clean_attention_outputs[
                            index
                        ]
                    ),
                    corrupt_attention_outputs=(
                        corrupt_attention_outputs[
                            index
                        ]
                    ),
                    clean_mlp_outputs=(
                        clean_mlp_outputs[index]
                    ),
                    corrupt_mlp_outputs=(
                        corrupt_mlp_outputs[index]
                    ),
                    clean_offsets=clean_offsets[index],
                    corrupt_offsets=(
                        corrupt_offsets[index]
                    ),
                    recipient_offsets=(
                        clean_offsets[index]
                    ),
                    mode="faithfulness",
                )

                faithful_amount = (
                    faithful_logit
                    - corrupt_logit
                )

                faithfulness = (
                    faithful_amount / total_effect
                    if abs(total_effect)
                    > args.minimum_total_effect
                    else float("nan")
                )

                faithfulness_values.append(
                    faithfulness
                )
                faithful_amounts.append(
                    faithful_amount
                )

                detail_rows.append(
                    {
                        "circuit_size": size,
                        "recipient_index": index,
                        "source_index": row[
                            "source_index"
                        ],
                        "clean_logit": clean_logit,
                        "corrupt_logit": corrupt_logit,
                        "total_effect": total_effect,
                        "faithful_logit": (
                            faithful_logit
                        ),
                        "faithful_amount": (
                            faithful_amount
                        ),
                        "faithfulness_fraction": (
                            faithfulness
                        ),
                    }
                )

                print(
                    f"\r{index + 1:03d}/"
                    f"{len(rows)} "
                    f"F={faithfulness:+.3f}",
                    end="",
                    flush=True,
                )

            print()

        finally:
            faith_module.CIRCUIT_HEADS = old_heads
            faith_module.CIRCUIT_MLP_LAYERS = old_mlps

        values = np.asarray(
            faithfulness_values,
            dtype=np.float64,
        )

        amounts = np.asarray(
            faithful_amounts,
            dtype=np.float64,
        )

        summary = finite_summary(values)

        curve_rows.append(
            {
                "circuit_size": size,
                "num_heads": len(circuit_heads),
                "num_mlps": len(circuit_mlps),
                "mean_faithfulness": (
                    summary["mean"]
                ),
                "median_faithfulness": (
                    summary["median"]
                ),
                "std_faithfulness": (
                    summary["std"]
                ),
                "minimum_faithfulness": (
                    summary["minimum"]
                ),
                "maximum_faithfulness": (
                    summary["maximum"]
                ),
                "fraction_positive": float(
                    np.mean(amounts > 0)
                ),
                "selected_nodes": [
                    node["name"]
                    for node in selected_nodes
                ],
            }
        )

        print(
            "Mean faithfulness:",
            summary["mean"],
        )

    summary = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_layer": SELECTED_LAYER,
        "cohort": str(args.cohort),
        "rows": len(rows),
        "ranking_source": {
            "heads": str(HEAD_RANKING_PATH),
            "mlps": str(MLP_RANKING_PATH),
            "metric": (
                "promoter_mean_damage, ascending because "
                "more-negative values indicate greater "
                "promoter damage; mean_absolute_logit_change "
                "used as secondary key"
            ),
        },
        "base_circuit": {
            "heads": [
                {"layer": layer, "head": head}
                for layer, head
                in sorted(BASE_HEADS)
            ],
            "mlp_layers": sorted(BASE_MLPS),
        },
        "growth_order": growth_order,
        "curve": curve_rows,
        "scope_note": (
            "Head-and-MLP component faithfulness "
            "through encoder Layer 2."
        ),
    }

    summary_path = OUTPUT_DIR / (
        f"{args.output_name}_summary.json"
    )

    curve_path = OUTPUT_DIR / (
        f"{args.output_name}_curve.csv"
    )

    detail_path = OUTPUT_DIR / (
        f"{args.output_name}_per_example.csv"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with curve_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "circuit_size",
            "num_heads",
            "num_mlps",
            "mean_faithfulness",
            "median_faithfulness",
            "std_faithfulness",
            "minimum_faithfulness",
            "maximum_faithfulness",
            "fraction_positive",
            "selected_nodes",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in curve_rows:
            serialized = dict(row)
            serialized["selected_nodes"] = json.dumps(
                row["selected_nodes"]
            )
            writer.writerow(serialized)

    with detail_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                detail_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(detail_rows)

    print("\n===== Faithfulness curve =====")

    for row in curve_rows:
        print(
            f"size={row['circuit_size']:02d} "
            f"mean={row['mean_faithfulness']:.6f} "
            f"median={row['median_faithfulness']:.6f}"
        )

    print("\nSaved:", summary_path)
    print("Saved:", curve_path)
    print("Saved:", detail_path)

    # Sibling provenance record. Written after the outputs exist so their hashes are real.
    manifest_path = write_run_manifest(
        summary_path,
        experiment_id="dnabert_promoter_faithfulness_curve",
        config=vars(args),
        outputs=[curve_path, detail_path],
        inputs=[PROBE_PATH],
        probe_path=PROBE_PATH,
        selected_layer=SELECTED_LAYER,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
    )
    print("Saved:", manifest_path)

    print("FROZEN FAITHFULNESS CURVE SUCCEEDED")


if __name__ == "__main__":
    main()
