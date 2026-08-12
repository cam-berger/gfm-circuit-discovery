"""Component-wise donor-class restoration matrix for frozen DNABERT.

This experiment distinguishes:

1. sequence-matched restoration;
2. same-class but sequence-mismatched restoration;
3. opposite-class restoration;
4. zero-component intervention;
5. self-corrupt identity behavior.

The assay uses the fixed Layer-2 masked-mean linear probe.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy.stats import wilcoxon

from gfmcd.adapters.dnabert import DnaBertAdapter

from run_head_restoration_pilot import (
    DEFAULT_COHORT,
    MODEL_NAME,
    MODEL_REVISION,
    OUTPUT_DIR,
    PROBE_PATH,
    assert_adapter_tokenization_matches,
    get_corrupt_sequence,
    tokenize_with_offsets,
)

from run_combined_mlp_head_restoration_controls import (
    SELECTED_LAYER,
    attention_module,
    make_attention_patch_hook,
    make_mlp_patch_hook,
    mlp_module,
    pool_selected_layer,
)


COMPONENT_CONDITIONS: dict[str, dict[str, Any]] = {
    "L0H8": {
        "heads": ((0, 8),),
        "mlp_layer": None,
    },
    "L0_MLP": {
        "heads": (),
        "mlp_layer": 0,
    },
    "L0_MLP_L0H8": {
        "heads": ((0, 8),),
        "mlp_layer": 0,
    },
    "FULL_EARLY_GROUP": {
        "heads": (
            (0, 8),
            (1, 8),
            (2, 9),
            (2, 3),
        ),
        "mlp_layer": 0,
    },
}

DEFAULT_SAME_CLASS_OFFSETS = (1, 3, 5, 7)
DEFAULT_OPPOSITE_CLASS_OFFSETS = (0, 3, 5, 7)


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )

    if not values:
        raise argparse.ArgumentTypeError(
            "Expected at least one integer offset"
        )

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        type=Path,
        default=DEFAULT_COHORT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/dnabert_promoter/"
            "frozen_claim_tests/components/donor_matrix"
        ),
    )

    parser.add_argument(
        "--output-name",
        default="validation_component_donor_matrix_128",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--same-class-offsets",
        type=parse_int_list,
        default=DEFAULT_SAME_CLASS_OFFSETS,
    )

    parser.add_argument(
        "--opposite-class-offsets",
        type=parse_int_list,
        default=DEFAULT_OPPOSITE_CLASS_OFFSETS,
    )

    parser.add_argument(
        "--components",
        default=",".join(COMPONENT_CONDITIONS),
        help=(
            "Comma-separated component conditions. "
            f"Available: {','.join(COMPONENT_CONDITIONS)}"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    args = parser.parse_args()

    args.components = tuple(
        item.strip()
        for item in args.components.split(",")
        if item.strip()
    )

    unknown = [
        item
        for item in args.components
        if item not in COMPONENT_CONDITIONS
    ]

    if unknown:
        raise ValueError(
            f"Unknown component conditions: {unknown}"
        )

    if not args.components:
        raise ValueError(
            "At least one component condition is required"
        )

    return args


def load_class_rows(
    path: Path,
    limit: int,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    promoters = [
        row
        for row in rows
        if int(row["label"]) == 1
    ]

    nonpromoters = [
        row
        for row in rows
        if int(row["label"]) == 0
    ]

    if len(promoters) < limit:
        raise ValueError(
            f"Only {len(promoters)} promoter rows available; "
            f"requested {limit}"
        )

    if len(nonpromoters) < limit:
        raise ValueError(
            f"Only {len(nonpromoters)} non-promoter rows available; "
            f"requested {limit}"
        )

    return promoters[:limit], nonpromoters[:limit]


def required_head_layers() -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                layer
                for condition in COMPONENT_CONDITIONS.values()
                for layer, _head in condition["heads"]
            }
        )
    )


def capture_all_components_and_logit(
    adapter: DnaBertAdapter,
    probe,
    sequence: str,
) -> tuple[
    dict[int, torch.Tensor],
    torch.Tensor,
    float,
]:
    attention_outputs: dict[int, torch.Tensor] = {}
    mlp_output: dict[str, torch.Tensor] = {}
    handles = []

    def make_capture_hook(layer: int):
        def hook(
            _module,
            _inputs,
            output: torch.Tensor,
        ) -> torch.Tensor:
            attention_outputs[layer] = (
                output.detach().clone()
            )
            return output

        return hook

    def capture_mlp_hook(
        _module,
        _inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        mlp_output["output"] = (
            output.detach().clone()
        )
        return output

    for layer in required_head_layers():
        handles.append(
            attention_module(
                adapter,
                layer,
            ).register_forward_hook(
                make_capture_hook(layer)
            )
        )

    handles.append(
        mlp_module(
            adapter,
        ).register_forward_hook(
            capture_mlp_hook
        )
    )

    try:
        tokenized = adapter.tokenize([sequence])

        with torch.inference_mode():
            feature = pool_selected_layer(
                adapter,
                tokenized,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing_layers = [
        layer
        for layer in required_head_layers()
        if layer not in attention_outputs
    ]

    if missing_layers:
        raise RuntimeError(
            f"Missing attention captures: {missing_layers}"
        )

    if "output" not in mlp_output:
        raise RuntimeError(
            "Layer-0 MLP output was not captured"
        )

    logit = float(
        probe.decision_function(
            feature.detach().cpu().numpy()
        )[0]
    )

    return (
        attention_outputs,
        mlp_output["output"],
        logit,
    )


def run_component_patch(
    adapter: DnaBertAdapter,
    probe,
    sequence: str,
    *,
    condition_name: str,
    donor_attention_outputs: dict[
        int,
        torch.Tensor,
    ],
    donor_mlp_output: torch.Tensor,
    donor_offsets: list[tuple[int, int]],
    recipient_offsets: list[tuple[int, int]],
) -> float:
    condition = COMPONENT_CONDITIONS[
        condition_name
    ]

    heads_by_layer: dict[int, list[int]] = (
        defaultdict(list)
    )

    for layer, head in condition["heads"]:
        heads_by_layer[layer].append(head)

    handles = []

    for layer, heads in sorted(
        heads_by_layer.items()
    ):
        handles.append(
            attention_module(
                adapter,
                layer,
            ).register_forward_hook(
                make_attention_patch_hook(
                    heads=heads,
                    donor_output=(
                        donor_attention_outputs[layer]
                    ),
                    donor_offsets=donor_offsets,
                    recipient_offsets=recipient_offsets,
                )
            )
        )

    if condition["mlp_layer"] is not None:
        if condition["mlp_layer"] != 0:
            raise ValueError(
                "Current helper supports only Layer-0 MLP"
            )

        handles.append(
            mlp_module(
                adapter,
            ).register_forward_hook(
                make_mlp_patch_hook(
                    donor_output=donor_mlp_output,
                    donor_offsets=donor_offsets,
                    recipient_offsets=recipient_offsets,
                )
            )
        )

    try:
        tokenized = adapter.tokenize([sequence])

        with torch.inference_mode():
            feature = pool_selected_layer(
                adapter,
                tokenized,
            )
    finally:
        for handle in handles:
            handle.remove()

    return float(
        probe.decision_function(
            feature.detach().cpu().numpy()
        )[0]
    )


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
            "fraction_positive": float("nan"),
            "fraction_negative": float("nan"),
        }

    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "std": (
            float(finite.std(ddof=1))
            if finite.size > 1
            else 0.0
        ),
        "median": float(np.median(finite)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
        "fraction_positive": float(
            np.mean(finite > 0)
        ),
        "fraction_negative": float(
            np.mean(finite < 0)
        ),
    }


def paired_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    alternative: str,
) -> dict[str, float]:
    valid = np.isfinite(a) & np.isfinite(b)

    a_valid = a[valid]
    b_valid = b[valid]
    delta = a_valid - b_valid

    if delta.size == 0:
        return {
            "n": 0,
            "mean_difference": float("nan"),
            "median_difference": float("nan"),
            "fraction_a_greater": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "p_value": float("nan"),
        }

    try:
        result = wilcoxon(
            delta,
            alternative=alternative,
            zero_method="wilcox",
        )

        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except ValueError:
        statistic = float("nan")
        p_value = 1.0

    return {
        "n": int(delta.size),
        "mean_a": float(a_valid.mean()),
        "mean_b": float(b_valid.mean()),
        "mean_difference": float(delta.mean()),
        "median_difference": float(
            np.median(delta)
        ),
        "fraction_a_greater": float(
            np.mean(delta > 0)
        ),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def holm_adjust(
    p_values: list[float],
) -> list[float]:
    p_array = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        p_array.shape,
        np.nan,
        dtype=float,
    )

    finite_indices = np.flatnonzero(
        np.isfinite(p_array)
    )

    if finite_indices.size == 0:
        return adjusted.tolist()

    finite_values = p_array[finite_indices]
    order = np.argsort(finite_values)

    running = 0.0
    m = len(finite_values)

    for rank, ordered_position in enumerate(order):
        original_index = finite_indices[
            ordered_position
        ]

        value = min(
            1.0,
            (m - rank)
            * p_array[original_index],
        )

        running = max(running, value)
        adjusted[original_index] = running

    return adjusted.tolist()


def recovery_fraction(
    patched_logit: float,
    corrupt_logit: float,
    total_damage: float,
) -> float:
    if abs(total_damage) <= 1e-8:
        return float("nan")

    return (
        patched_logit - corrupt_logit
    ) / total_damage


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    promoters, nonpromoters = load_class_rows(
        args.cohort,
        args.limit,
    )

    for offset in args.same_class_offsets:
        if offset % len(promoters) == 0:
            raise ValueError(
                f"Same-class offset {offset} maps "
                "recipients to themselves"
            )

    print("Promoter recipients:", len(promoters))
    print("Non-promoter donors:", len(nonpromoters))
    print(
        "Same-class offsets:",
        args.same_class_offsets,
    )
    print(
        "Opposite-class offsets:",
        args.opposite_class_offsets,
    )
    print(
        "Component conditions:",
        args.components,
    )

    print("\nLoading frozen DNABERT...")

    adapter = DnaBertAdapter.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=args.device,
    )

    adapter.model.eval()

    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)

    probe = joblib.load(PROBE_PATH)

    promoter_clean_sequences = [
        row["clean_sequence"]
        for row in promoters
    ]

    promoter_corrupt_sequences = [
        get_corrupt_sequence(row)
        for row in promoters
    ]

    nonpromoter_sequences = [
        row["clean_sequence"]
        for row in nonpromoters
    ]

    promoter_clean_offsets = []
    promoter_corrupt_offsets = []
    nonpromoter_offsets = []

    print("\nPreparing tokenizer offsets...")

    for index in range(args.limit):
        clean_ids, clean_offsets = (
            tokenize_with_offsets(
                adapter,
                promoter_clean_sequences[index],
            )
        )

        corrupt_ids, corrupt_offsets = (
            tokenize_with_offsets(
                adapter,
                promoter_corrupt_sequences[index],
            )
        )

        negative_ids, negative_offsets = (
            tokenize_with_offsets(
                adapter,
                nonpromoter_sequences[index],
            )
        )

        assert_adapter_tokenization_matches(
            adapter,
            promoter_clean_sequences[index],
            clean_ids,
        )

        assert_adapter_tokenization_matches(
            adapter,
            promoter_corrupt_sequences[index],
            corrupt_ids,
        )

        assert_adapter_tokenization_matches(
            adapter,
            nonpromoter_sequences[index],
            negative_ids,
        )

        promoter_clean_offsets.append(
            clean_offsets
        )
        promoter_corrupt_offsets.append(
            corrupt_offsets
        )
        nonpromoter_offsets.append(
            negative_offsets
        )

    promoter_clean_attention = []
    promoter_corrupt_attention = []
    nonpromoter_attention = []

    promoter_clean_mlp = []
    promoter_corrupt_mlp = []
    nonpromoter_mlp = []

    promoter_clean_logits = []
    promoter_corrupt_logits = []
    nonpromoter_logits = []

    print("\nCapturing activations...")

    for index in range(args.limit):
        (
            clean_attention,
            clean_mlp,
            clean_logit,
        ) = capture_all_components_and_logit(
            adapter,
            probe,
            promoter_clean_sequences[index],
        )

        (
            corrupt_attention,
            corrupt_mlp,
            corrupt_logit,
        ) = capture_all_components_and_logit(
            adapter,
            probe,
            promoter_corrupt_sequences[index],
        )

        (
            negative_attention,
            negative_mlp,
            negative_logit,
        ) = capture_all_components_and_logit(
            adapter,
            probe,
            nonpromoter_sequences[index],
        )

        promoter_clean_attention.append(
            clean_attention
        )
        promoter_corrupt_attention.append(
            corrupt_attention
        )
        nonpromoter_attention.append(
            negative_attention
        )

        promoter_clean_mlp.append(clean_mlp)
        promoter_corrupt_mlp.append(
            corrupt_mlp
        )
        nonpromoter_mlp.append(negative_mlp)

        promoter_clean_logits.append(
            clean_logit
        )
        promoter_corrupt_logits.append(
            corrupt_logit
        )
        nonpromoter_logits.append(
            negative_logit
        )

        print(
            f"\rCaptured {index + 1}/"
            f"{args.limit}",
            end="",
            flush=True,
        )

    print()

    result_rows: list[dict[str, Any]] = []

    total_patch_runs = (
        len(args.components)
        * args.limit
        * (
            1
            + len(args.same_class_offsets)
            + len(args.opposite_class_offsets)
            + 2
        )
    )

    completed_patch_runs = 0

    for condition_name in args.components:
        print(
            "\nRunning component condition:",
            condition_name,
        )

        for recipient_index, promoter_row in enumerate(
            promoters
        ):
            recipient_sequence = (
                promoter_corrupt_sequences[
                    recipient_index
                ]
            )

            recipient_offsets = (
                promoter_corrupt_offsets[
                    recipient_index
                ]
            )

            clean_logit = promoter_clean_logits[
                recipient_index
            ]

            corrupt_logit = (
                promoter_corrupt_logits[
                    recipient_index
                ]
            )

            total_damage = (
                clean_logit - corrupt_logit
            )

            matched_logit = run_component_patch(
                adapter,
                probe,
                recipient_sequence,
                condition_name=condition_name,
                donor_attention_outputs=(
                    promoter_clean_attention[
                        recipient_index
                    ]
                ),
                donor_mlp_output=(
                    promoter_clean_mlp[
                        recipient_index
                    ]
                ),
                donor_offsets=(
                    promoter_clean_offsets[
                        recipient_index
                    ]
                ),
                recipient_offsets=recipient_offsets,
            )

            result_rows.append(
                {
                    "recipient_index": (
                        recipient_index
                    ),
                    "recipient_source_index": (
                        promoter_row["source_index"]
                    ),
                    "component_condition": (
                        condition_name
                    ),
                    "donor_type": "matched_promoter",
                    "donor_offset": 0,
                    "donor_index": recipient_index,
                    "donor_source_index": (
                        promoter_row["source_index"]
                    ),
                    "donor_clean_logit": clean_logit,
                    "clean_logit": clean_logit,
                    "corrupt_logit": corrupt_logit,
                    "total_corruption_damage": (
                        total_damage
                    ),
                    "patched_logit": matched_logit,
                    "restored_amount": (
                        matched_logit
                        - corrupt_logit
                    ),
                    "recovery_fraction": (
                        recovery_fraction(
                            matched_logit,
                            corrupt_logit,
                            total_damage,
                        )
                    ),
                }
            )

            completed_patch_runs += 1

            for offset in args.same_class_offsets:
                donor_index = (
                    recipient_index + offset
                ) % args.limit

                patched_logit = run_component_patch(
                    adapter,
                    probe,
                    recipient_sequence,
                    condition_name=condition_name,
                    donor_attention_outputs=(
                        promoter_clean_attention[
                            donor_index
                        ]
                    ),
                    donor_mlp_output=(
                        promoter_clean_mlp[
                            donor_index
                        ]
                    ),
                    donor_offsets=(
                        promoter_clean_offsets[
                            donor_index
                        ]
                    ),
                    recipient_offsets=(
                        recipient_offsets
                    ),
                )

                result_rows.append(
                    {
                        "recipient_index": (
                            recipient_index
                        ),
                        "recipient_source_index": (
                            promoter_row[
                                "source_index"
                            ]
                        ),
                        "component_condition": (
                            condition_name
                        ),
                        "donor_type": (
                            "same_class_promoter"
                        ),
                        "donor_offset": offset,
                        "donor_index": donor_index,
                        "donor_source_index": (
                            promoters[donor_index][
                                "source_index"
                            ]
                        ),
                        "donor_clean_logit": (
                            promoter_clean_logits[
                                donor_index
                            ]
                        ),
                        "clean_logit": clean_logit,
                        "corrupt_logit": (
                            corrupt_logit
                        ),
                        "total_corruption_damage": (
                            total_damage
                        ),
                        "patched_logit": (
                            patched_logit
                        ),
                        "restored_amount": (
                            patched_logit
                            - corrupt_logit
                        ),
                        "recovery_fraction": (
                            recovery_fraction(
                                patched_logit,
                                corrupt_logit,
                                total_damage,
                            )
                        ),
                    }
                )

                completed_patch_runs += 1

            for offset in (
                args.opposite_class_offsets
            ):
                donor_index = (
                    recipient_index + offset
                ) % args.limit

                patched_logit = run_component_patch(
                    adapter,
                    probe,
                    recipient_sequence,
                    condition_name=condition_name,
                    donor_attention_outputs=(
                        nonpromoter_attention[
                            donor_index
                        ]
                    ),
                    donor_mlp_output=(
                        nonpromoter_mlp[
                            donor_index
                        ]
                    ),
                    donor_offsets=(
                        nonpromoter_offsets[
                            donor_index
                        ]
                    ),
                    recipient_offsets=(
                        recipient_offsets
                    ),
                )

                result_rows.append(
                    {
                        "recipient_index": (
                            recipient_index
                        ),
                        "recipient_source_index": (
                            promoter_row[
                                "source_index"
                            ]
                        ),
                        "component_condition": (
                            condition_name
                        ),
                        "donor_type": (
                            "opposite_class_nonpromoter"
                        ),
                        "donor_offset": offset,
                        "donor_index": donor_index,
                        "donor_source_index": (
                            nonpromoters[donor_index][
                                "source_index"
                            ]
                        ),
                        "donor_clean_logit": (
                            nonpromoter_logits[
                                donor_index
                            ]
                        ),
                        "clean_logit": clean_logit,
                        "corrupt_logit": (
                            corrupt_logit
                        ),
                        "total_corruption_damage": (
                            total_damage
                        ),
                        "patched_logit": (
                            patched_logit
                        ),
                        "restored_amount": (
                            patched_logit
                            - corrupt_logit
                        ),
                        "recovery_fraction": (
                            recovery_fraction(
                                patched_logit,
                                corrupt_logit,
                                total_damage,
                            )
                        ),
                    }
                )

                completed_patch_runs += 1

            zero_attention = {
                layer: torch.zeros_like(output)
                for layer, output in (
                    promoter_corrupt_attention[
                        recipient_index
                    ].items()
                )
            }

            zero_mlp = torch.zeros_like(
                promoter_corrupt_mlp[
                    recipient_index
                ]
            )

            zero_logit = run_component_patch(
                adapter,
                probe,
                recipient_sequence,
                condition_name=condition_name,
                donor_attention_outputs=(
                    zero_attention
                ),
                donor_mlp_output=zero_mlp,
                donor_offsets=recipient_offsets,
                recipient_offsets=recipient_offsets,
            )

            result_rows.append(
                {
                    "recipient_index": recipient_index,
                    "recipient_source_index": (
                        promoter_row["source_index"]
                    ),
                    "component_condition": (
                        condition_name
                    ),
                    "donor_type": "zero_activation",
                    "donor_offset": 0,
                    "donor_index": -1,
                    "donor_source_index": "",
                    "donor_clean_logit": (
                        float("nan")
                    ),
                    "clean_logit": clean_logit,
                    "corrupt_logit": corrupt_logit,
                    "total_corruption_damage": (
                        total_damage
                    ),
                    "patched_logit": zero_logit,
                    "restored_amount": (
                        zero_logit - corrupt_logit
                    ),
                    "recovery_fraction": (
                        recovery_fraction(
                            zero_logit,
                            corrupt_logit,
                            total_damage,
                        )
                    ),
                }
            )

            completed_patch_runs += 1

            self_corrupt_logit = (
                run_component_patch(
                    adapter,
                    probe,
                    recipient_sequence,
                    condition_name=condition_name,
                    donor_attention_outputs=(
                        promoter_corrupt_attention[
                            recipient_index
                        ]
                    ),
                    donor_mlp_output=(
                        promoter_corrupt_mlp[
                            recipient_index
                        ]
                    ),
                    donor_offsets=(
                        recipient_offsets
                    ),
                    recipient_offsets=(
                        recipient_offsets
                    ),
                )
            )

            result_rows.append(
                {
                    "recipient_index": recipient_index,
                    "recipient_source_index": (
                        promoter_row["source_index"]
                    ),
                    "component_condition": (
                        condition_name
                    ),
                    "donor_type": "self_corrupt",
                    "donor_offset": 0,
                    "donor_index": recipient_index,
                    "donor_source_index": (
                        promoter_row["source_index"]
                    ),
                    "donor_clean_logit": (
                        corrupt_logit
                    ),
                    "clean_logit": clean_logit,
                    "corrupt_logit": corrupt_logit,
                    "total_corruption_damage": (
                        total_damage
                    ),
                    "patched_logit": (
                        self_corrupt_logit
                    ),
                    "restored_amount": (
                        self_corrupt_logit
                        - corrupt_logit
                    ),
                    "recovery_fraction": (
                        recovery_fraction(
                            self_corrupt_logit,
                            corrupt_logit,
                            total_damage,
                        )
                    ),
                }
            )

            completed_patch_runs += 1

            print(
                f"\r{condition_name}: "
                f"recipient {recipient_index + 1}/"
                f"{args.limit}; "
                f"patch runs {completed_patch_runs}/"
                f"{total_patch_runs}",
                end="",
                flush=True,
            )

        print()

    per_example_path = (
        args.output_dir
        / f"{args.output_name}_per_example.csv"
    )

    with per_example_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                result_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(result_rows)

    summary: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_layer": SELECTED_LAYER,
        "rows_per_class": args.limit,
        "same_class_offsets": list(
            args.same_class_offsets
        ),
        "opposite_class_offsets": list(
            args.opposite_class_offsets
        ),
        "component_conditions": {
            name: COMPONENT_CONDITIONS[name]
            for name in args.components
        },
        "identity_max_absolute_difference": (
            float(
                max(
                    abs(row["restored_amount"])
                    for row in result_rows
                    if row["donor_type"]
                    == "self_corrupt"
                )
            )
        ),
        "results": {},
    }

    statistical_records = []

    for condition_name in args.components:
        condition_rows = [
            row
            for row in result_rows
            if row["component_condition"]
            == condition_name
        ]

        summary["results"][condition_name] = {}

        donor_types = sorted(
            {
                row["donor_type"]
                for row in condition_rows
            }
        )

        for donor_type in donor_types:
            donor_rows = [
                row
                for row in condition_rows
                if row["donor_type"]
                == donor_type
            ]

            recovery = np.asarray(
                [
                    row["recovery_fraction"]
                    for row in donor_rows
                ],
                dtype=float,
            )

            restored = np.asarray(
                [
                    row["restored_amount"]
                    for row in donor_rows
                ],
                dtype=float,
            )

            summary["results"][
                condition_name
            ][donor_type] = {
                "recovery_fraction": (
                    summarize(recovery)
                ),
                "restored_amount": (
                    summarize(restored)
                ),
            }

        matched_by_recipient = {
            row["recipient_index"]: (
                row["recovery_fraction"]
            )
            for row in condition_rows
            if row["donor_type"]
            == "matched_promoter"
        }

        same_by_recipient = defaultdict(list)
        opposite_by_recipient = defaultdict(list)

        for row in condition_rows:
            if row["donor_type"] == (
                "same_class_promoter"
            ):
                same_by_recipient[
                    row["recipient_index"]
                ].append(
                    row["recovery_fraction"]
                )

            if row["donor_type"] == (
                "opposite_class_nonpromoter"
            ):
                opposite_by_recipient[
                    row["recipient_index"]
                ].append(
                    row["recovery_fraction"]
                )

        recipient_indices = sorted(
            matched_by_recipient
        )

        matched = np.asarray(
            [
                matched_by_recipient[index]
                for index in recipient_indices
            ],
            dtype=float,
        )

        same_class_mean = np.asarray(
            [
                np.nanmean(
                    same_by_recipient[index]
                )
                for index in recipient_indices
            ],
            dtype=float,
        )

        opposite_class_mean = np.asarray(
            [
                np.nanmean(
                    opposite_by_recipient[index]
                )
                for index in recipient_indices
            ],
            dtype=float,
        )

        comparisons = {
            "matched_vs_same_class": (
                matched,
                same_class_mean,
            ),
            "same_class_vs_opposite_class": (
                same_class_mean,
                opposite_class_mean,
            ),
            "matched_vs_opposite_class": (
                matched,
                opposite_class_mean,
            ),
        }

        for comparison_name, (
            first,
            second,
        ) in comparisons.items():
            values = paired_test(
                first,
                second,
                alternative="greater",
            )

            statistical_records.append(
                {
                    "component_condition": (
                        condition_name
                    ),
                    "comparison": (
                        comparison_name
                    ),
                    **values,
                }
            )

    p_values = [
        record["p_value"]
        for record in statistical_records
    ]

    adjusted_values = holm_adjust(
        p_values
    )

    for record, adjusted in zip(
        statistical_records,
        adjusted_values,
        strict=True,
    ):
        record["holm_adjusted_p"] = adjusted

    summary_path = (
        args.output_dir
        / f"{args.output_name}_summary.json"
    )

    statistics_path = (
        args.output_dir
        / f"{args.output_name}_statistics.json"
    )

    manifest_path = (
        args.output_dir
        / f"{args.output_name}_manifest.json"
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

    statistics_path.write_text(
        json.dumps(
            statistical_records,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_layer": SELECTED_LAYER,
        "cohort": str(args.cohort),
        "limit": args.limit,
        "same_class_offsets": list(
            args.same_class_offsets
        ),
        "opposite_class_offsets": list(
            args.opposite_class_offsets
        ),
        "component_conditions": {
            name: COMPONENT_CONDITIONS[name]
            for name in args.components
        },
        "per_example_file": str(
            per_example_path
        ),
        "summary_file": str(summary_path),
        "statistics_file": str(
            statistics_path
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n===== Donor matrix summary =====")

    for condition_name in args.components:
        condition_summary = summary[
            "results"
        ][condition_name]

        print("\n", condition_name)

        for donor_type in (
            "matched_promoter",
            "same_class_promoter",
            "opposite_class_nonpromoter",
            "zero_activation",
            "self_corrupt",
        ):
            values = condition_summary[
                donor_type
            ]["recovery_fraction"]

            print(
                f"  {donor_type}: "
                f"mean={values['mean']:.6f}, "
                f"median={values['median']:.6f}, "
                f"n={values['n']}"
            )

    print(
        "\nIdentity maximum absolute difference:",
        summary[
            "identity_max_absolute_difference"
        ],
    )

    print("\nStatistics:")

    for record in statistical_records:
        print(
            record["component_condition"],
            record["comparison"],
            "mean_difference=",
            f"{record['mean_difference']:.6f}",
            "p=",
            f"{record['p_value']:.6g}",
            "Holm=",
            f"{record['holm_adjusted_p']:.6g}",
        )

    print("\nSaved:", per_example_path)
    print("Saved:", summary_path)
    print("Saved:", statistics_path)
    print("Saved:", manifest_path)
    print("COMPONENT DONOR MATRIX SUCCEEDED")


if __name__ == "__main__":
    main()
