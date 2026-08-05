"""Measure head-and-MLP complement faithfulness in frozen DNABERT.

The candidate circuit is kept clean:
- Layer-0 MLP
- L0H8
- L1H8
- L2H9
- L2H3

The component complement is patched to corrupt:
- every other attention head in encoder Layers 0--2
- Layer-1 MLP
- Layer-2 MLP

Important:
This is component-level faithfulness over attention-head and MLP outputs.
Embeddings and residual-stream inputs are not independently patched.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch

from gfmcd.adapters.dnabert import DnaBertAdapter

from run_combined_mlp_head_restoration_controls import (
    HEAD_GROUP,
    attention_module,
    pool_selected_layer,
    summarize,
)

from run_head_restoration_pilot import (
    DEFAULT_COHORT,
    HEAD_DIM,
    HIDDEN_SIZE,
    MODEL_NAME,
    MODEL_REVISION,
    NUM_HEADS,
    PROBE_PATH,
    SELECTED_LAYER,
    assert_adapter_tokenization_matches,
    build_overlap_mapping,
    get_corrupt_sequence,
    tokenize_with_offsets,
)


OUTPUT_DIR = Path(
    "results/dnabert_promoter/frozen_claim_tests/"
    "component_faithfulness"
)

PATCHED_LAYERS = tuple(
    range(SELECTED_LAYER + 1)
)

CIRCUIT_HEADS = frozenset(HEAD_GROUP)
CIRCUIT_MLP_LAYERS = frozenset({0})


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
        default="validation_component_faithfulness_128",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--minimum-total-effect",
        type=float,
        default=1e-8,
    )

    return parser.parse_args()


def load_promoters(
    path: Path,
    limit: int,
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    promoters = [
        row
        for row in rows
        if int(row["label"]) == 1
    ]

    if len(promoters) < limit:
        raise ValueError(
            f"Requested {limit} promoters but only "
            f"{len(promoters)} are available"
        )

    return promoters[:limit]


def mlp_module(
    adapter: DnaBertAdapter,
    layer: int,
):
    return (
        adapter.model.encoder.layer[layer]
        .mlp.wo
    )


def capture_all_components_and_logit(
    adapter: DnaBertAdapter,
    probe,
    sequence: str,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    float,
]:
    attention_outputs: dict[int, torch.Tensor] = {}
    mlp_outputs: dict[int, torch.Tensor] = {}
    handles = []

    def make_attention_capture(layer: int):
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

    def make_mlp_capture(layer: int):
        def hook(
            _module,
            _inputs,
            output: torch.Tensor,
        ) -> torch.Tensor:
            mlp_outputs[layer] = (
                output.detach().clone()
            )
            return output

        return hook

    for layer in PATCHED_LAYERS:
        handles.append(
            attention_module(
                adapter,
                layer,
            ).register_forward_hook(
                make_attention_capture(layer)
            )
        )

        handles.append(
            mlp_module(
                adapter,
                layer,
            ).register_forward_hook(
                make_mlp_capture(layer)
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

    missing_attention = [
        layer
        for layer in PATCHED_LAYERS
        if layer not in attention_outputs
    ]

    missing_mlps = [
        layer
        for layer in PATCHED_LAYERS
        if layer not in mlp_outputs
    ]

    if missing_attention:
        raise RuntimeError(
            f"Missing attention captures: "
            f"{missing_attention}"
        )

    if missing_mlps:
        raise RuntimeError(
            f"Missing MLP captures: {missing_mlps}"
        )

    logit = float(
        probe.decision_function(
            feature.detach().cpu().numpy()
        )[0]
    )

    return attention_outputs, mlp_outputs, logit


def mapped_vector(
    *,
    donor_output: torch.Tensor,
    contributors: list[tuple[int, float]],
    start: int,
    end: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    restored = torch.zeros(
        end - start,
        dtype=dtype,
        device=device,
    )

    for donor_token, weight in contributors:
        restored += (
            float(weight)
            * donor_output[
                donor_token,
                start:end,
            ].to(
                device=device,
                dtype=dtype,
            )
        )

    return restored


def mapped_full_vector(
    *,
    donor_output: torch.Tensor,
    contributors: list[tuple[int, float]],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    restored = torch.zeros(
        HIDDEN_SIZE,
        dtype=dtype,
        device=device,
    )

    for donor_token, weight in contributors:
        restored += (
            float(weight)
            * donor_output[
                donor_token
            ].to(
                device=device,
                dtype=dtype,
            )
        )

    return restored


def make_mixed_attention_hook(
    *,
    layer: int,
    clean_output: torch.Tensor,
    corrupt_output: torch.Tensor,
    clean_offsets: list[tuple[int, int]],
    corrupt_offsets: list[tuple[int, int]],
    recipient_offsets: list[tuple[int, int]],
    mode: str,
):
    clean_mapping = build_overlap_mapping(
        clean_offsets,
        recipient_offsets,
    )

    corrupt_mapping = build_overlap_mapping(
        corrupt_offsets,
        recipient_offsets,
    )

    def hook(
        _module,
        _inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if output.ndim != 2:
            raise ValueError(
                "Expected packed attention output"
            )

        if output.shape[-1] != HIDDEN_SIZE:
            raise ValueError(
                f"Expected hidden size {HIDDEN_SIZE}"
            )

        patched = output.clone()

        for recipient_token in range(
            len(recipient_offsets)
        ):
            for head in range(NUM_HEADS):
                if mode == "faithfulness":
                    use_clean = (
                        (layer, head)
                        in CIRCUIT_HEADS
                    )
                elif mode == "all_clean":
                    use_clean = True
                elif mode == "all_corrupt":
                    use_clean = False
                else:
                    raise ValueError(
                        f"Unknown mode: {mode}"
                    )

                if use_clean:
                    donor_output = clean_output
                    contributors = clean_mapping[
                        recipient_token
                    ]
                else:
                    donor_output = corrupt_output
                    contributors = corrupt_mapping[
                        recipient_token
                    ]

                if not contributors:
                    continue

                start = head * HEAD_DIM
                end = start + HEAD_DIM

                patched[
                    recipient_token,
                    start:end,
                ] = mapped_vector(
                    donor_output=donor_output,
                    contributors=contributors,
                    start=start,
                    end=end,
                    dtype=output.dtype,
                    device=output.device,
                )

        return patched

    return hook


def make_mixed_mlp_hook(
    *,
    layer: int,
    clean_output: torch.Tensor,
    corrupt_output: torch.Tensor,
    clean_offsets: list[tuple[int, int]],
    corrupt_offsets: list[tuple[int, int]],
    recipient_offsets: list[tuple[int, int]],
    mode: str,
):
    clean_mapping = build_overlap_mapping(
        clean_offsets,
        recipient_offsets,
    )

    corrupt_mapping = build_overlap_mapping(
        corrupt_offsets,
        recipient_offsets,
    )

    def hook(
        _module,
        _inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if output.ndim != 2:
            raise ValueError(
                "Expected packed MLP output"
            )

        if output.shape[-1] != HIDDEN_SIZE:
            raise ValueError(
                f"Expected hidden size {HIDDEN_SIZE}"
            )

        patched = output.clone()

        if mode == "faithfulness":
            use_clean = (
                layer in CIRCUIT_MLP_LAYERS
            )
        elif mode == "all_clean":
            use_clean = True
        elif mode == "all_corrupt":
            use_clean = False
        else:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        donor_output = (
            clean_output
            if use_clean
            else corrupt_output
        )

        mapping = (
            clean_mapping
            if use_clean
            else corrupt_mapping
        )

        for recipient_token, contributors in enumerate(
            mapping
        ):
            if not contributors:
                continue

            patched[recipient_token] = (
                mapped_full_vector(
                    donor_output=donor_output,
                    contributors=contributors,
                    dtype=output.dtype,
                    device=output.device,
                )
            )

        return patched

    return hook


def run_component_patch(
    adapter: DnaBertAdapter,
    probe,
    recipient_sequence: str,
    *,
    clean_attention_outputs: dict[
        int,
        torch.Tensor,
    ],
    corrupt_attention_outputs: dict[
        int,
        torch.Tensor,
    ],
    clean_mlp_outputs: dict[
        int,
        torch.Tensor,
    ],
    corrupt_mlp_outputs: dict[
        int,
        torch.Tensor,
    ],
    clean_offsets: list[tuple[int, int]],
    corrupt_offsets: list[tuple[int, int]],
    recipient_offsets: list[tuple[int, int]],
    mode: str,
) -> float:
    handles = []

    for layer in PATCHED_LAYERS:
        handles.append(
            attention_module(
                adapter,
                layer,
            ).register_forward_hook(
                make_mixed_attention_hook(
                    layer=layer,
                    clean_output=(
                        clean_attention_outputs[
                            layer
                        ]
                    ),
                    corrupt_output=(
                        corrupt_attention_outputs[
                            layer
                        ]
                    ),
                    clean_offsets=clean_offsets,
                    corrupt_offsets=corrupt_offsets,
                    recipient_offsets=recipient_offsets,
                    mode=mode,
                )
            )
        )

        handles.append(
            mlp_module(
                adapter,
                layer,
            ).register_forward_hook(
                make_mixed_mlp_hook(
                    layer=layer,
                    clean_output=(
                        clean_mlp_outputs[layer]
                    ),
                    corrupt_output=(
                        corrupt_mlp_outputs[layer]
                    ),
                    clean_offsets=clean_offsets,
                    corrupt_offsets=corrupt_offsets,
                    recipient_offsets=recipient_offsets,
                    mode=mode,
                )
            )
        )

    try:
        tokenized = adapter.tokenize(
            [recipient_sequence]
        )

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


def finite_summary(
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

    result = summarize(finite)
    result["n"] = int(finite.size)
    return result


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_promoters(
        args.cohort,
        args.limit,
    )

    print("Rows:", len(rows))
    print("Selected layer:", SELECTED_LAYER)
    print("Patched layers:", PATCHED_LAYERS)
    print("Candidate heads:", sorted(CIRCUIT_HEADS))
    print(
        "Candidate MLP layers:",
        sorted(CIRCUIT_MLP_LAYERS),
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

    print("\nRunning complement interventions...")

    result_rows = []

    for index, row in enumerate(rows):
        clean_logit = clean_logits[index]
        corrupt_logit = corrupt_logits[index]
        total_effect = clean_logit - corrupt_logit

        faithful_logit = run_component_patch(
            adapter,
            probe,
            clean_sequences[index],
            clean_attention_outputs=(
                clean_attention_outputs[index]
            ),
            corrupt_attention_outputs=(
                corrupt_attention_outputs[index]
            ),
            clean_mlp_outputs=(
                clean_mlp_outputs[index]
            ),
            corrupt_mlp_outputs=(
                corrupt_mlp_outputs[index]
            ),
            clean_offsets=clean_offsets[index],
            corrupt_offsets=corrupt_offsets[index],
            recipient_offsets=clean_offsets[index],
            mode="faithfulness",
        )

        all_clean_logit = run_component_patch(
            adapter,
            probe,
            clean_sequences[index],
            clean_attention_outputs=(
                clean_attention_outputs[index]
            ),
            corrupt_attention_outputs=(
                corrupt_attention_outputs[index]
            ),
            clean_mlp_outputs=(
                clean_mlp_outputs[index]
            ),
            corrupt_mlp_outputs=(
                corrupt_mlp_outputs[index]
            ),
            clean_offsets=clean_offsets[index],
            corrupt_offsets=corrupt_offsets[index],
            recipient_offsets=clean_offsets[index],
            mode="all_clean",
        )

        all_corrupt_logit = run_component_patch(
            adapter,
            probe,
            corrupt_sequences[index],
            clean_attention_outputs=(
                clean_attention_outputs[index]
            ),
            corrupt_attention_outputs=(
                corrupt_attention_outputs[index]
            ),
            clean_mlp_outputs=(
                clean_mlp_outputs[index]
            ),
            corrupt_mlp_outputs=(
                corrupt_mlp_outputs[index]
            ),
            clean_offsets=clean_offsets[index],
            corrupt_offsets=corrupt_offsets[index],
            recipient_offsets=corrupt_offsets[index],
            mode="all_corrupt",
        )

        faithful_amount = (
            faithful_logit - corrupt_logit
        )

        faithfulness_fraction = (
            faithful_amount / total_effect
            if abs(total_effect)
            > args.minimum_total_effect
            else float("nan")
        )

        result_rows.append(
            {
                "recipient_index": index,
                "source_index": row["source_index"],
                "clean_logit": clean_logit,
                "corrupt_logit": corrupt_logit,
                "total_effect": total_effect,
                "faithful_logit": faithful_logit,
                "faithful_amount": faithful_amount,
                "faithfulness_fraction": (
                    faithfulness_fraction
                ),
                "all_clean_logit": all_clean_logit,
                "all_clean_minus_baseline": (
                    all_clean_logit - clean_logit
                ),
                "all_corrupt_logit": (
                    all_corrupt_logit
                ),
                "all_corrupt_minus_baseline": (
                    all_corrupt_logit
                    - corrupt_logit
                ),
            }
        )

        print(
            f"{index + 1:03d}/{len(rows)} "
            f"TE={total_effect:+.3f} "
            f"F={faithfulness_fraction:+.3f}"
        )

    faithfulness = np.asarray(
        [
            row["faithfulness_fraction"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    faithful_amounts = np.asarray(
        [
            row["faithful_amount"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    all_clean_changes = np.asarray(
        [
            row["all_clean_minus_baseline"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    all_corrupt_changes = np.asarray(
        [
            row["all_corrupt_minus_baseline"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    summary = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_layer": SELECTED_LAYER,
        "cohort": str(args.cohort),
        "rows": len(rows),
        "patched_layers": list(PATCHED_LAYERS),
        "candidate_circuit": {
            "heads": [
                {
                    "layer": layer,
                    "head": head,
                }
                for layer, head
                in sorted(CIRCUIT_HEADS)
            ],
            "mlp_layers": sorted(
                CIRCUIT_MLP_LAYERS
            ),
        },
        "complement": {
            "attention_heads": (
                "All heads in Layers 0--2 "
                "except candidate heads"
            ),
            "mlp_layers": [
                layer
                for layer in PATCHED_LAYERS
                if layer
                not in CIRCUIT_MLP_LAYERS
            ],
        },
        "scope_note": (
            "Component-level faithfulness over "
            "attention.self head slices and mlp.wo "
            "outputs. Embeddings and residual-stream "
            "inputs are not independently patched."
        ),
        "definition": (
            "(faithful_logit - corrupt_logit) "
            "/ (clean_logit - corrupt_logit)"
        ),
        "faithful_amount": finite_summary(
            faithful_amounts
        ),
        "faithfulness_fraction": finite_summary(
            faithfulness
        ),
        "fraction_positive_faithfulness": float(
            np.mean(faithful_amounts > 0)
        ),
        "all_clean_max_absolute_difference": (
            float(
                np.max(
                    np.abs(all_clean_changes)
                )
            )
        ),
        "all_corrupt_max_absolute_difference": (
            float(
                np.max(
                    np.abs(all_corrupt_changes)
                )
            )
        ),
    }

    summary_path = OUTPUT_DIR / (
        f"{args.output_name}_summary.json"
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

    with detail_path.open(
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

    print("\n===== Component faithfulness =====")
    print(
        "Mean faithfulness:",
        summary[
            "faithfulness_fraction"
        ]["mean"],
    )
    print(
        "Median faithfulness:",
        summary[
            "faithfulness_fraction"
        ]["median"],
    )
    print(
        "Positive faithful amount:",
        summary[
            "fraction_positive_faithfulness"
        ],
    )
    print(
        "All-clean control:",
        summary[
            "all_clean_max_absolute_difference"
        ],
    )
    print(
        "All-corrupt control:",
        summary[
            "all_corrupt_max_absolute_difference"
        ],
    )
    print("\nSaved:", summary_path)
    print("Saved:", detail_path)
    print(
        "FROZEN COMPONENT FAITHFULNESS SUCCEEDED"
    )


if __name__ == "__main__":
    main()
