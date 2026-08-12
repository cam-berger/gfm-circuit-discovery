"""Run grouped head ablations upstream of the frozen Layer-3 probe."""

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


MODEL_NAME = "quietflamingo/dnabert2-no-flashattention"
MODEL_REVISION = "813031b2bf86d9e960a027e2734c908009f31601"

PROBE_PATH = Path(
    "results/dnabert_promoter/frozen/"
    "layer_probes/all_layer_probes/layer_3_probe.joblib"
)

DEFAULT_COHORT = Path(
    "results/dnabert_promoter/causal/"
    "localization_cohorts/validation_localization_256.csv"
)

OUTPUT_DIR = Path(
    "results/dnabert_promoter/causal/"
    "layer3_grouped_head_ablation"
)

SELECTED_LAYER = 3
NUM_HEADS = 12
HIDDEN_SIZE = 768
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS


GROUPS: dict[str, list[tuple[int, int]]] = {
    "L0H8": [
        (0, 8),
    ],
    "L1H8": [
        (1, 8),
    ],
    "L2_trio": [
        (2, 3),
        (2, 8),
        (2, 9),
    ],
    "L3H11": [
        (3, 11),
    ],
    "L0H8_L1H8": [
        (0, 8),
        (1, 8),
    ],
    "L0H8_L1H8_L2_trio": [
        (0, 8),
        (1, 8),
        (2, 3),
        (2, 8),
        (2, 9),
    ],
    "primary_path_all": [
        (0, 8),
        (1, 8),
        (2, 3),
        (2, 8),
        (2, 9),
        (3, 11),
    ],
    "expanded_path": [
        (0, 8),
        (0, 5),
        (0, 10),
        (0, 11),
        (1, 8),
        (1, 0),
        (2, 3),
        (2, 8),
        (2, 9),
        (3, 11),
        (3, 6),
        (3, 4),
    ],
    "all_attention_L0": [
        (0, head)
        for head in range(NUM_HEADS)
    ],
    "all_attention_L1": [
        (1, head)
        for head in range(NUM_HEADS)
    ],
    "all_attention_L2": [
        (2, head)
        for head in range(NUM_HEADS)
    ],
    "all_attention_L3": [
        (3, head)
        for head in range(NUM_HEADS)
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        type=Path,
        default=DEFAULT_COHORT,
    )

    parser.add_argument(
        "--output-name",
        default="validation_layer3_grouped_heads_256",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--pilot-limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(GROUPS),
        default=sorted(GROUPS),
    )

    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_balanced_pilot(
    rows: list[dict[str, str]],
    limit: int,
) -> list[dict[str, str]]:
    if limit < 2:
        raise ValueError("--pilot-limit must be at least 2")

    label_0 = [
        row for row in rows
        if int(row["label"]) == 0
    ]

    label_1 = [
        row for row in rows
        if int(row["label"]) == 1
    ]

    count_0 = limit // 2
    count_1 = limit - count_0

    if len(label_0) < count_0:
        raise ValueError("Not enough label-0 rows")

    if len(label_1) < count_1:
        raise ValueError("Not enough label-1 rows")

    return label_1[:count_1] + label_0[:count_0]


def identity_hook(
    _module,
    _inputs,
    output: torch.Tensor,
) -> torch.Tensor:
    return output


def make_multi_head_zero_hook(
    heads: list[int],
):
    unique_heads = sorted(set(heads))

    for head in unique_heads:
        if not 0 <= head < NUM_HEADS:
            raise ValueError(
                f"Invalid head index: {head}"
            )

    slices = [
        (
            head * HEAD_DIM,
            (head + 1) * HEAD_DIM,
        )
        for head in unique_heads
    ]

    def hook(
        _module,
        _inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(output):
            raise TypeError(
                "Self-attention output must be a tensor"
            )

        if output.ndim != 2:
            raise ValueError(
                "Expected packed tensor with shape "
                f"(tokens, hidden), got {tuple(output.shape)}"
            )

        if output.shape[-1] != HIDDEN_SIZE:
            raise ValueError(
                f"Expected hidden size {HIDDEN_SIZE}, "
                f"got {output.shape[-1]}"
            )

        ablated = output.clone()

        for start, end in slices:
            ablated[:, start:end] = 0

        return ablated

    return hook


def self_attention_module(
    adapter: DnaBertAdapter,
    layer: int,
):
    return adapter.model.encoder.layer[layer].attention.self


def pool_selected_layer(
    adapter: DnaBertAdapter,
    tokenized,
) -> torch.Tensor:
    output = adapter.forward(
        tokenized,
        output_hidden_states=True,
        output_attentions=False,
    )

    if output.hidden_states is None:
        raise RuntimeError("No hidden states returned")

    hidden = output.hidden_states[SELECTED_LAYER]

    biological_mask = adapter.biological_token_mask(
        tokenized
    ).to(
        device=hidden.device,
        dtype=hidden.dtype,
    )

    mask = biological_mask.unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)

    return (hidden * mask).sum(dim=1) / denominator


def extract_features(
    adapter: DnaBertAdapter,
    sequences: list[str],
    *,
    batch_size: int,
    hooks: list[tuple[torch.nn.Module, object]] | None = None,
) -> np.ndarray:
    handles = []
    batches = []

    if hooks is not None:
        for module, hook in hooks:
            handles.append(
                module.register_forward_hook(hook)
            )

    try:
        for start in range(0, len(sequences), batch_size):
            end = min(
                start + batch_size,
                len(sequences),
            )

            tokenized = adapter.tokenize(
                sequences[start:end]
            )

            with torch.inference_mode():
                pooled = pool_selected_layer(
                    adapter,
                    tokenized,
                )

            batches.append(
                pooled.cpu().to(torch.float32).numpy()
            )

            print(
                f"\rProcessed {end:,}/{len(sequences):,}",
                end="",
                flush=True,
            )

        print()

    finally:
        for handle in handles:
            handle.remove()

    return np.concatenate(batches, axis=0)


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("Cannot summarize empty values")

    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1))
        if values.size > 1
        else 0.0,
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "fraction_negative": float(
            np.mean(values < 0)
        ),
        "fraction_positive": float(
            np.mean(values > 0)
        ),
        "mean_absolute": float(
            np.mean(np.abs(values))
        ),
    }


def build_group_hooks(
    adapter: DnaBertAdapter,
    group: list[tuple[int, int]],
) -> list[tuple[torch.nn.Module, object]]:
    heads_by_layer: dict[int, list[int]] = defaultdict(list)

    for layer, head in group:
        heads_by_layer[layer].append(head)

    hooks = []

    for layer, heads in sorted(heads_by_layer.items()):
        hooks.append(
            (
                self_attention_module(
                    adapter,
                    layer,
                ),
                make_multi_head_zero_hook(heads),
            )
        )

    return hooks


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.cohort)

    if args.pilot_limit is not None:
        rows = select_balanced_pilot(
            rows,
            args.pilot_limit,
        )

    labels = np.asarray(
        [int(row["label"]) for row in rows],
        dtype=np.int64,
    )

    sequences = [
        row["clean_sequence"]
        for row in rows
    ]

    print("Rows:", len(rows))
    print(
        "Label counts:",
        {
            0: int(np.sum(labels == 0)),
            1: int(np.sum(labels == 1)),
        },
    )

    print("\nLoading DNABERT...")

    adapter = DnaBertAdapter.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=args.device,
    )

    adapter.model.eval()

    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)

    probe = joblib.load(PROBE_PATH)

    print("\n===== Clean baseline =====")

    clean_features = extract_features(
        adapter,
        sequences,
        batch_size=args.batch_size,
    )

    clean_logits = probe.decision_function(
        clean_features
    )

    print("\n===== Identity-hook sanity check =====")

    identity_features = extract_features(
        adapter,
        sequences,
        batch_size=args.batch_size,
        hooks=[
            (
                self_attention_module(
                    adapter,
                    0,
                ),
                identity_hook,
            )
        ],
    )

    identity_logits = probe.decision_function(
        identity_features
    )

    identity_difference = float(
        np.max(
            np.abs(
                identity_logits - clean_logits
            )
        )
    )

    print(
        "Identity-hook maximum logit difference:",
        identity_difference,
    )

    if identity_difference > 1e-5:
        raise RuntimeError(
            "Identity hook changed model output"
        )

    results = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_layer": SELECTED_LAYER,
        "rows": len(rows),
        "identity_max_logit_difference": (
            identity_difference
        ),
        "groups": [],
    }

    per_example = []

    for index, row in enumerate(rows):
        per_example.append(
            {
                "localization_index": row[
                    "localization_index"
                ],
                "source_index": row["source_index"],
                "label": int(row["label"]),
                "clean_logit": float(
                    clean_logits[index]
                ),
            }
        )

    for group_name in args.groups:
        group = GROUPS[group_name]

        print(f"\n===== {group_name} =====")
        print("Heads:", group)

        hooks = build_group_hooks(
            adapter,
            group,
        )

        features = extract_features(
            adapter,
            sequences,
            batch_size=args.batch_size,
            hooks=hooks,
        )

        logits = probe.decision_function(
            features
        )

        change = logits - clean_logits
        promoter_change = change[labels == 1]
        non_promoter_change = change[labels == 0]

        group_result = {
            "name": group_name,
            "heads": [
                {
                    "layer": layer,
                    "head": head,
                }
                for layer, head in group
            ],
            "num_heads": len(group),
            "change_from_clean": summarize(
                change
            ),
            "change_from_clean_by_label": {
                "label_0": summarize(
                    non_promoter_change
                ),
                "label_1": summarize(
                    promoter_change
                ),
            },
            "promoter_mean_damage": float(
                promoter_change.mean()
            ),
            "promoter_fraction_damaged": float(
                np.mean(promoter_change < 0)
            ),
            "mean_absolute_logit_change": float(
                np.mean(np.abs(change))
            ),
        }

        results["groups"].append(
            group_result
        )

        print(
            "Promoter mean change:",
            f"{group_result['promoter_mean_damage']:.6f}",
            "| damaged:",
            f"{100 * group_result['promoter_fraction_damaged']:.1f}%",
            "| mean |change|:",
            f"{group_result['mean_absolute_logit_change']:.6f}",
        )

        for index in range(len(rows)):
            per_example[index][group_name] = float(
                logits[index]
            )

            per_example[index][
                f"{group_name}_minus_clean"
            ] = float(change[index])

    results["groups"].sort(
        key=lambda item: item[
            "promoter_mean_damage"
        ]
    )

    summary_path = OUTPUT_DIR / (
        f"{args.output_name}_summary.json"
    )

    detail_path = OUTPUT_DIR / (
        f"{args.output_name}_per_example.csv"
    )

    summary_path.write_text(
        json.dumps(
            results,
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
                per_example[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(per_example)

    print("\n===== Ranked grouped ablations =====")

    for item in results["groups"]:
        print(
            f"{item['name']}: "
            f"heads={item['num_heads']}, "
            f"promoter_mean_damage="
            f"{item['promoter_mean_damage']:.6f}, "
            f"damaged="
            f"{100 * item['promoter_fraction_damaged']:.1f}%, "
            f"mean_abs="
            f"{item['mean_absolute_logit_change']:.6f}"
        )

    print("\nSaved:", summary_path)
    print("Saved:", detail_path)
    print("GROUPED HEAD ABLATION SUCCEEDED")


if __name__ == "__main__":
    main()
