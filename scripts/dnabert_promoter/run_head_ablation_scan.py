"""Run individual attention-head ablations for DNABERT Layers 0-2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from gfmcd.adapters.dnabert import DnaBertAdapter
from gfmcd.dnabert.manifest import write_run_manifest


MODEL_NAME = "quietflamingo/dnabert2-no-flashattention"
MODEL_REVISION = "813031b2bf86d9e960a027e2734c908009f31601"

PROBE_PATH = Path(
    "results/dnabert_promoter/frozen/"
    "layer_probes/selected_layer_2_probe.joblib"
)

DEFAULT_COHORT = Path(
    "results/dnabert_promoter/causal/"
    "localization_cohorts/validation_localization_256.csv"
)

OUTPUT_DIR = Path(
    "results/dnabert_promoter/causal/head_ablation"
)

SELECTED_LAYER = 2
NUM_HEADS = 12
HIDDEN_SIZE = 768
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
SCANNED_LAYERS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        type=Path,
        default=DEFAULT_COHORT,
    )

    parser.add_argument(
        "--output-name",
        default="validation_localization_256",
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
        "--layers",
        type=int,
        nargs="+",
        default=list(SCANNED_LAYERS),
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

    label_0_rows = [
        row for row in rows
        if int(row["label"]) == 0
    ]

    label_1_rows = [
        row for row in rows
        if int(row["label"]) == 1
    ]

    label_0_count = limit // 2
    label_1_count = limit - label_0_count

    if len(label_0_rows) < label_0_count:
        raise ValueError("Not enough label-0 rows")

    if len(label_1_rows) < label_1_count:
        raise ValueError("Not enough label-1 rows")

    return (
        label_1_rows[:label_1_count]
        + label_0_rows[:label_0_count]
    )


def identity_output_hook(
    _module,
    _inputs,
    output: torch.Tensor,
) -> torch.Tensor:
    return output


def make_head_zero_hook(
    head_index: int,
):
    if not 0 <= head_index < NUM_HEADS:
        raise ValueError(
            f"head_index must be in [0, {NUM_HEADS - 1}]"
        )

    start = head_index * HEAD_DIM
    end = start + HEAD_DIM

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
                "Expected packed self-attention output with "
                f"two dimensions, received {tuple(output.shape)}"
            )

        if output.shape[-1] != HIDDEN_SIZE:
            raise ValueError(
                f"Expected hidden size {HIDDEN_SIZE}, "
                f"received {output.shape[-1]}"
            )

        ablated = output.clone()
        ablated[:, start:end] = 0

        return ablated

    return hook


def get_self_attention_module(
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
        raise RuntimeError("Adapter returned no hidden states")

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
    hook=None,
    hook_module=None,
) -> np.ndarray:
    feature_batches = []

    handle = None

    if hook is not None:
        if hook_module is None:
            raise ValueError(
                "hook_module is required with a hook"
            )

        handle = hook_module.register_forward_hook(
            hook
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

            feature_batches.append(
                pooled.cpu().to(torch.float32).numpy()
            )

            print(
                f"\rProcessed {end:,}/{len(sequences):,}",
                end="",
                flush=True,
            )

        print()

    finally:
        if handle is not None:
            handle.remove()

    return np.concatenate(
        feature_batches,
        axis=0,
    )


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    if values.size == 0:
        raise ValueError(
            "Cannot summarize an empty array"
        )

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


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    invalid_layers = [
        layer
        for layer in args.layers
        if layer not in SCANNED_LAYERS
    ]

    if invalid_layers:
        raise ValueError(
            "Only Layers 0, 1, and 2 occur upstream "
            f"of the selected readout: {invalid_layers}"
        )

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

    clean_sequences = [
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
        clean_sequences,
        batch_size=args.batch_size,
    )

    clean_logits = probe.decision_function(
        clean_features
    )

    print("\n===== Identity-hook sanity check =====")

    identity_module = get_self_attention_module(
        adapter,
        layer=args.layers[0],
    )

    identity_features = extract_features(
        adapter,
        clean_sequences,
        batch_size=args.batch_size,
        hook=identity_output_hook,
        hook_module=identity_module,
    )

    identity_logits = probe.decision_function(
        identity_features
    )

    identity_max_difference = float(
        np.max(
            np.abs(
                identity_logits - clean_logits
            )
        )
    )

    print(
        "Identity-hook maximum logit difference:",
        identity_max_difference,
    )

    if identity_max_difference > 1e-5:
        raise RuntimeError(
            "Identity hook changed the output: "
            f"{identity_max_difference}"
        )

    results = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_layer": SELECTED_LAYER,
        "rows": len(rows),
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "scanned_layers": args.layers,
        "identity_max_logit_difference": (
            identity_max_difference
        ),
        "heads": [],
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

    for layer in args.layers:
        module = get_self_attention_module(
            adapter,
            layer,
        )

        for head in range(NUM_HEADS):
            name = f"layer_{layer}_head_{head}_zero"

            print(f"\n===== {name} =====")

            features = extract_features(
                adapter,
                clean_sequences,
                batch_size=args.batch_size,
                hook=make_head_zero_hook(head),
                hook_module=module,
            )

            logits = probe.decision_function(
                features
            )

            change = logits - clean_logits

            label_0_change = change[
                labels == 0
            ]

            label_1_change = change[
                labels == 1
            ]

            head_result = {
                "name": name,
                "layer": layer,
                "head": head,
                "change_from_clean": summarize(
                    change
                ),
                "change_from_clean_by_label": {
                    "label_0": summarize(
                        label_0_change
                    ),
                    "label_1": summarize(
                        label_1_change
                    ),
                },
                "promoter_mean_damage": float(
                    label_1_change.mean()
                ),
                "promoter_fraction_damaged": float(
                    np.mean(label_1_change < 0)
                ),
                "mean_absolute_logit_change": float(
                    np.mean(np.abs(change))
                ),
            }

            results["heads"].append(
                head_result
            )

            print(
                "Promoter mean change:",
                f"{head_result['promoter_mean_damage']:.6f}",
                "| Promoters damaged:",
                f"{100 * head_result['promoter_fraction_damaged']:.1f}%",
                "| Mean |change|:",
                f"{head_result['mean_absolute_logit_change']:.6f}",
            )

            for index in range(len(rows)):
                per_example[index][name] = float(
                    logits[index]
                )

                per_example[index][
                    f"{name}_minus_clean"
                ] = float(change[index])

    results["heads"].sort(
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

    print("\n===== Top promoter-supporting heads =====")

    for item in results["heads"][:15]:
        print(
            f"L{item['layer']}H{item['head']}: "
            f"promoter_mean_damage="
            f"{item['promoter_mean_damage']:.6f}, "
            f"damaged="
            f"{100 * item['promoter_fraction_damaged']:.1f}%, "
            f"mean_abs="
            f"{item['mean_absolute_logit_change']:.6f}"
        )

    print("\nSaved:", summary_path)
    print("Saved:", detail_path)

    # Sibling provenance record. Written after the outputs exist so their hashes are real.
    manifest_path = write_run_manifest(
        summary_path,
        experiment_id="dnabert_promoter_head_ablation",
        config=vars(args),
        outputs=[detail_path],
        inputs=[PROBE_PATH],
        probe_path=PROBE_PATH,
        selected_layer=SELECTED_LAYER,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
    )
    print("Saved:", manifest_path)

    print("HEAD ABLATION SCAN SUCCEEDED")


if __name__ == "__main__":
    main()
