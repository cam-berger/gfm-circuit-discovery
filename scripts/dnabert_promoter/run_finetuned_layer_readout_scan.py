"""Scan the native fine-tuned classifier readout across DNABERT layers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from finetuned_common import (
    DEFAULT_SEED,
    SHARED_ROOT,
    load_finetuned_classifier,
    load_manifest,
)


OUTPUT_DIR = Path(
    "results/dnabert_promoter/finetuned_claim_tests/"
    "layer_readout_scan"
)


class SequenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.sequences = frame["sequence"].astype(str).tolist()
        self.labels = frame["label"].astype(int).to_numpy()
        self.source_indices = (
            frame["source_index"].astype(int).to_numpy()
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return (
            self.sequences[index],
            int(self.labels[index]),
            int(self.source_indices[index]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--output-name",
        default=None,
    )

    return parser.parse_args()


def restore_padded_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Restore DNABERT packed layer output to [batch, tokens, hidden]."""

    if hidden.ndim == 3:
        return hidden

    if hidden.ndim != 2:
        raise ValueError(
            "Expected packed [tokens, hidden] or padded "
            f"[batch, tokens, hidden], got {tuple(hidden.shape)}"
        )

    batch_size, sequence_length = attention_mask.shape
    hidden_size = hidden.shape[-1]

    valid_positions = attention_mask.reshape(-1).bool()
    expected_tokens = int(valid_positions.sum().item())

    if hidden.shape[0] != expected_tokens:
        raise RuntimeError(
            "Packed hidden-state length does not match the "
            "number of valid attention-mask positions: "
            f"{hidden.shape[0]} versus {expected_tokens}"
        )

    padded_flat = hidden.new_zeros(
        batch_size * sequence_length,
        hidden_size,
    )

    padded_flat[valid_positions] = hidden

    return padded_flat.reshape(
        batch_size,
        sequence_length,
        hidden_size,
    )


def masked_mean(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    hidden = restore_padded_hidden(
        hidden,
        attention_mask,
    )

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

    return (
        (hidden * mask).sum(dim=1)
        / mask.sum(dim=1).clamp_min(1.0)
    )


def compute_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
) -> dict[str, float]:
    probabilities = torch.softmax(
        torch.from_numpy(logits),
        dim=-1,
    ).numpy()[:, 1]

    predictions = logits.argmax(axis=-1)

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "f1": float(
            f1_score(labels, predictions)
        ),
        "mean_promoter_margin": float(
            np.mean(logits[:, 1] - logits[:, 0])
        ),
        "mean_absolute_margin": float(
            np.mean(
                np.abs(
                    logits[:, 1] - logits[:, 0]
                )
            )
        ),
    }


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.read_csv(
        SHARED_ROOT / f"{args.split}.csv"
    )

    if args.limit is not None:
        frame = frame.iloc[: args.limit].copy()

    manifest = load_manifest(args.seed)

    model = load_finetuned_classifier(
        seed=args.seed,
        device=args.device,
    )

    dataset = SequenceDataset(frame)

    def collate(batch):
        sequences, labels, source_indices = zip(*batch)

        encoded = model.tokenizer(
            list(sequences),
            padding="max_length",
            truncation=True,
            max_length=int(
                manifest["max_length"]
            ),
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded[
                "attention_mask"
            ],
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
            "source_indices": torch.tensor(
                source_indices,
                dtype=torch.long,
            ),
        }

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    num_layers = len(
        model.encoder.encoder.layer
    )

    logits_by_layer = [
        []
        for _ in range(num_layers)
    ]

    all_labels = []
    all_source_indices = []

    print("Split:", args.split)
    print("Rows:", len(dataset))
    print("Seed:", args.seed)
    print("Encoder layers:", num_layers)

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            labels = batch.pop("labels")
            source_indices = batch.pop(
                "source_indices"
            )

            batch = {
                key: value.to(args.device)
                for key, value in batch.items()
            }

            captured_hidden: list[torch.Tensor | None] = [
                None for _ in range(num_layers)
            ]

            handles = []

            def make_hook(layer_index: int):
                def hook(module, inputs, output):
                    if torch.is_tensor(output):
                        hidden = output
                    elif (
                        isinstance(output, tuple)
                        and len(output) > 0
                        and torch.is_tensor(output[0])
                    ):
                        hidden = output[0]
                    else:
                        raise RuntimeError(
                            "Unsupported encoder-layer output "
                            f"at layer {layer_index}: "
                            f"{type(output)}"
                        )

                    captured_hidden[layer_index] = hidden

                return hook

            for layer_index, layer_module in enumerate(
                model.encoder.encoder.layer
            ):
                handles.append(
                    layer_module.register_forward_hook(
                        make_hook(layer_index)
                    )
                )

            try:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    model.encoder(
                        input_ids=batch["input_ids"],
                        attention_mask=batch[
                            "attention_mask"
                        ],
                        output_hidden_states=False,
                        return_dict=False,
                    )

                    if any(
                        hidden is None
                        for hidden in captured_hidden
                    ):
                        missing = [
                            index
                            for index, hidden
                            in enumerate(captured_hidden)
                            if hidden is None
                        ]

                        raise RuntimeError(
                            "Failed to capture encoder layers: "
                            f"{missing}"
                        )

                    for layer_index, hidden in enumerate(
                        captured_hidden
                    ):
                        assert hidden is not None

                        pooled = masked_mean(
                            hidden,
                            batch["attention_mask"],
                        )

                        logits = model.classifier(
                            model.dropout(pooled)
                        )

                        logits_by_layer[
                            layer_index
                        ].append(
                            logits.float().cpu().numpy()
                        )

            finally:
                for handle in handles:
                    handle.remove()

            all_labels.append(labels.numpy())
            all_source_indices.append(
                source_indices.numpy()
            )

            print(
                f"\rProcessed "
                f"{min(batch_index * args.batch_size, len(dataset))}"
                f"/{len(dataset)}",
                end="",
                flush=True,
            )

    print()

    labels = np.concatenate(all_labels)
    source_indices = np.concatenate(
        all_source_indices
    )

    layer_summaries = []
    per_example_rows = []

    for layer_index in range(num_layers):
        logits = np.concatenate(
            logits_by_layer[layer_index]
        )

        metrics = compute_metrics(
            labels,
            logits,
        )

        layer_summary = {
            "layer": layer_index,
            **metrics,
        }

        layer_summaries.append(layer_summary)

        predictions = logits.argmax(axis=-1)

        probabilities = torch.softmax(
            torch.from_numpy(logits),
            dim=-1,
        ).numpy()[:, 1]

        for row_index in range(len(labels)):
            per_example_rows.append(
                {
                    "layer": layer_index,
                    "source_index": int(
                        source_indices[row_index]
                    ),
                    "label": int(
                        labels[row_index]
                    ),
                    "logit_0": float(
                        logits[row_index, 0]
                    ),
                    "logit_1": float(
                        logits[row_index, 1]
                    ),
                    "promoter_margin": float(
                        logits[row_index, 1]
                        - logits[row_index, 0]
                    ),
                    "promoter_probability": float(
                        probabilities[row_index]
                    ),
                    "prediction": int(
                        predictions[row_index]
                    ),
                }
            )

    best_layer = max(
        layer_summaries,
        key=lambda row: row[
            "balanced_accuracy"
        ],
    )

    final_layer = layer_summaries[-1]

    output_name = (
        args.output_name
        or f"seed_{args.seed}_{args.split}"
    )

    summary_path = (
        OUTPUT_DIR
        / f"{output_name}_summary.json"
    )

    curve_path = (
        OUTPUT_DIR
        / f"{output_name}_curve.csv"
    )

    per_example_path = (
        OUTPUT_DIR
        / f"{output_name}_per_example.csv"
    )

    summary = {
        "seed": args.seed,
        "split": args.split,
        "rows": len(labels),
        "precision": "bfloat16 autocast",
        "readout": (
            "native learned classifier applied "
            "to masked-mean intermediate hidden states"
        ),
        "num_layers": num_layers,
        "best_layer_by_balanced_accuracy": (
            best_layer
        ),
        "final_layer": final_layer,
        "layers": layer_summaries,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    with curve_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                layer_summaries[0].keys()
            ),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(layer_summaries)

    with per_example_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                per_example_rows[0].keys()
            ),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(per_example_rows)

    print("\n===== Layer readout scan =====")

    for row in layer_summaries:
        print(
            f"Layer {row['layer']:02d}: "
            f"BA={row['balanced_accuracy']:.6f} "
            f"AUC={row['roc_auc']:.6f} "
            f"|margin|="
            f"{row['mean_absolute_margin']:.4f}"
        )

    print(
        "\nBest native-readout layer:",
        best_layer["layer"],
    )

    print(
        "Best balanced accuracy:",
        best_layer["balanced_accuracy"],
    )

    print(
        "Final-layer balanced accuracy:",
        final_layer["balanced_accuracy"],
    )

    print("\nSaved:", summary_path)
    print("Saved:", curve_path)
    print("Saved:", per_example_path)

    print(
        "FINETUNED LAYER READOUT SCAN SUCCEEDED"
    )


if __name__ == "__main__":
    main()
