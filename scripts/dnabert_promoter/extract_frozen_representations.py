"""Extract masked-mean representations from every frozen DNABERT layer."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from gfmcd.adapters.dnabert import DnaBertAdapter


DATA_DIR = Path("results/dnabert_promoter/shared/clean")
OUTPUT_DIR = Path("results/dnabert_promoter/frozen/representations")

MODEL_NAME = "quietflamingo/dnabert2-no-flashattention"
MODEL_REVISION = "813031b2bf86d9e960a027e2734c908009f31601"

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
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
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(split: str) -> tuple[list[str], np.ndarray, list[str]]:
    path = DATA_DIR / f"{split}.csv"

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    sequences = [row["sequence"] for row in rows]

    labels = np.asarray(
        [int(row["label"]) for row in rows],
        dtype=np.int64,
    )

    sequence_hashes = [
        row["sequence_sha256"]
        for row in rows
    ]

    return sequences, labels, sequence_hashes


def masked_mean_per_layer(
    hidden_states: tuple[torch.Tensor, ...],
    biological_mask: torch.Tensor,
) -> torch.Tensor:
    """Return [batch, layers, hidden] masked-mean representations."""

    mask = biological_mask.to(
        dtype=hidden_states[0].dtype,
    ).unsqueeze(-1)

    denominator = mask.sum(dim=1).clamp_min(1.0)

    pooled_layers = []

    for hidden in hidden_states:
        pooled = (hidden * mask).sum(dim=1) / denominator
        pooled_layers.append(pooled)

    return torch.stack(
        pooled_layers,
        dim=1,
    )


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / f"{args.split}.npz"
    metadata_path = OUTPUT_DIR / f"{args.split}_metadata.json"

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; use --overwrite to replace it"
        )

    sequences, labels, hashes = load_split(args.split)

    print(f"Split: {args.split}")
    print(f"Rows: {len(sequences):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")

    adapter = DnaBertAdapter.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=args.device,
    )

    model_parameters = list(adapter.model.parameters())

    if any(parameter.requires_grad for parameter in model_parameters):
        for parameter in model_parameters:
            parameter.requires_grad_(False)

    adapter.model.eval()

    all_features = []

    for start in range(
        0,
        len(sequences),
        args.batch_size,
    ):
        end = min(
            start + args.batch_size,
            len(sequences),
        )

        sequence_batch = sequences[start:end]

        tokenized = adapter.tokenize(
            sequence_batch
        )

        with torch.inference_mode():
            output = adapter.forward(
                tokenized,
                output_hidden_states=True,
                output_attentions=False,
            )

            if output.hidden_states is None:
                raise RuntimeError(
                    "adapter returned no hidden states"
                )

            biological_mask = (
                adapter.biological_token_mask(tokenized)
            )

            pooled = masked_mean_per_layer(
                output.hidden_states,
                biological_mask,
            )

        all_features.append(
            pooled.cpu().to(torch.float32).numpy()
        )

        print(
            f"\rProcessed {end:,}/{len(sequences):,}",
            end="",
            flush=True,
        )

    print()

    features = np.concatenate(
        all_features,
        axis=0,
    )

    expected_shape = (
        len(sequences),
        adapter.num_layers,
        adapter.hidden_size,
    )

    if features.shape != expected_shape:
        raise RuntimeError(
            f"unexpected feature shape {features.shape}; "
            f"expected {expected_shape}"
        )

    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        sequence_sha256=np.asarray(hashes),
    )

    metadata = {
        "split": args.split,
        "rows": len(sequences),
        "feature_shape": list(features.shape),
        "dtype": str(features.dtype),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "pooling": "masked_mean_biological_tokens",
        "num_layers": adapter.num_layers,
        "hidden_size": adapter.hidden_size,
        "batch_size": args.batch_size,
        "seed": SEED,
        "output_file": str(output_path),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Feature shape:", features.shape)
    print("Saved:", output_path)
    print("Saved:", metadata_path)
    print("FROZEN REPRESENTATION EXTRACTION SUCCEEDED")


if __name__ == "__main__":
    main()
