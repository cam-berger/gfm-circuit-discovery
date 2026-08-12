"""Run attention/MLP branch ablations upstream of the Layer 2 probe."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

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
    "results/dnabert_promoter/causal/branch_ablation"
)

SELECTED_LAYER = 2


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

    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def zero_output_hook(
    _module,
    _inputs,
    output: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(output):
        raise TypeError(
            f"Expected tensor output, received {type(output).__name__}"
        )

    return torch.zeros_like(output)


def identity_output_hook(
    _module,
    _inputs,
    output: torch.Tensor,
) -> torch.Tensor:
    return output


def get_branch_module(
    adapter: DnaBertAdapter,
    *,
    layer: int,
    branch: str,
):
    """Return the additive branch output before residual addition."""

    encoder_layer = adapter.model.encoder.layer[layer]

    if branch == "attention":
        return encoder_layer.attention.output.dense

    if branch == "mlp":
        return encoder_layer.mlp.wo

    raise ValueError(f"Unknown branch: {branch}")


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

    pooled = (hidden * mask).sum(dim=1) / denominator

    return pooled


def extract_features(
    adapter: DnaBertAdapter,
    sequences: list[str],
    *,
    batch_size: int,
    hook_factory: Callable | None = None,
    hook_module=None,
) -> np.ndarray:
    batches = []

    handle = None

    if hook_factory is not None:
        if hook_module is None:
            raise ValueError(
                "hook_module is required when hook_factory is provided"
            )

        handle = hook_module.register_forward_hook(
            hook_factory
        )

    try:
        for start in range(0, len(sequences), batch_size):
            end = min(start + batch_size, len(sequences))
            batch = sequences[start:end]

            tokenized = adapter.tokenize(batch)

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
        if handle is not None:
            handle.remove()

    return np.concatenate(batches, axis=0)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "fraction_negative": float(
            np.mean(values < 0)
        ),
        "fraction_positive": float(
            np.mean(values > 0)
        ),
    }


def summarize_by_label(
    values: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        "label_0": summarize(values[labels == 0]),
        "label_1": summarize(values[labels == 1]),
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.cohort)

    if args.pilot_limit is not None:
        if args.pilot_limit < 2:
            raise ValueError(
                "--pilot-limit must be at least 2"
            )

        label_0_rows = [
            row for row in rows
            if int(row["label"]) == 0
        ]

        label_1_rows = [
            row for row in rows
            if int(row["label"]) == 1
        ]

        label_0_count = args.pilot_limit // 2
        label_1_count = (
            args.pilot_limit - label_0_count
        )

        if len(label_0_rows) < label_0_count:
            raise ValueError(
                "Not enough label-0 rows for pilot"
            )

        if len(label_1_rows) < label_1_count:
            raise ValueError(
                "Not enough label-1 rows for pilot"
            )

        rows = (
            label_1_rows[:label_1_count]
            + label_0_rows[:label_0_count]
        )

    labels = np.asarray(
        [int(row["label"]) for row in rows],
        dtype=np.int64,
    )

    clean_sequences = [
        row["clean_sequence"]
        for row in rows
    ]

    corrupt_sequences = [
        row["mono_sequence"]
        for row in rows
    ]

    print("Loading DNABERT...")

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

    identity_module = get_branch_module(
        adapter,
        layer=0,
        branch="attention",
    )

    identity_features = extract_features(
        adapter,
        clean_sequences,
        batch_size=args.batch_size,
        hook_factory=identity_output_hook,
        hook_module=identity_module,
    )

    identity_logits = probe.decision_function(
        identity_features
    )

    identity_max_difference = float(
        np.max(np.abs(identity_logits - clean_logits))
    )

    print(
        "Identity-hook maximum logit difference:",
        identity_max_difference,
    )

    if identity_max_difference > 1e-5:
        raise RuntimeError(
            "Identity hook changed model output: "
            f"{identity_max_difference}"
        )

    print("\n===== Corrupt baseline =====")

    corrupt_features = extract_features(
        adapter,
        corrupt_sequences,
        batch_size=args.batch_size,
    )

    corrupt_logits = probe.decision_function(
        corrupt_features
    )

    baseline_damage = corrupt_logits - clean_logits

    results = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_layer": SELECTED_LAYER,
        "rows": len(rows),
        "baseline": {
            "clean_logit": summarize(clean_logits),
            "corrupt_logit": summarize(corrupt_logits),
            "corrupt_minus_clean": summarize(
                baseline_damage
            ),
            "corrupt_minus_clean_by_label": (
                summarize_by_label(
                    baseline_damage,
                    labels,
                )
            ),
        },
        "interventions": [],
    }

    per_example = []

    for index, row in enumerate(rows):
        per_example.append(
            {
                "localization_index": row["localization_index"],
                "source_index": row["source_index"],
                "label": int(row["label"]),
                "clean_logit": float(clean_logits[index]),
                "corrupt_logit": float(corrupt_logits[index]),
                "baseline_damage": float(
                    baseline_damage[index]
                ),
            }
        )

    for layer in range(3):
        for branch in ("attention", "mlp"):
            name = f"layer_{layer}_{branch}_zero"

            print(f"\n===== {name} =====")

            module = get_branch_module(
                adapter,
                layer=layer,
                branch=branch,
            )

            ablated_features = extract_features(
                adapter,
                clean_sequences,
                batch_size=args.batch_size,
                hook_factory=zero_output_hook,
                hook_module=module,
            )

            ablated_logits = probe.decision_function(
                ablated_features
            )

            change_from_clean = (
                ablated_logits - clean_logits
            )

            promoter_change = change_from_clean[
                labels == 1
            ]

            non_promoter_change = change_from_clean[
                labels == 0
            ]

            intervention = {
                "name": name,
                "layer": layer,
                "branch": branch,
                "change_from_clean": summarize(
                    change_from_clean
                ),
                "change_from_clean_by_label": {
                    "label_0": summarize(
                        non_promoter_change
                    ),
                    "label_1": summarize(
                        promoter_change
                    ),
                },
                "mean_absolute_logit_change": float(
                    np.mean(
                        np.abs(change_from_clean)
                    )
                ),
                "promoter_mean_damage": float(
                    promoter_change.mean()
                ),
            }

            results["interventions"].append(
                intervention
            )

            print(
                "Promoter mean logit change:",
                intervention["promoter_mean_damage"],
            )

            for index in range(len(rows)):
                per_example[index][name] = float(
                    ablated_logits[index]
                )
                per_example[index][
                    f"{name}_minus_clean"
                ] = float(
                    change_from_clean[index]
                )

    results["interventions"].sort(
        key=lambda item: item["promoter_mean_damage"]
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
            fieldnames=list(per_example[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_example)

    print("\n===== Ranked interventions =====")

    for item in results["interventions"]:
        print(
            f"{item['name']}: "
            f"promoter_mean_damage="
            f"{item['promoter_mean_damage']:.6f}, "
            f"mean_absolute_change="
            f"{item['mean_absolute_logit_change']:.6f}"
        )

    print("\nSaved:", summary_path)
    print("Saved:", detail_path)

    # Sibling provenance record. Written after the outputs exist so their hashes are real.
    manifest_path = write_run_manifest(
        summary_path,
        experiment_id="dnabert_promoter_branch_ablation",
        config=vars(args),
        outputs=[detail_path],
        inputs=[PROBE_PATH],
        probe_path=PROBE_PATH,
        selected_layer=SELECTED_LAYER,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
    )
    print("Saved:", manifest_path)

    print("BRANCH ABLATION SCAN SUCCEEDED")


if __name__ == "__main__":
    main()
