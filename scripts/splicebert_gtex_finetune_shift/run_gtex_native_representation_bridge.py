#!/usr/bin/env python
"""Gated layerwise bridge between base and GTEx-fine-tuned representations."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.coupling import substitute  # noqa: E402
from gfmcd.splicing.grouped_stats import grouped_bootstrap  # noqa: E402
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import (  # noqa: E402
    NativeClassifierTask,
    linear_cka,
    load_annotation_cohort,
)

EXPECTED_SEEDS = frozenset(range(2020, 2025))
REPLAY_ATOL = 2e-5
LAYER_METRICS = (
    "fine_clean_margin",
    "fine_partner_coupling",
    "base_clean_margin_same_head",
    "base_partner_coupling_same_head",
    "fine_minus_base_clean_margin",
    "fine_minus_base_partner_coupling",
)


def parse_checkpoints(values: list[str]) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for value in values:
        try:
            seed_text, path_text = value.split("=", 1)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError("--checkpoint must be SEED=PATH") from exc
        if seed in checkpoints:
            raise ValueError(f"duplicate seed {seed}")
        checkpoints[seed] = Path(path_text).expanduser().resolve()
    if set(checkpoints) != set(EXPECTED_SEEDS):
        raise ValueError(
            f"complete seed family required; found {sorted(checkpoints)}"
        )
    return checkpoints


def project_all_depths(
    model: Any,
    task: NativeClassifierTask,
    sequences: list[str],
    positions: list[int],
    *,
    retain_states: bool,
) -> tuple[np.ndarray, list[np.ndarray] | None]:
    """Apply the frozen final classifier to donor states at every encoder depth."""

    margin_chunks: list[np.ndarray] = []
    state_chunks: list[list[np.ndarray]] | None = (
        [] if retain_states else None
    )
    with torch.no_grad():
        for start in range(0, len(sequences), task.batch_size):
            chunk = sequences[start : start + task.batch_size]
            encoded = task.encode(chunk)
            output = model.bert(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = output.hidden_states
            if hidden_states is None:
                raise RuntimeError("encoder did not return hidden states")
            token_positions = torch.as_tensor(
                positions[start : start + len(chunk)],
                device=hidden_states[0].device,
                dtype=torch.long,
            ) + task.prefix_tokens
            rows = torch.arange(len(chunk), device=token_positions.device)
            layer_margins: list[np.ndarray] = []
            layer_states: list[np.ndarray] = []
            for hidden in hidden_states:
                selected = hidden[rows, token_positions]
                logits = model.classifier(model.dropout(selected)).float()
                margins = logits[:, 1] - logits[:, 0]
                if not torch.isfinite(margins).all():
                    raise RuntimeError("non-finite layerwise classifier projection")
                layer_margins.append(margins.detach().cpu().numpy())
                if retain_states:
                    layer_states.append(
                        selected.float().detach().cpu().numpy()
                    )
            margin_chunks.append(np.stack(layer_margins, axis=1))
            if state_chunks is not None:
                state_chunks.append(layer_states)

    margins = np.concatenate(margin_chunks, axis=0).astype(np.float64)
    if state_chunks is None:
        return margins, None
    states = [
        np.concatenate(
            [batch_layers[layer] for batch_layers in state_chunks],
            axis=0,
        )
        for layer in range(margins.shape[1])
    ]
    return margins, states


def model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model weights not found in {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-behavior", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != 400:
        raise ValueError("representation bridge is frozen at n=400")
    native_path = Path(args.native_behavior).resolve()
    native = json.loads(native_path.read_text())
    gate = native.get("native_coupling_gate", {})
    if not gate.get("PASS") or not gate.get("representation_bridge_authorized"):
        raise RuntimeError("native coupling gate did not authorize representation")
    protocol_path = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CLASSIFIER_V2.md"
    )
    base_model = Path(args.base_model).expanduser().resolve()
    checkpoints = parse_checkpoints(args.checkpoint)
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)

    examples = load_annotation_cohort(
        args.gtf,
        args.fasta,
        split="val",
        n=args.n,
        tag="sealed-unc-val",
    )
    example_ids = [example.example_id for example in examples]
    groups = np.asarray([example.gene_id for example in examples], dtype=object)
    donor_labels = [example.donor_pos - 1 for example in examples]
    clean_sequences = [example.sequence for example in examples]
    corrupt_sequences = [
        substitute(example.sequence, int(example.next_acceptor_pos) - 2, "CC")
        for example in examples
    ]

    from transformers import AutoModelForMaskedLM

    base_mlm = AutoModelForMaskedLM.from_pretrained(str(base_model))
    base_encoder_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in base_mlm.bert.state_dict().items()
    }
    del base_mlm
    gc.collect()

    seed_arrays: dict[str, dict[str, np.ndarray]] = {}
    seed_summaries: dict[str, Any] = {}
    replay_checks: dict[str, Any] = {}
    n_depths: int | None = None
    for seed, checkpoint in sorted(checkpoints.items()):
        print(f"seed {seed}: layerwise fine/base bridge on {len(examples)} examples")
        name = str(seed)
        expected_raw = native["raw"][name]
        if [row["example_id"] for row in expected_raw] != example_ids:
            raise RuntimeError(f"seed {seed} native artifact cohort/order mismatch")
        model, tokenizer, report = load_gtex_token_classifier(
            base_model,
            checkpoint,
            device=device,
        )
        task = NativeClassifierTask(
            model,
            tokenizer,
            device=device,
            batch_size=args.batch,
        )

        fine_clean, fine_states = project_all_depths(
            model,
            task,
            clean_sequences,
            donor_labels,
            retain_states=True,
        )
        fine_corrupt, _ = project_all_depths(
            model,
            task,
            corrupt_sequences,
            donor_labels,
            retain_states=False,
        )
        if fine_states is None:
            raise RuntimeError("fine states were not retained")
        fine_coupling = fine_clean - fine_corrupt

        expected_clean = np.asarray(
            [row["clean_donor_margin"] for row in expected_raw],
            dtype=np.float64,
        )
        expected_coupling = np.asarray(
            [row["partner_cc"] for row in expected_raw],
            dtype=np.float64,
        )
        clean_error = float(np.max(np.abs(fine_clean[:, -1] - expected_clean)))
        coupling_error = float(
            np.max(np.abs(fine_coupling[:, -1] - expected_coupling))
        )
        replay_checks[name] = {
            "clean_margin_max_abs_error": clean_error,
            "partner_coupling_max_abs_error": coupling_error,
            "atol": REPLAY_ATOL,
            "PASS": clean_error <= REPLAY_ATOL and coupling_error <= REPLAY_ATOL,
        }
        if not replay_checks[name]["PASS"]:
            raise RuntimeError(
                f"seed {seed} final layer does not replay native behavior"
            )

        model.bert.load_state_dict(base_encoder_state, strict=True)
        base_clean, base_states = project_all_depths(
            model,
            task,
            clean_sequences,
            donor_labels,
            retain_states=True,
        )
        base_corrupt, _ = project_all_depths(
            model,
            task,
            corrupt_sequences,
            donor_labels,
            retain_states=False,
        )
        if base_states is None:
            raise RuntimeError("base states were not retained")
        base_coupling = base_clean - base_corrupt
        n_depths = fine_clean.shape[1]
        if n_depths != 7 or base_clean.shape != fine_clean.shape:
            raise RuntimeError(
                f"unexpected representation depths/shapes: "
                f"fine={fine_clean.shape}, base={base_clean.shape}"
            )

        arrays = {
            "fine_clean_margin": fine_clean,
            "fine_partner_coupling": fine_coupling,
            "base_clean_margin_same_head": base_clean,
            "base_partner_coupling_same_head": base_coupling,
            "fine_minus_base_clean_margin": fine_clean - base_clean,
            "fine_minus_base_partner_coupling": fine_coupling - base_coupling,
        }
        seed_arrays[name] = arrays
        layer_rows = []
        for depth in range(n_depths):
            row = {
                "depth": depth,
                "block": "embeddings" if depth == 0 else f"layer_{depth - 1}",
                "linear_cka_clean_donor_states": linear_cka(
                    base_states[depth],
                    fine_states[depth],
                ),
            }
            for metric_index, metric in enumerate(LAYER_METRICS):
                row[metric] = grouped_bootstrap(
                    arrays[metric][:, depth],
                    groups,
                    n_boot=args.n_boot,
                    seed=seed + 100 * depth + metric_index,
                )
            layer_rows.append(row)
        seed_summaries[name] = {
            "checkpoint": report.to_dict(),
            "layers": layer_rows,
        }

        del task, model, tokenizer, fine_states, base_states
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    if n_depths is None:
        raise RuntimeError("no checkpoints were scored")
    fine_names = [str(seed) for seed in sorted(EXPECTED_SEEDS)]
    family_layers = []
    for depth in range(n_depths):
        row: dict[str, Any] = {
            "depth": depth,
            "block": "embeddings" if depth == 0 else f"layer_{depth - 1}",
        }
        for metric_index, metric in enumerate(LAYER_METRICS):
            values = np.mean(
                np.stack(
                    [seed_arrays[name][metric][:, depth] for name in fine_names]
                ),
                axis=0,
            )
            row[metric] = grouped_bootstrap(
                values,
                groups,
                n_boot=args.n_boot,
                seed=20000 + 100 * depth + metric_index,
            )
        cka_values = [
            seed_summaries[name]["layers"][depth][
                "linear_cka_clean_donor_states"
            ]
            for name in fine_names
        ]
        row["linear_cka_clean_donor_states"] = {
            "mean_across_seeds": float(np.mean(cka_values)),
            "min_across_seeds": float(np.min(cka_values)),
            "max_across_seeds": float(np.max(cka_values)),
            "seed_values": dict(zip(fine_names, cka_values)),
        }
        if depth == 0:
            row["fine_projected_coupling_increment"] = None
        else:
            current = np.mean(
                np.stack(
                    [
                        seed_arrays[name]["fine_partner_coupling"][:, depth]
                        for name in fine_names
                    ]
                ),
                axis=0,
            )
            previous = np.mean(
                np.stack(
                    [
                        seed_arrays[name]["fine_partner_coupling"][:, depth - 1]
                        for name in fine_names
                    ]
                ),
                axis=0,
            )
            row["fine_projected_coupling_increment"] = grouped_bootstrap(
                current - previous,
                groups,
                n_boot=args.n_boot,
                seed=30000 + depth,
            )
        family_layers.append(row)

    raw: dict[str, list[dict[str, Any]]] = {}
    for name, arrays in seed_arrays.items():
        raw[name] = [
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                **{
                    metric: arrays[metric][index].tolist()
                    for metric in LAYER_METRICS
                },
            }
            for index, example in enumerate(examples)
        ]

    artifact = {
        "study": "splicebert_gtex_finetune_shift",
        "stage": "native classifier representation bridge",
        "status": "REPRESENTATION_CANDIDATE",
        "protocol": str(protocol_path.relative_to(REPO)),
        "scope": (
            "Common frozen seed-specific classifier vector applied layerwise to "
            "fine and base encoder states. Intermediate projections and CKA are "
            "descriptive candidates, not necessity or mediation."
        ),
        "native_behavior_gate": gate,
        "final_layer_replay": replay_checks,
        "models": seed_summaries,
        "family_layers": family_layers,
        "raw_projected_margins": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")

    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_representation_bridge",
        sys.argv,
        repo=REPO,
        status=artifact["status"],
        config={
            "split": "val",
            "cohort_tag": "sealed-unc-val",
            "n": args.n,
            "batch": args.batch,
            "n_boot": args.n_boot,
            "device": device,
            "depths": n_depths,
            "classifier_policy": (
                "same frozen seed-specific final classifier at every fine/base depth"
            ),
            "donor_label_coordinate": "donor_motif_start_minus_1",
            "replay_atol": REPLAY_ATOL,
        },
        seeds={
            "published_checkpoints_start": 2020,
            "family_bootstrap_start": 20000,
        },
        notes=[
            "Run only after the native behavior artifact passed every frozen gate.",
            "All five published seeds were run sequentially.",
            "Intermediate classifier projections and linear CKA are candidate generators.",
            "The base-encoder/fine-head endpoint is a compatibility diagnostic, "
            "not native base behavior.",
        ],
    )
    for path in (
        native_path,
        protocol_path,
        model_weight(base_model),
        base_model / "config.json",
        base_model / "tokenizer.json",
        Path(args.gtf),
        Path(args.fasta),
        *checkpoints.values(),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
