#!/usr/bin/env python
"""Capture label-free final-MLP inputs and writes for six paired checkpoints."""

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

from gfmcd.adapters.splicebert import SpliceBertAdapter  # noqa: E402
from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.circuits.multiview_transcoder import (  # noqa: E402
    assert_label_free_payload,
    validate_label_free_state_rows,
)
from gfmcd.provenance import (  # noqa: E402
    ExperimentManifest,
    file_record,
    sha256_file,
)
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.gtex_finetune import (  # noqa: E402
    load_gtex_token_classifier,
)
from gfmcd.splicing.native_mechanism import (  # noqa: E402
    cache_final_layer_inputs,
    capture_final_layer_run,
    donor_token_positions,
)


EXPECTED_SEEDS = tuple(range(2020, 2025))
VIEWS = ("base", *(str(seed) for seed in EXPECTED_SEEDS))
NAMESPACE = "gtex-native-biological-features-v1"
FINAL_LAYER = 5
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 2048
INPUT_EQUALITY_TOLERANCE = 2e-5
WRITE_RECONSTRUCTION_TOLERANCE = 5e-4


def parse_checkpoints(values: list[str]) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in checkpoints:
            raise ValueError(f"duplicate checkpoint seed {seed}")
        checkpoints[seed] = Path(path_text).expanduser().resolve()
    if tuple(sorted(checkpoints)) != EXPECTED_SEEDS:
        raise ValueError("complete checkpoint family 2020--2024 required")
    return checkpoints


def model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = model_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"model weights not found in {model_dir}")


def encode_sequences(
    tokenizer: Any,
    sequences: list[str],
    *,
    device: str,
) -> Any:
    encoded = tokenizer(
        [" ".join(sequence) for sequence in sequences],
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(device)
    expected = max(map(len, sequences)) + 2
    if int(encoded["input_ids"].shape[1]) != expected:
        raise RuntimeError("tokenizer is not one-token-per-nucleotide")
    return encoded


def target_rows(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[rows, positions.to(device=tensor.device)]


def capture_batch(
    adapter: SpliceBertAdapter,
    model: Any,
    encoded: Any,
    target_positions: list[int],
) -> dict[str, torch.Tensor | float]:
    """Capture and verify the exact input/output of the final MLP."""

    layer = adapter.encoder_layers[FINAL_LAYER]
    inputs = cache_final_layer_inputs(model, encoded, layer=FINAL_LAYER)
    intermediate_inputs: list[torch.Tensor] = []

    def capture_input(_module: Any, args: Any) -> None:
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("BertIntermediate input capture failed")
        intermediate_inputs.append(args[0].detach().clone())

    handle = layer.intermediate.register_forward_pre_hook(capture_input)
    try:
        run = capture_final_layer_run(adapter, inputs, layer=FINAL_LAYER)
    finally:
        handle.remove()
    if len(intermediate_inputs) != 1:
        raise RuntimeError("expected exactly one BertIntermediate invocation")
    if run.mlp_residual is None or run.mlp_intermediate is None:
        raise RuntimeError("final MLP capture is incomplete")
    input_error = float(
        (intermediate_inputs[0].float() - run.mlp_residual.float())
        .abs()
        .max()
        .cpu()
    )
    positions = donor_token_positions(
        target_positions,
        prefix_tokens=adapter.prefix_tokens,
        device=adapter.device,
    )
    exact_input = target_rows(intermediate_inputs[0], positions)
    write = target_rows(run.mlp_write, positions)
    intermediate = target_rows(run.mlp_intermediate, positions)
    dense = layer.output.dense
    reconstructed = (
        intermediate.cpu().double()
        @ dense.weight.detach().cpu().double().T
        + dense.bias.detach().cpu().double()
    )
    write_error = float(
        (reconstructed - write.detach().cpu().double()).abs().max()
    )
    return {
        "input": exact_input.detach().cpu().float(),
        "write": write.detach().cpu().float(),
        "input_error": input_error,
        "write_error": write_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-package", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--arrays-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    torch.set_grad_enabled(False)

    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_BIOLOGICAL_FEATURES_V1.md"
    )
    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_SPARSE_TRANSCODER_V1.md"
    )
    package_path = Path(args.sequence_package).expanduser().resolve()
    package = json.loads(package_path.read_text())
    rows = validate_label_free_state_rows(
        package,
        namespace=NAMESPACE,
        partition="discovery",
        verify_source_files=True,
    )
    if package.get("sparse_training_authorized") is not True:
        raise RuntimeError(
            "sequence package has not passed the frozen biological-feature preflight"
        )
    assert_label_free_payload(package)
    checkpoints = parse_checkpoints(args.checkpoint)
    base_model_path = Path(args.base_model).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    arrays_output = guard_output_path(Path(args.arrays_out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    if len({output, arrays_output, manifest_path}) != 3:
        raise ValueError("result, arrays, and manifest outputs must be distinct")
    if output.suffix != ".json" or arrays_output.suffix != ".npz":
        raise ValueError("activation capture requires .json result and .npz arrays")
    device = pick_device(args.device)

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    arrays: dict[str, np.ndarray] = {}
    view_records: dict[str, Any] = {}
    maximum_input_error = 0.0
    maximum_write_error = 0.0

    for view in VIEWS:
        if view == "base":
            model = AutoModelForMaskedLM.from_pretrained(
                str(base_model_path),
                attn_implementation="eager",
            )
            view_tokenizer = tokenizer
            source_path = model_weight(base_model_path)
            checkpoint_seed = None
            checkpoint_sha256 = sha256_file(source_path)
        else:
            seed = int(view)
            model, view_tokenizer, report = load_gtex_token_classifier(
                base_model_path,
                checkpoints[seed],
                device=device,
                attn_implementation="eager",
            )
            source_path = checkpoints[seed]
            checkpoint_seed = seed
            checkpoint_sha256 = report.checkpoint_sha256
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        adapter = SpliceBertAdapter(
            model,
            view_tokenizer,
            source=str(source_path),
            device=device,
        )
        dense = adapter.encoder_layers[FINAL_LAYER].output.dense
        if (
            adapter.num_layers != 6
            or adapter.num_heads != 16
            or adapter.hidden_size != HIDDEN_SIZE
            or dense.in_features != INTERMEDIATE_SIZE
            or dense.out_features != HIDDEN_SIZE
        ):
            raise RuntimeError("six-view checkpoint architecture mismatch")

        input_chunks: list[np.ndarray] = []
        write_chunks: list[np.ndarray] = []
        view_input_error = 0.0
        view_write_error = 0.0
        for start in range(0, len(rows), args.batch):
            stop = min(start + args.batch, len(rows))
            chunk = rows[start:stop]
            encoded = encode_sequences(
                view_tokenizer,
                [str(row["sequence"]) for row in chunk],
                device=device,
            )
            captured = capture_batch(
                adapter,
                model,
                encoded,
                [int(row["target_position"]) for row in chunk],
            )
            input_chunks.append(captured["input"].numpy())
            write_chunks.append(captured["write"].numpy())
            view_input_error = max(
                view_input_error,
                float(captured["input_error"]),
            )
            view_write_error = max(
                view_write_error,
                float(captured["write_error"]),
            )
            print(
                f"view {view} {stop}/{len(rows)}",
                end="\r",
                flush=True,
            )
        print()
        if (
            view_input_error > INPUT_EQUALITY_TOLERANCE
            or view_write_error > WRITE_RECONSTRUCTION_TOLERANCE
        ):
            raise RuntimeError(
                f"activation capture preflight failed for {view}: "
                f"input={view_input_error:.9g}, write={view_write_error:.9g}"
            )
        arrays[f"{view}__input"] = np.concatenate(input_chunks).astype(
            np.float32,
            copy=False,
        )
        arrays[f"{view}__write"] = np.concatenate(write_chunks).astype(
            np.float32,
            copy=False,
        )
        view_records[view] = {
            "seed": checkpoint_seed,
            "checkpoint_path": str(Path(source_path).resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "architecture": {
                "layers": adapter.num_layers,
                "heads": adapter.num_heads,
                "hidden_size": adapter.hidden_size,
                "intermediate_size": dense.in_features,
            },
            "input_equality_max_abs_error": view_input_error,
            "write_reconstruction_max_abs_error": view_write_error,
        }
        maximum_input_error = max(maximum_input_error, view_input_error)
        maximum_write_error = max(maximum_write_error, view_write_error)
        del adapter, model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    if any(
        value.shape != (len(rows), HIDDEN_SIZE)
        or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise RuntimeError("captured activation array invariant failed")

    arrays_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_output, **arrays)
    state_metadata = [
        {key: value for key, value in row.items() if key != "sequence"}
        for row in rows
    ]
    result = {
        "study": "native GTEx six-view sparse final-MLP feature model",
        "stage": "label-free activation capture",
        "status": "COMPLETE",
        "evidence_tier": "instrument",
        "protocol": str(protocol.relative_to(REPO)),
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "cohort_namespace": NAMESPACE,
        "partition": "discovery",
        "label_free": True,
        "n_states": len(rows),
        "n_genes": len({row["gene_id"] for row in rows}),
        "views": list(VIEWS),
        "arrays": {
            "path": str(arrays_output),
            "sha256": sha256_file(arrays_output),
            "shape_per_view": [len(rows), HIDDEN_SIZE],
            "keys": sorted(arrays),
        },
        "states": state_metadata,
        "view_records": view_records,
        "instrument_gate": {
            "sequential_checkpoint_capture": True,
            "architecture_6x16x512x2048": True,
            "label_free_payload": True,
            "finite_arrays": True,
            "maximum_input_equality_error": maximum_input_error,
            "input_equality_tolerance": INPUT_EQUALITY_TOLERANCE,
            "maximum_write_reconstruction_error": maximum_write_error,
            "write_reconstruction_tolerance": WRITE_RECONSTRUCTION_TOLERANCE,
            "PASS": True,
        },
        "training_authorized": True,
    }
    assert_label_free_payload(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:sparse_transcoder_activation_capture_v1",
        sys.argv,
        repo=REPO,
        status="INSTRUMENT",
        config={
            "cohort_namespace": NAMESPACE,
            "partition": "discovery",
            "n_states": len(rows),
            "batch": args.batch,
            "device": device,
            "views": list(VIEWS),
        },
        seeds={"checkpoint_family": list(EXPECTED_SEEDS)},
        notes=[
            "Checkpoints were loaded and released sequentially.",
            "No classifier output, margin, label, or external outcome was evaluated.",
            "The trainer consumes only the recorded MLP inputs and writes.",
        ],
    )
    for path in (
        protocol,
        addendum,
        package_path,
        model_weight(base_model_path),
        base_model_path / "config.json",
        base_model_path / "tokenizer.json",
        *checkpoints.values(),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.extend([file_record(output), file_record(arrays_output)])
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "instrument_gate": result["instrument_gate"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
