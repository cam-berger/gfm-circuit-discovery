#!/usr/bin/env python
"""Capture tuned final-MLP activations for the fresh J-SAE cohort."""

from __future__ import annotations

import argparse
import gc
import importlib.util
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
from gfmcd.provenance import (  # noqa: E402
    ExperimentManifest,
    file_record,
    sha256_file,
)
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402


def load_script(name: str) -> Any:
    path = REPO / "scripts/splicebert" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    spec.loader.exec_module(module)
    return module


CAPTURE = load_script("run_gtex_sparse_transcoder_activation_capture.py")
COHORT = load_script("build_gtex_jacobian_sae_fresh_cohort.py")
EXPECTED_SEEDS = tuple(range(2020, 2025))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
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

    cohort_path = Path(args.cohort).expanduser().resolve()
    cohort = json.loads(cohort_path.read_text())
    rows = cohort.get("states")
    if (
        cohort.get("status") != "FROZEN"
        or cohort.get("namespace") != COHORT.NAMESPACE
        or cohort.get("capture_authorized") is not True
        or cohort.get("classifier_output_observed") is not False
        or not isinstance(rows, list)
        or len(rows) != 768
    ):
        raise RuntimeError("fresh cohort contract mismatch")
    checkpoints = CAPTURE.parse_checkpoints(args.checkpoint)
    base_model = Path(args.base_model).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    arrays_output = guard_output_path(Path(args.arrays_out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    if len({output, arrays_output, manifest_path}) != 3:
        raise ValueError("capture outputs must be distinct")
    if output.suffix != ".json" or arrays_output.suffix != ".npz":
        raise ValueError("capture requires .json and .npz outputs")
    device = pick_device(args.device)
    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_FRESH_DELTA_CONFIRMATION_V1.md"
    )

    arrays: dict[str, np.ndarray] = {}
    view_records: dict[str, Any] = {}
    maximum_input_error = 0.0
    maximum_write_error = 0.0
    for seed in EXPECTED_SEEDS:
        view = str(seed)
        model, tokenizer, report = load_gtex_token_classifier(
            base_model,
            checkpoints[seed],
            device=device,
            attn_implementation="eager",
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        adapter = SpliceBertAdapter(
            model,
            tokenizer,
            source=str(checkpoints[seed]),
            device=device,
        )
        dense = adapter.encoder_layers[CAPTURE.FINAL_LAYER].output.dense
        if (
            adapter.num_layers != 6
            or adapter.num_heads != 16
            or adapter.hidden_size != CAPTURE.HIDDEN_SIZE
            or dense.in_features != CAPTURE.INTERMEDIATE_SIZE
        ):
            raise RuntimeError("checkpoint architecture mismatch")
        input_chunks: list[np.ndarray] = []
        write_chunks: list[np.ndarray] = []
        view_input_error = 0.0
        view_write_error = 0.0
        for start in range(0, len(rows), args.batch):
            stop = min(start + args.batch, len(rows))
            chunk = rows[start:stop]
            encoded = CAPTURE.encode_sequences(
                tokenizer,
                [str(row["sequence"]) for row in chunk],
                device=device,
            )
            captured = CAPTURE.capture_batch(
                adapter,
                model,
                encoded,
                [int(row["target_position"]) for row in chunk],
            )
            input_chunks.append(captured["input"].numpy())
            write_chunks.append(captured["write"].numpy())
            view_input_error = max(
                view_input_error, float(captured["input_error"])
            )
            view_write_error = max(
                view_write_error, float(captured["write_error"])
            )
            print(f"seed {seed} {stop}/{len(rows)}", end="\r", flush=True)
        print()
        if (
            view_input_error > CAPTURE.INPUT_EQUALITY_TOLERANCE
            or view_write_error > CAPTURE.WRITE_RECONSTRUCTION_TOLERANCE
        ):
            raise RuntimeError(
                f"capture failed seed {seed}: input={view_input_error}, "
                f"write={view_write_error}"
            )
        arrays[f"{view}__input"] = np.concatenate(input_chunks).astype(
            np.float32, copy=False
        )
        arrays[f"{view}__write"] = np.concatenate(write_chunks).astype(
            np.float32, copy=False
        )
        view_records[view] = {
            "checkpoint_path": str(checkpoints[seed]),
            "checkpoint_sha256": report.checkpoint_sha256,
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
        value.shape != (len(rows), CAPTURE.HIDDEN_SIZE)
        or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise RuntimeError("fresh activation arrays failed shape/finite invariant")

    arrays_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_output, **arrays)
    state_metadata = [
        {key: value for key, value in row.items() if key != "sequence"}
        for row in rows
    ]
    result = {
        "study": "SpliceBERT GTEx layer-5 Jacobian SAE",
        "stage": "fresh-cohort label-free activation capture",
        "status": "COMPLETE",
        "evidence_tier": "instrument",
        "namespace": COHORT.NAMESPACE,
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "cohort": str(cohort_path),
        "cohort_sha256": sha256_file(cohort_path),
        "label_free": True,
        "classifier_output_observed": False,
        "n_states": len(rows),
        "n_genes": len({str(row["gene_id"]) for row in rows}),
        "arrays": {
            "path": str(arrays_output),
            "sha256": None,
            "shape_per_view": [len(rows), CAPTURE.HIDDEN_SIZE],
            "keys": sorted(arrays),
        },
        "states": state_metadata,
        "view_records": view_records,
        "instrument_gate": {
            "sequential_checkpoint_capture": True,
            "finite_arrays": True,
            "maximum_input_equality_error": maximum_input_error,
            "input_equality_tolerance": CAPTURE.INPUT_EQUALITY_TOLERANCE,
            "maximum_write_reconstruction_error": maximum_write_error,
            "write_reconstruction_tolerance": (
                CAPTURE.WRITE_RECONSTRUCTION_TOLERANCE
            ),
            "PASS": True,
        },
        "confirmation_authorized": True,
    }
    result["arrays"]["sha256"] = sha256_file(arrays_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:jacobian_sae_fresh_capture_v1",
        sys.argv,
        repo=REPO,
        status="INSTRUMENT",
        config={
            "namespace": COHORT.NAMESPACE,
            "batch": args.batch,
            "device": device,
            "n_states": len(rows),
        },
        seeds={"checkpoint_family": list(EXPECTED_SEEDS)},
        notes=[
            "Checkpoints were loaded and released sequentially.",
            "No classifier output, margin, or SAE feature was evaluated.",
        ],
    )
    for path in (addendum, cohort_path, *checkpoints.values()):
        manifest.inputs.append(file_record(path))
    manifest.outputs.extend([file_record(output), file_record(arrays_output)])
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "instrument_gate": result["instrument_gate"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
