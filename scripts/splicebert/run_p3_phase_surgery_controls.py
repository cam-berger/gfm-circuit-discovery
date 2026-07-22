#!/usr/bin/env python
"""Development-only matched-direction controls for local P3 carrier phase surgery.

Each null keeps the P3 spatial waveform and every intervention RMS exactly fixed while applying an
independently seeded signed permutation to the hidden axis.  The target therefore has to outperform
controls with the same band, phase, rank, spatial Gram matrix, block, and dose—not merely an unrelated
low-energy Fourier perturbation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    CouplingTask,
    _dataset,
    load_adapter,
    pick_device,
)
from gfmcd.splicing.p3_cohort import sequence_sha256  # noqa: E402
from gfmcd.splicing.p3_phase_surgery import (  # noqa: E402
    P3_BAND_1026,
    PhaseSurgerySpec,
    fourier_band_component,
    phase_shift_delta,
    signed_permutation_component,
)
from run_p3_phase_surgery_preflight import (  # noqa: E402
    evaluate_batch,
    load_development_package,
    max_abs_difference,
)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@contextmanager
def position_weight_state(embedding: torch.nn.Embedding, weight: torch.Tensor):
    """Temporarily replace the complete position table and restore it exactly."""

    if weight.shape != embedding.weight.shape:
        raise ValueError("replacement position table has the wrong shape")
    original = embedding.weight.detach().clone()
    with torch.no_grad():
        embedding.weight.copy_(weight.to(device=original.device, dtype=original.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            embedding.weight.copy_(original)


def validate_preflight(path: str | Path, cohort_path: str | Path) -> dict:
    result = json.loads(Path(path).read_text())
    if result.get("schema") != "gfmcd.p3_phase_surgery.preflight.v1":
        raise ValueError("wrong phase-surgery preflight schema")
    if result.get("status") != "DEVELOPMENT_OUTCOME_AWARE":
        raise RuntimeError("control calibration requires a development preflight")
    if result.get("cohort_sha256") != sha256_file(cohort_path):
        raise RuntimeError("preflight/cohort hash mismatch")
    instrument = result.get("instrument", {})
    gates = (
        float(instrument.get("carrier_reconstruction_error", np.inf)) <= 1e-6,
        float(instrument.get("determinism_max_abs", np.inf)) <= 1e-8,
        float(instrument.get("zero_shift_noop_max_abs", np.inf)) <= 1e-8,
        bool(instrument.get("attention_live_and_normalized", False)),
        float(instrument.get("health_ratio", -np.inf)) >= 0.8,
    )
    if not all(gates):
        raise RuntimeError("target phase-surgery preflight gates are not all green")
    return result


def build_variant_grid(
    records: list[dict], examples: list, *, prefix: int, target_context: str = "cds"
):
    variant_examples = []
    variant_specs = []
    variant_meta = []
    for record, example in zip(records, examples):
        for context in (target_context, "upstream_intron"):
            for block_index, block in enumerate(record["blocks"][context]):
                start, end = int(block[0]) + prefix, int(block[1]) + prefix
                for direction in ("left", "right"):
                    for amount in (2, 3, 4):
                        variant_examples.append(example)
                        variant_specs.append(PhaseSurgerySpec(start, end, amount, direction))
                        variant_meta.append({
                            "example_id": example.example_id,
                            "gene_id": example.gene_id,
                            "chrom": example.chrom,
                            "strand": example.strand,
                            "donor_phase": record["donor_phase"],
                            "context": context,
                            "block_index": block_index,
                            "block": block,
                            "amount": amount,
                            "direction": direction,
                        })
    return variant_examples, variant_specs, variant_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit-genes", type=int, default=30)
    parser.add_argument("--controls", type=int, default=19)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--out", default="results/phase-lattice/p3_phase_surgery_controls.json"
    )
    args = parser.parse_args()
    if min(args.batch, args.limit_genes, args.controls) <= 0:
        parser.error("batch, gene limit, and control count must be positive")
    if args.controls < 19:
        parser.error("at least 19 controls are required for add-one p <= 0.05")
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite phase-surgery controls: {out}")

    package = load_development_package(args.cohort)
    preflight = validate_preflight(args.preflight, args.cohort)
    records = [row for row in package["records"] if row.get("eligible")]
    allowed = {row["gene_id"] for row in records[:args.limit_genes]}
    records = [row for row in records if row["gene_id"] in allowed]
    source = {example.example_id: example for example in _dataset(args.gtf, args.fasta)}
    examples = []
    for record in records:
        example = source.get(record["example_id"])
        if example is None or sequence_sha256(example.sequence) != record["native_sequence_sha256"]:
            raise RuntimeError(f"source reconstruction failed for {record['example_id']}")
        examples.append(example)

    torch.set_grad_enabled(False)
    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=True)
    task = CouplingTask(adapter, batch=args.batch)
    embedding = adapter.model.bert.embeddings.position_embeddings
    original_weight = embedding.weight.detach().clone()
    component = fourier_band_component(original_weight.cpu(), P3_BAND_1026)
    variants, specs, meta = build_variant_grid(
        records,
        examples,
        prefix=int(package["config"]["prefix_tokens"]),
        target_context=str(package["config"].get("target_context", "cds")),
    )

    labels = [("p3", None, component)]
    for index in range(args.controls):
        control_seed = args.seed + index
        labels.append((
            f"signed_permutation_{index:02d}",
            control_seed,
            signed_permutation_component(component, seed=control_seed),
        ))

    started = time.time()
    rows = []
    control_checks = []
    deterministic = []
    for label_index, (label, control_seed, local_component) in enumerate(labels):
        max_rms_error = max(
            abs(float(torch.sqrt(torch.mean(phase_shift_delta(local_component, spec).float() ** 2)))
                - float(torch.sqrt(torch.mean(phase_shift_delta(component, spec).float() ** 2))))
            for spec in set(specs)
        )
        control_checks.append({
            "component": label,
            "seed": control_seed,
            "sha256": tensor_sha256(local_component),
            "l2_ratio_to_p3": float(local_component.norm() / component.norm()),
            "max_shift_rms_error": max_rms_error,
        })
        values = []
        print(
            f"component {label_index + 1}/{len(labels)} {label}: {len(variants)} variants",
            flush=True,
        )
        for start in range(0, len(variants), args.batch):
            stop = min(len(variants), start + args.batch)
            values.extend(evaluate_batch(
                adapter,
                task,
                variants[start:stop],
                specs=specs[start:stop],
                component=local_component,
            ))
        repeat_n = min(args.batch, len(variants))
        if label_index in (0, 1):
            repeated = evaluate_batch(
                adapter,
                task,
                variants[:repeat_n],
                specs=specs[:repeat_n],
                component=local_component,
            )
            deterministic.append({
                "component": label,
                "max_abs": max_abs_difference(values[:repeat_n], repeated),
            })
        rows.extend({
            **description,
            "component": label,
            "control_seed": control_seed,
            **value,
        } for description, value in zip(meta, values))

    positive_examples = examples[:min(12, len(examples))]
    native_positive = evaluate_batch(adapter, task, positive_examples)
    deleted_weight = original_weight - component.to(
        device=original_weight.device, dtype=original_weight.dtype
    )
    with position_weight_state(embedding, deleted_weight):
        deleted_positive = evaluate_batch(adapter, task, positive_examples)
    if not torch.equal(embedding.weight.detach(), original_weight):
        raise RuntimeError("global-deletion positive control did not restore checkpoint exactly")
    native_c = float(np.mean([row["coupling"] for row in native_positive]))
    deleted_c = float(np.mean([row["coupling"] for row in deleted_positive]))
    positive = {
        "n_examples": len(positive_examples),
        "native_mean_coupling": native_c,
        "p3_deleted_mean_coupling": deleted_c,
        "deleted_over_native": deleted_c / native_c if native_c > 0 else float("nan"),
        "pass": native_c > 0.05 and deleted_c / native_c <= 0.10,
        "note": "substantive C8-style positive control; not an algebraic reconstruction identity",
    }
    result = {
        "schema": "gfmcd.p3_phase_surgery.controls.v1",
        "status": "DEVELOPMENT_OUTCOME_AWARE",
        "cohort_sha256": sha256_file(args.cohort),
        "preflight_sha256": sha256_file(args.preflight),
        "target_context": str(package["config"].get("target_context", "cds")),
        "counts": {
            "genes": len(records),
            "components": len(labels),
            "controls": args.controls,
            "variants_per_component": len(variants),
            "rows": len(rows),
        },
        "control_design": {
            "kind": "signed_hidden_axis_permutation",
            "p3_band": list(P3_BAND_1026),
            "amounts": [2, 3, 4],
            "directions": ["left", "right"],
            "blocks_per_context": 2,
            "preserved": [
                "nucleotide_sequence", "Fourier_support", "spatial_Gram_matrix",
                "row_norms", "phase_shift_RMS", "block", "dose",
            ],
        },
        "instrument": {
            "target_preflight_instrument": preflight["instrument"],
            "control_geometry": control_checks,
            "determinism": deterministic,
            "global_p3_deletion_positive": positive,
        },
        "runtime_seconds": time.time() - started,
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "p3-phase-surgery-controls-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_OUTCOME_AWARE",
        config={
            "batch": args.batch,
            "limit_genes": args.limit_genes,
            "controls": args.controls,
            "seed": args.seed,
            "amounts": [2, 3, 4],
        },
        seeds={"control_seed_start": args.seed, "python_hash": 0},
        notes=[
            "Every nucleotide is unchanged.",
            "Nulls preserve P3 spatial geometry and intervention RMS exactly.",
        ],
    )
    for path in (
        args.cohort,
        args.preflight,
        args.gtf,
        args.fasta,
        REPO / "src/gfmcd/splicing/p3_phase_surgery.py",
        REPO / "scripts/splicebert/run_p3_phase_surgery_controls.py",
    ):
        manifest.add_input(path)
    model_path = Path(args.model) if Path(args.model).is_absolute() else REPO / args.model
    for checkpoint_file in sorted(model_path.glob("*")):
        if checkpoint_file.is_file():
            manifest.add_input(checkpoint_file)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({
        "counts": result["counts"],
        "global_p3_deletion_positive": positive,
        "max_geometry_error": max(row["max_shift_rms_error"] for row in control_checks),
        "determinism": deterministic,
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
