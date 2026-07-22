#!/usr/bin/env python
"""Development-only model preflight for the P3 closure experiment.

The runner consumes an outcome-blind cohort package, reconstructs every sequence from the pinned
dataset, verifies its SHA-256, and records native coupling plus the site-excluded L3H1 phasor.  It
hard-refuses confirmation-role records.  Threshold calibration and statistical summaries live in a
separate analysis command so this artifact remains an immutable row-level measurement.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

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
from gfmcd.splicing.p3_closure import ContextBlock, rotate_cyclic, site_excluded_phasor  # noqa: E402
from gfmcd.splicing.p3_cohort import sequence_sha256  # noqa: E402


READER = (3, 1)


def load_development_package(path: str | Path) -> dict:
    package = json.loads(Path(path).read_text())
    if package.get("schema") != "gfmcd.p3_closure.cohort.v1":
        raise ValueError("wrong P3 closure cohort schema")
    if package.get("model_outcomes_read") is not False:
        raise ValueError("cohort package is not outcome-blind")
    roles = {row.get("role") for row in package.get("records", ()) if row.get("eligible")}
    if roles - {"development"}:
        raise RuntimeError(f"preflight refuses non-development roles: {sorted(roles)}")
    return package


def reconstruct_variant(example, row: dict):
    block = ContextBlock(int(row["block"][0]), int(row["block"][1]), str(row["context"]))
    sequence = rotate_cyclic(example.sequence, block, int(row["amount"]), str(row["direction"]))
    if sequence_sha256(sequence) != row["sequence_sha256"]:
        raise RuntimeError(f"sequence hash mismatch for {example.example_id}: {row}")
    return replace(example, sequence=sequence)


def evaluate_examples(adapter, task: CouplingTask, examples: list) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for offset in range(0, len(examples), task.batch):
            chunk = examples[offset:offset + task.batch]
            clean, corrupt, donors = task.pair(chunk)
            output = adapter.model(**clean, output_attentions=True)
            clean_margin = task.margin(output.logits.float(), donors)
            corrupt_margin = task.margin(adapter.model(**corrupt).logits.float(), donors)
            attention = output.attentions[READER[0]][:, READER[1]].float().cpu().numpy()
            for index, example in enumerate(chunk):
                query = example.donor_pos + task.off
                full_row = attention[index, query]
                nucleotide_row = full_row[task.off:task.off + len(example.sequence)]
                phasor = site_excluded_phasor(
                    nucleotide_row,
                    example.donor_pos,
                    int(example.next_acceptor_pos),
                    site_radius=6,
                    reference_class=1,
                )
                rows.append({
                    "clean_margin": float(clean_margin[index]),
                    "corrupt_margin": float(corrupt_margin[index]),
                    "coupling": float(clean_margin[index] - corrupt_margin[index]),
                    "phasor_real": float(phasor.value.real),
                    "phasor_imag": float(phasor.value.imag),
                    "phasor_signed": phasor.signed_projection,
                    "phasor_amplitude": phasor.amplitude,
                    "phasor_phase": phasor.phase_angle,
                    "phasor_class_masses": list(phasor.class_masses),
                    "site_excluded_attention_mass": phasor.included_mass,
                    "attention_row_sum_full": float(full_row.sum()),
                    "attention_row_sum_nucleotide": float(nucleotide_row.sum()),
                })
    return rows


def max_abs_difference(left: list[dict[str, object]], right: list[dict[str, object]]) -> float:
    metrics = ("clean_margin", "corrupt_margin", "coupling", "phasor_real", "phasor_imag",
               "phasor_signed", "phasor_amplitude")
    if len(left) != len(right):
        raise ValueError("determinism rows have different lengths")
    return max(
        abs(float(a[metric]) - float(b[metric]))
        for a, b in zip(left, right) for metric in metrics
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit-genes", type=int, default=None)
    parser.add_argument("--health-genes", type=int, default=8)
    parser.add_argument("--health-probes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--out", default="results/phase-lattice/p3_closure_preflight.json")
    args = parser.parse_args()
    if args.batch <= 0 or args.health_genes <= 0 or args.health_probes <= 0:
        parser.error("batch and health counts must be positive")
    if args.limit_genes is not None and args.limit_genes <= 0:
        parser.error("--limit-genes must be positive")
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite preflight artifact: {out}")

    package = load_development_package(args.cohort)
    eligible_records = [row for row in package["records"] if row.get("eligible")]
    gene_order = []
    seen = set()
    for row in eligible_records:
        if row["gene_id"] not in seen:
            seen.add(row["gene_id"])
            gene_order.append(row["gene_id"])
    if args.limit_genes is not None:
        allowed_genes = set(gene_order[:args.limit_genes])
        eligible_records = [row for row in eligible_records if row["gene_id"] in allowed_genes]
    if not eligible_records:
        raise RuntimeError("development package contains no eligible records")

    source = {example.example_id: example for example in _dataset(args.gtf, args.fasta)}
    native_examples = []
    variant_examples = []
    variant_metadata = []
    for record in eligible_records:
        example = source.get(record["example_id"])
        if example is None:
            raise RuntimeError(f"cohort example absent from reconstructed dataset: {record['example_id']}")
        if sequence_sha256(example.sequence) != record["native_sequence_sha256"]:
            raise RuntimeError(f"native sequence hash mismatch: {example.example_id}")
        native_examples.append(example)
        for variant_row in record["variants"]:
            variant_examples.append(reconstruct_variant(example, variant_row))
            variant_metadata.append({
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "chrom": example.chrom,
                "strand": example.strand,
                "donor_phase": record["donor_phase"],
                "context": variant_row["context"],
                "block_index": variant_row["block_index"],
                "block": variant_row["block"],
                "amount": variant_row["amount"],
                "direction": variant_row["direction"],
                "sequence_sha256": variant_row["sequence_sha256"],
            })

    torch.set_grad_enabled(False)
    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=True)
    task = CouplingTask(adapter, batch=args.batch)
    started = time.time()
    print(
        f"P3 closure development preflight: genes={len(set(x.gene_id for x in native_examples))} "
        f"examples={len(native_examples)} variants={len(variant_examples)} device={device}",
        flush=True,
    )
    native_values = evaluate_examples(adapter, task, native_examples)
    variant_values = []
    for start in range(0, len(variant_examples), max(args.batch * 8, 1)):
        stop = min(len(variant_examples), start + args.batch * 8)
        variant_values.extend(evaluate_examples(adapter, task, variant_examples[start:stop]))
        print(f"measured {stop}/{len(variant_examples)} variants", flush=True)

    repeat_n = min(args.batch, len(variant_examples))
    repeated = evaluate_examples(adapter, task, variant_examples[:repeat_n])
    determinism_max_abs = max_abs_difference(variant_values[:repeat_n], repeated)
    attention_live = all(
        math.isfinite(float(row["attention_row_sum_full"]))
        and abs(float(row["attention_row_sum_full"]) - 1.0) <= 1e-4
        and float(row["site_excluded_attention_mass"]) > 0
        for row in native_values + variant_values
    )

    health_native_examples = native_examples[:args.health_genes]
    # A deterministic off-register CDS state is the load-bearing health comparison.
    health_variant_examples = [
        variant for variant, meta in zip(variant_examples, variant_metadata)
        if meta["context"] == "cds" and meta["block_index"] == 0
        and meta["amount"] == 2 and meta["direction"] == "left"
    ][:args.health_genes]
    native_health = task.mlm_health(
        health_native_examples, n_probe=args.health_probes, seed=args.seed
    )
    variant_health = task.mlm_health(
        health_variant_examples, n_probe=args.health_probes, seed=args.seed
    )
    health_ratio = variant_health / native_health if native_health > 0 else float("nan")

    result_rows = []
    for metadata, values in zip(variant_metadata, variant_values):
        result_rows.append({**metadata, **values})
    native_rows = []
    for example, values in zip(native_examples, native_values):
        native_rows.append({
            "example_id": example.example_id,
            "gene_id": example.gene_id,
            "chrom": example.chrom,
            "strand": example.strand,
            **values,
        })
    result = {
        "schema": "gfmcd.p3_closure.preflight.v1",
        "status": "DEVELOPMENT_OUTCOME_AWARE",
        "cohort_sha256": sha256_file(args.cohort),
        "device": device,
        "counts": {
            "examples": len(native_examples),
            "genes": len({x.gene_id for x in native_examples}),
            "variants": len(result_rows),
        },
        "instrument": {
            "attention_live_and_normalized": attention_live,
            "determinism_max_abs": determinism_max_abs,
            "native_mlm_health": native_health,
            "offregister_cds_mlm_health": variant_health,
            "health_ratio": health_ratio,
        },
        "runtime_seconds": time.time() - started,
        "native_rows": native_rows,
        "rows": result_rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        "p3-closure-preflight-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_OUTCOME_AWARE",
        config={
            "batch": args.batch,
            "limit_genes": args.limit_genes,
            "health_genes": args.health_genes,
            "health_probes": args.health_probes,
            "seed": args.seed,
            "device": device,
        },
        seeds={"analysis": args.seed, "python_hash": 0},
        notes=["Development roles only; runner hard-refuses confirmation records."],
    )
    for path in (
        args.cohort, args.gtf, args.fasta, args.model,
        REPO / "src/gfmcd/splicing/p3_closure.py",
        REPO / "scripts/splicebert/run_p3_closure_preflight.py",
    ):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({"counts": result["counts"], "instrument": result["instrument"],
                      "runtime_seconds": result["runtime_seconds"]}, indent=2))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
