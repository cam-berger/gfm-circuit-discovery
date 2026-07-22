#!/usr/bin/env python
"""Development-only causal preflight for local P3 carrier phase surgery."""
from __future__ import annotations

import argparse
import json
import math
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
from gfmcd.splicing.p3_closure import local_recovery_contrast, site_excluded_phasor  # noqa: E402
from gfmcd.splicing.p3_cohort import sequence_sha256  # noqa: E402
from gfmcd.splicing.p3_phase_surgery import (  # noqa: E402
    P3_BAND_1026,
    PhaseSurgerySpec,
    fourier_band_component,
    local_phase_surgery,
    surgery_rms,
)


READER = (3, 1)


def load_development_package(path: str | Path) -> dict:
    package = json.loads(Path(path).read_text())
    if package.get("schema") != "gfmcd.p3_phase_surgery.cohort.v1":
        raise ValueError("wrong P3 phase-surgery cohort schema")
    if package.get("model_outcomes_read") is not False:
        raise ValueError("phase-surgery cohort is not outcome-blind")
    roles = {row.get("role") for row in package.get("records", ()) if row.get("eligible")}
    if roles - {"development"}:
        raise RuntimeError(f"preflight refuses non-development roles: {sorted(roles)}")
    return package


def evaluate_batch(adapter, task: CouplingTask, examples: list, *, specs=None,
                   component=None) -> list[dict[str, object]]:
    if (specs is None) != (component is None):
        raise ValueError("surgery specs and component must be supplied together")
    if specs is not None and len(specs) != len(examples):
        raise ValueError("one surgery spec is required per example")
    embedding = adapter.model.bert.embeddings.position_embeddings
    context = (
        local_phase_surgery(embedding, specs, component)
        if specs is not None else torch.no_grad()
    )
    with torch.no_grad(), context:
        clean, corrupt, donors = task.pair(examples)
        output = adapter.model(**clean, output_attentions=True)
        clean_margin = task.margin(output.logits.float(), donors)
        corrupt_margin = task.margin(adapter.model(**corrupt).logits.float(), donors)
        attention = output.attentions[READER[0]][:, READER[1]].float().cpu().numpy()
    rows = []
    for index, example in enumerate(examples):
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
        })
    return rows


def max_abs_difference(left, right) -> float:
    metrics = ("clean_margin", "corrupt_margin", "coupling", "phasor_real", "phasor_imag",
               "phasor_signed", "phasor_amplitude")
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
    parser.add_argument(
        "--out", default="results/phase-lattice/p3_phase_surgery_preflight.json"
    )
    args = parser.parse_args()
    if min(args.batch, args.health_genes, args.health_probes) <= 0:
        parser.error("batch and health counts must be positive")
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite phase-surgery preflight: {out}")

    package = load_development_package(args.cohort)
    records = [row for row in package["records"] if row.get("eligible")]
    if args.limit_genes is not None:
        allowed = {row["gene_id"] for row in records[:args.limit_genes]}
        records = [row for row in records if row["gene_id"] in allowed]
    if not records:
        raise RuntimeError("no development records in phase-surgery package")
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
    if embedding.weight.shape[0] != package["config"]["position_table_length"]:
        raise RuntimeError("position-table length differs from cohort package")
    # Fourier decomposition is tiny and CPU execution avoids backend-specific FFT warnings on MPS.
    # The intervention hook moves the frozen component to the model device/dtype at application time.
    position_weight_cpu = embedding.weight.detach().cpu()
    component = fourier_band_component(position_weight_cpu, P3_BAND_1026)
    reconstruction_error = float(
        ((position_weight_cpu - component) + component - position_weight_cpu).abs().max()
    )

    native_rows = []
    for start in range(0, len(examples), args.batch):
        native_rows.extend(evaluate_batch(adapter, task, examples[start:start + args.batch]))

    variant_examples = []
    variant_specs = []
    variant_meta = []
    rms_rows = []
    prefix = int(package["config"]["prefix_tokens"])
    target_context = str(package["config"].get("target_context", "cds"))
    for record, example in zip(records, examples):
        for context in (target_context, "upstream_intron"):
            for block_index, block in enumerate(record["blocks"][context]):
                token_start, token_end = int(block[0]) + prefix, int(block[1]) + prefix
                for amount in range(2, 11):
                    for direction in ("left", "right"):
                        spec = PhaseSurgerySpec(token_start, token_end, amount, direction)
                        variant_examples.append(example)
                        variant_specs.append(spec)
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
                        rms_rows.append({
                            "context": context,
                            "amount": amount,
                            "direction": direction,
                            "rms": surgery_rms(component, spec),
                        })

    started = time.time()
    print(
        f"P3 phase surgery: genes={len(records)} variants={len(variant_examples)} device={device}",
        flush=True,
    )
    variant_values = []
    for start in range(0, len(variant_examples), args.batch):
        stop = min(len(variant_examples), start + args.batch)
        variant_values.extend(evaluate_batch(
            adapter,
            task,
            variant_examples[start:stop],
            specs=variant_specs[start:stop],
            component=component,
        ))
        if stop % (args.batch * 16) == 0 or stop == len(variant_examples):
            print(f"measured {stop}/{len(variant_examples)} variants", flush=True)

    repeat_n = min(args.batch, len(variant_examples))
    repeated = evaluate_batch(
        adapter, task, variant_examples[:repeat_n], specs=variant_specs[:repeat_n],
        component=component,
    )
    determinism = max_abs_difference(variant_values[:repeat_n], repeated)
    zero_specs = [
        PhaseSurgerySpec(spec.start, spec.end, 0, spec.direction)
        for spec in variant_specs[:repeat_n]
    ]
    zero_rows = evaluate_batch(
        adapter, task, variant_examples[:repeat_n], specs=zero_specs, component=component
    )
    native_by_example = {
        row["example_id"]: value for row, value in zip(records, native_rows)
    }
    no_op_error = max_abs_difference(
        [native_by_example[meta["example_id"]] for meta in variant_meta[:repeat_n]], zero_rows
    )

    mechanical = {}
    for direction in ("left", "right"):
        values = {
            amount: float(np.mean([
                row["rms"] for row in rms_rows
                if row["direction"] == direction and row["amount"] == amount
            ]))
            for amount in range(2, 11)
        }
        mechanical[direction] = {
            "mean_rms_by_amount": values,
            "off_minus_inregister_rms": -local_recovery_contrast(values),
        }

    health_examples = examples[:args.health_genes]
    native_health = task.mlm_health(health_examples, n_probe=args.health_probes, seed=args.seed)
    health_specs = []
    for record in records[:len(health_examples)]:
        block = record["blocks"][target_context][0]
        health_specs.append(PhaseSurgerySpec(block[0] + prefix, block[1] + prefix, 2, "left"))
    # mlm_health accepts only a global context factory, so evaluate one example/spec at a time.
    surgery_health_values = []
    for example, spec in zip(health_examples, health_specs):
        surgery_health_values.append(task.mlm_health(
            [example],
            ctx=lambda spec=spec: local_phase_surgery(embedding, [spec], component),
            n_probe=args.health_probes,
            seed=args.seed,
        ))
    surgery_health = float(np.mean(surgery_health_values))

    rows = [{**meta, **value} for meta, value in zip(variant_meta, variant_values)]
    native = [
        {
            "example_id": example.example_id,
            "gene_id": example.gene_id,
            "chrom": example.chrom,
            "strand": example.strand,
            "donor_phase": record["donor_phase"],
            **value,
        }
        for record, example, value in zip(records, examples, native_rows)
    ]
    result = {
        "schema": "gfmcd.p3_phase_surgery.preflight.v1",
        "status": "DEVELOPMENT_OUTCOME_AWARE",
        "cohort_sha256": sha256_file(args.cohort),
        "p3_band": list(P3_BAND_1026),
        "device": device,
        "target_context": target_context,
        "counts": {"genes": len(records), "examples": len(examples), "variants": len(rows)},
        "instrument": {
            "carrier_reconstruction_error": reconstruction_error,
            "determinism_max_abs": determinism,
            "zero_shift_noop_max_abs": no_op_error,
            "attention_live_and_normalized": all(
                math.isfinite(float(row["attention_row_sum_full"]))
                and abs(float(row["attention_row_sum_full"]) - 1.0) <= 1e-4
                and float(row["site_excluded_attention_mass"]) > 0
                for row in native + rows
            ),
            "native_mlm_health": native_health,
            "offregister_surgery_mlm_health": surgery_health,
            "health_ratio": surgery_health / native_health if native_health > 0 else float("nan"),
            "mechanical_modulo_profile": mechanical,
        },
        "runtime_seconds": time.time() - started,
        "native_rows": native,
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "p3-phase-surgery-preflight-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_OUTCOME_AWARE",
        config={"batch": args.batch, "limit_genes": args.limit_genes, "band": P3_BAND_1026,
                "seed": args.seed},
        seeds={"analysis": args.seed, "python_hash": 0},
        notes=["Every nucleotide and token ID is unchanged; development roles only."],
    )
    for path in (
        args.cohort, args.gtf, args.fasta,
        REPO / "src/gfmcd/splicing/p3_phase_surgery.py",
        REPO / "scripts/splicebert/run_p3_phase_surgery_preflight.py",
    ):
        manifest.add_input(path)
    model_path = Path(args.model) if Path(args.model).is_absolute() else REPO / args.model
    for checkpoint_file in sorted(model_path.glob("*")):
        if checkpoint_file.is_file():
            manifest.add_input(checkpoint_file)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({"counts": result["counts"], "instrument": result["instrument"],
                      "runtime_seconds": result["runtime_seconds"]}, indent=2))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
