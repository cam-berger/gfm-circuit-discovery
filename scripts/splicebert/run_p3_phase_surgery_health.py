#!/usr/bin/env python
"""Paired continuous MLM-health calibration for local P3 phase surgery."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

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
    local_phase_surgery,
)
from run_p3_phase_surgery_preflight import load_development_package  # noqa: E402


NONINFERIORITY_NATS = -0.05


def stable_probe_positions(example, *, n_probe: int, seed: int) -> tuple[int, ...]:
    candidates = [
        position for position in range(10, len(example.sequence) - 10)
        if abs(position - example.donor_pos) > 2
        and abs(position - int(example.next_acceptor_pos)) > 3
    ]
    if len(candidates) < n_probe:
        raise ValueError("not enough nonsite MLM probe positions")
    digest = hashlib.sha256(f"p3-health\0{seed}\0{example.example_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return tuple(sorted(int(x) for x in rng.choice(candidates, size=n_probe, replace=False)))


def paired_mean_ci(values: list[float], *, seed: int, draws: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 3 or not np.isfinite(array).all() or draws < 999:
        raise ValueError("health CI needs at least three genes and 999 draws")
    rng = np.random.default_rng(seed)
    means = np.mean(array[rng.integers(0, len(array), size=(draws, len(array)))], axis=1)
    return tuple(float(x) for x in np.quantile(means, (0.025, 0.975)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit-genes", type=int, default=None)
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--out", default="results/phase-lattice/p3_phase_surgery_health.json"
    )
    args = parser.parse_args()
    if min(args.batch, args.probes) <= 0 or args.bootstrap < 999:
        parser.error("batch/probes must be positive and bootstrap >=999")
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite phase-surgery health result: {out}")

    package = load_development_package(args.cohort)
    records = [row for row in package["records"] if row.get("eligible")]
    if args.limit_genes is not None:
        records = records[:args.limit_genes]
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
    component = fourier_band_component(embedding.weight.detach().cpu(), P3_BAND_1026)
    prefix = int(package["config"]["prefix_tokens"])
    target_context = str(package["config"].get("target_context", "cds"))
    rows = []
    for start in range(0, len(examples), args.batch):
        stop = min(len(examples), start + args.batch)
        batch_examples = examples[start:stop]
        batch_records = records[start:stop]
        donors = [example.donor_pos for example in batch_examples]
        encoded = task.encode([example.sequence for example in batch_examples], donors)
        probes = [
            stable_probe_positions(example, n_probe=args.probes, seed=args.seed)
            for example in batch_examples
        ]
        for index, positions in enumerate(probes):
            for position in positions:
                encoded["input_ids"][index, position + task.off] = task.mask_id
        specs = []
        for record in batch_records:
            block = record["blocks"][target_context][0]
            specs.append(PhaseSurgerySpec(
                int(block[0]) + prefix, int(block[1]) + prefix, 2, "left"
            ))
        with torch.no_grad():
            native_logits = adapter.model(**encoded).logits.float()
            with local_phase_surgery(embedding, specs, component):
                surgery_logits = adapter.model(**encoded).logits.float()
        for index, (example, record, positions) in enumerate(zip(
            batch_examples, batch_records, probes
        )):
            token_positions = torch.tensor(
                [position + task.off for position in positions], device=native_logits.device
            )
            targets = torch.tensor(
                [task.bid[example.sequence[position]] for position in positions],
                device=native_logits.device,
            )
            native_at = native_logits[index, token_positions]
            surgery_at = surgery_logits[index, token_positions]
            probe_index = torch.arange(len(targets), device=native_logits.device)
            native_lp = torch.log_softmax(native_at, -1)[probe_index, targets]
            surgery_lp = torch.log_softmax(surgery_at, -1)[probe_index, targets]
            rows.append({
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "chrom": example.chrom,
                "target_context": target_context,
                "n_probes": len(positions),
                "probe_positions": list(positions),
                "native_true_logprob": float(native_lp.mean()),
                "surgery_true_logprob": float(surgery_lp.mean()),
                "delta_true_logprob": float((surgery_lp - native_lp).mean()),
                "native_accuracy": float((native_at.argmax(-1) == targets).float().mean()),
                "surgery_accuracy": float((surgery_at.argmax(-1) == targets).float().mean()),
            })
        print(f"health {stop}/{len(examples)} genes", flush=True)

    deltas = [float(row["delta_true_logprob"]) for row in rows]
    ci = paired_mean_ci(deltas, seed=args.seed, draws=args.bootstrap)
    native_accuracy = float(np.mean([row["native_accuracy"] for row in rows]))
    surgery_accuracy = float(np.mean([row["surgery_accuracy"] for row in rows]))
    result = {
        "schema": "gfmcd.p3_phase_surgery.health.v1",
        "status": "DEVELOPMENT_OUTCOME_AWARE",
        "cohort_sha256": sha256_file(args.cohort),
        "target_context": target_context,
        "intervention": {"block_index": 0, "amount": 2, "direction": "left"},
        "noninferiority_floor_delta_true_logprob": NONINFERIORITY_NATS,
        "counts": {"genes": len(rows), "probes_per_gene": args.probes},
        "summary": {
            "native_mean_true_logprob": float(np.mean([
                row["native_true_logprob"] for row in rows
            ])),
            "surgery_mean_true_logprob": float(np.mean([
                row["surgery_true_logprob"] for row in rows
            ])),
            "paired_mean_delta_true_logprob": float(np.mean(deltas)),
            "paired_delta_ci": list(ci),
            "native_accuracy": native_accuracy,
            "surgery_accuracy": surgery_accuracy,
            "accuracy_ratio": surgery_accuracy / native_accuracy if native_accuracy > 0 else math.nan,
            "pass_continuous_noninferiority": ci[0] > NONINFERIORITY_NATS,
        },
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "p3-phase-surgery-health-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_OUTCOME_AWARE",
        config={
            "batch": args.batch,
            "probes": args.probes,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "noninferiority_nats": NONINFERIORITY_NATS,
        },
        seeds={"probe_and_bootstrap": args.seed},
        notes=["Paired continuous health replaces the 32-probe top-1 boundary diagnostic."],
    )
    for path in (
        args.cohort,
        args.gtf,
        args.fasta,
        REPO / "src/gfmcd/splicing/p3_phase_surgery.py",
        REPO / "scripts/splicebert/run_p3_phase_surgery_health.py",
    ):
        manifest.add_input(path)
    model_path = Path(args.model) if Path(args.model).is_absolute() else REPO / args.model
    for checkpoint_file in sorted(model_path.glob("*")):
        if checkpoint_file.is_file():
            manifest.add_input(checkpoint_file)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
