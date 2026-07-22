#!/usr/bin/env python
"""Sealed, resumable runner for the P3 phase-surgery confirmation.

The target phase writes immutable per-unit shards and prints progress counts only.  It never computes or
prints a biological summary.  The independent analyzer is the only code allowed to combine shards.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gfmcd.provenance import file_record, git_dirty, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import CouplingTask, _dataset, load_adapter, pick_device  # noqa: E402
from gfmcd.splicing.p3_cohort import sequence_sha256  # noqa: E402
from gfmcd.splicing.p3_phase_surgery import (  # noqa: E402
    P3_BAND_1026,
    PhaseSurgerySpec,
    fourier_band_component,
    signed_permutation_component,
)
from run_p3_phase_surgery_preflight import evaluate_batch  # noqa: E402


def current_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def is_ancestor(older: str, newer: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        cwd=REPO,
        check=False,
    ).returncode == 0


def verify_file_record(record: dict) -> None:
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
        raise RuntimeError(f"bound file missing or wrong size: {path}")
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"bound file hash mismatch: {path}")


def load_contract(package_path: str | Path, audit_path: str | Path, *, require_clean: bool) -> dict:
    package = json.loads(Path(package_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if package.get("schema") != "gfmcd.p3_phase_surgery.confirmation_package.v1":
        raise ValueError("wrong confirmation-package schema")
    if package.get("status") != "FROZEN_OUTCOME_BLIND_CONFIRMATION_PACKAGE":
        raise RuntimeError("confirmation package is not frozen")
    if audit.get("schema") != "gfmcd.p3_phase_surgery.confirmation_package_audit.v1":
        raise ValueError("wrong confirmation-package audit schema")
    if audit.get("status") != "PASS" or audit.get("package_sha256") != sha256_file(package_path):
        raise RuntimeError("confirmation package audit is absent, failed, or for another package")
    if require_clean and git_dirty(REPO):
        raise RuntimeError("sealed target runner requires a clean worktree")
    revision = str(package.get("freeze_revision") or "")
    if not revision or not is_ancestor(revision, current_head()):
        raise RuntimeError("frozen code revision is not an ancestor of the execution revision")
    for record in package["source_artifacts"].values():
        verify_file_record(record)
    for record in package["code"].values():
        verify_file_record(record)
    for key, record in package["data"].items():
        if key == "model_files":
            for item in record:
                verify_file_record(item)
        else:
            verify_file_record(record)
    return package


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"immutable shard collision: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary shard exists: {temporary}")
    temporary.write_text(encoded)
    temporary.replace(path)


def reconstruct_records(records: list[dict], source: dict[str, object]) -> list[object]:
    output = []
    for record in records:
        example = source.get(record["example_id"])
        if example is None or sequence_sha256(example.sequence) != record["native_sequence_sha256"]:
            raise RuntimeError(f"source reconstruction failed for {record['example_id']}")
        output.append(example)
    return output


def variant_grid(record: dict, example: object, *, amounts: list[int]):
    examples = []
    specs = []
    metadata = []
    prefix = 1
    target_context = str(record["target_context"])
    for context in (target_context, "upstream_intron"):
        for block_index, block in enumerate(record["blocks"][context]):
            for direction in ("left", "right"):
                for amount in amounts:
                    examples.append(example)
                    specs.append(PhaseSurgerySpec(
                        int(block[0]) + prefix,
                        int(block[1]) + prefix,
                        int(amount),
                        direction,
                    ))
                    metadata.append({
                        "example_id": record["example_id"],
                        "gene_id": record["gene_id"],
                        "chrom": record["chrom"],
                        "strand": record["strand"],
                        "donor_phase": record.get("donor_phase"),
                        "target_context": target_context,
                        "context": context,
                        "block_index": block_index,
                        "block": block,
                        "amount": amount,
                        "direction": direction,
                    })
    return examples, specs, metadata


def evaluate_variants(adapter, task, examples, specs, metadata, component, *, batch: int):
    values = []
    for start in range(0, len(examples), batch):
        values.extend(evaluate_batch(
            adapter,
            task,
            examples[start:start + batch],
            specs=specs[start:start + batch],
            component=component,
        ))
    return [{**meta, **value} for meta, value in zip(metadata, values)]


def primary_shard(adapter, task, record, example, component, package_sha, *, batch: int) -> dict:
    variants, specs, metadata = variant_grid(record, example, amounts=list(range(2, 11)))
    return {
        "schema": "gfmcd.p3_phase_surgery.confirmation_shard.v1",
        "arm": "primary",
        "package_sha256": package_sha,
        "unit_id": record["example_id"],
        "rows": evaluate_variants(
            adapter, task, variants, specs, metadata, component, batch=batch
        ),
    }


def control_shard(
    adapter, task, record, example, component, control_seeds, package_sha, *, batch: int
) -> dict:
    variants, specs, metadata = variant_grid(record, example, amounts=[2, 3, 4])
    rows = []
    components = [("p3", None, component)] + [
        (f"signed_permutation_{index:02d}", seed,
         signed_permutation_component(component, seed=seed))
        for index, seed in enumerate(control_seeds)
    ]
    geometry = []
    for label, seed, local_component in components:
        geometry.append({
            "component": label,
            "seed": seed,
            "l2_ratio_to_p3": float(local_component.norm() / component.norm()),
        })
        rows.extend({**row, "component": label, "control_seed": seed} for row in
                    evaluate_variants(
                        adapter, task, variants, specs, metadata, local_component, batch=batch
                    ))
    return {
        "schema": "gfmcd.p3_phase_surgery.confirmation_shard.v1",
        "arm": "direction_control",
        "package_sha256": package_sha,
        "unit_id": record["example_id"],
        "geometry": geometry,
        "rows": rows,
    }


def context_shard(
    adapter, task, pair, records, examples, component, package_sha, *, batch: int
) -> dict:
    rows = []
    for record, example in zip(records, examples):
        variants, specs, metadata = variant_grid(record, example, amounts=list(range(2, 11)))
        rows.extend(evaluate_variants(
            adapter, task, variants, specs, metadata, component, batch=batch
        ))
    return {
        "schema": "gfmcd.p3_phase_surgery.confirmation_shard.v1",
        "arm": "matched_context",
        "package_sha256": package_sha,
        "unit_id": pair["pair_id"],
        "pair": pair,
        "rows": rows,
    }


def write_arm_manifest(output_dir: Path, arm: str, expected: int, package_sha: str) -> None:
    shard_dir = output_dir / f"{arm}_shards"
    shards = sorted(shard_dir.glob("*.json"))
    if len(shards) != expected:
        raise RuntimeError(f"{arm} incomplete: {len(shards)}/{expected}")
    atomic_json(output_dir / f"{arm}_manifest.json", {
        "schema": "gfmcd.p3_phase_surgery.confirmation_arm_manifest.v1",
        "status": "COMPLETE",
        "arm": arm,
        "package_sha256": package_sha,
        "expected_shards": expected,
        "shards": {path.name: file_record(path) for path in shards},
    })


def run_target(package: dict, package_sha: str, output_dir: Path, *, device, batch: int) -> None:
    gtf = package["data"]["basic_gtf"]["path"]
    fasta = package["data"]["fasta"]["path"]
    model_bin = next(
        Path(row["path"]) for row in package["data"]["model_files"]
        if Path(row["path"]).name == "pytorch_model.bin"
    )
    source = {example.example_id: example for example in _dataset(gtf, fasta)}
    adapter = load_adapter(str(model_bin.parent), pick_device(device), eager=True)
    task = CouplingTask(adapter, batch=batch)
    component = fourier_band_component(
        adapter.model.bert.embeddings.position_embeddings.weight.detach().cpu(),
        tuple(package["design"]["p3_band_1026"]),
    )

    primary = package["cohorts"]["primary"]
    primary_examples = reconstruct_records(primary, source)
    primary_dir = output_dir / "primary_shards"
    for index, (record, example) in enumerate(zip(primary, primary_examples), 1):
        path = primary_dir / f"{stable_id(record['example_id'])}.json"
        if not path.exists():
            atomic_json(path, primary_shard(
                adapter, task, record, example, component, package_sha, batch=batch
            ))
        if index % 25 == 0 or index == len(primary):
            print(f"primary {index}/{len(primary)}", flush=True)
    write_arm_manifest(output_dir, "primary", len(primary), package_sha)

    control_ids = set(package["cohorts"]["direction_control_example_ids"])
    control_records = [row for row in primary if row["example_id"] in control_ids]
    control_examples = reconstruct_records(control_records, source)
    control_dir = output_dir / "direction_control_shards"
    for index, (record, example) in enumerate(zip(control_records, control_examples), 1):
        path = control_dir / f"{stable_id(record['example_id'])}.json"
        if not path.exists():
            atomic_json(path, control_shard(
                adapter,
                task,
                record,
                example,
                component,
                package["design"]["control_seeds"],
                package_sha,
                batch=batch,
            ))
        if index % 5 == 0 or index == len(control_records):
            print(f"direction-control {index}/{len(control_records)}", flush=True)
    write_arm_manifest(output_dir, "direction_control", len(control_records), package_sha)

    cds_by_id = {row["example_id"]: row for row in package["cohorts"]["context_cds"]}
    non_by_id = {row["example_id"]: row for row in package["cohorts"]["context_non_cds"]}
    context_dir = output_dir / "matched_context_shards"
    pairs = package["cohorts"]["context_pairs"]
    for index, pair in enumerate(pairs, 1):
        path = context_dir / f"{stable_id(pair['pair_id'])}.json"
        if not path.exists():
            records = [cds_by_id[pair["cds_example_id"]], non_by_id[pair["non_cds_example_id"]]]
            examples = reconstruct_records(records, source)
            atomic_json(path, context_shard(
                adapter, task, pair, records, examples, component, package_sha, batch=batch
            ))
        if index % 5 == 0 or index == len(pairs):
            print(f"matched-context {index}/{len(pairs)}", flush=True)
    write_arm_manifest(output_dir, "matched_context", len(pairs), package_sha)

    atomic_json(output_dir / "run_complete.json", {
        "schema": "gfmcd.p3_phase_surgery.confirmation_run.v1",
        "status": "TARGET_COMPLETE_UNANALYZED",
        "package_sha256": package_sha,
        "counts": {
            "primary": len(primary),
            "direction_control": len(control_records),
            "matched_context": len(pairs),
        },
        "outcome_summary_written": False,
    })


def run_development_smoke(
    package: dict, package_sha: str, output_dir: Path, *, device, batch: int
) -> None:
    cds = json.loads(Path(package["source_artifacts"]["development_cds_cohort"]["path"]).read_text())
    non = json.loads(
        Path(package["source_artifacts"]["development_non_cds_cohort"]["path"]).read_text()
    )
    gtf = package["data"]["basic_gtf"]["path"]
    fasta = package["data"]["fasta"]["path"]
    model_bin = next(
        Path(row["path"]) for row in package["data"]["model_files"]
        if Path(row["path"]).name == "pytorch_model.bin"
    )
    source = {example.example_id: example for example in _dataset(gtf, fasta)}
    adapter = load_adapter(str(model_bin.parent), pick_device(device), eager=True)
    task = CouplingTask(adapter, batch=batch)
    component = fourier_band_component(
        adapter.model.bert.embeddings.position_embeddings.weight.detach().cpu(),
        tuple(package["design"]["p3_band_1026"]),
    )
    cds_records = [
        {**row, "target_context": "cds"} for row in cds["records"] if row.get("eligible")
    ][:2]
    non_records = [
        {**row, "target_context": "non_cds_exon"}
        for row in non["records"] if row.get("eligible")
    ][:1]
    cds_examples = reconstruct_records(cds_records, source)
    non_examples = reconstruct_records(non_records, source)
    smoke = {
        "schema": "gfmcd.p3_phase_surgery.confirmation_smoke.v1",
        "status": "DEVELOPMENT_SMOKE_COMPLETE",
        "package_sha256": package_sha,
        "primary": [primary_shard(
            adapter, task, row, example, component, package_sha, batch=batch
        ) for row, example in zip(cds_records, cds_examples)],
        "direction_control": control_shard(
            adapter,
            task,
            cds_records[0],
            cds_examples[0],
            component,
            package["design"]["control_seeds"],
            package_sha,
            batch=batch,
        ),
        "non_cds": primary_shard(
            adapter, task, non_records[0], non_examples[0], component, package_sha, batch=batch
        ),
        "target_outcomes_read": False,
    }
    atomic_json(output_dir / "development_smoke.json", smoke)
    print("development smoke complete; no untouched outcome read", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("development-smoke", "target"), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("--batch must be positive")
    require_clean = args.phase == "target"
    package = load_contract(args.package, args.package_audit, require_clean=require_clean)
    package_sha = sha256_file(args.package)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.phase == "development-smoke":
        run_development_smoke(
            package, package_sha, output_dir, device=args.device, batch=args.batch
        )
    else:
        run_target(package, package_sha, output_dir, device=args.device, batch=args.batch)
    print(f"phase={args.phase} elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
