#!/usr/bin/env python
"""Execute the sealed Phase-III P3 register lattice without interpreting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.circuit import load_adapter, pick_device  # noqa: E402
from gfmcd.splicing.pbc_register_relocation import relocation_mechanics  # noqa: E402
from gfmcd.splicing.pbc_register_runner import (  # noqa: E402
    evaluate_register_lattice,
    position_embedding_module,
    register_cell_from_mapping,
)
from gfmcd.splicing.pbc_register_stage import (  # noqa: E402
    REGISTER_TARGET_SCHEMA,
    RESERVE_BLOCKS,
    sibling_manifest_path,
    validate_sibling_manifest,
    verify_register_target_seal,
)


COHORT_SCHEMA = "gfmcd.positional_boundary_coherence.cohort.v1"
SHARD_SCHEMA = "gfmcd.pbc.p3_register_target_shard.v1"


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    return output, manifest


def reserve_cells(
    cohort: Mapping[str, Any],
    *,
    split_role: str = "phase3_reserve",
    expected_blocks: int = RESERVE_BLOCKS,
) -> tuple[Any, ...]:
    """Validate the sealed outcome-fresh complete lattice before checkpoint access."""

    if cohort.get("schema") != COHORT_SCHEMA:
        raise RuntimeError("wrong positional-boundary-coherence cohort schema")
    rows = tuple(
        row
        for row in cohort.get("cells", [])
        if isinstance(row, Mapping) and str(row.get("split_role")) == split_role
    )
    by_block: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_block.setdefault(str(row.get("block_id", "")), []).append(row)
    if len(by_block) != int(expected_blocks):
        raise RuntimeError(
            f"reserve needs exactly {expected_blocks} blocks; found {len(by_block)}"
        )
    expected = {(donor, acceptor) for donor in range(3) for acceptor in range(3)}
    for block_id, local in by_block.items():
        if not block_id or len(local) != 9:
            raise RuntimeError(f"reserve block {block_id!r} is not a nine-cell lattice")
        if {
            (int(row["donor_phase"]), int(row["acceptor_phase"])) for row in local
        } != expected:
            raise RuntimeError(f"reserve block {block_id!r} lacks the complete 3x3 lattice")
    cells = tuple(register_cell_from_mapping(row) for row in rows)
    if any(
        len(cell.intact_sequence) != 1022
        or cell.donor_position != 392
        or cell.acceptor_position != 512
        for cell in cells
    ):
        raise RuntimeError("reserve cell drifted from the frozen 1022-nt/120-nt geometry")
    return cells


def deterministic_shard(
    cells: Sequence[Any], *, shard_index: int, shard_count: int
) -> tuple[Any, ...]:
    """Assign complete sorted blocks to one deterministic modulo shard."""

    count = int(shard_count)
    index = int(shard_index)
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid target shard index/count")
    block_ids = sorted({str(cell.block_id) for cell in cells})
    selected = {
        block_id for offset, block_id in enumerate(block_ids) if offset % count == index
    }
    output = tuple(cell for cell in cells if str(cell.block_id) in selected)
    if not output:
        raise RuntimeError("target shard contains no complete block")
    return output


def execute_target(
    *,
    seal: Mapping[str, Any],
    cohort: Mapping[str, Any],
    adapter: Any,
    batch_size: int,
    shard_index: int = 0,
) -> dict[str, Any]:
    """Evaluate raw target and control rows under the already verified seal."""

    verify_register_target_seal(seal)
    all_cells = reserve_cells(
        cohort,
        split_role=str(seal["split_role"]),
        expected_blocks=int(seal["reserve_blocks"]),
    )
    shard_count = int(seal["shard_count"])
    cells = deterministic_shard(
        all_cells, shard_index=shard_index, shard_count=shard_count
    )
    design = seal["design"]
    joint = seal["mode"] == "JOINT"
    rows, through_model = evaluate_register_lattice(
        adapter,
        cells,
        p3_bins=tuple(int(value) for value in design["p3_bins"]),
        span_start_token=int(design["span_token"][0]),
        span_end_token=int(design["span_token"][1]),
        decoy_span_token=tuple(int(value) for value in design["decoy_span_token"]),
        control_seeds=tuple(int(value) for value in design["control_seeds"]),
        batch_size=int(batch_size),
        direction=str(design["direction"]),
        joint=joint,
        p10_bins=tuple(int(value) for value in design["p10_bins"]),
        p10_dose=(
            float(design["p10_attenuation_dose"])
            if joint
            else 0.0
        ),
    )
    table = (
        position_embedding_module(adapter.model)
        .weight.detach()
        .cpu()
        .double()
        .numpy()
    )
    construction = relocation_mechanics(
        table,
        tuple(int(value) for value in design["p3_bins"]),
        start=int(design["span_token"][0]),
        end=int(design["span_token"][1]),
        direction=str(design["direction"]),
    )
    if not construction["passed"]:
        raise RuntimeError("register construction invariant failed at target time")
    if float(through_model["register_zero_no_hook_max_abs"]) > 1e-8:
        raise RuntimeError("register-zero target replay failed")
    metadata = adapter.metadata()
    model_metadata = (
        dict(metadata)
        if isinstance(metadata, Mapping)
        else dict(getattr(metadata, "__dict__", {}))
    )
    return {
        "schema": REGISTER_TARGET_SCHEMA if shard_count == 1 else SHARD_SCHEMA,
        "status": "COMPLETE",
        "stage": "phase3_reserve",
        "mode": str(seal["mode"]),
        "seal_sha256": str(seal["seal_sha256"]),
        "authorization": dict(seal["authorization"]),
        "design": dict(design),
        "sources": {
            label: dict(record) for label, record in seal["bindings"].items()
        },
        "model": model_metadata,
        "n_blocks": len({cell.block_id for cell in cells}),
        "reserve_blocks": int(seal["reserve_blocks"]),
        "n_cells": len(cells),
        "shard": {
            "index": int(shard_index),
            "count": shard_count,
            "block_ids": sorted({cell.block_id for cell in cells}),
        },
        "mechanics": {
            **through_model,
            "construction_invariants": construction,
        },
        "rows": rows,
    }


def _checkpoint_path(model_path: Path) -> Path:
    if not model_path.is_dir():
        raise RuntimeError("--model must name the official SpliceBERT checkpoint directory")
    return model_path / "pytorch_model.bin"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output, manifest_path = output_paths(args.out)
    seal_path = Path(args.seal).expanduser().resolve()
    seal_manifest_path = sibling_manifest_path(seal_path)
    cohort_path = Path(args.cohort).expanduser().resolve()
    seal = json.loads(seal_path.read_text())
    seal_manifest = json.loads(seal_manifest_path.read_text())
    verify_register_target_seal(seal)
    if not validate_sibling_manifest(
        seal_manifest,
        manifest_path=seal_manifest_path,
        artifact_path=seal_path,
    )["passed"]:
        raise RuntimeError("reserve seal lacks its valid producer manifest")
    shard_count = int(seal["shard_count"])
    if not 0 <= args.shard_index < shard_count:
        raise RuntimeError("target shard index is outside the frozen range")
    frozen_output = Path(str(seal["shard_paths"][args.shard_index])).resolve()
    if output != frozen_output:
        raise RuntimeError("target shard output path differs from the frozen seal")
    bindings = seal["bindings"]
    if Path(str(bindings["cohort"]["path"])).resolve() != cohort_path:
        raise RuntimeError("target cohort path differs from the frozen seal")
    checkpoint = _checkpoint_path(Path(args.model).expanduser().resolve())
    if Path(str(bindings["checkpoint"]["path"])).resolve() != checkpoint:
        raise RuntimeError("target checkpoint path differs from the frozen seal")
    cohort = json.loads(cohort_path.read_text())
    # Validate the sealed reserve before the first checkpoint load.
    all_cells = reserve_cells(
        cohort,
        split_role=str(seal["split_role"]),
        expected_blocks=int(seal["reserve_blocks"]),
    )
    deterministic_shard(
        all_cells, shard_index=args.shard_index, shard_count=shard_count
    )
    adapter = load_adapter(
        str(Path(args.model).expanduser().resolve()),
        pick_device(args.device),
        eager=False,
    )
    result = execute_target(
        seal=seal,
        cohort=cohort,
        adapter=adapter,
        batch_size=args.batch,
        shard_index=args.shard_index,
    )
    if shard_count == 1:
        result["sources"]["reserve_seal"] = file_record(seal_path)
        result["sources"]["reserve_seal_manifest"] = file_record(
            seal_manifest_path
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-p3-register-target-v1",
        sys.argv,
        repo=REPO,
        status="TARGET_SCORED_UNANALYZED",
        config={
            "stage": "phase3_reserve",
            "mode": result["mode"],
            "n_blocks": result["n_blocks"],
            "batch_size": args.batch,
            "shard": result["shard"],
            "design": result["design"],
        },
        seeds={
            f"signed_control_{index:02d}": int(seed)
            for index, seed in enumerate(result["design"]["control_seeds"])
        },
    )
    for path in (seal_path, seal_manifest_path, cohort_path, checkpoint):
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": result["mode"],
                "n_blocks": result["n_blocks"],
                "shard": result["shard"],
                "n_rows": len(result["rows"]),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
