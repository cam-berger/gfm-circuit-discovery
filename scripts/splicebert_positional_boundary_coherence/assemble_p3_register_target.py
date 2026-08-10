#!/usr/bin/env python
"""Assemble frozen, disjoint Phase-III register shards into one analyzable target."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.pbc_register_stage import (  # noqa: E402
    REGISTER_TARGET_SCHEMA,
    RESERVE_BLOCKS,
    sibling_manifest_path,
    validate_sibling_manifest,
    verify_register_target_seal,
)
from run_p3_register_target import SHARD_SCHEMA  # noqa: E402


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    return output, manifest


def _mechanics_signature(mechanics: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(mechanics))
    value.pop("no_hook_rows", None)
    value.pop("register_zero_no_hook_max_abs", None)
    return value


def assemble_payload(
    seal: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
    *,
    shard_records: Sequence[Mapping[str, Any]],
    expected_blocks: int = RESERVE_BLOCKS,
    expected_component_count: int = 34,
) -> dict[str, Any]:
    """Verify shard topology and return the exact target-schema payload."""

    verify_register_target_seal(seal)
    count = int(seal["shard_count"])
    if count < 2 or len(shards) != count or len(shard_records) != count:
        raise RuntimeError("assembler requires every frozen multi-shard output")
    by_index: dict[int, Mapping[str, Any]] = {}
    all_blocks: set[str] = set()
    all_rows: list[Mapping[str, Any]] = []
    no_hook_rows: list[Mapping[str, Any]] = []
    replay_errors: list[float] = []
    reference = shards[0]
    reference_mechanics = _mechanics_signature(reference["mechanics"])
    states = 2 if seal["mode"] == "JOINT" else 1
    for shard in shards:
        if (
            shard.get("schema") != SHARD_SCHEMA
            or shard.get("status") != "COMPLETE"
            or shard.get("seal_sha256") != seal["seal_sha256"]
            or shard.get("mode") != seal["mode"]
            or shard.get("design") != seal["design"]
            or shard.get("authorization") != seal["authorization"]
            or shard.get("sources") != seal["bindings"]
        ):
            raise RuntimeError("target shard drifted from the frozen seal")
        local = shard.get("shard")
        if not isinstance(local, Mapping) or int(local.get("count", -1)) != count:
            raise RuntimeError("target shard identity is invalid")
        index = int(local.get("index", -1))
        if not 0 <= index < count or index in by_index:
            raise RuntimeError("target shard index is missing or duplicated")
        blocks = {str(value) for value in local.get("block_ids", [])}
        if not blocks or blocks & all_blocks:
            raise RuntimeError("target shard block supports overlap or are empty")
        if int(shard.get("n_blocks", -1)) != len(blocks):
            raise RuntimeError("target shard block count disagrees with its support")
        rows = shard.get("rows")
        if not isinstance(rows, list) or len(rows) != (
            len(blocks) * expected_component_count * 3 * 9 * states
        ):
            raise RuntimeError("target shard row count is incomplete")
        if {str(row.get("block_id")) for row in rows} != blocks:
            raise RuntimeError("target shard rows leave their declared block support")
        mechanics = shard.get("mechanics")
        if not isinstance(mechanics, Mapping):
            raise RuntimeError("target shard lacks mechanics")
        if _mechanics_signature(mechanics) != reference_mechanics:
            raise RuntimeError("target shard mechanics/design signatures differ")
        local_native = mechanics.get("no_hook_rows")
        if not isinstance(local_native, list) or len(local_native) != len(blocks) * 9:
            raise RuntimeError("target shard no-hook replay rows are incomplete")
        by_index[index] = shard
        all_blocks.update(blocks)
        all_rows.extend(rows)
        no_hook_rows.extend(local_native)
        replay_errors.append(float(mechanics["register_zero_no_hook_max_abs"]))
    if set(by_index) != set(range(count)) or len(all_blocks) != int(expected_blocks):
        raise RuntimeError("target shards do not close the frozen reserve block support")
    mechanics = copy.deepcopy(dict(reference["mechanics"]))
    mechanics["no_hook_rows"] = no_hook_rows
    mechanics["register_zero_no_hook_max_abs"] = max(replay_errors)
    return {
        "schema": REGISTER_TARGET_SCHEMA,
        "status": "COMPLETE",
        "stage": "phase3_reserve",
        "mode": str(seal["mode"]),
        "seal_sha256": str(seal["seal_sha256"]),
        "authorization": dict(seal["authorization"]),
        "design": dict(seal["design"]),
        "sources": {
            label: dict(record) for label, record in seal["bindings"].items()
        },
        "shard_sources": [dict(record) for record in shard_records],
        "model": dict(reference.get("model", {})),
        "n_blocks": len(all_blocks),
        "n_cells": len(all_blocks) * 9,
        "mechanics": mechanics,
        "rows": all_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output, manifest_path = output_paths(args.out)
    seal_path = Path(args.seal).expanduser().resolve()
    seal_manifest_path = sibling_manifest_path(seal_path)
    seal = json.loads(seal_path.read_text())
    seal_manifest = json.loads(seal_manifest_path.read_text())
    verify_register_target_seal(seal)
    if not validate_sibling_manifest(
        seal_manifest,
        manifest_path=seal_manifest_path,
        artifact_path=seal_path,
    )["passed"]:
        raise RuntimeError("reserve seal lacks its valid producer manifest")
    if output != Path(str(seal["output_path"])).resolve():
        raise RuntimeError("assembled target path differs from the frozen seal")
    shard_paths = [Path(value).resolve() for value in seal["shard_paths"]]
    shards = [json.loads(path.read_text()) for path in shard_paths]
    records = [file_record(path) for path in shard_paths]
    result = assemble_payload(seal, shards, shard_records=records)
    result["sources"]["reserve_seal"] = file_record(seal_path)
    result["sources"]["reserve_seal_manifest"] = file_record(
        seal_manifest_path
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-p3-register-target-assembly-v1",
        sys.argv,
        repo=REPO,
        status="TARGET_SCORED_UNANALYZED",
        config={
            "stage": "phase3_reserve",
            "mode": result["mode"],
            "n_blocks": result["n_blocks"],
            "shard_count": seal["shard_count"],
        },
    )
    manifest.add_input(seal_path)
    manifest.add_input(seal_manifest_path)
    for path in shard_paths:
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_blocks": result["n_blocks"],
                "n_rows": len(result["rows"]),
                "shards": len(shard_paths),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
