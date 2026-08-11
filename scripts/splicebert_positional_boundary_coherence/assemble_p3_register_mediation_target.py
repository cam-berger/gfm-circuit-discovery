#!/usr/bin/env python
"""Assemble complete, disjoint mediation shards under the original reserve seal."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.pbc_register_mediation import (  # noqa: E402
    LEARNED_STATES,
    MEDIATION_TARGET_SCHEMA,
    PATH_NAMES,
    compact_mediation_rows,
    validate_compact_mediation,
)
from gfmcd.splicing.pbc_register_mediation_runner import MEDIATION_SHARD_SCHEMA  # noqa: E402
from gfmcd.splicing.pbc_register_stage import (  # noqa: E402
    RESERVE_BLOCKS,
    RESERVE_CELLS,
    verify_register_target_seal,
)


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    return output, manifest


def assemble_payload(
    seal: Mapping[str, Any],
    shards: Iterable[Mapping[str, Any]],
    *,
    shard_records: Sequence[Mapping[str, Any]],
    expected_blocks: int = RESERVE_BLOCKS,
) -> dict[str, Any]:
    verify_register_target_seal(seal)
    expected_count = int(seal["mediation"]["shard_count"])
    if expected_count < 1 or len(shard_records) != expected_count:
        raise RuntimeError("mediation assembler requires every frozen multi-shard result")
    expected_rows_per_block = (len(LEARNED_STATES) + 31 * len(PATH_NAMES)) * 3 * 9
    seen_indices: set[int] = set()
    block_ids: set[str] = set()
    endpoint_rows: list[Mapping[str, Any]] = []
    cell_ledger: list[Mapping[str, Any]] = []
    ledger_static: dict[str, dict[str, Any]] = {}
    ledger_events: dict[str, dict[str, dict[str, Any]]] = {}
    ledger_source_rows: dict[str, int] = {}
    stream = sha256()
    replay_intact_max = 0.0
    replay_corrupt_max = 0.0
    replay_coupling_max = 0.0
    hook_events = 0
    reference: dict[str, Any] | None = None
    observed_shards = 0
    for shard in shards:
        observed_shards += 1
        if reference is None:
            reference = {
                "authorization": dict(shard.get("authorization", {})),
                "design": dict(shard.get("design", {})),
                "sources": dict(shard.get("sources", {})),
                "model": dict(shard.get("model", {})),
            }
        if (
            shard.get("schema") != MEDIATION_SHARD_SCHEMA
            or shard.get("status") != "COMPLETE"
            or shard.get("stage") != "mediation"
            or shard.get("split_role") != "phase3_reserve"
            or shard.get("seal_sha256") != seal["seal_sha256"]
            or shard.get("mode") != seal["mode"]
            or shard.get("authorization") != reference["authorization"]
            or shard.get("design") != reference["design"]
            or shard.get("sources") != reference["sources"]
        ):
            raise RuntimeError("mediation shard drifted from the frozen design or sources")
        local = shard.get("shard")
        if not isinstance(local, Mapping) or int(local.get("count", -1)) != expected_count:
            raise RuntimeError("mediation shard identity is malformed")
        index = int(local.get("index", -1))
        if not 0 <= index < expected_count or index in seen_indices:
            raise RuntimeError("mediation shard index is missing or duplicated")
        local_blocks = {str(value) for value in local.get("block_ids", [])}
        if not local_blocks or local_blocks & block_ids:
            raise RuntimeError("mediation shard block supports overlap or are empty")
        local_rows = shard.get("rows")
        if not isinstance(local_rows, list) or len(local_rows) != len(local_blocks) * expected_rows_per_block:
            raise RuntimeError("mediation shard row count is incomplete")
        if {str(row.get("block_id")) for row in local_rows} != local_blocks:
            raise RuntimeError("mediation shard rows leave their declared block support")
        mechanics = shard.get("mechanics")
        if (
            not isinstance(mechanics, Mapping)
            or mechanics.get("passed") is not True
            or mechanics.get("same_sequence_patching") is not True
            or mechanics.get("same_corruption_state_patching") is not True
            or mechanics.get("full_token_support") != [393, 513]
            or int(mechanics.get("rows_per_block", -1)) != expected_rows_per_block
        ):
            raise RuntimeError("mediation shard mechanics did not pass")
        stream.update(bytes.fromhex(str(mechanics["actual_hook_tensor_stream_sha256"])))
        replay_intact_max = max(
            replay_intact_max,
            float(mechanics["raw_relocation_replay_intact_margin_max_abs"]),
        )
        replay_corrupt_max = max(
            replay_corrupt_max,
            float(mechanics["raw_relocation_replay_corrupt_margin_max_abs"]),
        )
        replay_coupling_max = max(
            replay_coupling_max,
            float(mechanics["raw_relocation_replay_coupling_max_abs"]),
        )
        hook_events += int(mechanics["actual_patch_forward_events"])
        compact = compact_mediation_rows(
            local_rows,
            freeze=reference["design"]["candidate_freeze"],
        )
        endpoint_rows.extend(compact["endpoint_rows"])
        cell_ledger.extend(compact["cell_ledger"])
        for condition, value in compact["provenance_ledger"].items():
            static = {
                key: item
                for key, item in value.items()
                if key not in {"source_rows", "events", "tensor_stream_sha256"}
            }
            if condition in ledger_static and ledger_static[condition] != static:
                raise RuntimeError("mediation shard provenance condition drifted")
            ledger_static[condition] = static
            ledger_source_rows[condition] = ledger_source_rows.get(condition, 0) + int(
                value["source_rows"]
            )
            local_events = ledger_events.setdefault(condition, {})
            for event in value["events"]:
                event_static = {
                    key: item for key, item in event.items() if key != "source_rows"
                }
                encoded = json.dumps(event_static, sort_keys=True, separators=(",", ":"))
                if encoded not in local_events:
                    local_events[encoded] = {**event_static, "source_rows": 0}
                local_events[encoded]["source_rows"] += int(event["source_rows"])
        seen_indices.add(index)
        block_ids.update(local_blocks)
    if observed_shards != expected_count or reference is None:
        raise RuntimeError("mediation shard family is incomplete")
    if seen_indices != set(range(expected_count)):
        raise RuntimeError("mediation shard index family is incomplete")
    if len(block_ids) != int(expected_blocks):
        raise RuntimeError("mediation shards do not close the 384-block reserve")
    expected_endpoint_rows = int(expected_blocks) * (
        len(LEARNED_STATES) + 31 * len(PATH_NAMES)
    ) * 3 * 3
    if len(endpoint_rows) != expected_endpoint_rows:
        raise RuntimeError("assembled compact mediation lattice has the wrong cardinality")
    provenance_ledger = {}
    for condition in sorted(ledger_static):
        events = [ledger_events[condition][key] for key in sorted(ledger_events[condition])]
        provenance_ledger[condition] = {
            **ledger_static[condition],
            "source_rows": ledger_source_rows[condition],
            "events": events,
            "tensor_stream_sha256": sha256(
                json.dumps(events, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    normalization = {
        "primitive_rows": int(expected_blocks) * expected_rows_per_block,
        "compact_endpoint_rows": len(endpoint_rows),
        "phase_pairs_per_endpoint": 3,
        "endpoint_reduction": "arithmetic_mean_with_math_fsum",
        "sufficient_for": ["S_r", "omega3", "D_path", "R_path", "Q_path"],
        "provenance_ledger_sha256": sha256(
            json.dumps(
                provenance_ledger, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    result = {
        "schema": MEDIATION_TARGET_SCHEMA,
        "status": "COMPLETE",
        "stage": "mediation",
        "split_role": "phase3_reserve",
        "mode": seal["mode"],
        "seal_sha256": seal["seal_sha256"],
        "authorization": dict(reference["authorization"]),
        "design": dict(reference["design"]),
        "sources": dict(reference["sources"]),
        "shard_sources": [dict(record) for record in shard_records],
        "model": dict(reference["model"]),
        "reserve_blocks": RESERVE_BLOCKS,
        "reserve_cells": RESERVE_CELLS,
        "n_blocks": len(block_ids),
        "n_cells": RESERVE_CELLS,
        "mechanics": {
            "actual_hook_tensor_stream_sha256": stream.hexdigest(),
            "actual_patch_forward_events": hook_events,
            "same_sequence_patching": True,
            "same_corruption_state_patching": True,
            "full_token_support": [393, 513],
            "raw_relocation_replay_intact_margin_max_abs": replay_intact_max,
            "raw_relocation_replay_corrupt_margin_max_abs": replay_corrupt_max,
            "raw_relocation_replay_coupling_max_abs": replay_coupling_max,
            "normalized_provenance_ledger_sha256": normalization[
                "provenance_ledger_sha256"
            ],
            "p3_attenuation_order": seal["mediation"]["p3_attenuation_order"],
            "rows_per_block": expected_rows_per_block,
            "passed": max(
                replay_intact_max, replay_corrupt_max, replay_coupling_max
            )
            <= 1e-8,
        },
        "endpoint_rows": endpoint_rows,
        "cell_ledger": cell_ledger,
        "provenance_ledger": provenance_ledger,
        "normalization": normalization,
    }
    validate_compact_mediation(
        result,
        freeze=reference["design"]["candidate_freeze"],
        enforce_reserve_contract=expected_blocks == RESERVE_BLOCKS,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output, manifest_path = output_paths(args.out)
    seal_path = Path(args.seal).expanduser().resolve()
    seal = json.loads(seal_path.read_text())
    verify_register_target_seal(seal)
    if output != Path(str(seal["mediation"]["target_output_path"])).resolve():
        raise RuntimeError("assembled mediation path differs from the pre-reserve seal")
    shard_paths = [Path(value).resolve() for value in seal["mediation"]["shard_paths"]]
    records = [file_record(path) for path in shard_paths]
    result = assemble_payload(
        seal,
        (json.loads(path.read_text()) for path in shard_paths),
        shard_records=records,
    )
    if result["mechanics"]["passed"] is not True:
        raise RuntimeError("assembled mediation target failed raw replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-p3-register-mediation-assembly-v1",
        sys.argv,
        repo=REPO,
        status="TARGET_SCORED_UNANALYZED",
        config={
            "stage": "mediation",
            "n_blocks": result["n_blocks"],
            "n_cells": result["n_cells"],
            "shard_count": len(shard_paths),
            "rows_per_block": result["mechanics"]["rows_per_block"],
        },
    )
    manifest.add_input(seal_path)
    for path in shard_paths:
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_blocks": result["n_blocks"],
                "n_endpoint_rows": len(result["endpoint_rows"]),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
