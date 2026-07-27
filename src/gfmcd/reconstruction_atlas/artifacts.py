"""Transactional, immutable result-plus-manifest writes for the atlas."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gfmcd.artifact_guard import guard_output_path
from gfmcd.provenance import file_record, git_commit, git_dirty


def manifest_path_for(result_path: str | Path) -> Path:
    result = Path(result_path)
    if not result.suffix:
        raise ValueError("atlas artifact path must have an extension")
    return result.with_name(f"{result.stem}.manifest.json")


def write_result_bundle(
    result_path: str | Path,
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
    status: str,
    model_identity: Mapping[str, Any],
    input_paths: Sequence[str | Path],
    parameters: Mapping[str, Any],
    seeds: Mapping[str, int],
    command: Sequence[str] | None = None,
    repo: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write a new result and manifest, refusing either pre-existing path."""

    result_text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    return write_bytes_bundle(
        result_path,
        result_text.encode(),
        experiment_id=experiment_id,
        status=status,
        model_identity=model_identity,
        input_paths=input_paths,
        parameters=parameters,
        seeds=seeds,
        command=command,
        repo=repo,
    )


def write_jsonl_bundle(
    result_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> tuple[Path, Path]:
    if Path(result_path).suffix != ".jsonl":
        raise ValueError("JSONL artifact path must end in .jsonl")
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    return write_bytes_bundle(result_path, payload.encode(), **kwargs)


def write_bytes_bundle(
    result_path: str | Path,
    result_bytes: bytes,
    *,
    experiment_id: str,
    status: str,
    model_identity: Mapping[str, Any],
    input_paths: Sequence[str | Path],
    parameters: Mapping[str, Any],
    seeds: Mapping[str, int],
    command: Sequence[str] | None = None,
    repo: str | Path | None = None,
) -> tuple[Path, Path]:
    result = guard_output_path(result_path)
    manifest = guard_output_path(manifest_path_for(result))
    if not experiment_id or status not in {
        "PREFLIGHT",
        "EXPLORATORY",
        "DEVELOPMENT",
        "SEALED",
        "NULL",
        "INSTRUMENT_UNRESOLVED",
    }:
        raise ValueError("invalid experiment identity or status")
    result_record = {
        "path": str(result),
        "size_bytes": len(result_bytes),
        "sha256": hashlib.sha256(result_bytes).hexdigest(),
    }
    data_records = [file_record(path) for path in input_paths]
    missing_inputs = [
        record["path"] for record in data_records if record.get("missing")
    ]
    if missing_inputs:
        raise FileNotFoundError(
            f"manifest input path(s) do not exist: {missing_inputs}"
        )
    manifest_payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command if command is not None else sys.argv),
        "checkpoint": dict(model_identity),
        "data": data_records,
        "parameters": dict(parameters),
        "seeds": {str(key): int(value) for key, value in seeds.items()},
        "code": {
            "git_commit": git_commit(repo),
            "git_dirty": git_dirty(repo),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "outputs": [result_record],
    }
    manifest_text = json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    result.parent.mkdir(parents=True, exist_ok=True)
    wrote_result = False
    wrote_manifest = False
    try:
        with result.open("xb") as handle:
            handle.write(result_bytes)
        wrote_result = True
        with manifest.open("x", encoding="utf-8") as handle:
            wrote_manifest = True
            handle.write(manifest_text)
    except Exception:
        if wrote_manifest:
            manifest.unlink(missing_ok=True)
        if wrote_result:
            result.unlink()
        raise
    return result, manifest
