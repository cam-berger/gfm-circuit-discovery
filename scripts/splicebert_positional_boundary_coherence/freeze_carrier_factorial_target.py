#!/usr/bin/env python
"""Freeze an audited Phase-II authorization into an exact target execution seal."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.pbc_carrier_factorial import (  # noqa: E402
    MIN_STRUCTURED_CONTROLS,
    build_phase_control_family,
    build_world_tables,
    make_target_seal,
)
from gfmcd.splicing.pbc_frozen_inputs import (  # noqa: E402
    validate_candidate_input_pins,
    validate_frozen_construction_configs,
    validate_official_checkpoint,
)
from audit_carrier_factorial_preflight import AUDIT_SCHEMA  # noqa: E402
from audit_protocol_code_conformance import (  # noqa: E402
    CONFORMANCE_EXPERIMENT_ID,
    CONFORMANCE_SCHEMA,
    conformance_code_files,
)
from run_carrier_factorial_preflight import (  # noqa: E402
    EXPECTED_CONSTRUCTION_AUDIT_EXPERIMENT_ID,
    EXPECTED_CONSTRUCTION_AUDIT_SCHEMA,
    EXPECTED_CONSTRUCTION_AUDIT_STATUS,
    PREFLIGHT_SCHEMA,
    TABLE_HEALTH_SCHEMA,
)
from run_carrier_factorial_target import SpliceBertCellScorer  # noqa: E402


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    return output, manifest


def target_output_plan(path: str | Path, shard_count: int) -> dict[str, Any]:
    assembled = guard_output_path(Path(path).expanduser().resolve())
    if shard_count < 1:
        raise ValueError("target shard count must be positive")
    if shard_count == 1:
        shards = [assembled]
    else:
        suffix = assembled.suffix or ".json"
        stem = assembled.name[: -len(suffix)] if assembled.suffix else assembled.name
        shards = [
            guard_output_path(
                assembled.with_name(
                    f"{stem}.shard_{index:03d}_of_{shard_count:03d}{suffix}"
                )
            )
            for index in range(shard_count)
        ]
    if len(set(shards)) != shard_count:
        raise RuntimeError("target shard output plan is not unique")
    return {
        "shard_count": int(shard_count),
        "shards": [str(path) for path in shards],
        "assembled": str(assembled),
    }


def _file_record_valid(record: object) -> bool:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        return False
    path = Path(str(record["path"])).expanduser().resolve()
    return bool(
        path.is_file()
        and type(record.get("size_bytes")) is int
        and path.stat().st_size == int(record["size_bytes"])
        and str(record.get("sha256", "")) == sha256_file(path)
    )


def _manifest_binds_output(manifest: Mapping[str, Any], path: Path) -> bool:
    outputs = manifest.get("outputs")
    return bool(
        isinstance(outputs, list)
        and any(
            _file_record_valid(record)
            and Path(str(record["path"])).expanduser().resolve() == path.resolve()
            for record in outputs
            if isinstance(record, Mapping)
        )
    )


def _manifest_binds_input(manifest: Mapping[str, Any], path: Path) -> bool:
    inputs = manifest.get("inputs")
    digest = sha256_file(path)
    return bool(
        isinstance(inputs, list)
        and any(
            _file_record_valid(record)
            and Path(str(record["path"])).expanduser().resolve() == path.resolve()
            and str(record.get("sha256")) == digest
            for record in inputs
            if isinstance(record, Mapping)
        )
    )


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_conformance_chain(
    audit: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    audit_path: Path,
    required_inputs: tuple[Path, ...],
    frozen_revision: str,
) -> tuple[Path, ...]:
    """Validate the independent protocol audit and return its pinned inputs."""

    body = dict(audit)
    observed_content_hash = str(body.pop("content_sha256", ""))
    coverage = audit.get("coverage_matrix")
    config = manifest.get("config")
    audited_records = audit.get("audited_inputs")
    manifest_records = manifest.get("inputs")
    if (
        audit.get("schema") != CONFORMANCE_SCHEMA
        or audit.get("stage") != "development"
        or audit.get("status") != "PASS"
        or audit.get("passed") is not True
        or audit.get("target_outcomes_read") is not False
        or audit.get("model_loaded") is not False
        or audit.get("tracked_tree_clean") is not True
        or audit.get("git_revision") != frozen_revision
        or audit.get("failures") != []
        or not isinstance(coverage, list)
        or not coverage
        or not all(
            isinstance(row, Mapping) and row.get("passed") is True for row in coverage
        )
        or observed_content_hash != _canonical_json_sha256(body)
        or manifest.get("experiment_id") != CONFORMANCE_EXPERIMENT_ID
        or manifest.get("status") != "PASS"
        or manifest.get("git_commit") != frozen_revision
        or not isinstance(config, Mapping)
        or config.get("stage") != "development"
        or config.get("target_outcomes_read") is not False
        or config.get("git_revision") != frozen_revision
        or not _manifest_binds_output(manifest, audit_path)
        or not isinstance(audited_records, list)
        or not audited_records
        or not isinstance(manifest_records, list)
        or not manifest_records
        or not all(_file_record_valid(record) for record in audited_records)
        or not all(_file_record_valid(record) for record in manifest_records)
    ):
        raise RuntimeError("protocol-to-code conformance producer chain is invalid")

    def identities(records: list[object]) -> set[tuple[str, int, str]]:
        return {
            (
                str(Path(str(record["path"])).expanduser().resolve()),
                int(record["size_bytes"]),
                str(record["sha256"]),
            )
            for record in records
            if isinstance(record, Mapping)
        }

    if identities(audited_records) != identities(manifest_records):
        raise RuntimeError("conformance artifact and manifest input inventories differ")
    if not all(_manifest_binds_input(manifest, path) for path in required_inputs):
        raise RuntimeError("conformance audit did not bind every final code/input file")
    return tuple(
        Path(str(record["path"])).expanduser().resolve()
        for record in manifest_records
        if isinstance(record, Mapping)
    )


def clean_tracked_revision(repo: Path = REPO) -> str:
    """Return HEAD only when tracked files exactly match the index and commit."""

    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    )
    if status.strip():
        raise RuntimeError("target freeze requires a clean tracked worktree")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if len(revision) != 40:
        raise RuntimeError("target freeze could not resolve a full Git revision")
    return revision


def _require_tracked(path: Path) -> None:
    try:
        relative = path.resolve().relative_to(REPO.resolve())
    except ValueError as exc:
        raise RuntimeError(f"freeze code lies outside the repository: {path}") from exc
    try:
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=REPO,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"freeze code is not tracked: {path}") from exc


def _model_files(model: Path) -> tuple[Path, ...]:
    names = (
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    )
    files = tuple(model / name for name in names if (model / name).is_file())
    if not any(path.name in {"pytorch_model.bin", "model.safetensors"} for path in files):
        raise RuntimeError("official checkpoint weight file is absent")
    if not any(path.name in {"tokenizer.json", "vocab.txt"} for path in files):
        raise RuntimeError("official checkpoint tokenizer inventory is absent")
    return files


def _table_hash(table: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(table).tobytes()).hexdigest()


def freeze_payload(
    *,
    preflight: Mapping[str, Any],
    audit: Mapping[str, Any],
    original: np.ndarray,
    pinned_files: Mapping[str, Mapping[str, Any]],
    preflight_authorization: Mapping[str, Any],
    runtime: Mapping[str, Any],
    target_outputs: Mapping[str, Any],
    frozen_revision: str,
) -> dict[str, Any]:
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("status")
        not in {"P3_ONLY_TARGET_AUTHORIZED", "FULL_FACTORIAL_TARGET_AUTHORIZED"}
        or preflight.get("evaluation", {}).get("passed") is not True
    ):
        raise RuntimeError("preflight did not authorize a target")
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("target_mode") != preflight.get("target_mode")
        or audit.get("target_outcomes_read") is not False
    ):
        raise RuntimeError("independent preflight audit did not authorize this mode")
    evaluation = preflight["evaluation"]
    p3_dose = float(evaluation["p3"]["selected_dose"])
    p10_dose = (
        float(evaluation["p10"]["selected_dose"])
        if evaluation["target_mode"] == "FULL_FACTORIAL"
        else None
    )
    controls, specs = build_phase_control_family(
        original,
        p3_dose=p3_dose,
        p10_dose=p10_dose,
        n_controls=MIN_STRUCTURED_CONTROLS,
        seed=int(preflight["control_seed"]),
    )
    worlds = build_world_tables(original, p3_dose=p3_dose, p10_dose=p10_dose)
    table_health = preflight.get("measurements", {}).get("authorized_table_health", {})
    health_rows = table_health.get("tables") if isinstance(table_health, Mapping) else None
    if (
        table_health.get("schema") != TABLE_HEALTH_SCHEMA
        or table_health.get("target_mode") != evaluation["target_mode"]
        or table_health.get("all_tables_pass") is not True
        or not isinstance(health_rows, Mapping)
        or set(health_rows) != {*worlds, *controls}
    ):
        raise RuntimeError("preflight table-health ledger does not cover exact frozen tables")
    exact_tables = {**worlds, **controls}
    for label, table in exact_tables.items():
        if (
            str(health_rows[label].get("table_sha256")) != _table_hash(table)
            or health_rows[label].get("pass") is not True
        ):
            raise RuntimeError(f"audited table-health hash mismatch for {label}")
    control_hashes = {
        label: {
            "family": str(specs[label]["family"]),
            "control_id": int(specs[label]["control_id"]),
            "world": str(specs[label]["world"]),
            "shape": list(table.shape),
            "dtype": str(table.dtype),
            "sha256": _table_hash(table),
        }
        for label, table in controls.items()
    }
    return make_target_seal(
        evaluation,
        pinned_files=pinned_files,
        control_seed=int(preflight["control_seed"]),
        structured_control_hashes=control_hashes,
        preflight_authorization=preflight_authorization,
        runtime=runtime,
        target_outputs=target_outputs,
        frozen_revision=frozen_revision,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--preflight-manifest", required=True)
    parser.add_argument("--preflight-audit", required=True)
    parser.add_argument("--preflight-audit-manifest", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--candidate-package", required=True)
    parser.add_argument("--construction-audit", required=True)
    parser.add_argument("--construction-audit-manifest", required=True)
    parser.add_argument("--conformance-audit", required=True)
    parser.add_argument("--conformance-audit-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-output", required=True)
    parser.add_argument("--target-shards", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in (
            ("preflight", args.preflight),
            ("preflight_manifest", args.preflight_manifest),
            ("preflight_audit", args.preflight_audit),
            ("preflight_audit_manifest", args.preflight_audit_manifest),
            ("cohort", args.cohort),
            ("candidate_package", args.candidate_package),
            ("construction_audit", args.construction_audit),
            ("construction_audit_manifest", args.construction_audit_manifest),
            ("conformance_audit", args.conformance_audit),
            ("conformance_audit_manifest", args.conformance_audit_manifest),
            ("protocol", args.protocol),
            (
                "source_spec",
                REPO
                / "configs/splicebert_positional_boundary_coherence/"
                "freshness_sources_v1.json",
            ),
            (
                "frozen_variance",
                REPO
                / "configs/splicebert_positional_boundary_coherence/"
                "c43_upper_variance_v1.json",
            ),
        )
    }
    model_path = Path(args.model).expanduser().resolve()
    validate_official_checkpoint(model_path)
    preflight = json.loads(paths["preflight"].read_text())
    preflight_manifest = json.loads(paths["preflight_manifest"].read_text())
    audit = json.loads(paths["preflight_audit"].read_text())
    audit_manifest = json.loads(paths["preflight_audit_manifest"].read_text())
    construction_audit = json.loads(paths["construction_audit"].read_text())
    construction_audit_manifest = json.loads(
        paths["construction_audit_manifest"].read_text()
    )
    conformance_audit = json.loads(paths["conformance_audit"].read_text())
    conformance_audit_manifest = json.loads(
        paths["conformance_audit_manifest"].read_text()
    )
    candidate_package = json.loads(paths["candidate_package"].read_text())
    biology_paths = validate_candidate_input_pins(candidate_package)
    construction_config_paths = validate_frozen_construction_configs(
        {
            "source_spec": paths["source_spec"],
            "frozen_variance": paths["frozen_variance"],
        }
    )
    construction_inputs = construction_audit_manifest.get("inputs")
    construction_outputs = construction_audit_manifest.get("outputs")
    construction_config = construction_audit_manifest.get("config")
    if (
        construction_audit.get("schema") != EXPECTED_CONSTRUCTION_AUDIT_SCHEMA
        or construction_audit.get("stage") != "construction"
        or construction_audit.get("status") != EXPECTED_CONSTRUCTION_AUDIT_STATUS
        or construction_audit.get("passed") is not True
        or construction_audit.get("target_execution_authorized") is not True
        or construction_audit_manifest.get("experiment_id")
        != EXPECTED_CONSTRUCTION_AUDIT_EXPERIMENT_ID
        or construction_audit_manifest.get("status")
        != EXPECTED_CONSTRUCTION_AUDIT_STATUS
        or not isinstance(construction_config, Mapping)
        or construction_config.get("stage") != "construction"
        or not isinstance(construction_outputs, list)
        or len(construction_outputs) != 1
        or not _manifest_binds_output(
            construction_audit_manifest, paths["construction_audit"]
        )
        or not isinstance(construction_inputs, list)
        or not construction_inputs
        or not all(_file_record_valid(record) for record in construction_inputs)
    ):
        raise RuntimeError("construction audit producer chain is invalid")
    construction_input_paths = tuple(
        Path(str(record["path"])).expanduser().resolve()
        for record in construction_inputs
    )
    if not _manifest_binds_output(preflight_manifest, paths["preflight"]):
        raise RuntimeError("preflight manifest does not bind the raw artifact")
    if not _manifest_binds_output(audit_manifest, paths["preflight_audit"]):
        raise RuntimeError("audit manifest does not bind the audit artifact")
    if not all(
        _manifest_binds_input(preflight_manifest, paths[name])
        for name in (
            "cohort",
            "candidate_package",
            "construction_audit",
            "construction_audit_manifest",
            "protocol",
        )
    ):
        raise RuntimeError("preflight manifest does not bind every supplied scientific input")
    preflight_input_records = preflight_manifest.get("inputs")
    if not isinstance(preflight_input_records, list) or not all(
        _file_record_valid(record) for record in preflight_input_records
    ):
        raise RuntimeError("preflight manifest contains a stale or malformed input")
    preflight_input_paths = tuple(
        Path(str(record["path"])).expanduser().resolve()
        for record in preflight_input_records
    )
    if not (
        audit.get("preflight_sha256") == sha256_file(paths["preflight"])
        and audit.get("raw_sha256") == preflight.get("raw_sha256")
        and audit.get("analysis_sha256") == preflight.get("analysis_sha256")
        and audit.get("preflight_manifest_sha256")
        == sha256_file(paths["preflight_manifest"])
    ):
        raise RuntimeError("preflight audit hash chain is invalid")
    construction_provenance = preflight.get("construction_audit_provenance")
    if not (
        isinstance(construction_provenance, Mapping)
        and construction_provenance.get("experiment_id")
        == EXPECTED_CONSTRUCTION_AUDIT_EXPERIMENT_ID
        and construction_provenance.get("artifact_sha256")
        == sha256_file(paths["construction_audit"])
        and construction_provenance.get("manifest_sha256")
        == sha256_file(paths["construction_audit_manifest"])
    ):
        raise RuntimeError("preflight does not bind the construction audit producer chain")
    code_files = conformance_code_files(REPO)
    revision = clean_tracked_revision()
    for path in code_files:
        _require_tracked(path)
        if not _manifest_binds_input(preflight_manifest, path):
            raise RuntimeError(f"preflight manifest did not bind final code: {path}")
    model_files = _model_files(model_path)
    if not all(_manifest_binds_input(preflight_manifest, path) for path in model_files):
        raise RuntimeError("preflight manifest did not bind every checkpoint file")
    conformance_input_paths = validate_conformance_chain(
        conformance_audit,
        conformance_audit_manifest,
        audit_path=paths["conformance_audit"],
        required_inputs=tuple(
            dict.fromkeys(
                (
                    paths["protocol"],
                    paths["candidate_package"],
                    *construction_config_paths.values(),
                    *biology_paths.values(),
                    *code_files,
                    *model_files,
                )
            )
        ),
        frozen_revision=revision,
    )
    pinned_paths = tuple(
        dict.fromkeys(
            (
                *paths.values(),
                *preflight_input_paths,
                *construction_input_paths,
                *conformance_input_paths,
                *biology_paths.values(),
                *construction_config_paths.values(),
                *code_files,
                *model_files,
            )
        )
    )
    pinned = {
        f"input_{index:02d}_{path.name}": file_record(path)
        for index, path in enumerate(pinned_paths)
    }
    authorization = {
        "audit_schema": AUDIT_SCHEMA,
        "audit_status": "PASS",
        "target_mode": str(audit["target_mode"]),
        "preflight_sha256": sha256_file(paths["preflight"]),
        "raw_sha256": str(audit["raw_sha256"]),
        "analysis_sha256": str(audit["analysis_sha256"]),
        "preflight_manifest_sha256": sha256_file(paths["preflight_manifest"]),
        "audit_sha256": sha256_file(paths["preflight_audit"]),
        "audit_manifest_sha256": sha256_file(paths["preflight_audit_manifest"]),
        "construction_audit_sha256": sha256_file(paths["construction_audit"]),
        "construction_audit_manifest_sha256": sha256_file(
            paths["construction_audit_manifest"]
        ),
        "conformance_audit_sha256": sha256_file(paths["conformance_audit"]),
        "conformance_audit_manifest_sha256": sha256_file(
            paths["conformance_audit_manifest"]
        ),
    }
    plan = target_output_plan(args.target_output, args.target_shards)
    if args.batch < 1:
        parser.error("--batch must be positive")
    resolved_device = pick_device(args.device)
    scorer = SpliceBertCellScorer(
        model_path, device=resolved_device, batch_size=args.batch, eager=False
    )
    runtime = dict(scorer.runtime_config())
    if runtime.get("device") != resolved_device or runtime.get("batch_size") != args.batch:
        raise RuntimeError("loaded scorer runtime differs from the requested freeze runtime")
    seal = freeze_payload(
        preflight=preflight,
        audit=audit,
        original=scorer.position_table(),
        pinned_files=pinned,
        preflight_authorization=authorization,
        runtime=runtime,
        target_outputs=plan,
        frozen_revision=revision,
    )
    output, manifest_path = output_paths(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-carrier-factorial-target-freeze-v1",
        sys.argv,
        repo=REPO,
        status=seal["status"],
        config={
            "stage": "target",
            "target_mode": seal["target_mode"],
            "target_outputs": plan,
            "runtime": runtime,
        },
        seeds={"structured_controls": int(seal["structured_control_seed"])},
    )
    for path in pinned_paths:
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": seal["status"],
                "target_mode": seal["target_mode"],
                "target_outputs": plan,
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
