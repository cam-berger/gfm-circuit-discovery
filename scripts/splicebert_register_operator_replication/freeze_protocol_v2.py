#!/usr/bin/env python3
"""Bind the independently reviewed A1 repair after its consumed-data replay passes."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_io import (guard_paths, read_json, write_artifact,
    verify_artifact_manifest, require_manifest_input, verify_freeze as verify_parent, manifest_path)
from gfmcd.splicing.register_operator_io_v2 import BASE_FREEZE_SHA256
from gfmcd.splicing.register_operator_statistics_v2 import METHOD
from calibrate_statistics_v2 import frozen_settings, STREAMS
from freeze_protocol import source_closure
from run_model_v2 import verify_runtime_profile


def validate_receipt(path, *, auditor=None):
    receipt = read_json(path)
    manifest = verify_artifact_manifest(path)
    if (receipt.get("status") != "PASS" or manifest.get("status") != "PASS" or receipt.get("passed") is not True
            or receipt.get("inference_method") != METHOD
            or receipt.get("base_freeze_sha256") != BASE_FREEZE_SHA256):
        raise ValueError(f"A1 prerequisite failed: {path}")
    if auditor is not None:
        if receipt.get("auditor_sha256") != sha256_file(auditor):
            raise ValueError("design auditor changed")
        require_manifest_input(manifest, auditor)
        for name, digest in receipt["files"].items():
            p = Path(name) if Path(name).is_absolute() else REPO / name
            if sha256_file(p) != digest:
                raise ValueError(f"reviewed source changed: {p}")
            require_manifest_input(manifest, p)
    return receipt, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-audit", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    guard_paths(args.out)
    study = REPO / "studies/splicebert_register_operator_replication"
    base_path = study / "protocol_freeze_v1.json"
    if sha256_file(base_path) != BASE_FREEZE_SHA256:
        raise ValueError("v1 parent changed")
    parent = verify_parent(base_path, REPO)
    auditor = Path(__file__).parent / "audit_repair_design.py"
    design, design_manifest = validate_receipt(args.design_audit, auditor=auditor)
    development, _ = validate_receipt(args.development)
    if (development.get("fresh_streams_opened") is not False
            or development.get("point_and_pseudo_memorials_passed") != 1768
            or development.get("acceptance", {}).get("all_passed") is not True
            or development.get("counts") != {"development": 256, "null": 1000, "plants": 512}):
        raise ValueError("complete consumed-data development replay must pass before fresh streams")
    tests = sorted((REPO / "tests").glob("test_register_operator*.py"))
    tests.append(REPO / "tests/test_motif_cleanliness.py")
    sources = source_closure([*Path(__file__).parent.glob("*.py"),
        *(REPO / "src/gfmcd/splicing").glob("register_operator_*.py"), *tests])
    amendment = study / "AMENDMENT_A1.md"
    # Every additive executable/source/test must be in the independent review.
    for source in sources | {amendment}:
        relative = str(source.relative_to(REPO)) if source.is_relative_to(REPO) else str(source)
        if relative not in parent["files"]:
            require_manifest_input(design_manifest, source)
    files = {Path(name) if Path(name).is_absolute() else REPO / name for name in parent["files"]}
    files |= sources | {amendment, base_path, manifest_path(base_path),
        args.design_audit.resolve(), manifest_path(args.design_audit).resolve(),
        args.development.resolve(), manifest_path(args.development).resolve()}
    preserved = [args.design_audit, args.development, base_path]
    for name in ("smoke_v1.json", "statistical_calibration_v1.json", "analysis_v1.audit.json", "result_v1.json"):
        path = REPO / "results/splicebert_register_operator_replication" / name
        preserved.append(path)
        files.update((path, manifest_path(path)))
    # A calibration manifest also owns its complete stream and attempt, and a
    # development receipt owns its companion evidence. Pin those bytes directly.
    for artifact in preserved:
        manifest = verify_artifact_manifest(artifact)
        for record in [*manifest.get("inputs", []), *manifest.get("outputs", [])]:
            path = Path(record["path"]).resolve()
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"preserved evidence changed: {path}")
            files.add(path)
    result = deepcopy(parent)
    result.update(schema="gfmcd.register_operator.freeze.v2", status="FROZEN_V2",
        inference_method=METHOD, base_freeze_path=str(base_path), base_freeze_sha256=BASE_FREEZE_SHA256,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), specification_path=str(amendment),
        design_audit_path=str(args.design_audit.resolve()), development_path=str(args.development.resolve()),
        calibration=frozen_settings(), tests=[str(p.relative_to(REPO)) for p in tests])
    result["owner_choices"]["exact_messages"].append("authorized")
    result["owner_choices"]["authorized_amendment"] = "A1 finite-stratum residual correction; no threshold, cohort or readout change"
    outputs = {key: value.replace("_v1", "_v2") for key, value in parent["outputs"].items()}
    result_dir = Path(outputs["calibration"]).parent
    outputs.update(calibration_audit=str(result_dir / "statistical_calibration_v2.audit.json"),
        measurement_audit=str(result_dir / "measurement_v2.audit.json"),
        calibration_attempt_manifest=str(result_dir / "statistical_calibration_v2.attempt.manifest.json"))
    for path in outputs.values():
        if Path(path).exists() or Path(path).is_symlink():
            raise FileExistsError(path)
        if not path.endswith(".manifest.json"):
            guard_paths(path)
    result["outputs"] = outputs
    result["files"] = {(str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)):
                        sha256_file(path) for path in sorted(files)}
    if (platform.python_version() != parent["runtime"]["python"] or any(
        importlib.metadata.version(k) != v for k, v in parent["runtime"]["versions"].items())):
        raise ValueError("runtime differs from the historical fidelity environment")
    verify_runtime_profile(result)
    print(json.dumps(write_artifact(args.out, result, inputs=sorted(files),
        experiment="register-operator:protocol-freeze-v2", config=result["owner_choices"], seeds=STREAMS)))


if __name__ == "__main__":
    main()
