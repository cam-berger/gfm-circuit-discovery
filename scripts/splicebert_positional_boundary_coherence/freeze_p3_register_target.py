#!/usr/bin/env python
"""Seal the one-shot Phase-III reserve after independent prerequisite audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_register_stage import (  # noqa: E402
    REGISTER_MANIFEST_CHAIN_SCHEMA,
    REGISTER_PREFLIGHT_ANALYSIS_SCHEMA,
    REGISTER_PREFLIGHT_AUDIT_SCHEMA,
    REGISTER_PREREQUISITE_ARTIFACTS,
    make_register_target_seal,
    manifest_binds_path,
    manifest_precedes,
    validate_phase2_authorization,
    validate_sibling_manifest,
)


CHECKPOINT_SHA256 = "2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d"
PHASE2_TARGET_SCHEMA = "gfmcd.pbc.carrier_factorial_target.v1"


def output_paths(path: str | Path, target_path: str | Path) -> tuple[Path, Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    reserved_target = guard_output_path(Path(target_path).expanduser().resolve())
    return output, manifest, reserved_target


def _tracked_tree_dirty(repo: Path) -> bool:
    for command in (
        ("git", "diff", "--quiet"),
        ("git", "diff", "--cached", "--quiet"),
    ):
        completed = subprocess.run(command, cwd=repo, check=False)
        if completed.returncode not in {0, 1}:
            raise RuntimeError("could not verify tracked-tree cleanliness")
        if completed.returncode == 1:
            return True
    return False


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def validate_prerequisite_manifest_chain(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Verify every Phase-III prerequisite artifact/manifest edge before sealing."""

    missing = {
        key
        for artifact in REGISTER_PREREQUISITE_ARTIFACTS
        for key in (artifact, f"{artifact}_manifest")
        if key not in paths
    }
    if missing:
        raise RuntimeError(f"prerequisite manifest chain is incomplete: {sorted(missing)}")
    manifests = {
        artifact: _load(paths[f"{artifact}_manifest"])
        for artifact in REGISTER_PREREQUISITE_ARTIFACTS
    }
    artifacts = {
        artifact: _load(paths[artifact])
        for artifact in REGISTER_PREREQUISITE_ARTIFACTS
    }
    checks: dict[str, bool] = {}
    records: dict[str, dict[str, Any]] = {}
    for artifact in REGISTER_PREREQUISITE_ARTIFACTS:
        validation = validate_sibling_manifest(
            manifests[artifact],
            manifest_path=paths[f"{artifact}_manifest"],
            artifact_path=paths[artifact],
        )
        checks[f"{artifact}_sibling_manifest"] = bool(validation["passed"])
        records[artifact] = {
            "artifact_sha256": sha256_file(paths[artifact]),
            "manifest_sha256": sha256_file(paths[f"{artifact}_manifest"]),
            "created_at_utc": validation["created_at_utc"],
            "git_commit": validation["git_commit"],
        }

    input_edges = (
        ("phase2_analysis", "phase2_target"),
        ("phase2_audit", "phase2_target"),
        ("phase2_audit", "phase2_target_manifest"),
        ("phase2_audit", "phase2_analysis"),
        ("phase2_audit", "phase2_analysis_manifest"),
        ("candidate_freeze", "candidate_screen"),
        ("path_mde", "candidate_screen"),
        ("path_mde", "candidate_freeze"),
        ("candidate_freeze_audit", "candidate_screen"),
        ("candidate_freeze_audit", "candidate_screen_manifest"),
        ("candidate_freeze_audit", "candidate_freeze"),
        ("candidate_freeze_audit", "candidate_freeze_manifest"),
        ("candidate_freeze_audit", "path_mde"),
        ("candidate_freeze_audit", "path_mde_manifest"),
        ("register_preflight", "phase2_analysis"),
        ("register_preflight", "phase2_audit"),
        ("register_preflight_analysis", "register_preflight"),
        ("register_preflight_analysis", "path_mde"),
        ("register_preflight_analysis", "candidate_freeze_audit"),
        ("register_preflight_audit", "register_preflight"),
        ("register_preflight_audit", "register_preflight_manifest"),
        ("register_preflight_audit", "register_preflight_analysis"),
        ("register_preflight_audit", "register_preflight_analysis_manifest"),
    )
    for consumer, source in input_edges:
        checks[f"{consumer}_consumes_{source}"] = manifest_binds_path(
            manifests[consumer], paths[source], field="inputs"
        )

    chronological_edges = (
        ("phase2_target", "phase2_analysis"),
        ("phase2_analysis", "phase2_audit"),
        ("candidate_screen", "candidate_freeze"),
        ("candidate_freeze", "path_mde"),
        ("path_mde", "candidate_freeze_audit"),
        ("phase2_audit", "register_preflight"),
        ("register_preflight", "register_preflight_analysis"),
        ("candidate_freeze_audit", "register_preflight_analysis"),
        ("register_preflight_analysis", "register_preflight_audit"),
    )
    for producer, consumer in chronological_edges:
        checks[f"{producer}_precedes_{consumer}"] = manifest_precedes(
            manifests[producer], manifests[consumer]
        )
    checks["single_code_revision"] = (
        len({str(manifest.get("git_commit")) for manifest in manifests.values()}) == 1
    )
    for label in ("phase2_audit", "candidate_freeze_audit", "register_preflight_audit"):
        audit_checks = artifacts[label].get("checks")
        manifest_checks = {
            str(name): value
            for name, value in audit_checks.items()
            if "manifest" in str(name) or "output_bound" in str(name)
        } if isinstance(audit_checks, Mapping) else {}
        checks[f"{label}_passed"] = artifacts[label].get("status") == "PASS"
        checks[f"{label}_verified_producer_manifests"] = bool(
            manifest_checks and all(value is True for value in manifest_checks.values())
        )
    return {
        "schema": REGISTER_MANIFEST_CHAIN_SCHEMA,
        "stage": "target",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "artifacts": records,
    }


def build_seal(
    *,
    phase2_target: Mapping[str, Any],
    phase2_analysis: Mapping[str, Any],
    phase2_audit: Mapping[str, Any],
    phase2_analysis_sha256: str,
    phase2_target_sha256: str,
    register_preflight_analysis: Mapping[str, Any],
    register_preflight_audit: Mapping[str, Any],
    register_preflight_analysis_sha256: str,
    bindings: Mapping[str, Mapping[str, Any]],
    include_p10: bool,
    target_path: str,
    shard_count: int = 1,
    target_analysis_path: str | None = None,
    target_audit_path: str | None = None,
    mediation_target_path: str | None = None,
    mediation_analysis_path: str | None = None,
    mediation_audit_path: str | None = None,
    mediation_shard_count: int | None = None,
    prerequisite_manifest_chain: Mapping[str, Any],
) -> dict[str, Any]:
    phase2_authorization = validate_phase2_authorization(
        phase2_analysis,
        phase2_audit,
        analysis_sha256=phase2_analysis_sha256,
    )
    if (
        phase2_target.get("schema") != PHASE2_TARGET_SCHEMA
        or phase2_target.get("status") != "TARGET_SCORED_UNANALYZED"
        or phase2_analysis.get("target_sha256") != phase2_target_sha256
    ):
        raise RuntimeError("Phase-II target is incomplete or not bound to its analysis")
    if (
        register_preflight_analysis.get("schema")
        != REGISTER_PREFLIGHT_ANALYSIS_SCHEMA
        or register_preflight_audit.get("schema") != REGISTER_PREFLIGHT_AUDIT_SCHEMA
        or register_preflight_audit.get("status") != "PASS"
        or register_preflight_audit.get("analysis_sha256")
        != register_preflight_analysis_sha256
    ):
        raise RuntimeError("register preflight analysis/audit binding is invalid")
    doses = phase2_target.get("selected_doses")
    if not isinstance(doses, Mapping):
        raise RuntimeError("Phase-II target does not record its frozen doses")
    return make_register_target_seal(
        phase2_authorization=phase2_authorization,
        preflight_analysis=register_preflight_analysis,
        preflight_audit=register_preflight_audit,
        bindings=bindings,
        include_p10=include_p10,
        p3_attenuation_dose=float(doses["p3"]),
        p10_attenuation_dose=(
            float(doses["p10"]) if include_p10 else None
        ),
        output_path=target_path,
        shard_count=shard_count,
        relocation_analysis_path=target_analysis_path,
        relocation_audit_path=target_audit_path,
        mediation_output_path=mediation_target_path,
        mediation_analysis_path=mediation_analysis_path,
        mediation_audit_path=mediation_audit_path,
        mediation_shard_count=mediation_shard_count,
        prerequisite_manifest_chain=prerequisite_manifest_chain,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-target", required=True)
    parser.add_argument("--phase2-target-manifest", required=True)
    parser.add_argument("--phase2-analysis", required=True)
    parser.add_argument("--phase2-analysis-manifest", required=True)
    parser.add_argument("--phase2-audit", required=True)
    parser.add_argument("--phase2-audit-manifest", required=True)
    parser.add_argument("--register-preflight", required=True)
    parser.add_argument("--register-preflight-manifest", required=True)
    parser.add_argument("--register-preflight-analysis", required=True)
    parser.add_argument("--register-preflight-analysis-manifest", required=True)
    parser.add_argument("--register-preflight-audit", required=True)
    parser.add_argument("--register-preflight-audit-manifest", required=True)
    parser.add_argument("--candidate-screen", required=True)
    parser.add_argument("--candidate-screen-manifest", required=True)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--candidate-freeze-manifest", required=True)
    parser.add_argument("--candidate-freeze-audit", required=True)
    parser.add_argument("--candidate-freeze-audit-manifest", required=True)
    parser.add_argument("--path-mde", required=True)
    parser.add_argument("--path-mde-manifest", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--include-p10", action="store_true")
    parser.add_argument("--shards", type=int, default=24)
    parser.add_argument("--target-out", required=True)
    parser.add_argument("--target-analysis-out", required=True)
    parser.add_argument("--target-audit-out", required=True)
    parser.add_argument("--mediation-target-out", required=True)
    parser.add_argument("--mediation-analysis-out", required=True)
    parser.add_argument("--mediation-audit-out", required=True)
    parser.add_argument("--mediation-shards", type=int, default=24)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output, manifest_path, target_path = output_paths(args.out, args.target_out)
    if _tracked_tree_dirty(REPO):
        raise RuntimeError("refusing to freeze a reserve target from a dirty tracked tree")
    paths = {
        "phase2_target": Path(args.phase2_target).expanduser().resolve(),
        "phase2_target_manifest": Path(args.phase2_target_manifest).expanduser().resolve(),
        "phase2_analysis": Path(args.phase2_analysis).expanduser().resolve(),
        "phase2_analysis_manifest": Path(args.phase2_analysis_manifest).expanduser().resolve(),
        "phase2_audit": Path(args.phase2_audit).expanduser().resolve(),
        "phase2_audit_manifest": Path(args.phase2_audit_manifest).expanduser().resolve(),
        "register_preflight": Path(args.register_preflight).expanduser().resolve(),
        "register_preflight_manifest": Path(args.register_preflight_manifest).expanduser().resolve(),
        "register_preflight_analysis": Path(args.register_preflight_analysis).expanduser().resolve(),
        "register_preflight_analysis_manifest": Path(args.register_preflight_analysis_manifest).expanduser().resolve(),
        "register_preflight_audit": Path(args.register_preflight_audit).expanduser().resolve(),
        "register_preflight_audit_manifest": Path(args.register_preflight_audit_manifest).expanduser().resolve(),
        "candidate_screen": Path(args.candidate_screen).expanduser().resolve(),
        "candidate_screen_manifest": Path(args.candidate_screen_manifest).expanduser().resolve(),
        "candidate_freeze": Path(args.candidate_freeze).expanduser().resolve(),
        "candidate_freeze_manifest": Path(args.candidate_freeze_manifest).expanduser().resolve(),
        "candidate_freeze_audit": Path(args.candidate_freeze_audit).expanduser().resolve(),
        "candidate_freeze_audit_manifest": Path(args.candidate_freeze_audit_manifest).expanduser().resolve(),
        "path_mde": Path(args.path_mde).expanduser().resolve(),
        "path_mde_manifest": Path(args.path_mde_manifest).expanduser().resolve(),
        "cohort": Path(args.cohort).expanduser().resolve(),
    }
    checkpoint_root = Path(args.model).expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise RuntimeError("--model must name the official SpliceBERT checkpoint directory")
    checkpoint = checkpoint_root / "pytorch_model.bin"
    if not checkpoint.is_file() or sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("official SpliceBERT.1024nt checkpoint hash mismatch")
    model_files = {
        "checkpoint": checkpoint,
        "checkpoint_config": checkpoint_root / "config.json",
        "tokenizer_json": checkpoint_root / "tokenizer.json",
        "tokenizer_config": checkpoint_root / "tokenizer_config.json",
        "special_tokens_map": checkpoint_root / "special_tokens_map.json",
        "vocabulary": checkpoint_root / "vocab.txt",
    }
    protocol = REPO / "studies/splicebert_positional_boundary_coherence/PREREGISTRATION.md"
    code_paths = {
        "protocol": protocol,
        "register_core": REPO / "src/gfmcd/splicing/pbc_register_relocation.py",
        "register_runner": REPO / "src/gfmcd/splicing/pbc_register_runner.py",
        "register_stage": REPO / "src/gfmcd/splicing/pbc_register_stage.py",
        "candidate_core": REPO / "src/gfmcd/splicing/pbc_register_candidates.py",
        "candidate_model_runner": REPO
        / "src/gfmcd/splicing/pbc_register_candidate_runner.py",
        "mediation_model_runner": REPO
        / "src/gfmcd/splicing/pbc_register_mediation_runner.py",
        "register_freezer": Path(__file__).resolve(),
        "register_preflight_runner": REPO
        / "scripts/splicebert_positional_boundary_coherence/run_p3_register_preflight.py",
        "register_preflight_analyzer": REPO
        / "scripts/splicebert_positional_boundary_coherence/analyze_p3_register_preflight.py",
        "register_preflight_auditor": REPO
        / "scripts/splicebert_positional_boundary_coherence/audit_p3_register_preflight.py",
        "target_runner": REPO
        / "scripts/splicebert_positional_boundary_coherence/run_p3_register_target.py",
        "target_assembler": REPO
        / "scripts/splicebert_positional_boundary_coherence/assemble_p3_register_target.py",
        "target_analyzer": REPO
        / "scripts/splicebert_positional_boundary_coherence/analyze_p3_register_relocation.py",
        "target_auditor": REPO
        / "scripts/splicebert_positional_boundary_coherence/audit_p3_register_relocation.py",
        "candidate_screen_runner": REPO
        / "scripts/splicebert_positional_boundary_coherence/run_p3_register_candidate_screen.py",
        "candidate_freeze_builder": REPO
        / "scripts/splicebert_positional_boundary_coherence/freeze_p3_register_candidates.py",
        "candidate_freeze_auditor": REPO
        / "scripts/splicebert_positional_boundary_coherence/audit_p3_register_candidate_freeze.py",
        "mediation_target_runner": REPO
        / "scripts/splicebert_positional_boundary_coherence/run_p3_register_mediation_target.py",
        "mediation_target_assembler": REPO
        / "scripts/splicebert_positional_boundary_coherence/assemble_p3_register_mediation_target.py",
        "mediation_target_analyzer": REPO
        / "scripts/splicebert_positional_boundary_coherence/analyze_p3_register_mediation.py",
        "mediation_target_auditor": REPO
        / "scripts/splicebert_positional_boundary_coherence/audit_p3_register_mediation.py",
        "register_tests": REPO / "tests/test_pbc_register_relocation.py",
        "runner_tests": REPO / "tests/test_pbc_register_runner.py",
        "stage_tests": REPO / "tests/test_pbc_register_stage.py",
        "target_runner_tests": REPO / "tests/test_pbc_register_target_runner.py",
        "preflight_tests": REPO / "tests/test_pbc_register_preflight.py",
        "candidate_execution_tests": REPO
        / "tests/test_pbc_register_candidate_execution.py",
        "mediation_execution_tests": REPO
        / "tests/test_pbc_register_mediation_execution.py",
    }
    bindings = {
        label: file_record(path)
        for label, path in {**paths, **model_files, **code_paths}.items()
    }
    if any(record.get("missing") for record in bindings.values()):
        raise RuntimeError("reserve freeze input is missing")
    prerequisite_manifest_chain = validate_prerequisite_manifest_chain(paths)
    if prerequisite_manifest_chain["status"] != "PASS":
        failures = sorted(
            name
            for name, passed in prerequisite_manifest_chain["checks"].items()
            if not passed
        )
        raise RuntimeError(f"prerequisite manifest chain failed: {failures}")
    payloads = {
        label: _load(paths[label])
        for label in REGISTER_PREREQUISITE_ARTIFACTS
    }
    seal = build_seal(
        phase2_target=payloads["phase2_target"],
        phase2_analysis=payloads["phase2_analysis"],
        phase2_audit=payloads["phase2_audit"],
        phase2_analysis_sha256=bindings["phase2_analysis"]["sha256"],
        phase2_target_sha256=bindings["phase2_target"]["sha256"],
        register_preflight_analysis=payloads["register_preflight_analysis"],
        register_preflight_audit=payloads["register_preflight_audit"],
        register_preflight_analysis_sha256=bindings["register_preflight_analysis"]["sha256"],
        bindings=bindings,
        include_p10=args.include_p10,
        target_path=str(target_path),
        shard_count=args.shards,
        target_analysis_path=str(
            guard_output_path(Path(args.target_analysis_out).expanduser().resolve())
        ),
        target_audit_path=str(
            guard_output_path(Path(args.target_audit_out).expanduser().resolve())
        ),
        mediation_target_path=str(
            guard_output_path(Path(args.mediation_target_out).expanduser().resolve())
        ),
        mediation_analysis_path=str(
            guard_output_path(Path(args.mediation_analysis_out).expanduser().resolve())
        ),
        mediation_audit_path=str(
            guard_output_path(Path(args.mediation_audit_out).expanduser().resolve())
        ),
        mediation_shard_count=args.mediation_shards,
        prerequisite_manifest_chain=prerequisite_manifest_chain,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-p3-register-target-freeze-v1",
        sys.argv,
        repo=REPO,
        status="TARGET_AUTHORIZED",
        config={
            "stage": "target",
            "mode": seal["mode"],
            "target_path": str(target_path),
            "seal_sha256": seal["seal_sha256"],
        },
    )
    for path in {**paths, **model_files, **code_paths}.values():
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": seal["status"],
                "mode": seal["mode"],
                "seal_sha256": seal["seal_sha256"],
                "target_out": str(target_path),
                "shards": seal["shard_count"],
                "mediation_shards": seal["mediation"]["shard_count"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
