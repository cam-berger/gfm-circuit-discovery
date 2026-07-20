#!/usr/bin/env python
"""Independent structural and numerical audit of stage-two mechanism shards."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ATOL = 1e-12
REFERENCE_ATOL = 1e-6
A3_REFERENCE_ATOL = 1e-4
A3_AMENDMENT = (
    REPO
    / "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A3.md"
)
A3_GATE = "target_whole_stage_reference"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO / value).resolve()


def resolve_portable_manifest_input(path: str | Path) -> Path:
    """Resolve a frozen absolute producer path against a portable repo copy.

    The bundled health manifest predates the Colab deployment and records the
    producer's absolute checkout path.  The overlay carries the same immutable
    bytes under the repository-relative ``results/...`` suffix.  Hash validation
    remains mandatory after this path-only relocation.
    """
    value = Path(path)
    if value.exists() or not value.is_absolute():
        return resolve(value)
    parts = value.parts
    for marker in ("results", "data", "studies"):
        if marker in parts:
            candidate = (REPO / Path(*parts[parts.index(marker):])).resolve()
            if candidate.exists():
                return candidate
    return value.resolve()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def nearest_rank(values: list[float], probability: float) -> float:
    """Deterministic nearest-rank quantile for an audit report."""
    if not values:
        raise ValueError("cannot summarize an empty numerical audit")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def reference_error_summary(values: list[float], effective_atol: float) -> dict:
    """Expose the full numerical disposition rather than only the maximum."""
    return {
        "n_comparisons": len(values),
        "n_above_original_atol": sum(value > REFERENCE_ATOL for value in values),
        "n_above_effective_atol": sum(value > effective_atol for value in values),
        "original_atol": REFERENCE_ATOL,
        "effective_atol": effective_atol,
        "median_abs_error": nearest_rank(values, 0.50),
        "p90_abs_error": nearest_rank(values, 0.90),
        "p99_abs_error": nearest_rank(values, 0.99),
        "max_abs_error": max(values),
    }


def audit_runtime_gate_disposition(manifest: dict, phase: str, *, a3: bool,
                                   failures: list[str]) -> tuple[dict[str, dict], float]:
    """Validate either the original zero-failure contract or A3's one-gate exception.

    A3 is intentionally narrow: it can only admit the already-recorded target
    whole-reference numerical failure.  It cannot waive any missing shard,
    coordinate, QKV, health, finite-value, or mechanism-family gate.
    """
    gates = {row["gate"]: row for row in manifest["gates"]}
    require(len(gates) == len(manifest["gates"]),
            f"{phase}: duplicate runtime gate", failures)
    amended_target = a3 and phase == "target"
    if not amended_target:
        require(manifest["status"] == "PASS" and manifest["n_failures"] == 0,
                f"{phase}: runtime is not a zero-failure pass", failures)
        require(all(row["passed"] for row in gates.values()),
                f"{phase}: failed runtime gate", failures)
        return gates, REFERENCE_ATOL

    gate = gates.get(A3_GATE)
    runtime_failures = manifest.get("failures", [])
    require(manifest["status"] == "FAIL" and manifest["n_failures"] == 1,
            "target: A3 requires the preserved one-failure runtime manifest", failures)
    require(len(runtime_failures) == 1 and gate is not None and runtime_failures[0] == gate,
            "target: A3 runtime failure is not exactly the recorded reference gate", failures)
    require(gate is not None and gate.get("passed") is False and
            float(gate.get("threshold", math.nan)) == REFERENCE_ATOL and
            REFERENCE_ATOL < float(gate.get("value", math.inf)) <= A3_REFERENCE_ATOL,
            "target: A3 reference failure is outside the licensed numerical band", failures)
    require(all(row["passed"] for name, row in gates.items() if name != A3_GATE),
            "target: A3 cannot waive any second runtime gate", failures)
    return gates, A3_REFERENCE_ATOL


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def endpoint_key(candidate_id: str, role: str, contrast: str | None) -> str:
    value = hashlib.sha256(
        f"{candidate_id}|{role}|{contrast or 'general'}".encode()
    ).hexdigest()[:20]
    return f"endpoint_{value}"


def expected_endpoints(candidates: list[dict]) -> list[dict]:
    rows = []
    for candidate in candidates:
        for whole in candidate["validated_whole_endpoints"]:
            role = whole["target_role"]
            contrast = whole.get("contrast")
            rows.append({
                "endpoint_key": endpoint_key(candidate["candidate_id"], role, contrast),
                "candidate_id": candidate["candidate_id"],
                "target_role": role,
                "contrast": contrast,
                "scope_kind": "contrast" if contrast else "target_role",
                "scope_value": contrast or role,
            })
    return rows


def expected_members(candidate: dict) -> dict[str, str]:
    members = {"candidate_whole": "diagnostic", "control_whole": "diagnostic"}
    if candidate["component_type"] == "attention_head":
        qkv = candidate["mechanism"]["population_qk_v"]
        if candidate["estimand"] == "B" and qkv["enabled"]:
            members.update({
                "candidate_QK": "family", "candidate_V_OV": "family",
                "control_QK": "family", "control_V_OV": "family",
                "candidate_QKV_BRIDGE": "diagnostic",
                "control_QKV_BRIDGE": "diagnostic",
            })
        for layer in candidate["mechanism"]["downstream_mlp_recovery"]["layers"]:
            members.update({
                f"candidate_MLP_recovery_L{layer}": "diagnostic",
                f"control_MLP_recovery_L{layer}": "diagnostic",
                f"MLP_recovery_difference_L{layer}": "family",
            })
    else:
        path = candidate["mechanism"]["upstream_attention_path"]
        if path["enabled"]:
            members.update({
                "upstream_candidate_mlp_recovery": "diagnostic",
                "upstream_control_mlp_recovery": "diagnostic",
                "upstream_attention_to_candidate_vs_control_mlp_recovery": "family",
            })
    return members


def packet_expected(packet: dict, endpoints: list[dict], phase: str) -> set[str]:
    if phase == "health":
        return {row["endpoint_key"] for row in endpoints}
    memberships = {
        f"{row['focal_category']}|{row['target_role']}"
        for row in packet["pair_membership"]
    }
    return {
        row["endpoint_key"] for row in endpoints
        if (row["scope_kind"] == "target_role" and packet["target_role"] == row["scope_value"])
        or (row["scope_kind"] == "contrast" and row["scope_value"] in memberships)
    }


def load_whole_reference(package: dict, phase: str) -> dict[tuple[str, str, str], float]:
    record = package["whole_validation_row_references"][phase]
    path = resolve(record["path"])
    if sha256_file(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
        raise RuntimeError(f"{phase} whole-reference hash/size mismatch")
    reference = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row["endpoint_role"] not in {"candidate", "importance_control"}:
                continue
            member = "candidate_whole" if row["endpoint_role"] == "candidate" else "control_whole"
            field = "effect_margin" if phase == "target" else "health_margin_damage"
            reference[(row["packet_id"], row["candidate_id"], member)] = float(row[field])
    return reference


def load_smoke_reference(path: Path, phase: str) -> dict[tuple[str, str, str], float]:
    reference = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            estimand = "A" if row["estimand"].startswith("A_") else "B"
            field = "effect_margin" if phase == "target" else "health_margin_damage"
            reference[(row["packet_id"], estimand, row["component_id"])] = float(row[field])
    return reference


def audit_phase(run_dir: Path, phase: str, package: dict, package_path: Path,
                package_audit_path: Path, *, smoke: bool,
                smoke_reference_path: Path | None, a3: bool,
                failures: list[str]) -> dict:
    manifest_path = run_dir / f"run_manifest.{phase}.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema"] == "head_context_atlas.stage2_mechanism_phase/1",
            f"{phase}: manifest schema mismatch", failures)
    require(manifest["smoke"] is smoke and manifest["whole_validation_loaded"] is (not smoke),
            f"{phase}: smoke/whole-validation boundary mismatch", failures)
    require(manifest["package"]["sha256"] == sha256_file(package_path) and
            manifest["package_audit"]["sha256"] == sha256_file(package_audit_path),
            f"{phase}: package/audit identity mismatch", failures)
    for record in manifest["cohort_inputs"].values():
        path = resolve_portable_manifest_input(record["path"])
        require(path.exists() and sha256_file(path) == record["sha256"],
                f"{phase}: cohort input changed: {path}", failures)

    candidates = manifest["candidates"]
    if not smoke:
        require(candidates == package["candidates"],
                f"{phase}: candidate grid drifted from frozen package", failures)
    candidate_map = {row["candidate_id"]: row for row in candidates}
    require(len(candidate_map) == len(candidates), f"{phase}: duplicate candidates", failures)
    endpoints = expected_endpoints(candidates)
    require(manifest["endpoints"] == endpoints,
            f"{phase}: endpoint grid does not reconstruct from candidates", failures)
    endpoint_map = {row["endpoint_key"]: row for row in endpoints}

    gates, effective_reference_atol = audit_runtime_gate_disposition(
        manifest, phase, a3=a3, failures=failures
    )
    require(f"{phase}_deterministic_repeat" in gates and f"{phase}_packet_grid" in gates and
            f"{phase}_qkv_candidate_gate_grid" in gates,
            f"{phase}: core runtime gate absent", failures)
    qkv_candidates = [
        row for row in candidates if row["component_type"] == "attention_head"
        and row["estimand"] == "B" and row["mechanism"]["population_qk_v"]["enabled"]
    ]
    for candidate in qkv_candidates:
        for component in (candidate["component_id"], candidate["importance_control"]["component_id"]):
            require(f"{phase}_qkv_whole_bridge:{component}" in gates and
                    f"{phase}_native_qkv_identity:{component}" in gates,
                    f"{phase}: QKV bridge/identity absent for {component}", failures)
    if phase == "health":
        require(gates.get("masked_health_accuracy", {}).get("passed") is True and
                gates.get("masked_health_logprob", {}).get("passed") is True,
                "health: masked-MLM health gate absent/failed", failures)
    else:
        require(gates.get("target_whole_stage_reference_grid", {}).get("passed") is True and
                (gates.get("target_whole_stage_reference", {}).get("passed") is (not a3)),
                "target: whole-reference gate disposition does not match the audit mode", failures)

    if smoke:
        if smoke_reference_path is None:
            raise RuntimeError("smoke audit requires --reference-rows")
        reference = load_smoke_reference(smoke_reference_path, phase)
    else:
        reference = load_whole_reference(package, phase)
    shard_dir = run_dir / f"{phase}_shards"
    actual = {path.name: path for path in shard_dir.glob("packet_*.json")}
    require(set(actual) == set(manifest["shards"]), f"{phase}: shard filename grid mismatch", failures)
    packet_ids = set()
    n_members = 0
    reference_errors = []
    bridge_errors = []
    algebra_errors = []
    for name, path in sorted(actual.items()):
        record = json.loads(path.read_text())
        declared = manifest["shards"][name]
        require(declared["sha256"] == sha256_file(path) and
                declared["size_bytes"] == path.stat().st_size,
                f"{phase}: shard hash/size mismatch {name}", failures)
        require(record["schema"] == "head_context_atlas.stage2_mechanism_shard/1" and
                record["phase"] == phase and record["contract_sha256"] == manifest["contract_sha256"],
                f"{phase}: shard contract mismatch {name}", failures)
        require(finite(record), f"{phase}: non-finite shard {name}", failures)
        packet = record["packet"]
        packet_id = packet["packet_id"]
        require(packet_id not in packet_ids, f"{phase}: duplicate packet {packet_id}", failures)
        packet_ids.add(packet_id)
        observed_keys = {row["endpoint_key"] for row in record["endpoints"]}
        require(len(observed_keys) == len(record["endpoints"]) and
                observed_keys == packet_expected(packet, endpoints, phase),
                f"{phase}: endpoint scope mismatch {packet_id}", failures)
        for endpoint in record["endpoints"]:
            expected_endpoint = endpoint_map[endpoint["endpoint_key"]]
            require({key: endpoint[key] for key in expected_endpoint} == expected_endpoint,
                    f"{phase}: endpoint metadata mismatch {packet_id}", failures)
            candidate = candidate_map[endpoint["candidate_id"]]
            expected_grid = expected_members(candidate)
            members = endpoint["members"]
            observed_grid = {row["member"]: row for row in members}
            require(len(observed_grid) == len(members) and
                    {key: row["member_role"] for key, row in observed_grid.items()} == expected_grid,
                    f"{phase}: member grid mismatch {packet_id}|{candidate['candidate_id']}", failures)
            n_members += len(members)
            for member, row in observed_grid.items():
                require(row["candidate_id"] == candidate["candidate_id"] and
                        row["component_id"] == candidate["component_id"] and
                        row["control_component_id"] == candidate["importance_control"]["component_id"] and
                        row["estimand"] == candidate["estimand"],
                        f"{phase}: member identity drift {packet_id}|{member}", failures)
            for member, component in (
                ("candidate_whole", candidate["component_id"]),
                ("control_whole", candidate["importance_control"]["component_id"]),
            ):
                key = ((packet_id, candidate["estimand"], component) if smoke else
                       (packet_id, candidate["candidate_id"], member))
                expected = reference.get(key)
                require(expected is not None,
                        f"{phase}: independent whole reference absent {packet_id}|{member}", failures)
                if expected is not None:
                    reference_errors.append(abs(observed_grid[member]["value"] - expected))
            if "candidate_QKV_BRIDGE" in observed_grid:
                bridge_errors.extend([
                    abs(observed_grid["candidate_QKV_BRIDGE"]["value"] -
                        observed_grid["candidate_whole"]["value"]),
                    abs(observed_grid["control_QKV_BRIDGE"]["value"] -
                        observed_grid["control_whole"]["value"]),
                ])
            for member, row in observed_grid.items():
                if member.startswith("MLP_recovery_difference_L"):
                    layer = member.rsplit("L", 1)[1]
                    expected = (observed_grid[f"candidate_MLP_recovery_L{layer}"]["value"] -
                                observed_grid[f"control_MLP_recovery_L{layer}"]["value"])
                    algebra_errors.append(abs(row["value"] - expected))
                elif member == "upstream_attention_to_candidate_vs_control_mlp_recovery":
                    expected = (observed_grid["upstream_candidate_mlp_recovery"]["value"] -
                                observed_grid["upstream_control_mlp_recovery"]["value"])
                    algebra_errors.append(abs(row["value"] - expected))

    require(packet_ids and len(packet_ids) == manifest["n_packets"],
            f"{phase}: packet count mismatch", failures)
    require(n_members == manifest["n_member_rows"], f"{phase}: member count mismatch", failures)
    require(reference_errors and max(reference_errors) <= effective_reference_atol,
            f"{phase}: independent whole-reference mismatch", failures)
    require(not bridge_errors or max(bridge_errors) <= ATOL,
            f"{phase}: QKV bridge-value mismatch", failures)
    require(not algebra_errors or max(algebra_errors) <= ATOL,
            f"{phase}: derived-member arithmetic mismatch", failures)
    numerical = reference_error_summary(reference_errors, effective_reference_atol)
    return {
        "status": "PASS" if not (a3 and phase == "target") else "PASS_WITH_A3_AMENDMENT",
        "runtime_status": manifest["status"], "n_packets": len(packet_ids),
        "n_member_rows": n_members, "n_candidates": len(candidates),
        "n_endpoints": len(endpoints),
        "max_whole_reference_abs_error": max(reference_errors),
        "whole_reference_numerical_audit": numerical,
        "max_qkv_bridge_value_abs_error": max(bridge_errors) if bridge_errors else None,
        "max_derived_algebra_abs_error": max(algebra_errors) if algebra_errors else 0.0,
        "run_manifest_sha256": sha256_file(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-audit", required=True)
    parser.add_argument("--reference-rows", default=None)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--a3-numerical-amendment", action="store_true",
        help="Admit only the preserved A3 target reference-tolerance failure.",
    )
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    run_dir = Path(args.run_dir)
    package_path = Path(args.package)
    package_audit_path = Path(args.package_audit)
    audit_path = Path(args.audit) if args.audit else run_dir / "validation_audit.json"
    if audit_path.exists():
        raise FileExistsError(f"refuse to overwrite immutable audit {audit_path}")
    package = json.loads(package_path.read_text())
    package_audit = json.loads(package_audit_path.read_text())
    failures: list[str] = []
    require(not args.a3_numerical_amendment or not args.smoke,
            "A3 is only defined for the completed full target run", failures)
    require(not args.a3_numerical_amendment or args.audit is not None,
            "A3 requires an explicit new audit path; the original audit name is preserved", failures)
    require(not args.a3_numerical_amendment or A3_AMENDMENT.is_file(),
            "A3 amendment document is missing", failures)
    require(package_audit["status"] == "PASS" and package_audit["n_failures"] == 0 and
            package_audit["package_sha256"] == sha256_file(package_path),
            "stage-two package audit is not a current zero-failure pass", failures)
    smoke_reference = Path(args.reference_rows) if args.reference_rows else None
    require((smoke_reference is not None) is args.smoke,
            "reference rows are required only for smoke", failures)
    reports = {}
    for phase in ("health", "target"):
        reports[phase] = audit_phase(
            run_dir, phase, package, package_path, package_audit_path,
            smoke=args.smoke, smoke_reference_path=smoke_reference,
            a3=args.a3_numerical_amendment, failures=failures,
        )
    target_manifest = json.loads((run_dir / "run_manifest.target.json").read_text())
    cohort = target_manifest["cohort"]
    if args.smoke:
        require(cohort["expected_analysis_split"] == "atlas_discovery" and
                target_manifest["whole_validation_loaded"] is False,
                "smoke crossed the validation boundary", failures)
    else:
        require(cohort["expected_analysis_split"] == "atlas_internal_validation" and
                cohort["gene_overlap_with_discovery"] == [] and
                cohort["sequence_overlap_with_discovery"] == [] and
                cohort["historically_or_sealed_chromosomes_present"] == [] and
                target_manifest["whole_validation_loaded"] is True,
                "full run split/provenance boundary failed", failures)
    report = {
        "schema": ("head_context_atlas.stage2_mechanism_validation_audit/2"
                   if args.a3_numerical_amendment else
                   "head_context_atlas.stage2_mechanism_validation_audit/1"),
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": (
            "independent A3 execution audit over preserved shards; no mechanism label"
            if args.a3_numerical_amendment else
            "independent stage-two execution audit; no mechanism label"
        ),
        "smoke": args.smoke, "whole_validation_loaded": not args.smoke,
        "stage2_outcomes_loaded": True, "phase_reports": reports,
        "package_sha256": sha256_file(package_path),
        "auditor_sha256": sha256_file(Path(__file__)),
        "amendment": ({
            "id": "A3",
            "path": str(A3_AMENDMENT.relative_to(REPO)),
            "sha256": sha256_file(A3_AMENDMENT),
            "original_runtime_manifest_preserved": True,
            "mechanism_labels_loaded_before_amendment": False,
            "original_reference_atol": REFERENCE_ATOL,
            "effective_reference_atol": A3_REFERENCE_ATOL,
        } if args.a3_numerical_amendment else None),
        "n_failures": len(failures), "failures": failures,
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
