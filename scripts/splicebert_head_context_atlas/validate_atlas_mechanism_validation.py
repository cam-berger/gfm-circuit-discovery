#!/usr/bin/env python
"""Independent structural and numerical audit of whole/group validation shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
REFERENCE_ATOL = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def expected_endpoint_ids(package: dict, *, smoke: bool) -> set[str]:
    if smoke:
        return {
            "smoke|candidate|A|L0H0", "smoke|control|A|L0H1",
            "smoke|candidate|B|L3H1", "smoke|control|B|L3H2",
            "smoke|candidate|A|L5MLP", "smoke|control|A|L4MLP",
            "smoke|candidate|B|L3MLP", "smoke|control|B|L2MLP",
            "smoke|group|A|mixed", "smoke|group|B|mixed",
            "positive_control|B|L3H1",
        }
    ids = set()
    for candidate in package["candidates"]:
        ids.add(f"candidate|{candidate['candidate_id']}")
        ids.add(f"importance_control|{candidate['candidate_id']}")
    ids.update(f"exact_group|{group}" for group in package["exact_groups"])
    positive = package["positive_control"]
    ids.add(f"positive_control|{positive['estimand']}|{positive['component_id']}")
    return ids


def packet_expected_ids(packet: dict, endpoints: list[dict], phase: str) -> set[str]:
    if phase == "health":
        return {endpoint["endpoint_id"] for endpoint in endpoints}
    memberships = {
        f"{row['focal_category']}|{row['target_role']}"
        for row in packet["pair_membership"]
    }
    expected = set()
    for endpoint in endpoints:
        if endpoint["scope_kind"] == "general":
            expected.add(endpoint["endpoint_id"])
        elif endpoint["scope_kind"] == "target_role" and packet["target_role"] == endpoint["scope_value"]:
            expected.add(endpoint["endpoint_id"])
        elif endpoint["scope_kind"] == "contrast" and endpoint["scope_value"] in memberships:
            expected.add(endpoint["endpoint_id"])
    return expected


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def audit_phase(run_dir: Path, phase: str, package: dict, *, smoke: bool,
                reference: dict[tuple[str, str, str], dict], failures: list[str]) -> dict:
    manifest_path = run_dir / f"run_manifest.{phase}.json"
    manifest = json.loads(manifest_path.read_text())
    expected_status = "HEALTH_PASS" if phase == "health" else "TARGET_COMPLETE"
    require(manifest["status"] == expected_status and manifest["n_failures"] == 0,
            f"{phase}: runtime manifest is not a zero-failure {expected_status}", failures)
    require(manifest["smoke"] is smoke and manifest["validation_loaded"] is (not smoke),
            f"{phase}: smoke/validation-loaded boundary mismatch", failures)
    endpoints = manifest["endpoints"]
    endpoint_ids = [endpoint["endpoint_id"] for endpoint in endpoints]
    expected_all = expected_endpoint_ids(package, smoke=smoke)
    require(len(endpoint_ids) == len(set(endpoint_ids)) and set(endpoint_ids) == expected_all,
            f"{phase}: endpoint family does not reconstruct from package", failures)
    computation_ids = {endpoint["computation_id"] for endpoint in endpoints}
    require(manifest["n_unique_computations"] == len(computation_ids),
            f"{phase}: computation deduplication count mismatch", failures)

    gates = {gate["gate"]: gate for gate in manifest["gates"]}
    require(all(gate["passed"] for gate in gates.values()),
            f"{phase}: at least one runtime gate failed", failures)
    require(f"{phase}_deterministic_repeat" in gates,
            f"{phase}: missing deterministic-repeat gate", failures)
    require(f"{phase}_all_component_native_source_identity" in gates,
            f"{phase}: missing all-component identity gate", failures)
    for computation_id in computation_ids:
        name = f"{phase}_native_source_identity:{computation_id}"
        require(name in gates and gates[name]["value"] <= gates[name]["threshold"],
                f"{phase}: missing/failing identity for {computation_id}", failures)
    if phase == "health":
        require(gates.get("masked_health_accuracy", {}).get("passed") is True,
                "health: masked-health accuracy gate absent/failed", failures)
        require(gates.get("masked_health_logprob", {}).get("passed") is True,
                "health: masked-health logprob gate absent/failed", failures)

    shard_dir = run_dir / f"{phase}_shards"
    declared = manifest["shards"]
    actual = {path.name: path for path in shard_dir.glob("packet_*.json")}
    require(set(actual) == set(declared), f"{phase}: shard filename set mismatch", failures)
    packet_ids = set()
    n_rows = 0
    reference_errors = []
    duplicate_errors = []
    for name, path in sorted(actual.items()):
        record = json.loads(path.read_text())
        require(declared[name]["sha256"] == sha256_file(path) and
                declared[name]["size_bytes"] == path.stat().st_size,
                f"{phase}: shard hash/size mismatch {name}", failures)
        require(record["schema"] == "head_context_atlas.mechanism_validation_shard/1" and
                record["phase"] == phase and
                record["contract_sha256"] == manifest["contract_sha256"],
                f"{phase}: shard contract mismatch {name}", failures)
        packet = record["packet"]
        packet_id = packet["packet_id"]
        require(packet_id not in packet_ids, f"{phase}: duplicate packet {packet_id}", failures)
        packet_ids.add(packet_id)
        results = record["results"]
        result_ids = {result["endpoint_id"] for result in results}
        require(len(result_ids) == len(results) and result_ids == packet_expected_ids(packet, endpoints, phase),
                f"{phase}: endpoint grid mismatch for {packet_id}", failures)
        require(finite(record), f"{phase}: non-finite shard {packet_id}", failures)
        n_rows += len(results)

        by_computation: dict[str, list[dict]] = defaultdict(list)
        for result in results:
            by_computation[result["computation_id"]].append(result)
            if phase == "target":
                require(abs(
                    (result["baseline_margin"] - result["intervened_margin"])
                    - result["effect_margin"]
                ) <= 1e-12, f"target: effect sign/arithmetic mismatch {packet_id}", failures)
            else:
                require(abs(
                    (result["baseline_health_margin"] - result["intervened_health_margin"])
                    - result["health_margin_damage"]
                ) <= 1e-12, f"health: effect sign/arithmetic mismatch {packet_id}", failures)

            if smoke and len(result["component_ids"]) == 1:
                component = result["component_ids"][0]
                ref = reference.get((packet_id, component, result["estimand"]))
                require(ref is not None,
                        f"{phase}: missing independent reference {packet_id}|{component}|{result['estimand']}",
                        failures)
                if ref is not None:
                    observed = (
                        result["effect_margin"] if phase == "target"
                        else result["health_margin_damage"]
                    )
                    expected = ref["effect_margin"] if phase == "target" else ref["health_margin_damage"]
                    reference_errors.append(abs(float(observed) - float(expected)))

        for computation_id, rows in by_computation.items():
            if len(rows) < 2:
                continue
            field = "effect_margin" if phase == "target" else "health_margin_damage"
            values = [float(row[field]) for row in rows]
            duplicate_errors.append(max(values) - min(values))

    require(len(packet_ids) == manifest["n_packets"],
            f"{phase}: packet count mismatch", failures)
    require(n_rows == manifest["n_endpoint_rows"],
            f"{phase}: endpoint-row count mismatch", failures)
    if reference_errors:
        require(max(reference_errors) <= REFERENCE_ATOL,
                f"{phase}: independent preflight mismatch max={max(reference_errors):.3e}", failures)
    if duplicate_errors:
        require(max(duplicate_errors) <= 1e-12,
                f"{phase}: reused computation emitted divergent aliases", failures)
    return {
        "status": manifest["status"],
        "n_packets": len(packet_ids),
        "n_endpoint_rows": n_rows,
        "n_endpoints": len(endpoints),
        "n_unique_computations": len(computation_ids),
        "max_reference_abs_error": max(reference_errors) if reference_errors else None,
        "max_alias_abs_error": max(duplicate_errors) if duplicate_errors else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-audit", required=True)
    parser.add_argument("--reference-rows", default=None)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--smoke", action="store_true")
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
    require(package_audit["status"] == "PASS" and package_audit["n_failures"] == 0,
            "mechanism-package audit is not a zero-failure pass", failures)
    require(package_audit["package_sha256"] == sha256_file(package_path),
            "mechanism package changed after audit", failures)

    reference = {}
    if args.smoke:
        require(args.reference_rows is not None, "smoke audit requires independent reference rows", failures)
        if args.reference_rows:
            for row in read_jsonl(Path(args.reference_rows)):
                estimand = "A" if row["estimand"].startswith("A_") else "B"
                reference[(row["packet_id"], row["component_id"], estimand)] = row
    else:
        require(args.reference_rows is None,
                "full validation must not use discovery preflight as outcome reference", failures)

    phase_reports = {}
    for phase in ("health", "target"):
        phase_reports[phase] = audit_phase(
            run_dir, phase, package, smoke=args.smoke,
            reference=reference, failures=failures,
        )
    target_manifest = json.loads((run_dir / "run_manifest.target.json").read_text())
    split = target_manifest["cohort"]
    if args.smoke:
        require(split["expected_analysis_split"] == "atlas_discovery" and
                target_manifest["validation_loaded"] is False,
                "smoke crossed discovery/validation boundary", failures)
    else:
        require(split["expected_analysis_split"] == "atlas_internal_validation" and
                split["gene_overlap_with_discovery"] == [] and
                split["sequence_overlap_with_discovery"] == [] and
                split["historically_or_sealed_chromosomes_present"] == [] and
                split["matched_cohort_audit"]["status"] == "PREFLIGHT_PASS" and
                target_manifest["validation_loaded"] is True,
                "full validation split/provenance audit failed", failures)

    report = {
        "schema": "head_context_atlas.mechanism_validation_audit/1",
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": "independent structural/numerical audit; no mechanism claim",
        "smoke": args.smoke,
        "validation_loaded": not args.smoke,
        "package_sha256": sha256_file(package_path),
        "phase_reports": phase_reports,
        "n_failures": len(failures),
        "failures": failures,
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
