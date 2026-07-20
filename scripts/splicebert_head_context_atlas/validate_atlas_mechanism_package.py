#!/usr/bin/env python
"""Independent sealing audit for a candidate-specific mechanism package."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MECHANISM_AMENDMENT = REPO / "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A1.md"
MECHANISM_AMENDMENT_A2 = REPO / "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A2.md"


def resolve_reference(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def ctype(component: str) -> str:
    return "mlp_write" if component.endswith("MLP") else "attention_head"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--analysis-prefix", required=True)
    parser.add_argument("--merged-prefix", required=True)
    parser.add_argument("--audit", default=None)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    package_path = Path(args.package)
    manifest_path = package_path.with_suffix(".manifest.json")
    analysis = Path(args.analysis_prefix)
    merged = Path(args.merged_prefix)
    audit_path = Path(args.audit) if args.audit else package_path.with_suffix(".audit.json")
    if audit_path.exists():
        raise FileExistsError(f"refuse to overwrite immutable audit {audit_path}")
    package = json.loads(package_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    shortlist = json.loads(Path(f"{analysis}.shortlist.json").read_text())
    summary = json.loads(Path(f"{analysis}.summary.json").read_text())
    analysis_audit = json.loads(Path(f"{analysis}.audit.json").read_text())
    pairs = read_jsonl(Path(f"{merged}.pairs.jsonl.gz"))
    failures: list[str] = []

    smoke = summary["status"] == "SMOKE"
    require(package["validation_loaded"] is False, "package validation_loaded is not false", failures)
    require(analysis_audit["status"] == "PASS" and analysis_audit["n_failures"] == 0,
            "source analysis audit is not a zero-failure pass", failures)
    require(package["status"] == (
        "SMOKE_NO_CANDIDATES" if smoke else "FROZEN_BEFORE_VALIDATION_ACCESS"
    ), "package status mismatch", failures)

    expected_general = [] if smoke else shortlist["general"]["selected"]
    expected_context = [] if smoke else shortlist["context"]["selected"]
    expected_ids = {
        f"general|{row['estimand']}|{row['component_id']}" for row in expected_general
    } | {
        f"context|{row['contrast']}|{row['estimand']}|{row['component_id']}"
        for row in expected_context
    }
    candidates = package["candidates"]
    candidate_ids = {candidate["candidate_id"] for candidate in candidates}
    require(len(candidate_ids) == len(candidates) == package["n_candidates"],
            "candidate IDs/count are not unique/consistent", failures)
    require(candidate_ids == expected_ids, "package candidates do not equal frozen shortlist", failures)
    selected_components = {candidate["component_id"] for candidate in candidates}
    for candidate in candidates:
        label = candidate["candidate_id"]
        require(candidate["component_type"] == ctype(candidate["component_id"]),
                f"{label}: component type mismatch", failures)
        control = candidate["importance_control"]
        require(control["component_id"] not in selected_components,
                f"{label}: control is itself shortlisted", failures)
        require(control["component_type"] == candidate["component_type"],
                f"{label}: control type mismatch", failures)
        require(control["absolute_match_gap"] >= 0,
                f"{label}: negative importance gap", failures)
        interventions = candidate["interventions"]
        expected_source = "zero_component_output" if candidate["estimand"] == "A" else "partner_destroyed_same_packet"
        require(interventions["whole_component"] == expected_source,
                f"{label}: whole source mismatch", failures)
        if candidate["component_type"] == "attention_head":
            qkv = interventions["qk_v"]
            require(qkv["conditions"] == ["QK", "V", "QKV"],
                    f"{label}: QK/V condition grid mismatch", failures)
            if candidate["estimand"] == "A":
                require(qkv["source"] == "matched_control_packet" and "not a decomposition" in qkv["interpretation"],
                        f"{label}: A interchange semantics drift", failures)
            else:
                require(qkv["source"] == "partner_destroyed_same_packet",
                        f"{label}: B QK/V source mismatch", failures)

    expected_groups: dict[str, list[str]] = {}
    for candidate in candidates:
        key = (
            f"general|{candidate['estimand']}"
            if candidate["kind"] == "general"
            else f"context|{candidate['contrast']}|{candidate['estimand']}"
        )
        expected_groups.setdefault(key, [])
        if candidate["component_id"] not in expected_groups[key]:
            expected_groups[key].append(candidate["component_id"])
    require(package["exact_groups"] == dict(sorted(expected_groups.items())),
            "exact group membership mismatch", failures)

    pair_by_id = {pair["pair_id"]: pair for pair in pairs}
    for candidate in candidates:
        rep = candidate["representative"]
        if candidate["kind"] == "context":
            pair = pair_by_id.get(rep["pair_id"])
            require(pair is not None, f"{candidate['candidate_id']}: representative pair absent", failures)
            if pair:
                require(rep["case_packet_id"] == pair["case_packet_id"] and
                        rep["control_packet_id"] == pair["control_packet_id"],
                        f"{candidate['candidate_id']}: representative join mismatch", failures)
        else:
            require(set(rep) == {"donor", "acceptor"},
                    f"{candidate['candidate_id']}: general representative roles mismatch", failures)
            for role, record in rep.items():
                pair = pair_by_id.get(record["matched_pair_id"])
                require(pair is not None and pair["case_packet_id"] == record["packet_id"] and
                        pair["control_packet_id"] == record["matched_negative_context_packet_id"] and
                        pair["target_role"] == role,
                        f"{candidate['candidate_id']}: {role} representative join mismatch", failures)

    require(package["positive_control"]["component_id"] == "L3H1" and
            package["positive_control"]["estimand"] == "B",
            "positive control drift", failures)
    require(package["validation_inference"]["tau_nats"] == 0.05,
            "validation tau drift", failures)
    require("Holm-FWER" in package["validation_inference"]["whole_component_family"],
            "whole validation correction drift", failures)
    amendment = package.get("documents", {}).get("mechanism_amendment", {})
    require(amendment.get("sha256") == sha256_file(MECHANISM_AMENDMENT),
            "mechanism-validation amendment hash drift", failures)
    amendment_a2 = package.get("documents", {}).get("mechanism_amendment_a2", {})
    require(amendment_a2.get("sha256") == sha256_file(MECHANISM_AMENDMENT_A2),
            "mechanism-validation amendment A2 hash drift", failures)

    opened_inputs = {str(resolve_reference(record["path"])) for record in manifest["inputs"]}
    require(all("internal_validation" not in path and ".validation." not in path
                for path in opened_inputs),
            "manifest opened internal-validation data", failures)
    sealed = package["sealed_validation_references_not_opened"]
    require(set(sealed) == {"pairs", "packets", "sequences"},
            "sealed validation reference grid mismatch", failures)
    for kind, path in sealed.items():
        require(path.endswith(f".validation.{kind}.jsonl"),
                f"sealed {kind} reference does not name the physical validation artifact", failures)
        require(str(resolve_reference(path)) not in opened_inputs,
                f"sealed validation reference was opened: {path}", failures)
    require(manifest["config"]["validation_loaded"] is False,
            "manifest validation_loaded is not false", failures)
    output = next((record for record in manifest["outputs"]
                   if str(Path(record["path"]).resolve()) == str(package_path.resolve())), None)
    require(output is not None, "manifest lacks mechanism package output", failures)
    if output:
        require(output["sha256"] == sha256_file(package_path) and output["size_bytes"] == package_path.stat().st_size,
                "manifest package hash/size mismatch", failures)

    report = {
        "schema": "head_context_atlas.mechanism_package_audit/1",
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": "package sealing audit; no validation outcome or mechanism claim",
        "source_analysis_status": summary["status"],
        "n_candidates": len(candidates),
        "n_groups": len(package["exact_groups"]),
        "validation_loaded": False,
        "n_failures": len(failures),
        "failures": failures,
        "package_sha256": sha256_file(package_path),
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
