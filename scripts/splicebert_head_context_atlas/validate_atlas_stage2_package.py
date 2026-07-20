#!/usr/bin/env python
"""Independent audit of the conditional QK/OV/MLP/path package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def kind(component: str) -> str:
    return "mlp_write" if component.endswith("MLP") else "attention_head"


def layer(component: str) -> int:
    marker = "MLP" if component.endswith("MLP") else "H"
    return int(component[1:].split(marker, 1)[0])


def group_id(candidate: dict) -> str:
    return (
        f"general|{candidate['estimand']}" if candidate["kind"] == "general"
        else f"context|{candidate['contrast']}|{candidate['estimand']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--whole-analysis-prefix", required=True)
    parser.add_argument("--stage1-package", required=True)
    parser.add_argument("--audit", default=None)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    package_path = Path(args.package)
    manifest_path = package_path.with_suffix(".manifest.json")
    analysis = Path(args.whole_analysis_prefix)
    stage1_path = Path(args.stage1_package)
    audit_path = Path(args.audit) if args.audit else package_path.with_suffix(".audit.json")
    if audit_path.exists():
        raise FileExistsError(f"refuse to overwrite immutable audit {audit_path}")
    package = json.loads(package_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(Path(f"{analysis}.summary.json").read_text())
    relations = json.loads(Path(f"{analysis}.candidates.json").read_text())["relations"]
    endpoints = []
    with Path(f"{analysis}.candidate_endpoints.jsonl").open() as handle:
        endpoints = [json.loads(line) for line in handle if line.strip()]
    whole_audit = json.loads(Path(f"{analysis}.audit.json").read_text())
    whole_manifest = json.loads(Path(f"{analysis}.manifest.json").read_text())
    stage1 = json.loads(stage1_path.read_text())
    failures: list[str] = []
    smoke = summary["status"] == "SMOKE"

    require(package["stage2_outcomes_loaded"] is False,
            "stage-two outcomes-loaded flag is not false", failures)
    require(whole_audit["status"] == "PASS" and whole_audit["n_failures"] == 0,
            "whole analysis audit is not a zero-failure pass", failures)
    expected_ids = [] if smoke else [
        row["candidate_id"] for row in relations if row["advance_to_qk_v_path"]
    ]
    require(package["validated_candidate_ids"] == expected_ids,
            "validated candidate list drift", failures)
    require(package["n_candidates"] == len(package["candidates"]) == len(expected_ids),
            "candidate count mismatch", failures)
    expected_status = (
        "SMOKE_NO_STAGE2_CANDIDATES" if smoke else
        "NO_VALIDATED_CANDIDATES" if not expected_ids else
        "FROZEN_AFTER_WHOLE_VALIDATION_BEFORE_STAGE2_FORWARD"
    )
    require(package["status"] == expected_status, "package status mismatch", failures)
    require(package["whole_validation_loaded"] is (not smoke),
            "whole-validation access flag mismatch", failures)

    stage1_candidates = {row["candidate_id"]: row for row in stage1["candidates"]}
    relation_map = {row["candidate_id"]: row for row in relations}
    endpoint_map = {
        candidate_id: [row for row in endpoints if row["candidate_id"] == candidate_id]
        for candidate_id in expected_ids
    }
    validated = [stage1_candidates[candidate_id] for candidate_id in expected_ids]
    observed = {row["candidate_id"]: row for row in package["candidates"]}
    require(len(observed) == len(package["candidates"]), "duplicate candidate IDs", failures)
    for source in validated:
        candidate_id = source["candidate_id"]
        row = observed.get(candidate_id)
        require(row is not None, f"missing stage-two candidate {candidate_id}", failures)
        if row is None:
            continue
        gid = group_id(source)
        require(row["component_id"] == source["component_id"] and
                row["estimand"] == source["estimand"] and row["exact_group_id"] == gid,
                f"{candidate_id}: source identity/group drift", failures)
        require(row["exact_group_components"] == stage1["exact_groups"][gid] and
                source["component_id"] in row["exact_group_components"],
                f"{candidate_id}: exact group mismatch", failures)
        require(row["importance_control"] == source["importance_control"] and
                row["representative"] == source["representative"],
                f"{candidate_id}: control/representative drift", failures)
        require(row["validated_whole_relation"] == relation_map[candidate_id] and
                row["validated_whole_endpoints"] == endpoint_map[candidate_id] and
                endpoint_map[candidate_id] and
                all(item["validated_whole_endpoint"] for item in endpoint_map[candidate_id]),
                f"{candidate_id}: whole validation evidence mismatch", failures)
        mechanism = row["mechanism"]
        if kind(source["component_id"]) == "attention_head":
            expected_layers = list(range(layer(source["component_id"]), 6))
            require(mechanism["downstream_mlp_recovery"]["layers"] == expected_layers,
                    f"{candidate_id}: downstream MLP enumeration mismatch", failures)
            if source["estimand"] == "B":
                qkv = mechanism["population_qk_v"]
                require(qkv["enabled"] is True and qkv["source"] == "partner_destroyed_same_packet"
                        and qkv["conditions"] == ["QK", "V_OV", "QKV_BRIDGE"]
                        and qkv["qkv_bridge_is_not_a_family_member"] is True
                        and mechanism["representative_state_interchange"] is None,
                        f"{candidate_id}: B QK/V contract drift", failures)
            else:
                require(mechanism["population_qk_v"]["enabled"] is False and
                        mechanism["representative_state_interchange"]["inference"] ==
                        "descriptive_only_no_population_label",
                        f"{candidate_id}: A non-decomposition boundary drift", failures)
        else:
            expected_heads = sorted({
                other["component_id"] for other in validated
                if kind(other["component_id"]) == "attention_head"
                and group_id(other) == gid
                and layer(other["component_id"]) <= layer(source["component_id"])
            }, key=lambda item: (layer(item), int(item.split("H", 1)[1])))
            path = mechanism["upstream_attention_path"]
            require(path["attention_group"] == expected_heads and
                    path["enabled"] is bool(expected_heads),
                    f"{candidate_id}: upstream attention group mismatch", failures)
            require((path["instrument_limit"] is None) is bool(expected_heads),
                    f"{candidate_id}: upstream instrument-limit mismatch", failures)

    require(package["inference"]["critical_value"] ==
            "Gaussian max-|t|, 200000 draws, seed=0" and
            package["inference"]["alpha"] == 0.05 and package["inference"]["tau_nats"] == 0.05,
            "stage-two simultaneous-family contract drift", failures)
    require(package["dag_export"]["visualizer_boundary"] ==
            "validated and independently audited relations only",
            "visualizer boundary drift", failures)
    require(package["validation_references_already_opened_for_whole_stage"] ==
            stage1["sealed_validation_references_not_opened"],
            "validation reference drift", failures)
    for kind_, suffix in {
        "target": ".target_rows.jsonl.gz",
        "health": ".health_rows.jsonl.gz",
        "pairs": ".pairs.jsonl.gz",
    }.items():
        matches = [record for record in whole_manifest["inputs"]
                   if str(record["path"]).endswith(suffix)]
        observed_ref = package["whole_validation_row_references"].get(kind_)
        require(len(matches) == 1 and observed_ref is not None
                and observed_ref["sha256"] == matches[0]["sha256"]
                and observed_ref["size_bytes"] == matches[0]["size_bytes"]
                and str(observed_ref["path"]).endswith(suffix),
                f"whole-validation {kind_} row reference drift", failures)

    opened = {str(Path(record["path"]).resolve()) for record in manifest["inputs"]}
    require(all("stage2" not in path.lower() or path.endswith("MECHANISM_VALIDATION_AMENDMENT_A2.md")
                for path in opened),
            "manifest appears to include a stage-two outcome", failures)
    require(manifest["config"]["stage2_outcomes_loaded"] is False and
            manifest["config"]["n_candidates"] == len(expected_ids),
            "manifest stage-two boundary/count drift", failures)
    output = next((record for record in manifest["outputs"]
                   if str(Path(record["path"]).resolve()) == str(package_path.resolve())), None)
    require(output is not None and output["sha256"] == sha256_file(package_path)
            and output["size_bytes"] == package_path.stat().st_size,
            "manifest package hash/size mismatch", failures)

    report = {
        "schema": "head_context_atlas.stage2_mechanism_package_audit/1",
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": "conditional mechanism package audit; no stage-two outcome or label",
        "whole_validation_loaded": not smoke,
        "stage2_outcomes_loaded": False,
        "n_candidates": len(expected_ids),
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
