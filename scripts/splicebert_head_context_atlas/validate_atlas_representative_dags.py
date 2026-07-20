#!/usr/bin/env python
"""Independent fail-closed audit of representative DAG/evidence bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ALLOWED = {"screening", "exact_component", "QK", "V_OV", "MLP_path", "negative_control", "remainder"}
ATOL = 1e-10


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


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def close(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= ATOL
    return a == b


def audit_remainder(graph: dict, label: str, failures: list[str]) -> None:
    values = [abs(float(row.get("screeningAttribution", 0))) for row in graph["edges"]]
    rem = [
        abs(float(row.get("screeningAttribution", 0))) for row in graph["edges"]
        if row.get("validationStatus") == "remainder"
    ]
    total = sum(values)
    expected = {
        "absolute_screening_mass": total,
        "hidden_remainder_absolute_mass": sum(rem),
        "hidden_remainder_fraction": 0 if total == 0 else sum(rem) / total,
    }
    for key, value in expected.items():
        require(close(graph["remainderReport"][key], value),
                f"{label}: remainder {key} mismatch", failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-audit", required=True)
    parser.add_argument("--analysis-prefix", required=True)
    parser.add_argument("--analysis-audit", default=None)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    bundle_path = Path(args.bundle)
    manifest_path = bundle_path.with_suffix(".manifest.json")
    package_path = Path(args.package)
    package_audit_path = Path(args.package_audit)
    analysis = Path(args.analysis_prefix)
    analysis_audit_path = Path(args.analysis_audit) if args.analysis_audit else Path(f"{analysis}.audit.json")
    audit_path = Path(args.audit) if args.audit else bundle_path.with_suffix(".audit.json")
    if audit_path.exists():
        raise FileExistsError(f"refuse to overwrite immutable DAG audit {audit_path}")
    bundle = json.loads(bundle_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    package = json.loads(package_path.read_text())
    package_audit = json.loads(package_audit_path.read_text())
    analysis_audit = json.loads(analysis_audit_path.read_text())
    relations = json.loads(Path(f"{analysis}.relations.json").read_text())["relations"]
    families = read_jsonl(Path(f"{analysis}.families.jsonl"))
    failures = []
    require(package_audit["status"] == "PASS" and package_audit["n_failures"] == 0 and
            package_audit["package_sha256"] == sha256_file(package_path),
            "package audit is stale or failed", failures)
    require(analysis_audit["status"] == "PASS" and analysis_audit["n_failures"] == 0 and
            analysis_audit["smoke"] is args.smoke,
            "analysis audit is stale, failed, or wrong boundary", failures)
    expected_status = "SMOKE_NOT_EXPORTABLE" if args.smoke else "VALIDATED_EXPORT_CANDIDATE"
    require(bundle["schema"] == "head_context_atlas.representative_dag_bundle/1" and
            bundle["status"] == expected_status and bundle["export_eligible"] is (not args.smoke),
            "bundle status/export boundary mismatch", failures)
    require(set(bundle["evidence_labels"]) == ALLOWED,
            "evidence-label vocabulary mismatch", failures)
    require(finite(bundle), "bundle contains non-finite values", failures)
    relation_ids = {row["candidate_id"] for row in relations}
    package_ids = {row["candidate_id"] for row in package["candidates"]}
    observed_ids = {row["candidate_id"] for row in bundle["representatives"]}
    if args.smoke:
        require(len(observed_ids) == 2 and all("smoke_stage2" in item for item in observed_ids),
                "smoke representative candidates drifted", failures)
    else:
        require(observed_ids == relation_ids == package_ids,
                "bundle includes an unvalidated or omits a validated candidate", failures)
    require(bundle["n_representatives"] == len(bundle["representatives"]),
            "representative count mismatch", failures)

    family_map = {
        (row["candidate_id"], row["target_role"], row["contrast"]): row for row in families
    }
    package_map = {row["candidate_id"]: row for row in package["candidates"]}
    for index, rep in enumerate(bundle["representatives"]):
        label = f"representative[{index}]"
        require(len(rep["sequence"]) == 1024 and len(rep["matched_negative_sequence"]) == 1024,
                f"{label}: sequence length mismatch", failures)
        require(rep["case_packet"]["masked_native_base"] == "G" and
                rep["matched_negative_packet"]["masked_native_base"] == "G",
                f"{label}: corrected mask-base contract failed", failures)
        family = family_map.get((rep["candidate_id"], rep["target_role"], rep["contrast"]))
        require(family is not None, f"{label}: no audited family", failures)
        for graph_name in ("screening_graph", "matched_negative_screening_graph"):
            graph = rep[graph_name]
            node_ids = {row["id"] for row in graph["nodes"]}
            require(len(node_ids) == len(graph["nodes"]), f"{label}: duplicate graph nodes", failures)
            require(all(row["sourceNodeId"] in node_ids and row["targetNodeId"] in node_ids
                        for row in graph["edges"]), f"{label}: dangling graph edge", failures)
            require(all(row["evidenceLabel"] in ALLOWED for row in graph["edges"]),
                    f"{label}: illegal edge evidence label", failures)
            if graph_name == "screening_graph":
                require(all(row["evidenceLabel"] in {"screening", "remainder"}
                            for row in graph["edges"]),
                        f"{label}: case screening edge overclaimed", failures)
            else:
                require(all(row["evidenceLabel"] in {"negative_control", "remainder"}
                            for row in graph["edges"]),
                        f"{label}: negative-control edge mislabeled", failures)
            audit_remainder(graph, f"{label}|{graph_name}", failures)
        gates = rep["correctness_gates"]
        require(gates["case"]["pass"] is True and gates["matched_negative"]["pass"] is True,
                f"{label}: screening correctness gate failed", failures)
        overlay = rep["screening_graph"]["populationEvidence"]
        require(overlay["routeValueLabel"] == family["conditional_labels"]["route_value_label"] and
                overlay["pathLabels"] == family["conditional_labels"]["path_labels"],
                f"{label}: conditional-label overlay drift", failures)
        member_map = {row["member"]: row for row in family["members"]}
        for edge in overlay["edges"]:
            require(edge["evidenceLabel"] in ALLOWED - {"screening", "remainder"},
                    f"{label}: overlay edge has weak/illegal label", failures)
            if edge["evidenceLabel"] == "QK":
                source = member_map["candidate_QK"]
            elif edge["evidenceLabel"] == "V_OV":
                source = member_map["candidate_V_OV"]
            elif edge["evidenceLabel"] == "MLP_path":
                layer = edge["id"].rsplit("mlp-", 1)[-1]
                source = member_map.get(f"MLP_recovery_difference_L{layer}")
                if source is None:
                    source = member_map.get(
                        "upstream_attention_to_candidate_vs_control_mlp_recovery"
                    )
            elif edge["id"] == "exact-group-output":
                source = member_map["exact_group_effect"]
            else:
                source = None  # whole-component row is checked by full package provenance upstream
            if source is not None:
                require(close(edge["signedContribution"], source["mean"]) and
                        close(edge["simultaneousCI"], source["ci_fwer_max_t"]),
                        f"{label}: overlay numerical evidence drift", failures)
        interchange = rep["A_matched_control_state_interchange"]
        if rep["estimand"] == "A" and not rep["component_id"].endswith("MLP"):
            require(interchange is not None and
                    interchange["inference"] == "representative_only_descriptive_no_population_label" and
                    all(len(values) == 1024 for values in interchange["maps"].values()),
                    f"{label}: A representative interchange absent/malformed", failures)
        else:
            require(interchange is None, f"{label}: illegal population-like interchange", failures)

    provenance = bundle["provenance"]
    require(provenance["package_sha256"] == sha256_file(package_path) and
            provenance["package_audit_sha256"] == sha256_file(package_audit_path) and
            provenance["analysis_audit_sha256"] == sha256_file(analysis_audit_path),
            "bundle provenance drift", failures)
    require(manifest["output_sha256"] == sha256_file(bundle_path) and
            manifest["export_eligible"] is (not args.smoke),
            "DAG manifest hash/export boundary mismatch", failures)
    report = {
        "schema": "head_context_atlas.representative_dag_audit/1",
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": "representative DAG structural/evidence audit; no new mechanism claim",
        "smoke": args.smoke, "export_eligible": not args.smoke,
        "n_representatives": len(bundle["representatives"]),
        "n_failures": len(failures), "failures": failures,
        "bundle_sha256": sha256_file(bundle_path),
        "auditor_sha256": sha256_file(Path(__file__)),
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
