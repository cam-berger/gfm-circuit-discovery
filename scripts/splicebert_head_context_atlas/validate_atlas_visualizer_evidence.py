#!/usr/bin/env python3
"""Independently audit the visualizer-facing Atlas evidence envelope."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_EVIDENCE_LABELS = [
    "screening", "exact_component", "QK", "V_OV", "MLP_path",
    "negative_control", "remainder",
]
EXPECTED_RELATIONS = {
    ("general|A|L1H12", "donor"), ("general|A|L1H12", "acceptor"),
    ("general|A|L2H15", "donor"), ("general|A|L2H15", "acceptor"),
    ("general|A|L3H1", "donor"), ("general|A|L3H1", "acceptor"),
    ("general|B|L2MLP", "donor"), ("general|B|L2MLP", "acceptor"),
}
EXPECTED_TRACE_METRICS = {"target-log-probability", "output-entropy"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def reaches_root(node_id: str, root_id: str, outgoing: dict[str, list[str]]) -> bool:
    stack = [node_id]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == root_id:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(outgoing.get(current, []))
    return False


def graph_failures(graph: dict[str, Any], prefix: str, require_population: bool) -> list[str]:
    failures: list[str] = []
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = [node.get("id") for node in nodes]
    node_set = set(node_ids)
    root = graph.get("rootNodeId")
    require(graph.get("source") == "measured", f"{prefix}: graph is not measured", failures)
    require(len(node_ids) == len(node_set), f"{prefix}: duplicate node ids", failures)
    require(root in node_set, f"{prefix}: missing root node", failures)
    outgoing: dict[str, list[str]] = {}
    for edge in edges:
        source = edge.get("sourceNodeId")
        target = edge.get("targetNodeId")
        require(source in node_set and target in node_set, f"{prefix}: dangling edge", failures)
        require(edge.get("validationStatus") in {"screening", "remainder"},
                f"{prefix}: screening edge evidence was upgraded", failures)
        permitted_labels = {"screening", "remainder"} if require_population else {
            "negative_control", "remainder",
        }
        require(edge.get("evidenceLabel") in permitted_labels,
                f"{prefix}: screening/control evidence label drift", failures)
        outgoing.setdefault(source, []).append(target)
    for node_id in node_set:
        require(reaches_root(node_id, root, outgoing), f"{prefix}: node does not reach root", failures)
    remainder = graph.get("remainderReport", {})
    fraction = remainder.get("hidden_remainder_fraction")
    require(isinstance(fraction, (int, float)) and 0 <= fraction <= 1,
            f"{prefix}: invalid hidden remainder fraction", failures)
    metrics = graph.get("metrics", [])
    metric_ids = {metric.get("id") for metric in metrics}
    require(metric_ids == EXPECTED_TRACE_METRICS,
            f"{prefix}: validated trace-metric set drift", failures)
    for edge in edges:
        effects = edge.get("metricEffects", {})
        require(set(effects) == metric_ids, f"{prefix}: incomplete edge metric effects", failures)
    population = graph.get("populationEvidence")
    if require_population:
        require(isinstance(population, dict), f"{prefix}: missing population evidence", failures)
        if isinstance(population, dict):
            population_node_ids = {node.get("id") for node in population.get("nodes", [])}
            for edge in population.get("edges", []):
                interval = edge.get("simultaneousCI")
                require(edge.get("evidenceLabel") in set(EXPECTED_EVIDENCE_LABELS) - {"screening", "remainder"},
                        f"{prefix}: invalid population evidence label", failures)
                require(edge.get("sourceNodeId") in population_node_ids
                        and edge.get("targetNodeId") in population_node_ids,
                        f"{prefix}: dangling population edge", failures)
                require(isinstance(interval, list) and len(interval) == 2
                        and interval[0] <= interval[1],
                        f"{prefix}: invalid simultaneous interval", failures)
    else:
        require(population is None, f"{prefix}: matched negative carries population evidence", failures)
    return failures


def representative_failures(representative: dict[str, Any]) -> list[str]:
    key = f"{representative.get('candidate_id')}|{representative.get('target_role')}"
    failures: list[str] = []
    for variant, seq_key, packet_key, graph_key, annotations_key, population in [
        ("case", "sequence", "case_packet", "screening_graph", "annotations", True),
        ("negative", "matched_negative_sequence", "matched_negative_packet",
         "matched_negative_screening_graph", "matched_negative_annotations", False),
    ]:
        sequence = representative.get(seq_key, "")
        packet = representative.get(packet_key, {})
        graph = representative.get(graph_key, {})
        prefix = f"{key}:{variant}"
        require(len(sequence) == 1024 and set(sequence) <= set("ACGT"),
                f"{prefix}: invalid sequence", failures)
        require(packet.get("window_end", 0) - packet.get("window_start", 0) == 1024,
                f"{prefix}: packet/window length mismatch", failures)
        require(packet.get("target_role") == representative.get("target_role"),
                f"{prefix}: target-role mismatch", failures)
        target = graph.get("targetPosition")
        require(target == packet.get("masked_base_win") and isinstance(target, int) and 0 <= target < 1024,
                f"{prefix}: target coordinate mismatch", failures)
        masked = graph.get("maskedSegment", {})
        require(masked.get("start") <= target < masked.get("end", -1),
                f"{prefix}: target outside joint mask", failures)
        for annotation in representative.get(annotations_key, []):
            require(0 <= annotation.get("start", -1) < annotation.get("end", -1) <= 1024,
                    f"{prefix}: annotation out of bounds", failures)
        failures.extend(graph_failures(graph, prefix, population))
    gates = representative.get("correctness_gates", {})
    require(gates.get("case", {}).get("pass") is True
            and gates.get("matched_negative", {}).get("pass") is True,
            f"{key}: correctness gate failed", failures)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--audit", default=None)
    args = parser.parse_args()
    evidence_path = Path(args.evidence)
    source_path = Path(args.source_bundle)
    source_audit_path = Path(args.source_audit)
    source_manifest_path = Path(args.source_manifest)
    manifest_path = evidence_path.with_suffix(".manifest.json")
    output_audit = Path(args.audit) if args.audit else evidence_path.with_suffix(".audit.json")
    if output_audit.exists():
        raise FileExistsError("refuse to overwrite immutable visualizer audit")

    evidence = json.loads(evidence_path.read_text())
    source = json.loads(source_path.read_text())
    source_audit = json.loads(source_audit_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []
    source_sha = sha256_file(source_path)
    source_audit_sha = sha256_file(source_audit_path)
    source_manifest_sha = sha256_file(source_manifest_path)
    require(
        evidence.get("schema") == "gfm_visualizer.atlas_evidence/1"
        and evidence.get("status") == "AUDITED_VALIDATED"
        and evidence.get("export_eligible") is True,
        "visualizer evidence boundary mismatch", failures,
    )
    require(
        source.get("schema") == "head_context_atlas.representative_dag_bundle/1"
        and source.get("export_eligible") is True
        and source.get("status") == "VALIDATED_EXPORT_CANDIDATE",
        "source bundle is not exportable", failures,
    )
    require(
        source_audit.get("schema") == "head_context_atlas.representative_dag_audit/1"
        and source_audit.get("status") == "PASS"
        and source_audit.get("n_failures") == 0
        and source_audit.get("failures") == []
        and source_audit.get("smoke") is False
        and source_audit.get("export_eligible") is True
        and source_audit.get("bundle_sha256") == source_sha,
        "source DAG audit is failed or stale", failures,
    )
    require(
        source_manifest.get("schema") == "head_context_atlas.representative_dag_manifest/1"
        and source_manifest.get("status") == "VALIDATED_EXPORT_CANDIDATE"
        and source_manifest.get("export_eligible") is True
        and source_manifest.get("output_sha256") == source_sha,
        "source DAG manifest is failed or stale", failures,
    )
    require(evidence.get("representatives") == source.get("representatives"),
            "representative payload drift", failures)
    require(evidence.get("evidence_labels") == EXPECTED_EVIDENCE_LABELS
            and evidence.get("evidence_labels") == source.get("evidence_labels"),
            "evidence vocabulary drift", failures)
    relations = {
        (representative.get("candidate_id"), representative.get("target_role"))
        for representative in evidence.get("representatives", [])
    }
    require(relations == EXPECTED_RELATIONS, "validated relation grid drift", failures)
    for representative in evidence.get("representatives", []):
        failures.extend(representative_failures(representative))
    require(
        evidence.get("audit") == {
            "status": "PASS", "n_failures": 0, "export_eligible": True,
            "bundle_sha256": source_sha,
            "source_audit_sha256": source_audit_sha,
            "source_manifest_sha256": source_manifest_sha,
        },
        "embedded audit boundary drift", failures,
    )
    provenance = evidence.get("provenance", {})
    require(
        provenance.get("source_bundle_sha256") == source_sha
        and provenance.get("source_audit_sha256") == source_audit_sha
        and provenance.get("source_manifest_sha256") == source_manifest_sha,
        "source provenance mismatch", failures,
    )
    require(finite(evidence), "non-finite visualizer evidence", failures)
    require(
        manifest.get("schema") == "gfm_visualizer.atlas_evidence_manifest/1"
        and manifest.get("status") == "AUDITED_VALIDATED"
        and manifest.get("export_eligible") is True
        and manifest.get("output_sha256") == sha256_file(evidence_path)
        and manifest.get("output_bytes") == evidence_path.stat().st_size
        and manifest.get("source_bundle_sha256") == source_sha
        and manifest.get("source_audit_sha256") == source_audit_sha
        and manifest.get("source_manifest_sha256") == source_manifest_sha
        and manifest.get("n_representatives") == len(source.get("representatives", []))
        and manifest.get("evidence_labels") == EXPECTED_EVIDENCE_LABELS,
        "visualizer manifest mismatch", failures,
    )
    report = {
        "schema": "gfm_visualizer.atlas_evidence_audit/1",
        "status": "PASS" if not failures else "FAIL",
        "export_eligible": not failures,
        "n_representatives": len(evidence.get("representatives", [])),
        "n_failures": len(failures),
        "failures": failures,
        "evidence_sha256": sha256_file(evidence_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_bundle_sha256": source_sha,
        "source_audit_sha256": source_audit_sha,
        "source_manifest_sha256": source_manifest_sha,
        "auditor_sha256": sha256_file(Path(__file__)),
        "claim_boundary": "visualization evidence transport audit; no new mechanism claim",
    }
    output_audit.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
