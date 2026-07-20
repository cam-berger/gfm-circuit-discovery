#!/usr/bin/env python
"""Build post-audit representative screening DAGs with validated evidence overlays.

Population labels come only from the independently audited stage-two analysis.
Activation×Jacobian graph edges remain screening evidence.  A-estimand QK/V state
interchanges are representative-only descriptive maps, never population estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/splicebert"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_segment_attribution_bundle import run_segment  # noqa: E402
from gfmcd.splicing.circuit import load_adapter, pick_device  # noqa: E402
from run_all_head_screen import ScreenModel  # noqa: E402
from run_atlas_model_preflight import encode_masks, target_scores  # noqa: E402
from run_atlas_stage2_mechanisms import (  # noqa: E402
    build_smoke_candidates, capture_qkv_layers, parse_head, patch_head_qkv,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evidence_edge(edge_id: str, source: str, target: str, evidence: str,
                  member: dict, *, detail: str) -> dict:
    return {
        "id": edge_id, "sourceNodeId": source, "targetNodeId": target,
        "evidenceLabel": evidence, "signedContribution": member["mean"],
        "widthValue": abs(member["mean"]), "simultaneousCI": member["ci_fwer_max_t"],
        "classification": member["classification"], "detail": detail,
    }


def evidence_overlay(candidate: dict, family: dict, whole: dict | None) -> dict:
    members = {row["member"]: row for row in family["members"]}
    nodes = [
        {"id": "partner-state", "kind": "state", "label": "Partner-sensitive state"},
        {"id": "candidate", "kind": candidate["component_type"],
         "label": candidate["component_id"]},
        {"id": "exact-group", "kind": "component-group",
         "label": candidate["exact_group_id"]},
        {"id": "output", "kind": "metric", "label": "Masked-base margin"},
    ]
    edges = []
    exact = members["exact_group_effect"]
    edges.append(evidence_edge(
        "exact-group-output", "exact-group", "output", "exact_component", exact,
        detail="Exact joint intervention on the frozen candidate group.",
    ))
    if whole is not None:
        edges.append({
            "id": "candidate-output", "sourceNodeId": "candidate", "targetNodeId": "output",
            "evidenceLabel": "exact_component", "signedContribution": whole["mean"],
            "widthValue": abs(whole["mean"]), "simultaneousCI": whole["ci_fwer_bonferroni"],
            "classification": whole["classification"],
            "detail": "Untouched whole-component validation endpoint.",
        })
    route_label = family["conditional_labels"]["route_value_label"]
    if route_label in {"partner_sensitive_routing", "mixed_partner_sensitive"}:
        edges.append(evidence_edge(
            "partner-qk-candidate", "partner-state", "candidate", "QK",
            members["candidate_QK"], detail="Population QK main effect; not a position-specific edge.",
        ))
    if route_label in {"partner_sensitive_value_write", "mixed_partner_sensitive"}:
        edges.append(evidence_edge(
            "partner-v-candidate", "partner-state", "candidate", "V_OV",
            members["candidate_V_OV"], detail="Population V/OV main effect; not binding.",
        ))
    for path_member in family["conditional_labels"]["path_labels"]:
        layer = path_member.rsplit("L", 1)[-1] if path_member.startswith("MLP_recovery") else None
        node_id = f"mlp-{layer or candidate['component_layer']}"
        nodes.append({"id": node_id, "kind": "mlp", "label": f"L{layer}MLP" if layer else "MLP path"})
        source = "candidate" if candidate["component_type"] == "attention_head" else "partner-state"
        member_name = (path_member if path_member in members else
                       "upstream_attention_to_candidate_vs_control_mlp_recovery")
        edges.append(evidence_edge(
            f"path-{node_id}", source, node_id, "MLP_path", members[member_name],
            detail="Candidate-minus-importance-control recovery; overlapping causal cut.",
        ))
    negative = []
    for name in ("control_QK", "control_V_OV"):
        if name in members:
            negative.append({
                "member": name, "evidenceLabel": "negative_control", **members[name],
            })
    total = sum(abs(float(edge["signedContribution"])) for edge in edges)
    for edge in edges:
        edge["widthFraction"] = 0.0 if total == 0 else edge["widthValue"] / total
    instrument_limits = []
    population_qk_v = candidate["mechanism"].get("population_qk_v")
    if population_qk_v is not None and not population_qk_v.get("enabled", False):
        instrument_limits.append(population_qk_v["reason"])
    upstream_path = candidate["mechanism"].get("upstream_attention_path")
    if upstream_path is not None and not upstream_path.get("enabled", False):
        instrument_limits.append(upstream_path["instrument_limit"])
    return {
        "nodes": nodes, "edges": edges, "negativeControls": negative,
        "routeValueLabel": route_label, "pathLabels": family["conditional_labels"]["path_labels"],
        "instrumentLimits": instrument_limits,
        "claimBoundary": "component-level audited evidence overlay; screening graph supplies positions",
    }


def remainder_report(graph: dict) -> dict:
    values = [abs(float(row.get("screeningAttribution", 0.0))) for row in graph["edges"]]
    remainder = [
        abs(float(row.get("screeningAttribution", 0.0))) for row in graph["edges"]
        if row.get("validationStatus") == "remainder"
    ]
    total = sum(values)
    return {
        "absolute_screening_mass": total,
        "hidden_remainder_absolute_mass": sum(remainder),
        "hidden_remainder_fraction": 0.0 if total == 0 else sum(remainder) / total,
        "note": "Not a completeness fraction; signed activation×Jacobian edges are non-additive.",
    }


def annotations(packet: dict) -> list[dict]:
    target = packet["target_anchor_win"]
    partner = packet["partner_anchor_win"]
    if packet["target_role"] == "donor":
        target_start, target_end, partner_start, partner_end = target, target + 2, partner - 2, partner
        target_type, partner_type = "donor", "acceptor"
    else:
        target_start, target_end, partner_start, partner_end = target - 2, target, partner, partner + 2
        target_type, partner_type = "acceptor", "donor"
    return [
        {"id": "target-site", "type": target_type, "label": f"Target {target_type}",
         "start": target_start, "end": target_end, "strand": packet["strand"]},
        {"id": "partner-site", "type": partner_type, "label": f"Partner {partner_type}",
         "start": partner_start, "end": partner_end, "strand": packet["strand"]},
    ]


def state_interchange(model: ScreenModel, target_packet: dict, target_sequence: str,
                      source_packet: dict, source_sequence: str, component: str) -> dict:
    layer, head = parse_head(component)
    target_enc = encode_masks(model, [target_sequence], [[target_packet["masked_base_win"]]])
    source_enc = encode_masks(model, [source_sequence], [[source_packet["masked_base_win"]]])
    native_logits = model.logits(target_enc)
    native_margin = float(target_scores(model, native_logits, [target_packet])[0]["margin"])
    source_qkv = capture_qkv_layers(model, source_enc, {layer})
    positions = torch.arange(target_enc["input_ids"].shape[1], device=model.a.device)[None]
    effects = {}
    for label, names in {"QK": ("query", "key"), "V_OV": ("value",)}.items():
        with patch_head_qkv(model.a, component, source_qkv, positions, names):
            logits = model.logits(target_enc)
        patched = float(target_scores(model, logits, [target_packet])[0]["margin"])
        effects[label] = {
            "patched_margin_minus_native": patched - native_margin,
            "native_margin": native_margin, "patched_margin": patched,
        }

    with torch.no_grad():
        target_raw = model.a.model(**target_enc, output_attentions=True, return_dict=True)
        source_raw = model.a.model(**source_enc, output_attentions=True, return_dict=True)
    q = target_packet["masked_base_win"] + model.a.prefix_tokens
    start = model.a.prefix_tokens
    end = start + len(target_sequence)
    native_attention = target_raw.attentions[layer][0, head, q, start:end].float()
    source_attention = source_raw.attentions[layer][0, head, q, start:end].float()
    source_value = source_qkv[(layer, "value")].view(
        1, source_qkv[(layer, "value")].shape[1], model.a.num_heads, model.a.head_dim
    )[0, start:end, head]
    dense = model.a.encoder_layers[layer].attention.output.dense.weight
    head_slice = slice(head * model.a.head_dim, (head + 1) * model.a.head_dim)
    ov = source_value @ dense[:, head_slice].T
    v_ov_norm = native_attention[:, None] * ov
    return {
        "status": "matched_control_state_interchange",
        "inference": "representative_only_descriptive_no_population_label",
        "target_packet_id": target_packet["packet_id"],
        "source_control_packet_id": source_packet["packet_id"],
        "component_id": component, "layer": layer, "head": head,
        "effects": effects,
        "maps": {
            "target_native_attention": native_attention.cpu().tolist(),
            "source_QK_attention_at_target_absolute_query": source_attention.cpu().tolist(),
            "V_OV_write_norm_under_target_attention": v_ov_norm.norm(dim=1).cpu().tolist(),
        },
        "warning": "Cross-sequence all-token state transplant; localization only, not A decomposition.",
    }


def choose_smoke_representatives(pairs: list[dict], packets: dict[str, dict]) -> dict[str, tuple[str, str]]:
    result = {}
    for role in ("donor", "acceptor"):
        block = [row for row in pairs if row["target_role"] == role]
        selected = min(block, key=lambda row: (stable_rank("dag-smoke", role, row["pair_id"]), row["pair_id"]))
        result[role] = selected["case_packet_id"], selected["control_packet_id"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-audit", required=True)
    parser.add_argument("--analysis-prefix", required=True)
    parser.add_argument("--analysis-audit", default=None)
    parser.add_argument("--stage2-run-dir", required=True)
    parser.add_argument("--cohort-prefix", required=True)
    parser.add_argument("--model", default=str(
        REPO / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
    ))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    torch.manual_seed(0)
    np.random.seed(0)

    package_path = Path(args.package)
    package_audit_path = Path(args.package_audit)
    analysis = Path(args.analysis_prefix)
    analysis_audit_path = Path(args.analysis_audit) if args.analysis_audit else Path(f"{analysis}.audit.json")
    output = Path(args.output)
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError("refuse to overwrite immutable DAG output")
    package = json.loads(package_path.read_text())
    package_audit = json.loads(package_audit_path.read_text())
    analysis_audit = json.loads(analysis_audit_path.read_text())
    summary = json.loads(Path(f"{analysis}.summary.json").read_text())
    relations = json.loads(Path(f"{analysis}.relations.json").read_text())["relations"]
    families = read_jsonl(Path(f"{analysis}.families.jsonl"))
    if package_audit["status"] != "PASS" or package_audit["package_sha256"] != sha256_file(package_path):
        raise RuntimeError("stage-two package audit is not current")
    if analysis_audit["status"] != "PASS" or analysis_audit["n_failures"]:
        raise RuntimeError("stage-two analysis audit did not pass")
    expected = "SMOKE" if args.smoke else "STAGE2_CLASSIFIED"
    if summary["status"] != expected or analysis_audit["smoke"] is not args.smoke:
        raise RuntimeError("DAG smoke/full boundary mismatch")
    run_manifest = json.loads((Path(args.stage2_run_dir) / "run_manifest.target.json").read_text())
    candidates = run_manifest["candidates"] if args.smoke else package["candidates"]
    if not args.smoke and {row["candidate_id"] for row in candidates} != {
        row["candidate_id"] for row in relations
    }:
        raise RuntimeError("validated relation grid does not equal DAG candidate grid")

    cohort = Path(args.cohort_prefix)
    packet_path = Path(f"{cohort}.discovery.packets.jsonl")
    sequence_path = Path(f"{cohort}.discovery.sequences.jsonl")
    pair_path = Path(f"{cohort}.discovery.pairs.jsonl")
    cohort_audit_path = cohort.with_suffix(".audit.json")
    cohort_audit = json.loads(cohort_audit_path.read_text())
    if cohort_audit["status"] != "PREFLIGHT_PASS" or cohort_audit["n_failures"]:
        raise RuntimeError("discovery cohort audit did not pass")
    packets = {row["packet_id"]: row for row in read_jsonl(packet_path)}
    sequences = {row["sequence_id"]: row["sequence"] for row in read_jsonl(sequence_path)}
    pairs = read_jsonl(pair_path)
    smoke_reps = choose_smoke_representatives(pairs, packets) if args.smoke else {}

    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=True)
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    model = ScreenModel(adapter, batch_size=1)
    family_map = {
        (row["candidate_id"], row["target_role"], row["contrast"]): row for row in families
    }
    whole_map = {
        (candidate["candidate_id"], row["target_role"], row.get("contrast")): row
        for candidate in candidates for row in candidate["validated_whole_endpoints"]
        if row.get("validated_whole_endpoint")
    }
    representatives = []
    selected_candidates = candidates
    if args.smoke:
        selected_candidates = [
            next(row for row in candidates if row["component_type"] == "attention_head" and row["estimand"] == "A"),
            next(row for row in candidates if row["component_type"] == "attention_head" and row["estimand"] == "B"),
        ]
    for candidate in selected_candidates:
        endpoint_rows = [
            row for row in run_manifest["endpoints"] if row["candidate_id"] == candidate["candidate_id"]
        ]
        if args.smoke:
            endpoint_rows = [row for row in endpoint_rows if row["target_role"] == "donor"]
        for endpoint in endpoint_rows:
            role = endpoint["target_role"]
            if args.smoke:
                case_id, control_id = smoke_reps[role]
            elif candidate["kind"] == "general":
                rep = candidate["representative"][role]
                case_id = rep["packet_id"]
                control_id = rep["matched_negative_context_packet_id"]
            else:
                rep = candidate["representative"]
                case_id, control_id = rep["case_packet_id"], rep["control_packet_id"]
            case, control = packets[case_id], packets[control_id]
            case_seq, control_seq = sequences[case["sequence_id"]], sequences[control["sequence_id"]]
            segment = run_segment(
                adapter, case_seq, f"{candidate['candidate_id']}|{role}",
                case["masked_base_win"], case["masked_base_win"] + 1,
                heads_per_position=2, sources_per_head=3, max_frontier=6,
            )
            graph = segment["graphs"][0]
            for edge in graph["edges"]:
                edge["evidenceLabel"] = edge.get("validationStatus", "screening")
            graph["remainderReport"] = remainder_report(graph)
            graph["populationEvidence"] = evidence_overlay(
                candidate, family_map[(candidate["candidate_id"], role, endpoint["contrast"])],
                whole_map.get((candidate["candidate_id"], role, endpoint["contrast"])),
            )
            relevant_heads = {candidate["component_id"]}
            path = candidate["mechanism"].get("upstream_attention_path")
            if path:
                relevant_heads.update(path["attention_group"])
            traces = [
                row for row in segment["traces"]
                if f"L{row['layer']}H{row['head']}" in relevant_heads
            ]
            control_segment = run_segment(
                adapter, control_seq, f"{candidate['candidate_id']}|{role}|negative-control",
                control["masked_base_win"], control["masked_base_win"] + 1,
                heads_per_position=2, sources_per_head=3, max_frontier=6,
            )
            control_graph = control_segment["graphs"][0]
            for edge in control_graph["edges"]:
                edge["evidenceLabel"] = (
                    "remainder" if edge.get("validationStatus") == "remainder" else "negative_control"
                )
                edge["measurementStatus"] = "screening"
            control_graph["remainderReport"] = remainder_report(control_graph)
            interchange = None
            if candidate["component_type"] == "attention_head" and candidate["estimand"] == "A":
                interchange = state_interchange(
                    model, case, case_seq, control, control_seq, candidate["component_id"]
                )
            representatives.append({
                "candidate_id": candidate["candidate_id"], "component_id": candidate["component_id"],
                "estimand": candidate["estimand"], "target_role": role,
                "contrast": endpoint["contrast"], "case_packet": case,
                "matched_negative_packet": control, "sequence": case_seq,
                "annotations": annotations(case), "screening_graph": graph,
                "matched_negative_sequence": control_seq,
                "matched_negative_annotations": annotations(control),
                "matched_negative_screening_graph": control_graph,
                "candidate_attention_traces": traces,
                "A_matched_control_state_interchange": interchange,
                "correctness_gates": {
                    "case": segment["gates"], "matched_negative": control_segment["gates"]
                },
            })

    bundle = {
        "schema": "head_context_atlas.representative_dag_bundle/1",
        "status": "SMOKE_NOT_EXPORTABLE" if args.smoke else "VALIDATED_EXPORT_CANDIDATE",
        "export_eligible": not args.smoke,
        "claim_boundary": (
            "screening DAG plus audited component-level overlays; not a feature-complete attribution graph"
        ),
        "model": {"id": "SpliceBERT.1024nt", "checkpoint_sha256": sha256_file(
            Path(args.model) / "pytorch_model.bin"
        )},
        "n_representatives": len(representatives), "representatives": representatives,
        "evidence_labels": package["dag_export"]["evidence_labels"],
        "provenance": {
            "package_sha256": sha256_file(package_path),
            "package_audit_sha256": sha256_file(package_audit_path),
            "analysis_summary_sha256": sha256_file(Path(f"{analysis}.summary.json")),
            "analysis_audit_sha256": sha256_file(analysis_audit_path),
            "cohort_audit_sha256": sha256_file(cohort_audit_path),
            "packet_sha256": sha256_file(packet_path), "sequence_sha256": sha256_file(sequence_path),
            "pairs_sha256": sha256_file(pair_path), "exporter_sha256": sha256_file(Path(__file__)),
            "screening_exporter_sha256": sha256_file(
                REPO / "scripts/splicebert/export_segment_attribution_bundle.py"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "head_context_atlas.representative_dag_manifest/1",
        "status": bundle["status"], "output": str(output),
        "output_sha256": sha256_file(output), "export_eligible": bundle["export_eligible"],
        "inputs": bundle["provenance"], "device": str(device),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": bundle["status"], "n_representatives": len(representatives),
        "export_eligible": bundle["export_eligible"],
    }, indent=2))


if __name__ == "__main__":
    main()
