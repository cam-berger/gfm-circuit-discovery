#!/usr/bin/env python
"""Freeze QK/OV/MLP/path instructions after whole effects validate.

The builder reads audited whole-effect classifications and the original discovery-
frozen candidate package.  It does not read any stage-two mechanism forward.  A
zero-survivor result is preserved as an empty package rather than forcing a mechanism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

AMENDMENT_A2 = REPO / "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A2.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def portable_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO.resolve()))
    except ValueError:
        return str(resolved)


def component_kind(component: str) -> str:
    return "mlp_write" if component.endswith("MLP") else "attention_head"


def component_layer(component: str) -> int:
    if not component.startswith("L"):
        raise ValueError(component)
    marker = "MLP" if component.endswith("MLP") else "H"
    return int(component[1:].split(marker, 1)[0])


def group_id(candidate: dict) -> str:
    return (
        f"general|{candidate['estimand']}"
        if candidate["kind"] == "general"
        else f"context|{candidate['contrast']}|{candidate['estimand']}"
    )


def stage2_spec(candidate: dict, *, validated_candidates: list[dict],
                exact_groups: dict[str, list[str]]) -> dict:
    """Construct one candidate's outcome-independent conditional mechanism grid."""
    component = candidate["component_id"]
    kind = component_kind(component)
    layer = component_layer(component)
    gid = group_id(candidate)
    if gid not in exact_groups or component not in exact_groups[gid]:
        raise RuntimeError(f"candidate {component} absent from exact group {gid}")
    base = {
        "candidate_id": candidate["candidate_id"],
        "kind": candidate["kind"],
        "component_id": component,
        "component_type": kind,
        "component_layer": layer,
        "estimand": candidate["estimand"],
        "contrast": candidate.get("contrast"),
        "target_role": candidate.get("target_role"),
        "importance_control": candidate["importance_control"],
        "exact_group_id": gid,
        "exact_group_components": exact_groups[gid],
        "representative": candidate["representative"],
    }
    if kind == "attention_head":
        downstream = list(range(layer, 6))
        mechanism = {
            "population_qk_v": (
                {
                    "enabled": True,
                    "source": "partner_destroyed_same_packet",
                    "conditions": ["QK", "V_OV", "QKV_BRIDGE"],
                    "candidate_and_importance_control": True,
                    "label_family_members": [
                        "candidate_QK", "candidate_V_OV", "control_QK", "control_V_OV"
                    ],
                    "qkv_bridge_is_not_a_family_member": True,
                }
                if candidate["estimand"] == "B" else {
                    "enabled": False,
                    "reason": "A zero-output effect has no population QK/V source",
                }
            ),
            "representative_state_interchange": (
                None if candidate["estimand"] == "B" else {
                    "enabled": True,
                    "source": "frozen_discovery_matched_control_representative",
                    "conditions": ["QK", "V_OV"],
                    "inference": "descriptive_only_no_population_label",
                }
            ),
            "downstream_mlp_recovery": {
                "layers": downstream,
                "candidate_and_importance_control": True,
                "family_members": [f"MLP_recovery_difference_L{value}" for value in downstream],
                "effect": "candidate recovery minus importance-control recovery",
            },
        }
    else:
        upstream = sorted({
            other["component_id"]
            for other in validated_candidates
            if component_kind(other["component_id"]) == "attention_head"
            and group_id(other) == gid
            and component_layer(other["component_id"]) <= layer
        }, key=lambda item: (component_layer(item), int(item.split("H", 1)[1])))
        mechanism = {
            "population_qk_v": {"enabled": False, "reason": "candidate is an MLP write"},
            "upstream_attention_path": {
                "enabled": bool(upstream),
                "attention_group": upstream,
                "candidate_mlp": component,
                "importance_control_mlp": candidate["importance_control"]["component_id"],
                "family_members": ["upstream_attention_to_candidate_vs_control_mlp_recovery"]
                if upstream else [],
                "instrument_limit": None if upstream else "upstream_path_instrument_limited",
            },
        }
    base["mechanism"] = mechanism
    family = ["exact_group_effect"]
    if kind == "attention_head":
        if candidate["estimand"] == "B":
            family.extend(mechanism["population_qk_v"]["label_family_members"])
        family.extend(mechanism["downstream_mlp_recovery"]["family_members"])
    else:
        family.extend(mechanism["upstream_attention_path"]["family_members"])
    base["within_endpoint_family_members"] = family
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whole-analysis-prefix", required=True)
    parser.add_argument("--whole-analysis-audit", default=None)
    parser.add_argument("--stage1-package", required=True)
    parser.add_argument("--stage1-package-audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-smoke-empty", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    analysis = Path(args.whole_analysis_prefix)
    analysis_audit_path = (
        Path(args.whole_analysis_audit) if args.whole_analysis_audit
        else Path(f"{analysis}.audit.json")
    )
    stage1_path = Path(args.stage1_package)
    stage1_audit_path = Path(args.stage1_package_audit)
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refuse to overwrite immutable stage-two package {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "whole_summary": Path(f"{analysis}.summary.json"),
        "whole_relations": Path(f"{analysis}.candidates.json"),
        "whole_endpoints": Path(f"{analysis}.candidate_endpoints.jsonl"),
        "whole_analysis_manifest": Path(f"{analysis}.manifest.json"),
        "whole_analysis_audit": analysis_audit_path,
        "stage1_package": stage1_path,
        "stage1_package_audit": stage1_audit_path,
        "amendment_a2": AMENDMENT_A2,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    summary = json.loads(paths["whole_summary"].read_text())
    relations = json.loads(paths["whole_relations"].read_text())["relations"]
    endpoint_rows = read_jsonl(paths["whole_endpoints"])
    whole_audit = json.loads(analysis_audit_path.read_text())
    stage1 = json.loads(stage1_path.read_text())
    stage1_audit = json.loads(stage1_audit_path.read_text())
    whole_manifest = json.loads(paths["whole_analysis_manifest"].read_text())
    smoke = summary["status"] == "SMOKE"
    if smoke and not args.allow_smoke_empty:
        raise RuntimeError("smoke stage-two packaging requires --allow-smoke-empty")
    if not smoke and summary["status"] != "VALIDATION_CLASSIFIED":
        raise RuntimeError("whole validation analysis is not complete")
    if whole_audit["status"] != "PASS" or whole_audit["n_failures"]:
        raise RuntimeError("whole validation analysis audit did not pass")
    if stage1_audit["status"] != "PASS" or stage1_audit["n_failures"]:
        raise RuntimeError("stage-one package audit did not pass")
    if stage1_audit["package_sha256"] != sha256_file(stage1_path):
        raise RuntimeError("stage-one package changed after audit")
    if not smoke and not summary["validation_loaded"]:
        raise RuntimeError("full stage-two package requires completed whole internal validation")

    relation_map = {row["candidate_id"]: row for row in relations}
    validated_ids = [] if smoke else list(summary["validated_candidate_ids"])
    if validated_ids != [
        row["candidate_id"] for row in relations if row["advance_to_qk_v_path"]
    ]:
        raise RuntimeError("validated candidate list disagrees with relation dispositions")
    stage1_candidates = {row["candidate_id"]: row for row in stage1["candidates"]}
    if not set(validated_ids) <= set(stage1_candidates):
        raise RuntimeError("whole validation advanced a candidate absent from stage one")
    validated_candidates = [stage1_candidates[candidate_id] for candidate_id in validated_ids]
    endpoint_by_candidate: dict[str, list[dict]] = {}
    for candidate_id in validated_ids:
        rows = [row for row in endpoint_rows if row["candidate_id"] == candidate_id]
        if not rows or not all(row["validated_whole_endpoint"] for row in rows):
            raise RuntimeError(f"advanced candidate lacks validated endpoint grid: {candidate_id}")
        endpoint_by_candidate[candidate_id] = rows

    candidates = []
    for candidate in validated_candidates:
        spec = stage2_spec(
            candidate, validated_candidates=validated_candidates,
            exact_groups=stage1["exact_groups"],
        )
        spec["validated_whole_relation"] = relation_map[candidate["candidate_id"]]
        spec["validated_whole_endpoints"] = endpoint_by_candidate[candidate["candidate_id"]]
        candidates.append(spec)

    merged_suffixes = {
        "target": ".target_rows.jsonl.gz",
        "health": ".health_rows.jsonl.gz",
        "pairs": ".pairs.jsonl.gz",
    }
    whole_rows = {}
    for kind_, suffix in merged_suffixes.items():
        matches = [
            record for record in whole_manifest["inputs"]
            if str(record["path"]).endswith(suffix)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"whole analysis manifest lacks unique {kind_} row input")
        record = matches[0]
        whole_rows[kind_] = {
            "path": portable_repo_path(Path(record["path"])),
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }

    package = {
        "schema": "head_context_atlas.stage2_mechanism_package/1",
        "status": (
            "SMOKE_NO_STAGE2_CANDIDATES" if smoke
            else "NO_VALIDATED_CANDIDATES" if not candidates
            else "FROZEN_AFTER_WHOLE_VALIDATION_BEFORE_STAGE2_FORWARD"
        ),
        "claim_boundary": "conditional mechanism instructions; no QK/OV/path outcome or label",
        "whole_validation_loaded": not smoke,
        "stage2_outcomes_loaded": False,
        "n_candidates": len(candidates),
        "validated_candidate_ids": validated_ids,
        "candidates": candidates,
        "inference": {
            "family": "within each validated whole endpoint",
            "critical_value": "Gaussian max-|t|, 200000 draws, seed=0",
            "covariance": "same gene-clustered or unique-dyad two-endpoint estimator as whole endpoint",
            "alpha": 0.05,
            "tau_nats": 0.05,
            "negative_eigenvalue_policy": "clip numerical negatives to zero and report",
        },
        "dag_export": {
            "representatives": "stage-one discovery-frozen representatives only",
            "evidence_labels": [
                "screening", "exact_component", "QK", "V_OV", "MLP_path",
                "negative_control", "remainder",
            ],
            "visualizer_boundary": "validated and independently audited relations only",
        },
        "validation_references_already_opened_for_whole_stage": stage1[
            "sealed_validation_references_not_opened"
        ],
        "whole_validation_row_references": whole_rows,
        "documents": {
            "amendment_a2": {"path": str(AMENDMENT_A2), "sha256": sha256_file(AMENDMENT_A2)},
        },
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
    }
    out.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")

    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_stage2_mechanism_package_v1",
        command=["python"] + sys.argv,
        status=package["status"],
    )
    manifest.config.update({
        "whole_validation_loaded": not smoke,
        "stage2_outcomes_loaded": False,
        "n_candidates": len(candidates),
    })
    for path in paths.values():
        manifest.add_input(str(path))
    manifest.add_output(str(out))
    manifest.write(str(manifest_path))
    print(json.dumps({
        "status": package["status"], "n_candidates": len(candidates),
        "stage2_outcomes_loaded": False, "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
