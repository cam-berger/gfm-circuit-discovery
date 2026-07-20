#!/usr/bin/env python
"""Apply the frozen whole-effect validation inference to audited merged rows."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_atlas_discovery import Z_POWER_80, dyadic_stats, gene_cluster_stats  # noqa: E402


TAU = 0.05
ALPHA = 0.05
AMENDMENT = REPO / "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A1.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def holm_adjust(pvalues: list[float] | np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    m = len(pvalues)
    if m == 0:
        return pvalues.copy()
    order = np.argsort(pvalues, kind="mergesort")
    scaled = np.asarray([(m - rank) * pvalues[index] for rank, index in enumerate(order)])
    adjusted_ordered = np.maximum.accumulate(scaled)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.clip(adjusted_ordered, 0.0, 1.0)
    return adjusted


def scalar_stats(stats: dict) -> dict:
    return {
        "mean": float(stats["point"][0]),
        "se": float(stats["se"][0]),
        "ci95_marginal": [float(stats["low"][0]), float(stats["high"][0])],
        "p_marginal_two_sided": float(stats["p"][0]),
        "mde80_marginal": float(stats["mde80"][0]),
    }


def gene_estimate(rows: list[dict], value_field: str) -> dict:
    if not rows:
        raise RuntimeError("empty gene-cluster endpoint")
    values = np.asarray([[float(row[value_field])] for row in rows], dtype=float)
    result = scalar_stats(gene_cluster_stats(values, [row["gene_id"] for row in rows]))
    result.update({
        "inference": "one_way_gene_clustered",
        "n_packets": len(rows),
        "n_genes": len({row["gene_id"] for row in rows}),
        "mean_abs_effect": float(np.mean(np.abs(values[:, 0]))),
    })
    return result


def context_estimate(rows: list[dict], pairs: list[dict], contrast: str,
                     value_field: str) -> dict:
    category, target_role = contrast.split("|", 1)
    block = [
        pair for pair in pairs
        if pair["category"] == category and pair["target_role"] == target_role
    ]
    values = {row["packet_id"]: float(row[value_field]) for row in rows}
    if len(values) != len(rows):
        raise RuntimeError(f"duplicate packet endpoint rows for {contrast}")
    missing = [
        pair["pair_id"] for pair in block
        if pair["case_packet_id"] not in values or pair["control_packet_id"] not in values
    ]
    if missing:
        raise RuntimeError(f"endpoint cannot join {len(missing)} pairs for {contrast}")
    differences = np.asarray([[
        values[pair["case_packet_id"]] - values[pair["control_packet_id"]]
    ] for pair in block])
    reuse = Counter(pair["control_packet_id"] for pair in block)
    weights = np.asarray([1.0 / reuse[pair["control_packet_id"]] for pair in block])
    stats = dyadic_stats(
        differences,
        [pair["case_gene_id"] for pair in block],
        [pair["control_gene_id"] for pair in block],
        weights,
    )
    result = scalar_stats(stats)
    result.update({
        "inference": "inverse_reuse_weighted_unique_dyad_two_endpoint_gene_sandwich",
        "contrast": contrast,
        "n_pairs": len(block),
        "n_endpoint_genes": int(stats["n_endpoint_genes"]),
        "n_effective_pairs": float(stats["n_effective"]),
        "total_weight": float(stats["total_weight"]),
        "mean_abs_pair_difference": float(np.average(np.abs(differences[:, 0]), weights=weights)),
    })
    return result


def estimate(rows: list[dict], pairs: list[dict], *, scope_kind: str,
             scope_value: str, target_role: str | None, value_field: str) -> dict:
    if scope_kind == "contrast":
        return context_estimate(rows, pairs, scope_value, value_field)
    role = target_role if target_role is not None else scope_value
    block = [row for row in rows if row["target_role"] == role]
    return gene_estimate(block, value_field)


def discovery_direction(candidate: dict, target_role: str | None) -> int:
    if candidate["kind"] == "general":
        value = float(candidate["discovery"][f"{target_role}_mean"])
    else:
        value = float(candidate["discovery"]["mean_difference"])
    if value == 0:
        raise RuntimeError(f"frozen discovery direction is exactly zero for {candidate['candidate_id']}")
    return 1 if value > 0 else -1


def classify(point: float, low: float, high: float, p_holm: float,
             expected_direction: int) -> tuple[str, bool]:
    direction_match = point * expected_direction > 0
    if p_holm <= ALPHA and low > TAU:
        label = "material_support"
    elif p_holm <= ALPHA and high < -TAU:
        label = "material_suppress"
    elif p_holm <= ALPHA and (low > 0 or high < 0):
        label = "detected_subfloor"
    elif low >= -TAU and high <= TAU:
        label = "bounded_small"
    else:
        label = "inconclusive"
    return label, direction_match


def health_annotation(health_by_endpoint: dict[str, list[dict]], endpoint_id: str,
                      target_role: str) -> dict:
    block = [
        row for row in health_by_endpoint.get(endpoint_id, [])
        if row["target_role"] == target_role
    ]
    estimate_ = gene_estimate(block, "health_margin_damage")
    return {
        "target_role": target_role,
        "mean_health_margin_damage": estimate_["mean"],
        "ci95_health_margin_damage": estimate_["ci95_marginal"],
        "n_health_packets": estimate_["n_packets"],
        "n_health_genes": estimate_["n_genes"],
        "broad_MLM_flag": (
            estimate_["ci95_marginal"][0] > TAU
            or estimate_["ci95_marginal"][1] < -TAU
        ),
    }


def candidate_specs(package: dict, target_rows: list[dict], *, smoke: bool) -> list[dict]:
    if not smoke:
        return package["candidates"]
    seen = {}
    for row in target_rows:
        if row["endpoint_role"] != "candidate":
            continue
        seen.setdefault(row["candidate_id"], {
            "candidate_id": row["candidate_id"],
            "kind": "general",
            "estimand": row["estimand"],
            "component_id": row["component_ids"][0],
            "discovery": {"donor_mean": 1.0, "acceptor_mean": 1.0},
        })
    return [seen[key] for key in sorted(seen)]


def diagnostic_specs(target_rows: list[dict]) -> list[dict]:
    specs = {}
    for row in target_rows:
        if row["endpoint_role"] == "candidate":
            continue
        key = row["endpoint_id"]
        specs.setdefault(key, {
            field: row[field] for field in (
                "endpoint_id", "endpoint_role", "candidate_id", "group_id", "estimand",
                "component_ids", "scope_kind", "scope_value", "intervention",
            )
        })
    return [specs[key] for key in sorted(specs)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-prefix", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-audit", required=True)
    parser.add_argument("--validation-audit", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    merged = Path(args.merged_prefix)
    package_path = Path(args.package)
    package_audit_path = Path(args.package_audit)
    validation_audit_path = Path(args.validation_audit)
    output_prefix = Path(args.prefix)
    paths = {
        "target": Path(f"{merged}.target_rows.jsonl.gz"),
        "health": Path(f"{merged}.health_rows.jsonl.gz"),
        "pairs": Path(f"{merged}.pairs.jsonl.gz"),
        "merge_summary": Path(f"{merged}.summary.json"),
        "merge_manifest": Path(f"{merged}.manifest.json"),
        "package": package_path,
        "package_audit": package_audit_path,
        "validation_audit": validation_audit_path,
        "amendment": AMENDMENT,
    }
    outputs = {
        "endpoints": Path(f"{output_prefix}.candidate_endpoints.jsonl"),
        "candidates": Path(f"{output_prefix}.candidates.json"),
        "diagnostics": Path(f"{output_prefix}.diagnostics.jsonl"),
        "summary": Path(f"{output_prefix}.summary.json"),
        "manifest": Path(f"{output_prefix}.manifest.json"),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"refuse to overwrite immutable analysis output {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    package = json.loads(package_path.read_text())
    package_audit = json.loads(package_audit_path.read_text())
    validation_audit = json.loads(validation_audit_path.read_text())
    merge_summary = json.loads(paths["merge_summary"].read_text())
    if package_audit["status"] != "PASS" or package_audit["n_failures"]:
        raise RuntimeError("package audit did not pass")
    if package_audit["package_sha256"] != sha256_file(package_path):
        raise RuntimeError("package changed after audit")
    if validation_audit["status"] != "PASS" or validation_audit["n_failures"]:
        raise RuntimeError("mechanism-validation audit did not pass")
    if validation_audit["smoke"] is not args.smoke or merge_summary["smoke"] is not args.smoke:
        raise RuntimeError("smoke/full boundary mismatch")

    target_rows = read_jsonl_gz(paths["target"])
    health_rows = read_jsonl_gz(paths["health"])
    pairs = read_jsonl_gz(paths["pairs"])
    target_by_endpoint: dict[str, list[dict]] = defaultdict(list)
    health_by_endpoint: dict[str, list[dict]] = defaultdict(list)
    for row in target_rows:
        target_by_endpoint[row["endpoint_id"]].append(row)
    for row in health_rows:
        health_by_endpoint[row["endpoint_id"]].append(row)

    candidates = candidate_specs(package, target_rows, smoke=args.smoke)
    endpoint_rows = []
    for candidate in candidates:
        endpoint_id = (
            f"smoke|candidate|{candidate['estimand']}|{candidate['component_id']}"
            if args.smoke else f"candidate|{candidate['candidate_id']}"
        )
        source_rows = target_by_endpoint[endpoint_id]
        if not source_rows:
            raise RuntimeError(f"candidate endpoint absent: {endpoint_id}")
        if candidate["kind"] == "general":
            endpoint_roles = ("donor", "acceptor")
            for role in endpoint_roles:
                estimate_ = estimate(
                    source_rows, pairs, scope_kind="general", scope_value="all",
                    target_role=role, value_field="effect_margin",
                )
                endpoint_rows.append({
                    "candidate_id": candidate["candidate_id"],
                    "endpoint_id": endpoint_id,
                    "kind": "general",
                    "component_id": candidate["component_id"],
                    "component_type": "mlp_write" if candidate["component_id"].endswith("MLP") else "attention_head",
                    "estimand": candidate["estimand"],
                    "target_role": role,
                    "contrast": None,
                    "expected_discovery_direction": discovery_direction(candidate, role),
                    **estimate_,
                    "masked_health": health_annotation(health_by_endpoint, endpoint_id, role),
                })
        else:
            contrast = candidate["contrast"]
            role = candidate["target_role"]
            estimate_ = estimate(
                source_rows, pairs, scope_kind="contrast", scope_value=contrast,
                target_role=role, value_field="effect_margin",
            )
            endpoint_rows.append({
                "candidate_id": candidate["candidate_id"],
                "endpoint_id": endpoint_id,
                "kind": "context",
                "component_id": candidate["component_id"],
                "component_type": "mlp_write" if candidate["component_id"].endswith("MLP") else "attention_head",
                "estimand": candidate["estimand"],
                "target_role": role,
                "contrast": contrast,
                "expected_discovery_direction": discovery_direction(candidate, None),
                **estimate_,
                "masked_health": health_annotation(health_by_endpoint, endpoint_id, role),
            })

    m = len(endpoint_rows)
    if m:
        adjusted = holm_adjust([row["p_marginal_two_sided"] for row in endpoint_rows])
        z_simultaneous = float(norm.ppf(1.0 - ALPHA / (2.0 * m)))
        for row, p_holm in zip(endpoint_rows, adjusted):
            half_width = z_simultaneous * row["se"]
            low, high = row["mean"] - half_width, row["mean"] + half_width
            family_mde = (z_simultaneous + float(norm.ppf(0.80))) * row["se"]
            label, direction_match = classify(
                row["mean"], low, high, float(p_holm), row["expected_discovery_direction"]
            )
            row.update({
                "p_holm_whole_family": float(p_holm),
                "ci_fwer_bonferroni": [float(low), float(high)],
                "simultaneous_interval_method": f"Bonferroni 1-{ALPHA}/m, m={m}",
                "mde80_familywise": float(family_mde),
                "tau": TAU,
                "classification": label,
                "direction_matches_discovery": direction_match,
                "validated_whole_endpoint": bool(
                    direction_match and label in {"material_support", "material_suppress"}
                ),
                "bounded_null_licensed": bool(
                    label == "bounded_small" and family_mde <= TAU
                ),
            })

    endpoint_by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in endpoint_rows:
        endpoint_by_candidate[row["candidate_id"]].append(row)
    relation_rows = []
    for candidate in candidates:
        rows = endpoint_by_candidate[candidate["candidate_id"]]
        expected_count = 2 if candidate["kind"] == "general" else 1
        if len(rows) != expected_count:
            raise RuntimeError(f"candidate endpoint count mismatch: {candidate['candidate_id']}")
        if all(row["validated_whole_endpoint"] for row in rows):
            status = "VALIDATED_WHOLE_EFFECT"
            advance = True
        elif any(
            row["classification"] in {"material_support", "material_suppress", "detected_subfloor"}
            and not row["direction_matches_discovery"] for row in rows
        ):
            status = "RETIRED_DIRECTION_REVERSAL"
            advance = False
        elif all(row["bounded_null_licensed"] for row in rows):
            status = "RETIRED_BOUNDED_SMALL"
            advance = False
        else:
            status = "NOT_VALIDATED_INCONCLUSIVE"
            advance = False
        relation_rows.append({
            "candidate_id": candidate["candidate_id"],
            "kind": candidate["kind"],
            "component_id": candidate["component_id"],
            "estimand": candidate["estimand"],
            "contrast": candidate.get("contrast"),
            "validation_status": status,
            "advance_to_qk_v_path": advance,
            "endpoint_classifications": {
                row["target_role"]: row["classification"] for row in rows
            },
            "claim_boundary": "whole causal relation only; mechanism unresolved",
        })

    diagnostic_rows = []
    for spec in diagnostic_specs(target_rows):
        source_rows = target_by_endpoint[spec["endpoint_id"]]
        if spec["scope_kind"] == "general":
            roles = ("donor", "acceptor")
        elif spec["scope_kind"] == "target_role":
            roles = (spec["scope_value"],)
        else:
            roles = (spec["scope_value"].split("|")[-1],)
        for role in roles:
            estimate_ = estimate(
                source_rows, pairs, scope_kind=spec["scope_kind"],
                scope_value=spec["scope_value"], target_role=role,
                value_field="effect_margin",
            )
            diagnostic_rows.append({
                **spec,
                "target_role": role,
                "contrast": spec["scope_value"] if spec["scope_kind"] == "contrast" else None,
                **estimate_,
                "masked_health": health_annotation(
                    health_by_endpoint, spec["endpoint_id"], role
                ),
                "multiplicity_status": (
                    "prespecified diagnostic; reserve for later within-endpoint max-|t| family"
                ),
                "claim_boundary": "cannot validate or rescue a candidate whole effect",
            })

    positive_rows = [
        row for row in diagnostic_rows if row["endpoint_role"] == "positive_control"
    ]
    if len(positive_rows) != 1:
        raise RuntimeError("expected one donor L3H1/B positive-control estimate")
    positive_control = {
        **positive_rows[0],
        "passed": positive_rows[0]["ci95_marginal"][0] > 0,
        "pass_rule": "one-way gene-clustered marginal 95% CI lower bound > 0",
    }

    write_jsonl(outputs["endpoints"], endpoint_rows)
    outputs["candidates"].write_text(json.dumps({
        "schema": "head_context_atlas.mechanism_validation_candidates/1",
        "status": "SMOKE" if args.smoke else "VALIDATION_CLASSIFIED",
        "relations": relation_rows,
    }, indent=2, sort_keys=True) + "\n")
    write_jsonl(outputs["diagnostics"], diagnostic_rows)
    summary = {
        "schema": "head_context_atlas.mechanism_validation_analysis/1",
        "status": "SMOKE" if args.smoke else "VALIDATION_CLASSIFIED",
        "claim_boundary": "whole component relations only; no QK/V/path/function claim",
        "smoke": args.smoke,
        "validation_loaded": not args.smoke,
        "tau_nats": TAU,
        "whole_family": {
            "n_endpoints": m,
            "detection": "two-sided Holm-FWER alpha=0.05",
            "simultaneous_intervals": "Bonferroni FWER alpha=0.05 over same endpoints",
        },
        "n_candidates": len(relation_rows),
        "n_validated_candidates": sum(row["advance_to_qk_v_path"] for row in relation_rows),
        "validated_candidate_ids": [
            row["candidate_id"] for row in relation_rows if row["advance_to_qk_v_path"]
        ],
        "candidate_status_counts": dict(Counter(
            row["validation_status"] for row in relation_rows
        )),
        "positive_control": positive_control,
        "next_stage_rule": (
            "build and audit candidate-specific QK/V/MLP/path package only for "
            "validated_candidate_ids; do not export discovery-only relations"
        ),
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_mechanism_validation_analysis_v1",
        command=["python"] + sys.argv,
        status=summary["status"],
    )
    manifest.config.update({
        "tau_nats": TAU, "alpha": ALPHA, "smoke": args.smoke,
        "validation_loaded": not args.smoke,
        "mechanism_amendment_sha256": sha256_file(AMENDMENT),
    })
    for path in paths.values():
        manifest.add_input(str(path))
    for kind in ("endpoints", "candidates", "diagnostics", "summary"):
        manifest.add_output(str(outputs[kind]))
    manifest.write(str(outputs["manifest"]))
    print(json.dumps({
        "status": summary["status"], "n_candidates": len(relation_rows),
        "n_validated_candidates": summary["n_validated_candidates"],
        "positive_control_pass": positive_control["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
