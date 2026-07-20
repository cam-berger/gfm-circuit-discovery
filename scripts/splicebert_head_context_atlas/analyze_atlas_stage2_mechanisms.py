#!/usr/bin/env python
"""Apply frozen joint max-|t| inference to audited stage-two mechanism families."""
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
ALPHA = 0.05
TAU = 0.05
N_MAXT = 200_000
MAXT_SEED = 0
AMENDMENT = REPO / "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A2.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def family_members(candidate: dict, stage_rows: list[dict]) -> list[str]:
    observed = sorted({
        row["member"] for row in stage_rows
        if row["candidate_id"] == candidate["candidate_id"] and row["member_role"] == "family"
    })
    expected = ["exact_group_effect", *observed]
    frozen = candidate.get("within_endpoint_family_members")
    if frozen is not None:
        if set(expected) != set(frozen) or len(expected) != len(frozen):
            raise RuntimeError(f"frozen family mismatch for {candidate['candidate_id']}")
        return list(frozen)
    return expected


def endpoint_sources(candidate: dict, endpoint: dict, members: list[str],
                     stage_rows: list[dict], group_rows: list[dict]) -> dict[str, list[dict]]:
    result = {}
    for member in members:
        if member == "exact_group_effect":
            result[member] = [
                {**row, "value": row["effect_margin"]}
                for row in group_rows if row["group_id"] == candidate["exact_group_id"]
            ]
        else:
            result[member] = [
                row for row in stage_rows
                if row["endpoint_key"] == endpoint["endpoint_key"] and row["member"] == member
            ]
        if not result[member]:
            raise RuntimeError(
                f"empty family source {candidate['candidate_id']}|{endpoint['target_role']}|{member}"
            )
    return result


def matrix_by_packet(sources: dict[str, list[dict]], packet_ids: list[str]) -> np.ndarray:
    columns = []
    for member, rows in sources.items():
        values = {row["packet_id"]: float(row["value"]) for row in rows}
        if len(values) != len(rows) or not set(packet_ids) <= set(values):
            raise RuntimeError(f"family packet grid mismatch for {member}")
        columns.append([values[packet_id] for packet_id in packet_ids])
    return np.asarray(columns, dtype=float).T


def gene_covariance(sources: dict[str, list[dict]], target_role: str) -> tuple[np.ndarray, np.ndarray, dict]:
    anchor = next(iter(sources.values()))
    packets = sorted(
        {row["packet_id"] for row in anchor if row["target_role"] == target_role}
    )
    if not packets:
        raise RuntimeError(f"empty gene endpoint for {target_role}")
    matrix = matrix_by_packet(sources, packets)
    row_lookup = {row["packet_id"]: row for row in anchor}
    genes = [row_lookup[packet_id]["gene_id"] for packet_id in packets]
    point = matrix.mean(axis=0)
    residual = matrix - point
    unique_genes = sorted(set(genes))
    index = {gene: i for i, gene in enumerate(unique_genes)}
    scores = np.zeros((len(unique_genes), matrix.shape[1]), dtype=float)
    np.add.at(scores, [index[gene] for gene in genes], residual)
    correction = len(unique_genes) / max(len(unique_genes) - 1, 1)
    covariance = correction * scores.T @ scores / (len(packets) ** 2)
    meta = {
        "inference": "joint_one_way_gene_clustered",
        "n_packets": len(packets), "n_genes": len(unique_genes),
    }
    return point, covariance, meta


def context_covariance(sources: dict[str, list[dict]], pairs: list[dict],
                       contrast: str) -> tuple[np.ndarray, np.ndarray, dict]:
    category, role = contrast.split("|", 1)
    block = [
        row for row in pairs if row["category"] == category and row["target_role"] == role
    ]
    if not block:
        raise RuntimeError(f"empty pair endpoint for {contrast}")
    packet_ids = sorted({
        row[field] for row in block for field in ("case_packet_id", "control_packet_id")
    })
    packet_matrix = matrix_by_packet(sources, packet_ids)
    lookup = {packet_id: packet_matrix[i] for i, packet_id in enumerate(packet_ids)}
    differences = np.asarray([
        lookup[row["case_packet_id"]] - lookup[row["control_packet_id"]]
        for row in block
    ])
    reuse = Counter(row["control_packet_id"] for row in block)
    weights = np.asarray([1.0 / reuse[row["control_packet_id"]] for row in block])
    total = float(weights.sum())
    point = np.sum(weights[:, None] * differences, axis=0) / total
    pair_scores = weights[:, None] * (differences - point)
    case_genes = [row["case_gene_id"] for row in block]
    control_genes = [row["control_gene_id"] for row in block]
    dyads = [tuple(sorted((a, b))) for a, b in zip(case_genes, control_genes)]
    if len(dyads) != len(set(dyads)):
        raise RuntimeError(f"repeated endpoint dyad in {contrast}")
    genes = sorted(set(case_genes) | set(control_genes))
    index = {gene: i for i, gene in enumerate(genes)}
    incident = np.zeros((len(genes), differences.shape[1]), dtype=float)
    np.add.at(incident, [index[gene] for gene in case_genes], pair_scores)
    np.add.at(incident, [index[gene] for gene in control_genes], pair_scores)
    meat = incident.T @ incident - pair_scores.T @ pair_scores
    correction = len(genes) / max(len(genes) - 1, 1)
    covariance = correction * meat / (total ** 2)
    meta = {
        "inference": "joint_inverse_reuse_weighted_unique_dyad_two_endpoint_gene_sandwich",
        "contrast": contrast, "n_pairs": len(block), "n_endpoint_genes": len(genes),
        "n_effective_pairs": float(total ** 2 / np.square(weights).sum()),
        "total_weight": total,
    }
    return point, covariance, meta


def marginal_health(rows: list[dict], role: str) -> dict:
    block = [row for row in rows if row["target_role"] == role]
    if not block:
        raise RuntimeError(f"empty health role {role}")
    values = np.asarray([float(row["value"]) for row in block])
    point = float(values.mean())
    residual = values - point
    genes = sorted({row["gene_id"] for row in block})
    index = {gene: i for i, gene in enumerate(genes)}
    scores = np.zeros(len(genes), dtype=float)
    np.add.at(scores, [index[row["gene_id"]] for row in block], residual)
    correction = len(genes) / max(len(genes) - 1, 1)
    se = math.sqrt(max(correction * float(scores @ scores), 0.0)) / len(block)
    low, high = point - 1.96 * se, point + 1.96 * se
    return {
        "mean_health_margin_damage": point, "se_health": se,
        "ci95_health_margin_damage": [low, high], "n_health_packets": len(block),
        "n_health_genes": len(genes), "broad_MLM_flag": low > TAU or high < -TAU,
    }


def health_sources(candidate: dict, endpoint: dict, members: list[str],
                   stage_health: list[dict], group_health: list[dict]) -> dict[str, list[dict]]:
    result = {}
    for member in members:
        if member == "exact_group_effect":
            result[member] = [
                {**row, "value": row["health_margin_damage"]}
                for row in group_health if row["group_id"] == candidate["exact_group_id"]
            ]
        else:
            result[member] = [
                row for row in stage_health
                if row["endpoint_key"] == endpoint["endpoint_key"] and row["member"] == member
            ]
    return result


def max_t(point: np.ndarray, covariance: np.ndarray) -> dict:
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.clip(eigenvalues, 0.0, None)
    covariance_psd = (eigenvectors * clipped) @ eigenvectors.T
    se = np.sqrt(np.clip(np.diag(covariance_psd), 0.0, None))
    correlation = np.zeros_like(covariance_psd)
    nonzero = se > 0
    correlation[np.ix_(nonzero, nonzero)] = (
        covariance_psd[np.ix_(nonzero, nonzero)] / np.outer(se[nonzero], se[nonzero])
    )
    correlation[nonzero, nonzero] = 1.0
    corr_eig, corr_vec = np.linalg.eigh((correlation + correlation.T) / 2.0)
    corr_sqrt = (corr_vec * np.sqrt(np.clip(corr_eig, 0.0, None))) @ corr_vec.T
    rng = np.random.default_rng(MAXT_SEED)
    maxima = np.empty(N_MAXT, dtype=float)
    cursor = 0
    while cursor < N_MAXT:
        n = min(20_000, N_MAXT - cursor)
        draws = rng.standard_normal((n, len(point))) @ corr_sqrt.T
        maxima[cursor:cursor + n] = np.max(np.abs(draws), axis=1)
        cursor += n
    critical = float(np.quantile(maxima, 1.0 - ALPHA, method="higher"))
    t = np.divide(point, se, out=np.where(point == 0, 0.0, np.sign(point) * np.inf), where=se > 0)
    adjusted = np.asarray([
        (1 + int(np.count_nonzero(maxima >= abs(value)))) / (N_MAXT + 1)
        for value in t
    ])
    return {
        "covariance": covariance_psd, "se": se, "t": t,
        "critical": critical, "p_max_t": adjusted,
        "min_raw_eigenvalue": float(eigenvalues.min()),
        "n_negative_eigenvalues_clipped": int(np.count_nonzero(eigenvalues < -1e-12)),
    }


def classify(point: float, low: float, high: float, p: float) -> str:
    if p <= ALPHA and low > TAU:
        return "material_support"
    if p <= ALPHA and high < -TAU:
        return "material_suppress"
    if p <= ALPHA and (low > 0 or high < 0):
        return "detected_subfloor"
    if low >= -TAU and high <= TAU:
        return "bounded_small"
    return "inconclusive"


def conditional_labels(candidate: dict, endpoint: dict, members: list[dict], *, smoke: bool) -> dict:
    """Apply only the categorical mechanism rules frozen in Amendment A2."""
    if smoke:
        return {
            "route_value_label": "SMOKE_NO_LABEL",
            "path_labels": [], "validated_whole_direction": None,
        }
    whole = next(
        row for row in candidate["validated_whole_endpoints"]
        if row["target_role"] == endpoint["target_role"]
        and row.get("contrast") == endpoint.get("contrast")
    )
    if not whole["validated_whole_endpoint"] or abs(float(whole["mean"])) <= TAU:
        raise RuntimeError("stage-two candidate endpoint lacks a material validated whole effect")
    direction = 1 if float(whole["mean"]) > 0 else -1
    lookup = {row["member"]: row for row in members}

    route_value = None
    if candidate["component_type"] == "attention_head" and candidate["estimand"] == "B":
        qk = lookup["candidate_QK"]
        value = lookup["candidate_V_OV"]
        qk_material = qk["classification"] == (
            "material_support" if direction > 0 else "material_suppress"
        )
        value_material = value["classification"] == (
            "material_support" if direction > 0 else "material_suppress"
        )
        qk_bounded = qk["classification"] == "bounded_small"
        value_bounded = value["classification"] == "bounded_small"
        if qk_material and value_bounded:
            route_value = "partner_sensitive_routing"
        elif value_material and qk_bounded:
            route_value = "partner_sensitive_value_write"
        elif qk_material and value_material:
            route_value = "mixed_partner_sensitive"
        else:
            route_value = "mechanism_unresolved"

    path_labels = []
    if candidate["component_type"] == "attention_head":
        path_labels = [
            row["member"] for row in members
            if row["member"].startswith("MLP_recovery_difference_L")
            and row["classification"] == "material_support"
        ]
    else:
        member = lookup.get("upstream_attention_to_candidate_vs_control_mlp_recovery")
        if member is not None and member["classification"] == "material_support":
            path_labels = ["upstream_attention_to_candidate_MLP_path"]
    return {
        "route_value_label": route_value, "path_labels": path_labels,
        "validated_whole_direction": direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-prefix", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-audit", required=True)
    parser.add_argument("--execution-audit", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    merged = Path(args.merged_prefix)
    package_path = Path(args.package)
    package_audit_path = Path(args.package_audit)
    execution_audit_path = Path(args.execution_audit)
    output = Path(args.prefix)
    paths = {
        "target": Path(f"{merged}.target_rows.jsonl.gz"),
        "health": Path(f"{merged}.health_rows.jsonl.gz"),
        "group_target": Path(f"{merged}.whole_group_target_rows.jsonl.gz"),
        "group_health": Path(f"{merged}.whole_group_health_rows.jsonl.gz"),
        "pairs": Path(f"{merged}.pairs.jsonl.gz"),
        "merge_summary": Path(f"{merged}.summary.json"),
        "merge_manifest": Path(f"{merged}.manifest.json"),
        "package": package_path, "package_audit": package_audit_path,
        "execution_audit": execution_audit_path, "amendment_a2": AMENDMENT,
    }
    outputs = {
        "families": Path(f"{output}.families.jsonl"),
        "relations": Path(f"{output}.relations.json"),
        "diagnostics": Path(f"{output}.diagnostics.jsonl"),
        "summary": Path(f"{output}.summary.json"),
        "manifest": Path(f"{output}.manifest.json"),
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
    execution_audit = json.loads(execution_audit_path.read_text())
    merge_summary = json.loads(paths["merge_summary"].read_text())
    if package_audit["status"] != "PASS" or package_audit["n_failures"] or \
            package_audit["package_sha256"] != sha256_file(package_path):
        raise RuntimeError("stage-two package audit did not pass or is stale")
    if execution_audit["status"] != "PASS" or execution_audit["n_failures"]:
        raise RuntimeError("stage-two execution audit did not pass")
    if execution_audit["smoke"] is not args.smoke or merge_summary["smoke"] is not args.smoke:
        raise RuntimeError("smoke/full boundary mismatch")

    stage_target = read_gz(paths["target"])
    stage_health = read_gz(paths["health"])
    group_target = read_gz(paths["group_target"])
    group_health = read_gz(paths["group_health"])
    pairs = read_gz(paths["pairs"])
    target_manifest_path = Path(args.run_dir) / "run_manifest.target.json"
    target_manifest = json.loads(target_manifest_path.read_text())
    candidates = target_manifest["candidates"]
    endpoints = target_manifest["endpoints"]
    endpoint_map: dict[str, list[dict]] = defaultdict(list)
    for row in endpoints:
        endpoint_map[row["candidate_id"]].append(row)

    family_rows = []
    diagnostics = []
    relation_rows = []
    for candidate in candidates:
        members = family_members(candidate, stage_target)
        candidate_families = []
        for endpoint in endpoint_map[candidate["candidate_id"]]:
            sources = endpoint_sources(candidate, endpoint, members, stage_target, group_target)
            if endpoint["scope_kind"] == "contrast":
                point, covariance, inference = context_covariance(
                    sources, pairs, endpoint["scope_value"]
                )
            else:
                point, covariance, inference = gene_covariance(sources, endpoint["target_role"])
            mt = max_t(point, covariance)
            health = health_sources(candidate, endpoint, members, stage_health, group_health)
            member_rows = []
            for i, member in enumerate(members):
                low = float(point[i] - mt["critical"] * mt["se"][i])
                high = float(point[i] + mt["critical"] * mt["se"][i])
                family_mde = float((mt["critical"] + norm.ppf(0.80)) * mt["se"][i])
                member_rows.append({
                    "member": member, "mean": float(point[i]), "se": float(mt["se"][i]),
                    "t": float(mt["t"][i]), "p_max_t": float(mt["p_max_t"][i]),
                    "ci_fwer_max_t": [low, high], "mde80_familywise": family_mde,
                    "tau": TAU, "classification": classify(
                        float(point[i]), low, high, float(mt["p_max_t"][i])
                    ),
                    "bounded_null_licensed": bool(
                        low >= -TAU and high <= TAU and family_mde <= TAU
                    ),
                    "masked_health": marginal_health(health[member], endpoint["target_role"]),
                })
            family = {
                "candidate_id": candidate["candidate_id"],
                "component_id": candidate["component_id"], "estimand": candidate["estimand"],
                "target_role": endpoint["target_role"], "contrast": endpoint["contrast"],
                "endpoint_key": endpoint["endpoint_key"], "scope_kind": endpoint["scope_kind"],
                "scope_value": endpoint["scope_value"], **inference,
                "family_members": members, "n_family_members": len(members),
                "max_t": {
                    "method": "Gaussian max-|t|", "draws": N_MAXT, "seed": MAXT_SEED,
                    "alpha": ALPHA, "critical_value": mt["critical"],
                    "min_raw_covariance_eigenvalue": mt["min_raw_eigenvalue"],
                    "n_negative_eigenvalues_clipped": mt["n_negative_eigenvalues_clipped"],
                    "covariance": mt["covariance"].tolist(),
                },
                "members": member_rows,
                "claim_boundary": "conditional intervention evidence; no binding or biological function label",
            }
            family["conditional_labels"] = conditional_labels(
                candidate, endpoint, member_rows, smoke=args.smoke
            )
            family_rows.append(family)
            candidate_families.append(family)

            diagnostic_members = sorted({
                row["member"] for row in stage_target
                if row["endpoint_key"] == endpoint["endpoint_key"]
                and row["member_role"] == "diagnostic"
            })
            for member in diagnostic_members:
                source = {
                    member: [
                        row for row in stage_target
                        if row["endpoint_key"] == endpoint["endpoint_key"] and row["member"] == member
                    ]
                }
                if endpoint["scope_kind"] == "contrast":
                    p, cov, meta = context_covariance(source, pairs, endpoint["scope_value"])
                else:
                    p, cov, meta = gene_covariance(source, endpoint["target_role"])
                se = math.sqrt(max(float(cov[0, 0]), 0.0))
                diagnostics.append({
                    "candidate_id": candidate["candidate_id"],
                    "component_id": candidate["component_id"], "estimand": candidate["estimand"],
                    "target_role": endpoint["target_role"], "contrast": endpoint["contrast"],
                    "endpoint_key": endpoint["endpoint_key"], "member": member,
                    "mean": float(p[0]), "se": se,
                    "ci95_marginal": [float(p[0] - 1.96 * se), float(p[0] + 1.96 * se)],
                    **meta, "multiplicity_status": "diagnostic_not_in_stage_two_family",
                })

        non_group = [
            row for family in candidate_families for row in family["members"]
            if row["member"] != "exact_group_effect"
        ]
        relation_rows.append({
            "candidate_id": candidate["candidate_id"], "kind": candidate["kind"],
            "component_id": candidate["component_id"], "component_type": candidate["component_type"],
            "estimand": candidate["estimand"], "contrast": candidate.get("contrast"),
            "target_role": candidate.get("target_role"),
            "stage2_status": "STAGE2_CONDITIONAL_INFERENCE_COMPLETE",
            "material_members_by_endpoint": {
                family["target_role"]: [
                    row["member"] for row in family["members"]
                    if row["classification"] in {"material_support", "material_suppress"}
                ] for family in candidate_families
            },
            "conditional_labels_by_endpoint": {
                family["target_role"]: family["conditional_labels"]
                for family in candidate_families
            },
            "n_material_non_group_members": sum(
                row["classification"] in {"material_support", "material_suppress"}
                for row in non_group
            ),
            "function_label": None,
            "claim_boundary": "validated whole relation plus conditional paths; biological function unresolved",
        })

    write_jsonl(outputs["families"], family_rows)
    write_jsonl(outputs["diagnostics"], diagnostics)
    outputs["relations"].write_text(json.dumps({
        "schema": "head_context_atlas.stage2_mechanism_relations/1",
        "status": "SMOKE" if args.smoke else "STAGE2_CLASSIFIED",
        "relations": relation_rows,
    }, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "head_context_atlas.stage2_mechanism_analysis/1",
        "status": "SMOKE" if args.smoke else "STAGE2_CLASSIFIED",
        "claim_boundary": "conditional mechanisms only; no binding or biological function claim",
        "smoke": args.smoke, "whole_validation_loaded": not args.smoke,
        "stage2_outcomes_loaded": True, "n_candidates": len(candidates),
        "n_endpoint_families": len(family_rows),
        "n_family_members": sum(row["n_family_members"] for row in family_rows),
        "n_diagnostics": len(diagnostics),
        "classification_counts": dict(Counter(
            member["classification"] for family in family_rows for member in family["members"]
        )),
        "inference": {
            "method": "Gaussian max-|t|", "draws": N_MAXT, "seed": MAXT_SEED,
            "alpha": ALPHA, "tau_nats": TAU,
            "covariance": "joint gene-clustered or inverse-reuse unique-dyad two-endpoint",
        },
        "next_stage_rule": (
            "independent reconstruction audit before representative A interchange, "
            "evidence-graded DAG generation, or visualizer export"
        ),
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_stage2_mechanism_analysis_v1",
        command=["python"] + sys.argv, status=summary["status"],
    )
    manifest.config.update({
        "smoke": args.smoke, "whole_validation_loaded": not args.smoke,
        "stage2_outcomes_loaded": True, "alpha": ALPHA, "tau_nats": TAU,
        "max_t_draws": N_MAXT, "max_t_seed": MAXT_SEED,
        "amendment_a2_sha256": sha256_file(AMENDMENT),
    })
    for path in paths.values():
        manifest.add_input(str(path))
    manifest.add_input(str(target_manifest_path))
    for key in ("families", "relations", "diagnostics", "summary"):
        manifest.add_output(str(outputs[key]))
    manifest.write(str(outputs["manifest"]))
    print(json.dumps({
        "status": summary["status"], "n_candidates": len(candidates),
        "n_endpoint_families": len(family_rows),
        "n_family_members": summary["n_family_members"],
    }, indent=2))


if __name__ == "__main__":
    main()
