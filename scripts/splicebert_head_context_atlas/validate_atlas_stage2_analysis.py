#!/usr/bin/env python
"""Independent reconstruction audit for stage-two joint mechanism inference."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm


ALPHA = 0.05
TAU = 0.05
N_MAXT = 200_000
MAXT_SEED = 0
ATOL = 1e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def close(a, b, atol: float = ATOL) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= atol
    return a == b


def compare(observed: dict, expected: dict, label: str, failures: list[str]) -> None:
    for key, value in expected.items():
        require(key in observed and close(observed.get(key), value),
                f"{label}: {key} mismatch: {observed.get(key)!r} != {value!r}", failures)


def member_grid(candidate: dict, stage_rows: list[dict]) -> list[str]:
    observed = sorted({
        row["member"] for row in stage_rows
        if row["candidate_id"] == candidate["candidate_id"] and row["member_role"] == "family"
    })
    grid = ["exact_group_effect", *observed]
    frozen = candidate.get("within_endpoint_family_members")
    if frozen is not None:
        if set(grid) != set(frozen) or len(grid) != len(frozen):
            raise RuntimeError("frozen family grid mismatch")
        return list(frozen)
    return grid


def sources(candidate: dict, endpoint: dict, members: list[str], stage: list[dict],
            group: list[dict], field: str) -> dict[str, list[dict]]:
    result = {}
    for member in members:
        if member == "exact_group_effect":
            result[member] = [
                {**row, "value": row[field]}
                for row in group if row["group_id"] == candidate["exact_group_id"]
            ]
        else:
            result[member] = [
                row for row in stage
                if row["endpoint_key"] == endpoint["endpoint_key"] and row["member"] == member
            ]
        if not result[member]:
            raise RuntimeError(f"empty source {candidate['candidate_id']}|{member}")
    return result


def packet_matrix(blocks: dict[str, list[dict]], packet_ids: list[str]) -> np.ndarray:
    matrix = np.empty((len(packet_ids), len(blocks)), dtype=float)
    for j, rows in enumerate(blocks.values()):
        lookup = {row["packet_id"]: float(row["value"]) for row in rows}
        if len(lookup) != len(rows):
            raise RuntimeError("duplicate packet in member source")
        matrix[:, j] = [lookup[packet_id] for packet_id in packet_ids]
    return matrix


def gene_joint(blocks: dict[str, list[dict]], role: str) -> tuple[np.ndarray, np.ndarray, dict]:
    anchor = next(iter(blocks.values()))
    packet_ids = sorted({row["packet_id"] for row in anchor if row["target_role"] == role})
    matrix = packet_matrix(blocks, packet_ids)
    lookup = {row["packet_id"]: row for row in anchor}
    genes = [lookup[packet_id]["gene_id"] for packet_id in packet_ids]
    point = matrix.mean(axis=0)
    residual = matrix - point
    unique = sorted(set(genes))
    index = {gene: i for i, gene in enumerate(unique)}
    score = np.zeros((len(unique), matrix.shape[1]))
    np.add.at(score, [index[gene] for gene in genes], residual)
    covariance = (len(unique) / max(len(unique) - 1, 1)) * score.T @ score / len(packet_ids) ** 2
    return point, covariance, {
        "inference": "joint_one_way_gene_clustered",
        "n_packets": len(packet_ids), "n_genes": len(unique),
    }


def dyad_joint(blocks: dict[str, list[dict]], pair_rows: list[dict],
               contrast: str) -> tuple[np.ndarray, np.ndarray, dict]:
    category, role = contrast.split("|", 1)
    pairs = [row for row in pair_rows if row["category"] == category and row["target_role"] == role]
    packet_ids = sorted({
        row[field] for row in pairs for field in ("case_packet_id", "control_packet_id")
    })
    matrix = packet_matrix(blocks, packet_ids)
    lookup = {packet_id: matrix[i] for i, packet_id in enumerate(packet_ids)}
    difference = np.asarray([
        lookup[row["case_packet_id"]] - lookup[row["control_packet_id"]] for row in pairs
    ])
    reuse = Counter(row["control_packet_id"] for row in pairs)
    weights = np.asarray([1 / reuse[row["control_packet_id"]] for row in pairs])
    total = float(weights.sum())
    point = np.sum(weights[:, None] * difference, axis=0) / total
    score = weights[:, None] * (difference - point)
    left = [row["case_gene_id"] for row in pairs]
    right = [row["control_gene_id"] for row in pairs]
    if len({tuple(sorted(item)) for item in zip(left, right)}) != len(pairs):
        raise RuntimeError("repeated endpoint dyad")
    genes = sorted(set(left) | set(right))
    index = {gene: i for i, gene in enumerate(genes)}
    incident = np.zeros((len(genes), difference.shape[1]))
    np.add.at(incident, [index[gene] for gene in left], score)
    np.add.at(incident, [index[gene] for gene in right], score)
    covariance = (len(genes) / max(len(genes) - 1, 1)) * (
        incident.T @ incident - score.T @ score
    ) / total ** 2
    return point, covariance, {
        "inference": "joint_inverse_reuse_weighted_unique_dyad_two_endpoint_gene_sandwich",
        "contrast": contrast, "n_pairs": len(pairs), "n_endpoint_genes": len(genes),
        "n_effective_pairs": float(total ** 2 / np.square(weights).sum()),
        "total_weight": total,
    }


def health(rows: list[dict], role: str) -> dict:
    block = [row for row in rows if row["target_role"] == role]
    values = np.asarray([float(row["value"]) for row in block])
    point = float(values.mean())
    residual = values - point
    genes = sorted({row["gene_id"] for row in block})
    index = {gene: i for i, gene in enumerate(genes)}
    score = np.zeros(len(genes))
    np.add.at(score, [index[row["gene_id"]] for row in block], residual)
    se = math.sqrt(max((len(genes) / max(len(genes) - 1, 1)) * float(score @ score), 0)) / len(block)
    low, high = point - 1.96 * se, point + 1.96 * se
    return {
        "mean_health_margin_damage": point, "se_health": se,
        "ci95_health_margin_damage": [low, high], "n_health_packets": len(block),
        "n_health_genes": len(genes), "broad_MLM_flag": low > TAU or high < -TAU,
    }


def max_t(point: np.ndarray, covariance: np.ndarray) -> dict:
    covariance = (covariance + covariance.T) / 2
    eig, vec = np.linalg.eigh(covariance)
    psd = (vec * np.clip(eig, 0, None)) @ vec.T
    se = np.sqrt(np.clip(np.diag(psd), 0, None))
    corr = np.zeros_like(psd)
    active = se > 0
    corr[np.ix_(active, active)] = psd[np.ix_(active, active)] / np.outer(se[active], se[active])
    corr[active, active] = 1
    ceig, cvec = np.linalg.eigh((corr + corr.T) / 2)
    root = (cvec * np.sqrt(np.clip(ceig, 0, None))) @ cvec.T
    rng = np.random.default_rng(MAXT_SEED)
    maxima = np.empty(N_MAXT)
    start = 0
    while start < N_MAXT:
        n = min(20_000, N_MAXT - start)
        maxima[start:start + n] = np.max(np.abs(rng.standard_normal((n, len(point))) @ root.T), axis=1)
        start += n
    critical = float(np.quantile(maxima, 1 - ALPHA, method="higher"))
    t = np.divide(point, se, out=np.where(point == 0, 0.0, np.sign(point) * np.inf), where=se > 0)
    p = np.asarray([
        (1 + np.count_nonzero(maxima >= abs(value))) / (N_MAXT + 1) for value in t
    ])
    return {
        "covariance": psd, "se": se, "t": t, "p": p, "critical": critical,
        "min_eigen": float(eig.min()),
        "n_clipped": int(np.count_nonzero(eig < -1e-12)),
    }


def classification(point: float, low: float, high: float, p: float) -> str:
    if p <= ALPHA and low > TAU:
        return "material_support"
    if p <= ALPHA and high < -TAU:
        return "material_suppress"
    if p <= ALPHA and (low > 0 or high < 0):
        return "detected_subfloor"
    if low >= -TAU and high <= TAU:
        return "bounded_small"
    return "inconclusive"


def expected_labels(candidate: dict, endpoint: dict, members: list[dict], smoke: bool) -> dict:
    if smoke:
        return {
            "route_value_label": "SMOKE_NO_LABEL", "path_labels": [],
            "validated_whole_direction": None,
        }
    whole = next(
        row for row in candidate["validated_whole_endpoints"]
        if row["target_role"] == endpoint["target_role"]
        and row.get("contrast") == endpoint.get("contrast")
    )
    if not whole["validated_whole_endpoint"] or abs(float(whole["mean"])) <= TAU:
        raise RuntimeError("non-material endpoint entered stage two")
    direction = 1 if float(whole["mean"]) > 0 else -1
    lookup = {row["member"]: row for row in members}
    route = None
    if candidate["component_type"] == "attention_head" and candidate["estimand"] == "B":
        expected_material = "material_support" if direction > 0 else "material_suppress"
        qk_material = lookup["candidate_QK"]["classification"] == expected_material
        value_material = lookup["candidate_V_OV"]["classification"] == expected_material
        qk_bounded = lookup["candidate_QK"]["classification"] == "bounded_small"
        value_bounded = lookup["candidate_V_OV"]["classification"] == "bounded_small"
        if qk_material and value_bounded:
            route = "partner_sensitive_routing"
        elif value_material and qk_bounded:
            route = "partner_sensitive_value_write"
        elif qk_material and value_material:
            route = "mixed_partner_sensitive"
        else:
            route = "mechanism_unresolved"
    if candidate["component_type"] == "attention_head":
        paths = [
            row["member"] for row in members
            if row["member"].startswith("MLP_recovery_difference_L")
            and row["classification"] == "material_support"
        ]
    else:
        upstream = lookup.get("upstream_attention_to_candidate_vs_control_mlp_recovery")
        paths = (["upstream_attention_to_candidate_MLP_path"]
                 if upstream is not None and upstream["classification"] == "material_support" else [])
    return {
        "route_value_label": route, "path_labels": paths,
        "validated_whole_direction": direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-prefix", required=True)
    parser.add_argument("--analysis-prefix", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    merged = Path(args.merged_prefix)
    analysis = Path(args.analysis_prefix)
    audit_path = Path(args.audit) if args.audit else Path(f"{analysis}.audit.json")
    if audit_path.exists():
        raise FileExistsError(f"refuse to overwrite immutable audit {audit_path}")
    paths = {
        "target": Path(f"{merged}.target_rows.jsonl.gz"),
        "health": Path(f"{merged}.health_rows.jsonl.gz"),
        "group_target": Path(f"{merged}.whole_group_target_rows.jsonl.gz"),
        "group_health": Path(f"{merged}.whole_group_health_rows.jsonl.gz"),
        "pairs": Path(f"{merged}.pairs.jsonl.gz"),
        "families": Path(f"{analysis}.families.jsonl"),
        "relations": Path(f"{analysis}.relations.json"),
        "diagnostics": Path(f"{analysis}.diagnostics.jsonl"),
        "summary": Path(f"{analysis}.summary.json"),
        "manifest": Path(f"{analysis}.manifest.json"),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    stage_target = read_rows(paths["target"])
    stage_health = read_rows(paths["health"])
    group_target = read_rows(paths["group_target"])
    group_health = read_rows(paths["group_health"])
    pairs = read_rows(paths["pairs"])
    observed_families = read_rows(paths["families"])
    observed_relations = json.loads(paths["relations"].read_text())["relations"]
    observed_diagnostics = read_rows(paths["diagnostics"])
    summary = json.loads(paths["summary"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    run_manifest = json.loads((Path(args.run_dir) / "run_manifest.target.json").read_text())
    candidates = run_manifest["candidates"]
    endpoints_by_candidate: dict[str, list[dict]] = defaultdict(list)
    for endpoint in run_manifest["endpoints"]:
        endpoints_by_candidate[endpoint["candidate_id"]].append(endpoint)
    failures: list[str] = []

    observed_family_map = {
        (row["candidate_id"], row["target_role"], row["contrast"]): row
        for row in observed_families
    }
    require(len(observed_family_map) == len(observed_families),
            "family endpoint grid is not unique", failures)
    expected_classifications = []
    expected_material: dict[str, dict[str, list[str]]] = defaultdict(dict)
    expected_conditional: dict[str, dict[str, dict]] = defaultdict(dict)
    expected_family_count = 0
    for candidate in candidates:
        members = member_grid(candidate, stage_target)
        for endpoint in endpoints_by_candidate[candidate["candidate_id"]]:
            expected_family_count += 1
            target_blocks = sources(
                candidate, endpoint, members, stage_target, group_target, "effect_margin"
            )
            if endpoint["scope_kind"] == "contrast":
                point, covariance, meta = dyad_joint(target_blocks, pairs, endpoint["scope_value"])
            else:
                point, covariance, meta = gene_joint(target_blocks, endpoint["target_role"])
            mt = max_t(point, covariance)
            health_blocks = sources(
                candidate, endpoint, members, stage_health, group_health, "health_margin_damage"
            )
            expected_members = []
            for i, member in enumerate(members):
                low = float(point[i] - mt["critical"] * mt["se"][i])
                high = float(point[i] + mt["critical"] * mt["se"][i])
                mde = float((mt["critical"] + norm.ppf(.80)) * mt["se"][i])
                label = classification(float(point[i]), low, high, float(mt["p"][i]))
                expected_classifications.append(label)
                expected_members.append({
                    "member": member, "mean": float(point[i]), "se": float(mt["se"][i]),
                    "t": float(mt["t"][i]), "p_max_t": float(mt["p"][i]),
                    "ci_fwer_max_t": [low, high], "mde80_familywise": mde,
                    "tau": TAU, "classification": label,
                    "bounded_null_licensed": low >= -TAU and high <= TAU and mde <= TAU,
                    "masked_health": health(health_blocks[member], endpoint["target_role"]),
                })
            key = candidate["candidate_id"], endpoint["target_role"], endpoint["contrast"]
            observed = observed_family_map.get(key)
            require(observed is not None, f"missing family endpoint {key}", failures)
            if observed:
                compare(observed, {
                    "component_id": candidate["component_id"], "estimand": candidate["estimand"],
                    "endpoint_key": endpoint["endpoint_key"], "scope_kind": endpoint["scope_kind"],
                    "scope_value": endpoint["scope_value"], **meta,
                    "family_members": members, "n_family_members": len(members),
                    "conditional_labels": expected_labels(
                        candidate, endpoint, expected_members, args.smoke
                    ),
                }, f"family {key}", failures)
                compare(observed["max_t"], {
                    "method": "Gaussian max-|t|", "draws": N_MAXT, "seed": MAXT_SEED,
                    "alpha": ALPHA, "critical_value": mt["critical"],
                    "min_raw_covariance_eigenvalue": mt["min_eigen"],
                    "n_negative_eigenvalues_clipped": mt["n_clipped"],
                    "covariance": mt["covariance"].tolist(),
                }, f"family max-t {key}", failures)
                require(len(observed["members"]) == len(expected_members),
                        f"family member count mismatch {key}", failures)
                for actual, expected in zip(observed["members"], expected_members):
                    compare(actual, expected, f"family member {key}|{expected['member']}", failures)
            expected_material[candidate["candidate_id"]][endpoint["target_role"]] = [
                row["member"] for row in expected_members
                if row["classification"] in {"material_support", "material_suppress"}
            ]
            expected_conditional[candidate["candidate_id"]][endpoint["target_role"]] = expected_labels(
                candidate, endpoint, expected_members, args.smoke
            )
    require(len(observed_families) == expected_family_count,
            "family endpoint count mismatch", failures)

    observed_diag_map = {
        (row["candidate_id"], row["endpoint_key"], row["member"]): row
        for row in observed_diagnostics
    }
    expected_diag_count = 0
    for candidate in candidates:
        for endpoint in endpoints_by_candidate[candidate["candidate_id"]]:
            members = sorted({
                row["member"] for row in stage_target
                if row["endpoint_key"] == endpoint["endpoint_key"]
                and row["member_role"] == "diagnostic"
            })
            for member in members:
                expected_diag_count += 1
                block = {member: [
                    row for row in stage_target
                    if row["endpoint_key"] == endpoint["endpoint_key"] and row["member"] == member
                ]}
                if endpoint["scope_kind"] == "contrast":
                    point, covariance, meta = dyad_joint(block, pairs, endpoint["scope_value"])
                else:
                    point, covariance, meta = gene_joint(block, endpoint["target_role"])
                se = math.sqrt(max(float(covariance[0, 0]), 0))
                key = candidate["candidate_id"], endpoint["endpoint_key"], member
                observed = observed_diag_map.get(key)
                require(observed is not None, f"missing diagnostic {key}", failures)
                if observed:
                    compare(observed, {
                        "component_id": candidate["component_id"], "estimand": candidate["estimand"],
                        "target_role": endpoint["target_role"], "contrast": endpoint["contrast"],
                        "mean": float(point[0]), "se": se,
                        "ci95_marginal": [float(point[0] - 1.96 * se), float(point[0] + 1.96 * se)],
                        **meta,
                    }, f"diagnostic {key}", failures)
    require(len(observed_diagnostics) == expected_diag_count,
            "diagnostic count mismatch", failures)

    observed_relation_map = {row["candidate_id"]: row for row in observed_relations}
    require(len(observed_relation_map) == len(observed_relations) == len(candidates),
            "relation grid mismatch", failures)
    for candidate in candidates:
        observed = observed_relation_map.get(candidate["candidate_id"])
        require(observed is not None, f"missing relation {candidate['candidate_id']}", failures)
        if observed:
            expected_non_group_material = sum(
                member != "exact_group_effect"
                for values in expected_material[candidate["candidate_id"]].values()
                for member in values
            )
            compare(observed, {
                "kind": candidate["kind"], "component_id": candidate["component_id"],
                "component_type": candidate["component_type"], "estimand": candidate["estimand"],
                "contrast": candidate.get("contrast"), "target_role": candidate.get("target_role"),
                "stage2_status": "STAGE2_CONDITIONAL_INFERENCE_COMPLETE",
                "material_members_by_endpoint": expected_material[candidate["candidate_id"]],
                "conditional_labels_by_endpoint": expected_conditional[candidate["candidate_id"]],
                "n_material_non_group_members": expected_non_group_material,
                "function_label": None,
            }, f"relation {candidate['candidate_id']}", failures)

    require(summary["smoke"] is args.smoke and summary["whole_validation_loaded"] is (not args.smoke)
            and summary["n_candidates"] == len(candidates)
            and summary["n_endpoint_families"] == expected_family_count
            and summary["n_family_members"] == sum(len(row["members"]) for row in observed_families)
            and summary["n_diagnostics"] == expected_diag_count,
            "summary counts/boundary mismatch", failures)
    require(summary["classification_counts"] == dict(Counter(expected_classifications)) and
            summary["inference"] == {
                "method": "Gaussian max-|t|", "draws": N_MAXT, "seed": MAXT_SEED,
                "alpha": ALPHA, "tau_nats": TAU,
                "covariance": "joint gene-clustered or inverse-reuse unique-dyad two-endpoint",
            }, "summary classification/inference mismatch", failures)

    manifest_outputs = {str(Path(row["path"]).resolve()): row for row in manifest["outputs"]}
    for key in ("families", "relations", "diagnostics", "summary"):
        path = paths[key]
        record = manifest_outputs.get(str(path.resolve()))
        require(record is not None and record["sha256"] == sha256_file(path)
                and record["size_bytes"] == path.stat().st_size,
                f"manifest output hash/size mismatch {key}", failures)
    require(manifest["config"]["max_t_draws"] == N_MAXT and
            manifest["config"]["max_t_seed"] == MAXT_SEED and
            manifest["config"]["alpha"] == ALPHA and manifest["config"]["tau_nats"] == TAU,
            "analysis manifest inference configuration drift", failures)

    report = {
        "schema": "head_context_atlas.stage2_mechanism_analysis_audit/1",
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": "independent joint-inference reconstruction; no function claim",
        "smoke": args.smoke, "whole_validation_loaded": not args.smoke,
        "stage2_outcomes_loaded": True, "n_candidates": len(candidates),
        "n_endpoint_families": expected_family_count,
        "n_family_members": sum(len(row["members"]) for row in observed_families),
        "n_diagnostics": expected_diag_count, "n_failures": len(failures),
        "failures": failures, "analysis_summary_sha256": sha256_file(paths["summary"]),
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
