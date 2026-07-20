#!/usr/bin/env python
"""Independent reconstruction audit for mechanism whole-effect inference.

This verifier intentionally does not import the analysis implementation.  It
recomputes gene-clustered and two-endpoint dyadic estimates, Holm adjustment,
Bonferroni simultaneous intervals, classifications, relation progression, health
annotations, diagnostics, and artifact hashes directly from audited merged rows.
"""
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


REPO = Path(__file__).resolve().parents[2]
TAU = 0.05
ALPHA = 0.05
Z_POWER_80 = float(norm.ppf(0.975) + norm.ppf(0.80))
ATOL = 1e-10


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


def close(a, b, atol: float = ATOL) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= atol
    return a == b


def pvalue(point: float, se: float) -> float:
    if se > 0:
        return float(2.0 * norm.sf(abs(point / se)))
    return 0.0 if abs(point) > 0 else 1.0


def gene_stats(rows: list[dict], field: str) -> dict:
    if not rows:
        raise RuntimeError("empty gene block")
    values = np.asarray([float(row[field]) for row in rows])
    point = float(values.mean())
    residual = values - point
    genes = sorted({row["gene_id"] for row in rows})
    index = {gene: i for i, gene in enumerate(genes)}
    scores = np.zeros(len(genes), dtype=float)
    np.add.at(scores, [index[row["gene_id"]] for row in rows], residual)
    correction = len(genes) / max(len(genes) - 1, 1)
    se = float(math.sqrt(max(float(np.square(scores).sum()) * correction, 0.0)) / len(rows))
    return {
        "mean": point,
        "se": se,
        "ci95_marginal": [point - 1.96 * se, point + 1.96 * se],
        "p_marginal_two_sided": pvalue(point, se),
        "mde80_marginal": Z_POWER_80 * se,
        "n_packets": len(rows),
        "n_genes": len(genes),
        "mean_abs_effect": float(np.mean(np.abs(values))),
        "inference": "one_way_gene_clustered",
    }


def dyadic_stats(rows: list[dict], pairs: list[dict], contrast: str, field: str) -> dict:
    category, role = contrast.split("|", 1)
    block = [pair for pair in pairs if pair["category"] == category and pair["target_role"] == role]
    values = {row["packet_id"]: float(row[field]) for row in rows}
    if len(values) != len(rows):
        raise RuntimeError("duplicate endpoint packet row")
    difference = np.asarray([
        values[pair["case_packet_id"]] - values[pair["control_packet_id"]]
        for pair in block
    ])
    reuse = Counter(pair["control_packet_id"] for pair in block)
    weights = np.asarray([1.0 / reuse[pair["control_packet_id"]] for pair in block])
    case_genes = [pair["case_gene_id"] for pair in block]
    control_genes = [pair["control_gene_id"] for pair in block]
    dyads = [tuple(sorted((a, b))) for a, b in zip(case_genes, control_genes)]
    if len(dyads) != len(set(dyads)):
        raise RuntimeError("repeated endpoint dyad")
    total = float(weights.sum())
    point = float(np.sum(weights * difference) / total)
    score = weights * (difference - point)
    genes = sorted(set(case_genes) | set(control_genes))
    index = {gene: i for i, gene in enumerate(genes)}
    incident = np.zeros(len(genes), dtype=float)
    np.add.at(incident, [index[gene] for gene in case_genes], score)
    np.add.at(incident, [index[gene] for gene in control_genes], score)
    meat = float(np.square(incident).sum() - np.square(score).sum())
    correction = len(genes) / max(len(genes) - 1, 1)
    se = float(math.sqrt(max(meat * correction, 0.0)) / total)
    return {
        "mean": point,
        "se": se,
        "ci95_marginal": [point - 1.96 * se, point + 1.96 * se],
        "p_marginal_two_sided": pvalue(point, se),
        "mde80_marginal": Z_POWER_80 * se,
        "n_pairs": len(block),
        "n_endpoint_genes": len(genes),
        "n_effective_pairs": float(total ** 2 / np.square(weights).sum()),
        "total_weight": total,
        "mean_abs_pair_difference": float(np.average(np.abs(difference), weights=weights)),
        "inference": "inverse_reuse_weighted_unique_dyad_two_endpoint_gene_sandwich",
        "contrast": contrast,
    }


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    if not m:
        return []
    order = sorted(range(m), key=lambda i: (pvalues[i], i))
    running = 0.0
    adjusted = [0.0] * m
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def classify(point: float, low: float, high: float, adjusted_p: float) -> str:
    if adjusted_p <= ALPHA and low > TAU:
        return "material_support"
    if adjusted_p <= ALPHA and high < -TAU:
        return "material_suppress"
    if adjusted_p <= ALPHA and (low > 0 or high < 0):
        return "detected_subfloor"
    if low >= -TAU and high <= TAU:
        return "bounded_small"
    return "inconclusive"


def candidate_specs(package: dict, target_rows: list[dict], smoke: bool) -> list[dict]:
    if not smoke:
        return package["candidates"]
    specs = {}
    for row in target_rows:
        if row["endpoint_role"] == "candidate":
            specs.setdefault(row["candidate_id"], {
                "candidate_id": row["candidate_id"], "kind": "general",
                "estimand": row["estimand"], "component_id": row["component_ids"][0],
                "discovery": {"donor_mean": 1.0, "acceptor_mean": 1.0},
            })
    return [specs[key] for key in sorted(specs)]


def expected_direction(candidate: dict, role: str) -> int:
    value = (
        candidate["discovery"][f"{role}_mean"] if candidate["kind"] == "general"
        else candidate["discovery"]["mean_difference"]
    )
    return 1 if float(value) > 0 else -1


def health_expected(health_rows: list[dict], endpoint_id: str, role: str) -> dict:
    rows = [
        row for row in health_rows
        if row["endpoint_id"] == endpoint_id and row["target_role"] == role
    ]
    stats = gene_stats(rows, "health_margin_damage")
    return {
        "target_role": role,
        "mean_health_margin_damage": stats["mean"],
        "ci95_health_margin_damage": stats["ci95_marginal"],
        "n_health_packets": stats["n_packets"],
        "n_health_genes": stats["n_genes"],
        "broad_MLM_flag": (
            stats["ci95_marginal"][0] > TAU or stats["ci95_marginal"][1] < -TAU
        ),
    }


def compare_stats(observed: dict, expected: dict, label: str, failures: list[str]) -> None:
    for key, value in expected.items():
        require(key in observed and close(observed.get(key), value),
                f"{label}: mismatch in {key}: observed={observed.get(key)!r}, expected={value!r}",
                failures)


def endpoint_meta(target_rows: list[dict]) -> dict[str, dict]:
    meta = {}
    fields = (
        "endpoint_id", "endpoint_role", "candidate_id", "group_id", "estimand",
        "component_ids", "scope_kind", "scope_value", "intervention",
    )
    for row in target_rows:
        value = {field: row[field] for field in fields}
        if row["endpoint_id"] in meta and meta[row["endpoint_id"]] != value:
            raise RuntimeError(f"endpoint metadata drift for {row['endpoint_id']}")
        meta[row["endpoint_id"]] = value
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-prefix", required=True)
    parser.add_argument("--analysis-prefix", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    merged = Path(args.merged_prefix)
    analysis = Path(args.analysis_prefix)
    package_path = Path(args.package)
    audit_path = Path(args.audit) if args.audit else Path(f"{analysis}.audit.json")
    if audit_path.exists():
        raise FileExistsError(f"refuse to overwrite immutable audit {audit_path}")
    paths = {
        "target": Path(f"{merged}.target_rows.jsonl.gz"),
        "health": Path(f"{merged}.health_rows.jsonl.gz"),
        "pairs": Path(f"{merged}.pairs.jsonl.gz"),
        "endpoints": Path(f"{analysis}.candidate_endpoints.jsonl"),
        "candidates": Path(f"{analysis}.candidates.json"),
        "diagnostics": Path(f"{analysis}.diagnostics.jsonl"),
        "summary": Path(f"{analysis}.summary.json"),
        "manifest": Path(f"{analysis}.manifest.json"),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    package = json.loads(package_path.read_text())
    target_rows = read_jsonl(paths["target"])
    health_rows = read_jsonl(paths["health"])
    pairs = read_jsonl(paths["pairs"])
    observed_endpoints = read_jsonl(paths["endpoints"])
    observed_candidates = json.loads(paths["candidates"].read_text())["relations"]
    observed_diagnostics = read_jsonl(paths["diagnostics"])
    summary = json.loads(paths["summary"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    failures: list[str] = []

    target_by_endpoint: dict[str, list[dict]] = defaultdict(list)
    for row in target_rows:
        target_by_endpoint[row["endpoint_id"]].append(row)
    candidates = candidate_specs(package, target_rows, args.smoke)
    expected_primary = []
    for candidate in candidates:
        endpoint_id = (
            f"smoke|candidate|{candidate['estimand']}|{candidate['component_id']}"
            if args.smoke else f"candidate|{candidate['candidate_id']}"
        )
        rows = target_by_endpoint[endpoint_id]
        roles = ("donor", "acceptor") if candidate["kind"] == "general" else (candidate["target_role"],)
        for role in roles:
            if candidate["kind"] == "general":
                stats = gene_stats([row for row in rows if row["target_role"] == role], "effect_margin")
                contrast = None
            else:
                stats = dyadic_stats(rows, pairs, candidate["contrast"], "effect_margin")
                contrast = candidate["contrast"]
            expected_primary.append({
                "candidate_id": candidate["candidate_id"], "endpoint_id": endpoint_id,
                "kind": candidate["kind"], "component_id": candidate["component_id"],
                "component_type": "mlp_write" if candidate["component_id"].endswith("MLP") else "attention_head",
                "estimand": candidate["estimand"], "target_role": role,
                "contrast": contrast,
                "expected_discovery_direction": expected_direction(candidate, role),
                **stats,
                "masked_health": health_expected(health_rows, endpoint_id, role),
            })

    m = len(expected_primary)
    adjusted = holm([row["p_marginal_two_sided"] for row in expected_primary])
    z = float(norm.ppf(1.0 - ALPHA / (2.0 * m))) if m else None
    for row, p_holm in zip(expected_primary, adjusted):
        low = row["mean"] - z * row["se"]
        high = row["mean"] + z * row["se"]
        family_mde = (z + float(norm.ppf(0.80))) * row["se"]
        label = classify(row["mean"], low, high, p_holm)
        direction_match = row["mean"] * row["expected_discovery_direction"] > 0
        row.update({
            "p_holm_whole_family": p_holm,
            "ci_fwer_bonferroni": [low, high],
            "simultaneous_interval_method": f"Bonferroni 1-{ALPHA}/m, m={m}",
            "mde80_familywise": family_mde, "tau": TAU,
            "classification": label,
            "direction_matches_discovery": direction_match,
            "validated_whole_endpoint": direction_match and label in {"material_support", "material_suppress"},
            "bounded_null_licensed": label == "bounded_small" and family_mde <= TAU,
        })

    observed_primary = {
        (row["candidate_id"], row["target_role"]): row for row in observed_endpoints
    }
    require(len(observed_primary) == len(observed_endpoints) == m,
            "candidate endpoint IDs/roles are not unique or family size drifted", failures)
    for expected in expected_primary:
        key = expected["candidate_id"], expected["target_role"]
        observed = observed_primary.get(key)
        require(observed is not None, f"missing candidate endpoint {key}", failures)
        if observed is not None:
            compare_stats(observed, expected, f"candidate endpoint {key}", failures)

    expected_relations = []
    primary_by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in expected_primary:
        primary_by_candidate[row["candidate_id"]].append(row)
    for candidate in candidates:
        rows = primary_by_candidate[candidate["candidate_id"]]
        if all(row["validated_whole_endpoint"] for row in rows):
            status, advance = "VALIDATED_WHOLE_EFFECT", True
        elif any(
            row["classification"] in {"material_support", "material_suppress", "detected_subfloor"}
            and not row["direction_matches_discovery"] for row in rows
        ):
            status, advance = "RETIRED_DIRECTION_REVERSAL", False
        elif all(row["bounded_null_licensed"] for row in rows):
            status, advance = "RETIRED_BOUNDED_SMALL", False
        else:
            status, advance = "NOT_VALIDATED_INCONCLUSIVE", False
        expected_relations.append({
            "candidate_id": candidate["candidate_id"], "kind": candidate["kind"],
            "component_id": candidate["component_id"], "estimand": candidate["estimand"],
            "contrast": candidate.get("contrast"), "validation_status": status,
            "advance_to_qk_v_path": advance,
            "endpoint_classifications": {row["target_role"]: row["classification"] for row in rows},
        })
    observed_relation_map = {row["candidate_id"]: row for row in observed_candidates}
    require(len(observed_relation_map) == len(observed_candidates) == len(expected_relations),
            "candidate relation grid mismatch", failures)
    for expected in expected_relations:
        observed = observed_relation_map.get(expected["candidate_id"])
        require(observed is not None, f"missing candidate relation {expected['candidate_id']}", failures)
        if observed:
            compare_stats(observed, expected, f"candidate relation {expected['candidate_id']}", failures)

    meta = endpoint_meta(target_rows)
    expected_diagnostics = []
    for endpoint_id in sorted(meta):
        spec = meta[endpoint_id]
        if spec["endpoint_role"] == "candidate":
            continue
        if spec["scope_kind"] == "general":
            roles = ("donor", "acceptor")
        elif spec["scope_kind"] == "target_role":
            roles = (spec["scope_value"],)
        else:
            roles = (spec["scope_value"].split("|")[-1],)
        for role in roles:
            rows = target_by_endpoint[endpoint_id]
            if spec["scope_kind"] == "contrast":
                stats = dyadic_stats(rows, pairs, spec["scope_value"], "effect_margin")
                contrast = spec["scope_value"]
            else:
                stats = gene_stats([row for row in rows if row["target_role"] == role], "effect_margin")
                contrast = None
            expected_diagnostics.append({
                **spec, "target_role": role, "contrast": contrast, **stats,
                "masked_health": health_expected(health_rows, endpoint_id, role),
            })
    observed_diag_map = {
        (row["endpoint_id"], row["target_role"]): row for row in observed_diagnostics
    }
    require(len(observed_diag_map) == len(observed_diagnostics) == len(expected_diagnostics),
            "diagnostic endpoint grid mismatch", failures)
    for expected in expected_diagnostics:
        key = expected["endpoint_id"], expected["target_role"]
        observed = observed_diag_map.get(key)
        require(observed is not None, f"missing diagnostic {key}", failures)
        if observed:
            compare_stats(observed, expected, f"diagnostic {key}", failures)

    positive = next(row for row in expected_diagnostics if row["endpoint_role"] == "positive_control")
    expected_positive_pass = positive["ci95_marginal"][0] > 0
    require(summary["positive_control"]["passed"] is expected_positive_pass,
            "positive-control pass disposition mismatch", failures)
    require(summary["whole_family"]["n_endpoints"] == m and
            summary["n_candidates"] == len(expected_relations) and
            summary["n_validated_candidates"] == sum(row["advance_to_qk_v_path"] for row in expected_relations),
            "summary family/candidate counts mismatch", failures)
    require(summary["validated_candidate_ids"] == [
        row["candidate_id"] for row in expected_relations if row["advance_to_qk_v_path"]
    ], "summary validated-candidate list mismatch", failures)
    require(summary["candidate_status_counts"] == dict(Counter(
        row["validation_status"] for row in expected_relations
    )), "summary candidate-status counts mismatch", failures)
    require(summary["smoke"] is args.smoke and summary["validation_loaded"] is (not args.smoke),
            "summary smoke/validation boundary mismatch", failures)

    manifest_outputs = {
        str(Path(record["path"]).resolve()): record for record in manifest["outputs"]
    }
    for kind in ("endpoints", "candidates", "diagnostics", "summary"):
        path = paths[kind]
        record = manifest_outputs.get(str(path.resolve()))
        require(record is not None and record["sha256"] == sha256_file(path)
                and record["size_bytes"] == path.stat().st_size,
                f"manifest output hash/size mismatch for {kind}", failures)
    require(manifest["config"]["tau_nats"] == TAU and manifest["config"]["alpha"] == ALPHA
            and manifest["config"]["smoke"] is args.smoke,
            "analysis manifest configuration drift", failures)

    report = {
        "schema": "head_context_atlas.mechanism_validation_analysis_audit/1",
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": "independent inference reconstruction; no mechanism claim",
        "smoke": args.smoke,
        "validation_loaded": not args.smoke,
        "n_candidate_endpoints": m,
        "n_candidates": len(expected_relations),
        "n_diagnostics": len(expected_diagnostics),
        "n_failures": len(failures),
        "failures": failures,
        "analysis_summary_sha256": sha256_file(paths["summary"]),
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
