#!/usr/bin/env python
"""Analyze the merged wider-atlas discovery matrix under ANALYSIS_PLAN_V1.

Produces gene-clustered general-use summaries, two-endpoint gene-dyadic matched
context contrasts, FDR-adjusted classifications, and a deterministic mechanism
shortlist.  This script has no internal-validation or sealed-data input.
"""
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

PLAN = REPO / "studies/splicebert_head_context_atlas/ANALYSIS_PLAN_V1.md"
AMENDMENT = REPO / "studies/splicebert_head_context_atlas/ANALYSIS_AMENDMENT_A1.md"
ESTIMANDS = {
    "A": "A_effect_margin",
    "B": "B_effect_margin",
}
HEALTH_METRICS = {
    "A": "A_health_margin_damage",
    "B": "B_health_margin_damage",
}
HEALTH_LOGPROB_METRICS = {
    "A": "A_health_logprob_damage",
    "B": "B_health_logprob_damage",
}
EXPECTED_COMPONENTS = tuple(
    [f"L{layer}H{head}" for layer in range(6) for head in range(16)]
    + [f"L{layer}MLP" for layer in range(6)]
)
EXPECTED_COMPONENT_SET = set(EXPECTED_COMPONENTS)
EXPECTED_CONTRASTS = {
    "single_pair|donor", "single_pair|acceptor",
    "donor_cluster|donor", "acceptor_cluster|acceptor",
    "multiple_independent_pairs|donor", "multiple_independent_pairs|acceptor",
    "cassette_exon|donor", "cassette_exon|acceptor",
    "complex_event|donor", "complex_event|acceptor",
}
Z_POWER_80 = float(norm.ppf(0.975) + norm.ppf(0.80))


def stable_rank(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    n = len(pvalues)
    if n == 0:
        return pvalues.copy()
    order = np.argsort(pvalues, kind="mergesort")
    ranked = pvalues[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(n, dtype=float)
    result[order] = np.clip(ranked, 0.0, 1.0)
    return result


def _pvalue(point: np.ndarray, se: np.ndarray) -> np.ndarray:
    p = np.ones_like(point, dtype=float)
    positive_se = se > 0
    p[positive_se] = 2.0 * norm.sf(np.abs(point[positive_se] / se[positive_se]))
    p[~positive_se & (np.abs(point) > 0)] = 0.0
    return p


def gene_cluster_stats(values: np.ndarray, genes: list[str]) -> dict[str, np.ndarray | int]:
    """Packet mean with one-way gene-clustered sandwich uncertainty."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(genes):
        raise ValueError("values must be [packets, components] and align to genes")
    n, n_components = values.shape
    if n == 0:
        raise ValueError("empty packet block")
    point = values.mean(axis=0)
    residual = values - point
    unique_genes = sorted(set(genes))
    gene_index = {gene: index for index, gene in enumerate(unique_genes)}
    cluster_scores = np.zeros((len(unique_genes), n_components), dtype=float)
    np.add.at(cluster_scores, [gene_index[gene] for gene in genes], residual)
    meat = np.square(cluster_scores).sum(axis=0)
    correction = len(unique_genes) / max(len(unique_genes) - 1, 1)
    se = np.sqrt(np.maximum(meat * correction, 0.0)) / n
    return {
        "point": point,
        "se": se,
        "low": point - 1.96 * se,
        "high": point + 1.96 * se,
        "p": _pvalue(point, se),
        "mde80": Z_POWER_80 * se,
        "n": n,
        "n_genes": len(unique_genes),
    }


def dyadic_stats(values: np.ndarray, case_genes: list[str], control_genes: list[str],
                  weights: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Weighted mean with two-endpoint gene-dyadic sandwich uncertainty.

    Each pair contributes its weighted residual score to both endpoint genes.  Subtracting
    one observation-level score square leaves a single diagonal contribution, while
    retaining cross-pair covariance whenever either endpoint gene is shared.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(case_genes) or len(case_genes) != len(control_genes):
        raise ValueError("dyadic arrays do not align")
    if len(weights) != values.shape[0] or np.any(weights <= 0):
        raise ValueError("weights must be positive and align to pairs")
    if any(case == control for case, control in zip(case_genes, control_genes)):
        raise ValueError("dyadic pair cannot use the same endpoint gene twice")
    endpoint_dyads = [tuple(sorted((case, control))) for case, control in zip(case_genes, control_genes)]
    if len(endpoint_dyads) != len(set(endpoint_dyads)):
        raise ValueError("repeated endpoint-gene dyad violates the frozen v1 variance formula")
    total_weight = weights.sum()
    point = np.sum(values * weights[:, None], axis=0) / total_weight
    score = weights[:, None] * (values - point)
    unique_genes = sorted(set(case_genes) | set(control_genes))
    gene_index = {gene: index for index, gene in enumerate(unique_genes)}
    incident = np.zeros((len(unique_genes), values.shape[1]), dtype=float)
    np.add.at(incident, [gene_index[gene] for gene in case_genes], score)
    np.add.at(incident, [gene_index[gene] for gene in control_genes], score)
    meat = np.square(incident).sum(axis=0) - np.square(score).sum(axis=0)
    correction = len(unique_genes) / max(len(unique_genes) - 1, 1)
    se = np.sqrt(np.maximum(meat * correction, 0.0)) / total_weight
    return {
        "point": point,
        "se": se,
        "low": point - 1.96 * se,
        "high": point + 1.96 * se,
        "p": _pvalue(point, se),
        "mde80": Z_POWER_80 * se,
        "n_pairs": values.shape[0],
        "n_endpoint_genes": len(unique_genes),
        "n_effective": float(total_weight ** 2 / np.square(weights).sum()),
        "total_weight": float(total_weight),
    }


def classify(point: float, low: float, high: float, qvalue: float, tau: float) -> str:
    if qvalue <= 0.05 and low > tau:
        return "material_support"
    if qvalue <= 0.05 and high < -tau:
        return "material_suppress"
    if qvalue <= 0.05 and (low > 0 or high < 0):
        return "detected_subfloor"
    if low >= -tau and high <= tau:
        return "bounded_small"
    return "inconclusive"


def gene_half(gene: str) -> int:
    return stable_rank("atlas-stability-v1", gene) % 2


def load_matrix(path: Path, metrics: dict[str, str]) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Load a packet-grouped gzip matrix into compact packet x component arrays."""
    component_index = {component: index for index, component in enumerate(EXPECTED_COMPONENTS)}
    packet_meta: list[dict] = []
    matrix_lists = {name: [] for name in metrics}
    current_packet = None
    current_meta = None
    current_values = None
    seen_components: set[str] = set()

    def finish_packet() -> None:
        nonlocal current_packet, current_meta, current_values, seen_components
        if current_packet is None:
            return
        if seen_components != EXPECTED_COMPONENT_SET:
            missing = sorted(EXPECTED_COMPONENT_SET - seen_components)
            extra = sorted(seen_components - EXPECTED_COMPONENT_SET)
            raise RuntimeError(f"component grid mismatch for {current_packet}: missing={missing}, extra={extra}")
        packet_meta.append(current_meta)
        for name in metrics:
            if not np.isfinite(current_values[name]).all():
                raise RuntimeError(f"non-finite/incomplete {name} for {current_packet}")
            matrix_lists[name].append(current_values[name])
        current_packet = None
        current_meta = None
        current_values = None
        seen_components = set()

    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            packet_id = row["packet_id"]
            if packet_id != current_packet:
                finish_packet()
                current_packet = packet_id
                current_meta = {
                    key: row[key] for key in (
                        "packet_id", "sequence_id", "gene_id", "chrom", "analysis_split",
                        "target_role", "tags",
                    )
                }
                current_values = {
                    name: np.full(len(EXPECTED_COMPONENTS), np.nan, dtype=np.float64)
                    for name in metrics
                }
            component = row["component_id"]
            if component not in component_index or component in seen_components:
                raise RuntimeError(f"unknown/duplicate component {component} for {packet_id}")
            seen_components.add(component)
            index = component_index[component]
            for name, field in metrics.items():
                current_values[name][index] = float(row[field])
        finish_packet()
    return packet_meta, {name: np.stack(rows) for name, rows in matrix_lists.items()}


def read_pairs(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=json_value) + "\n")


def build_general_rows(packet_meta: list[dict], target: dict[str, np.ndarray],
                       health_meta: list[dict], health: dict[str, np.ndarray],
                       tau: float) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    health_lookup: dict[tuple[str, str], dict] = {}
    for role in ("donor", "acceptor"):
        hidx = [index for index, meta in enumerate(health_meta) if meta["target_role"] == role]
        if not hidx:
            continue
        hgenes = [health_meta[index]["gene_id"] for index in hidx]
        for estimand in ESTIMANDS:
            margin_stats = gene_cluster_stats(health[estimand][hidx], hgenes)
            logprob_stats = gene_cluster_stats(health[f"{estimand}_logprob"][hidx], hgenes)
            for component_index, component in enumerate(EXPECTED_COMPONENTS):
                health_lookup[(role, estimand, component)] = {
                    "health_mean_margin_damage": margin_stats["point"][component_index],
                    "health_ci95_margin_damage": [
                        margin_stats["low"][component_index], margin_stats["high"][component_index]
                    ],
                    "health_mean_logprob_damage": logprob_stats["point"][component_index],
                    "health_ci95_logprob_damage": [
                        logprob_stats["low"][component_index], logprob_stats["high"][component_index]
                    ],
                    "health_n": margin_stats["n"],
                    "health_n_genes": margin_stats["n_genes"],
                }

    for estimand in ESTIMANDS:
        family_indices = []
        for role in ("donor", "acceptor"):
            indices = [index for index, meta in enumerate(packet_meta) if meta["target_role"] == role]
            genes = [packet_meta[index]["gene_id"] for index in indices]
            stats = gene_cluster_stats(target[estimand][indices], genes)
            for component_index, component in enumerate(EXPECTED_COMPONENTS):
                h = health_lookup.get((role, estimand, component), {})
                row = {
                    "estimand": estimand,
                    "target_role": role,
                    "component_id": component,
                    "component_type": "mlp_write" if component.endswith("MLP") else "attention_head",
                    "mean": stats["point"][component_index],
                    "se_gene_cluster": stats["se"][component_index],
                    "ci95": [stats["low"][component_index], stats["high"][component_index]],
                    "p": stats["p"][component_index],
                    "mde80": stats["mde80"][component_index],
                    "n_packets": stats["n"],
                    "n_genes": stats["n_genes"],
                    "tau": tau,
                    **h,
                }
                rows.append(row)
                family_indices.append(len(rows) - 1)
        qvalues = bh_adjust(np.asarray([rows[index]["p"] for index in family_indices]))
        for index, qvalue in zip(family_indices, qvalues):
            row = rows[index]
            row["q_bh_within_estimand"] = float(qvalue)
            row["classification"] = classify(
                row["mean"], row["ci95"][0], row["ci95"][1], float(qvalue), tau
            )
            hci = row.get("health_ci95_margin_damage")
            row["broad_MLM_flag"] = bool(
                hci and (hci[0] > tau or hci[1] < -tau)
            )

    by_profile: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_profile[(row["estimand"], row["component_id"])][row["target_role"]] = row
    profiles = []
    for (estimand, component), role_rows in sorted(by_profile.items()):
        donor, acceptor = role_rows.get("donor"), role_rows.get("acceptor")
        labels = {row["classification"] for row in role_rows.values()}
        if donor and acceptor and labels == {"material_support"}:
            profile = "cross_role_general_support"
            score = min(donor["ci95"][0] - tau, acceptor["ci95"][0] - tau)
        elif donor and acceptor and labels == {"material_suppress"}:
            profile = "cross_role_general_suppress"
            score = min(-tau - donor["ci95"][1], -tau - acceptor["ci95"][1])
        elif donor and acceptor and {
            donor["classification"], acceptor["classification"]
        } == {"material_support", "material_suppress"}:
            profile = "role_opposed"
            score = min(
                abs(donor["mean"]) - tau, abs(acceptor["mean"]) - tau
            )
        elif any(label in {"material_support", "material_suppress"} for label in labels):
            profile = "role_associated"
            score = max(abs(row["mean"]) - tau for row in role_rows.values())
        else:
            profile = "unresolved"
            score = None
        profiles.append({
            "estimand": estimand,
            "component_id": component,
            "component_type": "mlp_write" if component.endswith("MLP") else "attention_head",
            "profile": profile,
            "score_bound_excess": score,
            "donor": donor,
            "acceptor": acceptor,
            "general_shortlist_eligible": bool(
                profile in {"cross_role_general_support", "cross_role_general_suppress"}
                and donor["mde80"] <= tau and acceptor["mde80"] <= tau
                and donor.get("health_n", 0) > 0 and acceptor.get("health_n", 0) > 0
            ),
        })
    return rows, profiles


def build_context_rows(pairs: list[dict], packet_meta: list[dict],
                       target: dict[str, np.ndarray], tau: float,
                       min_stability_pairs: int) -> list[dict]:
    packet_index = {meta["packet_id"]: index for index, meta in enumerate(packet_meta)}
    blocks: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for pair in pairs:
        if pair["case_packet_id"] not in packet_index or pair["control_packet_id"] not in packet_index:
            raise RuntimeError(f"pair {pair['pair_id']} cannot join target matrix")
        blocks[(pair["category"], pair["target_role"])].append(pair)
    rows: list[dict] = []
    family_indices: dict[str, list[int]] = defaultdict(list)
    for (category, role), block in sorted(blocks.items()):
        case_idx = np.asarray([packet_index[pair["case_packet_id"]] for pair in block])
        control_idx = np.asarray([packet_index[pair["control_packet_id"]] for pair in block])
        case_genes = [pair["case_gene_id"] for pair in block]
        control_genes = [pair["control_gene_id"] for pair in block]
        reuse = Counter(pair["control_packet_id"] for pair in block)
        weights = np.asarray([1.0 / reuse[pair["control_packet_id"]] for pair in block])
        halves = np.asarray([
            gene_half(case) if gene_half(case) == gene_half(control) else -1
            for case, control in zip(case_genes, control_genes)
        ])
        for estimand in ESTIMANDS:
            case_values = target[estimand][case_idx]
            control_values = target[estimand][control_idx]
            difference = case_values - control_values
            stats = dyadic_stats(difference, case_genes, control_genes, weights)
            case_mean = np.average(case_values, axis=0, weights=weights)
            control_mean = np.average(control_values, axis=0, weights=weights)
            half_points = []
            half_counts = []
            for half in (0, 1):
                keep = halves == half
                half_counts.append(int(keep.sum()))
                if keep.any():
                    half_points.append(np.average(difference[keep], axis=0, weights=weights[keep]))
                else:
                    half_points.append(np.full(len(EXPECTED_COMPONENTS), np.nan))
            for component_index, component in enumerate(EXPECTED_COMPONENTS):
                point = float(stats["point"][component_index])
                support = all(count >= min_stability_pairs for count in half_counts)
                direction = bool(
                    support and point != 0
                    and np.sign(half_points[0][component_index]) == np.sign(point)
                    and np.sign(half_points[1][component_index]) == np.sign(point)
                )
                row = {
                    "category": category,
                    "target_role": role,
                    "contrast": f"{category}|{role}",
                    "estimand": estimand,
                    "component_id": component,
                    "component_type": "mlp_write" if component.endswith("MLP") else "attention_head",
                    "mean_case": float(case_mean[component_index]),
                    "mean_control": float(control_mean[component_index]),
                    "mean_difference": point,
                    "se_gene_dyadic": float(stats["se"][component_index]),
                    "ci95": [
                        float(stats["low"][component_index]),
                        float(stats["high"][component_index]),
                    ],
                    "p": float(stats["p"][component_index]),
                    "mde80": float(stats["mde80"][component_index]),
                    "n_pairs": stats["n_pairs"],
                    "n_endpoint_genes": stats["n_endpoint_genes"],
                    "n_effective_pairs": stats["n_effective"],
                    "max_control_reuse": max(reuse.values()),
                    "stability_half_counts": half_counts,
                    "stability_half_means": [
                        float(half_points[0][component_index]) if half_counts[0] else None,
                        float(half_points[1][component_index]) if half_counts[1] else None,
                    ],
                    "stability_support": support,
                    "stability_direction_pass": direction,
                    "tau": tau,
                }
                rows.append(row)
                family_indices[estimand].append(len(rows) - 1)
    for estimand, indices in family_indices.items():
        qvalues = bh_adjust(np.asarray([rows[index]["p"] for index in indices]))
        for index, qvalue in zip(indices, qvalues):
            row = rows[index]
            row["q_bh_within_estimand"] = float(qvalue)
            row["classification"] = classify(
                row["mean_difference"], row["ci95"][0], row["ci95"][1], float(qvalue), tau
            )
            row["context_associated"] = bool(
                row["classification"] in {"material_support", "material_suppress"}
                and row["stability_direction_pass"]
            )
            if row["classification"] == "material_support":
                row["score_bound_excess"] = row["ci95"][0] - tau
            elif row["classification"] == "material_suppress":
                row["score_bound_excess"] = -tau - row["ci95"][1]
            else:
                row["score_bound_excess"] = None
    return rows


def build_shortlist(general_profiles: list[dict], context_rows: list[dict],
                    tau: float, min_context_neff: float) -> dict:
    general_candidates = [row for row in general_profiles if row["general_shortlist_eligible"]]
    general_candidates.sort(key=lambda row: (
        -float(row["score_bound_excess"]), stable_rank("general", row["component_id"], row["estimand"])
    ))
    general_selected = []
    seen_components = set()
    for row in general_candidates:
        if row["component_id"] in seen_components:
            continue
        general_selected.append(row)
        seen_components.add(row["component_id"])
        if len(general_selected) == 4:
            break

    context_candidates = [
        row for row in context_rows
        if row["context_associated"]
        and row["n_effective_pairs"] >= min_context_neff
        and row["mde80"] <= tau
    ]
    context_candidates.sort(key=lambda row: (
        -float(row["score_bound_excess"]),
        stable_rank("context", row["contrast"], row["estimand"], row["component_id"]),
    ))
    context_selected = []
    contrast_count = Counter()
    component_count = Counter()

    def can_add(row: dict) -> bool:
        return contrast_count[row["contrast"]] < 2 and component_count[row["component_id"]] < 3

    eligible_roles = {row["target_role"] for row in context_candidates}
    for role in ("donor", "acceptor"):
        if role not in eligible_roles:
            continue
        row = next((item for item in context_candidates if item["target_role"] == role and can_add(item)), None)
        if row is not None:
            context_selected.append(row)
            contrast_count[row["contrast"]] += 1
            component_count[row["component_id"]] += 1
    for row in context_candidates:
        if len(context_selected) >= 8:
            break
        if row in context_selected or not can_add(row):
            continue
        context_selected.append(row)
        contrast_count[row["contrast"]] += 1
        component_count[row["component_id"]] += 1

    l3 = next((
        row for row in general_profiles
        if row["component_id"] == "L3H1" and row["estimand"] == "B"
    ), None)
    return {
        "schema": "head_context_atlas.discovery_shortlist/1",
        "status": "DISCOVERY_FROZEN_CANDIDATES",
        "tau": tau,
        "general": {
            "n_eligible": len(general_candidates),
            "eligible": general_candidates,
            "selected": general_selected,
            "maximum": 4,
        },
        "context": {
            "n_eligible": len(context_candidates),
            "eligible": context_candidates,
            "selected": context_selected,
            "maximum": 8,
            "max_per_contrast": 2,
            "max_per_component": 3,
            "minimum_effective_pairs": min_context_neff,
        },
        "positive_control_L3H1_B": l3,
        "claim_boundary": (
            "discovery shortlist for exact mechanism tests; not a validated head-function atlas"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-prefix", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--min-stability-pairs", type=int, default=40)
    parser.add_argument("--min-context-neff", type=float, default=80.0)
    parser.add_argument("--allow-incomplete-contexts", action="store_true", help="smoke artifacts only")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    if args.tau <= 0:
        raise ValueError("tau must be positive")

    merged = Path(args.merged_prefix)
    output = Path(args.out_prefix)
    input_paths = {
        "target": Path(f"{merged}.target_matrix.jsonl.gz"),
        "health": Path(f"{merged}.health_matrix.jsonl.gz"),
        "pairs": Path(f"{merged}.pairs.jsonl.gz"),
        "summary": Path(f"{merged}.summary.json"),
    }
    output_paths = {
        "general": Path(f"{output}.general.jsonl"),
        "profiles": Path(f"{output}.profiles.jsonl"),
        "context": Path(f"{output}.context.jsonl"),
        "shortlist": Path(f"{output}.shortlist.json"),
        "summary": Path(f"{output}.summary.json"),
        "manifest": Path(f"{output}.manifest.json"),
    }
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(f"refuse to overwrite immutable artifact {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    merged_summary = json.loads(input_paths["summary"].read_text())
    if merged_summary["status"] != "COMPLETE" or merged_summary["n_components"] != 102:
        raise RuntimeError("merged input is not a complete 102-component matrix")
    packet_meta, target = load_matrix(input_paths["target"], ESTIMANDS)
    health_fields = {
        **HEALTH_METRICS,
        **{f"{key}_logprob": field for key, field in HEALTH_LOGPROB_METRICS.items()},
    }
    health_meta, health = load_matrix(input_paths["health"], health_fields)
    pairs = read_pairs(input_paths["pairs"])
    contrasts = {f"{pair['category']}|{pair['target_role']}" for pair in pairs}
    if not args.allow_incomplete_contexts and contrasts != EXPECTED_CONTRASTS:
        raise RuntimeError(
            f"full analysis requires all ten contrasts; missing={sorted(EXPECTED_CONTRASTS-contrasts)}"
        )
    if any(meta["analysis_split"] != "atlas_discovery" for meta in packet_meta + health_meta):
        raise RuntimeError("non-discovery packet reached discovery analyzer")

    general_rows, profiles = build_general_rows(
        packet_meta, target, health_meta, health, args.tau
    )
    context_rows = build_context_rows(
        pairs, packet_meta, target, args.tau, args.min_stability_pairs
    )
    shortlist = build_shortlist(profiles, context_rows, args.tau, args.min_context_neff)
    if args.allow_incomplete_contexts:
        shortlist["status"] = "SMOKE_NOT_FOR_SELECTION"
        shortlist["general"]["selected"] = []
        shortlist["context"]["selected"] = []
    shortlist["analysis_plan"] = str(PLAN)
    shortlist["analysis_plan_sha256"] = sha256_file(PLAN)
    shortlist["analysis_amendment"] = str(AMENDMENT)
    shortlist["analysis_amendment_sha256"] = sha256_file(AMENDMENT)
    shortlist["input_hashes"] = {
        name: sha256_file(path) for name, path in input_paths.items()
    }

    write_jsonl(output_paths["general"], general_rows)
    write_jsonl(output_paths["profiles"], profiles)
    write_jsonl(output_paths["context"], context_rows)
    output_paths["shortlist"].write_text(
        json.dumps(shortlist, indent=2, sort_keys=True, default=json_value) + "\n"
    )
    summary = {
        "schema": "head_context_atlas.discovery_analysis/1",
        "status": "SMOKE" if args.allow_incomplete_contexts else "DISCOVERY_COMPLETE",
        "claim_boundary": "discovery classification and mechanism shortlist; not validated function",
        "tau": args.tau,
        "n_packets": len(packet_meta),
        "n_health_packets": len(health_meta),
        "n_pairs": len(pairs),
        "contrasts": sorted(contrasts),
        "n_general_rows": len(general_rows),
        "n_profiles": len(profiles),
        "n_context_rows": len(context_rows),
        "general_classifications": dict(Counter(row["classification"] for row in general_rows)),
        "profile_classifications": dict(Counter(row["profile"] for row in profiles)),
        "context_classifications": dict(Counter(row["classification"] for row in context_rows)),
        "n_general_shortlist": len(shortlist["general"]["selected"]),
        "n_context_shortlist": len(shortlist["context"]["selected"]),
        "analysis_plan_sha256": shortlist["analysis_plan_sha256"],
        "analysis_amendment_sha256": shortlist["analysis_amendment_sha256"],
    }
    output_paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_discovery_analysis_v1",
        command=["python"] + sys.argv,
        status=summary["status"],
    )
    manifest.config.update({
        "tau": args.tau,
        "min_stability_pairs": args.min_stability_pairs,
        "min_context_neff": args.min_context_neff,
        "analysis_plan_sha256": shortlist["analysis_plan_sha256"],
        "analysis_amendment_sha256": shortlist["analysis_amendment_sha256"],
    })
    for path in [*input_paths.values(), PLAN, AMENDMENT]:
        manifest.add_input(str(path))
    for name, path in output_paths.items():
        if name != "manifest":
            manifest.add_output(str(path))
    manifest.write(str(output_paths["manifest"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
