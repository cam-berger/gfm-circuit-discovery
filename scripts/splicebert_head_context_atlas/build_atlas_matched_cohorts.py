#!/usr/bin/env python
"""Build outcome-independent matched discovery/validation cohorts for the wider atlas.

Each context is contrasted against packets lacking that focal tag while matching target
role, strand, coding-region class, period-3 distance phase, site strengths, distance,
composition, motif burden, transcript support, packet position, and the *other* context
tags. Context-defining graph/packet variables are not adjusted away.

Matching is within analysis split with a predeclared cap on control reuse inside each
category/role contrast. Cases and controls must come from different genes. The files
contain no model outputs and internal validation is physically separate from discovery.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
CORE_CATEGORIES = (
    "single_pair",
    "donor_cluster",
    "acceptor_cluster",
    "multiple_independent_pairs",
    "cassette_exon",
    "complex_event",
)
CATEGORY_ROLES = {
    "single_pair": ("donor", "acceptor"),
    "donor_cluster": ("donor",),
    "acceptor_cluster": ("acceptor",),
    "multiple_independent_pairs": ("donor", "acceptor"),
    "cassette_exon": ("donor", "acceptor"),
    "complex_event": ("donor", "acceptor"),
}
NUMERIC_FEATURES = (
    "log_target_partner_distance",
    "target_strength",
    "partner_strength",
    "gc_window",
    "gc_target_local50",
    "gc_partner_local50",
    "pyrimidine_target_local50",
    "pyrimidine_partner_local50",
    "log_target_site_support",
    "log_junction_support",
    "target_position_fraction",
    "log_donor_motif_count",
    "log_acceptor_motif_count",
    "max_homopolymer",
    "distance_phase10_63_sin",
    "distance_phase10_63_cos",
)


def stable_rank(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def numeric_vector(row: dict) -> np.ndarray:
    return np.asarray([
        math.log1p(row["target_partner_distance"]),
        row["target_strength"],
        row["partner_strength"],
        row["gc_window"],
        row["gc_target_local50"],
        row["gc_partner_local50"],
        row["pyrimidine_target_local50"],
        row["pyrimidine_partner_local50"],
        math.log1p(row["n_annotated_transcripts_for_target_site"]),
        math.log1p(row["n_annotated_transcripts_for_junction"]),
        row["target_position_fraction"],
        math.log1p(row["donor_motif_count"]),
        math.log1p(row["acceptor_motif_count"]),
        row["max_homopolymer"],
        row["distance_phase10_63_sin"],
        row["distance_phase10_63_cos"],
    ], dtype=float)


def support_bin(value: int) -> str:
    if value <= 4:
        return str(value)
    if value <= 7:
        return "5_7"
    if value <= 12:
        return "8_12"
    if value <= 20:
        return "13_20"
    if value <= 40:
        return "21_40"
    return "41_plus"


def width_bin(value: float, width: float) -> str:
    lower = math.floor(value / width) * width
    return f"{lower:.3f}"


def exact_stratum(row: dict, category: str) -> tuple:
    tags = set(row["tags"])
    other_context_signature = tuple(
        int(tag in tags) for tag in CORE_CATEGORIES if tag != category
    )
    return (
        row["target_role"], row["strand"], row["target_boundary_region"],
        row["distance_mod3"],
        f"{(row['target_partner_distance'] // 50) * 50:03d}",
        support_bin(row["n_annotated_transcripts_for_target_site"]),
        support_bin(row["n_annotated_transcripts_for_junction"]),
        other_context_signature,
    )


def robust_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(matrix, axis=0)
    q25, q75 = np.percentile(matrix, [25, 75], axis=0)
    scale = q75 - q25
    fallback = matrix.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    return center, scale


def match_vector(row: dict, category: str, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    del category  # focal/other context tags are handled exactly by ``exact_stratum``
    return (numeric_vector(row) - center) / scale


def smd(case_values: np.ndarray, control_values: np.ndarray) -> np.ndarray:
    pooled = np.sqrt((case_values.var(axis=0, ddof=1) + control_values.var(axis=0, ddof=1)) / 2)
    pooled = np.where(pooled > 1e-8, pooled, 1.0)
    return (case_values.mean(axis=0) - control_values.mean(axis=0)) / pooled


def balance_stats(pairs: list[dict], by_id: dict[str, dict]) -> dict:
    if not pairs:
        return {
            "numeric_smd": {}, "other_context_tag_smd": {},
            "max_abs_numeric_smd": None, "max_abs_other_tag_smd": None,
            "match_distance_q": {},
        }
    category = pairs[0]["category"]
    case_numeric = np.stack([numeric_vector(by_id[pair["case_packet_id"]]) for pair in pairs])
    control_numeric = np.stack([numeric_vector(by_id[pair["control_packet_id"]]) for pair in pairs])
    numeric_smd = smd(case_numeric, control_numeric)
    other_tags = [tag for tag in CORE_CATEGORIES if tag != category]
    case_tags = np.asarray([
        [float(tag in by_id[pair["case_packet_id"]]["tags"]) for tag in other_tags]
        for pair in pairs
    ])
    control_tags = np.asarray([
        [float(tag in by_id[pair["control_packet_id"]]["tags"]) for tag in other_tags]
        for pair in pairs
    ])
    tag_smd = smd(case_tags, control_tags)
    return {
        "numeric_smd": dict(zip(NUMERIC_FEATURES, map(float, numeric_smd))),
        "other_context_tag_smd": dict(zip(other_tags, map(float, tag_smd))),
        "max_abs_numeric_smd": float(np.max(np.abs(numeric_smd))),
        "max_abs_other_tag_smd": float(np.max(np.abs(tag_smd))) if len(tag_smd) else 0.0,
        "match_distance_q": {
            "0.5": float(np.median([pair["match_distance"] for pair in pairs])),
            "0.9": float(np.percentile([pair["match_distance"] for pair in pairs], 90)),
            "0.99": float(np.percentile([pair["match_distance"] for pair in pairs], 99)),
        },
    }


def prune_to_balance(pairs: list[dict], by_id: dict[str, dict], *, threshold: float,
                     min_retain: int) -> tuple[list[dict], dict]:
    """Outcome-blind cardinality pruning toward the predeclared SMD threshold."""
    kept = list(pairs)
    while len(kept) > min_retain:
        balance = balance_stats(kept, by_id)
        if balance["max_abs_numeric_smd"] <= threshold:
            return kept, balance
        smds = np.asarray([balance["numeric_smd"][name] for name in NUMERIC_FEATURES])
        feature = int(np.argmax(np.abs(smds)))
        direction = 1.0 if smds[feature] > 0 else -1.0
        case_values = np.asarray([
            numeric_vector(by_id[pair["case_packet_id"]])[feature] for pair in kept
        ])
        control_values = np.asarray([
            numeric_vector(by_id[pair["control_packet_id"]])[feature] for pair in kept
        ])
        contribution = direction * (case_values - control_values)
        n_remove = min(max(1, len(kept) // 100), len(kept) - min_retain)
        remove = set(np.argsort(contribution)[-n_remove:])
        kept = [pair for index, pair in enumerate(kept) if index not in remove]
    return kept, balance_stats(kept, by_id)


def select_one_per_gene(rows: list[dict], category: str, role: str, n: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (stable_rank(category, role, row["packet_id"]), row["packet_id"]))
    selected, seen_genes = [], set()
    for row in ordered:
        if row["gene_id"] in seen_genes:
            continue
        selected.append(row)
        seen_genes.add(row["gene_id"])
        if len(selected) >= n:
            break
    return selected


def match_contrast(rows: list[dict], category: str, role: str, n_requested: int,
                   propensity_caliper: float, k_neighbors: int,
                   max_control_reuse: int, balance_threshold: float,
                   min_balanced: int) -> tuple[list[dict], dict]:
    role_rows = [row for row in rows if row["target_role"] == role]
    all_numeric = np.stack([numeric_vector(row) for row in role_rows])
    center, scale = robust_scale(all_numeric)
    standardized = (all_numeric - center) / scale
    cases_available = [row for row in role_rows if category in row["tags"]]
    controls_available = [row for row in role_rows if category not in row["tags"]]

    labels = np.asarray([int(category in row["tags"]) for row in role_rows])
    propensity_model = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=0
    )
    propensity_model.fit(standardized, labels)
    logits = propensity_model.decision_function(standardized)
    logit_sd = float(logits.std(ddof=1)) or 1.0
    logit_by_id = {row["packet_id"]: float(score) for row, score in zip(role_rows, logits)}

    control_by_stratum: dict[tuple, list[dict]] = defaultdict(list)
    for control in controls_available:
        control_by_stratum[exact_stratum(control, category)].append(control)
    control_logits = {}
    for stratum, controls in control_by_stratum.items():
        controls.sort(key=lambda row: (logit_by_id[row["packet_id"]], row["packet_id"]))
        control_logits[stratum] = [logit_by_id[row["packet_id"]] for row in controls]
    cases_common_support = [
        case for case in cases_available
        if control_by_stratum.get(exact_stratum(case, category), set())
        and any(
            control["gene_id"] != case["gene_id"]
            for control in control_by_stratum[exact_stratum(case, category)]
        )
        and control_logits[exact_stratum(case, category)][0] - propensity_caliper * logit_sd
            <= logit_by_id[case["packet_id"]]
            <= control_logits[exact_stratum(case, category)][-1] + propensity_caliper * logit_sd
    ]
    cases = select_one_per_gene(cases_common_support, category, role, n_requested)

    control_use = Counter()
    pairs = []
    failures = Counter()
    for case in cases:
        stratum = exact_stratum(case, category)
        controls = control_by_stratum.get(stratum, [])
        if not controls:
            failures["no_exact_stratum"] += 1
            continue
        scores = control_logits[stratum]
        case_logit = logit_by_id[case["packet_id"]]
        insertion = bisect.bisect_left(scores, case_logit)
        radius = min(max(k_neighbors // 2, 1), len(controls))
        candidate_indices = range(max(0, insertion - radius), min(len(controls), insertion + radius))
        candidates = sorted(
            ((abs(scores[index] - case_logit), index) for index in candidate_indices),
            key=lambda item: (item[0], controls[item[1]]["packet_id"]),
        )
        chosen = None
        for raw_distance, index in candidates:
            control = controls[int(index)]
            if raw_distance > propensity_caliper * logit_sd:
                break
            if control["gene_id"] == case["gene_id"]:
                continue
            if control_use[control["packet_id"]] >= max_control_reuse:
                continue
            chosen = (control, float(raw_distance / logit_sd))
            break
        if chosen is None:
            failures["caliper_or_reuse"] += 1
            continue
        control, distance = chosen
        control_use[control["packet_id"]] += 1
        pairs.append({
            "pair_id": f"{category}:{role}:{len(pairs):05d}",
            "category": category,
            "target_role": role,
            "analysis_split": case["analysis_split"],
            "case_packet_id": case["packet_id"],
            "case_gene_id": case["gene_id"],
            "control_packet_id": control["packet_id"],
            "control_gene_id": control["gene_id"],
            "match_distance": distance,
            "control_use_ordinal": control_use[control["packet_id"]],
            "exact_stratum": list(stratum),
        })

    by_id = {row["packet_id"]: row for row in role_rows}
    raw_pairs = len(pairs)
    pairs, balance = prune_to_balance(
        pairs, by_id, threshold=balance_threshold, min_retain=min(min_balanced, len(pairs))
    )
    summary = {
        "category": category,
        "target_role": role,
        "n_requested": n_requested,
        "n_case_packets_available": len(cases_available),
        "n_case_genes_available": len({row["gene_id"] for row in cases_available}),
        "n_case_packets_in_exact_other-tag_support": len(cases_common_support),
        "n_case_genes_in_exact_other-tag_support": len({row["gene_id"] for row in cases_common_support}),
        "n_controls_available": len(controls_available),
        "n_cases_sampled": len(cases),
        "n_matched_before_balance_pruning": raw_pairs,
        "n_pruned_for_balance": raw_pairs - len(pairs),
        "n_matched": len(pairs),
        "failures": dict(failures),
        "max_control_reuse": max(Counter(pair["control_packet_id"] for pair in pairs).values(), default=0),
        "n_unique_controls": len({pair["control_packet_id"] for pair in pairs}),
        "balance": balance,
        "balance_gate": bool(
            pairs
            and balance["max_abs_numeric_smd"] <= balance_threshold
            and balance["max_abs_other_tag_smd"] <= 0.15
            and len(pairs) >= min_balanced
        ),
        "propensity": {
            "model": "balanced logistic regression on standardized numeric confounds",
            "caliper_sd": propensity_caliper,
            "logit_sd": logit_sd,
        },
    }
    return pairs, summary


def write_selected(source: Path, selected: dict[str, set[str]], outputs: dict[str, Path]) -> None:
    handles = {split: outputs[split].open("w") for split in outputs}
    try:
        with source.open() as input_handle:
            for line in input_handle:
                row = json.loads(line)
                split = row["analysis_split"]
                if split in selected and row["packet_id"] in selected[split]:
                    handles[split].write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        for handle in handles.values():
            handle.close()


def write_selected_sequences(source: Path, selected_sequence_ids: dict[str, set[str]],
                             outputs: dict[str, Path]) -> None:
    handles = {split: outputs[split].open("w") for split in outputs}
    try:
        with source.open() as input_handle:
            for line in input_handle:
                row = json.loads(line)
                for split, ids in selected_sequence_ids.items():
                    if row["sequence_id"] in ids:
                        handles[split].write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        for handle in handles.values():
            handle.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_packets_enriched_v1.jsonl"
    ))
    ap.add_argument("--sequences", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_sequences_v1.jsonl"
    ))
    ap.add_argument("--prefix", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_matched_v1"
    ))
    ap.add_argument("--n-discovery", type=int, default=800, help="per category x target role")
    ap.add_argument("--n-validation", type=int, default=400, help="per category x target role")
    ap.add_argument("--propensity-caliper", type=float, default=0.2)
    ap.add_argument("--k-neighbors", type=int, default=128)
    ap.add_argument("--max-control-reuse", type=int, default=2)
    ap.add_argument("--balance-threshold", type=float, default=0.10)
    ap.add_argument("--min-balanced-discovery", type=int, default=400)
    ap.add_argument("--min-balanced-validation", type=int, default=150)
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    packet_path = Path(args.packets)
    sequence_path = Path(args.sequences)
    prefix = Path(args.prefix)
    paths = {}
    for short, analysis_split in (
        ("discovery", "atlas_discovery"),
        ("validation", "atlas_internal_validation"),
    ):
        paths[(short, "pairs")] = Path(f"{prefix}.{short}.pairs.jsonl")
        paths[(short, "packets")] = Path(f"{prefix}.{short}.packets.jsonl")
        paths[(short, "sequences")] = Path(f"{prefix}.{short}.sequences.jsonl")
    summary_path = Path(f"{prefix}.summary.json")
    manifest_path = Path(f"{prefix}.manifest.json")
    for path in list(paths.values()) + [summary_path, manifest_path]:
        if path.exists():
            raise FileExistsError(f"refuse to overwrite immutable artifact {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    compact = defaultdict(list)
    with packet_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["analysis_split"] in {"atlas_discovery", "atlas_internal_validation"}:
                compact[row["analysis_split"]].append(row)
    print(
        f"[atlas-match] discovery={len(compact['atlas_discovery'])} validation={len(compact['atlas_internal_validation'])}",
        file=sys.stderr,
    )

    all_pairs = {}
    summaries = {}
    selected_packet_ids = {}
    selected_sequence_ids = {}
    split_specs = (
        ("discovery", "atlas_discovery", args.n_discovery, args.min_balanced_discovery),
        ("validation", "atlas_internal_validation", args.n_validation, args.min_balanced_validation),
    )
    for short, analysis_split, n_requested, min_balanced in split_specs:
        rows = compact[analysis_split]
        pairs = []
        split_summary = {}
        for category in CORE_CATEGORIES:
            for role in CATEGORY_ROLES[category]:
                block, block_summary = match_contrast(
                    rows, category, role, n_requested,
                    propensity_caliper=args.propensity_caliper, k_neighbors=args.k_neighbors,
                    max_control_reuse=args.max_control_reuse,
                    balance_threshold=args.balance_threshold, min_balanced=min_balanced,
                )
                pairs.extend(block)
                split_summary[f"{category}|{role}"] = block_summary
                print(
                    f"[atlas-match] {short} {category}|{role}: {len(block)}/{n_requested} "
                    f"balance={block_summary['balance_gate']}",
                    file=sys.stderr,
                )
        all_pairs[analysis_split] = pairs
        summaries[analysis_split] = split_summary
        ids = {
            packet_id for pair in pairs
            for packet_id in (pair["case_packet_id"], pair["control_packet_id"])
        }
        selected_packet_ids[analysis_split] = ids
        by_id = {row["packet_id"]: row for row in rows}
        selected_sequence_ids[analysis_split] = {by_id[packet_id]["sequence_id"] for packet_id in ids}
        with paths[(short, "pairs")].open("w") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, sort_keys=True) + "\n")

    packet_outputs = {
        "atlas_discovery": paths[("discovery", "packets")],
        "atlas_internal_validation": paths[("validation", "packets")],
    }
    sequence_outputs = {
        "atlas_discovery": paths[("discovery", "sequences")],
        "atlas_internal_validation": paths[("validation", "sequences")],
    }
    write_selected(packet_path, selected_packet_ids, packet_outputs)
    write_selected_sequences(sequence_path, selected_sequence_ids, sequence_outputs)

    balance_failures = {
        split: [key for key, block in summary.items() if not block["balance_gate"]]
        for split, summary in summaries.items()
    }
    report = {
        "schema": "head_context_atlas.matched_cohorts/1",
        "status": "OUTCOME_INDEPENDENT_MATCHED_CANDIDATE",
        "categories": list(CORE_CATEGORIES),
        "matching": {
            "exact": [
                "analysis_split", "target_role", "strand", "target_boundary_region", "distance_mod3",
                "50-nt target-partner distance band", "target-site transcript-support band",
                "junction transcript-support band", "all non-focal core context tags"
            ],
            "continuous": list(NUMERIC_FEATURES),
            "other_context_tags": "all non-focal core tags matched exactly",
            "excluded_as_context_defining": [
                "visible_pair_count", "donor_degree", "acceptor_degree", "local_edge_count"
            ],
            "different_gene": True,
            "control_reuse_within_contrast": f"capped at {args.max_control_reuse}; repeated controls require analysis weights",
            "propensity": "balanced logistic regression over all continuous confounds",
            "propensity_caliper_sd": args.propensity_caliper,
            "k_neighbors": args.k_neighbors,
            "balance_gate": "max |numeric SMD| <=0.10 and max |other-tag SMD| <=0.15",
            "cardinality_pruning": (
                f"outcome-blind removal to numeric SMD <= {args.balance_threshold}; minimum retained "
                f"discovery={args.min_balanced_discovery}, validation={args.min_balanced_validation}"
            ),
        },
        "summaries": summaries,
        "selected": {
            split: {
                "n_pairs": len(all_pairs[split]),
                "n_unique_packets": len(selected_packet_ids[split]),
                "n_unique_sequences": len(selected_sequence_ids[split]),
                "n_genes": len({
                    row["gene_id"] for row in compact[split]
                    if row["packet_id"] in selected_packet_ids[split]
                }),
            }
            for split in all_pairs
        },
        "balance_failures": balance_failures,
        "global_balance_gate": not any(balance_failures.values()),
        "split_guard": (
            "discovery and internal validation are gene-hash-disjoint and physically separate; "
            "chr11/16/19 and chr8/9 are absent"
        ),
        "claim_boundary": "matching and cohort feasibility only; no model outcome or head function read",
    }
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    sys.path.insert(0, str(REPO / "src"))
    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_matched_cohorts_v1",
        command=["python"] + sys.argv,
        status="OUTCOME_INDEPENDENT_MATCHED_CANDIDATE",
    )
    for path in (packet_path, sequence_path):
        manifest.add_input(str(path))
    for path in list(paths.values()) + [summary_path]:
        manifest.add_output(str(path))
    manifest.write(str(manifest_path))
    print(json.dumps({
        "selected": report["selected"],
        "balance_failures": balance_failures,
        "global_balance_gate": report["global_balance_gate"],
    }, indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
