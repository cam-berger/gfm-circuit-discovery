#!/usr/bin/env python
"""Freeze candidate-specific mechanism instructions before validation access.

The builder consumes only audited discovery artifacts.  It never opens the
internal-validation cohort; it records the fixed later filenames and selection
rules as sealed references.  A smoke analysis can exercise the contracts but is
forced to emit an empty candidate package.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

ANALYSIS_PLAN = REPO / "studies/splicebert_head_context_atlas/ANALYSIS_PLAN_V1.md"
ANALYSIS_AMENDMENT = REPO / "studies/splicebert_head_context_atlas/ANALYSIS_AMENDMENT_A1.md"
MECHANISM_PLAN = REPO / "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_PLAN_V1.md"
MECHANISM_AMENDMENT = REPO / "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A1.md"
MECHANISM_AMENDMENT_A2 = REPO / "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A2.md"
ESTIMAND_FIELD = {"A": "A_effect_margin", "B": "B_effect_margin"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_repo_path(path: Path) -> str:
    """Use a repository-relative scientific reference when the artifact is in-repo."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO.resolve()))
    except ValueError:
        return str(resolved)


def stable_rank(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def component_type(component: str) -> str:
    return "mlp_write" if component.endswith("MLP") else "attention_head"


def component_layer(component: str) -> int:
    if not component.startswith("L"):
        raise ValueError(f"invalid component label {component}")
    marker = "MLP" if component.endswith("MLP") else "H"
    return int(component[1:].split(marker, 1)[0])


def general_importance(profile: dict) -> float:
    return float((abs(profile["donor"]["mean"]) + abs(profile["acceptor"]["mean"])) / 2)


def choose_importance_control(candidate: dict, pool: list[dict],
                              excluded_components: set[str], *, context: bool) -> dict:
    """Nearest effect-size control, same type and same layer for heads when possible."""
    component = candidate["component_id"]
    ctype = component_type(component)
    options = [
        row for row in pool
        if component_type(row["component_id"]) == ctype
        and row["component_id"] not in excluded_components
    ]
    if ctype == "attention_head":
        same_layer = [row for row in options if component_layer(row["component_id"]) == component_layer(component)]
        if same_layer:
            options = same_layer
    if not options:
        raise RuntimeError(f"no importance-matched control available for {component}")
    target = abs(float(candidate["mean_difference"])) if context else general_importance(candidate)

    def score(row: dict) -> tuple[float, int]:
        value = abs(float(row["mean_difference"])) if context else general_importance(row)
        return abs(value - target), stable_rank("importance-control", component, row["component_id"])

    selected = min(options, key=score)
    selected_value = abs(float(selected["mean_difference"])) if context else general_importance(selected)
    return {
        "component_id": selected["component_id"],
        "component_type": component_type(selected["component_id"]),
        "same_layer": component_layer(selected["component_id"]) == component_layer(component),
        "candidate_abs_importance": target,
        "control_abs_importance": selected_value,
        "absolute_match_gap": abs(selected_value - target),
        "tie_break": "sha256(importance-control|candidate|control)",
    }


def stream_effects(path: Path, requested: set[tuple[str, str]]) -> dict[tuple[str, str], list[dict]]:
    output: dict[tuple[str, str], list[dict]] = defaultdict(list)
    if not requested:
        return output
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            for estimand, field in ESTIMAND_FIELD.items():
                key = (row["component_id"], estimand)
                if key in requested:
                    output[key].append({
                        "packet_id": row["packet_id"],
                        "sequence_id": row["sequence_id"],
                        "gene_id": row["gene_id"],
                        "chrom": row["chrom"],
                        "target_role": row["target_role"],
                        "tags": row["tags"],
                        "effect": float(row[field]),
                    })
    missing = requested - set(output)
    if missing:
        raise RuntimeError(f"target matrix lacks requested component/estimand cells: {sorted(missing)}")
    return output


def choose_typical(rows: list[dict], target_mean: float, *, key: str) -> dict:
    if not rows:
        raise RuntimeError(f"no representative candidates for {key}")
    return min(rows, key=lambda row: (
        abs(float(row["effect"]) - target_mean),
        stable_rank("typical", key, row["packet_id"]),
    ))


def pair_lookup(pairs: list[dict]) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    by_case: dict[str, list[dict]] = defaultdict(list)
    by_id: dict[str, dict] = {}
    for pair in pairs:
        by_case[pair["case_packet_id"]].append(pair)
        by_id[pair["pair_id"]] = pair
    return by_case, by_id


def representative_general(profile: dict, effects: list[dict], pairs: list[dict]) -> dict:
    by_case, _ = pair_lookup(pairs)
    result = {}
    for role in ("donor", "acceptor"):
        candidates = [row for row in effects if row["target_role"] == role and row["packet_id"] in by_case]
        typical = choose_typical(
            candidates, float(profile[role]["mean"]),
            key=f"general|{profile['estimand']}|{profile['component_id']}|{role}",
        )
        edge = min(by_case[typical["packet_id"]], key=lambda pair: (
            float(pair["match_distance"]), stable_rank("general-edge", pair["pair_id"])
        ))
        result[role] = {
            **typical,
            "target_mean": float(profile[role]["mean"]),
            "distance_to_target_mean": abs(float(typical["effect"]) - float(profile[role]["mean"])),
            "matched_pair_id": edge["pair_id"],
            "matched_negative_context_packet_id": edge["control_packet_id"],
            "selection": "case packet closest to role mean; lowest match-distance edge; stable hash ties",
        }
    return result


def representative_context(candidate: dict, effects: list[dict], pairs: list[dict]) -> dict:
    values = {row["packet_id"]: row for row in effects}
    block = [
        pair for pair in pairs
        if f"{pair['category']}|{pair['target_role']}" == candidate["contrast"]
    ]
    records = []
    for pair in block:
        case = values[pair["case_packet_id"]]
        control = values[pair["control_packet_id"]]
        records.append({
            "pair_id": pair["pair_id"],
            "case_packet_id": pair["case_packet_id"],
            "control_packet_id": pair["control_packet_id"],
            "case_gene_id": pair["case_gene_id"],
            "control_gene_id": pair["control_gene_id"],
            "case_effect": case["effect"],
            "control_effect": control["effect"],
            "effect_difference": case["effect"] - control["effect"],
            "match_distance": float(pair["match_distance"]),
        })
    if not records:
        raise RuntimeError(f"no matched pairs for {candidate['contrast']}")
    target = float(candidate["mean_difference"])
    selected = min(records, key=lambda row: (
        abs(row["effect_difference"] - target),
        stable_rank("context-typical", candidate["contrast"], candidate["component_id"], row["pair_id"]),
    ))
    return {
        **selected,
        "target_mean_difference": target,
        "distance_to_target_mean": abs(selected["effect_difference"] - target),
        "negative_context": "the matched control packet in this same frozen pair",
        "selection": "pair difference closest to full matched mean; stable hash ties",
    }


def intervention_spec(candidate: dict) -> dict:
    estimand = candidate["estimand"]
    ctype = component_type(candidate["component_id"])
    whole_source = "zero_component_output" if estimand == "A" else "partner_destroyed_same_packet"
    spec = {
        "whole_component": whole_source,
        "exact_group": whole_source,
        "source_identity_tolerance": 1e-4,
        "all_component_source_bridge_tolerance": 1e-3,
        "tau_nats": 0.05,
    }
    if ctype == "attention_head":
        if estimand == "B":
            spec["qk_v"] = {
                "source": "partner_destroyed_same_packet",
                "conditions": ["QK", "V", "QKV"],
                "interpretation": "partner-sensitive routing/value main effects; no binding interaction",
            }
        else:
            spec["qk_v"] = {
                "source": "matched_control_packet",
                "conditions": ["QK", "V", "QKV"],
                "interpretation": "matched_control_state_interchange; not a decomposition of zero-output A",
            }
        spec["downstream"] = "prespecified downstream MLP write recovery plus importance-matched write control"
    else:
        spec["mlp_path"] = {
            "upstream": "frozen shortlisted attention group",
            "downstream": "write-level clamps only",
            "control": "importance-matched MLP/head group",
        }
    return spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-prefix", required=True)
    parser.add_argument("--merged-prefix", required=True)
    parser.add_argument("--cohort-prefix", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_matched_v7"
    ))
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-smoke-empty", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    analysis = Path(args.analysis_prefix)
    merged = Path(args.merged_prefix)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refuse to overwrite immutable package {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "shortlist": Path(f"{analysis}.shortlist.json"),
        "analysis_summary": Path(f"{analysis}.summary.json"),
        "analysis_audit": Path(f"{analysis}.audit.json"),
        "profiles": Path(f"{analysis}.profiles.jsonl"),
        "context": Path(f"{analysis}.context.jsonl"),
        "target": Path(f"{merged}.target_matrix.jsonl.gz"),
        "pairs": Path(f"{merged}.pairs.jsonl.gz"),
        "analysis_plan": ANALYSIS_PLAN,
        "analysis_amendment": ANALYSIS_AMENDMENT,
        "mechanism_plan": MECHANISM_PLAN,
        "mechanism_amendment": MECHANISM_AMENDMENT,
        "mechanism_amendment_a2": MECHANISM_AMENDMENT_A2,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    shortlist = json.loads(paths["shortlist"].read_text())
    summary = json.loads(paths["analysis_summary"].read_text())
    audit = json.loads(paths["analysis_audit"].read_text())
    smoke = summary["status"] == "SMOKE"
    if smoke and not args.allow_smoke_empty:
        raise RuntimeError("smoke analysis requires --allow-smoke-empty and cannot select candidates")
    if not smoke and summary["status"] != "DISCOVERY_COMPLETE":
        raise RuntimeError("analysis is neither a recognized smoke nor complete discovery output")
    if audit["status"] != "PASS" or audit["n_failures"]:
        raise RuntimeError("analysis must pass the independent audit before packaging")
    if shortlist["analysis_plan_sha256"] != sha256_file(ANALYSIS_PLAN):
        raise RuntimeError("analysis-plan hash drift")
    if shortlist["analysis_amendment_sha256"] != sha256_file(ANALYSIS_AMENDMENT):
        raise RuntimeError("analysis-amendment hash drift")

    profiles = read_jsonl(paths["profiles"])
    context_rows = read_jsonl(paths["context"])
    pairs = read_jsonl(paths["pairs"])
    selected_general = [] if smoke else shortlist["general"]["selected"]
    selected_context = [] if smoke else shortlist["context"]["selected"]
    selected_components = {
        row["component_id"] for row in selected_general + selected_context
    }

    candidates = []
    for row in selected_general:
        pool = [profile for profile in profiles if profile["estimand"] == row["estimand"]]
        candidates.append({
            "candidate_id": f"general|{row['estimand']}|{row['component_id']}",
            "kind": "general",
            "component_id": row["component_id"],
            "component_type": component_type(row["component_id"]),
            "estimand": row["estimand"],
            "profile": row["profile"],
            "discovery": {
                "score_bound_excess": row["score_bound_excess"],
                "donor_mean": row["donor"]["mean"],
                "acceptor_mean": row["acceptor"]["mean"],
            },
            "importance_control": choose_importance_control(
                row, pool, selected_components, context=False
            ),
        })
    for row in selected_context:
        pool = [
            item for item in context_rows
            if item["contrast"] == row["contrast"] and item["estimand"] == row["estimand"]
        ]
        candidates.append({
            "candidate_id": f"context|{row['contrast']}|{row['estimand']}|{row['component_id']}",
            "kind": "context",
            "component_id": row["component_id"],
            "component_type": component_type(row["component_id"]),
            "estimand": row["estimand"],
            "contrast": row["contrast"],
            "target_role": row["target_role"],
            "discovery": {
                "mean_difference": row["mean_difference"],
                "ci95": row["ci95"],
                "q": row["q_bh_within_estimand"],
                "mde80": row["mde80"],
                "n_effective_pairs": row["n_effective_pairs"],
            },
            "importance_control": choose_importance_control(
                row, pool, selected_components, context=True
            ),
        })

    requested = {(candidate["component_id"], candidate["estimand"]) for candidate in candidates}
    effects = stream_effects(paths["target"], requested)
    profile_map = {(row["estimand"], row["component_id"]): row for row in profiles}
    context_map = {
        (row["contrast"], row["estimand"], row["component_id"]): row
        for row in context_rows
    }
    for candidate in candidates:
        key = (candidate["component_id"], candidate["estimand"])
        if candidate["kind"] == "general":
            profile = profile_map[(candidate["estimand"], candidate["component_id"])]
            candidate["representative"] = representative_general(profile, effects[key], pairs)
        else:
            source = context_map[(
                candidate["contrast"], candidate["estimand"], candidate["component_id"]
            )]
            candidate["representative"] = representative_context(source, effects[key], pairs)
        candidate["interventions"] = intervention_spec(candidate)

    groups: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        group = (
            f"general|{candidate['estimand']}"
            if candidate["kind"] == "general"
            else f"context|{candidate['contrast']}|{candidate['estimand']}"
        )
        if candidate["component_id"] not in groups[group]:
            groups[group].append(candidate["component_id"])

    cohort_prefix = Path(args.cohort_prefix)
    portable_prefix = portable_repo_path(cohort_prefix)
    sealed_references = {
        # ``atlas_internal_validation`` is the value stored *inside* these rows.  The
        # cohort builder deliberately writes the physically separated files with the
        # shorter, immutable ``.validation.*`` stem.  Seal the real filenames here so
        # the later runner cannot choose a path after seeing the discovery shortlist.
        "pairs": f"{portable_prefix}.validation.pairs.jsonl",
        "packets": f"{portable_prefix}.validation.packets.jsonl",
        "sequences": f"{portable_prefix}.validation.sequences.jsonl",
    }
    package = {
        "schema": "head_context_atlas.mechanism_package/1",
        "status": "SMOKE_NO_CANDIDATES" if smoke else "FROZEN_BEFORE_VALIDATION_ACCESS",
        "claim_boundary": "candidate-specific intervention package; no validation outcome or mechanism claim",
        "validation_loaded": False,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "exact_groups": dict(sorted(groups.items())),
        "positive_control": {
            "component_id": "L3H1",
            "estimand": "B",
            "scope": "donor partner-coupling; independent of shortlist and does not consume a slot",
        },
        "validation_inference": {
            "whole_component_family": "Holm-FWER alpha=0.05 across every frozen atlas endpoint",
            "mechanism_family": "max-|t| simultaneous family within each validated endpoint",
            "tau_nats": 0.05,
            "context_se": "inverse-reuse weighted unique-dyad two-endpoint gene sandwich",
            "general_se": "one-way gene-clustered sandwich; roles separate",
        },
        "sealed_validation_references_not_opened": sealed_references,
        "documents": {
            "analysis_plan": {"path": str(ANALYSIS_PLAN), "sha256": sha256_file(ANALYSIS_PLAN)},
            "analysis_amendment": {"path": str(ANALYSIS_AMENDMENT), "sha256": sha256_file(ANALYSIS_AMENDMENT)},
            "mechanism_plan": {"path": str(MECHANISM_PLAN), "sha256": sha256_file(MECHANISM_PLAN)},
            "mechanism_amendment": {
                "path": str(MECHANISM_AMENDMENT), "sha256": sha256_file(MECHANISM_AMENDMENT)
            },
            "mechanism_amendment_a2": {
                "path": str(MECHANISM_AMENDMENT_A2), "sha256": sha256_file(MECHANISM_AMENDMENT_A2)
            },
        },
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
    }
    out.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")

    from gfmcd.provenance import ExperimentManifest
    manifest_path = out.with_suffix(".manifest.json")
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_mechanism_package_v1",
        command=["python"] + sys.argv,
        status=package["status"],
    )
    manifest.config.update({
        "validation_loaded": False,
        "n_candidates": len(candidates),
        "allow_smoke_empty": args.allow_smoke_empty,
    })
    for path in paths.values():
        manifest.add_input(str(path))
    manifest.add_output(str(out))
    manifest.write(str(manifest_path))
    print(json.dumps({
        "status": package["status"],
        "n_candidates": len(candidates),
        "n_groups": len(groups),
        "validation_loaded": False,
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
