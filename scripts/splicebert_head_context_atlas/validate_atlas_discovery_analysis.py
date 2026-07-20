#!/usr/bin/env python
"""Independent mechanical/statistical audit of a discovery-analysis artifact.

This verifier deliberately does not import ``analyze_atlas_discovery``.  It
reconstructs the packet matrices, estimators, multiplicity families,
classifications, stability split, and deterministic shortlist from the merged
screen and compares those independently computed quantities with the recorded
analysis outputs.
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
PLAN = REPO / "studies/splicebert_head_context_atlas/ANALYSIS_PLAN_V1.md"
AMENDMENT = REPO / "studies/splicebert_head_context_atlas/ANALYSIS_AMENDMENT_A1.md"
COMPONENTS = tuple(
    [f"L{layer}H{head}" for layer in range(6) for head in range(16)]
    + [f"L{layer}MLP" for layer in range(6)]
)
COMPONENT_SET = set(COMPONENTS)
ESTIMANDS = {"A": "A_effect_margin", "B": "B_effect_margin"}
HEALTH_FIELDS = {
    "A": "A_health_margin_damage",
    "B": "B_health_margin_damage",
    "A_logprob": "A_health_logprob_damage",
    "B_logprob": "B_health_logprob_damage",
}
FULL_CONTRASTS = {
    "single_pair|donor", "single_pair|acceptor",
    "donor_cluster|donor", "acceptor_cluster|acceptor",
    "multiple_independent_pairs|donor", "multiple_independent_pairs|acceptor",
    "cassette_exon|donor", "cassette_exon|acceptor",
    "complex_event|donor", "complex_event|acceptor",
}
Z_MDE = float(norm.ppf(0.975) + norm.ppf(0.80))
ATOL = 2e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_half(gene: str) -> int:
    return stable_rank("atlas-stability-v1", gene) % 2


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def close(actual: float, expected: float, message: str, failures: list[str]) -> None:
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        failures.append(f"{message}: non-finite value")
    elif not math.isclose(float(actual), float(expected), rel_tol=2e-9, abs_tol=ATOL):
        failures.append(f"{message}: {actual} != {expected}")


def close_pair(actual: list[float], expected: np.ndarray, message: str,
               failures: list[str]) -> None:
    require(len(actual) == 2, f"{message}: expected two bounds", failures)
    if len(actual) == 2:
        close(actual[0], expected[0], f"{message}[0]", failures)
        close(actual[1], expected[1], f"{message}[1]", failures)


def bh_reference(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="mergesort")
    n = len(pvalues)
    adjusted = pvalues[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty(n, dtype=float)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def p_reference(point: np.ndarray, se: np.ndarray) -> np.ndarray:
    output = np.ones_like(point)
    positive = se > 0
    output[positive] = 2 * norm.sf(np.abs(point[positive] / se[positive]))
    output[~positive & (np.abs(point) > 0)] = 0.0
    return output


def gene_reference(values: np.ndarray, genes: list[str]) -> dict[str, np.ndarray | int]:
    n = len(genes)
    point = values.sum(axis=0) / n
    residual = values - point
    unique = sorted(set(genes))
    scores = np.vstack([
        residual[np.asarray([gene == current for gene in genes])].sum(axis=0)
        for current in unique
    ])
    se = np.sqrt(np.square(scores).sum(axis=0) * len(unique) / max(len(unique) - 1, 1)) / n
    return {
        "point": point,
        "se": se,
        "ci": np.vstack([point - 1.96 * se, point + 1.96 * se]).T,
        "p": p_reference(point, se),
        "mde": Z_MDE * se,
        "n": n,
        "n_genes": len(unique),
    }


def dyadic_reference(values: np.ndarray, case_genes: list[str], control_genes: list[str],
                     weights: np.ndarray) -> dict[str, np.ndarray | float | int]:
    total = weights.sum()
    point = (values * weights[:, None]).sum(axis=0) / total
    score = weights[:, None] * (values - point)
    genes = sorted(set(case_genes) | set(control_genes))
    incident_rows = []
    for gene in genes:
        incident = np.asarray([
            case == gene or control == gene
            for case, control in zip(case_genes, control_genes)
        ])
        incident_rows.append(score[incident].sum(axis=0))
    incident_scores = np.vstack(incident_rows)
    meat = np.square(incident_scores).sum(axis=0) - np.square(score).sum(axis=0)
    se = np.sqrt(np.maximum(meat * len(genes) / max(len(genes) - 1, 1), 0.0)) / total
    return {
        "point": point,
        "se": se,
        "ci": np.vstack([point - 1.96 * se, point + 1.96 * se]).T,
        "p": p_reference(point, se),
        "mde": Z_MDE * se,
        "n_pairs": len(values),
        "n_genes": len(genes),
        "neff": float(total * total / np.square(weights).sum()),
    }


def classification(low: float, high: float, qvalue: float, tau: float) -> str:
    if qvalue <= 0.05 and low > tau:
        return "material_support"
    if qvalue <= 0.05 and high < -tau:
        return "material_suppress"
    if qvalue <= 0.05 and (low > 0 or high < 0):
        return "detected_subfloor"
    if low >= -tau and high <= tau:
        return "bounded_small"
    return "inconclusive"


def load_matrix(path: Path, fields: dict[str, str], failures: list[str], label: str
                ) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = read_jsonl(path)
    grouped: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        packet_id = row["packet_id"]
        if packet_id not in grouped:
            order.append(packet_id)
        grouped[packet_id].append(row)
    meta: list[dict] = []
    matrices = {key: [] for key in fields}
    for packet_id in order:
        packet_rows = grouped[packet_id]
        by_component = {row["component_id"]: row for row in packet_rows}
        require(
            len(packet_rows) == 102 and set(by_component) == COMPONENT_SET,
            f"{label}: incomplete/duplicate component grid for {packet_id}", failures,
        )
        if set(by_component) != COMPONENT_SET:
            continue
        first = packet_rows[0]
        require(first["analysis_split"] == "atlas_discovery",
                f"{label}: non-discovery packet {packet_id}", failures)
        meta.append({
            key: first[key] for key in (
                "packet_id", "sequence_id", "gene_id", "chrom", "analysis_split",
                "target_role", "tags",
            )
        })
        for key, field in fields.items():
            values = np.asarray([float(by_component[component][field]) for component in COMPONENTS])
            require(np.isfinite(values).all(), f"{label}: non-finite {key} for {packet_id}", failures)
            matrices[key].append(values)
    return meta, {key: np.vstack(value) for key, value in matrices.items()}


def verify_manifest(manifest_path: Path, expected_inputs: dict[str, Path],
                    expected_outputs: dict[str, Path], failures: list[str]) -> None:
    manifest = json.loads(manifest_path.read_text())
    actual_inputs = {record["path"]: record for record in manifest["inputs"]}
    actual_outputs = {record["path"]: record for record in manifest["outputs"]}
    for path in expected_inputs.values():
        record = actual_inputs.get(str(path))
        require(record is not None, f"manifest: missing input {path}", failures)
        if record:
            require(record["sha256"] == sha256_file(path) and record["size_bytes"] == path.stat().st_size,
                    f"manifest: stale input {path}", failures)
    for name, path in expected_outputs.items():
        if name == "manifest":
            continue
        record = actual_outputs.get(str(path))
        require(record is not None, f"manifest: missing output {path}", failures)
        if record:
            require(record["sha256"] == sha256_file(path) and record["size_bytes"] == path.stat().st_size,
                    f"manifest: stale output {path}", failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-prefix", required=True)
    parser.add_argument("--analysis-prefix", required=True)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--allow-incomplete-contexts", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    merged = Path(args.merged_prefix)
    analysis = Path(args.analysis_prefix)
    inputs = {
        "target": Path(f"{merged}.target_matrix.jsonl.gz"),
        "health": Path(f"{merged}.health_matrix.jsonl.gz"),
        "pairs": Path(f"{merged}.pairs.jsonl.gz"),
        "summary": Path(f"{merged}.summary.json"),
        "plan": PLAN,
        "amendment": AMENDMENT,
    }
    outputs = {
        "general": Path(f"{analysis}.general.jsonl"),
        "profiles": Path(f"{analysis}.profiles.jsonl"),
        "context": Path(f"{analysis}.context.jsonl"),
        "shortlist": Path(f"{analysis}.shortlist.json"),
        "summary": Path(f"{analysis}.summary.json"),
        "manifest": Path(f"{analysis}.manifest.json"),
    }
    audit_path = Path(args.audit) if args.audit else Path(f"{analysis}.audit.json")
    if audit_path.exists():
        raise FileExistsError(f"refuse to overwrite immutable audit {audit_path}")
    failures: list[str] = []
    for path in [*inputs.values(), *outputs.values()]:
        require(path.exists(), f"missing required artifact {path}", failures)
    if failures:
        raise FileNotFoundError("; ".join(failures))

    merged_summary = json.loads(inputs["summary"].read_text())
    summary = json.loads(outputs["summary"].read_text())
    shortlist = json.loads(outputs["shortlist"].read_text())
    tau = float(summary["tau"])
    require(merged_summary["status"] == "COMPLETE" and merged_summary["n_components"] == 102,
            "merged input is not COMPLETE/102-component", failures)
    require(shortlist["analysis_plan_sha256"] == sha256_file(PLAN),
            "shortlist plan hash mismatch", failures)
    require(shortlist["analysis_amendment_sha256"] == sha256_file(AMENDMENT),
            "shortlist amendment hash mismatch", failures)
    require(summary["analysis_amendment_sha256"] == sha256_file(AMENDMENT),
            "summary amendment hash mismatch", failures)
    for name in ("target", "health", "pairs", "summary"):
        require(shortlist["input_hashes"].get(name) == sha256_file(inputs[name]),
                f"shortlist input hash mismatch: {name}", failures)

    packet_meta, target = load_matrix(inputs["target"], ESTIMANDS, failures, "target")
    health_meta, health = load_matrix(inputs["health"], HEALTH_FIELDS, failures, "health")
    pairs = read_jsonl(inputs["pairs"])
    packet_index = {meta["packet_id"]: index for index, meta in enumerate(packet_meta)}
    health_index = {meta["packet_id"]: index for index, meta in enumerate(health_meta)}
    require(len(packet_index) == len(packet_meta), "duplicate target packet metadata", failures)
    require(len(health_index) == len(health_meta), "duplicate health packet metadata", failures)
    contrasts = {f"{pair['category']}|{pair['target_role']}" for pair in pairs}
    if not args.allow_incomplete_contexts:
        require(contrasts == FULL_CONTRASTS, "full contrast family mismatch", failures)
    require(summary["n_packets"] == len(packet_meta), "summary packet count mismatch", failures)
    require(summary["n_health_packets"] == len(health_meta), "summary health count mismatch", failures)
    require(summary["n_pairs"] == len(pairs), "summary pair count mismatch", failures)

    general_rows = read_jsonl(outputs["general"])
    general = {(row["estimand"], row["target_role"], row["component_id"]): row for row in general_rows}
    require(len(general_rows) == 408 and len(general) == 408,
            "general output does not have a unique 2x2x102 grid", failures)
    for estimand in ("A", "B"):
        family_keys = [(estimand, role, component) for role in ("donor", "acceptor") for component in COMPONENTS]
        pvalues = []
        references = {}
        for role in ("donor", "acceptor"):
            tidx = [index for index, meta in enumerate(packet_meta) if meta["target_role"] == role]
            hidx = [index for index, meta in enumerate(health_meta) if meta["target_role"] == role]
            target_ref = gene_reference(target[estimand][tidx], [packet_meta[index]["gene_id"] for index in tidx])
            health_ref = gene_reference(health[estimand][hidx], [health_meta[index]["gene_id"] for index in hidx])
            logprob_ref = gene_reference(health[f"{estimand}_logprob"][hidx], [health_meta[index]["gene_id"] for index in hidx])
            for component_index, component in enumerate(COMPONENTS):
                key = (estimand, role, component)
                references[key] = (target_ref, health_ref, logprob_ref, component_index)
                pvalues.append(float(target_ref["p"][component_index]))
        qvalues = bh_reference(np.asarray(pvalues))
        for key, qvalue in zip(family_keys, qvalues):
            row = general.get(key)
            if row is None:
                continue
            target_ref, health_ref, logprob_ref, index = references[key]
            label = "general:" + "|".join(key)
            close(row["mean"], target_ref["point"][index], label + ":mean", failures)
            close(row["se_gene_cluster"], target_ref["se"][index], label + ":se", failures)
            close_pair(row["ci95"], target_ref["ci"][index], label + ":ci", failures)
            close(row["p"], target_ref["p"][index], label + ":p", failures)
            close(row["mde80"], target_ref["mde"][index], label + ":mde", failures)
            close(row["q_bh_within_estimand"], qvalue, label + ":q", failures)
            close(row["health_mean_margin_damage"], health_ref["point"][index], label + ":health-margin", failures)
            close(row["health_mean_logprob_damage"], logprob_ref["point"][index], label + ":health-logprob", failures)
            expected_class = classification(target_ref["ci"][index, 0], target_ref["ci"][index, 1], qvalue, tau)
            require(row["classification"] == expected_class, label + ": classification mismatch", failures)

    profiles = read_jsonl(outputs["profiles"])
    profile_map = {(row["estimand"], row["component_id"]): row for row in profiles}
    require(len(profiles) == 204 and len(profile_map) == 204,
            "profile output does not have a unique 2x102 grid", failures)
    for (estimand, component), profile in profile_map.items():
        donor = general[(estimand, "donor", component)]
        acceptor = general[(estimand, "acceptor", component)]
        labels = {donor["classification"], acceptor["classification"]}
        if labels == {"material_support"}:
            expected_profile = "cross_role_general_support"
            score = min(donor["ci95"][0] - tau, acceptor["ci95"][0] - tau)
        elif labels == {"material_suppress"}:
            expected_profile = "cross_role_general_suppress"
            score = min(-tau - donor["ci95"][1], -tau - acceptor["ci95"][1])
        elif labels == {"material_support", "material_suppress"}:
            expected_profile = "role_opposed"
            score = min(abs(donor["mean"]) - tau, abs(acceptor["mean"]) - tau)
        elif labels & {"material_support", "material_suppress"}:
            expected_profile = "role_associated"
            score = max(abs(donor["mean"]) - tau, abs(acceptor["mean"]) - tau)
        else:
            expected_profile, score = "unresolved", None
        require(profile["profile"] == expected_profile,
                f"profile:{estimand}|{component}: profile mismatch", failures)
        if score is not None:
            close(profile["score_bound_excess"], score,
                  f"profile:{estimand}|{component}: score", failures)
        eligible = bool(
            expected_profile in {"cross_role_general_support", "cross_role_general_suppress"}
            and donor["mde80"] <= tau and acceptor["mde80"] <= tau
            and donor["health_n"] > 0 and acceptor["health_n"] > 0
        )
        require(profile["general_shortlist_eligible"] is eligible,
                f"profile:{estimand}|{component}: eligibility mismatch", failures)

    context_rows = read_jsonl(outputs["context"])
    context = {
        (row["contrast"], row["estimand"], row["component_id"]): row
        for row in context_rows
    }
    require(len(context_rows) == len(contrasts) * 2 * 102 and len(context) == len(context_rows),
            "context output grid mismatch", failures)
    blocks: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        contrast = f"{pair['category']}|{pair['target_role']}"
        blocks[contrast].append(pair)
        case = packet_index.get(pair["case_packet_id"])
        control = packet_index.get(pair["control_packet_id"])
        require(case is not None and control is not None, f"pair join failed: {pair['pair_id']}", failures)
        if case is not None and control is not None:
            case_meta, control_meta = packet_meta[case], packet_meta[control]
            require(case_meta["gene_id"] == pair["case_gene_id"] and control_meta["gene_id"] == pair["control_gene_id"],
                    f"pair gene mismatch: {pair['pair_id']}", failures)
            require(case_meta["target_role"] == pair["target_role"] == control_meta["target_role"],
                    f"pair role mismatch: {pair['pair_id']}", failures)
            require(pair["category"] in case_meta["tags"] and pair["category"] not in control_meta["tags"],
                    f"pair focal-tag mismatch: {pair['pair_id']}", failures)
    for contrast, block in blocks.items():
        endpoint_dyads = [
            tuple(sorted((pair["case_gene_id"], pair["control_gene_id"])))
            for pair in block
        ]
        require(len(endpoint_dyads) == len(set(endpoint_dyads)),
                f"{contrast}: repeated endpoint-gene dyad", failures)

    for estimand in ("A", "B"):
        family_keys = [(contrast, estimand, component) for contrast in sorted(blocks) for component in COMPONENTS]
        pvalues = []
        references = {}
        for contrast in sorted(blocks):
            block = blocks[contrast]
            case_idx = np.asarray([packet_index[pair["case_packet_id"]] for pair in block])
            control_idx = np.asarray([packet_index[pair["control_packet_id"]] for pair in block])
            case_genes = [pair["case_gene_id"] for pair in block]
            control_genes = [pair["control_gene_id"] for pair in block]
            reuse = Counter(pair["control_packet_id"] for pair in block)
            weights = np.asarray([1.0 / reuse[pair["control_packet_id"]] for pair in block])
            difference = target[estimand][case_idx] - target[estimand][control_idx]
            ref = dyadic_reference(difference, case_genes, control_genes, weights)
            case_mean = np.average(target[estimand][case_idx], axis=0, weights=weights)
            control_mean = np.average(target[estimand][control_idx], axis=0, weights=weights)
            halves = np.asarray([
                stable_half(case) if stable_half(case) == stable_half(control) else -1
                for case, control in zip(case_genes, control_genes)
            ])
            half_counts = [int(np.sum(halves == half)) for half in (0, 1)]
            half_means = [
                np.average(difference[halves == half], axis=0, weights=weights[halves == half])
                if half_counts[half] else np.full(102, np.nan)
                for half in (0, 1)
            ]
            for index, component in enumerate(COMPONENTS):
                key = (contrast, estimand, component)
                references[key] = (ref, case_mean, control_mean, half_counts, half_means, index)
                pvalues.append(float(ref["p"][index]))
        qvalues = bh_reference(np.asarray(pvalues))
        for key, qvalue in zip(family_keys, qvalues):
            row = context.get(key)
            if row is None:
                continue
            ref, case_mean, control_mean, half_counts, half_means, index = references[key]
            label = "context:" + "|".join(key)
            close(row["mean_case"], case_mean[index], label + ":case", failures)
            close(row["mean_control"], control_mean[index], label + ":control", failures)
            close(row["mean_difference"], ref["point"][index], label + ":difference", failures)
            close(row["se_gene_dyadic"], ref["se"][index], label + ":se", failures)
            close_pair(row["ci95"], ref["ci"][index], label + ":ci", failures)
            close(row["p"], ref["p"][index], label + ":p", failures)
            close(row["q_bh_within_estimand"], qvalue, label + ":q", failures)
            close(row["mde80"], ref["mde"][index], label + ":mde", failures)
            close(row["n_effective_pairs"], ref["neff"], label + ":neff", failures)
            require(row["stability_half_counts"] == half_counts,
                    label + ": half-count mismatch", failures)
            for half in (0, 1):
                if half_counts[half]:
                    close(row["stability_half_means"][half], half_means[half][index],
                          label + f":half-{half}", failures)
                else:
                    require(row["stability_half_means"][half] is None,
                            label + f": empty half-{half} should be null", failures)
            expected_class = classification(ref["ci"][index, 0], ref["ci"][index, 1], qvalue, tau)
            support = all(count >= 40 for count in half_counts)
            direction = bool(
                support and ref["point"][index] != 0
                and np.sign(half_means[0][index]) == np.sign(ref["point"][index])
                and np.sign(half_means[1][index]) == np.sign(ref["point"][index])
            )
            require(row["classification"] == expected_class, label + ": classification mismatch", failures)
            require(row["stability_support"] is support and row["stability_direction_pass"] is direction,
                    label + ": stability mismatch", failures)
            require(row["context_associated"] is bool(
                expected_class in {"material_support", "material_suppress"} and direction
            ), label + ": context-associated mismatch", failures)

    general_candidates = [profile for profile in profiles if profile["general_shortlist_eligible"]]
    general_candidates.sort(key=lambda row: (
        -float(row["score_bound_excess"]), stable_rank("general", row["component_id"], row["estimand"])
    ))
    general_selected = []
    seen = set()
    for row in general_candidates:
        if row["component_id"] not in seen:
            general_selected.append((row["component_id"], row["estimand"]))
            seen.add(row["component_id"])
        if len(general_selected) == 4:
            break
    context_candidates = [
        row for row in context_rows
        if row["context_associated"] and row["n_effective_pairs"] >= 80 and row["mde80"] <= tau
    ]
    context_candidates.sort(key=lambda row: (
        -float(row["score_bound_excess"]),
        stable_rank("context", row["contrast"], row["estimand"], row["component_id"]),
    ))
    context_selected = []
    contrast_count, component_count = Counter(), Counter()
    for role in ("donor", "acceptor"):
        row = next((candidate for candidate in context_candidates
                    if candidate["target_role"] == role
                    and contrast_count[candidate["contrast"]] < 2
                    and component_count[candidate["component_id"]] < 3), None)
        if row is not None:
            context_selected.append(row)
            contrast_count[row["contrast"]] += 1
            component_count[row["component_id"]] += 1
    for row in context_candidates:
        if len(context_selected) >= 8:
            break
        if row in context_selected or contrast_count[row["contrast"]] >= 2 or component_count[row["component_id"]] >= 3:
            continue
        context_selected.append(row)
        contrast_count[row["contrast"]] += 1
        component_count[row["component_id"]] += 1
    recorded_general = [(row["component_id"], row["estimand"]) for row in shortlist["general"]["selected"]]
    recorded_context = [
        (row["contrast"], row["estimand"], row["component_id"])
        for row in shortlist["context"]["selected"]
    ]
    if args.allow_incomplete_contexts:
        general_selected, context_selected = [], []
        require(shortlist["status"] == "SMOKE_NOT_FOR_SELECTION",
                "smoke shortlist status mismatch", failures)
    expected_context = [(row["contrast"], row["estimand"], row["component_id"]) for row in context_selected]
    require(recorded_general == general_selected, "general shortlist mismatch", failures)
    require(recorded_context == expected_context, "context shortlist mismatch", failures)

    verify_manifest(outputs["manifest"], inputs, outputs, failures)
    report = {
        "schema": "head_context_atlas.discovery_analysis_audit/1",
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": "independent analysis mechanics audit; not biological validation",
        "analysis_status": summary["status"],
        "n_packets": len(packet_meta),
        "n_health_packets": len(health_meta),
        "n_pairs": len(pairs),
        "n_components": 102,
        "n_contrasts": len(contrasts),
        "n_general_rows": len(general_rows),
        "n_context_rows": len(context_rows),
        "n_failures": len(failures),
        "failures": failures[:100],
        "analysis_plan_sha256": sha256_file(PLAN),
        "analysis_amendment_sha256": sha256_file(AMENDMENT),
        "analysis_manifest_sha256": sha256_file(outputs["manifest"]),
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
