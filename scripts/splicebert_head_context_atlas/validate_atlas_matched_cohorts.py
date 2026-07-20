#!/usr/bin/env python
"""Mechanically audit a Head--Context Atlas matched-cohort artifact.

This audit reads no model outputs.  It independently checks the relational, split,
sequence-coordinate, balance, reuse, and provenance invariants that must hold before a
cohort may be sent to model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_atlas_matched_cohorts import (  # noqa: E402
    CATEGORY_ROLES,
    CORE_CATEGORIES,
    NUMERIC_FEATURES,
    balance_stats,
    exact_stratum,
)
from gfmcd.splicing.atlas_context import atlas_analysis_split  # noqa: E402

EXCLUDED_CHROMS = {"chr8", "chr9", "chr11", "chr16", "chr19"}
EXPECTED_SPLITS = {
    "discovery": "atlas_discovery",
    "validation": "atlas_internal_validation",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_compatible(value):
    if isinstance(value, tuple):
        return [json_compatible(item) for item in value]
    return value


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def audit_split(short: str, prefix: Path, summary: dict, *, reuse_cap: int,
                balance_threshold: float) -> tuple[dict, set[str], set[str], list[str]]:
    expected_split = EXPECTED_SPLITS[short]
    paths = {
        kind: Path(f"{prefix}.{short}.{kind}.jsonl")
        for kind in ("pairs", "packets", "sequences")
    }
    pairs = read_jsonl(paths["pairs"])
    packets = read_jsonl(paths["packets"])
    sequences = read_jsonl(paths["sequences"])
    failures: list[str] = []

    packet_ids = [row["packet_id"] for row in packets]
    sequence_ids = [row["sequence_id"] for row in sequences]
    pair_ids = [row["pair_id"] for row in pairs]
    require(len(packet_ids) == len(set(packet_ids)), f"{short}: duplicate packet_id", failures)
    require(len(sequence_ids) == len(set(sequence_ids)), f"{short}: duplicate sequence_id", failures)
    require(len(pair_ids) == len(set(pair_ids)), f"{short}: duplicate pair_id", failures)
    by_packet = {row["packet_id"]: row for row in packets}
    by_sequence = {row["sequence_id"]: row for row in sequences}

    referenced_packets: set[str] = set()
    block_pairs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for pair in pairs:
        category = pair.get("category")
        role = pair.get("target_role")
        require(category in CORE_CATEGORIES, f"{short}: unknown category {category!r}", failures)
        if category not in CORE_CATEGORIES:
            continue
        require(role in CATEGORY_ROLES[category],
                f"{short}: role {role!r} invalid for {category}", failures)
        require(pair.get("analysis_split") == expected_split,
                f"{short}: pair {pair['pair_id']} split mismatch", failures)
        case_id, control_id = pair["case_packet_id"], pair["control_packet_id"]
        referenced_packets.update((case_id, control_id))
        require(case_id in by_packet, f"{short}: missing case packet {case_id}", failures)
        require(control_id in by_packet, f"{short}: missing control packet {control_id}", failures)
        if case_id not in by_packet or control_id not in by_packet:
            continue
        case, control = by_packet[case_id], by_packet[control_id]
        require(case["gene_id"] == pair["case_gene_id"],
                f"{short}: case gene mismatch in {pair['pair_id']}", failures)
        require(control["gene_id"] == pair["control_gene_id"],
                f"{short}: control gene mismatch in {pair['pair_id']}", failures)
        require(case["gene_id"] != control["gene_id"],
                f"{short}: within-gene match in {pair['pair_id']}", failures)
        require(case["analysis_split"] == control["analysis_split"] == expected_split,
                f"{short}: packet split mismatch in {pair['pair_id']}", failures)
        require(case["target_role"] == control["target_role"] == role,
                f"{short}: target role mismatch in {pair['pair_id']}", failures)
        require(category in case["tags"],
                f"{short}: case lacks focal tag in {pair['pair_id']}", failures)
        require(category not in control["tags"],
                f"{short}: control carries focal tag in {pair['pair_id']}", failures)
        case_stratum = json_compatible(exact_stratum(case, category))
        control_stratum = json_compatible(exact_stratum(control, category))
        require(case_stratum == control_stratum,
                f"{short}: exact-stratum mismatch in {pair['pair_id']}", failures)
        require(case_stratum == pair.get("exact_stratum"),
                f"{short}: stored stratum mismatch in {pair['pair_id']}", failures)
        block_pairs[(category, role)].append(pair)

    require(set(packet_ids) == referenced_packets,
            f"{short}: packet file is not exactly the pair-referenced set", failures)

    referenced_sequences = set()
    genes = set()
    chromosomes = set()
    for packet in packets:
        packet_id = packet["packet_id"]
        sequence_id = packet["sequence_id"]
        referenced_sequences.add(sequence_id)
        genes.add(packet["gene_id"])
        chromosomes.add(packet["chrom"])
        require(sequence_id in by_sequence,
                f"{short}: packet {packet_id} lacks sequence row", failures)
        require(packet["analysis_split"] == expected_split,
                f"{short}: packet {packet_id} split mismatch", failures)
        require(atlas_analysis_split(packet["gene_id"], packet["split"]) == expected_split,
                f"{short}: packet {packet_id} disagrees with deterministic split", failures)
        require(packet["chrom"] not in EXCLUDED_CHROMS,
                f"{short}: excluded chromosome {packet['chrom']} in {packet_id}", failures)
        if sequence_id not in by_sequence:
            continue
        sequence_row = by_sequence[sequence_id]
        seq = sequence_row["sequence"]
        donor = packet["donor_win"]
        acceptor = packet["acceptor_win"]
        masked = packet["masked_base_win"]
        require(len(seq) == packet["window_end"] - packet["window_start"],
                f"{short}: sequence length mismatch for {packet_id}", failures)
        require(not (set(seq) - set("ACGT")),
                f"{short}: non-ACGT sequence for {packet_id}", failures)
        require(hashlib.sha256(seq.encode()).hexdigest() == sequence_row["sequence_sha256"],
                f"{short}: sequence hash mismatch for {packet_id}", failures)
        require(seq[donor:donor + 2] in {"GT", "GC"},
                f"{short}: donor motif mismatch for {packet_id}", failures)
        require(seq[acceptor - 2:acceptor] == "AG",
                f"{short}: acceptor motif mismatch for {packet_id}", failures)
        require(0 <= masked < len(seq) and seq[masked] == "G",
                f"{short}: masked native base mismatch for {packet_id}", failures)
        expected_masked = donor if packet["target_role"] == "donor" else acceptor - 1
        require(masked == expected_masked,
                f"{short}: role-specific masked coordinate mismatch for {packet_id}", failures)
        for key in ("sequence_id", "chrom", "strand", "window_start", "window_end",
                    "donor_win", "acceptor_win"):
            require(packet[key] == sequence_row[key],
                    f"{short}: packet/sequence {key} mismatch for {packet_id}", failures)

    require(set(sequence_ids) == referenced_sequences,
            f"{short}: sequence file is not exactly the packet-referenced set", failures)

    block_report = {}
    summary_blocks = summary["summaries"][expected_split]
    for category in CORE_CATEGORIES:
        for role in CATEGORY_ROLES[category]:
            key = (category, role)
            label = f"{category}|{role}"
            current = block_pairs.get(key, [])
            require(bool(current), f"{short}: empty contrast {label}", failures)
            uses = Counter(pair["control_packet_id"] for pair in current)
            max_reuse = max(uses.values(), default=0)
            require(max_reuse <= reuse_cap,
                    f"{short}: control reuse {max_reuse}>{reuse_cap} in {label}", failures)
            balance = balance_stats(current, by_packet)
            require(balance["max_abs_numeric_smd"] <= balance_threshold + 1e-12,
                    f"{short}: numeric balance failed in {label}", failures)
            require(balance["max_abs_other_tag_smd"] <= 1e-12,
                    f"{short}: other-tag exactness failed in {label}", failures)
            stored = summary_blocks[label]
            require(len(current) == stored["n_matched"],
                    f"{short}: summary count mismatch in {label}", failures)
            require(max_reuse == stored["max_control_reuse"],
                    f"{short}: summary reuse mismatch in {label}", failures)
            for feature in NUMERIC_FEATURES:
                require(math.isclose(
                    balance["numeric_smd"][feature],
                    stored["balance"]["numeric_smd"][feature],
                    rel_tol=0.0, abs_tol=1e-12,
                ), f"{short}: stored balance mismatch for {label}/{feature}", failures)
            block_report[label] = {
                "n_pairs": len(current),
                "n_unique_case_genes": len({pair["case_gene_id"] for pair in current}),
                "n_unique_controls": len(uses),
                "max_control_reuse": max_reuse,
                "max_abs_numeric_smd": balance["max_abs_numeric_smd"],
                "max_abs_other_tag_smd": balance["max_abs_other_tag_smd"],
            }

    selected = summary["selected"][expected_split]
    require(len(pairs) == selected["n_pairs"], f"{short}: total pair count mismatch", failures)
    require(len(packets) == selected["n_unique_packets"],
            f"{short}: total packet count mismatch", failures)
    require(len(sequences) == selected["n_unique_sequences"],
            f"{short}: total sequence count mismatch", failures)
    require(len(genes) == selected["n_genes"], f"{short}: total gene count mismatch", failures)
    return ({
        "n_pairs": len(pairs),
        "n_packets": len(packets),
        "n_sequences": len(sequences),
        "n_genes": len(genes),
        "chromosomes": sorted(chromosomes),
        "blocks": block_report,
        "gate": not failures,
    }, genes, set(sequence_ids), failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--reuse-cap", type=int, default=2)
    parser.add_argument("--balance-threshold", type=float, default=0.10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    prefix = Path(args.prefix)
    summary_path = Path(f"{prefix}.summary.json")
    manifest_path = Path(f"{prefix}.manifest.json")
    output_path = Path(args.out) if args.out else Path(f"{prefix}.audit.json")
    audit_manifest_path = output_path.with_suffix(".manifest.json")
    for path in (output_path, audit_manifest_path):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite immutable artifact {path}")

    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []
    split_reports = {}
    split_genes = {}
    split_sequences = {}
    for short in EXPECTED_SPLITS:
        report, genes, sequences, split_failures = audit_split(
            short, prefix, summary,
            reuse_cap=args.reuse_cap,
            balance_threshold=args.balance_threshold,
        )
        split_reports[short] = report
        split_genes[short] = genes
        split_sequences[short] = sequences
        failures.extend(split_failures)

    require(not (split_genes["discovery"] & split_genes["validation"]),
            "discovery/validation gene leakage", failures)
    require(not (split_sequences["discovery"] & split_sequences["validation"]),
            "discovery/validation sequence leakage", failures)
    require(summary.get("global_balance_gate") is True,
            "source summary global balance gate is not true", failures)

    manifest_checks = []
    for kind in ("inputs", "outputs"):
        for item in manifest[kind]:
            path = Path(item["path"])
            if not path.is_absolute():
                path = REPO / path
            exists = path.exists()
            size_ok = exists and path.stat().st_size == item["size_bytes"]
            hash_ok = exists and sha256_file(path) == item["sha256"]
            manifest_checks.append({
                "path": str(path), "exists": exists, "size_ok": size_ok, "sha256_ok": hash_ok,
            })
            require(exists and size_ok and hash_ok,
                    f"manifest integrity failed for {path}", failures)

    report = {
        "schema": "head_context_atlas.matched_cohort_audit/1",
        "status": "PREFLIGHT_PASS" if not failures else "PREFLIGHT_FAIL",
        "source_prefix": str(prefix),
        "claim_boundary": "mechanical cohort audit only; no model output or mechanism read",
        "taxonomy_policy": {
            "core_labels_are_overlapping": True,
            "single_pair_is_packet_scoped": True,
            "role_specific": {
                "donor_cluster": ["donor"],
                "acceptor_cluster": ["acceptor"],
            },
            "validation_minimum_is_precision_not_equivalence": (
                "balanced arms below the discovery minimum remain internal-validation/exploratory "
                "until an outcome-specific MDE is computed"
            ),
        },
        "split_reports": split_reports,
        "cross_split": {
            "gene_disjoint": not (split_genes["discovery"] & split_genes["validation"]),
            "sequence_disjoint": not (split_sequences["discovery"] & split_sequences["validation"]),
        },
        "manifest_checks": manifest_checks,
        "n_failures": len(failures),
        "failures": failures,
        "gate": not failures,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    from gfmcd.provenance import ExperimentManifest
    audit_manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_matched_cohort_audit_v1",
        command=["python"] + sys.argv,
        status=report["status"],
    )
    for path in [
        summary_path, manifest_path,
        *[
            Path(f"{prefix}.{short}.{kind}.jsonl")
            for short in EXPECTED_SPLITS for kind in ("pairs", "packets", "sequences")
        ],
    ]:
        audit_manifest.add_input(str(path))
    audit_manifest.add_output(str(output_path))
    audit_manifest.write(str(audit_manifest_path))
    print(json.dumps({
        "status": report["status"],
        "n_failures": len(failures),
        "discovery_pairs": split_reports["discovery"]["n_pairs"],
        "validation_pairs": split_reports["validation"]["n_pairs"],
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
