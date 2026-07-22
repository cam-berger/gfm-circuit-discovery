#!/usr/bin/env python
"""Independently audit a frozen P3 phase-surgery package without model access."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import DEFAULT_FASTA, DEFAULT_GTF, _dataset  # noqa: E402
from gfmcd.splicing.p3_closure import stable_gene_role  # noqa: E402
from gfmcd.splicing.p3_cohort import sequence_sha256  # noqa: E402


ROLE_WEIGHTS = (("development", 2), ("confirmation", 2), ("reserve", 1))


def verify_file_record(record: dict) -> bool:
    path = Path(record["path"])
    return (
        path.is_file()
        and int(record["size_bytes"]) == path.stat().st_size
        and record["sha256"] == sha256_file(path)
    )


def flatten_file_records(package: dict) -> list[dict]:
    records = list(package["source_artifacts"].values()) + list(package["code"].values())
    records.extend(
        value for key, value in package["data"].items() if key != "model_files"
    )
    records.extend(package["data"]["model_files"])
    return records


def audit_package(package: dict, *, source_by_id: dict[str, object]) -> dict:
    if package.get("schema") != "gfmcd.p3_phase_surgery.confirmation_package.v1":
        raise ValueError("wrong confirmation-package schema")
    if package.get("model_outcomes_read") is not False:
        raise RuntimeError("confirmation package is not outcome-blind")
    cohorts = package["cohorts"]
    primary = cohorts["primary"]
    context_cds = cohorts["context_cds"]
    context_non = cohorts["context_non_cds"]
    pairs = cohorts["context_pairs"]
    excluded = set(package["exposure_firewall"]["excluded_chromosomes"])
    gates: dict[str, bool] = {}
    gates["all_bound_files_match"] = all(
        verify_file_record(record) for record in flatten_file_records(package)
    )
    gates["primary_count"] = len(primary) == package["counts"]["primary_genes"] == 943
    source_cds = json.loads(
        Path(package["source_artifacts"]["cds_cohort"]["path"]).read_text()
    )
    expected_primary = [
        {**row, "target_context": "cds"}
        for row in source_cds["records"]
        if row.get("eligible") and row.get("role") == "confirmation"
    ]
    expected_reserve = [
        row["gene_id"] for row in source_cds["records"]
        if row.get("eligible") and row.get("role") == "reserve"
    ]
    gates["primary_selection_exact_from_source"] = primary == expected_primary
    gates["primary_one_record_per_gene"] = (
        len({row["gene_id"] for row in primary}) == len(primary)
        and len({row["example_id"] for row in primary}) == len(primary)
    )
    gates["primary_roles_reconstructed"] = all(
        row["role"] == "confirmation"
        and stable_gene_role(
            row["gene_id"], namespace=package["exposure_firewall"]["stable_namespace"],
            roles=ROLE_WEIGHTS,
        ) == "confirmation"
        for row in primary
    )
    gates["reserve_count_and_disjoint"] = (
        len(cohorts["reserve_gene_ids_not_runnable"]) == 457
        and cohorts["reserve_gene_ids_not_runnable"] == expected_reserve
        and not ({row["gene_id"] for row in primary}
                 & set(cohorts["reserve_gene_ids_not_runnable"]))
    )
    gates["control_subset_exact"] = (
        cohorts["direction_control_example_ids"]
        == [row["example_id"] for row in primary[:150]]
    )
    gates["context_count"] = len(pairs) == len(context_non) == 52
    gates["context_balance_frozen"] = (
        int(cohorts["context_balance_spec"]["minimum_pairs"]) == 30
        and float(cohorts["context_balance_spec"]["max_abs_smd"]) == 0.25
        and float(cohorts["context_balance"]["max_abs_post_match_smd"]) <= 0.25
    )
    cds_ids = {row["example_id"] for row in context_cds}
    non_ids = {row["example_id"] for row in context_non}
    gates["context_pair_membership"] = all(
        pair["cds_example_id"] in cds_ids and pair["non_cds_example_id"] in non_ids
        for pair in pairs
    )
    gates["context_pairs_one_to_one"] = (
        len({pair["pair_id"] for pair in pairs}) == len(pairs)
        and len({pair["cds_example_id"] for pair in pairs}) == len(pairs)
        and len({pair["non_cds_example_id"] for pair in pairs}) == len(pairs)
        and len(cds_ids) == len(non_ids) == len(pairs)
    )
    all_rows = primary + context_cds + context_non
    gates["excluded_chromosomes_absent"] = all(row["chrom"] not in excluded for row in all_rows)
    sequence_failures = []
    for row in all_rows:
        example = source_by_id.get(row["example_id"])
        if example is None or sequence_sha256(example.sequence) != row["native_sequence_sha256"]:
            sequence_failures.append(row["example_id"])
    gates["source_sequences_reconstructed"] = not sequence_failures
    gates["context_labels"] = (
        all(row["target_context"] == "cds" for row in primary + context_cds)
        and all(row["target_context"] == "non_cds_exon" for row in context_non)
    )
    gates["control_seed_family"] = package["design"]["control_seeds"] == list(
        range(20260721, 20260740)
    )
    gates["single_primary_endpoint"] = package["decision_order"][1] == "primary_psi_c"
    return {
        "schema": "gfmcd.p3_phase_surgery.confirmation_package_audit.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "package_sha256": None,
        "gates": gates,
        "failures": sorted(name for name, passed in gates.items() if not passed),
        "sequence_failures": sorted(set(sequence_failures)),
        "counts_rederived": {
            "primary_genes": len({row["gene_id"] for row in primary}),
            "direction_control_genes": len(cohorts["direction_control_example_ids"]),
            "context_pairs": len(pairs),
            "same_gene_context_pairs": sum(bool(pair["same_gene"]) for pair in pairs),
            "chromosomes": dict(sorted(Counter(row["chrom"] for row in primary).items())),
        },
        "model_loaded": False,
        "model_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite package audit: {out}")
    package = json.loads(Path(args.package).read_text())
    source = {example.example_id: example for example in _dataset(args.gtf, args.fasta)}
    audit = audit_package(package, source_by_id=source)
    audit["package_sha256"] = sha256_file(args.package)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "p3-phase-surgery-confirmation-package-audit-v1",
        sys.argv,
        repo=REPO,
        status=audit["status"],
        notes=["Independent structural audit; no model loaded and no model outcome read."],
    )
    for path in (args.package, args.gtf, args.fasta):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({"status": audit["status"], "gates": audit["gates"]}, indent=2))
    if audit["status"] != "PASS":
        raise RuntimeError(f"confirmation-package audit failed: {audit['failures']}")


if __name__ == "__main__":
    main()
