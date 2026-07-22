#!/usr/bin/env python
"""Independently audit the sealed runner's development-only smoke artifact."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.p3_stats import gene_paired_phase_table  # noqa: E402
from analyze_p3_phase_surgery_confirmation import instrument_gates  # noqa: E402


METRICS = ("clean_margin", "corrupt_margin", "coupling", "phasor_signed")


def row_key(row: dict) -> tuple:
    return (
        row["example_id"], row["context"], int(row["block_index"]), row["direction"],
        int(row["amount"]),
    )


def audit_smoke(package: dict, smoke: dict) -> dict:
    package_sha = smoke.get("package_sha256")
    primary = smoke.get("primary", ())
    controls = smoke.get("direction_control", {})
    non_cds = smoke.get("non_cds", {})
    gates = {
        "development_only": smoke.get("target_outcomes_read") is False,
        "primary_grid": len(primary) == 2 and all(len(shard.get("rows", ())) == 72 for shard in primary),
        "control_grid": len(controls.get("rows", ())) == 480,
        "non_cds_grid": len(non_cds.get("rows", ())) == 72,
        "component_family": sorted({row["component"] for row in controls.get("rows", ())})
        == ["p3", *[f"signed_permutation_{index:02d}" for index in range(19)]],
    }
    rows = [row for shard in primary for row in shard.get("rows", ())]
    rows += list(controls.get("rows", ())) + list(non_cds.get("rows", ()))
    gates["finite"] = bool(rows) and all(
        math.isfinite(float(row[metric])) for row in rows for metric in METRICS
    )
    gates["attention_normalized"] = bool(rows) and all(
        abs(float(row["attention_row_sum_full"]) - 1.0) <= 1e-4 for row in rows
    )
    primary_first = {
        row_key(row): row for row in primary[0]["rows"] if int(row["amount"]) in (2, 3, 4)
    }
    control_p3 = {
        row_key(row): row for row in controls["rows"] if row["component"] == "p3"
    }
    gates["p3_path_reproduces_primary"] = (
        set(primary_first) == set(control_p3)
        and max(
            abs(float(primary_first[key][metric]) - float(control_p3[key][metric]))
            for key in primary_first for metric in METRICS
        ) <= 1e-8
    )
    gates["primary_contrast_reconstructs"] = len(gene_paired_phase_table(
        [row for shard in primary for row in shard["rows"]],
        metrics=("coupling",),
        target_context="cds",
    )) == 2
    gates["non_cds_contrast_reconstructs"] = len(gene_paired_phase_table(
        non_cds["rows"], metrics=("coupling",), target_context="non_cds_exon"
    )) == 1
    gates.update({f"frozen_instrument:{key}": value for key, value in instrument_gates(package).items()})
    return {
        "schema": "gfmcd.p3_phase_surgery.confirmation_smoke_audit.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "package_sha256": package_sha,
        "gates": gates,
        "failures": sorted(key for key, value in gates.items() if not value),
        "target_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-audit", required=True)
    parser.add_argument("--smoke", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite smoke audit: {out}")
    package = json.loads(Path(args.package).read_text())
    package_audit = json.loads(Path(args.package_audit).read_text())
    smoke = json.loads(Path(args.smoke).read_text())
    package_sha = sha256_file(args.package)
    if package_audit.get("status") != "PASS" or package_audit.get("package_sha256") != package_sha:
        raise RuntimeError("package audit is not green for this package")
    if smoke.get("package_sha256") != package_sha:
        raise RuntimeError("development smoke belongs to another package")
    audit = audit_smoke(package, smoke)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "p3-phase-surgery-confirmation-smoke-audit-v1",
        sys.argv,
        repo=REPO,
        status=audit["status"],
        notes=["Development roles only; no untouched outcome was read."],
    )
    for path in (args.package, args.package_audit, args.smoke):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({"status": audit["status"], "failures": audit["failures"]}, indent=2))
    if audit["status"] != "PASS":
        raise RuntimeError(f"development smoke audit failed: {audit['failures']}")


if __name__ == "__main__":
    main()
