#!/usr/bin/env python
"""Independent artifact audit for the QK/V mechanism preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


HEADS = {"L0H0", "L3H1", "L5H15"}
CONDITIONS = {"QK", "V", "QKV", "WHOLE"}
EXCLUDED_CHROMS = {"chr8", "chr9", "chr11", "chr16", "chr19"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--reference-rows", required=True)
    parser.add_argument("--audit", default=None)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    prefix = Path(args.prefix)
    rows_path = Path(f"{prefix}.rows.jsonl")
    summary_path = Path(f"{prefix}.summary.json")
    manifest_path = Path(f"{prefix}.manifest.json")
    audit_path = Path(args.audit) if args.audit else Path(f"{prefix}.audit.json")
    if audit_path.exists():
        raise FileExistsError(f"refuse to overwrite immutable audit {audit_path}")
    summary = json.loads(summary_path.read_text())
    rows = read_jsonl(rows_path)
    reference_rows = read_jsonl(Path(args.reference_rows))
    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []

    require(summary["status"] == "PREFLIGHT_PASS" and summary["n_failures"] == 0,
            "summary is not a zero-failure pass", failures)
    require(summary["validation_loaded"] is False, "validation_loaded is not false", failures)
    require(summary["n_rows"] == 48 == len(rows), "row count is not 4x3x4", failures)
    require(set(summary["heads"]) == HEADS, "fixed head set mismatch", failures)
    keys = {(row["packet_id"], row["component_id"], row["condition"]) for row in rows}
    require(len(keys) == len(rows), "duplicate packet/head/condition row", failures)
    packet_ids = {row["packet_id"] for row in rows}
    require(len(packet_ids) == 4, "expected four packets", failures)
    require({row["target_role"] for row in rows} == {"donor", "acceptor"},
            "both target roles are not represented", failures)
    require(all(row["analysis_split"] == "atlas_discovery" for row in rows),
            "non-discovery row present", failures)
    require(all(row["chrom"] not in EXCLUDED_CHROMS for row in rows),
            "excluded chromosome present", failures)
    require({row["component_id"] for row in rows} == HEADS, "row head set mismatch", failures)
    require({row["condition"] for row in rows} == CONDITIONS, "condition set mismatch", failures)
    require(all(row["estimand"] == "B" and row["source"] == "partner_destroyed_same_packet"
                for row in rows), "estimand/source mismatch", failures)

    by_key = {(row["packet_id"], row["component_id"], row["condition"]): row for row in rows}
    numerical = (
        "baseline_margin", "intervened_margin", "effect_margin",
        "baseline_logprob", "intervened_logprob", "baseline_health_margin",
        "intervened_health_margin", "health_margin_damage",
    )
    for row in rows:
        require(all(math.isfinite(float(row[field])) for field in numerical),
                f"non-finite row {row['packet_id']}|{row['component_id']}|{row['condition']}", failures)
        require(math.isclose(
            row["effect_margin"], row["baseline_margin"] - row["intervened_margin"],
            rel_tol=0.0, abs_tol=1e-12,
        ), f"effect arithmetic mismatch {row['packet_id']}|{row['component_id']}|{row['condition']}", failures)
    for packet_id in packet_ids:
        for component in HEADS:
            qkv = by_key[(packet_id, component, "QKV")]
            whole = by_key[(packet_id, component, "WHOLE")]
            for field in (
                "intervened_margin", "intervened_logprob", "effect_margin",
                "intervened_health_margin", "health_margin_damage",
            ):
                require(qkv[field] == whole[field],
                        f"QKV/whole bridge mismatch {packet_id}|{component}|{field}", failures)

    reference = {
        (row["packet_id"], row["component_id"]): float(row["effect_margin"])
        for row in reference_rows
        if row["estimand"] == "B_partner_dependence" and row["component_id"] in HEADS
    }
    require(len(reference) == 12, "independent reference grid is not 4x3", failures)
    for packet_id in packet_ids:
        for component in HEADS:
            require(by_key[(packet_id, component, "WHOLE")]["effect_margin"] == reference[(packet_id, component)],
                    f"independent whole-effect mismatch {packet_id}|{component}", failures)

    gate_names = {gate["gate"] for gate in summary["gates"]}
    require(len(gate_names) == 14 == summary["n_gates"], "gate grid mismatch", failures)
    require(all(gate["passed"] and gate["value"] <= gate["threshold"] for gate in summary["gates"]),
            "recorded gate failure/threshold violation", failures)

    output_records = {str(Path(record["path"]).resolve()): record for record in manifest["outputs"]}
    for path in (rows_path, summary_path):
        record = output_records.get(str(path.resolve()))
        require(record is not None, f"manifest missing output {path}", failures)
        if record:
            require(record["sha256"] == sha256_file(path) and record["size_bytes"] == path.stat().st_size,
                    f"manifest output hash/size mismatch {path}", failures)
    require(manifest["config"]["validation_loaded"] is False,
            "manifest validation_loaded is not false", failures)
    require(all("internal_validation" not in record["path"] for record in manifest["inputs"]),
            "manifest opened internal validation", failures)

    report = {
        "schema": "head_context_atlas.mechanism_preflight_audit/1",
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": "independent artifact/mechanics audit; no mechanism claim",
        "n_packets": len(packet_ids),
        "n_heads": len(HEADS),
        "n_rows": len(rows),
        "n_failures": len(failures),
        "failures": failures,
        "rows_sha256": sha256_file(rows_path),
        "summary_sha256": sha256_file(summary_path),
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
