#!/usr/bin/env python3
"""Replay the one A1 candidate on consumed v1 simulations; never fresh evidence."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_io import (read_json, write_artifact, guard_paths,
    verify_freeze, verify_artifact_manifest, manifest_path)
from gfmcd.splicing.register_operator_io_v2 import BASE_FREEZE_SHA256
from gfmcd.splicing import register_operator_statistics_v2 as statistics
import calibrate_statistics as old
import calibrate_statistics_v2 as revised

STUDY = REPO / "studies/splicebert_register_operator_replication"
RESULTS = REPO / "results/splicebert_register_operator_replication"
OLD_CAL_SHA = "4c13d30ee8dd79a1c16e52ced24f955fec7c8285b3bf2cd4400a776bf5959ee7"
OLD_AUDIT_SHA = "91bda8a6e3cced9f708a0b468e4c2af79f114368a1b45f7f84d29eb8eddbcb65"


def point_memorial(before, after):
    """Every measured block endpoint, mean and pseudo draw must be unchanged."""
    for key in ("block_endpoints", "matching_means", "competitors"):
        if before[key] != after[key]:
            raise ValueError(f"correction changed observed {key}")
    for key in ("seeds", "control_means", "maximum_control", "target_exceeds_all",
                "exact_add_one_one_sided_p"):
        if before["pseudo_phase"][key] != after["pseudo_phase"][key]:
            raise ValueError(f"correction changed pseudo {key}")
    if before["primary_psi"]["mean"] != after["primary_psi"]["mean"]:
        raise ValueError("correction changed primary mean")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out, _ = guard_paths(args.out)
    records_out = out.with_suffix(".records.jsonl.gz")
    if records_out.exists() or records_out.is_symlink():
        raise FileExistsError(records_out)
    attempt = out.with_suffix(".attempt.json")
    guard_paths(attempt)
    base = STUDY / "protocol_freeze_v1.json"
    calibration = RESULTS / "statistical_calibration_v1.json"
    audit_path = RESULTS / "analysis_v1.audit.json"
    if (sha256_file(base) != BASE_FREEZE_SHA256 or sha256_file(calibration) != OLD_CAL_SHA
            or sha256_file(audit_path) != OLD_AUDIT_SHA):
        raise ValueError("consumed v1 evidence differs")
    freeze = verify_freeze(base, REPO)
    verify_artifact_manifest(calibration)
    verify_artifact_manifest(audit_path)
    cal = read_json(calibration)
    audit = read_json(audit_path)
    if audit["status"] != "PASS" or cal["status"] != "INSTRUMENT_UNRESOLVED":
        raise ValueError("expected independently audited failed v1 instrument")
    records = Path(cal["records"]["path"])
    if sha256_file(records) != cal["records"]["sha256"]:
        raise ValueError("consumed realization stream changed")
    selection = Path(freeze["selection_path"])
    ids, strata = old.load_selection(selection)
    sources = [Path(__file__), Path(old.__file__), Path(revised.__file__),
        REPO / "src/gfmcd/splicing/register_operator_statistics_v2.py",
        REPO / "src/gfmcd/splicing/register_operator_io_v2.py",
        STUDY / "AMENDMENT_A1.md"]
    inputs = [base, calibration, manifest_path(calibration), audit_path, manifest_path(audit_path),
              selection, records, *sources]
    initial_hashes = {str(p): sha256_file(p) for p in inputs}
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    write_artifact(attempt, {"status": "ATTEMPT", "created_at_utc": started,
        "inference_method": statistics.METHOD, "fresh_streams_opened": False,
        "scope": "One candidate; complete already-consumed v1 replay as development only.",
        "input_hashes": initial_hashes}, inputs=inputs,
        experiment="register-operator:a1-development-attempt")
    context = statistics.make_inference_context(ids, strata)
    smoke = revised.run_smoke()
    counts, classes, seen = Counter(), Counter(), set()
    noncoverage = {name: 0 for name in old.FAMILIES}
    null_positive = detection = material = memorials = 0
    errors = []
    with records_out.open("xb") as raw_out, gzip.open(records, "rt") as source:
        with gzip.GzipFile(fileobj=raw_out, mode="wb", mtime=0) as destination:
            for line in source:
                row = json.loads(line)
                if "record_type" in row:
                    continue
                key = (row["phase"], row["index"], row["delta"])
                if key in seen:
                    raise ValueError("duplicate consumed realization")
                seen.add(key)
                counts[row["phase"]] += 1
                result = {"phase": key[0], "index": key[1], "delta": key[2],
                    "raw_four_logit_payload_sha256": row.get("raw_four_logit_payload_sha256")}
                try:
                    if row["status"] != "COMPLETE":
                        raise ValueError("v1 realization incomplete")
                    analysis = statistics.analyze(row["block_sufficient_statistics"], context=context)
                    point_memorial(row["analysis"], analysis)
                    memorials += 1
                    failures = old.family_noncoverage(analysis, row["fixture"]["population_truths"])
                    if key[0] == "null":
                        null_positive += analysis["classification"] in old.POSITIVE_CLASSES
                        for name, value in failures.items():
                            noncoverage[name] += int(value)
                    if key[0] == "plants":
                        if key[2] == .05:
                            detection += analysis["primary_psi"]["ci95"][0] > 0
                        else:
                            material += analysis["classification"] in old.MATERIAL_CLASSES
                    classes[key[0] + ":" + analysis["classification"]] += 1
                    result.update(status="COMPLETE", family_noncoverage=failures,
                                  analysis={k: v for k, v in analysis.items() if k != "block_endpoints"})
                except Exception as exc:
                    error = {"phase": key[0], "index": key[1], "delta": key[2],
                             "error": f"{type(exc).__name__}: {exc}"}
                    errors.append(error)
                    result.update(status="ERROR", error=error)
                destination.write(old.canonical_bytes(result) + b"\n")
                if sum(counts.values()) % 100 == 0:
                    print(json.dumps({"consumed_realizations_replayed": sum(counts.values())}), flush=True)
    expected = {("development", i, 0 if i % 4 < 2 else (.05 if i % 4 == 2 else .15)) for i in range(256)}
    expected |= {("null", i, 0) for i in range(1000)}
    expected |= {("plants", i, d) for i in range(256) for d in (.05, .15)}
    if seen != expected:
        errors.append({"error": "consumed realization inventory differs"})
    if smoke["status"] != "SMOKE_PASS":
        errors.append({"error": "deterministic memorial failed", "smoke": smoke})
    for path, digest in initial_hashes.items():
        if sha256_file(path) != digest:
            errors.append({"error": "input changed during execution", "path": path})
    acceptance = old.evaluate_acceptance(null_positive=int(null_positive), noncoverage=noncoverage,
        detection_successes=int(detection), material_successes=int(material), execution_errors=errors)
    passed = acceptance["all_passed"] and memorials == 1768
    result = {"schema": "gfmcd.register_operator.a1_development.v2",
        "status": "PASS" if passed else "INSTRUMENT_UNRESOLVED", "passed": passed,
        "inference_method": statistics.METHOD, "base_freeze_sha256": BASE_FREEZE_SHA256,
        "created_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - clock, "fresh_streams_opened": False,
        "model_loaded": False, "new_model_outputs": 0, "counts": dict(counts),
        "point_and_pseudo_memorials_passed": memorials, "smoke": smoke,
        "acceptance": acceptance, "classification_counts": dict(classes),
        "records": {"path": str(records_out.resolve()), "sha256": sha256_file(records_out)},
        "scope": "Already-consumed simulations; development evidence only, never independent validation."}
    write_artifact(out, result, inputs=[*inputs, attempt, manifest_path(attempt), records_out],
        experiment="register-operator:a1-development", seeds={"bootstrap": 20260803})
    print(json.dumps({"status": result["status"], "acceptance": acceptance}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
