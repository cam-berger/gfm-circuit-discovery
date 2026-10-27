#!/usr/bin/env python3
"""Materialize only an independently audited A1 conclusion, including honest stops."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_io import (read_json, write_artifact,
    verify_artifact_manifest, require_manifest_input, manifest_path)
from gfmcd.splicing.register_operator_io_v2 import verify_freeze
from gfmcd.splicing.register_operator_statistics_v2 import METHOD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    args = parser.parse_args()
    freeze = verify_freeze(args.freeze, REPO)
    fs = sha256_file(args.freeze)
    audit_path = Path(freeze["outputs"]["result_audit"])
    audit = read_json(audit_path)
    manifest = verify_artifact_manifest(audit_path)
    require_manifest_input(manifest, args.freeze)
    auditor = REPO / "scripts/splicebert_register_operator_replication/audit_results_v2.py"
    require_manifest_input(manifest, auditor)
    if (audit.get("schema") != "gfmcd.register_operator.result_audit.v2"
            or audit.get("status") != "PASS" or manifest.get("status") != "PASS" or audit.get("passed") is not True
            or audit.get("authorized_for_final_result") is not True
            or audit.get("inference_method") != METHOD or audit.get("freeze_sha256") != fs
            or audit.get("auditor_sha256") != sha256_file(auditor)):
        raise ValueError("current independent audit does not authorize a final result")
    complete = audit.get("model_evidence_complete") is True
    if not complete and audit.get("classification") != "INSTRUMENT_UNRESOLVED":
        raise ValueError("missing model evidence cannot support a behavioral classification")
    if complete:
        for name in ("native", "cells", "analysis", "measurement_audit"):
            path = Path(freeze["outputs"][name])
            if not path.is_file():
                raise ValueError(f"complete model evidence requires {name}")
            require_manifest_input(manifest, path)
        if not isinstance(audit.get("recomputed"), dict):
            raise ValueError("complete model evidence requires reconstructed endpoints")
    inputs = [args.freeze, audit_path, manifest_path(audit_path), Path(__file__)]
    receipts = {}
    for name in ("smoke", "calibration", "calibration_audit", "fidelity", "fidelity_audit",
                 "native", "cells", "analysis", "measurement_audit"):
        path = Path(freeze["outputs"][name])
        if path.is_file():
            verify_artifact_manifest(path)
            inputs.extend((path, manifest_path(path)))
            receipt = read_json(path)
            receipts[name] = {"status": receipt.get("status"), "sha256": sha256_file(path)}
        else:
            receipts[name] = {"status": "NOT_RUN", "sha256": None}
    result = {"schema": "gfmcd.register_operator.final_result.v2", "status": audit["classification"],
        "classification": audit["classification"], "inference_method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_sha256": fs,
        "independent_audit_sha256": sha256_file(audit_path),
        "primary_operator": "AG_to_CC:max_margin", "cohort_tier": "outcome-aware operator verification",
        "span_nt": 120, "n_selected_blocks": 342, "resolution_mde80_limit": .05,
        "model_evidence_complete": complete, "stages": receipts,
        "gates": audit["gates"], "failed_gates": audit["failed_gates"], "unrun_gates": audit["unrun_gates"],
        "observed_model_stages": audit.get("observed_model_stages", {}),
        "model_endpoints": audit.get("recomputed") if complete else None,
        "licensed_conclusion": ("Audited behavior in the frozen outcome-aware 342-block, 120-nt domain, subject to every recorded instrument gate."
            if complete and audit["classification"] != "INSTRUMENT_UNRESOLVED" else
            "The amended instrument is unresolved. No positive, null, inverted or equivalent model-behavior conclusion is licensed."),
        "scope": "This result does not establish mediation, genome-wide transfer, or phase independent of stop composition."}
    print(json.dumps(write_artifact(freeze["outputs"]["final"], result, inputs=inputs,
        experiment="register-operator:audited-result-v2")))


if __name__ == "__main__":
    main()
