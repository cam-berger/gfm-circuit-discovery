#!/usr/bin/env python3
"""Record an independently audited calibration stop without inventing model outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_io import (
    read_json, require_manifest_input, verify_artifact_manifest, verify_freeze,
    write_artifact,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    freeze = verify_freeze(args.freeze, REPO)
    freeze_sha = sha256_file(args.freeze)
    calibration_path = Path(freeze["outputs"]["calibration"])
    calibration = read_json(calibration_path)
    verify_artifact_manifest(calibration_path)
    audit = read_json(args.audit)
    audit_manifest = verify_artifact_manifest(args.audit)
    require_manifest_input(audit_manifest, args.freeze)
    require_manifest_input(audit_manifest, calibration_path)
    if (audit.get("status") != "PASS" or audit.get("passed") is not True
            or audit.get("classification") != "INSTRUMENT_UNRESOLVED"
            or audit.get("authorized_for_cc") is not False
            or audit.get("freeze_sha256") != freeze_sha):
        raise ValueError("independent audit does not verify this stopped run")
    if (calibration.get("status") != "INSTRUMENT_UNRESOLVED"
            or calibration.get("freeze_sha256") != freeze_sha
            or calibration.get("model_loaded") is not False
            or calibration.get("new_model_outputs") != 0
            or calibration["acceptance"]["all_passed"] is not False):
        raise ValueError("calibration is not the declared pre-model stop")
    unrun = ("fidelity", "fidelity_attempt", "fidelity_audit", "native",
             "native_attempt", "cells", "cells_attempt", "analysis")
    for name in unrun:
        path = Path(freeze["outputs"][name])
        if path.exists() or path.is_symlink():
            raise ValueError(f"cannot attest zero subsequent execution: {path}")
    value = {
        "schema": "gfmcd.register_operator.stopped_result.v1",
        "status": "INSTRUMENT_UNRESOLVED",
        "classification": "INSTRUMENT_UNRESOLVED",
        "stop_stage": "statistical_calibration",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": freeze_sha,
        "calibration_sha256": sha256_file(calibration_path),
        "independent_audit_sha256": sha256_file(args.audit),
        "primary_operator": "AG_to_CC:max_margin",
        "cohort_tier": "outcome-aware operator verification",
        "span_nt": 120,
        "n_selected_blocks": 342,
        "resolution_mde80_limit": 0.05,
        "new_model_outputs": 0,
        "authorized_for_cc": False,
        "calibration_acceptance": calibration["acceptance"],
        "stages": {
            "construction_audit": "PASS",
            "protocol_freeze": "FROZEN",
            "smoke": "PASS",
            "statistical_calibration": "FAIL",
            "historical_forward_fidelity": "NOT_RUN",
            "native_cc": "NOT_RUN",
            "crossed_cc": "NOT_RUN",
            "target_analysis": "NOT_RUN",
            "independent_stop_audit": "PASS",
        },
        "model_endpoints": {name: None for name in (
            "primary_psi", "matching_means", "directional_bounds",
            "pseudo_control_maximum", "pseudo_add_one_p", "ac_bridge",
            "stop_load_association", "actual_cc_mde80")},
        "target_gate_status": {
            "construction_and_prerequisites": "FAIL: statistical calibration; fidelity not run",
            **{name: "NOT_RUN" for name in (
                "source_edge_positive_control", "crossed_lattice_capability",
                "negative_control", "resolution", "corruption_robustness",
                "protected_mlm_health", "phase_blind_site_health", "numerics",
                "independent_target_audit")},
        },
        "licensed_conclusion": "The frozen statistical instrument failed its synthetic coverage gate. No new positive, null, inverted, or equivalent model-behavior conclusion is licensed.",
    }
    inputs = [args.freeze, calibration_path, args.audit, Path(__file__),
              Path(freeze["outputs"]["smoke"]), Path(freeze["construction_audit_path"])]
    print(json.dumps(write_artifact(freeze["outputs"]["final"], value,
        inputs=inputs, experiment="register-operator:audited-calibration-stop")))


if __name__ == "__main__":
    main()
