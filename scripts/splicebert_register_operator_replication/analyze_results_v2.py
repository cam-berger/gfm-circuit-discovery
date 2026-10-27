#!/usr/bin/env python3
"""Analyze all amended v2 arms; classification remains pending independent audit."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_io import read_json, guard_paths, write_artifact, verify_artifact_manifest, require_manifest_input
from gfmcd.splicing.register_operator_measurement import requests_to_score_payload, serialize_score_payload, deserialize_score_payload, reduce_score_payload
from gfmcd.splicing.register_operator_statistics_v2 import analyze, METHOD
from gfmcd.splicing.register_operator_io_v2 import verify_freeze
from run_model_v2 import verify_stage_authorization, prerequisite_hashes, verify_unchanged_approvals


def diagnostics(analysis: dict, package: dict) -> dict:
    rows = analysis["block_endpoints"]
    psi = np.array([r["cc_max"]["psi"] for r in rows])
    omega = np.array([package["stop_diagnostics"]["selected_omega_by_block"][r["block_id"]] for r in rows])
    median = package["stop_diagnostics"]["selected_omega_median"]
    statistic = spearmanr(omega, psi)
    groups = {}
    for label, take in (("omega_above_frozen_median", omega > median), ("omega_at_or_below_frozen_median", omega <= median)):
        groups[label] = {"n_blocks": int(take.sum()), "mean_psi": float(psi[take].mean())}
    chroms = np.array([r["stratum"].split(":")[0] for r in rows])
    strands = np.array([r["stratum"].split(":")[-1] for r in rows])
    return {"stop_diagnostic": {**package["stop_diagnostics"], "spearman_rho": float(statistic.statistic),
              "spearman_two_sided_p_descriptive": float(statistic.pvalue), "subgroups": groups,
              "scope": "Composite sequence diagnostic; does not identify phase independently of stop composition."},
            "sensitivity": {"leave_one_block": {"minimum": float(((psi.sum() - psi) / (len(psi) - 1)).min()),
                                                  "maximum": float(((psi.sum() - psi) / (len(psi) - 1)).max())},
                "leave_one_chromosome": {str(c): float(psi[chroms != c].mean()) for c in sorted(set(chroms))},
                "leave_one_strand": {str(s): float(psi[strands != s].mean()) for s in sorted(set(strands))}}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    args = p.parse_args()
    freeze = verify_freeze(args.freeze, REPO)
    out = Path(freeze["outputs"]["analysis"])
    guard_paths(out)
    raw_path = Path(freeze["outputs"]["cells"])
    fs = sha256_file(args.freeze)
    approvals = verify_stage_authorization(freeze, fs, "cells", freeze_path=args.freeze)
    snapshot = prerequisite_hashes(approvals)
    manifest = verify_artifact_manifest(raw_path)
    require_manifest_input(manifest, args.freeze)
    raw_sha = sha256_file(raw_path)
    raw = read_json(raw_path)
    if (raw.get("status") != "COMPLETE" or raw.get("freeze_sha256") != fs
            or raw.get("schema") != "gfmcd.register_operator.model.v2"
            or raw.get("inference_method") != METHOD):
        raise ValueError("analysis requires the complete frozen crossed run")
    package = read_json(freeze["construction_path"])
    chosen = set(package["selected_block_ids"])
    blocks = sorted((b for b in package["original_blocks"] if b["block_id"] in chosen), key=lambda b: b["block_id"])
    inputs = [args.freeze, raw_path, Path(freeze["construction_path"]), Path(__file__), *approvals]
    prereqs = {"construction": True, "independent_audit": False}
    for k, key in (("smoke", "smoke"), ("fidelity", "fidelity_audit"), ("calibration", "calibration")):
        path = Path(freeze["outputs"][key]); receipt = read_json(path)
        prereqs[k] = receipt.get("status") == "PASS" and receipt.get("freeze_sha256") == fs
        inputs.append(path)
    payload = requests_to_score_payload(raw, blocks, prerequisites=prereqs)
    serialized = serialize_score_payload(payload)
    reduced = reduce_score_payload(deserialize_score_payload(serialized))
    analysis = analyze(reduced)
    analysis.update(diagnostics(analysis, package))
    analysis.update(status="AWAITING_INDEPENDENT_AUDIT", freeze_sha256=fs, inference_method=METHOD,
                    raw_requests_sha256=raw_sha, primary_operator="AG_to_CC:max_margin",
                    cohort_tier="outcome-aware operator verification", span_nt=120)
    verify_freeze(args.freeze, REPO)
    if sha256_file(args.freeze) != fs:
        raise ValueError("freeze changed during v2 analysis")
    verify_unchanged_approvals(freeze, fs, "cells", snapshot, freeze_path=args.freeze)
    verify_artifact_manifest(raw_path)
    if sha256_file(raw_path) != raw_sha:
        raise ValueError("raw measurements changed during v2 analysis")
    print(json.dumps(write_artifact(out, analysis, inputs=inputs,
        experiment="register-operator-v2:analysis", seeds={"bootstrap": 20260803})))


if __name__ == "__main__":
    main()
