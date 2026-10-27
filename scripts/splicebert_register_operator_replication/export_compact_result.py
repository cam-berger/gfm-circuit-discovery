#!/usr/bin/env python3
"""Export a compact, source-bound account of the completed run without reclassifying it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RESULTS = "results/splicebert_register_operator_replication/"
SOURCES = {
    "final": RESULTS + "result_v2.json",
    "audit": RESULTS + "analysis_v2.audit.json",
    "measurement": RESULTS + "measurement_v2.audit.json",
    "calibration": RESULTS + "statistical_calibration_v2.json",
    "v1": RESULTS + "result_v1.json",
    "freeze": "studies/splicebert_register_operator_replication/protocol_freeze_v2.json",
}


def record(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def build() -> dict[str, dict]:
    bindings = {name: record(ROOT / relative) for name, relative in SOURCES.items()}
    docs = {name: json.loads((ROOT / relative).read_text()) for name, relative in SOURCES.items()}
    final, audit = docs["final"], docs["audit"]
    if final["independent_audit_sha256"] != bindings["audit"]["sha256"]:
        raise ValueError("final result/audit binding differs")
    if final["freeze_sha256"] != bindings["freeze"]["sha256"] or audit["status"] != "PASS":
        raise ValueError("freeze or audit prerequisite differs")
    for key in ("classification", "gates", "failed_gates", "unrun_gates"):
        if final[key] != audit[key]:
            raise ValueError(f"final result/audit mismatch: {key}")
    e = final["model_endpoints"]
    if e != audit["recomputed"]:
        raise ValueError("independently recomputed endpoints differ")
    arms = sorted(e["arms"])
    columns = ["block_id", "stratum"] + arms
    rows = [[row["block_id"], row["stratum"]] + [row[arm]["psi"] for arm in arms]
            for row in e["block_endpoints"]]
    table = {"schema": "gfmcd.register-operator.block-psi/1", "units": "logit units",
             "source": bindings["final"], "source_pointer": "/model_endpoints/block_endpoints",
             "columns": columns, "rows": rows}
    pseudo = {key: value for key, value in e["pseudo_phase"].items() if key != "seeds"}
    summary = {
        "schema": "gfmcd.register-operator.compact-result/1", "claim_aliases": ["C45", "C46"],
        "registration_kind": "retrospective artifact registration of completed frozen analyses",
        "classification": final["classification"], "cohort_tier": final["cohort_tier"],
        "primary_operator": final["primary_operator"], "span_nt": final["span_nt"],
        "n_blocks": e["n_blocks"], "n_source_genes": 7 * e["n_blocks"],
        "n_strata": e["n_strata"], "units": "logit units",
        "resolution_mde80_limit": final["resolution_mde80_limit"],
        "threshold_changed": False, "new_model_forwards": 0,
        "gates": final["gates"], "failed_gates": final["failed_gates"], "unrun_gates": final["unrun_gates"],
        "primary_psi": e["primary_psi"], "arms": e["arms"],
        "matching_means": e["matching_means"], "directions": e["directions"],
        "mismatch_equivalence": e["mismatch_equivalence"], "pseudo_phase": pseudo,
        "bridge_diagnostics": e["bridge_diagnostics"], "instrument": e["instrument"],
        "stop_diagnostic": {key: value for key, value in e["stop_diagnostic"].items()
                            if key != "selected_omega_by_block"},
        "inference": {"method": e["inference_method"], "draws": e["bootstrap_draws"],
                      "seed": e["bootstrap_seed"], "stratification": "chromosome x strand",
                      "residual_scale": "sqrt(n_h / (n_h - 1)) within each stratum",
                      "mde_factor": 1.96 + 0.8416212335729143},
        "audit": {"status": audit["status"], "raw_arithmetic": audit["raw_arithmetic"],
                  "producer_comparison": audit["producer_comparison"],
                  "measurement_status": docs["measurement"]["status"]},
        "calibration": {"v2_status": docs["calibration"]["status"],
                        "v2_acceptance": docs["calibration"]["acceptance"]},
        "preserved_v1": {"classification": docs["v1"]["classification"],
                         "stop_stage": docs["v1"]["stop_stage"], "new_model_outputs": 0,
                         "directional_noncoverage": 83, "realizations": 1000, "maximum_allowed": 71},
        "scope": ["Outcome-aware verification on a motif-clean subset of the original C43 cohort.",
                  "The positive estimate does not pass the frozen all-gate confirmation rule.",
                  "One 120-nt span; no independent biological replication or mediation inference.",
                  "Both mismatch intervals extend outside the equivalence band; 'only when matched' is unsupported.",
                  "Phase and stop composition remain confounded."],
        "reanalysis_scope": "The block table permits scalar Psi/CI/MDE replay for all six arms. Other gates and controls are retained audit findings; raw model forwards are not replayed by this export.",
        "sources": bindings,
    }
    return {"summary_v2.json": summary, "block_psi_v2.json": table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    documents = build()
    if args.verify:
        for name, expected in documents.items():
            if json.loads((args.out_dir / name).read_text()) != expected:
                raise ValueError(f"compact record differs from audited sources: {name}")
        print(json.dumps({"status": "SOURCE_EQUIVALENT", "records": len(documents), "new_model_forwards": 0}))
        return
    paths = [args.out_dir / name for name in documents]
    if any(path.exists() or path.with_suffix(".manifest.json").exists() for path in paths):
        raise FileExistsError("refusing to replace a compact evidence artifact")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        path = args.out_dir / name
        # Row-oriented tables avoid millions of indentation-only lines.
        text = json.dumps(document, indent=2, sort_keys=True) + "\n"
        if name == "block_psi_v2.json":
            header = {key: value for key, value in document.items() if key != "rows"}
            text = json.dumps(header, indent=2)[:-2] + ',\n  "rows": [\n' + ",\n".join(
                "    " + json.dumps(row) for row in document["rows"]) + "\n  ]\n}\n"
        with path.open("x") as handle:
            handle.write(text)
        manifest = {"experiment_id": "register-operator:compact-export-v2", "status": "DERIVED_REPORT",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
                    "command": ["scripts/splicebert_register_operator_replication/export_compact_result.py",
                                "--out-dir", args.out_dir.as_posix()],
                    "inputs": [record(ROOT / relative) for relative in SOURCES.values()] + [record(Path(__file__).resolve())],
                    "outputs": [record(path.resolve())],
                    "seeds": {"recorded_bootstrap": document.get("inference", {}).get("seed", 20260803)},
                    "config": {"new_model_forwards": 0, "threshold_changed": False,
                               "classification": "INSTRUMENT_UNRESOLVED"}}
        with path.with_suffix(".manifest.json").open("x") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "EXPORTED", "files": [str(p) for p in paths], "new_model_forwards": 0}))


if __name__ == "__main__":
    main()
