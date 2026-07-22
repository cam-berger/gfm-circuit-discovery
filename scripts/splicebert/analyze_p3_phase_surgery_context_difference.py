#!/usr/bin/env python
"""Direct paired CDS-minus-non-CDS inference for local P3 phase surgery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.p3_stats import (  # noqa: E402
    bootstrap_studentized_max_t,
    interval_classification,
    mde80,
)


ENDPOINTS = ("coupling", "phasor_signed")
SESOI = {"coupling": 0.05, "phasor_signed": 0.10}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cds-analysis", required=True)
    parser.add_argument("--non-cds-analysis", required=True)
    parser.add_argument("--match", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--out", default="results/phase-lattice/p3_phase_surgery_context_difference.json"
    )
    args = parser.parse_args()
    if args.bootstrap < 999:
        parser.error("--bootstrap must be at least 999")
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite context-difference analysis: {out}")
    cds = json.loads(Path(args.cds_analysis).read_text())
    non = json.loads(Path(args.non_cds_analysis).read_text())
    match = json.loads(Path(args.match).read_text())
    if cds.get("schema") != "gfmcd.p3_phase_surgery.preflight_analysis.v1":
        raise ValueError("wrong CDS analysis schema")
    if non.get("schema") != "gfmcd.p3_phase_surgery.preflight_analysis.v1":
        raise ValueError("wrong non-CDS analysis schema")
    if match.get("schema") != "gfmcd.p3_phase_surgery.context_match.v1":
        raise ValueError("wrong context-match schema")
    if match.get("model_outcomes_read") is not False:
        raise RuntimeError("context match was not outcome-blind")
    cds_rows = {str(row["gene_id"]): row for row in cds["per_gene"]}
    non_rows = {str(row["gene_id"]): row for row in non["per_gene"]}
    pair_rows = []
    for pair in match["pairs"]:
        left = cds_rows[pair["cds_gene_id"]]
        right = non_rows[pair["non_cds_gene_id"]]
        pair_rows.append({
            "pair_id": pair["pair_id"],
            "cds_gene_id": pair["cds_gene_id"],
            "non_cds_gene_id": pair["non_cds_gene_id"],
            "strand": pair["strand"],
            "match_distance": pair["distance"],
            **{
                f"delta_{endpoint}": float(left[f"psi_{endpoint}"])
                - float(right[f"psi_{endpoint}"])
                for endpoint in ENDPOINTS
            },
        })
    matrix = np.asarray([
        [float(row[f"delta_{endpoint}"]) for endpoint in ENDPOINTS]
        for row in pair_rows
    ])
    simultaneous = bootstrap_studentized_max_t(
        matrix, names=ENDPOINTS, seed=args.seed, n_bootstrap=args.bootstrap
    )
    endpoints = {}
    for index, endpoint in enumerate(ENDPOINTS):
        values = matrix[:, index]
        endpoints[endpoint] = {
            "cds_minus_non_cds_mean": simultaneous.means[index],
            "simultaneous_ci": [simultaneous.lower[index], simultaneous.upper[index]],
            "adjusted_p": simultaneous.adjusted_p[index],
            "sesoi": SESOI[endpoint],
            "classification_development_only": interval_classification(
                simultaneous.lower[index], simultaneous.upper[index], sesoi=SESOI[endpoint]
            ),
            "mde80": mde80(values, critical_value=simultaneous.critical_value),
            "pair_sd": float(values.std(ddof=1)),
            "sign_fraction_positive": float(np.mean(values > 0)),
        }
    result = {
        "schema": "gfmcd.p3_phase_surgery.context_difference.v1",
        "status": "DEVELOPMENT_CALIBRATION_NOT_CONFIRMATORY",
        "cds_analysis_sha256": sha256_file(args.cds_analysis),
        "non_cds_analysis_sha256": sha256_file(args.non_cds_analysis),
        "match_sha256": sha256_file(args.match),
        "n_pairs": len(pair_rows),
        "family": {
            "endpoints": list(ENDPOINTS),
            "critical_value": simultaneous.critical_value,
            "n_bootstrap": simultaneous.n_bootstrap,
        },
        "endpoints": endpoints,
        "match_balance": match["balance"],
        "pairs": pair_rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "p3-phase-surgery-context-difference-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_CALIBRATION_NOT_CONFIRMATORY",
        config={"bootstrap": args.bootstrap, "seed": args.seed, "sesoi": SESOI},
        seeds={"bootstrap": args.seed},
    )
    for path in (
        args.cds_analysis,
        args.non_cds_analysis,
        args.match,
        REPO / "src/gfmcd/splicing/p3_stats.py",
        REPO / "scripts/splicebert/analyze_p3_phase_surgery_context_difference.py",
    ):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({"n_pairs": len(pair_rows), "endpoints": endpoints}, indent=2))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
