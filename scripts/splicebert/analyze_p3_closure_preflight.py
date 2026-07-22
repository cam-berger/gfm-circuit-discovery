#!/usr/bin/env python
"""Independent development analysis for the P3 closure row artifact."""
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
    gene_paired_phase_table,
    interval_classification,
    mde80,
)


ENDPOINTS = ("coupling", "phasor_signed")
SESOI = {"coupling": 0.05, "phasor_signed": 0.10}


def dominant_class_tracking(result: dict) -> dict[str, object]:
    native = {
        row["example_id"]: int(np.argmax(row["phasor_class_masses"]))
        for row in result["native_rows"]
    }
    grouped = {}
    for context in ("cds", "upstream_intron"):
        grouped[context] = {}
        for direction in ("left", "right"):
            grouped[context][direction] = {}
            sign = -1 if direction == "left" else 1
            for amount in range(2, 11):
                rows = [
                    row for row in result["rows"]
                    if row["context"] == context and row["direction"] == direction
                    and int(row["amount"]) == amount
                ]
                observed = [int(np.argmax(row["phasor_class_masses"])) for row in rows]
                baseline = [native[row["example_id"]] for row in rows]
                expected = [(base + sign * amount) % 3 for base in baseline]
                grouped[context][direction][str(amount)] = {
                    "n": len(rows),
                    "dominant_class_counts": np.bincount(observed, minlength=3).tolist(),
                    "same_as_native_fraction": float(np.mean(np.asarray(observed) == baseline)),
                    "content_following_expected_fraction": float(
                        np.mean(np.asarray(observed) == expected)
                    ),
                    "note": "descriptive; downstream phasor need not obey a hard rotation law",
                }
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--out", default="results/phase-lattice/p3_closure_preflight_analysis.json")
    args = parser.parse_args()
    if args.bootstrap < 99:
        parser.error("--bootstrap must be at least 99")
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite analysis artifact: {out}")
    source = json.loads(Path(args.input).read_text())
    if source.get("schema") != "gfmcd.p3_closure.preflight.v1":
        raise ValueError("wrong P3 closure preflight schema")
    if source.get("status") != "DEVELOPMENT_OUTCOME_AWARE":
        raise RuntimeError("preflight analysis is development-only")

    per_gene = gene_paired_phase_table(source["rows"], metrics=ENDPOINTS)
    matrix = np.asarray([
        [float(row[f"psi_{endpoint}"]) for endpoint in ENDPOINTS]
        for row in per_gene
    ])
    simultaneous = bootstrap_studentized_max_t(
        matrix, names=ENDPOINTS, seed=args.seed, n_bootstrap=args.bootstrap
    )
    endpoints = {}
    for index, endpoint in enumerate(ENDPOINTS):
        values = matrix[:, index]
        endpoints[endpoint] = {
            "mean": simultaneous.means[index],
            "se": simultaneous.standard_errors[index],
            "simultaneous_ci": [simultaneous.lower[index], simultaneous.upper[index]],
            "adjusted_p": simultaneous.adjusted_p[index],
            "sesoi": SESOI[endpoint],
            "classification_development_only": interval_classification(
                simultaneous.lower[index], simultaneous.upper[index], sesoi=SESOI[endpoint]
            ),
            "mde80": mde80(values, critical_value=simultaneous.critical_value),
            "sign_fraction_positive": float(np.mean(values > 0)),
            "gene_sd": float(values.std(ddof=1)),
        }
    by_chromosome = {}
    for chromosome in sorted({str(row["chrom"]) for row in per_gene}):
        subset = [row for row in per_gene if row["chrom"] == chromosome]
        by_chromosome[chromosome] = {
            "n_genes": len(subset),
            **{
                f"mean_psi_{endpoint}": float(np.mean([
                    float(row[f"psi_{endpoint}"]) for row in subset
                ]))
                for endpoint in ENDPOINTS
            },
        }

    analysis = {
        "schema": "gfmcd.p3_closure.preflight_analysis.v1",
        "status": "DEVELOPMENT_CALIBRATION_NOT_CONFIRMATORY",
        "input_sha256": sha256_file(args.input),
        "n_genes": len(per_gene),
        "family": {
            "endpoints": list(ENDPOINTS),
            "critical_value": simultaneous.critical_value,
            "n_bootstrap": simultaneous.n_bootstrap,
            "shared_gene_resampling": True,
        },
        "endpoints": endpoints,
        "instrument": source["instrument"],
        "power_gate_candidate": {
            endpoint: endpoints[endpoint]["mde80"] <= SESOI[endpoint]
            for endpoint in ENDPOINTS
        },
        "dominant_mod3_tracking": dominant_class_tracking(source),
        "by_chromosome": by_chromosome,
        "per_gene": list(per_gene),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        "p3-closure-preflight-analysis-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_CALIBRATION_NOT_CONFIRMATORY",
        config={"bootstrap": args.bootstrap, "seed": args.seed, "sesoi": SESOI},
        seeds={"bootstrap": args.seed},
    )
    for path in (
        args.input,
        REPO / "src/gfmcd/splicing/p3_stats.py",
        REPO / "scripts/splicebert/analyze_p3_closure_preflight.py",
    ):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({
        "n_genes": analysis["n_genes"],
        "critical_value": simultaneous.critical_value,
        "endpoints": endpoints,
        "power_gate_candidate": analysis["power_gate_candidate"],
    }, indent=2))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
