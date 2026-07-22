#!/usr/bin/env python
"""Gene-first analysis of geometry-matched P3 phase-surgery controls."""
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


def add_one_rank_p(target: float, controls: list[float]) -> float:
    """One-sided exact add-one rank p for the preregistered positive direction."""

    if len(controls) < 19 or not np.isfinite([target, *controls]).all():
        raise ValueError("rank test requires at least 19 finite controls")
    return float((1 + sum(value >= target for value in controls)) / (len(controls) + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--out", default="results/phase-lattice/p3_phase_surgery_controls_analysis.json"
    )
    args = parser.parse_args()
    if args.bootstrap < 99:
        parser.error("--bootstrap must be at least 99")
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite control analysis: {out}")
    source = json.loads(Path(args.input).read_text())
    if source.get("schema") != "gfmcd.p3_phase_surgery.controls.v1":
        raise ValueError("wrong P3 phase-surgery controls schema")
    if source.get("status") != "DEVELOPMENT_OUTCOME_AWARE":
        raise RuntimeError("phase-surgery control analysis is development-only")

    labels = sorted({str(row["component"]) for row in source["rows"]})
    if "p3" not in labels or len(labels) < 20:
        raise RuntimeError("target plus at least 19 controls are required")
    controls = [label for label in labels if label != "p3"]
    target_context = str(source.get("target_context", "cds"))
    tables = {
        label: gene_paired_phase_table(
            [row for row in source["rows"] if row["component"] == label],
            metrics=ENDPOINTS,
            target_context=target_context,
            centers=(3,),
        )
        for label in labels
    }
    genes = tuple(str(row["gene_id"]) for row in tables["p3"])
    if any(tuple(str(row["gene_id"]) for row in tables[label]) != genes for label in labels):
        raise RuntimeError("component families do not contain identical ordered genes")

    by_component = {}
    for label in labels:
        by_component[label] = {
            endpoint: float(np.mean([
                float(row[f"psi_{endpoint}"]) for row in tables[label]
            ]))
            for endpoint in ENDPOINTS
        }
    target_minus_control = np.asarray([
        [
            float(tables["p3"][gene_index][f"psi_{endpoint}"])
            - float(np.mean([
                tables[label][gene_index][f"psi_{endpoint}"] for label in controls
            ]))
            for endpoint in ENDPOINTS
        ]
        for gene_index in range(len(genes))
    ])
    simultaneous = bootstrap_studentized_max_t(
        target_minus_control,
        names=ENDPOINTS,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
    )
    endpoints = {}
    for index, endpoint in enumerate(ENDPOINTS):
        target = by_component["p3"][endpoint]
        control_values = [by_component[label][endpoint] for label in controls]
        values = target_minus_control[:, index]
        endpoints[endpoint] = {
            "p3_mean_psi": target,
            "control_mean_psi": float(np.mean(control_values)),
            "control_range_psi": [float(np.min(control_values)), float(np.max(control_values))],
            "p3_add_one_rank_p_one_sided": add_one_rank_p(target, control_values),
            "p3_exceeds_every_control": bool(target > max(control_values)),
            "p3_minus_mean_control": simultaneous.means[index],
            "simultaneous_ci": [simultaneous.lower[index], simultaneous.upper[index]],
            "adjusted_p": simultaneous.adjusted_p[index],
            "sesoi": SESOI[endpoint],
            "classification_development_only": interval_classification(
                simultaneous.lower[index], simultaneous.upper[index], sesoi=SESOI[endpoint]
            ),
            "mde80": mde80(values, critical_value=simultaneous.critical_value),
            "gene_sd": float(values.std(ddof=1)),
        }
    geometry = source["instrument"]["control_geometry"]
    deterministic = source["instrument"]["determinism"]
    candidate_gates = {
        "global_p3_deletion_positive": bool(
            source["instrument"]["global_p3_deletion_positive"]["pass"]
        ),
        "control_geometry_exact": max(
            float(row["max_shift_rms_error"]) for row in geometry
        ) <= 1e-7,
        "component_l2_exact": max(
            abs(float(row["l2_ratio_to_p3"]) - 1.0) for row in geometry
        ) <= 5e-6,
        "deterministic": max(float(row["max_abs"]) for row in deterministic) <= 1e-8,
        "coupling_p3_exceeds_all_controls": endpoints["coupling"][
            "p3_exceeds_every_control"
        ],
        "coupling_paired_ci_excludes_zero": endpoints["coupling"][
            "simultaneous_ci"
        ][0] > 0,
    }
    analysis = {
        "schema": "gfmcd.p3_phase_surgery.controls_analysis.v1",
        "status": "DEVELOPMENT_CALIBRATION_NOT_CONFIRMATORY",
        "input_sha256": sha256_file(args.input),
        "n_genes": len(genes),
        "n_controls": len(controls),
        "target_context": target_context,
        "family": {
            "endpoints": list(ENDPOINTS),
            "critical_value": simultaneous.critical_value,
            "n_bootstrap": simultaneous.n_bootstrap,
            "shared_gene_resampling": True,
        },
        "endpoints": endpoints,
        "candidate_gates": candidate_gates,
        "by_component": by_component,
        "per_gene_target_minus_mean_control": [
            {
                "gene_id": gene,
                **{
                    f"delta_{endpoint}": float(target_minus_control[index, endpoint_index])
                    for endpoint_index, endpoint in enumerate(ENDPOINTS)
                },
            }
            for index, gene in enumerate(genes)
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "p3-phase-surgery-controls-analysis-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_CALIBRATION_NOT_CONFIRMATORY",
        config={"bootstrap": args.bootstrap, "seed": args.seed, "sesoi": SESOI},
        seeds={"bootstrap": args.seed},
    )
    for path in (
        args.input,
        REPO / "src/gfmcd/splicing/p3_stats.py",
        REPO / "scripts/splicebert/analyze_p3_phase_surgery_controls.py",
    ):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({
        "n_genes": len(genes),
        "n_controls": len(controls),
        "endpoints": endpoints,
        "candidate_gates": candidate_gates,
    }, indent=2))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
