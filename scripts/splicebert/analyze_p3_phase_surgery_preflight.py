#!/usr/bin/env python
"""Independent development analysis for nucleotide-invariant local P3 phase surgery."""
from __future__ import annotations

import argparse
import json
import math
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


def carrier_class_tracking(result: dict) -> dict[str, object]:
    """Describe whether downstream attention's dominant modulo-3 class moves with surgery.

    This is deliberately descriptive: changing a local position-key component does not algebraically
    force the full L3H1 attention row to rotate, and no class-tracking value enters the primary test.
    """

    native = {
        row["example_id"]: int(np.argmax(row["phasor_class_masses"]))
        for row in result["native_rows"]
    }
    grouped: dict[str, object] = {}
    target_context = str(result.get("target_context", "cds"))
    for context in (target_context, "upstream_intron"):
        grouped[context] = {}
        for direction in ("left", "right"):
            grouped[context][direction] = {}
            # C[x] <- C[x+r] (left) makes x carry the phase of x+r; right uses x-r.
            sign = 1 if direction == "left" else -1
            for amount in range(2, 11):
                rows = [
                    row for row in result["rows"]
                    if row["context"] == context and row["direction"] == direction
                    and int(row["amount"]) == amount
                ]
                observed = np.asarray([
                    int(np.argmax(row["phasor_class_masses"])) for row in rows
                ])
                baseline = np.asarray([native[row["example_id"]] for row in rows])
                expected = (baseline + sign * amount) % 3
                grouped[context][direction][str(amount)] = {
                    "n": len(rows),
                    "dominant_class_counts": np.bincount(observed, minlength=3).tolist(),
                    "same_as_native_fraction": float(np.mean(observed == baseline)),
                    "carrier_shift_expected_fraction": float(np.mean(observed == expected)),
                    "expected_modulo_class_delta": int((sign * amount) % 3),
                    "note": "descriptive only; not part of the confirmatory family",
                }
    return grouped


def phase_strata(per_gene: tuple[dict[str, object], ...], source: dict) -> dict[str, object]:
    phase_by_gene = {
        str(row["gene_id"]): int(row["donor_phase"])
        for row in source["native_rows"]
        if row.get("donor_phase") is not None
    }
    output = {}
    for phase in (0, 1, 2):
        rows = [row for row in per_gene if phase_by_gene.get(str(row["gene_id"])) == phase]
        output[str(phase)] = {
            "n_genes": len(rows),
            **{
                f"mean_psi_{endpoint}": (
                    float(np.mean([float(row[f"psi_{endpoint}"]) for row in rows]))
                    if rows else None
                )
                for endpoint in ENDPOINTS
            },
            "status": "exploratory_development_only",
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--out", default="results/phase-lattice/p3_phase_surgery_preflight_analysis.json"
    )
    args = parser.parse_args()
    if args.bootstrap < 99:
        parser.error("--bootstrap must be at least 99")
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite phase-surgery analysis: {out}")
    source = json.loads(Path(args.input).read_text())
    if source.get("schema") != "gfmcd.p3_phase_surgery.preflight.v1":
        raise ValueError("wrong P3 phase-surgery preflight schema")
    if source.get("status") != "DEVELOPMENT_OUTCOME_AWARE":
        raise RuntimeError("phase-surgery preflight analysis is development-only")

    target_context = str(source.get("target_context", "cds"))
    per_gene = gene_paired_phase_table(
        source["rows"], metrics=ENDPOINTS, target_context=target_context
    )
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
        rows = [row for row in per_gene if row["chrom"] == chromosome]
        by_chromosome[chromosome] = {
            "n_genes": len(rows),
            **{
                f"mean_psi_{endpoint}": float(np.mean([
                    float(row[f"psi_{endpoint}"]) for row in rows
                ]))
                for endpoint in ENDPOINTS
            },
        }

    instrument = source["instrument"]
    instrument_candidate_gates = {
        "carrier_reconstruction": float(instrument["carrier_reconstruction_error"]) <= 1e-6,
        "deterministic": float(instrument["determinism_max_abs"]) <= 1e-8,
        "zero_shift_exact_noop": float(instrument["zero_shift_noop_max_abs"]) <= 1e-8,
        "attention_live_and_normalized": bool(instrument["attention_live_and_normalized"]),
        "mlm_health_ratio_at_least_0_8": (
            math.isfinite(float(instrument["health_ratio"]))
            and float(instrument["health_ratio"]) >= 0.8
        ),
        "mechanical_modulo_recovery": all(
            float(instrument["mechanical_modulo_profile"][direction][
                "off_minus_inregister_rms"
            ]) > 0
            for direction in ("left", "right")
        ),
    }
    analysis = {
        "schema": "gfmcd.p3_phase_surgery.preflight_analysis.v1",
        "status": "DEVELOPMENT_CALIBRATION_NOT_CONFIRMATORY",
        "input_sha256": sha256_file(args.input),
        "n_genes": len(per_gene),
        "target_context": target_context,
        "family": {
            "endpoints": list(ENDPOINTS),
            "critical_value": simultaneous.critical_value,
            "n_bootstrap": simultaneous.n_bootstrap,
            "shared_gene_resampling": True,
        },
        "endpoints": endpoints,
        "instrument": instrument,
        "instrument_candidate_gates": instrument_candidate_gates,
        "power_gate_candidate": {
            endpoint: endpoints[endpoint]["mde80"] <= SESOI[endpoint]
            for endpoint in ENDPOINTS
        },
        "carrier_mod3_tracking": carrier_class_tracking(source),
        "donor_phase_strata": phase_strata(per_gene, source),
        "by_chromosome": by_chromosome,
        "per_gene": list(per_gene),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        "p3-phase-surgery-preflight-analysis-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_CALIBRATION_NOT_CONFIRMATORY",
        config={"bootstrap": args.bootstrap, "seed": args.seed, "sesoi": SESOI},
        seeds={"bootstrap": args.seed},
    )
    for path in (
        args.input,
        REPO / "src/gfmcd/splicing/p3_stats.py",
        REPO / "scripts/splicebert/analyze_p3_phase_surgery_preflight.py",
    ):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({
        "n_genes": analysis["n_genes"],
        "critical_value": simultaneous.critical_value,
        "endpoints": endpoints,
        "instrument_candidate_gates": instrument_candidate_gates,
        "power_gate_candidate": analysis["power_gate_candidate"],
    }, indent=2))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
