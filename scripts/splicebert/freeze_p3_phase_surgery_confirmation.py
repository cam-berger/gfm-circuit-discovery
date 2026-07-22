#!/usr/bin/env python
"""Freeze the outcome-blind package for the P3 phase-surgery confirmation.

This script never imports torch or loads SpliceBERT.  It selects only predeclared roles and stable
orders from already-written annotation cohorts, binds the planned context/control arms, and records
every source/code/data digest needed by the target runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, file_record, git_commit, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL  # noqa: E402


DEFAULT_COMPREHENSIVE_GTF = "/Users/cberg/Research/data/genomes/gencode.v50.annotation.gtf.gz"
DEFAULT_CDS_COHORT = "results/phase-lattice/p3_phase_surgery_untouched_census_20260721.json"
DEFAULT_NON_CDS_COHORT = (
    "results/phase-lattice/p3_phase_surgery_non_cds_untouched_census_20260721.json"
)
DEFAULT_CONTEXT_MATCH = (
    "results/phase-lattice/p3_phase_surgery_context_match_untouched_20260721.json"
)
DEFAULT_PREREG = (
    "studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_P3_CLOSURE.md"
)
DEFAULT_REDTEAM = (
    "studies/splicebert_partner_selection/redteam/REDTEAM_SYNTHESIS_P3_CLOSURE.md"
)

DEVELOPMENT_ARTIFACTS = {
    "development_cds_cohort": (
        "results/phase-lattice/p3_phase_surgery_development_cohort_n120_20260721.json"
    ),
    "development_non_cds_cohort": (
        "results/phase-lattice/p3_phase_surgery_non_cds_development_cohort_20260721.json"
    ),
    "development_context_match": (
        "results/phase-lattice/p3_phase_surgery_context_match_development_v2_20260721.json"
    ),
    "development_preflight": (
        "results/phase-lattice/p3_phase_surgery_preflight_development_n61_20260721.json"
    ),
    "development_preflight_analysis": (
        "results/phase-lattice/p3_phase_surgery_preflight_analysis_development_n61_20260721.json"
    ),
    "development_controls": (
        "results/phase-lattice/p3_phase_surgery_controls_development_n30_20260721.json"
    ),
    "development_controls_analysis": (
        "results/phase-lattice/p3_phase_surgery_controls_analysis_development_n30_v2_20260721.json"
    ),
    "development_health_cds": (
        "results/phase-lattice/p3_phase_surgery_health_cds_development_n61_20260721.json"
    ),
    "development_health_non_cds": (
        "results/phase-lattice/p3_phase_surgery_health_non_cds_development_n49_20260721.json"
    ),
    "development_context_analysis": (
        "results/phase-lattice/p3_phase_surgery_context_difference_development_n33_20260721.json"
    ),
}

CODE_FILES = (
    "src/gfmcd/provenance.py",
    "src/gfmcd/adapters/splicebert.py",
    "src/gfmcd/splicing/circuit.py",
    "src/gfmcd/splicing/p3_annotation.py",
    "src/gfmcd/splicing/p3_closure.py",
    "src/gfmcd/splicing/p3_cohort.py",
    "src/gfmcd/splicing/p3_phase_surgery.py",
    "src/gfmcd/splicing/p3_stats.py",
    "scripts/splicebert/freeze_p3_phase_surgery_confirmation.py",
    "scripts/splicebert/audit_p3_phase_surgery_confirmation_package.py",
    "scripts/splicebert/run_p3_phase_surgery_preflight.py",
    "scripts/splicebert/run_p3_phase_surgery_confirmation.py",
    "scripts/splicebert/analyze_p3_phase_surgery_confirmation.py",
    "tests/test_p3_phase_surgery.py",
    "tests/test_p3_stats.py",
    "tests/test_freeze_p3_phase_surgery_confirmation.py",
    "tests/test_audit_p3_phase_surgery_confirmation_package.py",
    "tests/test_run_p3_phase_surgery_confirmation.py",
    "tests/test_analyze_p3_phase_surgery_confirmation.py",
)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def eligible_records(package: dict, *, role: str | None = None) -> list[dict]:
    rows = [dict(row) for row in package.get("records", ()) if row.get("eligible")]
    if role is not None:
        rows = [row for row in rows if row.get("role") == role]
    return rows


def validate_source_packages(cds: dict, non_cds: dict, match: dict) -> None:
    for package in (cds, non_cds):
        if package.get("schema") != "gfmcd.p3_phase_surgery.cohort.v1":
            raise ValueError("wrong P3 cohort schema")
        if package.get("model_outcomes_read") is not False:
            raise RuntimeError("freeze requires outcome-blind cohorts")
    if match.get("schema") != "gfmcd.p3_phase_surgery.context_match.v1":
        raise ValueError("wrong P3 context-match schema")
    if match.get("model_outcomes_read") is not False:
        raise RuntimeError("freeze requires outcome-blind context matching")


def normalize_record(record: dict, *, target_context: str) -> dict:
    output = dict(record)
    output["target_context"] = target_context
    return output


def build_package(
    *,
    cds: dict,
    non_cds: dict,
    match: dict,
    source_records: dict[str, dict],
    code_records: dict[str, dict],
    data_records: dict[str, dict],
    revision: str | None,
) -> dict:
    validate_source_packages(cds, non_cds, match)
    primary = [
        normalize_record(row, target_context="cds")
        for row in eligible_records(cds, role="confirmation")
    ]
    reserve = [
        normalize_record(row, target_context="cds")
        for row in eligible_records(cds, role="reserve")
    ]
    primary_ids = {row["example_id"] for row in primary}
    cds_by_id = {
        row["example_id"]: normalize_record(row, target_context="cds")
        for row in eligible_records(cds)
    }
    non_by_id = {
        row["example_id"]: normalize_record(row, target_context="non_cds_exon")
        for row in eligible_records(non_cds)
    }
    pairs = list(match.get("pairs", ()))
    context_cds_ids = {pair["cds_example_id"] for pair in pairs}
    context_non_ids = {pair["non_cds_example_id"] for pair in pairs}
    missing_cds = sorted(context_cds_ids - set(cds_by_id))
    missing_non = sorted(context_non_ids - set(non_by_id))
    if missing_cds or missing_non:
        raise RuntimeError(f"context match references absent cohort rows: {missing_cds}, {missing_non}")
    if len(primary) != 943 or len({row["gene_id"] for row in primary}) != 943:
        raise RuntimeError("frozen primary must contain exactly 943 one-record genes")
    if len(reserve) != 457 or len({row["gene_id"] for row in reserve}) != 457:
        raise RuntimeError("frozen reserve census must contain exactly 457 genes")
    if len(pairs) != 52:
        raise RuntimeError("frozen context family must contain exactly 52 pairs")
    control = primary[:150]
    if len(control) != 150:
        raise RuntimeError("frozen control subset must contain exactly 150 primary genes")
    if any(row["example_id"] not in primary_ids for row in control):
        raise AssertionError("control subset escaped primary")
    excluded = {"chr8", "chr9", "chr11", "chr16", "chr19"}
    all_frozen = primary + [cds_by_id[key] for key in sorted(context_cds_ids)] + [
        non_by_id[key] for key in sorted(context_non_ids)
    ]
    if any(str(row["chrom"]) in excluded for row in all_frozen):
        raise RuntimeError("historically exposed chromosome entered frozen package")
    if len({row["example_id"] for row in primary}) != len(primary):
        raise RuntimeError("duplicate primary example")

    return {
        "schema": "gfmcd.p3_phase_surgery.confirmation_package.v1",
        "status": "FROZEN_OUTCOME_BLIND_CONFIRMATION_PACKAGE",
        "model_outcomes_read": False,
        "claim_under_test": (
            "SpliceBERT causally uses its learned period-3 position component in a "
            "coding-context-enriched, phase-aligned manner to support donor-acceptor coupling."
        ),
        "freeze_revision": revision,
        "source_artifacts": source_records,
        "code": code_records,
        "data": data_records,
        "design": {
            "p3_band_1026": list(range(339, 346)),
            "primary_amounts": list(range(2, 11)),
            "control_amounts": [2, 3, 4],
            "directions": ["left", "right"],
            "contexts": ["target", "upstream_intron"],
            "blocks_per_context": 2,
            "prefix_tokens": 1,
            "control_seeds": list(range(20260721, 20260740)),
            "bootstrap_seed": 20260721,
            "bootstrap_draws": 20000,
            "sesoi_coupling_nats": 0.05,
            "thresholds": {
                "reconstruction_max_abs": 1e-6,
                "determinism_max_abs": 1e-8,
                "zero_shift_noop_max_abs": 1e-8,
                "attention_row_abs_error": 1e-4,
                "mechanical_modulo_recovery_min_rms": 0.04,
                "continuous_health_delta_logprob_lower": -0.05,
                "control_rms_mismatch_max": 1e-7,
                "control_l2_ratio_error_max": 5e-6,
                "global_deletion_native_min": 0.05,
                "global_deletion_ratio_max": 0.10,
                "context_min_pairs": 30,
                "context_max_abs_smd": 0.25,
            },
        },
        "cohorts": {
            "primary": primary,
            "reserve_gene_ids_not_runnable": [row["gene_id"] for row in reserve],
            "direction_control_example_ids": [row["example_id"] for row in control],
            "context_cds": [cds_by_id[key] for key in sorted(context_cds_ids)],
            "context_non_cds": [non_by_id[key] for key in sorted(context_non_ids)],
            "context_pairs": pairs,
            "context_balance": match["balance"],
            "context_balance_spec": match["balance_spec"],
        },
        "counts": {
            "primary_genes": len(primary),
            "reserve_genes_not_runnable": len(reserve),
            "direction_control_genes": len(control),
            "context_pairs": len(pairs),
            "context_same_gene_pairs": sum(bool(pair["same_gene"]) for pair in pairs),
            "context_cds_already_primary": len(context_cds_ids & primary_ids),
        },
        "exposure_firewall": {
            "label": "P3-analysis-untouched",
            "excluded_chromosomes": sorted(excluded),
            "stable_namespace": "p3-closure-v1",
            "reserve_forbidden": True,
        },
        "decision_order": [
            "instrument_gates",
            "primary_psi_c",
            "learned_direction_specificity_if_primary_material",
            "matched_non_cds_context_if_primary_material",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cds-cohort", default=DEFAULT_CDS_COHORT)
    parser.add_argument("--non-cds-cohort", default=DEFAULT_NON_CDS_COHORT)
    parser.add_argument("--context-match", default=DEFAULT_CONTEXT_MATCH)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--comprehensive-gtf", default=DEFAULT_COMPREHENSIVE_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prereg", default=DEFAULT_PREREG)
    parser.add_argument("--redteam", default=DEFAULT_REDTEAM)
    parser.add_argument(
        "--out", default="results/phase-lattice/p3_phase_surgery_confirmation_package_v1.json"
    )
    args = parser.parse_args()
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen confirmation package: {out}")

    source_paths = {
        "cds_cohort": args.cds_cohort,
        "non_cds_cohort": args.non_cds_cohort,
        "context_match": args.context_match,
        "preregistration": args.prereg,
        "redteam": args.redteam,
        **DEVELOPMENT_ARTIFACTS,
    }
    missing = [path for path in source_paths.values() if not Path(path).is_file()]
    missing += [path for path in CODE_FILES if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"freeze inputs missing: {missing}")
    match = read_json(args.context_match)
    if match.get("cds_cohort_sha256") != sha256_file(args.cds_cohort):
        raise RuntimeError("context match is not bound to the supplied CDS cohort")
    if match.get("non_cds_cohort_sha256") != sha256_file(args.non_cds_cohort):
        raise RuntimeError("context match is not bound to the supplied non-CDS cohort")
    model_dir = Path(args.model)
    model_files = sorted(path for path in model_dir.iterdir() if path.is_file())
    if not model_files:
        raise FileNotFoundError(f"model checkpoint files missing under {model_dir}")

    package = build_package(
        cds=read_json(args.cds_cohort),
        non_cds=read_json(args.non_cds_cohort),
        match=match,
        source_records={key: file_record(path) for key, path in source_paths.items()},
        code_records={path: file_record(path) for path in CODE_FILES},
        data_records={
            "basic_gtf": file_record(args.gtf),
            "comprehensive_gtf": file_record(args.comprehensive_gtf),
            "fasta": file_record(args.fasta),
            "model_files": [file_record(path) for path in model_files],
        },
        revision=git_commit(REPO),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "p3-phase-surgery-confirmation-freeze-v1",
        sys.argv,
        repo=REPO,
        status="FROZEN_OUTCOME_BLIND_CONFIRMATION_PACKAGE",
        config=package["design"],
        seeds={"bootstrap": 20260721, "first_control": 20260721},
        notes=["No model was loaded and no untouched model outcome was read."],
    )
    for path in list(source_paths.values()) + list(CODE_FILES) + [
        args.gtf, args.comprehensive_gtf, args.fasta,
    ] + [str(path) for path in model_files]:
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps(package["counts"], indent=2, sort_keys=True))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
