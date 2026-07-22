#!/usr/bin/env python
"""Independent, gene-first analysis of the sealed P3 phase-surgery confirmation."""
from __future__ import annotations

import argparse
from collections import defaultdict
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


def verify_arm(output_dir: Path, arm: str, *, package_sha: str, expected: int) -> list[dict]:
    manifest_path = output_dir / f"{arm}_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing complete arm manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "gfmcd.p3_phase_surgery.confirmation_arm_manifest.v1"
        or manifest.get("status") != "COMPLETE"
        or manifest.get("package_sha256") != package_sha
        or int(manifest.get("expected_shards", -1)) != expected
    ):
        raise RuntimeError(f"invalid {arm} arm manifest")
    shards = []
    for name, record in sorted(manifest["shards"].items()):
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"missing or wrong-size shard: {path}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"shard hash mismatch: {path}")
        shard = json.loads(path.read_text())
        if shard.get("package_sha256") != package_sha:
            raise RuntimeError(f"shard belongs to another package: {path}")
        shards.append(shard)
    if len(shards) != expected:
        raise RuntimeError(f"wrong {arm} shard count")
    return shards


def single_endpoint_interval(values: np.ndarray, *, seed: int, bootstrap: int) -> dict:
    values = np.asarray(values, dtype=float)
    result = bootstrap_studentized_max_t(
        values[:, None], names=("coupling",), seed=seed, n_bootstrap=bootstrap
    )
    return {
        "mean": result.means[0],
        "se": result.standard_errors[0],
        "ci": [result.lower[0], result.upper[0]],
        "p_two_sided": result.adjusted_p[0],
        "critical_value": result.critical_value,
        "n_bootstrap": result.n_bootstrap,
        "gene_sd": float(values.std(ddof=1)),
        "mde80": mde80(values, critical_value=result.critical_value),
        "sign_fraction_positive": float(np.mean(values > 0)),
        "classification": interval_classification(
            result.lower[0], result.upper[0], sesoi=0.05
        ),
        "material_positive": bool(result.lower[0] > 0.05),
    }


def add_one_rank_p(target: float, controls: list[float]) -> float:
    if len(controls) != 19 or not np.isfinite([target, *controls]).all():
        raise ValueError("specificity rank requires exactly 19 finite controls")
    return float((1 + sum(value >= target for value in controls)) / 20)


def instrument_gates(package: dict) -> dict[str, bool]:
    source = package["source_artifacts"]
    preflight = json.loads(Path(source["development_preflight"]["path"]).read_text())
    controls = json.loads(Path(source["development_controls"]["path"]).read_text())
    health_cds = json.loads(Path(source["development_health_cds"]["path"]).read_text())
    health_non = json.loads(Path(source["development_health_non_cds"]["path"]).read_text())
    context = json.loads(Path(source["context_match"]["path"]).read_text())
    thresholds = package["design"]["thresholds"]
    instrument = preflight["instrument"]
    geometry = controls["instrument"]["control_geometry"]
    positive = controls["instrument"]["global_p3_deletion_positive"]
    return {
        "carrier_reconstruction": (
            float(instrument["carrier_reconstruction_error"])
            <= thresholds["reconstruction_max_abs"]
        ),
        "determinism": (
            float(instrument["determinism_max_abs"]) <= thresholds["determinism_max_abs"]
        ),
        "zero_shift_noop": (
            float(instrument["zero_shift_noop_max_abs"])
            <= thresholds["zero_shift_noop_max_abs"]
        ),
        "attention_normalized": bool(instrument["attention_live_and_normalized"]),
        "mechanical_modulo_recovery": all(
            float(instrument["mechanical_modulo_profile"][direction][
                "off_minus_inregister_rms"
            ]) >= thresholds["mechanical_modulo_recovery_min_rms"]
            for direction in ("left", "right")
        ),
        "continuous_health_cds": bool(
            health_cds["summary"]["pass_continuous_noninferiority"]
        ),
        "continuous_health_non_cds": bool(
            health_non["summary"]["pass_continuous_noninferiority"]
        ),
        "control_rms_matched": max(
            float(row["max_shift_rms_error"]) for row in geometry
        ) <= thresholds["control_rms_mismatch_max"],
        "control_l2_matched": max(
            abs(float(row["l2_ratio_to_p3"]) - 1.0) for row in geometry
        ) <= thresholds["control_l2_ratio_error_max"],
        "global_deletion_positive": (
            float(positive["native_mean_coupling"])
            > thresholds["global_deletion_native_min"]
            and float(positive["deleted_over_native"])
            <= thresholds["global_deletion_ratio_max"]
        ),
        "context_match": (
            int(context["counts"]["matched_pairs"]) >= thresholds["context_min_pairs"]
            and float(context["balance"]["max_abs_post_match_smd"])
            <= thresholds["context_max_abs_smd"]
        ),
    }


def descriptive_curve(rows: list[dict]) -> dict:
    by_gene = defaultdict(list)
    for row in rows:
        by_gene[(row["gene_id"], row["context"], int(row["amount"]))].append(
            float(row["coupling"])
        )
    output = {}
    contexts = sorted({str(row["context"]) for row in rows})
    for context in contexts:
        output[context] = {}
        for amount in range(2, 11):
            gene_means = [
                float(np.mean(values))
                for (gene, local_context, local_amount), values in by_gene.items()
                if local_context == context and local_amount == amount
            ]
            output[context][str(amount)] = {
                "mean_coupling": float(np.mean(gene_means)),
                "se_across_genes": float(np.std(gene_means, ddof=1) / math.sqrt(len(gene_means))),
                "n_genes": len(gene_means),
                "register": "in" if amount % 3 == 0 else "off",
            }
    return output


def primary_analysis(shards: list[dict], *, seed: int, bootstrap: int) -> tuple[dict, tuple]:
    rows = [row for shard in shards for row in shard["rows"]]
    table = gene_paired_phase_table(rows, metrics=("coupling",), target_context="cds")
    values = np.asarray([float(row["psi_coupling"]) for row in table])
    result = single_endpoint_interval(values, seed=seed, bootstrap=bootstrap)
    by_chrom = {}
    for chrom in sorted({str(row["chrom"]) for row in table}):
        local = [float(row["psi_coupling"]) for row in table if row["chrom"] == chrom]
        by_chrom[chrom] = {"n_genes": len(local), "mean": float(np.mean(local))}
    result.update({
        "n_genes": len(table),
        "per_gene": list(table),
        "by_chromosome": by_chrom,
        "leave_one_gene_mean_range": [
            float(min((values.sum() - value) / (len(values) - 1) for value in values)),
            float(max((values.sum() - value) / (len(values) - 1) for value in values)),
        ],
        "leave_one_chromosome": {
            chrom: float(np.mean([
                float(row["psi_coupling"]) for row in table if row["chrom"] != chrom
            ]))
            for chrom in sorted(by_chrom)
        },
        "figure_curve": descriptive_curve(rows),
    })
    return result, table


def direction_analysis(shards: list[dict], *, seed: int, bootstrap: int) -> dict:
    rows = [row for shard in shards for row in shard["rows"]]
    labels = sorted({str(row["component"]) for row in rows})
    if labels != ["p3", *[f"signed_permutation_{index:02d}" for index in range(19)]]:
        raise RuntimeError("direction-control component family differs from freeze")
    tables = {
        label: gene_paired_phase_table(
            [row for row in rows if row["component"] == label],
            metrics=("coupling",),
            target_context="cds",
            centers=(3,),
        )
        for label in labels
    }
    genes = [row["gene_id"] for row in tables["p3"]]
    if any([row["gene_id"] for row in tables[label]] != genes for label in labels):
        raise RuntimeError("direction-control genes are not identical across components")
    controls = [label for label in labels if label != "p3"]
    means = {
        label: float(np.mean([row["psi_coupling"] for row in tables[label]]))
        for label in labels
    }
    paired = np.asarray([
        float(tables["p3"][index]["psi_coupling"])
        - float(np.mean([tables[label][index]["psi_coupling"] for label in controls]))
        for index in range(len(genes))
    ])
    interval = single_endpoint_interval(paired, seed=seed, bootstrap=bootstrap)
    target = means["p3"]
    control_values = [means[label] for label in controls]
    return {
        "n_genes": len(genes),
        "n_controls": len(controls),
        "p3_mean": target,
        "control_means": {label: means[label] for label in controls},
        "control_range": [float(min(control_values)), float(max(control_values))],
        "p3_exceeds_every_control": bool(target > max(control_values)),
        "add_one_rank_p_one_sided": add_one_rank_p(target, control_values),
        "p3_minus_mean_control": interval,
        "specificity_pass": bool(target > max(control_values) and interval["ci"][0] > 0),
        "per_gene_target_minus_control": [
            {"gene_id": gene, "delta": float(value)} for gene, value in zip(genes, paired)
        ],
    }


def context_analysis(shards: list[dict], *, seed: int, bootstrap: int) -> dict:
    per_pair = []
    for shard in shards:
        pair = shard["pair"]
        rows = shard["rows"]
        values = {}
        for label, example_id, target_context in (
            ("cds", pair["cds_example_id"], "cds"),
            ("non_cds", pair["non_cds_example_id"], "non_cds_exon"),
        ):
            local = [
                {**row, "gene_id": row["example_id"]}
                for row in rows if row["example_id"] == example_id
            ]
            table = gene_paired_phase_table(
                local, metrics=("coupling",), target_context=target_context
            )
            if len(table) != 1:
                raise RuntimeError(f"context pair {pair['pair_id']} has wrong row grid")
            values[label] = float(table[0]["psi_coupling"])
        per_pair.append({
            "pair_id": pair["pair_id"],
            "same_gene": bool(pair["same_gene"]),
            "cds_psi": values["cds"],
            "non_cds_psi": values["non_cds"],
            "delta_context": values["cds"] - values["non_cds"],
        })
    values = np.asarray([row["delta_context"] for row in per_pair])
    interval = single_endpoint_interval(values, seed=seed, bootstrap=bootstrap)
    return {
        "n_pairs": len(per_pair),
        "n_same_gene_pairs": sum(row["same_gene"] for row in per_pair),
        "cds_minus_non_cds": interval,
        "coding_enrichment_pass": bool(interval["ci"][0] > 0),
        "per_pair": per_pair,
    }


def adjudicate(primary: dict, direction: dict, context: dict, gates: dict) -> dict:
    if not all(gates.values()):
        return {
            "code": "INSTRUMENT_LIMITED",
            "licensed_claim": "No biological interpretation; at least one frozen instrument gate failed.",
        }
    if primary["material_positive"]:
        if not direction["specificity_pass"]:
            return {
                "code": "MATERIAL_PHASE_EFFECT_DIRECTION_UNRESOLVED",
                "licensed_claim": (
                    "A same-geometry local position-embedding perturbation materially affects coupling; "
                    "specificity to SpliceBERT's learned P3 hidden direction is unresolved."
                ),
            }
        if context["coding_enrichment_pass"]:
            return {
                "code": "P3_USE_CODING_ENRICHED",
                "licensed_claim": (
                    "SpliceBERT's learned P3 direction causally supports donor-acceptor coupling in a "
                    "coding-context-enriched, phase-aligned manner."
                ),
            }
        return {
            "code": "P3_USE_CONTEXT_UNRESOLVED",
            "licensed_claim": (
                "SpliceBERT's learned P3 direction causally supports phase-aligned coupling in CDS "
                "blocks; coding-context enrichment is unresolved."
            ),
        }
    if primary["classification"] == "bounded_null":
        return {
            "code": "BOUNDED_NULL",
            "licensed_claim": "No primary effect larger than +/-0.05 nat in the frozen domain.",
        }
    if primary["ci"][0] > 0:
        return {
            "code": "DETECTABLE_MATERIALITY_UNRESOLVED",
            "licensed_claim": "P3 phase sensitivity is detectable; materiality remains unresolved.",
        }
    return {
        "code": "INCONCLUSIVE",
        "licensed_claim": "The frozen primary test is inconclusive.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-audit", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.bootstrap != 20000 or args.seed != 20260721:
        parser.error("confirmation requires the frozen bootstrap count and seed")
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite confirmation analysis: {out}")
    package = json.loads(Path(args.package).read_text())
    audit = json.loads(Path(args.package_audit).read_text())
    package_sha = sha256_file(args.package)
    if audit.get("status") != "PASS" or audit.get("package_sha256") != package_sha:
        raise RuntimeError("package audit is not green for this package")
    run_dir = Path(args.run_dir)
    complete = json.loads((run_dir / "run_complete.json").read_text())
    if (
        complete.get("status") != "TARGET_COMPLETE_UNANALYZED"
        or complete.get("package_sha256") != package_sha
        or complete.get("outcome_summary_written") is not False
    ):
        raise RuntimeError("target run is incomplete or was already summarized")
    counts = package["counts"]
    primary_shards = verify_arm(
        run_dir, "primary", package_sha=package_sha, expected=counts["primary_genes"]
    )
    control_shards = verify_arm(
        run_dir,
        "direction_control",
        package_sha=package_sha,
        expected=counts["direction_control_genes"],
    )
    context_shards = verify_arm(
        run_dir,
        "matched_context",
        package_sha=package_sha,
        expected=counts["context_pairs"],
    )
    gates = instrument_gates(package)
    primary, _table = primary_analysis(
        primary_shards, seed=args.seed, bootstrap=args.bootstrap
    )
    direction = direction_analysis(
        control_shards, seed=args.seed + 1, bootstrap=args.bootstrap
    )
    context = context_analysis(
        context_shards, seed=args.seed + 2, bootstrap=args.bootstrap
    )
    direction["opened_for_claim"] = bool(primary["material_positive"])
    context["opened_for_claim"] = bool(
        primary["material_positive"] and direction["specificity_pass"]
    )
    decision = adjudicate(primary, direction, context, gates)
    analysis = {
        "schema": "gfmcd.p3_phase_surgery.confirmation_analysis.v1",
        "status": "CONFIRMATORY_ADJUDICATED",
        "package_sha256": package_sha,
        "package_audit_sha256": sha256_file(args.package_audit),
        "run_complete_sha256": sha256_file(run_dir / "run_complete.json"),
        "instrument_gates": gates,
        "all_instrument_gates_pass": all(gates.values()),
        "primary": primary,
        "direction_specificity": direction,
        "matched_context": context,
        "decision": decision,
        "scope": (
            "Present-day use in the frozen SpliceBERT checkpoint; does not identify training origin, "
            "spliceosomal frame sensing, biological mutation effects, or transfer to another GFM."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "p3-phase-surgery-confirmation-analysis-v1",
        sys.argv,
        repo=REPO,
        status="CONFIRMATORY_ADJUDICATED",
        config={"bootstrap": args.bootstrap, "seed": args.seed, "sesoi": 0.05},
        seeds={"primary": args.seed, "direction": args.seed + 1, "context": args.seed + 2},
    )
    for path in (
        args.package,
        args.package_audit,
        run_dir / "run_complete.json",
        run_dir / "primary_manifest.json",
        run_dir / "direction_control_manifest.json",
        run_dir / "matched_context_manifest.json",
        REPO / "src/gfmcd/splicing/p3_stats.py",
        REPO / "scripts/splicebert/analyze_p3_phase_surgery_confirmation.py",
    ):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({
        "all_instrument_gates_pass": analysis["all_instrument_gates_pass"],
        "primary": {key: primary[key] for key in (
            "n_genes", "mean", "ci", "classification", "material_positive"
        )},
        "direction_specificity_pass": direction["specificity_pass"],
        "coding_enrichment_pass": context["coding_enrichment_pass"],
        "decision": decision,
    }, indent=2))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
