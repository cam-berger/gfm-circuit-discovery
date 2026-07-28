#!/usr/bin/env python
"""Run the gated native-base promoter source-region assay."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_native_execution import (
    score_native_edit_rows,
)
from gfmcd.reconstruction_atlas.promoter_performance import (
    summarize_masked_classification,
)
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    cluster_bootstrap_interval,
    holm_adjust,
    paired_cluster_sign_permutation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--development-report")
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=32)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    preflight = json.loads(Path(args.preflight).read_text())
    supply = json.loads(Path(args.supply_audit).read_text())
    _assert_inputs(config, config_hash, preflight, supply, args)
    development = None
    if args.split == "confirmation":
        if not args.development_report:
            raise SystemExit("--development-report is required for confirmation")
        development = json.loads(Path(args.development_report).read_text())
        if (
            development.get("split") != "development"
            or development.get("config_sha256") != config_hash
            or development.get("preflight_sha256") != sha256_file(args.preflight)
            or not development.get("qualified_on_split")
        ):
            raise SystemExit(
                "development did not qualify unchanged; confirmation remains sealed"
            )

    all_rows = _read_jsonl(args.cohort)
    rows = [row for row in all_rows if row["split"] == args.split]
    expected_n = int(config["cohort"]["pairs_per_split"])
    if (
        len(rows) != expected_n
        or len({str(row["homology_cluster"]) for row in rows}) != expected_n
    ):
        raise RuntimeError("target split does not match the frozen locus count")
    operations = tuple(config["measurement"]["primary_operations"])
    bundle = load_dnabert2_promoter_classifier(
        str(config["checkpoint"]["candidate"]), device=args.device
    )
    scored = score_native_edit_rows(
        bundle,
        rows,
        operations=operations,
        batch_pairs=args.batch_pairs,
        progress=lambda message: print(f"{args.split}: {message}", flush=True),
    )
    metadata = {str(row["example_id"]): row for row in rows}
    raw_rows = [
        {
            **row,
            "split": args.split,
            "biological_annotations": metadata[str(row["pair_id"])]["native_source"][
                "biological_annotations"
            ],
        }
        for row in scored
    ]
    inference = _inference(raw_rows, config)
    clean_behavior = _clean_behavior(raw_rows, config)
    performance = {
        operation: _performance(
            [row for row in raw_rows if row["operation"] == operation],
            config,
        )
        for operation in operations
    }
    biological_strata = {
        operation: _biological_strata(
            [row for row in raw_rows if row["operation"] == operation],
            config,
        )
        for operation in operations
    }
    qualified = clean_behavior["qualified"] and all(
        test["qualified"] for test in inference["tests"].values()
    )
    status = "DEVELOPMENT" if args.split == "development" else "SEALED"
    version = int(config["schema_version"])
    experiment = f"dnabert2_promoter_native_source_{args.split}_v{version}"
    input_paths = tuple(
        path
        for path in (
            args.cohort,
            args.supply_audit,
            args.checkpoint_audit,
            args.config,
            args.preflight,
            args.development_report,
        )
        if path is not None
    )
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id=f"{experiment}_raw",
        status=status,
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "measurement": config["measurement"],
        },
        seeds={"inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_native_source_v10",
        "split": args.split,
        "status": status,
        "config_sha256": config_hash,
        "preflight_sha256": sha256_file(args.preflight),
        "checkpoint_revision": bundle.identity()["revision"],
        "n_pairs": len(rows),
        "n_clusters": len(rows),
        "measurement": config["measurement"],
        "inference": inference,
        "clean_behavior": clean_behavior,
        "secondary_classification_performance": performance,
        "descriptive_biological_strata": biological_strata,
        "qualified_on_split": qualified,
        "confirmation_gate": (
            "opened_from_qualified_development"
            if args.split == "confirmation"
            else (
                "confirmation_may_open" if qualified else "confirmation_remains_sealed"
            )
        ),
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "rows": len(raw_rows),
        },
        "claim_boundary": config["scope"],
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=experiment,
        status=status,
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "gates": config["native_primary_gates"],
        },
        seeds={"inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _assert_inputs(
    config: dict[str, Any],
    config_hash: str,
    preflight: dict[str, Any],
    supply: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    operations = set(config["measurement"]["primary_operations"])
    if (
        preflight.get("status") != "QUALIFIED"
        or not preflight.get("qualified")
        or not all(preflight.get("checks", {}).values())
        or preflight.get("config_sha256") != config_hash
        or preflight.get("checkpoint_revision") != config["checkpoint"]["revision"]
        or supply.get("status") != "QUALIFIED"
        or not all(supply.get("checks", {}).values())
        or set(supply["native_edit_geometry"]) != operations
    ):
        raise RuntimeError("matching qualified frozen inputs are required")
    for split in ("development", "confirmation"):
        ids = supply["selection"]["selected_example_ids"][split]
        digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
        if digest != config["cohort"]["selected_id_sha256"][split]:
            raise RuntimeError("frozen cohort identity does not match supply")
    if sha256_file(args.config) != preflight["input_sha256"]["config"]:
        raise RuntimeError("preflight raw config file digest does not match")
    if sha256_file(args.supply_audit) != preflight["input_sha256"]["supply_audit"]:
        raise RuntimeError("preflight supply digest does not match")


def _inference(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_operation[str(row["operation"])].append(row)
    tests: dict[str, Any] = {}
    raw_p = {}
    gates = config["native_primary_gates"]
    seed = int(config["inference"]["seed"])
    for index, (operation, operation_rows) in enumerate(sorted(by_operation.items())):
        values = [float(row["target_specific_loss"]) for row in operation_rows]
        clusters = [str(row["homology_cluster"]) for row in operation_rows]
        observed, null = paired_cluster_sign_permutation(
            values,
            clusters,
            permutations=int(config["inference"]["sign_flip_draws"]),
            seed=seed ^ ((index + 1) * 0x1111),
        )
        pvalue = add_one_permutation_pvalue(observed, null, alternative="greater")
        interval = cluster_bootstrap_interval(
            values,
            clusters,
            draws=int(config["inference"]["bootstrap_draws"]),
            seed=seed ^ ((index + 1) * 0x2222),
        )
        raw_p[operation] = pvalue
        tests[operation] = {
            "n_pairs": len(values),
            "n_clusters": len(set(clusters)),
            "mean_target_specific_loss": observed,
            "cluster_bootstrap_CI95": list(interval),
            "p_add_one_cluster_sign": pvalue,
        }
    adjusted = holm_adjust(raw_p)
    for operation, pvalue in adjusted.items():
        test = tests[operation]
        checks = {
            "mean_at_least_SESOI": test["mean_target_specific_loss"]
            >= float(gates["SESOI_logit_margin"]),
            "positive_lower_CI": test["cluster_bootstrap_CI95"][0] > 0,
            "Holm_p_at_most_alpha": pvalue <= float(gates["familywise_alpha"]),
        }
        test.update(
            {
                "SESOI_logit_margin": float(gates["SESOI_logit_margin"]),
                "p_holm": pvalue,
                "checks": checks,
                "qualified": all(checks.values()),
            }
        )
    return {
        "tests": tests,
        "multiplicity": gates["Holm_family"],
        "unit": config["inference"]["unit"],
    }


def _clean_behavior(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    by_pair = {str(row["pair_id"]): row for row in rows}
    values = [
        float(row["target_clean_margin"]) - float(row["control_clean_margin"])
        for row in by_pair.values()
    ]
    clusters = [str(row["homology_cluster"]) for row in by_pair.values()]
    seed = int(config["inference"]["seed"]) ^ 0x3333
    observed, null = paired_cluster_sign_permutation(
        values,
        clusters,
        permutations=int(config["inference"]["sign_flip_draws"]),
        seed=seed,
    )
    pvalue = add_one_permutation_pvalue(observed, null, alternative="greater")
    interval = cluster_bootstrap_interval(
        values,
        clusters,
        draws=int(config["inference"]["bootstrap_draws"]),
        seed=seed ^ 0x4444,
    )
    sesoi = float(config["native_primary_gates"]["clean_target_control_margin_SESOI"])
    checks = {
        "mean_at_least_SESOI": observed >= sesoi,
        "positive_lower_CI": interval[0] > 0,
        "p_add_one_at_most_alpha": pvalue
        <= float(config["native_primary_gates"]["familywise_alpha"]),
    }
    return {
        "mean_clean_target_minus_control_margin": observed,
        "cluster_bootstrap_CI95": list(interval),
        "p_add_one_cluster_sign": pvalue,
        "SESOI_logit_margin": sesoi,
        "checks": checks,
        "qualified": all(checks.values()),
    }


def _performance(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    mapped = [
        {
            **row,
            "target_masked_margin": row["target_edited_margin"],
            "control_masked_margin": row["control_edited_margin"],
        }
        for row in rows
    ]
    summary = summarize_masked_classification(
        mapped,
        bootstrap_draws=int(config["inference"]["bootstrap_draws"]),
        seed=int(config["inference"]["seed"]) ^ 0x5555,
    )
    summary["edited"] = summary.pop("masked")
    summary["bootstrap"]["edited_ci95"] = summary["bootstrap"].pop("masked_ci95")
    summary["semantics"] = "masked fields were mapped to native-edited margins"
    return summary


def _biological_strata(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    annotations = (
        "CAGE_supported_TSS",
        "PLS_cCRE_at_source_region",
        "CpG_island_at_source_region",
        "DCE_II_CTGT_within_plus16_plus21",
    )
    output = {}
    for index, annotation in enumerate(annotations):
        selected = [
            row for row in rows if bool(row["biological_annotations"][annotation])
        ]
        values = [float(row["target_specific_loss"]) for row in selected]
        if len(values) < 5:
            output[annotation] = {
                "n": len(values),
                "mean": statistics.mean(values) if values else None,
                "CI95": None,
                "status": "descriptive_underpowered",
            }
            continue
        clusters = [str(row["homology_cluster"]) for row in selected]
        interval = cluster_bootstrap_interval(
            values,
            clusters,
            draws=int(config["inference"]["bootstrap_draws"]),
            seed=int(config["inference"]["seed"]) ^ ((index + 1) * 0x6666),
        )
        output[annotation] = {
            "n": len(values),
            "mean": statistics.mean(values),
            "CI95": list(interval),
            "status": "descriptive_not_multiplicity_qualified",
        }
    return output


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
