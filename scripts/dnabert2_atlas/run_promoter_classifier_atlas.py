#!/usr/bin/env python
"""Run the gated GENCODE promoter classifier margin/context assay."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    PROMOTER_CHECKPOINTS,
    Dnabert2PromoterClassifier,
    load_dnabert2_promoter_classifier,
)
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import (
    frozen_config_sha256,
    load_preflight_report,
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
    parser.add_argument("--cohort-audit", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--development-report")
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    seed = int(config["seed"])
    checkpoint_name = str(config["checkpoint"]["candidate"])
    checkpoint = PROMOTER_CHECKPOINTS[checkpoint_name]
    preflight = load_preflight_report(args.preflight_report)
    expected_inputs = {
        "gue_test": sha256_file(args.gue_test),
        "cohort": sha256_file(args.cohort),
        "cohort_audit": sha256_file(args.cohort_audit),
        "checkpoint_audit": sha256_file(args.checkpoint_audit),
    }
    preflight.assert_qualified(
        expected_config_sha256=config_hash,
        expected_checkpoint_revision=checkpoint.revision,
        expected_input_sha256=expected_inputs,
    )
    development = None
    if args.split == "confirmation":
        if not args.development_report:
            raise SystemExit(
                "--development-report is required before confirmation can open"
            )
        development = json.loads(Path(args.development_report).read_text())
        if (
            development.get("split") != "development"
            or development.get("preflight_report_id") != preflight.report_id
            or development.get("config_sha256") != config_hash
        ):
            raise RuntimeError(
                "development result does not bind the frozen promoter assay"
            )
        if not development.get("qualified_on_split"):
            raise SystemExit(
                "promoter development did not qualify; confirmation remains sealed"
            )
    records = [row for row in _read_jsonl(args.cohort) if row["split"] == args.split]
    minimum = int(config["behavior"]["minimum_clusters_per_split"])
    if len({row["homology_cluster"] for row in records}) < minimum:
        raise RuntimeError("promoter split is below the frozen minimum cluster count")
    requests, metadata = _requests(records)
    bundle = load_dnabert2_promoter_classifier(checkpoint_name, device=args.device)
    scorer = Dnabert2PromoterClassifier(bundle)
    scores = scorer.score(requests, batch_size=args.batch_size)
    by_id = {score.request_id: score for score in scores}
    effect_rows = _effects(records, by_id)
    inference = _inference(
        effect_rows,
        permutations=int(config["behavior"]["permutation_draws"]),
        bootstrap_draws=args.bootstrap_draws,
        seed=seed,
    )
    decisions = _decisions(inference, config)
    qualified = all(decision["qualified"] for decision in decisions.values())
    raw_rows = [{**score.to_dict(), **metadata[score.request_id]} for score in scores]
    status = "DEVELOPMENT" if args.split == "development" else "SEALED"
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id=f"dnabert2_promoter_classifier_{args.split}_raw_v1",
        status=status,
        model_identity=scorer.identity(),
        input_paths=tuple(
            path
            for path in (
                args.cohort,
                args.cohort_audit,
                args.checkpoint_audit,
                args.gue_test,
                args.config,
                args.preflight_report,
                args.development_report,
            )
            if path is not None
        ),
        parameters={
            "split": args.split,
            "measurement": config["measurement"],
            "n_examples": len(effect_rows),
            "n_scored_sequences": len(scores),
        },
        seeds={"inference": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "fine_tuned_promoter_margin_target_preserving_context_v1",
        "split": args.split,
        "status": status,
        "preflight_report_id": preflight.report_id,
        "preflight_decision": preflight.decision,
        "config_sha256": config_hash,
        "checkpoint_candidate": checkpoint_name,
        "n_examples": len(effect_rows),
        "n_clusters": len({row["homology_cluster"] for row in effect_rows}),
        "measurement": config["measurement"],
        "effect_definition": {
            "clean_enrichment": "target_clean_margin-control_clean_margin",
            "target_context_loss": "target_clean_margin-target_context_shuffle_margin",
            "control_context_loss": "control_clean_margin-control_context_shuffle_margin",
            "primary_context_difference": "target_context_loss-control_context_loss",
        },
        "summary": {
            "mean_target_clean_margin": statistics.mean(
                row["target_clean_margin"] for row in effect_rows
            ),
            "mean_control_clean_margin": statistics.mean(
                row["control_clean_margin"] for row in effect_rows
            ),
            "mean_target_context_loss": statistics.mean(
                row["target_context_loss"] for row in effect_rows
            ),
            "mean_control_context_loss": statistics.mean(
                row["control_context_loss"] for row in effect_rows
            ),
        },
        "inference": inference,
        "decisions": decisions,
        "qualified_on_split": qualified,
        "confirmation_gate": (
            "opened_from_qualified_development"
            if args.split == "confirmation"
            else (
                "confirmation_may_open" if qualified else "confirmation_remains_sealed"
            )
        ),
        "raw_scores": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(raw_rows),
        },
        "mechanistic_boundary": config["mechanistic_boundary"],
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=f"dnabert2_promoter_classifier_atlas_{args.split}_v1",
        status=status,
        model_identity=scorer.identity(),
        input_paths=tuple(
            path
            for path in (
                args.cohort,
                args.cohort_audit,
                args.checkpoint_audit,
                args.gue_test,
                args.config,
                args.preflight_report,
                args.development_report,
                raw_path,
                raw_manifest,
            )
            if path is not None
        ),
        parameters={
            "split": args.split,
            "bootstrap_draws": args.bootstrap_draws,
            "permutation_draws": int(config["behavior"]["permutation_draws"]),
            "multiple_testing": config["behavior"]["multiple_testing"],
            "measurement": config["measurement"],
        },
        seeds={"inference": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "n_examples": len(effect_rows),
                "qualified": qualified,
                "effects": {
                    name: {
                        "mean": test["observed_cluster_mean"],
                        "CI": test["cluster_bootstrap_CI"],
                        "p_holm": test["p_holm"],
                    }
                    for name, test in inference["tests"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def _requests(
    records: list[dict[str, Any]],
) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]]]:
    requests = []
    metadata = {}
    for row in records:
        variants = {
            "target_clean": row["target"]["clean_sequence"],
            "target_shuffle": row["target"]["context_shuffle_sequence"],
            "control_clean": row["control"]["clean_sequence"],
            "control_shuffle": row["control"]["context_shuffle_sequence"],
        }
        for variant, sequence in variants.items():
            request_id = f"{row['example_id']}|{variant}"
            requests.append((request_id, sequence))
            metadata[request_id] = {
                "example_id": row["example_id"],
                "homology_cluster": row["homology_cluster"],
                "gene_id": row["target"]["gene_id"],
                "transcript_id": row["target"]["transcript_id"],
                "variant": variant,
                "changed_nucleotides": (
                    0
                    if variant.endswith("clean")
                    else row["intervention"]["target_control_changed_nucleotides"]
                ),
            }
    return requests, metadata


def _effects(
    records: list[dict[str, Any]], scores: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for row in records:
        prefix = row["example_id"]
        margins = {
            variant: scores[f"{prefix}|{variant}"].margin
            for variant in (
                "target_clean",
                "target_shuffle",
                "control_clean",
                "control_shuffle",
            )
        }
        target_loss = margins["target_clean"] - margins["target_shuffle"]
        control_loss = margins["control_clean"] - margins["control_shuffle"]
        output.append(
            {
                "example_id": prefix,
                "homology_cluster": row["homology_cluster"],
                "gene_id": row["target"]["gene_id"],
                **{f"{key}_margin": value for key, value in margins.items()},
                "clean_enrichment": (
                    margins["target_clean"] - margins["control_clean"]
                ),
                "target_context_loss": target_loss,
                "control_context_loss": control_loss,
                "context_difference": target_loss - control_loss,
            }
        )
    return output


def _inference(
    rows: list[dict[str, Any]],
    *,
    permutations: int,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    clusters = [str(row["homology_cluster"]) for row in rows]
    tests: dict[str, Any] = {}
    raw_pvalues = {}
    for index, endpoint in enumerate(("clean_enrichment", "context_difference")):
        values = [float(row[endpoint]) for row in rows]
        observed, null = paired_cluster_sign_permutation(
            values,
            clusters,
            permutations=permutations,
            seed=seed ^ (index + 1) * 0x1111,
        )
        pvalue = add_one_permutation_pvalue(observed, null, alternative="greater")
        interval = cluster_bootstrap_interval(
            values,
            clusters,
            draws=bootstrap_draws,
            seed=seed ^ (index + 1) * 0x2222,
        )
        raw_pvalues[endpoint] = pvalue
        tests[endpoint] = {
            "n_rows": len(values),
            "n_clusters": len(set(clusters)),
            "observed_cluster_mean": observed,
            "cluster_bootstrap_CI": list(interval),
            "p_add_one": pvalue,
        }
    adjusted = holm_adjust(raw_pvalues)
    for endpoint, pvalue in adjusted.items():
        tests[endpoint]["p_holm"] = pvalue
    return {
        "tests": tests,
        "permutations": permutations,
        "bootstrap_draws": bootstrap_draws,
        "multiplicity": "Holm_across_two_preregistered_effects",
    }


def _decisions(inference: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    behavior = config["behavior"]
    sesoi = {
        "clean_enrichment": float(behavior["clean_enrichment_SESOI_logit_margin"]),
        "context_difference": float(behavior["context_difference_SESOI_logit_margin"]),
    }
    output = {}
    for endpoint, test in inference["tests"].items():
        checks = {
            "minimum_clusters": test["n_clusters"]
            >= int(behavior["minimum_clusters_per_split"]),
            "mean_at_least_SESOI": test["observed_cluster_mean"] >= sesoi[endpoint],
            "positive_lower_CI": test["cluster_bootstrap_CI"][0] > 0,
            "Holm_p_at_most_alpha": test["p_holm"]
            <= float(behavior["familywise_alpha"]),
        }
        output[endpoint] = {
            "SESOI_logit_margin": sesoi[endpoint],
            "checks": checks,
            "qualified": all(checks.values()),
        }
    return output


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
