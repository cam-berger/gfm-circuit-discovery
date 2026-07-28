#!/usr/bin/env python
"""Combine immutable development and confirmation context-margin results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gfmcd.reconstruction_atlas.artifacts import write_result_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    development = json.loads(Path(args.development).read_text())
    confirmation = json.loads(Path(args.confirmation).read_text())
    config = json.loads(Path(args.config).read_text())
    if development["split"] != "development":
        raise SystemExit("development artifact carries the wrong split")
    if confirmation["split"] != "confirmation":
        raise SystemExit("confirmation artifact carries the wrong split")
    for key in ("preflight_report_id", "config_sha256"):
        if development[key] != confirmation[key]:
            raise SystemExit(f"split artifacts disagree on {key}")

    rows = {}
    admitted = [str(x) for x in config["behavior"]["admitted_site_classes"]]
    quarantined = [str(x) for x in config["behavior"]["quarantined_site_classes"]]
    for site_class in admitted:
        dev_decision = development["behavior_decisions"].get(site_class)
        dev_test = development["inference"]["primary_tests"].get(site_class)
        conf_decision = confirmation["behavior_decisions"].get(site_class)
        conf_test = confirmation["inference"]["primary_tests"].get(site_class)
        if dev_decision is None or dev_test is None:
            status = "not_development_scored"
        elif not dev_decision["qualified_on_split"]:
            status = "development_not_qualified"
        elif conf_decision is None or conf_test is None:
            status = "confirmation_not_opened"
        elif conf_decision["qualified_on_split"]:
            status = "replicated_context_effect"
        else:
            status = "development_only_not_replicated"
        rows[site_class] = {
            "status": status,
            "development": _test_summary(dev_test),
            "confirmation": _test_summary(conf_test),
        }
    for site_class in quarantined:
        rows[site_class] = {
            "status": "preflight_instrument_unresolved",
            "development": None,
            "confirmation": None,
        }
    replicated = sorted(
        site_class
        for site_class, row in rows.items()
        if row["status"] == "replicated_context_effect"
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "geometry_qualified_margin_class_specific_context",
        "status": "SEALED",
        "preflight_report_id": development["preflight_report_id"],
        "config_sha256": development["config_sha256"],
        "primary_effect": "margin(irrelevant_distal_equal_dose_shuffle)-margin(class_specific_proximal_shuffle)",
        "primary_sesoi_nats_per_covered_nt": config["behavior"][
            "primary_sesoi_nats_per_covered_nt"
        ],
        "replicated_site_classes": replicated,
        "n_replicated_site_classes": len(replicated),
        "class_results": dict(sorted(rows.items())),
        "licensed_claim": "DNABERT-2 alternative margins for the replicated classes depend on the preregistered proximal context more than on an equal-width, equal-composition, equal-change-count distal context corruption.",
        "boundary": [
            "This is a model-behavior context effect, not biological function in cells.",
            "The distal null controls mutation dose and composition but not distance from the target; proximity-sensitive encoding remains part of the licensed effect.",
            "TF and enhancer targets are derived peak/cCRE anchors, not resolved motifs or causal bases.",
            "No head, MLP, residual direction, mediation path, or causal circuit has been tested.",
        ],
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_context_margin_atlas_replication",
        status="SEALED",
        model_identity={
            "hf_id": config["checkpoint"]["hf_id"],
            "revision": config["checkpoint"]["revision"],
        },
        input_paths=(
            args.development,
            args.confirmation,
            args.config,
            args.preflight_report,
        ),
        parameters={
            "combination": "development_selection_then_confirmation_replication",
            "no_new_model_scores": True,
        },
        seeds={"none": 0},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _test_summary(test: dict[str, Any] | None) -> dict[str, Any] | None:
    if test is None:
        return None
    return {
        "n_clusters": int(test["n_clusters"]),
        "effect_mean": float(test["observed_cluster_mean"]),
        "cluster_bootstrap_ci": [
            float(value) for value in test["cluster_bootstrap_ci"]
        ],
        "p_add_one": float(test["p_add_one"]),
        "p_holm": float(test["p_holm"]),
    }


if __name__ == "__main__":
    main()
