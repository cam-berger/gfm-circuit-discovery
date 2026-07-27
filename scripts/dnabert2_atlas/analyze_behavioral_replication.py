#!/usr/bin/env python
"""Combine frozen development and sealed confirmation atlas behavior screens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gfmcd.adapters.dnabert2 import DNABERT2_HF_ID, DNABERT2_REVISION
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.gates import (
    frozen_config_sha256,
    load_preflight_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--preflight-config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    development = json.loads(Path(args.development).read_text())
    confirmation = json.loads(Path(args.confirmation).read_text())
    if development["split"] != "development":
        raise SystemExit("development input carries the wrong split")
    if confirmation["split"] != "confirmation":
        raise SystemExit("confirmation input carries the wrong split")
    if development["preflight_report_id"] != confirmation["preflight_report_id"]:
        raise SystemExit("behavior screens do not share a preflight")
    report = load_preflight_report(args.preflight_report)
    if report.decision != "QUALIFIED":
        raise SystemExit("replication analysis requires a qualified preflight")
    frozen_config_sha256(args.preflight_config)
    config = json.loads(Path(args.preflight_config).read_text())
    admitted = [str(item) for item in config["behavior"]["admitted_site_classes"]]
    verdicts = {}
    for site_class in admitted:
        development_decision = development["behavior_decisions"].get(site_class)
        confirmation_decision = confirmation["behavior_decisions"].get(site_class)
        replicated = bool(
            development_decision
            and confirmation_decision
            and development_decision["qualified_on_split"]
            and confirmation_decision["qualified_on_split"]
        )
        verdicts[site_class] = {
            "status": (
                "SUPPORTED_BEHAVIOR" if replicated else "NOT_BEHAVIOR_QUALIFIED"
            ),
            "development": development_decision,
            "confirmation": confirmation_decision,
            "first_missing_bridge": (
                "token-invariant context-source specificity"
                if replicated
                else "replicated excess reconstruction over every frozen control"
            ),
        }
    supported = sorted(
        site_class
        for site_class, value in verdicts.items()
        if value["status"] == "SUPPORTED_BEHAVIOR"
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "evidence_level": "BEHAVIOR",
        "preflight_report_id": report.report_id,
        "admitted_site_classes": admitted,
        "supported_site_classes": supported,
        "verdicts": verdicts,
        "hypothesis_h1": ("SUPPORTED: BEHAVIOR" if supported else "NOT SUPPORTED"),
        "claim_boundary": (
            "Replicated excess masked reconstruction is a scoped model behavior. "
            "It does not identify an informative biological context region or a circuit."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_atlas_behavioral_replication",
        status="SEALED",
        model_identity={
            "hf_id": DNABERT2_HF_ID,
            "revision": DNABERT2_REVISION,
        },
        input_paths=(
            args.development,
            args.confirmation,
            args.preflight_config,
            args.preflight_report,
        ),
        parameters={
            "qualification": (
                "same frozen SESOI, minimum clusters, direction, and Holm-corrected "
                "familywise alpha on development and confirmation"
            )
        },
        seeds={},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
