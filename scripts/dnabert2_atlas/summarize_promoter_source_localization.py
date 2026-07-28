#!/usr/bin/env python
"""Seal the replicated promoter source-localization conclusion."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    development = json.loads(Path(args.development).read_text())
    confirmation = json.loads(Path(args.confirmation).read_text())
    supply = json.loads(Path(args.supply_audit).read_text())
    preflight = json.loads(Path(args.preflight).read_text())
    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    if (
        development.get("split") != "development"
        or confirmation.get("split") != "confirmation"
        or development.get("config_sha256") != config_hash
        or confirmation.get("config_sha256") != config_hash
        or supply.get("status") != "QUALIFIED"
        or supply.get("model_outcomes_computed") is not False
        or not preflight.get("qualified")
        or preflight.get("config_sha256") != config_hash
    ):
        raise RuntimeError("matching development, confirmation, and preflight required")
    dev_test = development["primary_localization"]["tests"]["mask|tile26"]
    conf_test = confirmation["primary_localization"]["tests"]["mask|tile26"]
    replicated = bool(dev_test["qualified"] and conf_test["qualified"])
    dev_pls = development["biological_region_qualification"]["tests"][
        "mask|tile26|ENCODE_PLS_cCRE"
    ]
    conf_pls = confirmation["biological_region_qualification"]["tests"][
        "mask|tile26|ENCODE_PLS_cCRE"
    ]
    development_value_tests = [
        test
        for test in development["primary_localization"]["tests"].values()
        if test["operation"] == "candidate_value_swap"
    ]
    cover = {
        "development": _cover_summary(Path(development["raw"]["path"]), tile_index=26),
        "confirmation": _cover_summary(
            Path(confirmation["raw"]["path"]), tile_index=26
        ),
    }
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_source_localization_replication_v1",
        "status": "SEALED",
        "config_sha256": config_hash,
        "strongest_supported_claim": (
            "Equal-dose post-tokenization masking localizes a replicated "
            "promoter-classifier-relevant source region to the nominal +11 to +21 "
            "nt interval relative to fresh GENCODE MANE TSSs. This is a behavioral "
            "source-region claim, not a motif, L11H2/L11H5 mediation, or cellular "
            "promoter-function claim."
            if replicated
            else "No promoter source region replicated under the frozen gates."
        ),
        "replicated_behavioral_localization": {
            "qualified": replicated,
            "operation": "two-BPE-token post-tokenization [MASK]",
            "nominal_relative_interval": [11, 21],
            "development": {
                "n_pairs": dev_test["n_pairs"],
                "target_specific_loss_mean": dev_test["target_specific_loss_mean"],
                "target_specific_loss_ci95": dev_test["target_specific_loss_ci95"],
                "localization_contrast_mean": dev_test["localization_contrast_mean"],
                "localization_contrast_ci95": dev_test["localization_contrast_ci95"],
                "familywise_max_sign_flip_p": dev_test["familywise_max_sign_flip_p"],
            },
            "confirmation": {
                "n_pairs": conf_test["n_pairs"],
                "target_specific_loss_mean": conf_test["target_specific_loss_mean"],
                "target_specific_loss_ci95": conf_test["target_specific_loss_ci95"],
                "familywise_sign_flip_p": conf_test["familywise_max_sign_flip_p"],
            },
            "actual_BPE_cover_nt": cover,
            "precision_boundary": config["tile_geometry"]["precision_boundary"],
        },
        "biological_qualification": {
            "FANTOM5_CAGE": {
                "status": "not_qualified",
                "reason": "too few tile-26 CAGE-overlapping loci under the frozen regional sample gate",
            },
            "ENCODE_PLS_cCRE": {
                "status": "development_only_not_replicated",
                "development": {
                    "n": dev_pls["n_annotated"],
                    "mean": dev_pls["annotated_mean"],
                    "ci95": dev_pls["annotated_ci95"],
                    "familywise_p": dev_pls["familywise_max_sign_flip_p"],
                    "qualified": dev_pls["qualified"],
                },
                "confirmation": {
                    "n": conf_pls["n_annotated"],
                    "mean": conf_pls["annotated_mean"],
                    "ci95": conf_pls["annotated_ci95"],
                    "familywise_p": conf_pls["familywise_max_sign_flip_p"],
                    "qualified": conf_pls["qualified"],
                },
            },
            "UCSC_CpG_island": {
                "status": "not_qualified",
                "reason": "development annotated-subset effect did not pass",
            },
            "EPD_motifs": {
                "status": "no_intersecting_instance",
                "reason": (
                    "No geometry-qualified predefined TATA, Inr, CCAAT, or "
                    "GC-box instance intersected the nominated +11 to +21 tile; "
                    "therefore no motif was relabeled or disrupted post hoc."
                ),
            },
        },
        "D7_payload_localization": {
            "status": "not_supported_at_two_token_resolution",
            "maximum_development_raw_target_specific_loss": max(
                test["target_specific_loss_mean"] for test in development_value_tests
            ),
            "maximum_development_localization_contrast": max(
                test["localization_contrast_mean"] for test in development_value_tests
            ),
            "frozen_raw_SESOI": config["primary_gates"][
                "candidate_value_absolute_SESOI_logit_margin"
            ],
            "frozen_localization_SESOI": config["primary_gates"][
                "candidate_value_localization_contrast_SESOI_logit_margin"
            ],
            "boundary": (
                "D7's whole-sequence value-payload dependence replicated, but this "
                "atlas did not localize it to any single two-token tile."
            ),
        },
        "clean_behavior": {
            "development": development["clean_behavior"],
            "confirmation": confirmation["clean_behavior"],
        },
        "evidence_decision": {
            "replicated_classifier_source_region": replicated,
            "replicated_independent_biological_annotation": False,
            "replicated_sequence_specific_motif_necessity": False,
            "regional_D7_value_payload_source": False,
            "mediation": False,
        },
        "next_phase": (
            "Treat +11 to +21 as a behavioral source-region candidate. Before "
            "circuit tracing, freeze geometry-qualified native-nucleotide "
            "substitutions/shuffles across that region and a mask-by-payload "
            "interaction or rescue. Do not call it a promoter motif."
        ),
        "inputs": {
            "development": args.development,
            "confirmation": args.confirmation,
            "supply": args.supply_audit,
            "preflight": args.preflight,
        },
    }
    model_identity = json.loads(
        Path(str(args.confirmation).replace(".json", ".manifest.json")).read_text()
    )["checkpoint"]
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_source_localization_replication_v1",
        status="SEALED",
        model_identity=model_identity,
        input_paths=(
            args.development,
            str(args.development).replace(".json", ".manifest.json"),
            args.confirmation,
            str(args.confirmation).replace(".json", ".manifest.json"),
            development["raw"]["path"],
            development["raw"]["manifest"],
            confirmation["raw"]["path"],
            confirmation["raw"]["manifest"],
            args.config,
            args.supply_audit,
            args.preflight,
        ),
        parameters={"config_sha256": config_hash},
        seeds={"inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cover_summary(path: Path, *, tile_index: int) -> dict[str, Any]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("row_type") != "tile_effect"
                or row.get("operation") != "mask"
                or int(row.get("tile_index", -1)) != tile_index
            ):
                continue
            values.extend(
                (
                    int(row["target_cover_end"]) - int(row["target_cover_start"]),
                    int(row["control_cover_end"]) - int(row["control_cover_start"]),
                )
            )
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
