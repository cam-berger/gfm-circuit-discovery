#!/usr/bin/env python
"""Preflight low-rank L11MLP projection patching on a fresh GUE slice."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.adapters.dnabert2.promoter_head_atlas import (
    paired_cls_projection_patch_tensor,
    paired_cls_write_patch_tensor,
)
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_head_scan import select_gue_pairs
from gfmcd.reconstruction_atlas.promoter_low_rank import (
    capture_mlp_writes_for_pairs,
    fit_uncentered_difference_basis,
    paired_mlp_projection_rows,
    random_orthonormal_basis,
    summarize_projection_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=16)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    if config.get("status") != "FROZEN":
        raise RuntimeError("low-rank preflight requires a frozen config")
    config_hash = frozen_config_sha256(args.config)
    supply = json.loads(Path(args.supply_audit).read_text())
    if (
        supply.get("status") != "QUALIFIED"
        or not all(supply.get("checks", {}).values())
        or supply.get("model_outcomes_computed") is not False
    ):
        raise RuntimeError("qualified outcome-blind D10 supply is required")
    preflight = config["preflight"]
    pairs = select_gue_pairs(
        args.gue_test,
        count=int(preflight["GUE_pairs"]),
        offset=int(preflight["GUE_pair_offset"]),
        seed=int(config["inference"]["seed"]),
    )
    fit_count = int(preflight["basis_fit_pairs"])
    fit_pairs = pairs[:fit_count]
    test_pairs = pairs[fit_count:]
    if len(test_pairs) != int(preflight["test_pairs"]):
        raise RuntimeError("GUE fit/test allocation does not match the frozen config")

    layer = int(config["scope"]["layer"])
    maximum_rank = int(config["basis"]["maximum_rank"])
    bundle = load_dnabert2_promoter_classifier(
        str(config["checkpoint"]["candidate"]), device=args.device
    )
    _fit_ids, _fit_margins, fit_writes = capture_mlp_writes_for_pairs(
        bundle,
        fit_pairs,
        layer=layer,
        batch_pairs=args.batch_pairs,
    )
    basis, diagnostics = fit_uncentered_difference_basis(
        fit_writes, maximum_rank=maximum_rank
    )
    repeated_basis, _repeated_diagnostics = fit_uncentered_difference_basis(
        fit_writes, maximum_rank=maximum_rank
    )
    basis_projector_drift = float(
        torch.max(torch.abs(basis @ basis.T - repeated_basis @ repeated_basis.T))
    )
    gram = basis.T @ basis
    orthonormality_error = float(torch.max(torch.abs(gram - torch.eye(gram.shape[0]))))

    test_ids, test_margins, test_writes = capture_mlp_writes_for_pairs(
        bundle,
        test_pairs,
        layer=layer,
        batch_pairs=len(test_pairs),
    )
    half_ids, half_margins, half_writes = capture_mlp_writes_for_pairs(
        bundle,
        test_pairs,
        layer=layer,
        batch_pairs=max(1, len(test_pairs) // 2),
    )
    if test_ids != half_ids:
        raise RuntimeError("batch-invariance capture changed pair order")
    batch_drift = max(
        float(torch.max(torch.abs(test_margins - half_margins))),
        float(torch.max(torch.abs(test_writes - half_writes))),
    )

    cls_rows = torch.arange(test_writes.shape[0])
    subspace = paired_cls_projection_patch_tensor(
        test_writes,
        cls_rows=cls_rows,
        basis=basis,
        direction="bidirectional_swap",
        component="subspace",
    )
    complement = paired_cls_projection_patch_tensor(
        test_writes,
        cls_rows=cls_rows,
        basis=basis,
        direction="bidirectional_swap",
        component="complement",
    )
    full_tensor = paired_cls_write_patch_tensor(
        test_writes,
        cls_rows=cls_rows,
        direction="bidirectional_swap",
    )
    reconstruction_error = float(
        torch.max(
            torch.abs(
                (subspace - test_writes)
                + (complement - test_writes)
                - (full_tensor - test_writes)
            )
        )
    )
    random_basis = random_orthonormal_basis(
        hidden_size=int(config["scope"]["hidden_size"]),
        rank=maximum_rank,
        seed=int(config["inference"]["seed"]) ^ 0xB4515,
    )
    random_orthonormality_error = float(
        torch.max(
            torch.abs(random_basis.T @ random_basis - torch.eye(random_basis.shape[1]))
        )
    )

    full_rows = paired_mlp_projection_rows(
        bundle,
        test_pairs,
        layer=layer,
        batch_pairs=args.batch_pairs,
        direction="bidirectional_swap",
        basis=None,
    )
    rank_zero_rows = paired_mlp_projection_rows(
        bundle,
        test_pairs,
        layer=layer,
        batch_pairs=args.batch_pairs,
        direction="bidirectional_swap",
        basis=torch.empty((int(config["scope"]["hidden_size"]), 0)),
    )
    inference = config["inference"]
    full_summary = summarize_projection_rows(
        full_rows,
        bootstrap_draws=int(inference["bootstrap_draws"]),
        sign_permutations=int(inference["sign_permutations"]),
        seed=int(inference["seed"]) ^ 0xF011,
    )
    rank_zero_drift = max(
        abs(float(row[native]) - float(row[patched]))
        for row in rank_zero_rows
        for native, patched in (
            ("native_target_margin", "patched_target_margin"),
            ("native_control_margin", "patched_control_margin"),
        )
    )
    finite = all(
        math.isfinite(float(row[key]))
        for row in (*full_rows, *rank_zero_rows)
        for key in (
            "native_target_margin",
            "native_control_margin",
            "patched_target_margin",
            "patched_control_margin",
            "class_margin_loss",
        )
    )
    clean = full_summary["clean"]
    checks = {
        "fresh_disjoint_GUE_slice": int(preflight["GUE_pair_offset"]) >= 960,
        "clean_accuracy": float(clean["accuracy"])
        >= float(preflight["minimum_clean_accuracy"]),
        "clean_AUROC": float(clean["auroc"]) >= float(preflight["minimum_clean_AUROC"]),
        "full_L11MLP_sensitivity": float(full_summary["class_margin_loss"])
        >= float(preflight["minimum_full_L11MLP_class_margin_loss"]),
        "rank_zero_noop": rank_zero_drift
        <= float(preflight["maximum_rank_zero_absolute_margin_difference"]),
        "projection_reconstruction": reconstruction_error
        <= float(preflight["maximum_projection_reconstruction_error"]),
        "batch_invariance": batch_drift
        <= float(preflight["maximum_batch_invariance_absolute_margin_difference"]),
        "basis_orthonormality": max(orthonormality_error, random_orthonormality_error)
        <= float(preflight["maximum_basis_orthonormality_error"]),
        "deterministic_basis": basis_projector_drift == 0.0,
        "finite_outputs": finite,
    }
    qualified = all(checks.values())
    artifact_status = "PREFLIGHT" if qualified else "INSTRUMENT_UNRESOLVED"
    input_paths = (
        args.gue_test,
        args.config,
        args.supply_audit,
        args.checkpoint_audit,
    )
    raw_rows = [
        *({"condition": "full_L11MLP_swap", **row} for row in full_rows),
        *({"condition": "rank_zero_subspace", **row} for row in rank_zero_rows),
    ]
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id="dnabert2_promoter_low_rank_preflight_raw_v1",
        status=artifact_status,
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "config_sha256": config_hash,
            "preflight": preflight,
            "basis": config["basis"],
        },
        seeds={"basis_and_inference": int(inference["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_L11MLP_low_rank_preflight_v1",
        "status": "QUALIFIED" if qualified else "INSTRUMENT_UNRESOLVED",
        "qualified": qualified,
        "target_cohort_outcomes_computed": False,
        "config_sha256": config_hash,
        "checkpoint_revision": config["checkpoint"]["revision"],
        "GUE_pair_ids": [pair[0] for pair in pairs],
        "basis_diagnostics": diagnostics,
        "measurements": {
            "clean": clean,
            "full_L11MLP": full_summary,
            "rank_zero_max_absolute_margin_difference": rank_zero_drift,
            "projection_reconstruction_max_absolute_error": reconstruction_error,
            "batch_invariance_max_absolute_difference": batch_drift,
            "basis_orthonormality_max_absolute_error": orthonormality_error,
            "random_basis_orthonormality_max_absolute_error": (
                random_orthonormality_error
            ),
            "deterministic_basis_projector_max_absolute_difference": (
                basis_projector_drift
            ),
        },
        "checks": checks,
        "input_sha256": {
            "gue_test": sha256_file(args.gue_test),
            "config": sha256_file(args.config),
            "supply_audit": sha256_file(args.supply_audit),
            "checkpoint_audit": sha256_file(args.checkpoint_audit),
        },
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "rows": len(raw_rows),
        },
        "boundary": (
            "Fresh GUE preflight validates projection geometry, sensitivity, and "
            "numerical invariants only. It reads no D10 target outcome."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_low_rank_preflight_v1",
        status=artifact_status,
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={
            "config_sha256": config_hash,
            "preflight": preflight,
            "basis": config["basis"],
            "batch_pairs": args.batch_pairs,
        },
        seeds={"basis_and_inference": int(inference["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "checks": checks,
                "full_L11MLP_class_margin_loss": full_summary["class_margin_loss"],
                "rank_zero_drift": rank_zero_drift,
                "projection_reconstruction_error": reconstruction_error,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
