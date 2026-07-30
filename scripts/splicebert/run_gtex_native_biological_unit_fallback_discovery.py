#!/usr/bin/env python
"""Freeze exact L5 native-unit rankings on biological discovery constructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.mlp_units import unit_group_write_delta  # noqa: E402
from gtex_native_biological_unit_fallback import (  # noqa: E402
    CONTROL_DRAWS,
    DISCOVERY_SCHEMA,
    EXPECTED_SEEDS,
    GROUP_SIZES,
    INTERMEDIATE_SIZE,
    activity_statistics,
    candidate_contract,
    exact_unit_ranking,
    load_rank4_basis,
    matched_native_unit_control_orders,
    normalize_discovery_audit,
    oriented_arrays,
    oriented_state_names,
    rank4_energy_fraction,
    uniform_native_unit_control_orders,
    unit_overlap,
)
from gtex_native_biological_unit_runtime import (  # noqa: E402
    capture_family,
    classifier_parameters,
    load_checkpoint_runtime,
    parse_checkpoints,
    parse_heads,
    preflight_family,
    read_lock,
    verify_checkpoints,
    verify_file,
)


def safe_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def record_arrays(
    archive: dict[str, np.ndarray],
    *,
    seed: int,
    endpoint: str,
    candidate: str,
    states: dict[str, Any],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    prefix = f"s{seed}_{endpoint}_{safe_key(candidate)}"
    for state, fields in states.items():
        for field, value in fields.items():
            key = f"{prefix}_{state}_{field}"
            archive[key] = np.asarray(value)
            mapping[f"{state}.{field}"] = key
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--discovery-construction-audit", required=True)
    parser.add_argument("--discovery-sequence-package", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gfs19-arrays", required=True)
    parser.add_argument("--gfs20-discovery", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--arrays-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("batch must be positive")
    torch.set_grad_enabled(False)

    lock_path = Path(args.lock).expanduser().resolve()
    lock = read_lock(lock_path)
    audit_path = verify_file(
        args.discovery_construction_audit,
        lock["sources"]["discovery_construction"],
    )
    sequence_path = verify_file(
        args.discovery_sequence_package,
        lock["sources"]["discovery_sequence_package"],
    )
    package = normalize_discovery_audit(
        json.loads(audit_path.read_text()),
        json.loads(sequence_path.read_text()),
    )
    if candidate_contract(package) != lock["candidate_contract"]["discovery"]:
        raise RuntimeError("discovery construction contract differs from Lock C")
    gfs20_path = verify_file(
        args.gfs20_discovery, lock["sources"]["gfs20_discovery"]
    )
    gfs19_arrays_path = verify_file(
        args.gfs19_arrays, lock["sources"]["gfs19_arrays"]
    )
    gfs20 = json.loads(gfs20_path.read_text())
    checkpoints = parse_checkpoints(args.checkpoint)
    verify_checkpoints(checkpoints, lock)
    base_model = Path(args.base_model).expanduser().resolve()
    if str(base_model) != lock["sources"]["base_model"]["path"]:
        raise RuntimeError("base model path differs from Lock C")
    for record in lock["sources"]["base_model"]["files"]:
        verify_file(record["path"], record)

    output = guard_output_path(Path(args.out).expanduser().resolve())
    arrays_output = guard_output_path(Path(args.arrays_out).expanduser().resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)
    rankings: dict[str, Any] = {}
    checkpoint_reports: dict[str, Any] = {}
    preflight: list[dict[str, Any]] = []
    archive: dict[str, np.ndarray] = {}
    array_index: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        model, tokenizer, task, adapter, report = load_checkpoint_runtime(
            base_model,
            checkpoints[seed],
            seed=seed,
            device=device,
            batch_size=args.batch,
        )
        layer = adapter.encoder_layers[5]
        output_weight = (
            layer.output.dense.weight.detach().cpu().double().numpy()
        )
        archive[f"s{seed}_writer_columns"] = output_weight.astype(np.float32)
        archive[f"s{seed}_writer_bias"] = (
            layer.output.dense.bias.detach().cpu().float().numpy()
        )
        output_norms = np.linalg.norm(output_weight, axis=0)
        rankings[str(seed)] = {}
        array_index[str(seed)] = {}
        array_index[str(seed)]["writer_columns"] = f"s{seed}_writer_columns"
        array_index[str(seed)]["writer_bias"] = f"s{seed}_writer_bias"
        for candidate in package["candidates"]:
            endpoint = candidate["endpoint"]
            candidate_id = candidate["candidate_id"]
            candidate_key = f"{endpoint}:{candidate_id}"
            bridge_lock = lock["directed_bridge"]["frozen_head_sets"][endpoint]
            candidate_heads = parse_heads(bridge_lock["candidate_heads"])
            control_heads = [
                parse_heads(draw)
                for draw in bridge_lock["matched_head_control_draws"]
            ]
            family = candidate["families"][candidate_id]
            states, capture_check = capture_family(
                model,
                adapter,
                task,
                family["pairs"],
                endpoint=endpoint,
                candidate_heads=candidate_heads,
                control_head_draws=control_heads,
                batch_size=args.batch,
                device=device,
            )
            parameters = classifier_parameters(model, layer, endpoint)
            algebra_check = preflight_family(states, output_weight, parameters)
            if capture_check["write_reconstruction_max_abs_error"] > 5e-4:
                raise RuntimeError("native MLP write reconstruction preflight failed")
            preflight.append(
                {
                    "seed": seed,
                    "candidate_key": candidate_key,
                    **capture_check,
                    **algebra_check,
                }
            )
            preferred, adverse = oriented_arrays(
                states,
                candidate["expected_high_minus_low_margin_sign"],
            )
            genes = np.asarray(
                [pair["gene_id"] for pair in family["pairs"]], dtype=object
            )
            ranked = exact_unit_ranking(
                preferred, adverse, output_weight, parameters, genes
            )
            order = ranked["order"][: max(GROUP_SIZES)]
            activity = activity_statistics(
                preferred["activation"], adverse["activation"]
            )
            tag = f"bio-native-unit:{seed}:{candidate_key}"
            matched_controls = matched_native_unit_control_orders(
                order,
                output_norms,
                activation_rms=activity["activation_rms"],
                positive_activation_rate=activity["positive_activation_rate"],
                sequence_change_rate=activity["sequence_change_rate"],
                tag=tag,
                draws=CONTROL_DRAWS,
            )
            uniform_controls = uniform_native_unit_control_orders(
                order,
                INTERMEDIATE_SIZE,
                tag=tag,
                draws=CONTROL_DRAWS,
            )
            activation_delta = (
                preferred["activation"] - adverse["activation"]
            )
            gfs20_endpoint = gfs20["rankings"][str(seed)][endpoint]
            basis = load_rank4_basis(gfs19_arrays_path, seed, endpoint)
            groups: dict[str, Any] = {}
            for size in GROUP_SIZES:
                selected = order[:size]
                selected_write = unit_group_write_delta(
                    activation_delta, output_weight, selected
                )
                groups[str(size)] = {
                    "candidates": selected.tolist(),
                    "matched_unit_controls": [
                        values[:size].tolist() for values in matched_controls
                    ],
                    "uniform_unit_controls": [
                        values[:size].tolist() for values in uniform_controls
                    ],
                    "candidate_output_norm_mean": float(
                        output_norms[selected].mean()
                    ),
                    "candidate_activation_rms_mean": float(
                        activity["activation_rms"][selected].mean()
                    ),
                    "candidate_positive_activation_rate_mean": float(
                        activity["positive_activation_rate"][selected].mean()
                    ),
                    "candidate_sequence_change_rate_mean": float(
                        activity["sequence_change_rate"][selected].mean()
                    ),
                    "gfs20_overlap": {
                        family_name: unit_overlap(
                            selected,
                            gfs20_endpoint[family_name]["unit_order"],
                        )
                        for family_name in ("context", "head_induced")
                    },
                    "gfs19_rank4_energy_fraction_mean": float(
                        rank4_energy_fraction(selected_write, basis).mean()
                    ),
                }
            rankings[str(seed)][candidate_key] = {
                "endpoint": endpoint,
                "candidate_id": candidate_id,
                "expected_high_minus_low_margin_sign": candidate[
                    "expected_high_minus_low_margin_sign"
                ],
                "orientation": dict(
                    zip(
                        ("preferred", "adverse"),
                        oriented_state_names(
                            candidate[
                                "expected_high_minus_low_margin_sign"
                            ]
                        ),
                    )
                ),
                "n_pairs": len(family["pairs"]),
                "n_genes": len(set(genes)),
                "ranking_rule": (
                    "minimum of gene-mean exact native-unit damage and recovery"
                ),
                "unit_order": ranked["order"].tolist(),
                "unit_score": ranked["score"].tolist(),
                "unit_mean_damage": ranked["mean_damage"].tolist(),
                "unit_mean_recovery": ranked["mean_recovery"].tolist(),
                "output_norms": output_norms.tolist(),
                "activity_covariates": {
                    key: value.tolist() for key, value in activity.items()
                },
                "groups": groups,
            }
            array_index[str(seed)][candidate_key] = record_arrays(
                archive,
                seed=seed,
                endpoint=endpoint,
                candidate=candidate_id,
                states=states,
            )
            print(
                f"seed {seed} {candidate_key}: {len(family['pairs'])} pairs",
                flush=True,
            )
        checkpoint_reports[str(seed)] = report.to_dict()
        del task, adapter, model, tokenizer
        if device == "mps":
            torch.mps.empty_cache()

    result = {
        "schema": DISCOVERY_SCHEMA,
        "status": "COMPLETE",
        "stage": "discovery rankings and control-set freeze",
        "evidence_tier": "exploratory causal native-unit discovery",
        "object": (
            "exact L5 post-GELU native-unit populations; architectural, "
            "distributed/non-monosemantic coordinates, not sparse features"
        ),
        "sparse_weights_used": False,
        "lock": file_record(lock_path),
        "device": device,
        "checkpoint_reports": checkpoint_reports,
        "candidate_contract": candidate_contract(package),
        "preflight": {
            "checks": preflight,
            "all_pass": all(row["PASS"] for row in preflight),
            "max_write_reconstruction_error": max(
                row["write_reconstruction_max_abs_error"] for row in preflight
            ),
            "max_composition_error": max(
                row["composition_max_abs_error"] for row in preflight
            ),
            "max_all_unit_delta_error": max(
                row["all_unit_delta_max_abs_error"] for row in preflight
            ),
        },
        "freeze": {
            "checkpoint_specific": True,
            "candidate_specific": True,
            "group_sizes": list(GROUP_SIZES),
            "nested_prefixes": True,
            "control_draws": CONTROL_DRAWS,
            "confirmation_outcomes_read": False,
            "confirmation_authorized": True,
        },
        "rankings": rankings,
        "arrays": {
            "path": str(arrays_output),
            "index": array_index,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_output, **archive)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_biological_unit_fallback_discovery_v1",
        sys.argv,
        repo=REPO,
        status="EXPLORATORY",
        config={
            "batch": args.batch,
            "device": device,
            "group_sizes": list(GROUP_SIZES),
            "control_draws": CONTROL_DRAWS,
        },
        seeds={"checkpoint_family": list(EXPECTED_SEEDS)},
        notes=[
            "Discovery rankings and both control families are frozen before confirmation.",
            "Controls match writer norm and discovery activation/activity, not norm alone.",
            "No sparse dictionary weights are loaded or interpreted.",
        ],
    )
    for path in (
        lock_path,
        audit_path,
        sequence_path,
        gfs19_arrays_path,
        gfs20_path,
        *checkpoints.values(),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.extend((file_record(output), file_record(arrays_output)))
    manifest.write(manifest_path)
    print(output)


if __name__ == "__main__":
    main()
