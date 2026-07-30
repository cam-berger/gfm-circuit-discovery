#!/usr/bin/env python
"""Confirm exact native-unit mediation of biological low/high constructions."""

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
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gtex_native_biological_unit_fallback import (  # noqa: E402
    CONFIRMATION_SCHEMA,
    CONTROL_DRAWS,
    DISCOVERY_SCHEMA,
    EXPECTED_SEEDS,
    GROUP_SIZES,
    INTERMEDIATE_SIZE,
    N_BOOT,
    N_PERM,
    apply_declared_bh_family,
    candidate_contract,
    load_rank4_basis,
    normalize_confirmation_states,
    oriented_arrays,
    population_effects,
    rank4_energy_fraction,
    summarize_five_seed_effect,
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


DIRECT_ESTIMANDS = (
    "selected",
    "matched_unit_null",
    "uniform_unit_null",
    "selected_minus_matched_unit",
    "selected_minus_uniform_unit",
)
BRIDGE_ESTIMANDS = (
    "head_to_selected_units",
    "matched_head_to_selected_units",
    "candidate_head_to_matched_units",
    "matched_head_to_matched_units",
    "bridge_minus_matched_head",
    "bridge_minus_matched_units",
    "bridge_minus_joint_matched",
)


def safe_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def record_capture(
    archive: dict[str, np.ndarray],
    *,
    seed: int,
    candidate_key: str,
    family_id: str,
    states: dict[str, Any],
) -> dict[str, str]:
    index: dict[str, str] = {}
    prefix = f"s{seed}_{safe_key(candidate_key)}_{safe_key(family_id)}"
    for state, fields in states.items():
        for field, values in fields.items():
            key = f"{prefix}_{state}_{field}"
            archive[key] = np.asarray(values)
            index[f"{state}.{field}"] = key
    return index


def mean_effects(values: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        direction: np.mean(
            np.stack([value[direction] for value in values], axis=0), axis=0
        )
        for direction in ("damage", "recovery")
    }


def subtract_effects(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        direction: left[direction] - right[direction]
        for direction in ("damage", "recovery")
    }


def stripped_effects(value: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {direction: value[direction] for direction in ("damage", "recovery")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--confirmation-construction", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gfs19-arrays", required=True)
    parser.add_argument("--expected-genes", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--device", default=None)
    parser.add_argument("--arrays-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("batch must be positive")
    if args.n_boot != N_BOOT or args.n_perm != N_PERM:
        raise ValueError("inference settings differ from frozen Lock C")
    torch.set_grad_enabled(False)

    lock_path = Path(args.lock).expanduser().resolve()
    lock = read_lock(lock_path)
    discovery_path = Path(args.discovery).expanduser().resolve()
    discovery = json.loads(discovery_path.read_text())
    if (
        discovery.get("schema") != DISCOVERY_SCHEMA
        or discovery.get("status") != "COMPLETE"
        or discovery.get("sparse_weights_used") is not False
        or discovery.get("freeze", {}).get("confirmation_authorized") is not True
        or discovery.get("lock", {}).get("sha256") != sha256_file(lock_path)
    ):
        raise RuntimeError("frozen native-unit discovery does not authorize confirmation")
    if tuple(discovery["freeze"]["group_sizes"]) != GROUP_SIZES:
        raise RuntimeError("discovery nested size family differs from Lock C")

    construction_path = verify_file(
        args.confirmation_construction,
        lock["sources"]["confirmation_construction"],
    )
    package = normalize_confirmation_states(
        json.loads(construction_path.read_text()),
        expected_genes=args.expected_genes,
    )
    if candidate_contract(package) != lock["candidate_contract"]["confirmation"]:
        raise RuntimeError("confirmation construction contract differs from Lock C")
    discovery_genes = {
        row["gene_id"]
        for candidate in lock["candidate_contract"]["discovery"]
        for family in candidate["families"].values()
        for row in family["pair_ids"]
    }
    if discovery_genes & set(package["gene_ids"]):
        raise RuntimeError("confirmation genes overlap native-unit discovery")

    gfs19_arrays_path = verify_file(
        args.gfs19_arrays, lock["sources"]["gfs19_arrays"]
    )
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
    archive: dict[str, np.ndarray] = {}
    capture_index: dict[str, Any] = {}
    effect_values: dict[str, Any] = {}
    geometry: dict[str, Any] = {}
    ceilings: dict[str, Any] = {}
    preflight: list[dict[str, Any]] = []
    checkpoint_reports: dict[str, Any] = {}
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
        capture_index[str(seed)] = {
            "writer_columns": f"s{seed}_writer_columns",
            "writer_bias": f"s{seed}_writer_bias",
        }
        effect_values[str(seed)] = {}
        geometry[str(seed)] = {}
        ceilings[str(seed)] = {}
        for candidate in package["candidates"]:
            endpoint = candidate["endpoint"]
            candidate_id = candidate["candidate_id"]
            candidate_key = f"{endpoint}:{candidate_id}"
            if candidate_key not in discovery["rankings"][str(seed)]:
                raise RuntimeError(f"discovery has no ranking for {seed}/{candidate_key}")
            frozen_ranking = discovery["rankings"][str(seed)][candidate_key]
            if (
                frozen_ranking["expected_high_minus_low_margin_sign"]
                != candidate["expected_high_minus_low_margin_sign"]
            ):
                raise RuntimeError("confirmation sign differs from frozen discovery")
            bridge_lock = lock["directed_bridge"]["frozen_head_sets"][endpoint]
            candidate_heads = parse_heads(bridge_lock["candidate_heads"])
            control_heads = [
                parse_heads(draw)
                for draw in bridge_lock["matched_head_control_draws"]
            ]
            basis = load_rank4_basis(gfs19_arrays_path, seed, endpoint)
            capture_index[str(seed)][candidate_key] = {}
            effect_values[str(seed)][candidate_key] = {}
            geometry[str(seed)][candidate_key] = {}
            ceilings[str(seed)][candidate_key] = {}
            for family_id, family in candidate["families"].items():
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
                algebra_check = preflight_family(
                    states, output_weight, parameters
                )
                if capture_check["write_reconstruction_max_abs_error"] > 5e-4:
                    raise RuntimeError("native MLP write reconstruction preflight failed")
                preflight.append(
                    {
                        "seed": seed,
                        "candidate_key": candidate_key,
                        "family_id": family_id,
                        **capture_check,
                        **algebra_check,
                    }
                )
                preferred, adverse = oriented_arrays(
                    states,
                    candidate["expected_high_minus_low_margin_sign"],
                )
                preferred_head_noising = (
                    preferred["activation"]
                    - preferred["candidate_head_patched_activation"]
                )
                adverse_head_recovery = (
                    adverse["candidate_head_patched_activation"]
                    - adverse["activation"]
                )
                matched_head_deltas = []
                for draw in range(CONTROL_DRAWS):
                    field = f"control_head_patched_activation_{draw}"
                    matched_head_deltas.append(
                        (
                            preferred["activation"] - preferred[field],
                            adverse[field] - adverse["activation"],
                        )
                    )
                behavior = preferred["margin"] - adverse["margin"]
                ceiling = population_effects(
                    preferred,
                    adverse,
                    output_weight,
                    range(INTERMEDIATE_SIZE),
                    parameters,
                )
                ceilings[str(seed)][candidate_key][family_id] = {
                    "behavior_mean": float(behavior.mean()),
                    "full_mlp_damage_mean": float(ceiling["damage"].mean()),
                    "full_mlp_recovery_mean": float(ceiling["recovery"].mean()),
                }
                capture_index[str(seed)][candidate_key][family_id] = record_capture(
                    archive,
                    seed=seed,
                    candidate_key=candidate_key,
                    family_id=family_id,
                    states=states,
                )
                effect_values[str(seed)][candidate_key][family_id] = {}
                geometry[str(seed)][candidate_key][family_id] = {}
                for size in GROUP_SIZES:
                    group = frozen_ranking["groups"][str(size)]
                    selected_units = group["candidates"]
                    matched_units = group["matched_unit_controls"]
                    uniform_units = group["uniform_unit_controls"]
                    selected_full = population_effects(
                        preferred,
                        adverse,
                        output_weight,
                        selected_units,
                        parameters,
                    )
                    selected = stripped_effects(selected_full)
                    matched = [
                        stripped_effects(
                            population_effects(
                                preferred,
                                adverse,
                                output_weight,
                                units,
                                parameters,
                            )
                        )
                        for units in matched_units
                    ]
                    uniform = [
                        stripped_effects(
                            population_effects(
                                preferred,
                                adverse,
                                output_weight,
                                units,
                                parameters,
                            )
                        )
                        for units in uniform_units
                    ]
                    matched_mean = mean_effects(matched)
                    uniform_mean = mean_effects(uniform)
                    bridge_full = population_effects(
                        preferred,
                        adverse,
                        output_weight,
                        selected_units,
                        parameters,
                        noising_activation_delta=preferred_head_noising,
                        recovery_activation_delta=adverse_head_recovery,
                    )
                    bridge = stripped_effects(bridge_full)
                    matched_head = []
                    candidate_head_matched_units = []
                    joint_matched = []
                    for draw in range(CONTROL_DRAWS):
                        head_noising, head_recovery = matched_head_deltas[draw]
                        matched_head.append(
                            stripped_effects(
                                population_effects(
                                    preferred,
                                    adverse,
                                    output_weight,
                                    selected_units,
                                    parameters,
                                    noising_activation_delta=head_noising,
                                    recovery_activation_delta=head_recovery,
                                )
                            )
                        )
                        candidate_head_matched_units.append(
                            stripped_effects(
                                population_effects(
                                    preferred,
                                    adverse,
                                    output_weight,
                                    matched_units[draw],
                                    parameters,
                                    noising_activation_delta=preferred_head_noising,
                                    recovery_activation_delta=adverse_head_recovery,
                                )
                            )
                        )
                        joint_matched.append(
                            stripped_effects(
                                population_effects(
                                    preferred,
                                    adverse,
                                    output_weight,
                                    matched_units[draw],
                                    parameters,
                                    noising_activation_delta=head_noising,
                                    recovery_activation_delta=head_recovery,
                                )
                            )
                        )
                    matched_head_mean = mean_effects(matched_head)
                    matched_units_bridge_mean = mean_effects(
                        candidate_head_matched_units
                    )
                    joint_matched_mean = mean_effects(joint_matched)
                    estimands = {
                        "selected": selected,
                        "matched_unit_null": matched_mean,
                        "uniform_unit_null": uniform_mean,
                        "selected_minus_matched_unit": subtract_effects(
                            selected, matched_mean
                        ),
                        "selected_minus_uniform_unit": subtract_effects(
                            selected, uniform_mean
                        ),
                        "head_to_selected_units": bridge,
                        "matched_head_to_selected_units": matched_head_mean,
                        "candidate_head_to_matched_units": matched_units_bridge_mean,
                        "matched_head_to_matched_units": joint_matched_mean,
                        "bridge_minus_matched_head": subtract_effects(
                            bridge, matched_head_mean
                        ),
                        "bridge_minus_matched_units": subtract_effects(
                            bridge, matched_units_bridge_mean
                        ),
                        "bridge_minus_joint_matched": subtract_effects(
                            bridge, joint_matched_mean
                        ),
                    }
                    effect_values[str(seed)][candidate_key][family_id][
                        str(size)
                    ] = estimands
                    geometry[str(seed)][candidate_key][family_id][str(size)] = {
                        "gfs19_rank4_direct_damage_write_energy_mean": float(
                            rank4_energy_fraction(
                                selected_full["noising_write"], basis
                            ).mean()
                        ),
                        "gfs19_rank4_direct_recovery_write_energy_mean": float(
                            rank4_energy_fraction(
                                selected_full["recovery_write"], basis
                            ).mean()
                        ),
                        "gfs19_rank4_bridge_damage_write_energy_mean": float(
                            rank4_energy_fraction(
                                bridge_full["noising_write"], basis
                            ).mean()
                        ),
                        "gfs19_rank4_bridge_recovery_write_energy_mean": float(
                            rank4_energy_fraction(
                                bridge_full["recovery_write"], basis
                            ).mean()
                        ),
                        "gfs20_overlap": group["gfs20_overlap"],
                    }
                print(
                    f"seed {seed} {candidate_key}/{family_id}: "
                    f"{len(family['pairs'])} pairs",
                    flush=True,
                )
        checkpoint_reports[str(seed)] = report.to_dict()
        del task, adapter, model, tokenizer
        if device == "mps":
            torch.mps.empty_cache()

    # One declared confirmation family.  Checkpoint effects are averaged within
    # example before grouped inference; per-checkpoint signs remain explicit.
    entries: list[dict[str, Any]] = []
    effect_array_index: dict[str, str] = {}
    infer_seed = 940_000
    for candidate in package["candidates"]:
        candidate_key = f"{candidate['endpoint']}:{candidate['candidate_id']}"
        for family_id, family in candidate["families"].items():
            groups = np.asarray(
                [pair["gene_id"] for pair in family["pairs"]], dtype=object
            )
            for size in GROUP_SIZES:
                for estimand in (*DIRECT_ESTIMANDS, *BRIDGE_ESTIMANDS):
                    verdict = (
                        "direct_native_unit_population"
                        if estimand in DIRECT_ESTIMANDS
                        else "directed_head_to_native_unit_bridge"
                    )
                    for direction in ("damage", "recovery"):
                        per_seed = np.stack(
                            [
                                effect_values[str(seed)][candidate_key][family_id][
                                    str(size)
                                ][estimand][direction]
                                for seed in EXPECTED_SEEDS
                            ],
                            axis=0,
                        )
                        infer_seed += 1
                        summary = summarize_five_seed_effect(
                            per_seed,
                            groups,
                            seed=infer_seed,
                            n_boot=N_BOOT,
                            n_perm=N_PERM,
                        )
                        array_key = (
                            f"effect_{safe_key(candidate_key)}_"
                            f"{safe_key(family_id)}_k{size}_"
                            f"{safe_key(estimand)}_{direction}"
                        )
                        archive[array_key] = per_seed.astype(np.float32)
                        effect_array_index[
                            f"{candidate_key}/{family_id}/{size}/"
                            f"{estimand}/{direction}"
                        ] = array_key
                        entries.append(
                            {
                                "candidate_key": candidate_key,
                                "endpoint": candidate["endpoint"],
                                "candidate_id": candidate["candidate_id"],
                                "sequence_family": family_id,
                                "sequence_family_role": family["role"],
                                "size": size,
                                "estimand": estimand,
                                "direction": direction,
                                "verdict_family": verdict,
                                "summary": summary,
                            }
                        )
    apply_declared_bh_family(entries)

    result = {
        "schema": CONFIRMATION_SCHEMA,
        "status": "COMPLETE",
        "stage": "held-out exact native-unit and directed-path confirmation",
        "evidence_tier": "confirmatory causal native-unit populations",
        "object": (
            "exact L5 post-GELU native-unit populations; architectural, "
            "distributed/non-monosemantic coordinates, not sparse features"
        ),
        "sparse_instrument_status": "INSTRUMENT_UNRESOLVED",
        "sparse_weights_used": False,
        "lock": file_record(lock_path),
        "discovery": file_record(discovery_path),
        "confirmation_construction": file_record(construction_path),
        "device": device,
        "n_frozen_confirmation_genes": package["n_genes"],
        "n_analysis_genes": len(package["analysis_gene_ids"]),
        "checkpoint_reports": checkpoint_reports,
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
        "inference": {
            "five_checkpoint_average_within_example": True,
            "five_checkpoint_sign_replication_reported": True,
            "grouped_bootstrap_draws": N_BOOT,
            "exact_add_one_gene_sign_flips": N_PERM,
            "bh_family_name": "full_native_unit_confirmation_family",
            "bh_family_size": len(entries),
            "bh_members": (
                "all endpoints, candidate/control sequence families, six nested "
                "sizes, damage/recovery, direct populations/nulls/contrasts, and "
                "directed head-to-unit bridges/nulls/contrasts"
            ),
        },
        "results": entries,
        "geometry": geometry,
        "ceilings": ceilings,
        "arrays": {
            "path": str(arrays_output),
            "capture_index": capture_index,
            "effect_index": effect_array_index,
        },
        "evidence_boundary": (
            "Positive results license causal claims about distributed native-unit "
            "populations and the tested head-to-unit path. They do not license "
            "monosemantic, sparse-feature, direct RBP-binding, or biological "
            "mechanism claims."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_output, **archive)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_biological_unit_fallback_confirmation_v1",
        sys.argv,
        repo=REPO,
        status="CONFIRMATORY",
        config={
            "expected_genes": args.expected_genes,
            "batch": args.batch,
            "device": device,
            "group_sizes": list(GROUP_SIZES),
            "control_draws": CONTROL_DRAWS,
            "n_boot": N_BOOT,
            "n_perm": N_PERM,
            "bh_family_size": len(entries),
        },
        seeds={"checkpoint_family": list(EXPECTED_SEEDS)},
        notes=[
            "Discovery rankings and controls were frozen before confirmation.",
            "All checkpoints are loaded sequentially and released between seeds.",
            "The directed bridge separates frozen GFS18 heads from downstream native units.",
            "No sparse dictionary weights are loaded or interpreted.",
        ],
    )
    for path in (
        lock_path,
        discovery_path,
        construction_path,
        gfs19_arrays_path,
        *checkpoints.values(),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.extend((file_record(output), file_record(arrays_output)))
    manifest.write(manifest_path)
    print(output)


if __name__ == "__main__":
    main()
