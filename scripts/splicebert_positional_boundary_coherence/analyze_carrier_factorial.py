#!/usr/bin/env python
"""Analyze the sealed positional-boundary-coherence carrier factorial."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_carrier_factorial import (  # noqa: E402
    MIN_PSEUDO_PHASE_CONTROLS,
    MIN_STRUCTURED_CONTROLS,
    SESOI,
    balanced_pseudo_phase_assignment_audit,
    balanced_pseudo_phase_matrix,
    block_psi,
    carrier_phase2_decision,
    exact_add_one_absolute_p,
    exact_add_one_upper_p,
    factorial_contrasts,
    mde80_from_bootstrap,
    percentile_stratified_interval,
    simultaneous_stratified_intervals,
)


TARGET_SCHEMA = "gfmcd.pbc.carrier_factorial_target.v1"
COMPACT_EVIDENCE_SCHEMA = "gfmcd.pbc.carrier_factorial_compact_blocks.v1"
ANALYSIS_SCHEMA = "gfmcd.pbc.carrier_factorial_analysis.v1"
CONFIRMATORY_BOOTSTRAP_DRAWS = 20_000
CONFIRMATORY_BOOTSTRAP_SEED = 2026081001
CONFIRMATORY_PSEUDO_CONTROLS = 999
CONFIRMATORY_PSEUDO_SEED = 2026081002


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    return output, manifest


def _condition_blocks(
    rows: Sequence[Mapping[str, Any]], *, value_key: str
) -> dict[str, dict[str, Any]]:
    by_block: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_block.setdefault(str(row["block_id"]), []).append(row)
    output: dict[str, dict[str, Any]] = {}
    for block_id, block_rows in by_block.items():
        summary = block_psi(block_rows, value_key=value_key)
        chroms = {str(row["chrom"]) for row in block_rows}
        strands = {str(row["strand"]) for row in block_rows}
        if len(chroms) != 1 or len(strands) != 1:
            raise RuntimeError(f"block {block_id} changes sampling stratum")
        output[block_id] = {
            **summary,
            "chrom": next(iter(chroms)),
            "strand": next(iter(strands)),
            "stratum": f"{next(iter(chroms))}|{next(iter(strands))}",
            "rows": list(block_rows),
        }
    return output


def _group_conditions(
    target: Mapping[str, Any], *, value_key: str
) -> dict[str, dict[str, dict[str, Any]]]:
    specs = target.get("condition_specs")
    evidence = target.get("compact_evidence")
    if not isinstance(specs, Mapping) or not isinstance(evidence, Mapping):
        raise RuntimeError("compact target condition evidence is absent")
    if evidence.get("schema") != COMPACT_EVIDENCE_SCHEMA:
        raise RuntimeError("wrong compact target evidence schema")
    block_rows = evidence.get("blocks")
    block_order = evidence.get("block_order")
    endpoints = evidence.get("condition_endpoints")
    if (
        not isinstance(block_rows, list)
        or not isinstance(block_order, list)
        or not isinstance(endpoints, Mapping)
        or len(block_rows) != int(target["n_blocks"])
        or [str(row.get("block_id")) for row in block_rows] != block_order
        or len(set(block_order)) != len(block_order)
        or set(endpoints) != set(specs) - {"Y00"}
    ):
        raise RuntimeError("compact target block/condition support is malformed")
    output: dict[str, dict[str, dict[str, Any]]] = {
        "Y00": {}
    }
    for block in block_rows:
        block_id = str(block["block_id"])
        cells = block.get("y00_cells")
        if not isinstance(cells, list) or len(cells) != 9:
            raise RuntimeError("compact Y00 block lacks nine phase cells")
        summary = block_psi(cells, value_key=value_key)
        chrom = str(block["chrom"])
        strand = str(block["strand"])
        output["Y00"][block_id] = {
            **summary,
            "chrom": chrom,
            "strand": strand,
            "stratum": f"{chrom}|{strand}",
            "rows": cells,
        }
    endpoint_key = "psi_ac" if value_key == "coupling_ac" else "psi_at"
    for condition, spec in specs.items():
        if condition == "Y00":
            continue
        if value_key == "coupling_at" and str(spec.get("family")) != "target":
            continue
        row = endpoints.get(condition)
        values = row.get(endpoint_key) if isinstance(row, Mapping) else None
        if (
            not isinstance(values, list)
            or len(values) != len(block_order)
            or not np.isfinite(np.asarray(values, dtype=float)).all()
        ):
            raise RuntimeError(f"compact endpoint {condition}/{endpoint_key} is malformed")
        output[str(condition)] = {
            str(block_id): {
                "Psi": float(values[index]),
                "chrom": str(block_rows[index]["chrom"]),
                "strand": str(block_rows[index]["strand"]),
                "stratum": (
                    f"{block_rows[index]['chrom']}|{block_rows[index]['strand']}"
                ),
                "rows": [],
            }
            for index, block_id in enumerate(block_order)
        }
    return output


def _world_arrays(
    blocks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    block_ids: Sequence[str],
    worlds: Sequence[str],
) -> dict[str, np.ndarray]:
    return {
        world: np.asarray(
            [float(blocks[world][block_id]["Psi"]) for block_id in block_ids],
            dtype=float,
        )
        for world in worlds
    }


def _contrast_arrays(worlds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    n3 = worlds["Y00"] - worlds["Y10"]
    result = {"N3": n3}
    if "Y01" in worlds and "Y11" in worlds:
        result.update(
            {
                "N10": worlds["Y00"] - worlds["Y01"],
                "Nboth": worlds["Y00"] - worlds["Y11"],
                "J": worlds["Y11"]
                - worlds["Y10"]
                - worlds["Y01"]
                + worlds["Y00"],
            }
        )
    return result


def _structured_nulls(
    target: Mapping[str, Any],
    blocks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    block_ids: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    specs = target["condition_specs"]
    y00 = np.asarray(
        [float(blocks["Y00"][block_id]["Psi"]) for block_id in block_ids]
    )
    singles: dict[str, dict[int, np.ndarray]] = {
        "p3_phase": {},
        "p3_axis": {},
        "p10_phase": {},
        "p10_axis": {},
    }
    joint_p3: dict[int, np.ndarray] = {}
    joint_p10: dict[int, np.ndarray] = {}
    joint: dict[int, np.ndarray] = {}
    for condition, spec in specs.items():
        family = str(spec["family"])
        if family == "target":
            continue
        control_id = int(spec["control_id"])
        values = np.asarray(
            [float(blocks[condition][block_id]["Psi"]) for block_id in block_ids]
        )
        if family in singles:
            singles[family][control_id] = y00 - values
        elif family == "joint_p3":
            joint_p3[control_id] = values
        elif family == "joint_p10":
            joint_p10[control_id] = values
        elif family == "joint":
            joint[control_id] = values
        else:
            raise RuntimeError(f"unknown structured-null family {family}")
    output: dict[str, dict[str, np.ndarray]] = {
        "N3": {
            family: np.stack([values for _, values in sorted(singles[family].items())])
            for family in ("p3_phase", "p3_axis")
        }
    }
    if target["target_mode"] == "FULL_FACTORIAL":
        if not (
            set(joint_p3)
            == set(joint_p10)
            == set(joint)
            and set(singles["p10_phase"])
            and set(singles["p10_axis"])
        ):
            raise RuntimeError("joint controls do not share the single-family IDs")
        output["N10"] = {
            family: np.stack(
                [values for _, values in sorted(singles[family].items())]
            )
            for family in ("p10_phase", "p10_axis")
        }
        output["J"] = {
            "joint": np.stack(
                [
                    joint[index]
                    - joint_p3[index]
                    - joint_p10[index]
                    + y00
                    for index in sorted(joint)
                ]
            )
        }
    return output


def _clean_pseudo_phase(
    blocks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    block_ids: Sequence[str],
    *,
    n_controls: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    rows = [blocks["Y00"][block_id]["rows"] for block_id in block_ids]
    target = float(np.mean([blocks["Y00"][block_id]["Psi"] for block_id in block_ids]))
    matrix = balanced_pseudo_phase_matrix(
        rows,
        n_controls=n_controls,
        seed=seed,
    )
    controls = matrix.mean(axis=1)
    return {
        "target": target,
        "n_controls": len(controls),
        "control_mean": float(controls.mean()),
        "control_max": float(controls.max()),
        "target_exceeds_every_control": bool(target > controls.max()),
        "add_one_p_one_sided": float(
            (1 + np.sum(controls >= target)) / (len(controls) + 1)
        ),
        "assignment_audit": balanced_pseudo_phase_assignment_audit(
            len(block_ids), n_controls=n_controls, seed=seed
        ),
    }, matrix


def _verdict(
    *,
    clean_behavior_classification: str,
    endpoints: Mapping[str, Mapping[str, Any]],
    specificity: Mapping[str, Mapping[str, Any]],
    target_mode: str,
    target_resolution: Mapping[str, Mapping[str, Any]],
) -> str:
    return str(
        carrier_phase2_decision(
            clean_behavior_classification=clean_behavior_classification,
            endpoints=endpoints,
            specificity=specificity,
            target_mode=target_mode,
            target_resolution=target_resolution,
        )["verdict"]
    )


def analyze_fixture_arrays(
    fixture: Mapping[str, Any],
    *,
    target_mode: str = "FULL_FACTORIAL",
    n_bootstrap: int = CONFIRMATORY_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = CONFIRMATORY_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run the frozen Phase-II families and verdict on synthetic block arrays.

    The calibration producer uses this entry point so its null and planted
    replicates traverse the same simultaneous intervals, exact add-one tests,
    MDE calculation, classifications, and decision engine as the target
    analysis.  ``fixture`` stores sufficient block-level inputs for the exact
    families; it never accepts caller-supplied pass flags.
    """

    values = fixture.get("values")
    directions = fixture.get("directions")
    specificity_inputs = fixture.get("specificity")
    strata_raw = fixture.get("strata")
    if not all(
        isinstance(item, Mapping)
        for item in (values, directions, specificity_inputs)
    ) or not isinstance(strata_raw, Sequence):
        raise ValueError("fixture arrays require values, directions, specificity, and strata")
    strata = tuple(str(value) for value in strata_raw)

    def vector(container: Mapping[str, Any], name: str) -> np.ndarray:
        array = np.asarray(container[name], dtype=float)
        if array.shape != (len(strata),) or not np.isfinite(array).all():
            raise ValueError(f"fixture vector {name} is not finite block support")
        return array

    clean = vector(values, "clean")
    n3 = vector(values, "N3")
    n10 = vector(values, "N10")
    interaction = vector(values, "J")
    direction_names = ("Dplus", "Dminus", "Psi0", "Psi1", "Psi2")
    direction_matrix = np.column_stack(
        [vector(directions, name) for name in direction_names]
    )

    def control(name: str, family: str) -> tuple[np.ndarray, np.ndarray]:
        endpoint = specificity_inputs.get(name)
        row = endpoint.get(family) if isinstance(endpoint, Mapping) else None
        if not isinstance(row, Mapping):
            raise ValueError(f"fixture specificity {name}/{family} is absent")
        means = np.asarray(row.get("control_means"), dtype=float)
        block_mean = np.asarray(row.get("block_mean"), dtype=float)
        expected = (
            MIN_PSEUDO_PHASE_CONTROLS
            if name == "clean"
            else MIN_STRUCTURED_CONTROLS
        )
        if (
            means.shape != (expected,)
            or block_mean.shape != (len(strata),)
            or not np.isfinite(means).all()
            or not np.isfinite(block_mean).all()
            or not math.isclose(
                float(means.mean()),
                float(block_mean.mean()),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"fixture specificity {name}/{family} is malformed")
        return means, block_mean

    if target_mode not in {"P3_ONLY", "FULL_FACTORIAL"}:
        raise ValueError("fixture target_mode must be P3_ONLY or FULL_FACTORIAL")
    pseudo_means, pseudo_block_mean = control("clean", "pseudo")
    pseudo_input = specificity_inputs["clean"]["pseudo"]
    assignment_exact = bool(pseudo_input.get("assignment_exact", False))
    n3_phase_means, n3_phase_block = control("N3", "phase")
    n3_axis_means, n3_axis_block = control("N3", "axis")
    n10_phase_means, n10_phase_block = control("N10", "phase")
    n10_axis_means, n10_axis_block = control("N10", "axis")
    j_means, j_block = control("J", "joint")

    p3_family = simultaneous_stratified_intervals(
        np.column_stack((clean, n3)),
        names=("Psi00", "N3"),
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    endpoints = p3_family.by_name(sesoi=SESOI)
    optional_family = simultaneous_stratified_intervals(
        np.column_stack((n10, interaction)),
        names=("N10", "J"),
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed + 1,
    )
    endpoints.update(optional_family.by_name(sesoi=SESOI))
    paired_p3 = simultaneous_stratified_intervals(
        np.column_stack(
            (
                clean - pseudo_block_mean,
                n3 - n3_phase_block,
                n3 - n3_axis_block,
            )
        ),
        names=("clean", "N3_phase", "N3_axis"),
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed + 4,
    ).by_name(sesoi=0.0)
    j_sign = 1.0 if float(interaction.mean()) >= 0.0 else -1.0
    paired_optional = simultaneous_stratified_intervals(
        np.column_stack(
            (
                n10 - n10_phase_block,
                n10 - n10_axis_block,
                j_sign * (interaction - j_block),
            )
        ),
        names=("N10_phase", "N10_axis", "J_signed"),
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed + 5,
    ).by_name(sesoi=0.0)
    direction_family = simultaneous_stratified_intervals(
        direction_matrix,
        names=direction_names,
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed + 6,
    ).by_name(sesoi=0.0)

    def ordinary_specificity(
        target: np.ndarray,
        controls: Sequence[tuple[np.ndarray, str]],
        paired: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for means, label in controls:
            paired_row = paired[label]
            rows[label] = {
                "add_one_p": exact_add_one_upper_p(float(target.mean()), means),
                "target_exceeds_every_control": bool(target.mean() > means.max()),
                "paired_ci": list(paired_row["ci"]),
                "paired_lower_above_zero": bool(paired_row["ci"][0] > 0),
            }
        return {
            "families": rows,
            "corrected_pass": bool(
                all(
                    row["add_one_p"] <= 1.0 / 32.0
                    and row["target_exceeds_every_control"]
                    and row["paired_lower_above_zero"]
                    for row in rows.values()
                )
            ),
        }

    clean_specificity = {
        "add_one_p": exact_add_one_upper_p(float(clean.mean()), pseudo_means),
        "target_exceeds_every_control": bool(clean.mean() > pseudo_means.max()),
        "paired_ci": list(paired_p3["clean"]["ci"]),
        "paired_lower_above_zero": bool(paired_p3["clean"]["ci"][0] > 0),
    }
    clean_specificity["corrected_pass"] = bool(
        clean_specificity["add_one_p"] <= 1.0 / 1000.0
        and clean_specificity["target_exceeds_every_control"]
        and clean_specificity["paired_lower_above_zero"]
        and assignment_exact
    )
    n3_specificity = ordinary_specificity(
        n3,
        ((n3_phase_means, "N3_phase"), (n3_axis_means, "N3_axis")),
        paired_p3,
    )
    n10_specificity = ordinary_specificity(
        n10,
        ((n10_phase_means, "N10_phase"), (n10_axis_means, "N10_axis")),
        paired_optional,
    )
    j_paired = paired_optional["J_signed"]
    j_specificity = {
        "absolute_add_one_p": exact_add_one_absolute_p(
            float(interaction.mean()), j_means
        ),
        "target_absolute_exceeds_every_control": bool(
            abs(float(interaction.mean())) > float(np.abs(j_means).max())
        ),
        "paired_ci": list(j_paired["ci"]),
        "paired_signed_lower_above_zero": bool(j_paired["ci"][0] > 0),
    }
    j_specificity["corrected_pass"] = bool(
        j_specificity["absolute_add_one_p"] <= 1.0 / 32.0
        and j_specificity["target_absolute_exceeds_every_control"]
        and j_specificity["paired_signed_lower_above_zero"]
    )
    specificity = {
        "Psi00": clean_specificity,
        "N3": n3_specificity,
        "N10": n10_specificity,
        "J": j_specificity,
    }
    resolution = {
        name: {
            "mde80": mde80_from_bootstrap(array, strata),
            "threshold": 0.04 if name in {"Psi00", "N3"} else 0.05,
        }
        for name, array in (
            ("Psi00", clean),
            ("N3", n3),
            ("N10", n10),
            ("J", interaction),
        )
    }
    for row in resolution.values():
        row["pass"] = bool(row["mde80"] <= row["threshold"])
    directional_lower = min(float(row["ci"][0]) for row in direction_family.values())
    clean_ci = endpoints["Psi00"]["ci"]
    if (
        clean_ci[0] > SESOI
        and directional_lower > 0
        and clean_specificity["corrected_pass"]
        and resolution["Psi00"]["pass"]
    ):
        clean_classification = "BALANCED_PREFERENCE_MATERIAL"
    elif (
        clean_ci[0] >= -SESOI
        and clean_ci[1] <= SESOI
        and resolution["Psi00"]["pass"]
    ):
        clean_classification = "BALANCED_PREFERENCE_BOUNDED_NULL"
    else:
        clean_classification = "BALANCED_PREFERENCE_INCONCLUSIVE"
    decision = carrier_phase2_decision(
        clean_behavior_classification=clean_classification,
        endpoints=endpoints,
        specificity=specificity,
        target_mode=target_mode,
        target_resolution=resolution,
    )

    def endpoint_record(
        name: str,
        family: Mapping[str, Any],
        paired_names: Sequence[str],
    ) -> dict[str, Any]:
        family_rows = family.get("families")
        if isinstance(family_rows, Mapping):
            rows = [family_rows[label] for label in paired_names]
            pvalues = [float(row["add_one_p"]) for row in rows]
            exceeds = all(bool(row["target_exceeds_every_control"]) for row in rows)
            paired_cis = [list(row["paired_ci"]) for row in rows]
        else:
            pvalues = [float(family["absolute_add_one_p"])]
            exceeds = bool(family["target_absolute_exceeds_every_control"])
            paired_cis = [list(family["paired_ci"])]
        return {
            "simultaneous_ci": list(endpoints[name]["ci"]),
            "classification": endpoints[name]["classification"],
            "add_one_p": pvalues,
            "target_exceeds_every_control": exceeds,
            "paired_cis": paired_cis,
            "paired_ci": [
                min(float(row[0]) for row in paired_cis),
                max(float(row[1]) for row in paired_cis),
            ],
            "mde80": float(resolution[name]["mde80"]),
            "corrected_specificity_pass": bool(family["corrected_pass"]),
        }

    return {
        "analysis_engine": "analyze_carrier_factorial.analyze_fixture_arrays",
        "analysis_parameters": {
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
            "n_blocks": len(strata),
        },
        "target_mode": target_mode,
        "clean": {
            "simultaneous_ci": list(endpoints["Psi00"]["ci"]),
            "classification": clean_classification,
            "minimum_directional_lower_bound": directional_lower,
            "pseudo_add_one_p": float(clean_specificity["add_one_p"]),
            "target_exceeds_every_pseudo": bool(
                clean_specificity["target_exceeds_every_control"]
            ),
            "paired_pseudo_ci": list(clean_specificity["paired_ci"]),
            "mde80": float(resolution["Psi00"]["mde80"]),
            "corrected_specificity_pass": bool(
                clean_specificity["corrected_pass"]
            ),
        },
        "N3": endpoint_record(
            "N3", n3_specificity, ("N3_phase", "N3_axis")
        ),
        "N10": endpoint_record(
            "N10", n10_specificity, ("N10_phase", "N10_axis")
        ),
        "J": endpoint_record("J", j_specificity, ("J_signed",)),
        "decision": decision,
        "engine_details": {
            "endpoints": endpoints,
            "specificity": specificity,
            "direction_family": direction_family,
            "directional_pass": bool(directional_lower > 0),
            "target_resolution": resolution,
            "clean_behavior_classification": clean_classification,
        },
    }


def analyze_factorial_rows(
    target: Mapping[str, Any],
    *,
    n_bootstrap: int = CONFIRMATORY_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = CONFIRMATORY_BOOTSTRAP_SEED,
    n_pseudo: int = CONFIRMATORY_PSEUDO_CONTROLS,
    pseudo_seed: int = CONFIRMATORY_PSEUDO_SEED,
) -> dict[str, Any]:
    """Re-derive all target estimands from compact cell rows."""

    if target.get("schema") != TARGET_SCHEMA:
        raise RuntimeError("wrong carrier-factorial target schema")
    if target.get("stage") != "target":
        raise RuntimeError("carrier-factorial artifact does not declare target stage")
    if target.get("status") != "TARGET_SCORED_UNANALYZED":
        raise RuntimeError("target is not in the analyzable state")
    if target.get("mechanical_audit", {}).get("passed") is not True:
        raise RuntimeError("target mechanical audit did not pass")
    if (
        n_bootstrap != CONFIRMATORY_BOOTSTRAP_DRAWS
        or bootstrap_seed != CONFIRMATORY_BOOTSTRAP_SEED
        or n_pseudo != CONFIRMATORY_PSEUDO_CONTROLS
        or pseudo_seed != CONFIRMATORY_PSEUDO_SEED
    ):
        raise RuntimeError("confirmatory resampling counts and seeds are frozen")
    mode = str(target["target_mode"])
    worlds = ("Y00", "Y10") if mode == "P3_ONLY" else ("Y00", "Y10", "Y01", "Y11")
    blocks_ac = _group_conditions(target, value_key="coupling_ac")
    blocks_at = _group_conditions(target, value_key="coupling_at")
    block_ids = tuple(sorted(blocks_ac["Y00"]))
    strata = tuple(blocks_ac["Y00"][block_id]["stratum"] for block_id in block_ids)
    world_ac = _world_arrays(blocks_ac, block_ids, worlds)
    world_at = _world_arrays(blocks_at, block_ids, worlds)
    contrasts_ac = _contrast_arrays(world_ac)
    contrasts_at = _contrast_arrays(world_at)

    p3_matrix = np.column_stack([world_ac["Y00"], contrasts_ac["N3"]])
    p3_family = simultaneous_stratified_intervals(
        p3_matrix,
        names=("Psi00", "N3"),
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    endpoints = p3_family.by_name(sesoi=SESOI)
    family_records: dict[str, Any] = {
        "p3_primary": {
            "members": ["Psi00", "N3"],
            "critical_value": p3_family.critical_value,
            "n_bootstrap": p3_family.n_bootstrap,
        }
    }
    if mode == "FULL_FACTORIAL":
        optional_matrix = np.column_stack(
            [contrasts_ac["N10"], contrasts_ac["J"]]
        )
        optional_family = simultaneous_stratified_intervals(
            optional_matrix,
            names=("N10", "J"),
            strata=strata,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + 1,
        )
        endpoints.update(optional_family.by_name(sesoi=SESOI))
        family_records["optional_p10"] = {
            "members": ["N10", "J"],
            "critical_value": optional_family.critical_value,
            "n_bootstrap": optional_family.n_bootstrap,
            "does_not_tax_p3_family": True,
        }
        nboth = percentile_stratified_interval(
            contrasts_ac["Nboth"],
            strata,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + 2,
        )
        endpoints["Nboth"] = {
            "estimate": nboth["estimate"],
            "ci": [nboth["ci_low"], nboth["ci_high"]],
            "classification": "DESCRIPTIVE_NOT_IN_CONFIRMATORY_FAMILY",
        }
    else:
        endpoints.update(
            {
                name: {"status": "NOT_RUN_OPTIONAL_P10_INSTRUMENT_UNRESOLVED"}
                for name in ("N10", "Nboth", "J")
            }
        )

    clean_pseudo, pseudo_matrix = _clean_pseudo_phase(
        blocks_ac,
        block_ids,
        n_controls=n_pseudo,
        seed=pseudo_seed,
    )
    nulls = _structured_nulls(target, blocks_ac, block_ids)
    target_means = {
        name: float(values.mean()) for name, values in contrasts_ac.items()
    }
    p3_phase_null = nulls["N3"]["p3_phase"]
    p3_axis_null = nulls["N3"]["p3_axis"]
    p3_paired_matrix = np.column_stack(
        [
            world_ac["Y00"] - pseudo_matrix.mean(axis=0),
            contrasts_ac["N3"] - p3_phase_null.mean(axis=0),
            contrasts_ac["N3"] - p3_axis_null.mean(axis=0),
        ]
    )
    p3_paired = simultaneous_stratified_intervals(
        p3_paired_matrix,
        names=("Psi00_pseudo_excess", "N3_phase_excess", "N3_axis_excess"),
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed + 4,
    ).by_name(sesoi=0.0)

    def single_family(
        target_values: np.ndarray,
        controls: np.ndarray,
        paired_name: str,
    ) -> dict[str, Any]:
        control_means = controls.mean(axis=1)
        target_mean = float(target_values.mean())
        paired_row = p3_paired[paired_name]
        return {
            "n_controls": int(len(controls)),
            "control_mean": float(control_means.mean()),
            "control_range": [
                float(control_means.min()),
                float(control_means.max()),
            ],
            "target_exceeds_every_control": bool(
                target_mean > float(control_means.max())
            ),
            "add_one_p": exact_add_one_upper_p(target_mean, control_means),
            "paired_target_minus_mean_control": paired_row,
            "paired_lower_above_zero": bool(paired_row["ci"][0] > 0),
        }

    specificity: dict[str, dict[str, Any]] = {
        "Psi00": {
            "family": "balanced_pseudo_phase",
            "add_one_p": float(clean_pseudo["add_one_p_one_sided"]),
            "target_exceeds_every_control": clean_pseudo[
                "target_exceeds_every_control"
            ],
            "paired_target_minus_mean_control": p3_paired[
                "Psi00_pseudo_excess"
            ],
            "paired_lower_above_zero": bool(
                p3_paired["Psi00_pseudo_excess"]["ci"][0] > 0
            ),
        },
        "N3": {
            "families": {
                "p3_phase": single_family(
                    contrasts_ac["N3"], p3_phase_null, "N3_phase_excess"
                ),
                "p3_axis": single_family(
                    contrasts_ac["N3"], p3_axis_null, "N3_axis_excess"
                ),
            }
        },
    }
    specificity["Psi00"]["corrected_pass"] = bool(
        specificity["Psi00"]["add_one_p"] <= 1.0 / (n_pseudo + 1)
        and specificity["Psi00"]["target_exceeds_every_control"]
        and specificity["Psi00"]["paired_lower_above_zero"]
        and clean_pseudo["assignment_audit"]["exactly_balanced"]
        and clean_pseudo["assignment_audit"]["all_assignments_unique"]
    )
    specificity["N3"]["corrected_pass"] = bool(
        all(
            row["add_one_p"] <= 1.0 / 32.0
            and row["target_exceeds_every_control"]
            and row["paired_lower_above_zero"]
            for row in specificity["N3"]["families"].values()
        )
    )
    if mode == "FULL_FACTORIAL":
        p10_phase_null = nulls["N10"]["p10_phase"]
        p10_axis_null = nulls["N10"]["p10_axis"]
        joint_null = nulls["J"]["joint"]
        j_sign = 1.0 if target_means["J"] >= 0 else -1.0
        optional_paired_matrix = np.column_stack(
            [
                contrasts_ac["N10"] - p10_phase_null.mean(axis=0),
                contrasts_ac["N10"] - p10_axis_null.mean(axis=0),
                j_sign * (contrasts_ac["J"] - joint_null.mean(axis=0)),
            ]
        )
        optional_paired = simultaneous_stratified_intervals(
            optional_paired_matrix,
            names=("N10_phase_excess", "N10_axis_excess", "J_signed_excess"),
            strata=strata,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + 5,
        ).by_name(sesoi=0.0)

        def optional_single_family(
            controls: np.ndarray, paired_name: str
        ) -> dict[str, Any]:
            control_means = controls.mean(axis=1)
            paired_row = optional_paired[paired_name]
            return {
                "n_controls": int(len(controls)),
                "control_mean": float(control_means.mean()),
                "control_range": [
                    float(control_means.min()),
                    float(control_means.max()),
                ],
                "target_exceeds_every_control": bool(
                    target_means["N10"] > float(control_means.max())
                ),
                "add_one_p": exact_add_one_upper_p(
                    target_means["N10"], control_means
                ),
                "paired_target_minus_mean_control": paired_row,
                "paired_lower_above_zero": bool(paired_row["ci"][0] > 0),
            }

        specificity["N10"] = {
            "families": {
                "p10_phase": optional_single_family(
                    p10_phase_null, "N10_phase_excess"
                ),
                "p10_axis": optional_single_family(
                    p10_axis_null, "N10_axis_excess"
                ),
            }
        }
        specificity["N10"]["corrected_pass"] = bool(
            all(
                row["add_one_p"] <= 1.0 / 32.0
                and row["target_exceeds_every_control"]
                and row["paired_lower_above_zero"]
                for row in specificity["N10"]["families"].values()
            )
        )
        joint_means = joint_null.mean(axis=1)
        j_paired = optional_paired["J_signed_excess"]
        specificity["J"] = {
            "family": "joint",
            "observed_direction": (
                "positive_sub_additive" if j_sign > 0 else "negative_super_additive"
            ),
            "n_controls": int(len(joint_null)),
            "absolute_control_range": [
                float(np.abs(joint_means).min()),
                float(np.abs(joint_means).max()),
            ],
            "target_absolute_exceeds_every_control": bool(
                abs(target_means["J"]) > float(np.abs(joint_means).max())
            ),
            "absolute_add_one_p": exact_add_one_absolute_p(
                target_means["J"], joint_means
            ),
            "paired_signed_excess": j_paired,
            "paired_signed_lower_above_zero": bool(j_paired["ci"][0] > 0),
        }
        specificity["J"]["corrected_pass"] = bool(
            specificity["J"]["absolute_add_one_p"] <= 1.0 / 32.0
            and specificity["J"]["target_absolute_exceeds_every_control"]
            and specificity["J"]["paired_signed_lower_above_zero"]
        )
    else:
        specificity.update(
            {
                name: {"status": "NOT_RUN_OPTIONAL_P10_INSTRUMENT_UNRESOLVED"}
                for name in ("N10", "J")
            }
        )

    direction_names = ("Dplus", "Dminus", "Psi0", "Psi1", "Psi2")
    direction_matrix = np.column_stack(
        [
            [float(blocks_ac["Y00"][block_id][name]) for block_id in block_ids]
            for name in direction_names
        ]
    )
    optional_n10 = (
        contrasts_ac["N10"] if mode == "FULL_FACTORIAL" else contrasts_ac["N3"]
    )
    optional_j = (
        contrasts_ac["J"] if mode == "FULL_FACTORIAL" else contrasts_ac["N3"]
    )
    optional_phase_null = (
        nulls["N10"]["p10_phase"]
        if mode == "FULL_FACTORIAL"
        else p3_phase_null
    )
    optional_axis_null = (
        nulls["N10"]["p10_axis"]
        if mode == "FULL_FACTORIAL"
        else p3_axis_null
    )
    optional_joint_null = (
        nulls["J"]["joint"] if mode == "FULL_FACTORIAL" else p3_phase_null
    )

    def sufficient(matrix: np.ndarray, *, assignment_exact: bool = True) -> dict[str, Any]:
        return {
            "control_means": [float(value) for value in matrix.mean(axis=1)],
            "block_mean": [float(value) for value in matrix.mean(axis=0)],
            "assignment_exact": bool(assignment_exact),
        }

    assignment = clean_pseudo["assignment_audit"]
    shared_engine = analyze_fixture_arrays(
        {
            "strata": list(strata),
            "values": {
                "clean": world_ac["Y00"],
                "N3": contrasts_ac["N3"],
                "N10": optional_n10,
                "J": optional_j,
            },
            "directions": {
                name: direction_matrix[:, index]
                for index, name in enumerate(direction_names)
            },
            "specificity": {
                "clean": {
                    "pseudo": sufficient(
                        pseudo_matrix,
                        assignment_exact=bool(
                            assignment["exactly_balanced"]
                            and assignment["all_assignments_unique"]
                        ),
                    )
                },
                "N3": {
                    "phase": sufficient(p3_phase_null),
                    "axis": sufficient(p3_axis_null),
                },
                "N10": {
                    "phase": sufficient(optional_phase_null),
                    "axis": sufficient(optional_axis_null),
                },
                "J": {"joint": sufficient(optional_joint_null)},
            },
        },
        target_mode=mode,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    direction_intervals = simultaneous_stratified_intervals(
        direction_matrix,
        names=direction_names,
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed + 6,
    ).by_name(sesoi=0.0)
    directional_pass = bool(
        all(row["ci"][0] > 0 for row in direction_intervals.values())
    )
    mismatch_names = ("Mplus", "Mminus")
    mismatch_matrix = np.column_stack(
        [
            [float(blocks_ac["Y00"][block_id][name]) for block_id in block_ids]
            for name in mismatch_names
        ]
    )
    mismatch_intervals = simultaneous_stratified_intervals(
        mismatch_matrix,
        names=mismatch_names,
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed + 7,
    ).by_name(sesoi=SESOI)
    mismatch_equivalence = bool(
        all(
            row["ci"][0] >= -SESOI and row["ci"][1] <= SESOI
            for row in mismatch_intervals.values()
        )
    )

    resolution_values: dict[str, np.ndarray] = {
        "Psi00": world_ac["Y00"],
        "N3": contrasts_ac["N3"],
    }
    if mode == "FULL_FACTORIAL":
        resolution_values.update(
            {"N10": contrasts_ac["N10"], "J": contrasts_ac["J"]}
        )
    target_resolution = {
        name: {
            "mde80": mde80_from_bootstrap(values, strata),
            "threshold": 0.04 if name in {"Psi00", "N3"} else 0.05,
        }
        for name, values in resolution_values.items()
    }
    for row in target_resolution.values():
        row["pass"] = bool(row["mde80"] <= row["threshold"])

    clean_interval = endpoints["Psi00"]["ci"]
    if (
        clean_interval[0] >= -SESOI
        and clean_interval[1] <= SESOI
        and target_resolution["Psi00"]["pass"]
    ):
        clean_behavior_classification = "BALANCED_PREFERENCE_BOUNDED_NULL"
    elif (
        clean_interval[0] > SESOI
        and directional_pass
        and specificity["Psi00"]["corrected_pass"]
        and target_resolution["Psi00"]["pass"]
    ):
        clean_behavior_classification = "BALANCED_PREFERENCE_MATERIAL"
    else:
        clean_behavior_classification = "BALANCED_PREFERENCE_INCONCLUSIVE"

    sensitivity_names = ["Psi00", "N3"]
    sensitivity_values = [
        world_ac["Y00"] - world_at["Y00"],
        contrasts_ac["N3"] - contrasts_at["N3"],
    ]
    if mode == "FULL_FACTORIAL":
        sensitivity_names.extend(("N10", "J"))
        sensitivity_values.extend(
            (
                contrasts_ac["N10"] - contrasts_at["N10"],
                contrasts_ac["J"] - contrasts_at["J"],
            )
        )
    sensitivity = simultaneous_stratified_intervals(
        np.column_stack(sensitivity_values),
        names=sensitivity_names,
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed + 3,
    ).by_name(sesoi=SESOI)
    sensitivity_pass = {
        name: bool(row["ci"][0] >= -SESOI and row["ci"][1] <= SESOI)
        for name, row in sensitivity.items()
    }

    shared_details = shared_engine["engine_details"]
    relevant_endpoints = (
        ("Psi00", "N3")
        if mode == "P3_ONLY"
        else ("Psi00", "N3", "N10", "J")
    )
    if any(
        endpoints[name] != shared_details["endpoints"][name]
        for name in relevant_endpoints
    ) or any(
        target_resolution[name] != shared_details["target_resolution"][name]
        for name in relevant_endpoints
    ) or direction_intervals != shared_details["direction_family"]:
        raise RuntimeError("shared target/fixture statistical engine diverged")
    if (
        clean_behavior_classification
        != shared_details["clean_behavior_classification"]
        or specificity["Psi00"]["corrected_pass"]
        != shared_details["specificity"]["Psi00"]["corrected_pass"]
        or specificity["N3"]["corrected_pass"]
        != shared_details["specificity"]["N3"]["corrected_pass"]
        or (
            mode == "FULL_FACTORIAL"
            and (
                specificity["N10"]["corrected_pass"]
                != shared_details["specificity"]["N10"]["corrected_pass"]
                or specificity["J"]["corrected_pass"]
                != shared_details["specificity"]["J"]["corrected_pass"]
            )
        )
    ):
        raise RuntimeError("shared target/fixture decision inputs diverged")
    endpoints.update(
        {name: shared_details["endpoints"][name] for name in relevant_endpoints}
    )
    direction_intervals = shared_details["direction_family"]
    directional_pass = bool(shared_details["directional_pass"])
    target_resolution.update(
        {
            name: shared_details["target_resolution"][name]
            for name in relevant_endpoints
        }
    )
    clean_behavior_classification = str(
        shared_details["clean_behavior_classification"]
    )
    phase2_decision = dict(shared_engine["decision"])
    p3_statistical_criteria = bool(
        phase2_decision["p3_phase3_statistical_criteria"]
    )
    joint_statistical_criteria = bool(
        phase2_decision["joint_phase3_statistical_criteria"]
    )
    authorization = {
        "p3_phase3_statistical_criteria": p3_statistical_criteria,
        "joint_phase3_statistical_criteria": joint_statistical_criteria,
        "authorize_p3_phase3": p3_statistical_criteria,
        "authorize_joint_phase3": joint_statistical_criteria,
        "requires_independent_result_audit": True,
        "authorization_is_provisional_until_audit_pass": True,
    }

    null_summaries = {
        endpoint: {
            family: {
                "n_controls": int(values.shape[0]),
                "n_blocks": int(values.shape[1]),
                "control_means": [
                    float(value) for value in values.mean(axis=1)
                ],
                "matrix_sha256": sha256(
                    np.ascontiguousarray(values).tobytes()
                ).hexdigest(),
            }
            for family, values in families.items()
        }
        for endpoint, families in nulls.items()
    }

    result = {
        "schema": ANALYSIS_SCHEMA,
        "stage": "target",
        "status": "COMPLETE",
        "target_mode": mode,
        "target_seal_sha256": str(target["target_seal_sha256"]),
        "n_blocks": len(block_ids),
        "n_cells": int(target["n_cells"]),
        "sesoi": SESOI,
        "analysis_parameters": {
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
            "n_pseudo": int(n_pseudo),
            "pseudo_seed": int(pseudo_seed),
        },
        "phase2_analysis_engine": "analyze_carrier_factorial.analyze_fixture_arrays",
        "phase2_decision_engine": "pbc_carrier_factorial.carrier_phase2_decision",
        "inference_families": family_records,
        "worlds": {
            world: {
                "estimate": float(values.mean()),
                "operator": "AG_TO_AC",
            }
            for world, values in world_ac.items()
        },
        "endpoints": endpoints,
        "clean_behavior_classification": clean_behavior_classification,
        "clean_directional_family": {
            "members": list(direction_names),
            "endpoints": direction_intervals,
            "all_simultaneous_lower_bounds_above_zero": directional_pass,
        },
        "mismatch_equivalence_family": {
            "members": list(mismatch_names),
            "endpoints": mismatch_intervals,
            "simultaneous_equivalence_within_sesoi": mismatch_equivalence,
            "governs_only_wording_not_p3_authorization": True,
        },
        "target_resolution": target_resolution,
        "structured_null_specificity": specificity,
        "structured_null_summaries": null_summaries,
        "clean_pseudo_phase": clean_pseudo,
        "operator_sensitivity_ac_minus_at": {
            "endpoints": sensitivity,
            "equivalence_pass": sensitivity_pass,
            "governs_target_authorization": False,
        },
        "authorization": authorization,
        "verdict": _verdict(
            clean_behavior_classification=clean_behavior_classification,
            endpoints=endpoints,
            specificity=specificity,
            target_mode=mode,
            target_resolution=target_resolution,
        ),
        "per_block": [
            {
                "block_id": block_id,
                "stratum": strata[index],
                **{
                    world: float(world_ac[world][index]) for world in worlds
                },
                **{
                    name: (
                        float(values[index]) if values is not None else None
                    )
                    for name, values in contrasts_ac.items()
                },
            }
            for index, block_id in enumerate(block_ids)
        ],
    }
    # Recompute the scalar map through the public helper as an arithmetic memorial.
    aggregate_worlds = {
        world: float(values.mean()) for world, values in world_ac.items()
    }
    recomputed_contrasts = factorial_contrasts(aggregate_worlds)
    recorded_contrasts = {
        name: (float(values.mean()) if values is not None else None)
        for name, values in {
            "N3": contrasts_ac["N3"],
            "N10": contrasts_ac.get("N10"),
            "Nboth": contrasts_ac.get("Nboth"),
            "J": contrasts_ac.get("J"),
        }.items()
    }
    if any(
        (recomputed_contrasts[name] is None) != (recorded_contrasts[name] is None)
        or (
            recomputed_contrasts[name] is not None
            and not math.isclose(
                float(recomputed_contrasts[name]),
                float(recorded_contrasts[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        for name in recorded_contrasts
    ):
        raise RuntimeError("factorial arithmetic memorial failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--bootstrap", type=int, default=CONFIRMATORY_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=CONFIRMATORY_BOOTSTRAP_SEED)
    parser.add_argument("--pseudo", type=int, default=CONFIRMATORY_PSEUDO_CONTROLS)
    parser.add_argument("--pseudo-seed", type=int, default=CONFIRMATORY_PSEUDO_SEED)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output, manifest_path = output_paths(args.out)
    source_path = Path(args.input).expanduser().resolve()
    target = json.loads(source_path.read_text())
    result = analyze_factorial_rows(
        target,
        n_bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        n_pseudo=args.pseudo,
        pseudo_seed=args.pseudo_seed,
    )
    result["target_sha256"] = sha256_file(source_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-carrier-factorial-analysis-v1",
        sys.argv,
        repo=REPO,
        status=result["verdict"],
        config={
            "stage": "target",
            "bootstrap": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "pseudo": args.pseudo,
            "pseudo_seed": args.pseudo_seed,
            "sesoi": SESOI,
            "target_mode": result["target_mode"],
            "families": result["inference_families"],
        },
        seeds={
            "bootstrap": args.bootstrap_seed,
            "pseudo_phase": args.pseudo_seed,
        },
    )
    for path in (
        source_path,
        REPO / "src/gfmcd/splicing/pbc_carrier_factorial.py",
        Path(__file__).resolve(),
    ):
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "authorization": result["authorization"],
                "endpoints": result["endpoints"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
