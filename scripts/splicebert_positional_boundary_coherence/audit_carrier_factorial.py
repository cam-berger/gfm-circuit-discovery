#!/usr/bin/env python
"""Independent Phase-II carrier-factorial arithmetic and provenance audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_carrier_factorial import (  # noqa: E402
    MIN_PSEUDO_PHASE_CONTROLS,
    MIN_STRUCTURED_CONTROLS,
    SESOI,
    balanced_pseudo_phase_assignment_audit,
    balanced_pseudo_phase_matrix,
    exact_add_one_absolute_p,
    exact_add_one_upper_p,
    mde80_from_bootstrap,
    percentile_stratified_interval,
    simultaneous_stratified_intervals,
    verify_target_seal,
)
from run_carrier_factorial_preflight import PREFLIGHT_SCHEMA  # noqa: E402


AUDIT_SCHEMA = "gfmcd.pbc.carrier_factorial_audit.v1"
SEAL_SCHEMA = "gfmcd.pbc.carrier_factorial_target_seal.v1"
TARGET_SCHEMA = "gfmcd.pbc.carrier_factorial_target.v1"
ANALYSIS_SCHEMA = "gfmcd.pbc.carrier_factorial_analysis.v1"
CONFIRMATORY_BOOTSTRAP_DRAWS = 20_000
CONFIRMATORY_BOOTSTRAP_SEED = 2026081001
CONFIRMATORY_PSEUDO_CONTROLS = 999
CONFIRMATORY_PSEUDO_SEED = 2026081002


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    return output, manifest


def _file_record_valid(record: object) -> bool:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        return False
    path = Path(str(record["path"])).expanduser().resolve()
    return bool(
        path.is_file()
        and type(record.get("size_bytes")) is int
        and path.stat().st_size == int(record["size_bytes"])
        and str(record.get("sha256", "")) == sha256_file(path)
    )


def _manifest_checks(
    manifest: Mapping[str, Any], artifact_path: Path
) -> dict[str, bool]:
    outputs = manifest.get("outputs", [])
    inputs = manifest.get("inputs", [])
    resolved = artifact_path.resolve()
    return {
        "artifact_output_bound": bool(
            isinstance(outputs, list)
            and any(
                _file_record_valid(record)
                and Path(str(record["path"])).expanduser().resolve() == resolved
                for record in outputs
                if isinstance(record, Mapping)
            )
        ),
        "all_manifest_inputs_valid": bool(
            isinstance(inputs, list)
            and inputs
            and all(_file_record_valid(record) for record in inputs)
        ),
        "git_revision_recorded": isinstance(manifest.get("git_commit"), str)
        and len(str(manifest["git_commit"])) == 40,
        "git_state_recorded": type(manifest.get("git_dirty")) is bool,
    }


def _manual_block_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    if len(rows) != 9:
        raise ValueError("manual block audit requires nine rows")
    values: dict[tuple[int, int], float] = {}
    for row in rows:
        pair = (int(row["donor_phase"]), int(row["acceptor_phase"]))
        if pair in values or pair[0] not in range(3) or pair[1] not in range(3):
            raise ValueError("invalid or duplicate phase cell")
        values[pair] = float(row[key])
    if len(values) != 9:
        raise ValueError("phase lattice incomplete")
    m0 = float(np.mean([values[(phase, phase)] for phase in range(3)]))
    mplus = float(
        np.mean([values[(phase, (phase + 1) % 3)] for phase in range(3)])
    )
    mminus = float(
        np.mean([values[(phase, (phase - 1) % 3)] for phase in range(3)])
    )
    result = {
        "M0": m0,
        "Mplus": mplus,
        "Mminus": mminus,
        "Psi": m0 - 0.5 * (mplus + mminus),
        "Dplus": m0 - mplus,
        "Dminus": m0 - mminus,
    }
    for phase in range(3):
        result[f"Psi{phase}"] = values[(phase, phase)] - 0.5 * (
            values[(phase, (phase + 1) % 3)]
            + values[(phase, (phase - 1) % 3)]
        )
    return result


def _row_contract(target: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    specs = target.get("condition_specs")
    evidence = target.get("compact_evidence")
    if (
        "rows" in target
        or not isinstance(specs, Mapping)
        or not isinstance(evidence, Mapping)
        or evidence.get("schema")
        != "gfmcd.pbc.carrier_factorial_compact_blocks.v1"
    ):
        return {"passed": False, "reason": "compact endpoint evidence absent"}, {}
    blocks = evidence.get("blocks")
    block_order = evidence.get("block_order")
    endpoints = evidence.get("condition_endpoints")
    if (
        not isinstance(blocks, list)
        or not isinstance(block_order, list)
        or not isinstance(endpoints, Mapping)
        or len(blocks) != int(target.get("n_blocks", -1))
        or [str(block.get("block_id")) for block in blocks] != block_order
        or len(set(block_order)) != len(block_order)
        or set(endpoints) != set(specs) - {"Y00"}
    ):
        return {"passed": False, "reason": "compact support malformed"}, {}
    finite = True
    lattice_valid = True
    strata_valid = True
    cell_ids: set[str] = set()
    y00_ac: list[float] = []
    for block in blocks:
        try:
            rows = block["y00_cells"]
            summary = _manual_block_summary(rows, "coupling_ac")
            _manual_block_summary(rows, "coupling_at")
            y00_ac.append(float(summary["Psi"]))
            finite &= all(
                math.isfinite(float(row[key]))
                for row in rows
                for key in ("coupling_ac", "coupling_at")
            )
            for row in rows:
                cell_id = str(row["cell_id"])
                if cell_id in cell_ids:
                    lattice_valid = False
                cell_ids.add(cell_id)
            strata_valid &= bool(str(block["chrom"]) and str(block["strand"]) in {"+", "-"})
        except (KeyError, TypeError, ValueError):
            lattice_valid = False
    family_counts = Counter(
        str(spec["family"])
        for spec in specs.values()
        if str(spec["family"]) != "target"
    )
    required_families = (
        {"p3_phase", "p3_axis"}
        if target.get("target_mode") == "P3_ONLY"
        else {
            "p3_phase",
            "p3_axis",
            "p10_phase",
            "p10_axis",
            "joint_p3",
            "joint_p10",
            "joint",
        }
    )
    n_controls = int(target.get("n_controls_per_family", -1))
    family_sizes_exact = set(family_counts) == required_families and all(
        family_counts[family] == n_controls for family in required_families
    )
    expected_worlds = (
        {"Y00", "Y10"}
        if target.get("target_mode") == "P3_ONLY"
        else {"Y00", "Y10", "Y01", "Y11"}
    )
    target_worlds_exact = {
        condition
        for condition, spec in specs.items()
        if str(spec["family"]) == "target"
    } == expected_worlds
    endpoint_support = True
    target_at_support = True
    world_values: dict[str, np.ndarray] = {"Y00": np.asarray(y00_ac, dtype=float)}
    for condition, spec in specs.items():
        if condition == "Y00":
            continue
        row = endpoints.get(condition)
        ac = row.get("psi_ac") if isinstance(row, Mapping) else None
        at = row.get("psi_at") if isinstance(row, Mapping) else None
        endpoint_support &= bool(
            isinstance(ac, list)
            and len(ac) == len(block_order)
            and np.isfinite(np.asarray(ac, dtype=float)).all()
        )
        is_target = str(spec.get("family")) == "target"
        target_at_support &= bool(
            (
                is_target
                and isinstance(at, list)
                and len(at) == len(block_order)
                and np.isfinite(np.asarray(at, dtype=float)).all()
            )
            or (not is_target and at is None)
        )
        if is_target and isinstance(ac, list):
            world_values[str(condition)] = np.asarray(ac, dtype=float)
    raw_row_count = int(evidence.get("raw_row_count", -1))
    expected_raw_rows = int(target.get("n_cells", -1)) * len(specs)
    stream_hash = str(evidence.get("raw_endpoint_stream_sha256", ""))
    raw_shards = evidence.get("raw_shards")
    provenance_valid = bool(
        raw_row_count == expected_raw_rows
        and len(stream_hash) == 64
        and isinstance(raw_shards, list)
        and evidence.get("raw_identity_support_verified") is True
        and float(evidence.get("raw_coupling_arithmetic_max_abs", math.inf))
        <= 1e-10
        and all(
            isinstance(record, Mapping)
            and len(str(record.get("sha256", ""))) == 64
            and type(record.get("size_bytes")) is int
            and isinstance(record.get("path"), str)
            for record in raw_shards
        )
    )
    passed = bool(
        finite
        and target.get("stage") == "target"
        and int(target.get("n_cells", -1)) == len(block_order) * 9
        and len(cell_ids) == len(block_order) * 9
        and family_sizes_exact
        and target_worlds_exact
        and lattice_valid
        and strata_valid
        and endpoint_support
        and target_at_support
        and provenance_valid
    )
    return {
        "passed": passed,
        "representation": "smallest_per_block_endpoint_table_v1",
        "n_blocks": len(block_order),
        "n_conditions": len(specs),
        "n_y00_cells": len(cell_ids),
        "raw_rows_absent": "rows" not in target,
        "condition_endpoint_support_complete": endpoint_support,
        "target_at_support_complete": target_at_support,
        "all_finite": bool(finite),
        "family_counts": dict(sorted(family_counts.items())),
        "family_sizes_exact": family_sizes_exact,
        "target_worlds_exact": target_worlds_exact,
        "y00_lattices_complete": lattice_valid,
        "block_strata_valid": strata_valid,
        "raw_provenance_valid": provenance_valid,
        "raw_row_count": raw_row_count,
        "raw_endpoint_stream_sha256": stream_hash,
    }, world_values


def _independent_decision_checks(
    analysis: Mapping[str, Any], world_values: Mapping[str, np.ndarray]
) -> dict[str, bool]:
    endpoints = analysis.get("endpoints", {})
    specificity = analysis.get("structured_null_specificity", {})
    resolution = analysis.get("target_resolution", {})
    directions = analysis.get("clean_directional_family", {})
    clean_interval = endpoints.get("Psi00", {}).get("ci", [math.inf, -math.inf])
    clean_bounded = bool(
        clean_interval[0] >= -SESOI
        and clean_interval[1] <= SESOI
        and resolution.get("Psi00", {}).get("pass") is True
    )
    clean_material = bool(
        clean_interval[0] > SESOI
        and directions.get("all_simultaneous_lower_bounds_above_zero") is True
        and specificity.get("Psi00", {}).get("corrected_pass") is True
        and resolution.get("Psi00", {}).get("pass") is True
    )
    expected_clean = (
        "BALANCED_PREFERENCE_BOUNDED_NULL"
        if clean_bounded
        else (
            "BALANCED_PREFERENCE_MATERIAL"
            if clean_material
            else "BALANCED_PREFERENCE_INCONCLUSIVE"
        )
    )
    p3 = bool(
        expected_clean == "BALANCED_PREFERENCE_MATERIAL"
        and endpoints.get("N3", {}).get("classification") == "MATERIAL_SUPPORTIVE"
        and specificity.get("N3", {}).get("corrected_pass") is True
        and resolution.get("N3", {}).get("pass") is True
    )
    full = analysis.get("target_mode") == "FULL_FACTORIAL"
    joint = bool(
        full
        and p3
        and endpoints.get("N10", {}).get("classification") == "MATERIAL_SUPPORTIVE"
        and str(endpoints.get("J", {}).get("classification", "")).startswith(
            "MATERIAL_"
        )
        and specificity.get("N10", {}).get("corrected_pass") is True
        and specificity.get("J", {}).get("corrected_pass") is True
        and resolution.get("N10", {}).get("pass") is True
        and resolution.get("J", {}).get("pass") is True
    )
    worlds_match = True
    recorded_worlds = analysis.get("worlds", {})
    for world, values in world_values.items():
        worlds_match &= abs(
            float(recorded_worlds.get(world, {}).get("estimate", math.inf))
            - float(values.mean())
        ) <= 1e-12
    contrasts_match = True
    if {"Y00", "Y10"}.issubset(world_values):
        n3 = world_values["Y00"] - world_values["Y10"]
        contrasts_match &= abs(
            float(endpoints.get("N3", {}).get("estimate", math.inf))
            - float(n3.mean())
        ) <= 1e-12
    if {"Y00", "Y10", "Y01", "Y11"}.issubset(world_values):
        n10 = world_values["Y00"] - world_values["Y01"]
        nboth = world_values["Y00"] - world_values["Y11"]
        interaction = (
            world_values["Y11"]
            - world_values["Y10"]
            - world_values["Y01"]
            + world_values["Y00"]
        )
        contrasts_match &= all(
            abs(float(endpoints[name]["estimate"]) - float(values.mean())) <= 1e-12
            for name, values in (("N10", n10), ("Nboth", nboth), ("J", interaction))
        )
        observed_direction = specificity.get("J", {}).get("observed_direction")
        expected_direction = (
            "positive_sub_additive"
            if float(interaction.mean()) >= 0
            else "negative_super_additive"
        )
        sign_convention = observed_direction == expected_direction
    else:
        sign_convention = True
    authorization = analysis.get("authorization", {})
    pseudo = analysis.get("clean_pseudo_phase", {})
    pseudo_audit = pseudo.get("assignment_audit", {})
    null_specificity = analysis.get("structured_null_specificity", {})
    def integer_grid(value: Any, denominator: int) -> bool:
        try:
            scaled = float(value) * denominator
        except (TypeError, ValueError):
            return False
        return bool(math.isfinite(scaled) and abs(scaled - round(scaled)) <= 1e-10)

    exact_denominators = bool(
        pseudo.get("n_controls") == MIN_PSEUDO_PHASE_CONTROLS
        and integer_grid(
            null_specificity.get("Psi00", {}).get("add_one_p"),
            MIN_PSEUDO_PHASE_CONTROLS + 1,
        )
    )
    for endpoint in ("N3", "N10"):
        row = null_specificity.get(endpoint, {})
        if "families" not in row:
            continue
        for family in row["families"].values():
            exact_denominators &= family.get("n_controls") == MIN_STRUCTURED_CONTROLS
            exact_denominators &= integer_grid(family.get("add_one_p"), 32)
    if full:
        j_row = null_specificity.get("J", {})
        exact_denominators &= j_row.get("n_controls") == MIN_STRUCTURED_CONTROLS
        exact_denominators &= integer_grid(j_row.get("absolute_add_one_p"), 32)
    return {
        "world_estimates_independently_recomputed": bool(worlds_match),
        "factorial_contrasts_independently_recomputed": bool(contrasts_match),
        "clean_classification_independently_recomputed": analysis.get(
            "clean_behavior_classification"
        )
        == expected_clean,
        "p3_authorization_independently_recomputed": authorization.get(
            "authorize_p3_phase3"
        )
        is p3,
        "joint_authorization_independently_recomputed": authorization.get(
            "authorize_joint_phase3"
        )
        is joint,
        "joint_sign_convention": sign_convention,
        "exact_add_one_denominators": exact_denominators,
        "pseudo_phase_exact_balance": pseudo_audit.get("exactly_balanced") is True,
        "pseudo_phase_assignments_unique": pseudo_audit.get(
            "all_assignments_unique"
        )
        is True,
        "clean_directional_family_complete": set(
            directions.get("members", [])
        )
        == {"Dplus", "Dminus", "Psi0", "Psi1", "Psi2"},
        "mismatch_family_complete": set(
            analysis.get("mismatch_equivalence_family", {}).get("members", [])
        )
        == {"Mplus", "Mminus"},
    }


def _audit_group_conditions(
    target: Mapping[str, Any], key: str
) -> dict[str, dict[str, dict[str, Any]]]:
    specs = target.get("condition_specs")
    evidence = target.get("compact_evidence")
    if not isinstance(specs, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("compact target condition evidence is absent")
    blocks = evidence.get("blocks")
    order = evidence.get("block_order")
    endpoints = evidence.get("condition_endpoints")
    if (
        not isinstance(blocks, list)
        or not isinstance(order, list)
        or not isinstance(endpoints, Mapping)
        or [str(block.get("block_id")) for block in blocks] != order
    ):
        raise ValueError("compact target support is malformed")
    output: dict[str, dict[str, dict[str, Any]]] = {"Y00": {}}
    for block in blocks:
        block_id = str(block["block_id"])
        rows = block.get("y00_cells")
        if not isinstance(rows, list):
            raise ValueError("compact Y00 cells absent")
        output["Y00"][block_id] = {
            **_manual_block_summary(rows, key),
            "stratum": f"{block['chrom']}|{block['strand']}",
            "rows": rows,
        }
    endpoint_key = "psi_ac" if key == "coupling_ac" else "psi_at"
    for condition, spec in specs.items():
        if condition == "Y00":
            continue
        if key == "coupling_at" and str(spec.get("family")) != "target":
            continue
        row = endpoints.get(condition)
        values = row.get(endpoint_key) if isinstance(row, Mapping) else None
        if not isinstance(values, list) or len(values) != len(order):
            raise ValueError(f"compact endpoint missing: {condition}/{endpoint_key}")
        output[str(condition)] = {
            str(block_id): {
                "Psi": float(values[index]),
                "stratum": f"{blocks[index]['chrom']}|{blocks[index]['strand']}",
                "rows": [],
            }
            for index, block_id in enumerate(order)
        }
    return output


def _audit_world_arrays(
    blocks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    block_ids: Sequence[str],
    worlds: Sequence[str],
) -> dict[str, np.ndarray]:
    return {
        world: np.asarray(
            [float(blocks[world][block_id]["Psi"]) for block_id in block_ids]
        )
        for world in worlds
    }


def _audit_contrasts(worlds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {"N3": worlds["Y00"] - worlds["Y10"]}
    if {"Y01", "Y11"}.issubset(worlds):
        output.update(
            {
                "N10": worlds["Y00"] - worlds["Y01"],
                "Nboth": worlds["Y00"] - worlds["Y11"],
                "J": worlds["Y11"]
                - worlds["Y10"]
                - worlds["Y01"]
                + worlds["Y00"],
            }
        )
    return output


def _audit_structured_nulls(
    target: Mapping[str, Any],
    blocks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    block_ids: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    y00 = np.asarray([float(blocks["Y00"][block]["Psi"]) for block in block_ids])
    singles: dict[str, dict[int, np.ndarray]] = {
        family: {}
        for family in ("p3_phase", "p3_axis", "p10_phase", "p10_axis")
    }
    joint_p3: dict[int, np.ndarray] = {}
    joint_p10: dict[int, np.ndarray] = {}
    joint: dict[int, np.ndarray] = {}
    for condition, spec in target["condition_specs"].items():
        family = str(spec["family"])
        if family == "target":
            continue
        index = int(spec["control_id"])
        values = np.asarray(
            [float(blocks[condition][block]["Psi"]) for block in block_ids]
        )
        if family in singles:
            singles[family][index] = y00 - values
        elif family == "joint_p3":
            joint_p3[index] = values
        elif family == "joint_p10":
            joint_p10[index] = values
        elif family == "joint":
            joint[index] = values
        else:
            raise ValueError(f"unknown null family {family}")
    output = {
        "N3": {
            family: np.stack([values for _, values in sorted(singles[family].items())])
            for family in ("p3_phase", "p3_axis")
        }
    }
    if target["target_mode"] == "FULL_FACTORIAL":
        if set(joint_p3) != set(joint_p10) or set(joint_p3) != set(joint):
            raise ValueError("joint null IDs differ")
        output["N10"] = {
            family: np.stack([values for _, values in sorted(singles[family].items())])
            for family in ("p10_phase", "p10_axis")
        }
        output["J"] = {
            "joint": np.stack(
                [
                    joint[index] - joint_p3[index] - joint_p10[index] + y00
                    for index in sorted(joint)
                ]
            )
        }
    return output


def _audit_decision(
    clean: str,
    endpoints: Mapping[str, Mapping[str, Any]],
    specificity: Mapping[str, Mapping[str, Any]],
    resolution: Mapping[str, Mapping[str, Any]],
    mode: str,
    direction: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    directional_scoped = bool(
        float(endpoints["Psi00"]["ci"][0]) > 0
        and float(endpoints["N3"]["ci"][0]) > 0
        and all(
            float(direction[name]["estimate"]) > 0
            for name in ("Dplus", "Dminus", "Psi0", "Psi1", "Psi2")
        )
        and specificity["Psi00"]["corrected_pass"] is True
        and specificity["N3"]["corrected_pass"] is True
        and resolution["Psi00"]["pass"] is True
        and float(resolution["Psi00"]["mde80"]) <= 0.04
        and resolution["N3"]["pass"] is True
        and float(resolution["N3"]["mde80"]) <= 0.04
    )
    n3 = str(endpoints["N3"]["classification"])
    if (
        clean == "BALANCED_PREFERENCE_MATERIAL"
        and n3 == "MATERIAL_SUPPORTIVE"
        and specificity["N3"]["corrected_pass"]
        and resolution["N3"]["pass"]
    ):
        p3 = "P3_NECESSITY_SUPPORTED"
    elif directional_scoped:
        p3 = "P3_DIRECTIONAL_SCOPED_SUPPORT"
    elif n3 == "BOUNDED_NULL" and resolution["N3"]["pass"]:
        p3 = "P3_ATTENUATION_BOUNDED_NULL"
    elif n3 == "MATERIAL_SUPPRESSIVE":
        p3 = "P3_SUPPRESSIVE"
    else:
        p3 = "P3_INCONCLUSIVE"
    strong = p3 == "P3_NECESSITY_SUPPORTED"
    if strong and mode == "P3_ONLY":
        verdict = "P3_ONLY_CLOSURE_PATH"
    elif strong and (
        endpoints["N10"]["classification"] == "MATERIAL_SUPPORTIVE"
        and specificity["N10"]["corrected_pass"]
        and str(endpoints["J"]["classification"]).startswith("MATERIAL_")
        and specificity["J"]["corrected_pass"]
        and resolution["N10"]["pass"]
        and resolution["J"]["pass"]
    ):
        verdict = "DUAL_SCALE_INTERACTION_SUPPORTED"
    elif strong and endpoints["N10"]["classification"] == "BOUNDED_NULL":
        verdict = "P3_ONLY_CLOSURE_PATH"
    elif strong:
        verdict = "P3_NECESSITY_SUPPORTED"
    elif directional_scoped:
        verdict = "P3_DIRECTIONAL_SCOPED_SUPPORT"
    elif clean == "BALANCED_PREFERENCE_BOUNDED_NULL":
        verdict = clean
    elif clean != "BALANCED_PREFERENCE_MATERIAL":
        verdict = "BALANCED_PREFERENCE_INCONCLUSIVE"
    else:
        verdict = p3 if p3 != "P3_INCONCLUSIVE" else "INCONCLUSIVE"
    support_tiers = []
    if directional_scoped:
        support_tiers.append("P3_DIRECTIONAL_SCOPED_SUPPORT")
    if strong:
        support_tiers.append("P3_NECESSITY_SUPPORTED")
    return {
        "verdict": verdict,
        "directional_scoped_support": directional_scoped,
        "support_tiers": support_tiers,
    }


def _nested_equal(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _nested_equal(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _nested_equal(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
        except (TypeError, ValueError):
            return False
    return left == right


def _independent_statistical_recompute(
    target: Mapping[str, Any], analysis: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute all inferential records without importing the analysis module."""

    parameters = analysis.get("analysis_parameters")
    expected_parameters = {
        "n_bootstrap": CONFIRMATORY_BOOTSTRAP_DRAWS,
        "bootstrap_seed": CONFIRMATORY_BOOTSTRAP_SEED,
        "n_pseudo": CONFIRMATORY_PSEUDO_CONTROLS,
        "pseudo_seed": CONFIRMATORY_PSEUDO_SEED,
    }
    if parameters != expected_parameters:
        raise ValueError("analysis changed frozen inference parameters")
    mode = str(target["target_mode"])
    worlds = ("Y00", "Y10") if mode == "P3_ONLY" else ("Y00", "Y10", "Y01", "Y11")
    blocks_ac = _audit_group_conditions(target, "coupling_ac")
    blocks_at = _audit_group_conditions(target, "coupling_at")
    block_ids = tuple(sorted(blocks_ac["Y00"]))
    strata = tuple(blocks_ac["Y00"][block]["stratum"] for block in block_ids)
    world_ac = _audit_world_arrays(blocks_ac, block_ids, worlds)
    world_at = _audit_world_arrays(blocks_at, block_ids, worlds)
    contrasts = _audit_contrasts(world_ac)
    contrasts_at = _audit_contrasts(world_at)

    primary = simultaneous_stratified_intervals(
        np.column_stack([world_ac["Y00"], contrasts["N3"]]),
        names=("Psi00", "N3"),
        strata=strata,
        n_bootstrap=CONFIRMATORY_BOOTSTRAP_DRAWS,
        seed=CONFIRMATORY_BOOTSTRAP_SEED,
    )
    endpoints = primary.by_name(sesoi=SESOI)
    inference_families: dict[str, Any] = {
        "p3_primary": {
            "members": ["Psi00", "N3"],
            "critical_value": primary.critical_value,
            "n_bootstrap": primary.n_bootstrap,
        }
    }
    if mode == "FULL_FACTORIAL":
        optional = simultaneous_stratified_intervals(
            np.column_stack([contrasts["N10"], contrasts["J"]]),
            names=("N10", "J"),
            strata=strata,
            n_bootstrap=CONFIRMATORY_BOOTSTRAP_DRAWS,
            seed=CONFIRMATORY_BOOTSTRAP_SEED + 1,
        )
        endpoints.update(optional.by_name(sesoi=SESOI))
        inference_families["optional_p10"] = {
            "members": ["N10", "J"],
            "critical_value": optional.critical_value,
            "n_bootstrap": optional.n_bootstrap,
            "does_not_tax_p3_family": True,
        }
        nboth = percentile_stratified_interval(
            contrasts["Nboth"],
            strata,
            n_bootstrap=CONFIRMATORY_BOOTSTRAP_DRAWS,
            seed=CONFIRMATORY_BOOTSTRAP_SEED + 2,
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

    y00_rows = [blocks_ac["Y00"][block]["rows"] for block in block_ids]
    pseudo_matrix = balanced_pseudo_phase_matrix(
        y00_rows,
        n_controls=CONFIRMATORY_PSEUDO_CONTROLS,
        seed=CONFIRMATORY_PSEUDO_SEED,
    )
    pseudo_controls = pseudo_matrix.mean(axis=1)
    clean_target = float(world_ac["Y00"].mean())
    clean_pseudo = {
        "target": clean_target,
        "n_controls": len(pseudo_controls),
        "control_mean": float(pseudo_controls.mean()),
        "control_max": float(pseudo_controls.max()),
        "target_exceeds_every_control": bool(clean_target > pseudo_controls.max()),
        "add_one_p_one_sided": float(
            (1 + np.sum(pseudo_controls >= clean_target))
            / (len(pseudo_controls) + 1)
        ),
        "assignment_audit": balanced_pseudo_phase_assignment_audit(
            len(block_ids),
            n_controls=CONFIRMATORY_PSEUDO_CONTROLS,
            seed=CONFIRMATORY_PSEUDO_SEED,
        ),
    }
    nulls = _audit_structured_nulls(target, blocks_ac, block_ids)
    p3_phase = nulls["N3"]["p3_phase"]
    p3_axis = nulls["N3"]["p3_axis"]
    paired_primary = simultaneous_stratified_intervals(
        np.column_stack(
            [
                world_ac["Y00"] - pseudo_matrix.mean(axis=0),
                contrasts["N3"] - p3_phase.mean(axis=0),
                contrasts["N3"] - p3_axis.mean(axis=0),
            ]
        ),
        names=("Psi00_pseudo_excess", "N3_phase_excess", "N3_axis_excess"),
        strata=strata,
        n_bootstrap=CONFIRMATORY_BOOTSTRAP_DRAWS,
        seed=CONFIRMATORY_BOOTSTRAP_SEED + 4,
    ).by_name(sesoi=0.0)

    def family_record(values: np.ndarray, endpoint: np.ndarray, paired: str) -> dict[str, Any]:
        means = values.mean(axis=1)
        paired_row = paired_primary[paired]
        return {
            "n_controls": int(len(values)),
            "control_mean": float(means.mean()),
            "control_range": [float(means.min()), float(means.max())],
            "target_exceeds_every_control": bool(float(endpoint.mean()) > float(means.max())),
            "add_one_p": exact_add_one_upper_p(float(endpoint.mean()), means),
            "paired_target_minus_mean_control": paired_row,
            "paired_lower_above_zero": bool(paired_row["ci"][0] > 0),
        }

    specificity: dict[str, Any] = {
        "Psi00": {
            "family": "balanced_pseudo_phase",
            "add_one_p": clean_pseudo["add_one_p_one_sided"],
            "target_exceeds_every_control": clean_pseudo["target_exceeds_every_control"],
            "paired_target_minus_mean_control": paired_primary["Psi00_pseudo_excess"],
            "paired_lower_above_zero": bool(
                paired_primary["Psi00_pseudo_excess"]["ci"][0] > 0
            ),
        },
        "N3": {
            "families": {
                "p3_phase": family_record(p3_phase, contrasts["N3"], "N3_phase_excess"),
                "p3_axis": family_record(p3_axis, contrasts["N3"], "N3_axis_excess"),
            }
        },
    }
    specificity["Psi00"]["corrected_pass"] = bool(
        specificity["Psi00"]["add_one_p"] <= 1 / 1000
        and specificity["Psi00"]["target_exceeds_every_control"]
        and specificity["Psi00"]["paired_lower_above_zero"]
        and clean_pseudo["assignment_audit"]["exactly_balanced"]
        and clean_pseudo["assignment_audit"]["all_assignments_unique"]
    )
    specificity["N3"]["corrected_pass"] = bool(
        all(
            row["add_one_p"] <= 1 / 32
            and row["target_exceeds_every_control"]
            and row["paired_lower_above_zero"]
            for row in specificity["N3"]["families"].values()
        )
    )
    if mode == "FULL_FACTORIAL":
        p10_phase = nulls["N10"]["p10_phase"]
        p10_axis = nulls["N10"]["p10_axis"]
        joint = nulls["J"]["joint"]
        sign = 1.0 if float(contrasts["J"].mean()) >= 0 else -1.0
        optional_paired = simultaneous_stratified_intervals(
            np.column_stack(
                [
                    contrasts["N10"] - p10_phase.mean(axis=0),
                    contrasts["N10"] - p10_axis.mean(axis=0),
                    sign * (contrasts["J"] - joint.mean(axis=0)),
                ]
            ),
            names=("N10_phase_excess", "N10_axis_excess", "J_signed_excess"),
            strata=strata,
            n_bootstrap=CONFIRMATORY_BOOTSTRAP_DRAWS,
            seed=CONFIRMATORY_BOOTSTRAP_SEED + 5,
        ).by_name(sesoi=0.0)

        def optional_family(values: np.ndarray, paired: str) -> dict[str, Any]:
            means = values.mean(axis=1)
            paired_row = optional_paired[paired]
            target_mean = float(contrasts["N10"].mean())
            return {
                "n_controls": int(len(values)),
                "control_mean": float(means.mean()),
                "control_range": [float(means.min()), float(means.max())],
                "target_exceeds_every_control": bool(target_mean > float(means.max())),
                "add_one_p": exact_add_one_upper_p(target_mean, means),
                "paired_target_minus_mean_control": paired_row,
                "paired_lower_above_zero": bool(paired_row["ci"][0] > 0),
            }

        specificity["N10"] = {
            "families": {
                "p10_phase": optional_family(p10_phase, "N10_phase_excess"),
                "p10_axis": optional_family(p10_axis, "N10_axis_excess"),
            }
        }
        specificity["N10"]["corrected_pass"] = bool(
            all(
                row["add_one_p"] <= 1 / 32
                and row["target_exceeds_every_control"]
                and row["paired_lower_above_zero"]
                for row in specificity["N10"]["families"].values()
            )
        )
        joint_means = joint.mean(axis=1)
        paired_joint = optional_paired["J_signed_excess"]
        specificity["J"] = {
            "family": "joint",
            "observed_direction": (
                "positive_sub_additive" if sign > 0 else "negative_super_additive"
            ),
            "n_controls": int(len(joint)),
            "absolute_control_range": [
                float(np.abs(joint_means).min()),
                float(np.abs(joint_means).max()),
            ],
            "target_absolute_exceeds_every_control": bool(
                abs(float(contrasts["J"].mean())) > float(np.abs(joint_means).max())
            ),
            "absolute_add_one_p": exact_add_one_absolute_p(
                float(contrasts["J"].mean()), joint_means
            ),
            "paired_signed_excess": paired_joint,
            "paired_signed_lower_above_zero": bool(paired_joint["ci"][0] > 0),
        }
        specificity["J"]["corrected_pass"] = bool(
            specificity["J"]["absolute_add_one_p"] <= 1 / 32
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
    direction = simultaneous_stratified_intervals(
        np.column_stack(
            [
                [float(blocks_ac["Y00"][block][name]) for block in block_ids]
                for name in direction_names
            ]
        ),
        names=direction_names,
        strata=strata,
        n_bootstrap=CONFIRMATORY_BOOTSTRAP_DRAWS,
        seed=CONFIRMATORY_BOOTSTRAP_SEED + 6,
    ).by_name(sesoi=0.0)
    directional_pass = all(row["ci"][0] > 0 for row in direction.values())
    mismatch_names = ("Mplus", "Mminus")
    mismatch = simultaneous_stratified_intervals(
        np.column_stack(
            [
                [float(blocks_ac["Y00"][block][name]) for block in block_ids]
                for name in mismatch_names
            ]
        ),
        names=mismatch_names,
        strata=strata,
        n_bootstrap=CONFIRMATORY_BOOTSTRAP_DRAWS,
        seed=CONFIRMATORY_BOOTSTRAP_SEED + 7,
    ).by_name(sesoi=SESOI)
    mismatch_pass = all(
        row["ci"][0] >= -SESOI and row["ci"][1] <= SESOI
        for row in mismatch.values()
    )
    resolution_values = {"Psi00": world_ac["Y00"], "N3": contrasts["N3"]}
    if mode == "FULL_FACTORIAL":
        resolution_values.update({"N10": contrasts["N10"], "J": contrasts["J"]})
    resolution = {
        name: {
            "mde80": mde80_from_bootstrap(values, strata),
            "threshold": 0.04 if name in {"Psi00", "N3"} else 0.05,
        }
        for name, values in resolution_values.items()
    }
    for row in resolution.values():
        row["pass"] = bool(row["mde80"] <= row["threshold"])
    clean_ci = endpoints["Psi00"]["ci"]
    if clean_ci[0] >= -SESOI and clean_ci[1] <= SESOI and resolution["Psi00"]["pass"]:
        clean = "BALANCED_PREFERENCE_BOUNDED_NULL"
    elif (
        clean_ci[0] > SESOI
        and directional_pass
        and specificity["Psi00"]["corrected_pass"]
        and resolution["Psi00"]["pass"]
    ):
        clean = "BALANCED_PREFERENCE_MATERIAL"
    else:
        clean = "BALANCED_PREFERENCE_INCONCLUSIVE"
    sensitivity_names = ["Psi00", "N3"]
    sensitivity_values = [
        world_ac["Y00"] - world_at["Y00"],
        contrasts["N3"] - contrasts_at["N3"],
    ]
    if mode == "FULL_FACTORIAL":
        sensitivity_names.extend(("N10", "J"))
        sensitivity_values.extend(
            (
                contrasts["N10"] - contrasts_at["N10"],
                contrasts["J"] - contrasts_at["J"],
            )
        )
    sensitivity = simultaneous_stratified_intervals(
        np.column_stack(sensitivity_values),
        names=sensitivity_names,
        strata=strata,
        n_bootstrap=CONFIRMATORY_BOOTSTRAP_DRAWS,
        seed=CONFIRMATORY_BOOTSTRAP_SEED + 3,
    ).by_name(sesoi=SESOI)
    sensitivity_pass = {
        name: bool(row["ci"][0] >= -SESOI and row["ci"][1] <= SESOI)
        for name, row in sensitivity.items()
    }
    p3_authorized = bool(
        clean == "BALANCED_PREFERENCE_MATERIAL"
        and endpoints["N3"]["classification"] == "MATERIAL_SUPPORTIVE"
        and specificity["N3"]["corrected_pass"]
        and resolution["N3"]["pass"]
    )
    joint_authorized = bool(
        mode == "FULL_FACTORIAL"
        and p3_authorized
        and endpoints["N10"]["classification"] == "MATERIAL_SUPPORTIVE"
        and str(endpoints["J"]["classification"]).startswith("MATERIAL_")
        and specificity["N10"]["corrected_pass"]
        and specificity["J"]["corrected_pass"]
        and resolution["N10"]["pass"]
        and resolution["J"]["pass"]
    )
    null_summaries = {
        endpoint: {
            family: {
                "n_controls": int(values.shape[0]),
                "n_blocks": int(values.shape[1]),
                "control_means": [float(value) for value in values.mean(axis=1)],
                "matrix_sha256": sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
            }
            for family, values in families.items()
        }
        for endpoint, families in nulls.items()
    }
    decision = _audit_decision(
        clean,
        endpoints,
        specificity,
        resolution,
        mode,
        direction,
    )
    expected = {
        "inference_families": inference_families,
        "worlds": {
            world: {"estimate": float(values.mean()), "operator": "AG_TO_AC"}
            for world, values in world_ac.items()
        },
        "endpoints": endpoints,
        "clean_behavior_classification": clean,
        "clean_directional_family": {
            "members": list(direction_names),
            "endpoints": direction,
            "all_simultaneous_lower_bounds_above_zero": directional_pass,
        },
        "mismatch_equivalence_family": {
            "members": list(mismatch_names),
            "endpoints": mismatch,
            "simultaneous_equivalence_within_sesoi": mismatch_pass,
            "governs_only_wording_not_p3_authorization": True,
        },
        "target_resolution": resolution,
        "structured_null_specificity": specificity,
        "structured_null_summaries": null_summaries,
        "clean_pseudo_phase": clean_pseudo,
        "operator_sensitivity_ac_minus_at": {
            "endpoints": sensitivity,
            "equivalence_pass": sensitivity_pass,
            "governs_target_authorization": False,
        },
        "authorization": {
            "p3_phase3_statistical_criteria": p3_authorized,
            "joint_phase3_statistical_criteria": joint_authorized,
            "authorize_p3_phase3": p3_authorized,
            "authorize_joint_phase3": joint_authorized,
            "requires_independent_result_audit": True,
            "authorization_is_provisional_until_audit_pass": True,
        },
        "directional_scoped_support": decision["directional_scoped_support"],
        "support_tiers": decision["support_tiers"],
        "verdict": decision["verdict"],
        "per_block": [
            {
                "block_id": block,
                "stratum": strata[index],
                **{world: float(world_ac[world][index]) for world in worlds},
                **{name: float(values[index]) for name, values in contrasts.items()},
            }
            for index, block in enumerate(block_ids)
        ],
    }
    field_checks = {
        field: _nested_equal(expected[field], analysis.get(field)) for field in expected
    }
    return {
        "passed": all(field_checks.values()),
        "field_checks": field_checks,
        "expected_verdict": expected["verdict"],
        "expected_p3_authorization": p3_authorized,
        "expected_joint_authorization": joint_authorized,
    }


def audit_payloads(
    preflight: Mapping[str, Any],
    seal: Mapping[str, Any],
    target: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    preflight_sha256: str,
    target_sha256: str,
    analysis_sha256: str,
    provenance_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Audit compact endpoints and independently recompute inference without model access."""

    checks: dict[str, bool] = {
        "preflight_schema_and_authorization": bool(
            preflight.get("schema") == PREFLIGHT_SCHEMA
            and preflight.get("status")
            in {"P3_ONLY_TARGET_AUTHORIZED", "FULL_FACTORIAL_TARGET_AUTHORIZED"}
            and preflight.get("evaluation", {}).get("passed") is True
        ),
        "seal_schema": seal.get("schema") == SEAL_SCHEMA,
        "target_schema_and_state": target.get("schema") == TARGET_SCHEMA
        and target.get("status") == "TARGET_SCORED_UNANALYZED",
        "analysis_schema_and_state": analysis.get("schema") == ANALYSIS_SCHEMA
        and analysis.get("status") == "COMPLETE"
        and analysis.get("stage") == "target",
        "target_stage_declared": target.get("stage") == "target",
        "target_hash_bound": analysis.get("target_sha256") == target_sha256,
        "seal_hash_bound": analysis.get("target_seal_sha256")
        == target.get("target_seal_sha256")
        == seal.get("seal_sha256"),
        "target_modes_agree": preflight.get("target_mode")
        == seal.get("target_mode")
        == target.get("target_mode")
        == analysis.get("target_mode"),
        "confirmation_role_and_size": target.get("split_role") == "confirmation"
        and int(target.get("n_blocks", -1)) == 600
        and int(target.get("n_cells", -1)) == 5400,
        "target_mechanical_audit": target.get("mechanical_audit", {}).get("passed")
        is True,
        "preflight_file_hash_pinned_by_seal": any(
            str(record.get("sha256", "")) == preflight_sha256
            for record in seal.get("pinned_files", {}).values()
            if isinstance(record, Mapping)
        ),
        "all_seal_file_pins_valid": bool(
            seal.get("pinned_files")
            and all(
                _file_record_valid(record)
                for record in seal.get("pinned_files", {}).values()
            )
        ),
        **{str(key): bool(value) for key, value in provenance_checks.items()},
    }
    try:
        verify_target_seal(seal)
        checks["seal_internal_hash_and_contract"] = True
    except Exception:
        checks["seal_internal_hash_and_contract"] = False
    try:
        row_contract, world_values = _row_contract(target)
    except Exception as exc:
        row_contract = {
            "passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
        world_values = {}
    checks["compact_endpoint_contract"] = bool(row_contract.get("passed"))
    try:
        decision_checks = _independent_decision_checks(analysis, world_values)
    except Exception:
        decision_checks = {"independent_decision_recompute_completed": False}
    checks.update(decision_checks)

    recompute_error: str | None = None
    statistical_recompute: dict[str, Any] = {"passed": False}
    try:
        statistical_recompute = _independent_statistical_recompute(target, analysis)
    except Exception as exc:  # preserve the failure rather than erasing it
        recompute_error = f"{type(exc).__name__}: {exc}"
    checks["analysis_independent_statistical_recompute"] = bool(
        statistical_recompute.get("passed")
    )

    passed = bool(all(checks.values()))
    analysis_authorization = analysis.get("authorization", {})
    authorization = {
        "authorize_p3_phase3": bool(
            passed and analysis_authorization.get("authorize_p3_phase3") is True
        ),
        "authorize_joint_phase3": bool(
            passed and analysis_authorization.get("authorize_joint_phase3") is True
        ),
        "analysis_authorization_hash_bound": passed,
    }
    return {
        "schema": AUDIT_SCHEMA,
        "stage": "target",
        "status": "PASS" if passed else "FAIL",
        "analysis_sha256": analysis_sha256,
        "target_sha256": target_sha256,
        "preflight_sha256": preflight_sha256,
        "checks": checks,
        "failures": sorted(name for name, value in checks.items() if not value),
        "row_contract": row_contract,
        "independent_decision_checks": decision_checks,
        "independent_statistical_recompute": statistical_recompute,
        "analysis_recompute_error": recompute_error,
        "authorization": authorization,
        "clean_checkout_reexecutable": bool(
            checks.get("all_seal_file_pins_valid")
            and all(
                value
                for name, value in checks.items()
                if "manifest_" in name or name.endswith("_output_bound")
            )
        ),
        "model_loaded": False,
    }


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected object JSON at {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--preflight-manifest", required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output, manifest_path = output_paths(args.out)
    preflight_path = Path(args.preflight).expanduser().resolve()
    preflight_manifest_path = Path(args.preflight_manifest).expanduser().resolve()
    seal_path = Path(args.seal).expanduser().resolve()
    target_path = Path(args.target).expanduser().resolve()
    target_manifest_path = Path(args.target_manifest).expanduser().resolve()
    analysis_path = Path(args.analysis).expanduser().resolve()
    analysis_manifest_path = Path(args.analysis_manifest).expanduser().resolve()
    preflight = _load(preflight_path)
    seal = _load(seal_path)
    target = _load(target_path)
    analysis = _load(analysis_path)
    preflight_manifest = _load(preflight_manifest_path)
    target_manifest = _load(target_manifest_path)
    analysis_manifest = _load(analysis_manifest_path)
    provenance: dict[str, bool] = {}
    for prefix, manifest, artifact in (
        ("preflight_manifest", preflight_manifest, preflight_path),
        ("target_manifest", target_manifest, target_path),
        ("analysis_manifest", analysis_manifest, analysis_path),
    ):
        provenance.update(
            {
                f"{prefix}_{name}": value
                for name, value in _manifest_checks(manifest, artifact).items()
            }
        )
    compact_shards = target.get("compact_evidence", {}).get("raw_shards", [])
    target_inputs = target_manifest.get("inputs", [])
    provenance["compact_raw_shards_manifest_bound"] = bool(
        isinstance(compact_shards, list)
        and isinstance(target_inputs, list)
        and all(
            any(
                isinstance(candidate, Mapping)
                and str(candidate.get("path")) == str(record.get("path"))
                and str(candidate.get("sha256")) == str(record.get("sha256"))
                and candidate.get("size_bytes") == record.get("size_bytes")
                for candidate in target_inputs
            )
            for record in compact_shards
            if isinstance(record, Mapping)
        )
        and all(isinstance(record, Mapping) for record in compact_shards)
    )
    try:
        times = [
            datetime.fromisoformat(str(manifest["created_at_utc"]))
            for manifest in (preflight_manifest, target_manifest, analysis_manifest)
        ]
        provenance["manifest_chronology"] = times == sorted(times)
        commits = {
            str(manifest.get("git_commit"))
            for manifest in (preflight_manifest, target_manifest, analysis_manifest)
        }
        provenance["manifest_single_code_revision"] = len(commits) == 1
    except (KeyError, TypeError, ValueError):
        provenance["manifest_chronology"] = False
        provenance["manifest_single_code_revision"] = False

    audit = audit_payloads(
        preflight,
        seal,
        target,
        analysis,
        preflight_sha256=sha256_file(preflight_path),
        target_sha256=sha256_file(target_path),
        analysis_sha256=sha256_file(analysis_path),
        provenance_checks=provenance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-carrier-factorial-audit-v1",
        sys.argv,
        repo=REPO,
        status=audit["status"],
        inputs=[
            file_record(path)
            for path in (
                preflight_path,
                preflight_manifest_path,
                seal_path,
                target_path,
                target_manifest_path,
                analysis_path,
                analysis_manifest_path,
                REPO / "studies/splicebert_positional_boundary_coherence/PREREGISTRATION.md",
                REPO / "src/gfmcd/splicing/pbc_carrier_factorial.py",
                SCRIPT_DIR / "analyze_carrier_factorial.py",
                Path(__file__).resolve(),
            )
        ],
        outputs=[file_record(output)],
        config={"stage": "target", "target_mode": audit["target_mode"]},
        notes=[
            "Independent compact-endpoint, multiplicity, decision, seal, provenance, "
            "and independent statistical recomputation; no model loaded."
        ],
    )
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "analysis_sha256": audit["analysis_sha256"],
                "authorization": audit["authorization"],
                "failures": audit["failures"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
