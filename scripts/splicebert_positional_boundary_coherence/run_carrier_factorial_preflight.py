#!/usr/bin/env python
"""Capability/health preflight measurements for the PBC carrier factorial.

Dose selection is phase-blind: it consumes carrier capability, cross-carrier
retention, mechanical checks, and masked-base health only.  Development Psi is
opened afterwards solely to check resolution; it cannot trigger dose reselection.
P3 is mandatory.  P10 is optional and may fail unresolved without erasing a
capable P3-only target.  This runner cannot issue target authorization: its
immutable output must first pass the independent preflight auditor and the
separate clean-revision freezer.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_frozen_inputs import (  # noqa: E402
    validate_official_checkpoint,
)
from gfmcd.splicing.pbc_carrier_factorial import (  # noqa: E402
    DOSE_GRID,
    MIN_STRUCTURED_CONTROLS,
    P10_BINS_1026,
    P10_CONTROL_SEED,
    P10_PERIOD_1026,
    P10_READERS_1026,
    P3_BINS_1026,
    P3_CONTROL_SEED,
    STRUCTURED_CONTROL_BASE_SEED,
    attenuate_band,
    block_psi,
    bounded_attention_carrier_score,
    build_phase_control_family,
    build_world_tables,
    canonical_json_sha256,
    evaluate_dose_row,
    evaluate_preflight,
    exact_add_one_upper_p,
    hidden_axis_control_tables,
    intervention_audit,
    mechanical_audit,
    percentile_stratified_interval,
    phase_control_tables,
    projected_mde80,
    reconstruct_attenuated_band,
    select_largest_passing_dose,
)
from gfmcd.splicing.positional_boundary_coherence import (  # noqa: E402
    CANDIDATE_SCHEMA,
    validate_candidate_package,
)
from run_carrier_factorial_target import (  # noqa: E402
    COHORT_SCHEMA,
    SpliceBertCellScorer,
    _split_cells,
)
from audit_protocol_code_conformance import conformance_code_files  # noqa: E402


PREFLIGHT_SCHEMA = "gfmcd.pbc.carrier_factorial_preflight.v1"
TABLE_HEALTH_SCHEMA = "gfmcd.pbc.carrier_factorial_table_health.v1"
EXPECTED_SHARED_GATE_SCHEMA = "gfmcd.pbc.carrier_factorial_shared_gates.v1"
EXPECTED_CONSTRUCTION_AUDIT_SCHEMA = (
    "gfmcd.positional_boundary_coherence.cohort_audit.v1"
)
EXPECTED_CONSTRUCTION_AUDIT_STATUS = "AUDIT_PASS_TARGET_AUTHORIZED"
EXPECTED_CONSTRUCTION_AUDIT_EXPERIMENT_ID = (
    "splicebert-positional-boundary-coherence:composition-balanced-cohort-audit-v1"
)
FIXTURE_CALIBRATION_SCHEMA = "gfmcd.pbc.carrier_factorial_fixture_calibration.v1"


def construction_audit_authorizes_target(audit: Mapping[str, Any]) -> bool:
    return bool(
        audit.get("schema") == EXPECTED_CONSTRUCTION_AUDIT_SCHEMA
        and audit.get("stage") == "construction"
        and audit.get("status") == EXPECTED_CONSTRUCTION_AUDIT_STATUS
        and audit.get("passed") is True
        and audit.get("target_execution_authorized") is True
    )


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


def _manifest_binds_output(manifest: Mapping[str, Any], path: Path) -> bool:
    outputs = manifest.get("outputs")
    return bool(
        isinstance(outputs, list)
        and any(
            _file_record_valid(record)
            and Path(str(record.get("path", ""))).expanduser().resolve()
            == path.resolve()
            for record in outputs
            if isinstance(record, Mapping)
        )
    )


def validate_construction_audit_chain(
    audit: Mapping[str, Any],
    *,
    audit_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, str]:
    """Require the exact, live producer manifest for the construction audit."""

    outputs = manifest.get("outputs")
    inputs = manifest.get("inputs")
    config = manifest.get("config")
    valid = bool(
        construction_audit_authorizes_target(audit)
        and manifest.get("experiment_id")
        == EXPECTED_CONSTRUCTION_AUDIT_EXPERIMENT_ID
        and manifest.get("status") == EXPECTED_CONSTRUCTION_AUDIT_STATUS
        and isinstance(config, Mapping)
        and config.get("stage") == "construction"
        and isinstance(outputs, list)
        and len(outputs) == 1
        and _manifest_binds_output(manifest, audit_path)
        and isinstance(inputs, list)
        and bool(inputs)
        and all(_file_record_valid(record) for record in inputs)
    )
    if not valid:
        raise RuntimeError(
            "construction audit lacks its exact, live sibling producer manifest"
        )
    return {
        "artifact_sha256": sha256_file(audit_path),
        "manifest_sha256": sha256_file(manifest_path),
        "experiment_id": EXPECTED_CONSTRUCTION_AUDIT_EXPERIMENT_ID,
    }


def _fixture_endpoint_pass(row: Mapping[str, Any], name: str) -> bool:
    endpoint = row.get(name)
    if not isinstance(endpoint, Mapping):
        return False
    try:
        ci = [float(value) for value in endpoint["simultaneous_ci"]]
        paired_low = float(endpoint["paired_ci"][0])
        pvalues = [float(value) for value in endpoint["add_one_p"]]
        resolution = float(endpoint["mde80"])
    except (KeyError, TypeError, ValueError):
        return False
    if len(ci) != 2 or len(pvalues) != (1 if name == "J" else 2):
        return False
    material = ci[0] > 0.05 if name != "J" else (ci[0] > 0.05 or ci[1] < -0.05)
    threshold = 0.04 if name == "N3" else 0.05
    return bool(
        material
        and all(value <= 1 / 32 for value in pvalues)
        and endpoint.get("target_exceeds_every_control") is True
        and endpoint.get("corrected_specificity_pass") is True
        and paired_low > 0
        and resolution <= threshold
    )


def _fixture_clean_pass(row: Mapping[str, Any]) -> bool:
    clean = row.get("clean")
    if not isinstance(clean, Mapping):
        return False
    try:
        ci = [float(value) for value in clean["simultaneous_ci"]]
        return bool(
            len(ci) == 2
            and ci[0] > 0.05
            and float(clean["minimum_directional_lower_bound"]) > 0
            and float(clean["pseudo_add_one_p"]) <= 1 / 1000
            and clean.get("target_exceeds_every_pseudo") is True
            and clean.get("corrected_specificity_pass") is True
            and float(clean["paired_pseudo_ci"][0]) > 0
            and float(clean["mde80"]) <= 0.04
        )
    except (KeyError, TypeError, ValueError):
        return False


def _fixture_directional_scoped_pass(row: Mapping[str, Any]) -> bool:
    clean = row.get("clean")
    n3 = row.get("N3")
    details = row.get("engine_details")
    directions = (
        details.get("direction_family") if isinstance(details, Mapping) else None
    )
    if not all(isinstance(value, Mapping) for value in (clean, n3, directions)):
        return False
    try:
        clean_ci = [float(value) for value in clean["simultaneous_ci"]]
        n3_ci = [float(value) for value in n3["simultaneous_ci"]]
        return bool(
            len(clean_ci) == len(n3_ci) == 2
            and clean_ci[0] > 0
            and n3_ci[0] > 0
            and all(
                float(directions[name]["estimate"]) > 0
                for name in ("Dplus", "Dminus", "Psi0", "Psi1", "Psi2")
            )
            and float(clean["pseudo_add_one_p"]) <= 1 / 1000
            and clean.get("target_exceeds_every_pseudo") is True
            and clean.get("corrected_specificity_pass") is True
            and float(clean["paired_pseudo_ci"][0]) > 0
            and float(clean["mde80"]) <= 0.04
            and all(float(value) <= 1 / 32 for value in n3["add_one_p"])
            and n3.get("target_exceeds_every_control") is True
            and n3.get("corrected_specificity_pass") is True
            and all(float(interval[0]) > 0 for interval in n3["paired_cis"])
            and float(n3["mde80"]) <= 0.04
        )
    except (KeyError, TypeError, ValueError):
        return False


def summarize_fixture_calibration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reapply frozen decisions to 1,000 raw null/planted fixture records."""

    if (
        value.get("schema") != FIXTURE_CALIBRATION_SCHEMA
        or value.get("analysis_engine")
        != "analyze_carrier_factorial.analyze_fixture_arrays"
        or value.get("decision_engine")
        != "pbc_carrier_factorial.carrier_phase2_decision"
        or value.get("stage") != "development"
        or value.get("status") != "COMPLETE"
        or int(value.get("replicates", -1)) != 1_000
        or value.get("effects") != [0.05, 0.075, 0.10]
        or value.get("generator")
        != {
            "algorithm": "independent_gaussian_block_fixtures_v1",
            "base_seed": 2026081063,
            "bit_generator": "PCG64",
            "n_blocks": 600,
            "strata": [
                "chr1|+", "chr1|-", "chr2|+", "chr2|-", "chr3|+", "chr3|-",
                "chr4|+", "chr4|-", "chr5|+", "chr5|-", "chr6|+", "chr6|-",
            ],
            "noise_sd": 0.16,
            "sign_symmetric_null": True,
            "stored_inputs": "control_and_block_margins_sufficient_for_exact_families",
        }
        or value.get("analysis_parameters")
        != {"n_bootstrap": 20_000, "bootstrap_seed": 2026081001}
    ):
        return {"passed": False, "reason": "fixture calibration header invalid"}
    null_rows = value.get("null_replicates")
    planted = value.get("planted_replicates")
    if (
        not isinstance(null_rows, list)
        or len(null_rows) != 1_000
        or any(not isinstance(row, Mapping) for row in null_rows)
        or not isinstance(planted, Mapping)
        or set(planted) != {"0.050", "0.075", "0.100"}
        or any(
            not isinstance(rows, list)
            or len(rows) != 1_000
            or any(not isinstance(row, Mapping) for row in rows)
            for rows in planted.values()
        )
    ):
        return {"passed": False, "reason": "fixture replicate support invalid"}

    def ordered(rows: Sequence[Mapping[str, Any]]) -> bool:
        return [int(row.get("replicate", -1)) for row in rows] == list(range(1_000))

    if not ordered(null_rows) or any(not ordered(rows) for rows in planted.values()):
        return {"passed": False, "reason": "fixture replicate IDs invalid"}
    indexed_groups = (
        (0, null_rows),
        (1, planted["0.050"]),
        (2, planted["0.075"]),
        (3, planted["0.100"]),
    )

    def engine_row(row: Mapping[str, Any], group_index: int) -> bool:
        replicate = int(row.get("replicate", -1))
        decision = row.get("decision")
        p3 = _fixture_clean_pass(row) and _fixture_endpoint_pass(row, "N3")
        joint = bool(
            p3
            and _fixture_endpoint_pass(row, "N10")
            and _fixture_endpoint_pass(row, "J")
        )
        directional = _fixture_directional_scoped_pass(row)
        return bool(
            row.get("analysis_engine")
            == "analyze_carrier_factorial.analyze_fixture_arrays"
            and row.get("analysis_parameters")
            == {
                "n_bootstrap": 20_000,
                "bootstrap_seed": 2026081001,
                "n_blocks": 600,
            }
            and int(row.get("generator_seed", -1))
            == 2026081063 + group_index * 1_000_000 + replicate
            and len(str(row.get("input_sha256", ""))) == 64
            and isinstance(decision, Mapping)
            and decision.get("authorize_p3_phase3") is p3
            and decision.get("authorize_joint_phase3") is joint
            and decision.get("directional_scoped_support") is directional
        )

    if any(not engine_row(row, group) for group, rows in indexed_groups for row in rows):
        return {"passed": False, "reason": "fixture analysis/decision trace invalid"}
    contract = value.get("contract_fixtures")
    if not isinstance(contract, Mapping) or set(contract) != {
        "identity", "no_op", "planted_positive", "sign_reversed"
    }:
        return {"passed": False, "reason": "contract fixtures absent"}
    expected_contract = {
        "identity": (10, False, False),
        "no_op": (11, False, False),
        "planted_positive": (12, True, True),
        "sign_reversed": (13, False, False),
    }
    for name, (group, expected_p3, expected_joint) in expected_contract.items():
        row = contract[name]
        decision = row.get("decision") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or not engine_row(row, group)
            or not isinstance(decision, Mapping)
            or decision.get("authorize_p3_phase3") is not expected_p3
            or decision.get("authorize_joint_phase3") is not expected_joint
        ):
            return {"passed": False, "reason": f"contract fixture failed: {name}"}
    null_hits = sum(
        int(
            _fixture_directional_scoped_pass(row)
            or _fixture_clean_pass(row)
            or any(_fixture_endpoint_pass(row, name) for name in ("N3", "N10", "J"))
        )
        for row in null_rows
    )
    planted_counts = {
        effect: {
            name: sum(
                int(_fixture_clean_pass(row) and _fixture_endpoint_pass(row, name))
                for row in rows
            )
            for name in ("N3", "N10", "J")
        }
        for effect, rows in planted.items()
    }
    fwer = null_hits / 1_000
    power = {
        effect: {name: count / 1_000 for name, count in counts.items()}
        for effect, counts in planted_counts.items()
    }
    return {
        "passed": bool(
            fwer <= 0.06
            and power["0.075"]["N3"] >= 0.80
        ),
        "null_false_positive_count": null_hits,
        "familywise_false_positive_rate": fwer,
        "planted_success_counts": planted_counts,
        "corrected_power": power,
        "n3_power_pass": power["0.075"]["N3"] >= 0.80,
        "optional_power_pass": bool(
            power["0.075"]["N10"] >= 0.80
            and power["0.075"]["J"] >= 0.80
        ),
        "fwer_pass": fwer <= 0.06,
        "replicates": 1_000,
        "effects": [0.05, 0.075, 0.10],
    }


class PreflightProvider(Protocol):
    def position_table(self) -> np.ndarray: ...

    def measure_dose(
        self,
        *,
        band: str,
        dose: float,
        cells: Sequence[Mapping[str, Any]],
        original: np.ndarray,
        edited: np.ndarray,
        controls: Mapping[str, Sequence[np.ndarray]],
    ) -> Mapping[str, Any]: ...

    def measure_shared_gates(
        self,
        *,
        cells: Sequence[Mapping[str, Any]],
        original: np.ndarray,
        source: Mapping[str, Any] | None,
        candidate_package: Mapping[str, Any],
        construction_audit_pass: bool,
    ) -> Mapping[str, Any]: ...

    def measure_authorized_table_health(
        self,
        *,
        cells: Sequence[Mapping[str, Any]],
        original: np.ndarray,
        tables: Mapping[str, np.ndarray],
        table_specs: Mapping[str, Mapping[str, Any]],
        candidate_package: Mapping[str, Any],
        target_mode: str,
    ) -> Mapping[str, Any]: ...

    def measure_joint_p10(
        self,
        *,
        cells: Sequence[Mapping[str, Any]],
        original: np.ndarray,
        p3_dose: float,
        p10_dose: float,
        n_controls: int,
        control_seed: int,
    ) -> Mapping[str, Any]: ...

    def development_mde(
        self,
        *,
        cells: Sequence[Mapping[str, Any]],
        original: np.ndarray,
        p3_dose: float,
        p10_dose: float | None,
    ) -> Mapping[str, Mapping[str, Any]]: ...

    def metadata(self) -> Mapping[str, Any]: ...


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    return output, manifest


def preflight_raw_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema",
            "split_role",
            "dose_grid",
            "bands",
            "n_controls_per_family",
            "control_seed",
            "dose_selection_used_phase_labels",
            "dose_selection_used_psi",
            "development_resolution_opened_after_dose_selection",
            "construction_audit_provenance",
            "measurements",
            "provider",
        )
        if key in payload
    }


def preflight_analysis_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in ("schema", "status", "target_mode", "evaluation")
        if key in payload
    }


def _block_phase_blind(
    rows: Sequence[Mapping[str, Any]], *, key: str
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    by_block: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_block.setdefault(str(row["block_id"]), []).append(row)
    block_ids = tuple(sorted(by_block))
    values = np.asarray(
        [np.mean([float(row[key]) for row in by_block[block]]) for block in block_ids],
        dtype=float,
    )
    strata = tuple(
        f"{by_block[block][0]['chrom']}|{by_block[block][0]['strand']}"
        for block in block_ids
    )
    return values, block_ids, strata


def _effect_interval(
    values: np.ndarray, strata: Sequence[str], *, seed: int
) -> tuple[float, list[float]]:
    interval = percentile_stratified_interval(
        values, strata, n_bootstrap=999, seed=seed
    )
    return float(interval["estimate"]), [
        float(interval["ci_low"]),
        float(interval["ci_high"]),
    ]


def _stratified_ratio_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    strata: Sequence[str],
    *,
    seed: int,
) -> tuple[float, list[float], dict[str, Any]]:
    """Whole-block bootstrap interval for a ratio of paired mean effects."""

    top = np.asarray(numerator, dtype=float)
    bottom = np.asarray(denominator, dtype=float)
    labels = np.asarray(strata, dtype=object)
    if top.shape != bottom.shape or top.shape != labels.shape or len(top) < 6:
        raise ValueError("ratio endpoints and strata must align over at least six blocks")
    if not np.isfinite(top).all() or not np.isfinite(bottom).all():
        raise ValueError("ratio endpoints must be finite")
    groups = [np.flatnonzero(labels == label) for label in sorted(set(labels))]
    rng = np.random.default_rng(seed)
    ratios = np.empty(999, dtype=float)
    denominator_means = np.empty(999, dtype=float)
    for draw in range(999):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        denominator_means[draw] = float(bottom[indices].mean())
        ratios[draw] = (
            float(top[indices].mean() / denominator_means[draw])
            if denominator_means[draw] != 0.0
            else np.nan
        )
    point_denominator = float(bottom.mean())
    point = (
        float(top.mean() / point_denominator)
        if point_denominator != 0.0
        else np.nan
    )
    finite = np.isfinite(ratios)
    sign = 1.0 if point_denominator >= 0 else -1.0
    sign_stability = float(np.mean(sign * denominator_means > 0))
    interval = (
        [
            float(np.quantile(ratios[finite], 0.025)),
            float(np.quantile(ratios[finite], 0.975)),
        ]
        if finite.any()
        else [-np.inf, np.inf]
    )
    return point, interval, {
        "n_bootstrap": 999,
        "seed": int(seed),
        "finite_fraction": float(finite.mean()),
        "denominator_sign_stability": sign_stability,
        "denominator_point": point_denominator,
        "pass": bool(finite.all() and sign_stability >= 0.95),
    }


def _control_family_summary(
    target_effect: np.ndarray,
    control_effects: Sequence[np.ndarray],
    strata: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    means = [float(values.mean()) for values in control_effects]
    paired = target_effect - np.mean(np.stack(control_effects), axis=0)
    _, paired_ci = _effect_interval(paired, strata, seed=seed)
    target = float(target_effect.mean())
    return {
        "n_controls": len(control_effects),
        "control_means": means,
        "target_exceeds_every_control": bool(target > max(means)),
        "add_one_p": exact_add_one_upper_p(target, means),
        "paired_excess": float(paired.mean()),
        "paired_excess_ci": paired_ci,
    }


def _replace_motif(sequence: str, start: int, expected: str, replacement: str) -> str:
    if sequence[start : start + len(expected)] != expected:
        raise RuntimeError(
            f"health assay expected {expected} at [{start},{start + len(expected)})"
        )
    return sequence[:start] + replacement + sequence[start + len(expected) :]


def _health_candidate_design(
    cells: Sequence[Mapping[str, Any]],
    candidate_package: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve authentic source windows and fixed bridge anchors without outcomes."""

    if candidate_package.get("schema") != CANDIDATE_SCHEMA:
        raise RuntimeError("table health requires the frozen Phase-I candidate package")
    validate_candidate_package(candidate_package)
    raw_candidates = candidate_package.get("candidates")
    if not isinstance(raw_candidates, list):
        raise RuntimeError("candidate package lacks candidate rows")
    by_id = {str(row.get("candidate_id")): row for row in raw_candidates}
    if len(by_id) != len(raw_candidates):
        raise RuntimeError("candidate package has duplicate source IDs")
    by_block: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        by_block.setdefault(str(cell["block_id"]), []).append(cell)

    native_sources: list[dict[str, Any]] = []
    bridge_cells: list[dict[str, Any]] = []
    representatives: list[Mapping[str, Any]] = []
    for block_id in sorted(by_block):
        rows = sorted(by_block[block_id], key=lambda row: str(row["cell_id"]))
        if len(rows) != 9:
            raise RuntimeError("health design requires complete nine-cell blocks")
        representative = rows[0]
        representatives.append(representative)
        stratum = f"{representative['chrom']}|{representative['strand']}"
        source_ids = {
            *(str(row["donor_source_id"]) for row in rows),
            *(str(row["acceptor_source_id"]) for row in rows),
        }
        if len(source_ids) != 6:
            raise RuntimeError("health design requires six authentic edge sources per block")
        for source_id in sorted(source_ids):
            source = by_id.get(source_id)
            if not isinstance(source, Mapping) or source.get("role") not in {
                "donor",
                "acceptor",
            }:
                raise RuntimeError(f"missing health source candidate {source_id}")
            sequence = str(source.get("native_sequence", "")).upper().replace("U", "T")
            if len(sequence) != 1022 or set(sequence) - set("ACGT"):
                raise RuntimeError(
                    f"{source_id}: native_sequence must be an authentic 1022-nt ACGT window"
                )
            donor_pos = int(source.get("native_donor_pos", -1))
            acceptor_pos = int(source.get("native_acceptor_pos", -1))
            if (
                sequence[donor_pos : donor_pos + 2] != "GT"
                or sequence[acceptor_pos - 2 : acceptor_pos] != "AG"
                or acceptor_pos != 512
                or not 0 <= donor_pos < acceptor_pos - 2
            ):
                raise RuntimeError(
                    f"{source_id}: native source-site geometry/motifs are invalid"
                )
            native_sources.append(
                {
                    "block_id": block_id,
                    "stratum": stratum,
                    "source_id": source_id,
                    "sequence": sequence,
                    "donor_pos": donor_pos,
                    "acceptor_pos": acceptor_pos,
                }
            )

        scaffold_id = str(representative["scaffold_source_id"])
        scaffold = by_id.get(scaffold_id)
        if not isinstance(scaffold, Mapping) or scaffold.get("role") != "scaffold":
            raise RuntimeError(f"missing scaffold health source {scaffold_id}")
        bridge_pos = int(scaffold.get("bridge_acceptor_pos", -1))
        for cell in rows:
            sequence = str(cell["sequence_intact"])
            if sequence[bridge_pos - 2 : bridge_pos] != "AG":
                raise RuntimeError(
                    f"{scaffold_id}: fixed bridge AG is absent from {cell['cell_id']}"
                )
            bridge_cells.append(
                {
                    "block_id": block_id,
                    "stratum": stratum,
                    "cell_id": str(cell["cell_id"]),
                    "sequence": sequence,
                    "donor_pos": int(cell["donor_pos"]),
                    "bridge_pos": bridge_pos,
                }
            )
    return {
        "native_sources": native_sources,
        "bridge_cells": bridge_cells,
        "representatives": representatives,
        "candidate_package_sha256": canonical_json_sha256(candidate_package),
    }


def _table_health_summary(
    blocks: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Summarize raw health blocks; the independent auditor repeats this arithmetic."""

    if not blocks:
        raise RuntimeError("table health has no complete blocks")
    strata = tuple(str(row["stratum"]) for row in blocks)

    def endpoint(name: str, offset: int) -> dict[str, Any]:
        values = np.asarray([float(row[name]) for row in blocks], dtype=float)
        result = percentile_stratified_interval(
            values,
            strata,
            n_bootstrap=999,
            seed=seed + offset,
        )
        return {
            "estimate": float(result["estimate"]),
            "ci": [float(result["ci_low"]), float(result["ci_high"])],
            "n_bootstrap": 999,
            "seed": seed + offset,
        }

    native = endpoint("native_source_ac", 1)
    crossed = endpoint("crossed_ac", 2)
    bridge = endpoint("bridge_effect", 3)
    ac_at = endpoint("crossed_ac_minus_at", 4)
    protected = endpoint("protected_log_probability_delta", 5)
    donor = endpoint("donor_destruction_response", 6)
    acceptor = endpoint("acceptor_destruction_response", 7)
    intact_correct = sum(int(row["protected_intact_correct"]) for row in blocks)
    table_correct = sum(int(row["protected_table_correct"]) for row in blocks)
    total = sum(int(row["protected_total"]) for row in blocks)
    intact_accuracy = intact_correct / total if total else 0.0
    table_accuracy = table_correct / total if total else 0.0
    accuracy_retention = table_accuracy / intact_accuracy if intact_accuracy > 0 else 0.0
    baseline_donor = float(
        np.mean([float(row["baseline_donor_destruction_response"]) for row in blocks])
    )
    baseline_acceptor = float(
        np.mean([float(row["baseline_acceptor_destruction_response"]) for row in blocks])
    )
    donor_retention = donor["estimate"] / baseline_donor if baseline_donor > 0 else -np.inf
    acceptor_retention = (
        acceptor["estimate"] / baseline_acceptor if baseline_acceptor > 0 else -np.inf
    )
    gates = {
        "native_source_positive": native["ci"][0] > 0.05,
        "phase_blind_crossed_positive": crossed["ci"][0] > 0.02,
        "bridge_negative": bridge["ci"][0] >= -0.05
        and bridge["ci"][1] <= 0.05,
        "ac_at_invariant": ac_at["ci"][0] >= -0.05
        and ac_at["ci"][1] <= 0.05,
        "protected_base_health": protected["ci"][0] > -0.10,
        "masked_accuracy_health": accuracy_retention >= 0.80,
        "donor_margin_health": donor["ci"][0] > 0.50
        and donor_retention >= 0.80,
        "acceptor_margin_health": acceptor["ci"][0] > 0.50
        and acceptor_retention >= 0.80,
        "numerical_complete": all(bool(row["numerical_complete"]) for row in blocks),
    }
    return {
        "endpoints": {
            "native_source_ac": native,
            "phase_blind_crossed_ac": crossed,
            "fixed_bridge_effect": bridge,
            "phase_blind_crossed_ac_minus_at": ac_at,
            "protected_true_base_log_probability_delta": protected,
            "donor_destruction_response": donor,
            "acceptor_destruction_response": acceptor,
        },
        "protected_accuracy": {
            "intact_correct": intact_correct,
            "table_correct": table_correct,
            "total": total,
            "intact_accuracy": intact_accuracy,
            "table_accuracy": table_accuracy,
            "retention": accuracy_retention,
        },
        "destruction_response_retention": {
            "donor": float(donor_retention),
            "acceptor": float(acceptor_retention),
            "baseline_donor": baseline_donor,
            "baseline_acceptor": baseline_acceptor,
        },
        "gates": gates,
        "pass": bool(all(gates.values())),
    }


class SpliceBertPreflightProvider(SpliceBertCellScorer):
    """Default model-backed phase-blind capability provider."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["eager"] = True
        super().__init__(*args, **kwargs)

    def _p10_attention_scores(
        self,
        cells: Sequence[Mapping[str, Any]],
        table: np.ndarray,
    ) -> np.ndarray:
        torch = self.torch
        # One outcome-blind representative per block avoids ninefold weighting.
        representatives: dict[str, Mapping[str, Any]] = {}
        for cell in cells:
            representatives.setdefault(str(cell["block_id"]), cell)
        sequences = [
            str(cell["sequence_intact"])[257:765]
            for _, cell in sorted(representatives.items())
        ]
        scores: list[float] = []
        with self._using_table(table):
            for sequence in sequences:
                batch = self.adapter.tokenize([sequence])
                with torch.no_grad():
                    output = self.adapter.model(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        output_attentions=True,
                        return_dict=True,
                    )
                if output.attentions is None:
                    raise RuntimeError("eager attention was not returned")
                length = int(batch.attention_mask.sum())
                query = np.arange(70, length - 70)
                offsets = np.concatenate(
                    [np.arange(-60, -5), np.arange(6, 61)]
                )
                head_scores: list[float] = []
                for label in P10_READERS_1026:
                    layer, head = (
                        int(part[1:])
                        for part in label.replace("H", " L").split()
                    )
                    attention = (
                        output.attentions[layer][0, head].detach().float().cpu().numpy()
                    )
                    profile = np.asarray(
                        [attention[query, query + offset].mean() for offset in offsets]
                    )
                    head_scores.append(
                        bounded_attention_carrier_score(
                            profile, offsets, period=P10_PERIOD_1026
                        )
                    )
                scores.append(float(np.mean(head_scores)))
        return np.asarray(scores, dtype=float)

    def _masked_probe_health(
        self,
        cells: Sequence[Mapping[str, Any]],
        edited: np.ndarray,
    ) -> dict[str, Any]:
        torch = self.torch
        representatives: dict[str, Mapping[str, Any]] = {}
        for cell in cells:
            representatives.setdefault(str(cell["block_id"]), cell)
        selected = [cell for _, cell in sorted(representatives.items())]
        positions = (64, 128, 192, 256, 320, 360, 528, 608, 704, 800, 896, 960)

        def score(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            log_prob: list[float] = []
            accuracy: list[float] = []
            with self._using_table(table):
                for cell in selected:
                    sequence = str(cell["sequence_intact"])
                    valid = [
                        position
                        for position in positions
                        if sequence[position] in "ACGT"
                        and abs(position - int(cell["donor_pos"])) > 3
                        and abs(position - int(cell["acceptor_pos"])) > 3
                    ]
                    batch = self.adapter.tokenize([sequence])
                    targets = [self.base_ids[sequence[position]] for position in valid]
                    for position in valid:
                        batch.input_ids[
                            0, self.adapter.nucleotide_to_token(position)
                        ] = self.mask_token_id
                    with torch.no_grad():
                        logits = self.adapter.model(
                            input_ids=batch.input_ids,
                            attention_mask=batch.attention_mask,
                            return_dict=True,
                        ).logits[0]
                    token_positions = [
                        self.adapter.nucleotide_to_token(position) for position in valid
                    ]
                    selected_logits = logits[token_positions].float()
                    log_probs = torch.log_softmax(selected_logits, dim=-1)
                    target_tensor = torch.tensor(targets, device=log_probs.device)
                    probe = torch.arange(len(valid), device=log_probs.device)
                    log_prob.append(
                        float(log_probs[probe, target_tensor].mean().detach().cpu())
                    )
                    accuracy.append(
                        float(
                            (selected_logits.argmax(dim=-1) == target_tensor)
                            .float()
                            .mean()
                            .detach()
                            .cpu()
                        )
                    )
            return np.asarray(log_prob), np.asarray(accuracy)

        intact_lp, intact_accuracy = score(self.position_table())
        edited_lp, edited_accuracy = score(edited)
        strata = tuple(
            f"{cell['chrom']}|{cell['strand']}" for cell in selected
        )
        delta = edited_lp - intact_lp
        interval = percentile_stratified_interval(
            delta, strata, n_bootstrap=999, seed=20260841
        )
        retention = float(edited_accuracy.mean()) / max(
            float(intact_accuracy.mean()), np.finfo(float).tiny
        )
        return {
            "mean_true_base_log_probability_delta": float(delta.mean()),
            "log_probability_delta_ci": [
                float(interval["ci_low"]),
                float(interval["ci_high"]),
            ],
            "intact_accuracy": float(intact_accuracy.mean()),
            "edited_accuracy": float(edited_accuracy.mean()),
            "accuracy_retention": retention,
            "pass": bool(interval["ci_low"] > -0.10 and retention >= 0.80),
        }

    def _score_health_requests(
        self,
        table: np.ndarray,
        requests: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Score independently masked source, bridge, and protected-base requests."""

        torch = self.torch
        if len({str(row["request_id"]) for row in requests}) != len(requests):
            raise RuntimeError("health request IDs are not unique")
        output: dict[str, dict[str, Any]] = {}
        with self._using_table(table), torch.no_grad():
            for start in range(0, len(requests), self.batch_size):
                chunk = requests[start : start + self.batch_size]
                sequences = [str(row["sequence"]) for row in chunk]
                batch = self.adapter.tokenize(sequences)
                if batch.input_ids.shape[1] != 1024 or not bool(
                    torch.all(batch.attention_mask.sum(dim=1) == 1024)
                ):
                    raise RuntimeError("health requests must tokenize to exactly 1024 tokens")
                token_positions: list[int] = []
                target_ids: list[int] = []
                for index, row in enumerate(chunk):
                    position = int(row["position"])
                    true_base = str(row["true_base"])
                    sequence = str(row["sequence"])
                    if sequence[position] != true_base or true_base not in self.base_ids:
                        raise RuntimeError("health request true base disagrees with sequence")
                    token_position = self.adapter.nucleotide_to_token(position)
                    batch.input_ids[index, token_position] = self.mask_token_id
                    token_positions.append(token_position)
                    target_ids.append(self.base_ids[true_base])
                raw = self.adapter.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    return_dict=True,
                )
                logits = raw.logits.float()
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("non-finite logits in a table-health request")
                row_index = torch.arange(len(chunk), device=logits.device)
                position_tensor = torch.tensor(token_positions, device=logits.device)
                selected = logits[row_index, position_tensor]
                base_tensor = torch.tensor(
                    [self.base_ids[base] for base in "ACGT"],
                    device=logits.device,
                )
                target_tensor = torch.tensor(target_ids, device=logits.device)
                base_logits = selected.index_select(1, base_tensor)
                target_logits = selected.gather(1, target_tensor[:, None]).squeeze(1)
                target_base_index = torch.tensor(
                    ["ACGT".index(str(row["true_base"])) for row in chunk],
                    device=logits.device,
                )
                other_sum = base_logits.sum(dim=1) - base_logits.gather(
                    1, target_base_index[:, None]
                ).squeeze(1)
                margins = target_logits - other_sum / 3.0
                log_probabilities = torch.log_softmax(selected, dim=-1).gather(
                    1, target_tensor[:, None]
                ).squeeze(1)
                predicted = selected.argmax(dim=-1)
                for index, request in enumerate(chunk):
                    output[str(request["request_id"])] = {
                        "margin": float(margins[index].detach().cpu()),
                        "true_base_log_probability": float(
                            log_probabilities[index].detach().cpu()
                        ),
                        "correct": bool(
                            int(predicted[index].detach().cpu()) == target_ids[index]
                        ),
                        "finite": True,
                    }
        return output

    @staticmethod
    def _health_requests(design: Mapping[str, Any]) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []
        for source in design["native_sources"]:
            source_id = str(source["source_id"])
            sequence = str(source["sequence"])
            donor_pos = int(source["donor_pos"])
            acceptor_pos = int(source["acceptor_pos"])
            acceptor_ac = _replace_motif(
                sequence, acceptor_pos - 2, "AG", "AC"
            )
            donor_ac = _replace_motif(sequence, donor_pos, "GT", "AC")
            requests.extend(
                (
                    {
                        "request_id": f"native:{source_id}:donor:intact",
                        "sequence": sequence,
                        "position": donor_pos,
                        "true_base": "G",
                    },
                    {
                        "request_id": f"native:{source_id}:donor:acceptor_ac",
                        "sequence": acceptor_ac,
                        "position": donor_pos,
                        "true_base": "G",
                    },
                    {
                        "request_id": f"native:{source_id}:acceptor:intact",
                        "sequence": sequence,
                        "position": acceptor_pos - 1,
                        "true_base": "G",
                    },
                    {
                        "request_id": f"native:{source_id}:acceptor:donor_ac",
                        "sequence": donor_ac,
                        "position": acceptor_pos - 1,
                        "true_base": "G",
                    },
                )
            )
        for cell in design["bridge_cells"]:
            cell_id = str(cell["cell_id"])
            sequence = _replace_motif(
                str(cell["sequence"]), int(cell["bridge_pos"]) - 2, "AG", "AC"
            )
            requests.append(
                {
                    "request_id": f"bridge:{cell_id}",
                    "sequence": sequence,
                    "position": int(cell["donor_pos"]),
                    "true_base": "G",
                }
            )
        protected_positions = (
            64,
            128,
            192,
            256,
            320,
            360,
            528,
            608,
            704,
            800,
            896,
            960,
        )
        for cell in design["representatives"]:
            sequence = str(cell["sequence_intact"])
            for position in protected_positions:
                true_base = sequence[position]
                if true_base not in "ACGT":
                    raise RuntimeError("protected health probe is not A/C/G/T")
                requests.append(
                    {
                        "request_id": f"protected:{cell['block_id']}:{position}",
                        "sequence": sequence,
                        "position": position,
                        "true_base": true_base,
                    }
                )
        return requests

    def measure_authorized_table_health(
        self,
        *,
        cells: Sequence[Mapping[str, Any]],
        original: np.ndarray,
        tables: Mapping[str, np.ndarray],
        table_specs: Mapping[str, Mapping[str, Any]],
        candidate_package: Mapping[str, Any],
        target_mode: str,
    ) -> Mapping[str, Any]:
        """Run every §7.4 assay on every exact named/control/joint table."""

        if set(tables) != set(table_specs):
            raise RuntimeError("table-health specs do not cover the exact table family")
        design = _health_candidate_design(cells, candidate_package)
        requests = self._health_requests(design)
        baseline_scores = self._score_health_requests(original, requests)
        scored_conditions = self.score_conditions(cells, tables)
        by_block_cells: dict[str, list[Mapping[str, Any]]] = {}
        for cell in cells:
            by_block_cells.setdefault(str(cell["block_id"]), []).append(cell)
        native_by_block: dict[str, list[Mapping[str, Any]]] = {}
        for source in design["native_sources"]:
            native_by_block.setdefault(str(source["block_id"]), []).append(source)
        bridge_by_block: dict[str, list[Mapping[str, Any]]] = {}
        for cell in design["bridge_cells"]:
            bridge_by_block.setdefault(str(cell["block_id"]), []).append(cell)

        output_tables: dict[str, Any] = {}
        for condition, table in tables.items():
            table_scores = (
                baseline_scores
                if np.array_equal(np.asarray(table), np.asarray(original))
                else self._score_health_requests(table, requests)
            )
            scored_rows = {
                str(row["cell_id"]): row for row in scored_conditions[condition]
            }
            raw_blocks: list[dict[str, Any]] = []
            for block_id in sorted(by_block_cells):
                block_cells = by_block_cells[block_id]
                stratum = f"{block_cells[0]['chrom']}|{block_cells[0]['strand']}"
                native_ac_values: list[float] = []
                donor_response_values: list[float] = []
                acceptor_response_values: list[float] = []
                baseline_donor_values: list[float] = []
                baseline_acceptor_values: list[float] = []
                for source in native_by_block[block_id]:
                    source_id = str(source["source_id"])
                    def margin(scores: Mapping[str, Mapping[str, Any]], suffix: str) -> float:
                        return float(scores[f"native:{source_id}:{suffix}"]["margin"])

                    acceptor_response = margin(table_scores, "donor:intact") - margin(
                        table_scores, "donor:acceptor_ac"
                    )
                    donor_response = margin(table_scores, "acceptor:intact") - margin(
                        table_scores, "acceptor:donor_ac"
                    )
                    baseline_acceptor = margin(
                        baseline_scores, "donor:intact"
                    ) - margin(baseline_scores, "donor:acceptor_ac")
                    baseline_donor = margin(
                        baseline_scores, "acceptor:intact"
                    ) - margin(baseline_scores, "acceptor:donor_ac")
                    native_ac_values.append(acceptor_response)
                    donor_response_values.append(donor_response)
                    acceptor_response_values.append(acceptor_response)
                    baseline_donor_values.append(baseline_donor)
                    baseline_acceptor_values.append(baseline_acceptor)

                cross = [scored_rows[str(cell["cell_id"])] for cell in block_cells]
                bridge_values = [
                    float(scored_rows[str(cell["cell_id"])]["f_intact"])
                    - float(table_scores[f"bridge:{cell['cell_id']}"]["margin"])
                    for cell in bridge_by_block[block_id]
                ]
                protected_ids = [
                    request_id
                    for request_id in table_scores
                    if request_id.startswith(f"protected:{block_id}:")
                ]
                protected_delta = [
                    float(table_scores[request_id]["true_base_log_probability"])
                    - float(baseline_scores[request_id]["true_base_log_probability"])
                    for request_id in protected_ids
                ]
                numerical = all(
                    bool(table_scores[request_id]["finite"])
                    for request_id in table_scores
                    if request_id.startswith(("native:", "bridge:", f"protected:{block_id}:"))
                ) and all(
                    np.isfinite(
                        [
                            float(row["f_intact"]),
                            float(row["f_ac"]),
                            float(row["f_at"]),
                            float(row["coupling_ac"]),
                            float(row["coupling_at"]),
                        ]
                    ).all()
                    for row in cross
                )
                raw_blocks.append(
                    {
                        "block_id": block_id,
                        "stratum": stratum,
                        "native_source_ac": float(np.mean(native_ac_values)),
                        "crossed_ac": float(
                            np.mean([float(row["coupling_ac"]) for row in cross])
                        ),
                        "bridge_effect": float(np.mean(bridge_values)),
                        "crossed_ac_minus_at": float(
                            np.mean(
                                [
                                    float(row["coupling_ac"])
                                    - float(row["coupling_at"])
                                    for row in cross
                                ]
                            )
                        ),
                        "protected_log_probability_delta": float(
                            np.mean(protected_delta)
                        ),
                        "protected_intact_correct": sum(
                            int(bool(baseline_scores[key]["correct"]))
                            for key in protected_ids
                        ),
                        "protected_table_correct": sum(
                            int(bool(table_scores[key]["correct"]))
                            for key in protected_ids
                        ),
                        "protected_total": len(protected_ids),
                        "donor_destruction_response": float(
                            np.mean(donor_response_values)
                        ),
                        "acceptor_destruction_response": float(
                            np.mean(acceptor_response_values)
                        ),
                        "baseline_donor_destruction_response": float(
                            np.mean(baseline_donor_values)
                        ),
                        "baseline_acceptor_destruction_response": float(
                            np.mean(baseline_acceptor_values)
                        ),
                        "numerical_complete": bool(numerical),
                    }
                )
            table_hash = sha256(np.ascontiguousarray(table).tobytes()).hexdigest()
            seed = 20261000 + int(table_hash[:8], 16) % 100_000_000
            summary = _table_health_summary(raw_blocks, seed=seed)
            output_tables[condition] = {
                "spec": dict(table_specs[condition]),
                "table_sha256": table_hash,
                "inference_seed": seed,
                "raw_blocks": raw_blocks,
                **summary,
            }
        return {
            "schema": TABLE_HEALTH_SCHEMA,
            "target_mode": target_mode,
            "candidate_package_sha256": design["candidate_package_sha256"],
            "n_blocks": len(by_block_cells),
            "n_tables": len(tables),
            "tables": output_tables,
            "all_tables_pass": bool(
                output_tables and all(row["pass"] for row in output_tables.values())
            ),
        }

    def measure_dose(
        self,
        *,
        band: str,
        dose: float,
        cells: Sequence[Mapping[str, Any]],
        original: np.ndarray,
        edited: np.ndarray,
        controls: Mapping[str, Sequence[np.ndarray]],
    ) -> Mapping[str, Any]:
        rescue = reconstruct_attenuated_band(
            edited,
            original,
            P3_BINS_1026 if band == "p3" else P10_BINS_1026,
            dose,
        )
        tables = {"intact": original, "edited": edited, "rescue": rescue}
        for family, family_tables in controls.items():
            tables.update(
                {
                    f"{family}_control_{index:03d}": table
                    for index, table in enumerate(family_tables, 1)
                }
            )
        scored = self.score_conditions(cells, tables)
        replay_fields = ("f_intact", "f_ac", "f_at")
        rescue_forward_max_abs_error = max(
            abs(float(intact_row[field]) - float(rescue_row[field]))
            for intact_row, rescue_row in zip(
                scored["intact"], scored["rescue"], strict=True
            )
            for field in replay_fields
        )
        intact, block_ids, strata = _block_phase_blind(
            scored["intact"], key="coupling_ac"
        )
        changed, changed_ids, _ = _block_phase_blind(
            scored["edited"], key="coupling_ac"
        )
        rescued, rescued_ids, _ = _block_phase_blind(
            scored["rescue"], key="coupling_ac"
        )
        if changed_ids != block_ids or rescued_ids != block_ids:
            raise RuntimeError("dose rows change block support")
        health = self._masked_probe_health(cells, edited)
        control_health = {
            family: [self._masked_probe_health(cells, table) for table in tables_]
            for family, tables_ in controls.items()
        }
        all_control_health_pass = all(
            row["pass"]
            for rows in control_health.values()
            for row in rows
        )
        if band == "p3":
            effect = intact - changed
            estimate, interval = _effect_interval(
                effect, strata, seed=20260820 + int(100 * dose)
            )
            family_effects: dict[str, list[np.ndarray]] = {}
            for family in controls:
                family_effects[family] = []
                for label in sorted(
                    key for key in scored if key.startswith(f"{family}_control_")
                ):
                    values, ids, _ = _block_phase_blind(
                        scored[label], key="coupling_ac"
                    )
                    if ids != block_ids:
                        raise RuntimeError("control changes block support")
                    family_effects[family].append(intact - values)
            control_summaries = {
                family: _control_family_summary(
                    effect,
                    values,
                    strata,
                    seed=20260900 + index,
                )
                for index, (family, values) in enumerate(sorted(family_effects.items()))
            }
            p10_intact = self._p10_attention_scores(cells, original)
            p10_edited = self._p10_attention_scores(cells, edited)
            retention, retention_ci, retention_ratio_health = (
                _stratified_ratio_interval(
                    p10_edited,
                    p10_intact,
                    strata,
                    seed=20260910,
                )
            )
            rescue_recovery, rescue_ci, rescue_ratio_health = (
                _stratified_ratio_interval(
                    rescued - changed,
                    effect,
                    strata,
                    seed=20260911,
                )
            )
            return {
                "capability_effect": estimate,
                "capability_ci": interval,
                "control_families": control_summaries,
                "cross_carrier_retention": retention,
                "cross_carrier_retention_ci": retention_ci,
                "cross_carrier_ratio_health": retention_ratio_health,
                "cross_carrier_ratio_raw": [
                    {
                        "block_id": block_id,
                        "stratum": stratum,
                        "numerator": float(numerator),
                        "denominator": float(denominator),
                    }
                    for block_id, stratum, numerator, denominator in zip(
                        block_ids,
                        strata,
                        p10_edited,
                        p10_intact,
                        strict=True,
                    )
                ],
                "rescue_recovery": rescue_recovery,
                "rescue_recovery_ci": rescue_ci,
                "rescue_ratio_health": rescue_ratio_health,
                "rescue_forward_max_abs_error": rescue_forward_max_abs_error,
                "health": health,
                "health_pass": health["pass"],
                "control_health": control_health,
                "all_control_health_pass": all_control_health_pass,
            }
        attention_intact = self._p10_attention_scores(cells, original)
        attention_changed = self._p10_attention_scores(cells, edited)
        intact_carrier_score, intact_carrier_ci = _effect_interval(
            attention_intact, strata, seed=20260829
        )
        reduction, interval, capability_ratio_health = _stratified_ratio_interval(
            attention_intact - attention_changed,
            attention_intact,
            strata,
            seed=20260830 + int(100 * dose),
        )
        denominator_mean = float(attention_intact.mean())
        normalized_reduction = (
            attention_intact - attention_changed
        ) / denominator_mean
        family_effects = {}
        for family, family_tables in controls.items():
            family_effects[family] = [
                (attention_intact - self._p10_attention_scores(cells, control))
                / denominator_mean
                for control in family_tables
            ]
        control_summaries = {
            family: _control_family_summary(
                normalized_reduction,
                values,
                strata,
                seed=20260920 + index,
            )
            for index, (family, values) in enumerate(sorted(family_effects.items()))
        }
        attention_rescue = self._p10_attention_scores(cells, rescue)
        rescue_recovery, rescue_ci, rescue_ratio_health = (
            _stratified_ratio_interval(
                attention_rescue - attention_changed,
                attention_intact - attention_changed,
                strata,
                seed=20260931,
            )
        )
        return {
            "intact_carrier_score": intact_carrier_score,
            "intact_carrier_score_ci": intact_carrier_ci,
            "denominator_finite_sign_stable_fraction": capability_ratio_health[
                "denominator_sign_stability"
            ],
            "capability_fraction_reduced": reduction,
            "capability_fraction_reduced_ci": interval,
            "capability_ratio_health": capability_ratio_health,
            "capability_ratio_raw": [
                {
                    "block_id": block_id,
                    "stratum": stratum,
                    "numerator": float(numerator),
                    "denominator": float(denominator),
                }
                for block_id, stratum, numerator, denominator in zip(
                    block_ids,
                    strata,
                    attention_intact - attention_changed,
                    attention_intact,
                    strict=True,
                )
            ],
            "control_families": control_summaries,
            "cross_carrier_retention": -1.0,
            "cross_carrier_retention_ci": [-1.0, -1.0],
            "joint_k3_evaluated": False,
            "joint_health_pass": False,
            "rescue_recovery": rescue_recovery,
            "rescue_recovery_ci": rescue_ci,
            "rescue_ratio_health": rescue_ratio_health,
            "rescue_forward_max_abs_error": rescue_forward_max_abs_error,
            "health": health,
            "health_pass": health["pass"],
            "control_health": control_health,
            "all_control_health_pass": all_control_health_pass,
        }

    def measure_joint_p10(
        self,
        *,
        cells: Sequence[Mapping[str, Any]],
        original: np.ndarray,
        p3_dose: float,
        p10_dose: float,
        n_controls: int,
        control_seed: int,
    ) -> Mapping[str, Any]:
        """Measure true K3 preservation after P10 with P3 held fixed."""

        worlds = build_world_tables(
            original, p3_dose=p3_dose, p10_dose=p10_dose
        )
        scored = self.score_conditions(cells, worlds)
        arrays: dict[str, np.ndarray] = {}
        block_ids: tuple[str, ...] | None = None
        strata: tuple[str, ...] | None = None
        for world in ("Y00", "Y10", "Y01", "Y11"):
            values, ids, world_strata = _block_phase_blind(
                scored[world], key="coupling_ac"
            )
            if block_ids is None:
                block_ids, strata = ids, world_strata
            elif ids != block_ids or world_strata != strata:
                raise RuntimeError("joint worlds change block support or strata")
            arrays[world] = values
        if strata is None:
            raise RuntimeError("joint K3 received no development blocks")
        k3_intact = arrays["Y00"] - arrays["Y10"]
        k3_with_p10 = arrays["Y01"] - arrays["Y11"]
        retention, retention_ci, ratio_health = _stratified_ratio_interval(
            k3_with_p10,
            k3_intact,
            strata,
            seed=20260930 + int(100 * p10_dose),
        )

        controls, specs = build_phase_control_family(
            original,
            p3_dose=p3_dose,
            p10_dose=p10_dose,
            n_controls=n_controls,
            seed=control_seed,
        )
        joint_tables = {
            label: table
            for label, table in controls.items()
            if str(specs[label]["family"]).startswith("joint")
        }
        joint_target_health = self._masked_probe_health(cells, worlds["Y11"])
        joint_control_health = {
            label: self._masked_probe_health(cells, table)
            for label, table in joint_tables.items()
        }
        all_joint_control_health_pass = all(
            row["pass"] for row in joint_control_health.values()
        )
        return {
            "cross_carrier_retention": retention,
            "cross_carrier_retention_ci": retention_ci,
            "joint_k3_evaluated": True,
            "joint_k3_intact": float(k3_intact.mean()),
            "joint_k3_with_p10": float(k3_with_p10.mean()),
            "joint_k3_ratio_health": ratio_health,
            "joint_target_health": joint_target_health,
            "joint_control_health": joint_control_health,
            "all_joint_control_health_pass": all_joint_control_health_pass,
            "joint_health_pass": bool(
                ratio_health["pass"]
                and joint_target_health["pass"]
                and all_joint_control_health_pass
            ),
        }

    def measure_shared_gates(
        self,
        *,
        cells: Sequence[Mapping[str, Any]],
        original: np.ndarray,
        source: Mapping[str, Any] | None,
        candidate_package: Mapping[str, Any],
        construction_audit_pass: bool,
    ) -> Mapping[str, Any]:
        # Independently repeat the intact condition for the determinism check.
        scored = self.score_conditions(
            cells, {"intact_a": original, "intact_b": original}
        )
        ac, _, strata = _block_phase_blind(scored["intact_a"], key="coupling_ac")
        at, _, _ = _block_phase_blind(scored["intact_a"], key="coupling_at")
        crossed = percentile_stratified_interval(
            ac, strata, n_bootstrap=999, seed=20260851
        )
        ac_at = percentile_stratified_interval(
            ac - at, strata, n_bootstrap=999, seed=20260852
        )
        fields = ("f_intact", "f_ac", "f_at", "coupling_ac", "coupling_at")
        repeat_error = max(
            abs(float(left[field]) - float(right[field]))
            for left, right in zip(
                scored["intact_a"], scored["intact_b"], strict=True
            )
            for field in fields
        )
        intact_health = self.measure_authorized_table_health(
            cells=cells,
            original=original,
            tables={"intact": original},
            table_specs={
                "intact": {"family": "shared_intact", "control_id": None, "world": "Y00"}
            },
            candidate_package=candidate_package,
            target_mode="PREFLIGHT_INTACT",
        )
        intact_row = intact_health["tables"]["intact"]
        health_gates = intact_row["gates"]
        calibration_source = (
            source.get("calibration") if isinstance(source, Mapping) else None
        )
        calibration = (
            summarize_fixture_calibration(calibration_source)
            if isinstance(calibration_source, Mapping)
            else {"passed": False, "reason": "fixture calibration absent"}
        )
        gates = {
            "identity_and_reconstruction": bool(repeat_error <= 1e-8),
            "native_source_positive": bool(health_gates["native_source_positive"]),
            "phase_blind_crossed_positive": bool(
                health_gates["phase_blind_crossed_positive"]
            ),
            "bridge_negative": bool(health_gates["bridge_negative"]),
            "ac_at_invariant": bool(health_gates["ac_at_invariant"]),
            "protected_base_health": bool(health_gates["protected_base_health"]),
            "masked_accuracy_health": bool(health_gates["masked_accuracy_health"]),
            "donor_margin_health": bool(health_gates["donor_margin_health"]),
            "acceptor_margin_health": bool(health_gates["acceptor_margin_health"]),
            "determinism": bool(repeat_error <= 1e-8),
            "construction_audit": bool(construction_audit_pass),
            "synthetic_familywise_fpr": calibration.get("fwer_pass") is True,
            "planted_power": calibration.get("n3_power_pass") is True,
        }
        return {
            **gates,
            "diagnostics": {
                "phase_blind_crossed": crossed,
                "phase_blind_ac_minus_at": ac_at,
                "repeat_forward_max_abs_error": repeat_error,
                "externally_measured_gate_source_present": source is not None,
                "intact_table_health": intact_row,
                "fixture_calibration": {
                    "source": calibration_source,
                    "summary": calibration,
                    "provenance": (
                        source.get("calibration_provenance")
                        if isinstance(source, Mapping)
                        else None
                    ),
                },
            },
        }

    def development_mde(
        self,
        *,
        cells: Sequence[Mapping[str, Any]],
        original: np.ndarray,
        p3_dose: float,
        p10_dose: float | None,
    ) -> Mapping[str, Mapping[str, Any]]:
        worlds = build_world_tables(
            original, p3_dose=p3_dose, p10_dose=p10_dose
        )
        scored = self.score_conditions(cells, worlds)
        by_world: dict[str, dict[str, dict[str, Any]]] = {}
        for world, rows in scored.items():
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for row in rows:
                grouped.setdefault(str(row["block_id"]), []).append(row)
            by_world[world] = {
                block_id: {
                    **block_psi(block_rows, value_key="coupling_ac"),
                    "stratum": f"{block_rows[0]['chrom']}|{block_rows[0]['strand']}",
                }
                for block_id, block_rows in grouped.items()
            }
        block_ids = tuple(sorted(by_world["Y00"]))
        strata = tuple(by_world["Y00"][block]["stratum"] for block in block_ids)
        arrays = {
            world: np.asarray(
                [by_world[world][block]["Psi"] for block in block_ids], dtype=float
            )
            for world in worlds
        }
        output = {
            "clean_psi": projected_mde80(
                arrays["Y00"], strata, seed=20260941
            ),
            "N3": projected_mde80(
                arrays["Y00"] - arrays["Y10"], strata, seed=20260942
            ),
        }
        if p10_dose is not None:
            output.update(
                {
                    "N10": projected_mde80(
                        arrays["Y00"] - arrays["Y01"], strata, seed=20260943
                    ),
                    "J": projected_mde80(
                        arrays["Y11"]
                        - arrays["Y10"]
                        - arrays["Y01"]
                        + arrays["Y00"],
                        strata,
                        seed=20260944,
                    ),
                }
            )
        return output


def load_provider(
    specification: str | None,
    *,
    model_path: str | Path,
    device: str | None,
    batch_size: int,
) -> PreflightProvider:
    if not specification:
        return SpliceBertPreflightProvider(
            model_path, device=device, batch_size=batch_size
        )
    module_name, attribute = specification.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        eager=True,
    )


def execute_preflight(
    *,
    cohort: Mapping[str, Any],
    candidate_package: Mapping[str, Any],
    provider: PreflightProvider,
    split_role: str,
    shared_gate_source: Mapping[str, Any] | None,
    construction_audit_pass: bool,
    n_controls: int,
    control_seed: int,
) -> dict[str, Any]:
    """Run phase-blind dose selection, then the resolution-only development open."""

    if cohort.get("schema") != COHORT_SCHEMA:
        raise RuntimeError("wrong PBC cohort schema")
    if n_controls != MIN_STRUCTURED_CONTROLS:
        raise RuntimeError("frozen preflight requires exactly 31 controls per family")
    if control_seed != STRUCTURED_CONTROL_BASE_SEED:
        raise RuntimeError(
            f"frozen preflight requires control seed {STRUCTURED_CONTROL_BASE_SEED}"
        )
    if split_role != "development":
        raise RuntimeError("carrier-factorial preflight may read only development")
    cells = _split_cells(cohort, split_role)
    if len({str(cell["block_id"]) for cell in cells}) != 120:
        raise RuntimeError("development preflight requires exactly 120 complete blocks")
    original = provider.position_table()
    dose_rows: dict[str, dict[str, Any]] = {"p3": {}, "p10": {}}
    for band, bins, family_seed in (
        ("p3", P3_BINS_1026, P3_CONTROL_SEED),
        ("p10", P10_BINS_1026, P10_CONTROL_SEED),
    ):
        for dose in DOSE_GRID:
            edited = attenuate_band(original, bins, dose)
            phase_controls, _ = phase_control_tables(
                original,
                bins,
                dose,
                n_controls=n_controls,
                seed=family_seed,
            )
            axis_controls, _ = hidden_axis_control_tables(
                original,
                bins,
                dose,
                n_controls=n_controls,
                seed=family_seed,
            )
            protect = P10_BINS_1026 if band == "p3" else P3_BINS_1026
            mechanics = intervention_audit(
                original,
                edited,
                bins=bins,
                dose=dose,
                protect_bins=protect,
            )
            measurement = dict(
                provider.measure_dose(
                    band=band,
                    dose=dose,
                    cells=cells,
                    original=original,
                    edited=edited,
                    controls={"phase": phase_controls, "axis": axis_controls},
                )
            )
            measurement["mechanical"] = mechanics
            measurement["mechanical_pass"] = bool(
                mechanics["finite_real"]
                and mechanics["outside_band_pass"]
                and mechanics["dose_pass"]
                and mechanics["protected_band_pass"]
            )
            dose_rows[band][f"{dose:.2f}"] = measurement

    shared = provider.measure_shared_gates(
        cells=cells,
        original=original,
        source=shared_gate_source,
        candidate_package=candidate_package,
        construction_audit_pass=construction_audit_pass,
    )
    phase_blind_provisional = evaluate_preflight(
        {"shared_gates": shared, "dose_rows": dose_rows},
        development_mde={
            "clean_psi": 0.0,
            "N3": 0.0,
            "N10": 0.0,
            "J": 0.0,
        },
    )
    p10_individual = select_largest_passing_dose(
        "p10", dose_rows["p10"], require_cross_carrier=False
    )
    joint_attempts: dict[str, Any] = {}
    if phase_blind_provisional["shared_pass"] and phase_blind_provisional["p3"]["passed"]:
        selected_p3 = float(phase_blind_provisional["p3"]["selected_dose"])
        for dose in reversed(DOSE_GRID):
            label = f"{dose:.2f}"
            if not p10_individual["evaluated"][label]["passed"]:
                joint_attempts[label] = {
                    "status": "NOT_JOINT_TESTED_INDIVIDUAL_P10_FAILED"
                }
                continue
            joint_measurement = dict(
                provider.measure_joint_p10(
                    cells=cells,
                    original=original,
                    p3_dose=selected_p3,
                    p10_dose=dose,
                    n_controls=n_controls,
                    control_seed=control_seed,
                )
            )
            candidate_controls, candidate_specs = build_phase_control_family(
                original,
                p3_dose=selected_p3,
                p10_dose=dose,
                n_controls=n_controls,
                seed=control_seed,
            )
            candidate_mechanics = mechanical_audit(
                original,
                p3_dose=selected_p3,
                p10_dose=dose,
                controls=candidate_controls,
                control_specs=candidate_specs,
            )
            joint_measurement["joint_mechanical_audit"] = candidate_mechanics
            joint_measurement["joint_health_pass"] = bool(
                joint_measurement.get("joint_health_pass", False)
                and candidate_mechanics["passed"]
                and shared.get("diagnostics", {})
                .get("fixture_calibration", {})
                .get("summary", {})
                .get("optional_power_pass")
                is True
            )
            dose_rows["p10"][label].update(joint_measurement)
            passed, gates = evaluate_dose_row(
                "p10", dose_rows["p10"][label], require_cross_carrier=True
            )
            joint_attempts[label] = {
                "status": "PASS" if passed else "FAIL_STEP_DOWN",
                "gates": gates,
                "cross_carrier_retention": joint_measurement.get(
                    "cross_carrier_retention"
                ),
                "cross_carrier_retention_ci": joint_measurement.get(
                    "cross_carrier_retention_ci"
                ),
                "joint_mechanical_pass": candidate_mechanics["passed"],
            }
            if passed:
                break

    post_joint_provisional = evaluate_preflight(
        {"shared_gates": shared, "dose_rows": dose_rows},
        development_mde={
            "clean_psi": 0.0,
            "N3": 0.0,
            "N10": 0.0,
            "J": 0.0,
        },
    )
    authorized_table_health: Mapping[str, Any] = {
        "schema": TABLE_HEALTH_SCHEMA,
        "target_mode": "SEALED",
        "candidate_package_sha256": canonical_json_sha256(candidate_package),
        "n_blocks": len({str(cell["block_id"]) for cell in cells}),
        "n_tables": 0,
        "tables": {},
        "all_tables_pass": False,
    }
    if post_joint_provisional["shared_pass"] and post_joint_provisional["p3"]["passed"]:
        health_p3 = float(post_joint_provisional["p3"]["selected_dose"])
        health_p10 = (
            float(post_joint_provisional["p10"]["selected_dose"])
            if post_joint_provisional["p10"]["passed"]
            else None
        )
        health_worlds = build_world_tables(
            original, p3_dose=health_p3, p10_dose=health_p10
        )
        health_controls, health_specs = build_phase_control_family(
            original,
            p3_dose=health_p3,
            p10_dose=health_p10,
            n_controls=n_controls,
            seed=control_seed,
        )
        health_table_specs: dict[str, Mapping[str, Any]] = {
            world: {"family": "target", "control_id": None, "world": world}
            for world in health_worlds
        }
        health_table_specs.update(health_specs)
        measured_health = provider.measure_authorized_table_health(
            cells=cells,
            original=original,
            tables={**health_worlds, **health_controls},
            table_specs=health_table_specs,
            candidate_package=candidate_package,
            target_mode=("FULL_FACTORIAL" if health_p10 is not None else "P3_ONLY"),
        )
        measured_tables = measured_health.get("tables", {})
        p3_labels = {
            label
            for label, spec in health_table_specs.items()
            if label in {"Y00", "Y10"}
            or str(spec.get("family")) in {"p3_phase", "p3_axis"}
        }
        p3_tables_pass = bool(
            p3_labels
            and set(p3_labels).issubset(measured_tables)
            and all(bool(measured_tables[label].get("pass")) for label in p3_labels)
        )
        if not p3_tables_pass:
            for row in dose_rows["p3"].values():
                row["health_pass"] = False
                row["all_control_health_pass"] = False
            authorized_table_health = dict(measured_health)
            authorized_table_health["all_tables_pass"] = False
        elif health_p10 is not None and not bool(measured_health.get("all_tables_pass")):
            # A failed ancillary table cannot contaminate the valid exact P3 subset.
            # It also cannot trigger an outcome-adaptive lower P10 dose.
            for row in dose_rows["p10"].values():
                row["joint_health_pass"] = False
            authorized_table_health = {
                **dict(measured_health),
                "target_mode": "P3_ONLY",
                "n_tables": len(p3_labels),
                "tables": {
                    label: measured_tables[label] for label in sorted(p3_labels)
                },
                "all_tables_pass": True,
                "ancillary_full_factorial_health_failed": True,
            }
        else:
            authorized_table_health = dict(measured_health)

    post_health_provisional = evaluate_preflight(
        {"shared_gates": shared, "dose_rows": dose_rows},
        development_mde={
            "clean_psi": 0.0,
            "N3": 0.0,
            "N10": 0.0,
            "J": 0.0,
        },
    )
    if not post_health_provisional["shared_pass"] or not post_health_provisional["p3"]["passed"]:
        mde: Mapping[str, Any] = {}
        final = evaluate_preflight(
            {"shared_gates": shared, "dose_rows": dose_rows},
            development_mde=mde,
        )
    else:
        selected_p3 = float(post_health_provisional["p3"]["selected_dose"])
        selected_p10 = (
            float(post_health_provisional["p10"]["selected_dose"])
            if post_health_provisional["p10"]["passed"]
            else None
        )
        controls, specs = build_phase_control_family(
            original,
            p3_dose=selected_p3,
            p10_dose=selected_p10,
            n_controls=n_controls,
            seed=control_seed,
        )
        joint_mechanics = mechanical_audit(
            original,
            p3_dose=selected_p3,
            p10_dose=selected_p10,
            controls=controls,
            control_specs=specs,
        )
        if not joint_mechanics["passed"]:
            for row in dose_rows["p3"].values():
                row["mechanical_pass"] = False
            mde = {}
        else:
            mde = provider.development_mde(
                cells=cells,
                original=original,
                p3_dose=selected_p3,
                p10_dose=selected_p10,
            )
        final = evaluate_preflight(
            {"shared_gates": shared, "dose_rows": dose_rows},
            development_mde=mde,
        )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "stage": "development",
        "status": final["status"],
        "target_mode": final["target_mode"],
        "split_role": split_role,
        "dose_grid": list(DOSE_GRID),
        "bands": {"p3": list(P3_BINS_1026), "p10": list(P10_BINS_1026)},
        "n_controls_per_family": n_controls,
        "control_seed": control_seed,
        "dose_selection_used_phase_labels": False,
        "dose_selection_used_psi": False,
        "development_resolution_opened_after_dose_selection": bool(
            post_joint_provisional["shared_pass"]
            and post_joint_provisional["p3"]["passed"]
        ),
        "evaluation": final,
        "measurements": {
            "shared_gates": shared,
            "dose_rows": dose_rows,
            "p10_individual_selection": p10_individual,
            "p10_joint_step_down": joint_attempts,
            "authorized_table_health": dict(authorized_table_health),
            "development_mde": dict(mde),
        },
        "provider": dict(provider.metadata()),
    }


def _model_files(model: Path) -> tuple[Path, ...]:
    names = (
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    )
    files = tuple(model / name for name in names if (model / name).is_file())
    if not any(path.name in {"pytorch_model.bin", "model.safetensors"} for path in files):
        raise FileNotFoundError(f"no model weight file in {model}")
    if not any(path.name in {"tokenizer.json", "vocab.txt"} for path in files):
        raise FileNotFoundError(f"no tokenizer inventory in {model}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--candidate-package", required=True)
    parser.add_argument("--construction-audit", required=True)
    parser.add_argument("--construction-audit-manifest", required=True)
    parser.add_argument("--shared-gates", default=None)
    parser.add_argument("--fixture-calibration", required=True)
    parser.add_argument("--fixture-calibration-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split-role", default="development")
    parser.add_argument("--provider", default=None, help="optional MODULE:FACTORY")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--controls", type=int, default=MIN_STRUCTURED_CONTROLS)
    parser.add_argument("--control-seed", type=int, default=20260810)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.controls != MIN_STRUCTURED_CONTROLS:
        parser.error(f"--controls must equal the frozen value {MIN_STRUCTURED_CONTROLS}")
    output, manifest_path = output_paths(args.out)
    cohort_path = Path(args.cohort).expanduser().resolve()
    candidate_path = Path(args.candidate_package).expanduser().resolve()
    audit_path = Path(args.construction_audit).expanduser().resolve()
    audit_manifest_path = Path(
        args.construction_audit_manifest
    ).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if args.provider is not None:
        parser.error("custom providers are fixture-only and cannot authorize a target")
    validate_official_checkpoint(model_path)
    cohort = json.loads(cohort_path.read_text())
    candidate_package = json.loads(candidate_path.read_text())
    validate_candidate_package(candidate_package)
    construction_audit = json.loads(audit_path.read_text())
    construction_audit_manifest = json.loads(audit_manifest_path.read_text())
    construction_audit_provenance = validate_construction_audit_chain(
        construction_audit,
        audit_path=audit_path,
        manifest=construction_audit_manifest,
        manifest_path=audit_manifest_path,
    )
    construction_audit_pass = True
    shared_path = (
        Path(args.shared_gates).expanduser().resolve()
        if args.shared_gates
        else None
    )
    shared_source = json.loads(shared_path.read_text()) if shared_path else None
    if shared_source is not None and shared_source.get("schema") != EXPECTED_SHARED_GATE_SCHEMA:
        raise RuntimeError("wrong shared-gate source schema")
    calibration_path = Path(args.fixture_calibration).expanduser().resolve()
    calibration_manifest_path = Path(
        args.fixture_calibration_manifest
    ).expanduser().resolve()
    calibration = json.loads(calibration_path.read_text())
    calibration_manifest = json.loads(calibration_manifest_path.read_text())
    if (
        calibration_manifest.get("experiment_id")
        != "pbc-carrier-factorial-fixture-calibration-v1"
        or calibration_manifest.get("status") != "COMPLETE"
        or calibration_manifest.get("config", {}).get("stage") != "development"
        or not _manifest_binds_output(calibration_manifest, calibration_path)
    ):
        raise RuntimeError("fixture calibration lacks its immutable producer manifest")
    shared_source = {
        **(dict(shared_source) if isinstance(shared_source, Mapping) else {}),
        "schema": EXPECTED_SHARED_GATE_SCHEMA,
        "calibration": calibration,
        "calibration_provenance": {
            "artifact_sha256": sha256_file(calibration_path),
            "manifest_sha256": sha256_file(calibration_manifest_path),
            "experiment_id": calibration_manifest["experiment_id"],
        },
    }
    provider = load_provider(
        args.provider,
        model_path=model_path,
        device=args.device,
        batch_size=args.batch,
    )
    result = execute_preflight(
        cohort=cohort,
        candidate_package=candidate_package,
        provider=provider,
        split_role=args.split_role,
        shared_gate_source=shared_source,
        construction_audit_pass=construction_audit_pass,
        n_controls=args.controls,
        control_seed=args.control_seed,
    )
    result["construction_audit_provenance"] = construction_audit_provenance
    result["raw_sha256"] = canonical_json_sha256(preflight_raw_view(result))
    result["analysis_sha256"] = canonical_json_sha256(
        preflight_analysis_view(result)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    code_files = conformance_code_files(REPO)
    pinned_paths = (
        cohort_path,
        candidate_path,
        audit_path,
        audit_manifest_path,
        protocol_path,
        calibration_path,
        calibration_manifest_path,
        *code_files,
        *_model_files(model_path),
        *((shared_path,) if shared_path is not None else ()),
        output,
    )
    manifest = ExperimentManifest.create(
        "pbc-carrier-factorial-preflight-v1",
        sys.argv,
        repo=REPO,
        status=result["status"],
        config={
            "stage": "development",
            "split_role": args.split_role,
            "dose_grid": list(DOSE_GRID),
            "bands": result["bands"],
            "n_controls_per_family": args.controls,
            "control_seed": args.control_seed,
            "target_mode": result["target_mode"],
        },
        seeds={"structured_controls": args.control_seed},
    )
    for path in pinned_paths[:-1]:
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "target_mode": result["target_mode"],
                "selected_p3_dose": result["evaluation"]["p3"]["selected_dose"],
                "selected_p10_dose": result["evaluation"]["p10"]["selected_dose"],
                "seal_status": "PENDING_INDEPENDENT_PREFLIGHT_AUDIT",
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
