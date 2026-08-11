#!/usr/bin/env python
"""Run the outcome-fresh development preflight for P3 register relocation.

This runner may read only the deterministic 120-block development subset.  It emits
raw model endpoints and construction/health measurements; it does not interpret the
relocation result and cannot open the Phase-III reserve.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_register_relocation import (  # noqa: E402
    REGISTERS,
    relocation_mechanics,
)
from gfmcd.splicing.pbc_register_runner import (  # noqa: E402
    _linear_masked_margin,
    carrier_edit_delta,
    embedding_delta_context,
    evaluate_register_lattice,
    position_embedding_module,
    register_cell_from_mapping,
    score_cell_batch,
)
from gfmcd.splicing.pbc_register_stage import (  # noqa: E402
    DECOY_SPAN_TOKEN,
    P10_BINS,
    P3_BINS,
    REGISTER_CONTROL_SEEDS,
    REGISTER_DIRECTION,
    REGISTER_PREFLIGHT_SCHEMA,
    SPAN_TOKEN,
    validate_phase2_authorization,
)
from gfmcd.splicing.positional_spectrum import band_component  # noqa: E402
from run_carrier_factorial_preflight import SpliceBertPreflightProvider  # noqa: E402
from gfmcd.splicing.positional_boundary_coherence import (  # noqa: E402
    sha256_payload,
    validate_candidate_package,
)


COHORT_SCHEMA = "gfmcd.positional_boundary_coherence.cohort.v1"
RAW_STATUS = "DEVELOPMENT_SCORED_UNANALYZED"
DEVELOPMENT_BLOCKS = 120
DEVELOPMENT_SELECTION_SALT = "pbc-p3-register-development-v1"
CHECKPOINT_SHA256 = "2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d"
EXPECTED_POSITION_SHAPE = (1026, 512)
CANDIDATE_AUDIT_SCHEMA = "gfmcd.positional_boundary_coherence.candidate_census_audit.v1"
TABLE_HEALTH_SCHEMA = "gfmcd.pbc.carrier_factorial_table_health.v1"


class RegisterPreflightProvider(Protocol):
    """Injectable model provider used by synthetic tests and the official runner."""

    def measure(
        self,
        cells: Sequence[Mapping[str, Any]],
        *,
        joint: bool,
        p10_dose: float,
        candidate_package: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def metadata(self) -> Mapping[str, Any]: ...


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    if output == manifest:
        raise ValueError("preflight and manifest paths must be distinct")
    return output, manifest


def validate_candidate_construction_audit(
    candidate_package: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, str]:
    """Require the independent construction audit for this exact candidate census."""

    validate_candidate_package(candidate_package)
    checks = audit.get("checks")
    candidate_hash = sha256_payload(candidate_package)
    if not (
        audit.get("schema") == CANDIDATE_AUDIT_SCHEMA
        and audit.get("stage") == "construction"
        and audit.get("status") == "AUDIT_PASS"
        and audit.get("passed") is True
        and isinstance(checks, Mapping)
        and checks
        and all(value is True for value in checks.values())
        and audit.get("candidate_census_sha256") == candidate_hash
        and audit.get("candidate_content_sha256")
        == candidate_package.get("content_sha256")
    ):
        raise RuntimeError("candidate package lacks its matching construction audit")
    return {
        "candidate_package_canonical_sha256": candidate_hash,
        "candidate_content_sha256": str(candidate_package.get("content_sha256")),
        "candidate_audit_status": "AUDIT_PASS",
    }


def _block_rank(block_id: str) -> tuple[str, str]:
    digest = sha256(
        f"{DEVELOPMENT_SELECTION_SALT}\0{block_id}".encode("utf-8")
    ).hexdigest()
    return digest, block_id


def development_cells(
    cohort: Mapping[str, Any],
    *,
    split_role: str = "development",
    expected_blocks: int = DEVELOPMENT_BLOCKS,
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any]]:
    """Select and validate exactly 120 complete development blocks before model access."""

    if cohort.get("schema") != COHORT_SCHEMA:
        raise RuntimeError("wrong positional-boundary-coherence cohort schema")
    if split_role != "development":
        raise RuntimeError("register preflight may read only the development role")
    candidates = [
        row
        for row in cohort.get("cells", [])
        if isinstance(row, Mapping) and str(row.get("split_role")) == split_role
    ]
    by_block: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidates:
        by_block.setdefault(str(row.get("block_id") or ""), []).append(row)
    if len(by_block) < int(expected_blocks):
        raise RuntimeError(
            f"register preflight needs at least {expected_blocks} development blocks; "
            f"found {len(by_block)}"
        )
    expected_lattice = {(donor, acceptor) for donor in range(3) for acceptor in range(3)}
    for block_id, local in by_block.items():
        if not block_id or len(local) != 9:
            raise RuntimeError(f"development block {block_id!r} is not a nine-cell lattice")
        observed = {
            (int(row["donor_phase"]), int(row["acceptor_phase"])) for row in local
        }
        if observed != expected_lattice:
            raise RuntimeError(f"development block {block_id!r} lacks the complete 3x3 lattice")
    selected_ids = tuple(
        sorted(by_block, key=_block_rank)[: int(expected_blocks)]
    )
    selected = tuple(
        row
        for block_id in selected_ids
        for row in sorted(
            by_block[block_id],
            key=lambda value: (int(value["donor_phase"]), int(value["acceptor_phase"])),
        )
    )
    parsed = tuple(register_cell_from_mapping(row) for row in selected)
    if any(
        len(cell.intact_sequence) != 1022
        or cell.donor_position != 392
        or cell.acceptor_position != 512
        for cell in parsed
    ):
        raise RuntimeError("development cell drifted from the frozen 1022-nt/120-nt geometry")
    return selected, {
        "method": "sha256_rank_whole_block_identity",
        "salt": DEVELOPMENT_SELECTION_SALT,
        "candidate_blocks": len(by_block),
        "selected_blocks": len(selected_ids),
        "selected_block_ids": list(selected_ids),
        "selected_block_ids_sha256": sha256(
            ("\n".join(selected_ids) + "\n").encode("utf-8")
        ).hexdigest(),
    }


def _endpoint_map(
    rows: Sequence[Mapping[str, Any]], *, component: str = "p3"
) -> dict[tuple[str, int, int, int], tuple[float, float, float]]:
    output: dict[tuple[str, int, int, int], tuple[float, float, float]] = {}
    for row in rows:
        if str(row.get("component")) != component or str(row.get("p10_state")) != "intact":
            continue
        key = (
            str(row["block_id"]),
            int(row["register"]),
            int(row["donor_phase"]),
            int(row["acceptor_phase"]),
        )
        output[key] = (
            float(row["intact_margin"]),
            float(row["corrupt_margin"]),
            float(row["coupling"]),
        )
    return output


def _maximum_endpoint_error(
    expected: Mapping[tuple[str, int, int, int], tuple[float, float, float]],
    observed: Mapping[tuple[str, int, int, int], tuple[float, float, float]],
) -> float:
    if set(expected) != set(observed):
        raise RuntimeError("replay changed the target register-cell support")
    return max(
        abs(left - right)
        for key in expected
        for left, right in zip(expected[key], observed[key], strict=True)
    )


class OfficialCheckpointRegisterProvider:
    """Official SpliceBERT provider with health, reader, and replay measurements."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None,
        batch_size: int,
        eager: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.backend = SpliceBertPreflightProvider(
            model_path, device=device, batch_size=batch_size, eager=eager
        )
        self.batch_size = int(batch_size)

    def metadata(self) -> Mapping[str, Any]:
        value = self.backend.metadata()
        return dict(value) if isinstance(value, Mapping) else {}

    def _intervention_deltas(self) -> dict[str, dict[int, torch.Tensor]]:
        embedding = position_embedding_module(self.backend.adapter.model)
        weight = embedding.weight.detach().cpu()
        from gfmcd.splicing.p3_phase_surgery import (
            fourier_band_component,
            signed_permutation_component,
        )

        p3 = fourier_band_component(weight, P3_BINS)
        components = {
            "p3": p3,
            **{
                f"p3_signed_{index:02d}": signed_permutation_component(p3, seed=seed)
                for index, seed in enumerate(REGISTER_CONTROL_SEEDS)
            },
        }
        target = {
            register: carrier_edit_delta(
                p3,
                p3_bins=P3_BINS,
                register=register,
                start=SPAN_TOKEN[0],
                end=SPAN_TOKEN[1],
                direction=REGISTER_DIRECTION,
            )
            for register in REGISTERS
        }
        output: dict[str, dict[int, torch.Tensor]] = {"p3": target}
        for label, component in components.items():
            if label == "p3":
                continue
            output[label] = {
                register: carrier_edit_delta(
                    component,
                    p3_bins=P3_BINS,
                    register=register,
                    start=SPAN_TOKEN[0],
                    end=SPAN_TOKEN[1],
                    direction=REGISTER_DIRECTION,
                    match_p3_rms=(
                        float(torch.sqrt(torch.mean(target[register].float().square())))
                        if register != 0
                        else None
                    ),
                )
                for register in REGISTERS
            }
        output["p3_common_mode"] = {
            register: carrier_edit_delta(
                p3,
                p3_bins=P3_BINS,
                register=register,
                start=SPAN_TOKEN[0],
                end=SPAN_TOKEN[1],
                direction=REGISTER_DIRECTION,
                phase_mode="common",
                match_p3_rms=(
                    float(torch.sqrt(torch.mean(target[register].float().square())))
                    if register != 0
                    else None
                ),
            )
            for register in REGISTERS
        }
        output["p3_decoy_span"] = {
            register: carrier_edit_delta(
                p3,
                p3_bins=P3_BINS,
                register=register,
                start=DECOY_SPAN_TOKEN[0],
                end=DECOY_SPAN_TOKEN[1],
                direction=REGISTER_DIRECTION,
                match_p3_rms=(
                    float(torch.sqrt(torch.mean(target[register].float().square())))
                    if register != 0
                    else None
                ),
            )
            for register in REGISTERS
        }
        return output

    def _replay_checks(
        self,
        cells: Sequence[Mapping[str, Any]],
        raw_rows: Sequence[Mapping[str, Any]],
        deltas: Mapping[int, torch.Tensor],
    ) -> dict[str, float]:
        parsed = tuple(register_cell_from_mapping(row) for row in cells)
        expected = _endpoint_map(raw_rows)
        reversed_cells = tuple(reversed(parsed))
        alternate_batch = max(1, self.batch_size - 1)
        replay: dict[tuple[str, int, int, int], tuple[float, float, float]] = {}
        corruption_replay: dict[
            tuple[str, int, int, int], tuple[float, float, float]
        ] = {}
        table_replay: dict[tuple[str, int, int, int], tuple[float, float, float]] = {}
        embedding = position_embedding_module(self.backend.adapter.model)
        for register in REGISTERS:
            delta = deltas[register]
            for start in range(0, len(reversed_cells), alternate_batch):
                chunk = reversed_cells[start : start + alternate_batch]
                scores = score_cell_batch(self.backend.adapter, chunk, delta=delta)
                for cell, score in zip(chunk, scores, strict=True):
                    key = (
                        cell.block_id,
                        register,
                        cell.donor_phase,
                        cell.acceptor_phase,
                    )
                    replay[key] = (
                        float(score["intact_margin"]),
                        float(score["corrupt_margin"]),
                        float(score["coupling"]),
                    )
            for start in range(0, len(parsed), self.batch_size):
                chunk = parsed[start : start + self.batch_size]
                positions = [cell.donor_position for cell in chunk]
                with torch.inference_mode(), embedding_delta_context(
                    embedding, [delta] * len(chunk)
                ):
                    corrupt = _linear_masked_margin(
                        self.backend.adapter,
                        [cell.acceptor_ac_sequence for cell in chunk],
                        positions,
                    )
                    intact = _linear_masked_margin(
                        self.backend.adapter,
                        [cell.intact_sequence for cell in chunk],
                        positions,
                    )
                for cell, left, right in zip(chunk, intact, corrupt, strict=True):
                    key = (
                        cell.block_id,
                        register,
                        cell.donor_phase,
                        cell.acceptor_phase,
                    )
                    corruption_replay[key] = (
                        float(left), float(right), float(left - right)
                    )
            original = self.backend.position_table()
            edited = original + delta.detach().cpu().double().numpy()
            with self.backend._using_table(edited):
                intact_values = self.backend._score_state(cells, "intact")
                corrupt_values = self.backend._score_state(cells, "ac")
            for row, left, right in zip(cells, intact_values, corrupt_values, strict=True):
                key = (
                    str(row["block_id"]),
                    register,
                    int(row["donor_phase"]),
                    int(row["acceptor_phase"]),
                )
                table_replay[key] = (float(left), float(right), float(left - right))
        batch_error = _maximum_endpoint_error(expected, replay)
        corruption_error = _maximum_endpoint_error(expected, corruption_replay)
        forward_error = _maximum_endpoint_error(expected, table_replay)
        return {
            "batch_order_replay_max_abs": batch_error,
            "corruption_order_replay_max_abs": corruption_error,
            "ramp_forward_replay_max_abs": forward_error,
            "order_replay_max_abs": max(batch_error, corruption_error, forward_error),
        }

    def _construction_checks(
        self,
        rows: Sequence[Mapping[str, Any]],
        through_model: Mapping[str, Any],
        deltas: Mapping[int, torch.Tensor],
    ) -> dict[str, Any]:
        original = self.backend.position_table()
        construction = relocation_mechanics(
            original,
            P3_BINS,
            start=SPAN_TOKEN[0],
            end=SPAN_TOKEN[1],
            direction=REGISTER_DIRECTION,
        )
        embedding = position_embedding_module(self.backend.adapter.model)
        # Independent float64 torch reconstruction avoids comparing the protocol's
        # numerical invariant against a deliberately float32 scoring helper.
        torch_values = embedding.weight.detach().cpu().double()
        torch_spectrum = torch.fft.rfft(
            torch_values - torch_values.mean(dim=0, keepdim=True), dim=0
        )
        torch_selected = torch.zeros_like(torch_spectrum)
        torch_selected[list(P10_BINS)] = torch_spectrum[list(P10_BINS)]
        torch_p10 = torch.fft.irfft(
            torch_selected, n=torch_values.shape[0], dim=0
        ).numpy()
        numpy_p10 = band_component(original, P10_BINS)
        p10_scale = max(float(np.max(np.abs(numpy_p10))), np.finfo(float).tiny)
        p10_reconstruction_error = float(np.max(np.abs(torch_p10 - numpy_p10)) / p10_scale)
        leakage: dict[str, float] = {}
        for register in REGISTERS:
            edited = original + deltas[register].detach().cpu().double().numpy()
            leaked = band_component(edited, P10_BINS) - numpy_p10
            leakage[str(register)] = float(np.sqrt(np.mean(np.square(leaked))))

        rms = through_model.get("delta_rms")
        rms_errors: list[float] = []
        if isinstance(rms, Mapping) and isinstance(rms.get("p3"), Mapping):
            for register in (-1, 1):
                target = float(rms["p3"]["intact"][str(register)])
                for label in (
                    *(f"p3_signed_{index:02d}" for index in range(31)),
                    "p3_common_mode",
                    "p3_decoy_span",
                ):
                    observed = float(rms[label]["intact"][str(register)])
                    rms_errors.append(abs(observed - target) / max(target, np.finfo(float).tiny))
        rms_error = max(rms_errors, default=math.inf)
        # Independent operator-level handedness audit on a two-channel pure P3 tone.
        # This calls the actual torch intervention and measures phase from atan2; it
        # does not use preferred_delta() to construct the observed displacement.
        length = 12
        coordinates = torch.arange(length, dtype=torch.float64)
        pure = torch.stack(
            (
                torch.cos(2 * torch.pi * coordinates / 3),
                torch.sin(2 * torch.pi * coordinates / 3),
            ),
            dim=1,
        )
        handedness: dict[str, dict[str, Any]] = {}
        handedness_pass = True
        for register in REGISTERS:
            delta = carrier_edit_delta(
                pure,
                p3_bins=(length // 3,),
                register=register,
                start=0,
                end=4,
                direction=REGISTER_DIRECTION,
            )
            edited = pure + delta
            observed = np.unwrap(
                np.arctan2(
                    edited[:4, 1].detach().cpu().numpy(),
                    edited[:4, 0].detach().cpu().numpy(),
                )
                - (2 * np.pi * np.arange(4) / 3)
            )
            expected = -2 * np.pi * register * np.linspace(0.0, 1.0, 4) / 3
            error = float(np.max(np.abs(observed - expected)))
            observed_source_displacement = int(
                round(float(observed[-1]) / (2 * np.pi / 3))
            )
            predicted_delta = observed_source_displacement % 3
            row_pass = bool(
                error <= 1e-6
                and predicted_delta == {-1: 1, 0: 0, 1: 2}[register]
                and np.all(np.diff(observed) >= -1e-9)
                if register < 0
                else (
                    error <= 1e-6
                    and predicted_delta == {-1: 1, 0: 0, 1: 2}[register]
                    and np.all(np.diff(observed) <= 1e-9)
                )
            )
            handedness[str(register)] = {
                "register": register,
                "measured_acceptor_phase_radians": float(observed[-1]),
                "requested_acceptor_phase_radians": float(expected[-1]),
                "phase_profile_max_abs_error_radians": error,
                "observed_source_displacement": observed_source_displacement,
                "predicted_annotated_phase_delta": predicted_delta,
                "passed": row_pass,
            }
            handedness_pass &= row_pass

        p3_numpy = band_component(original, P3_BINS)
        residual_before = original - p3_numpy
        residual_errors: dict[str, float] = {}
        p10_retention_errors: dict[str, float] = {}
        residual_p10_before = band_component(residual_before, P10_BINS)
        residual_p10_scale = max(
            float(np.max(np.abs(residual_p10_before))), np.finfo(float).tiny
        )
        for register in REGISTERS:
            actual_delta = deltas[register].detach().cpu().double().numpy()
            edited = original + actual_delta
            shifted_p3 = p3_numpy + actual_delta
            residual_after = edited - shifted_p3
            residual_errors[str(register)] = float(
                np.max(np.abs(residual_after - residual_before))
            )
            retained = band_component(residual_after, P10_BINS)
            p10_retention_errors[str(register)] = float(
                np.max(np.abs(retained - residual_p10_before)) / residual_p10_scale
            )
        residual_error = max(residual_errors.values())
        explicit_p10_error = max(p10_retention_errors.values())
        return {
            "construction_invariants": construction,
            "direction": REGISTER_DIRECTION,
            "handedness": handedness,
            "handedness_passed": bool(handedness_pass),
            "p10_independent_reconstruction_relative_max_abs": p10_reconstruction_error,
            "pointwise_non_p3_residual_max_abs": residual_error,
            "pointwise_non_p3_residual_by_register": residual_errors,
            "p10_explicit_component_retention_relative_max_abs": explicit_p10_error,
            "p10_explicit_component_retention_by_register": p10_retention_errors,
            "windowed_ramp_global_p10_leakage_rms": leakage,
            "rms_match_max_relative_error": float(rms_error),
            "register_zero_no_hook_max_abs": float(
                through_model["register_zero_no_hook_max_abs"]
            ),
            "passed": bool(
                construction["passed"]
                and handedness_pass
                and p10_reconstruction_error <= 1e-10
                and residual_error <= 1e-10
                and explicit_p10_error <= 1e-10
                and rms_error <= 1e-6
                and float(through_model["register_zero_no_hook_max_abs"]) <= 1e-8
            ),
        }

    def _health_checks(
        self,
        cells: Sequence[Mapping[str, Any]],
        deltas: Mapping[str, Mapping[int, torch.Tensor]],
        *,
        joint: bool,
        p10_dose: float,
        candidate_package: Mapping[str, Any],
    ) -> dict[str, Any]:
        original = self.backend.position_table()
        baseline_reader = self.backend._p10_attention_scores(cells, original)
        denominator = max(float(baseline_reader.mean()), np.finfo(float).tiny)
        p10_component = band_component(original, P10_BINS)
        table_states = ("intact", "attenuated") if joint else ("intact",)
        tables: dict[str, np.ndarray] = {}
        specs: dict[str, dict[str, Any]] = {}
        for component, registers in deltas.items():
            family = (
                "target"
                if component == "p3"
                else "signed_hidden_axis"
                if component.startswith("p3_signed_")
                else "common_mode"
                if component == "p3_common_mode"
                else "decoy_span"
            )
            control_id = (
                int(component.rsplit("_", 1)[1]) + 1
                if component.startswith("p3_signed_")
                else None
            )
            for state in table_states:
                base = (
                    original
                    if state == "intact"
                    else original - float(p10_dose) * p10_component
                )
                for register in REGISTERS:
                    table_label = f"{component}__{state}__r{register}"
                    tables[table_label] = (
                        base + registers[register].detach().cpu().double().numpy()
                    )
                    specs[table_label] = {
                        "family": family,
                        "component": component,
                        "control_id": control_id,
                        "p10_state": state,
                        "register": register,
                        "world": "register_p3" if state == "intact" else "register_joint",
                    }
        authorized = self.backend.measure_authorized_table_health(
            cells=cells,
            original=original,
            tables=tables,
            table_specs=specs,
            candidate_package=candidate_package,
            target_mode="REGISTER_JOINT" if joint else "REGISTER_P3_ONLY",
        )
        if (
            authorized.get("schema") != TABLE_HEALTH_SCHEMA
            or int(authorized.get("n_blocks", -1)) != DEVELOPMENT_BLOCKS
            or int(authorized.get("n_tables", -1)) != len(tables)
            or set(authorized.get("tables", {})) != set(tables)
        ):
            raise RuntimeError("full inherited table-health assay returned an incomplete family")
        states: dict[str, dict[str, Any]] = {}
        for label, table in tables.items():
            reader = (
                baseline_reader
                if np.array_equal(table, original)
                else self.backend._p10_attention_scores(cells, table)
            )
            retention = float(reader.mean()) / denominator
            phase2_pass = bool(authorized["tables"][label]["pass"])
            states[label] = {
                "spec": dict(specs[label]),
                "phase2_table_health_pass": phase2_pass,
                "p10_reader_score": float(reader.mean()),
                "p10_reader_retention": retention,
                "passed": bool(phase2_pass and retention >= 0.80),
            }
        minimum = min(float(row["p10_reader_retention"]) for row in states.values())
        control_states = {
            label: row
            for label, row in states.items()
            if row["spec"]["family"] != "target"
        }
        joint_states = {
            label: row
            for label, row in states.items()
            if row["spec"]["p10_state"] == "attenuated"
        }
        return {
            "schema": "gfmcd.pbc.p3_register_table_health.v1",
            "stage": "development",
            "authorized_table_health": authorized,
            "baseline_p10_reader_score": float(baseline_reader.mean()),
            "states": states,
            "control_states": control_states,
            "all_control_states_passed": bool(
                control_states and all(row["passed"] for row in control_states.values())
            ),
            "all_required_states_passed": bool(
                states and all(row["passed"] for row in states.values())
            ),
            "minimum_p10_reader_retention": minimum,
            "joint_state_passed": bool(
                joint and joint_states and all(row["passed"] for row in joint_states.values())
            ),
            "joint_state_reason": (
                "all exact P10-attenuated register tables passed full inherited health"
                if joint and joint_states and all(row["passed"] for row in joint_states.values())
                else "one or more exact P10-attenuated register tables failed health"
                if joint
                else "optional joint arm not requested"
            ),
        }

    def measure(
        self,
        cells: Sequence[Mapping[str, Any]],
        *,
        joint: bool,
        p10_dose: float,
        candidate_package: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        parsed = tuple(register_cell_from_mapping(row) for row in cells)
        rows, through_model = evaluate_register_lattice(
            self.backend.adapter,
            parsed,
            p3_bins=P3_BINS,
            span_start_token=SPAN_TOKEN[0],
            span_end_token=SPAN_TOKEN[1],
            decoy_span_token=DECOY_SPAN_TOKEN,
            control_seeds=REGISTER_CONTROL_SEEDS,
            batch_size=self.batch_size,
            direction=REGISTER_DIRECTION,
            joint=joint,
            p10_bins=P10_BINS,
            p10_dose=float(p10_dose),
        )
        deltas = self._intervention_deltas()
        construction = self._construction_checks(rows, through_model, deltas["p3"])
        replay = self._replay_checks(cells, rows, deltas["p3"])
        construction.update(replay)
        construction["passed"] = bool(
            construction["passed"] and replay["order_replay_max_abs"] <= 1e-8
        )
        return {
            "rows": rows,
            "mechanics": {**through_model, **construction},
            "health": self._health_checks(
                cells,
                deltas,
                joint=joint,
                p10_dose=p10_dose,
                candidate_package=candidate_package,
            ),
        }


def load_provider(
    specification: str | None,
    *,
    model_path: str | Path,
    device: str | None,
    batch_size: int,
) -> RegisterPreflightProvider:
    if not specification:
        return OfficialCheckpointRegisterProvider(
            model_path, device=device, batch_size=batch_size, eager=True
        )
    try:
        module_name, attribute = specification.split(":", 1)
    except ValueError as exc:
        raise ValueError("provider must be MODULE:FACTORY") from exc
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
    provider: RegisterPreflightProvider,
    phase2_authorization: Mapping[str, Any],
    candidate_package: Mapping[str, Any],
    candidate_construction_audit: Mapping[str, Any],
    include_p10: bool = False,
    p10_dose: float = 0.0,
) -> dict[str, Any]:
    """Measure the exact development lattice without inferential adjudication."""

    p3_authorized = phase2_authorization.get("phase2_p3_authorized")
    joint_authorized = phase2_authorization.get("phase2_joint_authorized")
    if type(p3_authorized) is not bool or type(joint_authorized) is not bool:
        raise RuntimeError("Phase-II authorization flags must be explicit booleans")
    if not p3_authorized:
        raise RuntimeError("Phase II did not authorize P3 register development")
    if include_p10 and not joint_authorized:
        raise RuntimeError("Phase II did not authorize the optional P10 register arm")
    if include_p10 and not 0 < float(p10_dose) <= 1:
        raise RuntimeError("joint register development needs the frozen Phase-II P10 dose")
    if not include_p10 and float(p10_dose) != 0:
        raise RuntimeError("P3-only preflight must not import a P10 dose")
    cells, selection = development_cells(cohort)
    construction_binding = validate_candidate_construction_audit(
        candidate_package, candidate_construction_audit
    )
    measured = provider.measure(
        cells,
        joint=include_p10,
        p10_dose=float(p10_dose),
        candidate_package=candidate_package,
    )
    rows = measured.get("rows")
    mechanics = measured.get("mechanics")
    health = measured.get("health")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("register provider emitted no raw rows")
    if not isinstance(mechanics, Mapping) or not isinstance(health, Mapping):
        raise RuntimeError("register provider omitted mechanics or health")
    return {
        "schema": REGISTER_PREFLIGHT_SCHEMA,
        "status": RAW_STATUS,
        "stage": "development",
        "mode": "JOINT" if include_p10 else "P3_ONLY",
        "split_role": "development",
        "n_blocks": DEVELOPMENT_BLOCKS,
        "n_cells": DEVELOPMENT_BLOCKS * 9,
        "selection": selection,
        "authorization": dict(phase2_authorization),
        "construction_binding": construction_binding,
        "design": {
            "p3_bins": list(P3_BINS),
            "p10_bins": list(P10_BINS),
            "registers": list(REGISTERS),
            "direction": REGISTER_DIRECTION,
            "span_token": list(SPAN_TOKEN),
            "decoy_span_token": list(DECOY_SPAN_TOKEN),
            "control_seeds": list(REGISTER_CONTROL_SEEDS),
            "p10_attenuation_dose": float(p10_dose) if include_p10 else None,
            "candidate_package_canonical_sha256": construction_binding[
                "candidate_package_canonical_sha256"
            ],
        },
        "mechanics": dict(mechanics),
        "health": dict(health),
        "model": dict(provider.metadata()),
        "rows": rows,
    }


def _checkpoint_path(path: Path) -> Path:
    return path / "pytorch_model.bin" if path.is_dir() else path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--candidate-package", required=True)
    parser.add_argument("--construction-audit", required=True)
    parser.add_argument("--phase2-analysis", required=True)
    parser.add_argument("--phase2-audit", required=True)
    parser.add_argument("--phase2-target")
    parser.add_argument("--include-p10", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output, manifest_path = output_paths(args.out)
    cohort_path = Path(args.cohort).expanduser().resolve()
    candidate_path = Path(args.candidate_package).expanduser().resolve()
    construction_audit_path = Path(args.construction_audit).expanduser().resolve()
    phase2_analysis_path = Path(args.phase2_analysis).expanduser().resolve()
    phase2_audit_path = Path(args.phase2_audit).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    checkpoint = _checkpoint_path(model_path)
    if not checkpoint.is_file() or sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("official SpliceBERT.1024nt checkpoint hash mismatch")
    cohort = json.loads(cohort_path.read_text())
    candidate_package = json.loads(candidate_path.read_text())
    construction_audit = json.loads(construction_audit_path.read_text())
    validate_candidate_construction_audit(candidate_package, construction_audit)
    phase2_analysis = json.loads(phase2_analysis_path.read_text())
    phase2_audit = json.loads(phase2_audit_path.read_text())
    authorization = validate_phase2_authorization(
        phase2_analysis,
        phase2_audit,
        analysis_sha256=sha256_file(phase2_analysis_path),
    )
    p10_dose = 0.0
    extra_inputs: list[Path] = []
    if args.include_p10:
        if not args.phase2_target:
            raise RuntimeError("--include-p10 requires --phase2-target")
        phase2_target_path = Path(args.phase2_target).expanduser().resolve()
        phase2_target = json.loads(phase2_target_path.read_text())
        doses = phase2_target.get("selected_doses")
        if not isinstance(doses, Mapping) or doses.get("p10") is None:
            raise RuntimeError("Phase-II target lacks its frozen P10 dose")
        p10_dose = float(doses["p10"])
        extra_inputs.append(phase2_target_path)

    # Selection and geometry validation precede the first checkpoint/model load.
    development_cells(cohort)
    provider = load_provider(
        args.provider,
        model_path=model_path,
        device=args.device,
        batch_size=args.batch,
    )
    result = execute_preflight(
        cohort=cohort,
        provider=provider,
        phase2_authorization=authorization,
        candidate_package=candidate_package,
        candidate_construction_audit=construction_audit,
        include_p10=args.include_p10,
        p10_dose=p10_dose,
    )
    result["sources"] = {
        "cohort": file_record(cohort_path),
        "candidate_package": file_record(candidate_path),
        "candidate_construction_audit": file_record(construction_audit_path),
        "phase2_analysis": file_record(phase2_analysis_path),
        "phase2_audit": file_record(phase2_audit_path),
        "checkpoint": file_record(checkpoint),
        **(
            {"phase2_target": file_record(extra_inputs[0])}
            if extra_inputs
            else {}
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        experiment_id="pbc-p3-register-development-preflight",
        command=sys.argv,
        repo=REPO,
        status=RAW_STATUS,
        config={
            "schema": REGISTER_PREFLIGHT_SCHEMA,
            "stage": "development",
            "mode": result["mode"],
            "n_blocks": DEVELOPMENT_BLOCKS,
            "batch_size": args.batch,
            "design": result["design"],
        },
        seeds={
            f"signed_control_{index:02d}": int(seed)
            for index, seed in enumerate(REGISTER_CONTROL_SEEDS)
        },
        inputs=[
            file_record(path)
            for path in (
                cohort_path,
                candidate_path,
                construction_audit_path,
                phase2_analysis_path,
                phase2_audit_path,
                checkpoint,
                *extra_inputs,
            )
        ],
        outputs=[file_record(output)],
        notes=[
            "Development-only raw scoring; no Phase-III reserve path is read or authorized here."
        ],
    )
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": result["mode"],
                "n_blocks": result["n_blocks"],
                "n_rows": len(result["rows"]),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
