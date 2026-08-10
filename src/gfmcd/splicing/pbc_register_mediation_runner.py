"""Model-backed execution kernel for frozen P3 register mediation.

The kernel consumes a pre-reserve candidate freeze and emits only primitive native
masked-donor margins.  It applies mixed head/MLP sets at their actual hook sites,
preserves sequence and AG/AC state, and records hashes of every tensor stream used by
an intervention.  Statistical interpretation remains in ``pbc_register_mediation``.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

import torch

from gfmcd.splicing.pbc_register_candidate_runner import (
    HfCandidateMeasurementProvider,
    PATCH_OPERATION_INJECT,
    PATCH_OPERATION_REPLACE,
    _coupling_rows,
    _subtract_tensors,
    sequence_sha256,
    tensor_mapping_sha256,
)
from gfmcd.splicing.pbc_register_candidates import (
    CANDIDATE_FREEZE_SCHEMA,
    TOKEN_SUPPORT,
)
from gfmcd.splicing.pbc_register_mediation import (
    LEARNED_SET,
    LEARNED_STATES,
    NULL_INTERVENTION_STATE,
    PATH_NAMES,
    STATE_OPERATIONS,
    STATE_TABLES,
    expected_null_labels,
    validate_candidate_freeze,
)
from gfmcd.splicing.pbc_register_relocation import REGISTERS, phase_delta
from gfmcd.splicing.pbc_register_runner import RegisterCell


MEDIATION_SHARD_SCHEMA = "gfmcd.pbc.p3_register_mediation_target_shard.v1"


def _component_ids(spec: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(row["component_id"]) for row in spec["components"])


def _digest_pair(
    intact: Mapping[str, torch.Tensor], corrupt: Mapping[str, torch.Tensor]
) -> tuple[str, str, str]:
    intact_digest = tensor_mapping_sha256(intact)
    corrupt_digest = tensor_mapping_sha256(corrupt)
    combined = sha256(
        bytes.fromhex(intact_digest) + bytes.fromhex(corrupt_digest)
    ).hexdigest()
    return intact_digest, corrupt_digest, combined


def _patch_provenance(
    *,
    components: Sequence[Mapping[str, object]],
    intact_source: Mapping[str, torch.Tensor],
    corrupt_source: Mapping[str, torch.Tensor],
    operation: str,
    source_register: int,
    source_baseline_register: int | None,
) -> dict[str, object]:
    intact, corrupt, combined = _digest_pair(intact_source, corrupt_source)
    return {
        "components": [str(row["component_id"]) for row in components],
        "hook_sites": sorted({str(row["hook_site"]) for row in components}),
        "operation": operation,
        "token_support": list(TOKEN_SUPPORT),
        "source_register": int(source_register),
        "source_baseline_register": source_baseline_register,
        "intact_patch_tensor_sha256": intact,
        "acceptor_ac_patch_tensor_sha256": corrupt,
        "combined_patch_tensor_sha256": combined,
    }


def _source_identity_fields(
    cell: RegisterCell, *, patched: bool
) -> dict[str, object]:
    identity = f"{cell.block_id}:{cell.donor_phase}:{cell.acceptor_phase}"
    intact_hash = sequence_sha256(cell.intact_sequence)
    corrupt_hash = sequence_sha256(cell.acceptor_ac_sequence)
    return {
        "intact_host_cell_id": identity,
        "acceptor_ac_host_cell_id": identity,
        "intact_host_sequence_sha256": intact_hash,
        "acceptor_ac_host_sequence_sha256": corrupt_hash,
        "intact_source_cell_id": identity if patched else None,
        "acceptor_ac_source_cell_id": identity if patched else None,
        "intact_source_sequence_sha256": intact_hash if patched else None,
        "acceptor_ac_source_sequence_sha256": corrupt_hash if patched else None,
    }


def _mediation_rows(
    cells: Sequence[RegisterCell],
    *,
    register: int,
    component_set: str,
    path_contrast: str,
    causal_state: str,
    intact: torch.Tensor,
    corrupt: torch.Tensor,
    patch_provenance: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    primitive = _coupling_rows(
        cells, register=register, intact=intact, corrupt=corrupt
    )
    patched = patch_provenance is not None
    return [
        {
            **row,
            "component_set": component_set,
            "path_contrast": path_contrast,
            "causal_state": causal_state,
            "intact_margin": float(left),
            "acceptor_ac_margin": float(right),
            "intact_host_corruption_state": "intact",
            "acceptor_ac_host_corruption_state": "acceptor_ac",
            "intact_source_corruption_state": "intact" if patched else None,
            "acceptor_ac_source_corruption_state": "acceptor_ac" if patched else None,
            "activation_operation": STATE_OPERATIONS[causal_state],
            "host_table_state": STATE_TABLES[causal_state],
            "patch_provenance": dict(patch_provenance) if patched else None,
            **_source_identity_fields(cell, patched=patched),
        }
        for row, cell, left, right in zip(primitive, cells, intact, corrupt)
    ]


def _raw_relocation_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int, int, int], Mapping[str, object]]:
    output: dict[tuple[str, int, int, int], Mapping[str, object]] = {}
    for row in rows:
        if row.get("component") != "p3" or row.get("p10_state") != "intact":
            continue
        key = (
            str(row["block_id"]),
            int(row["register"]),
            int(row["donor_phase"]),
            int(row["acceptor_phase"]),
        )
        if key in output:
            raise ValueError("relocation target repeats a learned raw cell")
        output[key] = row
    if not output:
        raise ValueError("relocation target has no learned intact-P10 raw rows")
    return output


def _raw_rows_from_relocation(
    cells: Sequence[RegisterCell],
    *,
    register: int,
    index: Mapping[tuple[str, int, int, int], Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in cells:
        source = index.get(
            (
                cell.block_id,
                int(register),
                cell.donor_phase,
                cell.acceptor_phase,
            )
        )
        if source is None:
            raise ValueError("relocation target is incomplete on this mediation shard")
        intact = float(source["intact_margin"])
        corrupt = float(source["corrupt_margin"])
        coupling = float(source["coupling"])
        if abs(coupling - (intact - corrupt)) > 1e-8:
            raise ValueError("relocation raw source fails margin arithmetic")
        rows.append(
            {
                "block_id": cell.block_id,
                "chrom": cell.chrom,
                "strand": cell.strand,
                "register": int(register),
                "donor_phase": cell.donor_phase,
                "acceptor_phase": cell.acceptor_phase,
                "phase_delta": phase_delta(cell.donor_phase, cell.acceptor_phase),
                "component_set": LEARNED_SET,
                "path_contrast": "learned",
                "causal_state": "raw",
                "intact_margin": intact,
                "acceptor_ac_margin": corrupt,
                "coupling": coupling,
                "intact_host_corruption_state": "intact",
                "acceptor_ac_host_corruption_state": "acceptor_ac",
                "intact_source_corruption_state": None,
                "acceptor_ac_source_corruption_state": None,
                "activation_operation": STATE_OPERATIONS["raw"],
                "host_table_state": STATE_TABLES["raw"],
                "patch_provenance": None,
                **_source_identity_fields(cell, patched=False),
            }
        )
    return rows


def evaluate_mediation_lattice(
    adapter: Any,
    cells: Sequence[RegisterCell],
    *,
    candidate_freeze: Mapping[str, object],
    relocation_rows: Sequence[Mapping[str, object]],
    p3_bins: Sequence[int],
    p3_attenuation_dose: float,
    batch_size: int,
    direction: str = "right",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Execute learned D/R/Q cells and their three independent null families."""

    if (
        candidate_freeze.get("schema") != CANDIDATE_FREEZE_SCHEMA
        or candidate_freeze.get("status") != "COMPLETE"
    ):
        raise ValueError("mediation runner requires the audited executed candidate freeze")
    validate_candidate_freeze(candidate_freeze)
    examples = tuple(cells)
    if not examples or batch_size <= 0:
        raise ValueError("mediation runner needs cells and a positive batch size")
    learned_spec = candidate_freeze["learned_set"]
    families = candidate_freeze["null_families"]
    learned_components = tuple(learned_spec["components"])
    relocation_index = _raw_relocation_index(relocation_rows)
    provider = HfCandidateMeasurementProvider(
        adapter,
        p3_bins=p3_bins,
        p3_attenuation_dose=p3_attenuation_dose,
        direction=direction,
    )
    rows: list[dict[str, object]] = []
    stream = sha256()
    patch_events = 0
    raw_replay_intact_max = 0.0
    raw_replay_corrupt_max = 0.0
    raw_replay_coupling_max = 0.0

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        intact_sequences = [cell.intact_sequence for cell in chunk]
        corrupt_sequences = [cell.acceptor_ac_sequence for cell in chunk]
        zero = provider._delta(0, attenuated=False)
        a0_intact, a0_corrupt = provider._forward_pair(
            chunk, delta=zero, capture=True, diagnostics=False
        )
        for register in REGISTERS:
            raw_delta = provider._delta(register, attenuated=False)
            attenuated_delta = provider._delta(register, attenuated=True)
            ar_intact, ar_corrupt = provider._forward_pair(
                chunk, delta=raw_delta, capture=True, diagnostics=False
            )
            br_intact, br_corrupt = provider._forward_pair(
                chunk, delta=attenuated_delta, capture=True, diagnostics=False
            )
            raw_source = _raw_rows_from_relocation(
                chunk, register=register, index=relocation_index
            )
            recomputed_raw = _coupling_rows(
                chunk,
                register=register,
                intact=ar_intact.margins,
                corrupt=ar_corrupt.margins,
            )
            for source, replay, intact_margin, corrupt_margin in zip(
                raw_source,
                recomputed_raw,
                ar_intact.margins,
                ar_corrupt.margins,
                strict=True,
            ):
                raw_replay_intact_max = max(
                    raw_replay_intact_max,
                    abs(float(source["intact_margin"]) - float(intact_margin)),
                )
                raw_replay_corrupt_max = max(
                    raw_replay_corrupt_max,
                    abs(float(source["acceptor_ac_margin"]) - float(corrupt_margin)),
                )
                raw_replay_coupling_max = max(
                    raw_replay_coupling_max,
                    abs(float(source["coupling"]) - float(replay["coupling"])),
                )
            rows.extend(raw_source)
            rows.extend(
                _mediation_rows(
                    chunk,
                    register=register,
                    component_set=LEARNED_SET,
                    path_contrast="learned",
                    causal_state="attenuated_host",
                    intact=br_intact.margins,
                    corrupt=br_corrupt.margins,
                    patch_provenance=None,
                )
            )
            rows.extend(
                _mediation_rows(
                    chunk,
                    register=register,
                    component_set=LEARNED_SET,
                    path_contrast="learned",
                    causal_state="no_ramp_host",
                    intact=a0_intact.margins,
                    corrupt=a0_corrupt.margins,
                    patch_provenance=None,
                )
            )
            learned_ids = _component_ids(learned_spec)
            learned_sources = {
                "scrub": (
                    {value: a0_intact.components[value] for value in learned_ids},
                    {value: a0_corrupt.components[value] for value in learned_ids},
                    raw_delta,
                    PATCH_OPERATION_REPLACE,
                    0,
                    None,
                ),
                "restore": (
                    {value: ar_intact.components[value] for value in learned_ids},
                    {value: ar_corrupt.components[value] for value in learned_ids},
                    attenuated_delta,
                    PATCH_OPERATION_REPLACE,
                    register,
                    None,
                ),
                "sufficiency": (
                    {
                        value: ar_intact.components[value] - a0_intact.components[value]
                        for value in learned_ids
                    },
                    {
                        value: ar_corrupt.components[value] - a0_corrupt.components[value]
                        for value in learned_ids
                    },
                    zero,
                    PATCH_OPERATION_INJECT,
                    register,
                    0,
                ),
            }
            for causal_state, (
                intact_source,
                corrupt_source,
                host_delta,
                operation,
                source_register,
                source_baseline,
            ) in learned_sources.items():
                provenance = _patch_provenance(
                    components=learned_components,
                    intact_source=intact_source,
                    corrupt_source=corrupt_source,
                    operation=operation,
                    source_register=source_register,
                    source_baseline_register=source_baseline,
                )
                stream.update(bytes.fromhex(provenance["combined_patch_tensor_sha256"]))
                intact_result, corrupt_result = provider._forward_pair(
                    chunk,
                    delta=host_delta,
                    capture=False,
                    diagnostics=False,
                    intact_patch=provider._patch(
                        components=learned_components,
                        source=intact_source,
                        sequences=intact_sequences,
                        state="intact",
                        operation=operation,
                    ),
                    corrupt_patch=provider._patch(
                        components=learned_components,
                        source=corrupt_source,
                        sequences=corrupt_sequences,
                        state="acceptor_ac",
                        operation=operation,
                    ),
                )
                rows.extend(
                    _mediation_rows(
                        chunk,
                        register=register,
                        component_set=LEARNED_SET,
                        path_contrast="learned",
                        causal_state=causal_state,
                        intact=intact_result.margins,
                        corrupt=corrupt_result.margins,
                        patch_provenance=provenance,
                    )
                )
                patch_events += 2

            for path_name in PATH_NAMES:
                causal_state = NULL_INTERVENTION_STATE[path_name]
                for label, spec in zip(expected_null_labels(path_name), families[path_name]):
                    if spec["label"] != label:
                        raise ValueError("candidate null family order drifted")
                    components = tuple(spec["components"])
                    ids = _component_ids(spec)
                    if path_name == "d_path":
                        intact_source = {value: a0_intact.components[value] for value in ids}
                        corrupt_source = {value: a0_corrupt.components[value] for value in ids}
                        host_delta = raw_delta
                        operation = PATCH_OPERATION_REPLACE
                        source_register = 0
                        source_baseline = None
                    elif path_name == "r_path":
                        intact_source = {value: ar_intact.components[value] for value in ids}
                        corrupt_source = {value: ar_corrupt.components[value] for value in ids}
                        host_delta = attenuated_delta
                        operation = PATCH_OPERATION_REPLACE
                        source_register = register
                        source_baseline = None
                    else:
                        intact_source = {
                            value: ar_intact.components[value] - a0_intact.components[value]
                            for value in ids
                        }
                        corrupt_source = {
                            value: ar_corrupt.components[value] - a0_corrupt.components[value]
                            for value in ids
                        }
                        host_delta = zero
                        operation = PATCH_OPERATION_INJECT
                        source_register = register
                        source_baseline = 0
                    provenance = _patch_provenance(
                        components=components,
                        intact_source=intact_source,
                        corrupt_source=corrupt_source,
                        operation=operation,
                        source_register=source_register,
                        source_baseline_register=source_baseline,
                    )
                    stream.update(bytes.fromhex(provenance["combined_patch_tensor_sha256"]))
                    intact_result, corrupt_result = provider._forward_pair(
                        chunk,
                        delta=host_delta,
                        capture=False,
                        diagnostics=False,
                        intact_patch=provider._patch(
                            components=components,
                            source=intact_source,
                            sequences=intact_sequences,
                            state="intact",
                            operation=operation,
                        ),
                        corrupt_patch=provider._patch(
                            components=components,
                            source=corrupt_source,
                            sequences=corrupt_sequences,
                            state="acceptor_ac",
                            operation=operation,
                        ),
                    )
                    rows.extend(
                        _mediation_rows(
                            chunk,
                            register=register,
                            component_set=label,
                            path_contrast=path_name,
                            causal_state=causal_state,
                            intact=intact_result.margins,
                            corrupt=corrupt_result.margins,
                            patch_provenance=provenance,
                        )
                    )
                    patch_events += 2
    if max(
        raw_replay_intact_max,
        raw_replay_corrupt_max,
        raw_replay_coupling_max,
    ) > 1e-8:
        raise RuntimeError(
            "mediation model forward does not replay the sealed relocation raw rows"
        )
    expected_rows_per_block = (len(LEARNED_STATES) + 31 * len(PATH_NAMES)) * 3 * 9
    block_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        block_counts[str(row["block_id"])] += 1
    if any(value != expected_rows_per_block for value in block_counts.values()):
        raise RuntimeError("mediation execution did not produce the complete causal lattice")
    return rows, {
        "actual_hook_tensor_stream_sha256": stream.hexdigest(),
        "actual_patch_forward_events": patch_events,
        "same_sequence_patching": True,
        "same_corruption_state_patching": True,
        "full_token_support": list(TOKEN_SUPPORT),
        "raw_relocation_replay_intact_margin_max_abs": raw_replay_intact_max,
        "raw_relocation_replay_corrupt_margin_max_abs": raw_replay_corrupt_max,
        "raw_relocation_replay_coupling_max_abs": raw_replay_coupling_max,
        "p3_attenuation_order": [
            "construct_intact_position_table",
            "attenuate_complete_frozen_p3_band",
            "apply_requested_full_span_analytic_register_ramp",
            "apply_same_state_activation_intervention",
        ],
        "rows_per_block": expected_rows_per_block,
        "passed": True,
    }


__all__ = [
    "MEDIATION_SHARD_SCHEMA",
    "evaluate_mediation_lattice",
]
