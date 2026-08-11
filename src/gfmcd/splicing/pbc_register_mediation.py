"""Model-independent analysis for Phase-III P3 register mediation.

The input to this module is a compact table of native masked-donor margins.  It does
not load SpliceBERT or infer anything from attention alone.  The table must contain a
complete block x imposed-register x donor-phase x acceptor-phase lattice for the
learned path and for three *separate* 31-set component-null families.

All activation patches are corruption-state preserving.  A row records both the
intact and ``AG->AC`` host/source bindings, which makes the wrong-state prohibition an
auditable property rather than an attestation.  Effects are reported only in raw logit
units: this module intentionally exposes no percentage-mediation estimator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from gfmcd.provenance import sha256_file
from gfmcd.splicing.pbc_register_relocation import (
    PHASES,
    REGISTERS,
    preferred_delta,
    stratified_studentized_max_t,
)


MEDIATION_TARGET_SCHEMA = "gfmcd.pbc.p3_register_mediation_target.v1"
RELOCATION_TARGET_SCHEMA = "gfmcd.pbc.p3_register_target.v1"
RELOCATION_ANALYSIS_SCHEMA = "gfmcd.pbc.p3_register_analysis.v1"
RELOCATION_AUDIT_SCHEMA = "gfmcd.pbc.p3_register_audit.v1"

INFERENCE_SEED = 2026081006
CONFIRMATORY_BOOTSTRAP_DRAWS = 20_000
NULL_SET_COUNT = 31
LEARNED_SET = "learned"
TOKEN_SUPPORT = (393, 513)
SESOI = 0.05
RESERVE_BLOCKS = 384
RESERVE_CELLS = 3_456

PATH_NAMES = ("d_path", "r_path", "q_path")
LEARNED_STATES = (
    "raw",
    "scrub",
    "restore",
    "attenuated_host",
    "sufficiency",
    "no_ramp_host",
)
PATH_STATE_PAIRS = {
    "d_path": ("raw", "scrub"),
    "r_path": ("restore", "attenuated_host"),
    "q_path": ("sufficiency", "no_ramp_host"),
}
NULL_INTERVENTION_STATE = {
    "d_path": "scrub",
    "r_path": "restore",
    "q_path": "sufficiency",
}
PATCH_STATES = frozenset({"scrub", "restore", "sufficiency"})
STATE_OPERATIONS = {
    "raw": "none",
    "scrub": "replace_a_r_with_same_state_a_0",
    "restore": "replace_with_same_state_a_r_from_intact_band",
    "attenuated_host": "none",
    "sufficiency": "inject_same_state_delta_a_r_minus_a_0",
    "no_ramp_host": "none",
}
STATE_TABLES = {
    "raw": "intact_band_ramp",
    "scrub": "intact_band_ramp",
    "restore": "attenuated_band_ramp",
    "attenuated_host": "attenuated_band_ramp",
    "sufficiency": "intact_band_no_ramp",
    "no_ramp_host": "intact_band_no_ramp",
}


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_freeze_sha256(freeze: Mapping[str, object]) -> str:
    """Return the canonical digest used to bind the candidate/null freeze."""

    return _canonical_sha256(freeze)


def null_label(path_name: str, index: int) -> str:
    if path_name not in PATH_NAMES:
        raise ValueError("unknown path contrast")
    if not 0 <= int(index) < NULL_SET_COUNT:
        raise ValueError("null-set index outside the frozen family")
    return f"{path_name}_null_{int(index):02d}"


def expected_null_labels(path_name: str) -> tuple[str, ...]:
    return tuple(null_label(path_name, index) for index in range(NULL_SET_COUNT))


def _component_signature(component: Mapping[str, object]) -> tuple[str, int, str, int]:
    component_id = str(component.get("component_id") or "")
    kind = str(component.get("kind") or "")
    if not component_id:
        raise ValueError("every frozen component needs a component_id")
    if kind not in {"head", "mlp"}:
        raise ValueError(f"component {component_id!r} has an invalid kind")
    try:
        layer = int(component["layer"])
        patch_tokens = int(component["patch_token_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"component {component_id!r} has invalid integer metadata") from exc
    if layer not in range(6):
        raise ValueError(f"component {component_id!r} has an invalid layer")
    if patch_tokens != TOKEN_SUPPORT[1] - TOKEN_SUPPORT[0]:
        raise ValueError(f"component {component_id!r} must patch the full frozen span")
    return component_id, layer, kind, patch_tokens


def _set_summary(spec: Mapping[str, object], *, expected_label: str) -> dict[str, object]:
    if str(spec.get("label") or "") != expected_label:
        raise ValueError(f"component-set label must be {expected_label!r}")
    components = spec.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError(f"component set {expected_label!r} is empty")
    signatures = [_component_signature(component) for component in components]
    ids = [signature[0] for signature in signatures]
    if len(ids) != len(set(ids)):
        raise ValueError(f"component set {expected_label!r} repeats a component")
    norm = _finite(spec.get("joint_output_delta_norm"), "joint output-delta norm")
    if norm <= 0:
        raise ValueError("joint output-delta norm must be positive")
    return {
        "ids": tuple(ids),
        "count": len(ids),
        "layers": Counter(signature[1] for signature in signatures),
        "kinds": Counter(signature[2] for signature in signatures),
        "patch_tokens": sum(signature[3] for signature in signatures),
        "joint_output_delta_norm": norm,
    }


def validate_candidate_freeze(freeze: Mapping[str, object]) -> dict[str, object]:
    """Validate candidate selection and all three separately matched null families."""

    if list(freeze.get("token_support", [])) != list(TOKEN_SUPPORT):
        raise ValueError("candidate freeze must use token support [393,513)")
    screen = freeze.get("candidate_screen")
    if not isinstance(screen, Mapping) or not (
        screen.get("attention_heads") == 96
        and screen.get("mlp_outputs") == 6
        and screen.get("all_components_screened") is True
        and screen.get("rank_metric") == "same_state_scrub_loss_in_omega3"
        and screen.get("prefixes_jointly_remeasured") is True
    ):
        raise ValueError("candidate screen must bind the complete 96-head/6-MLP universe")
    tensor_digest = str(freeze.get("patch_tensor_sha256") or "")
    if len(tensor_digest) != 64 or any(ch not in "0123456789abcdef" for ch in tensor_digest):
        raise ValueError("candidate freeze needs a lowercase patch_tensor_sha256")
    learned = freeze.get("learned_set")
    if not isinstance(learned, Mapping):
        raise ValueError("candidate freeze lacks learned_set")
    learned_summary = _set_summary(learned, expected_label=LEARNED_SET)
    if not 1 <= int(learned_summary["count"]) <= 8:
        raise ValueError("the frozen learned set must contain one to eight components")
    path_target_norms = freeze.get("path_target_output_delta_norms")
    if not isinstance(path_target_norms, Mapping) or set(path_target_norms) != set(
        PATH_NAMES
    ):
        raise ValueError("candidate freeze needs one learned output norm per path contrast")
    target_norms = {
        path_name: _finite(
            path_target_norms[path_name], f"{path_name} learned output-delta norm"
        )
        for path_name in PATH_NAMES
    }
    if any(value <= 0 for value in target_norms.values()):
        raise ValueError("learned path output-delta norms must be positive")

    selection = freeze.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("candidate freeze lacks selection metadata")
    if int(selection.get("candidate_count", -1)) != learned_summary["count"]:
        raise ValueError("selection candidate_count disagrees with learned_set")
    fraction = _finite(
        selection.get("jointly_remeasured_fraction"),
        "jointly remeasured register-effect fraction",
    )
    if not 0 <= fraction <= 1:
        raise ValueError("jointly remeasured fraction must lie in [0,1]")
    selection_class = str(selection.get("classification") or "")
    smallest = selection.get("smallest_prefix_reaches_80")
    if type(smallest) is not bool:
        raise ValueError("smallest_prefix_reaches_80 must be boolean")
    if fraction >= 0.80:
        if selection_class != "PREFIX_80" or not smallest:
            raise ValueError("an 80% prefix must be frozen as PREFIX_80")
    elif (
        selection_class != "DISTRIBUTED_CANDIDATE_SET"
        or smallest
        or learned_summary["count"] != 8
    ):
        raise ValueError("a sub-80% screen must freeze eight distributed candidates")

    families = freeze.get("null_families")
    if not isinstance(families, Mapping) or set(families) != set(PATH_NAMES):
        raise ValueError("candidate freeze needs separate D, R, and Q null families")
    learned_ids = set(learned_summary["ids"])
    matching: dict[str, dict[str, object]] = {}
    for path_name in PATH_NAMES:
        specs = families[path_name]
        if not isinstance(specs, list) or len(specs) != NULL_SET_COUNT:
            raise ValueError(f"{path_name} must contain exactly 31 component-null sets")
        expected = expected_null_labels(path_name)
        if [str(spec.get("label") or "") for spec in specs] != list(expected):
            raise ValueError(f"{path_name} null labels/order differ from the freeze")
        seen_sets: set[tuple[str, ...]] = set()
        ratios: dict[str, float] = {}
        for label, spec in zip(expected, specs):
            summary = _set_summary(spec, expected_label=label)
            if learned_ids.intersection(summary["ids"]):
                raise ValueError(f"null set {label!r} contains a learned component")
            canonical_ids = tuple(sorted(summary["ids"]))
            if canonical_ids in seen_sets:
                raise ValueError(f"{path_name} repeats a component-null set")
            seen_sets.add(canonical_ids)
            if summary["count"] != learned_summary["count"]:
                raise ValueError(f"null set {label!r} is not size matched")
            if summary["layers"] != learned_summary["layers"]:
                raise ValueError(f"null set {label!r} is not layer-count matched")
            if summary["kinds"] != learned_summary["kinds"]:
                raise ValueError(f"null set {label!r} is not head/MLP-type matched")
            if summary["patch_tokens"] != learned_summary["patch_tokens"]:
                raise ValueError(f"null set {label!r} is not patch-token matched")
            ratio = float(summary["joint_output_delta_norm"]) / target_norms[path_name]
            if not 0.90 <= ratio <= 1.10:
                raise ValueError(f"null set {label!r} is not output-norm matched")
            ratios[label] = ratio
        matching[path_name] = {
            "labels": list(expected),
            "count": NULL_SET_COUNT,
            "output_norm_ratios": ratios,
            "passed": True,
        }
    return {
        "candidate_count": learned_summary["count"],
        "selection_class": selection_class,
        "jointly_remeasured_fraction": fraction,
        "path_target_output_delta_norms": target_norms,
        "matching": matching,
        "passed": True,
    }


def _expected_label_states(freeze: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    labels: dict[str, tuple[str, ...]] = {LEARNED_SET: LEARNED_STATES}
    families = freeze["null_families"]
    assert isinstance(families, Mapping)
    for path_name in PATH_NAMES:
        state = NULL_INTERVENTION_STATE[path_name]
        for spec in families[path_name]:
            labels[str(spec["label"])] = (state,)
    return labels


def _path_for_label(label: str) -> str:
    if label == LEARNED_SET:
        return "learned"
    for path_name in PATH_NAMES:
        if label in expected_null_labels(path_name):
            return path_name
    raise ValueError(f"unknown component-set label {label!r}")


def _validate_state_binding(row: Mapping[str, object], state: str) -> None:
    if row.get("intact_host_corruption_state") != "intact":
        raise ValueError("intact margin must come from the intact host state")
    if row.get("acceptor_ac_host_corruption_state") != "acceptor_ac":
        raise ValueError("corrupt margin must come from the AG->AC host state")
    expected_source = state in PATCH_STATES
    intact_source = row.get("intact_source_corruption_state")
    corrupt_source = row.get("acceptor_ac_source_corruption_state")
    if expected_source:
        if intact_source != "intact" or corrupt_source != "acceptor_ac":
            raise ValueError("activation patch crosses corruption states")
    elif intact_source is not None or corrupt_source is not None:
        raise ValueError("unpatched state declares an activation source")
    if row.get("activation_operation") != STATE_OPERATIONS[state]:
        raise ValueError(f"causal state {state!r} has the wrong activation operation")
    if row.get("host_table_state") != STATE_TABLES[state]:
        raise ValueError(f"causal state {state!r} has the wrong host table")


def _valid_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _validate_patch_provenance(
    row: Mapping[str, object],
    *,
    state: str,
    register: int,
    expected_component_ids: Sequence[str],
) -> None:
    patched = state in PATCH_STATES
    host_cell = row.get("intact_host_cell_id")
    if (
        not isinstance(host_cell, str)
        or not host_cell
        or row.get("acceptor_ac_host_cell_id") != host_cell
        or not _valid_sha256(row.get("intact_host_sequence_sha256"))
        or not _valid_sha256(row.get("acceptor_ac_host_sequence_sha256"))
    ):
        raise ValueError("mediation row lacks exact host cell/sequence identity")
    source_fields = (
        "intact_source_cell_id",
        "acceptor_ac_source_cell_id",
        "intact_source_sequence_sha256",
        "acceptor_ac_source_sequence_sha256",
    )
    provenance = row.get("patch_provenance")
    if not patched:
        if any(row.get(name) is not None for name in source_fields) or provenance is not None:
            raise ValueError("unpatched state declares activation source provenance")
        return
    if (
        row.get("intact_source_cell_id") != host_cell
        or row.get("acceptor_ac_source_cell_id") != host_cell
        or row.get("intact_source_sequence_sha256")
        != row.get("intact_host_sequence_sha256")
        or row.get("acceptor_ac_source_sequence_sha256")
        != row.get("acceptor_ac_host_sequence_sha256")
    ):
        raise ValueError("activation patch crosses sequence or cell identity")
    if not isinstance(provenance, Mapping):
        raise ValueError("patched row lacks actual hook/tensor provenance")
    operation = (
        "inject_same_state_activation_delta"
        if state == "sufficiency"
        else "replace_same_state_activation"
    )
    expected_register = 0 if state == "scrub" else register
    expected_baseline = 0 if state == "sufficiency" else None
    if (
        provenance.get("operation") != operation
        or list(provenance.get("token_support", [])) != list(TOKEN_SUPPORT)
        or list(provenance.get("components", [])) != list(expected_component_ids)
        or int(provenance.get("source_register", -99)) != expected_register
        or provenance.get("source_baseline_register") != expected_baseline
        or not isinstance(provenance.get("hook_sites"), list)
        or not provenance.get("hook_sites")
        or any(
            value
            not in {
                "attention.self.context_pre_output_projection",
                "encoder.layer.output.dense_pre_residual_add",
            }
            for value in provenance["hook_sites"]
        )
        or any(
            not _valid_sha256(provenance.get(name))
            for name in (
                "intact_patch_tensor_sha256",
                "acceptor_ac_patch_tensor_sha256",
                "combined_patch_tensor_sha256",
            )
        )
    ):
        raise ValueError("activation patch provenance disagrees with the frozen operation")


def _phase_delta(donor: int, acceptor: int) -> int:
    if donor not in PHASES or acceptor not in PHASES:
        raise ValueError("donor and acceptor phases must lie in {0,1,2}")
    return (acceptor - donor) % 3


def validate_mediation_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    freeze: Mapping[str, object],
    arithmetic_atol: float = 1e-8,
) -> dict[str, object]:
    """Validate completeness, arithmetic, and same-corruption-state patching."""

    validate_candidate_freeze(freeze)
    if not rows:
        raise ValueError("mediation target has no rows")
    expected_label_states = _expected_label_states(freeze)
    expected_labels = set(expected_label_states)
    component_ids_by_label = {
        LEARNED_SET: tuple(
            str(row["component_id"])
            for row in freeze["learned_set"]["components"]
        )
    }
    for path_name in PATH_NAMES:
        for spec in freeze["null_families"][path_name]:
            component_ids_by_label[str(spec["label"])] = tuple(
                str(row["component_id"]) for row in spec["components"]
            )
    keys: set[tuple[str, str, str, int, int, int]] = set()
    block_metadata: dict[str, tuple[str, str]] = {}
    seen_labels: set[str] = set()
    arithmetic_max = 0.0
    for row in rows:
        block_id = str(row.get("block_id") or "")
        chrom = str(row.get("chrom") or "")
        strand = str(row.get("strand") or "")
        label = str(row.get("component_set") or "")
        state = str(row.get("causal_state") or "")
        if not block_id or not chrom or strand not in {"+", "-"}:
            raise ValueError("each row needs one block, chromosome, and valid strand")
        if label not in expected_labels or state not in expected_label_states[label]:
            raise ValueError(f"unexpected component-set/causal-state cell {label!r}/{state!r}")
        if row.get("path_contrast") != _path_for_label(label):
            raise ValueError("row path_contrast disagrees with component-set family")
        try:
            register = int(row["register"])
            donor = int(row["donor_phase"])
            acceptor = int(row["acceptor_phase"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("row register/phase fields are invalid") from exc
        if register not in REGISTERS:
            raise ValueError("register must lie in {-1,0,+1}")
        delta = _phase_delta(donor, acceptor)
        if int(row.get("phase_delta", -1)) != delta:
            raise ValueError("phase_delta disagrees with donor/acceptor phases")
        _validate_state_binding(row, state)
        _validate_patch_provenance(
            row,
            state=state,
            register=register,
            expected_component_ids=component_ids_by_label[label],
        )
        intact = _finite(row.get("intact_margin"), "intact margin")
        corrupt = _finite(row.get("acceptor_ac_margin"), "AG->AC margin")
        coupling = _finite(row.get("coupling"), "coupling")
        arithmetic_max = max(arithmetic_max, abs(coupling - (intact - corrupt)))
        if arithmetic_max > arithmetic_atol:
            raise ValueError("coupling does not equal intact minus AG->AC margin")
        metadata = (chrom, strand)
        if block_id in block_metadata and block_metadata[block_id] != metadata:
            raise ValueError(f"block {block_id!r} changes chromosome or strand")
        block_metadata[block_id] = metadata
        key = (label, state, block_id, register, donor, acceptor)
        if key in keys:
            raise ValueError("mediation target repeats an atomic lattice cell")
        keys.add(key)
        seen_labels.add(label)

    if seen_labels != expected_labels:
        missing = sorted(expected_labels - seen_labels)
        raise ValueError(f"mediation target misses component sets: {missing[:3]}")
    blocks = set(block_metadata)
    expected = {
        (label, state, block, register, donor, acceptor)
        for label, states in expected_label_states.items()
        for state in states
        for block in blocks
        for register in REGISTERS
        for donor in PHASES
        for acceptor in PHASES
    }
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(
            "mediation target has an incomplete lattice; "
            f"missing={missing[:2]} extra={extra[:2]}"
        )
    return {
        "n_rows": len(rows),
        "n_blocks": len(blocks),
        "n_component_sets": len(expected_labels),
        "expected_rows": len(expected),
        "arithmetic_max_abs": arithmetic_max,
        "same_state_patching": True,
        "complete_lattice": True,
        "passed": True,
    }


def mediation_condition_id(component_set: str, causal_state: str, register: int) -> str:
    """Stable normalized key for one component/state/register intervention."""

    if "|" in component_set or "|" in causal_state or register not in REGISTERS:
        raise ValueError("mediation condition fields cannot form a stable identifier")
    return f"{component_set}|{causal_state}|r{register}"


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot reduce an empty mediation endpoint")
    return float(math.fsum(float(value) for value in values) / len(values))


def compact_mediation_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    freeze: Mapping[str, object],
) -> dict[str, object]:
    """Normalize raw shard rows into sufficient endpoints and deduplicated provenance.

    The compact endpoint is the mean across the three donor/acceptor pairs sharing a
    phase delta.  Those nine register-by-delta means are sufficient to rederive every
    frozen S_r, Omega3, D/R/Q, and component-null statistic.
    """

    validation = validate_mediation_rows(rows, freeze=freeze)
    endpoint_groups: dict[
        tuple[str, str, str, int, int], list[tuple[float, float, float]]
    ] = defaultdict(list)
    cells: dict[tuple[str, int, int], dict[str, object]] = {}
    condition_static: dict[str, dict[str, object]] = {}
    condition_event_rows: dict[str, Counter[str]] = defaultdict(Counter)
    condition_rows: Counter[str] = Counter()

    for row in rows:
        block = str(row["block_id"])
        label = str(row["component_set"])
        state = str(row["causal_state"])
        register = int(row["register"])
        donor = int(row["donor_phase"])
        acceptor = int(row["acceptor_phase"])
        delta = int(row["phase_delta"])
        condition = mediation_condition_id(label, state, register)
        endpoint_groups[(block, condition, str(row["chrom"]), register, delta)].append(
            (
                float(row["intact_margin"]),
                float(row["acceptor_ac_margin"]),
                float(row["coupling"]),
            )
        )
        cell_key = (block, donor, acceptor)
        cell = {
            "cell_id": str(row["intact_host_cell_id"]),
            "block_id": block,
            "chrom": str(row["chrom"]),
            "strand": str(row["strand"]),
            "donor_phase": donor,
            "acceptor_phase": acceptor,
            "phase_delta": delta,
            "intact_sequence_sha256": str(row["intact_host_sequence_sha256"]),
            "acceptor_ac_sequence_sha256": str(
                row["acceptor_ac_host_sequence_sha256"]
            ),
        }
        if cell_key in cells and cells[cell_key] != cell:
            raise ValueError("mediation conditions change host cell or sequence identity")
        cells[cell_key] = cell

        provenance = row.get("patch_provenance")
        patched = state in PATCH_STATES
        static = {
            "condition_id": condition,
            "component_set": label,
            "path_contrast": str(row["path_contrast"]),
            "causal_state": state,
            "register": register,
            "activation_operation": str(row["activation_operation"]),
            "host_table_state": str(row["host_table_state"]),
            "patched": patched,
            "components": list(provenance["components"]) if patched else [],
            "hook_sites": list(provenance["hook_sites"]) if patched else [],
            "token_support": list(provenance["token_support"]) if patched else [],
            "source_register": int(provenance["source_register"]) if patched else None,
            "source_baseline_register": (
                provenance["source_baseline_register"] if patched else None
            ),
        }
        if condition in condition_static and condition_static[condition] != static:
            raise ValueError("mediation condition changes its intervention provenance")
        condition_static[condition] = static
        condition_rows[condition] += 1
        if patched:
            event = {
                name: str(provenance[name])
                for name in (
                    "intact_patch_tensor_sha256",
                    "acceptor_ac_patch_tensor_sha256",
                    "combined_patch_tensor_sha256",
                )
            }
            condition_event_rows[condition][
                json.dumps(event, sort_keys=True, separators=(",", ":"))
            ] += 1

    endpoint_rows: list[dict[str, object]] = []
    block_strands = {
        str(row["block_id"]): str(row["strand"])
        for row in rows
    }
    for (block, condition, chrom, register, delta), values in sorted(
        endpoint_groups.items()
    ):
        if len(values) != 3:
            raise ValueError("compact endpoint does not contain three phase-pair cells")
        intact = _mean([value[0] for value in values])
        corrupt = _mean([value[1] for value in values])
        coupling = _mean([value[2] for value in values])
        endpoint_rows.append(
            {
                "block_id": block,
                "chrom": chrom,
                "strand": block_strands[block],
                "condition_id": condition,
                "register": register,
                "phase_delta": delta,
                "n_phase_cells": 3,
                "mean_intact_margin": intact,
                "mean_acceptor_ac_margin": corrupt,
                "mean_coupling": coupling,
            }
        )

    ledger: dict[str, dict[str, object]] = {}
    for condition in sorted(condition_static):
        events = [
            {**json.loads(encoded), "source_rows": int(count)}
            for encoded, count in sorted(condition_event_rows[condition].items())
        ]
        ledger[condition] = {
            **condition_static[condition],
            "source_rows": int(condition_rows[condition]),
            "events": events,
            "tensor_stream_sha256": _canonical_sha256(events),
        }
    return {
        "endpoint_rows": endpoint_rows,
        "cell_ledger": [cells[key] for key in sorted(cells)],
        "provenance_ledger": ledger,
        "normalization": {
            "primitive_rows": int(validation["n_rows"]),
            "compact_endpoint_rows": len(endpoint_rows),
            "phase_pairs_per_endpoint": 3,
            "endpoint_reduction": "arithmetic_mean_with_math_fsum",
            "sufficient_for": ["S_r", "omega3", "D_path", "R_path", "Q_path"],
            "provenance_ledger_sha256": _canonical_sha256(ledger),
        },
    }


def validate_compact_mediation(
    payload: Mapping[str, object],
    *,
    freeze: Mapping[str, object],
    enforce_reserve_contract: bool = False,
    arithmetic_atol: float = 1e-8,
) -> dict[str, object]:
    """Validate the normalized endpoint, cell, and provenance ledgers."""

    validate_candidate_freeze(freeze)
    endpoints = payload.get("endpoint_rows")
    cell_ledger = payload.get("cell_ledger")
    ledger = payload.get("provenance_ledger")
    normalization = payload.get("normalization")
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("compact mediation target has no endpoint rows")
    if not isinstance(cell_ledger, list) or not cell_ledger:
        raise ValueError("compact mediation target has no cell ledger")
    if not isinstance(ledger, Mapping) or not ledger:
        raise ValueError("compact mediation target has no provenance ledger")
    if not isinstance(normalization, Mapping):
        raise ValueError("compact mediation target lacks normalization metadata")

    label_states = _expected_label_states(freeze)
    components = {
        LEARNED_SET: tuple(
            str(row["component_id"]) for row in freeze["learned_set"]["components"]
        )
    }
    for path_name in PATH_NAMES:
        for spec in freeze["null_families"][path_name]:
            components[str(spec["label"])] = tuple(
                str(row["component_id"]) for row in spec["components"]
            )
    expected_conditions = {
        mediation_condition_id(label, state, register)
        for label, states in label_states.items()
        for state in states
        for register in REGISTERS
    }
    if set(ledger) != expected_conditions:
        raise ValueError("compact provenance ledger has the wrong condition census")

    blocks: dict[str, tuple[str, str]] = {}
    cells: set[tuple[str, int, int]] = set()
    for cell in cell_ledger:
        block = str(cell.get("block_id") or "")
        chrom = str(cell.get("chrom") or "")
        strand = str(cell.get("strand") or "")
        donor = int(cell.get("donor_phase", -1))
        acceptor = int(cell.get("acceptor_phase", -1))
        if (
            not block
            or not chrom
            or strand not in {"+", "-"}
            or donor not in PHASES
            or acceptor not in PHASES
            or int(cell.get("phase_delta", -1)) != _phase_delta(donor, acceptor)
            or cell.get("cell_id") != f"{block}:{donor}:{acceptor}"
            or not _valid_sha256(cell.get("intact_sequence_sha256"))
            or not _valid_sha256(cell.get("acceptor_ac_sequence_sha256"))
        ):
            raise ValueError("compact cell ledger contains invalid identity metadata")
        if block in blocks and blocks[block] != (chrom, strand):
            raise ValueError("compact cell ledger changes block metadata")
        blocks[block] = (chrom, strand)
        key = (block, donor, acceptor)
        if key in cells:
            raise ValueError("compact cell ledger repeats a lattice cell")
        cells.add(key)
    expected_cells = {
        (block, donor, acceptor)
        for block in blocks
        for donor in PHASES
        for acceptor in PHASES
    }
    if cells != expected_cells:
        raise ValueError("compact cell ledger is not a complete 3x3 block lattice")

    for condition, value in ledger.items():
        if not isinstance(value, Mapping) or value.get("condition_id") != condition:
            raise ValueError("compact provenance condition is malformed")
        label = str(value.get("component_set") or "")
        state = str(value.get("causal_state") or "")
        register = int(value.get("register", -99))
        patched = state in PATCH_STATES
        expected_components = list(components.get(label, ()))
        expected_hooks = sorted(
            {
                str(row["hook_site"])
                for row in (
                    freeze["learned_set"]["components"]
                    if label == LEARNED_SET
                    else next(
                        spec["components"]
                        for path_name in PATH_NAMES
                        for spec in freeze["null_families"][path_name]
                        if spec["label"] == label
                    )
                )
            }
        )
        events = value.get("events")
        if (
            state not in label_states.get(label, ())
            or register not in REGISTERS
            or condition != mediation_condition_id(label, state, register)
            or value.get("path_contrast") != _path_for_label(label)
            or value.get("activation_operation") != STATE_OPERATIONS[state]
            or value.get("host_table_state") != STATE_TABLES[state]
            or value.get("patched") is not patched
            or int(value.get("source_rows", -1)) != len(blocks) * 9
            or not isinstance(events, list)
            or value.get("tensor_stream_sha256") != _canonical_sha256(events)
        ):
            raise ValueError("compact provenance condition disagrees with the design")
        if patched:
            expected_register = 0 if state == "scrub" else register
            expected_baseline = 0 if state == "sufficiency" else None
            if (
                list(value.get("components", [])) != expected_components
                or list(value.get("hook_sites", [])) != expected_hooks
                or list(value.get("token_support", [])) != list(TOKEN_SUPPORT)
                or int(value.get("source_register", -99)) != expected_register
                or value.get("source_baseline_register") != expected_baseline
                or not events
                or sum(int(event.get("source_rows", -1)) for event in events)
                != len(blocks) * 9
                or any(
                    int(event.get("source_rows", 0)) <= 0
                    or any(
                        not _valid_sha256(event.get(name))
                        for name in (
                            "intact_patch_tensor_sha256",
                            "acceptor_ac_patch_tensor_sha256",
                            "combined_patch_tensor_sha256",
                        )
                    )
                    for event in events
                )
            ):
                raise ValueError("compact patched provenance is incomplete")
        elif any(
            (
                value.get("components"),
                value.get("hook_sites"),
                value.get("token_support"),
                value.get("source_register"),
                value.get("source_baseline_register"),
                events,
            )
        ):
            raise ValueError("compact unpatched condition declares patch provenance")

    endpoint_keys: set[tuple[str, str, int]] = set()
    arithmetic_max = 0.0
    for row in endpoints:
        block = str(row.get("block_id") or "")
        condition = str(row.get("condition_id") or "")
        register = int(row.get("register", -99))
        delta = int(row.get("phase_delta", -1))
        if (
            block not in blocks
            or condition not in ledger
            or register != int(ledger[condition]["register"])
            or delta not in PHASES
            or int(row.get("n_phase_cells", -1)) != 3
            or (str(row.get("chrom")), str(row.get("strand"))) != blocks[block]
        ):
            raise ValueError("compact endpoint identity is invalid")
        intact = _finite(row.get("mean_intact_margin"), "compact intact mean")
        corrupt = _finite(row.get("mean_acceptor_ac_margin"), "compact corrupt mean")
        coupling = _finite(row.get("mean_coupling"), "compact coupling mean")
        arithmetic_max = max(arithmetic_max, abs(coupling - (intact - corrupt)))
        key = (block, condition, delta)
        if key in endpoint_keys:
            raise ValueError("compact mediation target repeats an endpoint")
        endpoint_keys.add(key)
    expected_endpoints = {
        (block, condition, delta)
        for block in blocks
        for condition in expected_conditions
        for delta in PHASES
    }
    if endpoint_keys != expected_endpoints or arithmetic_max > arithmetic_atol:
        raise ValueError("compact mediation endpoint lattice is incomplete or inconsistent")
    if (
        int(normalization.get("primitive_rows", -1))
        != len(blocks) * len(expected_conditions) * 9
        or int(normalization.get("compact_endpoint_rows", -1)) != len(endpoints)
        or int(normalization.get("phase_pairs_per_endpoint", -1)) != 3
        or normalization.get("provenance_ledger_sha256") != _canonical_sha256(ledger)
    ):
        raise ValueError("compact normalization counts or provenance digest drifted")
    if enforce_reserve_contract and (
        len(blocks) != RESERVE_BLOCKS or len(cells) != RESERVE_CELLS
    ):
        raise ValueError("confirmatory compact target is not the exact 384/3456 reserve")
    return {
        "n_blocks": len(blocks),
        "n_cells": len(cells),
        "n_conditions": len(expected_conditions),
        "n_endpoint_rows": len(endpoints),
        "primitive_rows_represented": int(normalization["primitive_rows"]),
        "arithmetic_max_abs": arithmetic_max,
        "complete_compact_lattice": True,
        "passed": True,
    }


def _register_summary(cell_values: Mapping[tuple[int, int, int], float]) -> dict[str, object]:
    scores: dict[str, float] = {}
    for register in REGISTERS:
        by_delta: dict[int, list[float]] = {delta: [] for delta in PHASES}
        for donor in PHASES:
            for acceptor in PHASES:
                by_delta[_phase_delta(donor, acceptor)].append(
                    float(cell_values[(register, donor, acceptor)])
                )
        means = {delta: float(np.mean(values)) for delta, values in by_delta.items()}
        preferred = preferred_delta(register)
        others = [means[delta] for delta in PHASES if delta != preferred]
        scores[f"s_{register}"] = means[preferred] - float(np.mean(others))
    return {
        "omega3": float(np.mean(list(scores.values()))),
        "s": scores,
    }


def block_mediation_effects(
    rows: Sequence[Mapping[str, object]],
    *,
    freeze: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Reduce valid primitive rows to block Ω3, S_r, and absolute path effects."""

    validate_mediation_rows(rows, freeze=freeze)
    values: dict[tuple[str, str, str], dict[tuple[int, int, int], float]] = defaultdict(dict)
    metadata: dict[str, tuple[str, str]] = {}
    for row in rows:
        block = str(row["block_id"])
        label = str(row["component_set"])
        state = str(row["causal_state"])
        coordinate = (
            int(row["register"]),
            int(row["donor_phase"]),
            int(row["acceptor_phase"]),
        )
        values[(block, label, state)][coordinate] = float(row["coupling"])
        metadata[block] = (str(row["chrom"]), str(row["strand"]))

    output: list[dict[str, object]] = []
    for block in sorted(metadata):
        state_summaries = {
            state: _register_summary(values[(block, LEARNED_SET, state)])
            for state in LEARNED_STATES
        }
        paths = {
            path_name: float(state_summaries[left]["omega3"])
            - float(state_summaries[right]["omega3"])
            for path_name, (left, right) in PATH_STATE_PAIRS.items()
        }
        register_paths: dict[str, dict[str, float]] = {}
        for path_name, (left, right) in PATH_STATE_PAIRS.items():
            register_paths[path_name] = {
                f"s_{register}": float(state_summaries[left]["s"][f"s_{register}"])
                - float(state_summaries[right]["s"][f"s_{register}"])
                for register in REGISTERS
            }
        null_paths: dict[str, dict[str, float]] = {}
        for path_name in PATH_NAMES:
            intervention_state = NULL_INTERVENTION_STATE[path_name]
            left, right = PATH_STATE_PAIRS[path_name]
            baseline_state = left if intervention_state == right else right
            baseline = float(state_summaries[baseline_state]["omega3"])
            local: dict[str, float] = {}
            for label in expected_null_labels(path_name):
                intervened = float(
                    _register_summary(values[(block, label, intervention_state)])["omega3"]
                )
                local[label] = (
                    baseline - intervened
                    if intervention_state == right
                    else intervened - baseline
                )
            null_paths[path_name] = local
        chrom, strand = metadata[block]
        output.append(
            {
                "block_id": block,
                "chrom": chrom,
                "strand": strand,
                "state_register_effects": state_summaries,
                "paths": paths,
                "path_register_effects": register_paths,
                "null_paths": null_paths,
            }
        )
    return tuple(output)


def compact_block_mediation_effects(
    payload: Mapping[str, object],
    *,
    freeze: Mapping[str, object],
    enforce_reserve_contract: bool = False,
) -> tuple[dict[str, object], ...]:
    """Rederive block effects directly from normalized register-by-delta means."""

    validate_compact_mediation(
        payload,
        freeze=freeze,
        enforce_reserve_contract=enforce_reserve_contract,
    )
    ledger = payload["provenance_ledger"]
    assert isinstance(ledger, Mapping)
    values: dict[tuple[str, str, str], dict[tuple[int, int], float]] = defaultdict(dict)
    metadata: dict[str, tuple[str, str]] = {}
    for row in payload["endpoint_rows"]:
        condition = ledger[str(row["condition_id"])]
        label = str(condition["component_set"])
        state = str(condition["causal_state"])
        block = str(row["block_id"])
        values[(block, label, state)][
            (int(row["register"]), int(row["phase_delta"]))
        ] = float(row["mean_coupling"])
        metadata[block] = (str(row["chrom"]), str(row["strand"]))

    def summary(local: Mapping[tuple[int, int], float]) -> dict[str, object]:
        scores = {}
        for register in REGISTERS:
            preferred = preferred_delta(register)
            scores[f"s_{register}"] = float(local[(register, preferred)]) - _mean(
                [
                    float(local[(register, delta)])
                    for delta in PHASES
                    if delta != preferred
                ]
            )
        return {"omega3": _mean(list(scores.values())), "s": scores}

    output: list[dict[str, object]] = []
    for block in sorted(metadata):
        state_summaries = {
            state: summary(values[(block, LEARNED_SET, state)])
            for state in LEARNED_STATES
        }
        paths = {
            path_name: float(state_summaries[left]["omega3"])
            - float(state_summaries[right]["omega3"])
            for path_name, (left, right) in PATH_STATE_PAIRS.items()
        }
        register_paths = {
            path_name: {
                f"s_{register}": float(state_summaries[left]["s"][f"s_{register}"])
                - float(state_summaries[right]["s"][f"s_{register}"])
                for register in REGISTERS
            }
            for path_name, (left, right) in PATH_STATE_PAIRS.items()
        }
        null_paths: dict[str, dict[str, float]] = {}
        for path_name in PATH_NAMES:
            intervention = NULL_INTERVENTION_STATE[path_name]
            left, right = PATH_STATE_PAIRS[path_name]
            baseline_state = left if intervention == right else right
            baseline = float(state_summaries[baseline_state]["omega3"])
            null_paths[path_name] = {}
            for label in expected_null_labels(path_name):
                intervened = float(summary(values[(block, label, intervention)])["omega3"])
                null_paths[path_name][label] = (
                    baseline - intervened
                    if intervention == right
                    else intervened - baseline
                )
        chrom, strand = metadata[block]
        output.append(
            {
                "block_id": block,
                "chrom": chrom,
                "strand": strand,
                "state_register_effects": state_summaries,
                "paths": paths,
                "path_register_effects": register_paths,
                "null_paths": null_paths,
            }
        )
    return tuple(output)


def exact_add_one_path_p(target: float, controls: Sequence[float]) -> float:
    values = [_finite(value, "component-null path effect") for value in controls]
    if len(values) != NULL_SET_COUNT:
        raise ValueError("path specificity requires exactly 31 component-null effects")
    learned = _finite(target, "learned path effect")
    return float((1 + sum(value >= learned for value in values)) / (len(values) + 1))


def path_specificity(
    blocks: Sequence[Mapping[str, object]],
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, object]:
    """Evaluate each path against its own 31-set family plus paired max-T bounds."""

    if not blocks:
        raise ValueError("path specificity needs block effects")
    rank_tests: dict[str, dict[str, object]] = {}
    paired_matrix: list[list[float]] = []
    for block in blocks:
        paired_matrix.append(
            [
                float(block["paths"][path_name])
                - float(np.mean(list(block["null_paths"][path_name].values())))
                for path_name in PATH_NAMES
            ]
        )
    for path_name in PATH_NAMES:
        target = float(np.mean([float(block["paths"][path_name]) for block in blocks]))
        labels = expected_null_labels(path_name)
        controls = {
            label: float(
                np.mean([float(block["null_paths"][path_name][label]) for block in blocks])
            )
            for label in labels
        }
        p_value = exact_add_one_path_p(target, list(controls.values()))
        rank_tests[path_name] = {
            "target": target,
            "controls": controls,
            "n_controls": len(controls),
            "target_exceeds_every_control": bool(target > max(controls.values())),
            "exact_add_one_p_one_sided": p_value,
            "threshold": 1 / 32,
            "passed": bool(target > max(controls.values()) and p_value <= 1 / 32),
        }
    names = tuple(f"{path_name}_minus_mean_null" for path_name in PATH_NAMES)
    strata = [f"{block['chrom']}:{block['strand']}" for block in blocks]
    paired = stratified_studentized_max_t(
        np.asarray(paired_matrix, dtype=float),
        names=names,
        strata=strata,
        seed=seed,
        n_bootstrap=n_bootstrap,
    ).as_dict()
    paired_passed = all(float(paired["lower"][name]) > 0 for name in names)
    return {
        "rank_tests": rank_tests,
        "paired_learned_minus_mean_null": paired,
        "separate_null_families": True,
        "passed": bool(
            all(bool(rank_tests[path_name]["passed"]) for path_name in PATH_NAMES)
            and paired_passed
        ),
    }


def _valid_file_record(record: object) -> tuple[Path, str]:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise ValueError("source file record is malformed")
    path = Path(str(record["path"]))
    if not path.is_file():
        raise ValueError(f"bound source does not exist: {path}")
    digest = sha256_file(path)
    if record.get("sha256") != digest or int(record.get("size_bytes", -1)) != path.stat().st_size:
        raise ValueError(f"bound source hash/size changed: {path}")
    return path, digest


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _validated_relocation_authorization(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object], str, dict[str, object], bool]:
    sources = payload.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("mediation target lacks source records")
    required = ("relocation_target", "relocation_analysis", "relocation_audit")
    if not set(required).issubset(sources):
        raise ValueError("mediation target lacks required relocation sources")
    records = {name: _valid_file_record(sources[name]) for name in required}
    relocation_target = _read_json(records["relocation_target"][0], "relocation target")
    relocation_analysis = _read_json(records["relocation_analysis"][0], "relocation analysis")
    relocation_audit = _read_json(records["relocation_audit"][0], "relocation audit")
    target_hash = records["relocation_target"][1]
    analysis_status = str(relocation_analysis.get("status") or "")
    if (
        relocation_target.get("schema") != RELOCATION_TARGET_SCHEMA
        or relocation_target.get("status") != "COMPLETE"
    ):
        raise ValueError("bound relocation target is incomplete")
    if (
        relocation_analysis.get("schema") != RELOCATION_ANALYSIS_SCHEMA
        or not analysis_status.startswith("P3_REGISTER_RELOCATION_SUPPORTED")
        or relocation_analysis.get("target_sha256") != target_hash
    ):
        raise ValueError("bound relocation analysis does not authorize mediation")
    decision = relocation_analysis.get("decision")
    if not isinstance(decision, Mapping) or decision.get("code") != analysis_status:
        raise ValueError("relocation analysis decision/status disagree")
    if (
        relocation_audit.get("schema") != RELOCATION_AUDIT_SCHEMA
        or relocation_audit.get("status") != "PASS"
        or relocation_audit.get("target_sha256") != target_hash
    ):
        raise ValueError("bound relocation audit is not PASS for this target")
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("mediation target lacks authorization")
    expected_bindings: dict[str, object] = {
        "relocation_target_sha256": records["relocation_target"][1],
        "relocation_analysis_sha256": records["relocation_analysis"][1],
        "relocation_audit_sha256": records["relocation_audit"][1],
        "relocation_status": analysis_status,
        "relocation_audit_status": "PASS",
    }
    for key, expected in expected_bindings.items():
        if authorization.get(key) != expected:
            raise ValueError(f"mediation authorization does not bind {key}")
    if type(authorization.get("path_instrument_capable")) is not bool:
        raise ValueError("path_instrument_capable must be boolean")
    return (
        relocation_target,
        analysis_status,
        expected_bindings,
        bool(authorization["path_instrument_capable"]),
    )


def validate_relocation_binding(
    payload: Mapping[str, object],
    *,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Bind mediation to the exact audited relocation target and replay its raw rows."""

    relocation_target, analysis_status, expected_bindings, capable = (
        _validated_relocation_authorization(payload)
    )

    original_rows = relocation_target.get("rows")
    if not isinstance(original_rows, list):
        raise ValueError("bound relocation target lacks primitive rows")
    original: dict[tuple[str, int, int, int], tuple[object, ...]] = {}
    for row in original_rows:
        if (
            isinstance(row, Mapping)
            and row.get("component") == "p3"
            and row.get("p10_state") == "intact"
        ):
            key = (
                str(row.get("block_id")),
                int(row.get("register", -99)),
                int(row.get("donor_phase", -99)),
                int(row.get("acceptor_phase", -99)),
            )
            if key in original:
                raise ValueError("bound relocation target repeats learned raw cells")
            original[key] = (
                str(row.get("chrom")),
                str(row.get("strand")),
                _finite(row.get("intact_margin"), "relocation intact margin"),
                _finite(row.get("corrupt_margin"), "relocation corrupt margin"),
                _finite(row.get("coupling"), "relocation coupling"),
            )
    replay: dict[tuple[str, int, int, int], tuple[object, ...]] = {}
    for row in rows:
        if row.get("component_set") == LEARNED_SET and row.get("causal_state") == "raw":
            key = (
                str(row.get("block_id")),
                int(row.get("register", -99)),
                int(row.get("donor_phase", -99)),
                int(row.get("acceptor_phase", -99)),
            )
            replay[key] = (
                str(row.get("chrom")),
                str(row.get("strand")),
                _finite(row.get("intact_margin"), "mediation raw intact margin"),
                _finite(row.get("acceptor_ac_margin"), "mediation raw corrupt margin"),
                _finite(row.get("coupling"), "mediation raw coupling"),
            )
    if replay != original or not replay:
        raise ValueError(
            "mediation raw cells do not exactly replay the authorized relocation target"
        )
    return {
        "relocation_status": analysis_status,
        "relocation_audit_status": "PASS",
        "raw_cells_exact_replay": True,
        "source_sha256": expected_bindings,
        "path_instrument_capable": capable,
        "passed": True,
    }


def validate_compact_relocation_binding(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Bind compact raw endpoints to the audited relocation sufficient endpoints."""

    relocation_target, analysis_status, expected_bindings, capable = (
        _validated_relocation_authorization(payload)
    )
    source_rows = relocation_target.get("rows")
    endpoints = payload.get("endpoint_rows")
    ledger = payload.get("provenance_ledger")
    if not isinstance(source_rows, list) or not isinstance(endpoints, list):
        raise ValueError("compact relocation replay lacks source or endpoint rows")
    if not isinstance(ledger, Mapping):
        raise ValueError("compact relocation replay lacks its condition ledger")
    grouped: dict[
        tuple[str, int, int], list[tuple[str, str, float, float, float]]
    ] = defaultdict(list)
    for row in source_rows:
        if (
            isinstance(row, Mapping)
            and row.get("component") == "p3"
            and row.get("p10_state") == "intact"
        ):
            donor = int(row.get("donor_phase", -99))
            acceptor = int(row.get("acceptor_phase", -99))
            grouped[
                (
                    str(row.get("block_id")),
                    int(row.get("register", -99)),
                    _phase_delta(donor, acceptor),
                )
            ].append(
                (
                    str(row.get("chrom")),
                    str(row.get("strand")),
                    _finite(row.get("intact_margin"), "relocation intact margin"),
                    _finite(row.get("corrupt_margin"), "relocation corrupt margin"),
                    _finite(row.get("coupling"), "relocation coupling"),
                )
            )
    original = {}
    for key, values in grouped.items():
        if len(values) != 3 or len({value[:2] for value in values}) != 1:
            raise ValueError("relocation source cannot form compact phase-delta endpoints")
        original[key] = (
            values[0][0],
            values[0][1],
            _mean([value[2] for value in values]),
            _mean([value[3] for value in values]),
            _mean([value[4] for value in values]),
        )
    replay = {}
    for row in endpoints:
        condition = ledger.get(str(row.get("condition_id")))
        if not isinstance(condition, Mapping):
            continue
        if (
            condition.get("component_set") == LEARNED_SET
            and condition.get("causal_state") == "raw"
        ):
            replay[
                (
                    str(row.get("block_id")),
                    int(row.get("register", -99)),
                    int(row.get("phase_delta", -1)),
                )
            ] = (
                str(row.get("chrom")),
                str(row.get("strand")),
                _finite(row.get("mean_intact_margin"), "compact raw intact mean"),
                _finite(row.get("mean_acceptor_ac_margin"), "compact raw corrupt mean"),
                _finite(row.get("mean_coupling"), "compact raw coupling mean"),
            )
    if not replay or replay != original:
        raise ValueError(
            "compact mediation raw endpoints do not exactly replay relocation means"
        )
    mechanics = payload.get("mechanics")
    replay_fields = (
        "raw_relocation_replay_intact_margin_max_abs",
        "raw_relocation_replay_corrupt_margin_max_abs",
        "raw_relocation_replay_coupling_max_abs",
    )
    if not isinstance(mechanics, Mapping) or any(
        _finite(mechanics.get(name), name) > 1e-8 for name in replay_fields
    ):
        raise ValueError("model-to-relocation replay does not pass all three raw endpoints")
    return {
        "relocation_status": analysis_status,
        "relocation_audit_status": "PASS",
        "raw_compact_endpoints_exact_replay": True,
        "raw_model_replay_max_abs": {
            name: float(mechanics[name]) for name in replay_fields
        },
        "source_sha256": expected_bindings,
        "path_instrument_capable": capable,
        "passed": True,
    }


def adjudicate_mediation(
    path_inference: Mapping[str, object],
    specificity: Mapping[str, object],
    *,
    relocation_authorized: bool,
    path_instrument_capable: bool,
) -> dict[str, str]:
    """Apply only the frozen Phase-III route classifications."""

    if not relocation_authorized:
        return {
            "code": "P3_REGISTER_INSTRUMENT_UNRESOLVED",
            "licensed_claim": "No positive, null, or inverted register conclusion.",
        }
    if not path_instrument_capable:
        return {
            "code": "P3_ROUTE_INSTRUMENT_UNRESOLVED",
            "licensed_claim": (
                "Register relocation may be reported; route interpretation is withheld."
            ),
        }
    lower = path_inference.get("lower")
    upper = path_inference.get("upper")
    if not isinstance(lower, Mapping) or not isinstance(upper, Mapping):
        raise ValueError("path inference lacks simultaneous intervals")
    if all(float(lower[name]) > SESOI for name in PATH_NAMES) and bool(
        specificity.get("passed")
    ):
        return {
            "code": "P3_REGISTER_MEDIATED",
            "licensed_claim": (
                "The frozen component path is necessary, rescuable, and separately "
                "sufficient for a scoped portion of the relocation."
            ),
        }
    if all(
        float(lower[name]) >= -SESOI and float(upper[name]) <= SESOI
        for name in PATH_NAMES
    ):
        return {
            "code": "P3_ROUTE_BOUNDED_NULL",
            "licensed_claim": (
                "The declared frozen path does not materially mediate relocation or "
                "supply a material sufficiency effect; alternative routes remain open."
            ),
        }
    return {
        "code": "INCONCLUSIVE",
        "licensed_claim": "Stop at the strongest lower evidence level.",
    }


def analyze_mediation(
    rows: Sequence[Mapping[str, object]],
    *,
    freeze: Mapping[str, object],
    seed: int = INFERENCE_SEED,
    n_bootstrap: int = CONFIRMATORY_BOOTSTRAP_DRAWS,
    relocation_authorized: bool = True,
    path_instrument_capable: bool = True,
) -> dict[str, object]:
    """Compute raw register replay, path effects, specificity, and classification."""

    blocks = block_mediation_effects(rows, freeze=freeze)
    strata = [f"{block['chrom']}:{block['strand']}" for block in blocks]
    raw_names = ("omega3", "s_-1", "s_0", "s_1")
    raw_matrix = np.asarray(
        [
            [
                float(block["state_register_effects"]["raw"]["omega3"]),
                *[
                    float(block["state_register_effects"]["raw"]["s"][f"s_{register}"])
                    for register in REGISTERS
                ],
            ]
            for block in blocks
        ],
        dtype=float,
    )
    raw_inference = stratified_studentized_max_t(
        raw_matrix,
        names=raw_names,
        strata=strata,
        seed=seed,
        n_bootstrap=n_bootstrap,
    ).as_dict()
    path_matrix = np.asarray(
        [[float(block["paths"][name]) for name in PATH_NAMES] for block in blocks],
        dtype=float,
    )
    path_inference = stratified_studentized_max_t(
        path_matrix,
        names=PATH_NAMES,
        strata=strata,
        seed=seed,
        n_bootstrap=n_bootstrap,
    ).as_dict()
    specificity = path_specificity(
        blocks,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    decision = adjudicate_mediation(
        path_inference,
        specificity,
        relocation_authorized=relocation_authorized,
        path_instrument_capable=path_instrument_capable,
    )
    return {
        "n_blocks": len(blocks),
        "inference_seed": int(seed),
        "bootstrap_draws": int(n_bootstrap),
        "relocation_replay_inference": raw_inference,
        "path_inference": path_inference,
        "specificity": specificity,
        "per_block": list(blocks),
        "decision": decision,
        "absolute_effect_units": "raw MLM logit-difference units",
    }


def analyze_compact_mediation(
    payload: Mapping[str, object],
    *,
    freeze: Mapping[str, object],
    seed: int = INFERENCE_SEED,
    n_bootstrap: int = CONFIRMATORY_BOOTSTRAP_DRAWS,
    relocation_authorized: bool = True,
    path_instrument_capable: bool = True,
    enforce_reserve_contract: bool = False,
) -> dict[str, object]:
    """Run frozen mediation inference from the compact sufficient endpoints."""

    blocks = compact_block_mediation_effects(
        payload,
        freeze=freeze,
        enforce_reserve_contract=enforce_reserve_contract,
    )
    strata = [f"{block['chrom']}:{block['strand']}" for block in blocks]
    raw_names = ("omega3", "s_-1", "s_0", "s_1")
    raw_matrix = np.asarray(
        [
            [
                float(block["state_register_effects"]["raw"]["omega3"]),
                *[
                    float(block["state_register_effects"]["raw"]["s"][f"s_{register}"])
                    for register in REGISTERS
                ],
            ]
            for block in blocks
        ],
        dtype=float,
    )
    raw_inference = stratified_studentized_max_t(
        raw_matrix,
        names=raw_names,
        strata=strata,
        seed=seed,
        n_bootstrap=n_bootstrap,
    ).as_dict()
    path_matrix = np.asarray(
        [[float(block["paths"][name]) for name in PATH_NAMES] for block in blocks],
        dtype=float,
    )
    path_inference = stratified_studentized_max_t(
        path_matrix,
        names=PATH_NAMES,
        strata=strata,
        seed=seed,
        n_bootstrap=n_bootstrap,
    ).as_dict()
    specificity = path_specificity(blocks, seed=seed, n_bootstrap=n_bootstrap)
    decision = adjudicate_mediation(
        path_inference,
        specificity,
        relocation_authorized=relocation_authorized,
        path_instrument_capable=path_instrument_capable,
    )
    return {
        "n_blocks": len(blocks),
        "inference_seed": int(seed),
        "bootstrap_draws": int(n_bootstrap),
        "relocation_replay_inference": raw_inference,
        "path_inference": path_inference,
        "specificity": specificity,
        "per_block": list(blocks),
        "decision": decision,
        "absolute_effect_units": "raw MLM logit-difference units",
    }


__all__ = [
    "CONFIRMATORY_BOOTSTRAP_DRAWS",
    "INFERENCE_SEED",
    "LEARNED_SET",
    "LEARNED_STATES",
    "MEDIATION_TARGET_SCHEMA",
    "NULL_SET_COUNT",
    "NULL_INTERVENTION_STATE",
    "PATH_NAMES",
    "PATH_STATE_PAIRS",
    "RESERVE_BLOCKS",
    "RESERVE_CELLS",
    "STATE_OPERATIONS",
    "STATE_TABLES",
    "SESOI",
    "TOKEN_SUPPORT",
    "adjudicate_mediation",
    "analyze_mediation",
    "analyze_compact_mediation",
    "block_mediation_effects",
    "candidate_freeze_sha256",
    "compact_block_mediation_effects",
    "compact_mediation_rows",
    "exact_add_one_path_p",
    "expected_null_labels",
    "null_label",
    "path_specificity",
    "validate_candidate_freeze",
    "validate_compact_mediation",
    "validate_compact_relocation_binding",
    "validate_mediation_rows",
    "validate_relocation_binding",
]
