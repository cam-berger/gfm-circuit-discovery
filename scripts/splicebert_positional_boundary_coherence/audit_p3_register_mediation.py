#!/usr/bin/env python
"""Independent raw-row and inference audit for P3 register mediation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_register_mediation import (  # noqa: E402
    CONFIRMATORY_BOOTSTRAP_DRAWS,
    INFERENCE_SEED,
    LEARNED_SET,
    LEARNED_STATES,
    MEDIATION_TARGET_SCHEMA,
    NULL_SET_COUNT,
    PATH_NAMES,
    PATH_STATE_PAIRS,
    RESERVE_BLOCKS,
    RESERVE_CELLS,
    SESOI,
    candidate_freeze_sha256,
    expected_null_labels,
    validate_candidate_freeze,
)
from gfmcd.splicing.pbc_register_relocation import (  # noqa: E402
    PHASES,
    REGISTERS,
    preferred_delta,
    stratified_studentized_max_t,
)
from gfmcd.splicing.pbc_register_stage import verify_register_target_seal  # noqa: E402


ANALYSIS_SCHEMA = "gfmcd.pbc.p3_register_mediation_analysis.v1"
AUDIT_SCHEMA = "gfmcd.pbc.p3_register_mediation_audit.v1"

_NULL_STATE = {
    "d_path": "scrub",
    "r_path": "restore",
    "q_path": "sufficiency",
}
_PATCH_STATES = {"scrub", "restore", "sufficiency"}
_OPERATIONS = {
    "raw": "none",
    "scrub": "replace_a_r_with_same_state_a_0",
    "restore": "replace_with_same_state_a_r_from_intact_band",
    "attenuated_host": "none",
    "sufficiency": "inject_same_state_delta_a_r_minus_a_0",
    "no_ramp_host": "none",
}
_TABLES = {
    "raw": "intact_band_ramp",
    "scrub": "intact_band_ramp",
    "restore": "attenuated_band_ramp",
    "attenuated_host": "attenuated_band_ramp",
    "sufficiency": "intact_band_no_ramp",
    "no_ramp_host": "intact_band_no_ramp",
}


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(path)
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    if output == manifest:
        raise ValueError("audit and manifest paths must be distinct")
    return output, manifest


def _finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite primitive endpoint")
    return result


def _phase_delta(donor: int, acceptor: int) -> int:
    if donor not in PHASES or acceptor not in PHASES:
        raise ValueError("invalid phase")
    return (acceptor - donor) % 3


def _sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _summary(values: Mapping[tuple[int, int, int], float]) -> dict[str, object]:
    scores: dict[str, float] = {}
    for register in REGISTERS:
        by_delta = {delta: [] for delta in PHASES}
        for donor in PHASES:
            for acceptor in PHASES:
                by_delta[_phase_delta(donor, acceptor)].append(
                    float(values[(register, donor, acceptor)])
                )
        means = {delta: float(np.mean(local)) for delta, local in by_delta.items()}
        preferred = preferred_delta(register)
        scores[f"s_{register}"] = means[preferred] - float(
            np.mean([means[delta] for delta in PHASES if delta != preferred])
        )
    return {"omega3": float(np.mean(list(scores.values()))), "s": scores}


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _condition(label: str, state: str, register: int) -> str:
    return f"{label}|{state}|r{register}"


def _independent_blocks(
    target: Mapping[str, object],
    freeze: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Independently validate and reduce the normalized compact lattice."""

    labels = {LEARNED_SET: tuple(LEARNED_STATES)}
    component_ids = {
        LEARNED_SET: tuple(
            str(row["component_id"]) for row in freeze["learned_set"]["components"]
        )
    }
    component_hooks = {
        LEARNED_SET: sorted(
            {str(row["hook_site"]) for row in freeze["learned_set"]["components"]}
        )
    }
    for path_name in PATH_NAMES:
        for spec in freeze["null_families"][path_name]:
            label = str(spec["label"])
            labels[label] = (_NULL_STATE[path_name],)
            component_ids[label] = tuple(
                str(row["component_id"]) for row in spec["components"]
            )
            component_hooks[label] = sorted(
                {str(row["hook_site"]) for row in spec["components"]}
            )
    expected_conditions = {
        _condition(label, state, register)
        for label, states in labels.items()
        for state in states
        for register in REGISTERS
    }
    ledger = target.get("provenance_ledger")
    cells = target.get("cell_ledger")
    endpoints = target.get("endpoint_rows")
    normalization = target.get("normalization")
    if (
        not isinstance(ledger, Mapping)
        or set(ledger) != expected_conditions
        or not isinstance(cells, list)
        or not isinstance(endpoints, list)
        or not isinstance(normalization, Mapping)
    ):
        raise ValueError("compact mediation ledgers are absent or incomplete")
    metadata: dict[str, tuple[str, str]] = {}
    cell_keys = set()
    for cell in cells:
        block = str(cell["block_id"])
        donor = int(cell["donor_phase"])
        acceptor = int(cell["acceptor_phase"])
        chrom, strand = str(cell["chrom"]), str(cell["strand"])
        if (
            donor not in PHASES
            or acceptor not in PHASES
            or int(cell["phase_delta"]) != _phase_delta(donor, acceptor)
            or cell.get("cell_id") != f"{block}:{donor}:{acceptor}"
            or strand not in {"+", "-"}
            or not _sha256(cell.get("intact_sequence_sha256"))
            or not _sha256(cell.get("acceptor_ac_sequence_sha256"))
        ):
            raise ValueError("invalid compact cell identity")
        if block in metadata and metadata[block] != (chrom, strand):
            raise ValueError("compact block metadata changes")
        metadata[block] = (chrom, strand)
        cell_keys.add((block, donor, acceptor))
    expected_cells = {
        (block, donor, acceptor)
        for block in metadata
        for donor in PHASES
        for acceptor in PHASES
    }
    if cell_keys != expected_cells or len(cell_keys) != len(cells):
        raise ValueError("compact cell ledger is incomplete or duplicated")

    for key, row in ledger.items():
        label = str(row["component_set"])
        state = str(row["causal_state"])
        register = int(row["register"])
        events = row.get("events")
        patched = state in _PATCH_STATES
        path = "learned" if label == LEARNED_SET else next(
            name for name in PATH_NAMES if label in expected_null_labels(name)
        )
        if (
            key != _condition(label, state, register)
            or state not in labels[label]
            or row.get("path_contrast") != path
            or row.get("activation_operation") != _OPERATIONS[state]
            or row.get("host_table_state") != _TABLES[state]
            or row.get("patched") is not patched
            or int(row.get("source_rows", -1)) != len(metadata) * 9
            or not isinstance(events, list)
            or row.get("tensor_stream_sha256") != _canonical(events)
        ):
            raise ValueError("compact condition provenance mismatch")
        if patched:
            if (
                list(row.get("components", [])) != list(component_ids[label])
                or list(row.get("hook_sites", [])) != component_hooks[label]
                or row.get("token_support") != [393, 513]
                or int(row.get("source_register", -99))
                != (0 if state == "scrub" else register)
                or row.get("source_baseline_register")
                != (0 if state == "sufficiency" else None)
                or not events
                or sum(int(event.get("source_rows", -1)) for event in events)
                != len(metadata) * 9
                or any(
                    int(event.get("source_rows", 0)) <= 0
                    or any(
                        not _sha256(event.get(name))
                        for name in (
                            "intact_patch_tensor_sha256",
                            "acceptor_ac_patch_tensor_sha256",
                            "combined_patch_tensor_sha256",
                        )
                    )
                    for event in events
                )
            ):
                raise ValueError("compact patch event ledger mismatch")
        elif events or row.get("components") or row.get("hook_sites"):
            raise ValueError("unpatched compact condition declares patch events")

    values: dict[tuple[str, str, str], dict[tuple[int, int], float]] = defaultdict(dict)
    endpoint_keys = set()
    for row in endpoints:
        block = str(row["block_id"])
        condition_id = str(row["condition_id"])
        register = int(row["register"])
        delta = int(row["phase_delta"])
        condition = ledger[condition_id]
        if (
            block not in metadata
            or register != int(condition["register"])
            or delta not in PHASES
            or int(row["n_phase_cells"]) != 3
            or (str(row["chrom"]), str(row["strand"])) != metadata[block]
        ):
            raise ValueError("invalid compact endpoint identity")
        intact = _finite(row["mean_intact_margin"])
        corrupt = _finite(row["mean_acceptor_ac_margin"])
        coupling = _finite(row["mean_coupling"])
        if abs(coupling - (intact - corrupt)) > 1e-8:
            raise ValueError("compact endpoint arithmetic mismatch")
        key = (block, condition_id, delta)
        if key in endpoint_keys:
            raise ValueError("duplicate compact endpoint")
        endpoint_keys.add(key)
        values[(block, str(condition["component_set"]), str(condition["causal_state"]))][
            (register, delta)
        ] = coupling
    expected_endpoint_keys = {
        (block, condition, delta)
        for block in metadata
        for condition in expected_conditions
        for delta in PHASES
    }
    if endpoint_keys != expected_endpoint_keys:
        raise ValueError("compact endpoint lattice is incomplete")
    if (
        int(normalization.get("primitive_rows", -1))
        != len(metadata) * len(expected_conditions) * 9
        or int(normalization.get("compact_endpoint_rows", -1)) != len(endpoints)
        or normalization.get("provenance_ledger_sha256") != _canonical(ledger)
    ):
        raise ValueError("compact normalization metadata drifted")

    def summary(local: Mapping[tuple[int, int], float]) -> dict[str, object]:
        scores = {}
        for register in REGISTERS:
            preferred = preferred_delta(register)
            others = [local[(register, delta)] for delta in PHASES if delta != preferred]
            scores[f"s_{register}"] = float(local[(register, preferred)]) - float(
                math.fsum(others) / len(others)
            )
        return {
            "omega3": float(math.fsum(scores.values()) / len(scores)),
            "s": scores,
        }

    output = []
    for block in sorted(metadata):
        states = {
            state: summary(values[(block, LEARNED_SET, state)])
            for state in LEARNED_STATES
        }
        paths = {
            name: float(states[left]["omega3"]) - float(states[right]["omega3"])
            for name, (left, right) in PATH_STATE_PAIRS.items()
        }
        register_paths = {
            name: {
                f"s_{register}": float(states[left]["s"][f"s_{register}"])
                - float(states[right]["s"][f"s_{register}"])
                for register in REGISTERS
            }
            for name, (left, right) in PATH_STATE_PAIRS.items()
        }
        null_paths = {}
        for name in PATH_NAMES:
            intervention = _NULL_STATE[name]
            left, right = PATH_STATE_PAIRS[name]
            baseline_state = left if intervention == right else right
            baseline = float(states[baseline_state]["omega3"])
            null_paths[name] = {}
            for label in expected_null_labels(name):
                intervened = float(summary(values[(block, label, intervention)])["omega3"])
                null_paths[name][label] = (
                    baseline - intervened if intervention == right else intervened - baseline
                )
        chrom, strand = metadata[block]
        output.append(
            {
                "block_id": block,
                "chrom": chrom,
                "strand": strand,
                "state_register_effects": states,
                "paths": paths,
                "path_register_effects": register_paths,
                "null_paths": null_paths,
            }
        )
    return tuple(output)


def _independent_inference(
    blocks: Sequence[Mapping[str, object]], seed: int, draws: int
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    strata = [f"{block['chrom']}:{block['strand']}" for block in blocks]
    raw_names = ("omega3", "s_-1", "s_0", "s_1")
    raw = stratified_studentized_max_t(
        np.asarray(
            [
                [
                    float(block["state_register_effects"]["raw"]["omega3"]),
                    *[
                        float(block["state_register_effects"]["raw"]["s"][f"s_{r}"])
                        for r in REGISTERS
                    ],
                ]
                for block in blocks
            ]
        ),
        names=raw_names,
        strata=strata,
        seed=seed,
        n_bootstrap=draws,
    ).as_dict()
    path = stratified_studentized_max_t(
        np.asarray([[float(block["paths"][name]) for name in PATH_NAMES] for block in blocks]),
        names=PATH_NAMES,
        strata=strata,
        seed=seed,
        n_bootstrap=draws,
    ).as_dict()
    rank_tests = {}
    paired = []
    for block in blocks:
        paired.append(
            [
                float(block["paths"][name])
                - float(np.mean(list(block["null_paths"][name].values())))
                for name in PATH_NAMES
            ]
        )
    for name in PATH_NAMES:
        target = float(np.mean([float(block["paths"][name]) for block in blocks]))
        controls = {
            label: float(
                np.mean([float(block["null_paths"][name][label]) for block in blocks])
            )
            for label in expected_null_labels(name)
        }
        p_value = (1 + sum(value >= target for value in controls.values())) / (
            NULL_SET_COUNT + 1
        )
        rank_tests[name] = {
            "target": target,
            "controls": controls,
            "n_controls": NULL_SET_COUNT,
            "target_exceeds_every_control": target > max(controls.values()),
            "exact_add_one_p_one_sided": p_value,
            "threshold": 1 / 32,
            "passed": target > max(controls.values()) and p_value <= 1 / 32,
        }
    paired_names = tuple(f"{name}_minus_mean_null" for name in PATH_NAMES)
    paired_inference = stratified_studentized_max_t(
        np.asarray(paired),
        names=paired_names,
        strata=strata,
        seed=seed,
        n_bootstrap=draws,
    ).as_dict()
    specificity = {
        "rank_tests": rank_tests,
        "paired_learned_minus_mean_null": paired_inference,
        "separate_null_families": True,
        "passed": all(test["passed"] for test in rank_tests.values())
        and all(float(paired_inference["lower"][name]) > 0 for name in paired_names),
    }
    return raw, path, specificity


def _independent_decision(
    path: Mapping[str, object], specificity: Mapping[str, object], capable: bool
) -> dict[str, str]:
    if not capable:
        return {
            "code": "P3_ROUTE_INSTRUMENT_UNRESOLVED",
            "licensed_claim": (
                "Register relocation may be reported; route interpretation is withheld."
            ),
        }
    lower = path["lower"]
    upper = path["upper"]
    if all(float(lower[name]) > SESOI for name in PATH_NAMES) and specificity["passed"]:
        return {
            "code": "P3_REGISTER_MEDIATED",
            "licensed_claim": (
                "The frozen component path is necessary, rescuable, and separately "
                "sufficient for a scoped portion of the relocation."
            ),
        }
    if all(
        float(lower[name]) >= -SESOI and float(upper[name]) <= SESOI for name in PATH_NAMES
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


def _source_record(record: object) -> tuple[Path, str]:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise ValueError("invalid source record")
    path = Path(record["path"])
    digest = sha256_file(path)
    if record.get("sha256") != digest or record.get("size_bytes") != path.stat().st_size:
        raise ValueError("source hash/size mismatch")
    return path, digest


def _independent_relocation_binding(target: Mapping[str, object]) -> bool:
    sources = target["sources"]
    records = {
        name: _source_record(sources[name])
        for name in ("relocation_target", "relocation_analysis", "relocation_audit")
    }
    relocation_target = json.loads(records["relocation_target"][0].read_text())
    relocation_analysis = json.loads(records["relocation_analysis"][0].read_text())
    relocation_audit = json.loads(records["relocation_audit"][0].read_text())
    target_hash = records["relocation_target"][1]
    status = str(relocation_analysis.get("status") or "")
    if not (
        relocation_target.get("schema") == "gfmcd.pbc.p3_register_target.v1"
        and relocation_target.get("status") == "COMPLETE"
        and relocation_analysis.get("schema") == "gfmcd.pbc.p3_register_analysis.v1"
        and status.startswith("P3_REGISTER_RELOCATION_SUPPORTED")
        and relocation_analysis.get("target_sha256") == target_hash
        and relocation_analysis.get("decision", {}).get("code") == status
        and relocation_audit.get("schema") == "gfmcd.pbc.p3_register_audit.v1"
        and relocation_audit.get("status") == "PASS"
        and relocation_audit.get("target_sha256") == target_hash
    ):
        return False
    authorization = target["authorization"]
    expected = {
        "relocation_target_sha256": records["relocation_target"][1],
        "relocation_analysis_sha256": records["relocation_analysis"][1],
        "relocation_audit_sha256": records["relocation_audit"][1],
        "relocation_status": status,
        "relocation_audit_status": "PASS",
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        return False
    grouped: dict[tuple[str, int, int], list[tuple[str, str, float, float, float]]] = defaultdict(list)
    for row in relocation_target["rows"]:
        if row.get("component") == "p3" and row.get("p10_state") == "intact":
            grouped[
                (
                    str(row["block_id"]),
                    int(row["register"]),
                    _phase_delta(int(row["donor_phase"]), int(row["acceptor_phase"])),
                )
            ].append(
                (
                    str(row["chrom"]),
                    str(row["strand"]),
                    _finite(row["intact_margin"]),
                    _finite(row["corrupt_margin"]),
                    _finite(row["coupling"]),
                )
            )
    original = {
        key: (
            values[0][0],
            values[0][1],
            *(
                math.fsum(value[index] for value in values) / len(values)
                for index in (2, 3, 4)
            ),
        )
        for key, values in grouped.items()
        if len(values) == 3 and len({value[:2] for value in values}) == 1
    }
    ledger = target.get("provenance_ledger")
    endpoints = target.get("endpoint_rows")
    if not isinstance(ledger, Mapping) or not isinstance(endpoints, list):
        return False
    replay = {}
    for row in endpoints:
        condition = ledger.get(str(row.get("condition_id")))
        if (
            isinstance(condition, Mapping)
            and condition.get("component_set") == LEARNED_SET
            and condition.get("causal_state") == "raw"
        ):
            key = (
                str(row["block_id"]),
                int(row["register"]),
                int(row["phase_delta"]),
            )
            replay[key] = (
                str(row["chrom"]),
                str(row["strand"]),
                _finite(row["mean_intact_margin"]),
                _finite(row["mean_acceptor_ac_margin"]),
                _finite(row["mean_coupling"]),
            )
    mechanics = target.get("mechanics")
    replay_fields = (
        "raw_relocation_replay_intact_margin_max_abs",
        "raw_relocation_replay_corrupt_margin_max_abs",
        "raw_relocation_replay_coupling_max_abs",
    )
    return bool(
        original
        and len(original) == len(grouped)
        and original == replay
        and isinstance(mechanics, Mapping)
        and all(float(mechanics.get(name, math.inf)) <= 1e-8 for name in replay_fields)
    )


def audit_payloads(
    target: Mapping[str, object],
    analysis: Mapping[str, object],
    *,
    target_sha256: str,
    enforce_confirmatory_contract: bool = True,
) -> dict[str, object]:
    recorded_seed = int(analysis.get("inference_seed", -1))
    recorded_draws = int(analysis.get("bootstrap_draws", -1))
    target_rows = target.get("cell_ledger")
    unique_cells = (
        {
            (
                str(row.get("block_id")),
                int(row.get("donor_phase", -1)),
                int(row.get("acceptor_phase", -1)),
            )
            for row in target_rows
            if isinstance(row, Mapping)
        }
        if isinstance(target_rows, list)
        else set()
    )
    target_design = target.get("design")
    mechanics = target.get("mechanics")
    reserve_contract = bool(
        target.get("stage") == "mediation"
        and target.get("split_role") == "phase3_reserve"
        and int(target.get("reserve_blocks", -1)) == RESERVE_BLOCKS
        and int(target.get("reserve_cells", -1)) == RESERVE_CELLS
        and int(target.get("n_blocks", -1)) == RESERVE_BLOCKS
        and int(target.get("n_cells", -1)) == RESERVE_CELLS
        and len(unique_cells) == RESERVE_CELLS
    )
    hook_contract = bool(
        isinstance(mechanics, Mapping)
        and mechanics.get("passed") is True
        and mechanics.get("same_sequence_patching") is True
        and mechanics.get("same_corruption_state_patching") is True
        and mechanics.get("full_token_support") == [393, 513]
        and all(
            float(mechanics.get(name, math.inf)) <= 1e-8
            for name in (
                "raw_relocation_replay_intact_margin_max_abs",
                "raw_relocation_replay_corrupt_margin_max_abs",
                "raw_relocation_replay_coupling_max_abs",
            )
        )
        and _sha256(mechanics.get("actual_hook_tensor_stream_sha256"))
        and int(mechanics.get("actual_patch_forward_events", 0)) > 0
    )
    design_contract = bool(
        isinstance(target_design, Mapping)
        and int(target_design.get("inference_seed", -1)) == INFERENCE_SEED
        and int(target_design.get("bootstrap_draws", -1))
        == CONFIRMATORY_BOOTSTRAP_DRAWS
    )
    checks: dict[str, bool] = {
        "target_schema": target.get("schema") == MEDIATION_TARGET_SCHEMA
        and target.get("status") == "COMPLETE",
        "analysis_schema_and_stage": analysis.get("schema") == ANALYSIS_SCHEMA
        and analysis.get("stage") == "mediation",
        "target_hash_bound": analysis.get("target_sha256") == target_sha256,
        "confirmatory_seed": (
            recorded_seed == INFERENCE_SEED or not enforce_confirmatory_contract
        ),
        "confirmatory_draws": (
            recorded_draws == CONFIRMATORY_BOOTSTRAP_DRAWS
            or not enforce_confirmatory_contract
        ),
        "confirmatory_reserve_384_blocks_3456_cells": (
            reserve_contract or not enforce_confirmatory_contract
        ),
        "confirmatory_target_design": (
            design_contract or not enforce_confirmatory_contract
        ),
        "executed_same_state_hook_contract": (
            hook_contract or not enforce_confirmatory_contract
        ),
    }
    error = None
    recomputed_status = None
    try:
        design = target["design"]
        freeze = design["candidate_freeze"]
        freeze_hash = candidate_freeze_sha256(freeze)
        checks["candidate_freeze_hash"] = (
            design.get("candidate_freeze_sha256") == freeze_hash
            and analysis.get("candidate_freeze_sha256") == freeze_hash
        )
        freeze_validation = validate_candidate_freeze(freeze)
        checks["candidate_and_null_matching"] = bool(freeze_validation["passed"])
        checks["relocation_binding"] = _independent_relocation_binding(target)
        blocks = _independent_blocks(target, freeze)
        checks["complete_same_state_lattice"] = True
        raw, path, specificity = _independent_inference(
            blocks, recorded_seed, recorded_draws
        )
        capable = bool(target["authorization"].get("path_instrument_capable"))
        if enforce_confirmatory_contract:
            capable = capable and reserve_contract and hook_contract and design_contract
        decision = _independent_decision(path, specificity, capable)
        recomputed_status = decision["code"]
        checks["per_block_exact_recompute"] = analysis.get("per_block") == list(blocks)
        checks["raw_inference_exact_recompute"] = (
            analysis.get("relocation_replay_inference") == raw
        )
        checks["path_inference_exact_recompute"] = analysis.get("path_inference") == path
        checks["specificity_exact_recompute"] = analysis.get("specificity") == specificity
        checks["decision_exact_recompute"] = (
            analysis.get("decision") == decision and analysis.get("status") == decision["code"]
        )
        firewall = analysis.get("reporting_firewall")
        checks["absolute_effect_firewall"] = bool(
            isinstance(firewall, Mapping)
            and firewall.get("absolute_effects_primary") is True
            and firewall.get("fractional_mediation_reported") is False
        )
        evidence_firewall = analysis.get("evidence_firewall")
        checks["evidence_level_firewall"] = bool(
            isinstance(evidence_firewall, Mapping)
            and evidence_firewall.get("output_mediation_adjudicated") is True
            and evidence_firewall.get("selector_claim_adjudicated") is False
            and evidence_firewall.get("attention_router_claim_adjudicated") is False
            and evidence_firewall.get("directed_edge_claim_adjudicated") is False
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        checks.setdefault("candidate_freeze_hash", False)
        checks.setdefault("candidate_and_null_matching", False)
        checks.setdefault("relocation_binding", False)
        checks.setdefault("complete_same_state_lattice", False)
        checks.setdefault("per_block_exact_recompute", False)
        checks.setdefault("raw_inference_exact_recompute", False)
        checks.setdefault("path_inference_exact_recompute", False)
        checks.setdefault("specificity_exact_recompute", False)
        checks.setdefault("decision_exact_recompute", False)
        checks.setdefault("absolute_effect_firewall", False)
        checks.setdefault("evidence_level_firewall", False)
    passed = all(checks.values())
    source_records = target.get("sources")
    source_paths = (
        [Path(str(record.get("path", ""))).resolve() for record in source_records.values()]
        if isinstance(source_records, Mapping)
        and all(isinstance(record, Mapping) for record in source_records.values())
        else []
    )
    inputs_under_repo = bool(source_paths) and all(
        path == REPO or REPO in path.parents for path in source_paths
    )
    return {
        "schema": AUDIT_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "stage": "mediation",
        "target_sha256": target_sha256,
        "analysis_sha256": str(analysis.get("self_sha256") or ""),
        "checks": checks,
        "recompute_error": error,
        "recomputed_status": recomputed_status,
        "clean_checkout_reexecutable": bool(passed and inputs_under_repo),
        "reexecution_basis": {
            "normalized_sufficient_endpoints_embedded": bool(
                checks.get("complete_same_state_lattice")
            ),
            "raw_shards_required": False,
            "bound_sources_under_repository": inputs_under_repo,
            "external_activation_capture_required": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    target_path = Path(args.target).resolve()
    analysis_path = Path(args.analysis).resolve()
    output_path, manifest_path = output_paths(Path(args.out).resolve())
    target = json.loads(target_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    sources = target.get("sources")
    seal_record = sources.get("reserve_seal") if isinstance(sources, Mapping) else None
    if not isinstance(seal_record, Mapping):
        raise RuntimeError("confirmatory mediation audit lacks the original reserve seal")
    seal_path = Path(str(seal_record.get("path", ""))).resolve()
    if (
        not seal_path.is_file()
        or seal_record.get("sha256") != sha256_file(seal_path)
        or seal_record.get("size_bytes") != seal_path.stat().st_size
    ):
        raise RuntimeError("mediation reserve seal changed before audit")
    seal = json.loads(seal_path.read_text())
    verify_register_target_seal(seal)
    if target_path != Path(str(seal["mediation"]["target_output_path"])).resolve():
        raise RuntimeError("mediation target path differs from the pre-reserve seal")
    if analysis_path != Path(str(seal["mediation"]["analysis_output_path"])).resolve():
        raise RuntimeError("mediation analysis path differs from the pre-reserve seal")
    if output_path != Path(str(seal["mediation"]["audit_output_path"])).resolve():
        raise RuntimeError("mediation audit path differs from the pre-reserve seal")
    analysis["self_sha256"] = sha256_file(analysis_path)
    audit = audit_payloads(target, analysis, target_sha256=sha256_file(target_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        experiment_id="pbc-p3-register-mediation-audit",
        command=sys.argv,
        repo=REPO,
        status="AUDIT",
        config={
            "schema": AUDIT_SCHEMA,
            "stage": "mediation",
            "seed": INFERENCE_SEED,
            "bootstrap_draws": CONFIRMATORY_BOOTSTRAP_DRAWS,
        },
        seeds={"whole_block_bootstrap": INFERENCE_SEED},
        inputs=[file_record(target_path), file_record(analysis_path)],
        outputs=[file_record(output_path)],
        notes=[
            "Independent primitive-row reduction, max-studentized inference, null "
            "ranking, and decision recomputation."
        ],
    )
    manifest.write(manifest_path)
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
