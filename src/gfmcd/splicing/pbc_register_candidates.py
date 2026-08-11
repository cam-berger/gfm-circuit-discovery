"""Candidate-screen and freeze machinery for P3 register mediation.

The development screen is observational/discovery-level.  This module turns its
recorded primitive endpoints into a deterministic candidate freeze; it never reads
the Phase-III reserve and never loads a model.  The independent audit intentionally
does not import the selection functions below.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from gfmcd.provenance import sha256_file


CANDIDATE_SCREEN_SCHEMA = "gfmcd.pbc.p3_register_candidate_screen.v1"
CANDIDATE_FREEZE_SCHEMA = "gfmcd.pbc.p3_register_candidate_freeze.v1"
CANDIDATE_FREEZE_AUDIT_SCHEMA = (
    "gfmcd.pbc.p3_register_candidate_freeze_audit.v1"
)
PATH_MDE_SCHEMA = "gfmcd.pbc.p3_register_path_mde.v1"

TOKEN_SUPPORT = (393, 513)
DEVELOPMENT_BLOCKS = 120
RESERVE_BLOCKS = 384
NULL_SET_COUNT = 31
PREFIX_CAP = 8
PREFIX_FRACTION = 0.80
SESOI = 0.05
PATH_NAMES = ("d_path", "r_path", "q_path")
NULL_SEEDS = {
    "d_path": 2026081201,
    "r_path": 2026081202,
    "q_path": 2026081203,
}


def canonical_json_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def component_universe() -> tuple[dict[str, object], ...]:
    """Return the frozen six-layer, sixteen-head plus six-MLP census."""

    heads = tuple(
        {
            "component_id": f"L{layer}H{head}",
            "layer": layer,
            "kind": "head",
            "head": head,
            "hook_site": "attention.self.context_pre_output_projection",
            "patch_token_count": TOKEN_SUPPORT[1] - TOKEN_SUPPORT[0],
        }
        for layer in range(6)
        for head in range(16)
    )
    mlps = tuple(
        {
            "component_id": f"MLP{layer}",
            "layer": layer,
            "kind": "mlp",
            "head": None,
            "hook_site": "encoder.layer.output.dense_pre_residual_add",
            "patch_token_count": TOKEN_SUPPORT[1] - TOKEN_SUPPORT[0],
        }
        for layer in range(6)
    )
    return heads + mlps


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _file_record_matches(record: Mapping[str, object]) -> bool:
    path = Path(str(record.get("path", ""))).expanduser()
    return bool(
        path.is_file()
        and type(record.get("size_bytes")) is int
        and path.stat().st_size == int(record["size_bytes"])
        and isinstance(record.get("sha256"), str)
        and sha256_file(path) == record["sha256"]
    )


def validate_candidate_screen(
    screen: Mapping[str, object],
    *,
    enforce_development_contract: bool = True,
) -> dict[str, object]:
    """Validate that a screen is an executed 102-component measurement record."""

    if (
        screen.get("schema") != CANDIDATE_SCREEN_SCHEMA
        or screen.get("status") != "COMPLETE"
        or screen.get("stage") != "development"
    ):
        raise ValueError("candidate screen is incomplete or has the wrong schema/stage")
    if list(screen.get("token_support", [])) != list(TOKEN_SUPPORT):
        raise ValueError("candidate screen must patch token support [393,513)")
    n_blocks = int(screen.get("n_blocks", -1))
    if n_blocks < 6 or (enforce_development_contract and n_blocks != DEVELOPMENT_BLOCKS):
        raise ValueError("candidate screen has the wrong development block count")
    universe = screen.get("candidate_universe")
    expected = component_universe()
    if universe != list(expected):
        raise ValueError("candidate screen does not contain the exact 102-component census")
    singletons = screen.get("singletons")
    prefixes = screen.get("joint_prefixes")
    if not isinstance(singletons, list) or len(singletons) != len(expected):
        raise ValueError("candidate screen needs exactly 102 singleton records")
    expected_ids = [str(row["component_id"]) for row in expected]
    observed_ids = [str(row.get("component_id", "")) for row in singletons]
    if set(observed_ids) != set(expected_ids) or len(set(observed_ids)) != len(expected_ids):
        raise ValueError("candidate screen singleton census is incomplete or duplicated")
    blocks: set[tuple[str, str, str]] | None = None
    for row in singletons:
        path_norms = row.get("path_output_delta_norms")
        if not isinstance(path_norms, Mapping) or set(path_norms) != set(PATH_NAMES):
            raise ValueError("every singleton needs measured D/R/Q output-delta norms")
        if any(_finite(path_norms[name], f"{name} output norm") <= 0 for name in PATH_NAMES):
            raise ValueError("singleton output-delta norms must be positive")
        per_block = row.get("per_block")
        if not isinstance(per_block, list) or len(per_block) != n_blocks:
            raise ValueError("every singleton needs one D/R/Q endpoint per development block")
        local: set[tuple[str, str, str]] = set()
        for endpoint in per_block:
            identity = (
                str(endpoint.get("block_id", "")),
                str(endpoint.get("chrom", "")),
                str(endpoint.get("strand", "")),
            )
            if not identity[0] or not identity[1] or identity[2] not in {"+", "-"}:
                raise ValueError("candidate endpoint lacks block/stratum identity")
            _finite(endpoint.get("D_path"), "singleton D_path")
            _finite(endpoint.get("R_path"), "singleton R_path")
            _finite(endpoint.get("Q_path"), "singleton Q_path")
            local.add(identity)
        if len(local) != n_blocks:
            raise ValueError("candidate singleton repeats a development block")
        if blocks is None:
            blocks = local
        elif local != blocks:
            raise ValueError("candidate singletons do not share one block support")
        measured_d = float(np.mean([float(value["D_path"]) for value in per_block]))
        measured_r = float(np.mean([float(value["R_path"]) for value in per_block]))
        measured_q = float(np.mean([float(value["Q_path"]) for value in per_block]))
        if abs(measured_d - _finite(row.get("same_state_scrub_loss_in_omega3"), "rank endpoint")) > 1e-10:
            raise ValueError("singleton rank endpoint is not rederived from its block rows")
        if abs(measured_r - _finite(row.get("same_state_restore_gain_in_omega3"), "restore endpoint")) > 1e-10:
            raise ValueError("singleton restoration endpoint is not rederived from its block rows")
        if abs(measured_q - _finite(row.get("same_state_sufficiency_gain_in_omega3"), "sufficiency endpoint")) > 1e-10:
            raise ValueError("singleton sufficiency endpoint is not rederived from its block rows")
    ranking = sorted(
        singletons,
        key=lambda row: (
            -float(row["same_state_scrub_loss_in_omega3"]),
            str(row["component_id"]),
        ),
    )
    ranked_ids = [str(row["component_id"]) for row in ranking]
    if not isinstance(prefixes, list) or len(prefixes) != PREFIX_CAP:
        raise ValueError("candidate screen needs jointly remeasured prefixes 1..8")
    for index, prefix in enumerate(prefixes, start=1):
        if int(prefix.get("prefix_size", -1)) != index:
            raise ValueError("joint prefixes are not ordered 1..8")
        if list(prefix.get("component_ids", [])) != ranked_ids[:index]:
            raise ValueError("joint prefix does not follow the singleton ranking")
        local = prefix.get("per_block")
        if not isinstance(local, list) or len(local) != n_blocks:
            raise ValueError("joint prefix lacks complete block endpoints")
        local_ids = {
            (str(row.get("block_id")), str(row.get("chrom")), str(row.get("strand")))
            for row in local
        }
        if local_ids != blocks:
            raise ValueError("joint prefix block support differs from the singleton screen")
        mean_d = float(np.mean([_finite(row.get("D_path"), "prefix D_path") for row in local]))
        mean_r = float(np.mean([_finite(row.get("R_path"), "prefix R_path") for row in local]))
        mean_q = float(np.mean([_finite(row.get("Q_path"), "prefix Q_path") for row in local]))
        if abs(mean_d - _finite(prefix.get("D_path"), "prefix D_path mean")) > 1e-10:
            raise ValueError("joint prefix D_path is not rederived from block endpoints")
        if abs(mean_r - _finite(prefix.get("R_path"), "prefix R_path mean")) > 1e-10:
            raise ValueError("joint prefix R_path is not rederived from block endpoints")
        if abs(mean_q - _finite(prefix.get("Q_path"), "prefix Q_path mean")) > 1e-10:
            raise ValueError("joint prefix Q_path is not rederived from block endpoints")
    tensor_stream = screen.get("tensor_stream")
    if not isinstance(tensor_stream, Mapping):
        raise ValueError("candidate screen lacks tensor-stream provenance")
    digest = str(tensor_stream.get("sha256", ""))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("candidate screen tensor stream has no valid sha256")
    if (
        tensor_stream.get("same_sequence_only") is not True
        or tensor_stream.get("same_corruption_state_only") is not True
        or tensor_stream.get("full_token_support_only") is not True
        or int(tensor_stream.get("patch_events", 0)) <= 0
    ):
        raise ValueError("candidate screen patch provenance is declarative or incomplete")
    levels = screen.get("evidence_levels")
    if not isinstance(levels, Mapping) or set(levels) != {
        "qk_pre_softmax",
        "attention_post_softmax",
        "component_output",
        "behavior_same_state_patch",
    }:
        raise ValueError("candidate screen does not keep the four evidence levels separate")
    baseline = screen.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("candidate screen lacks the raw register-effect baseline")
    raw = _finite(baseline.get("mean_raw_omega3"), "raw register effect")
    if raw <= 0:
        raise ValueError("candidate selection is unidentified without positive raw omega3")
    return {
        "n_blocks": n_blocks,
        "ranked_ids": ranked_ids,
        "block_support": sorted(blocks or ()),
        "mean_raw_omega3": raw,
        "tensor_stream_sha256": digest,
        "passed": True,
    }


def _joint_norm(
    components: Sequence[str],
    singleton_by_id: Mapping[str, Mapping[str, object]],
    path_name: str,
) -> float:
    # Components occupy distinct hook coordinates.  The frozen joint norm is the
    # Euclidean norm of their measured concatenated output-delta streams.
    squares = [
        _finite(singleton_by_id[value]["path_output_delta_norms"][path_name], path_name)
        ** 2
        for value in components
    ]
    return float(math.sqrt(math.fsum(squares)))


def _draw_layer_kind_matched_set(
    *,
    rng: np.random.Generator,
    universe: Sequence[Mapping[str, object]],
    learned_ids: set[str],
    layer_counts: Counter[int],
    kind_counts: Counter[str],
) -> tuple[str, ...] | None:
    """Draw one set matching marginal layer counts and marginal component type."""

    by_layer: dict[int, dict[str, list[str]]] = {}
    for layer in range(6):
        by_layer[layer] = {"head": [], "mlp": []}
    for row in universe:
        component_id = str(row["component_id"])
        if component_id not in learned_ids:
            by_layer[int(row["layer"])][str(row["kind"])].append(component_id)
    layers_with_slots = [layer for layer, count in layer_counts.items() if count]
    n_mlp = int(kind_counts.get("mlp", 0))
    feasible_mlp_layers = [
        layer
        for layer in layers_with_slots
        if by_layer[layer]["mlp"] and layer_counts[layer] >= 1
    ]
    if n_mlp > len(feasible_mlp_layers):
        return None
    if n_mlp:
        chosen_mlp_layers = set(
            int(value)
            for value in rng.choice(feasible_mlp_layers, size=n_mlp, replace=False)
        )
    else:
        chosen_mlp_layers = set()
    selected: list[str] = []
    for layer in sorted(layers_with_slots):
        count = int(layer_counts[layer])
        if layer in chosen_mlp_layers:
            selected.append(by_layer[layer]["mlp"][0])
            count -= 1
        heads = by_layer[layer]["head"]
        if count > len(heads):
            return None
        if count:
            selected.extend(
                str(value) for value in rng.choice(heads, size=count, replace=False)
            )
    if Counter(
        str(next(row for row in universe if row["component_id"] == value)["kind"])
        for value in selected
    ) != kind_counts:
        return None
    return tuple(sorted(selected))


def _matched_null_family(
    *,
    path_name: str,
    learned_ids: Sequence[str],
    universe: Sequence[Mapping[str, object]],
    singleton_by_id: Mapping[str, Mapping[str, object]],
    occupied_sets: set[tuple[str, ...]],
    max_attempts: int = 2_000_000,
) -> list[dict[str, object]]:
    learned_set = set(learned_ids)
    metadata = {str(row["component_id"]): row for row in universe}
    layers = Counter(int(metadata[value]["layer"]) for value in learned_ids)
    kinds = Counter(str(metadata[value]["kind"]) for value in learned_ids)
    target_norm = _joint_norm(learned_ids, singleton_by_id, path_name)
    rng = np.random.default_rng(NULL_SEEDS[path_name])
    accepted: list[tuple[tuple[str, ...], float]] = []
    local_seen: set[tuple[str, ...]] = set()
    for _ in range(max_attempts):
        candidate = _draw_layer_kind_matched_set(
            rng=rng,
            universe=universe,
            learned_ids=learned_set,
            layer_counts=layers,
            kind_counts=kinds,
        )
        if candidate is None:
            break
        if candidate in local_seen or candidate in occupied_sets:
            continue
        local_seen.add(candidate)
        norm = _joint_norm(candidate, singleton_by_id, path_name)
        ratio = norm / target_norm
        if 0.90 <= ratio <= 1.10:
            accepted.append((candidate, norm))
            occupied_sets.add(candidate)
            if len(accepted) == NULL_SET_COUNT:
                break
    if len(accepted) != NULL_SET_COUNT:
        raise ValueError(
            f"could not materialize 31 independent {path_name} null sets under frozen matching"
        )
    output: list[dict[str, object]] = []
    for index, (component_ids, norm) in enumerate(accepted):
        output.append(
            {
                "label": f"{path_name}_null_{index:02d}",
                "components": [dict(metadata[value]) for value in component_ids],
                "joint_output_delta_norm": norm,
                "matching_seed": NULL_SEEDS[path_name],
            }
        )
    return output


def projected_path_mde80(
    per_block: Sequence[Mapping[str, object]],
    *,
    reserve_blocks: int = RESERVE_BLOCKS,
) -> dict[str, float]:
    """Independent-of-effect normal projection from stratum-centered D/R/Q blocks."""

    rows = tuple(per_block)
    if len(rows) < 6 or reserve_blocks < 6:
        raise ValueError("path MDE projection needs development and reserve blocks")
    names = ("D_path", "R_path", "Q_path")
    matrix = np.asarray(
        [[_finite(row[name], name) for name in names] for row in rows], dtype=float
    )
    strata = np.asarray(
        [f"{row['chrom']}:{row['strand']}" for row in rows], dtype=object
    )
    centered = matrix.copy()
    for label in sorted(set(strata.tolist())):
        mask = strata == label
        if int(mask.sum()) < 2:
            raise ValueError(f"path MDE stratum {label!r} needs two development blocks")
        centered[mask] -= centered[mask].mean(axis=0)
    sd = centered.std(axis=0, ddof=1)
    multiplier = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    values = multiplier * sd / math.sqrt(int(reserve_blocks))
    return {name: float(value) for name, value in zip(names, values)}


def build_candidate_freeze(
    screen: Mapping[str, object],
    *,
    screen_record: Mapping[str, object],
    enforce_development_contract: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    """Derive the candidate, three null families, and D/R/Q resolution evidence."""

    validation = validate_candidate_screen(
        screen, enforce_development_contract=enforce_development_contract
    )
    if not _file_record_matches(screen_record):
        raise ValueError("candidate screen file record is not currently reproducible")
    universe = list(component_universe())
    singleton_by_id = {
        str(row["component_id"]): row for row in screen["singletons"]
    }
    ranked_ids = list(validation["ranked_ids"])
    raw = float(validation["mean_raw_omega3"])
    prefixes = list(screen["joint_prefixes"])
    selected_prefix = None
    for prefix in prefixes:
        fraction = float(prefix["D_path"]) / raw
        if fraction >= PREFIX_FRACTION:
            selected_prefix = prefix
            break
    sparse = selected_prefix is not None
    selected_prefix = selected_prefix or prefixes[-1]
    selected_ids = [str(value) for value in selected_prefix["component_ids"]]
    metadata = {str(row["component_id"]): row for row in universe}
    target_norms = {
        name: _joint_norm(selected_ids, singleton_by_id, name) for name in PATH_NAMES
    }
    occupied: set[tuple[str, ...]] = set()
    null_families = {
        name: _matched_null_family(
            path_name=name,
            learned_ids=selected_ids,
            universe=universe,
            singleton_by_id=singleton_by_id,
            occupied_sets=occupied,
        )
        for name in PATH_NAMES
    }
    learned_components = [dict(metadata[value]) for value in selected_ids]
    learned_norm = float(np.mean(list(target_norms.values())))
    fraction = float(selected_prefix["D_path"]) / raw
    freeze: dict[str, object] = {
        "schema": CANDIDATE_FREEZE_SCHEMA,
        "status": "COMPLETE",
        "stage": "development",
        "artifact_role": "candidate_freeze",
        "token_support": list(TOKEN_SUPPORT),
        "patch_tensor_sha256": validation["tensor_stream_sha256"],
        "sources": {"candidate_screen": dict(screen_record)},
        "candidate_screen": {
            "attention_heads": 96,
            "mlp_outputs": 6,
            "all_components_screened": True,
            "rank_metric": "same_state_scrub_loss_in_omega3",
            "prefixes_jointly_remeasured": True,
            "candidate_screen_sha256": screen_record["sha256"],
            "ranking": ranked_ids,
        },
        "selection": {
            "candidate_count": len(selected_ids),
            "jointly_remeasured_fraction": fraction,
            "classification": "PREFIX_80" if sparse else "DISTRIBUTED_CANDIDATE_SET",
            "smallest_prefix_reaches_80": sparse,
            "prefix_cap": PREFIX_CAP,
            "threshold": PREFIX_FRACTION,
            "full_register_effect": raw,
            "joint_prefix_D_path": float(selected_prefix["D_path"]),
            "joint_prefix_R_path": float(selected_prefix["R_path"]),
            "joint_prefix_Q_path": float(selected_prefix["Q_path"]),
            "source_prefix_sha256": canonical_json_sha256(selected_prefix),
        },
        "learned_set": {
            "label": "learned",
            "components": learned_components,
            "joint_output_delta_norm": learned_norm,
        },
        "path_target_output_delta_norms": target_norms,
        "null_families": null_families,
        "null_family_seeds": dict(NULL_SEEDS),
        "source_endpoint_sha256": canonical_json_sha256(
            {
                "singletons": screen["singletons"],
                "joint_prefixes": screen["joint_prefixes"],
                "baseline": screen["baseline"],
            }
        ),
        "outcome_firewall": {
            "phase3_reserve_read": False,
            "selection_uses_development_only": True,
        },
    }
    mde = projected_path_mde80(selected_prefix["per_block"])
    path_evidence: dict[str, object] = {
        "schema": PATH_MDE_SCHEMA,
        "status": "COMPLETE",
        "stage": "development",
        "artifact_role": "path_resolution",
        "development_blocks": int(screen["n_blocks"]),
        "reserve_blocks": RESERVE_BLOCKS,
        "mde80": mde,
        "threshold": SESOI,
        "passed": all(value <= SESOI for value in mde.values()),
        "candidate_screen_sha256": screen_record["sha256"],
        "candidate_freeze_sha256": canonical_json_sha256(freeze),
        "selected_prefix_sha256": canonical_json_sha256(selected_prefix),
        "method": "stratum_centered_normal_projection_80pct_power_two_sided_0.05",
    }
    return freeze, path_evidence


__all__ = [
    "CANDIDATE_FREEZE_AUDIT_SCHEMA",
    "CANDIDATE_FREEZE_SCHEMA",
    "CANDIDATE_SCREEN_SCHEMA",
    "DEVELOPMENT_BLOCKS",
    "NULL_SEEDS",
    "NULL_SET_COUNT",
    "PATH_MDE_SCHEMA",
    "PATH_NAMES",
    "PREFIX_CAP",
    "PREFIX_FRACTION",
    "RESERVE_BLOCKS",
    "SESOI",
    "TOKEN_SUPPORT",
    "build_candidate_freeze",
    "canonical_json_sha256",
    "component_universe",
    "projected_path_mde80",
    "validate_candidate_screen",
]
