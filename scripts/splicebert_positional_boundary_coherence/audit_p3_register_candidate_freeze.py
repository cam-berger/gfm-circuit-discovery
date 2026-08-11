#!/usr/bin/env python
"""Independent audit of the executed candidate screen and deterministic freeze.

This file deliberately duplicates the census, ranking, prefix, norm-matching, and
MDE calculations.  It does not import the candidate selection or verdict module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_register_stage import (  # noqa: E402
    manifest_binds_path,
    manifest_precedes,
    validate_sibling_manifest,
)


SCREEN_SCHEMA = "gfmcd.pbc.p3_register_candidate_screen.v1"
FREEZE_SCHEMA = "gfmcd.pbc.p3_register_candidate_freeze.v1"
AUDIT_SCHEMA = "gfmcd.pbc.p3_register_candidate_freeze_audit.v1"
PATH_MDE_SCHEMA = "gfmcd.pbc.p3_register_path_mde.v1"
PATH_NAMES = ("d_path", "r_path", "q_path")
SEEDS = {"d_path": 2026081201, "r_path": 2026081202, "q_path": 2026081203}
TOKEN_SUPPORT = [393, 513]
DEVELOPMENT_BLOCKS = 120
RESERVE_BLOCKS = 384
SESOI = 0.05


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    return output, manifest


def _canonical_hash(value: object) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _universe() -> list[dict[str, object]]:
    return [
        *(
            {
                "component_id": f"L{layer}H{head}",
                "layer": layer,
                "kind": "head",
                "head": head,
                "hook_site": "attention.self.context_pre_output_projection",
                "patch_token_count": 120,
            }
            for layer in range(6)
            for head in range(16)
        ),
        *(
            {
                "component_id": f"MLP{layer}",
                "layer": layer,
                "kind": "mlp",
                "head": None,
                "hook_site": "encoder.layer.output.dense_pre_residual_add",
                "patch_token_count": 120,
            }
            for layer in range(6)
        ),
    ]


def _record_matches(record: object, expected_path: Path) -> bool:
    if not isinstance(record, Mapping):
        return False
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    return bool(
        path == expected_path
        and path.is_file()
        and type(record.get("size_bytes")) is int
        and int(record["size_bytes"]) == path.stat().st_size
        and record.get("sha256") == sha256_file(path)
    )


def _joint_norm(
    ids: Sequence[str],
    singleton: Mapping[str, Mapping[str, Any]],
    path_name: str,
) -> float:
    return float(
        math.sqrt(
            math.fsum(
                float(singleton[value]["path_output_delta_norms"][path_name]) ** 2
                for value in ids
            )
        )
    )


def _mde(per_block: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    matrix = np.asarray(
        [
            [float(row["D_path"]), float(row["R_path"]), float(row["Q_path"])]
            for row in per_block
        ],
        dtype=float,
    )
    labels = np.asarray(
        [f"{row['chrom']}:{row['strand']}" for row in per_block], dtype=object
    )
    centered = matrix.copy()
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if int(mask.sum()) < 2:
            raise ValueError("candidate MDE stratum has fewer than two blocks")
        centered[mask] -= centered[mask].mean(axis=0)
    multiplier = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    values = multiplier * centered.std(axis=0, ddof=1) / math.sqrt(RESERVE_BLOCKS)
    return {
        "D_path": float(values[0]),
        "R_path": float(values[1]),
        "Q_path": float(values[2]),
    }


def audit_payloads(
    screen: Mapping[str, Any],
    freeze: Mapping[str, Any],
    path_mde: Mapping[str, Any],
    *,
    screen_path: Path,
    freeze_path: Path,
    path_mde_path: Path,
    enforce_development_contract: bool = True,
    provenance_checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    expected_universe = _universe()
    checks["screen_schema_and_stage"] = bool(
        screen.get("schema") == SCREEN_SCHEMA
        and screen.get("status") == "COMPLETE"
        and screen.get("stage") == "development"
        and screen.get("token_support") == TOKEN_SUPPORT
    )
    checks["exact_102_component_census"] = screen.get("candidate_universe") == expected_universe
    checks["development_role_exact"] = bool(
        int(screen.get("n_blocks", -1)) == DEVELOPMENT_BLOCKS
        if enforce_development_contract
        else int(screen.get("n_blocks", -1)) >= 6
    )
    tensor = screen.get("tensor_stream")
    checks["executed_same_state_full_span_stream"] = bool(
        isinstance(tensor, Mapping)
        and len(str(tensor.get("sha256", ""))) == 64
        and int(tensor.get("patch_events", 0)) > 0
        and tensor.get("same_sequence_only") is True
        and tensor.get("same_corruption_state_only") is True
        and tensor.get("full_token_support_only") is True
    )
    levels = screen.get("evidence_levels")
    checks["evidence_levels_separated"] = bool(
        isinstance(levels, Mapping)
        and set(levels)
        == {
            "qk_pre_softmax",
            "attention_post_softmax",
            "component_output",
            "behavior_same_state_patch",
        }
    )
    singletons = screen.get("singletons")
    prefixes = screen.get("joint_prefixes")
    expected_ids = [str(row["component_id"]) for row in expected_universe]
    singleton_ok = isinstance(singletons, list) and len(singletons) == 102
    singleton_by_id: dict[str, Mapping[str, Any]] = {}
    if singleton_ok:
        singleton_by_id = {str(row.get("component_id")): row for row in singletons}
        singleton_ok = set(singleton_by_id) == set(expected_ids) and len(singleton_by_id) == 102
    checks["singleton_endpoint_census"] = singleton_ok
    ranking: list[str] = []
    if singleton_ok:
        ranking = [
            str(row["component_id"])
            for row in sorted(
                singletons,
                key=lambda row: (
                    -float(row["same_state_scrub_loss_in_omega3"]),
                    str(row["component_id"]),
                ),
            )
        ]
        endpoint_ok = True
        support = None
        for row in singletons:
            local = row.get("per_block")
            if not isinstance(local, list) or len(local) != int(screen.get("n_blocks", -1)):
                endpoint_ok = False
                break
            identities = {
                (str(value.get("block_id")), str(value.get("chrom")), str(value.get("strand")))
                for value in local
            }
            if len(identities) != len(local) or (support is not None and identities != support):
                endpoint_ok = False
                break
            support = identities
            mean_d = float(np.mean([float(value["D_path"]) for value in local]))
            mean_r = float(np.mean([float(value["R_path"]) for value in local]))
            mean_q = float(np.mean([float(value["Q_path"]) for value in local]))
            if (
                abs(mean_d - float(row["same_state_scrub_loss_in_omega3"])) > 1e-10
                or abs(mean_r - float(row["same_state_restore_gain_in_omega3"])) > 1e-10
                or abs(mean_q - float(row["same_state_sufficiency_gain_in_omega3"])) > 1e-10
                or set(row.get("path_output_delta_norms", {})) != set(PATH_NAMES)
            ):
                endpoint_ok = False
                break
        checks["singleton_endpoints_recomputed"] = endpoint_ok
    else:
        checks["singleton_endpoints_recomputed"] = False
    prefix_ok = isinstance(prefixes, list) and len(prefixes) == 8 and bool(ranking)
    if prefix_ok:
        for size, row in enumerate(prefixes, start=1):
            local = row.get("per_block")
            if (
                int(row.get("prefix_size", -1)) != size
                or list(row.get("component_ids", [])) != ranking[:size]
                or not isinstance(local, list)
                or abs(float(row["D_path"]) - float(np.mean([float(value["D_path"]) for value in local]))) > 1e-10
                or abs(float(row["R_path"]) - float(np.mean([float(value["R_path"]) for value in local]))) > 1e-10
                or abs(float(row["Q_path"]) - float(np.mean([float(value["Q_path"]) for value in local]))) > 1e-10
            ):
                prefix_ok = False
                break
    checks["joint_prefixes_recomputed_1_through_8"] = prefix_ok

    checks["freeze_schema"] = bool(
        freeze.get("schema") == FREEZE_SCHEMA
        and freeze.get("status") == "COMPLETE"
        and freeze.get("stage") == "development"
        and freeze.get("token_support") == TOKEN_SUPPORT
    )
    source = freeze.get("sources")
    source_record = source.get("candidate_screen") if isinstance(source, Mapping) else None
    checks["screen_source_record_reproducible"] = _record_matches(source_record, screen_path)
    checks["screen_source_hash_bound"] = bool(
        isinstance(source_record, Mapping)
        and freeze.get("candidate_screen", {}).get("candidate_screen_sha256")
        == sha256_file(screen_path)
        and freeze.get("patch_tensor_sha256") == tensor.get("sha256")
    ) if isinstance(tensor, Mapping) else False
    selection = freeze.get("selection")
    selected_prefix = None
    raw = float(screen.get("baseline", {}).get("mean_raw_omega3", math.nan))
    if prefix_ok and math.isfinite(raw) and raw > 0:
        selected_prefix = next(
            (row for row in prefixes if float(row["D_path"]) / raw >= 0.80),
            prefixes[-1],
        )
        reaches = any(float(row["D_path"]) / raw >= 0.80 for row in prefixes)
        expected_class = "PREFIX_80" if reaches else "DISTRIBUTED_CANDIDATE_SET"
        checks["smallest_prefix_rule_recomputed"] = bool(
            isinstance(selection, Mapping)
            and int(selection.get("candidate_count", -1))
            == int(selected_prefix["prefix_size"])
            and selection.get("classification") == expected_class
            and selection.get("smallest_prefix_reaches_80") is reaches
            and abs(
                float(selection.get("jointly_remeasured_fraction", math.nan))
                - float(selected_prefix["D_path"]) / raw
            )
            <= 1e-12
            and selection.get("source_prefix_sha256") == _canonical_hash(selected_prefix)
        )
    else:
        checks["smallest_prefix_rule_recomputed"] = False

    learned = freeze.get("learned_set")
    learned_components = learned.get("components") if isinstance(learned, Mapping) else None
    learned_ids = (
        [str(row.get("component_id")) for row in learned_components]
        if isinstance(learned_components, list)
        else []
    )
    checks["learned_set_is_selected_prefix"] = bool(
        selected_prefix is not None
        and learned_ids == list(selected_prefix["component_ids"])
        and 1 <= len(learned_ids) <= 8
    )
    metadata = {str(row["component_id"]): row for row in expected_universe}
    target_norms: dict[str, float] = {}
    norms_ok = checks["learned_set_is_selected_prefix"] and singleton_ok
    if norms_ok:
        target_norms = {
            name: _joint_norm(learned_ids, singleton_by_id, name) for name in PATH_NAMES
        }
        recorded = freeze.get("path_target_output_delta_norms")
        norms_ok = bool(
            isinstance(recorded, Mapping)
            and all(abs(float(recorded[name]) - target_norms[name]) <= 1e-12 for name in PATH_NAMES)
        )
    checks["learned_output_norms_recomputed"] = norms_ok

    families = freeze.get("null_families")
    null_ok = isinstance(families, Mapping) and set(families) == set(PATH_NAMES) and norms_ok
    all_sets: set[tuple[str, ...]] = set()
    if null_ok:
        learned_layers = Counter(int(metadata[value]["layer"]) for value in learned_ids)
        learned_kinds = Counter(str(metadata[value]["kind"]) for value in learned_ids)
        for path_name in PATH_NAMES:
            rows = families[path_name]
            if not isinstance(rows, list) or len(rows) != 31:
                null_ok = False
                break
            for index, row in enumerate(rows):
                ids = tuple(sorted(str(value.get("component_id")) for value in row.get("components", [])))
                recomputed = _joint_norm(ids, singleton_by_id, path_name) if ids else math.nan
                if (
                    row.get("label") != f"{path_name}_null_{index:02d}"
                    or row.get("matching_seed") != SEEDS[path_name]
                    or len(ids) != len(learned_ids)
                    or set(ids) & set(learned_ids)
                    or any(value not in metadata for value in ids)
                    or Counter(int(metadata[value]["layer"]) for value in ids) != learned_layers
                    or Counter(str(metadata[value]["kind"]) for value in ids) != learned_kinds
                    or ids in all_sets
                    or abs(float(row.get("joint_output_delta_norm", math.nan)) - recomputed) > 1e-12
                    or not 0.90 <= recomputed / target_norms[path_name] <= 1.10
                ):
                    null_ok = False
                    break
                all_sets.add(ids)
            if not null_ok:
                break
    checks["three_independent_31_set_matched_null_families"] = null_ok

    expected_source_hash = _canonical_hash(
        {
            "singletons": screen.get("singletons"),
            "joint_prefixes": screen.get("joint_prefixes"),
            "baseline": screen.get("baseline"),
        }
    )
    checks["raw_endpoint_hash_recomputed"] = freeze.get("source_endpoint_sha256") == expected_source_hash
    recomputed_mde = _mde(selected_prefix["per_block"]) if selected_prefix else {}
    checks["path_mde_schema_and_sources"] = bool(
        path_mde.get("schema") == PATH_MDE_SCHEMA
        and path_mde.get("status") == "COMPLETE"
        and path_mde.get("stage") == "development"
        and int(path_mde.get("reserve_blocks", -1)) == RESERVE_BLOCKS
        and path_mde.get("candidate_screen_sha256") == sha256_file(screen_path)
        and path_mde.get("candidate_freeze_sha256") == _canonical_hash(freeze)
        and path_mde.get("selected_prefix_sha256")
        == (_canonical_hash(selected_prefix) if selected_prefix else None)
    )
    recorded_mde = path_mde.get("mde80")
    checks["path_mde_recomputed_and_capable"] = bool(
        isinstance(recorded_mde, Mapping)
        and set(recorded_mde) >= {"D_path", "R_path", "Q_path"}
        and all(
            abs(float(recorded_mde[name]) - recomputed_mde.get(name, math.inf)) <= 1e-12
            and float(recorded_mde[name]) <= SESOI
            for name in ("D_path", "R_path", "Q_path")
        )
        and path_mde.get("passed") is True
    )
    if provenance_checks is not None:
        checks.update(
            {
                f"producer_manifest_{name}": bool(value)
                for name, value in provenance_checks.items()
            }
        )
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema": AUDIT_SCHEMA,
        "status": status,
        "stage": "development",
        "candidate_screen_sha256": sha256_file(screen_path),
        "candidate_freeze_sha256": sha256_file(freeze_path),
        "candidate_freeze_canonical_sha256": _canonical_hash(freeze),
        "path_mde_sha256": sha256_file(path_mde_path),
        "checks": checks,
        "recomputed": {
            "ranking": ranking,
            "selected_ids": learned_ids,
            "path_target_output_delta_norms": target_norms,
            "path_mde80": recomputed_mde,
            "null_sets_total": len(all_sets),
        },
        "independence": {
            "imports_candidate_selection_logic": False,
            "recomputes_universe_ranking_prefix_matching_hashes_and_mde": True,
            "clean_checkout_reexecutable_from_bound_compact_inputs": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", required=True)
    parser.add_argument("--screen-manifest", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--path-mde", required=True)
    parser.add_argument("--path-mde-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output, manifest_path = output_paths(args.out)
    screen_path = Path(args.screen).expanduser().resolve()
    screen_manifest_path = Path(args.screen_manifest).expanduser().resolve()
    freeze_path = Path(args.freeze).expanduser().resolve()
    freeze_manifest_path = Path(args.freeze_manifest).expanduser().resolve()
    path_mde_path = Path(args.path_mde).expanduser().resolve()
    path_mde_manifest_path = Path(args.path_mde_manifest).expanduser().resolve()
    screen = json.loads(screen_path.read_text())
    screen_manifest = json.loads(screen_manifest_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    freeze_manifest = json.loads(freeze_manifest_path.read_text())
    path_mde = json.loads(path_mde_path.read_text())
    path_mde_manifest = json.loads(path_mde_manifest_path.read_text())
    manifests = {
        "screen": (screen_manifest, screen_manifest_path, screen_path),
        "freeze": (freeze_manifest, freeze_manifest_path, freeze_path),
        "path_mde": (path_mde_manifest, path_mde_manifest_path, path_mde_path),
    }
    provenance: dict[str, bool] = {}
    for label, (manifest, manifest_path, artifact_path) in manifests.items():
        validation = validate_sibling_manifest(
            manifest,
            manifest_path=manifest_path,
            artifact_path=artifact_path,
        )
        provenance[f"{label}_valid"] = bool(validation["passed"])
    provenance.update(
        {
            "freeze_consumes_screen": manifest_binds_path(
                freeze_manifest, screen_path, field="inputs"
            ),
            "path_mde_consumes_screen": manifest_binds_path(
                path_mde_manifest, screen_path, field="inputs"
            ),
            "path_mde_consumes_freeze": manifest_binds_path(
                path_mde_manifest, freeze_path, field="inputs"
            ),
            "screen_precedes_freeze": manifest_precedes(
                screen_manifest, freeze_manifest
            ),
            "freeze_precedes_path_mde": manifest_precedes(
                freeze_manifest, path_mde_manifest
            ),
            "single_code_revision": len(
                {
                    str(manifest.get("git_commit"))
                    for manifest, _, _ in manifests.values()
                }
            )
            == 1,
        }
    )
    audit = audit_payloads(
        screen,
        freeze,
        path_mde,
        screen_path=screen_path,
        freeze_path=freeze_path,
        path_mde_path=path_mde_path,
        provenance_checks=provenance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-p3-register-candidate-freeze-audit-v1",
        sys.argv,
        repo=REPO,
        status=audit["status"],
        config={"stage": "development", "checks": len(audit["checks"])},
    )
    for path in (
        screen_path,
        screen_manifest_path,
        freeze_path,
        freeze_manifest_path,
        path_mde_path,
        path_mde_manifest_path,
    ):
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(json.dumps({"status": audit["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
