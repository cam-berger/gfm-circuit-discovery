#!/usr/bin/env python
"""Run frozen whole-component and exact-group validation for the Head--Context Atlas.

This executor is intentionally narrower than the discovery screen.  It opens the
physically separated internal-validation cohort only after a candidate-specific
mechanism package and its independent audit both pass.  It then evaluates:

* each frozen candidate's exact whole-component intervention;
* its frozen importance-matched component control;
* every frozen exact joint group (heads and MLP writes patched simultaneously); and
* the independent L3H1/B donor-partner positive control.

The program has separate masked-health and target phases, resumes from immutable
per-packet shards, and computes identical interventions once even when several
overlapping biological contexts reuse them.  QK/V and downstream-path assays are a
later stage and are not run unless these whole effects validate.

``--smoke`` is a plumbing-only discovery run over the frozen four-packet cover.  It
uses built-in head/MLP/mixed-group operations and cannot open validation artifacts.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gfmcd.circuits.hf_bert import capture_head_outputs, edit_heads, patch_heads  # noqa: E402
from gfmcd.splicing.circuit import capture_writes, clamp_write, load_adapter, pick_device  # noqa: E402
from run_all_head_screen import ScreenModel  # noqa: E402
from run_atlas_discovery_screen import (  # noqa: E402
    file_sha256,
    membership_by_packet,
    packet_meta,
    select_health_panel,
    shard_name,
    write_atomic,
)
from run_atlas_model_preflight import (  # noqa: E402
    HEALTH_ACCURACY_FLOOR,
    HEALTH_LOGPROB_FLOOR,
    encode_masks,
    health_probes,
    health_scores,
    partner_destroyed,
    read_jsonl,
    select_cover_pairs,
    stable_rank,
    target_scores,
)


IDENTITY_ATOL = 1e-4
ALL_COMPONENT_IDENTITY_ATOL = 1e-3
EXPECTED_SPLITS = {False: "atlas_internal_validation", True: "atlas_discovery"}
HISTORICALLY_OR_SEALED = {"chr8", "chr9", "chr11", "chr16", "chr19"}
ALL_MLPS = tuple((layer, "mlp") for layer in range(6))


def stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


def parse_component(component_id: str) -> tuple[str, tuple[int, int] | tuple[int, str]]:
    """Return (type, hook key) for one canonical atlas component label."""
    if not component_id.startswith("L"):
        raise ValueError(f"invalid component ID {component_id!r}")
    if component_id.endswith("MLP"):
        layer_text = component_id[1:-3]
        if not layer_text.isdigit():
            raise ValueError(f"invalid MLP component ID {component_id!r}")
        layer = int(layer_text)
        if not 0 <= layer < 6:
            raise ValueError(f"MLP layer outside SpliceBERT grid: {component_id}")
        return "mlp_write", (layer, "mlp")
    if "H" not in component_id:
        raise ValueError(f"invalid attention component ID {component_id!r}")
    layer_text, head_text = component_id[1:].split("H", 1)
    if not layer_text.isdigit() or not head_text.isdigit():
        raise ValueError(f"invalid attention component ID {component_id!r}")
    layer, head = int(layer_text), int(head_text)
    if not 0 <= layer < 6 or not 0 <= head < 16:
        raise ValueError(f"head outside SpliceBERT grid: {component_id}")
    return "attention_head", (layer, head)


def component_sort_key(component_id: str) -> tuple[int, int, int]:
    ctype, key = parse_component(component_id)
    if ctype == "attention_head":
        layer, head = key
        return int(layer), 0, int(head)
    layer, _ = key
    return int(layer), 1, 0


def canonical_components(components: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    ordered = tuple(sorted(set(components), key=component_sort_key))
    if not ordered:
        raise ValueError("an intervention cannot have an empty component set")
    return ordered


def computation_id(estimand: str, components: tuple[str, ...]) -> str:
    if estimand not in {"A", "B"}:
        raise ValueError(f"unknown estimand {estimand!r}")
    return stable_id("comp", estimand, "+".join(components))


def _endpoint(*, endpoint_id: str, endpoint_role: str, estimand: str,
              components: list[str] | tuple[str, ...], scope_kind: str,
              scope_value: str, candidate_id: str | None = None,
              group_id: str | None = None) -> dict:
    components = canonical_components(components)
    return {
        "endpoint_id": endpoint_id,
        "endpoint_role": endpoint_role,
        "candidate_id": candidate_id,
        "group_id": group_id,
        "estimand": estimand,
        "component_ids": list(components),
        "computation_id": computation_id(estimand, components),
        "scope_kind": scope_kind,
        "scope_value": scope_value,
        "intervention": (
            "joint_zero_component_outputs" if estimand == "A"
            else "joint_partner_destroyed_source_patch"
        ),
    }


def build_endpoints(package: dict, *, smoke: bool) -> list[dict]:
    """Translate the frozen package to explicit, non-additive validation endpoints."""
    if smoke:
        specs = [
            _endpoint(endpoint_id="smoke|candidate|A|L0H0", endpoint_role="candidate",
                      estimand="A", components=["L0H0"], scope_kind="general",
                      scope_value="all", candidate_id="smoke|A|L0H0"),
            _endpoint(endpoint_id="smoke|control|A|L0H1", endpoint_role="importance_control",
                      estimand="A", components=["L0H1"], scope_kind="general",
                      scope_value="all", candidate_id="smoke|A|L0H0"),
            _endpoint(endpoint_id="smoke|candidate|B|L3H1", endpoint_role="candidate",
                      estimand="B", components=["L3H1"], scope_kind="general",
                      scope_value="all", candidate_id="smoke|B|L3H1"),
            _endpoint(endpoint_id="smoke|control|B|L3H2", endpoint_role="importance_control",
                      estimand="B", components=["L3H2"], scope_kind="general",
                      scope_value="all", candidate_id="smoke|B|L3H1"),
            _endpoint(endpoint_id="smoke|candidate|A|L5MLP", endpoint_role="candidate",
                      estimand="A", components=["L5MLP"], scope_kind="general",
                      scope_value="all", candidate_id="smoke|A|L5MLP"),
            _endpoint(endpoint_id="smoke|control|A|L4MLP", endpoint_role="importance_control",
                      estimand="A", components=["L4MLP"], scope_kind="general",
                      scope_value="all", candidate_id="smoke|A|L5MLP"),
            _endpoint(endpoint_id="smoke|candidate|B|L3MLP", endpoint_role="candidate",
                      estimand="B", components=["L3MLP"], scope_kind="general",
                      scope_value="all", candidate_id="smoke|B|L3MLP"),
            _endpoint(endpoint_id="smoke|control|B|L2MLP", endpoint_role="importance_control",
                      estimand="B", components=["L2MLP"], scope_kind="general",
                      scope_value="all", candidate_id="smoke|B|L3MLP"),
            _endpoint(endpoint_id="smoke|group|A|mixed", endpoint_role="exact_group",
                      estimand="A", components=["L0H0", "L5MLP"], scope_kind="general",
                      scope_value="all", group_id="smoke|A|mixed"),
            _endpoint(endpoint_id="smoke|group|B|mixed", endpoint_role="exact_group",
                      estimand="B", components=["L3H1", "L3MLP"], scope_kind="general",
                      scope_value="all", group_id="smoke|B|mixed"),
            _endpoint(endpoint_id="positive_control|B|L3H1", endpoint_role="positive_control",
                      estimand="B", components=["L3H1"], scope_kind="target_role",
                      scope_value="donor"),
        ]
        return specs

    endpoints: list[dict] = []
    for candidate in package["candidates"]:
        if candidate["kind"] == "general":
            scope_kind, scope_value = "general", "all"
        elif candidate["kind"] == "context":
            scope_kind, scope_value = "contrast", candidate["contrast"]
        else:
            raise ValueError(f"unknown candidate kind {candidate['kind']!r}")
        endpoints.append(_endpoint(
            endpoint_id=f"candidate|{candidate['candidate_id']}",
            endpoint_role="candidate", estimand=candidate["estimand"],
            components=[candidate["component_id"]], scope_kind=scope_kind,
            scope_value=scope_value, candidate_id=candidate["candidate_id"],
        ))
        endpoints.append(_endpoint(
            endpoint_id=f"importance_control|{candidate['candidate_id']}",
            endpoint_role="importance_control", estimand=candidate["estimand"],
            components=[candidate["importance_control"]["component_id"]],
            scope_kind=scope_kind, scope_value=scope_value,
            candidate_id=candidate["candidate_id"],
        ))

    for group_id, components in sorted(package["exact_groups"].items()):
        parts = group_id.split("|")
        if parts[0] == "general" and len(parts) == 2:
            scope_kind, scope_value, estimand = "general", "all", parts[1]
        elif parts[0] == "context" and len(parts) == 4:
            scope_kind = "contrast"
            scope_value = "|".join(parts[1:3])
            estimand = parts[3]
        else:
            raise ValueError(f"invalid exact-group ID {group_id!r}")
        endpoints.append(_endpoint(
            endpoint_id=f"exact_group|{group_id}", endpoint_role="exact_group",
            estimand=estimand, components=components, scope_kind=scope_kind,
            scope_value=scope_value, group_id=group_id,
        ))

    positive = package["positive_control"]
    endpoints.append(_endpoint(
        endpoint_id=f"positive_control|{positive['estimand']}|{positive['component_id']}",
        endpoint_role="positive_control", estimand=positive["estimand"],
        components=[positive["component_id"]], scope_kind="target_role",
        scope_value="donor",
    ))
    ids = [endpoint["endpoint_id"] for endpoint in endpoints]
    if len(ids) != len(set(ids)):
        raise RuntimeError("endpoint IDs are not unique")
    return endpoints


def computations_from_endpoints(endpoints: list[dict]) -> dict[str, dict]:
    computations: dict[str, dict] = {}
    for endpoint in endpoints:
        cid = endpoint["computation_id"]
        spec = {
            "computation_id": cid,
            "estimand": endpoint["estimand"],
            "component_ids": endpoint["component_ids"],
        }
        if cid in computations and computations[cid] != spec:
            raise RuntimeError(f"computation collision for {cid}")
        computations[cid] = spec
    return computations


def endpoint_packet_sets(endpoints: list[dict], packets: list[dict],
                         pairs: list[dict]) -> dict[str, set[str]]:
    all_ids = {packet["packet_id"] for packet in packets}
    role_ids: dict[str, set[str]] = defaultdict(set)
    for packet in packets:
        role_ids[packet["target_role"]].add(packet["packet_id"])
    contrast_ids: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        contrast = f"{pair['category']}|{pair['target_role']}"
        contrast_ids[contrast].update((pair["case_packet_id"], pair["control_packet_id"]))
    output = {}
    for endpoint in endpoints:
        if endpoint["scope_kind"] == "general":
            selected = set(all_ids)
        elif endpoint["scope_kind"] == "target_role":
            selected = set(role_ids[endpoint["scope_value"]])
        elif endpoint["scope_kind"] == "contrast":
            selected = set(contrast_ids[endpoint["scope_value"]])
            if not selected:
                raise RuntimeError(f"empty validation contrast {endpoint['scope_value']}")
        else:
            raise ValueError(f"unknown scope kind {endpoint['scope_kind']}")
        if not selected <= all_ids:
            raise RuntimeError(f"endpoint {endpoint['endpoint_id']} references absent packets")
        output[endpoint["endpoint_id"]] = selected
    return output


def split_path_from_validation(path: Path, split: str) -> Path:
    marker = ".validation."
    if marker not in str(path):
        raise ValueError(f"sealed path lacks {marker}: {path}")
    return Path(str(path).replace(marker, f".{split}.", 1))


def resolve_repo_reference(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO / value).resolve()


def load_and_gate_cohort(package: dict, *, smoke: bool, cohort_prefix: Path | None) -> tuple[
        list[dict], dict[str, str], list[dict], dict]:
    """Load one split only after package/audit validation has completed."""
    if smoke:
        if cohort_prefix is None:
            raise ValueError("smoke requires --cohort-prefix")
        paths = {
            kind: Path(f"{cohort_prefix}.discovery.{kind}.jsonl")
            for kind in ("pairs", "packets", "sequences")
        }
    else:
        paths = {
            kind: resolve_repo_reference(path)
            for kind, path in package["sealed_validation_references_not_opened"].items()
        }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    packets = read_jsonl(paths["packets"])
    pairs = read_jsonl(paths["pairs"])
    sequence_rows = read_jsonl(paths["sequences"])
    sequences = {row["sequence_id"]: row["sequence"] for row in sequence_rows}
    expected_split = EXPECTED_SPLITS[smoke]
    if not packets or not pairs or not sequences:
        raise RuntimeError("empty cohort artifact")
    if any(packet["analysis_split"] != expected_split for packet in packets):
        raise RuntimeError(f"cohort contains rows outside {expected_split}")
    packet_ids = {packet["packet_id"] for packet in packets}
    if len(packet_ids) != len(packets):
        raise RuntimeError("duplicate packet IDs")
    if any(pair["analysis_split"] != expected_split for pair in pairs):
        raise RuntimeError("pair split mismatch")
    if any(pair[field] not in packet_ids for pair in pairs
           for field in ("case_packet_id", "control_packet_id")):
        raise RuntimeError("pair cannot join packet table")
    if any(packet["sequence_id"] not in sequences for packet in packets):
        raise RuntimeError("packet cannot join sequence table")
    if any(sequences[packet["sequence_id"]][packet["masked_base_win"]] != "G"
           for packet in packets):
        raise RuntimeError("masked coordinate does not point to native G")
    if any(packet["masked_base_win"] != (
        packet["donor_win"] if packet["target_role"] == "donor"
        else packet["acceptor_win"] - 1
    ) for packet in packets):
        raise RuntimeError("donor/acceptor coordinate contract failed")

    split_audit = {
        "expected_analysis_split": expected_split,
        "n_pairs": len(pairs),
        "n_packets": len(packets),
        "n_sequences": len(sequences),
        "n_genes": len({packet["gene_id"] for packet in packets}),
        "gene_overlap_with_discovery": None,
        "sequence_overlap_with_discovery": None,
        "historically_or_sealed_chromosomes_present": sorted(
            {packet["chrom"] for packet in packets} & HISTORICALLY_OR_SEALED
        ),
    }
    if not smoke:
        discovery_packets_path = split_path_from_validation(paths["packets"], "discovery")
        discovery_sequences_path = split_path_from_validation(paths["sequences"], "discovery")
        discovery_packets = read_jsonl(discovery_packets_path)
        discovery_sequences = read_jsonl(discovery_sequences_path)
        discovery_genes = {row["gene_id"] for row in discovery_packets}
        discovery_sequence_ids = {row["sequence_id"] for row in discovery_sequences}
        split_audit["gene_overlap_with_discovery"] = sorted(
            {packet["gene_id"] for packet in packets} & discovery_genes
        )
        split_audit["sequence_overlap_with_discovery"] = sorted(
            set(sequences) & discovery_sequence_ids
        )
        if split_audit["gene_overlap_with_discovery"]:
            raise RuntimeError("internal-validation genes overlap discovery")
        if split_audit["sequence_overlap_with_discovery"]:
            raise RuntimeError("internal-validation sequences overlap discovery")
        if split_audit["historically_or_sealed_chromosomes_present"]:
            raise RuntimeError("excluded chromosomes entered internal validation")

        cohort_prefix_text = str(paths["packets"]).split(".validation.packets.jsonl", 1)[0]
        audit_path = Path(f"{cohort_prefix_text}.audit.json")
        source_manifest_path = Path(f"{cohort_prefix_text}.manifest.json")
        audit = json.loads(audit_path.read_text())
        source_manifest = json.loads(source_manifest_path.read_text())
        if audit["status"] != "PREFLIGHT_PASS" or audit["n_failures"] != 0:
            raise RuntimeError("matched-cohort audit is not a zero-failure pass")
        output_hashes = {
            str(resolve_repo_reference(record["path"])): record["sha256"]
            for record in source_manifest["outputs"]
        }
        for path in paths.values():
            expected = output_hashes.get(str(path.resolve()))
            if expected is None or expected != file_sha256(path):
                raise RuntimeError(f"cohort provenance hash mismatch: {path}")
        split_audit["matched_cohort_audit"] = {
            "path": str(audit_path), "sha256": file_sha256(audit_path),
            "status": audit["status"], "n_failures": audit["n_failures"],
        }
    return packets, sequences, pairs, {"paths": paths, "audit": split_audit}


def select_smoke(packets: list[dict], sequences: dict[str, str],
                 pairs: list[dict]) -> tuple[list[dict], dict[str, str], list[dict]]:
    packet_lookup = {packet["packet_id"]: packet for packet in packets}
    selected_pairs = select_cover_pairs(pairs, packet_lookup)
    ids = {
        pair[field] for pair in selected_pairs
        for field in ("case_packet_id", "control_packet_id")
    }
    selected_packets = sorted(
        (packet_lookup[packet_id] for packet_id in ids),
        key=lambda packet: (
            packet["target_role"], stable_rank("mechanism-smoke", packet["packet_id"]),
            packet["packet_id"],
        ),
    )
    if len(selected_packets) != 4 or {
        packet["target_role"] for packet in selected_packets
    } != {"donor", "acceptor"}:
        raise RuntimeError("smoke cover is not four packets spanning both roles")
    selected_sequences = {
        packet["sequence_id"]: sequences[packet["sequence_id"]]
        for packet in selected_packets
    }
    return selected_packets, selected_sequences, selected_pairs


def operation_parts(spec: dict) -> tuple[list[tuple[int, int]], list[tuple[int, str]]]:
    heads: list[tuple[int, int]] = []
    mlps: list[tuple[int, str]] = []
    for component in spec["component_ids"]:
        ctype, key = parse_component(component)
        if ctype == "attention_head":
            heads.append(key)  # type: ignore[arg-type]
        else:
            mlps.append(key)  # type: ignore[arg-type]
    return heads, mlps


@contextlib.contextmanager
def apply_operation(adapter, spec: dict, *, source_heads: dict[int, torch.Tensor],
                    source_mlps: dict[tuple[int, str], torch.Tensor], zero: bool):
    heads, mlps = operation_parts(spec)
    with contextlib.ExitStack() as stack:
        if heads:
            if zero:
                stack.enter_context(edit_heads(adapter, heads, mode="zero"))
            else:
                stack.enter_context(patch_heads(adapter, source_heads, heads))
        if mlps:
            if zero:
                zero_source = {key: torch.zeros_like(source_mlps[key]) for key in mlps}
                stack.enter_context(clamp_write(adapter, mlps, zero_source))
            else:
                stack.enter_context(clamp_write(adapter, mlps, source_mlps))
        yield


def capture_components(model: ScreenModel, enc) -> tuple[dict[int, torch.Tensor], dict]:
    mlp_cache: dict = {}
    head_cache = capture_head_outputs(
        model.a,
        lambda: mlp_cache.update(capture_writes(
            model.a, ALL_MLPS, lambda: model.logits(enc)
        )),
    )
    if set(mlp_cache) != set(ALL_MLPS):
        raise RuntimeError("failed to capture all six MLP writes")
    return head_cache, mlp_cache


def score_batch(model: ScreenModel, phase: str, logits: torch.Tensor,
                packets: list[dict], sequences: list[str], probes: list[list[int]] | None) -> list[dict]:
    if phase == "target":
        return target_scores(model, logits, packets)
    if probes is None:
        raise RuntimeError("health phase lacks probe coordinates")
    return health_scores(model, logits, sequences, probes)


def finite_nested(value) -> bool:
    if isinstance(value, dict):
        return all(finite_nested(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_nested(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def endpoint_result(endpoint: dict, phase: str, baseline: dict, intervened: dict,
                    counterfactual: dict) -> dict:
    common = {key: endpoint[key] for key in (
        "endpoint_id", "endpoint_role", "candidate_id", "group_id", "estimand",
        "component_ids", "computation_id", "scope_kind", "scope_value", "intervention",
    )}
    if phase == "target":
        return {
            **common,
            "baseline_margin": baseline["margin"],
            "intervened_margin": intervened["margin"],
            "effect_margin": baseline["margin"] - intervened["margin"],
            "baseline_logprob": baseline["logprob"],
            "intervened_logprob": intervened["logprob"],
            "effect_logprob": baseline["logprob"] - intervened["logprob"],
            "partner_destroyed_margin": counterfactual["margin"],
            "behavioral_partner_effect_margin": baseline["margin"] - counterfactual["margin"],
        }
    return {
        **common,
        "health_probe_count": baseline["n_probes"],
        "baseline_health_margin": baseline["mean_margin"],
        "intervened_health_margin": intervened["mean_margin"],
        "health_margin_damage": baseline["mean_margin"] - intervened["mean_margin"],
        "baseline_health_logprob": baseline["mean_logprob"],
        "intervened_health_logprob": intervened["mean_logprob"],
        "health_logprob_damage": baseline["mean_logprob"] - intervened["mean_logprob"],
        "baseline_health_accuracy": baseline["accuracy"],
        "intervened_health_accuracy": intervened["accuracy"],
        "health_accuracy_damage": baseline["accuracy"] - intervened["accuracy"],
        "partner_destroyed_health_margin": counterfactual["mean_margin"],
    }


def contract_payload(package_path: Path, package_audit_path: Path, phase: str,
                     model_path: Path, endpoints: list[dict], paths: dict[str, Path],
                     smoke: bool, health_n: int) -> dict:
    return {
        "schema": "head_context_atlas.mechanism_validation_contract/1",
        "package_sha256": file_sha256(package_path),
        "package_audit_sha256": file_sha256(package_audit_path),
        "phase": phase,
        "smoke": smoke,
        "health_n": health_n,
        "model_checkpoint_sha256": file_sha256(model_path / "pytorch_model.bin"),
        "cohort_input_sha256": {kind: file_sha256(path) for kind, path in paths.items()},
        "endpoints": endpoints,
        "identity_atol": IDENTITY_ATOL,
        "all_component_identity_atol": ALL_COMPONENT_IDENTITY_ATOL,
        "precision": "float32; TF32/AMP off",
        "seeds": {"PYTHONHASHSEED": 0, "torch": 0, "numpy": 0},
    }


def validate_existing_shard(path: Path, *, contract_sha256: str, phase: str,
                            packet_id: str, expected_endpoint_ids: set[str]) -> None:
    record = json.loads(path.read_text())
    if record.get("schema") != "head_context_atlas.mechanism_validation_shard/1":
        raise RuntimeError(f"unrecognized existing shard {path}")
    if record.get("contract_sha256") != contract_sha256 or record.get("phase") != phase:
        raise RuntimeError(f"existing shard belongs to a different frozen contract: {path}")
    if record.get("packet", {}).get("packet_id") != packet_id:
        raise RuntimeError(f"existing shard packet mismatch: {path}")
    if {row["endpoint_id"] for row in record.get("results", [])} != expected_endpoint_ids:
        raise RuntimeError(f"existing shard endpoint grid mismatch: {path}")


def make_groups(packets: list[dict], endpoint_sets: dict[str, set[str]], phase: str,
                endpoints: list[dict]) -> dict[tuple[str, ...], list[dict]]:
    if phase == "health":
        signature = tuple(sorted(endpoint["endpoint_id"] for endpoint in endpoints))
        return {signature: list(packets)}
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for packet in packets:
        signature = tuple(sorted(
            endpoint["endpoint_id"] for endpoint in endpoints
            if packet["packet_id"] in endpoint_sets[endpoint["endpoint_id"]]
        ))
        if signature:
            groups[signature].append(packet)
    return dict(groups)


def run_phase(*, phase: str, model: ScreenModel, packets: list[dict],
              sequences_by_id: dict[str, str], pairs: list[dict], endpoints: list[dict],
              computations: dict[str, dict], endpoint_sets: dict[str, set[str]],
              output_dir: Path, contract: dict, batch_size: int) -> dict:
    if phase not in {"health", "target"}:
        raise ValueError(phase)
    contract_sha256 = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    shard_dir = output_dir / f"{phase}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    endpoint_by_id = {endpoint["endpoint_id"]: endpoint for endpoint in endpoints}
    membership = membership_by_packet(pairs)
    for packet in packets:
        membership.setdefault(packet["packet_id"], [])
    gates: list[dict] = []

    def gate(name: str, value, threshold, passed: bool) -> None:
        gates.append({"gate": name, "value": value, "threshold": threshold, "passed": bool(passed)})

    signatures = make_groups(packets, endpoint_sets, phase, endpoints)
    expected_by_packet = {
        packet["packet_id"]: set(signature)
        for signature, block in signatures.items()
        for packet in block
    }
    expected_packet_ids = set(expected_by_packet)
    previous_manifest_path = output_dir / f"run_manifest.{phase}.json"
    if previous_manifest_path.exists():
        previous = json.loads(previous_manifest_path.read_text())
        expected_status = "HEALTH_PASS" if phase == "health" else "TARGET_COMPLETE"
        if (previous.get("contract_sha256") == contract_sha256
                and previous.get("status") == expected_status
                and previous.get("n_failures") == 0):
            actual_names = {path.name for path in shard_dir.glob("packet_*.json")}
            expected_names = {shard_name(packet_id) for packet_id in expected_packet_ids}
            if actual_names == expected_names:
                for packet_id, endpoint_ids in expected_by_packet.items():
                    validate_existing_shard(
                        shard_dir / shard_name(packet_id), contract_sha256=contract_sha256,
                        phase=phase, packet_id=packet_id,
                        expected_endpoint_ids=endpoint_ids,
                    )
                resumed = dict(previous)
                resumed["n_new_shards"] = 0
                resumed["n_resumed_shards"] = len(expected_packet_ids)
                resumed["resume_noop_verified"] = True
                return resumed
    first_batch = True
    n_new = 0
    start_time = time.time()
    for signature in sorted(signatures):
        group_packets = sorted(signatures[signature], key=lambda packet: (
            stable_rank("mechanism-validation-order", phase, packet["packet_id"]),
            packet["packet_id"],
        ))
        expected_ids = set(signature)
        pending = []
        for packet in group_packets:
            shard = shard_dir / shard_name(packet["packet_id"])
            if shard.exists():
                validate_existing_shard(
                    shard, contract_sha256=contract_sha256, phase=phase,
                    packet_id=packet["packet_id"], expected_endpoint_ids=expected_ids,
                )
            else:
                pending.append(packet)

        active_computation_ids = sorted({
            endpoint_by_id[endpoint_id]["computation_id"] for endpoint_id in signature
        })
        for start in range(0, len(pending), batch_size):
            batch_packets = pending[start:start + batch_size]
            batch_sequences = [sequences_by_id[packet["sequence_id"]] for packet in batch_packets]
            if phase == "target":
                masks = [[packet["masked_base_win"]] for packet in batch_packets]
                probes = None
            else:
                probes = [health_probes(packet, sequence) for packet, sequence in zip(
                    batch_packets, batch_sequences
                )]
                masks = probes
            native_enc = encode_masks(model, batch_sequences, masks)
            corrupt_sequences = [
                partner_destroyed(packet, sequence)
                for packet, sequence in zip(batch_packets, batch_sequences)
            ]
            corrupt_enc = encode_masks(model, corrupt_sequences, masks)
            native_logits = model.logits(native_enc)
            corrupt_logits = model.logits(corrupt_enc)
            baseline = score_batch(model, phase, native_logits, batch_packets, batch_sequences, probes)
            counterfactual = score_batch(
                model, phase, corrupt_logits, batch_packets, batch_sequences, probes
            )
            native_heads, native_mlps = capture_components(model, native_enc)
            corrupt_heads, corrupt_mlps = capture_components(model, corrupt_enc)

            if first_batch:
                repeat = model.logits(native_enc)
                repeat_error = float((repeat - native_logits).abs().max())
                gate(f"{phase}_deterministic_repeat", repeat_error, 1e-6, repeat_error <= 1e-6)
                if phase == "health":
                    accuracy = float(np.mean([row["accuracy"] for row in baseline]))
                    logprob = float(np.mean([row["mean_logprob"] for row in baseline]))
                    gate("masked_health_accuracy", accuracy, HEALTH_ACCURACY_FLOOR,
                         accuracy >= HEALTH_ACCURACY_FLOOR)
                    gate("masked_health_logprob", logprob, HEALTH_LOGPROB_FLOOR,
                         logprob >= HEALTH_LOGPROB_FLOOR)

                # Exercise every frozen computation on one common batch.  Context
                # scopes can otherwise make a later signature introduce an untested
                # head/group after the one-time plumbing gate.
                identity_computation_ids = sorted(computations)
                union = canonical_components(tuple(
                    component
                    for computation_id_ in identity_computation_ids
                    for component in computations[computation_id_]["component_ids"]
                ))
                union_spec = {"component_ids": list(union)}
                with apply_operation(
                    model.a, union_spec, source_heads=native_heads,
                    source_mlps=native_mlps, zero=False,
                ):
                    identity = model.logits(native_enc)
                all_error = float((identity - native_logits).abs().max())
                gate(f"{phase}_all_component_native_source_identity", all_error,
                     ALL_COMPONENT_IDENTITY_ATOL, all_error <= ALL_COMPONENT_IDENTITY_ATOL)
                for computation_id_ in identity_computation_ids:
                    spec = computations[computation_id_]
                    with apply_operation(
                        model.a, spec, source_heads=native_heads,
                        source_mlps=native_mlps, zero=False,
                    ):
                        identity = model.logits(native_enc)
                    error = float((identity - native_logits).abs().max())
                    gate(f"{phase}_native_source_identity:{computation_id_}", error,
                         IDENTITY_ATOL, error <= IDENTITY_ATOL)

            computed_scores: dict[str, list[dict]] = {}
            for computation_id_ in active_computation_ids:
                spec = computations[computation_id_]
                estimand = spec["estimand"]
                with apply_operation(
                    model.a, spec,
                    source_heads=native_heads if estimand == "A" else corrupt_heads,
                    source_mlps=native_mlps if estimand == "A" else corrupt_mlps,
                    zero=estimand == "A",
                ):
                    intervened_logits = model.logits(native_enc)
                computed_scores[computation_id_] = score_batch(
                    model, phase, intervened_logits, batch_packets, batch_sequences, probes
                )

            for index, packet in enumerate(batch_packets):
                results = []
                for endpoint_id in signature:
                    endpoint = endpoint_by_id[endpoint_id]
                    result = endpoint_result(
                        endpoint, phase, baseline[index],
                        computed_scores[endpoint["computation_id"]][index],
                        counterfactual[index],
                    )
                    if not finite_nested(result):
                        raise RuntimeError(f"non-finite endpoint result {endpoint_id}")
                    results.append(result)
                record = {
                    "schema": "head_context_atlas.mechanism_validation_shard/1",
                    "claim_boundary": (
                        "whole-component/group causal validation; no QK/V/path or "
                        "biological-function label"
                    ),
                    "phase": phase,
                    "contract_sha256": contract_sha256,
                    "packet": packet_meta(packet, membership),
                    "probe_positions": probes[index] if probes is not None else None,
                    "results": results,
                }
                write_atomic(shard_dir / shard_name(packet["packet_id"]), record)
                n_new += 1
            first_batch = False
            print(
                f"[atlas-mechanism:{phase}] {n_new} new shards; "
                f"signature={len(signature)} endpoints",
                file=sys.stderr,
            )

    shard_paths = sorted(shard_dir.glob("packet_*.json"))
    seen_packet_ids = set()
    n_rows = 0
    for path in shard_paths:
        row = json.loads(path.read_text())
        if row.get("contract_sha256") != contract_sha256 or row.get("phase") != phase:
            continue
        seen_packet_ids.add(row["packet"]["packet_id"])
        n_rows += len(row["results"])
    gate(f"{phase}_complete_packet_grid", len(seen_packet_ids), len(expected_packet_ids),
         seen_packet_ids == expected_packet_ids)
    finite_shards = all(
        finite_nested(json.loads(path.read_text())) for path in shard_paths
        if json.loads(path.read_text()).get("contract_sha256") == contract_sha256
    )
    gate(f"{phase}_finite_shards", int(finite_shards), 1, finite_shards)
    failures = [row for row in gates if not row["passed"]]
    elapsed = time.time() - start_time
    return {
        "schema": "head_context_atlas.mechanism_validation_phase/1",
        "status": (
            "HEALTH_PASS" if phase == "health" and not failures else
            "TARGET_COMPLETE" if phase == "target" and not failures else
            f"{phase.upper()}_FAIL"
        ),
        "phase": phase,
        "contract_sha256": contract_sha256,
        "n_packets": len(expected_packet_ids),
        "n_endpoint_rows": n_rows,
        "n_new_shards": n_new,
        "n_resumed_shards": len(expected_packet_ids) - n_new,
        "elapsed_seconds": elapsed,
        "seconds_per_new_packet": elapsed / n_new if n_new else None,
        "n_gates": len(gates),
        "n_failures": len(failures),
        "gates": gates,
        "failures": failures,
        "shards": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in shard_paths
            if json.loads(path.read_text()).get("contract_sha256") == contract_sha256
        },
    }


def package_preflight(package_path: Path, audit_path: Path, *, smoke: bool) -> tuple[dict, dict]:
    package = json.loads(package_path.read_text())
    audit = json.loads(audit_path.read_text())
    expected_status = "SMOKE_NO_CANDIDATES" if smoke else "FROZEN_BEFORE_VALIDATION_ACCESS"
    if package["status"] != expected_status:
        raise RuntimeError(f"package status must be {expected_status}")
    if package["validation_loaded"] is not False:
        raise RuntimeError("package was not sealed outcome-blind")
    if audit["status"] != "PASS" or audit["n_failures"] != 0:
        raise RuntimeError("mechanism package audit is not a zero-failure pass")
    if audit["package_sha256"] != file_sha256(package_path):
        raise RuntimeError("package changed after independent audit")
    if smoke and any(".validation." in value for value in
                     package["sealed_validation_references_not_opened"].values()):
        # Smoke packages may name the sealed files, but no caller may substitute them
        # as input; the actual load path is forced from --cohort-prefix below.
        pass
    return package, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-audit", default=None)
    parser.add_argument("--cohort-prefix", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_matched_v7"
    ), help="used only by --smoke; full runs use package-sealed validation paths")
    parser.add_argument("--model", default=str(
        REPO / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
    ))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("health", "target", "both"), default="health")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--health-n", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    if not args.smoke and (args.health_n < 40 or args.health_n % 2):
        raise ValueError("full-run --health-n must be an even integer >=40")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    package_path = Path(args.package)
    package_audit_path = (
        Path(args.package_audit) if args.package_audit
        else package_path.with_suffix(".audit.json")
    )
    # Critical ordering: validate the frozen package and audit before resolving,
    # opening, hashing, counting, or sampling any internal-validation artifact.
    package, package_audit = package_preflight(
        package_path, package_audit_path, smoke=args.smoke
    )
    endpoints = build_endpoints(package, smoke=args.smoke)
    computations = computations_from_endpoints(endpoints)
    packets, sequences, pairs, cohort = load_and_gate_cohort(
        package, smoke=args.smoke,
        cohort_prefix=Path(args.cohort_prefix) if args.smoke else None,
    )
    if args.smoke:
        packets, sequences, pairs = select_smoke(packets, sequences, pairs)

    endpoint_sets = endpoint_packet_sets(endpoints, packets, pairs)
    target_packet_ids = set().union(*endpoint_sets.values())
    full_packets = [
        packet for packet in packets if packet["packet_id"] in target_packet_ids
    ]
    if not full_packets:
        raise RuntimeError("frozen endpoint family addresses no target packets")
    if args.smoke:
        health_packets = list(packets)
    else:
        health_ids = set(select_health_panel(packets, pairs, args.health_n))
        health_packets = [packet for packet in packets if packet["packet_id"] in health_ids]
        if len(health_packets) != args.health_n:
            raise RuntimeError("health panel selection count mismatch")

    model_path = Path(args.model)
    if not (model_path / "pytorch_model.bin").exists():
        raise FileNotFoundError(model_path / "pytorch_model.bin")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    adapter = load_adapter(str(model_path), device, eager=False)
    if adapter.num_layers != 6 or adapter.num_heads != 16:
        raise RuntimeError(
            f"architecture mismatch: {adapter.num_layers} layers x {adapter.num_heads} heads"
        )
    model = ScreenModel(adapter, batch_size=args.batch)

    phases = ("health", "target") if args.phase == "both" else (args.phase,)
    reports = {}
    for phase in phases:
        if phase == "target" and "health" not in reports:
            health_manifest = output_dir / "run_manifest.health.json"
            if not health_manifest.exists():
                raise RuntimeError("target phase requires a completed health phase")
            health_report = json.loads(health_manifest.read_text())
            if health_report["status"] != "HEALTH_PASS" or health_report["n_failures"]:
                raise RuntimeError("target phase refused because health phase did not pass")
        phase_packets = health_packets if phase == "health" else full_packets
        phase_endpoint_sets = (
            {endpoint["endpoint_id"]: {packet["packet_id"] for packet in phase_packets}
             for endpoint in endpoints}
            if phase == "health" else endpoint_sets
        )
        contract = contract_payload(
            package_path, package_audit_path, phase, model_path, endpoints,
            cohort["paths"], args.smoke, len(health_packets),
        )
        report = run_phase(
            phase=phase, model=model, packets=phase_packets,
            sequences_by_id=sequences, pairs=pairs, endpoints=endpoints,
            computations=computations, endpoint_sets=phase_endpoint_sets,
            output_dir=output_dir, contract=contract, batch_size=args.batch,
        )
        report.update({
            "claim_boundary": (
                "whole-component/group causal validation; no QK/V/path or "
                "biological-function claim"
            ),
            "smoke": args.smoke,
            "validation_loaded": not args.smoke,
            "package": {
                "path": str(package_path), "sha256": file_sha256(package_path),
                "audit_path": str(package_audit_path),
                "audit_sha256": file_sha256(package_audit_path),
                "audit_status": package_audit["status"],
            },
            "cohort": cohort["audit"],
            "inputs": {
                kind: {"path": str(path), "size_bytes": path.stat().st_size,
                       "sha256": file_sha256(path)}
                for kind, path in cohort["paths"].items()
            },
            "model": {
                "path": str(model_path), "device": str(device),
                "checkpoint_sha256": file_sha256(model_path / "pytorch_model.bin"),
                "layers": adapter.num_layers, "heads_per_layer": adapter.num_heads,
                "precision": "float32; TF32/AMP off",
            },
            "endpoints": endpoints,
            "n_unique_computations": len(computations),
        })
        write_atomic(output_dir / f"run_manifest.{phase}.json", report)
        reports[phase] = report
        if report["n_failures"]:
            raise SystemExit(1)

    write_atomic(output_dir / "run_manifest.json", {
        "schema": "head_context_atlas.mechanism_validation_run/1",
        "status": "PASS" if all(report["n_failures"] == 0 for report in reports.values()) else "FAIL",
        "phase": args.phase,
        "smoke": args.smoke,
        "validation_loaded": not args.smoke,
        "reports": {
            phase: {
                "status": report["status"], "n_packets": report["n_packets"],
                "n_endpoint_rows": report["n_endpoint_rows"],
                "n_failures": report["n_failures"],
            }
            for phase, report in reports.items()
        },
    })
    print(json.dumps({
        "status": "PASS",
        "phase": args.phase,
        "smoke": args.smoke,
        "validation_loaded": not args.smoke,
        "reports": {phase: report["status"] for phase, report in reports.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
