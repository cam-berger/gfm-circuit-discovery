"""Pure analysis helpers for the GTEx native-unit biological fallback.

The failed sparse instrument is intentionally absent from this module.  The
objects analysed here are architectural coordinates: exact post-GELU units in
the released model's final (L5) BERT MLP.  A selected population may be causal
without any member being monosemantic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gfmcd.rigor import benjamini_hochberg
from gfmcd.splicing.grouped_stats import grouped_bootstrap, grouped_signflip_pvalue
from gfmcd.splicing.mlp_units import (
    composed_classifier_margin,
    exact_single_unit_effects,
    rank_units_bidirectional,
    unit_group_write_delta,
)


PACKAGE_SCHEMA = "gfmcd.gtex_native_biological_construction_package.v1"
CONFIRMATION_PACKAGE_SCHEMA = (
    "gfmcd.gtex_native_biological_confirmation_construction.v1"
)
LOCK_SCHEMA = "gfmcd.gtex_native_biological_unit_fallback_lock.v1"
DISCOVERY_SCHEMA = "gfmcd.gtex_native_biological_unit_fallback_discovery.v1"
CONFIRMATION_SCHEMA = "gfmcd.gtex_native_biological_unit_fallback_confirmation.v1"
EXPECTED_SEEDS = tuple(range(2020, 2025))
ENDPOINT_LABELS = {"donor": 1, "acceptor": 2}
GROUP_SIZES = (8, 16, 32, 64, 128, 256)
INTERMEDIATE_SIZE = 2048
CONTROL_DRAWS = 3
N_BOOT = 5000
N_PERM = 9999
ACTIVATION_EPS = 1e-12
DNA = frozenset("ACGTN")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sequence_from_pair(pair: Mapping[str, Any], state: str) -> str:
    direct = pair.get(f"{state}_sequence")
    if isinstance(direct, str):
        return direct.upper()
    nested = pair.get(state)
    if isinstance(nested, Mapping) and isinstance(nested.get("sequence"), str):
        return str(nested["sequence"]).upper()
    raise ValueError(f"pair is missing {state}_sequence")


def _families(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = candidate.get("families")
    if isinstance(raw, Mapping):
        values = []
        for family_id, value in raw.items():
            if not isinstance(value, Mapping):
                raise ValueError("family definitions must be objects")
            values.append({"family_id": str(family_id), **dict(value)})
        return values
    if isinstance(raw, list):
        return [dict(value) for value in raw if isinstance(value, Mapping)]
    raise ValueError("candidate families must be a list or object")


def normalize_construction_package(
    package: Mapping[str, Any],
    *,
    expected_partition: str | None = None,
    expected_genes: int | None = None,
) -> dict[str, Any]:
    """Validate and normalize an outcome-blind low/high construction package."""

    if package.get("schema") != PACKAGE_SCHEMA:
        raise ValueError(f"construction package must use schema {PACKAGE_SCHEMA}")
    if package.get("status") != "FROZEN_OUTCOME_BLIND":
        raise ValueError("construction package must be FROZEN_OUTCOME_BLIND")
    if package.get("outcome_blind") is not True:
        raise RuntimeError("construction package must state outcome_blind=true")
    if package.get("classifier_outcomes_read") is not False:
        raise RuntimeError("construction package must state classifier_outcomes_read=false")
    partition = str(package.get("partition", ""))
    if partition not in {"discovery", "confirmation"}:
        raise ValueError("construction partition must be discovery or confirmation")
    if expected_partition is not None and partition != expected_partition:
        raise ValueError(f"expected {expected_partition} construction package")
    candidates = package.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("construction package needs candidates")

    normalized_candidates: list[dict[str, Any]] = []
    candidate_keys: set[tuple[str, str]] = set()
    all_genes: set[str] = set()
    all_examples: dict[str, str] = {}
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("candidate definitions must be objects")
        candidate_id = str(raw_candidate.get("candidate_id", "")).strip()
        endpoint = str(raw_candidate.get("endpoint", "")).strip()
        sign = raw_candidate.get("expected_high_minus_low_margin_sign")
        controls = raw_candidate.get("matched_controls")
        if not candidate_id or endpoint not in ENDPOINT_LABELS or sign not in (-1, 1):
            raise ValueError("candidate id, endpoint, or frozen sign is invalid")
        if not isinstance(controls, list) or not controls:
            raise ValueError("candidate must freeze non-empty matched_controls")
        controls = [str(value) for value in controls]
        if len(controls) != len(set(controls)) or candidate_id in controls:
            raise ValueError("matched controls must be unique and exclude the candidate")
        key = (endpoint, candidate_id)
        if key in candidate_keys:
            raise ValueError(f"duplicate candidate {endpoint}/{candidate_id}")
        candidate_keys.add(key)

        family_rows: dict[str, dict[str, Any]] = {}
        for family in _families(raw_candidate):
            family_id = str(family.get("family_id", "")).strip()
            role = str(family.get("role", "")).strip()
            pairs = family.get("pairs", family.get("rows"))
            if (
                not family_id
                or role not in {"candidate", "matched_control"}
                or not isinstance(pairs, list)
                or not pairs
            ):
                raise ValueError(f"invalid family for {endpoint}/{candidate_id}")
            if family_id in family_rows:
                raise ValueError(f"duplicate family {family_id}")
            normalized_pairs: list[dict[str, Any]] = []
            seen_examples: set[str] = set()
            for raw_pair in pairs:
                if not isinstance(raw_pair, Mapping):
                    raise ValueError("construction pairs must be objects")
                example_id = str(raw_pair.get("example_id", "")).strip()
                gene_id = str(raw_pair.get("gene_id", "")).strip()
                target_position = raw_pair.get("target_position")
                if (
                    not example_id
                    or not gene_id
                    or not isinstance(target_position, int)
                    or target_position < 0
                ):
                    raise ValueError("pair identity or target_position is invalid")
                if example_id in seen_examples:
                    raise ValueError(f"duplicate pair example {example_id}")
                seen_examples.add(example_id)
                if example_id in all_examples and all_examples[example_id] != gene_id:
                    raise ValueError("example_id maps to inconsistent genes")
                all_examples[example_id] = gene_id
                low = _sequence_from_pair(raw_pair, "low")
                high = _sequence_from_pair(raw_pair, "high")
                if (
                    len(low) != len(high)
                    or not low
                    or not set(low) <= DNA
                    or not set(high) <= DNA
                    or target_position >= len(low)
                    or low == high
                ):
                    raise ValueError(f"invalid low/high sequences for {example_id}")
                low_hash = sha256_text(low)
                high_hash = sha256_text(high)
                for state, observed, expected in (
                    ("low", low_hash, raw_pair.get("low_sha256")),
                    ("high", high_hash, raw_pair.get("high_sha256")),
                ):
                    if expected is not None and str(expected) != observed:
                        raise ValueError(f"{state} sequence hash mismatch for {example_id}")
                all_genes.add(gene_id)
                normalized_pairs.append(
                    {
                        "example_id": example_id,
                        "gene_id": gene_id,
                        "target_position": target_position,
                        "low_sequence": low,
                        "high_sequence": high,
                        "low_sha256": low_hash,
                        "high_sha256": high_hash,
                    }
                )
            family_rows[family_id] = {
                "family_id": family_id,
                "role": role,
                "pairs": normalized_pairs,
            }
        expected_family_ids = {candidate_id, *controls}
        if set(family_rows) != expected_family_ids:
            raise ValueError(
                f"{endpoint}/{candidate_id} family set differs from candidate+controls"
            )
        if family_rows[candidate_id]["role"] != "candidate" or any(
            family_rows[name]["role"] != "matched_control" for name in controls
        ):
            raise ValueError("candidate/control family roles are inconsistent")
        normalized_candidates.append(
            {
                "candidate_id": candidate_id,
                "endpoint": endpoint,
                "expected_high_minus_low_margin_sign": int(sign),
                "matched_controls": controls,
                "families": family_rows,
            }
        )
    declared_genes = package.get("n_genes")
    if declared_genes is not None and int(declared_genes) != len(all_genes):
        raise ValueError("declared n_genes does not match construction records")
    if expected_genes is not None and len(all_genes) != expected_genes:
        raise ValueError(f"construction package must contain {expected_genes} frozen genes")
    return {
        "schema": PACKAGE_SCHEMA,
        "status": "FROZEN_OUTCOME_BLIND",
        "outcome_blind": True,
        "classifier_outcomes_read": False,
        "partition": partition,
        "n_genes": len(all_genes),
        "gene_ids": sorted(all_genes),
        "candidates": normalized_candidates,
    }


def _apply_codon_changes(sequence: str, arm: Mapping[str, Any]) -> str:
    changed = list(sequence)
    edits = arm.get("codon_changes")
    if not isinstance(edits, list) or not edits:
        raise ValueError("successful discovery arm has no codon changes")
    for edit in edits:
        if not isinstance(edit, Mapping):
            raise ValueError("codon change must be an object")
        start = int(edit["codon_start"])
        original = str(edit["original"]).upper()
        alternate = str(edit["alternate"]).upper()
        if (
            start < 0
            or start + 3 > len(changed)
            or len(original) != 3
            or len(alternate) != 3
            or "".join(changed[start : start + 3]) != original
        ):
            raise ValueError("discovery codon edit does not replay against clean sequence")
        changed[start : start + 3] = alternate
    return "".join(changed)


def normalize_discovery_audit(
    audit: Mapping[str, Any],
    sequence_package: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay corrected discovery codon edits against frozen clean sequences."""

    if audit.get("experiment") != "gtex_native_biological_construction_audit_v2":
        raise ValueError("discovery input must be corrected construction audit v2")
    if audit.get("status") != "COMPLETE" or audit.get("model_loaded") is not False:
        raise RuntimeError(
            "corrected discovery construction audit is not outcome-blind COMPLETE"
        )
    if sequence_package.get("label_free") is not True:
        raise RuntimeError("discovery sequence source must be label-free")
    states = sequence_package.get("states")
    if not isinstance(states, list):
        raise ValueError("discovery sequence package has no states")
    clean: dict[tuple[str, str], dict[str, Any]] = {}
    for state in states:
        if not isinstance(state, Mapping) or state.get("variant_kind") != "clean":
            continue
        key = (str(state["endpoint"]), str(state["example_id"]))
        if key in clean:
            raise ValueError(f"duplicate clean discovery state {key}")
        sequence = str(state["sequence"]).upper()
        if sha256_text(sequence) != state["sequence_sha256"]:
            raise ValueError(f"clean discovery sequence hash mismatch {key}")
        clean[key] = {
            "sequence": sequence,
            "gene_id": str(state["gene_id"]),
            "target_position": int(state["target_position"]),
        }
    raw_audit = audit.get("audit")
    if not isinstance(raw_audit, Mapping):
        raise ValueError("corrected construction audit has no audit table")
    candidates = []
    for endpoint in ("donor", "acceptor"):
        endpoint_audit = raw_audit.get(endpoint)
        if not isinstance(endpoint_audit, Mapping):
            raise ValueError(f"corrected audit missing {endpoint}")
        for candidate_name, candidate in endpoint_audit.items():
            if not isinstance(candidate, Mapping):
                raise ValueError("corrected candidate audit must be an object")
            if candidate.get("candidate_construction_pass") is not True:
                raise RuntimeError(
                    f"corrected discovery candidate failed: {endpoint}/{candidate_name}"
                )
            controls = [str(value) for value in candidate["matched_controls"]]
            family_values = []
            for family_name in (str(candidate_name), *controls):
                family = candidate["families"][family_name]
                if (
                    int(family.get("success_count", -1))
                    != sum(bool(row.get("success")) for row in family["rows"])
                    or int(family.get("success_gene_count", -1))
                    != len(
                        {
                            str(row["gene_id"])
                            for row in family["rows"]
                            if row.get("success")
                        }
                    )
                ):
                    raise ValueError(
                        f"corrected discovery family integrity failed: "
                        f"{endpoint}/{candidate_name}/{family_name}"
                    )
                pairs = []
                for row in family["rows"]:
                    if not row.get("success"):
                        continue
                    example_id = str(row["example_id"])
                    source = clean.get((endpoint, example_id))
                    if source is None:
                        raise ValueError(
                            f"missing frozen clean state for {endpoint}/{example_id}"
                        )
                    low = _apply_codon_changes(source["sequence"], row["low"])
                    high = _apply_codon_changes(source["sequence"], row["high"])
                    if (
                        sha256_text(low) != row["low_sha256"]
                        or sha256_text(high) != row["high_sha256"]
                    ):
                        raise ValueError(
                            f"corrected audit replay hash failed for {family_name}/{example_id}"
                        )
                    pairs.append(
                        {
                            "example_id": example_id,
                            "gene_id": source["gene_id"],
                            "target_position": source["target_position"],
                            "low_sequence": low,
                            "high_sequence": high,
                            "low_sha256": row["low_sha256"],
                            "high_sha256": row["high_sha256"],
                        }
                    )
                if not pairs:
                    raise ValueError(f"no successful discovery pairs for {family_name}")
                family_values.append(
                    {
                        "family_id": family_name,
                        "role": (
                            "candidate"
                            if family_name == candidate_name
                            else "matched_control"
                        ),
                        "pairs": pairs,
                    }
                )
            candidates.append(
                {
                    "candidate_id": str(candidate_name),
                    "endpoint": endpoint,
                    "expected_high_minus_low_margin_sign": int(
                        candidate["expected_high_minus_low_margin_sign"]
                    ),
                    "matched_controls": controls,
                    "families": family_values,
                }
            )
    package = {
        "schema": PACKAGE_SCHEMA,
        "status": "FROZEN_OUTCOME_BLIND",
        "outcome_blind": True,
        "classifier_outcomes_read": False,
        "partition": "discovery",
        "candidates": candidates,
    }
    normalized = normalize_construction_package(
        package, expected_partition="discovery"
    )
    normalized["source_schema"] = "corrected_construction_audit_v2_replay"
    return normalized


def normalize_confirmation_states(
    package: Mapping[str, Any],
    *,
    expected_genes: int = 200,
) -> dict[str, Any]:
    """Normalize the actual flat confirmation producer artifact.

    Only the producer's predeclared complete dose/nuisance-matched example IDs
    enter analysis.  This filtering is sequence-only and therefore happens
    before any checkpoint is loaded.
    """

    if package.get("schema") != CONFIRMATION_PACKAGE_SCHEMA:
        raise ValueError(
            "confirmation package must use the biological producer schema"
        )
    if (
        package.get("status") != "COMPLETE"
        or package.get("confirmation_authorized") is not True
        or package.get("model_loaded") is not False
        or package.get("confirmation_model_outputs_read") is not False
        or package.get("outcome_blind") is not True
    ):
        raise RuntimeError("confirmation construction is stopped or not outcome-blind")
    cohort = package.get("cohort")
    summaries = package.get("candidate_summaries")
    states = package.get("states")
    if (
        not isinstance(cohort, Mapping)
        or not isinstance(summaries, list)
        or not isinstance(states, list)
    ):
        raise ValueError("confirmation construction is missing cohort/summaries/states")
    frozen_examples = cohort.get("examples")
    if not isinstance(frozen_examples, list):
        raise ValueError("confirmation cohort examples are missing")
    frozen_genes = {str(row["gene_id"]) for row in frozen_examples}
    if len(frozen_genes) != expected_genes or int(cohort.get("n_genes", -1)) != expected_genes:
        raise ValueError(
            f"confirmation package must freeze exactly {expected_genes} genes"
        )
    by_key: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("confirmation state must be an object")
        sequence = str(state.get("sequence", "")).upper()
        if (
            state.get("direction") not in {"low", "high"}
            or sha256_text(sequence) != state.get("sequence_sha256")
        ):
            raise ValueError("confirmation state sequence/hash is invalid")
        key = (
            str(state["candidate_id"]),
            str(state["family_name"]),
            str(state["example_id"]),
            str(state["direction"]),
        )
        if key in by_key:
            raise ValueError(f"duplicate confirmation state {key}")
        by_key[key] = state
    candidates = []
    for summary in summaries:
        candidate_key = str(summary["candidate_id"])
        candidate_name = str(summary["candidate_name"])
        endpoint = str(summary["endpoint"])
        controls = [str(value) for value in summary["matched_controls"]]
        eligible = [str(value) for value in summary["eligible_example_ids"]]
        if not summary.get("confirmation_construction_gate") or not eligible:
            raise RuntimeError(f"confirmation candidate {candidate_key} is not authorized")
        family_values = []
        for family_name in (candidate_name, *controls):
            pairs = []
            for example_id in eligible:
                low = by_key.get((candidate_key, family_name, example_id, "low"))
                high = by_key.get((candidate_key, family_name, example_id, "high"))
                if (
                    low is None
                    or high is None
                    or low.get("pair_success") is not True
                    or high.get("pair_success") is not True
                ):
                    raise RuntimeError(
                        f"eligible confirmation pair incomplete: "
                        f"{candidate_key}/{family_name}/{example_id}"
                    )
                if (
                    low["gene_id"] != high["gene_id"]
                    or low["target_position"] != high["target_position"]
                ):
                    raise ValueError("confirmation low/high identity mismatch")
                pairs.append(
                    {
                        "example_id": example_id,
                        "gene_id": str(low["gene_id"]),
                        "target_position": int(low["target_position"]),
                        "low_sequence": str(low["sequence"]).upper(),
                        "high_sequence": str(high["sequence"]).upper(),
                        "low_sha256": str(low["sequence_sha256"]),
                        "high_sha256": str(high["sequence_sha256"]),
                    }
                )
            family_values.append(
                {
                    "family_id": family_name,
                    "role": (
                        "candidate"
                        if family_name == candidate_name
                        else "matched_control"
                    ),
                    "pairs": pairs,
                }
            )
        candidates.append(
            {
                "candidate_id": candidate_name,
                "endpoint": endpoint,
                "expected_high_minus_low_margin_sign": int(
                    summary["expected_high_minus_low_margin_sign"]
                ),
                "matched_controls": controls,
                "families": family_values,
            }
        )
    normalized = normalize_construction_package(
        {
            "schema": PACKAGE_SCHEMA,
            "status": "FROZEN_OUTCOME_BLIND",
            "outcome_blind": True,
            "classifier_outcomes_read": False,
            "partition": "confirmation",
            "candidates": candidates,
        },
        expected_partition="confirmation",
    )
    normalized["n_genes"] = expected_genes
    normalized["gene_ids"] = sorted(frozen_genes)
    normalized["analysis_gene_ids"] = sorted(
        {
            pair["gene_id"]
            for candidate in normalized["candidates"]
            for family in candidate["families"].values()
            for pair in family["pairs"]
        }
    )
    normalized["source_schema"] = CONFIRMATION_PACKAGE_SCHEMA
    return normalized


def candidate_contract(package: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the sequence-free candidate/family contract recorded in Lock C."""

    contract = []
    for candidate in package["candidates"]:
        contract.append(
            {
                "candidate_id": candidate["candidate_id"],
                "endpoint": candidate["endpoint"],
                "expected_high_minus_low_margin_sign": candidate[
                    "expected_high_minus_low_margin_sign"
                ],
                "matched_controls": list(candidate["matched_controls"]),
                "families": {
                    family_id: {
                        "role": family["role"],
                        "n_pairs": len(family["pairs"]),
                        "pair_ids": [
                            {
                                "example_id": pair["example_id"],
                                "gene_id": pair["gene_id"],
                                "target_position": pair["target_position"],
                                "low_sha256": pair["low_sha256"],
                                "high_sha256": pair["high_sha256"],
                            }
                            for pair in family["pairs"]
                        ],
                    }
                    for family_id, family in candidate["families"].items()
                },
            }
        )
    return contract


def oriented_state_names(expected_sign: int) -> tuple[str, str]:
    """Return (preferred, adverse) names on the frozen expected-effect scale."""

    if expected_sign == 1:
        return "high", "low"
    if expected_sign == -1:
        return "low", "high"
    raise ValueError("expected sign must be -1 or +1")


def oriented_arrays(
    states: Mapping[str, Mapping[str, np.ndarray]],
    expected_sign: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    preferred, adverse = oriented_state_names(expected_sign)
    return dict(states[preferred]), dict(states[adverse])


def activity_statistics(
    preferred_activation: np.ndarray,
    adverse_activation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Discovery-only activity covariates used for matched native-unit controls."""

    preferred = np.asarray(preferred_activation, dtype=np.float64)
    adverse = np.asarray(adverse_activation, dtype=np.float64)
    if preferred.shape != adverse.shape or preferred.ndim != 2:
        raise ValueError("preferred/adverse activations must be same-shaped 2D arrays")
    stacked = np.concatenate((preferred, adverse), axis=0)
    rms = np.sqrt(np.mean(np.square(stacked), axis=0))
    positive_rate = np.mean(stacked > 0.0, axis=0)
    change_rate = np.mean(np.abs(preferred - adverse) > ACTIVATION_EPS, axis=0)
    return {
        "activation_rms": rms,
        "positive_activation_rate": positive_rate,
        "sequence_change_rate": change_rate,
    }


def matched_native_unit_control_orders(
    candidates: Sequence[int],
    output_norms: np.ndarray,
    *,
    activation_rms: np.ndarray,
    positive_activation_rate: np.ndarray,
    sequence_change_rate: np.ndarray,
    tag: str,
    draws: int = CONTROL_DRAWS,
) -> list[np.ndarray]:
    """Match exact-size, unique controls on writer norm and discovery activity.

    Matching is greedy without replacement in robustly standardized
    ``(log writer norm, log activation RMS, positive rate, change rate)`` space.
    Candidate order is retained, so all returned prefixes form valid nested
    matched-size control curves.
    """

    selected = np.asarray(tuple(int(value) for value in candidates), dtype=np.int64)
    covariates = np.stack(
        (
            np.log(np.maximum(np.asarray(output_norms, dtype=np.float64), 1e-30)),
            np.log(np.maximum(np.asarray(activation_rms, dtype=np.float64), 1e-30)),
            np.asarray(positive_activation_rate, dtype=np.float64),
            np.asarray(sequence_change_rate, dtype=np.float64),
        ),
        axis=1,
    )
    if (
        selected.ndim != 1
        or selected.size < 1
        or len(np.unique(selected)) != selected.size
        or covariates.shape != (len(output_norms), 4)
        or not np.isfinite(covariates).all()
    ):
        raise ValueError("invalid candidates or matching covariates")
    if np.any(selected < 0) or np.any(selected >= covariates.shape[0]):
        raise ValueError("candidate native unit outside the MLP width")
    if selected.size > covariates.shape[0] // 2:
        raise ValueError("matched controls require an equally large noncandidate pool")
    median = np.median(covariates, axis=0)
    scale = np.median(np.abs(covariates - median), axis=0)
    scale = np.where(scale > 1e-12, scale, np.std(covariates, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (covariates - median) / scale
    available = np.setdiff1d(
        np.arange(covariates.shape[0]), selected, assume_unique=True
    )
    controls: list[np.ndarray] = []
    for draw in range(draws):
        seed = int.from_bytes(
            hashlib.sha256(f"{tag}:{draw}".encode("utf-8")).digest()[:8], "big"
        )
        rng = np.random.default_rng(seed)
        remaining = available.copy()
        jitter = rng.uniform(0.0, 1e-12, size=remaining.size)
        chosen: list[int] = []
        for candidate in selected:
            distance = np.sum(
                np.square(standardized[remaining] - standardized[candidate]), axis=1
            )
            index = int(np.argmin(distance + jitter))
            chosen.append(int(remaining[index]))
            remaining = np.delete(remaining, index)
            jitter = np.delete(jitter, index)
        control = np.asarray(chosen, dtype=np.int64)
        if len(control) != len(selected) or len(np.unique(control)) != len(control):
            raise RuntimeError("matched control construction lost size or uniqueness")
        if np.intersect1d(control, selected).size:
            raise RuntimeError("matched controls overlap candidate native units")
        controls.append(control)
    return controls


def uniform_native_unit_control_orders(
    candidates: Sequence[int],
    n_units: int,
    *,
    tag: str,
    draws: int = CONTROL_DRAWS,
) -> list[np.ndarray]:
    selected = np.asarray(tuple(int(value) for value in candidates), dtype=np.int64)
    available = np.setdiff1d(np.arange(n_units), selected, assume_unique=True)
    if selected.size > available.size:
        raise ValueError("uniform controls require an equally large noncandidate pool")
    controls = []
    for draw in range(draws):
        seed = int.from_bytes(
            hashlib.sha256(f"{tag}:uniform:{draw}".encode("utf-8")).digest()[:8],
            "big",
        )
        controls.append(
            np.random.default_rng(seed).choice(
                available, size=selected.size, replace=False
            )
        )
    return controls


def exact_unit_ranking(
    preferred: Mapping[str, np.ndarray],
    adverse: Mapping[str, np.ndarray],
    output_weight: np.ndarray,
    parameters: Mapping[str, Any],
    genes: np.ndarray,
) -> dict[str, np.ndarray]:
    delta = preferred["activation"] - adverse["activation"]
    effects = exact_single_unit_effects(
        delta,
        output_weight,
        clean_residual=preferred["mlp_input"],
        clean_write=preferred["mlp_write"],
        corrupt_residual=adverse["mlp_input"],
        corrupt_write=adverse["mlp_write"],
        clean_margin=preferred["margin"],
        corrupt_margin=adverse["margin"],
        parameters=dict(parameters),
    )
    ranked = rank_units_bidirectional(effects["damage"], effects["recovery"], genes)
    return {**ranked, **effects}


def population_effects(
    preferred: Mapping[str, np.ndarray],
    adverse: Mapping[str, np.ndarray],
    output_weight: np.ndarray,
    units: Sequence[int],
    parameters: Mapping[str, Any],
    *,
    noising_activation_delta: np.ndarray | None = None,
    recovery_activation_delta: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Exact write-minus/add intervention for a frozen native-unit population."""

    full_delta = preferred["activation"] - adverse["activation"]
    noising_delta = full_delta if noising_activation_delta is None else noising_activation_delta
    recovery_delta = full_delta if recovery_activation_delta is None else recovery_activation_delta
    noising_write = unit_group_write_delta(noising_delta, output_weight, units)
    recovery_write = unit_group_write_delta(recovery_delta, output_weight, units)
    noised = composed_classifier_margin(
        preferred["mlp_input"],
        preferred["mlp_write"] - noising_write,
        **dict(parameters),
    )
    recovered = composed_classifier_margin(
        adverse["mlp_input"],
        adverse["mlp_write"] + recovery_write,
        **dict(parameters),
    )
    return {
        "damage": preferred["margin"] - noised,
        "recovery": recovered - adverse["margin"],
        "noising_write": noising_write,
        "recovery_write": recovery_write,
    }


def rank4_energy_fraction(write: np.ndarray, basis: np.ndarray) -> np.ndarray:
    values = np.asarray(write, dtype=np.float64)
    directions = np.asarray(basis, dtype=np.float64)
    if values.ndim != 2 or directions.ndim != 2 or values.shape[1] != directions.shape[1]:
        raise ValueError("write vectors and basis have incompatible widths")
    gram_error = np.max(np.abs(directions @ directions.T - np.eye(len(directions))))
    if gram_error > 1e-4:
        raise ValueError("rank-4 task basis is not row-orthonormal")
    numerator = np.sum(np.square(values @ directions.T), axis=1)
    denominator = np.sum(np.square(values), axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float64).eps,
    )


def unit_overlap(selected: Sequence[int], prior_order: Sequence[int]) -> dict[str, Any]:
    current = set(map(int, selected))
    prior = set(map(int, prior_order[: len(current)]))
    overlap = len(current & prior)
    union = len(current | prior)
    return {
        "size": len(current),
        "overlap_count": overlap,
        "jaccard": float(overlap / union) if union else 1.0,
    }


def summarize_five_seed_effect(
    per_seed: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
) -> dict[str, Any]:
    """Average checkpoints within examples, then perform gene-grouped inference."""

    values = np.asarray(per_seed, dtype=np.float64)
    labels = np.asarray(groups, dtype=object)
    if values.ndim != 2 or values.shape[0] != len(EXPECTED_SEEDS):
        raise ValueError("effects must be [five checkpoints, examples]")
    if labels.shape != (values.shape[1],):
        raise ValueError("groups must match examples")
    checkpoint_means = values.mean(axis=1)
    averaged = values.mean(axis=0)
    summary = grouped_bootstrap(averaged, labels, n_boot=n_boot, seed=seed)
    summary["p_signflip_add_one"] = grouped_signflip_pvalue(
        averaged, labels, n_perm=n_perm, seed=seed + 100_000
    )
    summary["checkpoint_estimates"] = {
        str(checkpoint): float(value)
        for checkpoint, value in zip(EXPECTED_SEEDS, checkpoint_means)
    }
    summary["all_five_checkpoint_signs_positive"] = bool(
        np.all(checkpoint_means > 0.0)
    )
    return summary


def apply_declared_bh_family(
    entries: list[dict[str, Any]],
    *,
    field: str = "q_bh_full_native_unit_family",
) -> None:
    """Correct exactly one explicitly enumerated family in-place."""

    if not entries:
        raise ValueError("BH family cannot be empty")
    qvalues = benjamini_hochberg(
        [float(entry["summary"]["p_signflip_add_one"]) for entry in entries]
    )
    for index, (entry, qvalue) in enumerate(zip(entries, qvalues)):
        entry["summary"][field] = float(qvalue)
        entry["bh_family_index"] = index
        entry["bh_family_size"] = len(entries)


def load_rank4_basis(npz_path: str | Path, seed: int, endpoint: str) -> np.ndarray:
    key = f"basis_{seed}_{endpoint}"
    with np.load(npz_path, allow_pickle=False) as archive:
        if key not in archive:
            raise KeyError(f"GFS19 basis missing {key}")
        basis = np.asarray(archive[key][:4], dtype=np.float64)
    if basis.shape != (4, 512):
        raise ValueError(f"GFS19 {key} does not provide a 4x512 task code")
    return basis


__all__ = [
    "ACTIVATION_EPS",
    "CONFIRMATION_SCHEMA",
    "CONFIRMATION_PACKAGE_SCHEMA",
    "CONTROL_DRAWS",
    "DISCOVERY_SCHEMA",
    "ENDPOINT_LABELS",
    "EXPECTED_SEEDS",
    "GROUP_SIZES",
    "INTERMEDIATE_SIZE",
    "LOCK_SCHEMA",
    "N_BOOT",
    "N_PERM",
    "PACKAGE_SCHEMA",
    "activity_statistics",
    "apply_declared_bh_family",
    "candidate_contract",
    "canonical_json",
    "exact_unit_ranking",
    "load_rank4_basis",
    "matched_native_unit_control_orders",
    "normalize_construction_package",
    "normalize_confirmation_states",
    "normalize_discovery_audit",
    "oriented_arrays",
    "oriented_state_names",
    "population_effects",
    "rank4_energy_fraction",
    "sha256_text",
    "summarize_five_seed_effect",
    "uniform_native_unit_control_orders",
    "unit_overlap",
]
