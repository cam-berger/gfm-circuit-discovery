#!/usr/bin/env python3
"""Frozen, model-free calibration of the register-operator production instrument.

Full calibration requires a hash-bound FROZEN receipt. ``--smoke`` uses only small,
deterministic fixtures; it never consumes a development or held-out RNG stream.
No model is loaded, and no archived implementation is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta, binom

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.artifact_guard import guard_output_path
from gfmcd.provenance import ExperimentManifest, sha256_file

SCHEMA = "gfmcd.register_operator.statistical_calibration.v1"
PAYLOAD_SCHEMA = "gfmcd.register_operator.score_payload.v1"
MDE_FACTOR = 1.96 + 0.8416212335729143
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260803
PSEUDO_SEEDS = tuple(range(2026080401, 2026080500))
STREAMS = {"development": 2026101601, "null": 2026101602, "plants": 2026101603}
REPETITIONS = {"development": 256, "null": 1000, "plants": 256}
EXPECTED_N = 342
SELECTION_IDS_SHA256 = "972d388cc031b6debdbb5b1c531afb7a8619630462f6bf4bb380d6a989417436"
FPR_MAX = 38
NONCOVERAGE_MAX = 71
PLANT_SUCCESS_MIN = 216
TRUTH_TOLERANCE = 1e-10
FAMILIES = (
    "primary_psi", "directions", "mismatch_equivalence", "bridge_diagnostics",
    "historical_psi_ac_minus_at_linear", "historical_phase_blind_ac_minus_at_linear",
)
POSITIVE_CLASSES = frozenset({
    "ONLY_WHEN_COHERENT_IN_FROZEN_DOMAIN", "MATERIAL_PREFERENTIAL_POSITIVE",
    "DETECTABLE_MATERIALITY_UNRESOLVED", "DETECTABLE_PREFERENCE_MATERIALITY_UNRESOLVED",
    "DETECTABLE_BOUNDED_SUB_SESOI",
})
MATERIAL_CLASSES = frozenset({
    "ONLY_WHEN_COHERENT_IN_FROZEN_DOMAIN", "MATERIAL_PREFERENTIAL_POSITIVE",
})
CELL_STATES = ("intact", "cc", "ac", "at", "bridge_ac", "acceptor_intact")
NATIVE_STATES = ("intact", "cc", "ac", "acceptor_intact")
H = np.eye(3, dtype=float) - np.full((3, 3), 1 / 3)
CODE_FILES = (
    "scripts/splicebert_register_operator_replication/calibrate_statistics.py",
    "src/gfmcd/splicing/register_operator_measurement.py",
    "src/gfmcd/splicing/register_operator_statistics.py",
    "src/gfmcd/provenance.py",
)


def _production():
    from gfmcd.splicing import register_operator_measurement as measurement
    from gfmcd.splicing import register_operator_statistics as statistics
    return measurement, statistics


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def exact_binomial_interval(successes: int, total: int, confidence: float = .95) -> list[float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("binomial counts must satisfy 0 <= successes <= total")
    alpha = 1 - confidence
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [low, high]


def binomial_record(successes: int, total: int) -> dict[str, Any]:
    ci = exact_binomial_interval(successes, total)
    return {"count": successes, "total": total, "rate": successes / total, "ci95": ci,
            "lower95_one_sided": (0.0 if successes == 0 else
                float(beta.ppf(.05, successes, total - successes + 1))),
            "upper95_one_sided": (1.0 if successes == total else
                float(beta.ppf(.95, successes + 1, total - successes)))}


def frozen_settings() -> dict[str, Any]:
    return {
        "n_blocks": EXPECTED_N, "repetitions": REPETITIONS, "streams": STREAMS,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "pseudo_seeds": list(PSEUDO_SEEDS), "fpr_max": FPR_MAX,
        "noncoverage_max_each_family": NONCOVERAGE_MAX,
        "plant_success_min": PLANT_SUCCESS_MIN, "truth_tolerance": TRUTH_TOLERANCE,
        "plant_deltas": [.05, .15], "capable_fixture_nominal_mde": .035,
        "family_names": list(FAMILIES),
        "runtime_packages": {name: package_version(name) for name in ("numpy", "scipy", "torch")},
        "bit_generator": "PCG64",
        "development_schedule": ["additive_null", "heterogeneous_null", "plant_point05", "plant_point15"],
    }


def load_selection(path: Path, *, require_frozen_count: bool = True) -> tuple[list[str], list[str]]:
    selection = json.loads(path.read_text())
    ids = sorted(map(str, selection["block_ids"]))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("selection requires distinct nonempty block IDs")
    strata = []
    for block_id in ids:
        parts = block_id.split(":")
        if len(parts) != 4 or parts[2] not in ("+", "-"):
            raise ValueError("cannot derive chromosome/strand from block ID")
        strata.append(f"{parts[1]}:{parts[2]}")
    if len(ids) % 6 or min(Counter(strata).values()) < 2:
        raise ValueError("selection violates balanced-S3 or bootstrap-stratum requirements")
    if require_frozen_count:
        actual_ids_sha256 = hashlib.sha256(canonical_bytes(ids)).hexdigest()
        if (len(ids) != EXPECTED_N or selection.get("block_ids_sha256") != SELECTION_IDS_SHA256
                or actual_ids_sha256 != SELECTION_IDS_SHA256):
            raise ValueError("calibration must use the approved sequence-selected 342 blocks")
        if dict(Counter(strata)) != selection.get("counts_by_stratum"):
            raise ValueError("selection's recorded stratum counts do not recompute")
    return ids, strata


def verify_freeze(path: Path, selection_path: Path, specification_path: Path) -> dict[str, Any]:
    """Verify every declared file and required calibration code BEFORE opening RNG streams."""
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "FROZEN":
        raise ValueError("full calibration requires a FROZEN receipt")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("freeze receipt requires path-to-sha256 file bindings")
    bound = {}
    for name, expected in files.items():
        resolved = Path(name)
        if not resolved.is_absolute():
            resolved = REPO / resolved
        resolved = resolved.resolve()
        if not resolved.is_file() or sha256_file(resolved) != expected:
            raise ValueError(f"frozen file hash mismatch: {name}")
        bound[str(resolved)] = expected
    required = [selection_path.resolve(), specification_path.resolve(),
                *(REPO / relative for relative in CODE_FILES)]
    missing = [str(p) for p in required if str(p.resolve()) not in bound]
    if missing:
        raise ValueError(f"freeze omits required calibration inputs: {missing}")
    if receipt.get("selection_sha256") != sha256_file(selection_path):
        raise ValueError("freeze selection_sha256 does not match selected cohort")
    calibration = receipt.get("calibration", {})
    for key, expected in frozen_settings().items():
        if calibration.get(key) != expected:
            raise ValueError(f"freeze calibration constant differs: {key}")
    return receipt


@dataclass(frozen=True)
class Latents:
    row: np.ndarray
    column: np.ndarray
    common_offset: np.ndarray
    interactions: np.ndarray  # [block, operator(CC/AC/AT), donor, acceptor]
    spread_noise: np.ndarray  # [block, CC/shared-AC-AT, donor, acceptor]
    bridge: np.ndarray
    native: np.ndarray
    health_delta: np.ndarray
    rival_orders: np.ndarray
    acceptor_permutations: np.ndarray


def draw_latents(seed: np.random.SeedSequence, n: int) -> Latents:
    """Draw once; both planted amplitudes reuse this object byte for byte."""
    rng = np.random.default_rng(seed)
    if type(rng.bit_generator).__name__ != "PCG64":
        raise RuntimeError("frozen calibration requires the PCG64 bit generator")
    row = rng.normal(0, .10, size=(n, 3, 1))
    column = rng.normal(0, .10, size=(n, 1, 3))
    common = rng.normal(0, .10, size=(n, 1, 1))
    z = rng.normal(size=(n, 3, 3, 3))
    multiplier = rng.choice([.5, 1.5], size=(n, 1, 1)) / math.sqrt(1.25)
    sigma = .035 * math.sqrt(n) / MDE_FACTOR
    cc = z[:, 0]
    ac = .9 * z[:, 0] + math.sqrt(.19) * z[:, 1]
    at = .95 * ac + math.sqrt(.0975) * z[:, 2]
    residuals = np.stack((cc, ac, at), axis=1) * multiplier[:, None] * math.sqrt(2) * sigma
    noise = rng.uniform(-.005, .005, size=(n, 2, 3, 3))
    bridge = rng.normal(0, .01, size=n)
    native = rng.normal(0, .02, size=n)
    health = rng.normal(0, .01, size=(n, 1, 24))
    # 54 crossed + 24 native four-logit requests per block. Health is a separate assay.
    rival_orders = np.argsort(rng.random(size=(n * 78, 3)), axis=1)
    permutations = np.argsort(rng.random(size=(n, 3)), axis=1)
    return Latents(row, column, common, residuals, noise, bridge, native,
                   health, rival_orders, permutations)


def deterministic_latents(n: int, *, noise_scale: float = .001) -> Latents:
    """Analytic toy fields; never consume a frozen stochastic stream."""
    grid = np.arange(n * 9, dtype=float).reshape(n, 3, 3)
    residual = noise_scale * np.sin(grid + .3)
    zeros = np.zeros((n, 3, 3))
    return Latents(
        np.zeros((n, 3, 1)), np.zeros((n, 1, 3)), np.zeros((n, 1, 1)),
        np.stack([residual] * 3, axis=1), np.stack([zeros] * 2, axis=1),
        np.zeros(n), np.zeros(n), np.zeros((n, 1, 24)),
        np.asarray([np.roll(np.arange(3), (i + i // 6) % 3) for i in range(n * 78)]),
        np.tile(np.arange(3), (n, 1)),
    )


def fixed_tie_membership(n: int) -> np.ndarray:
    """Exactly floor(10% of margin requests), before the generating label permutation."""
    return np.arange((n * 78) // 10, dtype=int) * 10


def _four_logits(g: np.ndarray, spread: np.ndarray, orders: np.ndarray,
                 tie_indices: np.ndarray) -> np.ndarray:
    gflat, k = g.reshape(-1), spread.reshape(-1)
    rivals = np.stack([np.zeros_like(k), -1.5 * k, -1.5 * k], axis=1)
    rivals[tie_indices] = np.stack([np.zeros(len(tie_indices)), np.zeros(len(tie_indices)),
                                  -3 * k[tie_indices]], axis=1)
    rivals = np.take_along_axis(rivals, orders, axis=1)
    logits = np.empty((len(gflat), 4), dtype=float)
    logits[:, [0, 1, 3]] = rivals
    logits[:, 2] = gflat
    return logits


def make_payload(block_ids: Sequence[str], strata: Sequence[str], latent: Latents, *,
                 delta: float, null: bool, additive_only: bool = False,
                 matrices: np.ndarray | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    n = len(block_ids)
    if n != len(strata) or latent.row.shape[0] != n:
        raise ValueError("fixture identities, strata and latent blocks disagree")
    offsets = np.asarray([.10, .08, .08] if null else [.05, .05, .05])
    coupling = offsets[None, :, None, None] + latent.row[:, None] + latent.column[:, None]
    if additive_only:
        coupling = coupling + latent.common_offset[:, None]
    else:
        coupling = coupling + latent.interactions
    coupling = coupling + delta * H
    if matrices is not None:
        if matrices.shape != (n, 3, 3, 3):
            raise ValueError("explicit matrices must have block/operator/3/3 shape")
        coupling = np.asarray(matrices, dtype=float)
    cell_g = np.full((n, 9, 6), 3.0)
    cell_k = np.full_like(cell_g, 2.0)
    for operator, state in enumerate((1, 2, 3)):
        cell_g[:, :, state] -= coupling[:, operator].reshape(n, 9)
        spread = .01 * H + latent.spread_noise[:, 0 if operator == 0 else 1]
        cell_k[:, :, state] -= spread.reshape(n, 9)
    cell_g[:, :, 4] -= latent.bridge[:, None]
    native_g = np.full((n, 6, 4), 3.0)
    native_g[:, :, 1] -= .30 + latent.native[:, None]
    native_g[:, :, 2] -= .30 + latent.native[:, None]
    native_k = np.full_like(native_g, 2.0)
    all_g = np.concatenate((cell_g.reshape(-1), native_g.reshape(-1)))
    all_k = np.concatenate((cell_k.reshape(-1), native_k.reshape(-1)))
    logits = _four_logits(all_g, all_k, latent.rival_orders, fixed_tie_membership(n))
    cell_logits = logits[:n * 54].reshape(n, 9, 6, 4)
    native_logits = logits[n * 54:].reshape(n, 6, 4, 4)
    if null:
        inverse = np.argsort(latent.acceptor_permutations, axis=1)
        order = (3 * np.arange(3)[None, :, None] + inverse[:, None, :]).reshape(n, 9)
        cell_logits = np.take_along_axis(cell_logits, order[:, :, None, None], axis=1)
    source_logp = np.full((n, 1, 24), -1.0)
    construct_logp = source_logp + latent.health_delta
    if np.any(construct_logp >= 0):
        raise ValueError("synthetic true-base log probability is outside its domain")
    logp = np.stack((construct_logp, source_logp), axis=3)
    health_logits = np.zeros((n, 1, 24, 2, 4), dtype=float)
    health_logits[..., 2] = math.log(3) + logp - np.log1p(-np.exp(logp))
    phases = np.asarray([(p, q) for p in range(3) for q in range(3)], dtype=int)
    payload = {
        "schema": PAYLOAD_SCHEMA, "synthetic": True,
        "block_ids": list(block_ids), "strata": list(strata),
        "cell_phases": np.broadcast_to(phases, (n, 9, 2)).copy(),
        "cell_logits": cell_logits, "native_logits": native_logits,
        "health_logits": health_logits,
        "health_true_base_ids": np.full((n, 1, 24), 2, dtype=int),
        "health_base_ids": {"A": 0, "C": 1, "G": 2, "T": 3},
        "health_repeat_cells": 9,
        "prerequisites": {name: True for name in (
            "construction", "smoke", "fidelity", "calibration", "independent_audit")},
        "numerics": {"all_finite": True, "token_count": 1024,
                     "mask_mapping_valid": True, "repeat_max_abs": 0.0},
    }
    truths = {"primary_psi": [0.0 if null else delta],
              "directions": [0.0 if null else delta] * 5,
              "mismatch_equivalence": [float(offsets[0] - (0 if null else delta / 3))] * 2,
              "bridge_diagnostics": [0.] * 4 if null else [0., 0., -.01, -.01],
              "historical_psi_ac_minus_at_linear": [0.],
              "historical_phase_blind_ac_minus_at_linear": [0.]}
    details = {"delta": delta, "null": null, "additive_only": additive_only,
               "population_truths": truths, "nominal_fixture_mde": .035,
               "generating_acceptor_permutations": latent.acceptor_permutations.tolist() if null else None,
               "tie_request_count": len(fixed_tie_membership(n)),
               "tie_request_membership_sha256": hashlib.sha256(
                   canonical_bytes(fixed_tie_membership(n))).hexdigest()}
    return payload, details


def interval_arrays(summary: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Accept the production scalar/vector summary shapes; never infer a missing bound."""
    if "lower" in summary and "upper" in summary:
        lower = np.atleast_1d(np.asarray(summary["lower"], dtype=float))
        upper = np.atleast_1d(np.asarray(summary["upper"], dtype=float))
    elif "ci95" in summary:
        ci = np.asarray(summary["ci95"], dtype=float)
        if ci.shape != (2,):
            raise ValueError("scalar ci95 must have exactly two bounds")
        lower, upper = ci[:1], ci[1:]
    elif "simultaneous_ci95" in summary:
        ci = np.asarray(summary["simultaneous_ci95"], dtype=float)
        if ci.shape != (2,):
            raise ValueError("scalar simultaneous_ci95 must have exactly two bounds")
        lower, upper = ci[:1], ci[1:]
    elif "endpoints" in summary:
        intervals = [interval_arrays(value) for value in summary["endpoints"].values()]
        lower = np.concatenate([pair[0] for pair in intervals])
        upper = np.concatenate([pair[1] for pair in intervals])
    else:
        raise ValueError("production interval has neither scalar nor vector bounds")
    if lower.shape != upper.shape or not np.isfinite([lower, upper]).all() or np.any(lower > upper):
        raise ValueError("invalid production interval bounds")
    return lower, upper


def family_noncoverage(analysis: Mapping[str, Any], truths: Mapping[str, Any]) -> dict[str, bool]:
    failures = {}
    for family in FAMILIES:
        lower, upper = interval_arrays(analysis[family])
        truth = np.asarray(truths[family], dtype=float)
        if truth.shape != lower.shape:
            raise ValueError(f"known-truth shape differs from production family {family}")
        failures[family] = bool(np.any(truth < lower - TRUTH_TOLERANCE)
                                or np.any(truth > upper + TRUTH_TOLERANCE))
    return failures


def analyze_payload(payload: Mapping[str, Any], context: Any) -> tuple[dict, dict, str]:
    measurement, statistics = _production()
    measurement.validate_score_payload(payload)
    serialized = measurement.serialize_score_payload(payload)
    if isinstance(serialized, str):
        serialized = serialized.encode("utf-8")
    decoded = measurement.deserialize_score_payload(serialized)
    measurement.validate_score_payload(decoded)
    reduced = measurement.reduce_score_payload(decoded)
    analysis = statistics.analyze(reduced, context=context)
    allowed_classes = POSITIVE_CLASSES | {"BOUNDED_NULL", "MATERIAL_INVERTED", "INCONCLUSIVE",
                                         "INSTRUMENT_UNRESOLVED"}
    if analysis.get("classification") not in allowed_classes:
        raise ValueError(f"unknown production classification: {analysis.get('classification')}")
    # This also verifies that all mandatory families have finite, correctly shaped intervals.
    for family in FAMILIES:
        interval_arrays(analysis[family])
    return analysis, reduced, hashlib.sha256(serialized).hexdigest()


def _seed_record(seed: np.random.SeedSequence) -> dict[str, Any]:
    return {"entropy": seed.entropy, "spawn_key": list(seed.spawn_key),
            "pool_size": seed.pool_size, "bit_generator": "PCG64"}


def _write_record(handle: Any, record: dict) -> None:
    handle.write(canonical_bytes(record) + b"\n")


def _evaluate(block_ids: list[str], strata: list[str], latent: Latents, context: Any, *,
              phase: str, index: int, seed: np.random.SeedSequence | None,
              delta: float, null: bool, additive_only: bool = False,
              records: Any | None = None, matrices: np.ndarray | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"phase": phase, "index": index,
                           "seed_sequence": _seed_record(seed) if seed is not None else None,
                           "delta": delta, "status": "ERROR"}
    started = time.monotonic()
    try:
        payload, details = make_payload(block_ids, strata, latent, delta=delta, null=null,
                                        additive_only=additive_only, matrices=matrices)
        analysis, reduced, digest = analyze_payload(payload, context)
        noncoverage = family_noncoverage(analysis, details["population_truths"])
        lower, _ = interval_arrays(analysis["primary_psi"])
        row.update({"status": "COMPLETE", "fixture": details,
                    "raw_four_logit_payload_sha256": digest,
                    "analysis": analysis, "block_sufficient_statistics": reduced,
                    "family_noncoverage": noncoverage,
                    "positive_classification": analysis["classification"] in POSITIVE_CLASSES,
                    "material_classification": analysis["classification"] in MATERIAL_CLASSES,
                    "scalar_detection": bool(lower[0] > 0),
                    "failed_capability_gates": [name for name, passed in analysis["gates"].items()
                                                 if passed is not True]})
    except Exception as exc:
        row["error"] = {"type": type(exc).__name__, "message": str(exc)}
    row["wall_seconds"] = time.monotonic() - started
    if records is not None:
        _write_record(records, row)
    return row


def evaluate_acceptance(*, null_positive: int, noncoverage: Mapping[str, int],
                        detection_successes: int, material_successes: int,
                        execution_errors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if set(noncoverage) != set(FAMILIES):
        raise ValueError("calibration must report every frozen coverage family")
    null_record = binomial_record(null_positive, REPETITIONS["null"])
    coverage = {name: binomial_record(int(noncoverage[name]), REPETITIONS["null"])
                for name in FAMILIES}
    detection = binomial_record(detection_successes, REPETITIONS["plants"])
    material = binomial_record(material_successes, REPETITIONS["plants"])
    checks = {"no_execution_errors": not execution_errors,
              "full_decision_false_positive_rate": null_positive <= FPR_MAX,
              **{f"coverage:{name}": noncoverage[name] <= NONCOVERAGE_MAX for name in FAMILIES},
              "detect_point05": detection_successes >= PLANT_SUCCESS_MIN,
              "material_point15": material_successes >= PLANT_SUCCESS_MIN}
    return {"all_passed": all(checks.values()), "checks": checks,
            "failures": [name for name, passed in checks.items() if not passed],
            "full_decision_false_positive": null_record, "noncoverage_by_family": coverage,
            "point05_detection": detection, "point15_material": material,
            "execution_errors": list(execution_errors),
            "coverage_audit": {
                "nominal_noncoverage": .05, "audit_alpha_each": .01 / 6,
                "threshold": NONCOVERAGE_MAX,
                "exact_false_failure_each_at_nominal": float(binom.sf(NONCOVERAGE_MAX, 1000, .05)),
                "union_bound_false_failure": float(6 * binom.sf(NONCOVERAGE_MAX, 1000, .05)),
                "interpretation": "Binomial goodness-of-coverage acceptance; not proof of exact nominal coverage."}}


def run_calibration(block_ids: list[str], strata: list[str], records: Any) -> dict[str, Any]:
    _, statistics = _production()
    context = statistics.make_inference_context(block_ids, strata, draws=BOOTSTRAP_DRAWS,
                                                 seed=BOOTSTRAP_SEED, pseudo_seeds=PSEUDO_SEEDS)
    errors: list[dict[str, Any]] = []
    development_classes: Counter[str] = Counter()
    null_classes: Counter[str] = Counter()
    plant_classes: dict[str, Counter[str]] = {"point05": Counter(), "point15": Counter()}
    noncoverage = {name: 0 for name in FAMILIES}
    null_positive = detection = material = 0
    n = len(block_ids)
    header = {"record_type": "calibration_header", "settings": frozen_settings(),
              "block_ids": block_ids, "strata": strata, "model_loaded": False,
              "new_model_outputs": 0, "numpy": np.__version__,
              "tie_request_axis": "all cell block/cell/state requests, then all native block/source/state requests",
              "tie_request_flat_indices_before_generating_relabeling": fixed_tie_membership(n)}
    _write_record(records, header)

    def note_error(row: dict[str, Any]) -> None:
        errors.append({key: row[key] for key in ("phase", "index", "delta", "error")})

    dev_seeds = np.random.SeedSequence(STREAMS["development"]).spawn(REPETITIONS["development"])
    for i, seed in enumerate(dev_seeds):
        kind = i % 4
        is_null = kind < 2
        row = _evaluate(block_ids, strata, draw_latents(seed, n), context,
                        phase="development", index=i, seed=seed,
                        delta=0 if is_null else (.05 if kind == 2 else .15),
                        null=is_null, additive_only=kind == 0, records=records)
        if row["status"] == "ERROR":
            note_error(row)
        else:
            development_classes[row["analysis"]["classification"]] += 1
        if (i + 1) % 10 == 0:
            print(json.dumps({"phase": "development", "completed": i + 1}), flush=True)

    if errors:
        return {"schema": SCHEMA, "status": "INSTRUMENT_UNRESOLVED",
                "stage": "DEVELOPMENT_FAILED", "development_stream_opened": True,
                "heldout_stream_opened": False, "plant_stream_opened": False,
                "model_loaded": False, "new_model_outputs": 0,
                "settings": frozen_settings(), "execution_errors": errors,
                "classification_counts": {"development": dict(development_classes)},
                "reason": "Development path errors must be corrected before exposing held-out streams."}

    null_seeds = np.random.SeedSequence(STREAMS["null"]).spawn(REPETITIONS["null"])
    for i, seed in enumerate(null_seeds):
        row = _evaluate(block_ids, strata, draw_latents(seed, n), context,
                        phase="null", index=i, seed=seed, delta=0, null=True,
                        additive_only=i < 500, records=records)
        if row["status"] == "ERROR":
            note_error(row)
            for family in FAMILIES:
                noncoverage[family] += 1
        else:
            null_classes[row["analysis"]["classification"]] += 1
            null_positive += int(row["positive_classification"])
            for family in FAMILIES:
                noncoverage[family] += int(row["family_noncoverage"][family])
        if (i + 1) % 10 == 0:
            print(json.dumps({"phase": "null", "completed": i + 1}), flush=True)

    plant_seeds = np.random.SeedSequence(STREAMS["plants"]).spawn(REPETITIONS["plants"])
    for i, seed in enumerate(plant_seeds):
        latent = draw_latents(seed, n)
        for delta, label in ((.05, "point05"), (.15, "point15")):
            row = _evaluate(block_ids, strata, latent, context,
                            phase="plants", index=i, seed=seed, delta=delta,
                            null=False, records=records)
            if row["status"] == "ERROR":
                note_error(row)
            else:
                plant_classes[label][row["analysis"]["classification"]] += 1
                if delta == .05:
                    detection += int(row["scalar_detection"])
                else:
                    material += int(row["material_classification"])
        if (i + 1) % 10 == 0:
            print(json.dumps({"phase": "plants", "completed_seed_pairs": i + 1}), flush=True)
    acceptance = evaluate_acceptance(null_positive=null_positive, noncoverage=noncoverage,
                                     detection_successes=detection, material_successes=material,
                                     execution_errors=errors)
    return {"schema": SCHEMA, "status": "PASS" if acceptance["all_passed"] else "INSTRUMENT_UNRESOLVED",
            "synthetic": True, "model_loaded": False, "new_model_outputs": 0,
            "development_stream_opened": True, "heldout_stream_opened": True,
            "plant_stream_opened": True,
            "settings": frozen_settings(), "acceptance": acceptance,
            "classification_counts": {"development": dict(development_classes),
                                       "null": dict(null_classes),
                                       "plants": {key: dict(value) for key, value in plant_classes.items()}},
            "scope": "Synthetic statistical capability only; actual CC/max variance and model gates remain unmeasured."}


def run_smoke(*, draws: int = 512) -> dict[str, Any]:
    """Small deterministic path/branch memorials; no frozen stochastic stream is opened."""
    measurement, statistics = _production()
    ids = [f"synthetic:chr1:{'+' if i < 6 else '-'}:b{i:03d}" for i in range(12)]
    strata = ["chr1:+"] * 6 + ["chr1:-"] * 6
    context = statistics.make_inference_context(ids, strata, draws=draws,
                                                seed=BOOTSTRAP_SEED, pseudo_seeds=PSEUDO_SEEDS)
    latent = deterministic_latents(12)
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, **detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **jsonable(detail)})

    def classify_fixture(name: str, delta: float, expected: set[str], *,
                         matrices: np.ndarray | None = None) -> dict | None:
        payload, _ = make_payload(ids, strata, latent, delta=delta, null=False, matrices=matrices)
        try:
            analysis, _, _ = analyze_payload(payload, context)
            record(name, analysis["classification"] in expected,
                   classification=analysis["classification"], primary=analysis["primary_psi"],
                   gates=analysis["gates"])
            return analysis
        except Exception as exc:
            record(name, False, error=f"{type(exc).__name__}: {exc}")
            return None

    classify_fixture("material_preference", .15, set(MATERIAL_CLASSES))
    classify_fixture("bounded_null", 0, {"BOUNDED_NULL"})
    classify_fixture("material_inversion", -.15, {"MATERIAL_INVERTED"})
    classify_fixture("detectable_materiality_unresolved", .05,
                     {"DETECTABLE_PREFERENCE_MATERIALITY_UNRESOLVED", "DETECTABLE_MATERIALITY_UNRESOLVED"})
    classify_fixture("detectable_sub_sesoi", .02,
                     {"DETECTABLE_MATERIALITY_UNRESOLVED", "DETECTABLE_BOUNDED_SUB_SESOI"})
    # Every row remains phase-blind neutral, but donor phase0 is contrary to the target.
    conflict = .10 + np.asarray([-.15, .30, .30])[:, None] * H
    matrices = np.broadcast_to(conflict, (12, 3, 3, 3)).copy() + latent.interactions
    classify_fixture("conflicting_direction", 0, {"INCONCLUSIVE"}, matrices=matrices)
    mismatch = np.zeros((3, 3))
    for p in range(3):
        mismatch[p, p] = .15
        mismatch[p, (p + 1) % 3] = -.08
    matrices = np.broadcast_to(mismatch, (12, 3, 3, 3)).copy() + latent.interactions
    one_bad = classify_fixture("one_mismatch_fails_equivalence", 0,
                               {"MATERIAL_PREFERENTIAL_POSITIVE"}, matrices=matrices)
    if one_bad is not None:
        lower, upper = interval_arrays(one_bad["mismatch_equivalence"])
        record("exactly_one_mismatch_equivalent", int(np.sum((lower >= -.05) & (upper <= .05))) == 1)
    high = .10 + np.asarray([1., -1.] * 6)[:, None, None] * H
    high_matrices = np.repeat(high[:, None, :, :], 3, axis=1)
    high_result = classify_fixture("insufficient_resolution", 0, {"INSTRUMENT_UNRESOLVED"}, matrices=high_matrices)
    if high_result is not None:
        mde = high_result["primary_psi"].get("mde80")
        record("high_noise_observed_mde", mde is not None and mde > .05, observed_mde80=mde)

    payload, _ = make_payload(ids, strata, latent, delta=.15, null=False)
    # Reverse or perturb one field on the actual raw-logit path; source summaries cannot mask it.
    corruptions = {
        "misassigned_phase": lambda x: x["cell_phases"].__setitem__((0, 1), [1, 0]),
        "duplicate_phase": lambda x: x["cell_phases"].__setitem__((0, 1), [0, 0]),
        "missing_cell": lambda x: x.__setitem__("cell_logits", x["cell_logits"][:, :8]),
        "nonfinite_logit": lambda x: x["cell_logits"].__setitem__((0, 0, 0, 0), np.nan),
    }
    import copy
    for name, mutate in corruptions.items():
        bad = copy.deepcopy(payload)
        mutate(bad)
        try:
            measurement.validate_score_payload(bad)
        except (ValueError, KeyError, TypeError):
            record(name, True)
        else:
            record(name, False, error="invalid raw payload was accepted")
    unhealthy = copy.deepcopy(payload)
    # log p(true)=-2 in constructs versus -1 in sources; correctness changes are also measured.
    unhealthy["health_logits"][..., 0, 2] = math.log(3) - 2 - math.log1p(-math.exp(-2))
    try:
        result, _, _ = analyze_payload(unhealthy, context)
        record("failed_health", result["classification"] == "INSTRUMENT_UNRESOLVED",
               classification=result["classification"], gates=result["gates"])
    except Exception as exc:
        record("failed_health", False, error=str(exc))
    source_zero = copy.deepcopy(payload)
    source_zero["health_logits"][..., 1, 2] = -3.0
    try:
        result, _, _ = analyze_payload(source_zero, context)
        accuracy = result["instrument"]["protected_accuracy"]
        record("zero_source_accuracy_denominator", result["classification"] == "INSTRUMENT_UNRESOLVED"
               and accuracy["source_correct"] == 0 and accuracy["retention_ratio"] is None,
               accuracy=accuracy, gates=result["gates"])
    except Exception as exc:
        record("zero_source_accuracy_denominator", False, error=str(exc))
    # AC-minus-AT Psi=.20 while phase-blind sensitivity stays exactly zero.
    sensitivity_bad = copy.deepcopy(payload)
    sensitivity_bad["cell_logits"][:, :, 3, 2] += .20 * H.reshape(1, 9)
    try:
        result, _, _ = analyze_payload(sensitivity_bad, context)
        low, high_bound = interval_arrays(result["historical_psi_ac_minus_at_linear"])
        record("failed_historical_sensitivity", result["classification"] not in MATERIAL_CLASSES
               and bool(np.any(low < -.05) or np.any(high_bound > .05)),
               classification=result["classification"], sensitivity=result["historical_psi_ac_minus_at_linear"])
    except Exception as exc:
        record("failed_historical_sensitivity", False, error=str(exc))
    tied = np.full((12, 3, 3, 3), .125)
    tie_payload, _ = make_payload(ids, strata, deterministic_latents(12, noise_scale=0),
                                  delta=0, null=False, matrices=tied)
    try:
        result, _, _ = analyze_payload(tie_payload, context)
        pseudo = result.get("pseudo_phase", result.get("pseudo", {}))
        pvalue = pseudo.get("exact_add_one_one_sided_p")
        record("tied_pseudo", pvalue == 1.0 and pseudo.get("target_exceeds_all") is False,
               pseudo=pseudo, classification=result["classification"])
    except Exception as exc:
        record("tied_pseudo", False, error=str(exc))
    return {"schema": SCHEMA, "status": "SMOKE_PASS" if all(x["passed"] for x in checks) else "SMOKE_FAIL",
            "synthetic": True, "stochastic_calibration_executed": False,
            "development_stream_opened": False, "heldout_stream_opened": False,
            "model_loaded": False, "new_model_outputs": 0, "bootstrap_draws": draws,
            "checks": checks, "failures": [x["name"] for x in checks if not x["passed"]]}


def verify_output_bindings(receipt: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    declared = receipt.get("outputs", {})
    for name, actual in outputs.items():
        frozen = declared.get(name)
        if not isinstance(frozen, str) or not Path(frozen).is_absolute():
            raise ValueError(f"freeze omits absolute output authorization: {name}")
        if Path(frozen).resolve() != actual.resolve():
            raise ValueError(f"requested output differs from freeze: {name}")


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--specification", type=Path,
                        default=REPO / "studies/splicebert_register_operator_replication/STATISTICAL_CALIBRATION_DRAFT.md")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records-out", type=Path)
    parser.add_argument("--smoke", action="store_true", help="deterministic toy memorials only")
    args = parser.parse_args(argv)
    out = guard_output_path(args.out)
    manifest_path = guard_output_path(out.with_suffix(".manifest.json"))
    attempt_path = guard_output_path(out.with_suffix(".attempt.json"))
    records_path = guard_output_path(args.records_out or out.with_suffix(".records.jsonl.gz"))
    inputs = [REPO / relative for relative in CODE_FILES]
    started = time.monotonic()
    if args.smoke:
        result = run_smoke()
    else:
        if args.freeze is None or args.selection is None:
            parser.error("full calibration requires --freeze and --selection; no RNG stream was opened")
        initial_freeze_sha256 = sha256_file(args.freeze)
        initial_selection_sha256 = sha256_file(args.selection)
        receipt = verify_freeze(args.freeze, args.selection, args.specification)
        verify_output_bindings(receipt, {
            "calibration": out, "calibration_records": records_path,
            "calibration_attempt": attempt_path, "calibration_manifest": manifest_path})
        block_ids, strata = load_selection(args.selection)
        inputs += [args.freeze, args.selection, args.specification]
        _exclusive_json(attempt_path, {"schema": SCHEMA, "stage": "ATTEMPT",
                                      "freeze_sha256": initial_freeze_sha256,
                                      "selection_sha256": initial_selection_sha256,
                                      "frozen_files": receipt["files"],
                                      "settings": frozen_settings(), "model_loaded": False,
                                      "out": str(out.resolve()), "records_out": str(records_path.resolve())})
        records_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive output creation prevents an accidental repeat from replacing evidence.
        with records_path.open("xb") as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as handle:
                try:
                    smoke = run_smoke()
                    _write_record(handle, {"record_type": "deterministic_smoke", "result": smoke})
                    if smoke["status"] != "SMOKE_PASS":
                        result = {"schema": SCHEMA, "status": "INSTRUMENT_UNRESOLVED",
                                  "stage": "SMOKE_FAILED", "smoke": smoke,
                                  "development_stream_opened": False, "heldout_stream_opened": False,
                                  "model_loaded": False, "new_model_outputs": 0}
                    else:
                        result = run_calibration(block_ids, strata, handle)
                        result["smoke"] = smoke
                except Exception as exc:
                    result = {"schema": SCHEMA, "status": "INSTRUMENT_UNRESOLVED",
                              "fatal_error": {"type": type(exc).__name__, "message": str(exc)},
                              "settings": frozen_settings(), "model_loaded": False, "new_model_outputs": 0}
        try:
            if sha256_file(args.freeze) != initial_freeze_sha256:
                raise ValueError("freeze receipt changed during calibration")
            if sha256_file(args.selection) != initial_selection_sha256:
                raise ValueError("selected cohort changed during calibration")
            verify_freeze(args.freeze, args.selection, args.specification)
        except Exception as exc:
            result["status"] = "INSTRUMENT_UNRESOLVED"
            result["post_run_binding_failure"] = str(exc)
        result["records"] = {"path": str(records_path.resolve()), "sha256": sha256_file(records_path)}
        result["attempt_sha256"] = sha256_file(attempt_path)
        result["freeze_sha256"] = initial_freeze_sha256
        result["selection_sha256"] = initial_selection_sha256
    result["wall_seconds"] = time.monotonic() - started
    _exclusive_json(out, result)
    manifest = ExperimentManifest.create(
        "splicebert-register-operator-replication:statistical-calibration",
        [str(Path(__file__).resolve()), *(list(argv) if argv is not None else sys.argv[1:])],
        repo=REPO, status=result["status"], seeds=STREAMS,
        config={"numpy": np.__version__, "bit_generator": "PCG64", "smoke_only": args.smoke,
                "frozen_settings": frozen_settings()},
        notes=["No model is loaded; no new CC model outcome is computed.",
               "Synthetic calibration cannot establish actual CC/max variance or biological validity.",
               "Every stochastic realization is retained, including failures; outputs are exclusive-create."],
    )
    for path in inputs:
        manifest.add_input(path)
    manifest.add_output(out)
    if not args.smoke:
        manifest.add_output(attempt_path)
        manifest.add_output(records_path)
    _exclusive_json(manifest_path, manifest.to_dict())
    print(json.dumps({"status": result["status"], "out": str(out.resolve()),
                      "stochastic_calibration_executed": result.get("development_stream_opened")}), flush=True)
    return 0 if result["status"] in {"PASS", "SMOKE_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
