"""Frozen whole-block inference and classification for CC/max register verification."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import permutations

import numpy as np

MDE_FACTOR = 1.96 + 0.8416212335729143
PSEUDO_SEEDS = tuple(range(2026080401, 2026080500))
DIRECTIONS = ("m0_minus_plus", "m0_minus_minus", "donor_phase_0_excess",
              "donor_phase_1_excess", "donor_phase_2_excess")
BRIDGES = ("psi_cc_minus_ac_max", "psi_cc_minus_ac_linear",
           "psi_max_minus_linear_cc", "psi_max_minus_linear_ac")


def balanced_assignments(block_ids, seed: int) -> np.ndarray:
    if len(block_ids) % 6 or len(set(block_ids)) != len(block_ids) or not block_ids:
        raise ValueError("balanced controls need distinct IDs in a multiple of six")
    choices = tuple(permutations(range(3)))
    order = sorted(range(len(block_ids)), key=lambda i: (
        hashlib.sha256(f"crossed-phase-null-v1\0{seed}\0{block_ids[i]}".encode()).hexdigest(), block_ids[i]))
    assignments = np.zeros((len(block_ids), 3), dtype=int)
    for rank, i in enumerate(order):
        assignments[i] = choices[rank // (len(block_ids) // 6)]
    return assignments


def stratified_indices(strata, *, draws: int = 20000, seed: int = 20260803) -> np.ndarray:
    labels = np.asarray(strata, dtype=object)
    groups = [np.flatnonzero(labels == label) for label in sorted(set(strata))]
    if draws < 2 or not groups or any(len(g) < 2 for g in groups):
        raise ValueError("every stratum requires at least two independent blocks")
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(g, size=(draws, len(g)), replace=True) for g in groups], axis=1)


@dataclass
class InferenceContext:
    block_ids: tuple
    strata: tuple
    indices: np.ndarray
    weights: np.ndarray
    assignments: np.ndarray
    coefficients: np.ndarray
    draws: int
    seed: int
    pseudo_seeds: tuple


def make_inference_context(block_ids, strata, *, draws: int = 20000, seed: int = 20260803,
                           pseudo_seeds=PSEUDO_SEEDS) -> InferenceContext:
    if len(block_ids) != len(strata) or len(set(block_ids)) != len(block_ids):
        raise ValueError("invalid block identifiers/strata")
    indices = stratified_indices(strata, draws=draws, seed=seed)
    counts = np.zeros((draws, len(block_ids)), dtype=float)
    np.add.at(counts, (np.arange(draws)[:, None], indices), 1)
    assignments = np.asarray([balanced_assignments(block_ids, s) for s in pseudo_seeds])
    # pi maps ORIGINAL q to its permuted label; pi(q)==p defines a match.
    matches = assignments[:, :, None, :] == np.arange(3)[None, None, :, None]
    coefficients = np.where(matches, 1 / 3, -1 / 6)
    return InferenceContext(tuple(block_ids), tuple(strata), indices, counts / len(block_ids),
                            assignments, coefficients, draws, seed, tuple(pseudo_seeds))


def matching_endpoints(matrices) -> dict[str, np.ndarray]:
    array = np.asarray(matrices, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (3, 3) or not np.isfinite(array).all():
        raise ValueError("expected finite N by3 by3 lattice")
    p = np.arange(3)
    diagonal = array[:, p, p]
    plus = array[:, p, (p + 1) % 3]
    minus = array[:, p, (p - 1) % 3]
    m0, mp, mm = diagonal.mean(axis=1), plus.mean(axis=1), minus.mean(axis=1)
    result = {"m0": m0, "m_plus": mp, "m_minus": mm,
              "psi": m0 - .5 * (mp + mm), "m0_minus_plus": m0 - mp,
              "m0_minus_minus": m0 - mm, "pooled": array.mean(axis=(1, 2))}
    for i in range(3):
        result[f"donor_phase_{i}_excess"] = diagonal[:, i] - .5 * (plus[:, i] + minus[:, i])
    return result


def interval(values: np.ndarray, draws: np.ndarray) -> dict:
    mean = float(np.mean(values))
    sd = float(np.std(draws, ddof=1))
    return {"mean": mean, "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "bootstrap_sd": sd, "mde80": MDE_FACTOR * sd}


def simultaneous(values: np.ndarray, draws: np.ndarray, names: tuple | list) -> dict:
    means = values.mean(axis=0)
    sd = draws.std(axis=0, ddof=1)
    active = sd > 0
    critical = float(np.quantile(np.abs((draws[:, active] - means[active]) / sd[active]).max(axis=1), .95)) if active.any() else 0.0
    return {"names": list(names), "means": means.tolist(), "lower": (means - critical * sd).tolist(),
            "upper": (means + critical * sd).tolist(), "critical_value": critical,
            "bootstrap_sd": sd.tolist(), "method": "max-absolute-studentized whole-block bootstrap"}


def inside(row: dict, delta: float = .05) -> bool:
    if "ci95" in row:
        return bool(row["ci95"][0] >= -delta and row["ci95"][1] <= delta)
    return bool(all(v >= -delta for v in row["lower"]) and all(v <= delta for v in row["upper"]))


def classify(result: dict) -> str:
    if not all(value is True for value in result["gates"].values()):
        return "INSTRUMENT_UNRESOLVED"
    primary = result["primary_psi"]
    lower, upper = primary["ci95"]
    pseudo = result["pseudo_phase"]
    pattern = (all(v > 0 for v in result["directions"]["lower"])
               and pseudo["target_exceeds_all"] and pseudo["exact_add_one_one_sided_p"] <= .01
               and pseudo["target_minus_mean_control"]["lower95"] > 0
               and inside(result["historical_psi_ac_minus_at_linear"]))
    if lower > .05 and pattern:
        return "ONLY_WHEN_COHERENT_IN_FROZEN_DOMAIN" if inside(result["mismatch_equivalence"]) else "MATERIAL_PREFERENTIAL_POSITIVE"
    if lower > 0 and pattern:
        return "DETECTABLE_BOUNDED_SUB_SESOI" if inside(primary) else "DETECTABLE_PREFERENCE_MATERIALITY_UNRESOLVED"
    if inside(primary):
        return "BOUNDED_NULL"
    if upper < -.05:
        return "MATERIAL_INVERTED"
    return "INCONCLUSIVE"


def analyze(reduced: dict, *, context: InferenceContext | None = None) -> dict:
    ids, strata = reduced["block_ids"], reduced["strata"]
    context = context or make_inference_context(ids, strata)
    if tuple(ids) != context.block_ids or tuple(strata) != context.strata:
        raise ValueError("inference context does not match ordered blocks")
    endpoints = {key: matching_endpoints(value) for key, value in reduced["matrices"].items()}
    required = {f"{op}_{m}" for op in ("cc", "ac", "at") for m in ("max", "linear")}
    if set(endpoints) != required:
        raise ValueError("missing corruption/readout arm")
    cc = endpoints["cc_max"]
    arrays = {f"psi:{key}": item["psi"] for key, item in endpoints.items()}
    arrays.update({f"direction:{key}": cc[key] for key in DIRECTIONS})
    arrays.update({f"mismatch:{key}": cc[key] for key in ("m_plus", "m_minus")})
    arrays.update({"capability": cc["pooled"], "historical_phase_blind": endpoints["ac_linear"]["pooled"] - endpoints["at_linear"]["pooled"],
                   "historical_psi": endpoints["ac_linear"]["psi"] - endpoints["at_linear"]["psi"],
                   "historical_phase_blind_max": endpoints["ac_max"]["pooled"] - endpoints["at_max"]["pooled"],
                   "health": reduced["health"]["logp_delta"]})
    arrays.update({"native:" + k: v for k, v in reduced["native"].items()})
    arrays.update({"site:" + k: v for k, v in reduced["phase_blind"].items()})
    for key, left, right in zip(BRIDGES, ("cc_max", "cc_linear", "cc_max", "ac_max"),
                              ("ac_max", "ac_linear", "cc_linear", "ac_linear")):
        arrays[key] = endpoints[left]["psi"] - endpoints[right]["psi"]
    per_control = np.sum(context.coefficients * np.asarray(reduced["matrices"]["cc_max"])[None], axis=(2, 3))
    controls = per_control.mean(axis=1)
    arrays["paired_control"] = cc["psi"] - per_control.mean(axis=0)
    labels = np.asarray(strata, dtype=object)
    centered = cc["psi"].copy()
    for label in sorted(set(strata)):
        take = labels == label
        centered[take] -= centered[take].mean()
    arrays["centered"] = centered
    names = list(arrays)
    values = np.column_stack([arrays[k] for k in names])
    if values.shape[0] != len(ids) or not np.isfinite(values).all():
        raise ValueError("nonfinite or malformed inference inputs")
    # Counts contract the exact frozen resampling indices. Unit tests compare the
    # indexed mean path; only floating summation order differs, below1e-12.
    boot = context.weights @ values
    by_name = {name: i for i, name in enumerate(names)}

    def summary(name):
        i = by_name[name]
        return interval(values[:, i], boot[:, i])

    def family(keys, display):
        selected = [by_name[k] for k in keys]
        return simultaneous(values[:, selected], boot[:, selected], display)

    primary = summary("psi:cc_max")
    primary["mde80"] = float(MDE_FACTOR * boot[:, by_name["centered"]].std(ddof=1))
    primary["analytic_empirical_mde80"] = float(MDE_FACTOR * np.sqrt(np.sum(centered ** 2)) / len(ids))
    native = {key: summary("native:" + key) for key in reduced["native"]}
    sites = {key: summary("site:" + key) for key in reduced["phase_blind"]}
    native_floors = {key: min(0.0, native[key]["ci95"][0]) for key in ("donor_max", "acceptor_max", "donor_linear", "acceptor_linear")}
    health = summary("health")
    construct_correct = int(np.sum(reduced["health"]["construct_correct"]))
    source_correct = int(np.sum(reduced["health"]["source_correct"]))
    retention = None if source_correct == 0 else construct_correct / source_correct
    accuracy = {"construct_correct": construct_correct, "source_correct": source_correct,
                "probe_pairs": int(np.sum(reduced["health"]["probe_pairs"])), "retention_ratio": retention}
    historical_blind = family(["historical_phase_blind"], ["phase_blind_ac_minus_at_linear"])
    capability = summary("capability")
    prereqs = reduced["prerequisites"]
    numeric = reduced["numerics"]
    gates = {
        "construction": all(prereqs.get(k) is True for k in ("construction", "smoke", "fidelity", "calibration")),
        "source_edge_positive_control": native["cc_max"]["ci95"][0] > .05,
        "crossed_lattice_capability": capability["ci95"][0] > .02,
        "negative_control": inside(sites["bridge_max"]),
        "resolution": primary["mde80"] <= .05,
        "corruption_robustness": inside(historical_blind),
        "protected_mlm_health": health["ci95"][0] > -.10 and retention is not None and retention >= .80,
        "phase_blind_site_health": all(sites[k]["ci95"][0] > native_floors[k] for k in ("donor_max", "acceptor_max")),
        "numerics": numeric.get("all_finite") is True and numeric.get("token_count") == 1024
            and numeric.get("mask_mapping_valid") is True and numeric.get("repeat_max_abs", float("inf")) <= 1e-8,
        "independent_audit": prereqs.get("independent_audit") is True,
    }
    gates = {k: bool(v) for k, v in gates.items()}
    primary_mean = float(cc["psi"].mean())
    output = {"schema": "gfmcd.register_operator.analysis.v1", "synthetic": reduced["synthetic"],
        "n_blocks": len(ids), "n_strata": len(set(strata)), "bootstrap_draws": context.draws,
        "bootstrap_seed": context.seed, "primary_psi": primary,
        "directions": family(["direction:" + k for k in DIRECTIONS], DIRECTIONS),
        "mismatch_equivalence": family(["mismatch:m_plus", "mismatch:m_minus"], ["m_plus", "m_minus"]),
        "bridge_diagnostics": family(BRIDGES, BRIDGES),
        "historical_psi_ac_minus_at_linear": family(["historical_psi"], ["psi_ac_minus_at_linear"]),
        "historical_phase_blind_ac_minus_at_linear": historical_blind,
        "historical_phase_blind_ac_minus_at_max": family(["historical_phase_blind_max"], ["phase_blind_ac_minus_at_max"]),
        "arms": {key: {"psi": summary("psi:" + key),
            "matching_means": {k: float(row[k].mean()) for k in ("m0", "m_plus", "m_minus", "pooled")}}
            for key, row in endpoints.items()},
        "matching_means": {k: float(cc[k].mean()) for k in ("m0", "m_plus", "m_minus", "pooled")},
        "pseudo_phase": {"seeds": list(context.pseudo_seeds), "control_means": controls.tolist(),
            "maximum_control": float(controls.max()), "target_exceeds_all": primary_mean > float(controls.max()),
            "exact_add_one_one_sided_p": (1 + int((controls >= primary_mean).sum())) / (len(controls) + 1),
            "target_minus_mean_control": {"mean": float(arrays["paired_control"].mean()),
                "lower95": float(np.quantile(boot[:, by_name["paired_control"]], .05))}},
        "instrument": {"native": native, "native_floors": native_floors, "phase_blind_capability": capability,
            "sites": sites, "protected_logp_delta": health, "protected_accuracy": accuracy},
        "gates": gates, "failed_gates": [k for k, passed in gates.items() if not passed],
        "competitors": reduced.get("competitors", {}),
        "block_endpoints": [{"block_id": ids[i], "stratum": strata[i],
            **{key: {k: float(v[i]) for k, v in row.items()} for key, row in endpoints.items()}}
            for i in range(len(ids))]}
    output["classification"] = classify(output)
    return output
