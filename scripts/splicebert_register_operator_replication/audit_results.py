#!/usr/bin/env python3
"""Independent raw-row result audit, including honest pre-target stopped runs.

This program imports neither the producer reducer nor its inference/classifier.
It never loads a model. New audit source is bound by its own immutable manifest;
the approved production freeze is checked, never changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
from itertools import permutations
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import numpy as np
from scipy.stats import beta, binom, spearmanr
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import ExperimentManifest

SCHEMA = "gfmcd.register_operator.result_audit.v1"
FREEZE_SHA256 = "82bc168e2246dff82519215203e02747f0209c289fa00c3e538613d95573d854"
SELECTION_IDS_SHA256 = "972d388cc031b6debdbb5b1c531afb7a8619630462f6bf4bb380d6a989417436"
TOL = 1e-10
FACTOR = 1.96 + 0.8416212335729143
SEEDS = tuple(range(2026080401, 2026080500))
DIRECTIONS = ("m0_minus_plus", "m0_minus_minus", "donor_phase_0_excess",
              "donor_phase_1_excess", "donor_phase_2_excess")
BRIDGES = ("psi_cc_minus_ac_max", "psi_cc_minus_ac_linear",
           "psi_max_minus_linear_cc", "psi_max_minus_linear_ac")
FAMILIES = ("primary_psi", "directions", "mismatch_equivalence", "bridge_diagnostics",
            "historical_psi_ac_minus_at_linear", "historical_phase_blind_ac_minus_at_linear")
GATES = ("construction", "source_edge_positive_control", "crossed_lattice_capability",
         "negative_control", "resolution", "corruption_robustness", "protected_mlm_health",
         "phase_blind_site_health", "numerics", "independent_audit")
POSITIVE = {"ONLY_WHEN_COHERENT_IN_FROZEN_DOMAIN", "MATERIAL_PREFERENTIAL_POSITIVE",
            "DETECTABLE_BOUNDED_SUB_SESOI", "DETECTABLE_PREFERENCE_MATERIALITY_UNRESOLVED"}
MATERIAL = {"ONLY_WHEN_COHERENT_IN_FROZEN_DOMAIN", "MATERIAL_PREFERENTIAL_POSITIVE"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: str | Path) -> dict:
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
        return json.load(handle)


def resolved(path: str | Path) -> Path:
    p = Path(path)
    return (p if p.is_absolute() else REPO / p).resolve()


def manifest_path(path: str | Path) -> Path:
    p = Path(path)
    return (p.with_suffix("") if p.suffix == ".gz" else p).with_suffix(".manifest.json")


def check_manifest(path: Path, *, required_inputs=()) -> dict:
    result = read(manifest_path(path))
    matched = [r for r in result.get("outputs", []) if resolved(r["path"]) == path.resolve()]
    require(len(matched) == 1 and matched[0]["sha256"] == digest(path),
            f"manifest output binding failed: {path}")
    inputs = {}
    for item in result.get("inputs", []):
        p = resolved(item["path"])
        require(p.is_file() and digest(p) == item["sha256"], f"manifest input changed: {p}")
        require(p.stat().st_size == item.get("size_bytes", p.stat().st_size), f"input size changed: {p}")
        inputs[str(p)] = item["sha256"]
    for p in required_inputs:
        require(inputs.get(str(Path(p).resolve())) == digest(p), f"manifest omits required input: {p}")
    return result


def verify_freeze(path: Path) -> dict:
    require(digest(path) == FREEZE_SHA256, "the approved production freeze hash differs")
    freeze = read(path)
    require(freeze["status"] == "FROZEN", "production is not frozen")
    for name, expected in freeze["files"].items():
        p = resolved(name)
        require(p.is_file() and digest(p) == expected, f"frozen bytes changed: {p}")
    runtime = freeze["runtime"]
    require(platform.python_version() == runtime["python"], "auditor Python differs from frozen runtime")
    for package, expected in runtime["versions"].items():
        require(version(package) == expected, f"installed {package} differs from frozen runtime")
    require(freeze["owner_choices"]["blocks"] == 342 and freeze["owner_choices"]["span_nt"] == 120,
            "approved cohort/span differs")
    require(freeze["exposure"] == {"new_cc_rows": 0, "new_model_loaded": False,
                                    "stochastic_calibration_started": False}, "freeze exposure differs")
    return freeze


def compare(actual: Any, expected: Any, path: str = "root", *, tolerance: float = TOL) -> dict:
    """Compare only independently reconstructed fields; missing fields fail closed."""
    stats = {"numeric_checks": 0, "maximum_absolute_error": 0.0}
    def visit(a, e, key):
        if isinstance(e, Mapping):
            require(isinstance(a, Mapping), f"{key}: expected mapping")
            for name, value in e.items():
                require(name in a, f"{key}: missing {name}")
                visit(a[name], value, key + "." + name)
        elif isinstance(e, (list, tuple, np.ndarray)):
            require(isinstance(a, (list, tuple, np.ndarray)) and len(a) == len(e), f"{key}: shape mismatch")
            for i, (av, ev) in enumerate(zip(a, e)):
                visit(av, ev, f"{key}[{i}]")
        elif isinstance(e, (bool, np.bool_)) or e is None or isinstance(e, str):
            require(type(a) is type(e) and a == e if not isinstance(e, np.bool_) else a == bool(e),
                    f"{key}: categorical value differs ({a!r} != {e!r})")
        elif isinstance(e, (float, int, np.number)):
            require(isinstance(a, (float, int, np.number)) and not isinstance(a, bool), f"{key}: expected numeric")
            error = abs(float(a) - float(e))
            require(np.isfinite([a, e]).all() and error <= tolerance, f"{key}: numeric difference {error}")
            stats["numeric_checks"] += 1
            stats["maximum_absolute_error"] = max(stats["maximum_absolute_error"], error)
        else:
            raise TypeError(f"unsupported independent comparison at {key}: {type(e)}")
    visit(actual, expected, path)
    return stats


def raw_scores(rows: list[dict], token_ids: dict) -> tuple[dict[str, dict], dict]:
    """Recompute CPU float32 full-vocabulary and A/C/G/T arithmetic from raw logits."""
    require(rows and len({r["request_id"] for r in rows}) == len(rows), "empty/duplicate raw requests")
    require(token_ids == {"A": 6, "C": 7, "G": 8, "T": 9, "MASK": 4, "CLS": 2, "SEP": 3},
            "raw tokenizer map differs")
    x = torch.tensor([r["masked_logits"] for r in rows], dtype=torch.float32)
    require(x.ndim == 2 and x.shape[1] == 10 and bool(torch.isfinite(x).all()), "nonfinite/malformed raw logits")
    bases = x[:, [6, 7, 8, 9]]
    target_ordinals = ["ACGT".index(r["true_base"]) for r in rows]
    target = torch.tensor([token_ids[r["true_base"]] for r in rows])
    actual = x[torch.arange(len(rows)), target]
    linear = actual - (bases.sum(dim=1) - actual) / 3
    rivals = bases.clone()
    rivals[torch.arange(len(rows)), target_ordinals] = -torch.inf
    maximum = actual - rivals.max(dim=1).values
    lp = torch.log_softmax(x, dim=1)[torch.arange(len(rows)), target]
    correct = x.argmax(dim=1) == target
    output, maxima = {}, {key: 0. for key in ("linear_margin", "max_margin", "true_base_log_probability")}
    for i, row in enumerate(rows):
        require(row["token_count"] == 1024 and row["token_position"] == row["position"] + 1
                and row["all_forward_logits_finite"] is True, "raw numerical/mask invariant failed")
        require(np.array_equal(np.asarray(row["base_logits"], dtype=np.float32), bases[i].numpy()),
                "recorded base logits do not match full logits")
        values = {"linear_margin": float(linear[i]), "max_margin": float(maximum[i]),
                  "true_base_log_probability": float(lp[i]), "correct": bool(correct[i])}
        for key in maxima:
            error = abs(values[key] - float(row[key]))
            maxima[key] = max(maxima[key], error)
            require(error <= TOL, f"raw {key} arithmetic differs: {row['request_id']}")
        require(values["correct"] == row["correct"], "raw correctness differs")
        output[row["request_id"]] = {**values, "true_base": row["true_base"]}
    return output, {"requests_checked": len(rows), "maximum_absolute_differences": maxima}


def reduce_requests(raw: dict, blocks: list[dict]) -> tuple[dict, dict]:
    """Enumerate protocol identities independently; never consume producer reductions."""
    by_id, arithmetic = raw_scores(raw["cell_requests"] + raw["native_requests"], raw["tokenizer_ids"])
    used = set()
    def row(key):
        require(key in by_id, f"missing protocol request: {key}")
        used.add(key)
        return by_id[key]
    arrays = {f"{op}_{margin}": [] for op in ("cc", "ac", "at") for margin in ("max", "linear")}
    sites = {f"{role}_{margin}": [] for role in ("bridge", "donor", "acceptor") for margin in ("max", "linear")}
    native = {f"{role}_{margin}": [] for role in ("cc", "ac", "donor", "acceptor") for margin in ("max", "linear")}
    health = {k: [] for k in ("logp_delta", "construct_correct", "source_correct", "probe_pairs")}
    for block in blocks:
        coupling = {key: np.zeros((3, 3)) for key in arrays}
        cell_sites = {key: [] for key in sites}
        protected = []
        for cell in sorted(block["cells"], key=lambda c: (c["p"], c["q"])):
            p, q = cell["p"], cell["q"]
            prefix = f"{block['block_id']}:p{p}:q{q}:"
            states = {key: row(prefix + key) for key in ("donor_intact", "donor_true_acceptor_cc",
                       "donor_true_acceptor_ac", "donor_true_acceptor_at", "donor_bridge_ac", "acceptor_intact")}
            require(all(v["true_base"] == "G" for v in states.values()), "margin target must be G")
            for margin in ("max", "linear"):
                field = margin + "_margin"
                intact = states["donor_intact"][field]
                for op in ("cc", "ac", "at"):
                    coupling[f"{op}_{margin}"][p, q] = intact - states["donor_true_acceptor_" + op][field]
                cell_sites["bridge_" + margin].append(intact - states["donor_bridge_ac"][field])
                cell_sites["donor_" + margin].append(intact)
                cell_sites["acceptor_" + margin].append(states["acceptor_intact"][field])
            require(len(cell["protected_probes"]) == 24, "protected probe count differs")
            for probe in cell["protected_probes"]:
                role, position = probe["role"], probe["position"]
                construct = row(prefix + f"health:{role}:{position}:construct")
                source = row(f"source:{role}:{probe['source_id']}:health:{position}")
                require(construct["true_base"] == source["true_base"] == probe["source_base"],
                        "protected truth correspondence differs")
                protected.append((construct["true_base_log_probability"] - source["true_base_log_probability"],
                                  int(construct["correct"]), int(source["correct"])))
        for key in arrays:
            arrays[key].append(coupling[key])
        for key in sites:
            sites[key].append(float(np.mean(cell_sites[key])))
        protected = np.asarray(protected)
        health["logp_delta"].append(float(protected[:, 0].mean()))
        health["construct_correct"].append(int(protected[:, 1].sum()))
        health["source_correct"].append(int(protected[:, 2].sum()))
        health["probe_pairs"].append(len(protected))
        per_source = {key: [] for key in native}
        for role in ("donor", "acceptor"):
            for phase in range(3):
                prefix = f"native:{block['block_id']}:{role}:p{phase}:{block[role + 's'][str(phase)]['source_id']}:"
                states = {key: row(prefix + key) for key in ("donor_intact", "donor_true_acceptor_cc",
                                                           "donor_true_acceptor_ac", "acceptor_intact")}
                for margin in ("max", "linear"):
                    field = margin + "_margin"
                    per_source["donor_" + margin].append(states["donor_intact"][field])
                    per_source["acceptor_" + margin].append(states["acceptor_intact"][field])
                    for op in ("cc", "ac"):
                        per_source[op + "_" + margin].append(states["donor_intact"][field] - states["donor_true_acceptor_" + op][field])
        for key in native:
            native[key].append(float(np.mean(per_source[key])))
    require(used == set(by_id), "unused raw requests remain after independent reduction")
    return {"block_ids": [b["block_id"] for b in blocks],
            "strata": [f"{b['chrom']}:{b['strand']}" for b in blocks],
            "matrices": {k: np.asarray(v) for k, v in arrays.items()},
            "native": {k: np.asarray(v) for k, v in native.items()},
            "sites": {k: np.asarray(v) for k, v in sites.items()},
            "health": {k: np.asarray(v) for k, v in health.items()}}, arithmetic


def bootstrap_indices(strata, *, draws=20000, seed=20260803):
    labels = np.asarray(strata)
    groups = [np.where(labels == name)[0] for name in sorted(set(strata))]
    require(all(len(group) >= 2 for group in groups), "singleton bootstrap stratum")
    rng = np.random.default_rng(seed)
    return np.hstack([rng.choice(group, (draws, len(group)), replace=True) for group in groups])


def infer(reduced: dict, *, prereqs: dict, numeric_passed: bool, audit_passed: bool,
          draws=20000, seed=20260803) -> dict:
    ids, strata = reduced["block_ids"], reduced["strata"]
    n = len(ids)
    samples = bootstrap_indices(strata, draws=draws, seed=seed)
    endpoint = {}
    for arm, matrices in reduced["matrices"].items():
        m0 = np.column_stack([matrices[:, p, p] for p in range(3)])
        mp = np.column_stack([matrices[:, p, (p + 1) % 3] for p in range(3)])
        mm = np.column_stack([matrices[:, p, (p + 2) % 3] for p in range(3)])
        e = {"m0": m0.mean(axis=1), "m_plus": mp.mean(axis=1), "m_minus": mm.mean(axis=1),
             "pooled": matrices.reshape(n, 9).mean(axis=1)}
        e["psi"] = e["m0"] - (e["m_plus"] + e["m_minus"]) / 2
        e["m0_minus_plus"], e["m0_minus_minus"] = e["m0"] - e["m_plus"], e["m0"] - e["m_minus"]
        for p in range(3):
            e[f"donor_phase_{p}_excess"] = m0[:, p] - (mp[:, p] + mm[:, p]) / 2
        endpoint[arm] = e
    def scalar(values):
        values = np.asarray(values)
        boot = values[samples].mean(axis=1)
        sd = float(boot.std(ddof=1))
        return {"mean": float(values.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
                "bootstrap_sd": sd, "mde80": FACTOR * sd}
    def family(values, names):
        values = np.asarray(values)
        means = values.mean(axis=0)
        # Chunk columns rather than allocate B*N*all-endpoints; independent indexed path.
        boot = np.column_stack([values[:, j][samples].mean(axis=1) for j in range(values.shape[1])])
        sd = boot.std(axis=0, ddof=1)
        active = sd > 0
        critical = float(np.quantile(np.max(np.abs((boot[:, active] - means[active]) / sd[active]), axis=1), .95)) if active.any() else 0.
        return {"names": list(names), "means": means.tolist(), "lower": (means - critical * sd).tolist(),
                "upper": (means + critical * sd).tolist(), "bootstrap_sd": sd.tolist()}
    cc = endpoint["cc_max"]
    primary = scalar(cc["psi"])
    centered = cc["psi"].copy()
    for label in sorted(set(strata)):
        take = np.asarray(strata) == label
        centered[take] -= centered[take].mean()
    primary["mde80"] = FACTOR * float(centered[samples].mean(axis=1).std(ddof=1))
    primary["analytic_empirical_mde80"] = FACTOR * float(np.sqrt(np.dot(centered, centered))) / n
    contrasts = [("cc_max", "ac_max"), ("cc_linear", "ac_linear"),
                 ("cc_max", "cc_linear"), ("ac_max", "ac_linear")]
    result = {"n_blocks": n, "n_strata": len(set(strata)), "bootstrap_draws": draws,
              "bootstrap_seed": seed, "primary_psi": primary,
              "directions": family(np.column_stack([cc[k] for k in DIRECTIONS]), DIRECTIONS),
              "mismatch_equivalence": family(np.column_stack((cc["m_plus"], cc["m_minus"])), ("m_plus", "m_minus")),
              "bridge_diagnostics": family(np.column_stack([endpoint[a]["psi"] - endpoint[b]["psi"] for a, b in contrasts]), BRIDGES)}
    for suffix, field, margin in (("psi_ac_minus_at_linear", "psi", "linear"),
                                  ("phase_blind_ac_minus_at_linear", "pooled", "linear"),
                                  ("phase_blind_ac_minus_at_max", "pooled", "max")):
        result["historical_" + suffix] = family((endpoint["ac_" + margin][field] - endpoint["at_" + margin][field])[:, None], [suffix])
    per_controls = []
    choices = list(permutations((0, 1, 2)))
    require(n % 6 == 0, "pseudo controls require six equal global permutation counts")
    for pseudo_seed in SEEDS:
        order = sorted(range(n), key=lambda i: (
            hashlib.sha256(f"crossed-phase-null-v1\0{pseudo_seed}\0{ids[i]}".encode()).hexdigest(), ids[i]))
        control = np.zeros(n)
        for rank, i in enumerate(order):
            mapping = choices[rank // (n // 6)]
            matrix = reduced["matrices"]["cc_max"][i]
            diagonal = [matrix[p, q] for p in range(3) for q in range(3) if mapping[q] == p]
            off = [matrix[p, q] for p in range(3) for q in range(3) if mapping[q] != p]
            control[i] = np.mean(diagonal) - np.mean(off)
        per_controls.append(control)
    per_controls = np.asarray(per_controls)
    controls = per_controls.mean(axis=1)
    paired = cc["psi"] - per_controls.mean(axis=0)
    result["pseudo_phase"] = {"seeds": list(SEEDS), "control_means": controls.tolist(),
        "maximum_control": float(controls.max()), "target_exceeds_all": bool(cc["psi"].mean() > controls.max()),
        "exact_add_one_one_sided_p": (1 + int(np.count_nonzero(controls >= cc["psi"].mean()))) / 100,
        "target_minus_mean_control": {"mean": float(paired.mean()), "lower95": float(np.quantile(paired[samples].mean(axis=1), .05))}}
    native = {key: scalar(value) for key, value in reduced["native"].items()}
    sites = {key: scalar(value) for key, value in reduced["sites"].items()}
    native_floors = {key: min(0., native[key]["ci95"][0]) for key in ("donor_max", "acceptor_max", "donor_linear", "acceptor_linear")}
    health = scalar(reduced["health"]["logp_delta"])
    counts = {key: int(reduced["health"][key].sum()) for key in ("construct_correct", "source_correct", "probe_pairs")}
    counts["retention_ratio"] = counts["construct_correct"] / counts["source_correct"] if counts["source_correct"] else None
    capability = scalar(cc["pooled"])
    result["instrument"] = {"native": native, "native_floors": native_floors, "sites": sites,
        "protected_logp_delta": health, "protected_accuracy": counts, "phase_blind_capability": capability}
    result["gates"] = {
        "construction": all(prereqs.values()), "source_edge_positive_control": native["cc_max"]["ci95"][0] > .05,
        "crossed_lattice_capability": capability["ci95"][0] > .02, "negative_control": within(sites["bridge_max"]),
        "resolution": primary["mde80"] <= .05,
        "corruption_robustness": within(result["historical_phase_blind_ac_minus_at_linear"]),
        "protected_mlm_health": health["ci95"][0] > -.10 and counts["retention_ratio"] is not None and counts["retention_ratio"] >= .80,
        "phase_blind_site_health": all(sites[key]["ci95"][0] > native_floors[key] for key in ("donor_max", "acceptor_max")),
        "numerics": bool(numeric_passed), "independent_audit": bool(audit_passed)}
    result["failed_gates"] = [key for key, value in result["gates"].items() if not value]
    result["arms"] = {arm: {"psi": scalar(e["psi"]), "matching_means": {k: float(e[k].mean()) for k in ("m0", "m_plus", "m_minus", "pooled")}}
                      for arm, e in endpoint.items()}
    result["matching_means"] = result["arms"]["cc_max"]["matching_means"]
    result["block_endpoints"] = [{"block_id": ids[i], "stratum": strata[i],
        **{arm: {key: float(value[i]) for key, value in e.items()} for arm, e in endpoint.items()}} for i in range(n)]
    result["classification"] = classify(result)
    return result


def bounds(summary):
    return (np.asarray(summary["ci95"])[:1], np.asarray(summary["ci95"])[1:]) if "ci95" in summary else (
        np.asarray(summary["lower"]), np.asarray(summary["upper"]))


def within(summary):
    low, high = bounds(summary)
    return bool(np.all(low >= -.05) and np.all(high <= .05))


def classify(result):
    if not all(v is True for v in result["gates"].values()):
        return "INSTRUMENT_UNRESOLVED"
    lower, upper = result["primary_psi"]["ci95"]
    pseudo = result["pseudo_phase"]
    coherent = (all(x > 0 for x in result["directions"]["lower"])
                and pseudo["target_exceeds_all"] is True and pseudo["exact_add_one_one_sided_p"] <= .01
                and pseudo["target_minus_mean_control"]["lower95"] > 0
                and within(result["historical_psi_ac_minus_at_linear"]))
    if lower > .05 and coherent:
        return "ONLY_WHEN_COHERENT_IN_FROZEN_DOMAIN" if within(result["mismatch_equivalence"]) else "MATERIAL_PREFERENTIAL_POSITIVE"
    if lower > 0 and coherent:
        return "DETECTABLE_BOUNDED_SUB_SESOI" if within(result["primary_psi"]) else "DETECTABLE_PREFERENCE_MATERIALITY_UNRESOLVED"
    if within(result["primary_psi"]):
        return "BOUNDED_NULL"
    if upper < -.05:
        return "MATERIAL_INVERTED"
    return "INCONCLUSIVE"


def verify_repeats(raw: dict) -> dict:
    report = {}
    for stage, prefix in (("cells", "cell"), ("native", "native")):
        first, repeated = raw[prefix + "_requests"][:16], raw[prefix + "_repeats"]
        require(len(first) == len(repeated) == 16, f"{stage}: repeat count differs")
        maxima = {key: 0. for key in ("linear_margin", "max_margin", "true_base_log_probability", "masked_logits")}
        for a, b in zip(first, repeated):
            for key in ("request_id", "sequence_sha256", "position", "token_position", "true_base",
                        "input_ids_sha256", "attention_mask_sha256", "token_count", "correct"):
                require(a[key] == b[key], f"{stage}: repeat identity/numeric category differs: {key}")
            for key in maxima:
                maxima[key] = max(maxima[key], float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key])))))
        require(max(maxima.values()) <= 1e-8, f"{stage}: repeat exceeds frozen tolerance")
        expected = {"maximum_absolute_differences": maxima, "accuracy_identical": True,
                    "maximum": max(maxima.values()), "threshold": 1e-8, "passed": True}
        compare(raw["determinism"][stage], expected, f"determinism.{stage}")
        report[stage] = expected
    return report


def stop_diagnostics(blocks: list[dict], psi: np.ndarray, fasta_path: Path | None = None) -> dict:
    """Rebuild integer stop matrices; optional FASTA independently rederives codons."""
    genome = None
    if fasta_path is not None:
        from pyfaidx import Fasta
        genome = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True, rebuild=False)
    complement = str.maketrans("ACGT", "TGCA")
    def fragment(item):
        sequence = str(genome[item["chrom"]][item["genomic_start"]:item["genomic_end"]])
        if item["strand"] == "-":
            sequence = sequence.translate(complement)[::-1]
        require(hashlib.sha256(sequence.encode()).hexdigest() == item["sequence_sha256"], "stop source fragment hash differs")
        return sequence
    matrices = []
    try:
        for block in blocks:
            matrix = np.zeros((3, 3), dtype=int)
            require(len(block["cells"]) == 9, "stop lattice incomplete")
            donor = [fragment(block["donors"][str(p)]["donor_fragment"]) for p in range(3)] if genome else None
            acceptor = [fragment(block["acceptors"][str(q)]["acceptor_fragment"]) for q in range(3)] if genome else None
            for cell in block["cells"]:
                p, q = cell["p"], cell["q"]
                stop = cell["junction_translation"]
                if genome:
                    sequence = donor[p][392 - p:392] + acceptor[q][64:100 - p]
                    require(len(sequence) == 36, "stop translation must contain twelve codons")
                    indices = [i + 1 for i in range(12) if sequence[3 * i:3 * i + 3] in {"TAA", "TAG", "TGA"}]
                    require(hashlib.sha256(sequence.encode()).hexdigest() == stop["source_spliced_window_sha256"], "junction translation hash differs")
                else:
                    indices = stop["stop_codon_indices_1based"]
                require(len(indices) == len(set(indices)) and all(1 <= k <= 12 for k in indices), "invalid stop positions")
                require(indices == stop["stop_codon_indices_1based"] and len(indices) == stop["stop_count"], "stop count/positions differ")
                matrix[p, q] = len(indices)
            compare(block["stop_diagnostics"]["stop_count_matrix"], matrix.tolist(), "stop count matrix", tolerance=0)
            matrices.append(matrix)
    finally:
        if genome:
            genome.close()
    counts = np.asarray(matrices)
    diag = np.trace(counts, axis1=1, axis2=2)
    omega = (counts.sum(axis=(1, 2)) - 3 * diag) / 6
    median = float(np.median(omega))
    association = spearmanr(omega, psi)
    require(np.isfinite([association.statistic, association.pvalue]).all(), "stop association is undefined")
    groups = {}
    for label, take in (("omega_above_frozen_median", omega > median),
                        ("omega_at_or_below_frozen_median", omega <= median)):
        require(take.any(), "empty frozen stop subgroup")
        groups[label] = {"n_blocks": int(take.sum()), "mean_psi": float(psi[take].mean())}
    chroms, strands = np.array([b["chrom"] for b in blocks]), np.array([b["strand"] for b in blocks])
    loo = (psi.sum() - psi) / (len(psi) - 1)
    return {"stop_diagnostic": {
        "selected_omega_median": median,
        "selected_omega_by_block": {b["block_id"]: float(omega[i]) for i, b in enumerate(blocks)},
        "mean_stop_count_matrix": counts.mean(axis=0).tolist(),
        "spearman_rho": float(association.statistic), "spearman_two_sided_p_descriptive": float(association.pvalue),
        "subgroups": groups},
        "sensitivity": {"leave_one_block": {"minimum": float(loo.min()), "maximum": float(loo.max())},
            "leave_one_chromosome": {str(c): float(psi[chroms != c].mean()) for c in sorted(set(chroms))},
            "leave_one_strand": {str(s): float(psi[strands != s].mean()) for s in sorted(set(strands))}}}


def binomial_record(k: int, n: int) -> dict:
    return {"count": k, "total": n, "rate": k / n,
        "ci95": [0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
                 1. if k == n else float(beta.ppf(.975, k+1, n-k))],
        "lower95_one_sided": 0. if k == 0 else float(beta.ppf(.05, k, n-k+1)),
        "upper95_one_sided": 1. if k == n else float(beta.ppf(.95, k+1, n-k))}


def synthetic_reduction(payload: dict) -> dict:
    """Independent dense fixture reduction, after exact raw-payload hash validation."""
    def margins(logits):
        x = torch.tensor(np.asarray(logits), dtype=torch.float32)
        require(x.shape[-1] == 4 and bool(torch.isfinite(x).all()), "invalid replay four-logit input")
        g = x[..., 2]
        return {"max": (g - x[..., [0, 1, 3]].max(dim=-1).values).numpy().astype(float),
                "linear": (g - (x.sum(dim=-1) - g) / 3).numpy().astype(float)}
    cell, native = margins(payload["cell_logits"]), margins(payload["native_logits"])
    n = len(payload["block_ids"])
    reduced = {"block_ids": payload["block_ids"], "strata": payload["strata"],
               "matrices": {}, "native": {}, "phase_blind": {}}
    for m in ("max", "linear"):
        for index, op in ((1, "cc"), (2, "ac"), (3, "at")):
            reduced["matrices"][op + "_" + m] = (cell[m][..., 0] - cell[m][..., index]).reshape(n, 3, 3)
        reduced["phase_blind"]["bridge_" + m] = (cell[m][..., 0] - cell[m][..., 4]).mean(axis=1)
        reduced["phase_blind"]["donor_" + m] = cell[m][..., 0].mean(axis=1)
        reduced["phase_blind"]["acceptor_" + m] = cell[m][..., 5].mean(axis=1)
        for index, op in ((1, "cc"), (2, "ac")):
            reduced["native"][op + "_" + m] = (native[m][..., 0] - native[m][..., index]).mean(axis=1)
        reduced["native"]["donor_" + m] = native[m][..., 0].mean(axis=1)
        reduced["native"]["acceptor_" + m] = native[m][..., 3].mean(axis=1)
    h = torch.tensor(np.asarray(payload["health_logits"]), dtype=torch.float32)
    require(np.all(np.asarray(payload["health_true_base_ids"]) == 2), "synthetic true-base plan differs")
    lp = torch.log_softmax(h, dim=-1)[..., 2].numpy().astype(float)
    correct = h.argmax(dim=-1).numpy() == 2
    reduced["health"] = {"logp_delta": (lp[..., 0]-lp[..., 1]).mean(axis=(1, 2)),
        "construct_correct": correct[..., 0].sum(axis=(1, 2)) * 9,
        "source_correct": correct[..., 1].sum(axis=(1, 2)) * 9, "probe_pairs": np.full(n, 216)}
    return reduced


def replay_interval_families(reduced: dict, samples: np.ndarray) -> dict:
    """Explicit cell contrasts and directly indexed bootstrap, for all null replays."""
    end = {}
    for name, x in reduced["matrices"].items():
        diagonal = np.column_stack([x[:, p, p] for p in range(3)])
        plus = np.column_stack([x[:, p, (p+1) % 3] for p in range(3)])
        minus = np.column_stack([x[:, p, (p+2) % 3] for p in range(3)])
        end[name] = {"psi": diagonal.mean(axis=1) - (plus.mean(axis=1)+minus.mean(axis=1))/2,
            "pool": x.mean(axis=(1, 2)), "directions": np.column_stack((diagonal.mean(axis=1)-plus.mean(axis=1),
                diagonal.mean(axis=1)-minus.mean(axis=1), diagonal-(plus+minus)/2)),
            "mismatch": np.column_stack((plus.mean(axis=1), minus.mean(axis=1)))}
    cc = end["cc_max"]
    psi_boot = cc["psi"][samples].mean(axis=1)
    output = {"primary_psi": {"mean": float(cc["psi"].mean()), "ci95": np.quantile(psi_boot, [.025,.975]).tolist(),
                               "bootstrap_sd": float(psi_boot.std(ddof=1))}}
    arrays = {"directions": cc["directions"], "mismatch_equivalence": cc["mismatch"],
        "bridge_diagnostics": np.column_stack([end[a]["psi"]-end[b]["psi"] for a,b in
            (("cc_max","ac_max"),("cc_linear","ac_linear"),("cc_max","cc_linear"),("ac_max","ac_linear"))]),
        "historical_psi_ac_minus_at_linear": (end["ac_linear"]["psi"]-end["at_linear"]["psi"])[:,None],
        "historical_phase_blind_ac_minus_at_linear": (end["ac_linear"]["pool"]-end["at_linear"]["pool"])[:,None]}
    for name, values in arrays.items():
        average = values.mean(axis=0)
        boot = np.column_stack([values[:, j][samples].mean(axis=1) for j in range(values.shape[1])])
        se = boot.std(axis=0, ddof=1)
        active = se > 0
        critical = float(np.quantile(np.abs((boot[:,active]-average[active])/se[active]).max(axis=1),.95)) if active.any() else 0.
        output[name] = {"means": average.tolist(), "lower": (average-critical*se).tolist(),
                        "upper": (average+critical*se).tolist(), "bootstrap_sd": se.tolist()}
    return output


def fixture_generator():
    # Only the hash-pinned raw fixture generator is reused. No producer measurement,
    # reduction, inference or classification entry point is called by this auditor.
    import importlib.util
    path = REPO / "scripts/splicebert_register_operator_replication/calibrate_statistics.py"
    spec = importlib.util.spec_from_file_location("audit_frozen_fixture_generator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recount_calibration(records_path: Path, result: dict, settings: dict, ids: list[str], *, replay_nulls=False) -> dict:
    """Recount immutable per-realization evidence, without trusting reported counts."""
    counts = Counter()
    classifications = {"development": Counter(), "null": Counter(), "point05": Counter(), "point15": Counter()}
    failures = {key: 0 for key in FAMILIES}
    positive = detection = material = 0
    seen, errors = set(), []
    header = smoke = None
    generator = fixture_generator() if replay_nulls else None
    samples = None
    replay = {"null_raw_payloads_hash_verified": 0, "raw_reductions_checked": 0,
              "interval_families_checked": 0, "maximum_absolute_reduction_error": 0.,
              "maximum_absolute_interval_error": 0., "direction_noncoverage_by_endpoint": {"additive": [0]*5, "heterogeneous": [0]*5},
              "no_new_seed_or_alternative_settings": True}
    subgroup = {"additive": {"n": 0, "positive": 0, "noncoverage": Counter()},
                "heterogeneous": {"n": 0, "positive": 0, "noncoverage": Counter()}}
    scale_diagnostics = {name: {"direction_means": [], "direction_bootstrap_sd": [],
        "primary_means": [], "primary_bootstrap_sd": [], "only_first_two_directions_fail": 0,
        "only_last_three_directions_fail": 0, "both_direction_groups_fail": 0} for name in subgroup}
    with gzip.open(records_path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") == "deterministic_smoke":
                require(smoke is None, "duplicate calibration smoke record")
                smoke = row["result"]
                require(smoke["status"] == "SMOKE_PASS" and not smoke["failures"], "calibration smoke failed")
                continue
            if row.get("record_type") == "calibration_header":
                require(header is None, "duplicate calibration header")
                header = row
                compare(row["settings"], settings, "calibration settings", tolerance=0)
                require(row["block_ids"] == ids, "calibration uses different selected blocks")
                if replay_nulls:
                    samples = bootstrap_indices(row["strata"])
                continue
            phase, index, delta = row["phase"], row["index"], row["delta"]
            require(phase in ("development", "null", "plants"), "unknown calibration phase")
            key = (phase, index, delta)
            require(key not in seen, "duplicate calibration realization")
            seen.add(key); counts[phase] += 1
            expected_seed = settings["streams"][phase]
            require(row["seed_sequence"] == {"entropy": expected_seed, "spawn_key": [index], "pool_size": 4,
                                             "bit_generator": "PCG64"}, "calibration seed identity differs")
            require(0 <= index < settings["repetitions"][phase], "calibration realization index outside plan")
            expected_delta = (0 if index % 4 < 2 else (.05 if index % 4 == 2 else .15)) if phase == "development" else (0 if phase == "null" else delta)
            require(delta == expected_delta and (phase != "plants" or delta in (.05, .15)), "calibration amplitude differs")
            if row["status"] == "ERROR":
                errors.append({k: row[k] for k in ("phase", "index", "delta", "error")})
                if phase == "null":
                    failures = {k: v+1 for k, v in failures.items()}
                continue
            require(row["status"] == "COMPLETE", "unknown calibration realization status")
            is_null = phase == "null" or (phase == "development" and index % 4 < 2)
            expected_truth = {"primary_psi": [0. if is_null else delta],
                "directions": [0. if is_null else delta] * 5,
                "mismatch_equivalence": [.10 if is_null else .05-delta/3] * 2,
                "bridge_diagnostics": [0.] * 4 if is_null else [0., 0., -.01, -.01],
                "historical_psi_ac_minus_at_linear": [0.], "historical_phase_blind_ac_minus_at_linear": [0.]}
            compare(row["fixture"]["population_truths"], expected_truth, "fixture generating truth", tolerance=0)
            analysis = row["analysis"]
            if phase == "null" and replay_nulls:
                require(samples is not None, "null replay requires preceding calibration header")
                seed = np.random.SeedSequence(row["seed_sequence"]["entropy"], spawn_key=tuple(row["seed_sequence"]["spawn_key"]),
                                              pool_size=row["seed_sequence"]["pool_size"])
                latent = generator.draw_latents(seed, len(ids))
                payload, _ = generator.make_payload(ids, header["strata"], latent, delta=0, null=True, additive_only=index < 500)
                raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
                                 default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()).encode()
                require(hashlib.sha256(raw).hexdigest() == row["raw_four_logit_payload_sha256"], "regenerated raw null payload hash differs")
                replay["null_raw_payloads_hash_verified"] += 1
                reconstructed = synthetic_reduction(payload)
                diff = compare(row["block_sufficient_statistics"], reconstructed, "independent null reduction")
                replay["maximum_absolute_reduction_error"] = max(replay["maximum_absolute_reduction_error"], diff["maximum_absolute_error"])
                replay["raw_reductions_checked"] += 1
                rebuilt_intervals = replay_interval_families(reconstructed, samples)
                diff = compare(analysis, rebuilt_intervals, "independent null inference")
                replay["maximum_absolute_interval_error"] = max(replay["maximum_absolute_interval_error"], diff["maximum_absolute_error"])
                replay["interval_families_checked"] += len(FAMILIES)
                if (index + 1) % 10 == 0:
                    print(json.dumps({"stage": "independent_null_replay", "completed": index+1,
                                      "maximum_interval_error": replay["maximum_absolute_interval_error"]}), flush=True)
            require(analysis["n_blocks"] == len(ids) and analysis["bootstrap_draws"] == 20000
                    and analysis["bootstrap_seed"] == 20260803 and analysis["pseudo_phase"]["seeds"] == list(SEEDS),
                    "calibration inference settings differ")
            require(row["block_sufficient_statistics"]["block_ids"] == ids, "calibration reduced block order differs")
            require(analysis["classification"] == classify(analysis), "calibration classifier differs from independent rule")
            noncoverage = {}
            for family in FAMILIES:
                lo, hi = bounds(analysis[family]); truth = np.asarray(expected_truth[family])
                require(lo.shape == hi.shape == truth.shape and np.isfinite([lo, hi]).all() and np.all(lo <= hi), "invalid calibration family interval")
                noncoverage[family] = bool(np.any(truth < lo-1e-10) or np.any(truth > hi+1e-10))
            require(noncoverage == row["family_noncoverage"], "recorded noncoverage differs from generating truth")
            is_positive = analysis["classification"] in POSITIVE
            is_material = analysis["classification"] in MATERIAL
            detects = analysis["primary_psi"]["ci95"][0] > 0
            require(row["positive_classification"] == is_positive and row["material_classification"] == is_material
                    and row["scalar_detection"] == detects, "calibration success indicators differ")
            label = phase if phase != "plants" else ("point05" if delta == .05 else "point15")
            classifications[label][analysis["classification"]] += 1
            if phase == "null":
                positive += int(is_positive)
                group = subgroup["additive" if index < 500 else "heterogeneous"]
                group["n"] += 1; group["positive"] += int(is_positive)
                for family in FAMILIES:
                    failures[family] += int(noncoverage[family]); group["noncoverage"][family] += int(noncoverage[family])
                lower, upper = bounds(analysis["directions"])
                direction_failures = ((lower > 1e-10) | (upper < -1e-10)).astype(int)
                group_name = "additive" if index < 500 else "heterogeneous"
                replay["direction_noncoverage_by_endpoint"][group_name] = (
                    np.asarray(replay["direction_noncoverage_by_endpoint"][group_name]) + direction_failures).tolist()
                scale = scale_diagnostics[group_name]
                first, last = bool(direction_failures[:2].any()), bool(direction_failures[2:].any())
                scale["only_first_two_directions_fail"] += int(first and not last)
                scale["only_last_three_directions_fail"] += int(last and not first)
                scale["both_direction_groups_fail"] += int(first and last)
                scale["direction_means"].append(analysis["directions"]["means"])
                scale["direction_bootstrap_sd"].append(analysis["directions"]["bootstrap_sd"])
                scale["primary_means"].append(analysis["primary_psi"]["mean"])
                scale["primary_bootstrap_sd"].append(analysis["primary_psi"]["bootstrap_sd"])
            elif phase == "plants":
                detection += int(detects) if delta == .05 else 0
                material += int(is_material) if delta == .15 else 0
    require(smoke is not None, "calibration lacks smoke evidence")
    complete = counts == {"development": 256, "null": 1000, "plants": 512}
    if complete:
        expected_keys = {(phase, i, d) for phase, total in (("development", 256), ("null", 1000), ("plants", 256))
                         for i in range(total) for d in ((.05, .15) if phase == "plants" else
                         ((0 if i % 4 < 2 else (.05 if i % 4 == 2 else .15)),) if phase == "development" else (0,))}
        require(seen == expected_keys, "calibration complete counts hide missing/extra realizations")
        expected = {"full_decision_false_positive": binomial_record(positive, 1000),
            "noncoverage_by_family": {k: binomial_record(v, 1000) for k, v in failures.items()},
            "point05_detection": binomial_record(detection, 256), "point15_material": binomial_record(material, 256),
            "execution_errors": errors}
        checks = {"no_execution_errors": not errors, "full_decision_false_positive_rate": positive <= 38,
                  **{"coverage:" + k: v <= 71 for k, v in failures.items()},
                  "detect_point05": detection >= 216, "material_point15": material >= 216}
        expected.update(checks=checks, all_passed=all(checks.values()),
                        failures=[k for k, v in checks.items() if not v])
        compare(result["acceptance"], expected, "calibration acceptance")
        require((result["status"] == "PASS") == all(checks.values()), "calibration reported status differs from recount")
    else:
        require(result["status"] != "PASS", "incomplete stochastic calibration cannot PASS")
    summarized_scales = {}
    for name, data in scale_diagnostics.items():
        if len(data["primary_means"]) < 2:
            continue
        summary = {k: v for k, v in data.items() if not isinstance(v, list)}
        for kind in ("primary", "direction"):
            estimates = np.asarray(data[kind + "_means"])
            sd = np.asarray(data[kind + "_bootstrap_sd"])
            variance = estimates.var(axis=0, ddof=1)
            summary[kind + "_empirical_sampling_sd"] = np.asarray(np.sqrt(variance)).tolist()
            summary[kind + "_root_mean_bootstrap_variance"] = np.asarray(np.sqrt(np.mean(sd**2, axis=0))).tolist()
            summary[kind + "_bootstrap_to_sampling_variance_ratio"] = np.asarray(np.mean(sd**2, axis=0) / variance).tolist()
        summarized_scales[name] = summary
    return {"complete": complete, "realization_counts": dict(counts), "execution_errors": errors,
            "positive_count": positive, "noncoverage_counts": failures, "point05_detection_count": detection,
            "point15_material_count": material, "null_generator_diagnostics": subgroup,
            "classification_counts": {k: dict(v) for k, v in classifications.items()},
            "passed": bool(complete and result["status"] == "PASS"),
            "null_replay": replay,
            "variance_scale_diagnostics": summarized_scales,
            "stratified_bootstrap_iid_variance_expectation": {
                "n": len(ids), "strata": len(set(header["strata"])) if header else None,
                "variance_ratio": (len(ids)-len(set(header["strata"]))) / len(ids) if header else None,
                "scope": "Equal-variance iid planning identity (N-H)/N; diagnostic, not a correction or proof of the observed family failure."},
            "scope": "Independent recount; when requested, every consumed null raw payload and all six interval families independently replayed at frozen settings."}


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "chronology timestamp lacks timezone")
    return parsed


def chronology(freeze: dict, receipts: dict, manifests: dict, paths: dict) -> dict:
    frozen = utc(freeze["frozen_at_utc"])
    report = {"frozen_at_utc": freeze["frozen_at_utc"], "stages": {}}
    completed = {k: utc(m["created_at_utc"]) for k, m in manifests.items()}
    for key, time in completed.items():
        if key != "construction":
            require(time >= frozen, f"{key} predates freeze")
    if "calibration" in receipts:
        attempt = Path(freeze["outputs"]["calibration_attempt"])
        require(attempt.is_file(), "calibration attempt missing")
        a = read(attempt)
        require(a["freeze_sha256"] == FREEZE_SHA256 and receipts["calibration"]["attempt_sha256"] == digest(attempt),
                "calibration attempt binding differs")
        require(a["selection_sha256"] == freeze["selection_sha256"] and a["frozen_files"] == freeze["files"],
                "calibration attempt source binding differs")
        timestamp = datetime.fromtimestamp(attempt.stat().st_mtime, timezone.utc)
        require(timestamp >= frozen and timestamp <= completed["calibration"], "calibration filesystem chronology differs")
        if "smoke" in completed:
            require(timestamp >= completed["smoke"], "calibration attempt precedes smoke completion")
        report["stages"]["calibration"] = {"attempt_sha256": digest(attempt),
            "attempt_start_timestamp_recorded": False, "filesystem_mtime_utc_corroboration": timestamp.isoformat(),
            "completed_manifest_utc": manifests["calibration"]["created_at_utc"],
            "limitation": "Attempt JSON omits UTC creation time; start order uses filesystem mtime as corroboration."}
    for stage in ("fidelity", "native", "cells"):
        attempt_path = Path(freeze["outputs"][stage + "_attempt"])
        if not attempt_path.exists():
            report["stages"][stage] = {"attempt_present": False}
            continue
        required = [paths[k] for k in ("smoke", "calibration") if k in receipts]
        if stage != "fidelity" and "fidelity" in receipts:
            required.append(paths["fidelity"])
        am = check_manifest(attempt_path, required_inputs=required)
        attempt = read(attempt_path)
        require(attempt["freeze_sha256"] == FREEZE_SHA256 and attempt["stage"] == stage, "model attempt differs")
        start = utc(attempt["created_at_utc"])
        require(start >= frozen, "model attempt predates freeze")
        for prerequisite in ("smoke", "calibration") + (() if stage == "fidelity" else ("fidelity",)):
            require(prerequisite in receipts and receipts[prerequisite]["status"] == "PASS",
                    f"{stage} attempted without {prerequisite} PASS")
            require(start >= completed[prerequisite], f"{stage} started before {prerequisite} completed")
        output_path = Path(freeze["outputs"][stage])
        if output_path.exists():
            raw = read(output_path)
            require(raw["attempt_sha256"] == digest(attempt_path), "raw model output has wrong attempt")
            require(utc(raw["completed_at_utc"]) >= start, "model completion predates attempt")
        report["stages"][stage] = {"attempt_present": True, "created_at_utc": attempt["created_at_utc"],
                                  "attempt_sha256": digest(attempt_path), "manifest_created_at_utc": am["created_at_utc"]}
    return report


def audit(freeze_path: Path, paths: dict[str, Path], *, measurement_audit: Path | None = None) -> dict:
    freeze = verify_freeze(freeze_path)
    package_path = Path(freeze["construction_path"])
    package = read(package_path)
    ids = sorted(package["selected_block_ids"])
    require(len(ids) == len(set(ids)) == 342, "selected block coverage differs")
    require(hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() == SELECTION_IDS_SHA256,
            "actual selected IDs differ from approved set")
    blocks = sorted((b for b in package["original_blocks"] if b["block_id"] in set(ids)), key=lambda b: b["block_id"])
    require(len(blocks) == 342, "selected construction blocks incomplete")
    receipts, manifests, prereqs = {}, {}, {}
    report = {"schema": SCHEMA, "status": "FAIL", "passed": False, "freeze_sha256": digest(freeze_path),
        "auditor_sha256": digest(__file__), "model_loaded": False, "new_model_forwards": 0,
        "selected_blocks": 342, "span_nt": 120, "cohort_tier": "outcome-aware operator verification",
        "prerequisites": {}, "failures": []}
    for name in ("construction", "smoke", "calibration", "fidelity"):
        path = paths[name]
        expected_path = Path(freeze["construction_audit_path"] if name == "construction" else
                             freeze["outputs"]["fidelity_audit" if name == "fidelity" else name])
        require(path.resolve() == expected_path.resolve(), f"{name} receipt path differs from freeze")
        if not path.exists():
            prereqs[name] = False
            report["prerequisites"][name] = {"state": "UNRUN", "path": str(path)}
            continue
        required = [package_path] if name == "construction" else [freeze_path]
        manifest = check_manifest(path, required_inputs=required)
        receipt = read(path)
        receipts[name], manifests[name] = receipt, manifest
        if name == "construction":
            passed = receipt.get("status") == "AUDIT_PASS" and receipt.get("passed") is True
            require(not receipt.get("failures") if passed else True, "construction PASS contains failures")
        else:
            require(receipt.get("freeze_sha256") == FREEZE_SHA256, f"{name} freeze binding differs")
            passed = receipt.get("status") == "PASS"
            if name == "smoke" and passed:
                require(receipt["tests"]["exit_code"] == 0 and receipt["statistical_memorials"]["status"] == "SMOKE_PASS"
                        and not receipt["statistical_memorials"]["failures"] and receipt["new_cc_outputs"] == 0,
                        "smoke PASS is inconsistent")
            if name == "fidelity" and passed:
                require(receipt.get("passed") is True and receipt.get("authorized_for_cc") is True,
                        "fidelity PASS lacks complete authorization")
                require(receipt["fidelity_sha256"] == digest(freeze["outputs"]["fidelity"]), "fidelity raw hash differs")
                require(receipt["auditor_sha256"] == freeze["files"]["scripts/splicebert_register_operator_replication/audit_fidelity.py"],
                        "fidelity auditor source differs")
            if name == "calibration":
                record_path = Path(freeze["outputs"]["calibration_records"])
                require(receipt["records"]["path"] == str(record_path) and receipt["records"]["sha256"] == digest(record_path),
                        "calibration record binding differs")
                record_binding = [r for r in manifest["outputs"] if resolved(r["path"]) == record_path.resolve()]
                require(len(record_binding) == 1 and record_binding[0]["sha256"] == digest(record_path),
                        "calibration manifest omits records")
                report["calibration_recount"] = recount_calibration(record_path, receipt, freeze["calibration"], ids, replay_nulls=True)
                passed = report["calibration_recount"]["passed"]
        prereqs[name] = passed
        report["prerequisites"][name] = {"state": "PASS" if passed else "FAIL", "path": str(path),
                                            "sha256": digest(path), "reported_status": receipt.get("status")}
    report["chronology"] = chronology(freeze, receipts, manifests, paths)
    available = {}
    for stage in ("fidelity", "native", "cells"):
        p = Path(freeze["outputs"][stage])
        if p.exists():
            r = read(p)
            available[stage] = {"status": r.get("status"), "cell_requests": len(r.get("cell_requests", [])),
                                "native_requests": len(r.get("native_requests", [])), "failure": r.get("failure")}
    report["observed_model_stages"] = available
    complete = all(prereqs.values()) and all(paths[k].exists() for k in ("cells", "native", "analysis"))
    if not complete:
        report.update(status="PASS", passed=True, stopped_run=True, model_evidence_complete=False,
                      authorized_for_final_result=True, classification="INSTRUMENT_UNRESOLVED",
                      authorized_for_cc=False,
                      gates={key: (all(prereqs.values()) if key == "construction" else True if key == "independent_audit" else None) for key in GATES})
        report["failed_gates"] = [key for key, value in report["gates"].items() if value is False]
        report["unrun_gates"] = [key for key, value in report["gates"].items() if value is None]
        report["reason"] = "A failed or unrun prerequisite/target stage prevents a complete CC inference; no primary estimate is invented."
        report["failures"] = [{"prerequisite": k, "state": v["state"],
            "recorded_failures": receipts[k].get("acceptance", {}).get("failures", receipts[k].get("failures", []))}
            for k, v in report["prerequisites"].items() if v["state"] == "FAIL"]
        require(digest(freeze_path) == FREEZE_SHA256, "freeze changed during audit")
        return report
    require(measurement_audit is not None and measurement_audit.is_file(), "complete result requires independent measurement audit")
    check_manifest(measurement_audit, required_inputs=[freeze_path, paths["native"], paths["cells"]])
    measurement = read(measurement_audit)
    require(measurement.get("status") == "PASS" and measurement.get("passed") is True
            and measurement.get("authorized_for_inference") is True, "independent measurement audit failed")
    require(measurement["freeze_sha256"] == FREEZE_SHA256 and measurement["native_sha256"] == digest(paths["native"])
            and measurement["cells_sha256"] == digest(paths["cells"]), "measurement audit input binding differs")
    require(measurement["auditor_sha256"] == digest(REPO / "scripts/splicebert_register_operator_replication/audit_cc_measurement.py"),
            "measurement auditor source changed")
    for key in ("native", "cells", "analysis"):
        require(paths[key].resolve() == Path(freeze["outputs"][key]).resolve(), f"{key} path differs from freeze")
        check_manifest(paths[key], required_inputs=[freeze_path])
    raw, native, produced = read(paths["cells"]), read(paths["native"]), read(paths["analysis"])
    require(raw["status"] == native["status"] == "COMPLETE", "incomplete raw model evidence")
    require(raw["freeze_sha256"] == native["freeze_sha256"] == produced["freeze_sha256"] == FREEZE_SHA256,
            "model/analysis freeze binding differs")
    require(raw["native_result_sha256"] == digest(paths["native"]) and raw["native_requests"] == native["native_requests"],
            "crossed evidence native copy differs")
    require(produced["raw_requests_sha256"] == digest(paths["cells"]), "analysis raw source binding differs")
    require(produced["status"] == "AWAITING_INDEPENDENT_AUDIT", "producer classification was not audit-pending")
    report["determinism"] = verify_repeats(raw)
    reduced, report["raw_arithmetic"] = reduce_requests(raw, blocks)
    reconstructed = infer(reduced, prereqs=prereqs, numeric_passed=True, audit_passed=False)
    psi = np.asarray([r["cc_max"]["psi"] for r in reconstructed["block_endpoints"]])
    diagnostics = stop_diagnostics(blocks, psi, Path(freeze["fasta_path"]))
    compare(package["stop_diagnostics"], {k: diagnostics["stop_diagnostic"][k]
            for k in ("selected_omega_median", "selected_omega_by_block", "mean_stop_count_matrix")}, "frozen stop diagnostics")
    reconstructed.update(diagnostics)
    report["producer_comparison"] = compare(produced, reconstructed, "producer analysis")
    # Gate 10 changes only after all independent data, arithmetic and provenance checks.
    reconstructed["gates"]["independent_audit"] = True
    reconstructed["failed_gates"] = [k for k, v in reconstructed["gates"].items() if not v]
    reconstructed["classification"] = classify(reconstructed)
    report.update(status="PASS", passed=True, stopped_run=False, model_evidence_complete=True,
                  authorized_for_final_result=True, classification=reconstructed["classification"],
                  gates=reconstructed["gates"], failed_gates=reconstructed["failed_gates"], unrun_gates=[],
                  recomputed=reconstructed, measurement_audit_sha256=digest(measurement_audit))
    require(digest(freeze_path) == FREEZE_SHA256, "freeze changed during independent audit")
    return report


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    for name in ("construction", "smoke", "fidelity", "calibration", "cells", "native", "analysis", "measurement-audit"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for p in (args.out, manifest_path(args.out)):
        require(not p.exists() and not p.is_symlink(), f"immutable audit output exists: {p}")
    freeze = read(args.freeze)
    paths = {key: getattr(args, key) or Path(freeze["construction_audit_path"] if key == "construction" else
             freeze["outputs"]["fidelity_audit" if key == "fidelity" else key])
             for key in ("construction", "smoke", "fidelity", "calibration", "cells", "native", "analysis")}
    try:
        result = audit(args.freeze, paths, measurement_audit=args.measurement_audit)
    except Exception as exc:
        result = {"schema": SCHEMA, "status": "FAIL", "passed": False,
                  "authorized_for_final_result": False, "classification": "INSTRUMENT_UNRESOLVED",
                  "freeze_sha256": digest(args.freeze), "auditor_sha256": digest(__file__),
                  "model_loaded": False, "new_model_forwards": 0,
                  "failures": [{"type": type(exc).__name__, "message": str(exc)}]}
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_exclusive(args.out, result)
    manifest = ExperimentManifest.create("splicebert-register-operator-replication:independent-result-audit",
        [str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])], repo=REPO,
        status=result["status"], seeds={"bootstrap": 20260803},
        config={"bootstrap_draws": 20000, "comparison_absolute_tolerance": TOL},
        notes=["Auditor imports neither producer measurement/reduction nor producer inference/classifier.",
               "A stopped-run PASS means an honest independent audit; it does not turn a failed preflight into model evidence."])
    inputs = [args.freeze, Path(__file__), REPO / "tests/test_register_operator_result_audit.py",
              Path(freeze["construction_path"]), Path(freeze["selection_path"])]
    inputs += [p for p in paths.values() if p.is_file()]
    inputs += [manifest_path(p) for p in paths.values() if manifest_path(p).is_file()]
    inputs += [Path(freeze["outputs"][k]) for k in ("calibration_attempt", "calibration_records", "fidelity_attempt", "native_attempt", "cells_attempt")
               if Path(freeze["outputs"][k]).is_file()]
    if args.measurement_audit:
        inputs += [args.measurement_audit, manifest_path(args.measurement_audit)]
    for p in dict.fromkeys(inputs):
        if p.is_file():
            manifest.add_input(p)
    manifest.add_output(args.out)
    write_exclusive(manifest_path(args.out), manifest.to_dict())
    print(json.dumps({"status": result["status"], "classification": result["classification"],
                      "stopped_run": result.get("stopped_run"), "out": str(args.out.resolve()),
                      "failures": result.get("failures", [])}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
