#!/usr/bin/env python3
"""Independent, RNG-free conformance audit of the single A1 inference repair.

This gate compares calibration ASTs and frozen bytes, then supplies exhaustive
deterministic resamples to the production pure correction. It never calls a
random-number generator, a model, or the consumed-stream development runner.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import gzip
import hashlib
import importlib
from itertools import product
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402

BASE_SHA = "82bc168e2246dff82519215203e02747f0209c289fa00c3e538613d95573d854"
CALIBRATION_SHA = "4c13d30ee8dd79a1c16e52ced24f955fec7c8285b3bf2cd4400a776bf5959ee7"
PRESERVED_SHA = {
    "smoke_v1.json": "a56c986f3472588eb40c30c5114cf3ea9669734c75f389e79d60b6745128b5df",
    "statistical_calibration_v1.json": CALIBRATION_SHA,
    "analysis_v1.audit.json": "91bda8a6e3cced9f708a0b468e4c2af79f114368a1b45f7f84d29eb8eddbcb65",
    "result_v1.json": "1cd6df321679fb649bce5d49562a7a44b9c637c33d71e25ed7fd01e317b1dfc3",
}
METHOD = "finite-stratum-centered-residual-bootstrap-v2"
IDS_SHA = "972d388cc031b6debdbb5b1c531afb7a8619630462f6bf4bb380d6a989417436"
STREAMS = {"development": 2026101604, "null": 2026101605, "plants": 2026101606}
STUDY = Path("studies/splicebert_register_operator_replication")
SCRIPTS = Path("scripts/splicebert_register_operator_replication")
RESULTS = Path("results/splicebert_register_operator_replication")
GENERATOR_DEFINITIONS = (
    "Latents", "draw_latents", "deterministic_latents", "fixed_tie_membership",
    "_four_logits", "make_payload", "canonical_bytes", "jsonable", "load_selection",
    "analyze_payload", "interval_arrays", "family_noncoverage", "_seed_record",
    "_write_record", "_evaluate", "evaluate_acceptance", "exact_binomial_interval",
    "binomial_record", "run_calibration",
)
UNCHANGED_GLOBALS = (
    "PAYLOAD_SCHEMA", "MDE_FACTOR", "BOOTSTRAP_DRAWS", "BOOTSTRAP_SEED",
    "PSEUDO_SEEDS", "REPETITIONS", "EXPECTED_N", "SELECTION_IDS_SHA256", "FPR_MAX",
    "NONCOVERAGE_MAX", "PLANT_SUCCESS_MIN", "TRUTH_TOLERANCE", "FAMILIES",
    "POSITIVE_CLASSES", "MATERIAL_CLASSES", "CELL_STATES", "NATIVE_STATES", "H",
)
REQUIRED_ADDITIVE = (
    "src/gfmcd/splicing/register_operator_statistics_v2.py",
    "src/gfmcd/splicing/register_operator_io_v2.py",
    *(str(SCRIPTS / name) for name in (
        "calibrate_statistics_v2.py", "develop_repair.py", "freeze_protocol_v2.py",
        "run_smoke_v2.py", "run_model_v2.py", "analyze_results_v2.py",
        "audit_cc_measurement_v2.py", "audit_results_v2.py", "record_result_v2.py", "audit_repair_design.py")),
    "tests/test_register_operator_statistics_v2.py",
    "tests/test_register_operator_calibration_v2.py",
    "tests/test_register_operator_repair_design.py",
)


class DesignFailure(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DesignFailure(message)


def read_json(path: Path) -> dict:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def definitions(tree: ast.Module) -> dict[str, ast.AST]:
    return {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def assignments(tree: ast.Module) -> dict[str, ast.AST]:
    return {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)}


def dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def resolved(path: str | Path, repo: Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else repo / path).resolve()


def source_closure(initial, repo: Path) -> set[Path]:
    """Independently discover local Python dependencies without importing them."""
    found = {Path(path).resolve() for path in initial}
    pending = list(found)
    while pending:
        path = pending.pop()
        if path.suffix != ".py":
            continue
        names = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
                names.extend(node.module + "." + alias.name for alias in node.names)
        for name in names:
            if name.startswith("gfmcd"):
                base = repo / "src" / name.replace(".", "/")
                candidates = [base.with_suffix(".py"), base / "__init__.py"]
                candidates.extend(parent / "__init__.py" for parent in base.parents
                                  if parent.is_relative_to(repo / "src"))
            else:
                base = path.parent / name.replace(".", "/")
                candidates = [base.with_suffix(".py"), base / "__init__.py"]
            for candidate in candidates:
                if candidate.is_file() and candidate.resolve() not in found:
                    found.add(candidate.resolve())
                    pending.append(candidate.resolve())
    return found


def verify_manifest_evidence(artifact: Path, repo: Path, *, status: str,
                             required_inputs=()) -> tuple[dict, set[Path]]:
    """Verify every recorded input/output, including stream and attempt outputs."""
    artifact = resolved(artifact, repo)
    stem = artifact.with_suffix("") if artifact.suffix == ".gz" else artifact
    manifest_path = stem.with_suffix(".manifest.json")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == status, f"manifest status differs: {artifact}")
    output_matches = [r for r in manifest.get("outputs", []) if resolved(r["path"], repo) == artifact]
    require(len(output_matches) == 1, f"manifest does not uniquely bind artifact: {artifact}")
    paths = {artifact, manifest_path}
    for role in ("inputs", "outputs"):
        seen = set()
        for record in manifest.get(role, []):
            path = resolved(record["path"], repo)
            require(path not in seen, f"duplicate manifest {role}: {path}")
            seen.add(path)
            require(path.is_file() and sha256_file(path) == record["sha256"],
                    f"manifest evidence changed: {path}")
            require(path.stat().st_size == record.get("size_bytes", path.stat().st_size),
                    f"manifest evidence size changed: {path}")
            paths.add(path)
        if role == "inputs":
            require(all(resolved(p, repo) in seen for p in required_inputs),
                    f"manifest omits required input: {artifact}")
    return manifest, paths


def verify_generator_identity(v1_source: str, v2_source: str) -> dict:
    """The actual generator, scheduling, truth and acceptance ASTs must be exact."""
    old, new = ast.parse(v1_source), ast.parse(v2_source)
    old_defs, new_defs = definitions(old), definitions(new)
    for name in GENERATOR_DEFINITIONS:
        require(name in new_defs and dump(old_defs[name]) == dump(new_defs[name]),
                f"calibration definition changed: {name}")
    old_globals, new_globals = assignments(old), assignments(new)
    for name in UNCHANGED_GLOBALS:
        require(name in new_globals and dump(old_globals[name]) == dump(new_globals[name]),
                f"calibration constant changed: {name}")
    require(ast.literal_eval(new_globals["STREAMS"]) == STREAMS, "fresh roots differ")
    require(ast.literal_eval(new_globals["SCHEMA"]) == "gfmcd.register_operator.statistical_calibration.v2",
            "calibration schema is not v2")
    return {"identical_definitions": list(GENERATOR_DEFINITIONS),
            "identical_globals": list(UNCHANGED_GLOBALS), "fresh_streams": STREAMS,
            "definition_ast_sha256": {name: hashlib.sha256(dump(new_defs[name]).encode()).hexdigest()
                                      for name in GENERATOR_DEFINITIONS}}


def verify_source_contract(repo: Path) -> dict:
    """Check version barriers and scope without calling a runner or its RNG."""
    trees = {name: ast.parse((repo / SCRIPTS / name).read_text()) for name in (
        "calibrate_statistics_v2.py", "run_model.py", "run_model_v2.py", "analyze_results_v2.py",
        "audit_fidelity.py", "develop_repair.py", "freeze_protocol_v2.py", "run_smoke_v2.py")}
    io = ast.parse((repo / "src/gfmcd/splicing/register_operator_io_v2.py").read_text())
    verifier = definitions(io)["verify_freeze"]
    first_if = next(node for node in verifier.body if isinstance(node, ast.If))
    condition = ast.unparse(first_if.test)
    require(all(value in condition for value in ("FROZEN_V2", "gfmcd.register_operator.freeze.v2", "METHOD", "BASE_FREEZE_SHA256"))
            and any(isinstance(node, ast.Raise) for node in ast.walk(first_if)), "v2 version barrier weakened")
    require(ast.literal_eval(assignments(io)["BASE_FREEZE_SHA256"]) == BASE_SHA, "v2 parent pin changed")
    old_io = ast.parse((repo / "src/gfmcd/splicing/register_operator_io.py").read_text())
    old_verify = definitions(old_io)["verify_freeze"]
    old_if = next(node for node in old_verify.body if isinstance(node, ast.If))
    require("'FROZEN'" in ast.unparse(old_if.test) and "FROZEN_V2" not in ast.unparse(old_if.test)
            and any(isinstance(node, ast.Raise) for node in ast.walk(old_if)), "v1 no longer rejects v2 freeze")
    native = definitions(trees["run_model_v2.py"])["native_preflight"]
    native_imports = [node for node in ast.walk(native) if isinstance(node, ast.ImportFrom)]
    require(any(node.module == "gfmcd.splicing.register_operator_statistics_v2"
                and {alias.name for alias in node.names} >= {"make_inference_context", "interval"}
                for node in native_imports), "native preflight does not use corrected context")
    require("context.weights" in ast.unparse(native) and "CorrectedInferenceContext" in ast.unparse(native),
            "native preflight bypasses the corrected context")
    for name in ("run_model_v2.py", "analyze_results_v2.py"):
        require(any(isinstance(node, ast.ImportFrom) and node.module == "gfmcd.splicing.register_operator_io_v2"
                    and any(alias.name == "verify_freeze" for alias in node.names)
                    for node in trees[name].body), f"{name}: missing v2 barrier")
    require(any(isinstance(node, ast.ImportFrom) and node.module == "gfmcd.splicing.register_operator_statistics_v2"
                and any(alias.name == "analyze" for alias in node.names)
                for node in trees["analyze_results_v2.py"].body), "target analysis uses wrong method")
    old_main = definitions(trees["run_model.py"])["main"]
    new_main = definitions(trees["run_model_v2.py"])["main"]
    new_planner = definitions(trees["run_model_v2.py"])["prepare_requests"]
    class NormalizePlanner(ast.NodeTransformer):
        def __init__(self, *, historical):
            self.historical = historical
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "args" and node.attr == "stage":
                return ast.copy_location(ast.Name(id="requested_stage", ctx=ast.Load()), node)
            return self.generic_visit(node)
        def visit_Name(self, node):
            # The v1 main used an alias for the original cohort. The v2 helper
            # spells it inline and renames the comprehension's local variable.
            if self.historical and node.id == "original":
                return ast.parse('package["original_blocks"]', mode="eval").body
            mapping = ({"stage": "request_kind"} if self.historical else
                       {"stage": "requested_stage", "name": "request_kind"})
            return ast.copy_location(ast.Name(id=mapping.get(node.id, node.id), ctx=node.ctx), node)
    def named_assignment(function, name, *, historical):
        matches = [node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
        require(len(matches) == 1, f"runtime assignment missing or ambiguous: {name}")
        return dump(NormalizePlanner(historical=historical).visit(copy.deepcopy(matches[0])))
    require(named_assignment(old_main, "original", historical=True) ==
            dump(ast.parse('package["original_blocks"]', mode="eval").body), "historical cohort alias changed")
    for name in ("selected", "blocks", "stages", "plans", "expected"):
        require(named_assignment(old_main, name, historical=True) == named_assignment(new_planner, name, historical=False),
                f"historical/new request branch changed: {name}")
    require(any(isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value) == "prepare_requests(package, genome, args.stage)"
                and len(node.targets) == 1 and ast.unparse(node.targets[0]) == "(blocks, plans)"
                for node in ast.walk(new_main)), "runtime bypasses the reviewed request planner")
    require("register_operator_statistics_v2" not in ast.unparse(trees["audit_fidelity.py"]),
            "historical auditor imports repaired inference")
    calibration_main = definitions(trees["calibrate_statistics_v2.py"])["main"]
    calls = [(node.lineno, ast.unparse(node.func)) for node in ast.walk(calibration_main) if isinstance(node, ast.Call)]
    def first_call(name):
        lines = [line for line, function in calls if function == name]
        require(bool(lines), f"fresh calibration missing gate call: {name}")
        return min(lines)
    first_fresh = first_call("run_calibration")
    for call in ("verify_freeze", "verify_output_bindings", "verify_artifact_manifest", "require_manifest_input", "write_artifact"):
        require(first_call(call) < first_fresh, f"fresh stream precedes gate: {call}")
    require(any(isinstance(node, ast.If) and "smoke_receipt" in ast.unparse(node.test)
                and "PASS" in ast.unparse(node.test) and "METHOD" in ast.unparse(node.test)
                and any(isinstance(child, ast.Raise) for child in node.body)
                for node in ast.walk(calibration_main)), "fresh calibration does not require smoke PASS")
    development = definitions(trees["develop_repair.py"])["main"]
    require(not any(isinstance(node, ast.Call) and any(token in ast.unparse(node.func)
                    for token in ("SeedSequence", "default_rng", "draw_latents")) for node in ast.walk(development)),
            "consumed-stream development generates fresh payloads")
    stats = ast.parse((repo / "src/gfmcd/splicing/register_operator_statistics_v2.py").read_text())
    wrapper = definitions(stats)["make_inference_context"]
    require(dict(zip([arg.arg for arg in wrapper.args.kwonlyargs],
                     [ast.unparse(value) for value in wrapper.args.kw_defaults])) ==
            {"draws": "20000", "seed": "20260803", "pseudo_seeds": "PSEUDO_SEEDS"},
            "corrected context defaults changed")
    require(any(isinstance(node, ast.Call) and ast.unparse(node) ==
                "original.make_inference_context(block_ids, strata, draws=draws, seed=seed, pseudo_seeds=pseudo_seeds)"
                for node in ast.walk(wrapper)), "corrected context changed whole-block resampling")
    require(isinstance(wrapper.body[-1], ast.Return) and ast.unparse(wrapper.body[-1].value) ==
            "correct_inference_context(base)", "corrected context bypasses pure correction")
    for name in ("MDE_FACTOR", "PSEUDO_SEEDS", "DIRECTIONS", "BRIDGES", "interval", "simultaneous",
                 "classify", "inside", "matching_endpoints", "balanced_assignments", "stratified_indices"):
        require(ast.unparse(assignments(stats)[name]) == "original." + name,
                f"unchanged inference or decision helper replaced: {name}")
    receipt = definitions(trees["freeze_protocol_v2.py"])["validate_receipt"]
    receipt_text = ast.unparse(receipt)
    require(all(term in receipt_text for term in ("verify_artifact_manifest(path)", "receipt.get('passed') is not True",
                "receipt.get('status') != 'PASS'", "manifest.get('status') != 'PASS'", "BASE_FREEZE_SHA256",
                "receipt.get('inference_method') != METHOD", "sha256_file(auditor)",
                "require_manifest_input(manifest, auditor)", "require_manifest_input(manifest, p)")),
            "freeze does not bind PASS receipt and reviewed auditor/source inputs")
    freezer_text = ast.unparse(definitions(trees["freeze_protocol_v2.py"])["main"])
    require(all(term in freezer_text for term in ("validate_receipt(args.design_audit, auditor=auditor)",
                "validate_receipt(args.development)", "point_and_pseudo_memorials_passed", "1768",
                "development.get('fresh_streams_opened') is not False", "'null': 1000", "'plants': 512",
                "require_manifest_input(design_manifest, source)", "source_closure(", "deepcopy(parent)",
                "status='FROZEN_V2'", "sha256_file(path) != record['sha256']")),
            "freeze omits development, source closure, or preserved evidence gate")
    return {"v1_rejects_v2": True, "v2_requires_explicit_method_and_parent": True,
            "native_and_target_share_corrected_context": True, "historical_request_branches_unchanged": True,
            "fresh_calibration_requires_freeze_smoke_and_attempt": True,
            "consumed_stream_runner_does_not_generate_payloads": True,
            "same_resampling_defaults_and_original_classifier": True,
            "freeze_requires_source_bound_design_and_complete_development": True}


def _near(actual, expected, label: str, *, atol=2e-12) -> float:
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all(), f"{label}: shape or finite")
    difference = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    require(difference <= atol, f"{label}: maximum discrepancy {difference}")
    return difference


def enumerated_context(original):
    """All 2^2 × 4^4 resamples of unequal strata; no RNG is involved."""
    ids = tuple(f"audit:b{i}" for i in range(6))
    strata = ("chr1:+",) * 2 + ("chr2:-",) * 4
    indices = np.array([left + right for left in product(range(2), repeat=2)
                        for right in product(range(2, 6), repeat=4)], dtype=int)
    counts = np.array([[row.count(i) for i in range(6)] for row in indices.tolist()], dtype=float)
    seeds = tuple(range(2026080401, 2026080500))
    assignment = np.array([original.balanced_assignments(ids, seed) for seed in seeds])
    matches = assignment[:, :, None, :] == np.arange(3)[None, None, :, None]
    coefficients = np.where(matches, 1 / 3, -1 / 6)
    return original.InferenceContext(ids, strata, indices, counts / 6, assignment,
                                     coefficients, len(indices), 20260803, seeds)


def independent_transform(values, strata):
    x = np.asarray(values, dtype=float)
    transformed = x.copy()
    for label in sorted(set(strata)):
        members = [i for i, current in enumerate(strata) if current == label]
        center = np.mean(x[members], axis=0)
        transformed[members] = center + math.sqrt(len(members) / (len(members) - 1)) * (x[members] - center)
    return transformed


def deterministic_reduced(base):
    block = np.arange(6, dtype=float)[:, None, None]
    row = np.arange(3, dtype=float)[None, :, None]
    column = np.arange(3, dtype=float)[None, None, :]
    matrices = {}
    for index, key in enumerate(f"{op}_{margin}" for op in ("cc", "ac", "at") for margin in ("max", "linear")):
        matrices[key] = (.1 + .007 * (block + 1) ** 2 + .013 * row - .021 * column
                         + .004 * (index + 1) * (block + 1) * (1 + row * column)
                         + .002 * (block % 3) * (row - column) ** 2)
    native = {key: .3 + .003 * (i + 1) * np.arange(6) ** 2 for i, key in enumerate(
        ("cc_max", "ac_max", "donor_max", "acceptor_max", "cc_linear", "ac_linear", "donor_linear", "acceptor_linear"))}
    sites = {key: .1 + .002 * (i + 1) * np.arange(6) ** 2 for i, key in enumerate(
        ("donor_max", "acceptor_max", "bridge_max", "donor_linear", "acceptor_linear", "bridge_linear"))}
    return {"block_ids": list(base.block_ids), "strata": list(base.strata), "synthetic": True,
            "matrices": matrices, "native": native, "phase_blind": sites,
            "health": {"logp_delta": np.arange(6) ** 2 / 1000,
                       "construct_correct": np.full(6, 216), "source_correct": np.full(6, 216),
                       "probe_pairs": np.full(6, 216)},
            "prerequisites": {key: True for key in ("construction", "smoke", "fidelity", "calibration", "independent_audit")},
            "numerics": {"all_finite": True, "token_count": 1024, "mask_mapping_valid": True, "repeat_max_abs": 0.0}}


def verify_numeric_repair(revised) -> dict:
    """Verify actual pure correction and analysis against independent enumeration."""
    original = revised.original
    base = enumerated_context(original)
    base_before = {key: value.copy() for key, value in vars(base).items() if isinstance(value, np.ndarray)}
    corrected = revised.correct_inference_context(base)
    expected_scales = np.array([math.sqrt(2), math.sqrt(2), *([math.sqrt(4 / 3)] * 4)])
    _near(corrected.residual_scales, expected_scales, "per-stratum scales")
    _near(corrected.weights, base.weights * expected_scales + (1 - expected_scales) / 6,
          "affine coefficients")
    require(np.min(corrected.weights) < 0, "residual coefficients incorrectly clipped")
    for name in ("indices", "assignments", "coefficients"):
        require(np.array_equal(getattr(corrected, name), getattr(base, name)), f"correction changed {name}")
    x = np.array([[1., 4., -2.], [3., 1., 5.], [10., -4., 2.], [12., 2., 7.], [16., 3., -1.], [21., 8., 6.]])
    y = independent_transform(x, base.strata)
    _near(revised.corrected_values(x, base.strata), y, "direct residual map")
    expected_draws = y[base.indices].mean(axis=1)
    maximum = _near(corrected.weights @ x, expected_draws, "transformed indexed means")
    expected_covariance = np.zeros((3, 3))
    for members in ([0, 1], [2, 3, 4, 5]):
        residual = x[members] - x[members].mean(axis=0)
        expected_covariance += len(members) * (residual.T @ residual) / (len(members) - 1) / 36
        _near(y[members].mean(axis=0), x[members].mean(axis=0), "stratum point mean")
        _near(corrected.weights[:, members].sum(axis=1), np.full(len(base.indices), len(members) / 6), "replicate stratum weight")
    maximum = max(maximum, _near(np.cov(expected_draws, rowvar=False, ddof=0), expected_covariance,
                                  "exact conditional covariance"))
    _near(corrected.weights @ np.ones(6), np.ones(len(base.indices)), "constant-vector preservation")
    _near(revised.corrected_values(x + 7, base.strata), y + 7, "translation invariance")
    _near(revised.corrected_values(x[:, 0] - x[:, 1], base.strata), y[:, 0] - y[:, 1], "paired linearity")
    for name, values in base_before.items():
        require(np.array_equal(getattr(base, name), values), f"v1 context mutated: {name}")
    for call in (lambda: revised.correct_inference_context(corrected), lambda: revised.residual_scales(["a", "b", "b"])):
        try:
            call()
        except ValueError:
            pass
        else:
            raise DesignFailure("double correction or singleton was accepted")
    reduced = deterministic_reduced(base)
    prior = original.analyze(copy.deepcopy(reduced), context=base)
    result = revised.analyze(copy.deepcopy(reduced), context=corrected)
    require(result["inference_method"] == METHOD and result["schema"] == "gfmcd.register_operator.analysis.v2", "analysis method identity")
    for name in ("block_endpoints", "matching_means", "competitors"):
        require(result[name] == prior[name], f"point quantity changed: {name}")
    for name in ("seeds", "control_means", "maximum_control", "target_exceeds_all", "exact_add_one_one_sided_p"):
        require(result["pseudo_phase"][name] == prior["pseudo_phase"][name], f"pseudo quantity changed: {name}")
    checked = 0
    def scalar(values, observed, label):
        nonlocal checked, maximum
        draws = independent_transform(values, base.strata)[base.indices].mean(axis=1)
        maximum = max(maximum, _near(observed["ci95"], np.quantile(draws, [.025, .975]), label + " interval"))
        _near(observed["mean"], np.mean(values), label + " point mean")
        _near(observed["bootstrap_sd"], np.std(draws, ddof=1), label + " SD")
        checked += 1
        return draws
    def psi(matrix):
        diagonal = np.array([matrix[:, p, p] for p in range(3)]).mean(axis=0)
        off = np.array([matrix[:, p, q] for p in range(3) for q in range(3) if p != q]).mean(axis=0)
        return diagonal - off
    primary_values = psi(reduced["matrices"]["cc_max"])
    scalar(primary_values, result["primary_psi"], "primary")
    for arm, matrix in reduced["matrices"].items():
        scalar(psi(matrix), result["arms"][arm]["psi"], arm)
    for section, key in (("native", "native"), ("sites", "phase_blind")):
        for name, values in reduced[key].items():
            scalar(values, result["instrument"][section][name], section + ":" + name)
    for key in ("donor_max", "acceptor_max", "donor_linear", "acceptor_linear"):
        _near(result["instrument"]["native_floors"][key],
              min(0., result["instrument"]["native"][key]["ci95"][0]), "native floor:" + key)
    scalar(reduced["health"]["logp_delta"], result["instrument"]["protected_logp_delta"], "protected health")
    scalar(reduced["matrices"]["cc_max"].mean(axis=(1, 2)), result["instrument"]["phase_blind_capability"], "capability")
    per_control = np.sum(base.coefficients * reduced["matrices"]["cc_max"][None], axis=(2, 3))
    paired = primary_values - per_control.mean(axis=0)
    paired_draws = independent_transform(paired, base.strata)[base.indices].mean(axis=1)
    _near(result["pseudo_phase"]["target_minus_mean_control"]["lower95"], np.quantile(paired_draws, .05), "paired-control bound")
    residual = primary_values.copy()
    for label in sorted(set(base.strata)):
        members = [i for i, s in enumerate(base.strata) if s == label]
        residual[members] -= residual[members].mean()
    analytic = (1.96 + .8416212335729143) * math.sqrt(float(np.sum((expected_scales * residual) ** 2))) / 6
    _near(result["primary_psi"]["analytic_empirical_mde80"], analytic, "corrected analytic MDE")
    _near(result["primary_psi"]["uncorrected_analytic_empirical_mde80"], prior["primary_psi"]["analytic_empirical_mde80"], "retained uncorrected diagnostic")
    centered_draws = independent_transform(residual, base.strata)[base.indices].mean(axis=1)
    _near(result["primary_psi"]["mde80"], (1.96 + .8416212335729143) * centered_draws.std(ddof=1), "finite corrected MDE gate")
    def family(values, observed, label):
        values = np.column_stack(values)
        draws = independent_transform(values, base.strata)[base.indices].mean(axis=1)
        means, sd = values.mean(axis=0), draws.std(axis=0, ddof=1)
        active = sd > 0
        critical = np.quantile(np.max(np.abs((draws[:, active] - means[active]) / sd[active]), axis=1), .95)
        _near(observed["means"], means, label + " means")
        _near(observed["lower"], means - critical * sd, label + " lower")
        _near(observed["upper"], means + critical * sd, label + " upper")
        _near(observed["bootstrap_sd"], sd, label + " SD")
    matrix = reduced["matrices"]["cc_max"]
    diagonal = np.array([matrix[:, p, p] for p in range(3)]).T
    plus = np.array([matrix[:, p, (p + 1) % 3] for p in range(3)]).T
    minus = np.array([matrix[:, p, (p - 1) % 3] for p in range(3)]).T
    family([(diagonal - plus).mean(axis=1), (diagonal - minus).mean(axis=1),
            *[(diagonal[:, p] - .5 * (plus[:, p] + minus[:, p])) for p in range(3)]],
           result["directions"], "directions")
    family([plus.mean(axis=1), minus.mean(axis=1)], result["mismatch_equivalence"], "mismatch")
    arm_psi = {key: psi(value) for key, value in reduced["matrices"].items()}
    family([arm_psi[a] - arm_psi[b] for a, b in (("cc_max", "ac_max"), ("cc_linear", "ac_linear"),
            ("cc_max", "cc_linear"), ("ac_max", "ac_linear"))], result["bridge_diagnostics"], "operator diagnostics")
    family([arm_psi["ac_linear"] - arm_psi["at_linear"]], result["historical_psi_ac_minus_at_linear"], "AC/AT Psi")
    for margin in ("max", "linear"):
        values = (reduced["matrices"]["ac_" + margin] - reduced["matrices"]["at_" + margin]).mean(axis=(1, 2))
        family([values], result["historical_phase_blind_ac_minus_at_" + margin], "AC/AT pooled " + margin)
    require(original.analyze(copy.deepcopy(reduced), context=base) == prior, "historical inference was mutated")
    return {"enumerated_resamples": len(base.indices), "stratum_sizes": [2, 4],
            "scalar_paths_checked": checked, "simultaneous_families_checked": 6,
            "maximum_absolute_discrepancy": maximum,
            "rng_calls": 0, "point_and_pseudo_quantities_unchanged": True}


def audit_design(*, repo: Path = REPO, base_freeze: Path | None = None,
                 amendment: Path | None = None, development: Path | None = None) -> dict:
    """Source-conformance receipt, never authorization to open fresh streams."""
    repo = repo.resolve()
    base_path = (base_freeze or repo / STUDY / "protocol_freeze_v1.json").resolve()
    amendment_path = (amendment or repo / STUDY / "AMENDMENT_A1.md").resolve()
    result = {"schema": "gfmcd.register_operator.repair_design_audit.v2", "passed": False,
        "status": "FAIL", "inference_method": METHOD, "base_freeze_sha256": BASE_SHA,
        "auditor_sha256": sha256_file(__file__), "checks": {}, "failures": [], "files": {},
        "model_loaded": False, "fresh_streams_opened": False, "bootstrap_rng_called": False,
        "development_replayed": False, "authorized_for_model": False}
    paths: set[Path] = {base_path, amendment_path, Path(__file__).resolve()}
    try:
        require(sha256_file(base_path) == BASE_SHA, "v1 parent freeze changed")
        base = read_json(base_path)
        require(base["status"] == "FROZEN", "parent freeze status changed")
        bound = {str(base_path): BASE_SHA}
        for name, expected in base["files"].items():
            path = resolved(name, repo)
            require(path.is_file() and sha256_file(path) == expected, f"immutable v1 binding changed: {name}")
            paths.add(path)
            bound[str(path)] = expected
        result["checks"]["immutable_parent_closure"] = True
        for name in REQUIRED_ADDITIVE:
            require((repo / name).is_file(), f"missing additive input: {name}")
        sources = source_closure([path.resolve() for directory in (repo / SCRIPTS, repo / "src/gfmcd/splicing", repo / "tests")
            for path in directory.glob("*.py") if directory == repo / SCRIPTS or "register_operator" in path.name]
            + [repo / "tests/test_motif_cleanliness.py"], repo)
        paths.update(sources)
        source_start = {str(path): sha256_file(path) for path in sources | {amendment_path}}
        require(amendment_path.is_file(), "amendment missing")
        artifacts = [(base_path, "FROZEN", ())]
        for name, digest in PRESERVED_SHA.items():
            path = repo / RESULTS / name
            paths.update((path, path.with_suffix(".manifest.json")))
            require(path.is_file() and path.with_suffix(".manifest.json").is_file(), f"missing preserved v1 evidence: {name}")
            require(sha256_file(path) == digest, f"preserved v1 artifact changed: {name}")
            bound[str(path)] = digest
            artifacts.append((path, "PASS" if name in ("smoke_v1.json", "analysis_v1.audit.json") else
                              "INSTRUMENT_UNRESOLVED", (base_path,)))
        failed = read_json(repo / RESULTS / "statistical_calibration_v1.json")
        stop = read_json(repo / RESULTS / "analysis_v1.audit.json")
        final = read_json(repo / RESULTS / "result_v1.json")
        require(failed["status"] == "INSTRUMENT_UNRESOLVED" and failed["acceptance"]["failures"] == ["coverage:directions"]
                and failed["acceptance"]["noncoverage_by_family"]["directions"]["count"] == 83,
                "v1 failure not preserved")
        require(stop["status"] == "PASS" and stop["passed"] is True and stop["authorized_for_cc"] is False,
                "v1 stop audit not preserved")
        require(final["classification"] == "INSTRUMENT_UNRESOLVED" and final["new_model_outputs"] == 0,
                "v1 final record changed")
        for artifact in (failed, stop, final):
            require(artifact["freeze_sha256"] == BASE_SHA, "preserved evidence has wrong parent")
        selection = read_json(resolved(base["selection_path"], repo))
        ids = selection["block_ids"]
        require(len(ids) == 342 and canonical_hash(ids) == IDS_SHA, "approved cohort changed")
        strata = [":".join(b.split(":")[1:3]) for b in ids]
        require(len(set(strata)) == 32 and min(Counter(strata).values()) == 2
                and Counter(b.split(":")[2] for b in ids) == {"+": 171, "-": 171}, "strata changed")
        result["checks"]["v1_failure_and_selected_unit_preserved"] = True
        result["generator"] = verify_generator_identity((repo / SCRIPTS / "calibrate_statistics.py").read_text(),
                                                         (repo / SCRIPTS / "calibrate_statistics_v2.py").read_text())
        result["checks"]["generator_thresholds_truths_schedule_identical"] = True
        result["source_contract"] = verify_source_contract(repo)
        result["checks"]["version_barrier_scope_and_exposure_order"] = True
        result["numerical"] = verify_numeric_repair(importlib.import_module("gfmcd.splicing.register_operator_statistics_v2"))
        result["checks"]["independent_exact_enumeration_and_endpoint_paths"] = True
        if development is not None:
            development = development.resolve()
            observed = read_json(development)
            require(observed.get("status") == "PASS" and observed.get("passed") is True
                    and observed.get("inference_method") == METHOD and observed.get("fresh_streams_opened") is False
                    and observed.get("base_freeze_sha256") == BASE_SHA
                    and observed.get("model_loaded") is False and observed.get("new_model_outputs") == 0
                    and observed.get("point_and_pseudo_memorials_passed") == 1768
                    and observed.get("counts") == {"development": 256, "null": 1000, "plants": 512}
                    and observed.get("acceptance", {}).get("all_passed") is True,
                    "consumed-stream development has not passed")
            paths.update((development, development.with_suffix(".manifest.json")))
            artifacts.append((development, "PASS", (base_path, amendment_path,
                repo / "src/gfmcd/splicing/register_operator_statistics_v2.py",
                repo / SCRIPTS / "calibrate_statistics_v2.py", repo / SCRIPTS / "develop_repair.py")))
            result["development_evidence"] = file_record(development)
        for artifact, status, required_inputs in artifacts:
            manifest, evidence_paths = verify_manifest_evidence(artifact, repo, status=status, required_inputs=required_inputs)
            paths.update(evidence_paths)
            for record in [*manifest.get("inputs", []), *manifest.get("outputs", [])]:
                path = str(resolved(record["path"], repo))
                require(path not in bound or bound[path] == record["sha256"], f"conflicting immutable references: {path}")
                bound[path] = record["sha256"]
        result["checks"]["preserved_manifests_complete_inputs_outputs_and_development"] = True
        result["files"] = {str(path.resolve()): sha256_file(path) for path in sorted(paths)}
        for path, expected in bound.items():
            require(result["files"].get(path) == expected, f"bound input changed during audit: {path}")
        for path, expected in source_start.items():
            require(result["files"].get(path) == expected, f"reviewed source changed during audit: {path}")
        result["source_closure_count"] = len(sources)
        result["checks"]["sources_stable_during_review"] = True
        result["passed"] = True
        result["status"] = "PASS"
    except (DesignFailure, KeyError, ValueError, TypeError, OSError, ImportError) as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    return result


def write_audit(output: Path, **kwargs) -> tuple[dict, Path]:
    # Do not resolve a terminal symlink before checking the exclusive output.
    output = output.absolute()
    require(output.suffix == ".json", "audit output must end in .json")
    manifest_path = output.with_suffix(".manifest.json")
    for path in (output, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"immutable audit output exists: {path}")
    result = audit_design(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    manifest = ExperimentManifest.create("register-operator:a1-independent-design-audit", sys.argv,
        repo=kwargs.get("repo", REPO), status=result["status"],
        config={"inference_method": METHOD, "fresh_streams_opened": False, "rng_calls": 0},
        notes=["Pure correction and exhaustive enumeration only; no random stream or model was opened.",
               "PASS establishes source conformance, not fresh calibration or target capability."])
    for path in result["files"]:
        manifest.add_input(path)
    if not result["files"]:
        manifest.add_input(__file__)
    manifest.add_output(output)
    with manifest_path.open("x") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--base-freeze", type=Path)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result, manifest = write_audit(args.out, repo=args.repo, base_freeze=args.base_freeze,
                                   amendment=args.amendment, development=args.development)
    print(json.dumps({"status": result["status"], "failures": result["failures"],
                      "audit": str(args.out.resolve()), "manifest": str(manifest)}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
