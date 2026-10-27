"""Independent A1 audit must reject plausible implementation and scope defects."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gfmcd.splicing import register_operator_statistics_v2 as revised

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/splicebert_register_operator_replication"
spec = importlib.util.spec_from_file_location("repair_design_auditor_tests", SCRIPTS / "audit_repair_design.py")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


@pytest.fixture(autouse=True)
def forbid_random_streams(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("design review must not open a random stream")
    for name in ("default_rng", "SeedSequence", "RandomState"):
        monkeypatch.setattr(np.random, name, forbidden)


def proxy(**overrides):
    return SimpleNamespace(**(vars(revised) | overrides))


def test_exhaustive_unequal_strata_and_every_analysis_path_without_rng():
    result = audit.verify_numeric_repair(revised)
    assert result["enumerated_resamples"] == 2**2 * 4**4
    assert result["stratum_sizes"] == [2, 4]
    assert result["scalar_paths_checked"] == 23
    assert result["simultaneous_families_checked"] == 6
    assert result["maximum_absolute_discrepancy"] < 2e-12
    assert result["point_and_pseudo_quantities_unchanged"] is True
    assert result["rng_calls"] == 0


@pytest.mark.parametrize("defect", ["missing", "global", "uncentered", "clipped"])
def test_pure_correction_mutations_are_rejected(defect):
    def mutant(base):
        current = revised.correct_inference_context(base)
        if defect == "missing":
            weights = base.weights
        elif defect == "global":
            scale = np.sqrt(6 / 5)
            weights = base.weights * scale + (1 - scale) / 6
        elif defect == "uncentered":
            weights = base.weights * current.residual_scales
        else:
            weights = np.maximum(current.weights, 0)
        return replace(current, weights=weights)
    with pytest.raises(audit.DesignFailure, match="affine coefficients"):
        audit.verify_numeric_repair(proxy(correct_inference_context=mutant))


@pytest.mark.parametrize("path", ["native", "health", "directions", "paired"])
def test_one_uncorrected_inference_path_is_rejected(path):
    def mutant(reduced, *, context):
        result = revised.analyze(reduced, context=context)
        old = revised.original.InferenceContext(context.block_ids, context.strata, context.indices,
            context.original_weights, context.assignments, context.coefficients,
            context.draws, context.seed, context.pseudo_seeds)
        uncorrected = revised.original.analyze(reduced, context=old)
        if path == "native":
            result["instrument"]["native"] = uncorrected["instrument"]["native"]
        elif path == "health":
            result["instrument"]["protected_logp_delta"] = uncorrected["instrument"]["protected_logp_delta"]
        elif path == "directions":
            result["directions"] = uncorrected["directions"]
        else:
            result["pseudo_phase"]["target_minus_mean_control"] = uncorrected["pseudo_phase"]["target_minus_mean_control"]
        return result
    with pytest.raises(audit.DesignFailure, match="maximum discrepancy"):
        audit.verify_numeric_repair(proxy(analyze=mutant))


def test_points_and_permutation_ranks_cannot_be_repaired():
    def mutant(reduced, *, context):
        result = revised.analyze(reduced, context=context)
        result["pseudo_phase"]["exact_add_one_one_sided_p"] += .01
        return result
    with pytest.raises(audit.DesignFailure, match="pseudo quantity changed"):
        audit.verify_numeric_repair(proxy(analyze=mutant))


def test_generator_schedule_truth_and_thresholds_are_exact_ast_matches():
    result = audit.verify_generator_identity((SCRIPTS / "calibrate_statistics.py").read_text(),
                                             (SCRIPTS / "calibrate_statistics_v2.py").read_text())
    assert set(result["identical_definitions"]) == set(audit.GENERATOR_DEFINITIONS)
    assert result["fresh_streams"] == {"development": 2026101604, "null": 2026101605, "plants": 2026101606}


@pytest.mark.parametrize("before,after,reason", [
    ("NONCOVERAGE_MAX = 71", "NONCOVERAGE_MAX = 83", "constant changed"),
    ("TRUTH_TOLERANCE = 1e-10", "TRUTH_TOLERANCE = 1e-8", "constant changed"),
    ("2026101605", "2026101607", "fresh roots differ"),
    ("def draw_latents(", "def changed_draw_latents(", "definition changed"),
    ("def run_calibration(", "def changed_run_calibration(", "definition changed"),
])
def test_generator_or_gate_changes_cannot_hide_behind_v2(before, after, reason):
    old = (SCRIPTS / "calibrate_statistics.py").read_text()
    new = (SCRIPTS / "calibrate_statistics_v2.py").read_text()
    assert before in new
    with pytest.raises(audit.DesignFailure, match=reason):
        audit.verify_generator_identity(old, new.replace(before, after, 1))


def copy_contract_sources(tmp_path):
    for path in [*SCRIPTS.glob("*.py"),
                 ROOT / "src/gfmcd/splicing/register_operator_io.py",
                 ROOT / "src/gfmcd/splicing/register_operator_io_v2.py",
                 ROOT / "src/gfmcd/splicing/register_operator_statistics_v2.py"]:
        target = tmp_path / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return tmp_path


def test_actual_execution_source_contract_accepts_the_request_helper_refactor():
    result = audit.verify_source_contract(ROOT)
    assert result["historical_request_branches_unchanged"]
    assert result["native_and_target_share_corrected_context"]
    assert result["fresh_calibration_requires_freeze_smoke_and_attempt"]


@pytest.mark.parametrize("relative,before,after,reason", [
    ("src/gfmcd/splicing/register_operator_io_v2.py", '"FROZEN_V2"', '"FROZEN"', "version barrier weakened"),
    ("scripts/splicebert_register_operator_replication/run_model_v2.py", "else 342", "else 336", "request branch changed"),
    ("scripts/splicebert_register_operator_replication/run_model_v2.py", "blocks, plans = prepare_requests(package, genome, args.stage)",
        "blocks, plans = unchecked_requests(package, genome, args.stage)", "bypasses the reviewed request planner"),
    ("scripts/splicebert_register_operator_replication/analyze_results_v2.py", "register_operator_statistics_v2 import analyze",
        "register_operator_statistics import analyze", "target analysis uses wrong method"),
])
def test_source_scope_defects_are_rejected(tmp_path, relative, before, after, reason):
    copied = copy_contract_sources(tmp_path)
    path = copied / relative
    source = path.read_text()
    assert before in source
    path.write_text(source.replace(before, after))
    with pytest.raises(audit.DesignFailure, match=reason):
        audit.verify_source_contract(copied)


@pytest.mark.parametrize("occupied", ["output", "manifest", "broken_symlink"])
def test_audit_pair_is_immutable_before_work_starts(tmp_path, monkeypatch, occupied):
    out = tmp_path / "audit.json"
    if occupied == "broken_symlink":
        out.symlink_to(tmp_path / "missing")
    else:
        (out if occupied == "output" else out.with_suffix(".manifest.json")).write_text("preserved")
    monkeypatch.setattr(audit, "audit_design", lambda **kwargs: pytest.fail("must refuse before audit"))
    with pytest.raises(FileExistsError):
        audit.write_audit(out, repo=ROOT)


def test_changed_parent_fails_before_numeric_or_execution_work(tmp_path, monkeypatch):
    parent = tmp_path / "parent.json"
    parent.write_text('{"status":"FROZEN"}')
    monkeypatch.setattr(audit, "verify_numeric_repair", lambda *args: pytest.fail("changed parent reached inference"))
    result = audit.audit_design(repo=ROOT, base_freeze=parent)
    assert result["status"] == "FAIL" and not result["passed"]
    assert result["failures"] == ["DesignFailure: v1 parent freeze changed"]
    assert result["fresh_streams_opened"] is False
    assert result["model_loaded"] is False


def test_failed_design_receipt_and_manifest_are_preserved(tmp_path, monkeypatch):
    out = tmp_path / "audit.json"
    failure = {"status": "FAIL", "passed": False, "files": {}, "failures": ["injected defect"]}
    monkeypatch.setattr(audit, "audit_design", lambda **kwargs: failure)
    result, manifest_path = audit.write_audit(out, repo=ROOT)
    assert result == failure
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "FAIL"
    assert manifest["outputs"][0]["sha256"] == audit.sha256_file(out)
    assert manifest["inputs"][0]["sha256"] == audit.sha256_file(audit.__file__)
    with pytest.raises(FileExistsError):
        audit.write_audit(out, repo=ROOT)


def test_manifest_checker_verifies_companion_stream_outputs_and_required_sources(tmp_path):
    artifact, source, stream = (tmp_path / name for name in ("result.json", "source.py", "stream.gz"))
    artifact.write_text('{"status":"PASS"}')
    source.write_text("# bound source\n")
    stream.write_bytes(b"consumed stream bytes")
    manifest_path = artifact.with_suffix(".manifest.json")
    manifest = {"status": "PASS", "inputs": [audit.file_record(source)],
                "outputs": [audit.file_record(artifact), audit.file_record(stream)]}
    manifest_path.write_text(json.dumps(manifest))
    _, paths = audit.verify_manifest_evidence(artifact, tmp_path, status="PASS", required_inputs=[source])
    assert paths == {artifact, source, stream, manifest_path}
    with pytest.raises(audit.DesignFailure, match="omits required input"):
        audit.verify_manifest_evidence(artifact, tmp_path, status="PASS", required_inputs=[tmp_path / "missing.py"])
    stream.write_bytes(b"changed")
    with pytest.raises(audit.DesignFailure, match="manifest evidence changed"):
        audit.verify_manifest_evidence(artifact, tmp_path, status="PASS")


def test_independent_source_closure_includes_postfreeze_auditors_and_all_additive_tests():
    sources = audit.source_closure([*SCRIPTS.glob("*.py"),
        *ROOT.glob("src/gfmcd/splicing/register_operator*.py"),
        *ROOT.glob("tests/test_register_operator*.py")], ROOT)
    for name in audit.REQUIRED_ADDITIVE:
        assert ROOT / name in sources
    assert SCRIPTS / "audit_results.py" in sources
    assert SCRIPTS / "audit_cc_measurement.py" in sources
    assert ROOT / "tests/test_register_operator_execution_receipts_v2.py" in sources
    assert ROOT / "src/gfmcd/provenance.py" in sources
