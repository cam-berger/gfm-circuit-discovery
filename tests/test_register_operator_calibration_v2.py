"""Fresh roots are inaccessible until source-bound A1 smoke; generators stay exact."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from gfmcd.splicing.register_operator_io import write_artifact

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/splicebert_register_operator_replication"
spec = importlib.util.spec_from_file_location("calibration_v2_guard_tests", SCRIPTS / "calibrate_statistics_v2.py")
cal = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cal
spec.loader.exec_module(cal)


def test_generator_truth_and_acceptance_function_bodies_are_identical_to_v1():
    original = ast.parse((SCRIPTS / "calibrate_statistics.py").read_text())
    revised = ast.parse((SCRIPTS / "calibrate_statistics_v2.py").read_text())
    names = {"Latents", "draw_latents", "deterministic_latents", "fixed_tie_membership",
        "_four_logits", "make_payload", "family_noncoverage", "evaluate_acceptance",
        "interval_arrays", "canonical_bytes", "load_selection"}
    def selected(tree):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}
    assert selected(original) == selected(revised)
    assert names <= set(selected(revised))
    assert cal.STREAMS == {"development": 2026101604, "null": 2026101605, "plants": 2026101606}
    assert cal.REPETITIONS == {"development": 256, "null": 1000, "plants": 256}
    assert (cal.FPR_MAX, cal.NONCOVERAGE_MAX, cal.PLANT_SUCCESS_MIN, cal.TRUTH_TOLERANCE) == (38, 71, 216, 1e-10)


def fake_frozen_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "REPO", tmp_path)
    selection = tmp_path / "selected.json"
    selection.write_bytes((ROOT / "studies/splicebert_register_operator_replication/planning/fallback342_selection_v1.json").read_bytes())
    specification = tmp_path / "amendment.md"
    specification.write_text("test amendment")
    inputs = [selection, specification]
    for name in cal.CODE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("bound source")
        inputs.append(path)
    out = tmp_path / "result.json"
    receipt = {"status": "FROZEN_V2", "schema": "gfmcd.register_operator.freeze.v2",
        "inference_method": cal.METHOD, "calibration": cal.frozen_settings(),
        "files": {str(p): cal.sha256_file(p) for p in inputs},
        "selection_sha256": cal.sha256_file(selection), "outputs": {
            "calibration": str(out), "calibration_manifest": str(out.with_suffix(".manifest.json")),
            "calibration_attempt": str(out.with_suffix(".attempt.json")),
            "calibration_attempt_manifest": str(out.with_suffix(".attempt.manifest.json")),
            "calibration_records": str(out.with_suffix(".records.jsonl.gz")),
            "smoke": str(tmp_path / "smoke.json")}}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(receipt))
    # Isolate calibration's additional code, output, and smoke gates. The v2
    # parent/version barrier is separately tested against its real verifier.
    monkeypatch.setattr(cal, "verify_v2_receipt", lambda path, repo: json.loads(path.read_text()))
    return receipt, freeze, selection, specification, out


def test_no_fresh_rng_or_attempt_when_postfreeze_smoke_is_absent(tmp_path, monkeypatch):
    receipt, freeze, selection, specification, out = fake_frozen_inputs(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh RNG before smoke")
    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    with pytest.raises(FileNotFoundError):
        cal.main(["--freeze", str(freeze), "--selection", str(selection),
                  "--specification", str(specification), "--out", str(out)])
    assert not out.exists() and not out.with_suffix(".attempt.json").exists()


def test_attempt_has_utc_manifest_and_mutated_freeze_cannot_pass(tmp_path, monkeypatch):
    receipt, freeze, selection, specification, out = fake_frozen_inputs(tmp_path, monkeypatch)
    initial = cal.sha256_file(freeze)
    write_artifact(receipt["outputs"]["smoke"], {"status": "PASS", "inference_method": cal.METHOD,
        "freeze_sha256": initial}, inputs=[freeze], experiment="test-a1-smoke")
    monkeypatch.setattr(cal, "run_smoke", lambda: {"status": "SMOKE_PASS"})
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh RNG in provenance test")
    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    def fake_run(*args, **kwargs):
        attempt = json.loads(out.with_suffix(".attempt.json").read_text())
        assert attempt["created_at_utc"] == attempt["started_at_utc"]
        assert attempt["inference_method"] == cal.METHOD
        assert out.with_suffix(".attempt.manifest.json").is_file()
        receipt["changed"] = True
        freeze.write_text(json.dumps(receipt))
        return {"status": "PASS", "development_stream_opened": False}
    monkeypatch.setattr(cal, "run_calibration", fake_run)
    assert cal.main(["--freeze", str(freeze), "--selection", str(selection),
        "--specification", str(specification), "--out", str(out)]) == 1
    result = json.loads(out.read_text())
    assert result["status"] == "INSTRUMENT_UNRESOLVED"
    assert result["freeze_sha256"] == initial
    assert result["inference_method"] == cal.METHOD
    assert "freeze receipt changed" in result["post_run_binding_failure"]
