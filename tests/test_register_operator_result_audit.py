"""Independent-result auditor memorials; no model and no new calibration streams."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/splicebert_register_operator_replication/audit_results.py"
spec = importlib.util.spec_from_file_location("result_audit_test_module", SCRIPT)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def fixture():
    generator = audit.fixture_generator()
    ids = [f"synthetic:chr1:{'+' if i < 6 else '-'}:b{i:03d}" for i in range(12)]
    strata = ["chr1:+"] * 6 + ["chr1:-"] * 6
    return generator.make_payload(ids, strata, generator.deterministic_latents(12), delta=.15, null=False)[0]


def test_auditor_imports_no_producer_measurement_or_inference():
    tree = ast.parse(SCRIPT.read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and ("register_operator_measurement" in name or "register_operator_statistics" in name) for name in imports)


def test_synthetic_raw_reduction_independently_matches_production():
    from gfmcd.splicing.register_operator_measurement import reduce_score_payload
    payload = fixture()
    independent = audit.synthetic_reduction(payload)
    comparison = audit.compare(reduce_score_payload(payload), independent)
    assert comparison["maximum_absolute_error"] == 0
    corrupted = dict(independent)
    corrupted["matrices"] = dict(independent["matrices"])
    corrupted["matrices"]["cc_max"] = independent["matrices"]["cc_max"] + .001
    with pytest.raises(ValueError, match="numeric difference"):
        audit.compare(reduce_score_payload(payload), corrupted)


def test_independent_full_inference_matches_toy_producer_with_direct_samples():
    from gfmcd.splicing.register_operator_measurement import reduce_score_payload
    from gfmcd.splicing.register_operator_statistics import analyze, make_inference_context
    payload = fixture()
    reduced = reduce_score_payload(payload)
    reduced["prerequisites"]["independent_audit"] = False
    context = make_inference_context(payload["block_ids"], payload["strata"], draws=128)
    producer = analyze(reduced, context=context)
    independent = audit.synthetic_reduction(payload)
    independent["sites"] = independent["phase_blind"]
    actual = audit.infer(independent, prereqs={k: True for k in ("construction", "smoke", "calibration", "fidelity")},
                         numeric_passed=True, audit_passed=False, draws=128)
    comparison = audit.compare(producer, actual)
    assert comparison["maximum_absolute_error"] < 1e-10
    actual["gates"]["independent_audit"] = True
    assert audit.classify(actual) in audit.MATERIAL


def test_interval_replay_covers_every_family_and_matches_independent_full_inference():
    payload = fixture()
    reduced = audit.synthetic_reduction(payload)
    reduced["sites"] = reduced["phase_blind"]
    full = audit.infer(reduced, prereqs={"all": True}, numeric_passed=True, audit_passed=True, draws=128)
    replay = audit.replay_interval_families(reduced, audit.bootstrap_indices(payload["strata"], draws=128))
    assert set(replay) == set(audit.FAMILIES)
    audit.compare(full, replay)


def test_raw_scores_rederive_full_vocab_logp_both_margins_and_truth():
    import torch
    values = torch.tensor([[1., -2., .5, 0., 1., -1., 4., 1., 6., -2.]], dtype=torch.float32)
    lp = float(torch.log_softmax(values, dim=1)[0, 8])
    row = {"request_id": "test", "token_count": 1024, "token_position": 2, "position": 1,
           "true_base": "G", "all_forward_logits_finite": True, "masked_logits": values[0].tolist(),
           "base_logits": [4., 1., 6., -2.], "max_margin": 2., "linear_margin": 5.,
           "true_base_log_probability": lp, "correct": True}
    mapping = {"A": 6, "C": 7, "G": 8, "T": 9, "MASK": 4, "CLS": 2, "SEP": 3}
    output, comparison = audit.raw_scores([row], mapping)
    assert output["test"]["max_margin"] == 2
    assert comparison["maximum_absolute_differences"] == {k: 0. for k in ("linear_margin", "max_margin", "true_base_log_probability")}
    with pytest.raises(ValueError, match="duplicate"):
        audit.raw_scores([row, row], mapping)
    row["max_margin"] = 5.
    with pytest.raises(ValueError, match="max_margin"):
        audit.raw_scores([row], mapping)


def test_bootstrap_indices_keep_exact_stratum_counts_and_reject_singletons():
    strata = ["b"] * 4 + ["a"] * 2
    samples = audit.bootstrap_indices(strata, draws=16)
    assert np.all(samples[:, :2] >= 4) and np.all(samples[:, 2:] < 4)
    with pytest.raises(ValueError, match="singleton"):
        audit.bootstrap_indices(["a", "b", "b"], draws=4)


def test_known_binomial_boundaries_and_comparator_fail_closed():
    assert audit.binomial_record(38, 1000)["upper95_one_sided"] < .05
    assert audit.binomial_record(39, 1000)["upper95_one_sided"] > .05
    assert audit.binomial_record(216, 256)["lower95_one_sided"] > .8
    with pytest.raises(ValueError, match="missing"):
        audit.compare({}, {"mean": 0.})
    with pytest.raises(ValueError, match="numeric difference"):
        audit.compare(float("nan"), 0.)
    with pytest.raises(ValueError, match="categorical"):
        audit.compare(True, False)


def test_manifest_requires_actual_output_and_required_input(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("source")
    output = tmp_path / "output.json"
    output.write_text("{}")
    manifest = {"outputs": [{"path": str(output), "sha256": audit.digest(output)}],
                "inputs": [{"path": str(source), "sha256": audit.digest(source)}]}
    audit.manifest_path(output).write_text(json.dumps(manifest))
    audit.check_manifest(output, required_inputs=[source])
    source.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        audit.check_manifest(output)


def test_immutable_audit_writes_cannot_replace_existing_evidence(tmp_path):
    path = tmp_path / "audit.json"
    audit.write_exclusive(path, {"status": "FAIL"})
    with pytest.raises(FileExistsError):
        audit.write_exclusive(path, {"status": "PASS"})
    assert json.loads(path.read_text())["status"] == "FAIL"


def test_classifier_never_promotes_failed_capability():
    payload = fixture()
    reduced = audit.synthetic_reduction(payload)
    reduced["sites"] = reduced["phase_blind"]
    result = audit.infer(reduced, prereqs={"calibration": False}, numeric_passed=True,
                         audit_passed=True, draws=128)
    assert result["classification"] == "INSTRUMENT_UNRESOLVED"
    assert result["gates"]["construction"] is False
