"""On-path memorials for the new study's model-free calibration runner.

These tests never open the frozen development/held-out RNG streams and never load
model weights. Synthetic logit inputs exercise the production row path.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/splicebert_register_operator_replication/calibrate_statistics.py"
spec = importlib.util.spec_from_file_location("register_operator_calibration_test", SCRIPT)
calibration = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = calibration
spec.loader.exec_module(calibration)


def identities(n=12):
    return ([f"synthetic:chr1:{'+' if i < n // 2 else '-'}:b{i:03d}" for i in range(n)],
            ["chr1:+"] * (n // 2) + ["chr1:-"] * (n // 2))


def test_plant_traverses_serialized_raw_logits_and_both_real_margin_functions():
    from gfmcd.splicing import register_operator_measurement as measurement
    ids, strata = identities()
    latent = calibration.deterministic_latents(len(ids))
    left, _ = calibration.make_payload(ids, strata, latent, delta=.05, null=False)
    right, _ = calibration.make_payload(ids, strata, latent, delta=.15, null=False)
    a = measurement.reduce_score_payload(measurement.deserialize_score_payload(
        measurement.serialize_score_payload(left)))
    b = measurement.reduce_score_payload(measurement.deserialize_score_payload(
        measurement.serialize_score_payload(right)))
    for margin in ("max", "linear"):
        for operator in ("cc", "ac", "at"):
            np.testing.assert_allclose(b["matrices"][f"{operator}_{margin}"]
                                       - a["matrices"][f"{operator}_{margin}"],
                                       np.broadcast_to(.10 * calibration.H, (12, 3, 3)), atol=2e-6)
    # A fixture with equal rivals would allow the wrong readout to pass this test.
    difference = b["matrices"]["cc_linear"] - b["matrices"]["cc_max"]
    np.testing.assert_allclose(difference, np.broadcast_to(.01 * calibration.H, (12, 3, 3)), atol=2e-6)
    assert not np.allclose(difference, 0)
    assert np.array_equal(left["health_logits"], right["health_logits"])
    assert np.array_equal(left["cell_logits"][..., 0, :], right["cell_logits"][..., 0, :])
    assert sum(row["tied_requests"] for row in b["competitors"].values()) > 0
    assert all(sum(v > 0 for v in row["maximizer_counts_AC_T"]) >= 2
               for row in b["competitors"].values())


def test_raw_reduction_matches_absolute_analytic_truth_for_all_states_and_health():
    from gfmcd.splicing import register_operator_measurement as measurement
    ids, strata = identities()
    latent = replace(calibration.deterministic_latents(12, noise_scale=0),
                     bridge=np.linspace(-.01, .01, 12),
                     native=np.linspace(-.02, .02, 12),
                     health_delta=np.linspace(-.01, .01, 12)[:, None, None]
                         * np.ones((1, 1, 24)))
    payload, details = calibration.make_payload(ids, strata, latent, delta=.15, null=False)
    decoded = measurement.deserialize_score_payload(measurement.serialize_score_payload(payload))
    reduced = measurement.reduce_score_payload(decoded)
    assert reduced["block_ids"] == ids and reduced["strata"] == strata
    for operator in ("cc", "ac", "at"):
        for margin, delta in (("max", .15), ("linear", .16)):
            expected = np.broadcast_to(.05 + delta * calibration.H, (12, 3, 3))
            np.testing.assert_allclose(reduced["matrices"][f"{operator}_{margin}"],
                                       expected, rtol=0, atol=8e-7)
    for margin, intact in (("max", 3.), ("linear", 5.)):
        for operator in ("cc", "ac"):
            np.testing.assert_allclose(reduced["native"][f"{operator}_{margin}"],
                                       .30 + latent.native, rtol=0, atol=8e-7)
        np.testing.assert_allclose(reduced["phase_blind"][f"bridge_{margin}"],
                                   latent.bridge, rtol=0, atol=8e-7)
        for role in ("donor", "acceptor"):
            np.testing.assert_allclose(reduced["native"][f"{role}_{margin}"], intact,
                                       rtol=0, atol=8e-7)
            np.testing.assert_allclose(reduced["phase_blind"][f"{role}_{margin}"], intact,
                                       rtol=0, atol=8e-7)
    np.testing.assert_allclose(reduced["health"]["logp_delta"], latent.health_delta[:, 0].mean(axis=1),
                               rtol=0, atol=2e-7)
    for name in ("construct_correct", "source_correct", "probe_pairs"):
        np.testing.assert_array_equal(reduced["health"][name], np.full(12, 9 * 24))
    assert details["population_truths"] == {
        "primary_psi": [.15], "directions": [.15] * 5,
        "mismatch_equivalence": [.05 - .15 / 3] * 2,
        "bridge_diagnostics": [0., 0., -.01, -.01],
        "historical_psi_ac_minus_at_linear": [0.],
        "historical_phase_blind_ac_minus_at_linear": [0.],
    }


def test_generating_label_permutation_is_an_inverse_column_reindex_for_all_states():
    from gfmcd.splicing import register_operator_measurement as measurement
    ids, strata = identities()
    latent = replace(calibration.deterministic_latents(12),
                     acceptor_permutations=np.tile([1, 2, 0], (12, 1)))
    # Non-symmetric values distinguish pi from pi^{-1}; all arms travel together.
    matrices = np.arange(12 * 3 * 3 * 3, dtype=float).reshape(12, 3, 3, 3) / 1000
    original, _ = calibration.make_payload(ids, strata, latent, delta=0, null=False, matrices=matrices)
    permuted, _ = calibration.make_payload(ids, strata, latent, delta=0, null=True, matrices=matrices)
    a = measurement.reduce_score_payload(original)
    b = measurement.reduce_score_payload(permuted)
    for key in a["matrices"]:
        np.testing.assert_array_equal(b["matrices"][key], a["matrices"][key][:, :, [2, 0, 1]])
    # Native controls are not relabeled with the crossed lattice.
    np.testing.assert_array_equal(original["native_logits"], permuted["native_logits"])


def test_exact_binomial_acceptance_boundaries_are_frozen_and_all_failures_reported():
    zero = {name: 0 for name in calibration.FAMILIES}
    passed = calibration.evaluate_acceptance(null_positive=38, noncoverage={name: 71 for name in zero},
        detection_successes=216, material_successes=216, execution_errors=[])
    assert passed["all_passed"]
    assert passed["full_decision_false_positive"]["upper95_one_sided"] < .05
    assert passed["point05_detection"]["lower95_one_sided"] > .8
    failed = calibration.evaluate_acceptance(null_positive=39, noncoverage={name: 72 for name in zero},
        detection_successes=215, material_successes=215, execution_errors=[{"error": "planted"}])
    assert not failed["all_passed"]
    assert len(failed["failures"]) == 10
    assert calibration.binomial_record(0, 1000)["upper95_one_sided"] == pytest.approx(.002991249545095295)
    assert passed["coverage_audit"]["union_bound_false_failure"] < .01


def test_coverage_requires_the_entire_family_and_only_roundoff_tolerance():
    truths = {name: [0.] for name in calibration.FAMILIES}
    analysis = {name: {"ci95": [-.1, .1]} for name in calibration.FAMILIES}
    analysis["directions"] = {"means": [0., 0.], "lower": [-.1, .01], "upper": [.1, .2]}
    truths["directions"] = [0., 0.]
    failures = calibration.family_noncoverage(analysis, truths)
    assert failures["directions"] and sum(failures.values()) == 1
    analysis["directions"]["lower"][1] = .5e-10
    assert not calibration.family_noncoverage(analysis, truths)["directions"]
    analysis["directions"]["lower"][1] = 2e-10
    assert calibration.family_noncoverage(analysis, truths)["directions"]
    del analysis["mismatch_equivalence"]
    with pytest.raises(KeyError):
        calibration.family_noncoverage(analysis, truths)


def test_unfrozen_full_cli_does_not_open_any_random_stream(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("stochastic stream opened before freeze")
    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    with pytest.raises(SystemExit):
        calibration.main(["--out", str(tmp_path / "result.json")])
    assert not list(tmp_path.iterdir())


def test_freeze_binds_code_selection_and_thresholds_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "REPO", tmp_path)
    selection = tmp_path / "selected.json"
    specification = tmp_path / "calibration.md"
    selection.write_text("{}")
    specification.write_text("frozen specification")
    paths = [selection, specification]
    for name in calibration.CODE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("bound source")
        paths.append(path)
    receipt = {"status": "FROZEN", "files": {str(p): calibration.sha256_file(p) for p in paths},
               "selection_sha256": calibration.sha256_file(selection),
               "calibration": calibration.frozen_settings()}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(receipt))
    calibration.verify_freeze(freeze, selection, specification)
    receipt["calibration"]["fpr_max"] = 39
    freeze.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="constant differs"):
        calibration.verify_freeze(freeze, selection, specification)
    receipt["calibration"] = calibration.frozen_settings()
    freeze.write_text(json.dumps(receipt))
    paths[-1].write_text("changed after freeze")
    with pytest.raises(ValueError, match="hash mismatch"):
        calibration.verify_freeze(freeze, selection, specification)


def test_output_guard_preserves_existing_result(tmp_path):
    existing = tmp_path / "result.json"
    existing.write_text("preserve this")
    with pytest.raises(FileExistsError):
        calibration.main(["--smoke", "--out", str(existing)])
    assert existing.read_text() == "preserve this"


def test_deterministic_full_path_branch_memorials_without_frozen_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("frozen calibration streams must not be consumed by smoke")
    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    result = calibration.run_smoke(draws=128)
    assert result["status"] == "SMOKE_PASS", result["failures"]
    assert result["heldout_stream_opened"] is False
    assert result["development_stream_opened"] is False


def test_inference_agrees_with_direct_indexed_bootstrap_and_explicit_phase_matching():
    from gfmcd.splicing import register_operator_measurement as measurement
    from gfmcd.splicing import register_operator_statistics as statistics
    ids, strata = identities()
    payload, _ = calibration.make_payload(ids, strata, calibration.deterministic_latents(12),
                                          delta=.15, null=False)
    reduced = measurement.reduce_score_payload(payload)
    context = statistics.make_inference_context(ids, strata, draws=256)
    actual = statistics.analyze(reduced, context=context)
    matrices = reduced["matrices"]["cc_max"]
    # Independent coefficient oracle, not matching_endpoints or the cached count matrix.
    coefficients = np.full((3, 3), -1 / 6)
    np.fill_diagonal(coefficients, 1 / 3)
    psi = np.sum(matrices * coefficients, axis=(1, 2))
    bootstrap_means = psi[context.indices].mean(axis=1)
    np.testing.assert_allclose(actual["primary_psi"]["ci95"],
                               np.quantile(bootstrap_means, [.025, .975]), atol=1e-12, rtol=0)
    assert actual["primary_psi"]["mde80"] == pytest.approx(
        calibration.MDE_FACTOR * bootstrap_means.std(ddof=1), abs=1e-12)
    controls = []
    for assignments in context.assignments:
        per_block = []
        for matrix, permutation in zip(matrices, assignments):
            matched = [matrix[p, q] for p in range(3) for q in range(3) if permutation[q] == p]
            mismatched = [matrix[p, q] for p in range(3) for q in range(3) if permutation[q] != p]
            per_block.append(np.mean(matched) - np.mean(mismatched))
        controls.append(np.mean(per_block))
    np.testing.assert_allclose(actual["pseudo_phase"]["control_means"], controls, atol=1e-12, rtol=0)
    count = sum(control >= psi.mean() for control in controls)
    assert actual["pseudo_phase"]["exact_add_one_one_sided_p"] == (1 + count) / 100
    # Recompute every interval family from explicit cell expressions and directly
    # indexed block samples, without production endpoint or max-t helper calls.
    p = np.arange(3)
    independent = {}
    for name, values in reduced["matrices"].items():
        diag = np.stack([values[:, i, i] for i in range(3)], axis=1)
        plus = np.stack([values[:, i, (i + 1) % 3] for i in range(3)], axis=1)
        minus = np.stack([values[:, i, (i + 2) % 3] for i in range(3)], axis=1)
        independent[name] = {
            "psi": diag.mean(axis=1) - (plus.mean(axis=1) + minus.mean(axis=1)) / 2,
            "pool": values.reshape(12, 9).mean(axis=1),
            "directions": np.column_stack((diag.mean(axis=1) - plus.mean(axis=1),
                                            diag.mean(axis=1) - minus.mean(axis=1),
                                            diag - (plus + minus) / 2)),
            "mismatch": np.column_stack((plus.mean(axis=1), minus.mean(axis=1))),
        }
    endpoints = {
        "directions": independent["cc_max"]["directions"],
        "mismatch_equivalence": independent["cc_max"]["mismatch"],
        "bridge_diagnostics": np.column_stack([
            independent[left]["psi"] - independent[right]["psi"]
            for left, right in (("cc_max", "ac_max"), ("cc_linear", "ac_linear"),
                                ("cc_max", "cc_linear"), ("ac_max", "ac_linear"))]),
        "historical_psi_ac_minus_at_linear": (
            independent["ac_linear"]["psi"] - independent["at_linear"]["psi"])[:, None],
        "historical_phase_blind_ac_minus_at_linear": (
            independent["ac_linear"]["pool"] - independent["at_linear"]["pool"])[:, None],
    }
    for name, values in endpoints.items():
        resampled = values[context.indices].mean(axis=1)
        means = values.mean(axis=0)
        sd = resampled.std(axis=0, ddof=1)
        active = sd > 0
        critical = (np.quantile(np.max(np.abs((resampled[:, active] - means[active])
                                              / sd[active]), axis=1), .95)
                    if active.any() else 0.)
        for key, expected in (("means", means), ("lower", means - critical * sd),
                              ("upper", means + critical * sd)):
            np.testing.assert_allclose(actual[name][key], expected, rtol=0, atol=1e-12,
                                       err_msg=f"independent interval oracle: {name}/{key}")
    for kind, source in (("native", reduced["native"]), ("sites", reduced["phase_blind"])):
        for name, values in source.items():
            expected_ci = np.quantile(values[context.indices].mean(axis=1), [.025, .975])
            np.testing.assert_allclose(actual["instrument"][kind][name]["ci95"], expected_ci,
                                       rtol=0, atol=1e-12)


def test_known_nucleotide_margin_probe_catches_wrong_competitor_and_all_tie_cases():
    from gfmcd.splicing import register_operator_measurement as measurement
    # A,C,G,T. True G must never be one of its own max competitors.
    values = np.asarray([[4., 1., 6., -2.], [0., 5., 6., 2.],
                         [0., 0., 3., 0.], [-2., -2., 1., -2.]])
    actual = measurement.nucleotide_margins(values)
    np.testing.assert_array_equal(actual["max"], [2., 1., 3., 3.])
    np.testing.assert_allclose(actual["linear"], [5., 11 / 3, 3., 3.], atol=1e-6)
    assert actual["max"][0] != actual["linear"][0]


def test_output_authorization_cannot_be_retargeted(tmp_path):
    outputs = {"calibration": tmp_path / "result.json", "calibration_records": tmp_path / "rows.gz"}
    receipt = {"outputs": {key: str(path) for key, path in outputs.items()}}
    calibration.verify_output_bindings(receipt, outputs)
    with pytest.raises(ValueError, match="differs from freeze"):
        calibration.verify_output_bindings(receipt, {**outputs, "calibration": tmp_path / "other.json"})
    receipt["outputs"]["calibration"] = "relative.json"
    with pytest.raises(ValueError, match="absolute output authorization"):
        calibration.verify_output_bindings(receipt, outputs)


def test_approved_selection_hash_is_recomputed_not_trusted(tmp_path):
    original = REPO / "studies/splicebert_register_operator_replication/planning/fallback342_selection_v1.json"
    selection = json.loads(original.read_text())
    ids, strata = calibration.load_selection(original)
    assert len(ids) == 342 and min(__import__("collections").Counter(strata).values()) >= 2
    # Preserve every recorded metadata field and stratum while altering one actual ID.
    selection["block_ids"][0] = selection["block_ids"][0].rsplit(":", 1)[0] + ":b999"
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="approved sequence-selected"):
        calibration.load_selection(altered)


def test_editing_the_freeze_during_a_run_cannot_produce_bound_pass(tmp_path, monkeypatch):
    original_selection = REPO / "studies/splicebert_register_operator_replication/planning/fallback342_selection_v1.json"
    selection = tmp_path / "selected.json"
    selection.write_bytes(original_selection.read_bytes())
    specification = tmp_path / "spec.md"
    specification.write_text("frozen specification")
    monkeypatch.setattr(calibration, "REPO", tmp_path)
    paths = [selection, specification]
    for name in calibration.CODE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source binding")
        paths.append(path)
    out = tmp_path / "result.json"
    receipt = {"status": "FROZEN", "files": {str(p): calibration.sha256_file(p) for p in paths},
               "selection_sha256": calibration.sha256_file(selection),
               "calibration": calibration.frozen_settings(),
               "outputs": {"calibration": str(out),
                           "calibration_records": str(out.with_suffix(".records.jsonl.gz")),
                           "calibration_attempt": str(out.with_suffix(".attempt.json")),
                           "calibration_manifest": str(out.with_suffix(".manifest.json"))}}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(receipt))
    initial = calibration.sha256_file(freeze)
    monkeypatch.setattr(calibration, "run_smoke", lambda: {"status": "SMOKE_PASS"})

    def fake_run(*args, **kwargs):
        receipt["changed_during_execution"] = True
        freeze.write_text(json.dumps(receipt))
        return {"status": "PASS", "development_stream_opened": False}

    monkeypatch.setattr(calibration, "run_calibration", fake_run)
    assert calibration.main(["--selection", str(selection), "--specification", str(specification),
                             "--freeze", str(freeze), "--out", str(out)]) == 1
    result = json.loads(out.read_text())
    assert result["status"] == "INSTRUMENT_UNRESOLVED"
    assert result["freeze_sha256"] == initial
    assert "freeze receipt changed" in result["post_run_binding_failure"]
