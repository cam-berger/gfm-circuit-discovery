"""Independent A1 arithmetic and authorization memorials; no fresh calibration RNG."""
from __future__ import annotations

import ast
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts/splicebert_register_operator_replication/audit_results_v2.py'
spec = importlib.util.spec_from_file_location('result_v2_audit_test_module', SCRIPT)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)
torch.set_num_threads(1)


def fixture(*, unequal=False, delta=.15, additive=False):
    generator = audit.fixture_generator()
    split = 4 if unequal else 6
    ids = [f'synthetic:chr1:{"+" if i < split else "-"}:b{i:03d}' for i in range(12)]
    strata = ['chr1:+'] * split + ['chr1:-'] * (12 - split)
    return generator.make_payload(ids, strata, generator.deterministic_latents(12),
                                  delta=delta, null=False, additive_only=additive)[0]


def reduced_fixture(**kwargs):
    reduced = audit.synthetic_reduction(fixture(**kwargs))
    reduced['sites'] = reduced['phase_blind']
    return reduced


def test_no_direct_producer_arithmetic_imports_and_no_v1_mutation():
    tree = ast.parse(SCRIPT.read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and ('register_operator_measurement' in name or 'register_operator_statistics' in name)
                   for name in imports)
    assert audit.digest(REPO / 'scripts/splicebert_register_operator_replication/audit_results.py') == (
        'da6a653d1cca40378b4713335cf67d72fca5eff147b31c6a9640ba2a74a4eb34')
    audit.verify_generator_unchanged()


def test_exhaustive_unequal_strata_covariance_recovers_sample_covariance():
    strata = ['a'] * 2 + ['b'] * 4
    x = np.array([[1., -4.], [3., 2.], [0., 1.], [2., 3.], [7., -2.], [-1., 4.]])
    indices = np.array([a+b for a in product(range(2), repeat=2) for b in product(range(2, 6), repeat=4)])
    replicates = audit.bootstrap_means(x, strata, indices)
    actual = np.cov(replicates, rowvar=False, ddof=0)
    expected = (2 * np.cov(x[:2], rowvar=False, ddof=1) + 4 * np.cov(x[2:], rowvar=False, ddof=1)) / 36
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
    uncorrected = np.cov(x[indices].mean(axis=1), rowvar=False, ddof=0)
    assert not np.allclose(uncorrected, expected)
    assert not np.allclose(uncorrected * 6/4, expected)
    np.testing.assert_allclose(replicates.mean(axis=0), x.mean(axis=0), atol=1e-14, rtol=0)


def test_residual_map_preserves_stratum_means_translation_and_paired_linearity():
    strata = ['a'] * 2 + ['b'] * 4
    x = np.arange(18, dtype=float).reshape(6, 3) / 7
    z = np.sin(x)
    y = audit.residual_correct(x, strata)
    for sl in (slice(0, 2), slice(2, 6)):
        np.testing.assert_allclose(y[sl].mean(axis=0), x[sl].mean(axis=0), atol=1e-15, rtol=0)
    np.testing.assert_allclose(audit.residual_correct(x+2, strata), y+2, atol=1e-15, rtol=0)
    np.testing.assert_allclose(audit.residual_correct(x-z, strata), y-audit.residual_correct(z, strata), atol=1e-15, rtol=0)
    assert np.array_equal(audit.residual_correct(np.ones((6, 2)), strata), np.ones((6, 2)))
    with pytest.raises(ValueError, match='singleton'):
        audit.residual_correct([0., 1., 2.], ['a', 'b', 'b'])


def test_direct_endpoint_path_matches_independent_counts_and_producer_signed_weights():
    from gfmcd.splicing.register_operator_statistics_v2 import make_inference_context
    strata = ['a'] * 2 + ['b'] * 4
    ids = [f'block{i}' for i in range(6)]
    context = make_inference_context(ids, strata, draws=128)
    samples = audit.bootstrap_indices(strata, draws=128)
    assert np.array_equal(context.indices, samples)
    values = np.column_stack((np.arange(6.) / 7, np.cos(np.arange(6.)), np.arange(6.) * 1e-9))
    direct = audit.bootstrap_means(values, strata, samples)
    counted = audit.bootstrap_means(values, strata, samples, weights=audit.resample_weights(samples, 6))
    np.testing.assert_allclose(direct, counted, atol=1e-15, rtol=0)
    np.testing.assert_allclose(direct, context.weights @ values, atol=1e-15, rtol=0)
    assert (context.weights < 0).any()
    np.testing.assert_allclose(context.weights.sum(axis=1), 1, atol=1e-15, rtol=0)


@pytest.mark.parametrize('unequal', [False, True])
def test_full_inference_from_raw_logits_matches_corrected_producer(unequal):
    from gfmcd.splicing.register_operator_measurement import reduce_score_payload
    from gfmcd.splicing.register_operator_statistics_v2 import analyze, make_inference_context
    payload = fixture(unequal=unequal)
    reduced = reduce_score_payload(payload)
    context = make_inference_context(payload['block_ids'], payload['strata'], draws=128)
    expected = analyze(reduced, context=context)
    independent = audit.synthetic_reduction(payload)
    audit.compare(reduced, independent, tolerance=0)
    independent['sites'] = independent['phase_blind']
    actual = audit.infer(independent, prereqs={'all': True}, numeric_passed=True, audit_passed=True, draws=128)
    assert audit.compare(expected, actual)['maximum_absolute_error'] < 1e-10
    assert actual['classification'] in audit.MATERIAL
    assert actual['primary_psi']['analytic_empirical_mde80'] > actual['primary_psi']['uncorrected_analytic_empirical_mde80']
    fast = audit.infer(independent, prereqs={'all': True}, numeric_passed=True, audit_passed=True, draws=128,
                       samples=context.indices, weights=audit.resample_weights(context.indices, 12))
    assert audit.compare(fast, actual, tolerance=1e-12)['maximum_absolute_error'] < 1e-12


def test_pseudo_correction_must_follow_endpoint_formation():
    reduced = reduced_fixture(unequal=True)
    samples = audit.bootstrap_indices(reduced['strata'], draws=128)
    actual = audit.infer(reduced, prereqs={'all': True}, numeric_passed=True, audit_passed=True, draws=128)
    matrices = reduced['matrices']['cc_max']
    diagonal, off = audit.pseudo_plan(reduced['block_ids'])
    flat = matrices.reshape(1, 12, 9)
    control = np.take_along_axis(flat, diagonal, axis=2).mean(axis=2)-np.take_along_axis(flat, off, axis=2).mean(axis=2)
    psi = np.array([r['cc_max']['psi'] for r in actual['block_endpoints']])
    formed = psi-control.mean(axis=0)
    expected = np.quantile(audit.bootstrap_means(formed, reduced['strata'], samples), .05)
    assert abs(actual['pseudo_phase']['target_minus_mean_control']['lower95'] - expected) < 1e-15
    bad_matrix = audit.residual_correct(matrices, reduced['strata']).reshape(1, 12, 9)
    bad_control = np.take_along_axis(bad_matrix, diagonal, axis=2).mean(axis=2)-np.take_along_axis(bad_matrix, off, axis=2).mean(axis=2)
    assert not np.allclose(bad_control.mean(axis=0), audit.residual_correct(control.mean(axis=0), reduced['strata']), atol=1e-9, rtol=0)


def test_pseudo_rank_preserves_coefficient_summation_at_cancellation_scale():
    from gfmcd.splicing.register_operator_statistics import make_inference_context, matching_endpoints
    ids, strata = [f'block{i}' for i in range(12)], ['a']*4+['b']*8
    context = make_inference_context(ids, strata, draws=32)
    matrix = (np.arange(12.)[:, None, None]+np.arange(3.)[None, :, None]
              + .13*np.arange(3.)[None, None, :])/17
    plan = audit.pseudo_plan(ids)
    actual = audit.pseudo_values(matrix, plan).mean(axis=1)
    expected = np.sum(context.coefficients*matrix[None], axis=(2, 3)).mean(axis=1)
    assert np.array_equal(actual, expected)
    target = matching_endpoints(matrix)['psi'].mean()
    assert (1+np.count_nonzero(actual >= target))/100 == .34
    diagonal, off = plan
    separate = (np.take_along_axis(matrix.reshape(1, 12, 9), diagonal, axis=2).mean(axis=2)
                - np.take_along_axis(matrix.reshape(1, 12, 9), off, axis=2).mean(axis=2)).mean(axis=1)
    # Algebraic rearrangement changes strict rank tests despite sub-tolerance effects.
    assert np.max(np.abs(actual-separate)) < 1e-15
    assert (1+np.count_nonzero(separate >= target))/100 == .23


def test_mde_corrected_analytic_identity_and_constant_cancellation():
    reduced = reduced_fixture(unequal=True)
    result = audit.infer(reduced, prereqs={'all': True}, numeric_passed=True, audit_passed=True, draws=512)
    psi = np.array([r['cc_max']['psi'] for r in result['block_endpoints']])
    expected_variance = sum(len(group) * np.var(group, ddof=1) for group in (psi[:4], psi[4:])) / 144
    assert abs(result['primary_psi']['analytic_empirical_mde80']-audit.FACTOR*np.sqrt(expected_variance)) < 1e-15
    samples = audit.bootstrap_indices(reduced['strata'], draws=32)
    constant = np.zeros((12, 2))
    assert np.array_equal(audit.bootstrap_means(constant, reduced['strata'], samples), np.zeros((32, 2)))


def test_tiny_active_coordinates_retained_and_every_family_replayed():
    reduced = reduced_fixture(unequal=True, additive=True, delta=0)
    samples = audit.bootstrap_indices(reduced['strata'], draws=128)
    tiny = np.arange(12.) * 1e-9
    assert audit.bootstrap_means(tiny, reduced['strata'], samples).std(ddof=1) > 0
    full = audit.infer(reduced, prereqs={'all': True}, numeric_passed=True, audit_passed=True, draws=128)
    replay = audit.replay_interval_families(reduced, samples)
    assert set(replay) == set(audit.FAMILIES)
    audit.compare(full, replay, tolerance=1e-12)
    # There is no coordinate cutoff beyond the unchanged strictly positive SD rule.
    source = SCRIPT.read_text()
    assert 'active = sd > 0' in source and 'active = se > 0' in source


@pytest.mark.parametrize('correct,calibrated,authorized', [(True, True, True), (True, False, False), (False, True, False), (False, False, False)])
def test_audit_correctness_never_bypasses_calibration(correct, calibrated, authorized):
    receipt = audit.calibration_authorization(correct, calibrated)
    assert receipt['status'] == ('PASS' if correct else 'FAIL')
    assert receipt['passed'] is correct
    assert receipt['calibration_passed'] is calibrated
    assert receipt['authorized_for_fidelity'] is authorized
    assert receipt['authorized_for_cc'] is False


@pytest.mark.parametrize('mutation', [{'status': 'FROZEN'}, {'inference_method': 'original'}, {'schema': 'gfmcd.register_operator.freeze.v1'}])
def test_v2_guard_rejects_v1_or_wrong_method_before_reading_data(tmp_path, mutation):
    freeze = {'status': 'FROZEN_V2', 'schema': 'gfmcd.register_operator.freeze.v2',
              'inference_method': audit.METHOD, 'base_freeze_sha256': audit.BASE_FREEZE_SHA256, **mutation}
    path = tmp_path/'freeze.json'
    path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match='separately frozen'):
        audit.verify_freeze(path)


def test_immutable_evidence_and_snapshot_mutation_guards(tmp_path):
    source, missing = tmp_path/'source.json', tmp_path/'missing.json'
    audit.write_exclusive(source, {'status': 'FAIL'})
    snapshot = {str(source): audit.digest(source), str(missing): None}
    audit.verify_input_snapshot(snapshot)
    with pytest.raises(FileExistsError):
        audit.write_exclusive(source, {'status': 'PASS'})
    missing.write_text('{}')
    with pytest.raises(ValueError, match='input changed'):
        audit.verify_input_snapshot(snapshot)


def test_replay_coverage_accounts_for_errors_without_authorizing_failure():
    report = {'complete': True, 'passed': False, 'execution_errors': [{'phase': 'null', 'index': 0}],
              'null_replay': {'null_raw_payloads_hash_verified': 999, 'raw_reductions_checked': 999,
                             'complete_inferences_checked': 999, 'interval_families_checked': 5994,
                             'direct_index_memorials': 3}}
    audit.verify_replay_coverage(report)
    assert audit.calibration_authorization(True, report['passed'])['authorized_for_fidelity'] is False
    report['null_replay']['complete_inferences_checked'] = 998
    with pytest.raises(ValueError, match='coverage is incomplete'):
        audit.verify_replay_coverage(report)


def test_native_failure_is_preserved_as_scored_but_not_independently_audited():
    native = {'passed': False, 'inference_method': audit.METHOD,
              'endpoints': {'cc_max': {'mean': .02, 'ci95': [-.01, .05]}}}
    row = audit.observed_stage({'status': 'COMPLETE', 'native_requests': [{}, {}],
                               'native_preflight': native})
    assert row['native_requests'] == 2 and row['cell_requests'] == 0
    assert row['native_preflight']['reported'] == native
    assert row['native_preflight']['verification_status'] == 'REPORTED_NOT_INDEPENDENTLY_RECOMPUTED'
    assert 'native_preflight' not in audit.observed_stage({'status': 'FAILED'})


def test_calibration_utc_chronology_requires_source_bound_smoke_first(tmp_path, monkeypatch):
    freeze_path, smoke, cal = (tmp_path/name for name in ('freeze.json', 'smoke.json', 'cal.json'))
    for path in (freeze_path, smoke, cal):
        path.write_text('{}')
    attempt = tmp_path/'cal.attempt.json'
    outputs = {'calibration_attempt': str(attempt)}
    for stage in ('fidelity', 'native', 'cells'):
        outputs[stage] = str(tmp_path/(stage+'.json'))
        outputs[stage+'_attempt'] = str(tmp_path/(stage+'.attempt.json'))
    freeze = {'frozen_at_utc': '2026-10-16T01:00:00Z', 'outputs': outputs, 'selection_sha256': 'selected', 'files': {}}
    data = {'freeze_sha256': audit.digest(freeze_path), 'inference_method': audit.METHOD,
            'selection_sha256': 'selected', 'frozen_files': {}, 'created_at_utc': '2026-10-16T01:02:00Z'}
    attempt.write_text(json.dumps(data))
    receipts = {'smoke': {'status': 'PASS'}, 'calibration': {'attempt_sha256': audit.digest(attempt),
                 'completed_at_utc': '2026-10-16T01:04:00Z'}}
    manifests = {'smoke': {'created_at_utc': '2026-10-16T01:01:00Z'},
                 'calibration': {'created_at_utc': '2026-10-16T01:05:00Z'}}
    monkeypatch.setattr(audit, 'check_manifest', lambda path, **kw: {'created_at_utc': '2026-10-16T01:03:00Z'})
    result = audit.chronology(freeze, receipts, manifests, {'smoke': smoke}, freeze_path)
    assert result['stages']['calibration']['attempt_start_timestamp_recorded'] is True
    receipts['smoke']['status'] = 'FAIL'
    with pytest.raises(ValueError, match='without smoke PASS'):
        audit.chronology(freeze, receipts, manifests, {'smoke': smoke}, freeze_path)


def test_classifier_keeps_instrument_failure_even_with_audit_pass():
    result = audit.infer(reduced_fixture(), prereqs={'calibration': False}, numeric_passed=True,
                         audit_passed=True, draws=128)
    assert result['classification'] == 'INSTRUMENT_UNRESOLVED'
    assert result['gates']['construction'] is False
