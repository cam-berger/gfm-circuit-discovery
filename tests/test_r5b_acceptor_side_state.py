from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from gfmcd.splicing.r5_stack_binding import (
    R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS,
    R3C_INSTRUMENT_REVISION,
    R3C_PATH_HEALTH_RULE,
    R3C_PATH_NAMES,
    R3C_TARGET_HEALTH_RULE,
    R3C_TARGET_HEALTH_SCHEMA,
)


REPO = Path(__file__).resolve().parents[1]
RUNNER = (
    REPO
    / "scripts/splicebert_acceptor_coupling/run_r5b_acceptor_side_state.py"
)
SPEC = importlib.util.spec_from_file_location("run_r5b_acceptor_side_state", RUNNER)
assert SPEC is not None and SPEC.loader is not None
r5b = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r5b
SPEC.loader.exec_module(r5b)


def _strong_primitive() -> np.ndarray:
    offset = r5b._easy_causal_primitive_offset()
    return np.repeat(offset[:, None], r5b.TARGET_N, axis=1)


def test_acceptor_only_registries_are_exact_and_derived_rows_cannot_drift():
    assert r5b.RANK_GRID == (1, 2, 4, 8, 16, 32)
    assert len(r5b.causal_primitive_names()) == 70
    assert len(r5b.causal_family_names()) == 136
    assert len(r5b.causal_boundary_edge_registry()) == 76
    primitive = np.arange(70 * 3, dtype=float).reshape(70, 3)
    family, names = r5b.derive_causal_family(primitive)
    lookup_p = {name: i for i, name in enumerate(r5b.causal_primitive_names())}
    lookup_f = {name: i for i, name in enumerate(names)}
    assert np.array_equal(
        family[lookup_f["target|S|specific"]],
        primitive[lookup_p["target|S|partner"]]
        - primitive[lookup_p["target|S|pseudo"]],
    )
    assert np.array_equal(
        family[lookup_f["target_minus_random|draw=07|N|partner"]],
        primitive[lookup_p["target|N|partner"]]
        - primitive[lookup_p["random|draw=07|N|partner"]],
    )
    source = RUNNER.read_text()
    assert "masked donor G margin" not in source
    assert '"masked_donor_endpoint_constructed": False' in source


def test_exact_two_fold_assignment_has_64_unique_genes_per_fold_and_is_deterministic():
    genes = [f"gene-{index:03d}" for index in range(128)]
    first = r5b.exact_two_fold_ids(genes)
    second = r5b.exact_two_fold_ids(list(reversed(genes)))
    by_gene_first = dict(zip(genes, first))
    by_gene_second = dict(zip(reversed(genes), second))
    assert by_gene_first == by_gene_second
    assert tuple(np.bincount(first).tolist()) == (64, 64)
    with pytest.raises(ValueError, match="128 unique"):
        r5b.exact_two_fold_ids(genes[:-1] + [genes[0]])


def test_crossfit_really_fits_only_opposite_fold_gene_means():
    genes = [f"gene-{index:03d}" for index in range(128)]
    folds = r5b.exact_two_fold_ids(genes)
    rows = np.random.default_rng(4).normal(size=(128, 40))
    fitted = r5b.crossfit_svd(
        rows,
        genes,
        folds,
        candidate_ranks=r5b.RANK_GRID,
        diagnostic_energy_target=0.90,
        group_average=True,
    )
    assert len(fitted.folds) == 2
    for fold in fitted.folds:
        assert fold.svd.n_fit_rows == 64
        assert fold.derivation_matrix_policy == "gene_mean_rows"
        assert len(fold.derivation_groups) == 64
        assert len(fold.evaluation_groups) == 64
        assert not set(fold.derivation_groups) & set(fold.evaluation_groups)
    projected = r5b.crossfit_project(rows, fitted, rank=32)
    assert projected.shape == rows.shape


def test_private_refit_svd_enforces_frozen_relative_numerical_rank():
    rank_one = np.outer(np.arange(1.0, 65.0), np.arange(1.0, 41.0))
    with pytest.raises(r5b.InstrumentError, match="relative tolerance"):
        r5b._svd_basis(rank_one, 32)
    assert r5b.SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE == pytest.approx(1e-10)


def test_every_causal_boundary_case_is_primitive_coherent_and_single_edge_bound():
    cases = r5b.causal_boundary_case_specs()
    assert len(cases) == 76
    assert len({case["case"] for case in cases}) == 76
    assert all(len(case["primitive_offset_sha256"]) == 64 for case in cases)
    plan = r5b.prospective_calibration_trial_count()
    assert plan["N_cal"] >= 6 * 76
    assert plan["scientific_fwer_claimed_from_calibration"] is False


def test_rank_classifier_uses_boundary_iut_and_strict_complete_family_intervals():
    genes = [f"gene-{index:03d}" for index in range(128)]
    strong = r5b.infer_causal_rank_family(
        _strong_primitive(),
        genes,
        rank=1,
        n_draws=499,
        seed=0,
        namespace="test-r5b-strong",
    )
    assert strong["family_size"] == 136
    assert len(strong["edge_tests"]) == 76
    assert strong["path_iut_p_value"] == pytest.approx(1 / 500)
    assert strong["random_control_gate"] is True

    boundary_case = r5b.causal_boundary_case_specs()[0]
    boundary = np.repeat(
        np.asarray(boundary_case["primitive_offset"])[:, None], 128, axis=1
    )
    unresolved = r5b.infer_causal_rank_family(
        boundary,
        genes,
        rank=1,
        n_draws=499,
        seed=0,
        namespace="test-r5b-boundary",
    )
    assert unresolved["random_control_gate"] is False
    assert unresolved["path_iut_p_value"] > r5b.RANK_ALPHA


def test_overlap_family_uses_disjoint_fixed_calibration_pool_and_absolute_refit_gate():
    observed = {rank: 0.80 for rank in r5b.RANK_GRID}
    refit = r5b.RefitOverlapIntervals(
        observed_by_rank=observed,
        lower_by_rank={rank: 0.70 for rank in r5b.RANK_GRID},
        simultaneous_deviation_critical=0.10,
        bootstrap_sha256="a" * 64,
        draw_ids_sha256="b" * 64,
        n_draws=20,
        seed=0,
        namespace="test-refit",
    )
    result = r5b.overlap_family_inference(
        observed,
        refit,
        width=64,
        n_calibration=49,
        n_test=49,
        seed=0,
    )
    fixed = result["fixed_calibration"]
    assert fixed["calibration_pool_sha256"] != fixed["test_pool_sha256"]
    assert fixed["n_calibration_draws"] == 49
    assert fixed["n_test_draws"] == 49
    assert fixed["calibration_first_draw_identity"]["namespace"] == (
        r5b.OVERLAP_CALIBRATION_NAMESPACE
    )
    assert fixed["test_first_draw_identity"]["namespace"] == (
        r5b.OVERLAP_TEST_NAMESPACE
    )
    assert fixed["calibration_draw_identity_set_sha256"] != fixed[
        "test_draw_identity_set_sha256"
    ]
    assert all(
        row["absolute_overlap_pass"] for row in result["by_rank"].values()
    )

    touching = r5b.RefitOverlapIntervals(
        observed_by_rank=observed,
        lower_by_rank={rank: r5b.OVERLAP_SESOI for rank in r5b.RANK_GRID},
        simultaneous_deviation_critical=0.60,
        bootstrap_sha256="c" * 64,
        draw_ids_sha256="d" * 64,
        n_draws=20,
        seed=0,
        namespace="test-refit-touching",
    )
    touching_result = r5b.overlap_family_inference(
        observed,
        touching,
        width=64,
        n_calibration=49,
        n_test=49,
        seed=0,
    )
    assert not any(
        row["random_overlap_gate"] for row in touching_result["by_rank"].values()
    )


def test_overlap_refit_calibration_starts_from_raw_matrices_and_exercises_all_routes():
    calibration = r5b.run_overlap_refit_calibration(
        width=64,
        n_refit_draws=50,
        n_calibration=49,
        n_test=49,
        seed=0,
    )
    assert calibration["all_pass"] is True
    assert set(calibration["geometries"]) == {
        "no_shared_subspace",
        "absolute_overlap_boundary",
        "strong_shared_subspace",
    }
    assert calibration["checks"]["touching_absolute_boundary_not_accepted"]
    assert calibration["checks"]["strong_shared_subspace_detected_all_ranks"]
    for record in calibration["geometries"].values():
        assert len(record["raw_combined_sha256"]) == 64
        assert record["inference"]["calibration_and_test_pools_disjoint"] is True


class _ToyTransform(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value)


class _ToyPredictions(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transform = _ToyTransform()
        self.decoder = torch.nn.Linear(40, 8)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.transform(value))


class _ToyCLS(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictions = _ToyPredictions()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.predictions(value)


def test_full_mlm_capability_covers_dense_head_transform_decoder_and_margin():
    torch.manual_seed(2)
    cls = _ToyCLS()
    clean = np.random.default_rng(2).normal(size=(8, 40)).astype(np.float32)
    corrupt = np.random.default_rng(3).normal(size=(8, 40)).astype(np.float32)
    with torch.no_grad():
        clean_logits = cls(torch.from_numpy(clean)).numpy()
        corrupt_logits = cls(torch.from_numpy(corrupt)).numpy()
    base_ids = {"A": 0, "C": 1, "G": 2, "T": 3}
    capability = r5b.full_mlm_capability(
        cls,
        h_clean=clean,
        h_partner_corrupt=corrupt,
        native_clean_logits=clean_logits,
        native_partner_logits=corrupt_logits,
        dense_clean_patch_logits=clean_logits,
        dense_self_patch_logits=corrupt_logits,
        base_token_ids=base_ids,
        device="cpu",
        batch_size=3,
        atol=1e-6,
    )
    assert capability["all_controls_pass"] is True
    assert capability["masked_donor_endpoint_constructed"] is False
    assert capability["candidate_projector_used"] is False

    tampered = clean_logits.copy()
    tampered[0, 0] += 0.1
    failed = r5b.full_mlm_capability(
        cls,
        h_clean=clean,
        h_partner_corrupt=corrupt,
        native_clean_logits=clean_logits,
        native_partner_logits=corrupt_logits,
        dense_clean_patch_logits=tampered,
        dense_self_patch_logits=corrupt_logits,
        base_token_ids=base_ids,
        device="cpu",
        batch_size=4,
        atol=1e-6,
    )
    assert failed["all_controls_pass"] is False


def test_intervention_execution_scores_target_pseudo_adjacent_and_32_nested_controls():
    torch.manual_seed(8)
    cls = _ToyCLS()
    scorer = r5b.FullMLMScorer(
        cls,
        {"A": 0, "C": 1, "G": 2, "T": 3},
        device="cpu",
        batch_size=32,
    )
    genes = tuple(f"gene-{index:03d}" for index in range(128))
    folds = r5b.exact_two_fold_ids(genes)
    per_fold_seen = {0: 0, 1: 0}
    strands = []
    for fold in folds:
        strands.append("+" if per_fold_seen[int(fold)] % 2 == 0 else "-")
        per_fold_seen[int(fold)] += 1
    rng = np.random.default_rng(8)
    h_partner = rng.normal(size=(128, 40))
    delta_partner = rng.normal(size=(128, 40))
    h_clean = h_partner + 0.15 * delta_partner
    h_pseudo = h_clean - 0.01 * rng.normal(size=(128, 40))
    h_adjacent_partner = rng.normal(size=(128, 40))
    h_adjacent_clean = h_adjacent_partner + rng.normal(size=(128, 40))
    capture = r5b.ResidualCapture(
        genes=genes,
        strands=tuple(strands),
        example_ids=tuple(f"example-{index:03d}" for index in range(128)),
        h_clean=h_clean,
        h_partner_corrupt=h_partner,
        h_pseudo_corrupt=h_pseudo,
        h_adjacent_clean_mean=h_adjacent_clean,
        h_adjacent_partner_corrupt_mean=h_adjacent_partner,
        m_clean=scorer.margins(h_clean),
        m_partner_corrupt=scorer.margins(h_partner),
        m_pseudo_corrupt=scorer.margins(h_pseudo),
        capability={"all_controls_pass": True},
    )
    fitted = r5b.crossfit_svd(
        h_clean - h_partner,
        genes,
        folds,
        candidate_ranks=r5b.RANK_GRID,
        diagnostic_energy_target=0.90,
        group_average=True,
    )
    adjacent = r5b.crossfit_svd(
        h_adjacent_clean - h_adjacent_partner,
        genes,
        folds,
        candidate_ranks=r5b.RANK_GRID,
        diagnostic_energy_target=0.90,
        group_average=True,
    )
    primitives, controls = r5b.intervention_primitives_by_rank(
        capture,
        scorer,
        fitted,
        adjacent,
        folds,
        seed=0,
        binding="test-binding",
    )
    assert set(primitives) == set(r5b.RANK_GRID)
    assert all(matrix.shape == (70, 128) for matrix in primitives.values())
    target_projection = r5b.crossfit_project(
        h_clean - h_partner, fitted, rank=1
    )
    expected_s = scorer.margins(h_partner + target_projection) - capture.m_partner_corrupt
    expected_n = capture.m_clean - scorer.margins(h_clean - target_projection)
    assert np.allclose(primitives[1][0], expected_s)
    assert np.allclose(primitives[1][1], expected_n)
    assert controls["draws_per_rank"] == 32
    assert controls["nested_prefix_grid"] == list(r5b.RANK_GRID)
    assert all(
        draw["per_row_energy_match_max_abs"] < 1e-10
        for rank in controls["basis_records_by_rank"].values()
        for draw in rank.values()
    )


def test_runtime_bundle_binds_selected_slice_fold_map_and_exact_reservation(tmp_path):
    genes = tuple(f"runtime-gene-{index:03d}" for index in range(128))
    examples = tuple(f"runtime-example-{index:03d}" for index in range(128))
    folds = r5b.exact_two_fold_ids(genes)
    per_fold_seen = {0: 0, 1: 0}
    strands = []
    for fold in folds:
        strands.append("+" if per_fold_seen[int(fold)] % 2 == 0 else "-")
        per_fold_seen[int(fold)] += 1
    rng = np.random.default_rng(17)
    partner = rng.normal(size=(128, 40))
    clean = partner + rng.normal(size=(128, 40))
    capture = r5b.ResidualCapture(
        genes=genes,
        strands=tuple(strands),
        example_ids=examples,
        h_clean=clean,
        h_partner_corrupt=partner,
        h_pseudo_corrupt=clean.copy(),
        h_adjacent_clean_mean=clean.copy(),
        h_adjacent_partner_corrupt_mean=partner.copy(),
        m_clean=np.ones(128),
        m_partner_corrupt=np.zeros(128),
        m_pseudo_corrupt=np.zeros(128),
        capability={"all_controls_pass": True},
    )
    maximum, map_package = r5b._runtime_max_rank_arrays(
        capture, binding="runtime-test", seed=0
    )
    max_bytes = r5b._npz_bytes(maximum)
    max_path = tmp_path / "bucket1-runtime-max.npz"
    max_path.write_bytes(max_bytes)
    cohort_audit = {
        "selected_by_bucket": {
            "0": {
                "ordered_rows_sha256": "0" * 64,
                "construction_sha256": "1" * 64,
                "gene_ids_sha256": "2" * 64,
            },
            "1": {
                "ordered_rows_sha256": "3" * 64,
                "construction_sha256": "4" * 64,
                "gene_ids_sha256": "5" * 64,
            },
        }
    }
    source = r5b.runtime_max_source_record(
        maximum,
        max_bytes,
        bundle_path=max_path,
        bucket=1,
        cohort_audit=cohort_audit,
        checkpoint_sha256="6" * 64,
        map_record=map_package,
        r5_stack_fingerprint_sha256="7" * 64,
    )
    loaded = r5b.load_and_validate_runtime_max_source(source)
    r5b.validate_runtime_max_source_binding(
        source,
        loaded,
        map_package["crossfit_map"],
        bucket=1,
        cohort_audit=cohort_audit,
        checkpoint_sha256="6" * 64,
        r5_stack_fingerprint_sha256="7" * 64,
    )
    selected = r5b._selected_runtime_arrays(loaded, 8)
    assert set(selected) == {
        "selected_rank",
        "selected_bases_by_fold",
        "random_bases_by_draw_fold",
        "row_fold_ids",
        "ordered_row_ids",
    }
    assert selected["selected_rank"].shape == ()
    assert int(selected["selected_rank"]) == 8
    assert selected["selected_bases_by_fold"].shape == (2, 8, 40)
    assert selected["random_bases_by_draw_fold"].shape == (32, 2, 8, 40)
    selected_bytes = r5b._npz_bytes(selected)
    contract = r5b.runtime_bundle_contract(
        selected,
        selected_bytes,
        bundle_path=tmp_path / "bucket1-selected.npz",
        bucket=1,
        cohort_audit=cohort_audit,
        checkpoint_sha256="6" * 64,
        selected_rank=8,
        crossfit_map_sha256=map_package["crossfit_map_sha256"],
        source_payload_sha256=source["source_payload_sha256"],
        source_artifact_sha256=None,
        max_rank_source=source,
        r5_stack_fingerprint_sha256="7" * 64,
    )
    assert contract["schema"] == "gfmcd.r5b.r5c_runtime_contract/1"
    assert contract["bucket"] == 1
    assert contract["r5_stack_fingerprint_sha256"] == "7" * 64
    assert contract["cohort_sha256"] == "3" * 64
    assert set(contract["runtime_array_sha256"]) == {
        "selected_bases_by_fold",
        "random_bases_by_draw_fold",
        "row_fold_ids",
        "ordered_row_ids",
    }
    assert "selected_rank" not in contract["runtime_array_shapes"]
    changed = json.loads(json.dumps(source))
    changed["crossfit_map_sha256"] = "f" * 64
    with pytest.raises(r5b.InstrumentError, match="reservation binding differs"):
        r5b.validate_runtime_max_source_binding(
            changed,
            loaded,
            map_package["crossfit_map"],
            bucket=1,
            cohort_audit=cohort_audit,
            checkpoint_sha256="6" * 64,
            r5_stack_fingerprint_sha256="7" * 64,
        )


def test_genuine_preflight_runtime_source_authenticates_joint_stack_fingerprint(
    tmp_path, monkeypatch
):
    genes = tuple(f"preflight-gene-{index:03d}" for index in range(128))
    examples = tuple(f"preflight-example-{index:03d}" for index in range(128))
    rng = np.random.default_rng(31)
    partner = rng.normal(size=(128, 40))
    clean = partner + rng.normal(size=(128, 40))
    capture = r5b.ResidualCapture(
        genes=genes,
        strands=tuple("+" for _ in genes),
        example_ids=examples,
        h_clean=clean,
        h_partner_corrupt=partner,
        h_pseudo_corrupt=clean.copy(),
        h_adjacent_clean_mean=clean.copy(),
        h_adjacent_partner_corrupt_mean=partner.copy(),
        m_clean=np.ones(128),
        m_partner_corrupt=np.zeros(128),
        m_pseudo_corrupt=np.zeros(128),
        capability={"all_controls_pass": True},
    )
    maximum, map_package = r5b._runtime_max_rank_arrays(
        capture, binding="genuine-preflight", seed=0
    )
    max_bytes = r5b._npz_bytes(maximum)
    max_path = tmp_path / "preflight.runtime-max.npz"
    max_path.write_bytes(max_bytes)

    binding_payload = json.loads(json.dumps(r5b.instrument_binding_payload()))
    local_fingerprint = r5b.instrument_fingerprint(binding_payload)
    stack_fingerprint = binding_payload["r5_stack_fingerprint_sha256"]
    assert local_fingerprint != stack_fingerprint
    cohort_audit = {
        "selected_by_bucket": {
            "0": {
                "ordered_rows_sha256": "1" * 64,
                "construction_sha256": "2" * 64,
                "gene_ids_sha256": "3" * 64,
            }
        }
    }
    source = r5b.runtime_max_source_record(
        maximum,
        max_bytes,
        bundle_path=max_path,
        bucket=0,
        cohort_audit=cohort_audit,
        checkpoint_sha256="4" * 64,
        map_record=map_package,
        r5_stack_fingerprint_sha256=stack_fingerprint,
    )
    record = {
        "schema": "gfmcd.r5b.acceptor_side_state/1",
        "stage": "preflight",
        "target_bindable_preflight": True,
        "all_instrument_gates_pass": True,
        "calibration": {"all_pass": True},
        "model_free_audit": {"all_pass": True},
        "instrument_fingerprint_sha256": local_fingerprint,
        "instrument_binding_payload": binding_payload,
        "r5_stack_binding": binding_payload["r5_stack_binding"],
        "r5_stack_fingerprint_sha256": stack_fingerprint,
        "r5c_runtime_max_source_by_bucket": {"0": source},
        "r5c_runtime_crossfit_map_by_bucket": {
            "0": map_package["crossfit_map"]
        },
        "cohort_audit": cohort_audit,
        "checkpoint_identity": {"bundle_sha256": "4" * 64},
        "analysis": {
            "capture_digests": {
                "ordered_example_ids_sha256": source[
                    "ordered_row_ids_sha256"
                ]
            },
            "fold_assignment": {
                "fold_ids": map_package["crossfit_map"]["row_fold_ids"]
            },
        },
    }
    artifact = tmp_path / "preflight.json"
    artifact.write_text(json.dumps(record, sort_keys=True))
    artifact_hash = r5b.sha256_file(artifact)
    manifest = {
        "git_dirty": False,
        "outputs": [
            {
                "path": str(artifact),
                "size_bytes": artifact.stat().st_size,
                "sha256": artifact_hash,
            },
            {
                "path": str(max_path),
                "size_bytes": max_path.stat().st_size,
                "sha256": r5b.sha256_file(max_path),
            },
        ],
    }
    artifact.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(r5b, "FROZEN_PREFLIGHT_ARTIFACT_SHA256", artifact_hash)
    monkeypatch.setattr(
        r5b,
        "validate_analysis_replay",
        lambda _analysis: {"all_replayed": True},
    )

    verified, package = r5b.verify_preflight_artifact(
        artifact,
        fingerprint=local_fingerprint,
        binding_payload=binding_payload,
    )
    assert verified["r5_stack_fingerprint_sha256"] == stack_fingerprint
    assert package["replay"]["all_replayed"] is True


def test_checkpoint_digest_is_r3c_r5a_compatible_relative_path_bundle(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    weights = model / "weights.bin"
    weights.write_bytes(b"weights")
    identity = r5b.checkpoint_identity(str(model))
    expected = [
        {
            "relative_path": row["relative_path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in identity["files"]
    ]
    assert [row["relative_path"] for row in expected] == [
        "config.json",
        "weights.bin",
    ]
    assert identity["bundle_sha256"] == r5b.canonical_sha256(expected)
    assert "R3c-compatible" in identity["aggregate_schema"]


def test_entry_chain_consumes_shared_r5_contract_and_binds_checkpoint(
    tmp_path, monkeypatch
):
    checkpoint = "a" * 64
    source_identity = {
        "gtf": {"label": "test-gtf", "size_bytes": 1, "sha256": "1" * 64},
        "fasta": {"label": "test-fasta", "size_bytes": 2, "sha256": "2" * 64},
    }
    historical = ["historical"]
    bucket_genes = {"0": ["r3c-a"], "1": ["r3c-b"]}
    r3c_genes = ["r3c-a", "r3c-b"]
    census = {
        "historical_r3_gene_count": 1,
        "historical_r3_gene_ids_sha256": r5b.canonical_sha256(historical),
        "r3c_gene_count": 2,
        "r3c_gene_ids_sha256": r5b.canonical_sha256(r3c_genes),
    }
    stack = {"schema": "gfmcd.r5.stack_binding/1", "runtime_identity": {"test": True}}
    stack_fingerprint = r5b.canonical_sha256(stack)
    monkeypatch.setattr(r5b, "R5_EXPECTED_SOURCE_IDENTITY", source_identity)
    monkeypatch.setattr(r5b, "R5_EXPECTED_CENSUS_IDENTITY", census)
    monkeypatch.setattr(r5b, "EXPECTED_COHORT_CENSUS", {"test": "census"})
    monkeypatch.setattr(r5b, "build_r5_stack_binding", lambda **_kwargs: stack)
    monkeypatch.setattr(r5b, "r5_stack_fingerprint", lambda value: r5b.canonical_sha256(value))
    monkeypatch.setattr(
        r5b,
        "validate_serialized_r5_stack",
        lambda record, current: {"verified": record["r5_stack_binding"] == current},
    )
    monkeypatch.setattr(
        r5b,
        "validate_expected_r5_census_identity",
        lambda _audit: {"test": "census"},
    )
    health_suppressed = set(R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS)
    health_authorized = [
        name for name in R3C_PATH_NAMES if name not in health_suppressed
    ]
    r3c_record = {
        "schema": r5b.R3C_ENTRY_SCHEMA,
        "instrument_revision": R3C_INSTRUMENT_REVISION,
        "stage": "target",
        "status": "CONFIRMATORY",
        "tier": "HELD_OUT_TARGET",
        "split": "val",
        "gene_bucket": 1,
        "verdict": "HELD_OUT_ACCEPTOR_HEAD_CONFIRMATION_COMPLETE",
        "all_gates_pass": True,
        "sealed_split_opened": False,
        "protocol_status": "FROZEN_PROTOCOL_AND_INSTRUMENT_VERIFIED",
        "primary_estimand": "raw masked-acceptor-G logit-margin component drop",
        "paths": {
            name: {
                "classification": (
                    "HEALTH_FAILED" if name in health_suppressed else "SUPPORTED"
                ),
                "health": {
                    "ok": name not in health_suppressed,
                    "preflight_health_ok": name not in health_suppressed,
                    "current_health_ok": True,
                    "authorization_rule": R3C_PATH_HEALTH_RULE,
                },
            }
            for name in R3C_PATH_NAMES
        },
        "path_health_authorization": {
            "schema": R3C_TARGET_HEALTH_SCHEMA,
            "rule": R3C_TARGET_HEALTH_RULE,
            "can_only_shrink_from_preflight": True,
            "preflight_authorized_paths": health_authorized,
            "preflight_suppressed_paths": list(
                R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS
            ),
            "current_healthy_paths": list(R3C_PATH_NAMES),
            "effective_authorized_paths": health_authorized,
            "effective_suppressed_paths": list(
                R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS
            ),
        },
        "family": {
            "production_path_preflight": {
                "all_health_eligible_path_plants_powered": True,
                "health_eligible_paths": list(R3C_PATH_NAMES),
            }
        },
        "construction": {
            "schema": "gfmcd.r3c.acceptor_only_construction/1",
            "all_single_acceptor_G": True,
            "ordered_rows_sha256": "3" * 64,
            "policy": {
                "readout": "mask exactly one G at next_acceptor_pos - 1",
                "paired_operator": "GT -> AC at donor_pos",
                "donor_readout_constructed": False,
            },
        },
        "instrument_binding_payload": {"readout": "next_acceptor_pos - 1 only"},
        "instrument_fingerprint_sha256": "4" * 64,
        "input_snapshot_sha256": "5" * 64,
        "git_authorization_identity": {"commit": "6" * 40, "dirty": False},
        "gates": {"test_gate": True},
        "hashes": {
            "checkpoint_aggregate_sha256": checkpoint,
            "sources": source_identity,
        },
        "cohort_audit": {
            "historical_reconstruction": {"excluded_gene_ids": historical},
            "current_census": {
                "fixed_selection_gene_ids_by_bucket": {
                    bucket: {
                        "gene_ids": genes,
                        "gene_ids_sha256": r5b.canonical_sha256(genes),
                    }
                    for bucket, genes in bucket_genes.items()
                }
            },
        },
    }
    r3c_path = tmp_path / "r3c.json"
    r5a_path = tmp_path / "r5a.json"
    r3c_path.write_text(json.dumps(r3c_record))
    r3c_manifest_path = r3c_path.with_suffix(".manifest.json")
    r3c_manifest_path.write_text(
        json.dumps(
            {
                "experiment_id": "splicebert_acceptor_coupling:r3c_acceptor_head_confirmation",
                "status": "CONFIRMATORY",
                "git_dirty": False,
                "git_commit": "6" * 40,
                "config": {
                    "stage": "target",
                    "gene_bucket": 1,
                    "instrument_fingerprint_sha256": "4" * 64,
                    "construction_sha256": "3" * 64,
                    "input_snapshot_sha256": "5" * 64,
                },
                "outputs": [
                    {
                        "path": str(r3c_path),
                        "size_bytes": r3c_path.stat().st_size,
                        "sha256": r5b.sha256_file(r3c_path),
                    }
                ],
            }
        )
    )
    registry_payload = {
        "schema": "gfmcd.r5.exclusion_registry/1",
        "derivation": "authenticated R3c target cohort audit; no free gene-list input",
        "r3c_artifact": {
            "path": str(r3c_path),
            "size_bytes": r3c_path.stat().st_size,
            "sha256": r5b.sha256_file(r3c_path),
        },
        "r3c_manifest": {
            "path": str(r3c_manifest_path),
            "size_bytes": r3c_manifest_path.stat().st_size,
            "sha256": r5b.sha256_file(r3c_manifest_path),
        },
        "source_identity": source_identity,
        "historical_r3_gene_ids": historical,
        "r3c_gene_ids_by_bucket": bucket_genes,
        "r3c_gene_ids": r3c_genes,
        "expected_census_identity_sha256": r5b.canonical_sha256(census),
    }
    registry_record = registry_payload | {
        "record_sha256": r5b.canonical_sha256(registry_payload)
    }
    registry_path = tmp_path / "r5a.exclusion-registry.json"
    registry_path.write_text(json.dumps(registry_record))
    registry_identity = {
        "path": str(registry_path),
        "size_bytes": registry_path.stat().st_size,
        "sha256": r5b.sha256_file(registry_path),
        "record_sha256": registry_record["record_sha256"],
    }
    r5a_record = {
        "schema": "gfmcd.r5a.acceptor_address_binding/1",
        "stage": "target",
        "status": "L3H1_DONOR_LOCAL_READER_SUPPORTED",
        "all_instrument_gates_pass": True,
        "bucket": 1,
        "instrument_fingerprint_sha256": "7" * 64,
        "construction_sha256": "c" * 64,
        "git": {"commit": "8" * 40, "dirty": False},
        "r5_stack_binding": stack,
        "r5_stack_fingerprint_sha256": stack_fingerprint,
        "instrument_binding": {
            "r5_stack_binding": stack,
            "r5_stack_fingerprint_sha256": stack_fingerprint,
        },
        "runtime_identity": stack["runtime_identity"],
        "checkpoint": {"aggregate_sha256": checkpoint},
        "source_identity": source_identity,
        "cohort_audit": {"test": "audit"},
        "r3c_entry": {
            "path": str(r3c_path),
            "size_bytes": r3c_path.stat().st_size,
            "sha256": r5b.sha256_file(r3c_path),
            "manifest": {
                "path": str(r3c_manifest_path),
                "size_bytes": r3c_manifest_path.stat().st_size,
                "sha256": r5b.sha256_file(r3c_manifest_path),
            },
        },
        "r5_exclusion_registry": {
            "identity": registry_identity,
            "record": registry_record,
            "derived_from_authenticated_r3c": True,
        },
        "r5_entry_contract": {
            "schema": "gfmcd.r5.entry_contract/1",
            "instrument_gates_pass": True,
            "claim_eligible": True,
            "verdict": "L3H1_DONOR_LOCAL_READER_SUPPORTED",
            "acceptor_G_only": True,
            "masked_donor_endpoint_constructed": False,
            "checkpoint_sha256": checkpoint,
            "cohort_sha256": "b" * 64,
            "construction_sha256": "c" * 64,
            "r5_stack_fingerprint_sha256": stack_fingerprint,
            "upstream_r3c_artifact_sha256": r5b.sha256_file(r3c_path),
            "upstream_r3c_manifest_sha256": r5b.sha256_file(r3c_manifest_path),
            "exclusion_registry_sha256": registry_identity["sha256"],
            "exclusion_registry_record_sha256": registry_identity["record_sha256"],
        },
    }
    r5a_path.write_text(json.dumps(r5a_record))
    r5a_manifest_path = r5a_path.with_suffix(".manifest.json")
    r5a_manifest_path.write_text(
        json.dumps(
            {
                "experiment_id": "splicebert_acceptor_coupling:r5a_acceptor_address_binding",
                "status": "L3H1_DONOR_LOCAL_READER_SUPPORTED",
                "git_dirty": False,
                "git_commit": "8" * 40,
                "config": {
                    "stage": "target",
                    "bucket": 1,
                    "instrument_fingerprint_sha256": "7" * 64,
                    "r5_stack_fingerprint_sha256": stack_fingerprint,
                    "checkpoint_aggregate_sha256": checkpoint,
                    "construction_sha256": "c" * 64,
                },
                "outputs": [
                    {
                        "path": str(r5a_path),
                        "size_bytes": r5a_path.stat().st_size,
                        "sha256": r5b.sha256_file(r5a_path),
                    },
                    registry_identity,
                ],
            }
        )
    )
    monkeypatch.setattr(r5b, "FROZEN_R3C_ARTIFACT_SHA256", r5b.sha256_file(r3c_path))
    monkeypatch.setattr(r5b, "FROZEN_R5A_ARTIFACT_SHA256", r5b.sha256_file(r5a_path))
    entry = r5b.verify_entry_artifacts(r3c_path, r5a_path)
    assert entry["entry_chain_pass"] is True
    assert entry["checkpoint_sha256"] == checkpoint
    assert entry["r5a_cohort_sha256"] == "b" * 64
    assert entry["exclusion_registry"]["sha256"] == registry_identity["sha256"]

    changed = json.loads(r5a_path.read_text())
    changed["r5_entry_contract"]["checkpoint_sha256"] = "d" * 64
    r5a_path.write_text(json.dumps(changed))
    r5a_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "splicebert_acceptor_coupling:r5a_acceptor_address_binding",
                "status": "L3H1_DONOR_LOCAL_READER_SUPPORTED",
                "git_dirty": False,
                "git_commit": "8" * 40,
                "config": {
                    "stage": "target",
                    "bucket": 1,
                    "instrument_fingerprint_sha256": "7" * 64,
                    "r5_stack_fingerprint_sha256": stack_fingerprint,
                    "checkpoint_aggregate_sha256": checkpoint,
                    "construction_sha256": "c" * 64,
                },
                "outputs": [
                    {
                        "path": str(r5a_path),
                        "size_bytes": r5a_path.stat().st_size,
                        "sha256": r5b.sha256_file(r5a_path),
                    },
                    registry_identity,
                ],
            }
        )
    )
    monkeypatch.setattr(r5b, "FROZEN_R5A_ARTIFACT_SHA256", r5b.sha256_file(r5a_path))
    with pytest.raises(r5b.TargetSealError, match="checkpoint identities differ"):
        r5b.verify_entry_artifacts(r3c_path, r5a_path)


def test_runner_fingerprint_normalization_removes_only_literal_seal_values(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    template = (
        "FROZEN_PROTOCOL_SHA256: str | None = {value}\n"
        "FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = {value}\n"
        "UNCHANGED = 'scientific-method'\n"
    )
    first.write_text(template.format(value="None"))
    second.write_text(template.format(value=repr("a" * 64)))
    assert r5b.normalized_runner_record(first)["sha256"] == r5b.normalized_runner_record(
        second
    )["sha256"]
    second.write_text(second.read_text().replace("scientific-method", "changed-method"))
    assert r5b.normalized_runner_record(first)["sha256"] != r5b.normalized_runner_record(
        second
    )["sha256"]


def test_target_preflight_context_rejects_method_source_and_runtime_mismatch():
    stack = {"schema": "gfmcd.r5.stack_binding/1"}
    binding = {
        "schema": "gfmcd.r5b.instrument_binding/1",
        "r5_stack_binding": stack,
        "r5_stack_fingerprint_sha256": r5b.canonical_sha256(stack),
    }
    fingerprint = r5b.canonical_sha256(binding)
    entry = {
        "r3c": {"sha256": "1" * 64},
        "r5a": {"sha256": "2" * 64},
        "checkpoint_sha256": "3" * 64,
        "r5a_cohort_sha256": "4" * 64,
        "r5a_construction_sha256": "5" * 64,
        "r5_stack_binding": stack,
        "r5_stack_fingerprint_sha256": binding[
            "r5_stack_fingerprint_sha256"
        ],
        "exclusion_registry": {"sha256": "6" * 64},
        "entry_chain_pass": True,
    }
    source = {"gtf": {"sha256": "7" * 64}}
    cohort = {
        "selected_by_bucket": {
            "0": {
                "ordered_rows_sha256": "8" * 64,
                "construction_sha256": "9" * 64,
            }
        }
    }
    registry = {"sha256": "6" * 64}
    checkpoint = {"bundle_sha256": "3" * 64}
    runtime = {"python": "test"}
    record = {
        "bucket": 0,
        "instrument_binding_payload": binding,
        "instrument_fingerprint_sha256": fingerprint,
        "r5_stack_binding": stack,
        "r5_stack_fingerprint_sha256": binding[
            "r5_stack_fingerprint_sha256"
        ],
        "entry_chain": entry,
        "source_identity": source,
        "cohort_audit": cohort,
        "cohort_registry_identity": registry,
        "checkpoint_identity": checkpoint,
        "runtime_identity": runtime,
        "endpoint": "single masked acceptor G hard margin",
        "masked_donor_endpoint_constructed": False,
        "r5_entry_contract": {
            "cohort_sha256": "8" * 64,
            "construction_sha256": "9" * 64,
        },
    }
    assert r5b.validate_preflight_context(
        record,
        binding_payload=binding,
        fingerprint=fingerprint,
        entry_chain=entry,
        source_identity=source,
        cohort_audit=cohort,
        registry_identity=registry,
        checkpoint=checkpoint,
        runtime=runtime,
    )["all_match"]
    for field, changed in (
        ("instrument_fingerprint_sha256", "0" * 64),
        ("source_identity", {"gtf": {"sha256": "0" * 64}}),
        ("runtime_identity", {"python": "different"}),
    ):
        tampered = json.loads(json.dumps(record))
        tampered[field] = changed
        with pytest.raises(r5b.TargetSealError, match="context differs"):
            r5b.validate_preflight_context(
                tampered,
                binding_payload=binding,
                fingerprint=fingerprint,
                entry_chain=entry,
                source_identity=source,
                cohort_audit=cohort,
                registry_identity=registry,
                checkpoint=checkpoint,
                runtime=runtime,
            )


def test_sufficient_statistics_are_exact_one_row_per_gene_and_tamper_evident():
    genes = [f"gene-{index:03d}" for index in range(128)]
    primitive = np.random.default_rng(7).normal(size=(70, 128))
    record = r5b.gene_sufficient_statistics(
        primitive, r5b.causal_primitive_names(), genes
    )
    replay, names, replay_genes = r5b.reconstruct_gene_sufficient_statistics(record)
    assert tuple(names) == r5b.causal_primitive_names()
    assert replay_genes == sorted(genes)
    original_by_gene = {gene: primitive[:, i] for i, gene in enumerate(genes)}
    assert np.allclose(
        replay, np.stack([original_by_gene[gene] for gene in replay_genes], axis=1)
    )
    changed = json.loads(json.dumps(record))
    changed["genes"][0]["sums"][0] += 0.01
    with pytest.raises(ValueError, match="digest differs"):
        r5b.reconstruct_gene_sufficient_statistics(changed)


def test_model_free_audit_passes_without_model_data_or_bucket_access():
    audit = r5b.run_model_free_audit()
    assert audit["all_pass"] is True
    assert audit["checks"]["no_model_or_data_access"] is True
    assert audit["checks"]["no_masked_donor_endpoint_constructed"] is True
    assert audit["cohort_contract"]["schema"] == "gfmcd.r5.acceptor_mechanism_cohort/2"


@pytest.mark.parametrize("stage", ["preflight", "target"])
def test_model_stage_hard_stops_on_literal_none_pins_before_accessing_paths(
    tmp_path, stage
):
    missing = tmp_path / "must-not-be-opened.json"
    command = [
        sys.executable,
        str(RUNNER),
        "--stage",
        stage,
        "--gtf",
        str(missing),
        "--fasta",
        str(missing),
        "--exclusion-registry",
        str(missing),
        "--r3c-artifact",
        str(missing),
        "--r5a-artifact",
        str(missing),
        "--preflight-artifact",
        str(missing),
        "--out",
        str(tmp_path / f"{stage}.json"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode != 0
    combined = completed.stdout + completed.stderr
    assert "hard-blocked before path access" in combined
    assert "No such file" not in combined
    assert not (tmp_path / f"{stage}.json").exists()


def test_atomic_output_refuses_overwrite(tmp_path):
    destination = tmp_path / "artifact.json"
    r5b.atomic_write_new(destination, b"first\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        r5b.atomic_write_new(destination, b"second\n")
    assert destination.read_bytes() == b"first\n"


def test_result_manifest_authenticates_transient_runtime_bundle(tmp_path):
    output = tmp_path / "audit.json"
    bundle = tmp_path / "audit.runtime-max.npz"
    payload = b"authenticated-runtime-bundle"
    r5b._write_result_and_manifest(
        {"schema": "test", "stage": "audit"},
        output,
        stage="audit",
        input_paths=(RUNNER,),
        git_identity=None,
        extra_outputs={bundle: payload},
    )
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    by_path = {row["path"]: row for row in manifest["outputs"]}
    assert set(by_path) == {str(output), str(bundle)}
    assert by_path[str(bundle)]["size_bytes"] == len(payload)
    assert by_path[str(bundle)]["sha256"] == r5b.hashlib.sha256(payload).hexdigest()
    assert bundle.read_bytes() == payload
