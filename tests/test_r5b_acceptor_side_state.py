from __future__ import annotations

import importlib.util
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


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


def test_primary_state_is_the_upstream_l3h1_output_projected_local_writer():
    assert r5b.R5B_SCHEMA.endswith("/2")
    assert r5b.WRITER_LAYER == 3
    assert r5b.WRITER_HEAD == 1
    assert r5b.WRITER_SITE_ID == r5b.PRIMARY_LOCAL_WRITER_SITE_ID
    assert r5b.WRITER_SITE_ID == "L3H1.output_projected@acceptor_query"
    assert r5b.DENSE_PATH_ROLE == "primary"
    assert r5b.SUBSPACE_PATH_ROLE == "nested_candidate_only"
    assert r5b.MIN_PRODUCTION_INFERENCE_DRAWS >= 10_000
    source = RUNNER.read_text()
    assert "final masked-acceptor residual" not in source
    assert "final-residual U_A" not in source
    assert "L2H15_ADDRESS_REUSE" not in source
    assert "L2H15_mean_component" not in source


def test_output_projected_head_contribution_is_exact_and_excludes_shared_bias():
    torch.manual_seed(4)
    context = torch.randn(3, 7, 12, dtype=torch.float64)
    weight = torch.randn(12, 12, dtype=torch.float64)
    bias = torch.randn(12, dtype=torch.float64)
    contribution = r5b.output_projected_head_contribution(
        context,
        weight,
        head_index=1,
        num_heads=3,
    )
    isolated = torch.zeros_like(context)
    isolated[..., 4:8] = context[..., 4:8]
    expected = torch.nn.functional.linear(isolated, weight, None)
    assert torch.equal(contribution, expected)
    assert torch.allclose(
        torch.nn.functional.linear(context, weight, bias),
        sum(
            r5b.output_projected_head_contribution(
                context, weight, head_index=head, num_heads=3
            )
            for head in range(3)
        )
        + bias,
    )


def test_scored_provider_replays_true_post_wo_contribution_in_residual_width():
    rng = np.random.default_rng(5)
    n, head_width, residual_width = 8, 3, 12
    clean_context = rng.normal(size=(n, head_width))
    corrupt_context = rng.normal(size=(n, head_width))
    wo_block = rng.normal(size=(residual_width, head_width))
    clean_writer = clean_context @ wo_block.T
    corrupt_writer = corrupt_context @ wo_block.T
    replay = r5b.replay_output_projected_writer_provider(
        {
            "writer_site_id": r5b.WRITER_SITE_ID,
            "clean_head_context": clean_context,
            "corrupt_head_context": corrupt_context,
            "w_o_column_block": wo_block,
            "clean_writer_contribution": clean_writer,
            "corrupt_writer_contribution": corrupt_writer,
            "writer_delta": clean_writer - corrupt_writer,
            "numeric_provenance": {
                "schema": "gfmcd.r5b.post_wo_numeric_provenance/1",
                "checkpoint_operator_dtype": "float64",
                "head_context_dtype": "float64",
                "w_o_column_block_dtype": "float64",
                "writer_contribution_dtype": "float64",
                "replay_operator": "torch.nn.functional.linear",
                "relative_tolerance": 0.0,
                "absolute_tolerance": 1e-5,
            },
            "bias_included": False,
        }
    )
    assert replay["writer_delta"].shape == (n, residual_width)
    assert replay["residual_width"] == residual_width
    assert replay["head_width"] == head_width
    assert replay["bias_included"] is False
    assert replay["w_o_column_block_sha256"] == r5b.array_sha256(wo_block)
    changed = copy.deepcopy(replay["provider_record"])
    changed["writer_delta"][0][0] += 1.0
    with pytest.raises(r5b.InstrumentError, match="post-W_O writer delta"):
        r5b.replay_output_projected_writer_provider(changed)


def test_float32_checkpoint_projection_replays_at_realistic_splicebert_width():
    """JSON replay preserves the checkpoint's float32 64 -> 768 operator truth."""

    generator = torch.Generator().manual_seed(20260817)
    clean_context = torch.randn(5, 64, generator=generator, dtype=torch.float32)
    corrupt_context = torch.randn(5, 64, generator=generator, dtype=torch.float32)
    wo_block = torch.randn(768, 64, generator=generator, dtype=torch.float32)
    clean_writer = torch.nn.functional.linear(clean_context, wo_block, None)
    corrupt_writer = torch.nn.functional.linear(corrupt_context, wo_block, None)
    record = {
        "writer_site_id": r5b.WRITER_SITE_ID,
        "clean_head_context": clean_context.numpy().tolist(),
        "corrupt_head_context": corrupt_context.numpy().tolist(),
        "w_o_column_block": wo_block.numpy().tolist(),
        "clean_writer_contribution": clean_writer.numpy().tolist(),
        "corrupt_writer_contribution": corrupt_writer.numpy().tolist(),
        "writer_delta": (clean_writer - corrupt_writer).numpy().tolist(),
        "numeric_provenance": {
            "schema": "gfmcd.r5b.post_wo_numeric_provenance/1",
            "checkpoint_operator_dtype": "float32",
            "head_context_dtype": "float32",
            "w_o_column_block_dtype": "float32",
            "writer_contribution_dtype": "float32",
            "replay_operator": "torch.nn.functional.linear",
            "relative_tolerance": 0.0,
            "absolute_tolerance": 1e-5,
        },
        "bias_included": False,
    }

    replay = r5b.replay_output_projected_writer_provider(record)

    assert replay["numeric_provenance"] == record["numeric_provenance"]
    changed = copy.deepcopy(record)
    changed["clean_writer_contribution"][0][0] += 1e-3
    with pytest.raises(r5b.InstrumentError, match="post-W_O writer delta"):
        r5b.replay_output_projected_writer_provider(changed)
    changed_tolerance = copy.deepcopy(record)
    changed_tolerance["numeric_provenance"]["absolute_tolerance"] = 1e-3
    with pytest.raises(r5b.InstrumentError, match="numeric provenance"):
        r5b.replay_output_projected_writer_provider(changed_tolerance)


def test_disposition_schema_and_three_frozen_terminal_statuses_are_exported():
    assert r5b.R5B_DISPOSITION_SCHEMA == (
        "gfmcd.r5b.local_writer_basis_disposition/2"
    )
    assert r5b.R5B_DISPOSITION_STATUSES == (
        "VALIDATED_BASIS",
        "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS",
        "BASIS_INSTRUMENT_UNRESOLVED",
    )
    assert r5b.R5B_PATH_LOCAL_C2_BY_STATUS == {
        "VALIDATED_BASIS": (
            "EXECUTE_OPTIONAL_C2_WITH_AUTHENTICATED_FROZEN_LOCAL_WRITER_BASIS"
        ),
        "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS": (
            "SKIP_OPTIONAL_C2_NO_VALIDATED_BASIS_DENSE_PRIMARY_UNAFFECTED"
        ),
        "BASIS_INSTRUMENT_UNRESOLVED": (
            "SKIP_OPTIONAL_C2_BASIS_INSTRUMENT_UNRESOLVED_"
            "DENSE_PRIMARY_UNAFFECTED"
        ),
    }
    assert callable(r5b.validate_local_writer_basis_disposition)


def test_three_way_disposition_is_exact_and_dense_primary_never_depends_on_it():
    assert r5b._disposition_for_replays(
        {"instrument_failure": False},
        {
            "instrument_failure": False,
            "candidate_validation": {
                "status": "NESTED_CANDIDATE_VALIDATED_FOR_REFIT"
            },
        },
    ) == "VALIDATED_BASIS"
    assert r5b._disposition_for_replays(
        {"instrument_failure": False},
        {"instrument_failure": False, "candidate_validation": None},
    ) == "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS"
    assert r5b._disposition_for_replays(
        {"instrument_failure": True},
        {
            "instrument_failure": False,
            "candidate_validation": {
                "status": "NESTED_CANDIDATE_VALIDATED_FOR_REFIT"
            },
        },
    ) == "BASIS_INSTRUMENT_UNRESOLVED"


def test_replace_local_writer_changes_only_the_frozen_query_rows():
    dense = torch.arange(3 * 6 * 8, dtype=torch.float64).reshape(3, 6, 8)
    current = torch.full((3, 8), 2.0, dtype=torch.float64)
    replacement = torch.full((3, 8), 7.0, dtype=torch.float64)
    positions = torch.tensor([1, 4, 2])
    changed = r5b.replace_local_writer_at_queries(
        dense,
        current_writer=current,
        replacement_writer=replacement,
        query_positions=positions,
    )
    expected = dense.clone()
    expected[torch.arange(3), positions] += 5.0
    assert torch.equal(changed, expected)
    with pytest.raises(ValueError, match="one query position"):
        r5b.replace_local_writer_at_queries(
            dense,
            current_writer=current,
            replacement_writer=replacement,
            query_positions=torch.tensor([[1], [4], [2]]),
        )


def test_real_dense_hook_replaces_exact_l3h1_contribution_and_records_provenance():
    torch.manual_seed(9)
    dense = torch.nn.Linear(12, 12, bias=True, dtype=torch.float64)
    clean_context = torch.randn(2, 5, 12, dtype=torch.float64)
    corrupt_context = torch.randn(2, 5, 12, dtype=torch.float64)
    positions = torch.tensor([2, 3])
    rows = torch.arange(2)
    clean_all = r5b.output_projected_head_contribution(
        clean_context, dense.weight, head_index=1, num_heads=3
    )
    clean_writer = clean_all[rows, positions]
    expected = dense(corrupt_context).detach()
    corrupt_all = r5b.output_projected_head_contribution(
        corrupt_context, dense.weight, head_index=1, num_heads=3
    )
    expected[rows, positions] += clean_writer - corrupt_all[rows, positions]

    with r5b.local_writer_dense_hook(
        dense,
        query_positions=positions,
        replacement_writer=clean_writer,
        head_index=1,
        num_heads=3,
    ) as audit:
        observed = dense(corrupt_context)
    assert torch.equal(observed, expected)
    assert audit.calls == 1
    assert audit.writer_site_id == r5b.WRITER_SITE_ID
    assert audit.intervention_position == "before_L4_L5"
    assert audit.current_writer_sha256 == r5b.array_sha256(
        corrupt_all[rows, positions].detach().numpy()
    )


def _dense_record(leg: str, sufficiency: np.ndarray, necessity: np.ndarray):
    zero = np.zeros_like(sufficiency)
    return r5b.local_writer_dense_estimands(
        m_clean=necessity,
        m_corrupt=zero,
        m_corrupt_with_clean_writer=sufficiency,
        m_clean_with_corrupt_writer=zero,
        leg_id=leg,
        writer_site_id=r5b.WRITER_SITE_ID,
    )


def test_dense_primary_family_replays_cells_and_keeps_all_specificity_controls():
    n = 12
    partner = _dense_record("partner", np.full(n, 0.30), np.full(n, 0.28))
    pseudo = _dense_record("pseudo", np.full(n, 0.01), np.full(n, -0.01))
    adjacent = {
        offset: _dense_record(
            f"adjacent:{offset:+d}", np.zeros(n), np.zeros(n)
        )
        for offset in r5b.ADJACENT_TOKEN_OFFSETS
    }
    family = r5b.build_dense_local_writer_family(
        partner=partner,
        pseudo=pseudo,
        adjacent_by_offset=adjacent,
        gene_generic=_dense_record(
            "gene_generic", np.full(n, 0.20), np.full(n, 0.18)
        ),
        readout_axis=_dense_record(
            "readout_axis", np.full(n, 0.15), np.full(n, 0.14)
        ),
        health={"clean_noop": np.zeros(n), "corrupt_noop": np.zeros(n)},
    )
    assert family["writer_site_id"] == r5b.WRITER_SITE_ID
    assert family["path_role"] == "primary"
    endpoints = family["endpoints"]
    assert np.array_equal(endpoints["partner|S"], np.full(n, 0.30))
    assert np.array_equal(endpoints["partner|N"], np.full(n, 0.28))
    assert np.allclose(endpoints["specific|S"], np.full(n, 0.29))
    assert np.allclose(endpoints["specific|N"], np.full(n, 0.29))
    assert set(name for name in endpoints if name.startswith("adjacent")) == {
        f"adjacent|offset={offset:+d}|{estimand}"
        for offset in r5b.ADJACENT_TOKEN_OFFSETS
        for estimand in ("S", "N")
    }
    assert {"gene_generic|S", "gene_generic|N"} <= set(endpoints)
    assert {"readout_axis|S", "readout_axis|N"} <= set(endpoints)
    assert "basis_id" not in family

    changed = copy.deepcopy(partner)
    changed["estimands"]["S_writer"][0] += 1.0
    with pytest.raises(ValueError, match="replay"):
        r5b.build_dense_local_writer_family(
            partner=changed,
            pseudo=pseudo,
            adjacent_by_offset=adjacent,
            gene_generic=_dense_record("gene_generic", np.zeros(n), np.zeros(n)),
            readout_axis=_dense_record("readout_axis", np.zeros(n), np.zeros(n)),
            health={"clean_noop": np.zeros(n), "corrupt_noop": np.zeros(n)},
        )


def test_studentized_family_requires_mean_and_ten_percent_trimmed_mean():
    rng = np.random.default_rng(17)
    genes = [f"gene-{index:03d}" for index in range(96)]
    endpoints = {
        "material": rng.normal(0.24, 0.025, len(genes)),
        "equivalent": rng.normal(0.00, 0.006, len(genes)),
        "health": rng.normal(0.00, 0.002, len(genes)),
    }
    result = r5b.robust_studentized_family_inference(
        endpoints,
        {
            "material": "material_positive",
            "equivalent": "equivalent",
            "health": "noninferior",
        },
        genes,
        n_draws=1_999,
        alpha=0.05,
        seed=3,
        tier="development",
    )
    assert result["all_requirements_pass"] is True
    assert result["estimators"] == ["mean", "trimmed_mean_10pct"]
    assert result["inference"] == "studentized cluster bootstrap inversion"
    assert result["exact_test_claimed"] is False
    assert result["one_row_per_gene"] is True

    tail_driven = rng.normal(0.04, 0.003, len(genes))
    tail_driven[:2] = 12.0
    fragile = r5b.robust_studentized_family_inference(
        {"material": tail_driven},
        {"material": "material_positive"},
        genes,
        n_draws=1_999,
        alpha=0.05,
        seed=4,
        tier="development",
    )
    assert fragile["by_endpoint"]["material"]["mean"]["observed"] > r5b.SESOI
    assert fragile["by_endpoint"]["material"]["trimmed_mean_10pct"][
        "observed"
    ] < r5b.SESOI
    assert fragile["all_requirements_pass"] is False

    with pytest.raises(ValueError, match="at least 10000"):
        r5b.robust_studentized_family_inference(
            {"material": endpoints["material"]},
            {"material": "material_positive"},
            genes,
            n_draws=9_999,
            alpha=0.05,
            seed=5,
            tier="production",
        )


def test_dense_fixed_sequence_stops_on_health_before_scientific_patterns():
    rng = np.random.default_rng(21)
    n = 96
    genes = [f"gene-{index:03d}" for index in range(n)]
    partner = _dense_record(
        "partner", rng.normal(0.25, 0.02, n), rng.normal(0.24, 0.02, n)
    )
    pseudo = _dense_record(
        "pseudo", rng.normal(0.0, 0.005, n), rng.normal(0.0, 0.005, n)
    )
    adjacent = {
        offset: _dense_record(
            f"adjacent:{offset:+d}",
            rng.normal(0.0, 0.005, n),
            rng.normal(0.0, 0.005, n),
        )
        for offset in r5b.ADJACENT_TOKEN_OFFSETS
    }
    family = r5b.build_dense_local_writer_family(
        partner=partner,
        pseudo=pseudo,
        adjacent_by_offset=adjacent,
        gene_generic=_dense_record("gene_generic", np.zeros(n), np.zeros(n)),
        readout_axis=_dense_record("readout_axis", np.zeros(n), np.zeros(n)),
        health={
            "clean_noop": rng.normal(-0.04, 0.003, n),
            "corrupt_noop": rng.normal(0.0, 0.002, n),
        },
    )
    decision = r5b.classify_dense_local_writer(
        family, genes, n_draws=1_999, seed=9
    )
    assert decision["status"] == "INSTRUMENT_UNRESOLVED_NOOP_HEALTH"
    assert decision["fixed_sequence"][0]["stage"] == "paired_noop_health"
    assert decision["fixed_sequence"][0]["pass"] is False
    assert decision["claim_eligible"] is False


def test_low_rank_candidate_is_nested_derive_validate_then_union_refit_only():
    rng = np.random.default_rng(31)
    width = 40
    latent = np.linalg.qr(rng.normal(size=(width, 6)))[0].T
    bucket0 = rng.normal(size=(96, 6)) @ latent + rng.normal(
        scale=0.02, size=(96, width)
    )
    bucket1 = rng.normal(size=(92, 6)) @ latent + rng.normal(
        scale=0.02, size=(92, width)
    )
    derivation_fit = r5b.fit_local_writer_basis(
        bucket0,
        maximum_rank=max(r5b.RANK_GRID),
        stage="derivation",
        source_bucket=0,
    )
    outcomes = {
        rank: {
            "mean_partner_s": {1: 0.20, 2: 0.60, 4: 0.92, 8: 0.96, 16: 1.0, 32: 1.0}[rank],
            "trimmed_partner_s": {1: 0.18, 2: 0.58, 4: 0.91, 8: 0.95, 16: 0.99, 32: 0.99}[rank],
            "nested_path_pass": rank >= 4,
            "crossfold_stability_pass": rank >= 4,
            "all_rank_energy_random_controls_pass": rank >= 4,
        }
        for rank in r5b.RANK_GRID
    }
    candidate = r5b.select_derivation_candidate_rank(
        derivation_fit,
        outcomes,
        dense_development_status="DENSE_LOCAL_WRITER_DEVELOPMENT_SUPPORTED",
    )
    assert candidate["selected_rank"] == 4
    assert candidate["path_role"] == "nested_candidate_only"
    assert candidate["dense_path_depends_on_selection"] is False

    validation = r5b.validate_frozen_candidate_on_bucket1(
        candidate,
        derivation_fit,
        bucket1,
        evaluated_ranks=[4],
        nested_outcome_pass=True,
        random_controls_pass=True,
        n_overlap_draws=199,
        seed=6,
    )
    assert validation["status"] == "NESTED_CANDIDATE_VALIDATED_FOR_REFIT"
    assert validation["rank_refit_during_validation"] is False
    assert validation["rank_rescanned_during_validation"] is False

    row0 = [f"b0-{index:03d}" for index in range(len(bucket0))]
    row1 = [f"b1-{index:03d}" for index in range(len(bucket1))]
    frozen = r5b.mechanical_union_refit(
        bucket0,
        bucket1,
        row_ids_bucket0=row0,
        row_ids_bucket1=row1,
        candidate=candidate,
        validation=validation,
        readout_axis=np.linspace(-1.0, 1.0, width),
        seed=7,
    )
    assert frozen["selected_rank"] == 4
    assert frozen["basis"].shape == (4, width)
    assert frozen["random_bases"].shape == (
        r5b.N_RANDOM_CONTROLS,
        4,
        width,
    )
    assert frozen["source_buckets"] == [0, 1]
    assert frozen["fit_timing"] == "after_bucket1_validation_locked"
    assert frozen["target_refit_forbidden"] is True
    assert frozen["target_rank_selection_forbidden"] is True
    assert frozen["gene_generic_axis"].shape == (width,)
    assert frozen["readout_axis"].shape == (width,)


def test_bucket0_candidate_outcomes_are_crossfit_one_row_per_gene():
    genes = [f"gene-{index:03d}" for index in range(128)]
    reversed_genes = list(reversed(genes))
    first = r5b.exact_derivation_fold_ids(genes)
    second = r5b.exact_derivation_fold_ids(reversed_genes)
    assert dict(zip(genes, first)) == dict(zip(reversed_genes, second))
    assert tuple(np.bincount(first).tolist()) == (64, 64)
    rows = np.random.default_rng(37).normal(size=(128, 40))
    package = r5b.fit_crossfit_derivation_bases(rows, genes, maximum_rank=32)
    assert package["whole_bucket_fit"]["n_rows"] == 128
    for fold in package["evaluation_folds"]:
        assert fold["n_fit_rows"] == 64
        assert fold["n_evaluation_rows"] == 64
        assert not set(fold["fit_genes"]) & set(fold["evaluation_genes"])


def test_target_cannot_fit_select_or_rescan_the_nested_basis():
    rows = np.random.default_rng(41).normal(size=(64, 40))
    with pytest.raises(r5b.TargetSealError, match="target"):
        r5b.fit_local_writer_basis(
            rows,
            maximum_rank=8,
            stage="target",
            source_bucket=None,
        )
    fit = r5b.fit_local_writer_basis(
        rows, maximum_rank=32, stage="derivation", source_bucket=0
    )
    with pytest.raises(ValueError, match="exactly once"):
        r5b.validate_frozen_candidate_on_bucket1(
            {"selected_rank": 4, "schema": r5b.R5B_SCHEMA + ".candidate"},
            fit,
            rows,
            evaluated_ranks=[2, 4],
            nested_outcome_pass=True,
            random_controls_pass=True,
            n_overlap_draws=199,
            seed=0,
        )


def _subspace_record(
    leg: str,
    sufficiency: np.ndarray,
    necessity: np.ndarray,
    *,
    basis_id: str = "frozen-U_A",
):
    zero = np.zeros_like(sufficiency)
    return r5b.local_writer_subspace_estimands(
        m_clean=necessity,
        m_corrupt=zero,
        m_corrupt_with_projected_writer=sufficiency,
        m_clean_without_projected_writer=zero,
        leg_id=leg,
        writer_site_id=r5b.WRITER_SITE_ID,
        basis_id=basis_id,
    )


def test_nested_family_retains_all_rank_energy_nulls_and_does_not_gate_dense():
    rng = np.random.default_rng(51)
    n, width, rank = 64, 40, 4
    delta = rng.normal(size=(n, width))
    basis = np.linalg.qr(rng.normal(size=(width, rank)))[0].T
    random_bases = np.stack(
        [
            np.linalg.qr(rng.normal(size=(width, rank)))[0].T
            for _ in range(r5b.N_RANDOM_CONTROLS)
        ]
    )
    projections = r5b.energy_matched_rank_random_projections(
        delta, basis=basis, random_bases=random_bases
    )
    target_basis_id = projections["audit"]["target_basis_id"]
    target_norm = np.linalg.norm(projections["target_projection"], axis=1)
    random_norm = np.linalg.norm(projections["random_projections"], axis=2)
    assert np.allclose(random_norm, target_norm[None, :], rtol=1e-12, atol=1e-12)
    assert projections["rank"] == rank
    assert projections["n_random_controls"] == r5b.N_RANDOM_CONTROLS

    partner = _subspace_record(
        "partner", np.full(n, 0.2), np.full(n, 0.19), basis_id=target_basis_id
    )
    pseudo = _subspace_record(
        "pseudo", np.zeros(n), np.zeros(n), basis_id=target_basis_id
    )
    adjacent = {
        offset: _subspace_record(
            f"adjacent:{offset:+d}",
            np.zeros(n),
            np.zeros(n),
            basis_id=target_basis_id,
        )
        for offset in r5b.ADJACENT_TOKEN_OFFSETS
    }
    random_records = {
        draw: _subspace_record(
            f"random:{draw:02d}",
            np.full(n, 0.01),
            np.full(n, 0.01),
            basis_id=projections["audit"]["random_basis_ids"][draw],
        )
        for draw in range(r5b.N_RANDOM_CONTROLS)
    }
    family = r5b.build_nested_local_writer_family(
        partner=partner,
        pseudo=pseudo,
        adjacent_by_offset=adjacent,
        gene_generic=_subspace_record(
            "gene_generic", np.full(n, 0.12), np.full(n, 0.11), basis_id=target_basis_id
        ),
        readout_axis=_subspace_record(
            "readout_axis", np.full(n, 0.08), np.full(n, 0.07), basis_id=target_basis_id
        ),
        random_by_draw=random_records,
        rank=rank,
        projection_audit=projections["audit"],
    )
    assert family["path_role"] == "nested_candidate_only"
    assert family["dense_path_depends_on_this_family"] is False
    assert len(
        [name for name in family["endpoints"] if name.startswith("target_minus_random")]
    ) == 2 * r5b.N_RANDOM_CONTROLS
    assert family["rank_energy_random_control_count"] == r5b.N_RANDOM_CONTROLS
    decision = r5b.classify_nested_local_writer_family(
        family,
        [f"gene-{index:03d}" for index in range(n)],
        n_draws=199,
        seed=2,
    )
    assert decision["nested_path_pass"] is True
    assert decision["dense_path_status_changed"] is False
    assert decision["claim_eligible"] is False


def test_v2_public_registries_bind_dense_and_every_nested_control_edge():
    primitive = r5b.causal_primitive_names()
    family = r5b.causal_family_names()
    edges = r5b.causal_boundary_edge_registry()
    assert len(primitive) == len(set(primitive))
    assert len(family) == len(set(family))
    assert len({row["edge_id"] for row in edges}) == len(edges)
    assert any("dense|partner|S" == name for name in family)
    assert any("nested|rank=4|target_minus_random|draw=31|N" == name for name in family)
    assert all(row["estimator"] in r5b.ESTIMATOR_NAMES for row in edges)
    assert {
        row["requirement"] for row in edges
    } == {"material_positive", "positive", "equivalent", "noninferior"}
    calibration = r5b.prospective_calibration_trial_count()
    assert calibration["decision_edge_count"] == len(edges)
    assert calibration["trial_count"] >= 2 * len(edges)
    assert calibration["scientific_error_control_claimed"] is False


def test_frozen_runtime_bundle_authenticates_basis_controls_and_upstream_manifests(tmp_path):
    rng = np.random.default_rng(61)
    rank, width = 4, 40
    basis = np.linalg.qr(rng.normal(size=(width, rank)))[0].T
    random_bases = np.stack(
        [
            np.linalg.qr(rng.normal(size=(width, rank)))[0].T
            for _ in range(r5b.N_RANDOM_CONTROLS)
        ]
    )
    frozen = {
        "schema": r5b.R5B_BASIS_BUNDLE_SCHEMA,
        "writer_site_id": r5b.WRITER_SITE_ID,
        "selected_rank": rank,
        "basis": basis,
        "random_bases": random_bases,
        "gene_generic_axis": np.ones(width) / np.sqrt(width),
        "readout_axis": np.linspace(-1, 1, width) / np.linalg.norm(np.linspace(-1, 1, width)),
        "ordered_development_row_ids_sha256": "1" * 64,
        "source_buckets": [0, 1],
        "fit_timing": "after_bucket1_validation_locked",
        "target_refit_forbidden": True,
        "target_rank_selection_forbidden": True,
        "target_basis_mutation_forbidden": True,
    }
    arrays = r5b.frozen_basis_runtime_arrays(frozen)
    payload = r5b.npz_bytes(arrays)
    path = tmp_path / "r5b-frozen-basis.npz"
    path.write_bytes(payload)
    bindings = {
        name: {"artifact_sha256": char * 64, "manifest_sha256": char * 64}
        for name, char in (
            ("r5a_entry", "a"),
            ("r5b_derivation", "b"),
            ("r5b_validation", "c"),
        )
    }
    contract = r5b.frozen_basis_runtime_contract(
        frozen,
        bundle_path=path,
        bundle_bytes=payload,
        checkpoint_sha256="d" * 64,
        cohort_audit_sha256="e" * 64,
        r5_stack_fingerprint_sha256="f" * 64,
        artifact_bindings=bindings,
        post_wo_writer_contract={
            "writer_site_id": r5b.WRITER_SITE_ID,
            "representation": (
                "post_W_O_L3H1_additive_contribution_at_acceptor_query"
            ),
            "bias_included": False,
            "residual_width": width,
        },
    )
    assert contract["schema"] == r5b.R5B_BASIS_CONTRACT_SCHEMA
    assert contract["basis_id"] == "sha256:" + r5b.array_sha256(basis, dtype=None)
    assert contract["random_basis_ids"] == [
        "sha256:" + r5b.array_sha256(random_bases[draw], dtype=None)
        for draw in range(r5b.N_RANDOM_CONTROLS)
    ]
    assert contract["target_policy"]["refit"] == "forbidden"
    assert contract["target_policy"]["rank_selection"] == "forbidden"
    assert contract["upstream_artifact_bindings"] == bindings
    loaded = r5b.load_frozen_basis_runtime_bundle(path, contract=contract)
    assert np.array_equal(loaded["frozen_basis"], basis)
    assert np.array_equal(loaded["random_bases"], random_bases)

    changed = copy.deepcopy(contract)
    changed["upstream_artifact_bindings"]["r5b_validation"][
        "manifest_sha256"
    ] = "0" * 64
    with pytest.raises(r5b.InstrumentError, match="contract digest"):
        r5b.load_frozen_basis_runtime_bundle(path, contract=changed)

    inconsistent_arrays = dict(arrays)
    inconsistent_arrays["selected_rank"] = np.asarray(rank + 1, dtype="<i8")
    inconsistent_payload = r5b.npz_bytes(inconsistent_arrays)
    inconsistent_path = tmp_path / "r5b-inconsistent-rank.npz"
    inconsistent_path.write_bytes(inconsistent_payload)
    inconsistent_contract = copy.deepcopy(contract)
    inconsistent_contract["bundle"] = {
        "file_name": inconsistent_path.name,
        "size_bytes": len(inconsistent_payload),
        "sha256": hashlib.sha256(inconsistent_payload).hexdigest(),
    }
    selected = inconsistent_arrays["selected_rank"]
    inconsistent_contract["runtime_arrays"]["selected_rank"] = {
        "shape": list(selected.shape),
        "dtype": selected.dtype.str,
        "sha256": r5b.array_sha256(selected, dtype=None),
    }
    inconsistent_contract["contract_sha256"] = r5b.canonical_sha256(
        {
            key: value
            for key, value in inconsistent_contract.items()
            if key != "contract_sha256"
        }
    )
    with pytest.raises(r5b.InstrumentError, match="selected rank"):
        r5b.load_frozen_basis_runtime_bundle(
            inconsistent_path, contract=inconsistent_contract
        )


def test_r5b_declares_r5c_target_hooks_but_has_no_biological_target_stage():
    hook = r5b.r5c_target_attempt_hook_locations()
    assert hook["owner_stage"] == "R5c biological target"
    assert hook["reserve_after"] == "all predecessor and upstream packages authenticated"
    assert hook["reserve_before"] == "cohort/path/model/output inspection"
    assert hook["complete_after"] == "immutable result and manifest publication"
    assert hook["r5b_calls_target_ledger"] is False
    for stage in ("audit", "derivation", "validation", "freeze"):
        assert r5b.authorize_r5b_stage(stage) is None
    with pytest.raises(ValueError, match="unknown R5b development stage"):
        r5b.authorize_r5b_stage("target")


def test_residual_bucket_receipt_and_completion_are_create_only(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    identity = r5b.build_r5b_residual_attempt_identity(
        stage="derivation",
        checkpoint_sha256="1" * 64,
        cohort_input_sha256="2" * 64,
        cohort_audit_sha256="3" * 64,
    )
    completion_name = f"{identity['residual_attempt_key']}.completion.json"
    context = {
        "protocol_sha256": "4" * 64,
        "instrument_fingerprint_sha256": "5" * 64,
        "r5_stack_fingerprint_sha256": "6" * 64,
        "r3d_artifact_sha256": "7" * 64,
        "r3d_manifest_sha256": "8" * 64,
        "r5a_preflight_artifact_sha256": "9" * 64,
        "r5a_preflight_manifest_sha256": "a" * 64,
        "predecessor_artifact_sha256": None,
        "predecessor_manifest_sha256": None,
        "predecessor_completion_sha256": None,
        "requested_artifact_logical_name": "derivation.json",
        "requested_manifest_logical_name": "derivation.manifest.json",
        "declared_completion_logical_name": completion_name,
        "scoring_origin": "direct_checkpoint_scoring_in_r5b_process",
    }
    receipt = r5b.reserve_r5b_residual_attempt(
        repo=repo, identity=identity, method_context=context
    )
    with pytest.raises(r5b.ResidualAttemptSpentError, match="remains spent"):
        r5b.reserve_r5b_residual_attempt(
            repo=repo, identity=identity, method_context=context
        )

    artifact = tmp_path / "derivation.json"
    artifact_record = {
        "schema": r5b.R5B_DERIVATION_SCHEMA,
        "stage": "derivation",
        "status": "NO_DERIVATION_CANDIDATE_AT_FROZEN_THRESHOLDS",
        "direct_model_scoring": True,
        "external_provider_artifact_accepted": False,
        "residual_attempt": r5b.residual_attempt_publication_identity(receipt),
        "residual_completion": {"logical_name": completion_name},
    }
    artifact.write_text(json.dumps(artifact_record, sort_keys=True))
    manifest = tmp_path / "derivation.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "git_dirty": False,
                "outputs": [
                    {
                        "path": str(artifact),
                        "size_bytes": artifact.stat().st_size,
                        "sha256": r5b.sha256_file(artifact),
                    }
                ],
            },
            sort_keys=True,
        )
    )
    completion = r5b.complete_r5b_residual_attempt(
        receipt=receipt,
        artifact_path=artifact,
        manifest_path=manifest,
        official_status=artifact_record["status"],
    )
    authenticated = r5b.verify_r5b_residual_completion(
        artifact,
        manifest,
        completion["completion_path"],
        expected_stage="derivation",
    )
    assert authenticated["authenticated"] is True
    assert authenticated["completion_record"]["official_status"] == (
        artifact_record["status"]
    )
    with pytest.raises(r5b.ResidualAttemptSpentError, match="remains spent"):
        r5b.complete_r5b_residual_attempt(
            receipt=receipt,
            artifact_path=artifact,
            manifest_path=manifest,
            official_status=artifact_record["status"],
        )


def test_residual_bucket_receipt_is_durable_before_reservation_returns(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    identity = r5b.build_r5b_residual_attempt_identity(
        stage="derivation",
        checkpoint_sha256="1" * 64,
        cohort_input_sha256="2" * 64,
        cohort_audit_sha256="3" * 64,
    )
    context = {
        "protocol_sha256": "4" * 64,
        "instrument_fingerprint_sha256": "5" * 64,
        "r5_stack_fingerprint_sha256": "6" * 64,
        "r3d_artifact_sha256": "7" * 64,
        "r3d_manifest_sha256": "8" * 64,
        "r5a_preflight_artifact_sha256": "9" * 64,
        "r5a_preflight_manifest_sha256": "a" * 64,
        "predecessor_artifact_sha256": None,
        "predecessor_manifest_sha256": None,
        "predecessor_completion_sha256": None,
        "requested_artifact_logical_name": "derivation.json",
        "requested_manifest_logical_name": "derivation.manifest.json",
        "declared_completion_logical_name": (
            f"{identity['residual_attempt_key']}.completion.json"
        ),
        "scoring_origin": "direct_checkpoint_scoring_in_r5b_process",
    }
    sync_kinds = []
    real_fsync = r5b.os.fsync

    def record_fsync(descriptor):
        mode = r5b.os.fstat(descriptor).st_mode
        sync_kinds.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(r5b.os, "fsync", record_fsync)
    r5b.reserve_r5b_residual_attempt(
        repo=repo,
        identity=identity,
        method_context=context,
    )

    assert sync_kinds == ["file", "directory"]


def test_r5b_consumes_only_the_authenticated_r5a_v2_entry_contract(tmp_path, monkeypatch):
    stack_fingerprint = "1" * 64
    artifact = tmp_path / "r5a.json"
    record = {
        "schema": "gfmcd.r5a.acceptor_address_binding/2",
        "stage": "preflight",
        "status": "PREFLIGHT_PASS",
        "all_instrument_gates_pass": True,
        "behavioral_endpoint": "masked_acceptor_G_margin_only",
        "donor_readout_constructed": False,
        "masked_donor_endpoint_constructed": False,
        "r5a_opened_chr8_chr9": False,
        "scientific_entry_authorized": False,
        "observed_scientific_inference": None,
        "r5_stack_fingerprint_sha256": stack_fingerprint,
        "r5_entry_contract": {
            "schema": r5b.R5_ENTRY_SCHEMA,
            "instrument_gates_pass": True,
            "claim_eligible": False,
            "verdict": "PREFLIGHT_PASS",
            "scientific_entry_order": [
                "authenticated_R3d_Path_A_prior",
                "authenticated_R1c_validation_preflight_nonentry",
                "R5a_instrument_preflight",
            ],
            "r3d_path_b_used_to_gate": False,
            "r3c_used_as_scientific_entry": False,
            "acceptor_G_only": True,
            "masked_donor_endpoint_constructed": False,
            "checkpoint_sha256": "2" * 64,
            "target_gene_ids_sha256": "3" * 64,
            "cohort_sha256": "4" * 64,
            "ordered_row_ids_sha256": "9" * 64,
            "ordered_gene_ids_sha256": "a" * 64,
            "construction_sha256": "5" * 64,
            "r5_stack_fingerprint_sha256": stack_fingerprint,
            "upstream_package_chain_sha256": "6" * 64,
        },
    }
    artifact.write_text(json.dumps(record, sort_keys=True))
    manifest = artifact.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "git_dirty": False,
                "outputs": [
                    {
                        "path": "/relocated/original/r5a.json",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": r5b.sha256_file(artifact),
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(r5b, "FROZEN_R5A_ARTIFACT_SHA256", r5b.sha256_file(artifact))
    monkeypatch.setattr(r5b, "FROZEN_R5A_MANIFEST_SHA256", r5b.sha256_file(manifest))
    entry = r5b.verify_r5a_entry_package(
        artifact, manifest, expected_stack_fingerprint=stack_fingerprint
    )
    assert entry["entry_chain_pass"] is True
    assert entry["r3d_path_a_prior_consumed_via_r5a"] is True
    assert entry["r1c_validation_preflight_consumed_via_r5a"] is True
    assert entry["sealed_target_result_consumed"] is False
    assert entry["r3d_path_b_reopened"] is False
    assert entry["r3c_reopened_as_scientific_entry"] is False

    changed = copy.deepcopy(record)
    changed["r5_entry_contract"]["r3d_path_b_used_to_gate"] = True
    artifact.write_text(json.dumps(changed, sort_keys=True))
    manifest.write_text(
        json.dumps(
            {
                "git_dirty": False,
                "outputs": [
                    {
                        "path": "/relocated/original/r5a.json",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": r5b.sha256_file(artifact),
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(r5b, "FROZEN_R5A_ARTIFACT_SHA256", r5b.sha256_file(artifact))
    monkeypatch.setattr(r5b, "FROZEN_R5A_MANIFEST_SHA256", r5b.sha256_file(manifest))
    with pytest.raises(r5b.TargetSealError, match="Path B"):
        r5b.verify_r5a_entry_package(
            artifact, manifest, expected_stack_fingerprint=stack_fingerprint
        )


def test_model_free_audit_and_cli_never_need_model_genome_or_chr8_chr9(tmp_path):
    audit = r5b.run_model_free_audit(n_draws=199)
    assert audit["all_pass"] is True
    assert audit["model_loaded"] is False
    assert audit["genome_opened"] is False
    assert audit["sealed_chr8_chr9_opened"] is False
    assert audit["primary_state"]["writer_site_id"] == r5b.WRITER_SITE_ID
    assert audit["primary_state"]["dense_depends_on_low_rank"] is False

    destination = tmp_path / "audit.json"
    process = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--stage",
            "audit",
            "--out",
            str(destination),
            "--audit-draws",
            "199",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stderr
    record = json.loads(destination.read_text())
    assert record["schema"] == "gfmcd.r5b.model_free_audit/2"
    assert record["all_pass"] is True


def test_source_freeze_leaves_derivation_closed_until_r5a_preflight_exists():
    assert r5b.FROZEN_PROTOCOL_SHA256 == (
        "f84dd85570acfb010d77dd29f0a3f48a00967c9da9d69b0f907a98d4e141a6e8"
    )
    assert r5b.FROZEN_INSTRUMENT_FINGERPRINT_SHA256 == (
        "f41942fb08cdb11b90569b4202d93e797a4631c90134b32d1c8c7a2a5190706a"
    )
    assert r5b.FROZEN_COHORT_INPUT_SHA256 == (
        "b2cf20c92c9e3829c2eb00326748e3f96049b4f21f144c3fd79ed5076d881049"
    )
    assert r5b.FROZEN_COHORT_AUDIT_SHA256 == (
        "8e98aaf8bbeb120cacfc4d32928e49a80f3cc10edc3e793cae8cec0fcb68b923"
    )
    assert r5b.FROZEN_R3D_ARTIFACT_SHA256 == (
        "e33d37447cbba368a48beea108fc49c06c56adfbeef2c0f95f019c227d60cb88"
    )
    assert r5b.FROZEN_R3D_MANIFEST_SHA256 == (
        "53f13aa4884b53a91eba6efa2091c7e7638672ed260bf6558f20547d88f899cd"
    )
    assert r5b.FROZEN_CHECKPOINT_AGGREGATE_SHA256 == (
        "fb2a04e0c9b5247b44f03cd23581632d5924b86fcafaffa4c6f2b8d01ae1da8d"
    )
    assert "> **Status: FROZEN FOR PREFLIGHT ONLY" in r5b.PROTOCOL.read_text()
    assert r5b.FROZEN_R5A_ARTIFACT_SHA256 is None
    assert r5b.FROZEN_R5A_MANIFEST_SHA256 is None
    with pytest.raises(r5b.TargetSealError, match="literal freeze pins"):
        r5b.authorize_scored_stage("derivation", device="mps")
    with pytest.raises(ValueError, match="only defined for R5b scored stages"):
        r5b.authorize_scored_stage("target", device="mps")


def test_frozen_r5b_full_reservation_audit_reconstructs_from_bound_inputs():
    gtf = Path(os.environ.get("GFMCD_GTF", "/nonexistent/gencode.gtf.gz"))
    fasta = Path(os.environ.get("GFMCD_FASTA", "/nonexistent/hg38.fa"))
    if not gtf.is_file() or not fasta.is_file() or not Path(f"{fasta}.fai").is_file():
        pytest.skip("exact frozen GTF/FASTA/FAI are unavailable")

    results = REPO / "results/splicebert_acceptor_coupling"
    common_dir = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], cwd=REPO, text=True
        ).strip()
    )
    if not common_dir.is_absolute():
        common_dir = (REPO / common_dir).resolve()
    receipt_root = common_dir / "gfmcd_target_attempts/r3d"
    target_id = "b8ff2fedfd3add86c88d882a0b1c36788e2f41cafb583ff1179c4779d0d4f839"
    required = {
        "r3d_artifact": results / "r3d_acceptor_specificity_target_v1.json",
        "r3d_manifest": results / "r3d_acceptor_specificity_target_v1.manifest.json",
        "r3d_registry": results / "r3d_train_exposure_registry_v1.json",
        "r3d_registry_manifest": results / "r3d_train_exposure_registry_v1.manifest.json",
        "r3d_attempt": receipt_root / f"{target_id}.json",
        "r3d_completion": receipt_root / f"{target_id}.completion.json",
        "r3c_artifact": results / "r3c_acceptor_head_preflight_v2_1.json",
        "r3c_manifest": results / "r3c_acceptor_head_preflight_v2_1.manifest.json",
    }
    if not all(path.is_file() for path in required.values()):
        pytest.skip("exact authenticated historical R3 packages are unavailable")

    r3d = r5b.load_r3d_entry_package(
        required["r3d_artifact"],
        required["r3d_manifest"],
        registry_path=required["r3d_registry"],
        registry_manifest_path=required["r3d_registry_manifest"],
        attempt_path=required["r3d_attempt"],
        completion_path=required["r3d_completion"],
    )
    r3c = r5b.load_r3c_exclusion_package(
        required["r3c_artifact"], required["r3c_manifest"]
    )
    registry = r5b.derive_r5_exclusion_registry(r3c, r3d)
    reservations, audit, source = r5b.build_r5b_residual_reservations(
        gtf, fasta, exclusion_registry=registry
    )
    cohort_input = {
        "source_identity": source,
        "exclusion_registry_record_sha256": registry["record_sha256"],
        "construction": "build_r5_reservations@r5-acceptor-mechanism-v1",
    }
    assert {bucket: len(rows) for bucket, rows in reservations.items()} == {0: 128, 1: 128}
    assert audit["schema"] == "gfmcd.r5.acceptor_mechanism_cohort/3"
    assert audit["sealed_chr8_chr9_opened"] is False
    assert r5b.canonical_sha256(cohort_input) == r5b.FROZEN_COHORT_INPUT_SHA256
    assert r5b.canonical_sha256(audit) == r5b.FROZEN_COHORT_AUDIT_SHA256


def test_cli_has_no_external_scored_provider_or_self_attestation_path():
    assert not hasattr(r5b, "derive_residual_writer_provider")
    assert not hasattr(r5b, "validate_residual_writer_provider")
    process = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--stage",
            "derivation",
            "--provider-artifact",
            "fabricated-provider.json",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    assert process.returncode != 0
    assert "unrecognized arguments: --provider-artifact" in process.stderr


class _TinySelfAttention(torch.nn.Module):
    def forward(self, hidden):
        return hidden + hidden.mean(dim=1, keepdim=True)


class _TinyAttention(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self = _TinySelfAttention()
        self.output = torch.nn.Module()
        self.output.dense = torch.nn.Linear(width, width, bias=True)
        with torch.no_grad():
            self.output.dense.weight.copy_(torch.eye(width))
            self.output.dense.bias.zero_()


class _TinyLayer(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.attention = _TinyAttention(width)

    def forward(self, hidden):
        context = self.attention.self(hidden)
        return hidden + self.attention.output.dense(context)


class _TinyPredictions(torch.nn.Module):
    def __init__(self, width: int, vocab: int):
        super().__init__()
        self.decoder = torch.nn.Linear(width, vocab, bias=False)


class _TinyCls(torch.nn.Module):
    def __init__(self, width: int, vocab: int):
        super().__init__()
        self.predictions = _TinyPredictions(width, vocab)


class _TinyBert(torch.nn.Module):
    def __init__(self, *, width: int = 4, vocab: int = 6):
        super().__init__()
        torch.manual_seed(20260817)
        self.embedding = torch.nn.Embedding(vocab, width)
        self.layers = torch.nn.ModuleList([_TinyLayer(width) for _ in range(4)])
        self.cls = _TinyCls(width, vocab)
        self.forward_calls = 0

    def forward(self, input_ids):
        self.forward_calls += 1
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        # Keep the behavioral endpoints exactly null so this fast scorer test never
        # enters the optional six-rank path. The decoder remains real and supplies the
        # authenticated G-vs-A/C/T control axis.
        logits = self.cls.predictions.decoder(hidden) * 0.0
        return SimpleNamespace(logits=logits)


class _TinyAdapter:
    def __init__(self):
        self.model = _TinyBert()
        self.encoder_layers = self.model.layers
        self.attention_self_modules = [
            layer.attention.self for layer in self.encoder_layers
        ]
        self.num_layers = len(self.encoder_layers)
        self.num_heads = 2
        self.device = torch.device("cpu")


class _TinyAcceptorTask:
    off = 0
    bid = {"A": 0, "C": 1, "G": 2, "T": 3}
    mask_id = 4

    def encode(self, sequences, positions):
        ids = torch.tensor(
            [[self.bid.get(base, 5) for base in sequence] for sequence in sequences],
            dtype=torch.long,
        )
        for row, position in enumerate(positions):
            ids[row, int(position)] = self.mask_id
        return {"input_ids": ids}

    def margin(self, logits, positions):
        rows = torch.arange(len(positions), device=logits.device)
        columns = torch.tensor(positions, device=logits.device)
        values = logits[rows, columns]
        alternatives = torch.stack(
            [values[:, self.bid[base]] for base in "ACT"], dim=-1
        ).max(dim=-1).values
        return values[:, self.bid["G"]] - alternatives


def _tiny_residual_rows(prefix="b0"):
    return [
        SimpleNamespace(
            example=SimpleNamespace(
                gene_id=f"{prefix}-gene-{index:03d}",
                example_id=f"{prefix}-row-{index:03d}",
            ),
            acceptor_g=4,
            clean_sequence="AACGTAAA",
            paired_corrupt_sequence="ATCGTAAA",
            pseudo_corrupt_sequence="AAGGTAAA",
        )
        for index in range(r5b.PRODUCTION_N)
    ]


def _supported_dense_family(n):
    zero = np.zeros(n)
    return r5b.build_dense_local_writer_family(
        partner=_dense_record("partner", np.full(n, 0.30), np.full(n, 0.30)),
        pseudo=_dense_record("pseudo", zero, zero),
        adjacent_by_offset={
            offset: _dense_record(f"adjacent:{offset:+d}", zero, zero)
            for offset in r5b.ADJACENT_TOKEN_OFFSETS
        },
        gene_generic=_dense_record("gene_generic", zero, zero),
        readout_axis=_dense_record("readout_axis", zero, zero),
        health={"clean_noop": zero, "corrupt_noop": zero},
    )


def _captured_writer_fixture(*, rank_deficient, seed):
    rng = np.random.default_rng(seed)
    n, residual_width, head_width = r5b.PRODUCTION_N, 80, 40
    if rank_deficient:
        clean_context = np.outer(
            np.linspace(0.5, 1.5, n), rng.normal(size=head_width)
        )
    else:
        clean_context = rng.normal(size=(n, head_width))
    corrupt_context = np.zeros_like(clean_context)
    pseudo_context = clean_context.copy()
    block = np.random.default_rng(991).normal(size=(residual_width, head_width))
    clean_writer = clean_context @ block.T
    corrupt_writer = corrupt_context @ block.T
    pseudo_writer = pseudo_context @ block.T
    provider_record = {
        "writer_site_id": r5b.WRITER_SITE_ID,
        "clean_head_context": clean_context,
        "corrupt_head_context": corrupt_context,
        "w_o_column_block": block,
        "clean_writer_contribution": clean_writer,
        "corrupt_writer_contribution": corrupt_writer,
        "writer_delta": clean_writer - corrupt_writer,
        "pseudo_head_context": pseudo_context,
        "pseudo_writer_contribution": pseudo_writer,
        "pseudo_delta": clean_writer - pseudo_writer,
        "numeric_provenance": {
            "schema": "gfmcd.r5b.post_wo_numeric_provenance/1",
            "checkpoint_operator_dtype": "float64",
            "head_context_dtype": "float64",
            "w_o_column_block_dtype": "float64",
            "writer_contribution_dtype": "float64",
            "replay_operator": "torch.nn.functional.linear",
            "relative_tolerance": 0.0,
            "absolute_tolerance": 1e-5,
        },
        "bias_included": False,
    }
    return {
        "provider_record": provider_record,
        "clean_writer": clean_writer,
        "partner_writer": corrupt_writer,
        "pseudo_writer": pseudo_writer,
        "partner_delta": clean_writer - corrupt_writer,
        "pseudo_delta": clean_writer - pseudo_writer,
        "readout_axis": np.eye(1, residual_width, 0, dtype=np.float64)[0],
    }


def _nested_family_fixture(*, n, rank, projection_audit):
    target_id = projection_audit["target_basis_id"]
    zero = np.zeros(n)
    random_ids = projection_audit["random_basis_ids"]
    return r5b.build_nested_local_writer_family(
        partner=_subspace_record(
            "partner", np.full(n, 0.20), np.full(n, 0.20), basis_id=target_id
        ),
        pseudo=_subspace_record("pseudo", zero, zero, basis_id=target_id),
        adjacent_by_offset={
            offset: _subspace_record(
                f"adjacent:{offset:+d}", zero, zero, basis_id=target_id
            )
            for offset in r5b.ADJACENT_TOKEN_OFFSETS
        },
        gene_generic=_subspace_record("gene_generic", zero, zero, basis_id=target_id),
        readout_axis=_subspace_record("readout_axis", zero, zero, basis_id=target_id),
        random_by_draw={
            draw: _subspace_record(
                f"random:{draw:02d}",
                np.full(n, 0.01),
                np.full(n, 0.01),
                basis_id=random_ids[draw],
            )
            for draw in range(r5b.N_RANDOM_CONTROLS)
        },
        rank=rank,
        projection_audit=projection_audit,
    )


class _CleanTestManifest:
    git_dirty = False

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def to_dict(self):
        return {
            "git_dirty": False,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "fixture": "clean-manifest",
        }


def _publish_development_fixture(
    tmp_path, monkeypatch, record, *, derivation=None
):
    stage = record["stage"]
    attempt_repo = tmp_path / f"{stage}-attempt-repo"
    attempt_repo.mkdir()
    subprocess.run(["git", "init", "-q", str(attempt_repo)], check=True)
    identity = r5b.build_r5b_residual_attempt_identity(
        stage=stage,
        checkpoint_sha256="1" * 64,
        cohort_input_sha256="2" * 64,
        cohort_audit_sha256="3" * 64,
    )
    output = tmp_path / f"{stage}.json"
    predecessor = (None, None, None) if stage == "derivation" else (
        "b" * 64,
        "c" * 64,
        "d" * 64,
    )
    receipt = r5b.reserve_r5b_residual_attempt(
        repo=attempt_repo,
        identity=identity,
        method_context={
            "protocol_sha256": "4" * 64,
            "instrument_fingerprint_sha256": "5" * 64,
            "r5_stack_fingerprint_sha256": "6" * 64,
            "r3d_artifact_sha256": "7" * 64,
            "r3d_manifest_sha256": "8" * 64,
            "r5a_preflight_artifact_sha256": "9" * 64,
            "r5a_preflight_manifest_sha256": "a" * 64,
            "predecessor_artifact_sha256": predecessor[0],
            "predecessor_manifest_sha256": predecessor[1],
            "predecessor_completion_sha256": predecessor[2],
            "requested_artifact_logical_name": output.name,
            "requested_manifest_logical_name": output.with_suffix(
                ".manifest.json"
            ).name,
            "declared_completion_logical_name": (
                f"{identity['residual_attempt_key']}.completion.json"
            ),
            "scoring_origin": "direct_checkpoint_scoring_in_r5b_process",
        },
    )
    monkeypatch.setattr(r5b, "PRODUCTION_INFERENCE_DRAWS", 199)
    monkeypatch.setattr(r5b, "PRODUCTION_OVERLAP_DRAWS", 199)
    monkeypatch.setattr(
        r5b.ExperimentManifest,
        "create",
        lambda *_args, **_kwargs: _CleanTestManifest(),
    )
    return r5b.publish_residual_development_stage(
        record,
        derivation=derivation,
        receipt=receipt,
        output_path=output,
        command=["r5b-test"],
        input_paths=[],
        authorization={
            "instrument_fingerprint_sha256": "5" * 64,
            "r5_stack_fingerprint_sha256": "6" * 64,
        },
    )


def test_internal_derivation_scorer_runs_checkpoint_capture_and_post_wo_hooks():
    adapter = _TinyAdapter()
    record = r5b.score_r5b_derivation(
        adapter,
        _TinyAcceptorTask(),
        _tiny_residual_rows(),
        batch_size=r5b.PRODUCTION_N,
        n_draws=199,
        seed=0,
    )
    assert record["direct_model_scoring"] is True
    assert record["external_provider_artifact_accepted"] is False
    assert record["scored_provider"]["origin"] == (
        "direct_checkpoint_scoring_in_this_r5b_process"
    )
    assert record["post_wo_writer_audit"]["residual_width"] == 4
    assert record["post_wo_writer_audit"]["bias_included"] is False
    assert record["status"] == "NO_DERIVATION_CANDIDATE_AT_FROZEN_THRESHOLDS"
    assert adapter.model.forward_calls >= 20

    tampered = copy.deepcopy(record)
    tampered["scored_provider"]["genes"][0] = "different-gene"
    with pytest.raises(r5b.InstrumentError, match="provider header"):
        r5b._assert_development_record_replays(tampered)


def test_rank_deficient_derivation_completes_as_basis_instrument_unresolved(
    tmp_path, monkeypatch
):
    rows = _tiny_residual_rows("rank-deficient-b0")
    captured = _captured_writer_fixture(rank_deficient=True, seed=71)
    dense_family = _supported_dense_family(len(rows))
    monkeypatch.setattr(
        r5b,
        "capture_residual_writer_state",
        lambda *_args, **_kwargs: captured,
    )
    monkeypatch.setattr(
        r5b,
        "score_dense_local_writer_family",
        lambda *_args, **_kwargs: dense_family,
    )

    record = r5b.score_r5b_derivation(
        object(), object(), rows, batch_size=128, n_draws=199, seed=0
    )

    assert record["status"] == "BASIS_INSTRUMENT_UNRESOLVED"
    assert record["dense_decision"]["status"] == (
        "DENSE_LOCAL_WRITER_DEVELOPMENT_SUPPORTED"
    )
    assert record["nested_basis_instrument_failure"] == {
        "schema": "gfmcd.r5b.optional_basis_instrument_failure/1",
        "phase": "derivation",
        "code": "INSUFFICIENT_NUMERICAL_RANK",
        "message": (
            "local-writer SVD is numerically rank deficient at relative "
            "tolerance 1e-10"
        ),
        "optional_basis_only": True,
        "dense_primary_unaffected": True,
        "union_refit_performed": False,
    }
    package = _publish_development_fixture(tmp_path, monkeypatch, record)
    assert package["completion_record"]["official_status"] == (
        "BASIS_INSTRUMENT_UNRESOLVED"
    )

    validation = r5b.score_r5b_validation(
        object(),
        object(),
        _tiny_residual_rows("rank-deficient-b1"),
        package["record"],
        batch_size=128,
        n_draws=199,
        n_overlap_draws=199,
        seed=0,
    )
    assert validation["status"] == "BASIS_INSTRUMENT_UNRESOLVED"
    assert validation["candidate_validation"] is None
    assert validation["nested_basis_instrument_failure"] is None
    assert validation["scored_provider"]["nested_rank_families"] == {}
    validation_package = _publish_development_fixture(
        tmp_path, monkeypatch, validation, derivation=package["record"]
    )
    assert validation_package["completion_record"]["official_status"] == (
        "BASIS_INSTRUMENT_UNRESOLVED"
    )

    monkeypatch.setattr(
        r5b,
        "fit_crossfit_derivation_bases",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            r5b.InstrumentError("unclassified scorer/core failure")
        ),
    )
    with pytest.raises(r5b.InstrumentError, match="unclassified scorer/core failure"):
        r5b.score_r5b_derivation(
            object(), object(), rows, batch_size=128, n_draws=199, seed=0
        )


def test_rank_deficient_validation_completes_and_freezes_terminal_unresolved(
    tmp_path, monkeypatch
):
    rows0 = _tiny_residual_rows("full-rank-b0")
    rows1 = _tiny_residual_rows("rank-deficient-b1")
    captured0 = _captured_writer_fixture(rank_deficient=False, seed=73)
    captured1 = _captured_writer_fixture(rank_deficient=True, seed=79)
    dense_family = _supported_dense_family(r5b.PRODUCTION_N)

    def capture(_adapter, _task, rows, **_kwargs):
        return captured0 if rows[0].example.example_id.startswith("full-rank") else captured1

    monkeypatch.setattr(r5b, "capture_residual_writer_state", capture)
    monkeypatch.setattr(
        r5b,
        "score_dense_local_writer_family",
        lambda *_args, **_kwargs: dense_family,
    )
    monkeypatch.setattr(
        r5b,
        "score_nested_local_writer_family",
        lambda _adapter, _task, rows, _captured, **kwargs: _nested_family_fixture(
            n=len(rows),
            rank=kwargs["rank"],
            projection_audit=kwargs["projection_audit"],
        ),
    )

    derivation = r5b.score_r5b_derivation(
        object(), object(), rows0, batch_size=128, n_draws=199, seed=0
    )
    assert derivation["status"] == "DERIVATION_CANDIDATE_SELECTED"
    derivation_package = _publish_development_fixture(
        tmp_path, monkeypatch, derivation
    )
    frozen_derivation = derivation_package["record"]

    validation = r5b.score_r5b_validation(
        object(),
        object(),
        rows1,
        frozen_derivation,
        batch_size=128,
        n_draws=199,
        n_overlap_draws=199,
        seed=0,
    )

    assert validation["status"] == "BASIS_INSTRUMENT_UNRESOLVED"
    assert validation["dense_decision"]["status"] == (
        "DENSE_LOCAL_WRITER_DEVELOPMENT_SUPPORTED"
    )
    assert validation["nested_basis_instrument_failure"]["phase"] == "validation"
    assert validation["nested_basis_instrument_failure"]["code"] == (
        "INSUFFICIENT_NUMERICAL_RANK"
    )
    validation_package = _publish_development_fixture(
        tmp_path, monkeypatch, validation, derivation=frozen_derivation
    )
    assert validation_package["completion_record"]["official_status"] == (
        "BASIS_INSTRUMENT_UNRESOLVED"
    )

    disposition, bundle = r5b.build_local_writer_basis_disposition(
        frozen_derivation,
        validation_package["record"],
        development_packages={
            "derivation": {
                "artifact": derivation_package["artifact"],
                "manifest": derivation_package["manifest"],
            },
            "validation": {
                "artifact": validation_package["artifact"],
                "manifest": validation_package["manifest"],
            },
        },
        r3d_path_a_binding={
            "artifact_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
        },
        r5a_method_preflight_binding={
            "artifact_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
        },
        checkpoint_sha256="5" * 64,
        cohort_audit_sha256="6" * 64,
        r5_stack_fingerprint_sha256="7" * 64,
        bundle_path=None,
        seed=0,
    )
    assert disposition["status"] == "BASIS_INSTRUMENT_UNRESOLVED"
    assert disposition["frozen_basis_runtime_contract"] is None
    assert disposition["frozen_basis_sufficient_replay"] is None
    assert disposition["sufficient_replay"][
        "nested_basis_instrument_failures"
    ]["validation"]["code"] == "INSUFFICIENT_NUMERICAL_RANK"
    assert bundle is None

    monkeypatch.setattr(
        r5b,
        "energy_matched_rank_random_projections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            r5b.InstrumentError("unclassified validation scorer/core failure")
        ),
    )
    with pytest.raises(
        r5b.InstrumentError, match="unclassified validation scorer/core failure"
    ):
        r5b.score_r5b_validation(
            object(),
            object(),
            rows1,
            frozen_derivation,
            batch_size=128,
            n_draws=199,
            n_overlap_draws=199,
            seed=0,
        )


def test_derivation_cli_scores_after_authorization_then_publishes_and_completes(
    tmp_path, monkeypatch
):
    events = []
    artifact = tmp_path / "r5b-derivation.json"
    gtf = tmp_path / "genes.gtf"
    fasta = tmp_path / "genome.fa"
    fai = tmp_path / "genome.fa.fai"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_file = checkpoint / "weights.bin"
    for path in (gtf, fasta, fai, checkpoint_file):
        path.write_text(path.name)
    upstream_paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "r3d_artifact",
            "r3d_manifest",
            "r3d_registry",
            "r3d_registry_manifest",
            "r3d_attempt",
            "r3d_completion",
            "r3c_artifact",
            "r3c_manifest",
            "r5a_preflight_artifact",
            "r5a_preflight_manifest",
        )
    }
    for path in upstream_paths.values():
        path.write_text("{}")

    attempt_repo = tmp_path / "attempt-repo"
    attempt_repo.mkdir()
    subprocess.run(["git", "init", "-q", str(attempt_repo)], check=True)
    checkpoint_hash = "b" * 64
    protocol_hash = r5b.sha256_file(r5b.PROTOCOL)
    fingerprint = "c" * 64
    stack_fingerprint = "d" * 64
    cohort_input_hash = "e" * 64
    cohort_audit_hash = "f" * 64
    upstream = {
        "r3d": {
            "artifact": {"sha256": "1" * 64},
            "manifest": {"sha256": "2" * 64},
            "checkpoint_and_data": {
                "checkpoint_aggregate_sha256": checkpoint_hash
            },
        },
        "r5a": {
            "artifact": {"sha256": "3" * 64},
            "manifest": {"sha256": "4" * 64},
        },
        "exclusion_registry": {"record_sha256": "5" * 64},
        "chain_sha256": "6" * 64,
    }
    authorization = {
        "stage": "derivation",
        "instrument_fingerprint_sha256": fingerprint,
        "r5_stack_fingerprint_sha256": stack_fingerprint,
        "r5_stack_binding": {},
    }
    source_identity = {"fixture": "source"}
    cohort_audit = {"fixture": "audit"}
    adapter = _TinyAdapter()
    receipt_holder = {}

    def authorize(stage, *, device):
        events.append("authorize")
        assert stage == "derivation"
        return authorization

    def load_upstream(**_kwargs):
        events.append("upstream")
        return upstream

    def reserve(args, supplied_authorization, supplied_upstream, *, predecessor):
        events.append("reserve")
        assert supplied_authorization is authorization
        assert supplied_upstream is upstream
        assert predecessor is None
        identity = r5b.build_r5b_residual_attempt_identity(
            stage="derivation",
            checkpoint_sha256=checkpoint_hash,
            cohort_input_sha256=cohort_input_hash,
            cohort_audit_sha256=cohort_audit_hash,
        )
        receipt = r5b.reserve_r5b_residual_attempt(
            repo=attempt_repo,
            identity=identity,
            method_context={
                "protocol_sha256": protocol_hash,
                "instrument_fingerprint_sha256": fingerprint,
                "r5_stack_fingerprint_sha256": stack_fingerprint,
                "r3d_artifact_sha256": "1" * 64,
                "r3d_manifest_sha256": "2" * 64,
                "r5a_preflight_artifact_sha256": "3" * 64,
                "r5a_preflight_manifest_sha256": "4" * 64,
                "predecessor_artifact_sha256": None,
                "predecessor_manifest_sha256": None,
                "predecessor_completion_sha256": None,
                "requested_artifact_logical_name": artifact.name,
                "requested_manifest_logical_name": artifact.with_suffix(
                    ".manifest.json"
                ).name,
                "declared_completion_logical_name": (
                    f"{identity['residual_attempt_key']}.completion.json"
                ),
                "scoring_origin": "direct_checkpoint_scoring_in_r5b_process",
            },
        )
        receipt_holder["receipt"] = receipt
        return receipt

    class CleanManifest:
        git_dirty = False

        def __init__(self):
            self.inputs = []
            self.outputs = []

        def to_dict(self):
            return {
                "git_dirty": False,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "fixture": "clean-manifest",
            }

    def create_manifest(*_args, **_kwargs):
        events.append("manifest")
        return CleanManifest()

    monkeypatch.setattr(r5b, "PRODUCTION_BATCH", r5b.PRODUCTION_N)
    monkeypatch.setattr(r5b, "PRODUCTION_INFERENCE_DRAWS", 199)
    monkeypatch.setattr(r5b, "PRODUCTION_OVERLAP_DRAWS", 199)
    monkeypatch.setattr(r5b, "FROZEN_PROTOCOL_SHA256", protocol_hash)
    monkeypatch.setattr(r5b, "FROZEN_CHECKPOINT_AGGREGATE_SHA256", checkpoint_hash)
    monkeypatch.setattr(r5b, "FROZEN_COHORT_INPUT_SHA256", cohort_input_hash)
    monkeypatch.setattr(r5b, "FROZEN_COHORT_AUDIT_SHA256", cohort_audit_hash)
    monkeypatch.setattr(r5b, "PINNED_CHECKPOINT_AGGREGATE_SHA256", checkpoint_hash)
    monkeypatch.setattr(r5b, "authorize_scored_stage", authorize)
    monkeypatch.setattr(r5b, "load_r5b_upstream_chain", load_upstream)
    monkeypatch.setattr(r5b, "_reserve_stage_attempt", reserve)
    monkeypatch.setattr(
        r5b,
        "build_r5b_residual_reservations",
        lambda *_args, **_kwargs: (
            {0: tuple(_tiny_residual_rows()), 1: tuple()},
            cohort_audit,
            source_identity,
        ),
    )
    monkeypatch.setattr(
        r5b,
        "checkpoint_identity",
        lambda _path: {
            "root": str(checkpoint),
            "files": [
                {
                    "relative_path": checkpoint_file.name,
                    "size_bytes": checkpoint_file.stat().st_size,
                    "sha256": r5b.sha256_file(checkpoint_file),
                }
            ],
            "aggregate_schema": "fixture",
            "aggregate_sha256": checkpoint_hash,
        },
    )
    monkeypatch.setattr(r5b, "pick_device", lambda _value: torch.device("cpu"))

    def load_model(*_args, **_kwargs):
        events.append("model")
        return adapter

    monkeypatch.setattr(r5b, "load_adapter", load_model)
    monkeypatch.setattr(r5b, "CouplingTask", lambda *_args, **_kwargs: _TinyAcceptorTask())
    monkeypatch.setattr(
        r5b, "validate_expected_r5_source_identity", lambda *_args: source_identity
    )
    monkeypatch.setattr(r5b, "instrument_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(r5b, "_stack_source_paths", lambda _authorization: [])
    monkeypatch.setattr(r5b.ExperimentManifest, "create", create_manifest)

    argv = [
        "--stage",
        "derivation",
        "--out",
        str(artifact),
        "--model",
        str(checkpoint),
        "--gtf",
        str(gtf),
        "--fasta",
        str(fasta),
        "--batch",
        str(r5b.PRODUCTION_N),
        "--draws",
        "199",
        "--overlap-draws",
        "199",
    ]
    for name, path in upstream_paths.items():
        argv.extend([f"--{name.replace('_', '-')}", str(path)])
    r5b.main(argv)

    published = json.loads(artifact.read_text())
    receipt = receipt_holder["receipt"]
    completion = Path(receipt["receipt_path"]).with_suffix(".completion.json")
    assert events.index("authorize") < events.index("reserve") < events.index("model")
    assert events.index("model") < events.index("manifest")
    assert adapter.model.forward_calls >= 20
    assert published["direct_model_scoring"] is True
    assert published["external_provider_artifact_accepted"] is False
    assert published["scored_provider"]["origin"] == (
        "direct_checkpoint_scoring_in_this_r5b_process"
    )
    assert artifact.with_suffix(".manifest.json").is_file()
    assert completion.is_file()


def _write_clean_json_publication(directory, name, record):
    artifact = directory / name
    artifact.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    manifest = artifact.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "git_dirty": False,
                "outputs": [
                    {
                        "path": str(artifact),
                        "size_bytes": artifact.stat().st_size,
                        "sha256": r5b.sha256_file(artifact),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return artifact, manifest


def test_nonvalidated_disposition_authenticates_exact_development_packages(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(r5b, "PRODUCTION_INFERENCE_DRAWS", 199)
    monkeypatch.setattr(r5b, "PRODUCTION_OVERLAP_DRAWS", 199)
    derivation = r5b.score_r5b_derivation(
        _TinyAdapter(),
        _TinyAcceptorTask(),
        _tiny_residual_rows("b0"),
        batch_size=r5b.PRODUCTION_N,
        n_draws=199,
        seed=0,
    )
    validation = r5b.score_r5b_validation(
        _TinyAdapter(),
        _TinyAcceptorTask(),
        _tiny_residual_rows("b1"),
        derivation,
        batch_size=r5b.PRODUCTION_N,
        n_draws=199,
        n_overlap_draws=199,
        seed=0,
    )
    derivation_path, derivation_manifest = _write_clean_json_publication(
        tmp_path, "derivation.json", derivation
    )
    validation_path, validation_manifest = _write_clean_json_publication(
        tmp_path, "validation.json", validation
    )
    packages = {
        "derivation": {
            "artifact": r5b._publication_identity(derivation_path),
            "manifest": r5b._publication_identity(derivation_manifest),
        },
        "validation": {
            "artifact": r5b._publication_identity(validation_path),
            "manifest": r5b._publication_identity(validation_manifest),
        },
    }
    disposition, bundle = r5b.build_local_writer_basis_disposition(
        derivation,
        validation,
        development_packages=packages,
        r3d_path_a_binding={
            "artifact_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
        },
        r5a_method_preflight_binding={
            "artifact_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
        },
        checkpoint_sha256="5" * 64,
        cohort_audit_sha256="6" * 64,
        r5_stack_fingerprint_sha256="7" * 64,
        bundle_path=None,
        seed=0,
    )
    assert bundle is None
    disposition_path, disposition_manifest = _write_clean_json_publication(
        tmp_path, "disposition.json", disposition
    )
    authenticated = r5b.validate_local_writer_basis_disposition(
        disposition_path,
        disposition_manifest,
        derivation_artifact_path=derivation_path,
        derivation_manifest_path=derivation_manifest,
        validation_artifact_path=validation_path,
        validation_manifest_path=validation_manifest,
    )
    assert set(authenticated) == {
        "authenticated",
        "artifact",
        "manifest",
        "status",
        "path_local_c2_disposition",
        "frozen_basis_runtime_contract",
        "dense_primary_unaffected",
        "record",
    }
    assert authenticated["authenticated"] is True
    assert authenticated["status"] == (
        "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS"
    )
    assert authenticated["frozen_basis_runtime_contract"] is None
    assert authenticated["dense_primary_unaffected"] is True
