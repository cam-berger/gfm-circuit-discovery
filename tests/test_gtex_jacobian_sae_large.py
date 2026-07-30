import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from gfmcd.circuits.jacobian_sae import (
    JacobianSAEConfig,
    JacobianSAEPair,
    scalar_normalization,
)


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/splicebert/run_gtex_jacobian_sae_large.py"
SPEC = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def synthetic_states():
    states = []
    for endpoint in RUNNER.ENDPOINTS:
        for gene in ("gene-a", "gene-b"):
            clean_id = f"{endpoint}:{gene}:clean"
            states.append(
                {
                    "state_id": clean_id,
                    "reference_state_id": None,
                    "gene_id": gene,
                    "endpoint": endpoint,
                    "variant_kind": "clean",
                }
            )
            for variant in (
                "target_shuffle",
                "sham_shuffle",
                "max_recode_draw0",
                "max_recode_draw1",
                "synonymous_single",
            ):
                states.append(
                    {
                        "state_id": f"{endpoint}:{gene}:{variant}",
                        "reference_state_id": clean_id,
                        "gene_id": gene,
                        "endpoint": endpoint,
                        "variant_kind": variant,
                    }
                )
    return states


def small_pair() -> JacobianSAEPair:
    torch.manual_seed(11)
    inputs = torch.randn(12, 4)
    writes = torch.randn(12, 3)
    input_center, input_scale = scalar_normalization(inputs)
    output_center, output_scale = scalar_normalization(writes)
    return JacobianSAEPair(
        JacobianSAEConfig(
            input_dim=4,
            output_dim=3,
            mlp_dim=7,
            num_features=8,
            output_num_features=6,
            active_features=2,
        ),
        input_weight=torch.randn(7, 4),
        input_bias=torch.randn(7),
        output_weight=torch.randn(3, 7),
        output_bias=torch.randn(3),
        input_center=input_center,
        input_scale=input_scale,
        output_center=output_center,
        output_scale=output_scale,
    )


def test_reference_pairs_are_oriented_and_family_complete():
    states = synthetic_states()
    pairs = RUNNER.build_reference_pairs(
        states,
        np.ones(len(states), dtype=bool),
    )
    assert len(pairs) == 20
    assert {
        (row["endpoint"], row["family"]) for row in pairs
    } == {
        (endpoint, family)
        for endpoint in RUNNER.ENDPOINTS
        for family in RUNNER.PAIR_FAMILIES
    }
    for row in pairs:
        assert states[row["reference"]]["variant_kind"] == "clean"
        assert states[row["changed"]]["variant_kind"] != "clean"


def test_pair_schedule_is_deterministic_balanced_and_complete():
    states = synthetic_states()
    pairs = RUNNER.build_reference_pairs(
        states,
        np.ones(len(states), dtype=bool),
    )
    left, left_report = RUNNER.balanced_pair_schedule(
        pairs,
        draws=160,
        seed=9,
    )
    right, right_report = RUNNER.balanced_pair_schedule(
        pairs,
        draws=160,
        seed=9,
    )
    assert np.array_equal(left, right)
    assert left_report == right_report
    assert left_report["PASS"]
    assert left_report["coverage_fraction"] == 1.0
    counts = list(left_report["counts_by_stratum"].values())
    assert max(counts) - min(counts) <= 1


def test_fixed_state_epoch_pads_to_complete_batches_deterministically():
    genes = ["gene-a"] * 11 + ["gene-b"] * 3 + ["gene-c"] * 7
    indices = np.arange(len(genes))
    left = RUNNER.fixed_state_epoch(indices, genes, draws=64, seed=17)
    right = RUNNER.fixed_state_epoch(indices, genes, draws=64, seed=17)
    assert np.array_equal(left, right)
    assert len(left) == 64
    assert set(left).issubset(set(indices))
    assert len(left) % RUNNER.STATE_BATCH == 0


def test_delta_power_is_equal_stratum_then_equal_gene_weighted():
    class ScaleOnly:
        input_scale = torch.ones(())
        output_scale = torch.ones(())

    inputs = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [3.0]])
    writes = inputs * 2
    pairs = [
        {
            "reference": 0,
            "changed": 1,
            "gene_id": "gene-a",
            "endpoint": "donor",
            "family": "target_shuffle",
        },
        {
            "reference": 2,
            "changed": 3,
            "gene_id": "gene-a",
            "endpoint": "donor",
            "family": "target_shuffle",
        },
        {
            "reference": 4,
            "changed": 5,
            "gene_id": "gene-b",
            "endpoint": "donor",
            "family": "target_shuffle",
        },
    ]
    powers = RUNNER.delta_powers(ScaleOnly(), inputs, writes, pairs)
    assert powers["input"] == 5.0
    assert powers["output"] == 20.0


def test_paired_delta_loss_is_finite_and_trains_both_saes():
    model = small_pair()
    inputs = torch.randn(8, 4)
    writes = torch.randn(8, 3)
    indices = np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]])
    loss = RUNNER.paired_delta_loss(
        model,
        inputs,
        writes,
        indices,
        {"input": 1.0, "output": 1.0},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.input_sae.encoder_weight.grad is not None
    assert model.output_sae.encoder_weight.grad is not None
    assert model.mlp_input_weight.grad is None


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS accelerator is unavailable",
)
def test_cpu_activation_indices_work_with_accelerator_model():
    model = small_pair().to(torch.device("mps"))
    inputs = torch.randn(8, 4)
    writes = torch.randn(8, 3)
    indices = np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]])
    loss = RUNNER.paired_delta_loss(
        model,
        inputs,
        writes,
        indices,
        {"input": 1.0, "output": 1.0},
    )
    assert torch.isfinite(loss)
    pairs = [
        {
            "reference": int(reference),
            "changed": int(changed),
            "gene_id": f"gene-{row}",
            "endpoint": "donor",
            "family": "target_shuffle",
        }
        for row, (reference, changed) in enumerate(indices)
    ]
    report = RUNNER.delta_fve(model, inputs, writes, pairs)
    assert np.isfinite(
        report["by_stratum"]["donor:target_shuffle"][
            "output_delta_energy_fve"
        ]
    )


def test_large_pair_has_frozen_widths_and_active_graph_shape():
    torch.manual_seed(3)
    weights = {
        "input_weight": torch.randn(2048, 512) / 32,
        "input_bias": torch.zeros(2048),
        "output_weight": torch.randn(512, 2048) / 32,
        "output_bias": torch.zeros(512),
    }
    inputs = torch.randn(12, 512)
    writes = torch.randn(12, 512)
    torch.manual_seed(99)
    first = RUNNER.build_large_pair(
        weights,
        inputs,
        writes,
        device=torch.device("cpu"),
    )
    first_hash = RUNNER.state_hash(first)
    torch.manual_seed(99)
    second = RUNNER.build_large_pair(
        weights,
        inputs,
        writes,
        device=torch.device("cpu"),
    )
    assert first_hash == RUNNER.state_hash(second)
    run = first(inputs[:2])
    assert first.input_sae.num_features == 512
    assert first.output_sae.num_features == 128
    assert run["active_jacobian"].shape == (2, 16, 16)


def test_registered_factorial_and_representation_gate():
    assert {
        (
            row["delta_coefficient"] > 0,
            row["jacobian_coefficient"] > 0,
        )
        for row in RUNNER.CONDITIONS.values()
    } == {(False, False), (False, True), (True, False), (True, True)}

    validation = {
        "input_fve": 0.97,
        "output_write_fve": 0.98,
        "sandwich_output_fve": 0.95,
        "active_edge_mass_count": {"E90": {"median": 70.0}},
    }
    matched_validation = {
        **validation,
        "active_edge_mass_count": {"E90": {"median": 100.0}},
    }
    strata = {
        f"{endpoint}:{family}": {
            "input_delta_energy_fve": (
                0.92
                if family in {"target_shuffle", "sham_shuffle"}
                else 0.87
            ),
            "output_delta_energy_fve": (
                0.97
                if family in {"target_shuffle", "sham_shuffle"}
                else 0.92
            ),
        }
        for endpoint in RUNNER.ENDPOINTS
        for family in RUNNER.PAIR_FAMILIES
    }
    primary = {
        "train": {
            "input_dead_feature_fraction": 0.1,
            "output_dead_feature_fraction": 0.2,
        },
        "validation": validation,
        "delta_validation": {"by_stratum": strata},
        "graph_metrics": {
            "effective_active_edge_count": {"median": 60.0}
        },
        "autograd_control": {"PASS": True},
        "finite_difference_control": {"PASS": True},
        "decoder_invariants": {"PASS": True},
        "training": {"auxiliary": {"late_fraction": 0.1}},
    }
    matched = {
        "validation": matched_validation,
        "graph_metrics": {
            "effective_active_edge_count": {"median": 100.0}
        },
    }
    no_delta = {"delta_validation": {"by_stratum": strata}}
    gate = RUNNER.representation_gate(primary, matched, no_delta)
    assert gate["PASS"]
    primary["delta_validation"]["by_stratum"][
        "donor:sham_shuffle"
    ]["output_delta_energy_fve"] = 0.80
    assert not RUNNER.representation_gate(primary, matched, no_delta)["PASS"]
