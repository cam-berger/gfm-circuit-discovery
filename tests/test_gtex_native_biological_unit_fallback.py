from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from gfmcd.splicing.mlp_units import composed_classifier_margin


REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO / "scripts" / "splicebert"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "gtex_native_biological_unit_fallback",
    SCRIPT_DIR / "gtex_native_biological_unit_fallback.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def pair(example: str, gene: str, low: str, high: str, target: int = 2) -> dict:
    return {
        "example_id": example,
        "gene_id": gene,
        "target_position": target,
        "low_sequence": low,
        "high_sequence": high,
        "low_sha256": MODULE.sha256_text(low),
        "high_sha256": MODULE.sha256_text(high),
    }


def test_negative_candidate_orientation_is_preferred_low_adverse_high():
    states = {
        "low": {"margin": np.asarray([3.0]), "activation": np.asarray([[1.0]])},
        "high": {"margin": np.asarray([1.0]), "activation": np.asarray([[0.0]])},
    }
    preferred, adverse = MODULE.oriented_arrays(states, -1)
    assert MODULE.oriented_state_names(-1) == ("low", "high")
    assert preferred["margin"][0] - adverse["margin"][0] == 2.0
    assert MODULE.oriented_state_names(1) == ("high", "low")


def test_matched_controls_use_activity_and_preserve_size_uniqueness():
    candidates = np.asarray([0, 1])
    norms = np.ones(8)
    rms = np.asarray([1.0, 3.0, 1.0, 3.0, 8.0, 9.0, 10.0, 11.0])
    rate = np.asarray([0.2, 0.8, 0.2, 0.8, 0.0, 0.0, 1.0, 1.0])
    change = np.asarray([0.4, 0.6, 0.4, 0.6, 0.0, 1.0, 0.0, 1.0])
    controls = MODULE.matched_native_unit_control_orders(
        candidates,
        norms,
        activation_rms=rms,
        positive_activation_rate=rate,
        sequence_change_rate=change,
        tag="focused-test",
        draws=3,
    )
    for control in controls:
        assert control.tolist() == [2, 3]
        assert len(control) == len(candidates)
        assert len(np.unique(control)) == len(control)
        assert not np.intersect1d(control, candidates).size


def test_all_unit_write_sum_and_exact_composition_preflight():
    rng = np.random.default_rng(7)
    n, units, width = 5, 7, 4
    weight = rng.normal(size=(width, units))
    bias = rng.normal(size=width)
    parameters = {
        "layer_norm_weight": np.ones(width),
        "layer_norm_bias": np.zeros(width),
        "layer_norm_eps": 1e-12,
        "classifier_direction": rng.normal(size=width),
        "classifier_bias": 0.3,
    }
    states = {}
    for name in ("low", "high"):
        activation = rng.normal(size=(n, units))
        mlp_input = rng.normal(size=(n, width))
        mlp_write = activation @ weight.T + bias
        margin = composed_classifier_margin(
            mlp_input, mlp_write, **parameters
        )
        states[name] = {
            "activation": activation,
            "mlp_input": mlp_input,
            "mlp_write": mlp_write,
            "margin": margin,
        }
    import gtex_native_biological_unit_runtime as runtime

    old_size = runtime.INTERMEDIATE_SIZE
    runtime.INTERMEDIATE_SIZE = units
    try:
        result = runtime.preflight_family(states, weight, parameters)
    finally:
        runtime.INTERMEDIATE_SIZE = old_size
    # Runtime imported its own constant, so test the core exact identity directly.
    delta = states["high"]["activation"] - states["low"]["activation"]
    from gfmcd.splicing.mlp_units import unit_group_write_delta

    reconstructed = unit_group_write_delta(delta, weight, range(units))
    observed = states["high"]["mlp_write"] - states["low"]["mlp_write"]
    np.testing.assert_allclose(reconstructed, observed, atol=1e-12)
    assert result["PASS"] is True


def test_confirmation_flat_state_schema_filters_only_frozen_eligible_ids():
    cohort = [
        {"example_id": "e1", "gene_id": "g1"},
        {"example_id": "e2", "gene_id": "g2"},
    ]
    states = []
    for family, role in (("CAND", "candidate"), ("CTRL", "matched_control")):
        for example, gene in (("e1", "g1"), ("e2", "g2")):
            for direction, sequence in (("low", "AACG"), ("high", "AATG")):
                states.append(
                    {
                        "candidate_id": "donor:CAND",
                        "candidate_name": "CAND",
                        "endpoint": "donor",
                        "family_name": family,
                        "family_role": role,
                        "direction": direction,
                        "expected_high_minus_low_margin_sign": -1,
                        "example_id": example,
                        "gene_id": gene,
                        "target_position": 2,
                        "sequence": sequence,
                        "sequence_sha256": MODULE.sha256_text(sequence),
                        "pair_success": True,
                    }
                )
    package = {
        "schema": MODULE.CONFIRMATION_PACKAGE_SCHEMA,
        "status": "COMPLETE",
        "confirmation_authorized": True,
        "model_loaded": False,
        "confirmation_model_outputs_read": False,
        "outcome_blind": True,
        "cohort": {"n_genes": 2, "examples": cohort},
        "candidate_summaries": [
            {
                "candidate_id": "donor:CAND",
                "candidate_name": "CAND",
                "endpoint": "donor",
                "matched_controls": ["CTRL"],
                "expected_high_minus_low_margin_sign": -1,
                "eligible_example_ids": ["e1"],
                "confirmation_construction_gate": True,
            }
        ],
        "states": states,
    }
    normalized = MODULE.normalize_confirmation_states(package, expected_genes=2)
    assert normalized["n_genes"] == 2
    assert normalized["analysis_gene_ids"] == ["g1"]
    assert list(normalized["candidates"][0]["families"]) == ["CAND", "CTRL"]
    assert all(
        len(family["pairs"]) == 1
        for family in normalized["candidates"][0]["families"].values()
    )
    stopped = dict(package)
    stopped["status"] = "STOPPED_MINIMUM_COHORT"
    stopped["confirmation_authorized"] = False
    with pytest.raises(RuntimeError, match="stopped"):
        MODULE.normalize_confirmation_states(stopped, expected_genes=2)


def test_discovery_codon_replay_checks_hashes_and_candidate_pass():
    clean = "AAACCCGGG"
    low = "AAATCCGGG"
    high = "AAAGCCGGG"
    sequence_package = {
        "label_free": True,
        "states": [
            {
                "variant_kind": "clean",
                "endpoint": endpoint,
                "example_id": f"{endpoint}-e",
                "gene_id": f"{endpoint}-g",
                "target_position": 4,
                "sequence": clean,
                "sequence_sha256": MODULE.sha256_text(clean),
            }
            for endpoint in ("donor", "acceptor")
        ],
    }

    def family(endpoint: str, name: str) -> dict:
        row = {
            "success": True,
            "example_id": f"{endpoint}-e",
            "gene_id": f"{endpoint}-g",
            "low": {
                "codon_changes": [
                    {"codon_start": 3, "original": "CCC", "alternate": "TCC"}
                ]
            },
            "high": {
                "codon_changes": [
                    {"codon_start": 3, "original": "CCC", "alternate": "GCC"}
                ]
            },
            "low_sha256": MODULE.sha256_text(low),
            "high_sha256": MODULE.sha256_text(high),
        }
        return {
            "rows": [row],
            "success_count": 1,
            "success_gene_count": 1,
        }

    audit = {
        "experiment": "gtex_native_biological_construction_audit_v2",
        "status": "COMPLETE",
        "model_loaded": False,
        "audit": {
            endpoint: {
                f"{endpoint}-cand": {
                    "candidate_construction_pass": True,
                    "expected_high_minus_low_margin_sign": -1,
                    "matched_controls": [f"{endpoint}-ctrl"],
                    "families": {
                        f"{endpoint}-cand": family(endpoint, "cand"),
                        f"{endpoint}-ctrl": family(endpoint, "ctrl"),
                    },
                }
            }
            for endpoint in ("donor", "acceptor")
        },
    }
    normalized = MODULE.normalize_discovery_audit(audit, sequence_package)
    assert len(normalized["candidates"]) == 2
    candidate = normalized["candidates"][0]
    assert candidate["families"][candidate["candidate_id"]]["pairs"][0][
        "low_sequence"
    ] == low
    failed = json.loads(json.dumps(audit))
    first_endpoint = next(iter(failed["audit"]))
    first_candidate = next(iter(failed["audit"][first_endpoint].values()))
    first_candidate["candidate_construction_pass"] = False
    with pytest.raises(RuntimeError, match="candidate failed"):
        MODULE.normalize_discovery_audit(failed, sequence_package)


def test_real_corrected_discovery_audit_replays_without_model_compute():
    result_root = REPO / "results" / "splicebert_gtex_finetune_shift"
    audit_path = result_root / "native_biological_construction_audit_v2.json"
    sequence_path = result_root / "sparse_transcoder_sequence_package_v1.json"
    if not audit_path.is_file() or not sequence_path.is_file():
        pytest.skip("tracked biological discovery artifacts unavailable")
    normalized = MODULE.normalize_discovery_audit(
        json.loads(audit_path.read_text()),
        json.loads(sequence_path.read_text()),
    )
    assert len(normalized["candidates"]) == 4
    assert normalized["source_schema"] == "corrected_construction_audit_v2_replay"
    assert all(
        candidate["families"][candidate["candidate_id"]]["pairs"]
        for candidate in normalized["candidates"]
    )


def test_rank4_energy_and_five_seed_sign_replication():
    basis = np.eye(4)
    values = np.asarray([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]])
    np.testing.assert_allclose(MODULE.rank4_energy_fraction(values, basis), 1.0)
    effects = np.tile(np.asarray([[1.0, 2.0, 3.0]]), (5, 1))
    summary = MODULE.summarize_five_seed_effect(
        effects,
        np.asarray(["g1", "g2", "g3"], dtype=object),
        seed=9,
        n_boot=20,
        n_perm=19,
    )
    assert summary["all_five_checkpoint_signs_positive"] is True
    assert summary["p_signflip_add_one"] >= 1 / 20
