from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert_acceptor_coupling/run_r5a_acceptor_address_binding.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_r5a_acceptor_address_binding", SCRIPT
)
assert SPEC is not None
R5A = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(R5A)


def _noisy_plant(
    n: int = 40,
    *,
    position: tuple[float, float] = (0.40, 0.40),
    qk_partner: float = 0.40,
    qk_control: float = 0.0,
    av_partner: float = 0.40,
    l2_qk_partner: float = 0.0,
):
    return R5A._add_centered_primitive_noise(
        R5A._synthetic_measurements(
            n,
            position=position,
            qk_partner=qk_partner,
            qk_control=qk_control,
            av_partner=av_partner,
            l2_qk_partner=l2_qk_partner,
        ),
        rng=np.random.default_rng(11),
        scale=0.01,
    )


def test_v3_registry_is_margin_only_exact_400_and_has_separate_l2_secondary():
    rows = R5A.family_row_names()
    assert tuple(rows) == R5A.FAMILY_ORDER == (
        "position_margin",
        "qk_surgery",
        "attention_value",
        "l2h15_qk_secondary",
    )
    assert {name: len(value) for name, value in rows.items()} == {
        "position_margin": 12,
        "qk_surgery": 15,
        "attention_value": 69,
        "l2h15_qk_secondary": 15,
    }
    assert R5A.EXACT_ROWS == R5A.PRODUCTION_N == 400
    assert R5A.MIN_GENES >= 30
    assert tuple(R5A.named_path_specs()) == R5A.ALL_PATH_ORDER == (
        "G1_POSITION_CAUSAL_MARGIN",
        "G2_EXACT_QK_CAUSAL_MARGIN",
        "G3_FIXED_V_AXV_CAUSAL_MARGIN",
        "S1_L2H15_EXACT_QK_SECONDARY",
    )
    assert R5A.CORE_PATH_ORDER == R5A.ALL_PATH_ORDER[:3]
    assert R5A.SECONDARY_PATH_ORDER == R5A.ALL_PATH_ORDER[3:]
    decision_rows = [row for family in rows.values() for row in family]
    assert all(
        "measure=acceptor_margin" in row or not row.startswith("POSITION|")
        for row in decision_rows
    )
    assert not any(
        "qk_preference" in row or "attention_preference" in row
        for row in decision_rows
    )
    assert not any("adjacent" in row.lower() for row in decision_rows)
    assert all(
        "target=L3H1:paired" not in row
        for row in rows["l2h15_qk_secondary"]
        if row.startswith("QK_MARGIN_CONTRAST")
    )
    runner = Path(R5A.RUNNER).read_text()
    assert "L2H15" in runner
    assert "L1H12" not in runner


def test_raw_cells_replay_exact_factorials_and_target_minus_control_contrasts():
    matrices = R5A.build_family_matrices(
        R5A._synthetic_measurements(
            7,
            position=(0.40, 0.40),
            qk_partner=0.40,
            l2_qk_partner=0.40,
            av_partner=0.40,
        )
    )
    names = R5A.family_row_names()
    position = dict(zip(names["position_margin"], matrices["position_margin"]))
    np.testing.assert_allclose(
        position[
            R5A.position_row_name(
                "L3H1",
                "paired_pseudo",
                "acceptor_margin",
                "position_effect_at_native_content",
            )
        ],
        0.40,
    )
    np.testing.assert_allclose(
        position[
            R5A.position_margin_contrast_name(
                "L0H2:paired_pseudo",
                "position_effect_at_native_content",
            )
        ],
        0.40,
    )

    qk = dict(zip(names["qk_surgery"], matrices["qk_surgery"]))
    np.testing.assert_allclose(
        qk[R5A.qk_margin_contrast_name("L0H2:paired", "specific")],
        0.40,
    )

    av = dict(zip(names["attention_value"], matrices["attention_value"]))
    for quantity in (
        "address_effect_at_value_plus",
        "value_effect_at_address_plus",
    ):
        np.testing.assert_allclose(
            av[R5A.av_row_name("paired", quantity, "specific")],
            0.40,
        )
        np.testing.assert_allclose(
            av[R5A.av_margin_contrast_name("pseudo", quantity, "specific")],
            0.40,
        )

    l2 = dict(
        zip(
            names["l2h15_qk_secondary"],
            matrices["l2h15_qk_secondary"],
        )
    )
    np.testing.assert_allclose(
        l2[R5A.qk_row_name("L2H15", "paired", "specific")],
        0.40,
    )
    np.testing.assert_allclose(
        l2[
            R5A.qk_margin_contrast_name(
                "L0H2:paired", "specific", target="L2H15"
            )
        ],
        0.40,
    )


def test_internal_qk_and_attention_bridges_have_units_but_no_decision_threshold():
    primitive = R5A._synthetic_measurements(20, position=(0.40, 0.40))
    diagnostic = R5A.position_bridge_diagnostics(primitive)
    assert diagnostic["decision_bearing"] is False
    assert diagnostic["query_policy"] == (
        "same numerical clean query vector in every 2x2 cell"
    )
    assert diagnostic["shifted_native_query_included_in_decisions"] is False
    assert diagnostic["scale_specific_thresholds"] is None
    assert diagnostic["rows"]
    units = {record["unit"] for record in diagnostic["rows"].values()}
    assert units == {
        "pre_softmax_logit_difference",
        "post_softmax_probability_difference",
    }
    assert all(
        record["zero_null_or_margin_sesoi_applied"] is False
        for record in diagnostic["rows"].values()
    )


def test_dual_estimator_bootstrap_rejects_tail_only_mean_and_is_not_exact():
    values = np.zeros(100, dtype=float)
    values[-10:] = 1.0
    result = R5A.dual_estimator_studentized_bootstrap_inversion(
        values[None, :],
        [f"gene-{index:03d}" for index in range(100)],
        ["target"],
        {"target": "MATERIAL_POSITIVE"},
        n_draws=999,
        seed=0,
        namespace="tail-adversary",
        min_genes=100,
    )
    estimators = result["requirements"]["target"]["estimators"]
    assert estimators["mean"]["estimate"] > R5A.MARGIN_SESOI
    assert estimators["trimmed_mean_10pct"]["estimate"] == 0.0
    assert result["classification"] == "NOT_SUPPORTED"
    assert "not exact" in result["finite_draw_language"]
    assert result["B"] == 999
    assert R5A.PRODUCTION_RANDOMISATION >= 10_000


def test_strong_coherent_plant_passes_all_three_ordered_gates():
    n = 40
    result = R5A.infer_all_families(
        R5A.build_family_matrices(_noisy_plant(n)),
        [f"gene-{index:03d}" for index in range(n)],
        n_draws=199,
        seed=0,
        namespace="strong-plant",
        min_genes=n,
    )
    assert all(
        result["paths"][path]["classification"] == "SUPPORTED"
        for path in R5A.PATH_ORDER
    )
    assert result["decisions"] == {
        "reader": "L3H1_DONOR_LOCAL_READER_SUPPORTED",
        "opens_r5b": True,
        "l2h15_role": "L2H15_ROLE_NOT_SUPPORTED",
    }
    assert result["fixed_sequence"]["alpha_each_reached_gate"] == pytest.approx(
        0.05
    )
    assert result["fixed_sequence"]["alpha_fragmented"] is False


def test_l2h15_secondary_can_pass_without_becoming_headline_entry_requirement():
    n = 40
    result = R5A.infer_all_families(
        R5A.build_family_matrices(_noisy_plant(n, l2_qk_partner=0.40)),
        [f"gene-{index:03d}" for index in range(n)],
        n_draws=199,
        seed=0,
        namespace="l2-secondary-plant",
        min_genes=n,
    )
    assert result["decisions"] == {
        "reader": "L3H1_DONOR_LOCAL_READER_SUPPORTED",
        "opens_r5b": True,
        "l2h15_role": "L2H15_UPSTREAM_READER_ROLE_SUPPORTED",
    }
    secondary = result["paths"]["S1_L2H15_EXACT_QK_SECONDARY"]
    assert secondary["classification"] == "SUPPORTED"
    assert secondary["gates_l3_headline"] is False
    assert secondary["historical_r3c_health_used"] is False


def test_l2h15_secondary_failure_does_not_invalidate_supported_l3_core():
    n = 40
    result = R5A.infer_all_families(
        R5A.build_family_matrices(_noisy_plant(n)),
        [f"gene-{index:03d}" for index in range(n)],
        n_draws=199,
        seed=0,
        namespace="l2-secondary-null",
        min_genes=n,
    )
    assert result["decisions"]["reader"] == "L3H1_DONOR_LOCAL_READER_SUPPORTED"
    assert result["decisions"]["opens_r5b"] is True
    assert result["decisions"]["l2h15_role"] == "L2H15_ROLE_NOT_SUPPORTED"
    assert result["secondary_sequence"]["gates_l3_headline"] is False


def test_failed_gate_leaves_downstream_interventions_unexamined():
    n = 40
    result = R5A.infer_all_families(
        R5A.build_family_matrices(
            _noisy_plant(n, position=(0.0, 0.0))
        ),
        [f"gene-{index:03d}" for index in range(n)],
        n_draws=199,
        seed=0,
        namespace="fixed-sequence-stop",
        min_genes=n,
    )
    sequence = result["fixed_sequence"]
    assert sequence["tested_gates"] == ["G1_POSITION_CAUSAL_MARGIN"]
    assert sequence["stopped_after"] == "G1_POSITION_CAUSAL_MARGIN"
    for path in R5A.PATH_ORDER[1:]:
        assert result["paths"][path]["status"] == (
            "NOT_TESTED_DUE_TO_PRIOR_SCIENTIFIC_STOP"
        )


def test_reached_instrument_failure_does_not_claim_its_surgery_was_unexecuted():
    execution_log = [
        {
            "path": "G1_POSITION_CAUSAL_MARGIN",
            "model_intervention_executed": True,
            "capability_pass": False,
        }
    ]
    records = R5A.fixed_sequence_stop_records(
        {},
        execution_log,
        instrument_stop="position_margin",
        scientific_stop=None,
    )
    reached = records["G1_POSITION_CAUSAL_MARGIN"]
    assert reached["status"] == "INSTRUMENT_UNRESOLVED_AT_REACHED_GATE"
    assert reached["model_intervention_executed"] is True
    for path in R5A.PATH_ORDER[1:]:
        assert records[path]["status"] == (
            "NOT_TESTED_DUE_TO_PRIOR_INSTRUMENT_FAILURE"
        )
        assert records[path]["model_intervention_executed"] is False


def test_active_qk_control_is_scientific_failure_not_generic_unresolved():
    n = 40
    result = R5A.infer_all_families(
        R5A.build_family_matrices(
            _noisy_plant(n, qk_control=0.20)
        ),
        [f"gene-{index:03d}" for index in range(n)],
        n_draws=199,
        seed=0,
        namespace="active-control",
        min_genes=n,
    )
    qk = result["paths"]["G2_EXACT_QK_CAUSAL_MARGIN"]
    assert qk["classification"] == "NOT_SUPPORTED"
    assert qk["failure_mode"] == "NONSELECTIVE_OR_CONTROL_ACTIVE"
    assert result["paths"]["G3_FIXED_V_AXV_CAUSAL_MARGIN"]["status"] == (
        "NOT_TESTED_DUE_TO_PRIOR_SCIENTIFIC_STOP"
    )


def test_shift_swaps_exact_component_windows_and_is_involutive():
    captured = torch.arange(2 * 12 * 3, dtype=torch.float32).reshape(2, 12, 3)
    left = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    right = torch.tensor([[7, 8, 9, 10], [7, 8, 9, 10]])
    shifted = R5A.swap_component_tensor(captured, left, right)
    rows = torch.arange(2)[:, None]
    torch.testing.assert_close(shifted[rows, left], captured[rows, right])
    torch.testing.assert_close(shifted[rows, right], captured[rows, left])
    torch.testing.assert_close(
        R5A.swap_component_tensor(shifted, left, right), captured
    )
    with pytest.raises(ValueError, match="disjoint"):
        R5A.swap_component_tensor(captured, left, left)


def test_scientific_surgeries_require_only_paired_and_pseudo_windows():
    for function in (
        R5A.run_position_shift_batch,
        R5A.run_qk_batch,
        R5A.run_av_batch,
    ):
        assert "adjacent" not in inspect.signature(function).parameters
    for function in (R5A.run_position_shift_batch, R5A.run_qk_batch, R5A.run_av_batch):
        source = inspect.getsource(function)
        assert '"adjacent"' not in source
        assert "adjacent_positions" not in source


def test_qk_row_and_exact_four_logit_surgery_are_numerically_explicit():
    query = torch.zeros(2, 8, 4)
    key = torch.zeros_like(query)
    query[:, 5, 2:] = torch.tensor([1.0, 2.0])
    key[:, :, 2:] = (
        torch.arange(16, dtype=torch.float32)
        .reshape(2, 8, 1)
        .expand(-1, -1, 2)
    )
    logits, attention = R5A.qk_attention_row(
        query,
        key,
        head=1,
        query_positions=torch.tensor([5, 5]),
        attention_mask=torch.ones(2, 8),
        num_heads=2,
    )
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(2))
    source = logits + 10.0
    positions = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    edited = R5A.replace_selected_logits(logits, source, positions)
    rows = torch.arange(2)[:, None]
    torch.testing.assert_close(edited[rows, positions], source[rows, positions])
    mask = torch.zeros_like(edited, dtype=torch.bool)
    mask[rows, positions] = True
    torch.testing.assert_close(edited[~mask], logits[~mask])


def test_position_reader_uses_clean_query_with_shifted_keys():
    clean_query = torch.zeros(1, 6, 4)
    clean_query[0, 4, :2] = torch.tensor([1.0, 2.0])
    shifted_query = clean_query.clone()
    shifted_query[0, 4, :2] = torch.tensor([20.0, -10.0])
    shifted_key = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    encoded = {"attention_mask": torch.ones(1, 6)}
    query_tokens = torch.tensor([4])
    clean_state = {"query": clean_query, "key": shifted_key}
    shifted_state = {"query": shifted_query, "key": shifted_key}

    logits, attention = R5A._head_rows(
        shifted_state,
        encoded,
        query_tokens,
        head=0,
        num_heads=2,
        query_state=clean_state,
    )
    expected_logits, expected_attention = R5A.qk_attention_row(
        clean_query,
        shifted_key,
        head=0,
        query_positions=query_tokens,
        attention_mask=encoded["attention_mask"],
        num_heads=2,
    )
    shifted_native_logits, _ = R5A.qk_attention_row(
        shifted_query,
        shifted_key,
        head=0,
        query_positions=query_tokens,
        attention_mask=encoded["attention_mask"],
        num_heads=2,
    )
    torch.testing.assert_close(logits, expected_logits)
    torch.testing.assert_close(attention, expected_attention)
    assert not torch.equal(logits, shifted_native_logits)


def test_position_capability_records_numerically_fixed_query_per_cell():
    source = inspect.getsource(R5A.run_position_shift_batch)
    assert "query_state=clean_states[layer]" in source
    for field in (
        "fixed_query_sha256_by_head",
        "fixed_query_sha256_by_head_and_cell",
        "fixed_query_assignment_max_abs_error",
        "shifted_native_query_change_max_abs_by_head_and_cell",
    ):
        assert field in source


class _TinyMLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = torch.nn.Linear(3, 5, bias=False)

    def forward(self, input_ids, **_kwargs):
        hidden = torch.nn.functional.one_hot(input_ids % 3, num_classes=3).float()
        return type("Result", (), {"logits": self.cls(hidden)})()


def test_full_mlm_replay_uses_complete_cls_and_acceptor_margin_only():
    model = _TinyMLM()
    adapter = type("Adapter", (), {"model": model})()
    encoded = {"input_ids": torch.tensor([[0, 1, 2], [2, 1, 0]])}

    class Task:
        @staticmethod
        def margin(logits, positions):
            rows = torch.arange(len(positions))
            return logits[rows, positions, 2] - logits[rows, positions, 3]

    result = R5A.full_mlm_replay_capability(
        adapter, Task(), encoded, [1, 2]
    )
    assert result["ok"] is True
    assert result["uses_complete_model_cls"] is True
    assert result["donor_readout_constructed"] is False


def test_paired_noop_health_is_noninferiority_not_an_absolute_floor():
    groups = [f"gene-{index // 4:02d}" for index in range(64)]
    low = np.zeros(64)
    same = R5A.paired_noop_noninferiority(
        low,
        low,
        groups,
        n_draws=499,
        namespace="same-low-health",
    )
    assert same["noop_accuracy"] == 0.0
    assert same["pass"] is True
    damaged = R5A.paired_noop_noninferiority(
        np.ones(64),
        np.zeros(64),
        groups,
        n_draws=499,
        namespace="damaged-health",
    )
    assert damaged["pass"] is False
    assert damaged["criterion"] == "lower_bound > -noninferiority_margin"
    assert damaged["bootstrap_is_exact"] is False


def test_health_registry_gives_l2h15_its_own_prospective_gate():
    registry = R5A.health_policy_registry()
    assert registry == {
        "core": {
            "local_query_zero_L3H1": (3, 1),
            "local_query_zero_L0H2": (0, 2),
        },
        "secondary": {"local_query_zero_L2H15": (2, 15)},
        "historical_r3c_health_used": False,
        "secondary_failure_gates_l3_headline": False,
    }


def test_health_probe_positions_are_deterministic_and_exclude_guarded_sites():
    sequence = "ACGT" * 40
    protected = [40, 41, 42, 43, 80, 81, 82, 83]
    first = R5A.health_probe_positions(
        sequence,
        example_id="example-1",
        readout_position=60,
        protected_positions=protected,
    )
    second = R5A.health_probe_positions(
        sequence,
        example_id="example-1",
        readout_position=60,
        protected_positions=protected,
    )
    assert first == second
    guarded = set(range(57, 64))
    for position in protected:
        guarded.update(range(position - 4, position + 5))
    assert len(first) == R5A.HEALTH_PROBES_PER_EXAMPLE
    assert not set(first) & guarded


def test_r1c_valid_scientific_null_is_a_distinct_stop():
    result = R5A.classify_r1c_target_behavior_gate(
        {
            "schema": "gfmcd.r1c.verified_target_handoff/1",
            "verdict": "NO_CONFIRMED_PAIRED_DONOR_PATH",
            "all_gates_pass": True,
        }
    )
    assert result["scientific_gate_pass"] is False
    assert result["failure_is_scientific_not_instrumental"] is True
    assert result["instrument_failure"] is False
    with pytest.raises(RuntimeError, match="instrument was inconclusive"):
        R5A.classify_r1c_target_behavior_gate(
            {
                "schema": "gfmcd.r1c.verified_target_handoff/1",
                "verdict": "INCONCLUSIVE",
                "all_gates_pass": False,
            }
        )


def test_r1c_scientific_null_authenticates_all_six_files_without_building_target(
    monkeypatch, tmp_path
):
    names = (
        "target.json",
        "target.manifest.json",
        "attempt.json",
        "completion.json",
        "preflight.json",
        "preflight.manifest.json",
    )
    paths = []
    for index, name in enumerate(names):
        path = tmp_path / name
        path.write_text(json.dumps({"index": index}))
        paths.append(path)
    pin_names = (
        "FROZEN_R1C_TARGET_ARTIFACT_SHA256",
        "FROZEN_R1C_TARGET_MANIFEST_SHA256",
        "FROZEN_R1C_TARGET_ATTEMPT_SHA256",
        "FROZEN_R1C_TARGET_COMPLETION_SHA256",
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256",
        "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256",
    )
    for pin_name, path in zip(pin_names, paths, strict=True):
        monkeypatch.setattr(
            R5A, pin_name, hashlib.sha256(path.read_bytes()).hexdigest()
        )
    monkeypatch.setattr(
        R5A,
        "_verify_r1c_successor_target_package",
        lambda *_args, **_kwargs: {
            "schema": "gfmcd.r1c.verified_target_handoff/1",
            "verdict": "NO_CONFIRMED_PAIRED_DONOR_PATH",
            "all_gates_pass": True,
        },
    )
    monkeypatch.setattr(
        R5A,
        "load_r1c_capstone_cohort_handoff",
        lambda *_args, **_kwargs: pytest.fail(
            "scientific null must stop before capstone construction"
        ),
    )
    result = R5A.load_authenticated_r1c_target_handoff(*paths)
    assert result["handoff"] is None
    assert result["behavior_gate"]["failure_is_scientific_not_instrumental"]
    assert len(result["package_inputs"]) == 6


def test_r1c_validation_handoff_is_exact_400_and_never_scientific_entry(
    monkeypatch, tmp_path
):
    handoff = {
        "schema": "gfmcd.r5.preflight_cohort_handoff/1",
        "row_count": 400,
        "record_sha256": "1" * 64,
        "cohort_sha256": "2" * 64,
        "ordered_row_ids_sha256": "3" * 64,
        "ordered_gene_ids_sha256": "4" * 64,
        "sorted_unique_gene_ids_sha256": "5" * 64,
        "preflight_construction_sha256": "6" * 64,
        "source_package_sha256": "7" * 64,
        "scientific_entry_authorized": False,
        "sealed_split_opened": False,
    }
    verified = handoff | {
        "verified": True,
        "checkpoint_sha256": "8" * 64,
        "rows": [{"example_id": f"row-{index}"} for index in range(400)],
    }
    monkeypatch.setattr(
        R5A, "load_r1c_preflight_cohort_handoff", lambda *_args: handoff
    )
    monkeypatch.setattr(
        R5A, "validate_r5_preflight_cohort_handoff", lambda value: verified
    )
    artifact = tmp_path / "r1c.json"
    manifest = tmp_path / "r1c.manifest.json"
    artifact.write_text("{}")
    manifest.write_text("{}")
    monkeypatch.setattr(
        R5A,
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256",
        hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        R5A,
        "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    result = R5A.load_authenticated_r1c_preflight_handoff(artifact, manifest)
    assert result["handoff"] is handoff
    assert result["verified"] is verified
    assert result["scientific_entry_authorized"] is False
    assert result["row_count"] == 400
    assert result["exact_order_preserved"] is True


def test_r5a_target_identity_uses_exact_sealed_handoff_order():
    handoff = {
        "row_count": 400,
        "sorted_unique_gene_ids_sha256": "1" * 64,
        "ordered_row_ids_sha256": "2" * 64,
        "capstone_construction_sha256": "3" * 64,
    }
    identity = R5A.build_r5a_target_identity(
        handoff, checkpoint_sha256="4" * 64
    )
    assert identity["stage"] == "R5a"
    assert identity["target_gene_ids_sha256"] == "1" * 64
    assert identity["ordered_rows_sha256"] == "2" * 64
    assert identity["construction_sha256"] == "3" * 64
    assert identity["endpoint_policy"] == "masked_acceptor_G_margin_only"


def test_materializer_preserves_all_400_target_rows_in_handoff_order(monkeypatch):
    rows = [
        type(
            "Row",
            (),
            {
                "example": type(
                    "Example",
                    (),
                    {"example_id": f"row-{index:03d}"},
                )()
            },
        )()
        for index in range(400)
    ]
    handoff = {
        "rows": [{"example_id": f"row-{index:03d}"} for index in range(400)]
    }
    observed = {}
    monkeypatch.setattr(
        R5A, "validate_expected_r5_source_identity", lambda *_args: {"ok": True}
    )

    def fake_examples(*_args, **kwargs):
        observed.update(kwargs)
        return [object()]

    monkeypatch.setattr(R5A, "load_examples_unscored", fake_examples)
    monkeypatch.setattr(
        R5A,
        "materialize_r5_capstone_rows",
        lambda _examples, _handoff: (
            tuple(rows),
            {
                "handoff_order_preserved": True,
                "rows_filtered_or_reordered": False,
                "adjacent_window_required": False,
            },
        ),
    )
    materialized, audit, source = R5A.materialize_authenticated_cohort(
        "unused.gtf", "unused.fa", stage="target", handoff=handoff
    )
    assert materialized == tuple(rows)
    assert audit["rows_filtered_or_reordered"] is False
    assert source == {"ok": True}
    assert observed == {
        "tag": "r1c-target-r1c-single-g-primary-family-v2",
        "allowed_chromosomes": ("chr8", "chr9"),
    }

    reordered = list(rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    monkeypatch.setattr(
        R5A,
        "materialize_r5_capstone_rows",
        lambda _examples, _handoff: (
            tuple(reordered),
            {
                "handoff_order_preserved": True,
                "rows_filtered_or_reordered": False,
                "adjacent_window_required": False,
            },
        ),
    )
    with pytest.raises(RuntimeError, match="exact 400-row R1c handoff"):
        R5A.materialize_authenticated_cohort(
            "unused.gtf", "unused.fa", stage="target", handoff=handoff
        )


def test_sdpa_to_eager_common_leg_replay_is_exact_same_row_and_instrument_only():
    rows = [
        type(
            "Row",
            (),
            {
                "example": type(
                    "Example",
                    (),
                    {"example_id": f"row-{index}"},
                )()
            },
        )()
        for index in range(400)
    ]
    native = {
        "clean": np.linspace(-1.0, 1.0, 400),
        "partner": np.linspace(-0.5, 0.5, 400),
        "pseudo": np.linspace(-0.25, 0.25, 400),
    }
    common = {
        "rows": [
            {
                "example_id": f"row-{index}",
                "margin_levels": {
                    "P_clean": native["clean"][index],
                    "P_under_D": native["partner"][index],
                    "P_under_N": native["pseudo"][index],
                },
            }
            for index in range(400)
        ]
    }
    result = R5A.same_row_common_leg_replay(
        rows, native, common, stage="target"
    )
    assert result["pass"] is True
    assert result["source_execution"] == "R1c eager=False (SDPA)"
    assert result["replay_execution"] == "R5a eager=True (hook-capable)"
    assert result["atol"] == R5A.SDPA_EAGER_COMMON_LEG_ATOL == 1e-4
    assert result["mismatch_disposition"] == "INSTRUMENT_UNRESOLVED"
    common["rows"][0]["margin_levels"]["P_clean"] += 2e-4
    mismatch = R5A.same_row_common_leg_replay(
        rows, native, common, stage="target"
    )
    assert mismatch["pass"] is False
    assert mismatch["mismatch_is_scientific_null"] is False


def test_target_ledger_reservation_precedes_output_cohort_and_model_inspection():
    source = inspect.getsource(R5A.main)
    reserve = source.index("reserve_r5_target_attempt(")
    common_legs = source.index("load_r1c_verified_target_common_legs(")
    output = source.index("_output_paths(args)")
    cohort = source.index("materialize_authenticated_cohort(")
    run = source.index("run_model_stage(")
    complete = source.index("complete_r5_target_attempt(")
    assert reserve < common_legs < output < cohort < run < complete
    run_source = inspect.getsource(R5A.run_model_stage)
    assert "load_examples_unscored(" not in run_source
    assert "materialize_r5_capstone_rows(" not in run_source
    assert run_source.index("target_receipt is None") < run_source.index("load_adapter(")


def test_r3_provenance_authentication_precedes_r1c_without_duplicate_initial_load():
    source = inspect.getsource(R5A.main)
    r3 = source.index("load_exposure_packages(args)")
    r1c = source.index("load_authenticated_r1c_preflight_handoff(")
    combine = source.index("load_upstream_chain(")
    assert r3 < r1c < combine
    assert "load_exposure_packages(" not in inspect.getsource(R5A.load_upstream_chain)


def test_manifests_bind_every_exposure_file_and_target_attempt_receipt(tmp_path):
    keys = (
        "r3d_artifact",
        "r3d_manifest",
        "r3d_registry_artifact",
        "r3d_registry_manifest",
        "r3d_attempt",
        "r3d_completion",
        "r3c_artifact",
        "r3c_manifest",
    )
    exposed = {
        key: {"path": f"/{key}.json", "size_bytes": 1, "sha256": "0" * 64}
        for key in keys
    }
    exposed["exposure_chain_sha256"] = "1" * 64
    assert [
        item["path"]
        for item in R5A._model_free_exposure_manifest_inputs(
            {"authenticated_exposure_chain": exposed}
        )
    ] == [f"/{key}.json" for key in keys]

    r3d_keys = (
        "artifact",
        "manifest",
        "registry_artifact",
        "registry_manifest",
        "attempt",
        "completion",
    )
    r1c_inputs = [
        {"path": f"/r1c-{index}.json", "size_bytes": 1, "sha256": "2" * 64}
        for index in range(6)
    ]
    upstream_record = {
        "upstream_chain": {
            "r3d_scientific_entry": {
                "artifact": {"path": "/r3d-artifact.json"},
                "manifest": {"path": "/r3d-manifest.json"},
                "registry": {
                    "artifact": {"path": "/r3d-registry-artifact.json"},
                    "manifest": {"path": "/r3d-registry-manifest.json"},
                },
                "attempt": {"path": "/r3d-attempt.json"},
                "completion": {"path": "/r3d-completion.json"},
            },
            "r3c_exclusion_provenance": {
                "artifact": {"path": "/r3c-artifact.json"},
                "manifest": {"path": "/r3c-manifest.json"},
            },
            "r1c_package": {"package_inputs": r1c_inputs},
        }
    }
    manifest_inputs = R5A._upstream_manifest_inputs(upstream_record)
    assert len(manifest_inputs) == len(r3d_keys) + 2 + 6
    assert manifest_inputs[-6:] == r1c_inputs

    receipt_path = tmp_path / "attempt.json"
    persisted = {"schema": "gfmcd.r5.target_attempt_receipt/1", "token": "sealed"}
    receipt_path.write_text(json.dumps(persisted))
    identity = R5A._target_receipt_manifest_input(
        {
            "target_attempt_receipt": persisted
            | {"receipt_path": str(receipt_path)}
        }
    )
    assert identity == {
        "path": str(receipt_path.resolve()),
        "size_bytes": receipt_path.stat().st_size,
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }


def test_target_stops_before_nonexistent_user_paths_are_opened():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "model",
            "--stage",
            "target",
            "--gtf",
            "/definitely/not/r5a.gtf",
            "--fasta",
            "/definitely/not/r5a.fa",
            "--model",
            "/definitely/not/r5a-model",
            "--out",
            "/definitely/not/r5a-output.json",
        ],
        cwd=SCRIPT.parents[2],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "R5a target authorization failed" in combined
    assert "requires all literal" in combined
    assert "No such file" not in combined


def test_source_freeze_pins_only_precomputable_r5a_inputs():
    assert R5A.FROZEN_PROTOCOL_SHA256 == (
        "2981e40a65872914b5551d7b463d85b5203ef7630d66a1e9a0b24ae82c348ec6"
    )
    assert R5A.FROZEN_INSTRUMENT_FINGERPRINT_SHA256 == (
        "59077228c551f38cda5e136a94e5522f79c4c3984947c156eeae78a0ab61604b"
    )
    assert R5A.FROZEN_CHECKPOINT_AGGREGATE_SHA256 == (
        "fb2a04e0c9b5247b44f03cd23581632d5924b86fcafaffa4c6f2b8d01ae1da8d"
    )
    assert "> **Status: FROZEN FOR PREFLIGHT ONLY" in R5A.PROTOCOL.read_text()
    assert R5A.FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256 is None
    assert R5A.FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256 is None
    assert R5A.FROZEN_PREFLIGHT_ARTIFACT_SHA256 is None
    assert R5A.FROZEN_PREFLIGHT_MANIFEST_SHA256 is None


def test_checkpoint_aggregate_uses_ordered_relative_bundle(tmp_path):
    model = tmp_path / "model"
    nested = model / "nested"
    nested.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (nested / "weights.bin").write_bytes(b"weights")
    identity = R5A._checkpoint_identity(str(model))
    assert [row["relative_path"] for row in identity["ordered_bundle"]] == [
        "config.json",
        "nested/weights.bin",
    ]
    assert identity["aggregate_sha256"] == R5A.canonical_sha256(
        identity["ordered_bundle"]
    )


def test_frozen_r5a_runner_identity_is_invariant_only_to_literal_pin_values(tmp_path):
    source = SCRIPT.read_text()
    modified_pin = source.replace(
        (
            'FROZEN_PROTOCOL_SHA256: str | None = '
            '"2981e40a65872914b5551d7b463d85b5203ef7630d66a1e9a0b24ae82c348ec6"'
        ),
        'FROZEN_PROTOCOL_SHA256: str | None = "' + "0" * 64 + '"',
    )
    assert modified_pin != source
    modified_path = tmp_path / SCRIPT.name
    modified_path.write_text(modified_pin)
    assert R5A._normalized_runner_sha256(SCRIPT) == R5A._normalized_runner_sha256(
        modified_path
    )

    changed_code = source.replace("MARGIN_SESOI = 0.05", "MARGIN_SESOI = 0.051")
    assert changed_code != source
    modified_path.write_text(changed_code)
    assert R5A._normalized_runner_sha256(SCRIPT) != R5A._normalized_runner_sha256(
        modified_path
    )


def test_protocol_freezes_units_sequence_upstream_roles_and_nonlicenses():
    text = R5A.PROTOCOL.read_text()
    required = (
        "R3d",
        "Path A",
        "R1c",
        "chr8",
        "chr9",
        "paired",
        "pseudo",
        "L0H2",
        "L2H15",
        "fixed-V",
        "10%-trimmed",
        "not exact",
        "target-minus-control",
        "masked-acceptor",
        "masked-donor",
        "paired no-op",
        "scientific stop",
        "instrument",
    )
    for phrase in required:
        assert phrase.lower() in text.lower()
    assert "0.05/4" not in text
    assert "L1H12" not in text
    assert "residual validation" not in text.lower()
    assert "masked residual" not in text.lower()
    assert "byte-identical clean q" in text.lower()
    assert "shifted k" in text.lower()
    assert "shifted cell would natively produce" in text.lower()
    assert text.index("Authenticate the exact R3d target package") < text.index(
        "Authenticate the exact two-file R1c"
    )


def test_runner_binds_exact_upstream_artifact_and_manifest_pins():
    assert (
        R5A.FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256
        == R5A.SHARED_R1C_PREFLIGHT_ARTIFACT_SHA256
    )
    assert (
        R5A.FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256
        == R5A.SHARED_R1C_PREFLIGHT_MANIFEST_SHA256
    )
    assert (
        R5A.FROZEN_R1C_TARGET_ARTIFACT_SHA256
        == R5A.SHARED_R1C_TARGET_ARTIFACT_SHA256
    )
    assert (
        R5A.FROZEN_R1C_TARGET_MANIFEST_SHA256
        == R5A.SHARED_R1C_TARGET_MANIFEST_SHA256
    )
    assert (
        R5A.FROZEN_R1C_TARGET_ATTEMPT_SHA256
        == R5A.SHARED_R1C_TARGET_ATTEMPT_SHA256
    )
    assert (
        R5A.FROZEN_R1C_TARGET_COMPLETION_SHA256
        == R5A.SHARED_R1C_TARGET_COMPLETION_SHA256
    )
    assert R5A.FROZEN_R3D_ARTIFACT_SHA256 == R5A.R3D_TARGET_ARTIFACT_SHA256
    assert R5A.FROZEN_R3D_MANIFEST_SHA256 == R5A.R3D_TARGET_MANIFEST_SHA256
    assert (
        R5A.FROZEN_R3D_REGISTRY_ARTIFACT_SHA256
        == R5A.R3D_REGISTRY_ARTIFACT_SHA256
    )
    assert (
        R5A.FROZEN_R3D_REGISTRY_MANIFEST_SHA256
        == R5A.R3D_REGISTRY_MANIFEST_SHA256
    )
    assert R5A.FROZEN_R3D_ATTEMPT_SHA256 == R5A.R3D_TARGET_ATTEMPT_SHA256
    assert R5A.FROZEN_R3D_COMPLETION_SHA256 == R5A.R3D_TARGET_COMPLETION_SHA256
    assert R5A.FROZEN_R3C_ARTIFACT_SHA256 == R5A.R3C_PREFLIGHT_ARTIFACT_SHA256
    assert R5A.FROZEN_R3C_MANIFEST_SHA256 == R5A.R3C_PREFLIGHT_MANIFEST_SHA256
    assert R5A.R5_ENTRY_SCHEMA == "gfmcd.r5.entry_contract/2"
