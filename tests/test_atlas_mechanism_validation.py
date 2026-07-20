from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/splicebert_head_context_atlas/run_atlas_mechanism_validation.py"
SPEC = importlib.util.spec_from_file_location("atlas_mechanism_validation", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)

ANALYSIS_SCRIPT = REPO / "scripts/splicebert_head_context_atlas/analyze_atlas_mechanism_validation.py"
ANALYSIS_SPEC = importlib.util.spec_from_file_location("atlas_mechanism_analysis", ANALYSIS_SCRIPT)
analysis = importlib.util.module_from_spec(ANALYSIS_SPEC)
assert ANALYSIS_SPEC.loader is not None
ANALYSIS_SPEC.loader.exec_module(analysis)

VERIFY_SCRIPT = REPO / "scripts/splicebert_head_context_atlas/validate_atlas_mechanism_analysis.py"
VERIFY_SPEC = importlib.util.spec_from_file_location("atlas_mechanism_analysis_verify", VERIFY_SCRIPT)
verify = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
VERIFY_SPEC.loader.exec_module(verify)

STAGE2_SCRIPT = REPO / "scripts/splicebert_head_context_atlas/build_atlas_stage2_package.py"
STAGE2_SPEC = importlib.util.spec_from_file_location("atlas_stage2_package", STAGE2_SCRIPT)
stage2 = importlib.util.module_from_spec(STAGE2_SPEC)
assert STAGE2_SPEC.loader is not None
STAGE2_SPEC.loader.exec_module(stage2)

STAGE2_RUN_SCRIPT = REPO / "scripts/splicebert_head_context_atlas/run_atlas_stage2_mechanisms.py"
STAGE2_RUN_SPEC = importlib.util.spec_from_file_location("atlas_stage2_runner", STAGE2_RUN_SCRIPT)
stage2_runner = importlib.util.module_from_spec(STAGE2_RUN_SPEC)
assert STAGE2_RUN_SPEC.loader is not None
STAGE2_RUN_SPEC.loader.exec_module(stage2_runner)

STAGE2_ANALYSIS_SCRIPT = REPO / "scripts/splicebert_head_context_atlas/analyze_atlas_stage2_mechanisms.py"
STAGE2_ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "atlas_stage2_analysis", STAGE2_ANALYSIS_SCRIPT
)
stage2_analysis = importlib.util.module_from_spec(STAGE2_ANALYSIS_SPEC)
assert STAGE2_ANALYSIS_SPEC.loader is not None
STAGE2_ANALYSIS_SPEC.loader.exec_module(stage2_analysis)

STAGE2_VERIFY_SCRIPT = REPO / "scripts/splicebert_head_context_atlas/validate_atlas_stage2_mechanisms.py"
STAGE2_VERIFY_SPEC = importlib.util.spec_from_file_location(
    "atlas_stage2_execution_verify", STAGE2_VERIFY_SCRIPT
)
stage2_verify = importlib.util.module_from_spec(STAGE2_VERIFY_SPEC)
assert STAGE2_VERIFY_SPEC.loader is not None
STAGE2_VERIFY_SPEC.loader.exec_module(stage2_verify)

SCREEN_DAG_SCRIPT = REPO / "scripts/splicebert/export_segment_attribution_bundle.py"
SCREEN_DAG_SPEC = importlib.util.spec_from_file_location("segment_attribution_bundle", SCREEN_DAG_SCRIPT)
screen_dag = importlib.util.module_from_spec(SCREEN_DAG_SPEC)
assert SCREEN_DAG_SPEC.loader is not None
SCREEN_DAG_SPEC.loader.exec_module(screen_dag)


def _package() -> dict:
    return {
        "candidates": [
            {
                "candidate_id": "general|A|L0H0",
                "kind": "general",
                "estimand": "A",
                "component_id": "L0H0",
                "importance_control": {"component_id": "L0H1"},
            },
            {
                "candidate_id": "context|single_pair|donor|B|L3MLP",
                "kind": "context",
                "contrast": "single_pair|donor",
                "estimand": "B",
                "component_id": "L3MLP",
                "importance_control": {"component_id": "L2MLP"},
            },
        ],
        "exact_groups": {
            "general|A": ["L0H0"],
            "context|single_pair|donor|B": ["L3H1", "L3MLP"],
        },
        "positive_control": {"component_id": "L3H1", "estimand": "B"},
    }


def test_component_parser_and_canonical_mixed_group() -> None:
    assert runner.parse_component("L3H1") == ("attention_head", (3, 1))
    assert runner.parse_component("L5MLP") == ("mlp_write", (5, "mlp"))
    assert runner.canonical_components(["L5MLP", "L0H2", "L0H2"]) == (
        "L0H2", "L5MLP"
    )
    with pytest.raises(ValueError):
        runner.parse_component("L6H0")


def test_full_endpoints_include_candidates_controls_exact_groups_and_positive_control() -> None:
    endpoints = runner.build_endpoints(_package(), smoke=False)
    roles = {endpoint["endpoint_role"] for endpoint in endpoints}
    assert roles == {"candidate", "importance_control", "exact_group", "positive_control"}
    assert len(endpoints) == 7
    mixed = next(endpoint for endpoint in endpoints if endpoint["group_id"] == "context|single_pair|donor|B")
    assert mixed["component_ids"] == ["L3H1", "L3MLP"]
    assert mixed["scope_kind"] == "contrast"
    assert mixed["scope_value"] == "single_pair|donor"


def test_duplicate_physical_computation_is_reused_across_endpoint_semantics() -> None:
    endpoints = runner.build_endpoints(_package(), smoke=False)
    computations = runner.computations_from_endpoints(endpoints)
    l3h1_endpoints = [
        endpoint for endpoint in endpoints
        if endpoint["estimand"] == "B" and endpoint["component_ids"] == ["L3H1"]
    ]
    assert len(l3h1_endpoints) == 1  # positive control only in this fixture
    assert len(computations) < len(endpoints)  # general singleton group reuses L0H0/A


def test_context_scope_uses_only_its_matched_pair_union() -> None:
    endpoints = runner.build_endpoints(_package(), smoke=False)
    packets = [
        {"packet_id": "case", "target_role": "donor"},
        {"packet_id": "control", "target_role": "donor"},
        {"packet_id": "other", "target_role": "acceptor"},
    ]
    pairs = [{
        "category": "single_pair", "target_role": "donor",
        "case_packet_id": "case", "control_packet_id": "control",
    }]
    scopes = runner.endpoint_packet_sets(endpoints, packets, pairs)
    context = next(
        endpoint for endpoint in endpoints
        if endpoint["candidate_id"] == "context|single_pair|donor|B|L3MLP"
        and endpoint["endpoint_role"] == "candidate"
    )
    assert scopes[context["endpoint_id"]] == {"case", "control"}
    general = next(
        endpoint for endpoint in endpoints
        if endpoint["candidate_id"] == "general|A|L0H0"
        and endpoint["endpoint_role"] == "candidate"
    )
    assert scopes[general["endpoint_id"]] == {"case", "control", "other"}


def test_sealed_reference_resolves_portably_from_repo_root() -> None:
    relative = "results/splicebert_head_context_atlas/atlas_context_matched_v7.validation.packets.jsonl"
    assert runner.resolve_repo_reference(relative) == (REPO / relative).resolve()
    assert runner.split_path_from_validation(Path(relative), "discovery") == Path(
        relative.replace(".validation.", ".discovery.")
    )


def test_holm_step_down_is_reconstructed_independently() -> None:
    pvalues = [0.01, 0.04, 0.03]
    expected = [0.03, 0.06, 0.06]
    assert analysis.holm_adjust(pvalues).tolist() == pytest.approx(expected)
    assert verify.holm(pvalues) == pytest.approx(expected)


def test_simultaneous_classification_requires_interval_to_clear_material_floor() -> None:
    label, same_direction = analysis.classify(
        point=0.07, low=0.01, high=0.13, p_holm=0.01, expected_direction=1
    )
    assert label == "detected_subfloor"
    assert same_direction is True
    label, _ = analysis.classify(
        point=0.08, low=0.051, high=0.11, p_holm=0.01, expected_direction=1
    )
    assert label == "material_support"


def test_zero_shortlist_still_has_only_the_independent_positive_control() -> None:
    package = {
        "candidates": [], "exact_groups": {},
        "positive_control": {"component_id": "L3H1", "estimand": "B"},
    }
    endpoints = runner.build_endpoints(package, smoke=False)
    assert [endpoint["endpoint_id"] for endpoint in endpoints] == [
        "positive_control|B|L3H1"
    ]
    packets = [
        {"packet_id": "d", "target_role": "donor"},
        {"packet_id": "a", "target_role": "acceptor"},
    ]
    scopes = runner.endpoint_packet_sets(endpoints, packets, [])
    assert scopes["positive_control|B|L3H1"] == {"d"}


def _stage2_candidate(component: str, estimand: str, *, candidate_id: str | None = None) -> dict:
    return {
        "candidate_id": candidate_id or f"general|{estimand}|{component}",
        "kind": "general",
        "component_id": component,
        "estimand": estimand,
        "importance_control": {
            "component_id": "L3H2" if not component.endswith("MLP") else "L2MLP"
        },
        "representative": {"donor": {}, "acceptor": {}},
    }


def test_stage2_b_head_gets_population_qk_v_and_all_downstream_mlps() -> None:
    candidate = _stage2_candidate("L3H1", "B")
    spec = stage2.stage2_spec(
        candidate, validated_candidates=[candidate],
        exact_groups={"general|B": ["L3H1"]},
    )
    assert spec["mechanism"]["population_qk_v"]["conditions"] == [
        "QK", "V_OV", "QKV_BRIDGE"
    ]
    assert spec["mechanism"]["downstream_mlp_recovery"]["layers"] == [3, 4, 5]
    assert "QKV_BRIDGE" not in spec["within_endpoint_family_members"]


def test_stage2_a_head_is_representative_only_for_qk_v() -> None:
    candidate = _stage2_candidate("L1H4", "A")
    spec = stage2.stage2_spec(
        candidate, validated_candidates=[candidate],
        exact_groups={"general|A": ["L1H4"]},
    )
    assert spec["mechanism"]["population_qk_v"]["enabled"] is False
    assert spec["mechanism"]["representative_state_interchange"]["inference"] == (
        "descriptive_only_no_population_label"
    )


def test_stage2_mlp_uses_only_validated_same_group_upstream_heads() -> None:
    head_early = _stage2_candidate("L1H2", "B")
    head_late = _stage2_candidate("L5H3", "B")
    mlp = _stage2_candidate("L3MLP", "B")
    validated = [head_early, head_late, mlp]
    spec = stage2.stage2_spec(
        mlp, validated_candidates=validated,
        exact_groups={"general|B": ["L1H2", "L5H3", "L3MLP"]},
    )
    assert spec["mechanism"]["upstream_attention_path"]["attention_group"] == ["L1H2"]
    assert spec["mechanism"]["upstream_attention_path"]["enabled"] is True


def test_stage2_endpoint_identity_includes_candidate_not_only_role() -> None:
    candidates = stage2_runner.build_smoke_candidates()
    endpoints = stage2_runner.build_endpoints(candidates)
    assert len({row["endpoint_key"] for row in endpoints}) == 6
    donor = [row for row in endpoints if row["target_role"] == "donor"]
    assert len({row["endpoint_key"] for row in donor}) == 3


def test_stage2_reuses_whole_validation_batch_order_namespace() -> None:
    assert stage2_runner.WHOLE_REFERENCE_BATCH_ORDER_TAG == "mechanism-validation-order"


def test_stage2_family_uses_frozen_order_and_joint_gene_covariance() -> None:
    candidate = {
        "candidate_id": "c", "within_endpoint_family_members": [
            "exact_group_effect", "candidate_QK", "candidate_V_OV"
        ]
    }
    rows = [
        {"candidate_id": "c", "member": "candidate_V_OV", "member_role": "family"},
        {"candidate_id": "c", "member": "candidate_QK", "member_role": "family"},
    ]
    assert stage2_analysis.family_members(candidate, rows) == [
        "exact_group_effect", "candidate_QK", "candidate_V_OV"
    ]
    sources = {
        "x": [
            {"packet_id": "p1", "target_role": "donor", "gene_id": "g1", "value": 1.0},
            {"packet_id": "p2", "target_role": "donor", "gene_id": "g2", "value": 3.0},
        ],
        "y": [
            {"packet_id": "p1", "target_role": "donor", "gene_id": "g1", "value": 2.0},
            {"packet_id": "p2", "target_role": "donor", "gene_id": "g2", "value": 6.0},
        ],
    }
    point, covariance, meta = stage2_analysis.gene_covariance(sources, "donor")
    assert point.tolist() == pytest.approx([2.0, 4.0])
    assert covariance[0, 1] == pytest.approx(2.0 * covariance[0, 0])
    assert meta["n_genes"] == 2


def test_stage2_route_label_requires_directional_qk_and_bounded_value() -> None:
    candidate = {
        "component_type": "attention_head", "estimand": "B",
        "validated_whole_endpoints": [{
            "target_role": "donor", "contrast": None,
            "validated_whole_endpoint": True, "mean": 0.12,
        }],
    }
    endpoint = {"target_role": "donor", "contrast": None}
    members = [
        {"member": "candidate_QK", "classification": "material_support"},
        {"member": "candidate_V_OV", "classification": "bounded_small"},
        {"member": "MLP_recovery_difference_L3", "classification": "inconclusive"},
    ]
    labels = stage2_analysis.conditional_labels(candidate, endpoint, members, smoke=False)
    assert labels["route_value_label"] == "partner_sensitive_routing"
    assert labels["path_labels"] == []
    assert stage2_analysis.conditional_labels(candidate, endpoint, members, smoke=True) == {
        "route_value_label": "SMOKE_NO_LABEL", "path_labels": [],
        "validated_whole_direction": None,
    }


def _a3_runtime_manifest(value: float = 2.98e-5) -> dict:
    failed = {
        "gate": "target_whole_stage_reference", "value": value,
        "threshold": 1e-6, "passed": False,
    }
    return {
        "status": "FAIL", "n_failures": 1, "failures": [failed],
        "gates": [
            {"gate": "target_packet_grid", "value": 4019,
             "threshold": 4019, "passed": True},
            failed,
        ],
    }


def test_a3_admits_only_the_preserved_numerical_reference_failure() -> None:
    failures: list[str] = []
    gates, atol = stage2_verify.audit_runtime_gate_disposition(
        _a3_runtime_manifest(), "target", a3=True, failures=failures
    )
    assert failures == []
    assert atol == 1e-4
    assert gates["target_whole_stage_reference"]["passed"] is False


def test_a3_refuses_a_second_failure_or_error_above_the_frozen_identity_bound() -> None:
    second = _a3_runtime_manifest()
    second["gates"][0]["passed"] = False
    failures: list[str] = []
    stage2_verify.audit_runtime_gate_disposition(
        second, "target", a3=True, failures=failures
    )
    assert "target: A3 cannot waive any second runtime gate" in failures

    too_large = _a3_runtime_manifest(1.01e-4)
    failures = []
    stage2_verify.audit_runtime_gate_disposition(
        too_large, "target", a3=True, failures=failures
    )
    assert "target: A3 reference failure is outside the licensed numerical band" in failures


def test_a3_reference_report_exposes_distribution_and_both_thresholds() -> None:
    report = stage2_verify.reference_error_summary(
        [0.0, 1e-6, 2e-6, 2.98e-5], 1e-4
    )
    assert report["n_comparisons"] == 4
    assert report["n_above_original_atol"] == 2
    assert report["n_above_effective_atol"] == 0
    assert report["max_abs_error"] == pytest.approx(2.98e-5)


def test_stage2_auditor_resolves_frozen_producer_paths_by_repo_suffix() -> None:
    relative = Path(
        "results/splicebert_head_context_atlas/"
        "atlas_context_matched_v7.validation.packets.jsonl"
    )
    producer = Path("/Users/example/checkout") / relative
    assert stage2_verify.resolve_portable_manifest_input(producer) == (REPO / relative).resolve()


def test_version_stable_attention_projection_split() -> None:
    tensor = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
    observed = screen_dag.split_projection(tensor, 2)
    assert observed.shape == (2, 2, 3, 4)
    assert torch.equal(observed[1, 1, 2], tensor[1, 2, 4:8])
