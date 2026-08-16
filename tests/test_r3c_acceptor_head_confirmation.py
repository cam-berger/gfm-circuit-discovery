from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert_acceptor_coupling/run_r3c_acceptor_head_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_r3c_acceptor_head_confirmation", SCRIPT
)
R3C = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = R3C
SPEC.loader.exec_module(R3C)


def _measurements(n: int = 8, *, fill: float = 0.0):
    output = {mode: {} for mode in R3C.MODES}
    for mode in R3C.MODES:
        for component in R3C.COMPONENTS:
            head = R3C.component_name(component)
            partner = np.full(n, fill, dtype=float)
            pseudo = np.zeros(n, dtype=float)
            output[mode][head] = {
                **R3C.endpoint_vectors(partner, pseudo),
                "health": {"ok": True, "total": 10},
            }
    return output


def _all_health(ok: bool = True):
    return {
        name: {"ok": ok, "participants": [], "failed_participants": []}
        for name in R3C.PATH_NAMES
    }


def test_exact_head_registry_has_no_mlp_and_preserves_order():
    assert [R3C.component_name(component) for component in R3C.COMPONENTS] == [
        "L3H1",
        "L2H15",
        "L1H12",
        "L4H11",
        "L4H9",
        "L4H7",
        "L4H0",
        "L4H2",
        "L5H11",
        "L0H2",
    ]
    assert all(kind == "head" and head is not None for kind, _, head in R3C.COMPONENTS)
    assert all("MLP" not in name for name in R3C.family_row_names())


def test_exact_40_primitive_and_62_derived_row_family():
    measurements = _measurements(n=5)
    primitive, primitive_names = R3C.build_primitive_matrix(measurements)
    family, family_names = R3C.derive_family_from_primitives(primitive)

    assert primitive.shape == (40, 5)
    assert len(primitive_names) == len(set(primitive_names)) == 40
    assert family.shape == (62, 5)
    assert family_names == R3C.family_row_names()
    assert len(family_names) == len(set(family_names)) == 62
    assert family_names[-2:] == [
        R3C.delivery_gap_row_name("partner"),
        R3C.delivery_gap_row_name("specific"),
    ]
    assert all("delivery_gap" not in name for name in family_names[:60])


def test_endpoint_formulas_are_native_minus_intervened_or_patched_and_specific_is_derived():
    native_partner = np.asarray([1.0, 2.0])
    native_pseudo = np.asarray([0.2, 0.4])
    mean = R3C.mean_component_endpoint_drops(
        native_partner,
        native_pseudo,
        np.asarray([0.4, 0.5]),
        np.asarray([0.1, 0.2]),
    )
    np.testing.assert_allclose(mean["partner"], [0.6, 1.5])
    np.testing.assert_allclose(mean["pseudo"], [0.1, 0.2])
    np.testing.assert_allclose(mean["specific"], mean["partner"] - mean["pseudo"])

    noising = R3C.noising_component_endpoint_drops(
        np.asarray([3.0, 5.0]),
        np.asarray([2.5, 4.0]),
        np.asarray([2.9, 4.8]),
    )
    np.testing.assert_allclose(noising["partner"], [0.5, 1.0])
    np.testing.assert_allclose(noising["pseudo"], [0.1, 0.2])
    np.testing.assert_allclose(
        noising["specific"], noising["partner"] - noising["pseudo"]
    )


def test_specific_and_delivery_gaps_are_derived_after_primitive_edits():
    measurements = _measurements(n=3)
    for mode in R3C.MODES:
        for component in R3C.COMPONENTS:
            head = R3C.component_name(component)
            measurements[mode][head].update(
                R3C.endpoint_vectors(np.zeros(3), np.zeros(3))
            )
    measurements[R3C.NOISING_MODE]["L4H11"].update(
        R3C.endpoint_vectors(np.asarray([1.0, 2.0, 3.0]), np.asarray([0.1, 0.2, 0.3]))
    )
    measurements[R3C.NOISING_MODE]["L4H9"].update(
        R3C.endpoint_vectors(np.asarray([3.0, 4.0, 5.0]), np.asarray([0.3, 0.4, 0.5]))
    )
    measurements[R3C.NOISING_MODE]["L4H0"].update(
        R3C.endpoint_vectors(np.asarray([0.2, 0.4, 0.6]), np.asarray([0.0, 0.1, 0.2]))
    )
    measurements[R3C.NOISING_MODE]["L4H2"].update(
        R3C.endpoint_vectors(np.asarray([0.4, 0.6, 0.8]), np.asarray([0.1, 0.1, 0.1]))
    )
    measurements[R3C.NOISING_MODE]["L5H11"].update(
        R3C.endpoint_vectors(np.asarray([0.6, 0.8, 1.0]), np.asarray([0.2, 0.2, 0.2]))
    )
    primitive, _ = R3C.build_primitive_matrix(measurements)
    family, names = R3C.derive_family_from_primitives(primitive)
    by_name = {name: family[index] for index, name in enumerate(names)}

    for head in ["L4H11", "L4H9", "L4H0", "L4H2", "L5H11"]:
        np.testing.assert_allclose(
            by_name[R3C.family_row_name(R3C.NOISING_MODE, head, "specific")],
            by_name[R3C.family_row_name(R3C.NOISING_MODE, head, "partner")]
            - by_name[R3C.family_row_name(R3C.NOISING_MODE, head, "pseudo")],
        )
    expected_partner_gap = np.mean(
        [
            by_name[R3C.family_row_name(R3C.NOISING_MODE, head, "partner")]
            for head in R3C.DELIVERY_CANDIDATES
        ],
        axis=0,
    ) - np.mean(
        [
            by_name[R3C.family_row_name(R3C.NOISING_MODE, head, "partner")]
            for head in R3C.DONOR_ROUTE_COMPARATORS
        ],
        axis=0,
    )
    np.testing.assert_allclose(
        by_name[R3C.delivery_gap_row_name("partner")], expected_partner_gap
    )


def test_exact_seven_path_holm_family_and_route_controls():
    specs = R3C.named_path_specs()
    assert tuple(specs) == R3C.PATH_NAMES == (
        "L3H1_reused_component",
        "L2H15_mean_component",
        "L1H12_mean_component",
        "L4H11_noising_component",
        "L4H9_noising_component",
        "L4H7_noising_component",
        "distinct_acceptor_route_candidate",
    )
    assert len(specs["L3H1_reused_component"]["requirements"]) == 6
    route = specs["distinct_acceptor_route_candidate"]["requirements"]
    assert len(route) == 17
    for head in R3C.DONOR_ROUTE_COMPARATORS:
        for endpoint in R3C.ENDPOINTS:
            assert route[R3C.family_row_name(R3C.NOISING_MODE, head, endpoint)] == (
                "EQUIVALENT_WITHIN_SESOI"
            )
    assert not any("L0H2" in row for row in route)


def test_coherent_point_four_plant_and_28_isolated_boundary_faces_preserve_algebra():
    plant = R3C.coherent_path_offset(R3C.PLANT_DELTA)
    family, names = R3C.derive_family_from_primitives(plant[:, None])
    by_name = {name: float(family[index, 0]) for index, name in enumerate(names)}
    assert R3C.PLANT_DELTA == 0.40
    for spec in R3C.named_path_specs().values():
        for row, expected in spec["requirements"].items():
            assert by_name[row] == pytest.approx(
                0.40 if expected == "MATERIAL_POSITIVE" else 0.0
            )

    surfaces = R3C.boundary_face_surface_specs()
    assert len(surfaces) == 28
    covered: dict[str, list[tuple[str, str]]] = {
        path: [] for path in R3C.PATH_NAMES
    }
    for case, surface in surfaces.items():
        offset = np.asarray(surface["offset"], dtype=float)
        family, names = R3C.derive_family_from_primitives(offset[:, None])
        by_name = {name: float(family[index, 0]) for index, name in enumerate(names)}
        for path in surface["assigned_paths"]:
            targets = set(surface["targeted_boundaries"]) & set(
                R3C.named_path_specs()[path]["requirements"]
            )
            assert len(targets) == 1, (case, path, targets)
            target = next(iter(targets))
            covered[path].append((target, surface["targeted_boundaries"][target]))
            for row, expected in R3C.named_path_specs()[path]["requirements"].items():
                if row == target:
                    assert abs(by_name[row]) == pytest.approx(R3C.SESOI)
                elif expected == "MATERIAL_POSITIVE":
                    assert by_name[row] > R3C.SESOI
                else:
                    assert -R3C.SESOI < by_name[row] < R3C.SESOI
    assert {path: len(faces) for path, faces in covered.items()} == {
        "L3H1_reused_component": 8,
        "L2H15_mean_component": 4,
        "L1H12_mean_component": 4,
        "L4H11_noising_component": 4,
        "L4H9_noising_component": 4,
        "L4H7_noising_component": 4,
        "distinct_acceptor_route_candidate": 28,
    }
    assert R3C.boundary_face_suite_binding()["surface_count"] == 28
    assert (28 + 1) * R3C.PRODUCTION_TRIALS == 5800


def test_center_primitives_is_translation_invariant_and_shared_gene_flip_precedes_derivation():
    rng = np.random.default_rng(3)
    primitive = rng.normal(size=(40, 8))
    translations = rng.normal(size=(40, 1))
    centered = R3C.center_primitive_rows(primitive)
    shifted = R3C.center_primitive_rows(primitive + translations)
    np.testing.assert_allclose(centered, shifted, atol=1e-14)

    groups = ["g0", "g0", "g1", "g2", "g2", "g3", "g4", "g4"]
    flipped = R3C.gene_sign_flip(
        centered, groups, seed=8, namespace="r3c-test-shared-outer"
    )
    for column in range(centered.shape[1]):
        nonzero = np.abs(centered[:, column]) > 1e-12
        ratios = np.unique(flipped[nonzero, column] / centered[nonzero, column])
        assert ratios.size == 1
        if column == 1:
            first_ratio = ratios[0]
            previous = np.unique(flipped[:, 0] / centered[:, 0])
            assert first_ratio == previous[0]
    family, names = R3C.derive_family_from_primitives(flipped)
    by_name = {name: family[index] for index, name in enumerate(names)}
    np.testing.assert_allclose(
        by_name[R3C.family_row_name(R3C.MEAN_MODE, "L3H1", "specific")],
        by_name[R3C.family_row_name(R3C.MEAN_MODE, "L3H1", "partner")]
        - by_name[R3C.family_row_name(R3C.MEAN_MODE, "L3H1", "pseudo")],
    )


def test_boundary_face_preflight_serializes_each_exact_classifier_and_per_surface_cp():
    n = 40
    groups = [f"gene-{index:02d}" for index in range(n)]
    primitive = np.zeros((40, n), dtype=float)
    result = R3C.r3c_path_preflight(
        primitive,
        groups,
        path_health=_all_health(),
        n_calibration=49,
        n_trials=1,
        seed=0,
        namespace="r3c-test-boundary-face-preflight",
        min_genes=30,
    )
    assert result["production_classifier_calls"] == 29
    assert result["health_eligible_paths"] == list(R3C.PATH_NAMES)
    assert set(result["planted_power_by_path"]) == set(R3C.PATH_NAMES)
    assert all(
        item["health_eligible"] is True
        for item in result["planted_power_by_path"].values()
    )
    assert result["boundary_face_case_order"] == list(
        R3C.boundary_face_surface_specs()
    )
    assert "not an assertion" in result["boundary_face_role"]
    for case_name, case in result["boundary_face_cases"].items():
        assert len(case["classifier_records_by_trial"]) == 1
        assert case["classifier_records_sha256"] == R3C.canonical_sha256(
            case["classifier_records_by_trial"]
        )
        _, upper = R3C.one_sided_binomial_bounds(
            case["false_positive_trials"], 1, alpha=R3C.FAMILY_ALPHA
        )
        assert case["one_sided_clopper_pearson_upper"] == pytest.approx(upper)
        audit = case["classifier_records_by_trial"][0][
            "boundary_randomisation_audit"
        ]
        assert audit["B"] == 49
        assert audit["namespace"].startswith(
            f"r3c-test-boundary-face-preflight:boundary-face:{case_name}:classifier:0"
        )
    assert len(result["plant_classifier_records_by_trial"]) == 1


def test_strict_classifier_uses_one_sign_matrix_for_all_edges_and_holm_exactly_seven():
    n = 40
    groups = [f"gene-{index:02d}" for index in range(n)]
    primitive = np.repeat(R3C.coherent_path_offset(0.40)[:, None], n, axis=1)
    family, names = R3C.derive_family_from_primitives(primitive)
    result = R3C.strict_path_family_inference(
        family,
        groups,
        names,
        n_draws=399,
        seed=0,
        namespace="r3c-test-supported",
        min_genes=30,
        path_health=_all_health(),
    )
    assert tuple(result["paths"]) == R3C.PATH_NAMES
    assert all(
        result["paths"][name]["classification"] == "SUPPORTED"
        for name in R3C.PATH_NAMES
    )
    assert result["path_family"]["path_names"] == list(R3C.PATH_NAMES)
    assert all(
        record["holm_family_size"] == 7 for record in result["paths"].values()
    )
    audit = result["boundary_randomisation_audit"]
    assert audit["all_decision_edges_share_one_sign_matrix"] is True
    assert audit["B"] == 399
    sign_hashes = set()
    gene_hashes = set()
    for row in result["row_boundary_tests"].values():
        records = (
            [row["boundary_test"]]
            if "boundary_test" in row
            else [row["lower_edge"], row["upper_edge"]]
        )
        for edge in records:
            sign_hashes.add(edge["sign_matrix_sha256"])
            gene_hashes.add(edge["ordered_gene_labels_sha256"])
            assert edge["B"] == 399
    assert sign_hashes == {audit["sign_matrix_sha256"]}
    assert gene_hashes == {audit["ordered_gene_labels_sha256"]}
    assert audit["sign_matrix_sha256"] == R3C._expected_sign_matrix_sha256(
        namespace=audit["namespace"], seed=0, B=399, n_genes=n
    )


def test_observed_and_capability_inference_replay_exactly_from_gene_sufficient_statistics():
    n = 40
    groups = [f"gene-{index:02d}" for index in range(n)]
    primitive = np.repeat(R3C.coherent_path_offset(0.40)[:, None], n, axis=1)
    family, names = R3C.derive_family_from_primitives(primitive)
    stats = R3C.gene_sufficient_statistics(family, groups, names)
    canonical, canonical_groups = R3C._reconstruct_cluster_array(stats)
    first = R3C.strict_path_family_inference(
        canonical,
        canonical_groups,
        names,
        n_draws=199,
        seed=0,
        namespace="r3c-test-replay",
        min_genes=30,
        path_health=_all_health(),
    )
    replay, replay_groups = R3C._reconstruct_cluster_array(stats)
    second = R3C.strict_path_family_inference(
        replay,
        replay_groups,
        names,
        n_draws=199,
        seed=0,
        namespace="r3c-test-replay",
        min_genes=30,
        path_health=_all_health(),
    )
    assert first == second

    capability_stats = R3C.gene_sufficient_statistics(
        np.vstack((np.full(n, 0.40), np.zeros(n))),
        groups,
        ["all_L3_partner_recovery", "all_L3_pseudo_recovery"],
    )
    capability, capability_groups = R3C._reconstruct_cluster_array(capability_stats)
    result = R3C.hook_capability_inference(
        capability[0],
        capability[1],
        capability_groups,
        n_draws=199,
        seed=0,
        namespace="r3c-test-live-hook",
        min_genes=30,
    )
    assert result["classification"] == "LIVE_HOOK_CAPABILITY_CONFIRMED"
    audit = result["boundary_randomisation_audit"]
    assert audit["sign_matrix_sha256"] == R3C._expected_sign_matrix_sha256(
        namespace=audit["namespace"], seed=0, B=199, n_genes=n
    )


@pytest.mark.parametrize("tamper", ["p_value", "sign_digest", "ci"])
def test_observed_replay_rejects_tampered_decision_evidence(tamper):
    n = 40
    groups = [f"gene-{index:02d}" for index in range(n)]
    primitive = np.repeat(R3C.coherent_path_offset(0.40)[:, None], n, axis=1)
    family, names = R3C.derive_family_from_primitives(primitive)
    stats = R3C.gene_sufficient_statistics(family, groups, names)
    canonical, canonical_groups = R3C._reconstruct_cluster_array(stats)
    replay = R3C.strict_path_family_inference(
        canonical,
        canonical_groups,
        names,
        n_draws=199,
        seed=0,
        namespace="r3c-test-tamper",
        min_genes=30,
        path_health=_all_health(),
    )
    stored_inference = {
        key: deepcopy(value)
        for key, value in replay.items()
        if key not in {"paths", "row_names", "row_order_sha256"}
    }
    stored_paths = deepcopy(replay["paths"])
    stored_path_family = deepcopy(replay["path_family"])
    row = R3C.family_row_name(R3C.MEAN_MODE, "L3H1", "partner")
    if tamper == "p_value":
        stored_inference["row_boundary_tests"][row]["p_value"] = 0.25
        # Refresh the visible downstream path/Holm summaries as a sophisticated tamper
        # would; exact row-level replay must still reject it.
        stored_path_family["raw_iut_p_values"]["L3H1_reused_component"] = 0.25
        stored_path_family["holm_adjusted_p_values"] = R3C.holm_adjust_named(
            stored_path_family["raw_iut_p_values"]
        )
        stored_paths["L3H1_reused_component"]["iut_p_value"] = 0.25
        stored_paths["L3H1_reused_component"]["holm_adjusted_p"] = (
            stored_path_family["holm_adjusted_p_values"]["L3H1_reused_component"]
        )
    elif tamper == "sign_digest":
        stored_inference["boundary_randomisation_audit"][
            "sign_matrix_sha256"
        ] = "0" * 64
        for test in stored_inference["row_boundary_tests"].values():
            records = (
                [test["boundary_test"]]
                if "boundary_test" in test
                else [test["lower_edge"], test["upper_edge"]]
            )
            for edge in records:
                edge["sign_matrix_sha256"] = "0" * 64
    else:
        stored_inference["intervals"][row]["simultaneous_ci"] = [0.30, 0.31]
        stored_inference["intervals"][row]["strict_classification"] = (
            "MATERIAL_POSITIVE"
        )
        stored_paths["L3H1_reused_component"][
            "strict_simultaneous_ci_matches"
        ][row] = True
    with pytest.raises(ValueError, match="does not replay"):
        R3C._validate_observed_inference_replay(
            stats=stats,
            stored_inference=stored_inference,
            stored_paths=stored_paths,
            stored_path_family=stored_path_family,
            path_health=_all_health(),
            n_draws=199,
            seed=0,
            namespace="r3c-test-tamper",
            min_genes=30,
        )


def test_instrument_gates_accept_no_candidate_or_scientific_path_outcomes():
    expected = R3C.CURRENT_EXPECTED_CENSUS_IDENTITY
    cohort = {
        "historical_reconstruction": {
            "ordered_excluded_genes_sha256": expected[
                "historical_excluded_gene_sha256"
            ]
        },
        "current_census": {
            "screened_rows": expected["screened_rows"],
            "eligible_rows_by_bucket": expected["eligible_rows_by_bucket"],
            "eligible_genes_by_bucket": expected["eligible_genes_by_bucket"],
            "fixed_selection_by_bucket": expected["fixed_selection_by_bucket"],
        },
    }
    gates = R3C.build_preflight_gates(
        n_genes=30,
        cohort_audit=cohort,
        native_partner_mean=0.30,
        native_pseudo_mean=0.0,
        construction={
            "all_single_acceptor_G": True,
            "all_donor_targets_GT": True,
            "all_motif_clean": True,
            "policy": {"donor_readout_constructed": False},
        },
        health_executed=True,
        exact_family=True,
        calibration={
            "all_surface_fpr_diagnostics_ok": True,
            "all_health_eligible_path_plants_powered": True,
        },
        hook_capability={"all_controls_pass": True},
    )
    assert tuple(gates) == R3C.PREFLIGHT_GATE_NAMES
    assert all(gates.values())
    assert not ({"paths", "path_outcomes", "candidate_outcomes"} & set(
        inspect.signature(R3C.build_preflight_gates).parameters
    ))
    assert "known_L3H1" not in inspect.getsource(R3C.build_preflight_gates)


def test_exact_sesoi_boundary_is_unresolved_not_supported():
    n = 40
    groups = [f"gene-{index:02d}" for index in range(n)]
    primitive = np.repeat(R3C.coherent_path_offset(R3C.SESOI)[:, None], n, axis=1)
    family, names = R3C.derive_family_from_primitives(primitive)
    result = R3C.strict_path_family_inference(
        family,
        groups,
        names,
        n_draws=199,
        seed=1,
        namespace="r3c-test-strict-boundary",
        min_genes=30,
        path_health=_all_health(),
    )
    assert all(
        result["paths"][name]["classification"] != "SUPPORTED"
        for name in R3C.PATH_NAMES
    )
    assert R3C.strict_interval_decision(0.05, 0.05, R3C.SESOI) == "UNRESOLVED"


def test_health_suppresses_only_dependent_component_mode_paths():
    n = 40
    groups = [f"gene-{index:02d}" for index in range(n)]
    primitive = np.repeat(R3C.coherent_path_offset(0.40)[:, None], n, axis=1)
    family, names = R3C.derive_family_from_primitives(primitive)
    health = _all_health()
    health["L4H7_noising_component"] = {
        "ok": False,
        "participants": [f"{R3C.NOISING_MODE}|L4H7"],
        "failed_participants": [f"{R3C.NOISING_MODE}|L4H7"],
    }
    result = R3C.strict_path_family_inference(
        family,
        groups,
        names,
        n_draws=399,
        seed=2,
        namespace="r3c-test-local-health",
        min_genes=30,
        path_health=health,
    )
    assert result["paths"]["L4H7_noising_component"]["classification"] == (
        "HEALTH_FAILED"
    )
    assert result["paths"]["distinct_acceptor_route_candidate"]["classification"] == (
        "SUPPORTED"
    )
    assert result["paths"]["L3H1_reused_component"]["classification"] == "SUPPORTED"


def test_path_health_dependency_graph_is_local():
    measurements = _measurements()
    measurements[R3C.NOISING_MODE]["L4H7"]["health"]["ok"] = False
    health = R3C.path_health_eligibility(measurements)
    assert health["L4H7_noising_component"]["ok"] is False
    assert health["distinct_acceptor_route_candidate"]["ok"] is True
    assert health["L3H1_reused_component"]["ok"] is True

    measurements[R3C.NOISING_MODE]["L4H0"]["health"]["ok"] = False
    health = R3C.path_health_eligibility(measurements)
    assert health["distinct_acceptor_route_candidate"]["ok"] is False
    assert health["L4H11_noising_component"]["ok"] is True


def test_synthetic_calibration_exercises_all_seven_paths_regardless_of_live_health():
    calibration_health = R3C.calibration_path_health_eligibility()
    assert tuple(calibration_health) == R3C.PATH_NAMES
    assert all(item["ok"] is True for item in calibration_health.values())
    assert all(item["failed_participants"] == [] for item in calibration_health.values())

    compact = {
        "health_eligible_paths": list(R3C.PATH_NAMES),
        "planted_power_by_path": {
            name: {"health_eligible": True} for name in R3C.PATH_NAMES
        },
    }
    expected = {name: True for name in R3C.PATH_NAMES}
    R3C._validate_all_path_calibration_health(compact, expected)
    compact["planted_power_by_path"]["L2H15_mean_component"][
        "health_eligible"
    ] = False
    with pytest.raises(ValueError, match="all seven paths"):
        R3C._validate_all_path_calibration_health(compact, expected)


def test_target_path_health_can_only_shrink_from_reviewed_preflight():
    current = R3C.path_health_eligibility(_measurements())
    reviewed = R3C.reviewed_preflight_path_health(current)
    assert current["L2H15_mean_component"]["ok"] is True
    assert reviewed["L2H15_mean_component"]["ok"] is False
    assert reviewed["L2H15_mean_component"]["reviewed_v1_health_ceiling_ok"] is False
    assert reviewed["L3H1_reused_component"]["ok"] is True

    all_healthy_preflight = {
        name: {"health": dict(item)} for name, item in current.items()
    }
    current_failure = {name: dict(item) for name, item in current.items()}
    current_failure["L4H11_noising_component"]["ok"] = False
    current_failure["L4H11_noising_component"]["failed_participants"] = [
        f"{R3C.NOISING_MODE}|L4H11"
    ]
    effective_failure, _ = R3C.monotone_target_path_health(
        current_failure, all_healthy_preflight
    )
    assert effective_failure["L4H11_noising_component"]["preflight_health_ok"] is True
    assert effective_failure["L4H11_noising_component"]["current_health_ok"] is False
    assert effective_failure["L4H11_noising_component"]["ok"] is False

    preflight_paths = {
        name: {"health": dict(item)} for name, item in current.items()
    }
    preflight_paths["L2H15_mean_component"]["health"]["ok"] = False
    preflight_paths["L2H15_mean_component"]["health"]["failed_participants"] = [
        f"{R3C.MEAN_MODE}|L2H15"
    ]

    effective, authorization = R3C.monotone_target_path_health(
        current, preflight_paths
    )
    assert current["L2H15_mean_component"]["ok"] is True
    assert effective["L2H15_mean_component"]["ok"] is False
    assert effective["L2H15_mean_component"]["preflight_health_ok"] is False
    assert effective["L2H15_mean_component"]["current_health_ok"] is True
    assert effective["L3H1_reused_component"]["ok"] is True
    assert authorization["can_only_shrink_from_preflight"] is True
    assert authorization["preflight_suppressed_paths"] == [
        "L2H15_mean_component"
    ]
    assert "L2H15_mean_component" not in authorization[
        "effective_authorized_paths"
    ]


def test_acceptor_arm_masks_only_next_acceptor_g_and_constructs_only_donor_edits():
    sequence = list("A" * 100)
    sequence[20:22] = list("GT")
    sequence[40:42] = list("GT")
    sequence[78:80] = list("AG")
    example = SimpleNamespace(
        example_id="example",
        gene_id="gene",
        sequence="".join(sequence),
        donor_pos=20,
        next_acceptor_pos=80,
    )
    task = SimpleNamespace()
    arm = R3C.AcceptorArm(task, [example], [40])
    assert arm.mask == [79]
    assert arm.clean[0][arm.mask[0]] == "G"
    assert arm.partner_positions == [20]
    assert arm.pseudo_positions == [40]
    assert arm.partner_corrupt[0][20:22] == R3C.MOTIF_CLEAN_DONOR_ALT
    assert arm.pseudo_corrupt[0][40:42] == R3C.MOTIF_CLEAN_DONOR_ALT
    assert arm.partner_corrupt[0][78:80] == "AG"
    assert arm.pseudo_corrupt[0][78:80] == "AG"
    assert R3C.ENDPOINTS == ("partner", "pseudo", "specific")

    source = inspect.getsource(R3C.AcceptorArm)
    assert "next_acceptor_pos) - 1" in source
    assert "self.mask = [int(example.donor_pos)" not in source
    capability_source = inspect.getsource(R3C.run_hook_capability)
    for memorial in (
        "clean_to_clean_whole_head_self_patch",
        "post_attention_head_split_merge_identity",
        "same_example_source_target_alignment",
        "batch_size_invariance",
        "final_hidden_through_full_model_cls_reproduces_logits_and_margin",
        "all_L3_clean_to_corrupt_live_restoration",
    ):
        assert memorial in capability_source


def test_gene_bucket_preserves_r3b_salt():
    gene = "ENSG000001"
    digest = hashlib.sha256(f"r3b-gene|{gene}".encode()).digest()
    assert R3C.GENE_BUCKET_SALT == "r3b-gene"
    assert R3C.gene_bucket(gene) == int.from_bytes(digest[:8], "big") % 2


def test_cohort_scans_exhaustively_excludes_historical_genes_and_never_requires_current_acceptor_pseudo(
    monkeypatch,
):
    def example(identifier, gene, donor=20):
        return SimpleNamespace(
            example_id=identifier,
            gene_id=gene,
            sequence="A" * 100,
            donor_pos=donor,
            next_acceptor_pos=70,
            real_don=(),
        )

    historical = [example("old-complete", "old-gene"), example("old-missing", "other")]
    current = [
        example("old-new", "old-gene"),
        example("fresh-a", "fresh-a"),
        example("too-far", "fresh-far"),
        example("fresh-b", "fresh-b"),
    ]
    calls = []
    acceptor_calls = []

    def fake_load(_gtf, _fasta, split, n, *, tag, allowed_chromosomes):
        calls.append((split, n, tag, allowed_chromosomes))
        return historical if tag == R3C.HISTORICAL_TAG else current

    def fake_donor(row):
        if row.example_id == "old-missing":
            return None
        if row.example_id == "too-far":
            return 90
        return 30

    def fake_acceptor(row, tolerance):
        acceptor_calls.append((row.example_id, tolerance))
        return 80

    monkeypatch.setattr(R3C, "load_examples_unscored", fake_load)
    monkeypatch.setattr(R3C, "matched_decoy_donor", fake_donor)
    monkeypatch.setattr(R3C, "matched_decoy_acceptor", fake_acceptor)
    monkeypatch.setattr(R3C, "donor_alt_admissible", lambda *_args: True)
    monkeypatch.setattr(R3C, "planted_donor_motifs", lambda *_args: [])
    monkeypatch.setattr(R3C, "gene_bucket", lambda _gene: 0)
    monkeypatch.setattr(R3C, "MIN_GENES", 1)
    monkeypatch.setattr(R3C, "HISTORICAL_EXPECTED_ROWS", 1)
    monkeypatch.setattr(R3C, "HISTORICAL_EXPECTED_GENES", 1)
    monkeypatch.setattr(
        R3C,
        "HISTORICAL_EXPECTED_EXAMPLE_ID_SHA256",
        hashlib.sha256(b"old-complete").hexdigest(),
    )
    monkeypatch.setattr(R3C, "validate_current_census_identity", lambda audit: audit)
    args = SimpleNamespace(stage="preflight", n=2, gtf="unused", fasta="unused")
    selected, donors, bucket, audit = R3C.load_cohort(args)

    assert [row.example_id for row in selected] == ["fresh-a", "fresh-b"]
    assert donors == [30, 30] and bucket == 0
    assert calls == [
        (
            "val",
            R3C.HISTORICAL_SCREEN_N,
            R3C.HISTORICAL_TAG,
            R3C.VALIDATION_CHROMOSOMES,
        ),
        ("val", None, R3C.CURRENT_TAG, R3C.VALIDATION_CHROMOSOMES),
    ]
    assert acceptor_calls == [
        ("old-complete", R3C.DECOY_TOL),
        ("old-missing", R3C.DECOY_TOL),
    ]
    assert audit["current_census"]["eligibility_requires_matched_acceptor"] is False
    assert audit["current_census"]["eligibility_requires_only_donor_pseudo"] is True
    assert audit["current_census"]["exclusions"]["historical_R3_gene"] == 1
    assert (
        audit["current_census"]["exclusions"][
            "donor_pseudo_motif_outside_distance_caliper"
        ]
        == 1
    )
    assert audit["selection_reads_model_outputs"] is False
    assert audit["sealed_chr8_chr9_opened"] is False


def test_historical_identity_rejects_count_or_hash_drift(monkeypatch):
    rows = [
        {"example_id": "e1", "gene_id": "g1"},
        {"example_id": "e2", "gene_id": "g2"},
    ]
    identity = R3C.historical_r3_identity(rows)
    monkeypatch.setattr(R3C, "HISTORICAL_EXPECTED_ROWS", 2)
    monkeypatch.setattr(R3C, "HISTORICAL_EXPECTED_GENES", 2)
    monkeypatch.setattr(
        R3C, "HISTORICAL_EXPECTED_EXAMPLE_ID_SHA256", identity["example_id_sha256"]
    )
    assert R3C.validate_historical_r3_identity(rows) == identity
    monkeypatch.setattr(R3C, "HISTORICAL_EXPECTED_EXAMPLE_ID_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="historical R3"):
        R3C.validate_historical_r3_identity(rows)


def test_bound_full_gtf_census_identity_is_exact_and_fail_closed():
    expected = R3C.CURRENT_EXPECTED_CENSUS_IDENTITY
    assert expected["screened_rows"] == 6064
    assert expected["eligible_rows_by_bucket"] == {"0": 1010, "1": 1058}
    assert expected["eligible_genes_by_bucket"] == {"0": 417, "1": 443}
    assert expected["fixed_selection_by_bucket"]["0"]["genes"] == 284
    assert expected["fixed_selection_by_bucket"]["1"]["genes"] == 304
    bad = {
        "historical_reconstruction": {"ordered_excluded_genes_sha256": "0" * 64},
        "current_census": {
            "screened_rows": 1,
            "eligible_rows_by_bucket": {"0": 1, "1": 1},
            "eligible_genes_by_bucket": {"0": 1, "1": 1},
            "fixed_selection_by_bucket": {},
        },
    }
    with pytest.raises(RuntimeError, match="census identity drifted"):
        R3C.validate_current_census_identity(bad)


def test_unpinned_preflight_keeps_target_before_user_paths_or_outputs(tmp_path):
    output = tmp_path / "must-not-write.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--stage",
            "target",
            "--gtf",
            str(tmp_path / "must-not-read.gtf"),
            "--fasta",
            str(tmp_path / "must-not-read.fa"),
            "--model",
            str(tmp_path / "must-not-read-model"),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "preflight-artifact" in completed.stderr
    assert "missing" not in completed.stderr.lower()
    assert not output.exists()


def test_target_requires_protocol_instrument_and_exact_preflight_artifact_pins():
    with pytest.raises(R3C.TargetSealError, match="DRAFT"):
        R3C.authorize_target(
            "observed",
            protocol_text="> **Status: DRAFT",
            frozen_protocol_sha256=None,
        )
    frozen = "> **Status: FROZEN 2026-08-15.**"
    R3C.authorize_target(
        "same",
        protocol_text=frozen,
        frozen_protocol_sha256="same",
        frozen_preflight_artifact_sha256="a" * 64,
    )
    with pytest.raises(R3C.TargetSealError, match="preflight-artifact"):
        R3C.authorize_target(
            "same",
            protocol_text=frozen,
            frozen_protocol_sha256="same",
            frozen_preflight_artifact_sha256=None,
        )
    with pytest.raises(R3C.TargetSealError, match="fingerprint"):
        R3C.verify_frozen_instrument("observed", frozen_fingerprint=None)
    R3C.verify_frozen_instrument("same", frozen_fingerprint="same")
    assert (
        R3C.authorize_bindable_preflight(
            "p",
            "i",
            protocol_text="> **Status: DRAFT",
            frozen_protocol_sha256=None,
            frozen_fingerprint=None,
        )
        is False
    )
    with pytest.raises(R3C.TargetSealError, match="both literal pins"):
        R3C.authorize_bindable_preflight(
            "same",
            "same",
            protocol_text=frozen,
            frozen_protocol_sha256="same",
            frozen_fingerprint=None,
        )


def test_atomic_publication_refuses_overwrite(tmp_path):
    output = tmp_path / "fresh.json"
    R3C.atomic_write_new(output, b"first\n")
    assert output.read_bytes() == b"first\n"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        R3C.atomic_write_new(output, b"second\n")
    assert output.read_bytes() == b"first\n"


def test_preflight_manifest_requires_literal_exact_artifact_hash(tmp_path):
    artifact = tmp_path / "preflight.json"
    artifact.write_text(json.dumps({"stage": "preflight"}))
    digest = R3C.sha256_file(artifact)
    artifact.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": (
                    "splicebert_acceptor_coupling:r3c_acceptor_head_confirmation"
                ),
                "outputs": [{"path": str(artifact), "sha256": digest}],
            }
        )
    )
    record, package = R3C.verify_preflight_manifest(
        artifact, frozen_artifact_sha256=digest
    )
    assert record == {"stage": "preflight"}
    assert package["artifact"]["sha256"] == digest
    with pytest.raises(ValueError, match="literal reviewed"):
        R3C.verify_preflight_manifest(artifact, frozen_artifact_sha256=None)
    with pytest.raises(ValueError, match="differs"):
        R3C.verify_preflight_manifest(
            artifact, frozen_artifact_sha256="0" * 64
        )
def test_protocol_and_binding_pin_acceptor_only_contract():
    protocol = R3C.PROTOCOL.read_text()
    assert "Status: FROZEN" in protocol
    assert "4,000" in protocol
    assert "exactly 62" in protocol
    assert "Holm across exactly" in protocol
    assert "exactly 28" in protocol
    assert "5,800" in protocol
    assert "not a same-protocol double" in protocol
    assert R3C.FROZEN_PROTOCOL_SHA256 == R3C.sha256_file(R3C.PROTOCOL)
    assert isinstance(R3C.FROZEN_INSTRUMENT_FINGERPRINT_SHA256, str)
    assert len(R3C.FROZEN_INSTRUMENT_FINGERPRINT_SHA256) == 64
    assert R3C.FROZEN_PREFLIGHT_ARTIFACT_SHA256 == (
        "c0c9306fe894ce73bfd981a14c85e2aec6660da33f01d9716cee4327b3ff2d3e"
    )
    assert R3C.authorize_bindable_preflight(
        R3C.FROZEN_PROTOCOL_SHA256,
        R3C.FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
        protocol_text=protocol,
    )
    assert R3C.CURRENT_TAG == "r3c-acceptor-head-confirmation-v1"
    assert R3C.VALIDATION_CHROMOSOMES == ("chr11", "chr16", "chr19")
