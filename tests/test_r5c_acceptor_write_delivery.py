from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert_acceptor_coupling/run_r5c_acceptor_write_delivery.py"
)
SPEC = importlib.util.spec_from_file_location("run_r5c_acceptor_write_delivery", SCRIPT)
R5C = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = R5C
SPEC.loader.exec_module(R5C)


def _sha(character: str = "a") -> str:
    return character * 64


def _entry_chain():
    common = {
        "instrument_gates_pass": True,
        "claim_eligible": True,
        "acceptor_G_only": True,
        "masked_donor_endpoint_constructed": False,
        "checkpoint_sha256": _sha("a"),
    }
    return {
        "r3c": common
        | {"verdict": "HELD_OUT_ACCEPTOR_HEAD_CONFIRMATION_COMPLETE"},
        "r5a": common
        | {
            "schema": R5C.R5_ENTRY_CONTRACT_SCHEMA,
            R5C.STACK_FINGERPRINT_FIELD: _sha("9"),
            "verdict": "L3H1_DONOR_LOCAL_READER_SUPPORTED",
            "cohort_sha256": _sha("b"),
            "construction_sha256": _sha("c"),
        },
        "r5b": common
        | {
            "schema": R5C.R5_ENTRY_CONTRACT_SCHEMA,
            R5C.STACK_FINGERPRINT_FIELD: _sha("9"),
            "verdict": "ACCEPTOR_SIDE_STATE_SUPPORTED",
            "cohort_sha256": _sha("b"),
            "construction_sha256": _sha("c"),
            "selected_rank": 4,
            "crossfit_map_sha256": _sha("d"),
        },
    }


def test_upstream_instrument_stack_fingerprints_use_each_stage_actual_field():
    fingerprint = _sha("9")
    r5a = {"instrument_binding": {R5C.STACK_FINGERPRINT_FIELD: fingerprint}}
    r5b = {
        "instrument_binding_payload": {
            R5C.STACK_FINGERPRINT_FIELD: fingerprint
        }
    }
    R5C.validate_upstream_instrument_stack_fingerprints(
        r5a,
        r5b,
        current_stack_fingerprint=fingerprint,
    )

    wrong_r5b_shape = {
        "instrument_binding": {R5C.STACK_FINGERPRINT_FIELD: fingerprint}
    }
    with pytest.raises(R5C.InstrumentError, match="instrument bindings differ"):
        R5C.validate_upstream_instrument_stack_fingerprints(
            r5a,
            wrong_r5b_shape,
            current_stack_fingerprint=fingerprint,
        )


def test_exact_artifact_loader_requires_manifest_output_path(tmp_path):
    artifact = tmp_path / "entry.json"
    artifact.write_text(json.dumps({"schema": "test"}))
    artifact_sha256 = R5C.sha256_file(artifact)
    manifest_path = artifact.with_suffix(".manifest.json")
    manifest = {
        "git_dirty": False,
        "outputs": [
            {
                "path": str(tmp_path / "different.json"),
                "size_bytes": artifact.stat().st_size,
                "sha256": artifact_sha256,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(R5C.InstrumentError, match="manifest does not independently bind"):
        R5C._load_exact_json_artifact(
            artifact,
            expected_sha256=artifact_sha256,
            label="test entry",
        )

    manifest["outputs"][0]["path"] = str(artifact)
    manifest_path.write_text(json.dumps(manifest))
    record, loaded_manifest = R5C._load_exact_json_artifact(
        artifact,
        expected_sha256=artifact_sha256,
        label="test entry",
    )
    assert record == {"schema": "test"}
    assert loaded_manifest == manifest


def test_preflight_context_binds_exact_upstream_manifest_package():
    authorization = R5C.validate_entry_chain(_entry_chain())
    authorization["upstream_packages"] = {
        stage: {
            "artifact_sha256": _sha(str(index)),
            "artifact_size_bytes": 100 + index,
            "manifest_sha256": _sha(chr(ord("a") + index)),
            "manifest_size_bytes": 200 + index,
        }
        for index, stage in enumerate(("r3c", "r5a", "r5b"), start=1)
    }
    runtime_contract = {"runtime_bundle_sha256": _sha("f")}
    record = {
        "status": "PREFLIGHT_INSTRUMENT_PASS",
        "entry": {
            "artifact_sha256": R5C._entry_artifact_hashes(),
            "authorization": deepcopy(authorization),
        },
        "checkpoint": {
            "checkpoint_aggregate_sha256": authorization["checkpoint_sha256"]
        },
        "source_identity": R5C.R5_EXPECTED_SOURCE_IDENTITY,
        "runtime_identity": R5C.stable_runtime_identity(),
        "cohort": {"binding": R5C.expected_r5_binding(0)},
        "r5b_runtime_contract": runtime_contract,
        "runtime_bundle_sha256": runtime_contract["runtime_bundle_sha256"],
    }
    assert R5C.validate_preflight_context(
        record,
        entry_authorization=authorization,
        checkpoint_sha256=authorization["checkpoint_sha256"],
        runtime_contract=runtime_contract,
    )["status"] == "FROZEN_PREFLIGHT_CONTEXT_PASS"

    changed = deepcopy(record)
    changed["entry"]["authorization"]["upstream_packages"]["r5b"][
        "manifest_sha256"
    ] = _sha("0")
    with pytest.raises(R5C.InstrumentError, match="upstream authorization differs"):
        R5C.validate_preflight_context(
            changed,
            entry_authorization=authorization,
            checkpoint_sha256=authorization["checkpoint_sha256"],
            runtime_contract=runtime_contract,
        )


def _passing_diagnostic_records():
    registry = R5C.calibration_registry()
    diagnostics = {}
    for case in registry["coherent_face_registry"]:
        fields = R5C._row_fields(case["row"])
        signature = (
            case["family"],
            case["edge"],
            "gap" if "gap" in fields else fields.get("endpoint", "unknown"),
        )
        key = "|".join(signature)
        if key not in diagnostics:
            diagnostics[key] = {
                "representative_row": case["row"],
                "representative_edge": case["edge"],
                "requiring_paths": [
                    path
                    for path, requirements in R5C.path_registry()[case["family"]].items()
                    if any(requirement.row == case["row"] for requirement in requirements)
                ],
                "inference_sha256": _sha("d"),
            }
    binding = {
        "schema": "gfmcd.r5c.constituent_binding_suite/1",
        "production_classifier_reused": True,
        "n_draws": R5C.PRODUCTION_RANDOMISATION,
        "literal_faces_checked": registry["literal_edge_count"],
        "literal_face_matrices_sha256": _sha("a"),
        "signature_diagnostics": diagnostics,
        "signature_count": len(diagnostics),
        "all_alternative_plant_inference_sha256": {
            family: _sha("b") for family in R5C.FAMILY_NAMES
        },
        "all_boundary_signatures_rejected": True,
        "all_path_plants_supported": True,
        "engineering_not_scientific_fwer_evidence": True,
        "status": "PASS",
    }
    trials = R5C.computed_calibration_trials()
    _, false_upper = R5C.one_sided_binomial_bounds(0, trials, alpha=0.05)
    plant_lower, _ = R5C.one_sided_binomial_bounds(trials, trials, alpha=0.05)
    family_records = {}
    for family in R5C.FAMILY_NAMES:
        cases = [
            case
            for case in registry["coherent_face_registry"]
            if case["family"] == family
        ]
        family_records[family] = {
            "trials": trials,
            "false_positive_trials": 0,
            "false_positive_rate": 0.0,
            "one_sided_clopper_pearson_upper": false_upper,
            "boundary_gate_ok": True,
            "plant_success_trials": trials,
            "plant_success_rate": 1.0,
            "one_sided_clopper_pearson_lower": plant_lower,
            "plant_gate_ok": True,
            "false_positive_pattern_sha256": _sha("e"),
            "plant_success_pattern_sha256": _sha("f"),
            "balanced_case_cycle_sha256": R5C.canonical_sha256(
                [
                    [
                        cases[trial % len(cases)]["row"],
                        cases[trial % len(cases)]["edge"],
                    ]
                    for trial in range(trials)
                ]
            ),
        }
    calibration = {
        "schema": "gfmcd.r5c.engineering_calibration/1",
        "production_trial_count": True,
        "N_cal": trials,
        "n_draws_per_classifier": R5C.PRODUCTION_RANDOMISATION,
        "family_records": family_records,
        "all_family_gates_pass": True,
        "engineering_not_scientific_fwer_evidence": True,
        "status": "PASS",
    }
    return binding, calibration


def _sandwich(
    *,
    leg_id: str,
    component: str,
    m_minus: np.ndarray,
    m_c_plus: np.ndarray,
    effect: float,
):
    record = R5C.delivery_sandwich_estimands(
        m_minus=m_minus,
        m_c_plus=m_c_plus,
        m_c_plus_with_d_minus=m_c_plus - effect,
        m_minus_with_d_from_c_plus=m_minus + effect,
        leg_id=leg_id,
        component_id=component,
    )
    record["common_leg"] = R5C.common_leg_record(leg_id, m_minus, m_c_plus)
    return record


def _capture_identity(label: str):
    record = {
        "schema": "gfmcd.r5c.condition_capture_identity/1",
        "margin_sha256": R5C.canonical_sha256([label, "margin"]),
        "query_residual_sha256": R5C.canonical_sha256([label, "residual"]),
        "query_logits_sha256": R5C.canonical_sha256([label, "logits"]),
        "head_outputs_sha256_by_layer": {
            str(layer): R5C.canonical_sha256([label, "head", layer])
            for layer in range(6)
        },
        "full_mlm_margin_replay_max_abs_error": 0.0,
        "full_mlm_logits_replay_exact": True,
    }
    record["capture_sha256"] = R5C.canonical_sha256(record)
    return record


def _mutation_record(role, component, heads, *, n, query_index=1, self_patch=False):
    sequence_tokens = 3
    mask = np.zeros((n, sequence_tokens), dtype=bool)
    mask[:, query_index] = True
    layers = sorted({layer for layer, _ in heads})
    source = {
        layer: np.ones((n, 16, sequence_tokens, 1), dtype=np.float32)
        for layer in layers
    }
    before = {
        layer: np.zeros((n, 16, sequence_tokens, 1), dtype=np.float32)
        for layer in layers
    }
    if self_patch:
        source = {layer: value.copy() for layer, value in before.items()}
    after = {layer: value.copy() for layer, value in before.items()}
    for layer, head in heads:
        after[layer][:, head, query_index] = source[layer][:, head, query_index]
    return R5C._head_mutation_audit_record(
        intervention_id=f"{role}:{component}",
        role=role,
        component_id=component,
        heads=heads,
        token_mask=R5C.torch.from_numpy(mask),
        source_cache={
            layer: R5C.torch.from_numpy(value) for layer, value in source.items()
        },
        before_by_layer={
            layer: R5C.torch.from_numpy(value) for layer, value in before.items()
        },
        after_by_layer={
            layer: R5C.torch.from_numpy(value) for layer, value in after.items()
        },
    )


def _leg(leg_id: str, *, n: int = 5):
    partner = leg_id == "partner"
    m_minus = np.zeros(n)
    m_c_plus = np.full(n, 0.30 if partner else 0.0)
    u_effect = 0.25 if partner else 0.0
    u = R5C.l3_to_u_mediation_estimands(
        m_minus=m_minus,
        m_l3_plus=m_c_plus,
        m_minus_with_u_from_l3_plus=m_minus + u_effect,
        m_l3_plus_with_u_from_minus=m_c_plus - u_effect,
        leg_id=leg_id,
    )
    random_u = []
    for draw in range(32):
        effect = np.full(n, 0.01 if partner else 0.0)
        item = R5C.l3_to_u_mediation_estimands(
            m_minus=m_minus,
            m_l3_plus=m_c_plus,
            m_minus_with_u_from_l3_plus=m_minus + effect,
            m_l3_plus_with_u_from_minus=m_c_plus - effect,
            leg_id=leg_id,
        )
        item.update(
            {
                "draw": draw,
                "projector_sha256": f"{draw:064x}",
                "per_row_energy_match_max_abs_error": 0.0,
                "redrawn": False,
            }
        )
        random_u.append(item)
    sandwich = {}
    for ensemble in R5C.complete_downstream_registry():
        identity = R5C.ensemble_name(ensemble)
        is_candidate = bool(set(ensemble) & set(R5C.ACCEPTOR_CANDIDATES))
        effect = 0.20 if partner and is_candidate else 0.0
        sandwich[identity] = _sandwich(
            leg_id=leg_id,
            component=identity,
            m_minus=m_minus,
            m_c_plus=m_c_plus,
            effect=effect,
        )
    direct_cells = {
        "m_minus": m_minus,
        "m_direct_negative": m_minus.copy(),
    }
    minus_capture = _capture_identity(f"{leg_id}:minus")
    plus_capture = _capture_identity(f"{leg_id}:plus")
    aggregate_capture = {
        "m_minus_capture_sha256": R5C.canonical_sha256(
            [minus_capture["capture_sha256"]]
        ),
        "m_c_plus_capture_sha256": R5C.canonical_sha256(
            [plus_capture["capture_sha256"]]
        ),
    }
    common = R5C.common_leg_record(
        leg_id, m_minus, m_c_plus, **aggregate_capture
    )
    registered = tuple(
        sorted(
            {R5C.L3_READER, R5C.DIRECT_NEGATIVE}
            | {
                head
                for ensemble in R5C.complete_downstream_registry()
                for head in ensemble
            }
        )
    )
    mutation_records = [
        _mutation_record(
            "registered_clean_self_patch",
            "ALL_REGISTERED_HEADS",
            registered,
            n=n,
            self_patch=True,
        ),
        _mutation_record(
            "registered_minus_self_patch",
            "ALL_REGISTERED_HEADS",
            registered,
            n=n,
            self_patch=True,
        ),
        _mutation_record("l3_c_plus", "L3H1", (R5C.L3_READER,), n=n),
        _mutation_record(
            "direct_negative", "L0H2", (R5C.DIRECT_NEGATIVE,), n=n
        ),
    ]
    for ensemble in R5C.complete_downstream_registry():
        identity = R5C.ensemble_name(ensemble)
        audits = {
            "necessity_l3": _mutation_record(
                "sandwich_necessity_l3",
                identity,
                (R5C.L3_READER,),
                n=n,
            ),
            "necessity_delivery": _mutation_record(
                "sandwich_necessity_delivery", identity, ensemble, n=n
            ),
            "sufficiency_delivery": _mutation_record(
                "sandwich_sufficiency_delivery", identity, ensemble, n=n
            ),
        }
        sandwich[identity]["common_leg"] = dict(common)
        sandwich[identity]["mutation_audit_sha256_by_chunk"] = {
            key: [value["audit_sha256"]] for key, value in audits.items()
        }
        sandwich[identity]["condition_capture_sha256"] = {
            "necessity": _sha("a"),
            "sufficiency": _sha("b"),
        }
        mutation_records.extend(audits.values())
    capture_records = [minus_capture, plus_capture]
    replay_summary = {
        "condition_count": len(capture_records),
        "margin_max_abs_error": 0.0,
        "full_logits_exact": True,
        "condition_capture_registry_sha256": R5C.canonical_sha256(
            [record["capture_sha256"] for record in capture_records]
        ),
    }
    return {
        "leg_id": leg_id,
        "readout": "single masked acceptor G hard margin",
        "masked_donor_endpoint_constructed": False,
        "common_leg": common,
        "u_a": u,
        "direct_negative": {
            "component_id": "L0H2",
            "cells": direct_cells,
            "estimands": {"direct_negative_rescue": np.zeros(n)},
            "margin_sha256": R5C.array_sha256(m_minus),
        },
        "random_u_a": random_u,
        "sandwich": sandwich,
        "hook_capability": {
            "registered_heads": [R5C.head_name(head) for head in registered],
            "registered_query_self_patch_margin_max_abs_error": 0.0,
            "registered_query_self_patch_full_logits_exact": True,
            "l3_query_self_patch_included": True,
            "candidate_outcomes_used": False,
            "dense_full_residual_replay": replay_summary,
            "condition_capture_records": capture_records,
            "mutation_audit_chunks": [mutation_records],
            "expected_query_token_indices_by_chunk": [[1] * n],
        },
        "chunk_condition_hashes": [
            {
                "m_minus_capture_identity": minus_capture,
                "m_c_plus_capture_identity": plus_capture,
            }
        ],
    }


def _projector_capability_fixture():
    projector = np.repeat(
        np.diag([1.0, 0.0])[None, :, :], R5C.R5_ROWS_PER_BUCKET, axis=0
    )
    selected = R5C.validate_crossfit_projectors(projector, selected_rank=1)
    random = [
        {"draw": draw, **R5C.validate_crossfit_projectors(projector, selected_rank=1)}
        for draw in range(32)
    ]
    runtime_hashes = {
        "selected_bases_by_fold": _sha("1"),
        "random_bases_by_draw_fold": _sha("2"),
        "row_fold_ids": _sha("3"),
        "ordered_row_ids": _sha("4"),
    }
    record = {
        "schema": "gfmcd.r5c.crossfit_projector_capability/1",
        "selected": selected,
        "random": random,
        "authenticated_runtime_array_sha256": runtime_hashes,
        "observed_runtime_array_sha256": dict(runtime_hashes),
        "all_projectors_validated": 33,
        "audit_sha256": R5C.canonical_sha256(
            {"selected": selected, "random": random}
        ),
    }
    return record, runtime_hashes


def _capability_fixture(partner, pseudo):
    projector_audit, runtime_hashes = _projector_capability_fixture()
    residual = {
        "partner": R5C.validate_residual_replay_registry(partner),
        "pseudo": R5C.validate_residual_replay_registry(pseudo),
    }
    mutation = {
        "partner": R5C.validate_mutation_audit_registry(partner),
        "pseudo": R5C.validate_mutation_audit_registry(pseudo),
    }
    common = {
        "partner": R5C.validate_common_leg_capture_registry(partner),
        "pseudo": R5C.validate_common_leg_capture_registry(pseudo),
    }
    evidence = {
        "readout_validation": {
            "acceptor_g_mismatch_count": 0,
            "rows_checked": R5C.R5_ROWS_PER_BUCKET,
        },
        "residual_replay": residual,
        "self_patch": {
            "margin_max_abs_error": 0.0,
            "full_logits_exact": True,
            "l3_included": True,
            "registered_heads_exact": True,
        },
        "mutation_audit": mutation,
        "common_leg_capture": common,
        "crossfit_projectors": projector_audit,
        "random_projectors": {
            "energy_match_max_abs_error": 0.0,
            "redrawn_count": 0,
            "draws_per_leg": 32,
        },
        "health": {"all_arms_ok": True, "candidate_outcomes_used": False},
        "diagnostics": {
            "binding_suite_status": "PASS",
            "engineering_calibration_status": "PASS",
            "production_trial_count": True,
        },
    }
    record = {
        "gates": {name: True for name in R5C.CAPABILITY_GATES},
        "evidence": evidence,
        "candidate_outcomes_used": False,
    }
    return record, runtime_hashes


def _as_execution_chunk(leg):
    chunk = deepcopy(leg)
    hook = chunk["hook_capability"]
    hook["mutation_audits"] = hook.pop("mutation_audit_chunks")[0]
    hook["expected_query_token_indices"] = hook.pop(
        "expected_query_token_indices_by_chunk"
    )[0]
    conditions = chunk["chunk_condition_hashes"][0]
    m_minus = np.asarray(chunk["u_a"]["cells"]["m_minus"], dtype=float)
    m_c_plus = np.asarray(chunk["u_a"]["cells"]["m_l3_plus"], dtype=float)
    chunk["conditions"] = {
        "clean_margin": m_c_plus.copy(),
        "m_minus_margin": m_minus.copy(),
        "m_c_plus_margin": m_c_plus.copy(),
        **conditions,
    }
    raw_capture = {
        "m_minus_capture_sha256": conditions["m_minus_capture_identity"][
            "capture_sha256"
        ],
        "m_c_plus_capture_sha256": conditions["m_c_plus_capture_identity"][
            "capture_sha256"
        ],
    }
    for envelope in chunk["sandwich"].values():
        envelope["common_leg"] = R5C.common_leg_record(
            chunk["leg_id"],
            envelope["cells"]["m_minus"],
            envelope["cells"]["m_c_plus"],
            **raw_capture,
        )
        envelope["mutation_audit_sha256"] = {
            direction: values[0]
            for direction, values in envelope.pop(
                "mutation_audit_sha256_by_chunk"
            ).items()
        }
    chunk["common_leg"] = next(iter(chunk["sandwich"].values()))["common_leg"]
    chunk.pop("chunk_condition_hashes")
    return chunk


def test_registry_contains_fixed_heads_and_excludes_them_from_every_random_ensemble():
    registry = R5C.random_ensemble_registry()
    assert {key: len(value) for key, value in registry.items()} == {
        "pairs_l4_l4": 8,
        "pairs_l4_l5": 8,
        "triples_l4_l4_l4": 8,
        "triples_l4_l4_l5": 8,
    }
    flattened = [ensemble for values in registry.values() for ensemble in values]
    assert len({R5C.ensemble_name(ensemble) for ensemble in flattened}) == 32
    assert all(
        not (set(ensemble) & R5C.REGISTERED_DOWNSTREAM_HEADS)
        for ensemble in flattened
    )
    assert R5C.ACCEPTOR_CANDIDATES == ((4, 11), (4, 9), (4, 7))
    assert R5C.DONOR_ROUTE_COMPARATORS == ((4, 0), (4, 2), (5, 11))


def test_complete_registry_has_all_singletons_all_three_donor_pairs_and_fixed_triples():
    registry = R5C.complete_downstream_registry()
    assert len(registry) == 44
    assert registry[:6] == tuple(
        (head,) for head in R5C.ACCEPTOR_CANDIDATES + R5C.DONOR_ROUTE_COMPARATORS
    )
    assert all(pair in registry for pair in R5C.DONOR_ROUTE_PAIRS)
    assert R5C.SECONDARY_ACCEPTOR_TRIPLE in registry
    assert R5C.DONOR_ROUTE_TRIPLE in registry
    assert len({R5C.ensemble_name(item) for item in registry}) == len(registry)


def test_intervention_size_families_are_exact_separate_and_have_closed_paths():
    rows = R5C.family_row_names()
    assert {family: len(names) for family, names in rows.items()} == {
        "C1": 6,
        "C2": 134,
        "C3": 54,
        "C4": 158,
        "C5": 142,
    }
    assert all(len(names) == len(set(names)) for names in rows.values())
    assert all(name.startswith("C1|") for name in rows["C1"])
    assert all(name.startswith("C2|") for name in rows["C2"])
    assert all(name.startswith("C3|") for name in rows["C3"])
    assert all(name.startswith("C4|") for name in rows["C4"])
    assert all(name.startswith("C5|") for name in rows["C5"])
    paths = R5C.path_registry()
    assert tuple(paths) == R5C.FAMILY_NAMES
    assert tuple(paths["C3"]) == ("P_C3_L4H11", "P_C3_L4H9", "P_C3_L4H7")
    for family in R5C.FAMILY_NAMES:
        assert all(
            requirement.row in set(rows[family])
            for requirements in paths[family].values()
            for requirement in requirements
        )


def test_partner_pseudo_specific_and_all_gaps_are_derived_from_raw_legs():
    partner = _leg("partner")
    pseudo = _leg("pseudo")
    families = R5C.derive_family_matrices(partner, pseudo)
    assert {name: value.shape for name, value in families.items()} == {
        "C1": (6, 5),
        "C2": (134, 5),
        "C3": (54, 5),
        "C4": (158, 5),
        "C5": (142, 5),
    }
    rows = R5C.family_row_names()
    c3 = dict(zip(rows["C3"], families["C3"]))
    for estimand in R5C.SANDWICH_ESTIMANDS:
        partner_name = R5C._sandwich_row("C3", "L4H11", estimand, "partner")
        pseudo_name = R5C._sandwich_row("C3", "L4H11", estimand, "pseudo")
        specific_name = R5C._sandwich_row("C3", "L4H11", estimand, "specific")
        np.testing.assert_allclose(
            c3[specific_name], c3[partner_name] - c3[pseudo_name]
        )
        gap = R5C._gap_row("C3", "L4H11", "L4H0", estimand)
        np.testing.assert_allclose(c3[gap], 0.20)


def test_chunk_concatenation_replays_primitives_and_capture_identities():
    partner = _leg("partner")
    pseudo = _leg("pseudo")
    expected = R5C.derive_family_matrices(partner, pseudo)
    concatenated_partner = R5C.concatenate_leg_chunks(
        [_as_execution_chunk(partner)]
    )
    concatenated_pseudo = R5C.concatenate_leg_chunks([_as_execution_chunk(pseudo)])
    observed = R5C.derive_family_matrices(
        concatenated_partner, concatenated_pseudo
    )
    for family in R5C.FAMILY_NAMES:
        np.testing.assert_array_equal(observed[family], expected[family])


def test_derived_families_reject_caller_supplied_specific_rows_and_common_leg_drift():
    partner = _leg("partner")
    pseudo = _leg("pseudo")
    # There is deliberately nowhere to inject a caller-supplied "specific" vector.
    partner["specific"] = np.full(5, 999.0)
    families = R5C.derive_family_matrices(partner, pseudo)
    assert not np.any(families["C1"] == 999.0)

    broken = deepcopy(partner)
    first = next(iter(broken["sandwich"]))
    broken["sandwich"][first]["common_leg"]["m_minus_sha256"] = _sha("f")
    with pytest.raises(R5C.InstrumentError, match="common legs differ"):
        R5C.derive_family_matrices(broken, pseudo)

    stored_delivery_tamper = deepcopy(partner)
    component = next(iter(stored_delivery_tamper["sandwich"]))
    stored_delivery_tamper["sandwich"][component]["estimands"]["N_d"][0] = 99.0
    with pytest.raises(R5C.InstrumentError, match="primitive replay failed"):
        R5C.derive_family_matrices(stored_delivery_tamper, pseudo)

    stored_u_tamper = deepcopy(partner)
    stored_u_tamper["u_a"]["estimands"]["S_U"][0] = 99.0
    with pytest.raises(R5C.InstrumentError, match="primitive replay failed"):
        R5C.derive_family_matrices(stored_u_tamper, pseudo)


def test_common_leg_helper_and_gap_helper_fail_on_different_baselines():
    base = np.asarray([0.0, 1.0])
    rescued = np.asarray([0.5, 1.5])
    first = _sandwich(
        leg_id="partner",
        component="L4H11",
        m_minus=base,
        m_c_plus=rescued,
        effect=0.2,
    )
    second = _sandwich(
        leg_id="partner",
        component="L4H0",
        m_minus=base,
        m_c_plus=rescued,
        effect=0.0,
    )
    np.testing.assert_allclose(
        R5C.delivery_candidate_route_gaps(
            R5C._replay_delivery_envelope(first),
            R5C._replay_delivery_envelope(second),
        )[
            "N_candidate_minus_donor_route"
        ],
        0.2,
    )
    second["cells"]["m_minus"] = np.asarray([0.0, 1.1])
    with pytest.raises(R5C.InstrumentError, match="primitive replay failed"):
        R5C._replay_delivery_envelope(second)


def test_crossfitted_projectors_are_row_specific_symmetric_idempotent_and_rank_exact():
    projector = np.diag([1.0, 0.0, 0.0])
    projectors = np.repeat(projector[None, :, :], R5C.R5_ROWS_PER_BUCKET, axis=0)
    audit = R5C.validate_crossfit_projectors(projectors, selected_rank=1)
    assert audit["n_projectors"] == 128
    delta = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(
        R5C.row_project(delta, projectors[:2]), [[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    )
    broken = projectors.copy()
    broken[0, 0, 1] = 1.0
    with pytest.raises(R5C.InstrumentError, match="symmetry/idempotence/rank"):
        R5C.validate_crossfit_projectors(broken, selected_rank=1)
    with pytest.raises(R5C.InstrumentError, match="positive integer"):
        R5C.validate_crossfit_projectors(projectors, selected_rank=True)


def test_r5b_runtime_bundle_is_byte_array_shape_and_row_order_authenticated(tmp_path):
    rank = 2
    width = 4
    selected = np.repeat(np.eye(width)[:rank][None, :, :], 2, axis=0)
    random = np.repeat(selected[None, :, :, :], 32, axis=0)
    folds = np.asarray([index % 2 for index in range(R5C.R5_ROWS_PER_BUCKET)])
    row_ids = np.asarray(
        [f"example-{index:03d}" for index in range(R5C.R5_ROWS_PER_BUCKET)]
    )
    path = tmp_path / "r5b-runtime.npz"
    np.savez_compressed(
        path,
        selected_rank=np.asarray(rank),
        selected_bases_by_fold=selected,
        random_bases_by_draw_fold=random,
        row_fold_ids=folds,
        ordered_row_ids=row_ids,
    )
    contract = {
        "selected_rank": rank,
        "runtime_bundle_sha256": R5C.sha256_file(path),
        "runtime_bundle_size_bytes": path.stat().st_size,
        "runtime_array_sha256": {
            name: R5C.r5b_runtime_array_sha256(name, value)
            for name, value in {
                "selected_bases_by_fold": selected,
                "random_bases_by_draw_fold": random,
                "row_fold_ids": folds,
                "ordered_row_ids": row_ids,
            }.items()
        },
        "runtime_array_shapes": {
            "selected_bases_by_fold": [2, rank, width],
            "random_bases_by_draw_fold": [32, 2, rank, width],
            "row_fold_ids": [R5C.R5_ROWS_PER_BUCKET],
            "ordered_row_ids": [R5C.R5_ROWS_PER_BUCKET],
        },
        "ordered_row_ids_sha256": R5C.r5b_runtime_array_sha256(
            "ordered_row_ids", row_ids
        ),
        "crossfit_map_sha256": _sha("d"),
    }
    runtime = R5C.load_r5b_runtime_bundle(
        path,
        r5b_record=contract,
        ordered_row_ids=row_ids.tolist(),
    )
    assert runtime.selected_rank == rank
    assert runtime.bundle_sha256 == contract["runtime_bundle_sha256"]
    broken = deepcopy(contract)
    broken["runtime_array_sha256"]["row_fold_ids"] = _sha("e")
    with pytest.raises(R5C.InstrumentError, match="array hashes differ"):
        R5C.load_r5b_runtime_bundle(
            path,
            r5b_record=broken,
            ordered_row_ids=row_ids.tolist(),
        )


def test_r5b_runtime_contract_requires_both_distinct_bucket_bindings():
    entry = R5C.validate_entry_chain(_entry_chain())

    def contract(bucket):
        binding = R5C.expected_r5_binding(bucket)
        return {
            "schema": R5C.R5B_RUNTIME_CONTRACT_SCHEMA,
            "bucket": bucket,
            "cohort_sha256": binding["ordered_rows_sha256"],
            "construction_sha256": binding["construction_sha256"],
            "gene_ids_sha256": binding["gene_ids_sha256"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "selected_rank": entry["selected_rank"],
            R5C.STACK_FINGERPRINT_FIELD: entry[R5C.STACK_FINGERPRINT_FIELD],
            "crossfit_map_sha256": (
                entry["crossfit_map_sha256"] if bucket == 1 else _sha("e")
            ),
            "runtime_bundle_sha256": _sha("f"),
            "runtime_array_sha256": {
                "selected_bases_by_fold": _sha("1"),
                "random_bases_by_draw_fold": _sha("2"),
                "row_fold_ids": _sha("3"),
                "ordered_row_ids": _sha("4"),
            },
        }

    record = {"r5c_runtime_contract_by_bucket": {"0": contract(0), "1": contract(1)}}
    assert R5C.r5b_runtime_contract_for_bucket(
        record,
        bucket=0,
        cohort_binding=R5C.expected_r5_binding(0),
        entry_authorization=entry,
    )["bucket"] == 0
    with pytest.raises(R5C.InstrumentError, match="distinct bucket-0 and bucket-1"):
        R5C.r5b_runtime_contract_for_bucket(
            {"r5c_runtime_contract_by_bucket": {"1": contract(1)}},
            bucket=1,
            cohort_binding=R5C.expected_r5_binding(1),
            entry_authorization=entry,
        )


def test_query_patch_mask_selects_exactly_the_masked_acceptor_query_and_rejects_bool():
    mask = R5C._masked_query_token_mask([3, 5], sequence_tokens=9, token_offset=1)
    assert mask.tolist() == [
        [False, False, False, False, True, False, False, False, False],
        [False, False, False, False, False, False, True, False, False],
    ]
    with pytest.raises(TypeError, match="non-Boolean integers"):
        R5C._masked_query_token_mask([True], sequence_tokens=9, token_offset=1)


def test_corrected_boundary_tost_iut_classifier_supports_plant_and_rejects_touching_edge():
    names = R5C.family_row_names()["C1"]
    planted = np.zeros((len(names), R5C.MIN_GENES))
    index = {name: row for row, name in enumerate(names)}
    for endpoint in ("partner", "specific"):
        planted[index[R5C._direct_row(R5C.L3_READER, endpoint)]] = 0.20
    groups = [f"gene-{item}" for item in range(R5C.MIN_GENES)]
    result = R5C.strict_family_inference(
        "C1", planted, groups, n_draws=199, namespace="r5c-test-plant"
    )
    assert result["paths"]["P_C1_L3_DIRECT"]["classification"] == "SUPPORTED"
    assert result["boundary_randomisation_audit"][
        "all_decision_edges_share_one_sign_matrix"
    ]

    touching = planted.copy()
    touching[index[R5C._direct_row(R5C.L3_READER, "partner")]] = R5C.SESOI
    unresolved = R5C.strict_family_inference(
        "C1", touching, groups, n_draws=199, namespace="r5c-test-boundary"
    )
    assert unresolved["paths"]["P_C1_L3_DIRECT"]["classification"] != "SUPPORTED"


def test_three_candidate_paths_are_holm_corrected_as_one_singleton_family():
    names = R5C.family_row_names()["C3"]
    matrix = np.zeros((len(names), R5C.MIN_GENES))
    index = {name: row for row, name in enumerate(names)}
    for candidate in R5C.ACCEPTOR_CANDIDATES:
        identity = R5C.head_name(candidate)
        for endpoint in ("partner", "specific"):
            for estimand in R5C.SANDWICH_ESTIMANDS:
                matrix[index[R5C._sandwich_row("C3", identity, estimand, endpoint)]] = 0.20
        for route in R5C.DONOR_ROUTE_COMPARATORS:
            for estimand in R5C.SANDWICH_ESTIMANDS:
                matrix[
                    index[
                        R5C._gap_row(
                            "C3", identity, R5C.head_name(route), estimand
                        )
                    ]
                ] = 0.20
    groups = [f"gene-{item}" for item in range(R5C.MIN_GENES)]
    result = R5C.strict_family_inference(
        "C3", matrix, groups, n_draws=399, namespace="r5c-test-holm"
    )
    assert set(result["paths"]) == {"P_C3_L4H11", "P_C3_L4H9", "P_C3_L4H7"}
    assert all(
        path["holm_adjusted_p"] >= path["iut_p_value"]
        and path["classification"] == "SUPPORTED"
        for path in result["paths"].values()
    )


def test_entry_chain_distinguishes_scientific_stop_from_instrument_failure():
    entry = _entry_chain()
    assert R5C.validate_entry_chain(entry)["selected_rank"] == 4
    no_rank = deepcopy(entry)
    no_rank["r5b"]["selected_rank"] = None
    with pytest.raises(R5C.ScientificStop, match="selected no non-null"):
        R5C.validate_entry_chain(no_rank)
    donor_endpoint = deepcopy(entry)
    donor_endpoint["r5a"]["masked_donor_endpoint_constructed"] = True
    with pytest.raises(R5C.InstrumentError, match="forbidden masked-donor"):
        R5C.validate_entry_chain(donor_endpoint)
    bad_hash = deepcopy(entry)
    bad_hash["r5b"]["crossfit_map_sha256"] = "editable"
    with pytest.raises(R5C.InstrumentError, match="not literally bound"):
        R5C.validate_entry_chain(bad_hash)


def test_r3c_l3_method_disagreement_cannot_authorize_r5c():
    record = {
        "stage": "target",
        "status": "CONFIRMATORY",
        "verdict": "HELD_OUT_ACCEPTOR_HEAD_CONFIRMATION_COMPLETE",
        "instrument_revision": R5C.R3C_INSTRUMENT_REVISION,
        "all_gates_pass": True,
        "paths": {
            "L3H1_reused_component": {
                "classification": "INCONCLUSIVE_METHOD_DISAGREEMENT"
            }
        },
        "construction": {
            "all_single_acceptor_G": True,
            "policy": {"donor_readout_constructed": False},
        },
        "hashes": {"checkpoint_aggregate_sha256": _sha("a")},
    }
    contract = R5C._r3c_entry_contract(record)
    assert contract["instrument_gates_pass"] is True
    assert contract["claim_eligible"] is False

    entry = _entry_chain()
    entry["r3c"] = contract
    with pytest.raises(R5C.ScientificStop, match="r3c is not claim eligible"):
        R5C.validate_entry_chain(entry)


def test_capability_is_candidate_independent_and_missing_or_false_gate_fails():
    partner = _leg("partner", n=R5C.R5_ROWS_PER_BUCKET)
    pseudo = _leg("pseudo", n=R5C.R5_ROWS_PER_BUCKET)
    record, runtime_hashes = _capability_fixture(partner, pseudo)
    assert (
        R5C.validate_capability(
            record,
            partner_leg=partner,
            pseudo_leg=pseudo,
            expected_runtime_array_sha256=runtime_hashes,
        )["status"]
        == "R5C_CAPABILITY_CONFIRMED"
    )
    bad = deepcopy(record)
    bad["gates"]["dense_full_residual_replay"] = False
    with pytest.raises(R5C.InstrumentError, match="do not rederive"):
        R5C.validate_capability(
            bad,
            partner_leg=partner,
            pseudo_leg=pseudo,
            expected_runtime_array_sha256=runtime_hashes,
        )
    leaked = deepcopy(record)
    leaked["candidate_outcomes_used"] = True
    with pytest.raises(R5C.InstrumentError, match="do not rederive"):
        R5C.validate_capability(
            leaked,
            partner_leg=partner,
            pseudo_leg=pseudo,
            expected_runtime_array_sha256=runtime_hashes,
        )


def test_mutation_audit_rejects_wrong_token_and_extra_head():
    audit = _mutation_record("l3_c_plus", "L3H1", (R5C.L3_READER,), n=4)
    assert R5C.validate_head_mutation_audit(
        audit,
        expected_heads=(R5C.L3_READER,),
        expected_role="l3_c_plus",
        expected_component_id="L3H1",
    )["requested_token_indices"] == [1, 1, 1, 1]

    wrong_token = deepcopy(audit)
    wrong_token["requested_token_indices"][0] = 2
    wrong_token["audit_sha256"] = R5C.canonical_sha256(
        {key: value for key, value in wrong_token.items() if key != "audit_sha256"}
    )
    with pytest.raises(R5C.InstrumentError, match="token mask does not replay"):
        R5C.validate_head_mutation_audit(
            wrong_token, expected_heads=(R5C.L3_READER,)
        )

    extra_head = _mutation_record(
        "l3_c_plus", "L3H1", (R5C.L3_READER, (3, 2)), n=4
    )
    with pytest.raises(R5C.InstrumentError, match="wrong or extra head"):
        R5C.validate_head_mutation_audit(
            extra_head, expected_heads=(R5C.L3_READER,)
        )


def test_audited_patch_observes_the_actual_hook_before_and_after_tensors():
    module = R5C.torch.nn.Identity()
    adapter = SimpleNamespace(
        num_layers=1,
        num_heads=2,
        attention_self_modules=[module],
    )
    source = {0: R5C.torch.ones((2, 2, 3, 1))}
    target_heads = R5C.torch.zeros((2, 2, 3, 1))
    target_context = target_heads.permute(0, 2, 1, 3).reshape(2, 3, 2)
    mask = R5C.torch.zeros((2, 3), dtype=R5C.torch.bool)
    mask[:, 1] = True
    audit = {}
    with R5C.audited_patch_heads(
        adapter,
        source,
        ((0, 1),),
        token_mask=mask,
        intervention_id="test-hook",
        role="test",
        component_id="L0H1",
        audit_sink=audit,
    ):
        output = module(target_context)
    split = output.view(2, 3, 2, 1).permute(0, 2, 1, 3)
    assert R5C.torch.count_nonzero(split[:, 1, 1]).item() == 2
    assert R5C.torch.count_nonzero(split[:, 1, [0, 2]]).item() == 0
    R5C.validate_head_mutation_audit(
        audit,
        expected_heads=((0, 1),),
        expected_role="test",
        expected_component_id="L0H1",
    )


def test_capability_rejects_altered_projector_and_common_leg_capture_divergence():
    partner = _leg("partner", n=R5C.R5_ROWS_PER_BUCKET)
    pseudo = _leg("pseudo", n=R5C.R5_ROWS_PER_BUCKET)
    record, runtime_hashes = _capability_fixture(partner, pseudo)

    altered_projector = deepcopy(record)
    altered_projector["evidence"]["crossfit_projectors"]["selected"][
        "idempotence_max_abs"
    ] = 1.0
    with pytest.raises(R5C.InstrumentError, match="projector diagnostic"):
        R5C.validate_capability(
            altered_projector,
            partner_leg=partner,
            pseudo_leg=pseudo,
            expected_runtime_array_sha256=runtime_hashes,
        )

    divergent = deepcopy(partner)
    component = next(iter(divergent["sandwich"]))
    common = divergent["sandwich"][component]["common_leg"]
    common["m_minus_capture_sha256"] = _sha("f")
    common["common_leg_pair_sha256"] = R5C.canonical_sha256(
        {
            key: value
            for key, value in common.items()
            if key != "common_leg_pair_sha256"
        }
    )
    with pytest.raises(R5C.InstrumentError, match="common legs differ"):
        R5C.validate_capability(
            record,
            partner_leg=divergent,
            pseudo_leg=pseudo,
            expected_runtime_array_sha256=runtime_hashes,
        )


def test_calibration_registry_is_coherent_and_does_not_claim_impossible_isolated_faces():
    calibration = R5C.calibration_registry()
    assert calibration["engineering_not_scientific_fwer_evidence"] is True
    assert calibration["isolated_edge_maximum_claimed"] is False
    assert calibration["primitive_generator"].startswith("centered partner/pseudo")
    assert calibration["N_cal"] == R5C.computed_calibration_trials() == 386
    assert calibration["literal_edge_count"] == len(calibration["coherent_face_registry"])
    assert all(
        case["primitive_face_policy"].endswith("linked null edges permitted")
        for case in calibration["coherent_face_registry"]
    )


def test_diagnostic_records_are_replayed_and_tampering_fails_closed():
    binding, calibration = _passing_diagnostic_records()
    result = R5C.validate_binding_and_calibration_records(binding, calibration)
    assert result["status"] == "R5C_BINDING_AND_CALIBRATION_RECORDS_VALID"

    tampered = deepcopy(calibration)
    tampered["family_records"]["C3"]["false_positive_trials"] = 1
    with pytest.raises(R5C.InstrumentError, match="C3 calibration replay differs"):
        R5C.validate_binding_and_calibration_records(binding, tampered)


def test_constituent_binding_suite_reaches_every_face_through_production_inference():
    suite = R5C.run_constituent_binding_suite(n_draws=399)
    assert suite["status"] == "PASS"
    assert suite["literal_faces_checked"] == R5C.calibration_registry()[
        "literal_edge_count"
    ]
    assert suite["all_boundary_signatures_rejected"] is True
    assert suite["all_path_plants_supported"] is True
    assert suite["production_classifier_reused"] is True


def test_scientific_verdict_never_relables_healthy_negative_as_instrument_failure():
    base = dict(
        instrument_ok=True,
        entry_behavior="PASS",
        c1="SUPPORTED",
        c2="SUPPORTED",
        c3_paths={"P_C3_L4H11": "SUPPORTED"},
        c4="SUPPORTED",
        c5="SUPPORTED",
        controls_active=False,
        randoms_beaten=True,
    )
    assert R5C.scientific_verdict(**base).startswith("R5C_WRITE")
    assert R5C.scientific_verdict(**(base | {"controls_active": True})) == (
        "NONSELECTIVE_OR_SHARED_DELIVERY"
    )
    assert R5C.scientific_verdict(**(base | {"entry_behavior": "NOT_REPLICATED"})) == (
        "ENTRY_BEHAVIOR_NOT_REPLICATED"
    )
    assert R5C.scientific_verdict(**(base | {"instrument_ok": False})) == (
        "INSTRUMENT_UNRESOLVED"
    )
    assert R5C.scientific_verdict(
        **(base | {"c2": "SCIENTIFICALLY_UNRESOLVED_METHOD_DISAGREEMENT"})
    ) == "SCIENTIFICALLY_UNRESOLVED"


def test_model_free_audit_binds_shared_census_and_never_loads_data_or_model():
    audit = R5C.model_free_registry_audit()
    assert audit["status"] == "DRAFT_MODEL_FREE_REGISTRY_AUDIT_PASS"
    assert audit["model_loaded"] is False
    assert audit["annotation_read"] is False
    assert audit["fasta_read"] is False
    assert audit["cohort_read"] is False
    assert audit["chr8_chr9_opened"] is False
    assert audit["masked_donor_endpoint_constructed"] is False
    assert audit["shared_cohort"]["schema"] == "gfmcd.r5.acceptor_mechanism_cohort/2"
    assert audit["shared_cohort"]["expected_census_identity"][
        "available_genes_by_bucket"
    ] == {"0": 142, "1": 130}


def test_target_hard_stops_on_none_pins_before_user_paths_are_opened(tmp_path):
    sentinel = tmp_path / "must-not-be-created-or-read.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--stage",
            "target",
            "--r3c-artifact",
            str(sentinel),
            "--r5a-artifact",
            str(sentinel),
            "--r5b-artifact",
            str(sentinel),
            "--r5b-runtime",
            str(sentinel),
            "--preflight-artifact",
            str(sentinel),
            "--model",
            str(sentinel),
            "--gtf",
            str(sentinel),
            "--fasta",
            str(sentinel),
            "--out",
            str(sentinel),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "target is hard closed" in result.stderr
    assert "literal seals remain None" in result.stderr
    assert not sentinel.exists()


def test_preflight_hard_stops_on_none_pins_before_user_paths_are_opened(tmp_path):
    sentinel = tmp_path / "must-not-be-created-or-read.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--stage",
            "preflight",
            "--r3c-artifact",
            str(sentinel),
            "--r5a-artifact",
            str(sentinel),
            "--r5b-artifact",
            str(sentinel),
            "--r5b-runtime",
            str(sentinel),
            "--model",
            str(sentinel),
            "--gtf",
            str(sentinel),
            "--fasta",
            str(sentinel),
            "--out",
            str(sentinel),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "preflight is hard closed" in result.stderr
    assert "literal seals remain None" in result.stderr
    assert not sentinel.exists()


def test_authorized_cli_routes_to_the_end_to_end_model_executor(monkeypatch, tmp_path, capsys):
    authorization = {
        "FROZEN_PROTOCOL_SHA256": _sha("a"),
        "FROZEN_INSTRUMENT_FINGERPRINT_SHA256": _sha("b"),
        "FROZEN_R3C_ENTRY_ARTIFACT_SHA256": _sha("c"),
        "FROZEN_R5A_ENTRY_ARTIFACT_SHA256": _sha("d"),
        "FROZEN_R5B_ENTRY_ARTIFACT_SHA256": _sha("e"),
    }
    calls = []
    monkeypatch.setattr(R5C, "authorize_preflight", lambda: authorization)

    def execute(args, *, authorization):
        calls.append((args.stage, dict(authorization)))
        return (
            {"status": "PREFLIGHT_INSTRUMENT_PASS"},
            {"output": str(tmp_path / "result.json"), "result_sha256": _sha("f")},
        )

    monkeypatch.setattr(R5C, "run_authorized_model_stage", execute)
    R5C.main(
        [
            "--stage",
            "preflight",
            "--r3c-artifact",
            "r3c.json",
            "--r5a-artifact",
            "r5a.json",
            "--r5b-artifact",
            "r5b.json",
            "--r5b-runtime",
            "runtime.npz",
            "--out",
            str(tmp_path / "result.json"),
        ]
    )
    assert calls == [("preflight", authorization)]
    assert "PREFLIGHT_INSTRUMENT_PASS" in capsys.readouterr().out


def test_target_preflight_replay_rederives_families_from_primitive_legs():
    partner = _leg("partner", n=R5C.R5_ROWS_PER_BUCKET)
    pseudo = _leg("pseudo", n=R5C.R5_ROWS_PER_BUCKET)
    partner["native_conditions"] = {
        "clean_margin": np.full(R5C.R5_ROWS_PER_BUCKET, 0.30),
        "m_minus_margin": np.zeros(R5C.R5_ROWS_PER_BUCKET),
        "m_c_plus_margin": np.full(R5C.R5_ROWS_PER_BUCKET, 0.30),
    }
    pseudo["native_conditions"] = {
        "clean_margin": np.full(R5C.R5_ROWS_PER_BUCKET, 0.30),
        "m_minus_margin": np.full(R5C.R5_ROWS_PER_BUCKET, 0.30),
        "m_c_plus_margin": np.zeros(R5C.R5_ROWS_PER_BUCKET),
    }
    groups = [f"gene-{index:03d}" for index in range(R5C.R5_ROWS_PER_BUCKET)]
    matrices = R5C.derive_family_matrices(partner, pseudo)
    family_records = {}
    for family in R5C.FAMILY_NAMES:
        inference = R5C.strict_family_inference(
            family,
            matrices[family],
            groups,
            n_draws=R5C.PRODUCTION_RANDOMISATION,
            namespace=f"r5c-observed:{family}",
        )
        family_records[family] = {
            "row_names": list(R5C.family_row_names()[family]),
            "raw_values": matrices[family].tolist(),
            "raw_values_sha256": R5C.array_sha256(matrices[family]),
            "inference": inference,
            "inference_sha256": R5C.canonical_sha256(inference),
        }
    native = R5C.native_behavior_inference(
        np.full(R5C.R5_ROWS_PER_BUCKET, 0.30),
        np.zeros(R5C.R5_ROWS_PER_BUCKET),
        groups,
        n_draws=R5C.PRODUCTION_RANDOMISATION,
        seed=R5C.PRODUCTION_SEED,
        namespace="r5c-native-entry-behavior",
    )
    capability, runtime_hashes = _capability_fixture(partner, pseudo)
    measurement = {
        "groups": groups,
        "ordered_gene_ids_sha256": R5C.canonical_sha256(groups),
        "native_behavior": native,
        "partner_leg": partner,
        "pseudo_leg": pseudo,
        "families": family_records,
        "gene_sufficient_statistics": {
            "one_row_per_gene": True,
            "n_genes": R5C.R5_ROWS_PER_BUCKET,
            "raw_gene_values_stored_in_families": True,
            "family": {
                family: {
                    "row_sum": np.sum(matrices[family], axis=1).tolist(),
                    "row_sum_squares": np.sum(
                        np.square(matrices[family]), axis=1
                    ).tolist(),
                    "raw_values_sha256": R5C.array_sha256(matrices[family]),
                }
                for family in R5C.FAMILY_NAMES
            },
        },
        "capability": capability,
        "capability_status": {"status": "R5C_CAPABILITY_CONFIRMED"},
        "instrument_ok": True,
    }
    binding = R5C.instrument_binding_payload()
    fingerprint = R5C.canonical_sha256(binding)
    binding_suite, engineering_calibration = _passing_diagnostic_records()
    record = {
        "schema": R5C.SCHEMA,
        "stage": "preflight",
        "bucket": 0,
        "instrument_fingerprint_sha256": fingerprint,
        "instrument_binding": binding,
        R5C.STACK_BINDING_FIELD: binding[R5C.STACK_BINDING_FIELD],
        R5C.STACK_FINGERPRINT_FIELD: binding[R5C.STACK_FINGERPRINT_FIELD],
        "registry_audit": R5C.model_free_registry_audit(),
        "sealed_chr8_chr9_opened": False,
        "r5b_runtime_contract": {
            "runtime_array_sha256": runtime_hashes,
            R5C.STACK_FINGERPRINT_FIELD: binding[R5C.STACK_FINGERPRINT_FIELD],
        },
        "measurement": measurement,
        "binding_suite": binding_suite,
        "engineering_calibration": engineering_calibration,
    }
    replay = R5C.validate_preflight_replay(record, fingerprint=fingerprint)
    assert replay["primitive_family_rederivation_pass"] is True

    tampered = deepcopy(record)
    tampered_values = np.asarray(
        tampered["measurement"]["families"]["C1"]["raw_values"], dtype=float
    )
    tampered_values[0, 0] += 1.0
    tampered["measurement"]["families"]["C1"][
        "raw_values"
    ] = tampered_values.tolist()
    tampered["measurement"]["families"]["C1"][
        "raw_values_sha256"
    ] = R5C.array_sha256(tampered_values)
    with pytest.raises(R5C.InstrumentError, match="do not rederive"):
        R5C.validate_preflight_replay(tampered, fingerprint=fingerprint)


def test_normalized_runner_hash_changes_for_code_but_not_literal_pin_values():
    source = SCRIPT.read_text()
    modified_pin = source.replace(
        "FROZEN_PROTOCOL_SHA256: str | None = None",
        f'FROZEN_PROTOCOL_SHA256: str | None = "{_sha("e")}"',
    )
    assert R5C.normalized_runner_sha256(source) == R5C.normalized_runner_sha256(
        modified_pin
    )
    modified_code = source.replace("SESOI = 0.05", "SESOI = 0.051")
    assert R5C.normalized_runner_sha256(source) != R5C.normalized_runner_sha256(
        modified_code
    )


def test_audit_cli_writes_a_fresh_model_free_artifact_and_manifest(tmp_path):
    output = tmp_path / "r5c-audit.json"
    manifest_path = output.with_suffix(".manifest.json")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--stage", "audit", "--out", str(output)],
        text=True,
        capture_output=True,
        check=True,
    )
    record = json.loads(output.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert record["schema"] == R5C.AUDIT_SCHEMA
    assert record["instrument_binding_payload"]["family_alpha"] == pytest.approx(0.01)
    assert (
        R5C.r5_stack_fingerprint(
            record["instrument_binding_payload"][R5C.STACK_BINDING_FIELD]
        )
        == record["instrument_binding_payload"][R5C.STACK_FINGERPRINT_FIELD]
    )
    assert manifest["config"]["model_loaded"] is False
    assert manifest["outputs"][0]["sha256"] == R5C.sha256_file(output)
    second = subprocess.run(
        [sys.executable, str(SCRIPT), "--stage", "audit", "--out", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode != 0
    assert "outputs are immutable" in second.stderr
