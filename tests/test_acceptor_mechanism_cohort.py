from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from gfmcd.splicing import acceptor_mechanism_cohort as cohort_module
from gfmcd.splicing.acceptor_mechanism_cohort import (
    R1C_CELL_NAMES,
    R1C_CLAIM_BOUNDARY,
    R1C_COHORT_ORDER_VERSION,
    R1C_CONFIRMED_BEHAVIOR_VERDICTS,
    R1C_CONSTRUCTION_VERSION,
    R1C_DYAD_DISTANCE_CALIPER,
    R1C_MATCHED_DECOY_DONOR_CALIPER,
    R1C_PRIMARY_FAMILY_NAMES,
    R1C_TARGET_ARTIFACT_SHA256,
    R1C_TARGET_ATTEMPT_SHA256,
    R1C_TARGET_COMPLETION_SHA256,
    R1C_TARGET_MANIFEST_SHA256,
    R1C_TARGET_SCHEMA,
    R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
    R5_CAPSTONE_ROW_MATERIALIZATION_SCHEMA,
    R5_VERIFIED_R1C_COMMON_LEGS_SCHEMA,
    R5_COHORT_SCHEMA,
    R5_EXPECTED_CENSUS_IDENTITY,
    R5_EXPECTED_SOURCE_IDENTITY,
    R5_RESIDUAL_VALIDATION_COHORT_ROLE,
    R5_RESIDUAL_VALIDATION_ROWS_PER_BUCKET,
    R5_RESIDUAL_VALIDATION_STAGE_CONSUMERS,
    R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
    R5_PREFLIGHT_ROW_MATERIALIZATION_SCHEMA,
    R5A_R5C_PREFLIGHT_COHORT_ROLE,
    R5_ROWS_PER_BUCKET,
    R5C_REQUIRED_COHORT_ROLE,
    R5CapstoneCohortError,
    build_r5_reservations,
    build_r1c_verified_target_common_legs,
    build_r5_capstone_cohort_handoff,
    build_r5_preflight_cohort_handoff,
    canonical_sha256,
    deterministic_adjacent_window,
    evaluate_r5_row,
    r5_gene_bucket,
    r5_gene_priority,
    r5_row_priority,
    load_r1c_capstone_cohort_handoff,
    load_r1c_preflight_cohort_handoff,
    load_r1c_verified_target_common_legs,
    materialize_r5_capstone_rows,
    materialize_r5_preflight_rows,
    validate_expected_r5_source_identity,
    validate_r5_reservation_audit,
    validate_r5_capstone_cohort_handoff,
    validate_r5_preflight_cohort_handoff,
)
from gfmcd.splicing.coupling import CouplingExample
from gfmcd.splicing.r5_stack_binding import SHARED_R1C_PACKAGE_PIN_NAMES


REPO = Path(__file__).resolve().parents[1]


def _example(index: int, *, gene: str | None = None, strand: str = "+") -> CouplingExample:
    sequence = list("C" * 120)
    donor = 40
    pseudo = 50
    acceptor = 90
    sequence[donor : donor + 2] = "GT"
    sequence[pseudo : pseudo + 2] = "GT"
    sequence[acceptor - 2 : acceptor] = "AG"
    return CouplingExample(
        example_id=f"example-{index}",
        sequence="".join(sequence),
        acceptor_pos=20,
        donor_pos=donor,
        chrom="chr11",
        strand=strand,
        gene_id=gene or f"gene-{index}",
        split="val",
        exon_length=20,
        intron_length_down=50,
        n_isoforms_donor=1,
        n_isoforms_acceptor=1,
        gene_type="protein_coding",
        other_gt=(pseudo,),
        next_acceptor_pos=acceptor,
    )


def test_adjacent_window_and_three_acceptor_only_sequence_states_are_exact():
    example = _example(0)
    adjacent = deterministic_adjacent_window(
        example_id=example.example_id,
        sequence=example.sequence,
        paired_donor=example.donor_pos,
        pseudo_donor=50,
        acceptor_g=89,
    )
    assert adjacent is not None
    assert len(adjacent) == 4
    assert max(abs(position - example.donor_pos) for position in adjacent) <= 34
    row, reason = evaluate_r5_row(example, 50, excluded_gene_ids=frozenset())
    assert reason is None and row is not None
    assert row.clean_sequence[89] == "G"
    assert row.paired_corrupt_sequence[40:42] == "AC"
    assert row.paired_corrupt_sequence[50:52] == "GT"
    assert row.pseudo_corrupt_sequence[40:42] == "GT"
    assert row.pseudo_corrupt_sequence[50:52] == "AC"
    assert row.identity_record().keys().isdisjoint({"donor_readout", "masked_donor"})


def test_eligibility_fails_closed_on_exclusion_caliper_motif_and_adjacent_control():
    example = _example(1)
    row, reason = evaluate_r5_row(
        example, 50, excluded_gene_ids=frozenset({example.gene_id})
    )
    assert row is None and reason == "excluded_gene"
    row, reason = evaluate_r5_row(example, 100, excluded_gene_ids=frozenset())
    assert row is None and reason == "pseudo_donor_outside_40nt_caliper"
    no_pseudo = replace(example, sequence=example.sequence[:50] + "CC" + example.sequence[52:])
    row, reason = evaluate_r5_row(no_pseudo, 50, excluded_gene_ids=frozenset())
    assert row is None and reason == "pseudo_donor_not_GT"

    blocked = list(example.sequence)
    # Put motifs throughout every padded shell on both sides.
    for start in range(6, 72, 2):
        blocked[start : start + 2] = "AG" if start % 4 else "GT"
    blocked[example.donor_pos : example.donor_pos + 2] = "GT"
    blocked[50:52] = "GT"
    blocked[88:90] = "AG"
    row, reason = evaluate_r5_row(
        replace(example, sequence="".join(blocked)),
        50,
        excluded_gene_ids=frozenset(),
    )
    assert row is None and reason == "missing_admissible_adjacent_window"


def test_reservations_choose_one_row_per_gene_and_bind_exact_constructions():
    examples = []
    pseudo = {}
    for index in range(80):
        for duplicate in range(2):
            item = _example(
                index * 2 + duplicate,
                gene=f"gene-{index}",
                strand="+" if index % 2 == 0 else "-",
            )
            examples.append(item)
            pseudo[item.example_id] = 50
    selected, audit = build_r5_reservations(
        examples,
        pseudo,
        historical_r3_gene_ids=["gene-excluded-old"],
        r3c_gene_ids=["gene-excluded-current"],
        r3d_registry_gene_ids=["gene-r3d-registry"],
        r3d_target_gene_ids=["gene-r3d-target"],
        rows_per_bucket=10,
    )
    for bucket in (0, 1):
        assert len(selected[bucket]) == 10
        genes = [row.example.gene_id for row in selected[bucket]]
        assert len(set(genes)) == 10
        assert genes == sorted(genes, key=r5_gene_priority)
        for row in selected[bucket]:
            siblings = [
                example
                for example in examples
                if example.gene_id == row.example.gene_id
            ]
            assert row.example.example_id == min(
                (example.example_id for example in siblings), key=r5_row_priority
            )
            assert r5_gene_bucket(row.example.gene_id) == bucket
    validate_r5_reservation_audit(audit, rows_per_bucket=10)
    assert audit["selection_reads_model_outputs"] is False
    assert audit["sealed_chr8_chr9_opened"] is False
    assert audit["schema"] == R5_COHORT_SCHEMA == (
        "gfmcd.r5.acceptor_mechanism_cohort/3"
    )
    assert audit["cohort_role"] == R5_RESIDUAL_VALIDATION_COHORT_ROLE
    assert audit["authorized_stage_consumers"] == list(
        R5_RESIDUAL_VALIDATION_STAGE_CONSUMERS
    )
    assert R5_RESIDUAL_VALIDATION_STAGE_CONSUMERS == (
        "R5b",
    )
    assert audit["bucket_roles"] == {
        "0": "R5b_derivation",
        "1": "R5b_validation_and_union_refit",
    }
    assert audit["r5a_r5c_preflight_authorized"] is False
    assert audit["r5c_capstone_authorized"] is False
    assert audit["r5c_required_cohort_role"] == R5C_REQUIRED_COHORT_ROLE
    with pytest.raises(ValueError, match="cannot authorize"):
        validate_r5_reservation_audit(
            audit | {"r5c_capstone_authorized": True}, rows_per_bucket=10
        )
    assert audit["prior_exposure_sets_pairwise_disjoint"] is True
    assert audit["r3d_validation_row_overlap_by_source"] == {
        "r3d_direct_exposure_registry": 0,
        "r3d_target": 0,
    }
    assert audit["global_gene_freshness_claimed"] is False
    assert audit["eligible_rows"] == 160
    assert audit["eligible_genes"] == 80


def test_capacity_and_two_strand_gates_cannot_be_self_attested():
    examples = [_example(index, strand="+") for index in range(12)]
    pseudo = {example.example_id: 50 for example in examples}
    with pytest.raises(RuntimeError, match="capacity is insufficient"):
        build_r5_reservations(
            examples,
            pseudo,
            historical_r3_gene_ids=[],
            r3c_gene_ids=[],
            rows_per_bucket=R5_ROWS_PER_BUCKET,
        )
    _, audit = build_r5_reservations(
        examples,
        pseudo,
        historical_r3_gene_ids=[],
        r3c_gene_ids=[],
        rows_per_bucket=2,
        require_capacity=False,
    )
    with pytest.raises(ValueError, match="both strands"):
        validate_r5_reservation_audit(audit, rows_per_bucket=2)


def test_expected_full_census_identity_is_literal_and_capacity_is_not_rounded_up():
    expected = R5_EXPECTED_CENSUS_IDENTITY
    assert R5_ROWS_PER_BUCKET == R5_RESIDUAL_VALIDATION_ROWS_PER_BUCKET == 128
    assert expected["r5c_capstone_authorized"] is False
    assert expected["input_rows"] == 6064
    assert expected["eligible_rows"] == 366
    assert expected["eligible_genes"] == 272
    assert expected["available_genes_by_bucket"] == {"0": 142, "1": 130}
    assert expected["r3d_direct_exposure_registry_gene_count"] == 1112
    assert expected["r3d_target_gene_count"] == 3909
    assert expected["prior_exposure_union_gene_count"] == 6254
    assert len(expected["prior_exposure_pairwise_intersection_counts"]) == 6
    assert not any(expected["prior_exposure_pairwise_intersection_counts"].values())
    assert expected["prior_exposure_union_gene_ids_sha256"] == (
        "14f2cbb7d1aa4764da9825a481bdb1f2002e2d47022b0292f809494e163e9ed3"
    )
    assert expected["r3d_registry_and_target_union_gene_ids_sha256"] == (
        "2d64dab36515e9d6344f73164a91c668e15d50f15003cfdd3a5a4ecc87e9fd13"
    )
    assert expected["r3d_validation_row_overlap_by_source"] == {
        "r3d_direct_exposure_registry": 0,
        "r3d_target": 0,
    }
    assert expected["selected_by_bucket"]["0"]["rows"] == 128
    assert expected["selected_by_bucket"]["1"]["rows"] == 128
    assert expected["selected_by_bucket"]["1"]["gene_ids_sha256"].startswith(
        "e6c30992"
    )


def test_source_identity_pins_comprehensive_annotation_and_rejects_other_bytes(
    tmp_path,
):
    assert R5_EXPECTED_SOURCE_IDENTITY["gtf"] == {
        "label": "GENCODE_v50_comprehensive",
        "size_bytes": 124_527_720,
        "sha256": "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453",
    }
    assert R5_EXPECTED_SOURCE_IDENTITY["fasta_fai"] == {
        "label": "gfmcd_local_hg38.fa.fai@local_GRCh38_reference_v1",
        "size_bytes": 19_381,
        "sha256": "3b425de206296a5c8053023fa5ca61da43cfe78c1737c12e58c83367c7e83c21",
    }
    gtf = tmp_path / "annotation.gtf.gz"
    fasta = tmp_path / "reference.fa"
    gtf.write_bytes(b"not the pinned comprehensive annotation")
    fasta.write_bytes(b">chr11\nACGT\n")
    fasta.with_suffix(".fa.fai").write_bytes(b"not the pinned index")
    with pytest.raises(RuntimeError, match="source identity drifted"):
        validate_expected_r5_source_identity(gtf, fasta)


def test_r3d_exposure_sources_are_forbidden_from_overlapping_validation_rows():
    example = _example(500, gene="leaked-r3d-gene")
    with pytest.raises(RuntimeError, match="R3d exposure.*validation"):
        build_r5_reservations(
            [example],
            {example.example_id: 50},
            historical_r3_gene_ids=[],
            r3c_gene_ids=[],
            r3d_registry_gene_ids=[example.gene_id],
            r3d_target_gene_ids=[],
            rows_per_bucket=1,
            require_capacity=False,
        )


def _r1c_target_fixture(*, one_chromosome: bool = False):
    rows = []
    for index in range(400):
        rows.append(
            {
                "example_id": f"sealed-example-{index:03d}",
                "gene_id": f"sealed-gene-{index % 137:03d}",
                "chrom": (
                    "chr8"
                    if one_chromosome or index % 2 == 0
                    else "chr9"
                ),
                "strand": "+" if index % 3 else "-",
                "positions": {
                    "prior_real_acceptor": 100,
                    "paired_donor": 200,
                    "partner_acceptor": 300,
                    "matched_decoy_donor": 220,
                    "matched_decoy_donor_distance_mismatch": 20,
                    "dyad_distance_mismatch": index % 21,
                },
                # Real R1c rows contain outcomes too; the handoff must discard them.
                "margin_levels": {
                    "P_clean": float(index),
                    "P_under_D": float(index) - 0.2,
                    "P_under_N": float(index) - 0.1,
                },
                "family": {name: 0.0 for name in R1C_PRIMARY_FAMILY_NAMES},
                "cell_effects": {name: 0.0 for name in R1C_CELL_NAMES},
            }
        )
    source_cohort = [
        {
            "example_id": row["example_id"],
            "gene_id": row["gene_id"],
            "decoy_donor": row["positions"]["matched_decoy_donor"],
            "decoy_donor_distance_mismatch": row["positions"][
                "matched_decoy_donor_distance_mismatch"
            ],
            "dyad_distance_mismatch": row["positions"]["dyad_distance_mismatch"],
        }
        for row in rows
    ]
    ordered_ids = [row["example_id"] for row in rows]
    ordered_gene_ids = [row["gene_id"] for row in rows]
    six_position_rows = [
        {
            "example_id": row["example_id"],
            "gene_id": row["gene_id"],
            "chrom": row["chrom"],
            "strand": row["strand"],
            "positions": row["positions"],
        }
        for row in rows
    ]
    target_cohort_identity = {
        "source_cohort_sha256": canonical_sha256(source_cohort),
        "ordered_example_ids_sha256": canonical_sha256(ordered_ids),
        "target_gene_ids_sha256": canonical_sha256(sorted(set(ordered_gene_ids))),
        "ordered_gene_ids_sha256": canonical_sha256(ordered_gene_ids),
        "ordered_six_position_rows_sha256": canonical_sha256(six_position_rows),
        "construction_sha256": canonical_sha256(
            {
                "construction_version": R1C_CONSTRUCTION_VERSION,
                "cohort_order_version": R1C_COHORT_ORDER_VERSION,
                "ordered_six_position_rows": six_position_rows,
            }
        ),
        "ordered_partner_margin_levels_sha256": canonical_sha256(
            [
                {
                    "example_id": row["example_id"],
                    "gene_id": row["gene_id"],
                    "margin_levels": row["margin_levels"],
                }
                for row in rows
            ]
        ),
    }
    target_identity_payload = {
        "schema": "gfmcd.r1c.biological_target_identity/1",
        "program": "r1c-sealed-reverse-behavior-v2",
        "stage": "R1c",
        "holdout_id": "r1c-canonical-internal-short-intron-chr8-chr9-v1",
        "target_chromosomes": ["chr8", "chr9"],
        "target_n": 400,
        "checkpoint_sha256": "d" * 64,
        "gtf_sha256": "7" * 64,
        "fasta_sha256": "8" * 64,
        "fasta_fai_sha256": "9" * 64,
    }
    target_identity = target_identity_payload | {
        "biological_target_key": canonical_sha256(target_identity_payload)
    }
    artifact = {
        "logical_name": "r1c_target.json",
        "size_bytes": 1234,
        "sha256": "a" * 64,
    }
    manifest_identity = {
        "logical_name": "r1c_target.manifest.json",
        "size_bytes": 4321,
        "sha256": "b" * 64,
    }
    attempt_identity = {
        "logical_name": "r1c_target_attempt.json",
        "size_bytes": 987,
        "sha256": "c" * 64,
    }
    completion_identity = {
        "logical_name": "r1c_target_attempt.completion.json",
        "size_bytes": 654,
        "sha256": "f" * 64,
    }
    attempt_payload = {
        "schema": "gfmcd.r1c.target_attempt/2",
        "created_at_utc": "2026-08-16T12:00:00+00:00",
        "status": "AUTHORIZED_SEALED_OUTCOMES_MAY_BE_SPENT",
        "biological_target_key": target_identity["biological_target_key"],
        "target_identity": target_identity,
        "method_context": {
            "protocol_sha256": "1" * 64,
            "instrument_fingerprint_sha256": "2" * 64,
            "r5_stack_fingerprint_sha256": "3" * 64,
            "checkpoint_bundle_sha256": target_identity["checkpoint_sha256"],
            "construction_contract_sha256": "0" * 64,
            "endpoint_policy": (
                "single masked acceptor G hard margin; paired versus "
                "distance-matched donor GT->AC 2x2"
            ),
            "preflight_artifact": {
                "logical_name": "r1c_preflight.json",
                "size_bytes": 111,
                "sha256": "4" * 64,
            },
            "preflight_manifest": {
                "logical_name": "r1c_preflight.manifest.json",
                "size_bytes": 222,
                "sha256": "5" * 64,
            },
            "input_snapshot_sha256": "6" * 64,
            "requested_artifact_logical_name": artifact["logical_name"],
            "requested_manifest_logical_name": manifest_identity["logical_name"],
            "declared_completion_logical_name": completion_identity[
                "logical_name"
            ],
        },
        "policy": "create-only biological target receipt",
        "sealed_cohort_constructed_before_record": False,
        "model_loaded_before_record": False,
        "output_inspected_before_record": False,
    }
    attempt = attempt_payload | {"attempt_id": canonical_sha256(attempt_payload)}
    record = {
        "schema": R1C_TARGET_SCHEMA,
        "stage": "target",
        "split": "test",
        "n": 400,
        "sealed_split_opened": True,
        "preconstruction_allowed_chromosomes": ["chr8", "chr9"],
        "all_gates_pass": True,
        "gates": {
            "construction": True,
            "health": True,
            "inference": True,
            "raw_trimmed_gene_robustness": True,
        },
        "verdict": R1C_CONFIRMED_BEHAVIOR_VERDICTS[0],
        "instrument_fingerprint": {
            "fingerprint_sha256": "2" * 64,
            "checkpoint": {"bundle_sha256": target_identity["checkpoint_sha256"]},
            "gtf": {"sha256": target_identity["gtf_sha256"]},
            "fasta": {"sha256": target_identity["fasta_sha256"]},
            "fasta_fai": {"sha256": target_identity["fasta_fai_sha256"]},
        },
        "r5_stack_fingerprint_sha256": "3" * 64,
        "observed_decision_paths": {
            "core_paired_donor": True,
            "stronger_downstream_preference": False,
        },
        "boundary_inference": {
            "path_tests": {
                "core_paired_donor": {"decision": "DECLARED"},
                "stronger_downstream_preference": {
                    "decision": "NOT_DECLARED"
                },
            }
        },
        "gene_robustness": {
            "schema": "gfmcd.r1c.equal_gene_robustness/1",
            "randomisation_unit": "unweighted gene",
            "trim_fraction": 0.10,
            "n_draws": 10_000,
            "minimum_draws": 10_000,
            "all_required_robust": True,
            "paths": {
                "core_paired_donor": {"pass": True},
                "stronger_downstream_preference": {"pass": False},
            },
            "endpoints": {
                name: {
                    "joint_boundary_pass": name != "I",
                    "trim_fraction": 0.10,
                    "raw": {"estimator": "unweighted_gene_mean"},
                    "trimmed": {
                        "estimator": "10%_trimmed_unweighted_gene_mean"
                    },
                }
                for name in ("Y_PD", "S_P", "I")
            }
            | {
                "Y_PN": {
                    "equivalence_pass": True,
                    "lower_edge": {
                        "joint_boundary_pass": True,
                        "trim_fraction": 0.10,
                        "raw": {"estimator": "unweighted_gene_mean"},
                        "trimmed": {
                            "estimator": "10%_trimmed_unweighted_gene_mean"
                        },
                    },
                    "upper_edge": {
                        "joint_boundary_pass": True,
                        "trim_fraction": 0.10,
                        "raw": {"estimator": "unweighted_gene_mean"},
                        "trimmed": {
                            "estimator": "10%_trimmed_unweighted_gene_mean"
                        },
                    },
                }
            },
        },
        "config": {
            "single_G_acceptor_endpoint": True,
            "outcome_conditioning": None,
            "construction_version": R1C_CONSTRUCTION_VERSION,
            "cohort_order_version": R1C_COHORT_ORDER_VERSION,
            "matched_decoy_donor_caliper": R1C_MATCHED_DECOY_DONOR_CALIPER,
            "dyad_distance_caliper": R1C_DYAD_DISTANCE_CALIPER,
            "primary_family_names": list(R1C_PRIMARY_FAMILY_NAMES),
            "cell_names": list(R1C_CELL_NAMES),
        },
        "claim_boundary": R1C_CLAIM_BOUNDARY,
        "cohort_audit": {
            "selected": 400,
            "selection_reads_model_outputs": False,
            "selected_cohort_sha256": canonical_sha256(source_cohort),
            "selected_example_ids_sha256": canonical_sha256(ordered_ids),
        },
        "cohort_sha256": canonical_sha256(source_cohort),
        "target_cohort_identity": target_cohort_identity,
        "target_attempt": {
            "logical_name": attempt_identity["logical_name"],
            "size_bytes": attempt_identity["size_bytes"],
            "sha256": attempt_identity["sha256"],
            "attempt_id": attempt["attempt_id"],
            "biological_target_key": target_identity["biological_target_key"],
            "status": attempt["status"],
        },
        "target_completion": {"logical_name": completion_identity["logical_name"]},
        "rows": rows,
    }
    manifest = {
        "git_dirty": False,
        "inputs": [
            {
                "path": f"/historical/ledger/{attempt_identity['logical_name']}",
                "size_bytes": attempt_identity["size_bytes"],
                "sha256": attempt_identity["sha256"],
            }
        ],
        "outputs": [
            {
                "path": "/historical/worktree/r1c_target.json",
                "size_bytes": artifact["size_bytes"],
                "sha256": artifact["sha256"],
            }
        ],
    }
    completion_payload = {
        "schema": "gfmcd.r1c.target_completion/1",
        "created_at_utc": "2026-08-16T12:30:00+00:00",
        "status": "TARGET_PUBLICATION_COMPLETE",
        "attempt_id": attempt["attempt_id"],
        "biological_target_key": target_identity["biological_target_key"],
        "method_fingerprint_sha256": "2" * 64,
        "r5_stack_fingerprint_sha256": "3" * 64,
        "cohort_digests": target_cohort_identity,
        "artifact": artifact,
        "manifest": manifest_identity,
        "attempt": attempt_identity,
        "official_verdict": record["verdict"],
        "all_gates_pass": True,
        "gates": record["gates"],
    }
    completion = completion_payload | {
        "completion_id": canonical_sha256(completion_payload)
    }
    return (
        record,
        manifest,
        attempt,
        completion,
        artifact,
        manifest_identity,
        attempt_identity,
        completion_identity,
    )


def _r1c_target_successor_kwargs(
    record,
    attempt,
    completion,
    artifact,
    manifest_identity,
    attempt_identity,
    completion_identity,
):
    preflight_artifact = attempt["method_context"]["preflight_artifact"]
    preflight_manifest = attempt["method_context"]["preflight_manifest"]
    attestation = {
        "schema": "gfmcd.r1c.verified_target_handoff/1",
        "attempt_id": attempt["attempt_id"],
        "completion_id": completion["completion_id"],
        "biological_target_key": attempt["biological_target_key"],
        "instrument_fingerprint_sha256": attempt["method_context"][
            "instrument_fingerprint_sha256"
        ],
        "r5_stack_fingerprint_sha256": attempt["method_context"][
            "r5_stack_fingerprint_sha256"
        ],
        "target_cohort_identity": record["target_cohort_identity"],
        "observed_decision_paths": record["observed_decision_paths"],
        "verdict": record["verdict"],
        "all_gates_pass": record["all_gates_pass"],
        "gates": record["gates"],
        "package_identities": {
            "artifact": artifact,
            "manifest": manifest_identity,
            "attempt": attempt_identity,
            "completion": completion_identity,
        },
        "preflight_package": {
            "artifact": preflight_artifact,
            "manifest": preflight_manifest,
        },
    }
    return {
        "preflight_artifact_identity": preflight_artifact,
        "preflight_manifest_identity": preflight_manifest,
        "successor_attestation": attestation
        | {"attestation_sha256": canonical_sha256(attestation)},
    }


def _r1c_preflight_fixture():
    (
        target,
        _target_manifest,
        _attempt,
        _completion,
        _artifact,
        _manifest_identity,
        _attempt_identity,
        _completion_identity,
    ) = _r1c_target_fixture()
    record = copy.deepcopy(target)
    for index, row in enumerate(record["rows"]):
        row["example_id"] = f"validation-example-{index:03d}"
        row["gene_id"] = f"validation-gene-{index % 137:03d}"
        row["chrom"] = ("chr11", "chr16", "chr19")[index % 3]
    rows = record["rows"]
    source_cohort = [
        {
            "example_id": row["example_id"],
            "gene_id": row["gene_id"],
            "decoy_donor": row["positions"]["matched_decoy_donor"],
            "decoy_donor_distance_mismatch": row["positions"][
                "matched_decoy_donor_distance_mismatch"
            ],
            "dyad_distance_mismatch": row["positions"]["dyad_distance_mismatch"],
        }
        for row in rows
    ]
    identities = [
        {
            "example_id": row["example_id"],
            "gene_id": row["gene_id"],
            "chrom": row["chrom"],
            "strand": row["strand"],
            "positions": row["positions"],
        }
        for row in rows
    ]
    ordered_ids = [row["example_id"] for row in rows]
    ordered_genes = [row["gene_id"] for row in rows]
    target_cohort_identity = {
        "source_cohort_sha256": canonical_sha256(source_cohort),
        "ordered_example_ids_sha256": canonical_sha256(ordered_ids),
        "target_gene_ids_sha256": canonical_sha256(sorted(set(ordered_genes))),
        "ordered_gene_ids_sha256": canonical_sha256(ordered_genes),
        "ordered_six_position_rows_sha256": canonical_sha256(identities),
        "construction_sha256": canonical_sha256(
            {
                "construction_version": R1C_CONSTRUCTION_VERSION,
                "cohort_order_version": R1C_COHORT_ORDER_VERSION,
                "ordered_six_position_rows": identities,
            }
        ),
        "ordered_partner_margin_levels_sha256": canonical_sha256(
            [
                {
                    "example_id": row["example_id"],
                    "gene_id": row["gene_id"],
                    "margin_levels": row["margin_levels"],
                }
                for row in rows
            ]
        ),
    }
    record.update(
        {
            "experiment": "R1c_corrected_reverse_behavior",
            "stage": "preflight",
            "status": "DEVELOPMENT",
            "evidence_rung": "BEHAVIOR_ONLY",
            "split": "val",
            "sealed_split_opened": False,
            "n_genes": 137,
            "preconstruction_allowed_chromosomes": ["chr11", "chr16", "chr19"],
            "verdict": "PREFLIGHT_GO",
            "cohort_sha256": canonical_sha256(source_cohort),
            "target_cohort_identity": target_cohort_identity,
            "target_attempt": None,
            "target_completion": None,
            "target_attempt_disposition": None,
            "instrument_fingerprint": {
                "fingerprint_sha256": "2" * 64,
                "checkpoint": {"bundle_sha256": "d" * 64},
                "gtf": {
                    "sha256": R5_EXPECTED_SOURCE_IDENTITY["gtf"]["sha256"]
                },
                "fasta": {
                    "sha256": R5_EXPECTED_SOURCE_IDENTITY["fasta"]["sha256"]
                },
                "fasta_fai": {
                    "sha256": R5_EXPECTED_SOURCE_IDENTITY["fasta_fai"]["sha256"]
                },
            },
            "cohort_audit": {
                "selected": 400,
                "selected_genes": 137,
                "selection_reads_model_outputs": False,
                "selected_cohort_sha256": canonical_sha256(source_cohort),
                "selected_example_ids_sha256": canonical_sha256(ordered_ids),
            },
        }
    )
    artifact = {
        "logical_name": "r1c_preflight.json",
        "size_bytes": 111,
        "sha256": "4" * 64,
    }
    manifest_identity = {
        "logical_name": "r1c_preflight.manifest.json",
        "size_bytes": 222,
        "sha256": "5" * 64,
    }
    manifest = {
        "git_dirty": False,
        "config": {"stage": "preflight"},
        "outputs": [
            {
                "path": f"/relocated/{artifact['logical_name']}",
                "size_bytes": artifact["size_bytes"],
                "sha256": artifact["sha256"],
            }
        ],
    }
    attestation = {
        "schema": "gfmcd.r1c.verified_preflight_handoff/1",
        "artifact": artifact,
        "manifest": manifest_identity,
        "instrument_fingerprint_sha256": "2" * 64,
        "r5_stack_fingerprint_sha256": "3" * 64,
        "n": 400,
        "n_genes": 137,
        "target_cohort_identity": target_cohort_identity,
        "verdict": "PREFLIGHT_GO",
        "all_gates_pass": True,
        "gates": record["gates"],
    }
    return (
        record,
        manifest,
        artifact,
        manifest_identity,
        attestation | {"attestation_sha256": canonical_sha256(attestation)},
    )


def test_r1c_preflight_handoff_preserves_only_common_leg_and_never_scientific_entry():
    record, manifest, artifact, manifest_identity, attestation = (
        _r1c_preflight_fixture()
    )
    handoff = build_r5_preflight_cohort_handoff(
        record,
        manifest,
        artifact_identity=artifact,
        manifest_identity=manifest_identity,
        successor_attestation=attestation,
    )
    assert handoff["schema"] == R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA
    assert handoff["cohort_role"] == R5A_R5C_PREFLIGHT_COHORT_ROLE
    assert handoff["scientific_entry_authorized"] is False
    assert handoff["sealed_split_opened"] is False
    assert handoff["residual_validation_cohort_used"] is False
    assert handoff["completion_receipt_available"] is False
    assert handoff["preserved_outcome_fields"] == [
        "P_clean",
        "P_under_D",
        "P_under_N",
    ]
    assert set(handoff["rows"][0]) == {
        "example_id",
        "gene_id",
        "chrom",
        "strand",
        "positions",
    }
    assert set(handoff["common_leg_rows"][0]) == {
        "example_id",
        "gene_id",
        "margin_levels",
    }
    verified = validate_r5_preflight_cohort_handoff(handoff)
    assert verified["verified"] is True
    assert verified["scientific_entry_authorized"] is False
    assert verified["row_count"] == 400

    tampered = copy.deepcopy(handoff)
    tampered["common_leg_rows"][0]["family"] = {"Y_PD": 1.0}
    unsigned = {
        key: value for key, value in tampered.items() if key != "record_sha256"
    }
    tampered["record_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(R5CapstoneCohortError, match="non-common-leg outcomes"):
        validate_r5_preflight_cohort_handoff(tampered)

    examples = _r1c_target_examples(record, split="val")
    rows, audit = materialize_r5_preflight_rows(tuple(reversed(examples)), handoff)
    assert len(rows) == 400
    assert audit["schema"] == R5_PREFLIGHT_ROW_MATERIALIZATION_SCHEMA
    assert audit["cohort_kind"] == "preflight"
    assert audit["caller_must_supply_examples_after_target_ledger_authorization"] is False
    assert audit["ordered_row_ids_sha256"] == handoff["ordered_row_ids_sha256"]


def test_r1c_preflight_loader_fails_before_paths_while_prospective_pins_are_unset():
    with pytest.raises(R5CapstoneCohortError, match="pins remain unset"):
        load_r1c_preflight_cohort_handoff(
            "/does/not/exist/r1c_preflight.json",
            "/does/not/exist/r1c_preflight.manifest.json",
        )


def test_reviewed_r1c_preflight_is_authentic_but_frozen_r5_handoff_rejects_its_six_margin_rows():
    runner_path = (
        REPO
        / "scripts/splicebert_acceptor_coupling/run_r1c_sealed_reverse_behavior.py"
    )
    spec = importlib.util.spec_from_file_location(
        "r1c_terminal_handoff_audit", runner_path
    )
    assert spec is not None and spec.loader is not None
    r1c = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = r1c
    spec.loader.exec_module(r1c)

    artifact_path = (
        REPO
        / "results/splicebert_acceptor_coupling/r1c_reverse_behavior_preflight_v4.json"
    )
    manifest_path = artifact_path.with_suffix(".manifest.json")
    assert artifact_path.stat().st_size == 74_352_692
    assert (
        hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        == "1b27c4132c71a576c435b7b71f32831af9a44fbca6325931f35716ae4133cbdd"
    )
    assert manifest_path.stat().st_size == 11_370
    assert (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        == "7412f3159e82f6ab0bbaa95e2796f41aa038eef4233efa7f09ec65867e39c45c"
    )
    verified = r1c.verify_successor_preflight_package(
        artifact_path, manifest_path
    )
    record = verified["record"]
    attestation = verified["attestation"]
    assert (
        attestation["attestation_sha256"]
        == "04be60c2239c98eefb01521f7a8a94a68f4d2cc4db5db8f1dddb5622f7378ce1"
    )
    manifest = json.loads(manifest_path.read_text())

    assert attestation["verdict"] == "PREFLIGHT_GO"
    assert attestation["all_gates_pass"] is True
    assert r1c.FROZEN_PREFLIGHT_ARTIFACT_SHA256 is None
    assert all(
        getattr(cohort_module, name) is None
        for name in SHARED_R1C_PACKAGE_PIN_NAMES
    )
    assert set(record["rows"][0]["margin_levels"]) == {
        "P_clean",
        "P_under_D",
        "P_under_N",
        "X_clean",
        "X_under_D",
        "X_under_N",
    }
    with pytest.raises(
        R5CapstoneCohortError, match="R1c partner margin levels differ"
    ):
        build_r5_preflight_cohort_handoff(
            record,
            manifest,
            artifact_identity=attestation["artifact"],
            manifest_identity=attestation["manifest"],
            successor_attestation=attestation,
        )


def test_ac9_is_the_single_terminal_outcome_and_indexes_only_instrument_evidence():
    claims = (
        REPO / "studies/splicebert_acceptor_coupling/CLAIMS.md"
    ).read_text()
    decisions = (
        REPO / "studies/splicebert_acceptor_coupling/DECISIONS.md"
    ).read_text()
    manifest = json.loads((REPO / "results/MANIFEST.json").read_text())

    assert claims.count("| AC9 |") == 1
    assert "terminal `INSTRUMENT_UNRESOLVED`" in claims
    assert "b79570d" in claims
    assert "provenance-broken held-out verification" in claims
    assert "ACR18" in claims and "ACR19" in claims
    assert "this goal launches no GC/GT" in decisions
    assert "early-head, or MLP follow-up" in decisions

    ac9_rows = [
        row for row in manifest["rows"] if row.get("claim_id") == "AC9"
    ]
    assert len(ac9_rows) == 2
    artifact_row = next(row for row in ac9_rows if row["evidence_level"])
    assert artifact_row["evidence_level"] == "Instrument"
    assert artifact_row["polarity"] == "na"
    assert artifact_row["status"] == "canonical"
    assert artifact_row["artifact"].endswith(
        "r1c_reverse_behavior_preflight_v4.json"
    )

    result_dir = REPO / "results/splicebert_acceptor_coupling"
    assert not list(result_dir.glob("r1c_*target*.json"))
    assert not list(result_dir.glob("r5a_*.json"))
    assert not list(result_dir.glob("r5b_*.json"))
    assert not list(result_dir.glob("r5c_*.json"))


def test_capstone_handoff_rederives_exact_r1c_rows_without_residual_bucket_or_genome():
    (
        record,
        manifest,
        attempt,
        completion,
        artifact,
        manifest_identity,
        attempt_identity,
        completion_identity,
    ) = _r1c_target_fixture()
    handoff = build_r5_capstone_cohort_handoff(
        record,
        manifest,
        attempt,
        completion,
        artifact_identity=artifact,
        manifest_identity=manifest_identity,
        attempt_identity=attempt_identity,
        completion_identity=completion_identity,
        **_r1c_target_successor_kwargs(
            record,
            attempt,
            completion,
            artifact,
            manifest_identity,
            attempt_identity,
            completion_identity,
        ),
    )
    assert handoff["schema"] == R5_CAPSTONE_COHORT_HANDOFF_SCHEMA
    assert handoff["cohort_role"] == R5C_REQUIRED_COHORT_ROLE
    assert handoff["residual_validation_bucket_1_used"] is False
    assert handoff["handoff_reads_genome"] is False
    assert handoff["handoff_reads_chr8_chr9_sequences"] is False
    assert handoff["completion_receipt_available"] is True
    assert handoff["source"]["attempt"]["attempt_id"] == attempt["attempt_id"]
    assert handoff["source_package_sha256"] == canonical_sha256(
        {
            "artifact": artifact,
            "manifest": manifest_identity,
            "attempt": attempt_identity
            | {
                "attempt_id": attempt["attempt_id"],
                "biological_target_key": attempt["biological_target_key"],
                "status": attempt["status"],
            },
            "completion": completion_identity
            | {
                "completion_id": completion["completion_id"],
                "status": completion["status"],
            },
            "preflight_artifact": attempt["method_context"][
                "preflight_artifact"
            ],
            "preflight_manifest": attempt["method_context"][
                "preflight_manifest"
            ],
        }
    )
    assert len(handoff["rows"]) == 400
    assert set(handoff["rows"][0]) == {
        "example_id",
        "gene_id",
        "chrom",
        "strand",
        "positions",
    }
    assert handoff["source"]["source_cohort_sha256"] == record["cohort_sha256"]
    assert handoff["source_legacy_cohort_sha256"] == record["cohort_sha256"]
    assert handoff["unique_gene_count"] == 137
    assert handoff["ordered_gene_ids_sha256"] == canonical_sha256(
        [row["gene_id"] for row in handoff["rows"]]
    )
    assert handoff["sorted_unique_gene_ids_sha256"] == canonical_sha256(
        sorted({row["gene_id"] for row in handoff["rows"]})
    )
    assert len(handoff["capstone_construction_sha256"]) == 64
    validated = validate_r5_capstone_cohort_handoff(handoff)
    assert validated["verified"] is True
    assert validated["row_count"] == 400

    tampered = copy.deepcopy(handoff)
    tampered["rows"][0]["positions"]["paired_donor"] += 1
    payload = {key: value for key, value in tampered.items() if key != "record_sha256"}
    tampered["record_sha256"] = canonical_sha256(payload)
    with pytest.raises(R5CapstoneCohortError, match="row identity"):
        validate_r5_capstone_cohort_handoff(tampered)

    residual_reuse = copy.deepcopy(handoff)
    residual_reuse["residual_validation_bucket_1_used"] = True
    payload = {
        key: value for key, value in residual_reuse.items() if key != "record_sha256"
    }
    residual_reuse["record_sha256"] = canonical_sha256(payload)
    with pytest.raises(R5CapstoneCohortError, match="contract"):
        validate_r5_capstone_cohort_handoff(residual_reuse)

    (
        one_chromosome,
        one_chromosome_manifest,
        one_chromosome_attempt,
        one_chromosome_completion,
        one_chromosome_artifact,
        one_chromosome_manifest_identity,
        one_chromosome_attempt_identity,
        one_chromosome_completion_identity,
    ) = _r1c_target_fixture(one_chromosome=True)
    one_chromosome_handoff = build_r5_capstone_cohort_handoff(
        one_chromosome,
        one_chromosome_manifest,
        one_chromosome_attempt,
        one_chromosome_completion,
        artifact_identity=one_chromosome_artifact,
        manifest_identity=one_chromosome_manifest_identity,
        attempt_identity=one_chromosome_attempt_identity,
        completion_identity=one_chromosome_completion_identity,
        **_r1c_target_successor_kwargs(
            one_chromosome,
            one_chromosome_attempt,
            one_chromosome_completion,
            one_chromosome_artifact,
            one_chromosome_manifest_identity,
            one_chromosome_attempt_identity,
            one_chromosome_completion_identity,
        ),
    )
    assert {row["chrom"] for row in one_chromosome_handoff["rows"]} == {"chr8"}


def test_verified_target_common_legs_are_exact_ordered_and_never_select_rows():
    (
        record,
        manifest,
        attempt,
        completion,
        artifact,
        manifest_identity,
        attempt_identity,
        completion_identity,
    ) = _r1c_target_fixture()
    successor_kwargs = _r1c_target_successor_kwargs(
        record,
        attempt,
        completion,
        artifact,
        manifest_identity,
        attempt_identity,
        completion_identity,
    )
    handoff = build_r5_capstone_cohort_handoff(
        record,
        manifest,
        attempt,
        completion,
        artifact_identity=artifact,
        manifest_identity=manifest_identity,
        attempt_identity=attempt_identity,
        completion_identity=completion_identity,
        **successor_kwargs,
    )
    common = build_r1c_verified_target_common_legs(
        record, handoff, successor_kwargs["successor_attestation"]
    )
    assert common["schema"] == R5_VERIFIED_R1C_COMMON_LEGS_SCHEMA
    assert common["row_count"] == 400
    assert common["model_outputs_present"] is True
    assert common["may_be_used_for_cohort_selection"] is False
    assert common["caller_must_verify_downstream_target_receipt_before_load"] is True
    assert set(common["rows"][0]) == {"example_id", "gene_id", "margin_levels"}
    assert common["rows"][0]["example_id"] == handoff["rows"][0]["example_id"]
    assert common["ordered_partner_margin_levels_sha256"] == record[
        "target_cohort_identity"
    ]["ordered_partner_margin_levels_sha256"]

    tampered_record = copy.deepcopy(record)
    tampered_record["rows"][0]["margin_levels"]["P_clean"] += 0.25
    with pytest.raises(R5CapstoneCohortError, match="completed target digest"):
        build_r1c_verified_target_common_legs(
            tampered_record,
            handoff,
            successor_kwargs["successor_attestation"],
        )

    wrong_attestation = copy.deepcopy(successor_kwargs["successor_attestation"])
    wrong_attestation["verdict"] = "NO_CONFIRMED_PAIRED_DONOR_PATH"
    wrong_payload = {
        key: value
        for key, value in wrong_attestation.items()
        if key != "attestation_sha256"
    }
    wrong_attestation["attestation_sha256"] = canonical_sha256(wrong_payload)
    with pytest.raises(R5CapstoneCohortError, match="exact successful target"):
        build_r1c_verified_target_common_legs(
            record, handoff, wrong_attestation
        )


def test_capstone_loader_is_sealed_before_r1c_literal_pins_and_source_stops_fail():
    assert R1C_TARGET_ARTIFACT_SHA256 is None
    assert R1C_TARGET_MANIFEST_SHA256 is None
    assert R1C_TARGET_ATTEMPT_SHA256 is None
    assert R1C_TARGET_COMPLETION_SHA256 is None
    with pytest.raises(R5CapstoneCohortError, match="pins remain unset"):
        load_r1c_capstone_cohort_handoff(
            "/does/not/exist/r1c.json",
            "/does/not/exist/r1c.manifest.json",
            "/does/not/exist/r1c.attempt.json",
            "/does/not/exist/r1c.completion.json",
            "/does/not/exist/r1c_preflight.json",
            "/does/not/exist/r1c_preflight.manifest.json",
        )
    with pytest.raises(R5CapstoneCohortError, match="pins remain unset"):
        load_r1c_verified_target_common_legs(
            "/does/not/exist/r1c.json",
            "/does/not/exist/r1c.manifest.json",
            "/does/not/exist/r1c.attempt.json",
            "/does/not/exist/r1c.completion.json",
            "/does/not/exist/r1c_preflight.json",
            "/does/not/exist/r1c_preflight.manifest.json",
        )

    (
        record,
        manifest,
        attempt,
        completion,
        artifact,
        manifest_identity,
        attempt_identity,
        completion_identity,
    ) = _r1c_target_fixture()
    record["verdict"] = "NO_CONFIRMED_PAIRED_DONOR_PATH"
    with pytest.raises(R5CapstoneCohortError, match="source contract"):
        build_r5_capstone_cohort_handoff(
            record,
            manifest,
            attempt,
            completion,
            artifact_identity=artifact,
            manifest_identity=manifest_identity,
            attempt_identity=attempt_identity,
            completion_identity=completion_identity,
            **_r1c_target_successor_kwargs(
                record,
                attempt,
                completion,
                artifact,
                manifest_identity,
                attempt_identity,
                completion_identity,
            ),
        )

def test_capstone_attempt_bytes_and_embedded_identity_are_mandatory():
    (
        record,
        manifest,
        attempt,
        completion,
        artifact,
        manifest_identity,
        attempt_identity,
        completion_identity,
    ) = _r1c_target_fixture()
    tampered_attempt = attempt | {"status": "NOT_AUTHORIZED"}
    with pytest.raises(R5CapstoneCohortError, match="source contract"):
        build_r5_capstone_cohort_handoff(
            record,
            manifest,
            tampered_attempt,
            completion,
            artifact_identity=artifact,
            manifest_identity=manifest_identity,
            attempt_identity=attempt_identity,
            completion_identity=completion_identity,
            **_r1c_target_successor_kwargs(
                record,
                tampered_attempt,
                completion,
                artifact,
                manifest_identity,
                attempt_identity,
                completion_identity,
            ),
        )

    tampered_completion = copy.deepcopy(completion)
    tampered_completion["cohort_digests"][
        "ordered_partner_margin_levels_sha256"
    ] = "e" * 64
    completion_payload = {
        key: value
        for key, value in tampered_completion.items()
        if key != "completion_id"
    }
    tampered_completion["completion_id"] = canonical_sha256(completion_payload)
    with pytest.raises(R5CapstoneCohortError, match="cohort digests"):
        build_r5_capstone_cohort_handoff(
            record,
            manifest,
            attempt,
            tampered_completion,
            artifact_identity=artifact,
            manifest_identity=manifest_identity,
            attempt_identity=attempt_identity,
            completion_identity=completion_identity,
            **_r1c_target_successor_kwargs(
                record,
                attempt,
                tampered_completion,
                artifact,
                manifest_identity,
                attempt_identity,
                completion_identity,
            ),
        )

    stale_robustness = copy.deepcopy(record)
    stale_robustness["gene_robustness"]["all_required_robust"] = False
    stale_robustness["gene_robustness"]["paths"]["core_paired_donor"][
        "pass"
    ] = False
    # Deliberately leave verdict, top-level gates, and completion unchanged: semantic
    # authentication must catch the raw+trim equal-gene failure itself.
    with pytest.raises(R5CapstoneCohortError, match="source contract"):
        build_r5_capstone_cohort_handoff(
            stale_robustness,
            manifest,
            attempt,
            completion,
            artifact_identity=artifact,
            manifest_identity=manifest_identity,
            attempt_identity=attempt_identity,
            completion_identity=completion_identity,
            **_r1c_target_successor_kwargs(
                stale_robustness,
                attempt,
                completion,
                artifact,
                manifest_identity,
                attempt_identity,
                completion_identity,
            ),
        )

    missing_attempt_input = manifest | {"inputs": []}
    with pytest.raises(R5CapstoneCohortError, match="source contract"):
        build_r5_capstone_cohort_handoff(
            record,
            missing_attempt_input,
            attempt,
            completion,
            artifact_identity=artifact,
            manifest_identity=manifest_identity,
            attempt_identity=attempt_identity,
            completion_identity=completion_identity,
            **_r1c_target_successor_kwargs(
                record,
                attempt,
                completion,
                artifact,
                manifest_identity,
                attempt_identity,
                completion_identity,
            ),
        )


def _r1c_target_examples(record, *, split="test"):
    examples = []
    for index, serialized in enumerate(record["rows"]):
        positions = serialized["positions"]
        sequence = list("C" * 512)
        sequence[
            positions["prior_real_acceptor"] - 2 : positions["prior_real_acceptor"]
        ] = "AG"
        sequence[
            positions["partner_acceptor"] - 2 : positions["partner_acceptor"]
        ] = "AG"
        sequence[
            positions["paired_donor"] : positions["paired_donor"] + 2
        ] = "GT"
        sequence[
            positions["matched_decoy_donor"] : positions["matched_decoy_donor"] + 2
        ] = "GT"
        examples.append(
            CouplingExample(
                example_id=serialized["example_id"],
                sequence="".join(sequence),
                acceptor_pos=positions["prior_real_acceptor"],
                donor_pos=positions["paired_donor"],
                chrom=serialized["chrom"].removeprefix("chr"),
                strand=serialized["strand"],
                gene_id=serialized["gene_id"],
                split=split,
                exon_length=100,
                intron_length_down=100 + positions["dyad_distance_mismatch"],
                n_isoforms_donor=1,
                n_isoforms_acceptor=1,
                gene_type="protein_coding",
                other_gt=(positions["matched_decoy_donor"],),
                next_acceptor_pos=positions["partner_acceptor"],
            )
        )
    return examples


def test_capstone_materializer_maps_exact_handoff_order_and_keeps_missing_adjacent():
    (
        record,
        manifest,
        attempt,
        completion,
        artifact,
        manifest_identity,
        attempt_identity,
        completion_identity,
    ) = _r1c_target_fixture()
    handoff = build_r5_capstone_cohort_handoff(
        record,
        manifest,
        attempt,
        completion,
        artifact_identity=artifact,
        manifest_identity=manifest_identity,
        attempt_identity=attempt_identity,
        completion_identity=completion_identity,
        **_r1c_target_successor_kwargs(
            record,
            attempt,
            completion,
            artifact,
            manifest_identity,
            attempt_identity,
            completion_identity,
        ),
    )
    examples = _r1c_target_examples(record)
    blocked = list(examples[0].sequence)
    for start in range(164, 238, 2):
        blocked[start : start + 2] = "AG" if start % 4 else "GT"
    positions = record["rows"][0]["positions"]
    blocked[positions["prior_real_acceptor"] - 2 : positions["prior_real_acceptor"]] = "AG"
    blocked[positions["partner_acceptor"] - 2 : positions["partner_acceptor"]] = "AG"
    blocked[positions["paired_donor"] : positions["paired_donor"] + 2] = "GT"
    blocked[
        positions["matched_decoy_donor"] : positions["matched_decoy_donor"] + 2
    ] = "GT"
    examples[0] = replace(examples[0], sequence="".join(blocked))

    rows, audit = materialize_r5_capstone_rows(tuple(reversed(examples)), handoff)
    assert len(rows) == 400
    assert tuple(row.example.example_id for row in rows) == tuple(
        serialized["example_id"] for serialized in handoff["rows"]
    )
    assert rows[0].adjacent_window is None
    assert rows[1].adjacent_window is not None
    assert rows[0].identity_record()["adjacent_window"] is None
    assert audit["schema"] == R5_CAPSTONE_ROW_MATERIALIZATION_SCHEMA
    assert audit["row_count"] == 400
    assert audit["handoff_order_preserved"] is True
    assert audit["rows_filtered_or_reordered"] is False
    assert audit["adjacent_window_required"] is False
    assert audit["adjacent_window_unavailable_count"] == 1
    assert audit["ordered_row_ids_sha256"] == handoff["ordered_row_ids_sha256"]
    assert len(audit["materialization_construction_sha256"]) == 64
    assert len(audit["record_sha256"]) == 64


def test_capstone_materializer_rejects_missing_duplicate_or_coordinate_drift():
    (
        record,
        manifest,
        attempt,
        completion,
        artifact,
        manifest_identity,
        attempt_identity,
        completion_identity,
    ) = _r1c_target_fixture()
    handoff = build_r5_capstone_cohort_handoff(
        record,
        manifest,
        attempt,
        completion,
        artifact_identity=artifact,
        manifest_identity=manifest_identity,
        attempt_identity=attempt_identity,
        completion_identity=completion_identity,
        **_r1c_target_successor_kwargs(
            record,
            attempt,
            completion,
            artifact,
            manifest_identity,
            attempt_identity,
            completion_identity,
        ),
    )
    examples = _r1c_target_examples(record)
    with pytest.raises(R5CapstoneCohortError, match="missing authenticated"):
        materialize_r5_capstone_rows(examples[1:], handoff)
    with pytest.raises(R5CapstoneCohortError, match="unique example IDs"):
        materialize_r5_capstone_rows(examples + [examples[0]], handoff)
    drifted = list(examples)
    drifted[0] = replace(drifted[0], donor_pos=drifted[0].donor_pos + 1)
    with pytest.raises(R5CapstoneCohortError, match="coordinates differ"):
        materialize_r5_capstone_rows(drifted, handoff)
