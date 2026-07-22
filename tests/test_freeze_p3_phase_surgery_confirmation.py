from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert/freeze_p3_phase_surgery_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location("freeze_p3_phase_surgery_confirmation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def record(index: int, role: str, context: str = "cds") -> dict:
    return {
        "eligible": True,
        "example_id": f"{context}-example-{index}",
        "gene_id": f"{context}-gene-{index}",
        "chrom": "chr1",
        "strand": "+",
        "role": role,
        "native_sequence_sha256": f"sha-{index}",
        "blocks": {context: [[20, 56], [80, 116]], "upstream_intron": [[200, 236], [260, 296]]},
    }


def test_build_package_freezes_exact_primary_control_and_context_counts():
    primary = [record(index, "confirmation") for index in range(943)]
    reserve = [record(index + 943, "reserve") for index in range(457)]
    non = [record(index, "confirmation", "non_cds_exon") for index in range(52)]
    pairs = [{
        "pair_id": f"pair-{index:04d}",
        "same_gene": False,
        "cds_example_id": primary[index]["example_id"],
        "non_cds_example_id": non[index]["example_id"],
    } for index in range(52)]
    cohort = {
        "schema": "gfmcd.p3_phase_surgery.cohort.v1",
        "model_outcomes_read": False,
        "records": primary + reserve,
    }
    non_cohort = {
        "schema": "gfmcd.p3_phase_surgery.cohort.v1",
        "model_outcomes_read": False,
        "records": non,
    }
    match = {
        "schema": "gfmcd.p3_phase_surgery.context_match.v1",
        "model_outcomes_read": False,
        "pairs": pairs,
        "balance": {"max_abs_post_match_smd": 0.2},
        "balance_spec": {"minimum_pairs": 30, "max_abs_smd": 0.25},
    }
    package = MODULE.build_package(
        cds=cohort,
        non_cds=non_cohort,
        match=match,
        source_records={},
        code_records={},
        data_records={},
        revision="abc",
    )
    assert package["counts"]["primary_genes"] == 943
    assert package["counts"]["reserve_genes_not_runnable"] == 457
    assert package["counts"]["direction_control_genes"] == 150
    assert package["counts"]["context_pairs"] == 52
    assert package["cohorts"]["direction_control_example_ids"] == [
        row["example_id"] for row in primary[:150]
    ]


def test_build_package_refuses_outcome_read_cohort():
    bad = {
        "schema": "gfmcd.p3_phase_surgery.cohort.v1",
        "model_outcomes_read": True,
        "records": [],
    }
    match = {
        "schema": "gfmcd.p3_phase_surgery.context_match.v1",
        "model_outcomes_read": False,
        "pairs": [],
        "balance": {"max_abs_post_match_smd": 0.2},
        "balance_spec": {"minimum_pairs": 30, "max_abs_smd": 0.25},
    }
    with pytest.raises(RuntimeError, match="outcome-blind"):
        MODULE.build_package(
            cds=bad,
            non_cds=bad,
            match=match,
            source_records={},
            code_records={},
            data_records={},
            revision=None,
        )
