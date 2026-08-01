from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from gfmcd.artifact_guard import ArtifactExistsError


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "splicebert_gtex_finetune_shift"
    / "freeze_gtex_native_biological_features_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "freeze_gtex_native_biological_features_confirmation",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@dataclass(frozen=True)
class Example:
    example_id: str
    gene_id: str
    sequence: str = "GAA" * 70
    chrom: str = "chr1"
    strand: str = "+"
    donor_pos: int = 90
    next_acceptor_pos: int = 150


def nuisance(value: float = 0.0) -> dict[str, float]:
    return {name: float(value) for name in MODULE.NUISANCE_NAMES}


def successful_pair(
    clean: str,
    *,
    requested: tuple[int, int] | None,
    nuisance_value: float = 0.0,
) -> dict:
    arms = {}
    for direction, delivered, score in (("low", 2, 1.0), ("high", 3, 2.0)):
        requested_dose = (
            None
            if requested is None
            else requested[0 if direction == "low" else 1]
        )
        arms[direction] = {
            "sequence": clean,
            "sequence_sha256": MODULE.sha256_text(clean),
            "success": True,
            "failure_reason": None,
            "requested_changed_bases": requested_dose,
            "delivered_changed_bases": delivered,
            "changed_positions": list(range(delivered)),
            "codon_changes": [],
            "objective_clean": 1.5,
            "objective_score": score,
            "invariants": {"synthetic_test_invariant": True},
            "maxent": {},
        }
    return {
        "pair_success": True,
        "failure_reason": None,
        "arms": arms,
        "nuisance": nuisance(nuisance_value),
        "pair_score_high_minus_low": 1.0,
    }


def run_candidate_package(
    monkeypatch,
    *,
    n_examples: int,
    one_control_nuisance_failure: bool = False,
):
    calls: list[tuple[str, str, tuple[int, int] | None]] = []

    def fake_construct_family_pair(*, example, model, exact_doses, **_kwargs):
        calls.append((example.example_id, model.name, exact_doses))
        value = 0.0
        if (
            one_control_nuisance_failure
            and example.example_id == "example-0"
            and model.name == "control-a"
        ):
            value = 1.0
        return successful_pair(
            example.sequence,
            requested=exact_doses,
            nuisance_value=value,
        )

    monkeypatch.setattr(
        MODULE, "construct_family_pair", fake_construct_family_pair
    )
    examples = [
        Example(example_id=f"example-{index}", gene_id=f"gene-{index}")
        for index in range(n_examples)
    ]
    psams = {
        name: SimpleNamespace(name=name)
        for name in ("candidate", "control-a", "control-b", "control-c")
    }
    summary, states = MODULE.construct_candidate_package(
        examples=examples,
        endpoint="donor",
        candidate_name="candidate",
        controls=("control-a", "control-b", "control-c"),
        expected_sign=-1,
        caliper_record={
            "calipers": {name: 0.0 for name in MODULE.NUISANCE_NAMES}
        },
        cds_intervals={},
        codon_counts={},
        codon_pair_counts={},
        psams=psams,
        exon_score=lambda _sequence: 0.0,
        score3=lambda _sequence: 0.0,
        score5=lambda _sequence: 0.0,
    )
    return summary, states, calls


def test_controls_are_constructed_at_each_candidate_low_and_high_dose(monkeypatch):
    summary, states, calls = run_candidate_package(
        monkeypatch,
        n_examples=MODULE.MIN_CONFIRMATION_EXAMPLES,
    )
    candidate_calls = [row for row in calls if row[1] == "candidate"]
    control_calls = [row for row in calls if row[1] != "candidate"]
    assert all(exact_doses is None for _, _, exact_doses in candidate_calls)
    assert all(exact_doses == (2, 3) for _, _, exact_doses in control_calls)
    assert summary["confirmation_construction_gate"] is True
    assert (
        summary["complete_dose_and_nuisance_matched_set_count"]
        == MODULE.MIN_CONFIRMATION_EXAMPLES
    )
    control_states = [
        row for row in states if row["family_role"] == "matched_control"
    ]
    assert control_states
    for row in control_states:
        expected = 2 if row["direction"] == "low" else 3
        assert row["requested_changed_bases"] == expected
        assert row["delivered_changed_bases"] == expected
        assert row["sequence"]
        assert row["sequence_sha256"] == MODULE.sha256_text(row["sequence"])


def test_one_nuisance_mismatch_below_minimum_preserves_stopped_gate(monkeypatch):
    summary, _states, _calls = run_candidate_package(
        monkeypatch,
        n_examples=MODULE.MIN_CONFIRMATION_EXAMPLES,
        one_control_nuisance_failure=True,
    )
    assert (
        summary["complete_dose_and_nuisance_matched_set_count"]
        == MODULE.MIN_CONFIRMATION_EXAMPLES - 1
    )
    assert summary["confirmation_construction_gate"] is False


def test_gc3_uses_third_codon_base_not_codon_start():
    low = "GCTGAA"
    high = "GCCGAG"
    assert MODULE.gc3_high_minus_low(high, low, (0, 3)) == 2
    # The first positions are both G in both sequences; the legacy start-index
    # implementation would incorrectly report no change.
    assert sum(high[start] in "GC" for start in (0, 3)) == sum(
        low[start] in "GC" for start in (0, 3)
    )


def test_nuisance_caliper_is_inclusive_and_fails_above_band():
    candidate = nuisance()
    control = nuisance()
    control["mfe_high_minus_low"] = 0.5
    calipers = {name: 0.0 for name in MODULE.NUISANCE_NAMES}
    calipers["mfe_high_minus_low"] = 0.5
    passed, differences, metrics = MODULE.nuisance_match(
        candidate, control, calipers
    )
    assert passed is True
    assert differences["mfe_high_minus_low"] == 0.5
    assert metrics["mfe_high_minus_low"] is True
    control["mfe_high_minus_low"] = 0.500001
    passed, _differences, metrics = MODULE.nuisance_match(
        candidate, control, calipers
    )
    assert passed is False
    assert metrics["mfe_high_minus_low"] is False


def test_schema_and_outcome_firewall_fail_closed():
    MODULE._forbid_outcome_payload(
        {"expected_high_minus_low_margin_sign": -1, "sequence": "ACGT"}
    )
    with pytest.raises(RuntimeError, match="outcome-bearing"):
        MODULE._forbid_outcome_payload(
            {"states": [{"sequence": "ACGT", "native_margin": 2.0}]}
        )
    with pytest.raises(RuntimeError, match="package keys changed"):
        MODULE.validate_confirmation_package(
            {"schema": MODULE.SCHEMA, "unexpected": True}
        )


def test_higher_quantile_is_frozen_and_conservative():
    assert MODULE.higher_quantile([0.0, 1.0, 2.0, 100.0], 0.95) == 100.0
    assert MODULE.higher_quantile([0.0, 1.0, 2.0], 0.50) == 1.0


def test_result_and_manifest_paths_are_both_immutable(tmp_path):
    result = tmp_path / "confirmation.json"
    result.write_text("{}\n")
    with pytest.raises(ArtifactExistsError):
        MODULE.output_paths(result)

    result.unlink()
    manifest = tmp_path / "confirmation.manifest.json"
    manifest.write_text("{}\n")
    with pytest.raises(ArtifactExistsError):
        MODULE.output_paths(result)
