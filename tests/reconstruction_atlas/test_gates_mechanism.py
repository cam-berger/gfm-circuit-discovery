from __future__ import annotations

import pytest

from gfmcd.reconstruction_atlas.gates import (
    MANDATORY_GATES,
    GateOutcome,
    GateState,
    PreflightReport,
    frozen_config_sha256,
)
from gfmcd.reconstruction_atlas.mechanism import (
    BehaviorQualification,
    CausalInterventionSpec,
    InterventionType,
    require_mechanism_qualification,
)


def report(
    state: GateState = GateState.PASS, *, frozen: bool = True
) -> PreflightReport:
    return PreflightReport(
        report_id="pf1",
        config_sha256="abc",
        checkpoint_revision="rev",
        dataset_release="data-v1",
        input_sha256={"collection": "a" * 64},
        split="development",
        outcomes=tuple(
            GateOutcome(name, state, {}, {}) for name in sorted(MANDATORY_GATES)
        ),
        frozen=frozen,
    )


def test_preflight_requires_every_pass_and_a_frozen_lock() -> None:
    assert report().decision == "QUALIFIED"
    assert report(GateState.FAIL).decision == "INSTRUMENT_UNRESOLVED"
    assert report(frozen=False).decision == "INSTRUMENT_UNRESOLVED"
    with pytest.raises(RuntimeError, match="not qualified"):
        report(GateState.ANOMALOUS).assert_qualified(
            expected_config_sha256="abc",
            expected_checkpoint_revision="rev",
            expected_input_sha256={"collection": "a" * 64},
        )
    with pytest.raises(RuntimeError, match="data inputs"):
        report().assert_qualified(
            expected_config_sha256="abc",
            expected_checkpoint_revision="rev",
            expected_input_sha256={"collection": "b" * 64},
        )


def test_internal_screen_requires_behavior_and_context_qualification() -> None:
    qualification = BehaviorQualification(
        "splice.donor", True, True, False, "result.json"
    )
    with pytest.raises(RuntimeError, match="prohibited"):
        require_mechanism_qualification(qualification)


def test_causal_null_sets_must_match_intervention_size() -> None:
    with pytest.raises(ValueError, match="match candidate size"):
        CausalInterventionSpec(
            intervention_id="a",
            intervention_type=InterventionType.ABLATION,
            candidate_components=("L1H1", "L2H2"),
            matched_null_sets=(("L3H3",),),
            protected_metrics=("general_mlm",),
            behavior_artifact="behavior.json",
        )


def test_draft_config_cannot_authorize_a_target_run(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        '{"status":"DRAFT_NOT_FROZEN","mandatory_gates":{},"behavior":{}}'
    )
    with pytest.raises(RuntimeError, match="not FROZEN"):
        frozen_config_sha256(config)
