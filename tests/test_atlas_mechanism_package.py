from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/splicebert_head_context_atlas/build_atlas_mechanism_package.py"
SPEC = importlib.util.spec_from_file_location("atlas_mechanism_package", SCRIPT)
package = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(package)


def _profile(component: str, importance: float, estimand: str = "A") -> dict:
    return {
        "component_id": component,
        "estimand": estimand,
        "donor": {"mean": importance},
        "acceptor": {"mean": importance},
    }


def test_importance_control_prefers_same_layer_and_excludes_shortlist() -> None:
    candidate = _profile("L2H4", 0.20)
    pool = [
        _profile("L2H5", 0.19),
        _profile("L2H6", 0.18),
        _profile("L1H0", 0.199),
    ]
    control = package.choose_importance_control(
        candidate, pool, {"L2H4", "L2H5"}, context=False
    )
    assert control["component_id"] == "L2H6"
    assert control["same_layer"] is True


def test_mlp_control_uses_nearest_other_mlp() -> None:
    candidate = _profile("L3MLP", 0.10, estimand="B")
    pool = [
        _profile("L1MLP", 0.16, estimand="B"),
        _profile("L4MLP", 0.11, estimand="B"),
        _profile("L3H1", 0.101, estimand="B"),
    ]
    control = package.choose_importance_control(candidate, pool, {"L3MLP"}, context=False)
    assert control["component_id"] == "L4MLP"


def test_typical_packet_is_closest_to_frozen_mean() -> None:
    rows = [
        {"packet_id": "a", "effect": 0.01},
        {"packet_id": "b", "effect": 0.21},
        {"packet_id": "c", "effect": 0.12},
    ]
    assert package.choose_typical(rows, 0.10, key="x")["packet_id"] == "c"


def test_intervention_semantics_do_not_call_a_interchange_a_qk() -> None:
    a = package.intervention_spec({"component_id": "L1H2", "estimand": "A"})
    b = package.intervention_spec({"component_id": "L1H2", "estimand": "B"})
    assert a["qk_v"]["source"] == "matched_control_packet"
    assert "not a decomposition" in a["qk_v"]["interpretation"]
    assert b["qk_v"]["source"] == "partner_destroyed_same_packet"
    assert b["qk_v"]["conditions"] == ["QK", "V", "QKV"]


def test_physical_validation_artifact_stem_is_not_analysis_split_label() -> None:
    """The rows say atlas_internal_validation; files are written as .validation.*."""
    cohort_prefix = REPO / "results/splicebert_head_context_atlas/atlas_context_matched_v7"
    expected = {
        "pairs": Path(f"{cohort_prefix}.validation.pairs.jsonl"),
        "packets": Path(f"{cohort_prefix}.validation.packets.jsonl"),
        "sequences": Path(f"{cohort_prefix}.validation.sequences.jsonl"),
    }
    assert all(path.exists() for path in expected.values())
    assert not any(
        Path(f"{cohort_prefix}.internal_validation.{kind}.jsonl").exists()
        for kind in expected
    )
