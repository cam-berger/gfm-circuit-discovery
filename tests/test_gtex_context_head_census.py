import json
from pathlib import Path


RESULTS = (
    Path(__file__).resolve().parents[1]
    / "results/splicebert_gtex_finetune_shift"
)


def test_complete_l5_discovery_freezes_expected_endpoint_candidates():
    result = json.loads(
        (RESULTS / "native_context_head_census_discovery_v1.json").read_text()
    )
    assert result["status"] == "COMPLETE"
    assert result["multiplicity_family"] == 64
    assert len(result["head_summaries"]["donor"]) == 16
    assert len(result["head_summaries"]["acceptor"]) == 16
    assert all(
        all(gates.values()) for gates in result["behavior_gates"].values()
    )
    assert result["freeze"]["donor"]["candidate_heads"] == [
        "L5H14",
        "L5H8",
        "L5H12",
    ]
    assert result["freeze"]["acceptor"]["candidate_heads"] == [
        "L5H11",
        "L5H14",
        "L5H8",
    ]


def test_fresh_five_seed_confirmation_establishes_shared_and_specific_heads():
    result = json.loads(
        (
            RESULTS / "native_context_head_census_confirmation_v1.json"
        ).read_text()
    )
    assert result["status"] == "COMPLETE"
    assert result["n"] == 200
    assert result["n_groups"] == 184
    assert result["discovery_confirmation_gene_overlap"] == 0
    assert result["family_seeds"] == [2020, 2021, 2022, 2023, 2024]
    assert result["multiplicity"] == {
        "behavior_tests": 2,
        "candidate_tests": 12,
        "control_contrasts": 12,
    }
    assert result["confirmed_heads"] == {
        "donor": ["L5H14", "L5H8", "L5H12"],
        "acceptor": ["L5H11", "L5H14", "L5H8"],
    }
    assert all(
        verdict["PASS"]
        for endpoint in result["verdicts"].values()
        for verdict in endpoint.values()
    )

    for task_name, heads in result["confirmed_heads"].items():
        for head in heads:
            for direction in ("damage", "recovery"):
                effect = result["family"]["heads"][task_name][head][direction]
                contrast = result["family"]["candidate_vs_controls"][
                    task_name
                ][head][direction]
                effect_q = next(
                    value
                    for key, value in effect.items()
                    if key.startswith("q_bh_")
                )
                contrast_q = next(
                    value
                    for key, value in contrast.items()
                    if key.startswith("q_bh_")
                )
                assert effect["estimate"] >= 0.05
                assert effect["ci_low"] > 0
                assert effect_q <= 0.05
                assert all(
                    seed_estimate > 0
                    for seed_estimate in effect["seed_estimates"].values()
                )
                assert contrast["ci_low"] > 0
                assert contrast_q <= 0.05
