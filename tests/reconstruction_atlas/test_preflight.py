from __future__ import annotations

from gfmcd.reconstruction_atlas.preflight import (
    build_preflight_variants,
    summarize_preflight_effects,
)
from gfmcd.reconstruction_atlas.schema import Interval


def fixed_encoder(sequence: str):
    ids = [101] + [200 + ord(base) for base in sequence] + [102]
    offsets = [(0, 0)] + [(i, i + 1) for i in range(len(sequence))] + [(0, 0)]
    return ids, offsets


def test_copy_plant_is_graded_and_preserves_target_geometry() -> None:
    sequence = "A" * 48 + "GT" + ("ACGTTCA" * 30)
    variants = build_preflight_variants(
        sequence,
        Interval(48, 50),
        encoder=fixed_encoder,
        seed=7,
    )
    assert variants is not None
    assert all(
        value[48:50] == "GT" and len(value) == len(sequence)
        for value in variants.sequences().values()
    )
    assert variants.copy_dose_4.count("GT") > variants.copy_dose_1.count("GT")


def test_preflight_summary_is_cluster_first_and_reports_mde() -> None:
    rows = [
        {
            "site_class": "splice.donor",
            "homology_cluster": f"g{index}",
            "reference": -2.0,
            "copy_dose_1": -1.99,
            "copy_dose_2": -1.98,
            "copy_dose_4": -1.95 + index * 0.0001,
            "irrelevant_exchange": -2.0,
        }
        for index in range(20)
    ]
    summary = summarize_preflight_effects(
        rows,
        bootstrap_draws=100,
        seed=7,
        power=0.8,
    )["splice.donor"]
    assert summary["n_clusters"] == 20
    assert summary["copy_means_monotonic"]
    assert summary["minimum_detectable_effect_nats_per_covered_nt"] > 0
