"""LayerNorm faithfulness calibration (TO_FIX.md §16).

Pins the qualitative shape of the frozen-LN gap, not brittle exact values: LN-free is exact (the
control), LayerNorm breaks conservation, edge structure survives, and freezing LN distorts causal
magnitude most for small interventions while roughly preserving direction.
"""

from __future__ import annotations

from gfmcd.tracer.layernorm import calibrate


def test_ln_free_is_exact_control():
    r = calibrate()["ln_free"]
    assert r["conservation_max_err"] < 1e-6
    assert r["edges"]["recall"] == 1.0 and r["edges"]["precision"] == 1.0
    for row in r["gap_curve"]:
        assert row["faithfulness_gap"] < 1e-9  # no LN => no gap, exactly
        assert row["direction_cos"] > 0.999


def test_layernorm_breaks_conservation_but_keeps_structure():
    r = calibrate()["layernorm"]
    assert r["conservation_max_err"] > 0.1, "LN should break the LN-ignoring reconstruction"
    assert r["edges"]["recall"] == 1.0, "planted OV edge structure should survive LN"


def test_frozen_ln_gap_worst_for_small_interventions():
    curve = calibrate()["layernorm"]["gap_curve"]
    small, large = curve[0], curve[-1]
    # relative magnitude error is largest where the true effect is smallest
    assert small["faithfulness_gap"] > large["faithfulness_gap"]
    # direction is roughly preserved and improves with magnitude
    assert small["direction_cos"] < large["direction_cos"]
    assert all(row["direction_cos"] > 0.5 for row in curve)
