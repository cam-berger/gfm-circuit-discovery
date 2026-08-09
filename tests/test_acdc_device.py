"""Precision policy, batch autotuning, and the device floor that sets R4's tau grid.

Genome-free and GPU-free: every path is exercised on CPU.

The load-bearing test here is ``test_a_tau_grid_cannot_be_derived_from_a_zero_floor``.  A panel
that fails to discriminate returns floor 0, which would silently yield ``tau_min = 0`` and a
grid of zeros -- an ACDC run that removes every edge whose delta is under nothing.  Refusing
that is the difference between a measured grid and a grid-shaped hole.
"""

from __future__ import annotations

import pytest
import torch

from gfmcd.circuits.acdc_device import (
    CPU_REFERENCE_BATCH, TAU_FLOOR_MULTIPLE, TAU_GRID_POINTS, assert_decision_determinism,
    autotune_batch, cache_bytes_per_example, measure_device_floor, set_precision_policy,
    tau_grid_from_floor,
)


# --------------------------------------------------------------------------------------
# D7.1 -- the refused optimisation.
# --------------------------------------------------------------------------------------
def test_precision_policy_disables_tf32_on_both_backends():
    policy = set_precision_policy()
    assert policy.matmul_tf32 is False
    assert policy.cudnn_tf32 is False


def test_precision_policy_reports_state_read_back_not_requested():
    """A policy that failed to apply must not be recorded as applied."""
    set_precision_policy()
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        policy = set_precision_policy()
        assert policy.matmul_tf32 is bool(torch.backends.cuda.matmul.allow_tf32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False


def test_precision_policy_is_manifest_serialisable():
    payload = set_precision_policy().to_dict()
    assert payload["dtype"] == "float32"
    assert set(payload) == {
        "matmul_tf32", "cudnn_tf32", "matmul_precision", "dtype", "deterministic_algorithms",
    }


# --------------------------------------------------------------------------------------
# D7.5 -- batch autotuning.
# --------------------------------------------------------------------------------------
def test_cache_bytes_scale_with_nodes_sequence_and_hidden():
    base = cache_bytes_per_example(103, 1026, 512)
    assert cache_bytes_per_example(206, 1026, 512) == 2 * base
    assert cache_bytes_per_example(103, 2052, 512) == 2 * base
    assert cache_bytes_per_example(103, 1026, 1024) == 2 * base


def test_cache_bytes_match_the_design_estimate():
    """D7.5 records ~216 MB per example for R4's graph; drift here changes every VRAM claim."""
    per_example = cache_bytes_per_example(103, 1026, 512) / 2      # caches only, no transient
    assert 200e6 < per_example < 230e6


def test_autotune_returns_the_reference_batch_without_cuda():
    assert autotune_batch(103, 1026, 512, device="cpu") == CPU_REFERENCE_BATCH


def test_autotune_never_returns_zero():
    assert autotune_batch(10**6, 10**4, 10**4, device="cpu") >= 1


def test_autotune_rejects_an_out_of_range_safety_factor():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="safety must be"):
            autotune_batch(103, 1026, 512, device="cpu", safety=bad)


# --------------------------------------------------------------------------------------
# R4 section 3 -- tau is measured, not chosen.
# --------------------------------------------------------------------------------------
def test_tau_grid_starts_at_three_times_the_floor():
    grid = tau_grid_from_floor(1e-4)
    assert grid[0] == pytest.approx(TAU_FLOOR_MULTIPLE * 1e-4)


def test_tau_grid_has_the_declared_shape_and_spans_two_decades():
    grid = tau_grid_from_floor(1e-4)
    assert len(grid) == TAU_GRID_POINTS
    assert grid[-1] == pytest.approx(grid[0] * 100.0)
    assert list(grid) == sorted(grid)


def test_a_tau_grid_cannot_be_derived_from_a_zero_floor():
    """A non-discriminating panel must refuse, not return a grid of zeros."""
    with pytest.raises(ValueError, match="did not discriminate"):
        tau_grid_from_floor(0.0)


def test_negative_floor_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        tau_grid_from_floor(-1e-6)


def test_device_floor_is_the_max_absolute_disagreement():
    panel = [torch.zeros(1), torch.ones(1), torch.full((1,), 2.0)]
    reference = [0.0, 1.0, 2.0]
    result = measure_device_floor(
        lambda m: float(m.sum()) + 0.002, panel, reference,
        reference_device="mps", run_device="cuda",
    )
    assert result.floor == pytest.approx(0.002)
    assert result.n_panel == 3


def test_a_floor_above_the_chosen_tau_fails_the_gate():
    """The Atlas cross-device tolerance is 2e-3; a 3e-4 grid would sit under it."""
    result = measure_device_floor(
        lambda m: float(m.sum()) + 2e-3, [torch.zeros(1)], [0.0],
        reference_device="mps", run_device="cuda",
    )
    assert result.tau_min == pytest.approx(6e-3)
    assert result.tau_min > 3e-4              # the naive a-priori grid is below the floor
    assert result.passes is True              # ... but the derived grid clears it by construction


def test_device_floor_rejects_a_mismatched_reference():
    with pytest.raises(ValueError, match="reference has"):
        measure_device_floor(lambda m: 0.0, [torch.zeros(1)], [0.0, 1.0],
                             reference_device="mps", run_device="cuda")


def test_device_floor_rejects_an_empty_panel():
    with pytest.raises(ValueError, match="cannot bound anything"):
        measure_device_floor(lambda m: 0.0, [], [], reference_device="mps", run_device="cuda")


def test_device_floor_serialises_with_its_verdict():
    payload = measure_device_floor(
        lambda m: 1e-5, [torch.zeros(1)], [0.0], reference_device="mps", run_device="cuda"
    ).to_dict()
    assert payload["PASS"] is True
    assert payload["tau_floor_multiple"] == TAU_FLOOR_MULTIPLE
    assert len(payload["tau_grid"]) == TAU_GRID_POINTS


# --------------------------------------------------------------------------------------
# Determinism -- one nondeterministic call changes every later decision.
# --------------------------------------------------------------------------------------
def test_determinism_passes_on_a_stable_prefix():
    report = assert_decision_determinism(lambda: [True, False, True])
    assert report["PASS"] is True
    assert report["n_decisions"] == 3


def test_determinism_fails_and_names_the_first_divergence():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        return [True, False, calls["n"] == 1]

    report = assert_decision_determinism(flaky)
    assert report["PASS"] is False
    assert report["first_mismatch_index"] == 2
