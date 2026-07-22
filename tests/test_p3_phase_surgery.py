from __future__ import annotations

import numpy as np
import pytest
import torch

from gfmcd.splicing.p3_phase_surgery import (
    PhaseSurgerySpec,
    fourier_band_component,
    local_phase_surgery,
    modulo_recovery_profile,
    phase_shift_delta,
    signed_permutation_component,
    surgery_rms,
)


def _period3_table(length=12):
    x = torch.arange(length, dtype=torch.float32)
    return torch.stack((torch.cos(2 * torch.pi * x / 3), torch.sin(2 * torch.pi * x / 3)), 1)


def test_fourier_component_recovers_exact_grid_tone():
    table = _period3_table(12)
    component = fourier_band_component(table, (4,))
    assert torch.allclose(component, table, atol=2e-6)


def test_phase_shift_is_exact_and_recovers_every_three_positions():
    component = _period3_table(30)
    one = phase_shift_delta(component, PhaseSurgerySpec(4, 10, 1, "left"))
    assert torch.allclose(component[4:10] + one, component[5:11], atol=1e-6)
    three = phase_shift_delta(component, PhaseSurgerySpec(4, 10, 3, "left"))
    assert torch.allclose(three, torch.zeros_like(three), atol=2e-6)
    profile = modulo_recovery_profile(component, start=4, end=10, direction="left")
    assert profile[3] < 1e-6 and profile[6] < 1e-6
    assert profile[2] > 0.5 and profile[4] > 0.5


def test_embedding_hook_is_batch_specific_and_leaves_other_positions_exact():
    embedding = torch.nn.Embedding(18, 2)
    with torch.no_grad():
        embedding.weight.copy_(_period3_table(18) + 0.1)
    component = _period3_table(18)
    ids = torch.arange(12).unsqueeze(0)
    native = embedding(ids)
    specs = (
        PhaseSurgerySpec(3, 7, 1, "left"),
        PhaseSurgerySpec(5, 9, 1, "right"),
    )
    with local_phase_surgery(embedding, specs, component):
        edited = embedding(ids)
    assert edited.shape == (2, 12, 2)
    assert torch.equal(edited[0, :3], native[0, :3])
    assert torch.equal(edited[0, 7:], native[0, 7:])
    assert torch.allclose(edited[0, 3:7], native[0, 3:7] + component[4:8] - component[3:7])
    assert torch.allclose(edited[1, 5:9], native[0, 5:9] + component[4:8] - component[5:9])
    assert torch.equal(embedding(ids), native)


def test_surgery_rejects_out_of_table_source():
    component = _period3_table(12)
    with pytest.raises(ValueError, match="source positions"):
        phase_shift_delta(component, PhaseSurgerySpec(1, 5, 3, "right"))


def test_signed_permutation_preserves_spatial_geometry_and_shift_dose():
    component = torch.randn(30, 8, generator=torch.Generator().manual_seed(3))
    control = signed_permutation_component(component, seed=17)
    assert not torch.equal(control, component)
    assert torch.allclose(control @ control.T, component @ component.T, atol=1e-5)
    for direction in ("left", "right"):
        for amount in (2, 3, 4, 9):
            spec = PhaseSurgerySpec(10, 16, amount, direction)
            assert surgery_rms(control, spec) == pytest.approx(surgery_rms(component, spec))


def test_signed_permutation_is_seed_deterministic():
    component = _period3_table(30)
    assert torch.equal(
        signed_permutation_component(component, seed=2),
        signed_permutation_component(component, seed=2),
    )
    assert not torch.equal(
        signed_permutation_component(component, seed=2),
        signed_permutation_component(component, seed=3),
    )
