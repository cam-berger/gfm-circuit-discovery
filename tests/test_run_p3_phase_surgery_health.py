from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/splicebert/run_p3_phase_surgery_health.py"
SPEC = importlib.util.spec_from_file_location("run_p3_phase_surgery_health", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Example:
    example_id = "e"
    sequence = "A" * 100
    donor_pos = 40
    next_acceptor_pos = 70


def test_probe_positions_are_deterministic_and_exclude_sites():
    first = MODULE.stable_probe_positions(Example(), n_probe=12, seed=3)
    second = MODULE.stable_probe_positions(Example(), n_probe=12, seed=3)
    assert first == second
    assert all(abs(position - 40) > 2 and abs(position - 70) > 3 for position in first)


def test_paired_mean_ci_is_deterministic():
    values = list(np.linspace(-0.02, 0.01, 20))
    assert MODULE.paired_mean_ci(values, seed=4, draws=999) == MODULE.paired_mean_ci(
        values, seed=4, draws=999
    )
    with pytest.raises(ValueError):
        MODULE.paired_mean_ci(values, seed=4, draws=10)
