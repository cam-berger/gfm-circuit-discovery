"""C36's confirmation analysis re-executes model-free from the committed shards, byte for byte."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/splicebert/replay_p3_phase_surgery_confirmation.py"


def _replay_module():
    spec = importlib.util.spec_from_file_location("replay_c36", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c36_confirmation_analysis_replays_bitwise() -> None:
    replayed, committed = _replay_module().replay()
    assert replayed == committed


def test_staging_refuses_bytes_that_do_not_match_the_pin() -> None:
    module = _replay_module()
    record = {"path": "results/phase-lattice/p3_phase_surgery_untouched_census_20260721.json",
              "sha256": "0" * 64}
    with pytest.raises(FileNotFoundError, match="no committed file matches"):
        module.locate(record)
