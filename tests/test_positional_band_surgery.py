from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert_gtex_finetune_shift/run_gtex_positional_band_surgery.py"
)
SPEC = spec_from_file_location("run_gtex_positional_band_surgery", RUNNER)
assert SPEC is not None and SPEC.loader is not None
SURGERY = module_from_spec(SPEC)
SPEC.loader.exec_module(SURGERY)


def test_arm_tables_delete_and_replace_only_nominated_band():
    rng = np.random.default_rng(5)
    tuned = rng.normal(size=(66, 8))
    base = rng.normal(size=(66, 8))
    targets = [{"name": "gain", "bins": [10, 11, 12]}]
    controls = {
        "gain": [[3, 4, 5], [18, 19, 20], [25, 26, 27]],
    }
    tables, specs = SURGERY._arm_tables(tuned, base, targets, controls)
    assert len(tables) == 9
    assert set(tables) == set(specs)
    for name, table in tables.items():
        if name == "intact":
            continue
        audit = SURGERY._support_audit(tuned, table, specs[name]["bins"])
        assert audit["outside_band_pass"]
        assert audit["edited_is_real"]


def test_effect_arrays_use_intact_target_and_mean_of_three_controls():
    coupling = {
        "intact": np.asarray([3.0, 4.0]),
        "gain__delete": np.asarray([2.0, 2.0]),
        "gain__base_replace": np.asarray([2.5, 3.0]),
    }
    for index in range(1, 4):
        coupling[f"gain__control_{index}__delete"] = np.asarray([2.9, 3.9])
        coupling[f"gain__control_{index}__base_replace"] = np.asarray([3.0, 4.0])
    effects = SURGERY._effect_arrays(coupling, {"name": "gain"})
    assert effects["necessity"]["target"].tolist() == [1.0, 2.0]
    assert effects["base_band_rescue"]["target"].tolist() == [0.5, 1.0]
    assert effects["rewrite_contrast"]["target"].tolist() == [0.5, 1.0]
    assert effects["necessity"]["matched_control"] == pytest.approx([0.1, 0.1])
    assert effects["base_band_rescue"]["matched_control"] == pytest.approx([0.1, 0.1])
    assert effects["rewrite_contrast"]["matched_control"] == pytest.approx([0.0, 0.0])


def test_full_reconstruction_is_exact_at_float_precision():
    rng = np.random.default_rng(7)
    table = rng.normal(size=(66, 8))
    reconstructed = SURGERY._full_reconstruction(table)
    assert np.max(np.abs(reconstructed - table)) <= 1e-14


def test_accelerator_table_transfer_moves_to_cpu_before_float64():
    class TransferSpy:
        def __init__(self):
            self.events = []

        def detach(self):
            self.events.append("detach")
            return self

        def cpu(self):
            self.events.append("cpu")
            return self

        def to(self, *, dtype):
            assert self.events[-1] == "cpu"
            self.events.append(("to", dtype))
            return self

        def numpy(self):
            self.events.append("numpy")
            return np.zeros((1026, 512), dtype=np.float64)

    spy = TransferSpy()
    table = SURGERY._embedding_table_cpu_float64(spy)
    assert table.shape == (1026, 512)
    assert spy.events[0:2] == ["detach", "cpu"]
    assert spy.events[-1] == "numpy"
