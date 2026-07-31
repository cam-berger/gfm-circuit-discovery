import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert_gtex_finetune_shift/run_gtex_coupling_shift.py"
)
SPEC = importlib.util.spec_from_file_location("run_gtex_coupling_shift", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_grouped_bootstrap_tracks_groups_and_is_deterministic():
    values = np.asarray([1.0, 3.0, 10.0])
    groups = np.asarray(["a", "a", "b"], dtype=object)
    first = MODULE.grouped_bootstrap(values, groups, n_boot=200, seed=4)
    second = MODULE.grouped_bootstrap(values, groups, n_boot=200, seed=4)
    assert first == second
    assert first["estimate"] == pytest.approx(14 / 3)
    assert first["n_groups"] == 2


def test_signflip_uses_add_one_and_detects_large_consistent_effect():
    groups = np.asarray([f"g{i}" for i in range(20)], dtype=object)
    values = np.ones(20)
    pvalue = MODULE.grouped_signflip_pvalue(
        values,
        groups,
        n_perm=999,
        seed=2,
    )
    assert pvalue >= 1 / 1000
    assert pvalue <= 0.01


def test_ratio_bootstrap_is_ratio_of_paired_means():
    groups = np.asarray(["a", "a", "b"], dtype=object)
    result = MODULE.grouped_ratio_bootstrap(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 4.0, 6.0]),
        groups,
        n_boot=100,
        seed=3,
    )
    assert result["estimate"] == pytest.approx(0.5)
    assert result["ci_low"] == pytest.approx(0.5)
    assert result["ci_high"] == pytest.approx(0.5)


def test_fine_model_parser_requires_complete_seed_family(tmp_path: Path):
    complete = [f"{seed}={tmp_path / str(seed)}" for seed in range(2020, 2025)]
    assert set(MODULE.parse_fine_models(complete)) == set(range(2020, 2025))
    with pytest.raises(ValueError):
        MODULE.parse_fine_models(complete[:-1])


def test_json_default_handles_numpy_scalars_without_stringifying():
    encoded = json.dumps(
        {
            "integer": np.int64(4),
            "floating": np.float32(0.5),
            "array": np.asarray([1, 2]),
        },
        default=MODULE.json_default,
    )
    assert json.loads(encoded) == {
        "integer": 4,
        "floating": 0.5,
        "array": [1, 2],
    }
