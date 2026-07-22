import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/splicebert/analyze_p3_shift_mod3.py"
SPEC = importlib.util.spec_from_file_location("analyze_p3_shift_mod3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_relative_to_absolute_mod3_is_exact_rotation():
    rel = np.array([[0.7, 0.2, 0.1], [0.7, 0.2, 0.1], [0.7, 0.2, 0.1]])
    got = MODULE.absolute_mass(rel, np.array([0, 1, 2]))
    assert np.allclose(got[0], [0.7, 0.2, 0.1])
    assert np.allclose(got[1], [0.1, 0.7, 0.2])
    assert np.allclose(got[2], [0.2, 0.1, 0.7])
