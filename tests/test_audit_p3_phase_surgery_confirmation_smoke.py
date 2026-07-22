from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert/audit_p3_phase_surgery_confirmation_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("audit_p3_phase_surgery_confirmation_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_row_key_ignores_outcome_values():
    row = {
        "example_id": "e",
        "context": "cds",
        "block_index": 0,
        "direction": "left",
        "amount": 3,
        "coupling": 0.1,
    }
    changed = {**row, "coupling": 999.0}
    assert MODULE.row_key(row) == MODULE.row_key(changed)
