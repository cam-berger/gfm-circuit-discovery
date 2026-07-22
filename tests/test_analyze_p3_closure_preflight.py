from __future__ import annotations
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/splicebert/analyze_p3_closure_preflight.py"
SPEC = importlib.util.spec_from_file_location("analyze_p3_closure_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_dominant_class_tracking_uses_modular_direction():
    source = {
        "native_rows": [{"example_id": "x", "phasor_class_masses": [1, 0, 0]}],
        "rows": [],
    }
    for context in ("cds", "upstream_intron"):
        for direction in ("left", "right"):
            sign = -1 if direction == "left" else 1
            for amount in range(2, 11):
                expected = (sign * amount) % 3
                masses = [0, 0, 0]
                masses[expected] = 1
                source["rows"].append({
                    "example_id": "x", "context": context, "direction": direction,
                    "amount": amount, "phasor_class_masses": masses,
                })
    result = MODULE.dominant_class_tracking(source)
    assert result["cds"]["left"]["2"]["content_following_expected_fraction"] == 1.0
    assert result["cds"]["right"]["4"]["content_following_expected_fraction"] == 1.0
