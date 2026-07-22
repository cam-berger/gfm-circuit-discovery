from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert/analyze_p3_phase_surgery_preflight.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_p3_phase_surgery_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_carrier_class_tracking_uses_surgery_direction_and_modulo_three():
    source = {
        "native_rows": [{"example_id": "x", "phasor_class_masses": [1, 0, 0]}],
        "rows": [],
    }
    for context in ("cds", "upstream_intron"):
        for direction in ("left", "right"):
            sign = 1 if direction == "left" else -1
            for amount in range(2, 11):
                expected = (sign * amount) % 3
                masses = [0, 0, 0]
                masses[expected] = 1
                source["rows"].append({
                    "example_id": "x",
                    "context": context,
                    "direction": direction,
                    "amount": amount,
                    "phasor_class_masses": masses,
                })
    result = MODULE.carrier_class_tracking(source)
    assert result["cds"]["left"]["2"]["carrier_shift_expected_fraction"] == 1.0
    assert result["cds"]["right"]["4"]["carrier_shift_expected_fraction"] == 1.0
    assert result["cds"]["left"]["3"]["expected_modulo_class_delta"] == 0


def test_phase_strata_are_descriptive_and_allow_empty_phase():
    source = {"native_rows": [{"gene_id": "g1", "donor_phase": 1}]}
    per_gene = ({"gene_id": "g1", "psi_coupling": 0.2, "psi_phasor_signed": -0.1},)
    result = MODULE.phase_strata(per_gene, source)
    assert result["1"]["mean_psi_coupling"] == 0.2
    assert result["0"]["mean_psi_coupling"] is None
    assert result["2"]["status"] == "exploratory_development_only"
