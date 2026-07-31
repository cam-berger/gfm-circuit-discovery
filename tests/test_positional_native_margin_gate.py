from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "splicebert_gtex_finetune_shift"
    / "run_gtex_positional_native_margin_gate.py"
)
SPEC = spec_from_file_location("positional_native_margin_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_arm_family_has_frozen_phase_power_and_deletion_dose() -> None:
    rng = np.random.default_rng(14)
    tuned = rng.normal(size=(1026, 512))
    base = rng.normal(size=(1026, 512))

    tables, specs = MODULE._arm_tables(tuned, base)
    audits = MODULE._mechanical_audits(tuned, tables, specs)

    assert len(tables) == 13
    assert all(row["outside_band_pass"] for row in audits.values())
    phase_names = [
        name
        for name, spec in specs.items()
        if spec["intervention"] == "phase_control"
    ]
    assert len(phase_names) == 6
    assert all(audits[name]["band_power_pass"] for name in phase_names)
    assert all(audits[name]["dose_match_pass"] for name in phase_names)


def test_effect_arrays_implement_frozen_paired_contrasts() -> None:
    endpoint = "donor_margin"
    raw = {
        "intact": {endpoint: np.asarray([10.0, 20.0])},
        "period3__delete": {endpoint: np.asarray([7.0, 18.0])},
        "period3__half": {endpoint: np.asarray([8.5, 19.0])},
        "period3__base_replace": {endpoint: np.asarray([9.0, 19.5])},
        "period3__phase_all_positive": {
            endpoint: np.asarray([9.5, 19.0])
        },
        "period3__phase_alternating": {
            endpoint: np.asarray([9.0, 19.5])
        },
        "period3__phase_mixed": {endpoint: np.asarray([8.5, 20.0])},
    }

    effects = MODULE.effect_arrays(raw, "period3", endpoint)

    np.testing.assert_allclose(effects["deletion_damage"], [3.0, 2.0])
    np.testing.assert_allclose(effects["half_deletion_damage"], [1.5, 1.0])
    np.testing.assert_allclose(effects["base_band_rescue"], [2.0, 1.5])
    np.testing.assert_allclose(effects["base_rewrite"], [1.0, 0.5])
    np.testing.assert_allclose(effects["mean_phase_damage"], [1.0, 0.5])
    np.testing.assert_allclose(effects["deletion_minus_phase"], [2.0, 1.5])
