"""Memorial: figure 1's Level-A retention ranges must equal the artifact's.

Why this exists
---------------
`make_fig1_summary.py` used to type its four Level-A retention numbers as string literals.
One of them drifted: the carrier-deleted row read ``"40-44%"`` while the controlling data
package `results/phase-lattice/fig_report_data.json` gives **27-44%** across the eight
Level-A heads -- five of the eight sit below 40%. The figure therefore contradicted
`figures/splicebert/EVIDENCE.md`, which carried the correct range, and overstated how much
attention structure survives carrier deletion.

The generator now derives the ranges. This memorial makes the class of drift FAIL rather
than sit undetected: it recomputes the ranges from the artifact and asserts the generator's
own formatter reproduces them. It is on the measurement path (it calls the function the
figure calls), and it can fail -- re-introducing a literal, or a data change, breaks it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "results/phase-lattice/fig_report_data.json"

# The range the artifact licenses, as of the C11/C13 Level-A screen. Pinned so that a change
# to the underlying data is a deliberate, visible edit rather than a silent re-render.
EXPECTED = {
    "kept_carrier": "27-44%",
    "kept_reciprocal": "89-97%",
    "kept_shuffled": "93-107%",
}
EXPECTED_N_LEVEL_A = 8


def _level_a_rows() -> list[dict]:
    rows = json.loads(DATA.read_text())["levelA"]["multispecies"]
    return [r for r in rows if r["LEVEL_A"]]


@pytest.mark.skipif(not DATA.exists(), reason="fig_report_data.json not present")
def test_level_a_head_count_is_eight() -> None:
    assert len(_level_a_rows()) == EXPECTED_N_LEVEL_A


@pytest.mark.skipif(not DATA.exists(), reason="fig_report_data.json not present")
@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_artifact_range_matches_pin(key: str) -> None:
    v = [r[key] for r in _level_a_rows()]
    assert f"{min(v) * 100:.0f}-{max(v) * 100:.0f}%" == EXPECTED[key]


@pytest.mark.skipif(not DATA.exists(), reason="fig_report_data.json not present")
def test_carrier_deleted_range_is_not_the_retracted_literal() -> None:
    """The specific defect: the floor is 27%, not 40%."""
    v = [r["kept_carrier"] for r in _level_a_rows()]
    assert min(v) < 0.40, "at least one Level-A head retains <40%; '40-44%' understates the collapse"
    assert sum(1 for x in v if x < 0.40) == 6


@pytest.mark.skipif(not DATA.exists(), reason="fig_report_data.json not present")
def test_generator_formatter_agrees_with_artifact() -> None:
    """The figure generator must derive these, not hard-code them.

    Checks the card rows specifically, so that prose mentioning the retracted literal (in a
    docstring or a correction note) does not trip the guard.
    """
    src = (REPO / "scripts/splicebert/make_fig1_summary.py").read_text()
    for row in ("Kept after carrier deleted", "Kept after reciprocal band", "Kept after content shuffle"):
        assert f'("{row}", kept_range(' in src, f"{row!r} must be derived, not a literal"
    assert "kept_range(" in src, "figure 1 must derive its retention ranges from the artifact"
