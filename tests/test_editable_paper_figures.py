from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pytest


REPO = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO / "figures/splicebert"
if str(FIGURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_DIR))

from editable_paper_figures import (  # noqa: E402
    audit_figure,
    default_core_layout,
    default_phase_layout,
    default_style,
    export_figure,
    load_core_data,
    load_phase_data,
    promote_to_manuscript,
    render_core_figure,
    render_phase_figure,
)


def test_notebook_renderers_use_verified_artifacts_and_final_column_width() -> None:
    style = default_style()
    core_data = load_core_data()
    phase_data = load_phase_data()
    core = render_core_figure(core_data, style=style, layout=default_core_layout())
    phase = render_phase_figure(phase_data, style=style, layout=default_phase_layout())
    try:
        core_audit = audit_figure(core, name="core")
        phase_audit = audit_figure(phase, name="phase")
        assert core_audit["width_in"] == pytest.approx(3.25)
        assert phase_audit["width_in"] == pytest.approx(3.25)
        assert core_audit["minimum_text_pt"] >= 6.0
        assert phase_audit["minimum_text_pt"] >= 6.0
        assert not core_audit["text_sizes_below_threshold"]
        assert not phase_audit["text_sizes_below_threshold"]
        assert len(core_data["sources"]) == 6
        assert len(phase_data["sources"]) == 2
    finally:
        plt.close(core)
        plt.close(phase)


def test_edit_export_writes_vector_raster_and_source_hashed_provenance(tmp_path: Path) -> None:
    style = default_style()
    layout = default_core_layout()
    data = load_core_data()
    figure = render_core_figure(data, style=style, layout=layout)
    try:
        outputs = export_figure(
            figure,
            name="core_preview",
            sources=data["sources"],
            claim_boundary="test boundary",
            style=style,
            layout=layout,
            output_dir=tmp_path,
            dpi=120,
        )
    finally:
        plt.close(figure)

    assert set(outputs) == {"pdf", "svg", "png", "provenance"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())
    provenance = json.loads(outputs["provenance"].read_text())
    assert provenance["schema"] == "gfmcd.figure_edit_provenance.v1"
    assert provenance["claim_boundary"] == "test boundary"
    assert len(provenance["sources"]) == 6
    assert all(len(source["sha256"]) == 64 for source in provenance["sources"])


def test_promotion_requires_deliberate_exact_confirmation() -> None:
    with pytest.raises(RuntimeError, match="promotion disabled"):
        promote_to_manuscript(core_outputs={}, phase_outputs={}, confirmation="")
