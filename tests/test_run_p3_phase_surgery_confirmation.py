from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert/run_p3_phase_surgery_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location("run_p3_phase_surgery_confirmation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_variant_grid_is_exact_frozen_primary_grid():
    record = {
        "example_id": "e",
        "gene_id": "g",
        "chrom": "chr1",
        "strand": "+",
        "donor_phase": 0,
        "target_context": "cds",
        "blocks": {
            "cds": [[20, 56], [80, 116]],
            "upstream_intron": [[200, 236], [260, 296]],
        },
    }
    examples, specs, metadata = MODULE.variant_grid(
        record, object(), amounts=list(range(2, 11))
    )
    assert len(examples) == len(specs) == len(metadata) == 72
    assert {row["amount"] for row in metadata} == set(range(2, 11))
    assert {row["direction"] for row in metadata} == {"left", "right"}
    assert {row["context"] for row in metadata} == {"cds", "upstream_intron"}


def test_atomic_json_is_resumable_but_immutable(tmp_path):
    path = tmp_path / "shard.json"
    MODULE.atomic_json(path, {"x": 1})
    MODULE.atomic_json(path, {"x": 1})
    import pytest

    with pytest.raises(RuntimeError, match="collision"):
        MODULE.atomic_json(path, {"x": 2})
