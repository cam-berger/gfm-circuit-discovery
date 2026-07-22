from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/splicebert/run_p3_phase_surgery_controls.py"
SPEC = importlib.util.spec_from_file_location("run_p3_phase_surgery_controls", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_position_weight_state_restores_exactly():
    embedding = torch.nn.Embedding(4, 3)
    original = embedding.weight.detach().clone()
    replacement = torch.zeros_like(original)
    with MODULE.position_weight_state(embedding, replacement):
        assert torch.equal(embedding.weight, replacement)
    assert torch.equal(embedding.weight, original)


def test_preflight_validation_rejects_hash_mismatch(tmp_path):
    cohort = tmp_path / "cohort.json"
    cohort.write_text("{}")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({
        "schema": "gfmcd.p3_phase_surgery.preflight.v1",
        "status": "DEVELOPMENT_OUTCOME_AWARE",
        "cohort_sha256": "wrong",
        "instrument": {},
    }))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        MODULE.validate_preflight(preflight, cohort)


def test_variant_grid_is_one_complete_cycle_with_two_blocks():
    class Example:
        example_id = "e"
        gene_id = "g"
        chrom = "chr1"
        strand = "+"

    record = {
        "donor_phase": 0,
        "blocks": {"cds": [[20, 56], [60, 96]], "upstream_intron": [[100, 136], [140, 176]]},
    }
    examples, specs, meta = MODULE.build_variant_grid([record], [Example()], prefix=1)
    assert len(examples) == len(specs) == len(meta) == 24
    assert {row["amount"] for row in meta} == {2, 3, 4}
    assert {row["context"] for row in meta} == {"cds", "upstream_intron"}
