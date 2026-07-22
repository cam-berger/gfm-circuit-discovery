from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import pytest

from gfmcd.splicing.coupling import CouplingExample
from gfmcd.splicing.p3_closure import ContextBlock, rotate_cyclic
from gfmcd.splicing.p3_cohort import sequence_sha256


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/splicebert/run_p3_closure_preflight.py"
SPEC = importlib.util.spec_from_file_location("run_p3_closure_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_preflight_package_refuses_confirmation(tmp_path):
    path = tmp_path / "package.json"
    path.write_text(json.dumps({
        "schema": "gfmcd.p3_closure.cohort.v1",
        "model_outcomes_read": False,
        "records": [{"eligible": True, "role": "confirmation"}],
    }))
    with pytest.raises(RuntimeError, match="refuses"):
        MODULE.load_development_package(path)


def test_reconstruct_variant_verifies_hash():
    example = CouplingExample(
        example_id="x", sequence="AACCGGTT", acceptor_pos=2, donor_pos=4,
        chrom="chr1", strand="+", gene_id="g", split="train", exon_length=2,
        intron_length_down=10, n_isoforms_donor=1, n_isoforms_acceptor=1,
        gene_type="protein_coding", next_acceptor_pos=7,
    )
    block = ContextBlock(0, 4)
    expected = rotate_cyclic(example.sequence, block, 1, "left")
    row = {
        "block": [0, 4], "context": "cds", "amount": 1, "direction": "left",
        "sequence_sha256": sequence_sha256(expected),
    }
    assert MODULE.reconstruct_variant(example, row).sequence == expected
    row["sequence_sha256"] = "bad"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        MODULE.reconstruct_variant(example, row)
