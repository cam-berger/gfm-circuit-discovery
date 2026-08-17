"""Memorial for R47's bounded roll-attention padding defect."""

from __future__ import annotations

import ast
import inspect
import warnings
from pathlib import Path

from gfmcd.splicing import coupling


REPO = Path(__file__).resolve().parents[1]


def _roll_callers() -> list[str]:
    callers: list[str] = []
    for root in (REPO / "src", REPO / "scripts"):
        for path in root.rglob("*.py"):
            source = path.read_text()
            if "roll_attention" not in source:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else None
                )
                if name == "roll_attention":
                    callers.append(path.relative_to(REPO).as_posix())
    return sorted(callers)


def test_roll_attention_remains_scoped_to_fixed_length_unpadded_caller() -> None:
    assert _roll_callers() == ["scripts/splicebert/run_comb_shift.py"]
    assert coupling.CouplingConfig().window_length == 1024
    builder = inspect.getsource(coupling.build_coupling_dataset)
    assert "if len(seq) != N" in builder

    runner = (REPO / "scripts/splicebert/run_comb_shift.py").read_text()
    assert "load_examples(" in runner
    assert "SEQLEN = len(X[0].sequence)" in runner


def test_r47_records_padding_defect_and_nonretraction_scope() -> None:
    corrections = (REPO / "studies/splicebert_partner_selection/CORRECTIONS.md").read_text()
    section = corrections.split("## R47", 1)[1]
    assert "not padding-safe" in section
    assert "does **not** retract the recorded comb-shift result" in section
    assert "variable-length caller" in section
