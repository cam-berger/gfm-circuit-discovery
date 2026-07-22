from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert/audit_p3_phase_surgery_confirmation_package.py"
)
SPEC = importlib.util.spec_from_file_location("audit_p3_phase_surgery_confirmation_package", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_verify_file_record_detects_content_change(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("frozen")
    from gfmcd.provenance import file_record

    record = file_record(path)
    assert MODULE.verify_file_record(record)
    path.write_text("changed")
    assert not MODULE.verify_file_record(record)
