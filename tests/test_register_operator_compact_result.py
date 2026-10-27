"""The compact report must retain every gate and agree with the original audited record."""
import importlib.util
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_unabridged_execution_report_preserves_original_bytes():
    path = ROOT / "studies/splicebert_register_operator_replication/RESULTS_DETAIL.md"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "0515cca8bcbd245786344b0efa0923f27e665bf525f2001ce9c981f6042fb4fd"
    )


def test_compact_exports_equal_original_audited_sources():
    path = ROOT / "scripts/splicebert_register_operator_replication/export_compact_result.py"
    spec = importlib.util.spec_from_file_location("compact_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = ROOT / "results/splicebert_register_operator_replication/canonical"
    for name, document in module.build().items():
        assert json.loads((directory / name).read_text()) == document


def test_registered_resolution_failure_is_not_positive_evidence():
    path = ROOT / "results/splicebert_register_operator_replication/canonical/summary_v2.json"
    summary = json.loads(path.read_text())
    assert summary["classification"] == "INSTRUMENT_UNRESOLVED"
    assert summary["resolution_mde80_limit"] == 0.05
    assert summary["failed_gates"] == ["resolution"]
    assert len(summary["gates"]) == 10 and sum(summary["gates"].values()) == 9
    assert summary["preserved_v1"]["directional_noncoverage"] == 83
    assert summary["new_model_forwards"] == 0 and summary["threshold_changed"] is False
