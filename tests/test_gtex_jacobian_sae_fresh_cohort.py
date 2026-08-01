import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO / "scripts/splicebert_gtex_finetune_shift/build_gtex_jacobian_sae_fresh_cohort.py"
)
SPEC = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def test_gene_root_removes_only_version_suffix():
    assert RUNNER.gene_root("ENSG000001.12") == "ENSG000001"
    assert RUNNER.gene_root("ENSG000001") == "ENSG000001"


def test_stable_order_key_is_deterministic_and_example_specific():
    first = RUNNER.stable_order_key("ENSG000001.1", "example-a")
    assert first == RUNNER.stable_order_key("ENSG000001.9", "example-a")
    assert first != RUNNER.stable_order_key("ENSG000001.1", "example-b")
