from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUILDER = (
    REPO
    / "studies/splicebert_head_context_atlas/colab_atlas_package/build_stage2_overlay.py"
)
SCRIPT_ROOT = REPO / "scripts/splicebert_head_context_atlas"


def test_stage2_overlay_contains_sibling_import_closure_and_no_target_outcomes(tmp_path):
    output = tmp_path / "stage2-overlay.zip"
    subprocess.run(
        [sys.executable, str(BUILDER), "--out", str(output)],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        prefix = "scripts/splicebert_head_context_atlas/"
        packaged_modules = {
            Path(name).stem for name in names if name.startswith(prefix) and name.endswith(".py")
        }
        local_modules = {path.stem for path in SCRIPT_ROOT.glob("*.py")}
        for name in sorted(names):
            if not name.startswith(prefix) or not name.endswith(".py"):
                continue
            tree = ast.parse(archive.read(name), filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in local_modules:
                    assert node.module in packaged_modules, (
                        f"{name} imports sibling {node.module}.py, but the overlay omits it"
                    )

        health = [
            name
            for name in names
            if "/health_shards/packet_" in name and name.endswith(".json")
        ]
        assert len(health) == 256
        assert not any("/target_shards/" in name for name in names)
        assert not any(name.endswith("run_manifest.target.json") for name in names)
        manifest = json.loads(archive.read("STAGE2_OVERLAY_MANIFEST.json"))
        assert manifest["schema"] == "head_context_atlas.stage2_colab_overlay/6"
        assert (
            "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A3.md"
            in names
        )
        assert "scripts/splicebert_head_context_atlas/build_atlas_representative_dags.py" in names
        assert "scripts/splicebert_head_context_atlas/validate_atlas_representative_dags.py" in names
        assert "scripts/splicebert/export_segment_attribution_bundle.py" in names

        stage2_package = json.loads(
            archive.read(
                "results/splicebert_head_context_atlas/atlas_stage2_package_v1.json"
            )
        )
        for reference in stage2_package["whole_validation_row_references"].values():
            relative = reference["path"]
            assert relative in names, f"overlay omits runtime reference {relative}"
            payload = archive.read(relative)
            assert len(payload) == reference["size_bytes"]
            assert hashlib.sha256(payload).hexdigest() == reference["sha256"]

    sidecar = json.loads(output.with_suffix(".zip.sha256.json").read_text())
    assert sidecar["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert sidecar["n_files"] == len(names)
