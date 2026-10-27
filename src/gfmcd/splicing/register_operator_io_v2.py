"""Version barrier for A1; v1 verification deliberately rejects FROZEN_V2."""
from pathlib import Path

from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_io import read_json, verify_freeze as verify_v1
from gfmcd.splicing.register_operator_statistics_v2 import METHOD

BASE_FREEZE_SHA256 = "82bc168e2246dff82519215203e02747f0209c289fa00c3e538613d95573d854"


def verify_freeze(path, repo) -> dict:
    freeze = read_json(path)
    if (freeze.get("status") != "FROZEN_V2"
            or freeze.get("schema") != "gfmcd.register_operator.freeze.v2"
            or freeze.get("inference_method") != METHOD
            or freeze.get("base_freeze_sha256") != BASE_FREEZE_SHA256):
        raise ValueError("execution requires the separately frozen A1 v2 instrument")
    base_path = Path(freeze["base_freeze_path"])
    if sha256_file(base_path) != BASE_FREEZE_SHA256:
        raise ValueError("immutable v1 parent freeze changed")
    verify_v1(base_path, repo)
    for name, expected in freeze["files"].items():
        source = Path(name)
        if not source.is_absolute():
            source = Path(repo) / source
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"frozen v2 file changed or missing: {source}")
    return freeze
