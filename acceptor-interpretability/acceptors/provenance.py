"""Sibling `*.manifest.json` for every result file.

The repository standard is that a result is only interpretable alongside a
record of the conditions that produced it (`CLAUDE.md`: "Every result writes a
sibling `*.manifest.json`"). This arc shipped without one, which is part of why
its committed JSON, its figures, and its write-up turned out to be three
different unrecoverable runs.

What a manifest has to capture here, specifically:

* the **cohort**, by content — the GUE split is fetched over the network at run
  time, so a row count is not enough; we hash the exact sampled sequences.
* the **checkpoint revision**, so a re-upload cannot silently change the model.
* the **seed and the RNG discipline**, so the corruption draw is reconstructible
  (see `acceptors.data.sequence_rng`).
* enough **environment** to explain a numerical difference if one appears.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str):
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def cohort_digest(sequences) -> dict:
    """Content hash of the exact cohort, order-sensitive.

    Two runs agree on this iff they scored the same sequences in the same
    order — which is the thing a row count cannot tell you when the dataset is
    fetched from a remote that may change under you.
    """
    h = hashlib.sha256()
    n = 0
    for seq in sequences:
        h.update(seq.encode("ascii"))
        h.update(b"\n")
        n += 1
    return {"n": n, "sha256": h.hexdigest()}


def _namespace_to_plain(obj):
    if hasattr(obj, "__dict__"):
        return {k: _namespace_to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_namespace_to_plain(v) for v in obj]
    return obj


def write_run_manifest(result_path, *, script: str, config, sequences, model=None, extra=None):
    """Write `<result_path>.manifest.json` beside a result file.

    `result_path` is the result this manifest describes; the manifest lands at
    the same path with `.manifest.json` appended to the stem.
    """
    result_path = Path(result_path)
    manifest_path = result_path.with_suffix(".manifest.json")

    cfg = _namespace_to_plain(config)
    data_cfg = cfg.get("data", {})
    resolved_url = None
    if {"base_url", "dataset", "config", "split"} <= set(data_cfg):
        resolved_url = data_cfg["base_url"].format(
            dataset=data_cfg["dataset"], config=data_cfg["config"], split=data_cfg["split"]
        )

    manifest = {
        "result": result_path.name,
        "script": script,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(_git("status", "--porcelain"))},
        "cohort": {**cohort_digest(sequences), "source_url": resolved_url},
        "model": {
            "checkpoint": getattr(model, "checkpoint", None) or cfg.get("model", {}).get("checkpoint"),
            "revision": getattr(model, "revision", None) or cfg.get("model", {}).get("revision"),
            "device": str(getattr(model, "device", None)) if model is not None else None,
        },
        "randomness": {
            "seed": cfg.get("phase0", {}).get("seed"),
            "per_sequence_rng": "acceptors.data.sequence_rng (blake2b of 'seed:sequence')",
            "hash_randomization_independent": True,
        },
        "config": cfg,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }
    if extra:
        manifest["extra"] = extra

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"[manifest] wrote {manifest_path}")
    return manifest_path


def _package_versions():
    names = ["torch", "transformers", "multimolecule", "pandas", "numpy", "scipy"]
    out = {}
    for name in names:
        try:
            from importlib.metadata import version

            out[name] = version(name)
        except Exception:
            out[name] = None
    return out
