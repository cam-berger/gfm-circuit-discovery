#!/usr/bin/env python
"""Validate an all-head-screen output directory against its run_manifest (mechanical gates 8-11)."""
from __future__ import annotations

import argparse, hashlib, json, math, sys
from pathlib import Path


def sha(text): return hashlib.sha256(text.encode()).hexdigest()


def validate(out_dir: Path) -> dict:
    man = json.loads((out_dir / "run_manifest.json").read_text())
    fails, checked = [], 0
    for name, expected_sha in man["completed_shards"].items():
        p = out_dir / name
        if not p.exists():
            fails.append(f"{name}: missing"); continue
        body = p.read_text()
        if sha(body) != expected_sha:
            fails.append(f"{name}: sha mismatch")
        d = json.loads(body)
        for cond in ("B", "C", "D"):
            he = d["conditions"][cond]["head_effects"]
            if len(he) != 96:
                fails.append(f"{name}/{cond}: {len(he)}!=96 head records")
            lh = {(r["layer"], r["head"]) for r in he}
            if lh != {(l, h) for l in range(6) for h in range(16)}:
                fails.append(f"{name}/{cond}: layer/head coverage != L0H0..L5H15")
            if len(lh) != len(he):
                fails.append(f"{name}/{cond}: duplicate (layer,head)")
            if not all(math.isfinite(r["effect_margin"]) for r in he):
                fails.append(f"{name}/{cond}: non-finite effect_margin")
        checked += 1
    return {"checked": checked, "expected": len(man["expected_shards"]),
            "completed": len(man["completed_shards"]), "manifest_gate_failures": man.get("gate_failures", []),
            "validation_failures": fails, "ok": not fails and not man.get("gate_failures")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r = validate(Path(args.out))
    print(json.dumps({k: (v if k != "validation_failures" else v[:20]) for k, v in r.items()}, indent=2))
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
