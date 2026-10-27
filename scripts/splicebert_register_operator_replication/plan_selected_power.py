#!/usr/bin/env python3
"""Recompute sign-free resolution on an already fixed sequence-only ID selection.

This does not select blocks, change a protocol, or compute any new model output.
The fixed-stratum 20k-draw arithmetic is transcribed from the C43 inference rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from plan_power import MDE_FACTOR, REPO, variance_summary
from gfmcd.artifact_guard import guard_output_path
from gfmcd.provenance import ExperimentManifest


def selected_resolution(audit: dict, selection: dict) -> dict:
    ids = selection["block_ids"]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("selection requires nonempty unique IDs")
    lookup = {r["block_id"]: r for r in audit["recomputed"]["target"]["block_endpoints"]}
    if not set(ids) <= lookup.keys():
        raise ValueError("selection contains an unknown historical block")
    # The historical inference consumes lexically sorted whole-block rows.
    rows = [lookup[i] for i in sorted(ids)]
    strata = np.asarray([r["stratum"] for r in rows], dtype=object)
    groups = [np.flatnonzero(strata == s) for s in sorted(set(strata.tolist()))]
    if any(len(g) < 2 for g in groups):
        raise ValueError("each bootstrap stratum needs at least two independent blocks")
    centered = np.asarray([r["ac"]["psi_match"] for r in rows], dtype=float)
    if not np.isfinite(centered).all():
        raise ValueError("nonfinite historical value")
    for group in groups:
        centered[group] -= centered[group].mean()
    rng = np.random.default_rng(20260803)
    indices = np.concatenate([rng.choice(g, size=(20000, len(g)), replace=True)
                              for g in groups], axis=1)
    mde = float(MDE_FACTOR * centered[indices].mean(axis=1).std(ddof=1))
    return {
        "schema": "gfmcd.register_operator_replication.selected_power.v1",
        "status": "HISTORICAL_VARIANCE_PLANNING_ONLY",
        "new_model_outcomes": 0, "model_loaded": False,
        "selection_modified": False,
        "selection_block_ids_sha256": selection.get("block_ids_sha256"),
        "ordered_ids_newline_sha256": hashlib.sha256(
            ("\n".join(sorted(ids)) + "\n").encode()).hexdigest(),
        "variance": variance_summary(rows),
        "stratum_counts": dict(Counter(strata.tolist())),
        "bootstrap_draws": 20000, "bootstrap_seed": 20260803,
        "historical_ac_linear_mde80": mde,
        "would_pass_resolution_if_cc_max_had_this_variance": mde <= 0.05,
        "maximum_sd_ratio_for_pass_under_sqrt_scaling": 0.05 / mde,
        "scope": "Prospective planning proxy only. Actual CC/max variance is unknown.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = guard_output_path(args.out)
    manifest_path = guard_output_path(out.with_suffix(".manifest.json"))
    result = selected_resolution(json.loads(args.audit.read_text()),
                                 json.loads(args.selection.read_text()))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    manifest = ExperimentManifest.create(
        "splicebert-register-operator-replication:selected-design-power", sys.argv,
        repo=REPO, status=result["status"], seeds={"bootstrap": 20260803},
        config={"draws": 20000, "numpy": np.__version__, "model_loaded": False},
        notes=["Selected IDs were fixed by sequence hygiene before this variance calculation.",
               "No signed target estimate is reported or used for selection."],
    )
    for p in (args.audit, args.selection, Path(__file__),
              Path(__file__).with_name("plan_power.py"), REPO / "src/gfmcd/provenance.py"):
        manifest.add_input(p)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({"out": str(out), "n": result["variance"]["blocks"],
                      "historical_ac_linear_mde80": result["historical_ac_linear_mde80"]}))


if __name__ == "__main__":
    main()
