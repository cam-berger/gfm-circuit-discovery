#!/usr/bin/env python3
"""Sign-free planning only; never load a checkpoint or compute a new model outcome.

The exact empirical fixed-stratum bootstrap variance is sum(centered x**2)/N**2.
Keep the historical finite-bootstrap gate and its verdict untouched. Projections here
assume the new operator has the stated SD multiplier and the same sampling proportions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402

MDE_FACTOR = 1.96 + 0.8416212335729143


def variance_summary(rows: list[dict]) -> dict:
    """Discard stratum means; retain no signed target estimate in the output."""
    if len(rows) < 2 or len({r["block_id"] for r in rows}) != len(rows):
        raise ValueError("need at least two uniquely identified blocks")
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row["ac"]["psi_match"])
        if not math.isfinite(value):
            raise ValueError("nonfinite historical block value")
        groups[str(row["stratum"])].append(value)
    n = len(rows)
    df = n - len(groups)
    if df <= 0:
        raise ValueError("no within-stratum variance degrees of freedom")
    sse = math.fsum(
        (v - math.fsum(values) / len(values)) ** 2
        for values in groups.values() for v in values
    )
    se = math.sqrt(sse) / n
    return {
        "blocks": n, "strata": len(groups), "residual_df": df,
        "within_stratum_sse": sse, "residual_sd": math.sqrt(sse / df),
        "exact_empirical_bootstrap_se": se,
        "exact_empirical_bootstrap_mde80": MDE_FACTOR * se,
    }


def project_mde(reference: float, reference_n: int, n: int, sd_multiplier: float) -> float:
    if min(reference_n, n) < 2 or not math.isfinite(sd_multiplier) or sd_multiplier <= 0:
        raise ValueError("invalid count or SD multiplier")
    if not math.isfinite(reference) or reference < 0:
        raise ValueError("invalid reference MDE")
    return reference * math.sqrt(reference_n / n) * sd_multiplier


def build_plan(audit: dict, counts: list[int]) -> dict:
    target = audit["recomputed"]["target"]
    summary = variance_summary(target["block_endpoints"])
    recorded = float(target["primary_psi"]["mde80"])
    multipliers = [1.0, 1.1, 1.2, 1.3]
    return {
        "schema": "gfmcd.register_operator_replication.power_plan.v1",
        "status": "DESIGN_ONLY_NOT_EXECUTION_AUTHORIZATION",
        "new_model_outcomes": 0, "model_loaded": False,
        "historical_variance": summary,
        "historical_recorded_mde80": recorded,
        "historical_gate_unchanged": True,
        "resolution_limit": 0.05,
        "sd_multipliers": multipliers,
        "projections": [{
            "blocks": n, "genes_required": 7 * n,
            "equal_six_permutation_balance_possible": n % 6 == 0,
            "recorded_mde_scaling": [project_mde(recorded, summary["blocks"], n, r)
                                     for r in multipliers],
            "analytic_mde_scaling": [project_mde(
                summary["exact_empirical_bootstrap_mde80"], summary["blocks"], n, r
            ) for r in multipliers],
        } for n in counts],
        "minimum_multiple_six_blocks": [{
            "sd_multiplier": r,
            "recorded_mde_scaling": 6 * math.ceil(
                summary["blocks"] * (recorded * r / 0.05) ** 2 / 6),
            "analytic_mde_scaling": 6 * math.ceil(summary["blocks"] * (
                summary["exact_empirical_bootstrap_mde80"] * r / 0.05) ** 2 / 6),
        } for r in multipliers],
        "limits": [
            "AC/linear residual variance is a planning proxy, not CC/max-margin evidence.",
            "No target sign or magnitude is used for selection or threshold tuning.",
            "Square-root scaling assumes stable stratum proportions and variance.",
            "This projects the scalar resolution gate, not joint power of all decision criteria.",
            "No sample size, gate, historical verdict, or execution authorization is changed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+",
                        default=[342, 348, 350, 366, 384, 420, 480, 576, 768, 960])
    args = parser.parse_args()
    out = guard_output_path(args.out)
    manifest_path = guard_output_path(out.with_suffix(".manifest.json"))
    plan = build_plan(json.loads(args.audit.read_text()), args.blocks)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    manifest = ExperimentManifest.create(
        "splicebert-register-operator-replication:design-power", sys.argv,
        repo=REPO, status="DESIGN_ONLY_NOT_EXECUTION_AUTHORIZATION",
        config={"checkpoint_loaded": False, "new_model_outcomes": 0,
                "blocks": args.blocks, "resolution_limit": 0.05},
        notes=["Deterministic analytic calculation; no RNG seed or new data/model call.",
               "Historical AC/linear values are used only after within-stratum centering."],
    )
    for path in (args.audit, Path(__file__), REPO / "src/gfmcd/provenance.py"):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({"out": str(out), "status": plan["status"],
                      "historical_variance": plan["historical_variance"]}, indent=2))


if __name__ == "__main__":
    main()
