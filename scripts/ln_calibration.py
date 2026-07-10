"""Measure the frozen-LayerNorm faithfulness gap and write the artifact (TO_FIX.md §16).

    uv run python scripts/ln_calibration.py

Writes results/ln_calibration/ln_faithfulness.json (provenance-stamped) and prints the table that
goes into docs/RESULTS.md. Closes the last Gate-0 hole with a measured number, not an assumption.
"""

from __future__ import annotations

import json

from gfmcd import provenance
from gfmcd.paths import project_paths
from gfmcd.tracer.layernorm import calibrate


def main() -> None:
    report = calibrate()
    stamped = provenance.stamp(report, status="SETTLED", seed=report["config"]["seed"], script="ln_calibration.py")

    out_dir = project_paths().results / "ln_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ln_faithfulness.json"
    path.write_text(json.dumps(stamped, indent=2))

    for label in ("ln_free", "layernorm"):
        r = report[label]
        print(f"\n== {label} ==")
        print(f"  conservation max err : {r['conservation_max_err']:.3e}")
        print(f"  OV edge recall/prec  : {r['edges']['recall']:.3f} / {r['edges']['precision']:.3f}")
        print("  frozen-LN gap by alpha:")
        for row in r["gap_curve"]:
            print(f"    alpha={row['alpha']:.2f}  gap={row['faithfulness_gap']:.4f}  |true effect|={row['true_effect_norm']:.3e}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
