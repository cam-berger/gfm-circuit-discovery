"""Phase 2 (mandatory, separates gating from artifact): absolute m_acc vs. the floor.

Phase 0 only reported *differences* (Delta = clean - corrupted). A Delta can
look like "the effect vanished" for two very different reasons: (a) a real
interaction/gate, or (b) the margin already hit its floor (the random-AG
score) once the dominant driver (the tract) was removed, so there's nothing
left to lose. Absolute m_acc, compared directly to the random-AG floor,
distinguishes them.

This reuses results/phase0_per_sample_acceptor.csv - clean, bp_only,
ppt_only, both, and rand_ag are already there as *absolute* m_acc per
sequence (Phase 0 only took differences of them at the end), so no model
re-run is needed here.

Run:
    python experiments/phase2_absolute_floor.py
Writes results/phase2_results.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acceptors.stats import bootstrap_ci
from acceptors.config import load_config
from acceptors.provenance import write_run_manifest

CONDITIONS = [
    ("clean", "clean (intact BP + PPT)"),
    ("bp_only", "BP disrupted"),
    ("ppt_only", "PPT disrupted"),
    ("both", "BP + PPT both disrupted"),
    ("rand_ag", "random-AG floor"),
]


def _sha256(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ci_overlap(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def main():
    c = load_config()
    out_dir = Path(c.output.results_dir)
    upstream = out_dir / "phase0_per_sample_acceptor.csv"
    if not upstream.exists():
        raise SystemExit(
            f"[phase2] {upstream} not found. Phase 2 derives from Phase 0's per-sample "
            "absolute m_acc and runs no model of its own - run phase0_pilot.py first.")
    acc_pd = pd.read_csv(upstream)

    print("[phase2] absolute m_acc by condition (mean, 95% bootstrap CI):")
    summary = {}
    cis = {}
    for key, label in CONDITIONS:
        mean, lo, hi = bootstrap_ci(acc_pd[key], c.phase0.n_boot, c.phase0.alpha, c.phase0.seed)
        summary[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(acc_pd)}
        cis[key] = (lo, hi)
        print(f"  {label:28s} {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    floor_ci = cis["rand_ag"]
    print("\n[phase2] verdicts vs. the random-AG floor:")
    verdicts = {}
    for key, label in CONDITIONS[1:4]:
        overlaps = ci_overlap(cis[key], floor_ci)
        above = cis[key][0] > floor_ci[1]
        if above:
            verdict = "clearly ABOVE floor - retains real acceptor signal"
        elif overlaps:
            verdict = "AT floor - indistinguishable from no-context baseline"
        else:
            verdict = "BELOW floor (unexpected - inspect)"
        verdicts[key] = {"overlaps_floor": overlaps, "clearly_above_floor": above, "verdict": verdict}
        print(f"  {label:28s} -> {verdict}")

    gate_real = verdicts["bp_only"]["clearly_above_floor"] and verdicts["both"]["overlaps_floor"]
    print(f"\n[phase2] AND-gate vs. floor-artifact call: "
          f"{'GATE IS PLAUSIBLE (BP-only signal survives above floor; both collapses to floor)' if gate_real else 'NOT the AND-gate signature - see per-condition verdicts above'}")

    summary["verdicts"] = verdicts
    summary["gate_plausible"] = gate_real

    result_path = out_dir / "phase2_results.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    # No model run here: Phase 2 is a re-analysis of Phase 0's per-sample file,
    # so its provenance is that file's identity, not a fresh cohort draw.
    write_run_manifest(result_path, script=Path(__file__).name, config=c,
                       sequences=[], model=None,
                       extra={"derived_from": upstream.name,
                              "derived_from_sha256": _sha256(upstream),
                              "model_run": False})


if __name__ == "__main__":
    main()
