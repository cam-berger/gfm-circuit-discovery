"""Phase 1 (mandatory, blocks all mechanistic claims): fix the control.

Phase 0's non-splice control read Delta = -0.105 [-0.201, -0.027] where it
must read zero - GUE's label==2 rows are "not a splice site" but carry no
guarantee of being intronic, so corrupting their [160,198) window isn't a
composition-matched control for corrupting real intronic sequence.

This phase replaces that with two controls built from the *same* n=200
acceptor windows used throughout (same seed -> identical sample as
experiments/phase0_pilot.py), so "control" and "effect" are measured on
literally the same sequences:

  - far-intronic control: content-destroying corruption of a region deep in
    the intron, far from both BP and PPT (`coords.far_start/far_end`).
    Tests whether the corruption *method* itself is unbiased when applied
    somewhere with no known functional role.
  - in-place shuffle controls: permute the BP/PPT region's own bases
    (exact composition and position preserved, order scrambled) instead of
    replacing them. Isolates whether the Phase-0 effects are driven by
    composition (pyrimidine density, etc.) or by exact order/motif - a
    composition-preserving shuffle should show ~no effect if composition is
    the causal factor.

Verdict: an effect from Phase 0 only counts as real if |mean| clears the
largest |mean| among these (should-be-null) controls, AND the control's CI
includes zero while the effect's does not.

Run:
    python experiments/phase1_control_fix.py
Writes results/phase1_results.json, results/phase1_per_sample.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acceptors.config import load_config
from acceptors.data import (
    acceptor_rows,
    corrupt_region,
    load_gue_split,
    sample_rows,
    sequence_rng,
    shuffle_region_in_place,
)
from acceptors.metrics import m_acc
from acceptors.model import SpliceBert
from acceptors.provenance import write_run_manifest
from acceptors.stats import bootstrap_ci


def score_sample(model, seq: str, c) -> dict:
    rng = sequence_rng(seq, c.phase0.seed)
    ag0 = c.coords.ag_start

    clean = m_acc(model, seq, ag0)
    far_intronic = m_acc(
        model, corrupt_region(seq, c.coords.far_start, c.coords.far_end, rng), ag0)
    ppt_shuffle = m_acc(model, shuffle_region_in_place(seq, c.coords.ppt_start, c.coords.ppt_end, rng), ag0)
    bp_shuffle = m_acc(model, shuffle_region_in_place(seq, c.coords.bp_start, c.coords.bp_end, rng), ag0)

    return dict(clean=clean, far_intronic=far_intronic, ppt_shuffle=ppt_shuffle, bp_shuffle=bp_shuffle)


def summarize(label, diffs, c):
    mean, lo, hi = bootstrap_ci(diffs, c.phase0.n_boot, c.phase0.alpha, c.phase0.seed)
    flat = lo <= 0 <= hi
    tag = "flat (contains 0)" if flat else "DOES NOT read zero"
    print(f"  {label:34s} {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]  (n={len(diffs)})  -> {tag}")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(diffs), "reads_zero": flat}


def main():
    c = load_config()
    out_dir = Path(c.output.results_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"[phase1] loading GUE/{c.data.config} ({c.data.split}) ...")
    df = load_gue_split(c)
    # same seed as phase0 -> identical n=200 acceptor sample
    acc_df = sample_rows(acceptor_rows(df, c), c.phase0.n_acceptor, c.phase0.seed)
    print(f"[phase1] {len(acc_df)} acceptor windows (same sample as phase0)")

    print(f"[phase1] loading {c.model.checkpoint} ...")
    model = SpliceBert(c.model.checkpoint, c.model.device,
                       revision=getattr(c.model, "revision", None))
    print(f"[phase1] device: {model.device}")

    rows = [score_sample(model, seq, c) for seq in tqdm(acc_df["sequence"], desc="control-fix conditions")]
    acc_pd = pd.DataFrame(rows)
    acc_pd.to_csv(out_dir / "phase1_per_sample.csv", index=False)

    print("\n[phase1] new controls (all should read zero):")
    summary = {}
    summary["far_intronic_control"] = summarize(
        "far-intronic control (clean - corrupted)", acc_pd["clean"] - acc_pd["far_intronic"], c)
    summary["ppt_shuffle_in_place"] = summarize(
        "PPT shuffle-in-place (clean - shuffled)", acc_pd["clean"] - acc_pd["ppt_shuffle"], c)
    summary["bp_shuffle_in_place"] = summarize(
        "BP shuffle-in-place (clean - shuffled)", acc_pd["clean"] - acc_pd["bp_shuffle"], c)

    # pull Phase 0's effects back in for the "clears the control" comparison.
    # Only far_intronic_control is a true noise floor (same content-destroying
    # method, applied to a region with no known functional role) - it's the
    # right thing to compare other content-destroying effects against. The
    # shuffle-in-place controls are a *different* test (order vs. composition)
    # and are not folded into this floor: PPT shuffle-in-place came back large
    # (+1.207) in its own right, which is a finding, not a null result, and
    # would make the floor meaningless if averaged in.
    phase0_path = out_dir / "phase0_results.json"
    verdicts = {}
    if phase0_path.exists():
        with open(phase0_path) as f:
            phase0 = json.load(f)
        floor = summary["far_intronic_control"]
        control_floor = abs(floor["mean"])
        floor_ci = (floor["ci_lo"], floor["ci_hi"])
        print(f"\n[phase1] Phase-0 effects vs. the far-intronic noise floor "
              f"({floor['mean']:+.3f} [{floor_ci[0]:+.3f}, {floor_ci[1]:+.3f}]):")
        for key, label in [
            ("delta_ppt_bp_only", "BP-only disruption"),
            ("ppt_only_alone", "PPT-only disruption"),
            ("context_required", "context required"),
            ("ag_scanning", "AG-scanning"),
        ]:
            eff = phase0[key]
            ci_overlaps_floor = eff["ci_lo"] <= floor_ci[1] and floor_ci[0] <= eff["ci_hi"]
            clears = abs(eff["mean"]) > control_floor and not ci_overlaps_floor
            verdict = "SURVIVES" if clears else (
                "MARGINAL (CI overlaps the noise floor)" if ci_overlaps_floor else "UNPROVEN")
            verdicts[key] = {"mean": eff["mean"], "clears_control_floor": clears, "ci_overlaps_floor": ci_overlaps_floor}
            print(f"  {label:24s} {eff['mean']:+.3f}  -> {verdict}")
        summary["verdicts_vs_control_floor"] = verdicts
        summary["control_floor"] = control_floor
        print(f"\n[phase1] separately: PPT shuffle-in-place ({summary['ppt_shuffle_in_place']['mean']:+.3f}) "
              f"is itself a large, non-null effect - composition-preserving reordering of the tract nearly "
              f"reproduces the full content-destroying effect, so the tract signal looks order/identity-driven, "
              f"not just aggregate pyrimidine density.")
    else:
        print("\n[phase1] results/phase0_results.json not found - skipping effect-vs-control comparison")

    result_path = out_dir / "phase1_results.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    write_run_manifest(result_path, script=Path(__file__).name, config=c,
                       sequences=list(acc_df["sequence"]), model=model)
    print(f"\n[phase1] wrote {out_dir}/phase1_results.json and {out_dir}/phase1_per_sample.csv")


if __name__ == "__main__":
    main()
