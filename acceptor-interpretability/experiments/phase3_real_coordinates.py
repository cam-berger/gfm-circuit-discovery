"""Phase 3: re-locate the branch point and tract per-sequence, not by fixed offset.

Phase 0/1's BP=[160,183)/PPT=[183,198) windows are a single fixed-offset
guess applied to every sequence. If real branch points don't sit at a
uniform distance from the AG, that guess mislocates the BP window in most
sequences - which alone could explain why the branch-point effect measured
so much smaller than the tract effect (we may mostly be corrupting the
wrong 23nt).

This locates, per sequence, the best-scoring candidate branch-point
adenosine in the canonical search window (18-45nt upstream of the AG)
against a "yUnAy" consensus scorer (`acceptors.data.locate_branch_point` -
a documented stand-in for branchpointer/LaBranchoR, which need either
genomic coordinates GUE doesn't carry or a separate pretrained model; see
README), then derives the tract as the region between that branch point and
the AG. Both regions now vary in position (and the tract in length) per
sequence, instead of being the same 23nt/15nt window for everyone.

Run:
    python experiments/phase3_real_coordinates.py
Writes results/phase3_results.json, results/phase3_per_sample.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd
import random as pyrandom
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acceptors.config import load_config
from acceptors.data import (
    acceptor_rows,
    corrupt_region,
    load_gue_split,
    locate_branch_point,
    locate_tract,
    sample_rows,
)
from acceptors.metrics import m_acc
from acceptors.model import SpliceBert
from acceptors.stats import bootstrap_ci

BP_LEFT, BP_RIGHT = 2, 3  # BP corruption window: [bp_pos-BP_LEFT, bp_pos+BP_RIGHT)
FALLBACK_BP_OFFSET = 30   # used only if no 'A' found in the search window (rare)


def score_sample(model, seq: str, c) -> dict:
    rng = pyrandom.Random(hash(seq) & 0xFFFFFFFF)
    ag0 = c.coords.ag_start

    bp_pos = locate_branch_point(seq, ag0, c.coords.bp_search_from, c.coords.bp_search_to)
    fallback = bp_pos is None
    if fallback:
        bp_pos = ag0 - FALLBACK_BP_OFFSET

    bp_start, bp_end = max(0, bp_pos - BP_LEFT), bp_pos + BP_RIGHT
    ppt_start, ppt_end = locate_tract(seq, bp_pos, ag0, min_gap=BP_RIGHT)

    clean = m_acc(model, seq, ag0)
    bp_real = m_acc(model, corrupt_region(seq, bp_start, bp_end, rng), ag0)
    ppt_real = m_acc(model, corrupt_region(seq, ppt_start, ppt_end, rng), ag0)
    both_real = m_acc(model, corrupt_region(seq, bp_start, ppt_end, rng), ag0)

    return dict(
        clean=clean, bp_real=bp_real, ppt_real=ppt_real, both_real=both_real,
        bp_pos=bp_pos, bp_distance_from_ag=ag0 - bp_pos, ppt_len=ppt_end - ppt_start,
        fallback=fallback,
    )


def summarize(label, diffs, c):
    mean, lo, hi = bootstrap_ci(diffs, c.phase0.n_boot, c.phase0.alpha, c.phase0.seed)
    print(f"  {label:34s} {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]  (n={len(diffs)})")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(diffs)}


def main():
    c = load_config()
    out_dir = Path(c.output.results_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"[phase3] loading GUE/{c.data.config} ({c.data.split}) ...")
    df = load_gue_split(c)
    acc_df = sample_rows(acceptor_rows(df, c), c.phase0.n_acceptor, c.phase0.seed)
    print(f"[phase3] {len(acc_df)} acceptor windows (same sample as phase0/phase1)")

    print(f"[phase3] loading {c.model.checkpoint} ...")
    model = SpliceBert(c.model.checkpoint, c.model.device)
    print(f"[phase3] device: {model.device}")

    rows = [score_sample(model, seq, c) for seq in tqdm(acc_df["sequence"], desc="real-coordinate conditions")]
    acc_pd = pd.DataFrame(rows)
    acc_pd.to_csv(out_dir / "phase3_per_sample.csv", index=False)

    n_fallback = int(acc_pd["fallback"].sum())
    print(f"\n[phase3] branch-point localization: {n_fallback}/{len(acc_pd)} sequences had no candidate "
          f"'A' in the search window (fallback to fixed offset)")
    print(f"[phase3] located BP distance from AG: mean {acc_pd['bp_distance_from_ag'].mean():.1f}nt "
          f"(search window was fixed {c.coords.bp_search_from}-{c.coords.bp_search_to}nt)")
    print(f"[phase3] located tract length: mean {acc_pd['ppt_len'].mean():.1f}nt "
          f"(fixed-offset design used a constant {c.coords.ppt_end - c.coords.ppt_start}nt)")

    print("\n[phase3] real-coordinate effects vs. Phase 0's fixed-offset effects:")
    summary = {}
    summary["bp_real"] = summarize("BP disruption (real coords)", acc_pd["clean"] - acc_pd["bp_real"], c)
    summary["ppt_real"] = summarize("PPT disruption (real coords)", acc_pd["clean"] - acc_pd["ppt_real"], c)
    summary["both_real"] = summarize("BP+PPT both (real coords)", acc_pd["clean"] - acc_pd["both_real"], c)
    summary["n_fallback"] = n_fallback
    summary["mean_bp_distance_from_ag"] = float(acc_pd["bp_distance_from_ag"].mean())
    summary["mean_ppt_len"] = float(acc_pd["ppt_len"].mean())

    phase0_path = out_dir / "phase0_results.json"
    if phase0_path.exists():
        with open(phase0_path) as f:
            phase0 = json.load(f)
        print("\n  comparison (fixed-offset -> real-coordinate):")
        for real_key, fixed_key, label in [
            ("bp_real", "delta_ppt_bp_only", "BP-only"),
            ("ppt_real", "ppt_only_alone", "PPT-only"),
            ("both_real", "both_disrupted", "BP+PPT both"),
        ]:
            fx, rl = phase0[fixed_key]["mean"], summary[real_key]["mean"]
            print(f"    {label:14s} {fx:+.3f}  ->  {rl:+.3f}  (delta = {rl - fx:+.3f})")

    with open(out_dir / "phase3_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[phase3] wrote {out_dir}/phase3_results.json and {out_dir}/phase3_per_sample.csv")


if __name__ == "__main__":
    main()
