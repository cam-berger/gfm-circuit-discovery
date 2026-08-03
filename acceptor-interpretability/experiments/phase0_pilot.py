"""Phase 0: acceptor-recognition pilot on real 3'SS windows (GUE splice_reconstructed).

Reproduces, on real GENCODE-derived human splice windows, the four findings
of the original synthetic-window Phase-0 pilot described in the extension
proposal:

  1. branch-point disruption lowers m_acc (Delta_ppt), flat on a matched
     non-splice control (not general model degradation)
  2. context is required - a bare AG (random-AG control) scores far below a
     real acceptor window
  3. AG-scanning - the true (first, proximal) AG outscores a downstream
     decoy AG found in the same real sequence
  4. interaction, not additivity - PPT-only disruption alone does ~nothing,
     and disrupting BP+PPT together also does ~nothing (the BP effect
     requires an intact PPT context - an AND-gated computation)

Run:
    python experiments/phase0_pilot.py

Writes results/phase0_results.json (summary + CIs) and
results/phase0_per_sample.csv (raw per-sample m_acc values, for reuse by
the decisive 2x2 factorial in next-steps item 1).
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
    find_decoy_ag,
    load_gue_split,
    negative_rows,
    random_ag_variant,
    sample_rows,
)
from acceptors.metrics import m_acc
from acceptors.model import SpliceBert
from acceptors.stats import bootstrap_ci


def score_acceptor_sample(model, seq: str, c) -> dict:
    rng = pyrandom.Random(hash(seq) & 0xFFFFFFFF)
    ag0 = c.coords.ag_start

    clean = m_acc(model, seq, ag0)

    bp_only = m_acc(model, corrupt_region(seq, c.coords.bp_start, c.coords.bp_end, rng), ag0)
    ppt_only = m_acc(model, corrupt_region(seq, c.coords.ppt_start, c.coords.ppt_end, rng), ag0)
    both = m_acc(model, corrupt_region(seq, c.coords.bp_start, c.coords.ppt_end, rng), ag0)
    rand_ag = m_acc(model, random_ag_variant(seq, ag0, c.coords.ag_len, rng), ag0)

    decoy_pos = find_decoy_ag(seq, c.coords.decoy_search_from, c.coords.decoy_search_to)
    decoy = m_acc(model, seq, decoy_pos) if decoy_pos is not None else None

    return dict(
        clean=clean, bp_only=bp_only, ppt_only=ppt_only, both=both,
        rand_ag=rand_ag, decoy=decoy, has_decoy=decoy_pos is not None,
    )


def score_negative_sample(model, seq: str, c) -> dict:
    rng = pyrandom.Random(hash(seq) & 0xFFFFFFFF)
    ag0 = c.coords.ag_start
    clean = m_acc(model, seq, ag0)
    bp_only = m_acc(model, corrupt_region(seq, c.coords.bp_start, c.coords.bp_end, rng), ag0)
    ppt_only = m_acc(model, corrupt_region(seq, c.coords.ppt_start, c.coords.ppt_end, rng), ag0)
    both = m_acc(model, corrupt_region(seq, c.coords.bp_start, c.coords.ppt_end, rng), ag0)
    return dict(clean=clean, bp_only=bp_only, ppt_only=ppt_only, both=both)


def summarize(label, diffs, c):
    mean, lo, hi = bootstrap_ci(diffs, c.phase0.n_boot, c.phase0.alpha, c.phase0.seed)
    print(f"  {label:28s} {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]  (n={len(diffs)})")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(diffs)}


def main():
    c = load_config()
    out_dir = Path(c.output.results_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"[phase0] loading GUE/{c.data.config} ({c.data.split}) ...")
    df = load_gue_split(c)
    acc_df = sample_rows(acceptor_rows(df, c), c.phase0.n_acceptor, c.phase0.seed)
    neg_df = sample_rows(negative_rows(df, c), c.phase0.n_negative, c.phase0.seed)
    print(f"[phase0] {len(acc_df)} acceptor windows, {len(neg_df)} non-splice control windows")

    print(f"[phase0] loading {c.model.checkpoint} ...")
    model = SpliceBert(c.model.checkpoint, c.model.device)
    print(f"[phase0] device: {model.device}")

    acc_rows = [
        score_acceptor_sample(model, seq, c)
        for seq in tqdm(acc_df["sequence"], desc="acceptor windows")
    ]
    neg_rows = [
        score_negative_sample(model, seq, c)
        for seq in tqdm(neg_df["sequence"], desc="non-splice controls")
    ]

    acc_pd = pd.DataFrame(acc_rows)
    neg_pd = pd.DataFrame(neg_rows)
    acc_pd.to_csv(out_dir / f"{c.output.prefix}_per_sample_acceptor.csv", index=False)
    neg_pd.to_csv(out_dir / f"{c.output.prefix}_per_sample_negative.csv", index=False)

    print("\n[phase0] results (mean Delta, 95% bootstrap CI):")
    summary = {}
    summary["delta_ppt_bp_only"] = summarize(
        "Delta_ppt (BP disruption)", acc_pd["clean"] - acc_pd["bp_only"], c)
    summary["delta_ppt_control_nonsplice"] = summarize(
        "  -> non-splice control", neg_pd["clean"] - neg_pd["bp_only"], c)
    summary["context_required"] = summarize(
        "context required (clean - random-AG)", acc_pd["clean"] - acc_pd["rand_ag"], c)

    decoy_df = acc_pd[acc_pd["has_decoy"]]
    summary["ag_scanning"] = summarize(
        "AG-scanning (true - decoy)", decoy_df["clean"] - decoy_df["decoy"], c)

    summary["ppt_only_alone"] = summarize(
        "PPT-only disruption alone", acc_pd["clean"] - acc_pd["ppt_only"], c)
    summary["ppt_only_control_nonsplice"] = summarize(
        "  -> non-splice control", neg_pd["clean"] - neg_pd["ppt_only"], c)
    summary["both_disrupted"] = summarize(
        "BP+PPT both disrupted", acc_pd["clean"] - acc_pd["both"], c)
    summary["both_control_nonsplice"] = summarize(
        "  -> non-splice control", neg_pd["clean"] - neg_pd["both"], c)

    with open(out_dir / f"{c.output.prefix}_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[phase0] wrote {out_dir}/{c.output.prefix}_results.json "
          f"and per-sample CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
