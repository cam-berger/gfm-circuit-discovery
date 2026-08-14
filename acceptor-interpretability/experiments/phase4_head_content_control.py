"""Phase 4 (next-steps item 4): head content-control, retargeted to tract-readers.

Phases 1-3 established that the polypyrimidine tract, not the branch point,
is the dominant driver of `m_acc`, and that the tract signal is order-driven
(shuffle-in-place nearly matches full content-destroying corruption). The
candidate heads carried over from the donor-circuit proposal (L0H2, L1H14,
L2H7, L2H0, L3H10, L1H1) still need the check that distinguishes a genuine
content reader from a positional-locality artifact: attention from the
masked-AG query position into the PPT/BP region must *drop* when that
region's content is destroyed. Given the tract's order-sensitivity, a
composition-preserving shuffle-in-place is tested alongside full corruption
- a genuine motif/order reader should show attention drop under both, not
just full corruption.

For each candidate head and each region (PPT, BP):
  - clean attention mass (masked-AG query -> region), vs. a uniform-attention
    baseline (region length / sequence length) - is the head even looking
    there?
  - delta = clean - corrupted (content-destroying), paired bootstrap CI
  - delta = clean - shuffled (composition-preserving), paired bootstrap CI
  - verdict: content reader / positional-locality artifact / no attention

Also ranks every (layer, head) by clean-condition attention mass to each
region (no extra forward passes - reused from the same clean pass) as a
sanity check on whether the candidate list still looks complete now that
the tract is the primary object.

Run:
    python experiments/phase4_head_content_control.py
Writes results/phase4_results.json, results/phase4_per_sample.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd
import random as pyrandom
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acceptors.attention import region_mass_all_heads, region_mass_head
from acceptors.config import load_config
from acceptors.data import (
    acceptor_rows,
    corrupt_region,
    load_gue_split,
    sample_rows,
    shuffle_region_in_place,
)
from acceptors.model import SpliceBert
from acceptors.stats import bootstrap_ci

DONOR_CIRCUIT_CANDIDATES = [(0, 2), (1, 14), (2, 7), (2, 0), (3, 10), (1, 1)]

# Discovered in a first pass of this script (top-10 clean-attention heads per
# region - see results/phase4_results.json: top_ppt_heads/top_bp_heads): all
# six donor-circuit candidates turned out to be near-saturated positional-
# locality heads for the PPT region (>=90% attention mass regardless of
# content). The top BP-attending heads are far from saturated (24-74%,
# vs. ~99% for PPT), leaving real room for content-sensitivity that the
# original candidate list never covered - added here rather than in a
# separate script since they reuse the same forward passes at ~zero extra
# cost (region-mass extraction is indexing, not inference).
DISCOVERED_PPT_ATTENDERS = [(0, 14), (2, 8), (4, 7), (4, 3), (0, 1)]
DISCOVERED_BP_ATTENDERS = [
    (1, 6), (2, 12), (4, 12), (4, 8), (5, 8), (5, 14), (0, 13), (3, 5), (1, 12), (0, 0),
]
CANDIDATE_HEADS = sorted(set(DONOR_CIRCUIT_CANDIDATES + DISCOVERED_PPT_ATTENDERS + DISCOVERED_BP_ATTENDERS))
NO_ATTENTION_MULTIPLE = 1.5  # clean mass must exceed uniform baseline by this much to be "attending"


def score_sample(model, seq: str, c):
    rng = pyrandom.Random(hash(seq) & 0xFFFFFFFF)
    raw_ag = [c.coords.ag_start, c.coords.ag_start + 1]

    attn_clean, tok = model.masked_attentions_at(seq, raw_ag)
    attn_ppt_corrupt, _ = model.masked_attentions_at(
        corrupt_region(seq, c.coords.ppt_start, c.coords.ppt_end, rng), raw_ag)
    attn_ppt_shuffle, _ = model.masked_attentions_at(
        shuffle_region_in_place(seq, c.coords.ppt_start, c.coords.ppt_end, rng), raw_ag)
    attn_bp_corrupt, _ = model.masked_attentions_at(
        corrupt_region(seq, c.coords.bp_start, c.coords.bp_end, rng), raw_ag)
    attn_bp_shuffle, _ = model.masked_attentions_at(
        shuffle_region_in_place(seq, c.coords.bp_start, c.coords.bp_end, rng), raw_ag)

    row = {}
    for layer, head in CANDIDATE_HEADS:
        key = f"L{layer}H{head}"
        row[f"{key}_ppt_clean"] = region_mass_head(attn_clean, layer, head, tok, c.coords.ppt_start, c.coords.ppt_end)
        row[f"{key}_ppt_corrupt"] = region_mass_head(attn_ppt_corrupt, layer, head, tok, c.coords.ppt_start, c.coords.ppt_end)
        row[f"{key}_ppt_shuffle"] = region_mass_head(attn_ppt_shuffle, layer, head, tok, c.coords.ppt_start, c.coords.ppt_end)
        row[f"{key}_bp_clean"] = region_mass_head(attn_clean, layer, head, tok, c.coords.bp_start, c.coords.bp_end)
        row[f"{key}_bp_corrupt"] = region_mass_head(attn_bp_corrupt, layer, head, tok, c.coords.bp_start, c.coords.bp_end)
        row[f"{key}_bp_shuffle"] = region_mass_head(attn_bp_shuffle, layer, head, tok, c.coords.bp_start, c.coords.bp_end)

    ppt_all = region_mass_all_heads(attn_clean, tok, c.coords.ppt_start, c.coords.ppt_end)
    bp_all = region_mass_all_heads(attn_clean, tok, c.coords.bp_start, c.coords.bp_end)
    return row, ppt_all, bp_all


def classify(clean_mean, baseline, delta_corrupt, delta_shuffle):
    if clean_mean < baseline * NO_ATTENTION_MULTIPLE:
        return "no attention to region"
    corrupt_drops = delta_corrupt["ci_lo"] > 0
    shuffle_drops = delta_shuffle["ci_lo"] > 0
    if corrupt_drops and shuffle_drops:
        return "content reader (order-sensitive)"
    if corrupt_drops:
        return "content reader (composition/position, order-insensitive)"
    return "positional-locality artifact"


def summarize(label, diffs, c):
    mean, lo, hi = bootstrap_ci(diffs, c.phase0.n_boot, c.phase0.alpha, c.phase0.seed)
    print(f"    {label:20s} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(diffs)}


def main():
    c = load_config()
    out_dir = Path(c.output.results_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"[phase4] loading GUE/{c.data.config} ({c.data.split}) ...")
    df = load_gue_split(c)
    acc_df = sample_rows(acceptor_rows(df, c), c.phase0.n_acceptor, c.phase0.seed)
    print(f"[phase4] {len(acc_df)} acceptor windows (same sample as phase0/1/3)")

    print(f"[phase4] loading {c.model.checkpoint} ...")
    model = SpliceBert(c.model.checkpoint, c.model.device)
    print(f"[phase4] device: {model.device}")

    n_layers = model.model.config.num_hidden_layers
    n_heads = model.model.config.num_attention_heads
    ppt_sum = torch.zeros(n_layers, n_heads)
    bp_sum = torch.zeros(n_layers, n_heads)

    rows = []
    for seq in tqdm(acc_df["sequence"], desc="head content-control"):
        row, ppt_all, bp_all = score_sample(model, seq, c)
        rows.append(row)
        ppt_sum += ppt_all
        bp_sum += bp_all

    acc_pd = pd.DataFrame(rows)
    acc_pd.to_csv(out_dir / "phase4_per_sample.csv", index=False)
    n = len(acc_df)

    seq_len_tokens = len(acc_df["sequence"].iloc[0]) + 2  # <cls> + bases + <eos>
    ppt_baseline = (c.coords.ppt_end - c.coords.ppt_start) / seq_len_tokens
    bp_baseline = (c.coords.bp_end - c.coords.bp_start) / seq_len_tokens

    print("\n[phase4] per-head content-control (delta = clean - corrupted/shuffled attention mass):")
    heads_summary = {}
    for layer, head in CANDIDATE_HEADS:
        key = f"L{layer}H{head}"
        origin = []
        if (layer, head) in DONOR_CIRCUIT_CANDIDATES:
            origin.append("donor_circuit")
        if (layer, head) in DISCOVERED_PPT_ATTENDERS:
            origin.append("top_ppt_attender")
        if (layer, head) in DISCOVERED_BP_ATTENDERS:
            origin.append("top_bp_attender")
        head_summary = {"origin": origin}
        for region, baseline in [("ppt", ppt_baseline), ("bp", bp_baseline)]:
            clean = acc_pd[f"{key}_{region}_clean"]
            corrupt = acc_pd[f"{key}_{region}_corrupt"]
            shuffle = acc_pd[f"{key}_{region}_shuffle"]
            print(f"  {key} -> {region.upper()} (clean mass {clean.mean():.4f}, baseline {baseline:.4f}):")
            delta_corrupt = summarize("clean - corrupt", clean - corrupt, c)
            delta_shuffle = summarize("clean - shuffle", clean - shuffle, c)
            verdict = classify(clean.mean(), baseline, delta_corrupt, delta_shuffle)
            print(f"    -> {verdict}")
            head_summary[region] = {
                "clean_mass": float(clean.mean()),
                "baseline_mass": baseline,
                "delta_corrupt": delta_corrupt,
                "delta_shuffle": delta_shuffle,
                "verdict": verdict,
            }
        heads_summary[key] = head_summary

    def top_heads(mass_matrix, k=10):
        mass_matrix = mass_matrix / n
        flat = [
            {"head": f"L{l}H{h}", "mass": float(mass_matrix[l, h])}
            for l in range(n_layers) for h in range(n_heads)
        ]
        flat.sort(key=lambda d: -d["mass"])
        return flat[:k]

    top_ppt = top_heads(ppt_sum)
    top_bp = top_heads(bp_sum)
    print(f"\n[phase4] top 10 heads by clean PPT attention mass (baseline {ppt_baseline:.4f}):")
    for d in top_ppt:
        print(f"    {d['head']:6s} {d['mass']:.4f}")
    print(f"\n[phase4] top 10 heads by clean BP attention mass (baseline {bp_baseline:.4f}):")
    for d in top_bp:
        print(f"    {d['head']:6s} {d['mass']:.4f}")

    summary = {
        "candidate_heads": heads_summary,
        "ppt_baseline_mass": ppt_baseline,
        "bp_baseline_mass": bp_baseline,
        "top_ppt_heads": top_ppt,
        "top_bp_heads": top_bp,
        "n": n,
    }
    with open(out_dir / "phase4_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[phase4] wrote {out_dir}/phase4_results.json and {out_dir}/phase4_per_sample.csv")


if __name__ == "__main__":
    main()
