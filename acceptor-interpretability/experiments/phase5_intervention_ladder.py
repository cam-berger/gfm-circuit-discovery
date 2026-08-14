"""Phase 5 (next-steps item 5): the intervention ladder on Phase 4's
discovered tract-reading heads.

Phase 4 showed 5 heads (L5H8, L4H12, L3H5, L0H13, L0H14) have attention into
the polypyrimidine tract that genuinely tracks content - but attention drops
under corruption are a *behavioral* signature (of the head's own softmax),
not proof the head's output actually matters for the model's confidence.
This phase runs the standard causal ladder from the activation-patching
literature (mean-ablation over zero-ablation to stay on-distribution;
donor-sequence restoration patching for mediation; the query/value split
from path patching - Wang et al. 2022, Heimersheim & Nanda 2024, Goldowsky-
Dill et al. 2023 - for routing vs. content) on the two strongest,
order-sensitive hits (L5H8, L4H12), against the far-intronic-style logic
used throughout this project rather than an assumed-zero baseline:

  1. Mean-ablation (necessity): replace the head's output at the masked-AG
     query positions with its dataset mean (over the clean sample), on the
     otherwise-clean sequence. Does m_acc fall? Includes a negative control:
     L0H2 (Phase 4's most attention-saturated donor-circuit head, 0/6 that
     passed content-control) - if raw attention mass predicted causal
     importance, ablating it should hurt just as much; if not, this confirms
     Phase 4's attention-based verdict was the right filter, not just a
     correlate.
  2. Restoration patching (mediation): on the corrupted (and separately, the
     shuffled) sequence, patch in the *same sequence's* clean-run output for
     this head only. If m_acc recovers, the head mediates the tract effect -
     not just correlates with it.
  3. Q/V split (routing vs. content): still on the corrupted sequence,
     Q-patch feeds the head the clean routing (attention pattern) but the
     corrupted values; V-patch feeds it the corrupted routing but the clean
     values. Comparing their recovery fractions asks whether the head's
     causal contribution comes from attending correctly (routing) or from
     what it reads once it attends (content) - or both.

Secondary heads L3H5 and L0H13 (composition-sensitive, not order-sensitive
in Phase 4) get the ablation step only, as a cheaper necessity check.

Recovery fraction = (patched - degraded) / (clean - degraded), a ratio of
bootstrap-resampled means (`acceptors.stats.bootstrap_recovery_ci`) - 100%
means the single patched head fully explains the clean-vs-degraded gap, 0%
means it explains none of it.

Run:
    python experiments/phase5_intervention_ladder.py
Writes results/phase5_results.json, results/phase5_per_sample.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd
import random as pyrandom
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acceptors import patching
from acceptors.config import load_config
from acceptors.data import (
    acceptor_rows,
    corrupt_region,
    load_gue_split,
    sample_rows,
    shuffle_region_in_place,
)
from acceptors.metrics import m_acc_from_logits
from acceptors.model import SpliceBert
from acceptors.stats import bootstrap_ci, bootstrap_recovery_ci

# Phase 4's order-sensitive hits get the full ladder; composition-sensitive
# hits get ablation only; L0H2 is the negative control (saturated attention,
# 0/6 donor-circuit heads that passed content-control).
PRIMARY_HEADS = {"L5H8": (5, 8), "L4H12": (4, 12)}
SECONDARY_HEADS = {"L3H5": (3, 5), "L0H13": (0, 13)}
CONTROL_HEADS = {"L0H2": (0, 2)}
ALL_ABLATION_HEADS = {**PRIMARY_HEADS, **SECONDARY_HEADS, **CONTROL_HEADS}


def layer_to_heads(head_dict):
    out = {}
    for layer, head in head_dict.values():
        out.setdefault(layer, []).append(head)
    return out


def score_sample(model, seq: str, c, mean_vectors):
    rng = pyrandom.Random(hash(seq) & 0xFFFFFFFF)
    ag0 = c.coords.ag_start
    raw_ag = [ag0, ag0 + 1]
    layers = sorted(set(l for l, _ in ALL_ABLATION_HEADS.values()))

    corrupt_seq = corrupt_region(seq, c.coords.ppt_start, c.coords.ppt_end, rng)
    shuffle_seq = shuffle_region_in_place(seq, c.coords.ppt_start, c.coords.ppt_end, rng)

    clean_cap = patching.capture_layers(model, seq, raw_ag, layers)
    corrupt_cap = patching.capture_layers(model, corrupt_seq, raw_ag, layers)
    shuffle_cap = patching.capture_layers(model, shuffle_seq, raw_ag, layers)

    row = {
        "clean_m_acc": m_acc_from_logits(clean_cap["logits"], model.base_ids),
        "corrupt_m_acc": m_acc_from_logits(corrupt_cap["logits"], model.base_ids),
        "shuffle_m_acc": m_acc_from_logits(shuffle_cap["logits"], model.base_ids),
    }

    # 1. Mean-ablation (necessity), on the clean sequence - single heads,
    # then jointly (do individually-null heads matter as a group?).
    for name, (layer, head) in ALL_ABLATION_HEADS.items():
        ablated_logits = patching.run_with_replacement(
            model, seq, raw_ag, layer, head, mean_vectors[(layer, head)])
        row[f"{name}_ablated_m_acc"] = m_acc_from_logits(ablated_logits, model.base_ids)

    real_heads = {**PRIMARY_HEADS, **SECONDARY_HEADS}
    joint_real = {hl: mean_vectors[hl] for hl in real_heads.values()}
    row["joint_real_ablated_m_acc"] = m_acc_from_logits(
        patching.run_with_multi_replacement(model, seq, raw_ag, joint_real), model.base_ids)
    joint_all = {hl: mean_vectors[hl] for hl in ALL_ABLATION_HEADS.values()}
    row["joint_all_ablated_m_acc"] = m_acc_from_logits(
        patching.run_with_multi_replacement(model, seq, raw_ag, joint_all), model.base_ids)

    # 2. Restoration patching (mediation): clean donor into corrupted/shuffled recipient.
    for name, (layer, head) in PRIMARY_HEADS.items():
        full_repl = patching.full_patch_replacement(model, clean_cap, layer, head)
        patched_corrupt = patching.run_with_replacement(model, corrupt_seq, raw_ag, layer, head, full_repl)
        patched_shuffle = patching.run_with_replacement(model, shuffle_seq, raw_ag, layer, head, full_repl)
        row[f"{name}_patch_corrupt_m_acc"] = m_acc_from_logits(patched_corrupt, model.base_ids)
        row[f"{name}_patch_shuffle_m_acc"] = m_acc_from_logits(patched_shuffle, model.base_ids)

        # 3. Q/V split, on the corrupted recipient only.
        q_repl = patching.q_patch_replacement(model, clean_cap, corrupt_cap, layer, head)
        v_repl = patching.v_patch_replacement(model, corrupt_cap, clean_cap, layer, head)
        q_logits = patching.run_with_replacement(model, corrupt_seq, raw_ag, layer, head, q_repl)
        v_logits = patching.run_with_replacement(model, corrupt_seq, raw_ag, layer, head, v_repl)
        row[f"{name}_qpatch_m_acc"] = m_acc_from_logits(q_logits, model.base_ids)
        row[f"{name}_vpatch_m_acc"] = m_acc_from_logits(v_logits, model.base_ids)

    return row


def summarize(label, diffs, c):
    mean, lo, hi = bootstrap_ci(diffs, c.phase0.n_boot, c.phase0.alpha, c.phase0.seed)
    print(f"    {label:34s} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(diffs)}


def summarize_recovery(label, patched, degraded, clean, c):
    point, lo, hi = bootstrap_recovery_ci(patched, degraded, clean, c.phase0.n_boot, c.phase0.alpha, c.phase0.seed)
    print(f"    {label:34s} {point:+.1%}  [{lo:+.1%}, {hi:+.1%}]")
    return {"mean": point, "ci_lo": lo, "ci_hi": hi, "n": len(patched)}


def main():
    c = load_config()
    out_dir = Path(c.output.results_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"[phase5] loading GUE/{c.data.config} ({c.data.split}) ...")
    df = load_gue_split(c)
    acc_df = sample_rows(acceptor_rows(df, c), c.phase0.n_acceptor, c.phase0.seed)
    print(f"[phase5] {len(acc_df)} acceptor windows (same sample as phase0-4)")

    print(f"[phase5] loading {c.model.checkpoint} ...")
    model = SpliceBert(c.model.checkpoint, c.model.device)
    print(f"[phase5] device: {model.device}")

    ag0 = c.coords.ag_start
    raw_ag = [ag0, ag0 + 1]

    print("[phase5] computing mean-ablation vectors over the clean sample ...")
    mean_vectors = patching.mean_ablation_vectors(
        model, list(acc_df["sequence"]), raw_ag, layer_to_heads(ALL_ABLATION_HEADS))

    rows = [
        score_sample(model, seq, c, mean_vectors)
        for seq in tqdm(acc_df["sequence"], desc="intervention ladder")
    ]
    acc_pd = pd.DataFrame(rows)
    acc_pd.to_csv(out_dir / "phase5_per_sample.csv", index=False)

    summary = {}

    print("\n[phase5] step 1 - mean-ablation (necessity), clean sequence:")
    ablation = {}
    for name in ALL_ABLATION_HEADS:
        kind = "control" if name in CONTROL_HEADS else "target"
        print(f"  {name} ({kind}):")
        ablation[name] = summarize(
            "clean - ablated", acc_pd["clean_m_acc"] - acc_pd[f"{name}_ablated_m_acc"], c)
        ablation[name]["kind"] = kind
    print("  joint (L5H8+L4H12+L3H5+L0H13 together):")
    ablation["joint_real"] = summarize(
        "clean - jointly ablated", acc_pd["clean_m_acc"] - acc_pd["joint_real_ablated_m_acc"], c)
    print("  joint (all 5, incl. L0H2 control):")
    ablation["joint_all"] = summarize(
        "clean - jointly ablated", acc_pd["clean_m_acc"] - acc_pd["joint_all_ablated_m_acc"], c)
    summary["ablation"] = ablation

    print("\n[phase5] step 2 - restoration patching (mediation), recovery fraction:")
    mediation = {}
    for name in PRIMARY_HEADS:
        mediation[name] = {}
        for cond in ("corrupt", "shuffle"):
            print(f"  {name} -> {cond}:")
            mediation[name][cond] = summarize_recovery(
                "recovery fraction",
                acc_pd[f"{name}_patch_{cond}_m_acc"], acc_pd[f"{cond}_m_acc"], acc_pd["clean_m_acc"], c)
    summary["mediation"] = mediation

    print("\n[phase5] step 3 - Q/V split (routing vs. content), corrupted recipient:")
    qv_split = {}
    for name in PRIMARY_HEADS:
        print(f"  {name}:")
        qv_split[name] = {
            "q_patch": summarize_recovery(
                "Q-patch (routing) recovery",
                acc_pd[f"{name}_qpatch_m_acc"], acc_pd["corrupt_m_acc"], acc_pd["clean_m_acc"], c),
            "v_patch": summarize_recovery(
                "V-patch (content) recovery",
                acc_pd[f"{name}_vpatch_m_acc"], acc_pd["corrupt_m_acc"], acc_pd["clean_m_acc"], c),
        }
    summary["qv_split"] = qv_split

    summary["n"] = len(acc_df)
    with open(out_dir / "phase5_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[phase5] wrote {out_dir}/phase5_results.json and {out_dir}/phase5_per_sample.csv")


if __name__ == "__main__":
    main()
