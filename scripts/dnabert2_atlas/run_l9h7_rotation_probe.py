#!/usr/bin/env python
"""D17 — is L9H7 a content detector, a fixed-offset reader, or conjunctive?

D16 found L9H7's [CLS] attention moved onto proximal_upstream (-50:-2) after fine-tuning. But the
task geometry is fixed (TSS always at index 249), and ALiBi encodes distance from [CLS], so
-50:-2 is also window positions 199..247 -- a constant offset. Content detection and positional
habit are not distinguished by D16.

Circular rotation separates them: s' = s[k:] + s[:k] preserves the content EXACTLY and moves it to
a new offset. Two competing readouts per rotation:
    F(k)  attention on the NUCLEOTIDES originally at 199..247, now at (199-k)..(247-k)
    P(k)  attention on whatever now occupies WINDOW positions 199..247

Design red-team (2026-07-29) forced three changes:
  * k in {10, 25} are NOT discriminators -- the F and P bands overlap 39/49 and 24/49 nt, so the
    positional hypothesis itself predicts F ~ 0.444 at k=10, above any sensible content threshold.
    Bands are disjoint only for k >= 49. Only disjoint k decide.
  * The junction (between new indices 299-k and 300-k) can never fall inside F but CAN fall inside
    P, so junction-attracted attention inflates P and never F -- a bias toward the positional
    verdict. k=50 puts it exactly on the canonical TSS index and k=99 puts it inside the P band;
    both are excluded. J(k) is reported as a memorial.
  * A joint collapse of F and P is NOT "inconclusive": it is the signature of CONJUNCTIVE
    addressing (content recognised only at the trained offset). Separating that from generic OOD
    collapse needs (a) an OOD-severity-matched shuffled-content arm, giving the identified
    difference dF = F_intact - F_shuffled, and (b) a global-stability readout.

Protocol: studies/dnabert2_reconstruction_atlas/PREREGISTRATION_D17_L9H7_ROTATION_PROBE.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    load_dnabert2_promoter_classifier,
    promoter_margin_tensor,
)
from gfmcd.adapters.dnabert2.promoter_head_atlas import tokenize_promoter_pairs
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.statistics import cluster_bootstrap_interval

REPO = Path(__file__).resolve().parents[2]
COHORT = REPO / "data/processed/dnabert2_atlas/promoter_source_localization_v1/cohort.jsonl"

WINDOW_NT = 300
TSS_INDEX = 249
BAND_LO, BAND_HI = 199, 248           # proximal_upstream = TSS-relative -50:-2, half-open [199,248)
SEED = 20260729
DRAWS = 8000

# k=0 identity reference; k=25 declared NON-discriminating OOD-dose anchor (49% band overlap);
# k in {125,149,175,199} are the disjoint discriminators with the junction clear of both bands.
ROTATIONS = (0, 25, 125, 149, 175, 199)
DISCRIMINATING = (125, 149, 175, 199)
FOCUS = ("L9H7", "L11H2", "L11H5")

DF_CONTENT_MIN = 0.15   # frozen: identified content effect to call CONTENT-ADDRESSED
P_INDEX_MIN = 0.35      # frozen: retained positional mass to call INDEX-ADDRESSED
NEAR_ZERO = 0.05        # frozen: dF below this is "no identified content addressing"
COLLAPSE = 0.20         # frozen: P below this is "positional selectivity lost"
J_CEILING = 0.10        # memorial: junction must not attract more than this


def stable_seed(text: str) -> int:
    """builtin hash() on str varies with PYTHONHASHSEED and would make the shuffle irreproducible."""
    import hashlib

    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def rotate(sequence: str, k: int) -> str:
    return sequence[k:] + sequence[:k]


def shuffled_band(sequence: str, seed: int) -> str:
    """Mononucleotide shuffle of [BAND_LO, BAND_HI) in place. Composition exactly preserved."""
    band = list(sequence[BAND_LO:BAND_HI])
    rng = random.Random(seed)
    rng.shuffle(band)
    return sequence[:BAND_LO] + "".join(band) + sequence[BAND_HI:]


def overlap_fractions(offsets, lo: int, hi: int) -> list[float]:
    """Per-token fraction of its nucleotides lying in [lo, hi). Special tokens get 0."""
    out = []
    for start, end in offsets:
        if end <= start:
            out.append(0.0)
            continue
        out.append(max(0, min(end, hi) - max(start, lo)) / (end - start))
    return out


def capture_cls_attention(model, config, batch, device):
    """[CLS]-row attention probabilities per head per sequence. Same path validated in D16."""
    import torch

    qkv_by_layer: dict[int, Any] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            qkv_by_layer[layer] = output.detach()
            return output

        return hook

    try:
        for layer, block in enumerate(model.bert.encoder.layer):
            handles.append(block.attention.self.Wqkv.register_forward_hook(make_hook(layer)))
        with torch.no_grad():
            model(**{name: tensor.to(device) for name, tensor in batch.encoded.items()})
    finally:
        for handle in handles:
            handle.remove()

    n_layers = int(config.num_hidden_layers)
    n_heads = int(config.num_attention_heads)
    head_size = int(config.hidden_size) // n_heads
    spans, cursor = [], 0
    for count in batch.token_counts:
        spans.append((cursor, cursor + count))
        cursor += count
    alibi = model.bert.encoder.alibi
    probabilities: dict[int, dict[tuple[int, int], Any]] = defaultdict(dict)
    for layer in range(n_layers):
        qkv_all = qkv_by_layer[layer]
        for seq_index, ((start, end), count) in enumerate(zip(spans, batch.token_counts)):
            qkv = qkv_all[start:end].view(count, 3, n_heads, head_size)
            scores = torch.einsum("hd,shd->hs", qkv[0, 0], qkv[:, 1]) / math.sqrt(head_size)
            scores = scores + alibi[0, :, 0, :count].to(device=scores.device, dtype=scores.dtype)
            for head in range(n_heads):
                probabilities[seq_index][(layer, head)] = torch.softmax(
                    scores[head], dim=0
                ).cpu().tolist()
    return probabilities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-pairs", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in COHORT.open()]
    if args.limit:
        rows = rows[: args.limit]
    cluster_of = {r["example_id"]: r["homology_cluster"] for r in rows}
    bundle = load_dnabert2_promoter_classifier()
    tokenizer = bundle.tokenizer

    import torch

    per_locus: dict[str, dict[str, float]] = defaultdict(dict)
    margins: dict[str, dict[str, float]] = defaultdict(dict)
    band_composition_ok = True
    rotation_composition_ok = True
    position_bandmass: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    conditions = [(k, arm) for k in ROTATIONS for arm in ("intact", "shuffled")]

    for k, arm in conditions:
        built = []
        for record in rows:
            native = record["target"]["clean_sequence"]
            base = native if arm == "intact" else shuffled_band(
                native, SEED + stable_seed(record["example_id"])
            )
            if arm == "shuffled" and sorted(base) != sorted(native):
                band_composition_ok = False
            sequence = rotate(base, k)
            if sorted(sequence) != sorted(base):
                rotation_composition_ok = False
            built.append((record["example_id"], sequence, record["control"]["clean_sequence"]))

        for start in range(0, len(built), args.batch_pairs):
            chunk = built[start : start + args.batch_pairs]
            batch = tokenize_promoter_pairs(bundle, chunk)
            with torch.no_grad():
                margin = promoter_margin_tensor(bundle, batch).detach()
            probabilities = capture_cls_attention(
                bundle.model, bundle.config, batch, bundle.device
            )
            for i, pair_id in enumerate(batch.pair_ids):
                seq_index = 2 * i  # target branch only
                offsets = batch.offset_mapping[
                    seq_index, : batch.token_counts[seq_index]
                ].tolist()
                probs = probabilities[seq_index]
                f_frac = overlap_fractions(offsets, BAND_LO - k, BAND_HI - k)
                p_frac = overlap_fractions(offsets, BAND_LO, BAND_HI)
                j_frac = (
                    overlap_fractions(offsets, 299 - k, 301 - k) if k else [0.0] * len(offsets)
                )
                for head in FOCUS:
                    layer, index = int(head.split("H")[0][1:]), int(head.split("H")[1])
                    attention = probs[(layer, index)]
                    per_locus[pair_id][f"F|{head}|{k}|{arm}"] = sum(
                        a * w for a, w in zip(attention, f_frac)
                    )
                    per_locus[pair_id][f"P|{head}|{k}|{arm}"] = sum(
                        a * w for a, w in zip(attention, p_frac)
                    )
                    per_locus[pair_id][f"J|{head}|{k}|{arm}"] = sum(
                        a * w for a, w in zip(attention, j_frac)
                    )
                if arm == "intact":
                    margins[pair_id][str(k)] = float(margin[seq_index].cpu())
                    for (layer, index), attention in probs.items():
                        position_bandmass[k][f"L{layer}H{index}"].append(
                            sum(a * w for a, w in zip(attention, p_frac))
                        )
        print(f"[D17] k={k} arm={arm} done", flush=True)

    ids = sorted(per_locus)
    clusters = [cluster_of[i] for i in ids]

    def summarize(key: str) -> dict[str, Any]:
        values = [per_locus[i][key] for i in ids if key in per_locus[i]]
        subset = [cluster_of[i] for i in ids if key in per_locus[i]]
        if len(values) < 2:
            return {"n": len(values)}
        low, high = cluster_bootstrap_interval(values, subset, draws=DRAWS, seed=SEED)
        return {
            "n": len(values),
            "mean": statistics.mean(values),
            "ci95": [low, high],
            "median": statistics.median(values),
        }

    endpoints: dict[str, Any] = {}
    for head in FOCUS:
        for k in ROTATIONS:
            fi = [per_locus[i][f"F|{head}|{k}|intact"] for i in ids]
            fs = [per_locus[i][f"F|{head}|{k}|shuffled"] for i in ids]
            delta = [a - b for a, b in zip(fi, fs)]
            low, high = cluster_bootstrap_interval(delta, clusters, draws=DRAWS, seed=SEED)
            endpoints[f"{head}|k{k}"] = {
                "F_intact": summarize(f"F|{head}|{k}|intact"),
                "F_shuffled": summarize(f"F|{head}|{k}|shuffled"),
                "dF": {"mean": statistics.mean(delta), "ci95": [low, high]},
                "P_intact": summarize(f"P|{head}|{k}|intact"),
                "J_intact": summarize(f"J|{head}|{k}|intact"),
                "discriminating": k in DISCRIMINATING,
            }

    # ---------------------------------------------------------------- global stability
    stability = {}
    for k in ROTATIONS:
        drift = statistics.mean(
            abs(statistics.mean(position_bandmass[k][h]) - statistics.mean(position_bandmass[0][h]))
            for h in position_bandmass[0]
        )
        stability[str(k)] = {
            "mean_promoter_margin": statistics.mean(margins[i][str(k)] for i in ids),
            "mean_abs_P_band_drift_over_144_heads": drift,
        }

    # ---------------------------------------------------------------- frozen trichotomy
    disc_dF = statistics.mean(endpoints[f"L9H7|k{k}"]["dF"]["mean"] for k in DISCRIMINATING)
    disc_P = statistics.mean(
        endpoints[f"L9H7|k{k}"]["P_intact"]["mean"] for k in DISCRIMINATING
    )
    disc_dF_lower = min(endpoints[f"L9H7|k{k}"]["dF"]["ci95"][0] for k in DISCRIMINATING)
    margin_ok = stability[str(DISCRIMINATING[-1])]["mean_promoter_margin"] > 0

    if disc_dF >= DF_CONTENT_MIN and disc_dF_lower > 0:
        verdict = "CONTENT_ADDRESSED"
    elif disc_P >= P_INDEX_MIN and disc_dF < NEAR_ZERO:
        verdict = "INDEX_ADDRESSED"
    elif disc_P < COLLAPSE and disc_dF < NEAR_ZERO and margin_ok:
        verdict = "CONJUNCTIVE"
    else:
        verdict = "MIXED_OR_UNINTERPRETABLE"

    memorials = {
        "M1_identity_at_k0": {"architecture_forced": True,
                              "note": "k=0 rotation is the identity; forced, flagged not counted"},
        "M2_rotation_preserves_composition": {"passed": rotation_composition_ok},
        "M3_shuffle_preserves_composition": {"passed": band_composition_ok},
        "M5_L11H2_stays_pinned": {
            "P_by_k": {str(k): endpoints[f"L11H2|k{k}"]["P_intact"]["mean"] for k in ROTATIONS},
            "note": "L11H2 has reach ~8 tokens and is ALiBi-pinned near [CLS]; its P mass is tiny "
                    "and must stay tiny under every rotation. A change here breaks the framing.",
        },
        "M6_junction_below_ceiling": {
            "ceiling": J_CEILING,
            "J_by_k": {str(k): endpoints[f"L9H7|k{k}"]["J_intact"]["mean"] for k in ROTATIONS},
        },
    }
    memorials["M6_junction_below_ceiling"]["passed"] = all(
        v <= J_CEILING for v in memorials["M6_junction_below_ceiling"]["J_by_k"].values()
    )

    payload = {
        "study": "dnabert2_reconstruction_atlas",
        "experiment": "D17_l9h7_rotation_probe",
        "n_loci": len(ids),
        "config": {
            "rotations": list(ROTATIONS),
            "discriminating_rotations": list(DISCRIMINATING),
            "band_nt": [BAND_LO, BAND_HI],
            "thresholds": {
                "dF_content_min": DF_CONTENT_MIN,
                "P_index_min": P_INDEX_MIN,
                "near_zero": NEAR_ZERO,
                "collapse": COLLAPSE,
                "junction_ceiling": J_CEILING,
            },
        },
        "memorials": memorials,
        "endpoints": endpoints,
        "global_stability": stability,
        "frozen_verdict": verdict,
        "verdict_inputs": {
            "mean_dF_over_discriminating_k": disc_dF,
            "min_dF_ci_lower": disc_dF_lower,
            "mean_P_over_discriminating_k": disc_P,
            "promoter_margin_positive_at_max_k": margin_ok,
        },
        "per_locus": [{"locus": i, "cluster": cluster_of[i], **per_locus[i]} for i in ids],
    }

    write_result_bundle(
        args.out,
        payload,
        experiment_id="D17_l9h7_rotation_probe",
        status="EXPLORATORY",
        model_identity={
            "checkpoint": bundle.checkpoint.hf_id,
            "revision": bundle.checkpoint.revision,
        },
        input_paths=[COHORT],
        parameters={"rotations": list(ROTATIONS), "band": [BAND_LO, BAND_HI], "draws": DRAWS},
        seeds={"analysis": SEED},
        command=sys.argv,
        repo=REPO,
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "per_locus"}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
