#!/usr/bin/env python
"""D15 — regional information density vs the ALiBi-forced attention prior.

Equal-dose contiguous span masking. Every arm masks exactly the same number of BPE tokens in
target and control, with matched nucleotide cover, so any artifact that depends only on "k tokens
were replaced by [MASK]" cancels exactly in the between-arm differences.

  S       -9:+31    dose 7    the region D8/D12/D14 showed carries the behaviour
  F1..F5  -249:-49  dose 7    five disjoint 40-nt far-upstream windows
  U       -49:-9    dose 7    specificity check (development sign was negative)
  B       -249:-49  dose 30   the whole far-upstream band, deliberately over-dosed

Protocol: studies/dnabert2_reconstruction_atlas/PREREGISTRATION_D15_ATTENTION_CAUSALITY_DISSOCIATION.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    explicit_token_mask_effects_for_batch,
    load_dnabert2_promoter_classifier,
    promoter_margin_tensor,
)
from gfmcd.adapters.dnabert2.promoter_head_atlas import tokenize_promoter_pairs
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    cluster_bootstrap_interval,
    holm_adjust,
    paired_cluster_sign_permutation,
)

REPO = Path(__file__).resolve().parents[2]
COHORT = REPO / "data/processed/dnabert2_atlas/promoter_source_localization_v1/cohort.jsonl"

TSS_INDEX = 249
K_WINDOW = 7
K_BAND = 30
MAX_COVER_DIFF_NT = 8
SEED = 20260729
DRAWS = 8000
PERMUTATIONS = 9999
SESOI = 0.25
CLEAN_ANCHOR_MINIMUM = 1.5

SIGNAL = ("S", 240, 280)
FARWINS = [(f"F{i + 1}", 40 * i, 40 * i + 40) for i in range(5)]
UPSTREAM = ("U", 200, 240)
BAND = ("B", 0, 200)
ARMS = [SIGNAL, *FARWINS, UPSTREAM, BAND]
FAR_NAMES = [name for name, _, _ in FARWINS]


def contained_tokens(offsets, lo: int, hi: int, k: int):
    """The k genomic tokens with greatest nucleotide overlap with [lo, hi).

    Returns (sorted token indices, covered nucleotides) or None when fewer than k tokens
    overlap the window at all.
    """
    scored = []
    for index, (start, end) in enumerate(offsets):
        if end <= start:  # special token, never maskable
            continue
        overlap = max(0, min(end, hi) - max(start, lo))
        if overlap > 0:
            scored.append((overlap, index))
    if len(scored) < k:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return sorted(index for _, index in scored[:k]), sum(ov for ov, _ in scored[:k])


def score_arms(bundle, pairs, batch_pairs: int, progress=True):
    """Return per-locus target-specific loss for every arm, plus supply and cover records."""
    import torch

    per_locus: dict[str, dict[str, float]] = defaultdict(dict)
    supply = {name: {"attempted": 0, "qualified": 0} for name, _, _ in ARMS}
    covers: dict[str, list[tuple[int, int]]] = defaultdict(list)
    clean: dict[str, tuple[float, float]] = {}

    for start in range(0, len(pairs), batch_pairs):
        chunk = pairs[start : start + batch_pairs]
        batch = tokenize_promoter_pairs(bundle, chunk)
        offsets = [
            batch.offset_mapping[i, : batch.token_counts[i]].tolist()
            for i in range(2 * batch.n_pairs)
        ]
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, batch).detach()
        for i, pair_id in enumerate(batch.pair_ids):
            clean[pair_id] = (float(native[2 * i].cpu()), float(native[2 * i + 1].cpu()))

        for name, lo, hi in ARMS:
            dose = K_BAND if name == "B" else K_WINDOW
            selected, keep = [], []
            for i in range(batch.n_pairs):
                supply[name]["attempted"] += 1
                target = contained_tokens(offsets[2 * i], lo, hi, dose)
                control = contained_tokens(offsets[2 * i + 1], lo, hi, dose)
                if target is None or control is None:
                    continue
                if abs(target[1] - control[1]) > MAX_COVER_DIFF_NT:
                    continue
                supply[name]["qualified"] += 1
                keep.append(i)
                selected.append((target[0], control[0]))
                covers[name].append((target[1], control[1]))
            if not keep:
                continue
            sub = tokenize_promoter_pairs(bundle, [chunk[i] for i in keep])
            with torch.no_grad():
                sub_native = promoter_margin_tensor(bundle, sub).detach()
            token_indices: list[list[int]] = []
            for target_indices, control_indices in selected:
                token_indices.extend((target_indices, control_indices))
            effects = explicit_token_mask_effects_for_batch(
                bundle, sub, token_indices, native_margins=sub_native
            )
            for j, pair_id in enumerate(sub.pair_ids):
                per_locus[pair_id][name] = float(effects["target_loss"][j]) - float(
                    effects["control_loss"][j]
                )
        if progress:
            print(f"[D15] {min(start + batch_pairs, len(pairs))}/{len(pairs)}", flush=True)
    return per_locus, supply, covers, clean


def summarize(values, clusters, name: str) -> dict[str, Any]:
    # cluster_bootstrap_interval returns (low, high) ONLY -- the point estimate is not part of
    # its return value and must be computed separately.
    low, high = cluster_bootstrap_interval(values, clusters, draws=DRAWS, seed=SEED, alpha=0.05)
    mean = statistics.mean(values)
    observed, null = paired_cluster_sign_permutation(
        values, clusters, permutations=PERMUTATIONS, seed=SEED
    )
    ordered = sorted(values)
    trim = int(0.1 * len(ordered))
    return {
        "endpoint": name,
        "n": len(values),
        "mean": mean,
        "ci95": [low, high],
        "median": statistics.median(values),
        "trimmed10": statistics.mean(ordered[trim : len(ordered) - trim]) if trim else mean,
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "positive_fraction": sum(1 for v in values if v > 0) / len(values),
        "raw_p": add_one_permutation_pvalue(observed, null, alternative="two-sided"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["development", "confirmation"])
    parser.add_argument("--smoke", action="store_true", help="memorials and mechanics only")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-pairs", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in COHORT.open()]
    loci = [r for r in rows if r["split"] == args.split]
    if args.limit:
        loci = loci[: args.limit]
    pairs = [
        (r["example_id"], r["target"]["clean_sequence"], r["control"]["clean_sequence"])
        for r in loci
    ]
    cluster_of = {r["example_id"]: r["homology_cluster"] for r in loci}

    bundle = load_dnabert2_promoter_classifier()
    per_locus, supply, covers, clean = score_arms(bundle, pairs, args.batch_pairs)

    # ---------------------------------------------------------------- memorials
    clean_delta = [t - c for t, c in clean.values()]
    memorials: dict[str, Any] = {
        "M1_clean_anchor": {
            "mean_target_minus_control": statistics.mean(clean_delta),
            "minimum_required": CLEAN_ANCHOR_MINIMUM,
            "passed": statistics.mean(clean_delta) >= CLEAN_ANCHOR_MINIMUM,
        },
        "M2a_equal_dose": {
            "note": "instrument-enforced: explicit_token_mask_effects_for_batch raises on "
                    "unequal dose. FLAGGED as architecture-forced, not counted as evidence.",
            "architecture_forced": True,
        },
        # The per-pair bound |cover_t - cover_c| <= MAX_COVER_DIFF_NT is enforced by the
        # qualification filter, so "worst <= tolerance" is FORCED and validates nothing. The
        # failable quantity is whether the SURVIVING pairs are systematically biased: the filter
        # never constrains the per-arm MEAN difference, which a tokenization asymmetry between
        # target and pseudo-TSS sequence would move.
        "M2b_cover_balance": {
            "per_arm_mean_cover_difference_nt": {
                k: round(statistics.mean(a - b for a, b in v), 3) for k, v in covers.items()
            },
            "worst_absolute_mean_difference_nt": max(
                (abs(statistics.mean(a - b for a, b in v)) for v in covers.values()), default=0.0
            ),
            "tolerance_nt": 2.0,
            "mean_cover_by_arm": {
                k: [
                    round(statistics.mean(a for a, _ in v), 2),
                    round(statistics.mean(b for _, b in v), 2),
                ]
                for k, v in covers.items()
            },
            "per_pair_bound_note": (
                f"the per-pair |cover_t - cover_c| <= {MAX_COVER_DIFF_NT} nt bound is enforced by "
                "the qualification filter and is therefore FORCED; it is reported, not counted"
            ),
        },
    }
    memorials["M2b_cover_balance"]["passed"] = (
        memorials["M2b_cover_balance"]["worst_absolute_mean_difference_nt"] <= 2.0
    )

    arm_means = {
        name: statistics.mean([per_locus[p][name] for p in per_locus if name in per_locus[p]])
        for name, _, _ in ARMS
        if any(name in per_locus[p] for p in per_locus)
    }
    far_abs = statistics.mean([abs(arm_means[n]) for n in FAR_NAMES if n in arm_means])
    memorials["M3_non_degeneracy"] = {
        "abs_band_mean": abs(arm_means.get("B", 0.0)),
        "mean_abs_far_window": far_abs,
        "passed": abs(arm_means.get("B", 0.0)) > far_abs,
        "note": "if masking 4x more far-upstream sequence does not do more, the band arm is "
                "saturated and E_band is uninterpretable (not 'failed')",
    }

    # M4 batch invariance: rescore a fixed probe at a different batch size
    probe = pairs[:16]
    probe_locus, _, _, _ = score_arms(bundle, probe, batch_pairs=4, progress=False)
    worst = 0.0
    for pair_id, arms in probe_locus.items():
        for name, value in arms.items():
            if name in per_locus.get(pair_id, {}):
                worst = max(worst, abs(value - per_locus[pair_id][name]))
    memorials["M4_batch_invariance"] = {
        "worst_absolute_difference": worst,
        "tolerance": 1e-4,
        "passed": worst < 1e-4,
        "probe_loci": len(probe),
    }

    # M1 and M3 are statements about POPULATION MEANS (clean anchor, dose response). At smoke
    # scale they fail on sampling noise alone -- gating mechanics on them is a category error.
    # They are evaluated only when the split supplies enough loci. Amendment 1, pre-outcome.
    POPULATION_MEMORIALS = {"M1_clean_anchor", "M3_non_degeneracy"}
    MINIMUM_LOCI_FOR_POPULATION_MEMORIALS = 100
    evaluable = len(clean) >= MINIMUM_LOCI_FOR_POPULATION_MEMORIALS
    for key in POPULATION_MEMORIALS:
        memorials[key]["population_memorial"] = True
        memorials[key]["evaluated"] = evaluable
        if not evaluable:
            memorials[key]["note_scale"] = (
                f"not evaluated: {len(clean)} loci < {MINIMUM_LOCI_FOR_POPULATION_MEMORIALS}"
            )
    blocking = [
        k
        for k, v in memorials.items()
        if isinstance(v, dict)
        and v.get("passed") is False
        and not v.get("architecture_forced")
        and v.get("evaluated", True)
    ]

    payload: dict[str, Any] = {
        "study": "dnabert2_reconstruction_atlas",
        "experiment": "D15_region_density_vs_attention_prior",
        "split": args.split,
        "mode": "smoke" if args.smoke else "confirmatory",
        "config": {
            "tss_index": TSS_INDEX,
            "K_window": K_WINDOW,
            "K_band": K_BAND,
            "max_cover_diff_nt": MAX_COVER_DIFF_NT,
            "SESOI": SESOI,
            "arms": {name: [lo - TSS_INDEX, hi - TSS_INDEX] for name, lo, hi in ARMS},
        },
        "memorials": memorials,
        "blocking_memorial_failures": blocking,
        "supply": {
            k: {**v, "rate": round(v["qualified"] / max(1, v["attempted"]), 4)}
            for k, v in supply.items()
        },
        "arm_means": arm_means,
    }

    if args.smoke or blocking:
        payload["note"] = (
            "SMOKE: memorials and mechanics only, no endpoints scored."
            if args.smoke
            else "BLOCKED: memorial failure; endpoints not scored."
        )
        status = "PREFLIGHT"
    else:
        # ------------------------------------------------------------ endpoints
        e_reg_rows = [
            (p, d["S"] - statistics.mean([d[f] for f in FAR_NAMES if f in d]))
            for p, d in per_locus.items()
            if "S" in d and any(f in d for f in FAR_NAMES)
        ]
        e_band_rows = [(p, d["S"] - d["B"]) for p, d in per_locus.items() if "S" in d and "B" in d]
        n1_rows = [(p, d["F2"] - d["F4"]) for p, d in per_locus.items() if "F2" in d and "F4" in d]
        u_rows = [(p, d["S"] - d["U"]) for p, d in per_locus.items() if "S" in d and "U" in d]

        results = {}
        for key, rows_ in (
            ("E_reg", e_reg_rows),
            ("E_band", e_band_rows),
            ("N1_far_vs_far", n1_rows),
            ("SPEC_S_minus_U", u_rows),
        ):
            if rows_:
                results[key] = summarize(
                    [v for _, v in rows_], [cluster_of[p] for p, _ in rows_], key
                )
        adjusted = holm_adjust({k: results[k]["raw_p"] for k in ("E_reg", "E_band") if k in results})
        for k, v in adjusted.items():
            results[k]["holm_p"] = v
            results[k]["qualifies"] = bool(
                results[k]["mean"] >= SESOI and results[k]["ci95"][0] > 0 and v < 0.05
            )

        density = {
            "signal_per_token": arm_means["S"] / K_WINDOW,
            "far_window_per_token": statistics.mean(arm_means[n] for n in FAR_NAMES) / K_WINDOW,
            "band_per_token": arm_means["B"] / K_BAND,
        }
        density["signal_over_far_window_ratio"] = (
            density["signal_per_token"] / density["far_window_per_token"]
            if density["far_window_per_token"]
            else None
        )
        payload["endpoints"] = results
        payload["density"] = density
        payload["additivity"] = {
            "five_times_mean_far_window": 5 * statistics.mean(arm_means[n] for n in FAR_NAMES),
            "measured_band": arm_means["B"],
        }
        status = "SEALED" if args.split == "confirmation" else "EXPLORATORY"

    payload["per_locus"] = [
        {"locus": p, "cluster": cluster_of[p], **per_locus[p]} for p in sorted(per_locus)
    ]

    write_result_bundle(
        args.out,
        payload,
        experiment_id="D15_region_density_vs_attention_prior",
        status=status,
        model_identity={
            "name": bundle.checkpoint.name,
            "checkpoint": bundle.checkpoint.hf_id,
            "revision": bundle.checkpoint.revision,
            "upstream_sha256": dict(bundle.checkpoint.upstream_sha256),
        },
        input_paths=[COHORT],
        parameters={
            "split": args.split,
            "K_window": K_WINDOW,
            "K_band": K_BAND,
            "max_cover_diff_nt": MAX_COVER_DIFF_NT,
            "SESOI": SESOI,
            "draws": DRAWS,
            "permutations": PERMUTATIONS,
        },
        seeds={"analysis": SEED},
        command=sys.argv,
        repo=REPO,
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "per_locus"}, indent=2))
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
