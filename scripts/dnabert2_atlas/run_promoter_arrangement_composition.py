#!/usr/bin/env python
"""D14 — what carries the D12 TSS polarity: arrangement or composition?

Three arms applied per 10-nt tile at D12's E2 tiles:
  A  mask                  (already run by D12; REPLAYED here, not re-estimated)
  B  dinucleotide_reorder  (destroys arrangement, holds mono+dinucleotide counts exactly)
  C  gc_depletion          (changes GC with real bases, no [MASK])

Protocol: studies/dnabert2_reconstruction_atlas/PREREGISTRATION_D14_ARRANGEMENT_VS_COMPOSITION.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.promoter_native_execution import score_native_edit_rows
from gfmcd.reconstruction_atlas.promoter_native_source import (
    geometry_qualified_native_edit_ensemble,
)
from gfmcd.reconstruction_atlas.promoter_source_execution import (
    score_equal_dose_masked_tile,
)
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    cluster_bootstrap_interval,
    holm_adjust,
    paired_cluster_sign_permutation,
)

DOWNSTREAM = (25, 27)
UPSTREAM = (21, 22, 23)
E2_TILES = UPSTREAM + DOWNSTREAM
INERT_TILE = 5  # specificity control, TSS-relative -199:-189
SEED = 20260729
DRAWS = 8000
SESOI = 0.15
ARMS = ("dinucleotide_reorder", "gc_depletion")

TILE_WIDTH = 10
TOKEN_DOSE = 2
MAX_COVER_NT = 18
MAX_COVER_DIFF_NT = 2


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def build_rows(bundle, loci, policy, tiles, operation) -> tuple[list, dict]:
    """Construct geometry-qualified native edits for one arm, one tile set."""
    rows, supply = [], defaultdict(lambda: {"attempted": 0, "qualified": 0})
    for locus in loci:
        geometry = {t["tile_index"]: t for t in locus["localization"]["tiles"]}
        target = locus["target"]["clean_sequence"]
        control = locus["control"]["clean_sequence"]
        for tile_index in tiles:
            tile = geometry.get(tile_index)
            if tile is None or not tile.get("eligible"):
                continue
            supply[tile_index]["attempted"] += 1
            try:
                ensemble = geometry_qualified_native_edit_ensemble(
                    bundle.tokenizer,
                    target,
                    control,
                    start=int(tile["nominal_start"]),
                    end=int(tile["nominal_end"]),
                    gc_depletion_dose=int(policy["gc_depletion_dose"]),
                    policy=policy,
                    seed=SEED ^ stable_seed(f"{locus['example_id']}#{tile_index}"),
                    replicates=1,
                    operations=(operation,),
                )
            except Exception:
                continue
            if not ensemble.get(operation):
                continue
            supply[tile_index]["qualified"] += 1
            rows.append(
                {
                    "example_id": f"{locus['example_id']}#t{tile_index}",
                    "homology_cluster": locus["homology_cluster"],
                    "_locus": locus["example_id"],
                    "_tile": tile_index,
                    "_edit": ensemble[operation][0],
                    "target": {"clean_sequence": target},
                    "control": {"clean_sequence": control},
                    "native_source": {"edit_ensemble": {operation: ensemble[operation]}},
                }
            )
    return rows, {str(k): dict(v) for k, v in supply.items()}


def check_memorials(rows_by_arm) -> dict[str, Any]:
    """M1-M4: on the measurement path, each able to fail."""
    out: dict[str, Any] = {}
    for arm, rows in rows_by_arm.items():
        mono = dinuc = nonident = equal_dose = gc_exact = True
        for row in rows:
            t, c = row["_edit"]["target"], row["_edit"]["control"]
            nonident &= int(t["changed_nucleotides"]) > 0
            equal_dose &= int(t["changed_nucleotides"]) == int(c["changed_nucleotides"])
            if arm == "dinucleotide_reorder":
                mono &= bool(t["mononucleotide_preserved"])
                dinuc &= bool(t["dinucleotide_preserved"])
            if arm == "gc_depletion":
                gc_exact &= int(t["gc_count_change"]) < 0
        out[arm] = {
            "M3_non_identity": bool(nonident),
            "M4_equal_hamming_target_control": bool(equal_dose),
            "mean_hamming_dose_nt": round(
                statistics.mean(int(r["_edit"]["target"]["changed_nucleotides"]) for r in rows), 3
            )
            if rows
            else None,
        }
        # Report each arm-specific memorial ONLY on the arm where it can fail.
        # Reporting M1 on gc_depletion (or M2 on reorder) would be vacuously true
        # and would read as evidence while validating nothing.
        if arm == "dinucleotide_reorder":
            out[arm]["M1_dinucleotide_rule"] = bool(mono and dinuc)
        if arm == "gc_depletion":
            out[arm]["M2_GC_depletion_rule"] = bool(gc_exact)
    return out


def e2_by_locus(losses: dict[str, dict[int, float]]) -> dict[str, float]:
    out = {}
    for locus, tiles in losses.items():
        down = [tiles[t] for t in DOWNSTREAM if t in tiles]
        up = [tiles[t] for t in UPSTREAM if t in tiles]
        if down and up:
            out[locus] = statistics.mean(down) - statistics.mean(up)
    return out


def interval(values, clusters):
    lo, hi = cluster_bootstrap_interval(values, clusters, draws=DRAWS, seed=SEED)
    return {
        "mean": round(statistics.mean(values), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "n": len(values),
        "sd": round(statistics.stdev(values), 4) if len(values) > 1 else None,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="confirmation", choices=("development", "confirmation"))
    parser.add_argument("--smoke", action="store_true", help="memorials + replay only, no endpoints")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=32)
    parser.add_argument("--replay-loci", type=int, default=32)
    parser.add_argument(
        "--limit-loci",
        type=int,
        help="smoke only: cap loci so the gate is cheap and scores no endpoint",
    )
    args = parser.parse_args()
    if args.limit_loci and not args.smoke:
        raise SystemExit("--limit-loci is a smoke-phase flag; the confirmatory uses every locus")

    repo = Path(__file__).resolve().parents[2]
    cohort_path = repo / "data/processed/dnabert2_atlas/promoter_source_localization_v1/cohort.jsonl"
    d12_raw = repo / "data/processed/dnabert2_atlas/promoter_domain_v1/confirmation.raw.jsonl"
    policy_path = repo / "configs/dnabert2_atlas/promoter_native_source_rescue_v10.frozen.json"
    policy = json.loads(policy_path.read_text())["native_interventions"]

    cohort = [json.loads(l) for l in cohort_path.read_text().splitlines() if l.strip()]
    loci = [r for r in cohort if r.get("split") == args.split]
    if args.limit_loci:
        loci = loci[: args.limit_loci]
    print(f"{args.split}: {len(loci)} loci", flush=True)

    bundle = load_dnabert2_promoter_classifier(candidate="prom_300_all", device=args.device)
    print("model loaded", flush=True)

    # ---- positive control: EXACT replay of D12's stored arm-A losses ----
    stored: dict[tuple[str, int], float] = {}
    for line in d12_raw.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("operation") == "mask" and row.get("tile_index") in E2_TILES:
            stored[(row["pair_id"], row["tile_index"])] = float(row["target_specific_loss"])

    replay_subset = loci[: args.replay_loci]
    pairs = [
        (r["example_id"], r["target"]["clean_sequence"], r["control"]["clean_sequence"])
        for r in replay_subset
    ]
    deltas = []
    for tile_index in E2_TILES:
        for scored_row in score_equal_dose_masked_tile(
            bundle,
            pairs,
            tile_index=tile_index,
            tile_width_nt=TILE_WIDTH,
            token_dose=TOKEN_DOSE,
            maximum_cover_nt=MAX_COVER_NT,
            maximum_cover_difference_nt=MAX_COVER_DIFF_NT,
            batch_pairs=args.batch_pairs,
        ):
            key = (scored_row["pair_id"], tile_index)
            if key in stored:
                deltas.append(abs(scored_row["target_specific_loss"] - stored[key]))
    replay = {
        "compared_rows": len(deltas),
        "max_absolute_difference": round(max(deltas), 10) if deltas else None,
        "tolerance": 1e-4,
        "passed": bool(deltas) and max(deltas) < 1e-4,
    }
    print(f"replay positive control: {replay}", flush=True)

    # ---- build both native arms ----
    rows_by_arm, supply_by_arm = {}, {}
    for arm in ARMS:
        rows, supply = build_rows(bundle, loci, policy, E2_TILES, arm)
        rows_by_arm[arm], supply_by_arm[arm] = rows, supply
        print(f"{arm}: {len(rows)} qualified tile-edits", flush=True)

    memorials = check_memorials(rows_by_arm)
    memorials["M2_record_anchor_exact_replay"] = replay

    # ---- M6 clean-behaviour anchor + M5 batch invariance ----
    scored_by_arm: dict[str, list] = {}
    for arm, rows in rows_by_arm.items():
        scored_by_arm[arm] = score_native_edit_rows(
            bundle, rows, operations=(arm,), batch_pairs=args.batch_pairs
        )
    clean_gap = [
        s["target_clean_margin"] - s["control_clean_margin"]
        for s in scored_by_arm[ARMS[0]]
    ]
    memorials["M6_clean_behaviour_anchor"] = {
        "mean_target_minus_control_clean_margin": round(statistics.mean(clean_gap), 4),
        "passed": statistics.mean(clean_gap) > 0,
    }
    invariance_rows = rows_by_arm[ARMS[0]][:64]
    alt = score_native_edit_rows(bundle, invariance_rows, operations=(ARMS[0],), batch_pairs=8)
    base = {s["pair_id"]: s["target_specific_loss"] for s in scored_by_arm[ARMS[0]]}
    drift = max(abs(base[s["pair_id"]] - s["target_specific_loss"]) for s in alt)
    memorials["M5_batch_invariance"] = {
        "compared_rows": len(alt),
        "max_absolute_difference": float(f"{drift:.3e}"),
        "tolerance": 1e-4,
        "passed": drift < 1e-4,
    }

    # ---- specificity control: arm B at an inert far-upstream tile ----
    inert_rows, _ = build_rows(bundle, loci, policy, (INERT_TILE,), ARMS[0])
    specificity: dict[str, Any] = {"tile": INERT_TILE, "n": len(inert_rows)}
    if inert_rows:
        inert_scored = score_native_edit_rows(
            bundle, inert_rows, operations=(ARMS[0],), batch_pairs=args.batch_pairs
        )
        values = [s["target_specific_loss"] for s in inert_scored]
        clusters = [s["homology_cluster"] for s in inert_scored]
        specificity.update(interval(values, clusters))
        specificity["passed"] = abs(specificity["mean"]) < SESOI

    gates_pass = (
        replay["passed"]
        and memorials["M5_batch_invariance"]["passed"]
        and memorials["M6_clean_behaviour_anchor"]["passed"]
        and all(
            memorials[a][k]
            for a in ARMS
            for k in memorials[a]
            if k.startswith("M") and isinstance(memorials[a][k], bool)
        )
    )

    payload: dict[str, Any] = {
        "assay": "promoter_arrangement_vs_composition_v1",
        "study": "D14",
        "split": args.split,
        "phase": "smoke" if args.smoke else "confirmatory",
        "protocol": "studies/dnabert2_reconstruction_atlas/PREREGISTRATION_D14_ARRANGEMENT_VS_COMPOSITION.md",
        "memorials": memorials,
        "specificity_control_inert_tile": specificity,
        "supply_by_arm": supply_by_arm,
        "all_gates_passed": bool(gates_pass),
    }

    if args.smoke:
        payload["endpoints"] = "WITHHELD - smoke phase scores no endpoint"
        payload["smoke_verdict"] = "passed" if gates_pass else "failed"
        status = "PREFLIGHT" if gates_pass else "INSTRUMENT_UNRESOLVED"
    elif not gates_pass:
        payload["endpoints"] = "WITHHELD - a preregistered gate failed"
        status = "INSTRUMENT_UNRESOLVED"
    else:
        losses: dict[str, dict[str, dict[int, float]]] = {}
        for arm, scored in scored_by_arm.items():
            index = {r["example_id"]: r for r in rows_by_arm[arm]}
            per_locus: dict[str, dict[int, float]] = defaultdict(dict)
            for s in scored:
                meta = index[s["pair_id"]]
                per_locus[meta["_locus"]][meta["_tile"]] = s["target_specific_loss"]
            losses[arm] = per_locus
        mask_losses: dict[str, dict[int, float]] = defaultdict(dict)
        for (pair_id, tile_index), value in stored.items():
            mask_losses[pair_id][tile_index] = value

        e2 = {
            "mask": e2_by_locus(mask_losses),
            "reorder": e2_by_locus(losses[ARMS[0]]),
            "gc_depletion": e2_by_locus(losses[ARMS[1]]),
        }
        keep = sorted(set(e2["mask"]) & set(e2["reorder"]) & set(e2["gc_depletion"]))
        if len(keep) < 3:
            raise SystemExit(
                f"three-arm intersection is {len(keep)} loci; refusing to score endpoints"
            )
        cluster_of = {r["example_id"]: r["homology_cluster"] for r in loci}
        clusters = [cluster_of[p] for p in keep]

        endpoints = {
            "arms": {
                name: interval([values[p] for p in keep], clusters)
                for name, values in e2.items()
            },
            "P3_reorder_arrangement_contribution": interval(
                [e2["reorder"][p] for p in keep], clusters
            ),
            "P1_mask_minus_reorder": interval(
                [e2["mask"][p] - e2["reorder"][p] for p in keep], clusters
            ),
            "P2_gc_depletion_minus_reorder": interval(
                [e2["gc_depletion"][p] - e2["reorder"][p] for p in keep], clusters
            ),
            "S1_mask_minus_gc_depletion": interval(
                [e2["mask"][p] - e2["gc_depletion"][p] for p in keep], clusters
            ),
            "n_three_arm_intersection": len(keep),
        }
        # Holm across the three preregistered primaries, on add-one sign-flip p-values.
        contrasts = {
            "P3_reorder_arrangement_contribution": [e2["reorder"][p] for p in keep],
            "P1_mask_minus_reorder": [e2["mask"][p] - e2["reorder"][p] for p in keep],
            "P2_gc_depletion_minus_reorder": [
                e2["gc_depletion"][p] - e2["reorder"][p] for p in keep
            ],
        }
        # paired_cluster_sign_permutation returns (observed, null_distribution);
        # the p-value must come from add_one_permutation_pvalue over that null.
        # Two-sided: the protocol prespecified no direction for these contrasts.
        raw_p = {}
        for name, values in contrasts.items():
            observed, null = paired_cluster_sign_permutation(
                values, clusters, permutations=9999, seed=SEED
            )
            raw_p[name] = float(
                add_one_permutation_pvalue(observed, null, alternative="two-sided")
            )
        adjusted = holm_adjust(raw_p)
        for name in raw_p:
            endpoints[name]["p_sign_flip"] = round(raw_p[name], 5)
            endpoints[name]["p_holm"] = round(float(adjusted[name]), 5)
            endpoints[name]["holm_significant"] = bool(adjusted[name] <= 0.05)
        endpoints["holm_family"] = sorted(raw_p)
        payload["endpoints"] = endpoints

        # Per-locus values are embedded IN the tracked result, not written as a sibling
        # .jsonl: `results/dnabert2_reconstruction_atlas/**/*.jsonl` is gitignored for this
        # study, and a reproducibility record that a fresh checkout cannot see is not a
        # record. 83 rows is small enough to carry inline.
        payload["per_locus"] = [
            {
                "locus": p,
                "homology_cluster": cluster_of[p],
                "E2_mask": round(e2["mask"][p], 6),
                "E2_reorder": round(e2["reorder"][p], 6),
                "E2_gc_depletion": round(e2["gc_depletion"][p], 6),
            }
            for p in keep
        ]

        p1 = endpoints["P1_mask_minus_reorder"]
        p3 = endpoints["P3_reorder_arrangement_contribution"]
        p2 = endpoints["P2_gc_depletion_minus_reorder"]
        payload["verdict"] = {
            "arrangement_is_not_the_carrier": bool(
                p1["excludes_zero"] and not p3["excludes_zero"]
            ),
            "achieved_upper_bound_on_arrangement": max(abs(x) for x in p3["ci95"]),
            "bound_is_below_SESOI": max(abs(x) for x in p3["ci95"]) < SESOI,
            "composition_supported_as_carrier": bool(p2["excludes_zero"]),
            "mask_artifact_required": bool(
                endpoints["S1_mask_minus_gc_depletion"]["excludes_zero"]
                and endpoints["S1_mask_minus_gc_depletion"]["mean"] > 0
            ),
            "dose_monotonicity_check": {
                "reorder_mean_hamming": memorials[ARMS[0]]["mean_hamming_dose_nt"],
                "gc_depletion_mean_hamming": memorials[ARMS[1]]["mean_hamming_dose_nt"],
                "reorder_effect": endpoints["arms"]["reorder"]["mean"],
                "gc_depletion_effect": endpoints["arms"]["gc_depletion"]["mean"],
                "effect_is_non_monotone_in_dose": None,
            },
        }
        dm = payload["verdict"]["dose_monotonicity_check"]
        if dm["reorder_mean_hamming"] and dm["gc_depletion_mean_hamming"]:
            dm["effect_is_non_monotone_in_dose"] = bool(
                dm["reorder_mean_hamming"] > dm["gc_depletion_mean_hamming"]
                and abs(dm["reorder_effect"]) < abs(dm["gc_depletion_effect"])
            )
        status = "SEALED" if payload["verdict"]["arrangement_is_not_the_carrier"] else "NULL"

    result_path, manifest_path = write_result_bundle(
        args.out,
        payload,
        experiment_id="D14_arrangement_vs_composition",
        status=status,
        model_identity={
            "checkpoint": "magicslabnu/DNABERT-2-finetuned-prom_300_all",
            "revision": "7f2a5f5b8e7e6fffb2dcb396a77d1bc78b7f5f12",
        },
        input_paths=[cohort_path, d12_raw, policy_path],
        parameters={
            "split": args.split,
            "smoke": args.smoke,
            "tiles": list(E2_TILES),
            "downstream": list(DOWNSTREAM),
            "upstream": list(UPSTREAM),
            "arms": list(ARMS),
            "SESOI": SESOI,
            "bootstrap_draws": DRAWS,
            "batch_pairs": args.batch_pairs,
        },
        seeds={"analysis": SEED},
        command=sys.argv,
        repo=repo,
    )
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {result_path}\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
