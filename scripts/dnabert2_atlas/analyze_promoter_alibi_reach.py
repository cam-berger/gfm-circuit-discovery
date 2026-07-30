#!/usr/bin/env python
"""D15 §2/§3 — ALiBi-forced [CLS] reach, measured-minus-prior attention, behavioural band profile.

ZERO model forward passes. Every input is a committed artifact; only the pinned tokenizer is loaded
(no model weights) so that the ALiBi-only positional prior can be computed on real BPE offsets.

This is the reproducible source of the §2 and §3 tables in
studies/dnabert2_reconstruction_atlas/preregistrations/PREREGISTRATION_D15_ATTENTION_CAUSALITY_DISSOCIATION.md.

Status is EXPLORATORY: §2 is architecture (not evidence) and §3 is a hypothesis-generating analysis
of already-open development data and reused head-atlas artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.adapters.dnabert2.promoter_head_atlas import PROMOTER_SOURCE_BANDS
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    cluster_bootstrap_interval,
    paired_cluster_sign_permutation,
)

REPO = Path(__file__).resolve().parents[2]
COHORT = REPO / "data/processed/dnabert2_atlas/promoter_source_localization_v1/cohort.jsonl"
DEVRAW = REPO / "data/processed/dnabert2_atlas/promoter_source_localization_v1/development.raw.jsonl"
ATLAS = REPO / "results/dnabert2_reconstruction_atlas/promoter_head_atlas_{}_v1.json"

BAND_ORDER = ["special", *PROMOTER_SOURCE_BANDS]
TSS_INDEX = 249
SEED = 20260729
DRAWS = 8000
PERMUTATIONS = 9999
SIGNAL_TILES = (24, 25, 26, 27)  # -9:+31
D5_HEADS = ("L2H2", "L4H6", "L4H7", "L9H7", "L10H1", "L10H5", "L11H2", "L11H5")


def alibi_slopes(n_heads: int) -> list[float]:
    """Verbatim re-implementation of bert_layers.py:372 _get_alibi_head_slopes."""

    def power_of_two(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return power_of_two(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return power_of_two(closest) + alibi_slopes(2 * closest)[0::2][: n_heads - closest]


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    slopes = alibi_slopes(12)
    # The CLASSIFIER's own tokenizer, not load_dnabert2_tokenizer() -- that one defaults to the
    # base DNABERT-2 checkpoint, and the attention this script compares against was measured with
    # the fine-tuned classifier's tokenizer. No forward passes are run.
    bundle = load_dnabert2_promoter_classifier()
    tokenizer = bundle.tokenizer

    rows = [json.loads(line) for line in COHORT.open()]
    cluster_of = {r["example_id"]: r["homology_cluster"] for r in rows}

    # ---------------------------------------------------- ALiBi-only prior on real BPE offsets
    prior_samples: dict[int, dict[str, list[float]]] = {
        h: defaultdict(list) for h in range(12)
    }
    token_counts: list[int] = []
    for record in rows:
        encoded = tokenizer(
            record["target"]["clean_sequence"], return_offsets_mapping=True, add_special_tokens=True
        )
        offsets = encoded["offset_mapping"]
        n_tokens = len(offsets)
        token_counts.append(n_tokens)
        allocation = []
        for start, end in offsets:
            row = dict.fromkeys(BAND_ORDER, 0.0)
            if end <= start:
                row["special"] = 1.0
            else:
                relative_start, relative_end = start - TSS_INDEX, end - TSS_INDEX
                length = relative_end - relative_start
                for name, (low, high) in PROMOTER_SOURCE_BANDS.items():
                    row[name] = (
                        max(0, min(relative_end, high) - max(relative_start, low)) / length
                    )
            allocation.append(row)
        for head, slope in enumerate(slopes):
            weights = [math.exp(-slope * j) for j in range(n_tokens)]
            total = sum(weights)
            for name in BAND_ORDER:
                prior_samples[head][name].append(
                    sum(weights[j] * allocation[j][name] for j in range(n_tokens)) / total
                )
    prior = {
        h: {k: statistics.mean(v) for k, v in prior_samples[h].items()} for h in range(12)
    }

    payload: dict[str, Any] = {
        "study": "dnabert2_reconstruction_atlas",
        "experiment": "D15_alibi_reach_and_attention_prior",
        "status_note": "Section 2 is ARCHITECTURE, not evidence. Section 3 is hypothesis-generating "
                       "on already-open development data and reused head-atlas artifacts.",
        "geometry": {
            "n_sequences": len(rows),
            "mean_window_tokens": statistics.mean(token_counts),
            "tss_index": TSS_INDEX,
            "cls_window_index": 0,
            "cls_tss_relative_nt": -TSS_INDEX,
            "bands": {k: list(v) for k, v in PROMOTER_SOURCE_BANDS.items()},
        },
        "section2_alibi_prior": {
            f"H{h}": {
                "slope": slopes[h],
                "effective_reach_tokens": 1.0 / slopes[h],
                **{k: round(v, 6) for k, v in prior[h].items()},
            }
            for h in range(12)
        },
    }

    # ---------------------------------------------------- measured minus prior, all 144 heads
    per_head: dict[str, dict[str, Any]] = {}
    for split in ("development", "confirmation"):
        atlas = json.loads(Path(str(ATLAS).format(split)).read_text())
        for summary in atlas["head_summaries"]:
            label, index = summary["head_label"], summary["head"]
            record = per_head.setdefault(
                label, {"layer": summary["layer"], "head_index": index, "slope": slopes[index]}
            )
            record[f"{split}_necessity"] = summary["necessity_mean"]
            record[f"{split}_sufficiency"] = summary["sufficiency_mean"]
            for name, values in (summary.get("attention") or {}).items():
                record[f"{split}_measured_{name}"] = values["target_mean"]
                record[f"{split}_excess_{name}"] = values["target_mean"] - prior[index][name]
    payload["section3a_measured_minus_prior"] = per_head

    # ---------------------------------------------------- reach vs causal effect, within layer
    reach_tests: dict[str, Any] = {}
    for split in ("development", "confirmation"):
        for endpoint in ("necessity", "sufficiency"):
            rhos = []
            for layer in range(12):
                heads = [v for v in per_head.values() if v["layer"] == layer]
                rhos.append(
                    spearman(
                        [math.log10(1 / v["slope"]) for v in heads],
                        [v[f"{split}_{endpoint}"] for v in heads],
                    )
                )
            positive = sum(1 for r in rhos if r > 0)
            reach_tests[f"{split}|{endpoint}"] = {
                "per_layer_spearman": [round(r, 4) for r in rhos],
                "mean_within_layer_spearman": round(statistics.mean(rhos), 4),
                "layers_positive": positive,
                "sign_test_p_two_sided": round(
                    min(
                        1.0,
                        2
                        * sum(math.comb(12, k) for k in range(max(positive, 12 - positive), 13))
                        / 2**12,
                    ),
                    5,
                ),
            }
    payload["section3b_reach_vs_causal_effect"] = {
        "null_hypothesis": "head reach is unrelated to exact-patch causal effect, within layer",
        "tests": reach_tests,
        "verdict": "NULL — nothing qualifies. The D14 prereg §2 prediction that only long-reach "
                   "heads could carry downstream evidence to [CLS] FAILS.",
    }

    # ---------------------------------------------------- behavioural band profile (development)
    mask_rows = [
        json.loads(line)
        for line in DEVRAW.open()
    ]
    mask_rows = [r for r in mask_rows if r["operation"] == "mask"]
    by_tile: dict[int, list[tuple[str, float]]] = defaultdict(list)
    tile_relative: dict[int, tuple[int, int]] = {}
    for r in mask_rows:
        by_tile[r["tile_index"]].append((r["pair_id"], r["target_specific_loss"]))
        tile_relative[r["tile_index"]] = (r["relative_start"], r["relative_end"])

    def tile_band(tile: int) -> str:
        midpoint = sum(tile_relative[tile]) / 2
        for name, (low, high) in PROMOTER_SOURCE_BANDS.items():
            if low <= midpoint < high:
                return name
        return "outside"

    band_locus: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for tile, values in by_tile.items():
        band = tile_band(tile)
        for pair_id, value in values:
            band_locus[band][pair_id].append(value)

    band_profile = {}
    for band, per_locus in band_locus.items():
        ids = sorted(per_locus)
        values = [statistics.mean(per_locus[i]) for i in ids]
        clusters = [cluster_of.get(i, i) for i in ids]
        low, high = cluster_bootstrap_interval(values, clusters, draws=DRAWS, seed=SEED)
        mean = statistics.mean(values)
        observed, null = paired_cluster_sign_permutation(
            values, clusters, permutations=PERMUTATIONS, seed=SEED
        )
        band_profile[band] = {
            "nt_range": list(PROMOTER_SOURCE_BANDS[band]),
            "n_tiles": sum(1 for t in by_tile if tile_band(t) == band),
            "n_loci": len(values),
            "mean_per_tile_loss": round(mean, 4),
            "ci95": [round(low, 4), round(high, 4)],
            "sign_flip_p": round(
                add_one_permutation_pvalue(observed, null, alternative="two-sided"), 4
            ),
        }
    payload["section3c_behavioural_band_profile"] = {
        "source": "D8 development 30-tile mask scan (already open)",
        "bands": band_profile,
        "per_tile": {
            str(t): {
                "relative": list(tile_relative[t]),
                "n": len(v),
                "mean": round(statistics.mean(x for _, x in v), 4),
            }
            for t, v in sorted(by_tile.items())
        },
    }

    # ---------------------------------------------------- paired per-tile dissociation
    far_tiles = [t for t in by_tile if tile_relative[t][1] <= -50]
    far_by_locus: dict[str, list[float]] = defaultdict(list)
    signal_by_locus: dict[str, list[float]] = defaultdict(list)
    for tile, values in by_tile.items():
        for pair_id, value in values:
            if tile in far_tiles:
                far_by_locus[pair_id].append(value)
            if tile in SIGNAL_TILES:
                signal_by_locus[pair_id].append(value)
    shared = sorted(set(far_by_locus) & set(signal_by_locus))
    differences = [
        statistics.mean(signal_by_locus[i]) - statistics.mean(far_by_locus[i]) for i in shared
    ]
    clusters = [cluster_of.get(i, i) for i in shared]
    low, high = cluster_bootstrap_interval(differences, clusters, draws=DRAWS, seed=SEED)
    mean = statistics.mean(differences)
    observed, null = paired_cluster_sign_permutation(
        differences, clusters, permutations=PERMUTATIONS, seed=SEED
    )
    payload["section3c_paired_per_tile_contrast"] = {
        "definition": "per-locus mean mask loss over tiles -9:+31 minus mean over far-upstream "
                      f"tiles (<= -50 nt; {len(far_tiles)} tiles)",
        "n_loci": len(shared),
        "mean": round(mean, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "sign_flip_p": round(
            add_one_permutation_pvalue(observed, null, alternative="two-sided"), 4
        ),
        "caveat": "A per-tile null over a wide band is NOT a band-level null; see prereg §3d. "
                  "The D15 confirmatory arms test the band level.",
    }

    # ---------------------------------------------------- downgraded post-hoc enrichment
    observed_statistic = statistics.mean(
        math.log10(1 / slopes[int(h.split("H")[1])]) for h in D5_HEADS
    )
    payload["section3b_downgraded_posthoc_enrichment"] = {
        "statistic": "mean log10(1/slope) over D5's eight frozen head indices",
        "observed": round(observed_statistic, 4),
        "null_mean": round(statistics.mean(math.log10(1 / m) for m in slopes), 4),
        "status": "DOWNGRADED — the statistic was chosen after seeing the head list, and the "
                  "better-powered within-layer test in section3b is NULL.",
    }

    write_result_bundle(
        args.out,
        payload,
        experiment_id="D15_alibi_reach_and_attention_prior",
        status="EXPLORATORY",
        model_identity={
            "name": bundle.checkpoint.name,
            "checkpoint": bundle.checkpoint.hf_id,
            "revision": bundle.checkpoint.revision,
            "note": "tokenizer only; no model weights loaded and no forward passes run",
        },
        input_paths=[
            COHORT,
            DEVRAW,
            Path(str(ATLAS).format("development")),
            Path(str(ATLAS).format("confirmation")),
        ],
        parameters={"draws": DRAWS, "permutations": PERMUTATIONS, "tss_index": TSS_INDEX},
        seeds={"analysis": SEED},
        command=sys.argv,
        repo=REPO,
    )
    print(json.dumps(payload["section3b_reach_vs_causal_effect"]["tests"], indent=2))
    print(json.dumps(payload["section3c_behavioural_band_profile"]["bands"], indent=2))
    print(json.dumps(payload["section3c_paired_per_tile_contrast"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
