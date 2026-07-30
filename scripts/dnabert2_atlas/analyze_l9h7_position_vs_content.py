#!/usr/bin/env python
"""D17 companion — the two zero-forward-pass analyses that settle the L9H7 question.

(1) THE CONTROL ARM, recovered from the committed record. D5's head atlas measured [CLS] band
    attention on BOTH branches: real TSS windows and same-locus pseudo-TSS windows with a 300-nt
    TSS exclusion radius. D16's comparison script computed the control values and then dropped
    them at the writer, emitting target_* only. They answer the question directly: if L9H7 attends
    to window positions 199-247 just as hard in windows containing no TSS, it is not reading a TSS.

(2) PRIOR-SUBTRACTED ROTATION EXCESS. The D17 decision rule compares raw band mass, which is
    invalid across rotations because a head's ALiBi prior is not uniform over the window: when the
    displaced content band lands near [CLS], a short-reach head gains mass for free. This is why
    L11H2 -- the archetype positional head -- scores F=0.664 at k=199 under the raw rule. Excess
    over the per-rotation ALiBi prior removes that artifact.
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

REPO = Path(__file__).resolve().parents[2]
COHORT = REPO / "data/processed/dnabert2_atlas/promoter_source_localization_v1/cohort.jsonl"
ATLAS = REPO / "results/dnabert2_reconstruction_atlas/promoter_head_atlas_{}_v1.json"

BAND_LO, BAND_HI = 199, 248
ROTATIONS = (0, 25, 125, 149, 175, 199)
FOCUS = ("L9H7", "L11H2", "L11H5")


def alibi_slopes(n: int) -> list[float]:
    def p2(m: int) -> list[float]:
        s = 2 ** (-(2 ** -(math.log2(m) - 3)))
        return [s * s**i for i in range(m)]

    if math.log2(n).is_integer():
        return p2(n)
    c = 2 ** math.floor(math.log2(n))
    return p2(c) + alibi_slopes(2 * c)[0::2][: n - c]


def overlap_fractions(offsets, lo: int, hi: int) -> list[float]:
    out = []
    for start, end in offsets:
        out.append(0.0 if end <= start else max(0, min(end, hi) - max(start, lo)) / (end - start))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "study": "dnabert2_reconstruction_atlas",
        "experiment": "D17_l9h7_position_vs_content",
        "status_note": "Zero forward passes. (1) recovers the pseudo-TSS control arm already "
                       "present in the committed head atlas; (2) recomputes the D17 rotation "
                       "reference prior so raw band mass is not compared across rotations.",
    }

    # ---------------------------------------------------------------- (1) the control arm
    control: dict[str, Any] = {}
    for split in ("development", "confirmation"):
        atlas = json.loads(Path(str(ATLAS).format(split)).read_text())
        bands = atlas["candidate_attention_source_bands"]
        rows = {}
        for head, per_band in bands.items():
            rows[head] = {
                band: {
                    "target": values["target_mean"],
                    "pseudo_tss_control": values["control_mean"],
                    "target_minus_control": values["target_minus_control"],
                }
                for band, values in per_band.items()
            }
        control[split] = {
            "n_unique_locus_blocks": atlas.get("n_unique_locus_blocks"),
            "heads": rows,
        }
    l9 = {
        s: control[s]["heads"]["L9H7"]["proximal_upstream"] for s in ("development", "confirmation")
    }
    payload["control_arm"] = {
        "what_the_control_is": "same-locus pseudo-TSS windows, 300 nt, TSS exclusion radius 300 nt, "
                               "GC-matched. They contain no TSS.",
        "L9H7_proximal_upstream": l9,
        "fraction_of_L9H7_band_attention_present_without_a_TSS": {
            s: l9[s]["pseudo_tss_control"] / l9[s]["target"] for s in l9
        },
        "reading": "If ~90% of L9H7's acquired band concentration is present in windows containing "
                   "no TSS, the concentration is not TSS recognition.",
        "all_heads": control,
    }

    # ---------------------------------------------------------------- (2) rotation prior
    bundle = load_dnabert2_promoter_classifier()
    tokenizer = bundle.tokenizer
    slopes = alibi_slopes(12)
    records = [json.loads(line) for line in COHORT.open()]

    prior_F: dict[int, dict[int, list[float]]] = {k: defaultdict(list) for k in ROTATIONS}
    prior_P: dict[int, dict[int, list[float]]] = {k: defaultdict(list) for k in ROTATIONS}
    for record in records:
        native = record["target"]["clean_sequence"]
        for k in ROTATIONS:
            rotated = native[k:] + native[:k]
            encoded = tokenizer(rotated, return_offsets_mapping=True, add_special_tokens=True)
            offsets = encoded["offset_mapping"]
            f_frac = overlap_fractions(offsets, BAND_LO - k, BAND_HI - k)
            p_frac = overlap_fractions(offsets, BAND_LO, BAND_HI)
            for head_index, slope in enumerate(slopes):
                weights = [math.exp(-slope * j) for j in range(len(offsets))]
                total = sum(weights)
                prior_F[k][head_index].append(
                    sum(w * f for w, f in zip(weights, f_frac)) / total
                )
                prior_P[k][head_index].append(
                    sum(w * f for w, f in zip(weights, p_frac)) / total
                )

    probe = json.loads(
        (REPO / "results/dnabert2_reconstruction_atlas/l9h7_rotation_probe_v1.json").read_text()
    )
    corrected: dict[str, Any] = {}
    for head in FOCUS:
        head_index = int(head.split("H")[1])
        rows = {}
        for k in ROTATIONS:
            endpoint = probe["endpoints"][f"{head}|k{k}"]
            pf = statistics.mean(prior_F[k][head_index])
            pp = statistics.mean(prior_P[k][head_index])
            rows[str(k)] = {
                "F_measured": endpoint["F_intact"]["mean"],
                "F_prior": pf,
                "F_excess": endpoint["F_intact"]["mean"] - pf,
                "P_measured": endpoint["P_intact"]["mean"],
                "P_prior": pp,
                "P_excess": endpoint["P_intact"]["mean"] - pp,
                "discriminating": k in (125, 149, 175, 199),
            }
        corrected[head] = rows
    payload["prior_corrected_rotation"] = {
        "why": "raw F is not comparable across rotations: when the displaced band lands near [CLS] "
               "a short-reach head gains prior mass for free. L11H2 scores F=0.664 at k=199 under "
               "the raw rule with zero content sensitivity -- an architecture-forced false positive "
               "that excess removes.",
        "heads": corrected,
    }
    payload["rule_defect"] = {
        "id": "D17-F3",
        "statement": "The frozen rule's INDEX_ADDRESSED gate thresholds RAW P at 0.35, and 27.1% "
                     "of L9H7's measured mean P (0.1458 of 0.5376) is ALiBi prior. Prior-corrected, "
                     "P excess is +0.3918 -- still above the bar, but by 0.042 rather than 0.188. "
                     "Separately, raw F is not comparable across rotations (L11H2 reaches F=0.6637 "
                     "at k=199 with an excess of only +0.0253, 96% architecture), which is why "
                     "every reported F is an excess.",
        "affects_L9H7_verdict": False,
        "why_not": "L9H7's slope is 0.0039 (near-uniform prior), so its prior on the displaced "
                   "band is substantial while its measured mass there is near zero -- its excess "
                   "is strongly NEGATIVE. Correcting the defect strengthens INDEX_ADDRESSED "
                   "rather than weakening it.",
        "supersedes_v1_statement": "v1 of this artifact asserted that the frozen rule thresholds "
                                   "RAW F and would score L11H2 CONTENT_ADDRESSED. That was FALSE: "
                                   "the frozen rule thresholds dF and P and never touches raw F; "
                                   "applied as frozen L11H2 scores CONJUNCTIVE (mean dF -0.0075, "
                                   "mean P 0.0043). The F>=0.35 bar belonged to the DRAFT rule that "
                                   "the pre-freeze red-team removed. v1 is retained as superseded.",
    }

    write_result_bundle(
        args.out,
        payload,
        experiment_id="D17_l9h7_position_vs_content",
        status="EXPLORATORY",
        model_identity={
            "checkpoint": bundle.checkpoint.hf_id,
            "revision": bundle.checkpoint.revision,
            "note": "tokenizer only for the prior; no forward passes",
        },
        input_paths=[
            COHORT,
            Path(str(ATLAS).format("development")),
            Path(str(ATLAS).format("confirmation")),
            REPO / "results/dnabert2_reconstruction_atlas/l9h7_rotation_probe_v1.json",
        ],
        parameters={"rotations": list(ROTATIONS), "band": [BAND_LO, BAND_HI]},
        seeds={},
        command=sys.argv,
        repo=REPO,
    )
    print(json.dumps(payload["control_arm"]["L9H7_proximal_upstream"], indent=2))
    print(json.dumps(
        payload["control_arm"]["fraction_of_L9H7_band_attention_present_without_a_TSS"], indent=2))
    for head in FOCUS:
        print(f"\n{head} prior-corrected:")
        for k, row in corrected[head].items():
            print(f"  k={k:>4} F_meas {row['F_measured']:.4f} F_prior {row['F_prior']:.4f} "
                  f"F_excess {row['F_excess']:+.4f} | P_excess {row['P_excess']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
