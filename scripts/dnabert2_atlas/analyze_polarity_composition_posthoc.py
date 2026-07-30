#!/usr/bin/env python
"""D13 Task 1 — exploratory post-hoc re-analysis motivating composition-vs-position.

D12 confirmed a TSS polarity effect in mask sensitivity: downstream tiles {25, 27} lose more
target-specific classifier margin under masking than upstream tiles {21, 22, 23}
(`E2_tss_polarity = +0.3262, 95% CI [+0.1110, +0.5564], Holm p = 0.0096, n = 149`, recorded in
`results/dnabert2_reconstruction_atlas/promoter_domain_confirmation_v1.json`). The field's default
explanation for promoter-classifier positional effects is composition (GC/CpG content), not position
per se. This script re-aggregates the D12 confirmation rows -- already opened, spending nothing new
-- to ask, observationally, whether the target-minus-control GC/CpG composition differential tracks
the polarity. It does not intervene on composition and licenses no exclusion of the composition
explanation either way; it exists only to motivate and power D13
(`studies/dnabert2_reconstruction_atlas/preregistrations/PREREGISTRATION_D13_COMPOSITION_VS_POSITION.md`).

No model is loaded. This is pure arithmetic over two already-written JSONL files:
  --confirmation-raw   data/processed/dnabert2_atlas/promoter_domain_v1/confirmation.raw.jsonl
                        (D12's per-locus, per-tile `mask` rows)
  --cohort              data/processed/dnabert2_atlas/promoter_source_localization_v1/cohort.jsonl
                        (D8's annotated cohort; supplies target/control clean_sequence and the
                        homology_cluster used for cluster-robust inference)

Per locus, over downstream tiles {25, 27} and upstream tiles {21, 22, 23} (at least one tile
present per side -- the same completeness rule D12's E2 uses, and E2's exact tile set):

  polarity   = mean(target_specific_loss, present downstream tiles)
               - mean(target_specific_loss, present upstream tiles)
  delta_gc   = mean(delta_GC, present downstream tiles) - mean(delta_GC, present upstream tiles)
  delta_cpg  = mean(delta_CpG, present downstream tiles) - mean(delta_CpG, present upstream tiles)

where, per tile, `delta_GC = GC(target cover) - GC(control cover)` and `delta_CpG` is the same
contrast for CpG dinucleotide rate. Loci are then split at the median of `abs(delta_gc)` into a
best-matched half (lower composition differential) and a worst-matched half (higher), and the
polarity is recomputed within each. This script writes ``"status": "EXPLORATORY_POSTHOC"`` and
refuses to overwrite an existing output.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gfmcd.adapters.dnabert2 import PROMOTER_CHECKPOINTS
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.statistics import cluster_bootstrap_interval

# The exact D12 E2_tss_polarity tile set: downstream {25, 27}, upstream {21, 22, 23}. Tile 26
# is deliberately excluded -- its confirmation value was already opened by D8/D12's own record
# anchor and E2 excludes it for the same reason.
DOWNSTREAM_TILES: tuple[int, ...] = (25, 27)
UPSTREAM_TILES: tuple[int, ...] = (21, 22, 23)

BOOTSTRAP_DRAWS = 5000
# A fixed, documented base seed for this exploratory script's own bootstrap CIs. This script
# predates the D13 frozen config (it is written first, per task ordering) and is not gated by
# it, so the seed lives here rather than being read from a config. Per-summary sub-seeds are
# derived by XOR, mirroring the convention in scripts/dnabert2_atlas/run_promoter_domain.py.
BASE_SEED = 20260729

CLAIM_BOUNDARY = (
    "Computed on D12 confirmation rows that were already opened. Observational, not causal: it "
    "adjusts for composition, it does not intervene on it. It licenses no exclusion of the "
    "composition explanation and exists only to motivate and power D13."
)


def gc_fraction(seq: str) -> float:
    """GC fraction of one cover sequence."""

    seq = seq.upper()
    if not seq:
        raise ValueError("empty cover sequence")
    if set(seq) - set("ACGT"):
        raise ValueError(f"cover sequence has non-ACGT characters: {seq!r}")
    return sum(1 for base in seq if base in "GC") / len(seq)


def cpg_per_nt(seq: str) -> float:
    """CpG dinucleotide count per nucleotide (count of 'CG' steps / (length - 1))."""

    seq = seq.upper()
    if len(seq) < 2:
        raise ValueError(f"cover sequence too short for a dinucleotide rate: {seq!r}")
    if set(seq) - set("ACGT"):
        raise ValueError(f"cover sequence has non-ACGT characters: {seq!r}")
    count = sum(1 for i in range(len(seq) - 1) if seq[i] == "C" and seq[i + 1] == "G")
    return count / (len(seq) - 1)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError(
            "Pearson inputs must be aligned and contain at least two values"
        )
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        raise ValueError("Pearson correlation is undefined for a constant input")
    return numerator / (denom_x * denom_y)


def load_cohort(path: str | Path) -> dict[str, dict[str, Any]]:
    """example_id -> {target_sequence, control_sequence, cluster, split}."""

    cohort: dict[str, dict[str, Any]] = {}
    with Path(path).open() as handle:
        for line in handle:
            record = json.loads(line)
            example_id = str(record["example_id"])
            cohort[example_id] = {
                "target_sequence": str(record["target"]["clean_sequence"]),
                "control_sequence": str(record["control"]["clean_sequence"]),
                "cluster": str(record["homology_cluster"]),
                "split": str(record["split"]),
            }
    return cohort


def load_tile_covers(
    raw_path: str | Path, cohort: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[int, dict[str, float]]], int]:
    """pair_id -> tile_index -> {target_specific_loss, delta_gc, delta_cpg}.

    Restricted to `operation == "mask"` rows over the downstream/upstream tile set. Returns the
    per-pair-per-tile map and the count of raw rows consumed.
    """

    tiles_of_interest = set(DOWNSTREAM_TILES) | set(UPSTREAM_TILES)
    by_pair: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    n_rows = 0
    with Path(raw_path).open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("row_type") != "tile_effect":
                continue
            if record.get("operation") != "mask":
                continue
            tile_index = int(record["tile_index"])
            if tile_index not in tiles_of_interest:
                continue
            pair_id = str(record["pair_id"])
            locus = cohort.get(pair_id)
            if locus is None:
                raise KeyError(
                    f"pair_id {pair_id!r} (tile {tile_index}) missing from cohort {raw_path}"
                )
            if locus["split"] != "confirmation":
                raise ValueError(
                    f"pair_id {pair_id!r} is cohort split {locus['split']!r}, not "
                    "'confirmation' -- refusing to mix splits in the D12 re-analysis"
                )
            if tile_index in by_pair[pair_id]:
                raise ValueError(
                    f"duplicate mask row for pair_id {pair_id!r} tile {tile_index}"
                )
            target_cover = locus["target_sequence"][
                int(record["target_cover_start"]) : int(record["target_cover_end"])
            ]
            control_cover = locus["control_sequence"][
                int(record["control_cover_start"]) : int(record["control_cover_end"])
            ]
            by_pair[pair_id][tile_index] = {
                "target_specific_loss": float(record["target_specific_loss"]),
                "delta_gc": gc_fraction(target_cover) - gc_fraction(control_cover),
                "delta_cpg": cpg_per_nt(target_cover) - cpg_per_nt(control_cover),
            }
            n_rows += 1
    return dict(by_pair), n_rows


@dataclass(frozen=True)
class LocusRecord:
    pair_id: str
    cluster: str
    polarity: float
    delta_gc: float
    delta_cpg: float


def build_locus_records(
    by_pair: Mapping[str, Mapping[int, Mapping[str, float]]],
    cohort: Mapping[str, Mapping[str, Any]],
) -> list[LocusRecord]:
    """One record per locus with >=1 downstream and >=1 upstream tile present."""

    records: list[LocusRecord] = []
    for pair_id, tiles in sorted(by_pair.items()):
        inner_present = [t for t in DOWNSTREAM_TILES if t in tiles]
        outer_present = [t for t in UPSTREAM_TILES if t in tiles]
        if not inner_present or not outer_present:
            continue
        inner_tsl = statistics.mean(tiles[t]["target_specific_loss"] for t in inner_present)
        outer_tsl = statistics.mean(tiles[t]["target_specific_loss"] for t in outer_present)
        inner_gc = statistics.mean(tiles[t]["delta_gc"] for t in inner_present)
        outer_gc = statistics.mean(tiles[t]["delta_gc"] for t in outer_present)
        inner_cpg = statistics.mean(tiles[t]["delta_cpg"] for t in inner_present)
        outer_cpg = statistics.mean(tiles[t]["delta_cpg"] for t in outer_present)
        records.append(
            LocusRecord(
                pair_id=pair_id,
                cluster=cohort[pair_id]["cluster"],
                polarity=inner_tsl - outer_tsl,
                delta_gc=inner_gc - outer_gc,
                delta_cpg=inner_cpg - outer_cpg,
            )
        )
    return records


def cluster_balanced_mean(records: Sequence[LocusRecord], attribute: str) -> float:
    """Mean of per-cluster means -- the same convention as
    gfmcd.reconstruction_atlas.statistics.paired_cluster_sign_permutation's point estimate.
    """

    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.cluster].append(getattr(record, attribute))
    return statistics.mean(statistics.mean(values) for values in grouped.values())


def summarize_polarity(records: Sequence[LocusRecord], *, seed: int) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize polarity over zero loci")
    values = [record.polarity for record in records]
    clusters = [record.cluster for record in records]
    ci_low, ci_high = cluster_bootstrap_interval(
        values, clusters, draws=BOOTSTRAP_DRAWS, seed=seed
    )
    return {
        "mean": cluster_balanced_mean(records, "polarity"),
        "ci95": [ci_low, ci_high],
        "n": len(records),
        "n_clusters": len(set(clusters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-raw", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cohort = load_cohort(args.cohort)
    by_pair, n_rows = load_tile_covers(args.confirmation_raw, cohort)
    if n_rows == 0:
        raise SystemExit(
            "no eligible mask rows found over tiles "
            f"{sorted(set(DOWNSTREAM_TILES) | set(UPSTREAM_TILES))} -- refusing to summarize "
            "an empty re-analysis"
        )

    records = build_locus_records(by_pair, cohort)
    if not records:
        raise SystemExit("no loci had >=1 downstream and >=1 upstream tile -- refusing")

    overall = summarize_polarity(records, seed=BASE_SEED ^ 0x1111)

    polarities = [record.polarity for record in records]
    delta_gcs = [record.delta_gc for record in records]
    delta_cpgs = [record.delta_cpg for record in records]
    r_polarity_delta_gc = pearson_correlation(polarities, delta_gcs)
    r_polarity_delta_cpg = pearson_correlation(polarities, delta_cpgs)

    # Median split on abs(delta_gc): sort ascending, lower floor(n/2) is the best-matched
    # (composition-similar) half, the remainder is the worst-matched half. This is a strict
    # rank split (not a "<= median value" split), so it partitions deterministically into
    # floor(n/2) / ceil(n/2) regardless of ties at the median.
    ranked = sorted(records, key=lambda record: abs(record.delta_gc))
    half = len(ranked) // 2
    matched_half = ranked[:half]
    unmatched_half = ranked[half:]

    matched_summary = summarize_polarity(matched_half, seed=BASE_SEED ^ 0x2222)
    matched_summary["mean_abs_delta_gc"] = statistics.mean(
        abs(record.delta_gc) for record in matched_half
    )
    unmatched_summary = summarize_polarity(unmatched_half, seed=BASE_SEED ^ 0x3333)
    unmatched_summary["mean_abs_delta_gc"] = statistics.mean(
        abs(record.delta_gc) for record in unmatched_half
    )

    checkpoint = PROMOTER_CHECKPOINTS["prom_300_all"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "polarity_composition_posthoc_v1",
        "status": "EXPLORATORY_POSTHOC",
        "model_outcomes_computed": False,
        "provenance": {
            "confirmed_effect": (
                "results/dnabert2_reconstruction_atlas/promoter_domain_confirmation_v1.json "
                "(E2_tss_polarity = +0.3262, 95% CI [+0.1110, +0.5564], Holm p = 0.0096, n=149)"
            ),
            "confirmation_raw": str(args.confirmation_raw),
            "cohort": str(args.cohort),
        },
        "tiles": {
            "downstream": list(DOWNSTREAM_TILES),
            "upstream": list(UPSTREAM_TILES),
        },
        "definitions": {
            "polarity": (
                "per locus: mean(target_specific_loss, present downstream tiles) - "
                "mean(target_specific_loss, present upstream tiles); >=1 tile required per "
                "side (D12 E2_tss_polarity's own completeness rule and tile set)"
            ),
            "delta_gc": (
                "per tile cover: GC_fraction(target cover) - GC_fraction(control cover); "
                "then per locus, mean over present downstream tiles minus mean over present "
                "upstream tiles, identically to how polarity aggregates target_specific_loss"
            ),
            "delta_cpg": (
                "per tile cover: CpG_dinucleotide_count(target cover) / (length - 1) minus "
                "the same rate for the control cover; aggregated to a per-locus contrast "
                "identically to delta_gc"
            ),
            "matched_unmatched_split": (
                "loci ranked ascending by abs(delta_gc); the lower floor(n/2) form the "
                "best-matched half, the remaining ceil(n/2) form the worst-matched half"
            ),
        },
        "n_rows_consumed": n_rows,
        "n_loci": len(records),
        "polarity": overall,
        "mean_delta_gc": statistics.mean(delta_gcs),
        "mean_delta_cpg": statistics.mean(delta_cpgs),
        "correlations": {
            "r_polarity_delta_gc": r_polarity_delta_gc,
            "r_polarity_delta_cpg": r_polarity_delta_cpg,
            "n": len(records),
        },
        "matched_half": matched_summary,
        "unmatched_half": unmatched_summary,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "base_seed": BASE_SEED},
        "claim_boundary": CLAIM_BOUNDARY,
    }

    result_path, manifest_path = write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_polarity_composition_posthoc_v1",
        status="EXPLORATORY",
        model_identity={
            "hf_id": checkpoint.hf_id,
            "revision": checkpoint.revision,
            "note": (
                "no model loaded by this script; identity of the checkpoint that produced "
                "the consumed D12 confirmation rows, recorded for provenance only"
            ),
        },
        input_paths=(args.confirmation_raw, args.cohort),
        parameters={
            "downstream_tiles": list(DOWNSTREAM_TILES),
            "upstream_tiles": list(UPSTREAM_TILES),
        },
        seeds={"bootstrap_base_seed": BASE_SEED},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(f"wrote {result_path}", flush=True)
    print(f"wrote {manifest_path}", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
