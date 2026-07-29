#!/usr/bin/env python
"""Outcome-blind supply AND matching-feasibility audit for the D13 composition cohort.

Reports, per split, without computing any model score: pair count, unique
1-Mb blocks, zero GUE hits, zero block overlap with prior cohorts, and --
the real risk this study faces -- how many loci admit a composition-matched
downstream/upstream cover pair inside the frozen tight caliper (GC 0.05,
CpG 0.03), and how many admit an unmatched (planted-positive) pair at the
minimum GC separation (0.20). Uses the tokenizer only, to bind BPE cover
geometry via `gfmcd.reconstruction_atlas.promoter_composition`
(`enumerate_region_covers`, `select_matched_pair`, `select_unmatched_pair`).
No classifier margin or loss is read; `model_outcomes_computed` is `false`
throughout.

The calipers are FROZEN
(`configs/dnabert2_atlas/promoter_composition_v1.frozen.json`) and are never
widened here, including when the matched-pair count comes in short: per the
task brief, a shortfall is reported as an achieved caliper-vs-supply curve
and left for a decision, not papered over by loosening the caliper that this
study exists to hold fixed.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    PROMOTER_GUE_DATASET_ID,
    PROMOTER_GUE_DATASET_REVISION,
    load_dnabert2_promoter_classifier,
    tokenize_promoter_pairs,
)
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.promoter_assay import read_gue_sequences
from gfmcd.reconstruction_atlas.promoter_composition import (
    enumerate_region_covers,
    select_matched_pair,
    select_unmatched_pair,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("development", "confirmation")
NOMINAL_WIDTH_NT = 10

# Reported alongside the frozen caliper purely for documentation continuity
# with the controller's pre-measurement (see progress.md / PREREGISTRATION_D13
# "Background"). NEVER used to gate `status`; only `matching.tight_caliper_GC`
# / `tight_caliper_CpG` from the frozen config gates anything.
REFERENCE_CALIPER_CURVE = (
    (0.05, 0.03),
    (0.10, 0.05),
    (0.15, 0.08),
)

SOURCE_DEFINITION_CONFOUND = {
    "statement": (
        "The D13 composition cohort draws from GENCODE v50 protein-coding TSSs "
        "including non-MANE-Select transcripts, because MANE Select supply is "
        "exhausted under the block-disjointness admission rule (every block "
        "already used by a prior cohort is excluded). Non-MANE transcripts are "
        "alternative promoters -- a different promoter class from the MANE "
        "Select TSSs the D12 TSS-polarity effect was confirmed on."
    ),
    "asymmetry": (
        "A QUALIFIED (positive) P1/P2 result on this cohort generalises the "
        "confirmed effect beyond MANE Select promoters to alternative "
        "(non-MANE) promoters. A negative or null result on this cohort is "
        "AMBIGUOUS: it could mean the underlying effect is false, or that "
        "alternative promoters behave differently from the MANE Select "
        "promoters the effect was originally confirmed on. Neither this audit "
        "nor any downstream P1/P2 result may treat a negative outcome here as "
        "a clean refutation of the MANE-Select-confirmed effect."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-pairs", type=int, default=32)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    if config.get("status") != "FROZEN":
        raise SystemExit("composition supply audit requires a FROZEN configuration")

    rows = _read_jsonl(args.cohort)
    by_split = {split: [row for row in rows if row["split"] == split] for split in SPLITS}

    regions = config["regions"]
    matching_cfg = config["matching"]
    tight_caliper_gc = float(matching_cfg["tight_caliper_GC"])
    tight_caliper_cpg = float(matching_cfg["tight_caliper_CpG"])
    min_unmatched_sep = float(matching_cfg["unmatched_arm_minimum_GC_separation"])
    token_dose = int(matching_cfg["token_dose_per_cover"])
    maximum_cover_nt = int(matching_cfg["maximum_cover_nt"])
    maximum_cover_diff_nt = int(matching_cfg["maximum_cover_length_difference_nt"])
    downstream_region = tuple(int(v) for v in regions["downstream_relative"])
    upstream_region = tuple(int(v) for v in regions["upstream_relative"])

    # ---- supply invariants, recomputed independently of the builder -------
    cohort_path = Path(args.cohort).resolve()
    prior_cohort_paths = _prior_cohort_paths(cohort_path)
    consumed_blocks = _consumed_blocks(prior_cohort_paths)
    gue_root = _resolve_gue_root()
    gue_paths = [gue_root / f"{split}.csv" for split in ("train", "dev", "test")]
    gue_index = read_gue_sequences(gue_paths)

    supply: dict[str, Any] = {}
    selected_ids: dict[str, list[str]] = {}
    for split in SPLITS:
        split_rows = by_split[split]
        blocks = [str(row["homology_cluster"]) for row in split_rows]
        gue_hits = sum(
            bool(gue_index.get(row["target"]["clean_sequence"]))
            or bool(gue_index.get(row["control"]["clean_sequence"]))
            for row in split_rows
        )
        block_overlap = sum(1 for block in blocks if block in consumed_blocks)
        mane_rows = sum(1 for row in split_rows if row["target"].get("is_MANE_Select"))
        supply[split] = {
            "pairs": len(split_rows),
            "unique_locus_blocks": len(set(blocks)),
            "duplicate_blocks": len(blocks) - len(set(blocks)),
            "GUE_exact_overlaps": gue_hits,
            "blocks_overlapping_prior_cohorts": block_overlap,
            "MANE_Select_rows": mane_rows,
            "non_MANE_rows": len(split_rows) - mane_rows,
        }
        selected_ids[split] = [str(row["example_id"]) for row in split_rows]
    split_leakage = len(
        {str(row["homology_cluster"]) for row in by_split["development"]}
        & {str(row["homology_cluster"]) for row in by_split["confirmation"]}
    )
    selected_id_sha256 = {
        split: hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
        for split, ids in selected_ids.items()
    }

    # ---- matching feasibility, tokenizer-only, no model forward pass ------
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    feasibility: dict[str, dict[str, Any]] = {
        split: {
            "loci": 0,
            "covers_both_sides": 0,
            "matched_pair_frozen_caliper": 0,
            "unmatched_pair_planted_positive": 0,
            "both_arms": 0,
            "neither_arm": 0,
        }
        for split in SPLITS
    }
    caliper_curve: dict[str, dict[str, int]] = {
        split: {f"{gc}/{cpg}": 0 for gc, cpg in REFERENCE_CALIPER_CURVE} for split in SPLITS
    }

    for split in SPLITS:
        split_rows = by_split[split]
        for start in range(0, len(split_rows), args.batch_pairs):
            chunk = split_rows[start : start + args.batch_pairs]
            pairs = [
                (
                    str(row["example_id"]),
                    str(row["target"]["clean_sequence"]),
                    str(row["control"]["clean_sequence"]),
                )
                for row in chunk
            ]
            batch = tokenize_promoter_pairs(bundle, pairs)
            for pair_index, row in enumerate(chunk):
                tss_index = int(row["localization"]["TSS_index"])
                target_offsets = batch.offset_mapping[
                    2 * pair_index, : batch.token_counts[2 * pair_index]
                ].tolist()
                control_offsets = batch.offset_mapping[
                    2 * pair_index + 1, : batch.token_counts[2 * pair_index + 1]
                ].tolist()
                target_seq = str(row["target"]["clean_sequence"])
                control_seq = str(row["control"]["clean_sequence"])
                downstream_covers = enumerate_region_covers(
                    target_offsets,
                    control_offsets,
                    target_seq,
                    control_seq,
                    region_start=downstream_region[0],
                    region_end=downstream_region[1],
                    tss_index=tss_index,
                    nominal_width=NOMINAL_WIDTH_NT,
                    token_dose=token_dose,
                    maximum_cover_nt=maximum_cover_nt,
                    maximum_cover_length_difference_nt=maximum_cover_diff_nt,
                )
                upstream_covers = enumerate_region_covers(
                    target_offsets,
                    control_offsets,
                    target_seq,
                    control_seq,
                    region_start=upstream_region[0],
                    region_end=upstream_region[1],
                    tss_index=tss_index,
                    nominal_width=NOMINAL_WIDTH_NT,
                    token_dose=token_dose,
                    maximum_cover_nt=maximum_cover_nt,
                    maximum_cover_length_difference_nt=maximum_cover_diff_nt,
                )
                stats = feasibility[split]
                stats["loci"] += 1
                both_sides = bool(downstream_covers) and bool(upstream_covers)
                stats["covers_both_sides"] += int(both_sides)
                if not both_sides:
                    stats["neither_arm"] += int(
                        not downstream_covers and not upstream_covers
                    )
                    continue
                matched = select_matched_pair(
                    downstream_covers,
                    upstream_covers,
                    caliper_gc=tight_caliper_gc,
                    caliper_cpg=tight_caliper_cpg,
                )
                unmatched = select_unmatched_pair(
                    downstream_covers,
                    upstream_covers,
                    minimum_gc_separation=min_unmatched_sep,
                )
                stats["matched_pair_frozen_caliper"] += int(matched is not None)
                stats["unmatched_pair_planted_positive"] += int(unmatched is not None)
                stats["both_arms"] += int(matched is not None and unmatched is not None)
                for gc, cpg in REFERENCE_CALIPER_CURVE:
                    curve_pair = select_matched_pair(
                        downstream_covers,
                        upstream_covers,
                        caliper_gc=gc,
                        caliper_cpg=cpg,
                    )
                    caliper_curve[split][f"{gc}/{cpg}"] += int(curve_pair is not None)
            print(
                f"{split}: matching-feasibility scanned "
                f"{min(start + len(chunk), len(split_rows))}/{len(split_rows)}",
                flush=True,
            )

    minimum_matched = int(config["supply"]["minimum_matched_pairs_per_split"])
    target_raw = int(config["supply"]["target_raw_pairs_per_split"])
    checks = {
        "exact_pairs_per_split": {
            split: supply[split]["pairs"] == target_raw for split in SPLITS
        },
        "unique_locus_blocks_equal_pairs": {
            split: supply[split]["duplicate_blocks"] == 0 for split in SPLITS
        },
        "zero_GUE_exact_overlaps": {
            split: supply[split]["GUE_exact_overlaps"] == 0 for split in SPLITS
        },
        "zero_block_overlap_with_prior_cohorts": {
            split: supply[split]["blocks_overlapping_prior_cohorts"] == 0
            for split in SPLITS
        },
        "zero_split_leakage_clusters": split_leakage == 0,
        "matched_pairs_meet_minimum": {
            split: feasibility[split]["matched_pair_frozen_caliper"] >= minimum_matched
            for split in SPLITS
        },
    }
    supply_ok = all(
        all(value.values()) if isinstance(value, dict) else value
        for key, value in checks.items()
        if key != "matched_pairs_meet_minimum"
    )
    matching_ok = all(checks["matched_pairs_meet_minimum"].values())
    status = "QUALIFIED" if supply_ok and matching_ok else "BELOW_MATCHING_TARGET"
    if not supply_ok:
        status = "SUPPLY_INVARIANT_VIOLATED"

    common_inputs = (
        args.cohort,
        args.config,
        *gue_paths,
        *prior_cohort_paths,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "phase": "PROMOTER_COMPOSITION_OUTCOME_BLIND_SUPPLY_AND_MATCHING_FEASIBILITY",
        "status": status,
        "model_outcomes_computed": False,
        "boundary": (
            "This artifact uses the tokenizer only, to bind BPE cover geometry. "
            "It computes no classifier margin, loss, or target outcome."
        ),
        "supply": supply,
        "split_leakage_clusters": split_leakage,
        "matching_feasibility": {
            "frozen_caliper": {"GC": tight_caliper_gc, "CpG": tight_caliper_cpg},
            "unmatched_arm_minimum_GC_separation": min_unmatched_sep,
            "minimum_matched_pairs_per_split": minimum_matched,
            "by_split": feasibility,
        },
        "caliper_vs_supply_curve": {
            "note": (
                "Documentation only, reported for continuity with the "
                "controller's pre-measurement. The frozen caliper "
                f"{tight_caliper_gc}/{tight_caliper_cpg} in matching_feasibility "
                "is the only one that gates `status`; the wider calipers below "
                "are NEVER used to select or score a pair and must not be "
                "adopted to rescue a shortfall."
            ),
            "by_split": caliper_curve,
        },
        "checks": checks,
        "selection": {
            "selected_example_ids": selected_ids,
            "selected_id_sha256": selected_id_sha256,
        },
        "source_definition_confound": SOURCE_DEFINITION_CONFOUND,
        "cohort": {
            "path": str(cohort_path.relative_to(REPO_ROOT)),
            "rows": len(rows),
        },
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_composition_supply_audit_v1",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=common_inputs,
        parameters={
            "model_outcomes_computed": False,
            "regions": regions,
            "matching": matching_cfg,
            "nominal_width_nt": NOMINAL_WIDTH_NT,
            "minimum_matched_pairs_per_split": minimum_matched,
            "target_raw_pairs_per_split": target_raw,
        },
        seeds={"inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=REPO_ROOT,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prior_cohort_paths(cohort_path: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            Path(path)
            for path in glob.glob(
                str(REPO_ROOT / "data/processed/dnabert2_atlas/*/cohort.jsonl")
            )
            if Path(path).resolve() != cohort_path
        )
    )


def _consumed_blocks(paths: tuple[Path, ...]) -> set[str]:
    blocks: set[str] = set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cluster = row.get("homology_cluster")
                if cluster:
                    blocks.add(str(cluster))
    return blocks


def _resolve_gue_root() -> Path:
    owner, name = PROMOTER_GUE_DATASET_ID.split("/", 1)
    pattern = os.path.expanduser(
        f"~/.cache/huggingface/hub/datasets--{owner}--{name}/snapshots/"
        f"{PROMOTER_GUE_DATASET_REVISION}/GUE/prom_300_all"
    )
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(
            f"GUE prom_300_all not found at pinned revision "
            f"{PROMOTER_GUE_DATASET_REVISION} under {pattern}"
        )
    return Path(hits[0])


if __name__ == "__main__":
    main()
