#!/usr/bin/env python
"""Build the D13 fresh, non-MANE-preferred promoter composition cohort.

Reuses `build_promoter_classifier_cohort.py`'s admission rules verbatim: exact
forward/reverse-complement GUE exclusion, same-locus composition-matched
pseudo-TSS control, and outcome-blind hash-based split assignment. The window
geometry, control-offset ladder, and matching-balance tolerances are read from
that same admission-rule configuration
(`configs/dnabert2_atlas/promoter_classifier_v2.frozen.json`), unchanged.

The only intended change (v1, per
`.superpowers/sdd/2026-07-29-composition-vs-position/task-4-brief.md`) was the
transcript filter: GENCODE v50 protein-coding TSSs, one per gene, drawn from
`iter_gencode_protein_coding_tss` (which prefers a non-MANE-Select transcript
whenever a gene has one), restricted to locus blocks disjoint from every prior
cohort under `data/processed/dnabert2_atlas/*/cohort.jsonl`. MANE Select is
exhausted under the block-disjointness rule alone, so this extends the source
definition rather than reusing D4/D7-D12's MANE-only pool.

**Block size and exclusion rule (v2, Amendment 1,
`studies/dnabert2_reconstruction_atlas/preregistrations/PREREGISTRATION_D13_COMPOSITION_VS_POSITION.md`):**
the locus-block size is read from the run's config
(`boundaries.block_size_nt`; defaults to 1,000,000 -- v1's hardcoded value --
when a config carries no `boundaries` section, so this script still reproduces
v1 byte-for-byte). "Disjoint from every prior cohort" is **own-block
exclusion**: a candidate block is unavailable only if *that block itself* (at
the current run's block size) contains a TSS already consumed by a prior
cohort. Every prior cohort's consumed footprint is recomputed from its raw
`target.chrom` / `target.tss`, not from that cohort's own stored
`homology_cluster` string, so relaxing block size (e.g. v1's 1 Mb to v2's
250 kb) is applied uniformly to prior cohorts regardless of the block size
each was itself built under. See `boundaries.independence_cost` in v2's config
and preregistration Amendment 1 for why this loosens the cluster-bootstrap
independence assumption and is not a free relaxation.

Declared confound (see the supply audit and
`studies/dnabert2_reconstruction_atlas/preregistrations/PREREGISTRATION_D13_COMPOSITION_VS_POSITION.md`):
non-MANE transcripts are alternative promoters, a different promoter class
from the MANE Select TSSs the original TSS-polarity effect was confirmed on.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import PROMOTER_GUE_DATASET_ID, PROMOTER_GUE_DATASET_REVISION
from gfmcd.reconstruction_atlas.artifacts import write_jsonl_bundle
from gfmcd.reconstruction_atlas.assembly import split_for_cluster
from gfmcd.reconstruction_atlas.promoter_assay import (
    all_gencode_tss_by_chrom,
    has_nearby_tss,
    iter_gencode_protein_coding_tss,
    oriented_tss_window,
    read_gue_sequences,
    sequence_composition,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Window geometry, the control-offset ladder, and matching-balance tolerances
# are the reference builder's admission rules, reused verbatim (frozen
# alongside the MANE classifier cohort, unrelated to the transcript filter
# this script changes).
REFERENCE_COLLECTION_CONFIG = (
    REPO_ROOT / "configs/dnabert2_atlas/promoter_classifier_v2.frozen.json"
)

SPLITS = ("development", "confirmation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        from pyfaidx import Fasta
    except ImportError as exc:
        raise SystemExit("pyfaidx is required by the pinned atlas runtime") from exc

    config = json.loads(Path(args.config).read_text())
    if config.get("status") != "FROZEN":
        raise SystemExit("promoter composition cohort requires a FROZEN configuration")
    reference = json.loads(REFERENCE_COLLECTION_CONFIG.read_text())
    if reference.get("status") != "FROZEN":
        raise SystemExit("reference admission-rule configuration must be FROZEN")

    collection_config = reference["collection"]
    seed = int(reference["seed"])
    upstream = int(collection_config["window_upstream_nt"])
    downstream = int(collection_config["window_downstream_nt"])
    if upstream + downstream + 1 != int(collection_config["input_length_nt"]):
        raise SystemExit("composition cohort window geometry is inconsistent")

    block_size_nt = int(config.get("boundaries", {}).get("block_size_nt", 1_000_000))
    target_per_split = int(config["supply"]["target_raw_pairs_per_split"])
    out_path = Path(args.out).resolve()

    gue_root = _resolve_gue_root()
    gue_paths = [gue_root / f"{split}.csv" for split in ("train", "dev", "test")]
    gue_index = read_gue_sequences(gue_paths)
    tss_by_chrom = all_gencode_tss_by_chrom(args.gtf)
    candidates = iter_gencode_protein_coding_tss(args.gtf)
    prior_cohort_paths = _prior_cohort_paths(out_path)
    consumed_blocks = _consumed_blocks(prior_cohort_paths, block_size_nt)
    fasta = Fasta(args.genome, as_raw=True, sequence_always_upper=True)

    counters: Counter[str] = Counter()
    contamination: Counter[str] = Counter()
    candidate_rows: list[dict[str, Any]] = []
    for feature in candidates:
        counters["protein_coding_gene_candidates_offered"] += 1
        block = _block(feature.chrom, feature.tss, block_size_nt)
        if block in consumed_blocks:
            counters["block_already_consumed_by_prior_cohort"] += 1
            continue
        target = oriented_tss_window(
            fasta,
            chrom=feature.chrom,
            tss=feature.tss,
            strand=feature.strand,
            upstream_nt=upstream,
            downstream_nt=downstream,
        )
        if target is None:
            counters["target_sequence_ineligible"] += 1
            continue
        target_hits = gue_index.get(target, [])
        if target_hits:
            counters["target_GUE_exact_overlap"] += 1
            for split, label in target_hits:
                contamination[f"{split}|label{label}"] += 1
            continue
        target_composition = sequence_composition(target)
        controls: list[tuple[float, int, int, str, dict[str, float]]] = []
        for offset in collection_config["control_offsets_transcription_nt"]:
            offset = int(offset)
            control_tss = (
                feature.tss + offset if feature.strand == "+" else feature.tss - offset
            )
            if has_nearby_tss(
                tss_by_chrom,
                feature.chrom,
                control_tss,
                radius_nt=int(collection_config["control_TSS_exclusion_radius_nt"]),
            ):
                continue
            control = oriented_tss_window(
                fasta,
                chrom=feature.chrom,
                tss=control_tss,
                strand=feature.strand,
                upstream_nt=upstream,
                downstream_nt=downstream,
            )
            if control is None or gue_index.get(control):
                continue
            composition = sequence_composition(control)
            gc_difference = abs(
                target_composition["gc_fraction"] - composition["gc_fraction"]
            )
            cpg_difference = abs(
                target_composition["cpg_fraction"] - composition["cpg_fraction"]
            )
            distance = gc_difference + 2 * cpg_difference
            controls.append(
                (
                    distance,
                    abs(offset),
                    offset,
                    control,
                    {
                        "absolute_gc_fraction_difference": gc_difference,
                        "absolute_cpg_fraction_difference": cpg_difference,
                        **{
                            f"control_{key}": value
                            for key, value in composition.items()
                        },
                    },
                )
            )
        if not controls:
            counters["no_GUE_clean_TSS_excluded_control"] += 1
            continue
        _, _, offset, control, control_metrics = min(controls)
        if control_metrics["absolute_gc_fraction_difference"] > float(
            collection_config["maximum_absolute_GC_fraction_difference"]
        ) or control_metrics["absolute_cpg_fraction_difference"] > float(
            collection_config["maximum_absolute_CpG_fraction_difference"]
        ):
            counters["matching_balance_ineligible"] += 1
            continue
        control_tss = (
            feature.tss + offset if feature.strand == "+" else feature.tss - offset
        )
        split = split_for_cluster(
            block,
            seed=seed,
            development_percent=int(collection_config["development_percent"]),
        )
        row = {
            "example_id": _example_id(feature.transcript_id, feature.chrom, feature.tss),
            "split": split,
            "homology_cluster": block,
            "source": {
                "id": "GENCODE",
                "release": str(reference["source"]["gencode_release"]),
                "assembly": str(reference["source"]["assembly"]),
                "target_definition": (
                    "protein_coding_transcript_TSS_one_per_gene_non_MANE_preferred_"
                    "block_disjoint_from_prior_cohorts"
                ),
            },
            "target": {
                "chrom": feature.chrom,
                "tss": feature.tss,
                "strand": feature.strand,
                "gene_id": feature.gene_id,
                "transcript_id": feature.transcript_id,
                "gene_name": feature.gene_name,
                "clean_sequence": target,
                "is_MANE_Select": bool(feature.is_mane),
            },
            "control": {
                "family": str(collection_config["control_family"]),
                "chrom": feature.chrom,
                "pseudo_tss": control_tss,
                "strand": feature.strand,
                "transcription_offset_nt": offset,
                "clean_sequence": control,
            },
            "matching": {
                **{f"target_{key}": value for key, value in target_composition.items()},
                **control_metrics,
                "GUE_exact_target_hits": 0,
                "GUE_exact_control_hits": 0,
                "control_TSS_exclusion_radius_nt": int(
                    collection_config["control_TSS_exclusion_radius_nt"]
                ),
            },
            "localization": {"TSS_index": upstream},
            "selection_rank": _rank(seed, feature.transcript_id),
        }
        candidate_rows.append(row)

    selected = _select_locus_blocked(candidate_rows, maximum_per_split=target_per_split)
    split_counts = Counter(row["split"] for row in selected)
    if any(split_counts[split] < target_per_split for split in SPLITS):
        raise RuntimeError(
            f"insufficient fresh composition cohort supply: {dict(split_counts)}, "
            f"need {target_per_split} per split"
        )

    audit = _audit(selected, gue_index, consumed_blocks)
    if any(audit["invariant_violations"].values()):
        raise RuntimeError(f"composition cohort invariant violation: {audit}")

    selected_ids = {
        split: [str(row["example_id"]) for row in selected if row["split"] == split]
        for split in SPLITS
    }
    selected_id_sha256 = {
        split: hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
        for split, ids in selected_ids.items()
    }
    mane_counts = {
        split: sum(
            1
            for row in selected
            if row["split"] == split and row["target"]["is_MANE_Select"]
        )
        for split in SPLITS
    }

    source_inputs = (
        *gue_paths,
        args.gtf,
        args.genome,
        args.config,
        REFERENCE_COLLECTION_CONFIG,
        *prior_cohort_paths,
    )
    model_identity = {**config["checkpoint"], "used_for_collection_build": False}
    matching_gc = [row["matching"]["absolute_gc_fraction_difference"] for row in selected]
    matching_cpg = [
        row["matching"]["absolute_cpg_fraction_difference"] for row in selected
    ]
    assay_version = str(config["assay"]).rsplit("_", 1)[-1]
    cohort_path, cohort_manifest = write_jsonl_bundle(
        args.out,
        selected,
        experiment_id=f"dnabert2_promoter_composition_cohort_{assay_version}",
        status="PREFLIGHT",
        model_identity=model_identity,
        input_paths=source_inputs,
        parameters={
            "outcome_blind": True,
            "model_outcomes_computed": False,
            "collection_reference": str(
                REFERENCE_COLLECTION_CONFIG.relative_to(REPO_ROOT)
            ),
            "source_definition": (
                "GENCODE v50 protein-coding TSSs including non-MANE-Select "
                "transcripts (one per gene, non-MANE preferred), restricted to "
                f"own-block-excluded {_format_block_size(block_size_nt)} blocks unused "
                "by any prior cohort under data/processed/dnabert2_atlas/*/cohort.jsonl "
                "(own-block exclusion: a block is unavailable only if that block itself "
                "contains a TSS already consumed by a prior cohort)"
            ),
            "block_size_nt": block_size_nt,
            "target_raw_pairs_per_split": target_per_split,
            "n_examples": len(selected),
            "split_counts": dict(sorted(split_counts.items())),
            "unique_locus_blocks": len({row["homology_cluster"] for row in selected}),
            "consumed_blocks_excluded": len(consumed_blocks),
            "prior_cohort_files": [
                str(Path(path).relative_to(REPO_ROOT)) for path in prior_cohort_paths
            ],
            "MANE_Select_rows_in_selected_cohort": mane_counts,
            "selected_id_sha256": selected_id_sha256,
            "source_counts": dict(sorted(counters.items())),
            "GUE_exclusion_membership": dict(sorted(contamination.items())),
            "matching_balance": {
                "absolute_GC_fraction_difference": _summary(matching_gc),
                "absolute_CpG_fraction_difference": _summary(matching_cpg),
            },
            "invariant_violations": audit["invariant_violations"],
        },
        seeds={"collection_and_split": seed},
        command=sys.argv,
        repo=REPO_ROOT,
    )
    print(
        json.dumps(
            {
                "cohort_path": str(cohort_path),
                "cohort_manifest": str(cohort_manifest),
                "selected": dict(sorted(split_counts.items())),
                "MANE_Select_rows_in_selected_cohort": mane_counts,
                "unique_locus_blocks": len({row["homology_cluster"] for row in selected}),
                "invariant_violations": audit["invariant_violations"],
                "selected_id_sha256": selected_id_sha256,
                "matching_GC": _summary(matching_gc),
                "matching_CpG": _summary(matching_cpg),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _prior_cohort_paths(out_path: Path) -> tuple[Path, ...]:
    """Every prior cohort's block list, excluding this run's own output path."""

    paths = sorted(
        Path(path)
        for path in glob.glob(
            str(REPO_ROOT / "data/processed/dnabert2_atlas/*/cohort.jsonl")
        )
        if Path(path).resolve() != out_path
    )
    if not paths:
        raise RuntimeError(
            "no prior cohort.jsonl files found; block-disjointness cannot be verified"
        )
    return tuple(paths)


def _consumed_blocks(paths: tuple[Path, ...], block_size_nt: int) -> set[str]:
    """Own-block exclusion: recompute, at *this run's* block size, the block each prior
    cohort's target TSS falls in -- never trust a prior cohort's own stored
    `homology_cluster`, which may have been computed at a different (e.g. coarser 1-Mb)
    block size. A block is "consumed" only if it itself contains a prior target TSS;
    unconsumed sibling sub-blocks under the same former, coarser parent block remain
    available. See `boundaries.exclusion_rule_definition` in the composition v2 config
    and PREREGISTRATION_D13_COMPOSITION_VS_POSITION.md Amendment 1."""
    blocks: set[str] = set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                target = row.get("target") or {}
                chrom = target.get("chrom")
                tss = target.get("tss")
                if chrom is None or tss is None:
                    continue
                blocks.add(_block(str(chrom), int(tss), block_size_nt))
    return blocks


def _format_block_size(block_size_nt: int) -> str:
    if block_size_nt % 1_000_000 == 0:
        return f"{block_size_nt // 1_000_000}-Mb"
    if block_size_nt % 1_000 == 0:
        return f"{block_size_nt // 1_000}-kb"
    return f"{block_size_nt}-nt"


def _resolve_gue_root() -> Path:
    """Locate the pinned GUE prom_300_all directory in the local HF cache; never download."""

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


def _block(chrom: str, tss: int, block_size_nt: int) -> str:
    """The block id embeds the block size, so ids computed at different sizes never
    collide as strings (e.g. `locus1000000:...` vs `locus250000:...`) -- consumed-block
    membership must always be recomputed at the current run's block size, never compared
    across differently-sized runs by string equality alone."""
    return f"locus{block_size_nt}:{chrom}:{tss // block_size_nt}"


def _select_locus_blocked(
    rows: list[dict[str, Any]], *, maximum_per_split: int
) -> list[dict[str, Any]]:
    """One row per (split, locus block -- size set by the run's config), ranked by a
    seeded hash -- copied from `build_promoter_classifier_cohort.py`'s
    `_select_locus_blocked`, unchanged (the algorithm is size-agnostic; it only ever
    treats `homology_cluster` as an opaque cluster key)."""

    by_split_cluster: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        split = str(row["split"])
        cluster = str(row["homology_cluster"])
        previous = by_split_cluster[split].get(cluster)
        if previous is None or row["selection_rank"] < previous["selection_rank"]:
            by_split_cluster[split][cluster] = row
    selected = []
    for split in SPLITS:
        ranked = sorted(
            by_split_cluster[split].values(),
            key=lambda row: (row["selection_rank"], row["example_id"]),
        )
        selected.extend(ranked[:maximum_per_split])
    for row in selected:
        row.pop("selection_rank", None)
    return selected


def _audit(
    rows: list[dict[str, Any]],
    gue_index: dict[str, list[tuple[str, int]]],
    consumed_blocks: set[str],
) -> dict[str, Any]:
    split_clusters: dict[str, set[str]] = defaultdict(set)
    violations: Counter[str] = Counter()
    for row in rows:
        split_clusters[str(row["split"])].add(str(row["homology_cluster"]))
        violations["GUE_exact_overlaps"] += int(
            bool(gue_index.get(row["target"]["clean_sequence"]))
            or bool(gue_index.get(row["control"]["clean_sequence"]))
        )
        violations["block_already_consumed_by_prior_cohort"] += int(
            str(row["homology_cluster"]) in consumed_blocks
        )
    violations["split_leakage_clusters"] = len(
        split_clusters["development"] & split_clusters["confirmation"]
    )
    return {
        "invariant_violations": {
            key: violations[key]
            for key in (
                "GUE_exact_overlaps",
                "split_leakage_clusters",
                "block_already_consumed_by_prior_cohort",
            )
        },
    }


def _rank(seed: int, transcript_id: str) -> str:
    return hashlib.sha256(f"{seed}|select|{transcript_id}".encode()).hexdigest()


def _example_id(transcript_id: str, chrom: str, tss: int) -> str:
    digest = hashlib.sha256(f"{transcript_id}|{chrom}|{tss}".encode()).hexdigest()
    return f"promoter-composition:{digest[:16]}"


def _summary(values: list[float | int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
