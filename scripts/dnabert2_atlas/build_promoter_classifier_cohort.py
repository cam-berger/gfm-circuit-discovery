#!/usr/bin/env python
"""Build a GUE-excluded, locus-blocked GENCODE promoter classifier cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.assembly import split_for_cluster
from gfmcd.reconstruction_atlas.promoter_assay import (
    all_gencode_tss_by_chrom,
    context_indices,
    has_nearby_tss,
    iter_gencode_mane_tss,
    matched_context_shuffles,
    oriented_tss_window,
    read_gue_sequences,
    sequence_composition,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--gue-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohort-out", required=True)
    parser.add_argument("--audit-out", required=True)
    args = parser.parse_args()

    try:
        from pyfaidx import Fasta
    except ImportError as exc:
        raise SystemExit("pyfaidx is required by the pinned atlas runtime") from exc

    config = json.loads(Path(args.config).read_text())
    if config.get("status") != "FROZEN":
        raise SystemExit("promoter classifier cohort requires a FROZEN configuration")
    collection_config = config["collection"]
    intervention = config["context_intervention"]
    seed = int(config["seed"])
    upstream = int(collection_config["window_upstream_nt"])
    downstream = int(collection_config["window_downstream_nt"])
    if upstream + downstream + 1 != int(collection_config["input_length_nt"]):
        raise SystemExit("classifier window geometry is inconsistent")
    gue_paths = [
        Path(args.gue_root) / f"{split}.csv" for split in ("train", "dev", "test")
    ]
    gue_index = read_gue_sequences(gue_paths)
    tss_by_chrom = all_gencode_tss_by_chrom(args.gtf)
    mane = list(iter_gencode_mane_tss(args.gtf))
    fasta = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)
    indices, anchor = context_indices(
        tss_index=upstream,
        target_width_nt=int(intervention["target_anchor_width_nt"]),
        upstream_context_nt=int(intervention["upstream_context_nt"]),
        downstream_context_nt=int(intervention["downstream_context_nt"]),
    )
    counters: Counter[str] = Counter()
    contamination: Counter[str] = Counter()
    candidate_rows: list[dict[str, Any]] = []
    for feature in mane:
        counters["mane_offered"] += 1
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
        row_seed = _row_seed(seed, feature.transcript_id)
        shuffles = matched_context_shuffles(
            target,
            control,
            indices=indices,
            seed=row_seed,
            candidates=int(intervention["candidate_shuffles"]),
        )
        if shuffles is None:
            counters["dose_matched_shuffle_ineligible"] += 1
            continue
        target_shuffle, control_shuffle, changed = shuffles
        control_tss = (
            feature.tss + offset if feature.strand == "+" else feature.tss - offset
        )
        cluster = f"locus1000000:{feature.chrom}:{feature.tss // 1_000_000}"
        split = split_for_cluster(
            cluster,
            seed=seed,
            development_percent=int(collection_config["development_percent"]),
        )
        row = {
            "example_id": _example_id(
                feature.transcript_id, feature.chrom, feature.tss
            ),
            "split": split,
            "homology_cluster": cluster,
            "source": {
                "id": "GENCODE",
                "release": str(config["source"]["gencode_release"]),
                "assembly": str(config["source"]["assembly"]),
                "target_definition": str(config["source"]["target_definition"]),
            },
            "target": {
                "chrom": feature.chrom,
                "tss": feature.tss,
                "strand": feature.strand,
                "gene_id": feature.gene_id,
                "transcript_id": feature.transcript_id,
                "gene_name": feature.gene_name,
                "clean_sequence": target,
                "context_shuffle_sequence": target_shuffle,
            },
            "control": {
                "family": str(collection_config["control_family"]),
                "chrom": feature.chrom,
                "pseudo_tss": control_tss,
                "strand": feature.strand,
                "transcription_offset_nt": offset,
                "clean_sequence": control,
                "context_shuffle_sequence": control_shuffle,
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
            "intervention": {
                "context_indices": list(indices),
                "preserved_anchor": {"start": anchor[0], "end": anchor[1]},
                "target_control_changed_nucleotides": changed,
                "target_anchor_unchanged": (
                    target[anchor[0] : anchor[1]]
                    == target_shuffle[anchor[0] : anchor[1]]
                ),
                "control_anchor_unchanged": (
                    control[anchor[0] : anchor[1]]
                    == control_shuffle[anchor[0] : anchor[1]]
                ),
                "target_context_composition_unchanged": _context_composition_equal(
                    target, target_shuffle, indices
                ),
                "control_context_composition_unchanged": _context_composition_equal(
                    control, control_shuffle, indices
                ),
                "target_changed_nucleotides": _hamming(target, target_shuffle),
                "control_changed_nucleotides": _hamming(control, control_shuffle),
            },
            "selection_rank": _rank(seed, feature.transcript_id),
        }
        candidate_rows.append(row)

    selected = _select_locus_blocked(
        candidate_rows,
        maximum_per_split=int(collection_config["maximum_clusters_per_split"]),
    )
    split_counts = Counter(row["split"] for row in selected)
    minimum = int(collection_config["minimum_clusters_per_split"])
    if any(split_counts[split] < minimum for split in ("development", "confirmation")):
        raise RuntimeError(
            f"insufficient promoter cohort after exclusions: {split_counts}"
        )
    audit = _audit(selected, gue_index)
    if any(audit["invariant_violations"].values()):
        raise RuntimeError(
            f"promoter cohort invariant violation: {audit['invariant_violations']}"
        )
    source_inputs = (*gue_paths, args.gtf, args.fasta, args.config)
    model_identity = {
        **config["checkpoint"],
        "used_for_collection_build": False,
    }
    cohort_path, cohort_manifest = write_jsonl_bundle(
        args.cohort_out,
        selected,
        experiment_id="dnabert2_promoter_classifier_gencode_cohort_v1",
        status="PREFLIGHT",
        model_identity=model_identity,
        input_paths=source_inputs,
        parameters={
            "outcome_blind": True,
            "collection": collection_config,
            "context_intervention": intervention,
            "n_examples": len(selected),
            "split_counts": dict(sorted(split_counts.items())),
        },
        seeds={"collection_and_intervention": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    matching_gc = [
        row["matching"]["absolute_gc_fraction_difference"] for row in selected
    ]
    matching_cpg = [
        row["matching"]["absolute_cpg_fraction_difference"] for row in selected
    ]
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "phase": "PROMOTER_CLASSIFIER_OUTCOME_BLIND_COHORT",
        "status": "PREFLIGHT",
        "model_outcomes_computed": False,
        "source_counts": {
            **dict(sorted(counters.items())),
            "all_GENCODE_transcript_TSSs_for_control_exclusion": sum(
                len(values) for values in tss_by_chrom.values()
            ),
        },
        "GUE_exclusion_membership": dict(sorted(contamination.items())),
        "selected": {
            "n": len(selected),
            "by_split": dict(sorted(split_counts.items())),
            "unique_locus_blocks": len({row["homology_cluster"] for row in selected}),
        },
        "matching_balance": {
            "absolute_GC_fraction_difference": _summary(matching_gc),
            "absolute_CpG_fraction_difference": _summary(matching_cpg),
        },
        **audit,
        "cohort": {
            "path": str(cohort_path),
            "sha256": sha256_file(cohort_path),
            "manifest": str(cohort_manifest),
        },
    }
    write_result_bundle(
        args.audit_out,
        payload,
        experiment_id="dnabert2_promoter_classifier_cohort_audit_v1",
        status="PREFLIGHT",
        model_identity=model_identity,
        input_paths=(*source_inputs, cohort_path, cohort_manifest),
        parameters={
            "outcome_blind": True,
            "collection": collection_config,
            "context_intervention": intervention,
        },
        seeds={"collection_and_intervention": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "selected": dict(sorted(split_counts.items())),
                "GUE_exact_overlaps": audit["invariant_violations"][
                    "GUE_exact_overlaps"
                ],
                "matching_GC": _summary(matching_gc),
                "matching_CpG": _summary(matching_cpg),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _select_locus_blocked(
    rows: list[dict[str, Any]], *, maximum_per_split: int
) -> list[dict[str, Any]]:
    by_split_cluster: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        split = str(row["split"])
        cluster = str(row["homology_cluster"])
        previous = by_split_cluster[split].get(cluster)
        if previous is None or row["selection_rank"] < previous["selection_rank"]:
            by_split_cluster[split][cluster] = row
    selected = []
    for split in ("development", "confirmation"):
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
) -> dict[str, Any]:
    split_clusters: dict[str, set[str]] = defaultdict(set)
    violations: Counter[str] = Counter()
    changed_counts = []
    for row in rows:
        split_clusters[str(row["split"])].add(str(row["homology_cluster"]))
        target = row["target"]
        control = row["control"]
        intervention = row["intervention"]
        violations["GUE_exact_overlaps"] += int(
            bool(gue_index.get(target["clean_sequence"]))
            or bool(gue_index.get(control["clean_sequence"]))
        )
        violations["anchor_changes"] += int(
            not intervention["target_anchor_unchanged"]
            or not intervention["control_anchor_unchanged"]
        )
        violations["context_composition_changes"] += int(
            not intervention["target_context_composition_unchanged"]
            or not intervention["control_context_composition_unchanged"]
        )
        violations["target_control_dose_mismatches"] += int(
            intervention["target_changed_nucleotides"]
            != intervention["control_changed_nucleotides"]
        )
        changed_counts.append(intervention["target_changed_nucleotides"])
    violations["split_leakage_clusters"] = len(
        split_clusters["development"] & split_clusters["confirmation"]
    )
    return {
        "invariant_violations": {
            key: violations[key]
            for key in (
                "GUE_exact_overlaps",
                "split_leakage_clusters",
                "anchor_changes",
                "context_composition_changes",
                "target_control_dose_mismatches",
            )
        },
        "intervention_changed_nucleotides": _summary(changed_counts),
    }


def _context_composition_equal(
    reference: str, changed: str, indices: tuple[int, ...]
) -> bool:
    return sorted(reference[index] for index in indices) == sorted(
        changed[index] for index in indices
    )


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def _row_seed(seed: int, transcript_id: str) -> int:
    digest = hashlib.sha256(f"{seed}|shuffle|{transcript_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _rank(seed: int, transcript_id: str) -> str:
    return hashlib.sha256(f"{seed}|select|{transcript_id}".encode()).hexdigest()


def _example_id(transcript_id: str, chrom: str, tss: int) -> str:
    digest = hashlib.sha256(f"{transcript_id}|{chrom}|{tss}".encode()).hexdigest()
    return f"promoter-classifier:{digest[:16]}"


def _summary(values: list[float | int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
