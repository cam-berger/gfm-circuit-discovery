#!/usr/bin/env python
"""Build the annotation-only multi-context Head-Context Atlas census and packet table.

No genome sequence, model checkpoint, attention, margin, or attribution is read.  This
stage establishes category prevalence, overlaps, split integrity, and coarse
annotation-only common support before sequence enrichment or compute is authorized.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.atlas_context import (  # noqa: E402
    AtlasConfig,
    build_atlas_index,
    enumerate_atlas_packets,
    match_stratum,
)
from gfmcd.splicing.gtf import load_transcripts  # noqa: E402

DEFAULT_GTF: str | None = None  # supply via --gtf
STD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
CORE_CATEGORIES = (
    "single_pair",
    "donor_cluster",
    "acceptor_cluster",
    "multiple_independent_pairs",
    "cassette_exon",
    "complex_event",
)
CONTROL_TAG = "isolated_pair_control_candidate"


def quantiles(values, qs=(0.5, 0.9, 0.99)):
    values = sorted(values)
    if not values:
        return {str(q): None for q in qs}
    return {
        str(q): values[min(round(q * (len(values) - 1)), len(values) - 1)]
        for q in qs
    }


def count_block(packets) -> dict:
    return {
        "n_packets": len(packets),
        "n_genes": len({packet.gene_id for packet in packets}),
        "n_junctions": len({packet.junction_id for packet in packets}),
        "target_partner_distance_q": quantiles([packet.target_partner_distance for packet in packets]),
        "visible_pair_count_q": quantiles([packet.visible_pair_count for packet in packets]),
        "target_site_support_q": quantiles([
            packet.n_annotated_transcripts_for_target_site for packet in packets
        ]),
    }


def common_support_report(packets) -> dict:
    """Can each focal packet find a different-gene isolated-pair control in its stratum?"""
    controls: dict[tuple, set[str]] = defaultdict(set)
    for packet in packets:
        if CONTROL_TAG in packet.tags:
            controls[match_stratum(packet)].add(packet.gene_id)
    out = {}
    for category in CORE_CATEGORIES:
        focal = [packet for packet in packets if category in packet.tags]
        eligible = [
            packet for packet in focal
            if controls[match_stratum(packet)] - {packet.gene_id}
        ]
        out[category] = {
            "n_focal": len(focal),
            "n_focal_genes": len({packet.gene_id for packet in focal}),
            "n_with_different_gene_control": len(eligible),
            "n_genes_with_different_gene_control": len({packet.gene_id for packet in eligible}),
            "fraction_with_common_support": len(eligible) / len(focal) if focal else None,
            "note": (
                "coarse annotation-only support on split, target role, distance, transcript support, "
                "visible-pair count, and target-position bin; sequence strength/composition/health still required"
            ),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--out", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_census_v1.json"
    ))
    ap.add_argument("--packets", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_packets_v1.jsonl"
    ))
    ap.add_argument("--limit-transcripts", type=int, default=None, help="smoke only")
    ap.add_argument("--window-length", type=int, default=1024)
    ap.add_argument("--min-offset", type=int, default=100)
    ap.add_argument("--max-offset", type=int, default=300)
    ap.add_argument("--min-context", type=int, default=100)
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    out_path = Path(args.out)
    packet_path = Path(args.packets)
    for path in (out_path, packet_path, out_path.with_suffix(".manifest.json")):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite immutable artifact {path}")

    cfg = AtlasConfig(
        window_length=args.window_length,
        min_offset=args.min_offset,
        max_offset=args.max_offset,
        min_context=args.min_context,
    )
    cfg.validate()
    transcripts = load_transcripts(
        args.gtf,
        transcript_types=("protein_coding",),
        allowed_chromosomes=STD_CHROMS,
    )
    if args.limit_transcripts is not None:
        transcripts = transcripts[:args.limit_transcripts]
    print(f"[atlas-census] {len(transcripts)} transcripts; building splice graph", file=sys.stderr)
    index = build_atlas_index(transcripts)
    packets = list(enumerate_atlas_packets(index, cfg))
    print(
        f"[atlas-census] {len(index.junction_support)} unique junctions; {len(packets)} constructible readout packets",
        file=sys.stderr,
    )

    # Chromosome split must imply gene disjointness for every emitted packet.
    split_genes = {
        split: {packet.gene_id for packet in packets if packet.split == split}
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train_val": sorted(split_genes["train"] & split_genes["val"]),
        "train_test": sorted(split_genes["train"] & split_genes["test"]),
        "val_test": sorted(split_genes["val"] & split_genes["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"chromosome split is not gene-disjoint: {overlaps}")

    blocks = {}
    for split in ("train", "val", "test", "ALL"):
        for role in ("donor", "acceptor", "BOTH"):
            subset = [
                packet for packet in packets
                if (split == "ALL" or packet.split == split)
                and (role == "BOTH" or packet.target_role == role)
            ]
            blocks[f"{split}|{role}"] = count_block(subset)
            for category in CORE_CATEGORIES + (CONTROL_TAG,):
                blocks[f"{split}|{role}"][category] = count_block([
                    packet for packet in subset if category in packet.tags
                ])

    overlap_matrix = {}
    for left in CORE_CATEGORIES:
        overlap_matrix[left] = {}
        for right in CORE_CATEGORIES:
            joint = [packet for packet in packets if left in packet.tags and right in packet.tags]
            overlap_matrix[left][right] = {
                "n_packets": len(joint),
                "n_genes": len({packet.gene_id for packet in joint}),
            }

    label_sets = Counter(tuple(tag for tag in packet.tags if tag in CORE_CATEGORIES) for packet in packets)
    top_combinations = [
        {"categories": list(tags), "n_packets": count}
        for tags, count in sorted(label_sets.items(), key=lambda item: (-item[1], item[0]))[:25]
    ]

    train_packets = [packet for packet in packets if packet.split == "train"]
    common_support = common_support_report(train_packets)
    report = {
        "schema": "head_context_atlas.context_census/1",
        "status": "ANNOTATION_ONLY_NO_MODEL_NO_SEQUENCE",
        "objective": (
            "multi-label substrate for general-versus-context-associated component use across all 96 heads and 6 MLPs"
        ),
        "config": {
            "window_length": cfg.window_length,
            "min_offset": cfg.min_offset,
            "max_offset": cfg.max_offset,
            "min_context": cfg.min_context,
            "max_span": cfg.max_span,
            "transcript_types": ["protein_coding"],
        },
        "definitions_source": "src/gfmcd/splicing/atlas_context.py",
        "n_transcripts": len(transcripts),
        "n_unique_junctions": len(index.junction_support),
        "n_constructible_packets": len(packets),
        "n_constructible_junctions": len({packet.junction_id for packet in packets}),
        "junction_window_fit_fraction": (
            len({packet.junction_id for packet in packets}) / len(index.junction_support)
            if index.junction_support else None
        ),
        "gene_disjoint_split_gate": {
            "pass": not any(overlaps.values()),
            "overlaps": {name: len(genes) for name, genes in overlaps.items()},
        },
        "categories": list(CORE_CATEGORIES),
        "control_candidate_tag": CONTROL_TAG,
        "counts": blocks,
        "category_overlap_matrix": overlap_matrix,
        "top_category_combinations": top_combinations,
        "train_common_support_vs_isolated_pair_controls": common_support,
        "deferred_before_matching": [
            "genome-sequence construction and canonical motif validation",
            "MaxEnt strength and within-gene strength gap",
            "GC/PPT/k-mer/low-complexity covariates",
            "proper masked-MLM health probes",
            "final without-replacement matched-set selection",
        ],
        "claim_boundary": (
            "category prevalence and annotation-only common support only; no head function, expression, "
            "competition specificity, or biological mechanism inferred"
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    with packet_path.open("w") as handle:
        for packet in packets:
            handle.write(json.dumps(packet.to_dict(), sort_keys=True) + "\n")
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_context_census_v1",
        command=["python"] + sys.argv,
        status="ANNOTATION_ONLY_NO_MODEL_NO_SEQUENCE",
        config=report["config"],
    )
    manifest.add_input(args.gtf)
    manifest.add_output(str(packet_path))
    manifest.add_output(str(out_path))
    manifest.write(str(out_path.with_suffix(".manifest.json")))

    print("\n================ MULTI-CONTEXT ATLAS CENSUS ================")
    for split in ("train", "val", "test"):
        block = blocks[f"{split}|BOTH"]
        print(f"[{split}] packets={block['n_packets']} genes={block['n_genes']}")
        for category in CORE_CATEGORIES:
            item = block[category]
            print(f"  {category:30s} {item['n_packets']:7d} packets / {item['n_genes']:5d} genes")
    print("\n[train common support vs different-gene isolated-pair controls]")
    for category, item in common_support.items():
        fraction = item["fraction_with_common_support"]
        print(f"  {category:30s} {fraction:.3f}" if fraction is not None else f"  {category:30s} --")
    print(f"\nWrote {packet_path}\nWrote {out_path}")


if __name__ == "__main__":
    main()
