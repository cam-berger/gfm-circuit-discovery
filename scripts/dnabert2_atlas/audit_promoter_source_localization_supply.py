#!/usr/bin/env python
"""Bind fresh loci, BPE tile geometry, and independent promoter annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    load_dnabert2_promoter_classifier,
    tokenize_promoter_pairs,
)
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.promoter_head_analysis import select_locus_rows
from gfmcd.reconstruction_atlas.promoter_head_scan import read_jsonl
from gfmcd.reconstruction_atlas.promoter_source_localization import (
    BinnedIntervalIndex,
    EpdMotif,
    composition_preserving_paired_disruption,
    iter_encode_pls,
    iter_fantom5_dominant_ctss,
    iter_ucsc_cpg_islands,
    oriented_interval_to_genome,
    paired_fixed_dose_tile,
    parse_epd_motif,
    scan_epd_motif,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-cohort", required=True)
    parser.add_argument("--original-cohort", required=True)
    parser.add_argument("--previous-supply-audit", required=True)
    parser.add_argument("--cage", required=True)
    parser.add_argument("--ccre", required=True)
    parser.add_argument("--cpg", required=True)
    parser.add_argument("--epd-dir", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--annotated-cohort-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pairs-per-split", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--tile-width-nt", type=int, default=10)
    parser.add_argument("--token-dose", type=int, default=2)
    parser.add_argument("--maximum-cover-nt", type=int, default=18)
    parser.add_argument("--maximum-cover-difference-nt", type=int, default=2)
    parser.add_argument("--cage-tolerance-nt", type=int, default=5)
    parser.add_argument("--motif-shuffle-candidates", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    expanded = read_jsonl(args.expanded_cohort)
    original = read_jsonl(args.original_cohort)
    previous_supply = json.loads(Path(args.previous_supply_audit).read_text())
    selected = _select_fresh_rows(
        expanded,
        original,
        previous_supply,
        pairs_per_split=args.pairs_per_split,
        seed=args.seed,
    )
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    motifs = _load_motifs(Path(args.epd_dir))
    cage = BinnedIntervalIndex(iter_fantom5_dominant_ctss(args.cage))
    pls = BinnedIntervalIndex(iter_encode_pls(args.ccre))
    cpg = BinnedIntervalIndex(iter_ucsc_cpg_islands(args.cpg))

    enriched: list[dict[str, Any]] = []
    for split in ("development", "confirmation"):
        rows = selected[split]
        for start in range(0, len(rows), 32):
            chunk = rows[start : start + 32]
            pairs = [
                (
                    str(row["example_id"]),
                    str(row["target"]["clean_sequence"]),
                    str(row["control"]["clean_sequence"]),
                )
                for row in chunk
            ]
            batch = tokenize_promoter_pairs(bundle, pairs)
            for pair_index, raw_row in enumerate(chunk):
                row = dict(raw_row)
                row["localization"] = _annotate_row(
                    row,
                    bundle=bundle,
                    target_offsets=batch.offset_mapping[
                        2 * pair_index, : batch.token_counts[2 * pair_index]
                    ].tolist(),
                    control_offsets=batch.offset_mapping[
                        2 * pair_index + 1, : batch.token_counts[2 * pair_index + 1]
                    ].tolist(),
                    target_ids=batch.encoded["input_ids"][
                        2 * pair_index, : batch.token_counts[2 * pair_index]
                    ].tolist(),
                    control_ids=batch.encoded["input_ids"][
                        2 * pair_index + 1, : batch.token_counts[2 * pair_index + 1]
                    ].tolist(),
                    motifs=motifs,
                    cage=cage,
                    pls=pls,
                    cpg=cpg,
                    tile_width_nt=args.tile_width_nt,
                    token_dose=args.token_dose,
                    maximum_cover_nt=args.maximum_cover_nt,
                    maximum_cover_difference_nt=args.maximum_cover_difference_nt,
                    cage_tolerance_nt=args.cage_tolerance_nt,
                    seed=args.seed,
                    motif_shuffle_candidates=args.motif_shuffle_candidates,
                )
                enriched.append(row)
            print(
                f"{split}: annotated {min(start + len(chunk), len(rows))}/{len(rows)}",
                flush=True,
            )

    selected_ids = {
        split: [str(row["example_id"]) for row in selected[split]]
        for split in ("development", "confirmation")
    }
    checks = _checks(
        enriched,
        selected,
        expanded,
        original,
        previous_supply,
        pairs_per_split=args.pairs_per_split,
        tile_count=300 // args.tile_width_nt,
    )
    if not all(checks.values()):
        raise RuntimeError(f"promoter source-localization supply failed: {checks}")
    common_inputs = (
        args.expanded_cohort,
        args.original_cohort,
        args.previous_supply_audit,
        args.cage,
        args.ccre,
        args.cpg,
        *(str(path) for path in sorted(Path(args.epd_dir).glob("*.html"))),
        args.source_manifest,
    )
    cohort_path, cohort_manifest = write_jsonl_bundle(
        args.annotated_cohort_out,
        enriched,
        experiment_id="dnabert2_promoter_source_localization_annotated_cohort_v1",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=common_inputs,
        parameters={
            "model_outcomes_computed": False,
            "selection_tag": "promoter-source-localization-v1",
            "pairs_per_split": args.pairs_per_split,
            "tile_width_nt": args.tile_width_nt,
            "token_dose": args.token_dose,
            "maximum_cover_nt": args.maximum_cover_nt,
            "maximum_cover_difference_nt": args.maximum_cover_difference_nt,
            "cage_tolerance_nt": args.cage_tolerance_nt,
            "motif_shuffle_candidates": args.motif_shuffle_candidates,
        },
        seeds={"selection_and_disruption": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    summary = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "phase": "PROMOTER_SOURCE_LOCALIZATION_OUTCOME_BLIND_SUPPLY",
        "status": "QUALIFIED",
        "model_outcomes_computed": False,
        "selection": {
            "tag": "promoter-source-localization-v1",
            "pairs_per_split": args.pairs_per_split,
            "selected_example_ids": selected_ids,
            "selected_id_sha256": {
                split: hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()
                for split, values in selected_ids.items()
            },
            "unique_locus_blocks": {
                split: len({str(row["homology_cluster"]) for row in selected[split]})
                for split in ("development", "confirmation")
            },
        },
        "geometry": {
            "nominal_tile_width_nt": args.tile_width_nt,
            "tile_count": 300 // args.tile_width_nt,
            "fixed_BPE_token_dose": args.token_dose,
            "maximum_actual_cover_nt": args.maximum_cover_nt,
            "maximum_target_control_cover_difference_nt": (
                args.maximum_cover_difference_nt
            ),
            "by_split_and_tile": _tile_summary(enriched),
        },
        "independent_annotations": _annotation_summary(enriched, motifs),
        "checks": checks,
        "annotated_cohort": {
            "path": str(cohort_path),
            "manifest": str(cohort_manifest),
            "rows": len(enriched),
        },
        "boundary": (
            "This artifact uses the tokenizer only to bind BPE geometry. It computes "
            "no classifier margin, attention value, head effect, or target outcome. "
            "CAGE, PLS, CpG, and motif evidence remain separate."
        ),
    }
    write_result_bundle(
        args.out,
        summary,
        experiment_id="dnabert2_promoter_source_localization_supply_audit_v1",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=(*common_inputs, cohort_path, cohort_manifest),
        parameters={
            "model_outcomes_computed": False,
            "selection": summary["selection"],
            "geometry": {
                key: value
                for key, value in summary["geometry"].items()
                if key != "by_split_and_tile"
            },
        },
        seeds={"selection_and_disruption": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _select_fresh_rows(
    expanded: list[dict[str, Any]],
    original: list[dict[str, Any]],
    previous_supply: dict[str, Any],
    *,
    pairs_per_split: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    excluded_ids = {str(row["example_id"]) for row in original}
    excluded_clusters = {str(row["homology_cluster"]) for row in original}
    previous_ids = {
        str(value)
        for values in previous_supply["selection"]["selected_example_ids"].values()
        for value in values
    }
    expanded_by_id = {str(row["example_id"]): row for row in expanded}
    excluded_ids.update(previous_ids)
    excluded_clusters.update(
        str(expanded_by_id[value]["homology_cluster"]) for value in previous_ids
    )
    selected = {}
    for split in ("development", "confirmation"):
        available = [
            row
            for row in expanded
            if row["split"] == split
            and str(row["example_id"]) not in excluded_ids
            and str(row["homology_cluster"]) not in excluded_clusters
        ]
        selected[split] = select_locus_rows(
            available,
            split=split,
            maximum=pairs_per_split,
            seed=seed,
            tag="promoter-source-localization-v1",
        )
    return selected


def _load_motifs(epd_dir: Path) -> dict[str, EpdMotif]:
    definitions = (
        ("TATA", "core_promoter", "tata_old.html"),
        ("Inr", "core_promoter", "init.html"),
        ("CCAAT", "promoter_associated", "ccaat.html"),
        ("GC_box", "promoter_associated", "gc.html"),
    )
    return {
        name: parse_epd_motif(epd_dir / filename, name=name, family=family)
        for name, family, filename in definitions
    }


def _annotate_row(
    row: dict[str, Any],
    *,
    bundle: Any,
    target_offsets: list[list[int]],
    control_offsets: list[list[int]],
    target_ids: list[int],
    control_ids: list[int],
    motifs: dict[str, EpdMotif],
    cage: BinnedIntervalIndex,
    pls: BinnedIntervalIndex,
    cpg: BinnedIntervalIndex,
    tile_width_nt: int,
    token_dose: int,
    maximum_cover_nt: int,
    maximum_cover_difference_nt: int,
    cage_tolerance_nt: int,
    seed: int,
    motif_shuffle_candidates: int,
) -> dict[str, Any]:
    target = str(row["target"]["clean_sequence"])
    control = str(row["control"]["clean_sequence"])
    chrom = str(row["target"]["chrom"])
    tss = int(row["target"]["tss"])
    strand = str(row["target"]["strand"])
    tss_index = 249
    tss_cage = cage.overlapping(
        chrom,
        max(0, tss - cage_tolerance_nt),
        tss + cage_tolerance_nt + 1,
        strand=strand,
    )
    tss_pls = pls.overlapping(chrom, tss, tss + 1)
    tss_cpg = cpg.overlapping(chrom, tss, tss + 1)
    target_calls = {
        name: scan_epd_motif(target, motif, tss_index=tss_index)
        for name, motif in motifs.items()
    }
    control_calls = {
        name: scan_epd_motif(control, motif, tss_index=tss_index)
        for name, motif in motifs.items()
    }
    all_target_calls = [call for calls in target_calls.values() for call in calls]
    tiles = []
    for tile_index, nominal_start in enumerate(range(0, 300, tile_width_nt)):
        nominal_end = nominal_start + tile_width_nt
        geometry = paired_fixed_dose_tile(
            target_offsets,
            control_offsets,
            nominal_start=nominal_start,
            nominal_end=nominal_end,
            token_dose=token_dose,
            maximum_cover_nt=maximum_cover_nt,
            maximum_cover_length_difference_nt=maximum_cover_difference_nt,
        )
        genomic_start, genomic_end = oriented_interval_to_genome(
            tss=tss,
            strand=strand,
            start=nominal_start,
            end=nominal_end,
            tss_index=tss_index,
        )
        tiles.append(
            {
                "tile_index": tile_index,
                "nominal_start": nominal_start,
                "nominal_end": nominal_end,
                "relative_start": nominal_start - tss_index,
                "relative_end": nominal_end - tss_index,
                "target_genomic_start": genomic_start,
                "target_genomic_end": genomic_end,
                "eligible": geometry is not None,
                "target_geometry": (
                    asdict(geometry[0]) if geometry is not None else None
                ),
                "control_geometry": (
                    asdict(geometry[1]) if geometry is not None else None
                ),
                "annotation_overlap": {
                    "CAGE_dominant_CTSS": bool(
                        cage.overlapping(
                            chrom, genomic_start, genomic_end, strand=strand
                        )
                    ),
                    "PLS_cCRE": bool(
                        pls.overlapping(chrom, genomic_start, genomic_end)
                    ),
                    "CpG_island": bool(
                        cpg.overlapping(chrom, genomic_start, genomic_end)
                    ),
                    "motifs": sorted(
                        {
                            call.motif
                            for call in all_target_calls
                            if call.start < nominal_end and nominal_start < call.end
                        }
                    ),
                },
            }
        )

    original_geometry = {
        "target_offsets": [tuple(values) for values in target_offsets],
        "control_offsets": [tuple(values) for values in control_offsets],
        "target_ids": list(target_ids),
        "control_ids": list(control_ids),
    }
    disruptions = []
    for motif_index, (name, motif) in enumerate(motifs.items()):
        if not target_calls[name] or control_calls[name]:
            continue
        for instance_index, call in enumerate(target_calls[name]):
            geometry_check = _geometry_check_batch(
                bundle,
                target,
                control,
                original_geometry,
                start=call.start,
                end=call.end,
            )
            disruption = composition_preserving_paired_disruption(
                target,
                control,
                start=call.start,
                end=call.end,
                motif=motif,
                tss_index=tss_index,
                seed=_row_seed(
                    seed,
                    str(row["example_id"]),
                    name,
                    motif_index,
                    instance_index,
                ),
                geometry_qualified_batch=geometry_check,
                candidates=motif_shuffle_candidates,
            )
            if disruption is not None:
                disruptions.append(
                    {
                        "motif": name,
                        "family": motif.family,
                        "position0_relative_tss": call.position0_relative_tss,
                        "cutoff": motif.cutoff,
                        **disruption,
                    }
                )
                break
    return {
        "TSS_index": tss_index,
        "TSS_support": {
            "CAGE_within_tolerance": bool(tss_cage),
            "CAGE_tolerance_nt": cage_tolerance_nt,
            "CAGE_peak_ids": [value.name for value in tss_cage],
            "within_PLS_cCRE": bool(tss_pls),
            "PLS_ids": [value.name for value in tss_pls],
            "within_CpG_island": bool(tss_cpg),
            "CpG_ids": [value.name for value in tss_cpg],
        },
        "motif_calls": {
            "target": {
                name: [asdict(call) for call in calls]
                for name, calls in target_calls.items()
            },
            "control": {
                name: [asdict(call) for call in calls]
                for name, calls in control_calls.items()
            },
        },
        "motif_disruptions": disruptions,
        "tiles": tiles,
    }


def _geometry_check_batch(
    bundle: Any,
    target: str,
    control: str,
    original: dict[str, Any],
    *,
    start: int,
    end: int,
):
    def qualified(
        paired_sequences: list[tuple[str, str]],
    ) -> list[bool]:
        sequences = [sequence for pair in paired_sequences for sequence in pair]
        encoded = bundle.tokenizer(
            sequences,
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        results = []
        for pair_index in range(len(paired_sequences)):
            pair_qualified = True
            for branch_offset, branch in enumerate(("target", "control")):
                sequence_index = 2 * pair_index + branch_offset
                count = int(
                    sum(
                        int(value)
                        for value in encoded["attention_mask"][sequence_index]
                    )
                )
                offsets = [
                    tuple(int(value) for value in pair)
                    for pair in encoded["offset_mapping"][sequence_index][:count]
                ]
                ids = [
                    int(value) for value in encoded["input_ids"][sequence_index][:count]
                ]
                if offsets != original[f"{branch}_offsets"]:
                    pair_qualified = False
                    break
                old_ids = original[f"{branch}_ids"]
                for offset, old_id, new_id in zip(offsets, old_ids, ids):
                    token_start, token_end = offset
                    affected = token_start < end and start < token_end
                    if not affected and old_id != new_id:
                        pair_qualified = False
                        break
                if not pair_qualified:
                    break
            results.append(pair_qualified)
        return results

    return qualified


def _row_seed(seed: int, *values: Any) -> int:
    digest = hashlib.sha256(
        "|".join([str(seed), *(str(value) for value in values)]).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _checks(
    enriched: list[dict[str, Any]],
    selected: dict[str, list[dict[str, Any]]],
    expanded: list[dict[str, Any]],
    original: list[dict[str, Any]],
    previous_supply: dict[str, Any],
    *,
    pairs_per_split: int,
    tile_count: int,
) -> dict[str, bool]:
    original_ids = {str(row["example_id"]) for row in original}
    original_blocks = {str(row["homology_cluster"]) for row in original}
    previous_ids = {
        str(value)
        for values in previous_supply["selection"]["selected_example_ids"].values()
        for value in values
    }
    expanded_by_id = {str(row["example_id"]): row for row in expanded}
    previous_blocks = {
        str(expanded_by_id[value]["homology_cluster"])
        for value in previous_ids
        if value in expanded_by_id
    }
    split_blocks = {
        split: {str(row["homology_cluster"]) for row in rows}
        for split, rows in selected.items()
    }
    return {
        "exact_pairs_per_split": all(
            len(selected[split]) == pairs_per_split
            for split in ("development", "confirmation")
        ),
        "unique_locus_blocks": all(
            len(split_blocks[split]) == pairs_per_split
            for split in ("development", "confirmation")
        ),
        "split_disjoint": not (
            split_blocks["development"] & split_blocks["confirmation"]
        ),
        "fresh_from_original": all(
            str(row["example_id"]) not in original_ids
            and str(row["homology_cluster"]) not in original_blocks
            for row in enriched
        ),
        "fresh_from_previous_D7_ids": all(
            str(row["example_id"]) not in previous_ids for row in enriched
        ),
        "fresh_from_previous_D7_blocks": all(
            str(row["homology_cluster"]) not in previous_blocks for row in enriched
        ),
        "complete_tile_catalog": all(
            len(row["localization"]["tiles"]) == tile_count for row in enriched
        ),
        "motif_disruptions_composition_preserving": all(
            disruption["composition_preserved"]
            for row in enriched
            for disruption in row["localization"]["motif_disruptions"]
        ),
    }


def _tile_summary(enriched: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("development", "confirmation"):
        rows = [row for row in enriched if row["split"] == split]
        split_summary = {}
        for tile_index in range(len(rows[0]["localization"]["tiles"])):
            tiles = [row["localization"]["tiles"][tile_index] for row in rows]
            eligible = [tile for tile in tiles if tile["eligible"]]
            covers = [
                tile[branch]["cover_end"] - tile[branch]["cover_start"]
                for tile in eligible
                for branch in ("target_geometry", "control_geometry")
            ]
            split_summary[str(tile_index)] = {
                "relative_start": tiles[0]["relative_start"],
                "relative_end": tiles[0]["relative_end"],
                "eligible_pairs": len(eligible),
                "actual_cover_nt_min": min(covers) if covers else None,
                "actual_cover_nt_max": max(covers) if covers else None,
                "annotation_overlap_pairs": {
                    feature: sum(
                        bool(tile["annotation_overlap"][feature]) for tile in eligible
                    )
                    for feature in (
                        "CAGE_dominant_CTSS",
                        "PLS_cCRE",
                        "CpG_island",
                    )
                },
                "motif_overlap_pairs": dict(
                    sorted(
                        Counter(
                            motif
                            for tile in eligible
                            for motif in tile["annotation_overlap"]["motifs"]
                        ).items()
                    )
                ),
            }
        output[split] = split_summary
    return output


def _annotation_summary(
    enriched: list[dict[str, Any]], motifs: dict[str, EpdMotif]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "motif_definitions": {
            name: {
                "family": motif.family,
                "length": motif.length,
                "position_offsets": list(motif.position_offsets),
                "preferred_position0_relative_TSS": [
                    motif.preferred_position0_min,
                    motif.preferred_position0_max,
                ],
                "cutoff": motif.cutoff,
                "source_sha256": motif.source_sha256,
            }
            for name, motif in motifs.items()
        }
    }
    for split in ("development", "confirmation"):
        rows = [row for row in enriched if row["split"] == split]
        output[split] = {
            "pairs": len(rows),
            "TSS_support_counts": {
                key: sum(bool(row["localization"]["TSS_support"][key]) for row in rows)
                for key in (
                    "CAGE_within_tolerance",
                    "within_PLS_cCRE",
                    "within_CpG_island",
                )
            },
            "target_motif_positive_pairs": {
                name: sum(
                    bool(row["localization"]["motif_calls"]["target"][name])
                    for row in rows
                )
                for name in motifs
            },
            "control_motif_positive_pairs": {
                name: sum(
                    bool(row["localization"]["motif_calls"]["control"][name])
                    for row in rows
                )
                for name in motifs
            },
            "geometry_qualified_paired_disruptions": {
                name: sum(
                    any(
                        value["motif"] == name
                        for value in row["localization"]["motif_disruptions"]
                    )
                    for row in rows
                )
                for name in motifs
            },
        }
    return output


if __name__ == "__main__":
    main()
