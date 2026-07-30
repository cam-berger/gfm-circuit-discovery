#!/usr/bin/env python
"""Bind fresh loci and native-edit geometry before promoter target outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.promoter_head_analysis import select_locus_rows
from gfmcd.reconstruction_atlas.promoter_head_scan import read_jsonl
from gfmcd.reconstruction_atlas.promoter_native_source import (
    geometry_qualified_native_edit_ensemble,
    native_edit_invariants,
)
from gfmcd.reconstruction_atlas.promoter_source_localization import (
    BinnedIntervalIndex,
    iter_encode_pls,
    iter_fantom5_dominant_ctss,
    iter_ucsc_cpg_islands,
    oriented_interval_to_genome,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-cohort", required=True)
    parser.add_argument("--original-cohort", required=True)
    parser.add_argument("--D7-supply", required=True)
    parser.add_argument("--D8-supply", required=True)
    parser.add_argument("--cohort-audit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cage", required=True)
    parser.add_argument("--ccre", required=True)
    parser.add_argument("--cpg", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--cohort-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    if config.get("status") not in {"DRAFT", "FROZEN"}:
        raise RuntimeError("native-source config must be DRAFT or FROZEN")
    expanded = read_jsonl(args.expanded_cohort)
    version = int(config["schema_version"])
    original = read_jsonl(args.original_cohort)
    D7 = json.loads(Path(args.D7_supply).read_text())
    D8 = json.loads(Path(args.D8_supply).read_text())
    fresh = _fresh_rows(expanded, original, D7, D8)
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    start, end = (
        int(value)
        for value in config["scope"]["candidate_source_region_sequence_indices"]
    )
    dose = int(config["native_interventions"]["gc_depletion_dose"])
    cage = BinnedIntervalIndex(iter_fantom5_dominant_ctss(args.cage))
    pls = BinnedIntervalIndex(iter_encode_pls(args.ccre))
    cpg = BinnedIntervalIndex(iter_ucsc_cpg_islands(args.cpg))

    qualified: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "confirmation": [],
    }
    attrition: dict[str, Counter[str]] = {
        "development": Counter(),
        "confirmation": Counter(),
    }
    for split in ("development", "confirmation"):
        offered = [row for row in fresh if row["split"] == split]
        for index, raw_row in enumerate(offered, start=1):
            row = dict(raw_row)
            target = str(row["target"]["clean_sequence"])
            control = str(row["control"]["clean_sequence"])
            edit_ensemble = geometry_qualified_native_edit_ensemble(
                bundle.tokenizer,
                target,
                control,
                start=start,
                end=end,
                gc_depletion_dose=dose,
                policy=config["native_interventions"],
                seed=_row_seed(
                    int(config["inference"]["seed"]),
                    str(row["example_id"]),
                ),
                replicates=int(
                    config["native_interventions"].get(
                        "edit_replicates_per_locus_and_family",
                        1,
                    )
                ),
                operations=tuple(config["measurement"]["primary_operations"]),
            )
            missing = sorted(
                set(config["measurement"]["primary_operations"]) - set(edit_ensemble)
            )
            if missing:
                for operation in missing:
                    attrition[split][f"missing_{operation}"] += 1
                continue
            for edits in edit_ensemble.values():
                for edit_pair in edits:
                    for branch in ("target", "control"):
                        edit = edit_pair[branch]
                        invariants = native_edit_invariants(
                            target if branch == "target" else control,
                            edit,
                            start=start,
                            end=end,
                        )
                        if not all(invariants.values()):
                            raise RuntimeError(
                                f"native edit invariant failed: {invariants}"
                            )
                        if not edit["token_geometry"]["qualified"]:
                            raise RuntimeError(
                                "selected native edit lost BPE qualification"
                            )
            row["native_source"] = {
                "region": {
                    "start": start,
                    "end": end,
                    "relative_start": start - 249,
                    "relative_end": end - 249,
                },
                "edit_ensemble": edit_ensemble,
                "biological_annotations": _annotations(
                    row,
                    cage=cage,
                    pls=pls,
                    cpg=cpg,
                    start=start,
                    end=end,
                ),
            }
            qualified[split].append(row)
            if index % 20 == 0 or index == len(offered):
                print(
                    f"{split}: audited {index}/{len(offered)}, "
                    f"{len(qualified[split])} geometry-qualified",
                    flush=True,
                )

    selected = {
        split: select_locus_rows(
            qualified[split],
            split=split,
            maximum=int(config["cohort"]["pairs_per_split"]),
            seed=int(config["inference"]["seed"]),
            tag=str(config["cohort"]["selection_tag"]),
        )
        for split in ("development", "confirmation")
    }
    rows = [*selected["development"], *selected["confirmation"]]
    checks = _checks(
        rows,
        expanded,
        original,
        D7,
        D8,
        pairs_per_split=int(config["cohort"]["pairs_per_split"]),
        edit_replicates=int(
            config["native_interventions"].get(
                "edit_replicates_per_locus_and_family",
                1,
            )
        ),
        required_operations=tuple(config["measurement"]["primary_operations"]),
        start=start,
        end=end,
    )
    if not all(checks.values()):
        raise RuntimeError(f"native-source supply failed: {checks}")
    selected_ids = {
        split: [str(row["example_id"]) for row in selected[split]]
        for split in ("development", "confirmation")
    }
    input_paths = (
        args.expanded_cohort,
        args.original_cohort,
        args.D7_supply,
        args.D8_supply,
        args.cohort_audit,
        args.config,
        args.cage,
        args.ccre,
        args.cpg,
        args.source_manifest,
    )
    cohort_path, cohort_manifest = write_jsonl_bundle(
        args.cohort_out,
        rows,
        experiment_id=f"dnabert2_promoter_native_source_cohort_v{version}",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "model_outcomes_computed": False,
            "selection_tag": config["cohort"]["selection_tag"],
            "pairs_per_split": config["cohort"]["pairs_per_split"],
            "native_interventions": config["native_interventions"],
        },
        seeds={"selection_and_edits": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "phase": "PROMOTER_NATIVE_SOURCE_OUTCOME_BLIND_SUPPLY",
        "status": "QUALIFIED",
        "model_outcomes_computed": False,
        "selection": {
            "tag": config["cohort"]["selection_tag"],
            "pairs_per_split": int(config["cohort"]["pairs_per_split"]),
            "fresh_offered": {
                split: sum(row["split"] == split for row in fresh)
                for split in ("development", "confirmation")
            },
            "geometry_qualified": {
                split: len(qualified[split])
                for split in ("development", "confirmation")
            },
            "attrition": {
                split: dict(sorted(attrition[split].items()))
                for split in ("development", "confirmation")
            },
            "selected_example_ids": selected_ids,
            "selected_id_sha256": {
                split: hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()
                for split, values in selected_ids.items()
            },
        },
        "native_edit_geometry": _edit_summary(rows),
        "biological_annotations": _annotation_summary(rows),
        "checks": checks,
        "cohort": {
            "path": str(cohort_path),
            "manifest": str(cohort_manifest),
            "rows": len(rows),
        },
        "boundary": (
            "This outcome-blind artifact uses the pinned tokenizer only. It binds "
            "fresh loci, native edits, matched local BPE-retokenization geometry, "
            "dose, and independent promoter annotations without reading a "
            "classifier margin."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=(f"dnabert2_promoter_native_source_supply_audit_v{version}"),
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=(*input_paths, cohort_path, cohort_manifest),
        parameters={
            "model_outcomes_computed": False,
            "selection": {
                key: value
                for key, value in payload["selection"].items()
                if key != "selected_example_ids"
            },
            "native_interventions": config["native_interventions"],
        },
        seeds={"selection_and_edits": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _fresh_rows(
    expanded: Sequence[Mapping[str, Any]],
    original: Sequence[Mapping[str, Any]],
    D7: Mapping[str, Any],
    D8: Mapping[str, Any],
) -> list[dict[str, Any]]:
    excluded_ids = {str(row["example_id"]) for row in original}
    excluded_clusters = {str(row["homology_cluster"]) for row in original}
    by_id = {str(row["example_id"]): row for row in expanded}
    for audit in (D7, D8):
        for values in audit["selection"]["selected_example_ids"].values():
            for value in values:
                value = str(value)
                excluded_ids.add(value)
                if value not in by_id:
                    raise RuntimeError(
                        f"prior selected ID absent from expansion: {value}"
                    )
                excluded_clusters.add(str(by_id[value]["homology_cluster"]))
    return [
        dict(row)
        for row in expanded
        if str(row["example_id"]) not in excluded_ids
        and str(row["homology_cluster"]) not in excluded_clusters
    ]


def _annotations(
    row: Mapping[str, Any],
    *,
    cage: BinnedIntervalIndex,
    pls: BinnedIntervalIndex,
    cpg: BinnedIntervalIndex,
    start: int,
    end: int,
) -> dict[str, Any]:
    target = row["target"]
    chrom = str(target["chrom"])
    strand = str(target["strand"])
    tss = int(target["tss"])
    genome_start, genome_end = oriented_interval_to_genome(
        tss=tss,
        strand=strand,
        start=start,
        end=end,
        tss_index=249,
    )
    tss_start, tss_end = oriented_interval_to_genome(
        tss=tss,
        strand=strand,
        start=244,
        end=255,
        tss_index=249,
    )
    sequence = str(target["clean_sequence"])
    dce_window = sequence[249 + 16 : 249 + 22]
    return {
        "CAGE_supported_TSS": bool(
            cage.overlapping(chrom, tss_start, tss_end, strand=strand)
        ),
        "PLS_cCRE_at_source_region": bool(
            pls.overlapping(chrom, genome_start, genome_end)
        ),
        "CpG_island_at_source_region": bool(
            cpg.overlapping(chrom, genome_start, genome_end)
        ),
        "DCE_II_CTGT_within_plus16_plus21": "CTGT" in dce_window,
        "human_DPR_overlap": max(start - 249, 17) < min(end - 249, 35),
        "source_region_genomic_start": genome_start,
        "source_region_genomic_end": genome_end,
        "DCE_II_search_window": dce_window,
    }


def _checks(
    rows: Sequence[Mapping[str, Any]],
    expanded: Sequence[Mapping[str, Any]],
    original: Sequence[Mapping[str, Any]],
    D7: Mapping[str, Any],
    D8: Mapping[str, Any],
    *,
    pairs_per_split: int,
    edit_replicates: int,
    required_operations: Sequence[str],
    start: int,
    end: int,
) -> dict[str, bool]:
    original_ids = {str(row["example_id"]) for row in original}
    original_clusters = {str(row["homology_cluster"]) for row in original}
    expanded_by_id = {str(row["example_id"]): row for row in expanded}
    prior_ids = {
        str(value)
        for audit in (D7, D8)
        for values in audit["selection"]["selected_example_ids"].values()
        for value in values
    }
    prior_clusters = {
        str(expanded_by_id[value]["homology_cluster"]) for value in prior_ids
    }
    all_invariants = []
    for row in rows:
        for edits in row["native_source"]["edit_ensemble"].values():
            for operation in edits:
                all_invariants.append(bool(operation["equal_hamming_dose"]))
                for branch in ("target", "control"):
                    original_sequence = str(row[branch]["clean_sequence"])
                    all_invariants.extend(
                        native_edit_invariants(
                            original_sequence,
                            operation[branch],
                            start=start,
                            end=end,
                        ).values()
                    )
                    geometry = operation[branch]["token_geometry"]
                    all_invariants.extend(
                        (
                            bool(geometry["qualified"]),
                            bool(geometry["retokenization_is_local"]),
                        )
                    )
    split_clusters = {
        split: {str(row["homology_cluster"]) for row in rows if row["split"] == split}
        for split in ("development", "confirmation")
    }
    return {
        "exact_pairs_per_split": all(
            sum(row["split"] == split for row in rows) == pairs_per_split
            for split in ("development", "confirmation")
        ),
        "unique_locus_blocks": len({str(row["homology_cluster"]) for row in rows})
        == len(rows),
        "split_disjoint": not (
            split_clusters["development"] & split_clusters["confirmation"]
        ),
        "fresh_from_original_ids": all(
            str(row["example_id"]) not in original_ids for row in rows
        ),
        "fresh_from_original_blocks": all(
            str(row["homology_cluster"]) not in original_clusters for row in rows
        ),
        "fresh_from_D7_D8_ids": all(
            str(row["example_id"]) not in prior_ids for row in rows
        ),
        "fresh_from_D7_D8_blocks": all(
            str(row["homology_cluster"]) not in prior_clusters for row in rows
        ),
        "required_native_operations": all(
            set(row["native_source"]["edit_ensemble"]) == set(required_operations)
            for row in rows
        ),
        "exact_edit_replicates": all(
            len(edits) == edit_replicates
            for row in rows
            for edits in row["native_source"]["edit_ensemble"].values()
        ),
        "native_edit_and_BPE_invariants": all(all_invariants),
    }


def _edit_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    operations = sorted(rows[0]["native_source"]["edit_ensemble"])
    for operation in operations:
        records = [
            edit_pair[branch]
            for row in rows
            for edit_pair in row["native_source"]["edit_ensemble"][operation]
            for branch in ("target", "control")
        ]
        output[operation] = {
            "records": len(records),
            "edits_per_locus": len(
                rows[0]["native_source"]["edit_ensemble"][operation]
            ),
            "hamming_dose": _counts(
                int(record["changed_nucleotides"]) for record in records
            ),
            "changed_token_count": _counts(
                int(record["token_geometry"]["edited_affected_token_count"])
                for record in records
            ),
            "token_count_delta": _counts(
                int(record["token_geometry"]["token_count_delta"]) for record in records
            ),
            "affected_cover_length": _counts(
                int(record["token_geometry"]["affected_cover_length"])
                for record in records
            ),
            "GC_count_change": _counts(
                int(record["gc_count_change"]) for record in records
            ),
        }
    return output


def _annotation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = tuple(row["native_source"]["biological_annotations"] for row in rows[:1])
    if not names:
        return {}
    keys = [key for key, value in names[0].items() if isinstance(value, bool)]
    return {
        split: {
            key: sum(
                row["split"] == split
                and bool(row["native_source"]["biological_annotations"][key])
                for row in rows
            )
            for key in keys
        }
        for split in ("development", "confirmation")
    }


def _counts(values: Sequence[int] | Any) -> dict[str, Any]:
    counter = Counter(int(value) for value in values)
    return {str(key): counter[key] for key in sorted(counter)}


def _row_seed(seed: int, pair_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}|{pair_id}".encode()).digest()[:8],
        "big",
    )


if __name__ == "__main__":
    main()
