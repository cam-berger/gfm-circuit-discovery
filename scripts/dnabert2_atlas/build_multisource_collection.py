#!/usr/bin/env python
"""Build the frozen, multi-source DNABERT-2 reconstruction atlas collection."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path

from gfmcd.adapters.dnabert2 import DNABERT2_HF_ID, DNABERT2_REVISION
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.assembly import (
    GenomicFeature,
    assemble_annotated_windows,
    gencode_features_from_gtf,
    select_gencode_gene_ids,
)
from gfmcd.reconstruction_atlas.sources import (
    derive_polypyrimidine_tracts,
    iter_encode_dels_cores,
    iter_encode_tf_peak_cores,
    iter_epdnew_tss_cores,
    iter_polyasite_elements,
    liftover_branchpoints_roundtrip,
    sample_locus_blocked_features,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--epdnew-bed", required=True)
    parser.add_argument("--polyasite", required=True)
    parser.add_argument("--tf-clusters", required=True)
    parser.add_argument("--ccre", required=True)
    parser.add_argument("--branchpoints-hg19", required=True)
    parser.add_argument("--hg19-to-hg38-chain", required=True)
    parser.add_argument("--hg38-to-hg19-chain", required=True)
    parser.add_argument("--liftover-binary", required=True)
    parser.add_argument("--preflight-config", required=True)
    parser.add_argument("--target-out", required=True)
    parser.add_argument("--baseline-out", required=True)
    parser.add_argument("--annotation-bed-out", required=True)
    parser.add_argument("--source-audit-out", required=True)
    args = parser.parse_args()

    try:
        from pyfaidx import Fasta
    except ImportError as exc:
        raise SystemExit(
            "pyfaidx is required; run with the study's pinned requirements"
        ) from exc

    config = json.loads(Path(args.preflight_config).read_text())
    if config.get("status") != "FROZEN":
        raise SystemExit("multi-source collection requires a FROZEN configuration")
    collection_config = config["collection"]
    source_config = config["sources"]
    seed = int(config["seed"])
    development_percent = int(collection_config["split_percentages"]["development"])
    baseline_percent = int(collection_config["split_percentages"]["baseline_training"])
    maximum_per_class = int(
        collection_config["maximum_clusters_per_site_class_per_split"]
    )
    flank = int(collection_config["window_upstream_nt"])
    if flank != int(collection_config["window_downstream_nt"]):
        raise SystemExit("locus-blocked v2 builder requires symmetric windows")

    fasta = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)
    selected_by_provisional_split = select_gencode_gene_ids(
        args.gtf,
        seed=seed,
        maximum_per_split=int(
            collection_config["sampled_gencode_genes_per_provisional_split"]
        ),
        development_percent=development_percent,
        baseline_training_percent=baseline_percent,
        gene_types=frozenset(collection_config["gene_types"]),
        chromosomes=frozenset(collection_config["chromosomes"]),
    )
    selected_gene_ids = frozenset(
        gene_id
        for gene_ids in selected_by_provisional_split.values()
        for gene_id in gene_ids
    )
    gencode = gencode_features_from_gtf(
        args.gtf,
        source_version=str(source_config["gencode"]["release"]),
        include_gene_ids=selected_gene_ids,
    )
    ppt = derive_polypyrimidine_tracts(
        gencode,
        fasta,
        tract_width=int(source_config["polypyrimidine_tract"]["target_width_nt"]),
        search_upstream_nt=int(
            source_config["polypyrimidine_tract"]["search_upstream_nt"]
        ),
        acceptor_buffer_nt=int(
            source_config["polypyrimidine_tract"]["acceptor_buffer_nt"]
        ),
    )
    branchpoints, branchpoint_audit = liftover_branchpoints_roundtrip(
        args.branchpoints_hg19,
        forward_chain=args.hg19_to_hg38_chain,
        reverse_chain=args.hg38_to_hg19_chain,
        liftover_binary=args.liftover_binary,
        fasta=fasta,
    )
    source_iterables: tuple[tuple[str, Iterable[GenomicFeature]], ...] = (
        ("GENCODE", chain(gencode, ppt)),
        (
            "EPDnew",
            iter_epdnew_tss_cores(
                args.epdnew_bed,
                source_version=str(source_config["epdnew"]["release"]),
                core_width=int(source_config["epdnew"]["target_width_nt"]),
            ),
        ),
        (
            "PolyASite",
            iter_polyasite_elements(
                args.polyasite,
                source_version=str(source_config["polyasite"]["release"]),
                minimum_fraction_samples=float(
                    source_config["polyasite"]["minimum_fraction_samples"]
                ),
                minimum_protocols=int(source_config["polyasite"]["minimum_protocols"]),
                allowed_annotations=frozenset(
                    source_config["polyasite"]["allowed_annotations"]
                ),
            ),
        ),
        (
            "ENCODE-TF",
            iter_encode_tf_peak_cores(
                args.tf_clusters,
                factors=tuple(source_config["encode_tf"]["factors"]),
                source_version=str(source_config["encode_tf"]["release"]),
                minimum_score=int(source_config["encode_tf"]["minimum_score"]),
                minimum_source_count=int(
                    source_config["encode_tf"]["minimum_source_count"]
                ),
                core_width=int(source_config["encode_tf"]["target_width_nt"]),
            ),
        ),
        ("Mercer-branchpoints", branchpoints),
        (
            "ENCODE-cCRE",
            iter_encode_dels_cores(
                args.ccre,
                source_version=str(source_config["encode_ccre"]["release"]),
                core_width=int(source_config["encode_ccre"]["target_width_nt"]),
            ),
        ),
    )
    source_counts: Counter[str] = Counter()
    admitted_classes = frozenset(config["behavior"]["admitted_site_classes"])
    annotation_path = Path(args.annotation_bed_out)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    if annotation_path.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable annotation file: {annotation_path}"
        )
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{annotation_path.name}.",
        suffix=".tmp",
        dir=annotation_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as annotations:
            selected, sampling_audit = sample_locus_blocked_features(
                _tee_features(
                    source_iterables,
                    annotations,
                    source_counts,
                    admitted_classes=admitted_classes,
                ),
                seed=seed,
                maximum_clusters_per_site_class=maximum_per_class,
                development_percent=development_percent,
                baseline_training_percent=baseline_percent,
                window_flank_nt=flank,
                locus_block_nt=int(collection_config["locus_block_nt"]),
            )
        os.replace(temporary_path, annotation_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    _assert_expected_classes(
        selected,
        admitted_classes,
        maximum_per_class=maximum_per_class,
        seed=seed,
        development_percent=development_percent,
        baseline_percent=baseline_percent,
    )
    records = assemble_annotated_windows(
        selected,
        fasta,
        annotation_features=selected,
        assembly=str(collection_config["assembly"]),
        upstream_nt=flank,
        downstream_nt=flank,
        split_seed=seed,
        development_percent=development_percent,
        baseline_training_percent=baseline_percent,
        immediate_flank_nt=int(collection_config["immediate_flank_nt"]),
    )
    target_records = [
        record for record in records if record.split in {"development", "confirmation"}
    ]
    baseline_records = [
        record for record in records if record.split == "baseline_training"
    ]
    if not target_records or not baseline_records:
        raise SystemExit(
            "multi-source collection did not populate every required split"
        )

    source_paths = (
        args.gtf,
        args.fasta,
        args.epdnew_bed,
        args.polyasite,
        args.tf_clusters,
        args.ccre,
        args.branchpoints_hg19,
        args.hg19_to_hg38_chain,
        args.hg38_to_hg19_chain,
        args.liftover_binary,
        args.preflight_config,
    )
    source_audit = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "phase": "MULTISOURCE_COLLECTION_BUILD",
        "source_feature_counts": dict(sorted(source_counts.items())),
        "branchpoint_liftover": branchpoint_audit,
        "sampling": sampling_audit,
        "annotation_bed": {
            "path": str(annotation_path),
            "sha256": sha256_file(annotation_path),
            "coordinate_system": "GRCh38_zero_based_half_open",
            "purpose": "conservative control exclusion only",
        },
        "selected_gencode_gene_counts_by_provisional_split": {
            split: len(values)
            for split, values in sorted(selected_by_provisional_split.items())
        },
        "selected_examples": {
            "development": sum(
                record.split == "development" for record in target_records
            ),
            "baseline_training": len(baseline_records),
            "confirmation": sum(
                record.split == "confirmation" for record in target_records
            ),
        },
        "target_definitions_frozen_before_model_outcomes": True,
        "model_outcomes_computed": False,
    }
    audit_result, audit_manifest = write_result_bundle(
        args.source_audit_out,
        source_audit,
        experiment_id="dnabert2_atlas_multisource_build_v2",
        status="PREFLIGHT",
        model_identity={
            "hf_id": DNABERT2_HF_ID,
            "revision": DNABERT2_REVISION,
            "used_for_collection_build": False,
        },
        input_paths=source_paths,
        parameters={
            "source_configuration": source_config,
            "collection_configuration": collection_config,
        },
        seeds={"collection": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    common_inputs = (*source_paths, annotation_path, audit_result, audit_manifest)
    common_parameters = {
        "assembly": collection_config["assembly"],
        "source_catalog_release": collection_config["source_catalog_release"],
        "site_classes": sorted(config["behavior"]["admitted_site_classes"]),
        "split_percentages": collection_config["split_percentages"],
        "maximum_clusters_per_site_class_per_split": maximum_per_class,
        "locus_block_nt": collection_config["locus_block_nt"],
        "window_upstream_nt": flank,
        "window_downstream_nt": flank,
        "source_feature_counts": dict(sorted(source_counts.items())),
        "sampling_audit": sampling_audit,
    }
    write_jsonl_bundle(
        args.target_out,
        [record.to_dict() for record in target_records],
        experiment_id="dnabert2_atlas_multisource_targets_v2",
        status="DEVELOPMENT",
        model_identity={
            "hf_id": DNABERT2_HF_ID,
            "revision": DNABERT2_REVISION,
            "used_for_collection_build": False,
        },
        input_paths=common_inputs,
        parameters={
            **common_parameters,
            "included_splits": ["development", "confirmation"],
            "n_examples": len(target_records),
        },
        seeds={"collection": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    write_jsonl_bundle(
        args.baseline_out,
        [record.to_dict() for record in baseline_records],
        experiment_id="dnabert2_atlas_multisource_baseline_v2",
        status="DEVELOPMENT",
        model_identity={
            "hf_id": DNABERT2_HF_ID,
            "revision": DNABERT2_REVISION,
            "used_for_collection_build": False,
        },
        input_paths=common_inputs,
        parameters={
            **common_parameters,
            "included_splits": ["baseline_training"],
            "n_examples": len(baseline_records),
        },
        seeds={"collection": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )


def _tee_features(
    sources: Iterable[tuple[str, Iterable[GenomicFeature]]],
    annotations,
    counts: Counter[str],
    *,
    admitted_classes: frozenset[str],
) -> Iterator[GenomicFeature]:
    for source_name, features in sources:
        for feature in features:
            counts[source_name] += 1
            annotation_start = int(
                feature.attributes.get("source_interval_start", feature.start)
            )
            annotation_end = int(
                feature.attributes.get("source_interval_end", feature.end)
            )
            annotations.write(
                "\t".join(
                    (
                        feature.chrom,
                        str(annotation_start),
                        str(annotation_end),
                        feature.feature_id,
                        feature.site_class,
                    )
                )
                + "\n"
            )
            if feature.site_class in admitted_classes:
                yield feature
            else:
                counts[f"{source_name}:annotation_only"] += 1


def _assert_expected_classes(
    features: Iterable[GenomicFeature],
    expected: frozenset[str],
    *,
    maximum_per_class: int,
    seed: int,
    development_percent: int,
    baseline_percent: int,
) -> None:
    from gfmcd.reconstruction_atlas.assembly import split_for_cluster

    counts = Counter(
        (
            split_for_cluster(
                feature.homology_cluster,
                seed=seed,
                development_percent=development_percent,
                baseline_training_percent=baseline_percent,
            ),
            feature.site_class,
        )
        for feature in features
    )
    observed = {site_class for _split, site_class in counts}
    if observed != expected:
        raise SystemExit(
            "selected site classes disagree with frozen config: "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )
    incomplete = {
        f"{split}|{site_class}": counts[(split, site_class)]
        for split in ("development", "baseline_training", "confirmation")
        for site_class in sorted(expected)
        if counts[(split, site_class)] < maximum_per_class
    }
    if incomplete:
        raise SystemExit(f"source supply failed frozen per-class quota: {incomplete}")


if __name__ == "__main__":
    main()
