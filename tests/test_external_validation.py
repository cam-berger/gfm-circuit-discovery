from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from gfmcd.provenance import sha256_file
from gfmcd.splicing.external_validation import (
    COMPASS_DESIGN_FIELDS,
    COMPASS_LIBRARY_PREFIX,
    COMPASS_LIBRARY_SUFFIX,
    COMPASS_CUSTODIAN_SCHEMA,
    DESIGN_DOWNLOADS,
    DESIGN_POOL_SCHEMA,
    FEATURE_SCORE_SCHEMA,
    FROZEN_WINDOWS,
    LOCK_A_SCHEMA,
    compass_projection_report,
    freeze_candidate_cohort,
    freeze_design_pool,
    git_blob_sha1,
    iter_compass_design_records,
    iter_vex_design_records,
    read_compass_fasta_index,
    read_compass_metadata,
    read_lock_a,
    validate_lock_a,
)


def lock_a(tmp_path: Path) -> dict:
    artifacts = []
    for artifact_id, role in (
        ("candidate-evidence", "candidate_definition"),
        ("native-unit-evidence", "native_unit_definition"),
        ("causal-evidence", "causal_mediation"),
    ):
        path = tmp_path / f"{artifact_id}.json"
        path.write_text(json.dumps({"artifact_id": artifact_id, "role": role}))
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "role": role,
                "path": path.name,
                "sha256": sha256_file(path),
                "schema": "gfmcd.test.internal_artifact.v1",
            }
        )
    return {
        "schema": LOCK_A_SCHEMA,
        "status": "FROZEN_LOCK_A",
        "external_assay_outcomes_inspected_or_used": False,
        "windows": FROZEN_WINDOWS,
        "internal_artifacts": artifacts,
        "model": {
            "checkpoint_id": "splicebert-gtex-test",
            "checkpoint_sha256": "a" * 64,
            "tokenizer_id": "splicebert-tokenizer-test",
            "tokenizer_sha256": "b" * 64,
            "native_margin_definition": "target class logit minus max rival class logit",
            "class_labels": {"acceptor": 0, "donor": 1, "neither": 2},
        },
        "native_units": [
            {
                "native_unit_id": "l5-feature-17",
                "kind": "mlp_feature",
                "layer": 5,
                "hook_location": "encoder.layer.5.output",
                "position_scope": "splice-classification token",
                "aggregation": "signed scalar activation",
                "score_definition": "alternate minus reference activation",
                "artifact_ids": ["native-unit-evidence", "causal-evidence"],
            }
        ],
        "candidates": [
            {
                "candidate_id": "feature-17",
                "hypothesis": "a frozen L5 feature carries enhancer-like exonic evidence",
                "feature_definition": "signed l5-feature-17 activation",
                "evidence_label": "internally causal candidate; biology unresolved",
                "native_unit_ids": ["l5-feature-17"],
                "artifact_ids": ["candidate-evidence", "causal-evidence"],
                "window_roles": ["acceptor", "donor"],
                "expected_feature_delta_sign": 1,
                "expected_model_effect_sign": 1,
                "expected_reporter_effect_sign_by_dataset": {
                    "vex_seq": 1,
                    "compass": 1,
                },
                "carrier_thresholds": {
                    "reference_score_min": 0.5,
                    "signed_feature_delta_min": 0.5,
                    "carrier_model_margin_abs_min": 0.5,
                    "noncarrier_abs_feature_delta_max": 0.1,
                    "noncarrier_model_margin_abs_max": 0.1,
                },
            }
        ],
        "model_endpoint": {
            "name": "alt_minus_ref_class_margin",
            "contrast": "alternate_minus_reference",
            "units": "logit",
            "class_by_window": {"acceptor": "acceptor", "donor": "donor"},
        },
        "reporter_endpoints": {
            "vex_seq": {
                "name": "delta_psi",
                "contrast": "alternate_minus_reference",
                "units": "proportion",
                "cell_aggregation": "author_mean_K562_HepG2",
                "missingness_rule": "report all frozen variants; no outcome-based replacement",
            },
            "compass": {
                "name": "delta_logit_pooled",
                "contrast": "alternate_minus_reference",
                "units": "logit_PSI",
                "cell_aggregation": "mean_across_five_prespecified_cells",
                "missingness_rule": "available-cell mean; individual cells secondary",
            },
        },
        "control_design": {
            "positive_controls": [
                {
                    "control_id": "canonical-site-destruction",
                    "definition": "canonical splice-site variants must shift the native margin",
                    "pass_condition": "signed median effect exceeds frozen SESOI",
                }
            ],
            "negative_controls": [
                {
                    "control_id": "feature-neutral-edit",
                    "definition": "feature- and margin-neutral variants",
                    "pass_condition": "effect remains within frozen equivalence band",
                }
            ],
            "matching": {
                "controls_per_carrier": 1,
                "without_replacement": True,
                "tie_breaker": "record_id",
                "exact_fields": ["dataset", "window_role"],
                "numeric_calipers": {
                    "relative_offset": 1.0,
                    "reference_feature_score": 2.0,
                    "model_margin_delta": 2.0,
                    "sequence_gc": 1.0,
                },
                "distance_fields": [
                    "relative_offset",
                    "reference_feature_score",
                    "model_margin_delta",
                    "sequence_gc",
                ],
            },
        },
        "grouping_resampling": {
            "inference_unit": "exon_or_gene_not_variant",
            "group_by_dataset": {"vex_seq": "assayed_exon", "compass": "gene_exon"},
            "bootstrap": {
                "method": "cluster_bootstrap",
                "draws": 2_000,
                "seed": 20260730,
            },
            "permutation": {
                "method": "within_group_label_shuffle",
                "draws": 2_000,
                "seed": 20260731,
                "add_one_pvalue": True,
            },
        },
        "effect_thresholds": {
            "feature_signed_delta_min": 0.5,
            "model_margin_abs_min": 0.5,
            "vex_delta_psi_abs_min": 0.05,
            "compass_delta_logit_abs_min": 0.2,
        },
        "multiplicity": {
            "method": "benjamini_hochberg",
            "alpha": 0.05,
            "family_dimensions": [
                "candidate",
                "window",
                "dataset",
                "reporter_endpoint",
            ],
        },
        "decision_rule": (
            "advance only same-signed internally mediated effects that replicate in both "
            "Vex-seq and COMPASS under the complete frozen family"
        ),
    }


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_sequence(
    length: int,
    variant_index: int,
    alternate: str,
    *,
    acceptor_index: int | None = None,
    donor_index: int | None = None,
) -> str:
    sequence = ["C"] * length
    if acceptor_index is not None:
        sequence[acceptor_index - 2 : acceptor_index] = list("AG")
    if donor_index is not None:
        sequence[donor_index : donor_index + 2] = list("GT")
    sequence[variant_index] = alternate
    return "".join(sequence)


def vex_files(tmp_path: Path) -> tuple[Path, Path]:
    positions = tmp_path / "vex_positions.tsv"
    sequences = tmp_path / "vex_sequences.tsv"
    write_tsv(
        positions,
        ["variant", "3SS", "5SS"],
        [
            {"variant": "chr1_1060_A_G", "3SS": "20", "5SS": "-60"},
            {"variant": "chr2_2100_A_G", "3SS": "20", "5SS": "-60"},
            {"variant": "chr3_3060_A_G", "3SS": "90", "5SS": "-20"},
            {"variant": "chr4_4060_AT_G", "3SS": "20", "5SS": "-60"},
        ],
    )
    fieldnames = [
        "chromosome",
        "position",
        "reference",
        "alternative",
        "Upstream Int Adj",
        "Down Int Adj",
        "Exon up",
        "Exon down",
        "strand",
        "Sequence",
        "Final Sequence",
        "Barcode",
    ]
    plus = {
        "chromosome": "chr1",
        "position": "1060",
        "reference": "A",
        "alternative": "G",
        "Upstream Int Adj": "1000",
        "Down Int Adj": "1160",
        "Exon up": "1041",
        "Exon down": "1119",
        "strand": "+",
        "Sequence": make_sequence(
            161,
            60,
            "G",
            acceptor_index=41,
            donor_index=120,
        ),
        "Final Sequence": "not-used",
        "Barcode": "AAAA",
    }
    minus = {
        "chromosome": "chr2",
        "position": "2100",
        "reference": "A",
        "alternative": "G",
        "Upstream Int Adj": "2000",
        "Down Int Adj": "2160",
        "Exon up": "2041",
        "Exon down": "2119",
        "strand": "-",
        "Sequence": make_sequence(
            161,
            60,
            "C",
            acceptor_index=41,
            donor_index=120,
        ),
        "Final Sequence": "not-used",
        "Barcode": "CCCC",
    }
    write_tsv(sequences, fieldnames, [plus, {**plus, "Barcode": "TTTT"}, minus])
    return positions, sequences


def compass_sequence(
    *,
    variant_index: int,
    oriented_alternate: str,
) -> tuple[str, str, str, str]:
    intron1 = list("C" * 40)
    intron1[-2:] = list("AG")
    exon = list("C" * 80)
    intron2 = list("C" * 41)
    intron2[:2] = list("GT")
    if variant_index < len(intron1):
        intron1[variant_index] = oriented_alternate
    elif variant_index < len(intron1) + len(exon):
        exon[variant_index - len(intron1)] = oriented_alternate
    else:
        intron2[variant_index - len(intron1) - len(exon)] = oriented_alternate
    full = intron1 + exon + intron2
    return "".join(intron1), "".join(exon), "".join(intron2), "".join(full)


def compass_rows() -> list[dict]:
    i1, exon, i2, plus = compass_sequence(variant_index=60, oriented_alternate="G")
    j1, exon2, j2, minus = compass_sequence(variant_index=60, oriented_alternate="C")
    base = {
        "gene_exon": "GENE exon 2",
        "snp": "20:A>G",
        "source": "synthetic",
        "seq_type": "clinvar_single",
        "intron1": i1,
        "exon": exon,
        "intron2": i2,
        "transcript_id": "ENST000001",
        "K562_dpsi_pooled": "DO_NOT_SERIALIZE",
        "K562_delta_logit_pooled": "DO_NOT_SERIALIZE_EITHER",
    }
    return [
        {
            **base,
            "Reference": "1",
            "event_id": "chr1:1000-1160:+",
            "full_seq": plus,
            "exon_start_hg38": "1040",
            "exon_end_hg38": "1119",
            "variant_hg38": "chr1:1060:A>G",
        },
        {
            **base,
            "Reference": "2",
            "event_id": "chr2:2000-2160:-",
            "intron1": j1,
            "exon": exon2,
            "intron2": j2,
            "full_seq": minus,
            "exon_start_hg38": "2041",
            "exon_end_hg38": "2120",
            "variant_hg38": "chr2:2100:A>G",
        },
        {
            **base,
            "Reference": "3",
            "event_id": "chr1:1000-1160:+",
            "snp": "none",
            "full_seq": plus,
            "exon_start_hg38": "1040",
            "exon_end_hg38": "1119",
            "variant_hg38": "none",
        },
    ]


def write_compass(
    path: Path,
    rows: list[dict] | None = None,
    *,
    include_outcomes: bool = True,
) -> None:
    rows = rows or compass_rows()
    fields = list(COMPASS_DESIGN_FIELDS)
    if include_outcomes:
        fields.extend(("K562_dpsi_pooled", "K562_delta_logit_pooled"))
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{field: row.get(field, "") for field in fields} for row in rows]
        )


def write_compass_metadata(path: Path, *, include_outcomes: bool = True) -> None:
    fields = list(COMPASS_DESIGN_FIELDS)
    if include_outcomes:
        fields.extend(("K562_dpsi_pooled", "K562_delta_logit_pooled"))
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Column", "Definition"],
            delimiter="\t",
        )
        writer.writeheader()
        for field in fields:
            writer.writerow({"Column": field, "Definition": f"definition of {field}"})


def _fasta_variant_descriptor(variant_hg38: str) -> str:
    chrom, position, change = variant_hg38.split(":")
    reference, alternate = change.split(">")
    return f"{chrom.removeprefix('chr')}:{position}:{reference}:{alternate}"


def write_compass_fasta(path: Path, rows: list[dict] | None = None) -> None:
    rows = rows or compass_rows()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            if row["variant_hg38"] == "none":
                continue
            reference_id = row["Reference"]
            chrom = row["event_id"].split(":", 1)[0].removeprefix("chr")
            strand = row["event_id"].rsplit(":", 1)[1]
            snp = _fasta_variant_descriptor(row["variant_hg38"])
            handle.write(
                f">{reference_id}_1 Event ID:SE:{chrom}:900-999:1100-1200:{strand}"
                f"|Exon ID:ENSGTEST_2|Gene Name:GENE|SNP:{snp};{row['seq_type']}\n"
            )
            library = (
                COMPASS_LIBRARY_PREFIX
                + row["full_seq"]
                + COMPASS_LIBRARY_SUFFIX
                + "A" * 20
            )
            handle.write(library + "\n")


def write_custodian_manifest(
    path: Path,
    *,
    design_path: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": COMPASS_CUSTODIAN_SCHEMA,
                "source_accession": "GSE307247",
                "source_filename": "GSE307247_Processed_PSIs_All_Cells.csv.gz",
                "source_sha256": "c" * 64,
                "design_export_sha256": sha256_file(design_path),
                "retained_columns": list(COMPASS_DESIGN_FIELDS),
                "outcome_values_used_for_export": False,
                "custodian_id": "test-custodian",
                "exporter_version": "test-v1",
            }
        )
    )


def create_design_pool(tmp_path: Path) -> tuple[Path, Path]:
    lock_path = tmp_path / "lock_a.json"
    lock_path.write_text(json.dumps(lock_a(tmp_path)))
    positions, sequences = vex_files(tmp_path)
    compass = tmp_path / "compass.csv.gz"
    metadata = tmp_path / "compass_metadata.tsv.gz"
    fasta = tmp_path / "compass_reference.fasta.gz"
    write_compass(compass)
    write_compass_metadata(metadata)
    write_compass_fasta(fasta)
    output = tmp_path / "design_pool"
    freeze_design_pool(
        lock_a_path=lock_path,
        vex_positions_path=positions,
        vex_sequences_path=sequences,
        compass_input_path=compass,
        compass_input_kind="combined_container",
        compass_metadata_path=metadata,
        compass_fasta_path=fasta,
        compass_custodian_manifest_path=None,
        output_dir=output,
        enforce_pinned_sources=False,
    )
    return lock_path, output / "design_pool.json"


def write_feature_scores(
    path: Path,
    *,
    design_pool_path: Path,
    omit_last: bool = False,
) -> None:
    package = json.loads(design_pool_path.read_text())
    records = []
    for artifact in package["design_pools"].values():
        source = design_pool_path.parent / artifact["path"]
        records.extend(json.loads(line) for line in source.read_text().splitlines())
    rows = []
    carrier_ids = {"vex:chr1_1060_A_G", "compass:1"}
    for record in sorted(records, key=lambda value: value["record_id"]):
        carrier = record["record_id"] in carrier_ids
        reference_score = 1.0 if carrier else 0.0
        alternate_score = 2.0 if carrier else 0.0
        for role in record["window_labels"]:
            rows.append(
                {
                    "schema": FEATURE_SCORE_SCHEMA,
                    "design_record_id": record["record_id"],
                    "candidate_id": "feature-17",
                    "window_role": role,
                    "model_endpoint_name": "alt_minus_ref_class_margin",
                    "native_unit_ids": ["l5-feature-17"],
                    "reference_feature_score": reference_score,
                    "alternate_feature_score": alternate_score,
                    "feature_delta": alternate_score - reference_score,
                    "model_margin_delta": 1.0 if carrier else 0.0,
                    "sequence_gc": (
                        sum(base in {"G", "C"} for base in record["reference_sequence"])
                        / len(record["reference_sequence"])
                    ),
                }
            )
    if omit_last:
        rows.pop()
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_feature_score_manifest(
    path: Path,
    *,
    scores_path: Path,
    lock_path: Path,
    design_pool_path: Path,
) -> None:
    frozen = json.loads(lock_path.read_text())
    path.write_text(
        json.dumps(
            {
                "schema": "gfmcd.external_validation.feature_score_manifest.v1",
                "status": "OUTCOME_BLIND_MODEL_FEATURE_SCORES",
                "external_assay_outcomes_inspected_or_used": False,
                "score_file_sha256": sha256_file(scores_path),
                "lock_a_sha256": sha256_file(lock_path),
                "design_pool_sha256": sha256_file(design_pool_path),
                "model_checkpoint_sha256": frozen["model"]["checkpoint_sha256"],
                "model_endpoint": frozen["model_endpoint"],
                "candidate_ids": ["feature-17"],
                "native_unit_ids": ["l5-feature-17"],
                "score_tuple_family": "all_design_records_x_candidate_window_roles",
                "scoring_code_sha256": "d" * 64,
                "scoring_command": "fixture-only model scoring",
            }
        )
    )


def test_lock_a_requires_substantive_hashed_outcome_blind_contract(tmp_path):
    frozen = lock_a(tmp_path)
    validate_lock_a(frozen, base_dir=tmp_path, verify_artifacts=True)
    lock_path = tmp_path / "lock_a.json"
    lock_path.write_text(json.dumps(frozen))
    assert read_lock_a(lock_path)["schema"] == LOCK_A_SCHEMA

    exposed = lock_a(tmp_path)
    exposed["external_assay_outcomes_inspected_or_used"] = True
    with pytest.raises(RuntimeError, match="inspected_or_used"):
        validate_lock_a(exposed)
    drifted = lock_a(tmp_path)
    drifted["windows"] = {**FROZEN_WINDOWS, "acceptor": {"start": 15, "end": 48}}
    with pytest.raises(ValueError, match="windows"):
        validate_lock_a(drifted)
    duplicate = lock_a(tmp_path)
    duplicate["candidates"] *= 2
    with pytest.raises(ValueError, match="unique"):
        validate_lock_a(duplicate)
    incomplete = lock_a(tmp_path)
    del incomplete["multiplicity"]
    with pytest.raises(ValueError, match="multiplicity"):
        validate_lock_a(incomplete)


def test_lock_a_verifies_internal_artifact_hashes(tmp_path):
    frozen = lock_a(tmp_path)
    lock_path = tmp_path / "lock_a.json"
    lock_path.write_text(json.dumps(frozen))
    (tmp_path / "candidate-evidence.json").write_text("tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        read_lock_a(lock_path)


def test_vex_projection_collapses_barcodes_and_reverse_complements_minus(tmp_path):
    positions, sequences = vex_files(tmp_path)
    stats: dict = {}
    rows = list(iter_vex_design_records(positions, sequences, stats=stats))
    assert [row["record_id"] for row in rows] == [
        "vex:chr1_1060_A_G",
        "vex:chr2_2100_A_G",
    ]
    assert stats == {
        "selected_position_variants": 2,
        "matching_barcode_rows": 3,
        "barcode_collapsed_records": 2,
        "junction_rejected_records": 0,
        "junction_rejection_reasons": {},
    }
    plus, minus = rows
    assert plus["variant_index_transcript_0based"] == 60
    assert plus["reference_sequence"][60] == "A"
    assert plus["alternate_sequence"][60] == "G"
    assert minus["variant_index_transcript_0based"] == 60
    assert minus["reference_sequence"][60] == "T"
    assert minus["alternate_sequence"][60] == "C"
    assert plus["acceptor_target_index_transcript_0based"] == 41
    assert plus["donor_target_index_transcript_0based"] == 120
    assert plus["reference_sequence"][39:41] == "AG"
    assert plus["reference_sequence"][120:122] == "GT"
    assert set(plus["window_labels"]) == {"acceptor", "donor"}


def test_vex_projection_rejects_coordinate_or_allele_mismatch(tmp_path):
    positions, sequences = vex_files(tmp_path)
    text = sequences.read_text()
    sequences.write_text(text.replace("\t1000\t1160\t", "\t1000\t1161\t", 1))
    with pytest.raises(ValueError, match="1-based inclusive"):
        list(iter_vex_design_records(positions, sequences))


def test_vex_projection_excludes_noncanonical_junctions(tmp_path):
    positions, sequences = vex_files(tmp_path)
    rows = list(csv.DictReader(sequences.open(), delimiter="\t"))
    for row in rows:
        if row["chromosome"] == "chr2":
            sequence = list(row["Sequence"])
            sequence[39:41] = list("CC")
            row["Sequence"] = "".join(sequence)
    write_tsv(sequences, list(rows[0]), rows)
    stats: dict = {}
    projected = list(iter_vex_design_records(positions, sequences, stats=stats))
    assert [row["record_id"] for row in projected] == ["vex:chr1_1060_A_G"]
    assert stats["junction_rejected_records"] == 1
    assert stats["junction_rejection_reasons"] == {"noncanonical_acceptor": 1}


def test_compass_projection_strips_outcomes_and_checks_both_strands(tmp_path):
    source = tmp_path / "compass.csv.gz"
    write_compass(source)
    metadata = tmp_path / "metadata.tsv.gz"
    fasta = tmp_path / "reference.fasta.gz"
    write_compass_metadata(metadata)
    write_compass_fasta(fasta)
    definitions = read_compass_metadata(metadata)
    fasta_index, fasta_stats = read_compass_fasta_index(fasta)
    assert len(fasta_index) == 2
    assert fasta_stats["library_length"] == 1115
    report = compass_projection_report(
        source,
        input_kind="combined_container",
        metadata_columns=definitions,
    )
    assert report["stripped_outcome_columns"] == [
        "K562_dpsi_pooled",
        "K562_delta_logit_pooled",
    ]
    assert report["complete_source_rows_parsed_into_temporary_csv_lists"] is True
    assert report["outcome_columns_projected_into_row_mappings"] is False
    stats: dict = {}
    rows = list(
        iter_compass_design_records(
            source,
            metadata_path=metadata,
            fasta_path=fasta,
            input_kind="combined_container",
            stats=stats,
        )
    )
    assert len(rows) == 2
    assert stats["row_counts"] == {"selected": 2, "source_rows": 3, "wild_type": 1}
    plus, minus = rows
    assert plus["acceptor_offset"] == 20
    assert plus["donor_offset"] == -60
    assert plus["variant_index_transcript_0based"] == 60
    assert minus["acceptor_offset"] == 20
    assert minus["donor_offset"] == -60
    assert minus["variant_index_transcript_0based"] == 60
    serialized = json.dumps(rows)
    assert "DO_NOT_SERIALIZE" not in serialized
    assert "dpsi" not in serialized.lower()
    assert minus["reference_sequence"][60] == "T"
    assert minus["alternate_sequence"][60] == "C"


def test_compass_projection_fails_closed_on_unknown_or_invalid_coordinates(tmp_path):
    source = tmp_path / "unknown.csv.gz"
    rows = compass_rows()
    fields = [
        *COMPASS_DESIGN_FIELDS,
        "K562_dpsi_pooled",
        "K562_delta_logit_pooled",
        "mystery_measurement",
    ]
    with gzip.open(source, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**rows[0], "mystery_measurement": "secret"})
    with pytest.raises(RuntimeError, match="schema drift"):
        compass_projection_report(source, input_kind="combined_container")

    invalid = tmp_path / "invalid.csv.gz"
    bad = compass_rows()[0]
    bad["exon_end_hg38"] = "1118"
    write_compass(invalid, [bad])
    metadata = tmp_path / "metadata.tsv.gz"
    fasta = tmp_path / "reference.fasta.gz"
    write_compass_metadata(metadata)
    write_compass_fasta(fasta, [bad])
    with pytest.raises(ValueError, match="exon is not 1-based inclusive"):
        list(
            iter_compass_design_records(
                invalid,
                metadata_path=metadata,
                fasta_path=fasta,
                input_kind="combined_container",
            )
        )


def test_compass_projection_validates_metadata_fasta_and_reference_identity(tmp_path):
    source = tmp_path / "compass.csv.gz"
    metadata = tmp_path / "metadata.tsv.gz"
    fasta = tmp_path / "reference.fasta.gz"
    write_compass(source)
    write_compass_metadata(metadata)
    wrong = compass_rows()
    wrong[0]["full_seq"] = wrong[0]["full_seq"][:-1] + "A"
    write_compass_fasta(fasta, wrong)
    with pytest.raises(ValueError, match="disagrees with library FASTA insert"):
        list(
            iter_compass_design_records(
                source,
                metadata_path=metadata,
                fasta_path=fasta,
                input_kind="combined_container",
            )
        )


def test_compass_design_only_requires_exact_allowlist_and_custodian_hash(tmp_path):
    design = tmp_path / "compass_design.csv.gz"
    write_compass(design, include_outcomes=False)
    metadata = tmp_path / "metadata.tsv.gz"
    write_compass_metadata(metadata, include_outcomes=False)
    report = compass_projection_report(
        design,
        input_kind="design_only",
        metadata_columns=read_compass_metadata(metadata),
    )
    assert report["source_container_outcome_values_present"] is False
    assert report["complete_source_rows_parsed_into_temporary_csv_lists"] is False
    custodian = tmp_path / "custodian.json"
    write_custodian_manifest(custodian, design_path=design)
    manifest = json.loads(custodian.read_text())
    manifest["design_export_sha256"] = "0" * 64
    custodian.write_text(json.dumps(manifest))
    from gfmcd.splicing.external_validation import validate_compass_custodian_manifest

    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_compass_custodian_manifest(custodian, design_path=design)


def test_freeze_writes_hashed_immutable_truthful_design_pool(tmp_path):
    lock_path = tmp_path / "lock_a.json"
    lock_path.write_text(json.dumps(lock_a(tmp_path)))
    positions, sequences = vex_files(tmp_path)
    compass = tmp_path / "compass.csv.gz"
    write_compass(compass)
    metadata = tmp_path / "compass_metadata.tsv.gz"
    fasta = tmp_path / "compass_reference.fasta.gz"
    write_compass_metadata(metadata)
    write_compass_fasta(fasta)
    output = tmp_path / "design_pool"
    freeze_design_pool(
        lock_a_path=lock_path,
        vex_positions_path=positions,
        vex_sequences_path=sequences,
        compass_input_path=compass,
        compass_input_kind="combined_container",
        compass_metadata_path=metadata,
        compass_fasta_path=fasta,
        compass_custodian_manifest_path=None,
        output_dir=output,
        command=["design-pool-test"],
        repo=tmp_path,
        enforce_pinned_sources=False,
    )
    package = json.loads((output / "design_pool.json").read_text())
    manifest = json.loads((output / "design_pool.manifest.json").read_text())
    assert package["schema"] == DESIGN_POOL_SCHEMA
    assert package["status"] == "FROZEN_DESIGN_POOL_NOT_CANDIDATE_COHORT"
    assert package["candidate_selection_performed"] is False
    assert (
        package["compass_input_audit"][
            "complete_source_rows_parsed_into_temporary_csv_lists"
        ]
        is True
    )
    assert package["compass_input_audit"]["outcome_values_serialized"] is False
    assert manifest["design_pool_id"] == package["design_pool_id"]
    for artifact in ("vex_design_pool", "compass_design_pool", "design_pool"):
        record = manifest["outputs"][artifact]
        assert sha256_file(output / record["path"]) == record["sha256"]
    all_output = "".join(path.read_text() for path in output.glob("*.json*"))
    assert "DO_NOT_SERIALIZE" not in all_output
    assert any(
        column.endswith("delta_logit_pooled")
        for column in package["compass_input_audit"]["compass_stripped_outcome_columns"]
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        freeze_design_pool(
            lock_a_path=lock_path,
            vex_positions_path=positions,
            vex_sequences_path=sequences,
            compass_input_path=compass,
            compass_input_kind="combined_container",
            compass_metadata_path=metadata,
            compass_fasta_path=fasta,
            compass_custodian_manifest_path=None,
            output_dir=output,
            enforce_pinned_sources=False,
        )


def test_freeze_design_pool_supports_custodian_design_only_input(tmp_path):
    lock_path = tmp_path / "lock_a.json"
    lock_path.write_text(json.dumps(lock_a(tmp_path)))
    positions, sequences = vex_files(tmp_path)
    compass = tmp_path / "compass_design.csv.gz"
    metadata = tmp_path / "compass_metadata.tsv.gz"
    fasta = tmp_path / "compass_reference.fasta.gz"
    custodian = tmp_path / "custodian.json"
    write_compass(compass, include_outcomes=False)
    write_compass_metadata(metadata, include_outcomes=False)
    write_compass_fasta(fasta)
    write_custodian_manifest(custodian, design_path=compass)
    output = tmp_path / "design_pool"
    freeze_design_pool(
        lock_a_path=lock_path,
        vex_positions_path=positions,
        vex_sequences_path=sequences,
        compass_input_path=compass,
        compass_input_kind="design_only",
        compass_metadata_path=metadata,
        compass_fasta_path=fasta,
        compass_custodian_manifest_path=custodian,
        output_dir=output,
        enforce_pinned_sources=False,
    )
    package = json.loads((output / "design_pool.json").read_text())
    assert package["compass_input_audit"]["custodian_manifest_validated"] is True
    assert (
        package["compass_input_audit"][
            "complete_source_rows_parsed_into_temporary_csv_lists"
        ]
        is False
    )


def test_freeze_rejects_unpinned_sources_by_default(tmp_path):
    lock_path = tmp_path / "lock_a.json"
    lock_path.write_text(json.dumps(lock_a(tmp_path)))
    positions, sequences = vex_files(tmp_path)
    compass = tmp_path / "not-opened.csv.gz"
    metadata = tmp_path / "not-opened-metadata.tsv.gz"
    fasta = tmp_path / "not-opened.fasta.gz"
    write_compass(compass)
    write_compass_metadata(metadata)
    write_compass_fasta(fasta)
    with pytest.raises(RuntimeError, match="pinned upstream identities"):
        freeze_design_pool(
            lock_a_path=lock_path,
            vex_positions_path=positions,
            vex_sequences_path=sequences,
            compass_input_path=compass,
            compass_input_kind="combined_container",
            compass_metadata_path=metadata,
            compass_fasta_path=fasta,
            compass_custodian_manifest_path=None,
            output_dir=tmp_path / "design_pool",
        )


def test_freeze_candidate_cohort_requires_complete_scores_and_matches_without_outcomes(
    tmp_path,
):
    lock_path, design_pool_path = create_design_pool(tmp_path)
    scores = tmp_path / "feature_scores.jsonl"
    score_manifest = tmp_path / "feature_scores.manifest.json"
    write_feature_scores(scores, design_pool_path=design_pool_path)
    write_feature_score_manifest(
        score_manifest,
        scores_path=scores,
        lock_path=lock_path,
        design_pool_path=design_pool_path,
    )
    output = tmp_path / "candidate_cohort"
    freeze_candidate_cohort(
        lock_a_path=lock_path,
        design_pool_path=design_pool_path,
        feature_scores_path=scores,
        feature_score_manifest_path=score_manifest,
        output_dir=output,
    )
    package = json.loads((output / "candidate_cohort.json").read_text())
    manifest = json.loads((output / "candidate_cohort.manifest.json").read_text())
    assert package["status"] == "FROZEN_OUTCOME_BLIND_CARRIER_MATCHED_COHORT"
    assert package["external_assay_outcomes_inspected_or_used"] is False
    assert package["match_report"]["feature-17:acceptor"]["matched_pairs"] == 2
    assert package["match_report"]["feature-17:donor"]["matched_pairs"] == 2
    assert manifest["candidate_cohort_id"] == package["candidate_cohort_id"]
    serialized = "".join(path.read_text() for path in output.glob("*.json*"))
    assert "DO_NOT_SERIALIZE" not in serialized
    assert "delta_psi" not in serialized.lower()


def test_candidate_cohort_rejects_incomplete_model_score_family(tmp_path):
    lock_path, design_pool_path = create_design_pool(tmp_path)
    scores = tmp_path / "incomplete_scores.jsonl"
    score_manifest = tmp_path / "incomplete_scores.manifest.json"
    write_feature_scores(scores, design_pool_path=design_pool_path, omit_last=True)
    write_feature_score_manifest(
        score_manifest,
        scores_path=scores,
        lock_path=lock_path,
        design_pool_path=design_pool_path,
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        freeze_candidate_cohort(
            lock_a_path=lock_path,
            design_pool_path=design_pool_path,
            feature_scores_path=scores,
            feature_score_manifest_path=score_manifest,
            output_dir=tmp_path / "candidate_cohort",
        )


def test_git_blob_hash_and_download_allowlist_are_outcome_free(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"abc\n")
    expected = hashlib.sha1(b"blob 4\0abc\n").hexdigest()
    assert git_blob_sha1(path) == expected
    serialized_specs = json.dumps(
        [{"name": spec.name, "url": spec.url} for spec in DESIGN_DOWNLOADS]
    ).lower()
    assert "delta_psi" not in serialized_specs
    assert "processed_psis" not in serialized_specs
