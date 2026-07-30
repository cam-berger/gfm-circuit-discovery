"""Outcome-blind design-pool construction for Stage-4 biological validation.

This module implements the firewall between three distinct artifacts:

``Lock A``
    The internally derived candidate features, causal/native-unit evidence, expected directions,
    endpoints, carrier thresholds, controls, matching, inference, and multiplicity rules.  It
    must exist before external assay designs are inspected.

``Design pool``
    Immutable assay sequences and coordinates that are merely eligible for later scoring.  This
    is not a candidate cohort.  Prefer a custodian-produced design-only
    COMPASS export.  If the combined COMPASS container is used, :mod:`csv` necessarily parses
    complete source rows into temporary lists; only allowlisted indices are projected and no
    outcome value is used or serialized.

``Candidate cohort``
    A later outcome-blind carrier/matched-noncarrier freeze built from the design pool and an
    independently produced model-feature score artifact.  The scaffold here validates and
    matches those scores but never runs a model or reads reporter outcomes.

The module has no model dependency.  It deliberately has no downloader for Vex-seq delta-PSI or
the COMPASS processed outcome container.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Iterator, Mapping, TextIO
import urllib.request

from gfmcd.provenance import file_record, git_commit, git_dirty, sha256_file, sha256_text


LOCK_A_SCHEMA = "gfmcd.external_validation.lock_a.v2"
DESIGN_POOL_SCHEMA = "gfmcd.external_validation.design_pool.v1"
DESIGN_RECORD_SCHEMA = "gfmcd.external_validation.design_record.v1"
FEATURE_SCORE_SCHEMA = "gfmcd.external_validation.feature_score.v1"
CANDIDATE_COHORT_SCHEMA = "gfmcd.external_validation.candidate_cohort.v1"
COMPASS_CUSTODIAN_SCHEMA = "gfmcd.external_validation.compass_design_export.v1"

FROZEN_WINDOWS = {
    "donor": {"start": -80, "end": -48},
    "acceptor": {"start": 16, "end": 48},
}

VEX_COMMIT = "36d9c4a6e5e74def46b3cada69d3166f08c54bd5"
COMPASS_ACCESSION = "GSE307247"
VEX_POSITION_BLOB_SHA1 = "a8b4bd8a41c8139a4ac2c58db8ed310429209a45"
VEX_SEQUENCE_BLOB_SHA1 = "406e7a972f3b98bea3711c916d08f83362bf791e"
VEX_POSITION_SHA256 = "9b165057203da5dd6c4d7191c2eb91fb6b6d798c71f2ba0212789b283441b4c3"
VEX_SEQUENCE_SHA256 = "1eec399464a1ed35ec745033df8164d6f9011f328b8cee0b5a1074777bd87d2f"
COMPASS_METADATA_FILENAME = "GSE307247_column_descriptions_metadata.tsv.gz"
COMPASS_METADATA_SHA256 = "4a9ee91dbab3852df4acc1c8ffeeb69f0410474af71451a47c07e261f4a4987d"
COMPASS_FASTA_FILENAME = (
    "GSE307247_ESL_concat_2023_09_19_subsampleparams_d1c_ms75_"
    "shorter3p_iterate_mincov5_reference.fasta.gz"
)
COMPASS_FASTA_SHA256 = "9aa80696e83c17b2e72c5a5a0d41522a55eb9c0be7b86bb16b6ba0c4d7bc4b1e"
COMPASS_COMBINED_FILENAME = "GSE307247_Processed_PSIs_All_Cells.csv.gz"
COMPASS_CODE_COMMIT = "67791aa433bf66ac0b7b095ede01d650c5de9cb1"

# From the pinned COMPASS reference-FASTA assembly script.  Each 1,115-nt library record is
# 286-nt shared 5' sequence + 161-nt assay insert + 648-nt shared 3' sequence + 20-nt barcode.
COMPASS_LIBRARY_PREFIX = (
    "CTTCAAGATCCGCCACAACATCGAGGTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTG"
    "TAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAA"
    "AAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATA"
    "TTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCT"
    "ATCTATATATAGCTATCTATATCTA"
)
COMPASS_LIBRARY_SUFFIX = (
    "AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAA"
    "AAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCC"
    "AAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTG"
    "TGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTA"
    "GGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTC"
    "TGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAAT"
    "TCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAGGACGGCAGCGTGCAGCTCGCCGACCACTAC"
    "CAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAG"
    "TCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGC"
    "CGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGGACTGATAGTAAGGCCCATTACCTGC"
)
COMPASS_INSERT_START = len(COMPASS_LIBRARY_PREFIX)
COMPASS_INSERT_LENGTH = 161
COMPASS_BARCODE_LENGTH = 20
COMPASS_LIBRARY_LENGTH = (
    len(COMPASS_LIBRARY_PREFIX)
    + COMPASS_INSERT_LENGTH
    + len(COMPASS_LIBRARY_SUFFIX)
    + COMPASS_BARCODE_LENGTH
)


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    filename: str
    url: str
    expected_git_blob_sha1: str | None = None
    expected_sha256: str | None = None
    accession: str | None = None


# This allowlist intentionally contains design sources only.  In particular, it contains neither
# Vex-seq delta_PSI_values.tsv nor the COMPASS Processed_PSIs_All_Cells.csv.gz table.
DESIGN_DOWNLOADS = (
    DownloadSpec(
        name="vex_positions",
        filename="Vex-seq_positions.tsv",
        url=(
            "https://raw.githubusercontent.com/scottiadamson/Vex-seq/"
            f"{VEX_COMMIT}/processed_files/Vex-seq_positions.tsv"
        ),
        expected_git_blob_sha1=VEX_POSITION_BLOB_SHA1,
        expected_sha256=VEX_POSITION_SHA256,
    ),
    DownloadSpec(
        name="vex_sequences",
        filename="68_97_final_sequences_fixed.tsv",
        url=(
            "https://raw.githubusercontent.com/scottiadamson/Vex-seq/"
            f"{VEX_COMMIT}/preprocess_data/68_97_final_sequences_fixed.tsv"
        ),
        expected_git_blob_sha1=VEX_SEQUENCE_BLOB_SHA1,
        expected_sha256=VEX_SEQUENCE_SHA256,
    ),
    DownloadSpec(
        name="compass_column_metadata",
        filename=COMPASS_METADATA_FILENAME,
        url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE307nnn/GSE307247/suppl/"
            "GSE307247_column_descriptions_metadata.tsv.gz"
        ),
        expected_sha256=COMPASS_METADATA_SHA256,
        accession=COMPASS_ACCESSION,
    ),
    DownloadSpec(
        name="compass_reference_fasta",
        filename=COMPASS_FASTA_FILENAME,
        url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE307nnn/GSE307247/suppl/"
            "GSE307247_ESL_concat_2023_09_19_subsampleparams_d1c_ms75_"
            "shorter3p_iterate_mincov5_reference.fasta.gz"
        ),
        expected_sha256=COMPASS_FASTA_SHA256,
        accession=COMPASS_ACCESSION,
    ),
)

VEX_POSITION_FIELDS = ("variant", "3SS", "5SS")
VEX_SEQUENCE_FIELDS = (
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
)
COMPASS_DESIGN_FIELDS = (
    "Reference",
    "event_id",
    "gene_exon",
    "snp",
    "source",
    "seq_type",
    "intron1",
    "exon",
    "intron2",
    "full_seq",
    "transcript_id",
    "exon_start_hg38",
    "exon_end_hg38",
    "variant_hg38",
)

_DNA = frozenset("ACGTN")
_SNV = re.compile(r"^(chr[^:]+):(\d+):([ACGT])>([ACGT])$", re.IGNORECASE)
_EVENT = re.compile(r"^(chr[^:]+):(\d+)-(\d+):([+-])$")
_COMPASS_FASTA_HEADER = re.compile(
    r"^(?P<reference>\d+)(?:_(?P<barcode_index>\d+))? "
    r"Event ID:(?P<event>[^|]+)\|Exon ID:(?P<exon_id>[^|]*)"
    r"\|Gene Name:(?P<gene>[^|]+)\|SNP:(?P<snp>.+);(?P<seq_type>[^;]+)$"
)
_OUTCOME_COLUMN_MARKERS = (
    "included",
    "excluded",
    "total_pooled",
    "psi",
    "logit",
    "_sd_",
    "read_count",
    "readcount",
)
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_NATIVE_KINDS = {"head", "mlp_feature", "direction", "subspace", "residual"}
_ALLOWED_MATCH_EXACT_FIELDS = {
    "dataset",
    "window_role",
    "strand",
    "reference_genomic",
    "alternate_genomic",
    "sequence_type",
}
_ALLOWED_MATCH_NUMERIC_FIELDS = {
    "relative_offset",
    "reference_feature_score",
    "model_margin_delta",
    "sequence_gc",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def git_blob_sha1(path: str | Path) -> str:
    """Return Git's content hash (``sha1(b"blob <n>\\0" + data)``) for *path*."""
    data = Path(path).read_bytes()
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data).hexdigest()  # noqa: S324 - Git object identity


def _same_windows(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        normalized = {
            role: {"start": int(value[role]["start"]), "end": int(value[role]["end"])}
            for role in FROZEN_WINDOWS
        }
    except (KeyError, TypeError, ValueError):
        return False
    return normalized == FROZEN_WINDOWS and set(value) == set(FROZEN_WINDOWS)


def _required_mapping(parent: Mapping[str, object], key: str, *, context: str) -> Mapping:
    value = parent.get(key)
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{context} must contain a non-empty {key} object")
    return value


def _required_string(parent: Mapping[str, object], key: str, *, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must contain a non-empty {key}")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _finite_number(value: object, *, context: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _validate_artifact_bindings(
    artifacts: object,
    *,
    base_dir: Path | None,
    verify_artifacts: bool,
) -> dict[str, Mapping]:
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Lock A must contain a non-empty internal_artifacts list")
    by_id: dict[str, Mapping] = {}
    for index, artifact in enumerate(artifacts):
        context = f"internal_artifacts[{index}]"
        if not isinstance(artifact, Mapping):
            raise ValueError(f"{context} must be an object")
        artifact_id = _required_string(artifact, "artifact_id", context=context)
        _required_string(artifact, "role", context=context)
        path_text = _required_string(artifact, "path", context=context)
        expected = _require_sha256(artifact.get("sha256"), context=f"{context}.sha256")
        _required_string(artifact, "schema", context=context)
        if artifact_id in by_id:
            raise ValueError(f"duplicate internal artifact_id {artifact_id!r}")
        by_id[artifact_id] = artifact
        if verify_artifacts:
            path = Path(path_text)
            if not path.is_absolute():
                if base_dir is None:
                    raise ValueError("base_dir is required to verify relative artifact paths")
                path = base_dir / path
            if not path.is_file():
                raise FileNotFoundError(f"Lock A internal artifact missing: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"Lock A internal artifact hash mismatch for {artifact_id}: "
                    f"expected {expected}, found {actual}"
                )
    return by_id


def _validate_matching(matching: Mapping[str, object]) -> None:
    ratio = matching.get("controls_per_carrier")
    if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 1:
        raise ValueError("control_design.matching.controls_per_carrier must be a positive integer")
    if matching.get("without_replacement") is not True:
        raise ValueError("control matching must freeze without_replacement=true")
    if matching.get("tie_breaker") != "record_id":
        raise ValueError("control matching tie_breaker must be record_id")
    exact = matching.get("exact_fields")
    if not isinstance(exact, list) or not exact:
        raise ValueError("control matching must freeze non-empty exact_fields")
    if set(exact) - _ALLOWED_MATCH_EXACT_FIELDS:
        raise ValueError("control matching contains unsupported exact_fields")
    calipers = matching.get("numeric_calipers")
    if not isinstance(calipers, Mapping) or not calipers:
        raise ValueError("control matching must freeze numeric_calipers")
    if set(calipers) - _ALLOWED_MATCH_NUMERIC_FIELDS:
        raise ValueError("control matching contains unsupported numeric calipers")
    for field, value in calipers.items():
        _finite_number(value, context=f"numeric_calipers.{field}", minimum=0.0)
    distance = matching.get("distance_fields")
    if not isinstance(distance, list) or not distance or not set(distance) <= set(calipers):
        raise ValueError("distance_fields must be a non-empty subset of numeric_calipers")


def validate_lock_a(
    lock_a: Mapping[str, object],
    *,
    base_dir: str | Path | None = None,
    verify_artifacts: bool = False,
) -> None:
    """Fail closed unless *lock_a* is a substantive outcome-blind candidate contract."""
    if lock_a.get("schema") != LOCK_A_SCHEMA:
        raise ValueError(f"Lock A must use schema {LOCK_A_SCHEMA!r}")
    if lock_a.get("status") != "FROZEN_LOCK_A":
        raise ValueError("Lock A status must be FROZEN_LOCK_A")
    if lock_a.get("external_assay_outcomes_inspected_or_used") is not False:
        raise RuntimeError(
            "Lock A must state external_assay_outcomes_inspected_or_used=false"
        )
    if not _same_windows(lock_a.get("windows")):
        raise ValueError(f"Lock A windows must equal the frozen windows {FROZEN_WINDOWS}")

    artifact_by_id = _validate_artifact_bindings(
        lock_a.get("internal_artifacts"),
        base_dir=Path(base_dir) if base_dir is not None else None,
        verify_artifacts=verify_artifacts,
    )
    model = _required_mapping(lock_a, "model", context="Lock A")
    for field in ("checkpoint_id", "tokenizer_id", "native_margin_definition"):
        _required_string(model, field, context="model")
    _require_sha256(model.get("checkpoint_sha256"), context="model.checkpoint_sha256")
    _require_sha256(model.get("tokenizer_sha256"), context="model.tokenizer_sha256")
    class_labels = model.get("class_labels")
    if (
        not isinstance(class_labels, Mapping)
        or set(class_labels) != {"acceptor", "donor", "neither"}
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in class_labels.values()
        )
    ):
        raise ValueError("model.class_labels must bind integer acceptor/donor/neither labels")

    native_units = lock_a.get("native_units")
    if not isinstance(native_units, list) or not native_units:
        raise ValueError("Lock A must contain a non-empty native_units list")
    native_by_id: dict[str, Mapping] = {}
    for index, unit in enumerate(native_units):
        context = f"native_units[{index}]"
        if not isinstance(unit, Mapping):
            raise ValueError(f"{context} must be an object")
        unit_id = _required_string(unit, "native_unit_id", context=context)
        if unit_id in native_by_id:
            raise ValueError(f"duplicate native_unit_id {unit_id!r}")
        native_by_id[unit_id] = unit
        if unit.get("kind") not in _ALLOWED_NATIVE_KINDS:
            raise ValueError(f"{context}.kind is unsupported")
        layer = unit.get("layer")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError(f"{context}.layer must be a non-negative integer")
        for field in ("hook_location", "position_scope", "aggregation", "score_definition"):
            _required_string(unit, field, context=context)
        artifact_ids = unit.get("artifact_ids")
        if not isinstance(artifact_ids, list) or not artifact_ids:
            raise ValueError(f"{context}.artifact_ids must be non-empty")
        if set(artifact_ids) - set(artifact_by_id):
            raise ValueError(f"{context} references unknown internal artifacts")

    candidates = lock_a.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Lock A must contain a non-empty candidates list")
    ids: list[str] = []
    for index, candidate in enumerate(candidates):
        context = f"candidates[{index}]"
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{context} is not an object")
        candidate_id = _required_string(candidate, "candidate_id", context=context)
        ids.append(candidate_id)
        for field in ("hypothesis", "feature_definition", "evidence_label"):
            _required_string(candidate, field, context=context)
        unit_ids = candidate.get("native_unit_ids")
        artifact_ids = candidate.get("artifact_ids")
        if not isinstance(unit_ids, list) or not unit_ids or set(unit_ids) - set(native_by_id):
            raise ValueError(f"{context}.native_unit_ids are missing or unresolved")
        if (
            not isinstance(artifact_ids, list)
            or not artifact_ids
            or set(artifact_ids) - set(artifact_by_id)
        ):
            raise ValueError(f"{context}.artifact_ids are missing or unresolved")
        roles = candidate.get("window_roles")
        if not isinstance(roles, list) or not roles or set(roles) - set(FROZEN_WINDOWS):
            raise ValueError(f"{context}.window_roles are invalid")
        for sign_field in (
            "expected_feature_delta_sign",
            "expected_model_effect_sign",
        ):
            if candidate.get(sign_field) not in {-1, 1}:
                raise ValueError(f"{context}.{sign_field} must be -1 or +1")
        reporter_signs = candidate.get("expected_reporter_effect_sign_by_dataset")
        if (
            not isinstance(reporter_signs, Mapping)
            or set(reporter_signs) != {"vex_seq", "compass"}
            or any(value not in {-1, 1} for value in reporter_signs.values())
        ):
            raise ValueError(f"{context} must freeze Vex and COMPASS reporter signs")
        thresholds = _required_mapping(candidate, "carrier_thresholds", context=context)
        threshold_values = {
            field: _finite_number(
                thresholds.get(field),
                context=f"{context}.carrier_thresholds.{field}",
                minimum=0.0,
            )
            for field in (
                "reference_score_min",
                "signed_feature_delta_min",
                "carrier_model_margin_abs_min",
                "noncarrier_abs_feature_delta_max",
                "noncarrier_model_margin_abs_max",
            )
        }
        if (
            threshold_values["noncarrier_abs_feature_delta_max"]
            >= threshold_values["signed_feature_delta_min"]
        ):
            raise ValueError(f"{context} carrier and noncarrier feature thresholds overlap")
    if len(ids) != len(set(ids)):
        raise ValueError("Lock A candidate_id values must be unique")

    model_endpoint = _required_mapping(lock_a, "model_endpoint", context="Lock A")
    if (
        model_endpoint.get("name") != "alt_minus_ref_class_margin"
        or model_endpoint.get("contrast") != "alternate_minus_reference"
        or set(model_endpoint.get("class_by_window", {})) != {"acceptor", "donor"}
    ):
        raise ValueError("model_endpoint must freeze the alt-ref class margin by window")
    _required_string(model_endpoint, "units", context="model_endpoint")

    reporters = _required_mapping(lock_a, "reporter_endpoints", context="Lock A")
    if set(reporters) != {"vex_seq", "compass"}:
        raise ValueError("reporter_endpoints must freeze exactly Vex-seq and COMPASS")
    for dataset, endpoint in reporters.items():
        if not isinstance(endpoint, Mapping):
            raise ValueError(f"reporter_endpoints.{dataset} must be an object")
        for field in ("name", "contrast", "units", "cell_aggregation", "missingness_rule"):
            _required_string(endpoint, field, context=f"reporter_endpoints.{dataset}")
        if endpoint.get("contrast") != "alternate_minus_reference":
            raise ValueError(f"reporter_endpoints.{dataset}.contrast must be alt-ref")

    control_design = _required_mapping(lock_a, "control_design", context="Lock A")
    for control_kind in ("positive_controls", "negative_controls"):
        controls = control_design.get(control_kind)
        if not isinstance(controls, list) or not controls:
            raise ValueError(f"control_design.{control_kind} must be non-empty")
        for index, control in enumerate(controls):
            if not isinstance(control, Mapping):
                raise ValueError(f"{control_kind}[{index}] must be an object")
            for field in ("control_id", "definition", "pass_condition"):
                _required_string(control, field, context=f"{control_kind}[{index}]")
    _validate_matching(_required_mapping(control_design, "matching", context="control_design"))

    resampling = _required_mapping(lock_a, "grouping_resampling", context="Lock A")
    _required_string(resampling, "inference_unit", context="grouping_resampling")
    groups = resampling.get("group_by_dataset")
    if (
        not isinstance(groups, Mapping)
        or set(groups) != {"vex_seq", "compass"}
        or any(not isinstance(value, str) or not value for value in groups.values())
    ):
        raise ValueError("grouping_resampling.group_by_dataset must freeze both datasets")
    for method_name, expected_method in (
        ("bootstrap", "cluster_bootstrap"),
        ("permutation", "within_group_label_shuffle"),
    ):
        method = _required_mapping(resampling, method_name, context="grouping_resampling")
        if method.get("method") != expected_method:
            raise ValueError(f"grouping_resampling.{method_name}.method is not frozen")
        draws = method.get("draws")
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1_000:
            raise ValueError(f"grouping_resampling.{method_name}.draws must be >=1000")
        if isinstance(method.get("seed"), bool) or not isinstance(method.get("seed"), int):
            raise ValueError(f"grouping_resampling.{method_name}.seed must be integer")
    if resampling["permutation"].get("add_one_pvalue") is not True:
        raise ValueError("permutation must freeze exact add-one p-values")

    effects = _required_mapping(lock_a, "effect_thresholds", context="Lock A")
    for field in (
        "feature_signed_delta_min",
        "model_margin_abs_min",
        "vex_delta_psi_abs_min",
        "compass_delta_logit_abs_min",
    ):
        _finite_number(effects.get(field), context=f"effect_thresholds.{field}", minimum=0.0)

    multiplicity = _required_mapping(lock_a, "multiplicity", context="Lock A")
    if multiplicity.get("method") != "benjamini_hochberg":
        raise ValueError("multiplicity.method must be benjamini_hochberg")
    alpha = _finite_number(multiplicity.get("alpha"), context="multiplicity.alpha")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("multiplicity.alpha must be in (0,1]")
    dimensions = multiplicity.get("family_dimensions")
    required_dimensions = {"candidate", "window", "dataset", "reporter_endpoint"}
    if not isinstance(dimensions, list) or set(dimensions) != required_dimensions:
        raise ValueError("multiplicity.family_dimensions must freeze the complete scan family")
    _required_string(lock_a, "decision_rule", context="Lock A")


def read_lock_a(path: str | Path) -> dict:
    lock_path = Path(path)
    value = json.loads(lock_path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Lock A must be a JSON object")
    validate_lock_a(value, base_dir=lock_path.parent, verify_artifacts=True)
    return value


def in_frozen_window(role: str, offset: int) -> bool:
    window = FROZEN_WINDOWS[role]
    return window["start"] <= offset < window["end"]


def window_labels(*, acceptor_offset: int, donor_offset: int) -> list[str]:
    return [
        role
        for role, offset in (("acceptor", acceptor_offset), ("donor", donor_offset))
        if in_frozen_window(role, offset)
    ]


@contextmanager
def open_text(path: str | Path) -> Iterator[TextIO]:
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8", newline="") as handle:
            yield handle
    else:
        with p.open("r", encoding="utf-8", newline="") as handle:
            yield handle


def table_header(path: str | Path, *, delimiter: str) -> list[str]:
    with open_text(path) as handle:
        header = next(csv.reader(handle, delimiter=delimiter), None)
    if not header:
        raise ValueError(f"empty table: {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"duplicate columns in {path}")
    return header


def _iter_projected_rows(
    path: str | Path,
    *,
    delimiter: str,
    fields: tuple[str, ...],
) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield only allowlisted fields; non-design values never enter a mapping."""
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, None)
        if not header:
            raise ValueError(f"empty table: {path}")
        if len(header) != len(set(header)):
            raise ValueError(f"duplicate columns in {path}")
        missing = sorted(set(fields) - set(header))
        if missing:
            raise ValueError(f"{path} is missing required design columns: {missing}")
        indices = tuple(header.index(field) for field in fields)
        for line_no, values in enumerate(reader, 2):
            if len(values) != len(header):
                raise ValueError(
                    f"{path}:{line_no}: expected {len(header)} columns, found {len(values)}"
                )
            yield line_no, {field: values[index] for field, index in zip(fields, indices)}


def looks_like_outcome_column(column: str) -> bool:
    lower = column.strip().lower()
    return any(marker in lower for marker in _OUTCOME_COLUMN_MARKERS)


def read_compass_metadata(path: str | Path) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for line_no, row in _iter_projected_rows(
        path,
        delimiter="\t",
        fields=("Column", "Definition"),
    ):
        column = row["Column"].strip()
        definition = row["Definition"].strip()
        if not column or not definition:
            raise ValueError(f"{path}:{line_no}: blank COMPASS metadata field")
        if column in definitions:
            raise ValueError(f"{path}:{line_no}: duplicate COMPASS metadata column {column}")
        definitions[column] = definition
    if set(COMPASS_DESIGN_FIELDS) - set(definitions):
        raise ValueError("COMPASS metadata does not define every required design field")
    return definitions


def _validate_compass_library_sequence(sequence: str, *, header: str) -> str:
    if len(sequence) != COMPASS_LIBRARY_LENGTH:
        raise ValueError(f"COMPASS FASTA record {header!r} is not 1,115 nt")
    if not set(sequence) <= _DNA:
        raise ValueError(f"COMPASS FASTA record {header!r} contains non-DNA characters")
    insert_end = COMPASS_INSERT_START + COMPASS_INSERT_LENGTH
    if sequence[:COMPASS_INSERT_START] != COMPASS_LIBRARY_PREFIX:
        raise ValueError(f"COMPASS FASTA record {header!r} has the wrong 5' backbone")
    if sequence[insert_end:-COMPASS_BARCODE_LENGTH] != COMPASS_LIBRARY_SUFFIX:
        raise ValueError(f"COMPASS FASTA record {header!r} has the wrong 3' backbone")
    return sequence[COMPASS_INSERT_START:insert_end]


def read_compass_fasta_index(path: str | Path) -> tuple[dict[str, dict], dict]:
    """Return a compact Reference-ID index after validating the complete library FASTA."""
    index: dict[str, dict] = {}
    header: str | None = None
    parts: list[str] = []
    fasta_records = 0

    def consume(record_header: str, sequence_parts: list[str]) -> None:
        nonlocal fasta_records
        match = _COMPASS_FASTA_HEADER.fullmatch(record_header)
        if match is None:
            raise ValueError(f"invalid COMPASS FASTA header {record_header!r}")
        sequence = "".join(sequence_parts).upper()
        insert = _validate_compass_library_sequence(sequence, header=record_header)
        fields = match.groupdict()
        event_parts = fields["event"].split(":")
        if len(event_parts) < 4 or event_parts[0] != "SE" or event_parts[-1] not in {"+", "-"}:
            raise ValueError(f"invalid COMPASS FASTA event descriptor {fields['event']!r}")
        identity = {
            "reference_id": fields["reference"],
            "event_descriptor": fields["event"],
            "event_chrom": f"chr{event_parts[1].removeprefix('chr')}",
            "event_strand": event_parts[-1],
            "exon_id": fields["exon_id"],
            "gene_name": fields["gene"],
            "snp_descriptor": fields["snp"],
            "sequence_type": fields["seq_type"],
            "insert_sequence": insert,
            "insert_sha256": sha256_text(insert),
            "barcode_count": 1,
        }
        previous = index.get(fields["reference"])
        if previous is None:
            index[fields["reference"]] = identity
        else:
            comparable = {key: value for key, value in identity.items() if key != "barcode_count"}
            prior = {key: value for key, value in previous.items() if key != "barcode_count"}
            if comparable != prior:
                raise ValueError(
                    f"COMPASS FASTA Reference {fields['reference']} has inconsistent identity"
                )
            previous["barcode_count"] += 1
        fasta_records += 1

    with open_text(path) as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    consume(header, parts)
                header = line[1:]
                parts = []
            else:
                if header is None:
                    raise ValueError(f"{path}:{line_no}: sequence before FASTA header")
                parts.append(line)
    if header is None:
        raise ValueError(f"empty COMPASS FASTA: {path}")
    consume(header, parts)
    return index, {
        "library_records": fasta_records,
        "unique_reference_ids": len(index),
        "library_length": COMPASS_LIBRARY_LENGTH,
        "insert_start_0based": COMPASS_INSERT_START,
        "insert_length": COMPASS_INSERT_LENGTH,
        "barcode_length": COMPASS_BARCODE_LENGTH,
        "assembly_code_commit": COMPASS_CODE_COMMIT,
    }


def validate_compass_custodian_manifest(
    path: str | Path,
    *,
    design_path: str | Path,
) -> dict:
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != COMPASS_CUSTODIAN_SCHEMA:
        raise ValueError(f"custodian manifest must use {COMPASS_CUSTODIAN_SCHEMA}")
    if manifest.get("source_accession") != COMPASS_ACCESSION:
        raise ValueError("custodian manifest has the wrong COMPASS accession")
    if manifest.get("source_filename") != COMPASS_COMBINED_FILENAME:
        raise ValueError("custodian manifest has the wrong source filename")
    _require_sha256(
        manifest.get("source_sha256"),
        context="custodian_manifest.source_sha256",
    )
    export_sha = _require_sha256(
        manifest.get("design_export_sha256"),
        context="custodian_manifest.design_export_sha256",
    )
    if sha256_file(design_path) != export_sha:
        raise RuntimeError("custodian design export hash mismatch")
    if manifest.get("retained_columns") != list(COMPASS_DESIGN_FIELDS):
        raise ValueError("custodian manifest retained_columns do not match the allowlist")
    if manifest.get("outcome_values_used_for_export") is not False:
        raise RuntimeError("custodian export must not use outcome values")
    _required_string(manifest, "custodian_id", context="custodian manifest")
    _required_string(manifest, "exporter_version", context="custodian manifest")
    return manifest


def compass_projection_report(
    path: str | Path,
    *,
    input_kind: str,
    metadata_columns: Iterable[str] | None = None,
) -> dict:
    if input_kind not in {"design_only", "combined_container"}:
        raise ValueError("COMPASS input_kind must be design_only or combined_container")
    header = table_header(path, delimiter=",")
    missing = sorted(set(COMPASS_DESIGN_FIELDS) - set(header))
    if missing:
        raise ValueError(f"COMPASS table is missing required design columns: {missing}")
    if metadata_columns is not None:
        undocumented = sorted(set(header) - set(metadata_columns))
        if undocumented:
            raise RuntimeError(f"COMPASS source columns absent from metadata: {undocumented}")
    stripped = [column for column in header if column not in COMPASS_DESIGN_FIELDS]
    unknown = [column for column in stripped if not looks_like_outcome_column(column)]
    if unknown:
        raise RuntimeError(
            "COMPASS schema drift: non-allowlisted columns are not recognized outcome fields: "
            f"{unknown}"
        )
    if input_kind == "design_only" and stripped:
        raise RuntimeError("design_only COMPASS input contains non-design outcome columns")
    if input_kind == "combined_container" and not stripped:
        raise RuntimeError("combined_container COMPASS input has no outcome-bearing columns")
    return {
        "input_kind": input_kind,
        "source_column_count": len(header),
        "retained_design_columns": list(COMPASS_DESIGN_FIELDS),
        "stripped_outcome_columns": stripped,
        "stripped_outcome_column_count": len(stripped),
        "unknown_non_design_columns": [],
        "source_container_outcome_values_present": bool(stripped),
        "complete_source_rows_parsed_into_temporary_csv_lists": (
            input_kind == "combined_container"
        ),
        "outcome_columns_projected_into_row_mappings": False,
        "outcome_values_used_for_filtering_matching_or_output": False,
        "outcome_values_serialized": False,
    }


def _normalize_dna(value: str, *, field: str, line_no: int) -> str:
    sequence = value.upper()
    if not sequence or not set(sequence) <= _DNA:
        raise ValueError(f"line {line_no}: {field} contains non-DNA characters")
    return sequence


def _oriented_allele(allele: str, strand: str) -> str:
    allele = allele.upper()
    return allele if strand == "+" else allele.translate(_COMPLEMENT)


def _base_record(
    *,
    dataset: str,
    record_id: str,
    assembly: str,
    chrom: str,
    position: int,
    strand: str,
    reference: str,
    alternate: str,
    variant_index: int,
    reference_sequence: str,
    alternate_sequence: str,
    acceptor_offset: int,
    donor_offset: int,
) -> dict:
    labels = window_labels(
        acceptor_offset=acceptor_offset,
        donor_offset=donor_offset,
    )
    if not labels:
        raise ValueError("attempted to construct an out-of-window design record")
    return {
        "schema": DESIGN_RECORD_SCHEMA,
        "dataset": dataset,
        "record_id": record_id,
        "assembly": assembly,
        "chrom": chrom,
        "genomic_position_1based": position,
        "strand": strand,
        "reference_genomic": reference,
        "alternate_genomic": alternate,
        "variant_index_transcript_0based": variant_index,
        "sequence_orientation": "transcript",
        "reference_sequence": reference_sequence,
        "alternate_sequence": alternate_sequence,
        "acceptor_offset": acceptor_offset,
        "donor_offset": donor_offset,
        "window_labels": labels,
    }


def _parse_vex_variant(value: str) -> tuple[str, int, str, str]:
    parts = value.rsplit("_", 3)
    if len(parts) != 4:
        raise ValueError(f"invalid Vex-seq variant identifier: {value!r}")
    chrom, position, reference, alternate = parts
    try:
        pos = int(position)
    except ValueError as exc:
        raise ValueError(f"invalid Vex-seq position in {value!r}") from exc
    reference = reference.upper()
    alternate = alternate.upper()
    if reference not in "ACGT" or alternate not in "ACGT":
        raise ValueError(f"not a single-nucleotide Vex-seq variant: {value!r}")
    return chrom, pos, reference, alternate


def read_vex_positions(path: str | Path) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for line_no, row in _iter_projected_rows(
        path,
        delimiter="\t",
        fields=VEX_POSITION_FIELDS,
    ):
        variant = row["variant"]
        try:
            chrom, position, reference, alternate = _parse_vex_variant(variant)
        except ValueError:
            # The external analysis is frozen to SNVs; indels remain outside this design pool.
            continue
        try:
            acceptor_offset = int(row["3SS"])
            donor_offset = int(row["5SS"])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: non-integer splice offset") from exc
        labels = window_labels(
            acceptor_offset=acceptor_offset,
            donor_offset=donor_offset,
        )
        if not labels:
            continue
        if variant in selected:
            raise ValueError(f"{path}:{line_no}: duplicate Vex-seq variant {variant}")
        selected[variant] = {
            "chrom": chrom,
            "position": position,
            "reference": reference,
            "alternate": alternate,
            "acceptor_offset": acceptor_offset,
            "donor_offset": donor_offset,
        }
    return selected


def iter_vex_design_records(
    positions_path: str | Path,
    sequences_path: str | Path,
    *,
    stats: dict | None = None,
) -> Iterator[dict]:
    """Join Vex design offsets to barcode-collapsed, transcript-oriented assay sequences."""
    selected = read_vex_positions(positions_path)
    joined: dict[str, dict] = {}
    rejected: dict[str, str] = {}
    observed_rows = 0
    for line_no, row in _iter_projected_rows(
        sequences_path,
        delimiter="\t",
        fields=VEX_SEQUENCE_FIELDS,
    ):
        reference = row["reference"].upper()
        alternate = row["alternative"].upper()
        if reference not in "ACGT" or alternate not in "ACGT":
            continue
        variant = "_".join(
            (row["chromosome"], row["position"], reference, alternate)
        )
        if variant not in selected:
            continue
        observed_rows += 1
        design = selected[variant]
        if (
            row["chromosome"] != design["chrom"]
            or int(row["position"]) != design["position"]
            or reference != design["reference"]
            or alternate != design["alternate"]
        ):
            raise ValueError(f"{sequences_path}:{line_no}: Vex join invariant failed")
        strand = row["strand"]
        if strand not in {"+", "-"}:
            raise ValueError(f"{sequences_path}:{line_no}: invalid strand {strand!r}")
        sequence = _normalize_dna(row["Sequence"], field="Sequence", line_no=line_no)
        span_start = int(row["Upstream Int Adj"])
        span_end = int(row["Down Int Adj"])
        if span_start > span_end or len(sequence) != span_end - span_start + 1:
            raise ValueError(
                f"{sequences_path}:{line_no}: Vex assay span is not 1-based inclusive"
            )
        position = design["position"]
        variant_index = (
            position - span_start if strand == "+" else span_end - position
        )
        if not 0 <= variant_index < len(sequence):
            raise ValueError(f"{sequences_path}:{line_no}: variant lies outside assay sequence")
        oriented_alt = _oriented_allele(alternate, strand)
        oriented_ref = _oriented_allele(reference, strand)
        if sequence[variant_index] != oriented_alt:
            raise ValueError(
                f"{sequences_path}:{line_no}: alternate allele does not match assay sequence"
            )
        reference_sequence = (
            sequence[:variant_index] + oriented_ref + sequence[variant_index + 1 :]
        )
        # Adamson et al.'s 3SS offset is relative to the last intronic acceptor base, while its
        # 5SS offset is relative to the first downstream intronic base.  Convert both to the
        # transcript-oriented target indices used by the present study and verify the intact WT
        # junctions rather than silently trusting either convention.
        acceptor_target_index = variant_index - design["acceptor_offset"] + 1
        donor_target_index = variant_index - design["donor_offset"]
        targets_in_bounds = (
            2 <= acceptor_target_index < donor_target_index
            and donor_target_index + 2 <= len(reference_sequence)
        )
        canonical_acceptor = (
            targets_in_bounds
            and reference_sequence[
                acceptor_target_index - 2 : acceptor_target_index
            ]
            == "AG"
        )
        canonical_donor = (
            targets_in_bounds
            and reference_sequence[donor_target_index : donor_target_index + 2]
            == "GT"
        )
        if not targets_in_bounds or not canonical_acceptor or not canonical_donor:
            reason = (
                "target_out_of_bounds"
                if not targets_in_bounds
                else "noncanonical_acceptor"
                if not canonical_acceptor
                else "noncanonical_donor"
            )
            previous_reason = rejected.setdefault(variant, reason)
            if previous_reason != reason:
                raise ValueError(
                    f"{sequences_path}:{line_no}: barcode rows disagree on junction validity"
                )
            continue
        record = _base_record(
            dataset="vex_seq",
            record_id=f"vex:{variant}",
            assembly="GRCh37.p13",
            chrom=design["chrom"],
            position=position,
            strand=strand,
            reference=reference,
            alternate=alternate,
            variant_index=variant_index,
            reference_sequence=reference_sequence,
            alternate_sequence=sequence,
            acceptor_offset=design["acceptor_offset"],
            donor_offset=design["donor_offset"],
        )
        record.update(
            {
                "source_variant_id": variant,
                "assay_span_start_1based_inclusive": span_start,
                "assay_span_end_1based_inclusive": span_end,
                "exon_upstream_coordinate_source": int(row["Exon up"]),
                "exon_downstream_coordinate_source": int(row["Exon down"]),
                "acceptor_target_index_transcript_0based": acceptor_target_index,
                "donor_target_index_transcript_0based": donor_target_index,
                "canonical_acceptor": True,
                "canonical_donor": True,
                "vex_offset_convention": {
                    "3SS": "variant relative to last intronic acceptor base",
                    "5SS": "variant relative to first downstream intronic base",
                },
            }
        )
        previous = joined.get(variant)
        if previous is not None and previous != record:
            raise ValueError(
                f"{sequences_path}:{line_no}: barcode rows disagree for {variant}"
            )
        joined[variant] = record
    missing = sorted(set(selected) - set(joined) - set(rejected))
    if missing:
        preview = missing[:5]
        raise RuntimeError(
            f"{len(missing)} selected Vex-seq variants lack assay sequences; first={preview}"
        )
    if stats is not None:
        stats.update(
            {
                "selected_position_variants": len(selected),
                "matching_barcode_rows": observed_rows,
                "barcode_collapsed_records": len(joined),
                "junction_rejected_records": len(rejected),
                "junction_rejection_reasons": dict(sorted(Counter(rejected.values()).items())),
            }
        )
    for variant in sorted(joined):
        yield joined[variant]


def _parse_event(value: str) -> tuple[str, int, int, str]:
    match = _EVENT.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid COMPASS event_id {value!r}")
    chrom, start, end, strand = match.groups()
    start_i, end_i = int(start), int(end)
    if start_i > end_i:
        raise ValueError(f"reversed COMPASS event interval {value!r}")
    return chrom, start_i, end_i, strand


def _normalize_fasta_snv_descriptor(value: str) -> str:
    parts = value.split(":")
    if len(parts) != 4:
        return value.lower()
    chrom, position, reference, alternate = parts
    return (
        f"chr{chrom.removeprefix('chr')}:{position}:"
        f"{reference.upper()}>{alternate.upper()}"
    )


def _validate_compass_reference_identity(
    row: Mapping[str, str],
    *,
    full_sequence: str,
    chrom: str,
    strand: str,
    fasta_index: Mapping[str, Mapping[str, object]],
    line_no: int,
) -> Mapping[str, object]:
    reference_id = row["Reference"]
    identity = fasta_index.get(reference_id)
    if identity is None:
        raise ValueError(f"line {line_no}: Reference {reference_id} absent from library FASTA")
    if identity["insert_sequence"] != full_sequence:
        raise ValueError(f"line {line_no}: full_seq disagrees with library FASTA insert")
    if identity["event_chrom"] != chrom or identity["event_strand"] != strand:
        raise ValueError(f"line {line_no}: event chromosome/strand disagrees with FASTA")
    gene_exon = row["gene_exon"].strip()
    if not (
        gene_exon == identity["gene_name"]
        or gene_exon.startswith(f"{identity['gene_name']} ")
    ):
        raise ValueError(f"line {line_no}: gene_exon disagrees with FASTA Gene Name")
    if identity["sequence_type"] != row["seq_type"]:
        raise ValueError(f"line {line_no}: seq_type disagrees with FASTA description")
    if _normalize_fasta_snv_descriptor(str(identity["snp_descriptor"])) != row[
        "variant_hg38"
    ]:
        raise ValueError(f"line {line_no}: variant_hg38 disagrees with FASTA description")
    return identity


def normalize_compass_row(
    row: Mapping[str, str],
    *,
    line_no: int,
    fasta_index: Mapping[str, Mapping[str, object]],
) -> tuple[dict | None, str]:
    """Normalize one projected COMPASS row, returning ``(record, exclusion_reason)``."""
    variant_text = row["variant_hg38"].strip()
    if variant_text.lower() == "none" or row["snp"].strip().lower() == "none":
        return None, "wild_type"
    match = _SNV.fullmatch(variant_text)
    if match is None or "double" in row["seq_type"].lower():
        return None, "not_single_snv"
    chrom, pos_text, reference, alternate = match.groups()
    position = int(pos_text)
    reference, alternate = reference.upper(), alternate.upper()
    event_chrom, event_start, event_end, strand = _parse_event(row["event_id"])
    if chrom != event_chrom:
        raise ValueError(f"line {line_no}: variant/event chromosome mismatch")
    exon_start = int(row["exon_start_hg38"])
    exon_end = int(row["exon_end_hg38"])
    if not (event_start <= exon_start <= exon_end <= event_end):
        raise ValueError(f"line {line_no}: exon lies outside event")
    intron1 = _normalize_dna(row["intron1"], field="intron1", line_no=line_no)
    exon = _normalize_dna(row["exon"], field="exon", line_no=line_no)
    intron2 = _normalize_dna(row["intron2"], field="intron2", line_no=line_no)
    full_sequence = _normalize_dna(row["full_seq"], field="full_seq", line_no=line_no)
    if full_sequence != intron1 + exon + intron2:
        raise ValueError(f"line {line_no}: full_seq != intron1 + exon + intron2")
    if len(full_sequence) != event_end - event_start + 1:
        raise ValueError(f"line {line_no}: COMPASS event is not 1-based inclusive")
    if len(full_sequence) != 161:
        raise ValueError(f"line {line_no}: COMPASS assay sequence is not 161 nt")
    if len(exon) != exon_end - exon_start + 1:
        raise ValueError(f"line {line_no}: COMPASS exon is not 1-based inclusive")
    expected_intron1 = (
        exon_start - event_start if strand == "+" else event_end - exon_end
    )
    expected_intron2 = (
        event_end - exon_end if strand == "+" else exon_start - event_start
    )
    if len(intron1) != expected_intron1 or len(intron2) != expected_intron2:
        raise ValueError(f"line {line_no}: sequence parts disagree with strand coordinates")
    if not event_start <= position <= event_end:
        raise ValueError(f"line {line_no}: variant lies outside event")
    variant_index = (
        position - event_start if strand == "+" else event_end - position
    )
    oriented_ref = _oriented_allele(reference, strand)
    oriented_alt = _oriented_allele(alternate, strand)
    if full_sequence[variant_index] != oriented_alt:
        raise ValueError(
            f"line {line_no}: alternate allele does not match transcript-oriented full_seq"
        )
    reference_sequence = (
        full_sequence[:variant_index]
        + oriented_ref
        + full_sequence[variant_index + 1 :]
    )
    library_identity = _validate_compass_reference_identity(
        row,
        full_sequence=full_sequence,
        chrom=chrom,
        strand=strand,
        fasta_index=fasta_index,
        line_no=line_no,
    )
    acceptor_index = len(intron1)
    donor_index = acceptor_index + len(exon)
    if reference_sequence[acceptor_index - 2 : acceptor_index] != "AG":
        raise ValueError(f"line {line_no}: non-canonical COMPASS acceptor")
    if reference_sequence[donor_index : donor_index + 2] != "GT":
        raise ValueError(f"line {line_no}: non-canonical COMPASS donor")
    if strand == "+":
        acceptor_offset = position - exon_start
        donor_offset = position - (exon_end + 1)
    else:
        acceptor_offset = exon_end - position
        donor_offset = (exon_start - 1) - position
    labels = window_labels(
        acceptor_offset=acceptor_offset,
        donor_offset=donor_offset,
    )
    if not labels:
        return None, "outside_frozen_windows"
    record = _base_record(
        dataset="compass",
        record_id=f"compass:{row['Reference']}",
        assembly="GRCh38",
        chrom=chrom,
        position=position,
        strand=strand,
        reference=reference,
        alternate=alternate,
        variant_index=variant_index,
        reference_sequence=reference_sequence,
        alternate_sequence=full_sequence,
        acceptor_offset=acceptor_offset,
        donor_offset=donor_offset,
    )
    record.update(
        {
            "source_reference_id": row["Reference"],
            "event_id": row["event_id"],
            "gene_exon": row["gene_exon"],
            "source_variant_id": row["snp"],
            "source_category": row["source"],
            "sequence_type": row["seq_type"],
            "transcript_id": row["transcript_id"],
            "library_insert_sha256": library_identity["insert_sha256"],
            "library_barcode_count": library_identity["barcode_count"],
            "library_event_descriptor": library_identity["event_descriptor"],
            "library_exon_id": library_identity["exon_id"],
            "event_start_hg38_1based_inclusive": event_start,
            "event_end_hg38_1based_inclusive": event_end,
            "exon_start_hg38_1based_inclusive": exon_start,
            "exon_end_hg38_1based_inclusive": exon_end,
            "event_start_bed_0based": event_start - 1,
            "event_end_bed_0based_exclusive": event_end,
            "exon_start_bed_0based": exon_start - 1,
            "exon_end_bed_0based_exclusive": exon_end,
        }
    )
    return record, "selected"


def iter_compass_design_records(
    path: str | Path,
    *,
    metadata_path: str | Path,
    fasta_path: str | Path,
    input_kind: str,
    stats: dict | None = None,
) -> Iterator[dict]:
    """Stream an allowlist projection of COMPASS and yield only frozen-window SNVs."""
    metadata = read_compass_metadata(metadata_path)
    fasta_index, fasta_stats = read_compass_fasta_index(fasta_path)
    projection = compass_projection_report(
        path,
        input_kind=input_kind,
        metadata_columns=metadata,
    )
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for line_no, row in _iter_projected_rows(
        path,
        delimiter=",",
        fields=COMPASS_DESIGN_FIELDS,
    ):
        counts["source_rows"] += 1
        record, reason = normalize_compass_row(
            row,
            line_no=line_no,
            fasta_index=fasta_index,
        )
        counts[reason] += 1
        if record is None:
            continue
        if record["record_id"] in seen_ids:
            raise ValueError(f"{path}:{line_no}: duplicate COMPASS Reference identifier")
        seen_ids.add(record["record_id"])
        yield record
    if stats is not None:
        stats.update(projection)
        stats["metadata_column_count"] = len(metadata)
        stats["fasta"] = fasta_stats
        stats["row_counts"] = dict(sorted(counts.items()))


def assert_design_record(record: Mapping[str, object]) -> None:
    if record.get("schema") != DESIGN_RECORD_SCHEMA:
        raise ValueError("wrong external-validation design-record schema")
    if record.get("dataset") not in {"vex_seq", "compass"}:
        raise ValueError("unknown external-validation dataset")
    labels = record.get("window_labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("design record has no frozen-window label")
    if set(labels) - set(FROZEN_WINDOWS):
        raise ValueError("design record has an unknown window label")
    for key in record:
        if looks_like_outcome_column(key):
            raise RuntimeError(f"outcome-like field escaped into design record: {key}")
    reference_sequence = record.get("reference_sequence")
    alternate_sequence = record.get("alternate_sequence")
    index = record.get("variant_index_transcript_0based")
    if (
        not isinstance(reference_sequence, str)
        or not isinstance(alternate_sequence, str)
        or len(reference_sequence) != len(alternate_sequence)
        or not isinstance(index, int)
        or not 0 <= index < len(reference_sequence)
    ):
        raise ValueError("invalid paired reference/alternate sequences")
    differences = [
        i
        for i, (reference, alternate) in enumerate(
            zip(reference_sequence, alternate_sequence)
        )
        if reference != alternate
    ]
    if differences != [index]:
        raise ValueError("reference and alternate assay sequences must differ at exactly the SNV")


def _write_jsonl(records: Iterable[dict], path: Path) -> dict:
    counts: Counter[str] = Counter()
    record_ids: set[str] = set()
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            assert_design_record(record)
            record_id = str(record["record_id"])
            if record_id in record_ids:
                raise ValueError(f"duplicate normalized record_id {record_id}")
            record_ids.add(record_id)
            handle.write(canonical_json(record) + "\n")
            counts["records"] += 1
            counts[f"strand:{record['strand']}"] += 1
            for label in record["window_labels"]:
                counts[f"window:{label}"] += 1
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "counts": dict(sorted(counts.items())),
    }


@contextmanager
def _atomic_new_directory(path: str | Path) -> Iterator[Path]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        yield temporary
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _source_record(path: str | Path, **upstream: object) -> dict:
    record = file_record(path)
    if record.get("missing"):
        raise FileNotFoundError(path)
    record.update(upstream)
    return record


def _artifact_record(path: Path) -> dict:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _code_records(command: Iterable[str]) -> tuple[list[str], dict[str, dict]]:
    command_list = list(command)
    paths = {Path(__file__).resolve()}
    for item in command_list:
        candidate = Path(item)
        if candidate.suffix == ".py" and candidate.is_file():
            paths.add(candidate.resolve())
    return command_list, {
        str(path): file_record(path)
        for path in sorted(paths, key=str)
    }


def freeze_design_pool(
    *,
    lock_a_path: str | Path,
    vex_positions_path: str | Path,
    vex_sequences_path: str | Path,
    compass_input_path: str | Path,
    compass_input_kind: str,
    compass_metadata_path: str | Path,
    compass_fasta_path: str | Path,
    compass_custodian_manifest_path: str | Path | None,
    output_dir: str | Path,
    command: Iterable[str] = (),
    repo: str | Path | None = None,
    enforce_pinned_sources: bool = True,
) -> Path:
    """Create an immutable design pool; this does not select carriers or controls."""
    lock_a = read_lock_a(lock_a_path)
    command_list, code = _code_records(command)
    vex_position_blob = git_blob_sha1(vex_positions_path)
    vex_sequence_blob = git_blob_sha1(vex_sequences_path)
    source_checks = {
        "vex_positions_git_blob": (vex_position_blob, VEX_POSITION_BLOB_SHA1),
        "vex_sequences_git_blob": (vex_sequence_blob, VEX_SEQUENCE_BLOB_SHA1),
        "vex_positions_sha256": (sha256_file(vex_positions_path), VEX_POSITION_SHA256),
        "vex_sequences_sha256": (sha256_file(vex_sequences_path), VEX_SEQUENCE_SHA256),
        "compass_metadata_sha256": (
            sha256_file(compass_metadata_path),
            COMPASS_METADATA_SHA256,
        ),
        "compass_fasta_sha256": (
            sha256_file(compass_fasta_path),
            COMPASS_FASTA_SHA256,
        ),
    }
    if enforce_pinned_sources:
        mismatches = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in source_checks.items()
            if actual != expected
        }
        if mismatches:
            raise RuntimeError(
                f"design sources do not match pinned upstream identities: {mismatches}"
            )
    if compass_input_kind == "design_only":
        if compass_custodian_manifest_path is None:
            raise ValueError("design_only COMPASS input requires a custodian manifest")
        custodian = validate_compass_custodian_manifest(
            compass_custodian_manifest_path,
            design_path=compass_input_path,
        )
    elif compass_input_kind == "combined_container":
        if compass_custodian_manifest_path is not None:
            raise ValueError("combined_container must not be paired with a custodian manifest")
        custodian = None
    else:
        raise ValueError("compass_input_kind must be design_only or combined_container")
    inputs = {
        "lock_a": _source_record(lock_a_path, schema=LOCK_A_SCHEMA),
        "vex_positions": _source_record(
            vex_positions_path,
            upstream_commit=VEX_COMMIT,
            upstream_git_blob_sha1=vex_position_blob,
            expected_upstream_git_blob_sha1=VEX_POSITION_BLOB_SHA1,
            expected_sha256=VEX_POSITION_SHA256,
            upstream_hash_verified=(
                vex_position_blob == VEX_POSITION_BLOB_SHA1
                and sha256_file(vex_positions_path) == VEX_POSITION_SHA256
            ),
        ),
        "vex_sequences": _source_record(
            vex_sequences_path,
            upstream_commit=VEX_COMMIT,
            upstream_git_blob_sha1=vex_sequence_blob,
            expected_upstream_git_blob_sha1=VEX_SEQUENCE_BLOB_SHA1,
            expected_sha256=VEX_SEQUENCE_SHA256,
            upstream_hash_verified=(
                vex_sequence_blob == VEX_SEQUENCE_BLOB_SHA1
                and sha256_file(vex_sequences_path) == VEX_SEQUENCE_SHA256
            ),
        ),
        "compass_input": _source_record(
            compass_input_path,
            accession=COMPASS_ACCESSION,
            input_kind=compass_input_kind,
            expected_source_filename=COMPASS_COMBINED_FILENAME,
            published_checksum_available=False,
            note=(
                "custodian design-only export"
                if compass_input_kind == "design_only"
                else (
                    "combined container: csv.reader parses complete row lists; only design "
                    "indices are projected and outcome values are never used or serialized"
                )
            ),
        ),
        "compass_column_metadata": _source_record(
            compass_metadata_path,
            accession=COMPASS_ACCESSION,
            expected_filename=COMPASS_METADATA_FILENAME,
            expected_sha256=COMPASS_METADATA_SHA256,
            upstream_hash_verified=(
                sha256_file(compass_metadata_path) == COMPASS_METADATA_SHA256
            ),
        ),
        "compass_reference_fasta": _source_record(
            compass_fasta_path,
            accession=COMPASS_ACCESSION,
            expected_filename=COMPASS_FASTA_FILENAME,
            expected_sha256=COMPASS_FASTA_SHA256,
            upstream_code_commit=COMPASS_CODE_COMMIT,
            upstream_hash_verified=(
                sha256_file(compass_fasta_path) == COMPASS_FASTA_SHA256
            ),
        ),
    }
    if compass_custodian_manifest_path is not None:
        inputs["compass_custodian_manifest"] = _source_record(
            compass_custodian_manifest_path,
            schema=COMPASS_CUSTODIAN_SCHEMA,
        )
    lock_a_sha256 = inputs["lock_a"]["sha256"]
    vex_stats: dict = {}
    compass_stats: dict = {}
    with _atomic_new_directory(output_dir) as temporary:
        vex_path = temporary / "vex_design_pool.jsonl"
        compass_path = temporary / "compass_design_pool.jsonl"
        vex_artifact = _write_jsonl(
            iter_vex_design_records(
                vex_positions_path,
                vex_sequences_path,
                stats=vex_stats,
            ),
            vex_path,
        )
        compass_artifact = _write_jsonl(
            iter_compass_design_records(
                compass_input_path,
                metadata_path=compass_metadata_path,
                fasta_path=compass_fasta_path,
                input_kind=compass_input_kind,
                stats=compass_stats,
            ),
            compass_path,
        )
        if vex_artifact["counts"].get("records", 0) == 0:
            raise RuntimeError("Vex-seq design pool is empty")
        if compass_artifact["counts"].get("records", 0) == 0:
            raise RuntimeError("COMPASS design pool is empty")
        binding = {
            "lock_a_sha256": lock_a_sha256,
            "source_sha256": {
                name: record["sha256"] for name, record in sorted(inputs.items())
            },
            "code_sha256": {
                name: record["sha256"] for name, record in sorted(code.items())
            },
            "design_pool_sha256": {
                "vex_seq": vex_artifact["sha256"],
                "compass": compass_artifact["sha256"],
            },
            "windows": FROZEN_WINDOWS,
        }
        package = {
            "schema": DESIGN_POOL_SCHEMA,
            "status": "FROZEN_DESIGN_POOL_NOT_CANDIDATE_COHORT",
            "design_pool_id": sha256_text(canonical_json(binding)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "lock_a": {
                **inputs["lock_a"],
                "candidate_ids": [
                    candidate["candidate_id"] for candidate in lock_a["candidates"]
                ],
            },
            "windows": FROZEN_WINDOWS,
            "sources": {key: value for key, value in inputs.items() if key != "lock_a"},
            "code": code,
            "design_pools": {
                "vex_seq": vex_artifact,
                "compass": compass_artifact,
            },
            "filter_reports": {
                "vex_seq": vex_stats,
                "compass": compass_stats,
            },
            "compass_input_audit": {
                "input_kind": compass_input_kind,
                "policy": "explicit source-column allowlist plus normalized-record key audit",
                "compass_retained_columns": list(COMPASS_DESIGN_FIELDS),
                "compass_stripped_outcome_columns": compass_stats[
                    "stripped_outcome_columns"
                ],
                "source_container_outcome_values_present": compass_stats[
                    "source_container_outcome_values_present"
                ],
                "complete_source_rows_parsed_into_temporary_csv_lists": compass_stats[
                    "complete_source_rows_parsed_into_temporary_csv_lists"
                ],
                "outcome_columns_projected_into_row_mappings": False,
                "outcome_values_used_for_filtering_matching_or_output": False,
                "outcome_values_serialized": False,
                "custodian_manifest_validated": custodian is not None,
            },
            "vex_outcome_file_downloaded_or_parsed": False,
            "candidate_selection_performed": False,
            "binding": binding,
        }
        package_path = temporary / "design_pool.json"
        package_path.write_text(
            json.dumps(package, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema": "gfmcd.external_validation.design_pool_manifest.v1",
            "status": "FROZEN_DESIGN_POOL_NOT_CANDIDATE_COHORT",
            "design_pool_id": package["design_pool_id"],
            "created_at_utc": package["created_at_utc"],
            "command": command_list,
            "git_commit": git_commit(repo),
            "git_dirty": git_dirty(repo),
            "inputs": inputs,
            "outputs": {
                "vex_design_pool": _artifact_record(vex_path),
                "compass_design_pool": _artifact_record(compass_path),
                "design_pool": _artifact_record(package_path),
            },
        }
        manifest_path = temporary / "design_pool.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return Path(output_dir)


def _read_jsonl_objects(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row is not an object")
            rows.append(value)
    return rows


def _load_design_pool(package_path: str | Path) -> tuple[dict, dict[str, dict]]:
    path = Path(package_path)
    package = json.loads(path.read_text())
    if (
        not isinstance(package, dict)
        or package.get("schema") != DESIGN_POOL_SCHEMA
        or package.get("status") != "FROZEN_DESIGN_POOL_NOT_CANDIDATE_COHORT"
    ):
        raise ValueError("candidate freeze requires a valid frozen design pool")
    records: dict[str, dict] = {}
    pools = package.get("design_pools")
    if not isinstance(pools, Mapping) or set(pools) != {"vex_seq", "compass"}:
        raise ValueError("design pool does not bind both datasets")
    for dataset, artifact in pools.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"invalid {dataset} design-pool artifact")
        source = path.parent / str(artifact["path"])
        if sha256_file(source) != artifact.get("sha256"):
            raise RuntimeError(f"{dataset} design-pool hash mismatch")
        for record in _read_jsonl_objects(source):
            assert_design_record(record)
            if record.get("dataset") != dataset:
                raise ValueError(f"{dataset} pool contains a mislabeled record")
            record_id = str(record["record_id"])
            if record_id in records:
                raise ValueError(f"duplicate design record {record_id}")
            records[record_id] = record
    if not records:
        raise ValueError("design pool is empty")
    return package, records


def _validate_feature_score_manifest(
    path: str | Path,
    *,
    score_path: str | Path,
    lock_a_path: str | Path,
    design_pool_path: str | Path,
    lock_a: Mapping[str, object],
) -> dict:
    manifest = json.loads(Path(path).read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema")
        != "gfmcd.external_validation.feature_score_manifest.v1"
        or manifest.get("status") != "OUTCOME_BLIND_MODEL_FEATURE_SCORES"
    ):
        raise ValueError("invalid model-feature score manifest")
    if manifest.get("external_assay_outcomes_inspected_or_used") is not False:
        raise RuntimeError("feature scoring must be external-outcome-blind")
    expected = {
        "score_file_sha256": sha256_file(score_path),
        "lock_a_sha256": sha256_file(lock_a_path),
        "design_pool_sha256": sha256_file(design_pool_path),
        "model_checkpoint_sha256": lock_a["model"]["checkpoint_sha256"],
    }
    mismatches = {
        field: {"manifest": manifest.get(field), "actual": actual}
        for field, actual in expected.items()
        if manifest.get(field) != actual
    }
    if mismatches:
        raise RuntimeError(f"feature-score manifest binding mismatch: {mismatches}")
    if manifest.get("model_endpoint") != lock_a["model_endpoint"]:
        raise ValueError("feature-score manifest model_endpoint disagrees with Lock A")
    expected_candidates = [candidate["candidate_id"] for candidate in lock_a["candidates"]]
    expected_units = [unit["native_unit_id"] for unit in lock_a["native_units"]]
    if manifest.get("candidate_ids") != expected_candidates:
        raise ValueError("feature-score manifest candidate_ids disagree with Lock A")
    if manifest.get("native_unit_ids") != expected_units:
        raise ValueError("feature-score manifest native_unit_ids disagree with Lock A")
    if manifest.get("score_tuple_family") != "all_design_records_x_candidate_window_roles":
        raise ValueError("feature-score manifest does not bind the complete score tuple family")
    _require_sha256(
        manifest.get("scoring_code_sha256"),
        context="feature_score_manifest.scoring_code_sha256",
    )
    _required_string(manifest, "scoring_command", context="feature score manifest")
    return manifest


def _score_numeric(row: Mapping[str, object], field: str, *, line_no: int) -> float:
    return _finite_number(row.get(field), context=f"score row {line_no}.{field}")


def _validate_and_join_scores(
    *,
    score_path: str | Path,
    lock_a: Mapping[str, object],
    design_records: Mapping[str, dict],
) -> list[dict]:
    candidates = {candidate["candidate_id"]: candidate for candidate in lock_a["candidates"]}
    units = {
        unit["native_unit_id"]: unit
        for unit in lock_a["native_units"]
    }
    expected = {
        (record_id, candidate_id, role)
        for record_id, record in design_records.items()
        for candidate_id, candidate in candidates.items()
        for role in record["window_labels"]
        if role in candidate["window_roles"]
    }
    seen: set[tuple[str, str, str]] = set()
    joined: list[dict] = []
    for line_no, row in enumerate(_read_jsonl_objects(score_path), 1):
        if row.get("schema") != FEATURE_SCORE_SCHEMA:
            raise ValueError(f"score row {line_no} has the wrong schema")
        for key in row:
            if looks_like_outcome_column(key):
                raise RuntimeError(f"outcome-like field in model score row: {key}")
        record_id = _required_string(row, "design_record_id", context=f"score row {line_no}")
        candidate_id = _required_string(row, "candidate_id", context=f"score row {line_no}")
        role = _required_string(row, "window_role", context=f"score row {line_no}")
        if row.get("model_endpoint_name") != lock_a["model_endpoint"]["name"]:
            raise ValueError(f"score row {line_no} model endpoint disagrees with Lock A")
        key = (record_id, candidate_id, role)
        if key in seen:
            raise ValueError(f"duplicate feature-score tuple {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"feature-score tuple is outside the frozen design family: {key}")
        candidate = candidates[candidate_id]
        native_ids = row.get("native_unit_ids")
        if (
            not isinstance(native_ids, list)
            or native_ids != candidate["native_unit_ids"]
            or any(unit_id not in units for unit_id in native_ids)
        ):
            raise ValueError(f"score row {line_no} native units disagree with Lock A")
        reference_score = _score_numeric(row, "reference_feature_score", line_no=line_no)
        alternate_score = _score_numeric(row, "alternate_feature_score", line_no=line_no)
        feature_delta = _score_numeric(row, "feature_delta", line_no=line_no)
        model_delta = _score_numeric(row, "model_margin_delta", line_no=line_no)
        sequence_gc = _score_numeric(row, "sequence_gc", line_no=line_no)
        if abs((alternate_score - reference_score) - feature_delta) > 1e-9:
            raise ValueError(f"score row {line_no} feature_delta is not alt-ref")
        record = design_records[record_id]
        expected_gc = sum(base in {"G", "C"} for base in record["reference_sequence"]) / len(
            record["reference_sequence"]
        )
        if abs(sequence_gc - expected_gc) > 1e-9:
            raise ValueError(f"score row {line_no} sequence_gc is not sequence-derived")
        joined.append(
            {
                **row,
                "reference_feature_score": reference_score,
                "alternate_feature_score": alternate_score,
                "feature_delta": feature_delta,
                "model_margin_delta": model_delta,
                "sequence_gc": sequence_gc,
                "design": record,
            }
        )
    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(
            f"model-feature score artifact is incomplete: {len(missing)} missing tuples; "
            f"first={missing[:3]}"
        )
    return joined


def _classify_score(score: Mapping[str, object], candidate: Mapping[str, object]) -> str:
    thresholds = candidate["carrier_thresholds"]
    signed_feature = candidate["expected_feature_delta_sign"] * score["feature_delta"]
    signed_model = candidate["expected_model_effect_sign"] * score["model_margin_delta"]
    if (
        score["reference_feature_score"] >= thresholds["reference_score_min"]
        and signed_feature >= thresholds["signed_feature_delta_min"]
        and signed_model >= thresholds["carrier_model_margin_abs_min"]
    ):
        return "carrier"
    if (
        abs(score["feature_delta"]) <= thresholds["noncarrier_abs_feature_delta_max"]
        and abs(score["model_margin_delta"])
        <= thresholds["noncarrier_model_margin_abs_max"]
    ):
        return "noncarrier"
    return "unused"


def _match_value(score: Mapping[str, object], field: str) -> object:
    design = score["design"]
    if field == "dataset":
        return design["dataset"]
    if field == "window_role":
        return score["window_role"]
    if field in {"strand", "reference_genomic", "alternate_genomic", "sequence_type"}:
        return design.get(field, "")
    if field == "relative_offset":
        return design[f"{score['window_role']}_offset"]
    if field == "reference_feature_score":
        return score["reference_feature_score"]
    if field == "model_margin_delta":
        return score["model_margin_delta"]
    if field == "sequence_gc":
        return score["sequence_gc"]
    raise ValueError(f"unsupported matching field {field}")


def _within_calipers(
    carrier: Mapping[str, object],
    control: Mapping[str, object],
    calipers: Mapping[str, object],
) -> bool:
    return all(
        abs(float(_match_value(carrier, field)) - float(_match_value(control, field)))
        <= float(caliper)
        for field, caliper in calipers.items()
    )


def _match_distance(
    carrier: Mapping[str, object],
    control: Mapping[str, object],
    *,
    fields: Iterable[str],
    calipers: Mapping[str, object],
) -> float:
    return sum(
        abs(float(_match_value(carrier, field)) - float(_match_value(control, field)))
        / max(float(calipers[field]), 1e-12)
        for field in fields
    )


def build_candidate_matches(
    *,
    lock_a: Mapping[str, object],
    joined_scores: list[dict],
) -> tuple[list[dict], dict]:
    """Deterministically match frozen carriers to model-neutral controls without outcomes."""
    candidates = {candidate["candidate_id"]: candidate for candidate in lock_a["candidates"]}
    matching = lock_a["control_design"]["matching"]
    exact_fields = matching["exact_fields"]
    calipers = matching["numeric_calipers"]
    distance_fields = matching["distance_fields"]
    ratio = matching["controls_per_carrier"]
    classified: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for score in joined_scores:
        key = (score["candidate_id"], score["window_role"])
        buckets = classified.setdefault(key, {"carrier": [], "noncarrier": [], "unused": []})
        buckets[_classify_score(score, candidates[score["candidate_id"]])].append(score)
    matches: list[dict] = []
    report: dict[str, dict] = {}
    for candidate in lock_a["candidates"]:
        candidate_id = candidate["candidate_id"]
        for role in candidate["window_roles"]:
            key = (candidate_id, role)
            buckets = classified.get(key, {"carrier": [], "noncarrier": [], "unused": []})
            carriers = sorted(
                buckets["carrier"],
                key=lambda row: row["design_record_id"],
            )
            available = {
                row["design_record_id"]: row
                for row in buckets["noncarrier"]
            }
            matched_carriers = 0
            for carrier in carriers:
                exact_key = tuple(_match_value(carrier, field) for field in exact_fields)
                eligible = [
                    control
                    for control in available.values()
                    if tuple(_match_value(control, field) for field in exact_fields) == exact_key
                    and _within_calipers(carrier, control, calipers)
                ]
                eligible.sort(
                    key=lambda control: (
                        _match_distance(
                            carrier,
                            control,
                            fields=distance_fields,
                            calipers=calipers,
                        ),
                        control["design_record_id"],
                    )
                )
                selected = eligible[:ratio]
                if len(selected) != ratio:
                    continue
                matched_carriers += 1
                for control_index, control in enumerate(selected, 1):
                    control_id = control["design_record_id"]
                    del available[control_id]
                    matches.append(
                        {
                            "schema": "gfmcd.external_validation.candidate_match.v1",
                            "match_id": (
                                f"{candidate_id}:{role}:{carrier['design_record_id']}:"
                                f"{control_index}"
                            ),
                            "candidate_id": candidate_id,
                            "window_role": role,
                            "carrier_record_id": carrier["design_record_id"],
                            "noncarrier_record_id": control_id,
                            "exact_match_values": {
                                field: _match_value(carrier, field) for field in exact_fields
                            },
                            "distance": _match_distance(
                                carrier,
                                control,
                                fields=distance_fields,
                                calipers=calipers,
                            ),
                        }
                    )
            report[f"{candidate_id}:{role}"] = {
                "eligible_carriers": len(carriers),
                "eligible_noncarriers": len(buckets["noncarrier"]),
                "matched_carriers": matched_carriers,
                "matched_pairs": matched_carriers * ratio,
                "unused": len(buckets["unused"]),
            }
            if matched_carriers == 0:
                raise RuntimeError(f"no complete carrier/control match for {candidate_id}:{role}")
    return matches, report


def freeze_candidate_cohort(
    *,
    lock_a_path: str | Path,
    design_pool_path: str | Path,
    feature_scores_path: str | Path,
    feature_score_manifest_path: str | Path,
    output_dir: str | Path,
    command: Iterable[str] = (),
    repo: str | Path | None = None,
) -> Path:
    """Freeze a carrier/matched-noncarrier cohort from complete model scores, without outcomes."""
    lock_a = read_lock_a(lock_a_path)
    design_pool, records = _load_design_pool(design_pool_path)
    if design_pool["lock_a"]["sha256"] != sha256_file(lock_a_path):
        raise RuntimeError("design pool was not built from the supplied Lock A")
    score_manifest = _validate_feature_score_manifest(
        feature_score_manifest_path,
        score_path=feature_scores_path,
        lock_a_path=lock_a_path,
        design_pool_path=design_pool_path,
        lock_a=lock_a,
    )
    scores = _validate_and_join_scores(
        score_path=feature_scores_path,
        lock_a=lock_a,
        design_records=records,
    )
    matches, match_report = build_candidate_matches(
        lock_a=lock_a,
        joined_scores=scores,
    )
    command_list, code = _code_records(command)
    inputs = {
        "lock_a": _source_record(lock_a_path, schema=LOCK_A_SCHEMA),
        "design_pool": _source_record(design_pool_path, schema=DESIGN_POOL_SCHEMA),
        "feature_scores": _source_record(
            feature_scores_path,
            schema=FEATURE_SCORE_SCHEMA,
        ),
        "feature_score_manifest": _source_record(
            feature_score_manifest_path,
            schema=score_manifest["schema"],
        ),
    }
    with _atomic_new_directory(output_dir) as temporary:
        match_path = temporary / "candidate_matches.jsonl"
        with match_path.open("x", encoding="utf-8") as handle:
            for match in matches:
                handle.write(canonical_json(match) + "\n")
        binding = {
            "input_sha256": {
                name: record["sha256"] for name, record in sorted(inputs.items())
            },
            "matches_sha256": sha256_file(match_path),
            "code_sha256": {
                name: record["sha256"] for name, record in sorted(code.items())
            },
        }
        package = {
            "schema": CANDIDATE_COHORT_SCHEMA,
            "status": "FROZEN_OUTCOME_BLIND_CARRIER_MATCHED_COHORT",
            "external_assay_outcomes_inspected_or_used": False,
            "candidate_cohort_id": sha256_text(canonical_json(binding)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": inputs,
            "matching": lock_a["control_design"]["matching"],
            "match_report": match_report,
            "matches": _artifact_record(match_path),
            "binding": binding,
        }
        package_path = temporary / "candidate_cohort.json"
        package_path.write_text(
            json.dumps(package, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema": "gfmcd.external_validation.candidate_cohort_manifest.v1",
            "status": package["status"],
            "external_assay_outcomes_inspected_or_used": False,
            "candidate_cohort_id": package["candidate_cohort_id"],
            "created_at_utc": package["created_at_utc"],
            "command": command_list,
            "git_commit": git_commit(repo),
            "git_dirty": git_dirty(repo),
            "code": code,
            "inputs": inputs,
            "outputs": {
                "matches": _artifact_record(match_path),
                "candidate_cohort": _artifact_record(package_path),
            },
        }
        (temporary / "candidate_cohort.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return Path(output_dir)


def _download_one(spec: DownloadSpec, destination: Path, *, max_bytes: int) -> dict:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite downloaded source: {destination}")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "gfmcd-stage4-freezer/1"})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("xb") as output:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > max_bytes:
            raise RuntimeError(
                f"design source {spec.name} is {declared} bytes, above limit {max_bytes}"
            )
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > max_bytes:
                raise RuntimeError(
                    f"design source {spec.name} exceeded download limit {max_bytes}"
                )
            digest.update(block)
            output.write(block)
    record = {
        "path": destination.name,
        "url": spec.url,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }
    if spec.expected_sha256 is not None and record["sha256"] != spec.expected_sha256:
        raise RuntimeError(
            f"{spec.name} SHA-256 mismatch: expected {spec.expected_sha256}, "
            f"found {record['sha256']}"
        )
    if spec.expected_sha256 is not None:
        record["expected_sha256"] = spec.expected_sha256
    if spec.accession is not None:
        record["accession"] = spec.accession
    if spec.expected_git_blob_sha1 is not None:
        actual = git_blob_sha1(destination)
        if actual != spec.expected_git_blob_sha1:
            raise RuntimeError(
                f"{spec.name} Git blob mismatch: expected {spec.expected_git_blob_sha1}, "
                f"found {actual}"
            )
        record["upstream_git_blob_sha1"] = actual
        record["upstream_commit"] = VEX_COMMIT
    return record


def download_design_sources(
    *,
    lock_a_path: str | Path,
    output_dir: str | Path,
    max_bytes_per_file: int = 10_000_000,
    command: Iterable[str] = (),
    repo: str | Path | None = None,
) -> Path:
    """Download only small, outcome-free design sources and bind them to frozen Lock A."""
    lock_a = read_lock_a(lock_a_path)
    command_list, code = _code_records(command)
    if max_bytes_per_file <= 0:
        raise ValueError("max_bytes_per_file must be positive")
    with _atomic_new_directory(output_dir) as temporary:
        records = {
            spec.name: _download_one(
                spec,
                temporary / spec.filename,
                max_bytes=max_bytes_per_file,
            )
            for spec in DESIGN_DOWNLOADS
        }
        manifest = {
            "schema": "gfmcd.external_validation.design_download_manifest.v1",
            "status": "OUTCOME_BLIND_DESIGN_SOURCES",
            "external_assay_outcome_values_downloaded_or_parsed": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": command_list,
            "git_commit": git_commit(repo),
            "git_dirty": git_dirty(repo),
            "code": code,
            "lock_a": {
                **file_record(lock_a_path),
                "candidate_ids": [
                    candidate["candidate_id"] for candidate in lock_a["candidates"]
                ],
            },
            "max_bytes_per_file": max_bytes_per_file,
            "sources": records,
            "explicitly_excluded": [
                "Vex-seq delta_PSI_values.tsv",
                "GSE307247_Processed_PSIs_All_Cells.csv.gz",
            ],
        }
        (temporary / "download.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return Path(output_dir)
