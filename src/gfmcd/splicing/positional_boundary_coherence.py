"""Outcome-blind construction and audit primitives for boundary-phase cohorts.

The cohort is a complete 3 x 3 lattice: each block contributes one donor module
and one acceptor module from each annotated phase.  A seventh, gene-disjoint
scaffold fixes sequence length and donor-to-acceptor geometry.  Construction
uses sequence/annotation covariates only; model outputs are rejected at the
input boundary.

This module deliberately does not import any archived exon-edge implementation.
It defines a fresh, injectable data contract so synthetic fixtures can exercise
the complete construction and audit path without a genome or model checkpoint.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import norm


CANDIDATE_SCHEMA = "gfmcd.positional_boundary_coherence.candidates.v1"
CANDIDATE_AUDIT_SCHEMA = "gfmcd.positional_boundary_coherence.candidate_census_audit.v1"
COHORT_SCHEMA = "gfmcd.positional_boundary_coherence.cohort.v1"
CAPACITY_STOP_SCHEMA = "gfmcd.positional_boundary_coherence.phase1_capacity_stop.v1"
CANDIDATE_CENSUS_RECEIPT_SCHEMA = (
    "gfmcd.positional_boundary_coherence.candidate_census_receipt.v1"
)
CANONICAL_CANDIDATE_PRODUCER_REVISION = "addd7be93c321373118b02b82bc3d95eeb0bd20a"
AUDIT_SCHEMA = "gfmcd.positional_boundary_coherence.cohort_audit.v1"
ALGORITHM_VERSION = "pbc-composition-match-v1"
NAMESPACE = "splicebert-positional-boundary-coherence-v1"
SMOKE_NAMESPACE = "splicebert-positional-boundary-coherence-smoke-v1"
BASES = "ACGT"
STOPS = frozenset({"TAA", "TAG", "TGA"})
SMD_ZERO_VARIANCE_SENTINEL = 1e12
DEFAULT_SPAN_NT = 120
DEFAULT_CONSTRUCT_LENGTH = 1022
DEFAULT_DONOR_POS = 392
DEFAULT_ACCEPTOR_POS = 512
DEFAULT_BRIDGE_ACCEPTOR_POS = 437
DEFAULT_DONOR_WINDOW = (302, 428)  # [d-90,d+36)
DEFAULT_ACCEPTOR_WINDOW = (462, 602)  # [a-50,a+90)
DEFAULT_BLOCK_COUNTS = {
    "development": 120,
    "confirmation": 600,
    "phase3_reserve": 384,
}
DEFAULT_MINIMUM_BLOCKS = 384  # public sample-planning fallback
DEFAULT_SPLIT_WEIGHTS = (
    ("development", 2),
    ("confirmation", 2),
    ("phase3_reserve", 1),
)
DEFAULT_ALLOWED_CHROMS = frozenset(
    {*(f"chr{i}" for i in range(1, 8)), *(f"chr{i}" for i in range(10, 23)), "chrX"}
)
DEFAULT_FORBIDDEN_CHROMS = frozenset({"chr8", "chr9", "chrY", "chrM", "MT"})
_AUC_OUTER_WORKERS = 5
_AUC_BOOTSTRAP_CHUNK_SIZE = 256
_AUC_BOOTSTRAP_EQUIVALENCE_ATOL = 1e-12
_AUC_FROZEN_GATE_THRESHOLDS = (0.50, 0.55, 0.70)
ROLE_WINDOWS = {
    "donor": {
        "core": (-3, 6),
        "exonic": (-90, 0),
        "proximal_intronic": (6, 36),
    },
    "acceptor": {
        "core": (-20, 3),
        "proximal_intronic": (-50, -20),
        "exonic": (0, 90),
    },
}

# Candidate construction must not condition on any SpliceBERT readout.  External
# sequence scores are allowed under ``scores``; these tokens identify model-state
# or model-output fields rather than generic biological scores.
FORBIDDEN_OUTCOME_KEY_TOKENS = (
    "activation",
    "attention",
    "coupling",
    "hidden_state",
    "logit",
    "margin",
    "model_loss",
    "outcome",
    "prediction",
)


class CohortConstructionError(RuntimeError):
    """Raised when an outcome-blind cohort cannot be constructed safely."""


@dataclass(frozen=True)
class SampleSizePlan:
    variance: float
    standard_deviation: float
    sesoi: float
    alpha: float
    power: float
    design_effect: float
    calculated_blocks: int
    minimum_blocks: int
    planned_blocks: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "variance": self.variance,
            "standard_deviation": self.standard_deviation,
            "sesoi": self.sesoi,
            "alpha": self.alpha,
            "power": self.power,
            "design_effect": self.design_effect,
            "calculated_blocks": self.calculated_blocks,
            "minimum_blocks": self.minimum_blocks,
            "planned_blocks": self.planned_blocks,
        }


def canonical_json(payload: Any) -> str:
    """Stable JSON used by every content seal."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _is_lower_hex(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def gene_root(gene_id: str) -> str:
    """Version-normalize an Ensembl-like identifier for all split/blacklist checks."""

    root = str(gene_id).strip().split(".", 1)[0]
    if not root:
        raise ValueError("gene_id is empty")
    return root


def normalize_chromosome(chrom: str) -> str:
    value = str(chrom).strip()
    if not value:
        raise ValueError("chromosome is empty")
    return value if value.startswith("chr") else f"chr{value}"


def stable_digest(*parts: object, namespace: str = NAMESPACE) -> str:
    text = "\0".join((namespace, *(str(part) for part in parts)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_split(
    gene_id: str,
    *,
    namespace: str = NAMESPACE,
    weights: Sequence[tuple[str, int]] = DEFAULT_SPLIT_WEIGHTS,
) -> str:
    """Assign a version-normalized gene to one stable, mutually exclusive split."""

    if not weights or any(int(weight) <= 0 for _name, weight in weights):
        raise ValueError("split weights must be non-empty positive integers")
    names = [str(name) for name, _weight in weights]
    if len(set(names)) != len(names):
        raise ValueError("split names must be unique")
    total = sum(int(weight) for _name, weight in weights)
    bucket = int(stable_digest(gene_root(gene_id), namespace=namespace), 16) % total
    cursor = 0
    for name, weight in weights:
        cursor += int(weight)
        if bucket < cursor:
            return str(name)
    raise AssertionError("unreachable split bucket")


def stable_block_split(
    gene_ids: Iterable[str],
    *,
    namespace: str = NAMESPACE,
    weights: Sequence[tuple[str, int]] = DEFAULT_SPLIT_WEIGHTS,
) -> str:
    """Assign the complete sorted seven-gene identity to one study role.

    Role assignment is intentionally block-level.  Hashing seven genes
    independently and requiring every gene to land in the same role would make
    a valid block exponentially rare and would not implement the protocol.
    """

    normalized = sorted({gene_root(gene) for gene in gene_ids})
    if len(normalized) != 7:
        raise ValueError("block role assignment requires exactly seven unique genes")
    identity = "|".join(normalized)
    if not weights or any(int(weight) <= 0 for _name, weight in weights):
        raise ValueError("split weights must be positive")
    total = sum(int(weight) for _name, weight in weights)
    bucket = int(stable_digest(identity, namespace=f"{namespace}:block-role"), 16) % total
    cursor = 0
    for name, weight in weights:
        cursor += int(weight)
        if bucket < cursor:
            return str(name)
    raise AssertionError("unreachable block-role bucket")


def _walk_keys(payload: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield name
            yield from _walk_keys(value, name)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _walk_keys(value, f"{prefix}[{index}]")


def assert_outcome_blind(candidate_package: Mapping[str, Any]) -> None:
    """Fail closed if construction input contains model outcome/state fields."""

    if candidate_package.get("model_outcomes_read") is not False:
        raise CohortConstructionError("candidate package must declare model_outcomes_read=false")
    offenders = []
    for path in _walk_keys(candidate_package):
        leaf = path.rsplit(".", 1)[-1].split("[", 1)[0].lower()
        if leaf == "model_outcomes_read":
            continue
        if any(token in leaf for token in FORBIDDEN_OUTCOME_KEY_TOKENS):
            offenders.append(path)
    if offenders:
        raise CohortConstructionError(
            "outcome-bearing candidate keys are forbidden: " + ", ".join(offenders[:8])
        )


def load_blacklist(payload: Any) -> set[str]:
    """Accept a list or a ``gene_ids`` object and return normalized hard exclusions."""

    if isinstance(payload, Mapping):
        values = payload.get("gene_ids")
    else:
        values = payload
    if not isinstance(values, list):
        raise ValueError("blacklist must be a list or an object containing gene_ids")
    return {gene_root(str(value)) for value in values}


def _finite_numeric_mapping(value: Any, *, field: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object of numeric scalars")
    out: dict[str, float] = {}
    for key, raw in value.items():
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"{field}.{key} must be finite")
        out[str(key)] = number
    return dict(sorted(out.items()))


def _validate_sequence(sequence: Any, *, candidate_id: str) -> str:
    value = str(sequence).upper()
    if not value or set(value) - set(BASES):
        raise ValueError(f"{candidate_id}: sequence must contain only A/C/G/T")
    return value


def validate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize one injectable candidate record."""

    required = {
        "candidate_id",
        "role",
        "gene_id",
        "chrom",
        "strand",
        "sequence",
        "source_start",
        "source_end",
        "source_exon_length",
    }
    missing = required - set(candidate)
    if missing:
        raise ValueError(f"candidate missing fields: {sorted(missing)}")
    candidate_id = str(candidate["candidate_id"])
    role = str(candidate["role"])
    if role not in {"scaffold", "donor", "acceptor"}:
        raise ValueError(f"{candidate_id}: invalid role {role!r}")
    strand = str(candidate["strand"])
    if strand not in {"+", "-"}:
        raise ValueError(f"{candidate_id}: strand must be '+' or '-'")
    sequence = _validate_sequence(candidate["sequence"], candidate_id=candidate_id)
    gene_id_versioned = str(candidate["gene_id"]).strip()
    source_start, source_end = int(candidate["source_start"]), int(candidate["source_end"])
    source_exon_length = int(candidate["source_exon_length"])
    if source_start < 0 or source_end <= source_start or source_exon_length <= 0:
        raise ValueError(f"{candidate_id}: invalid source coordinates/exon length")
    out: dict[str, Any] = {
        "candidate_id": candidate_id,
        "role": role,
        "gene_id": gene_root(gene_id_versioned),
        "gene_id_versioned": gene_id_versioned,
        "chrom": normalize_chromosome(str(candidate["chrom"])),
        "strand": strand,
        "source_start": source_start,
        "source_end": source_end,
        "source_exon_length": source_exon_length,
        "transcript_id": (
            None if candidate.get("transcript_id") is None else str(candidate["transcript_id"])
        ),
        "sequence": sequence,
        "sequence_sha256": sequence_sha256(sequence),
        "geometry": _finite_numeric_mapping(candidate.get("geometry"), field="geometry"),
        "scores": _finite_numeric_mapping(candidate.get("scores"), field="scores"),
        "eligibility_checks": {
            str(key): bool(value)
            for key, value in dict(candidate.get("eligibility_checks", {})).items()
        },
    }
    if not out["eligibility_checks"] or not all(out["eligibility_checks"].values()):
        raise ValueError(f"{candidate_id}: every declared eligibility check must pass")
    if role == "scaffold":
        for field in (
            "donor_window",
            "acceptor_window",
            "donor_pos",
            "acceptor_pos",
            "bridge_acceptor_pos",
        ):
            if field not in candidate:
                raise ValueError(f"{candidate_id}: scaffold missing {field}")
        donor_window = tuple(int(x) for x in candidate["donor_window"])
        acceptor_window = tuple(int(x) for x in candidate["acceptor_window"])
        if len(donor_window) != 2 or len(acceptor_window) != 2:
            raise ValueError(f"{candidate_id}: replacement windows need [start,end]")
        ds, de = donor_window
        aas, ae = acceptor_window
        donor_pos = int(candidate["donor_pos"])
        acceptor_pos = int(candidate["acceptor_pos"])
        bridge_acceptor_pos = int(candidate["bridge_acceptor_pos"])
        if not (0 <= ds < de <= aas < ae <= len(sequence)):
            raise ValueError(f"{candidate_id}: windows must be ordered and non-overlapping")
        if not (ds <= donor_pos < de and aas < acceptor_pos <= ae):
            raise ValueError(f"{candidate_id}: anchors must fall inside replacement windows")
        if (
            bridge_acceptor_pos != DEFAULT_BRIDGE_ACCEPTOR_POS
            or sequence[bridge_acceptor_pos - 2 : bridge_acceptor_pos] != "AG"
            or not de <= bridge_acceptor_pos - 2 < bridge_acceptor_pos <= aas
        ):
            raise ValueError(
                f"{candidate_id}: fixed bridge must be AG at "
                f"[{DEFAULT_BRIDGE_ACCEPTOR_POS - 2},{DEFAULT_BRIDGE_ACCEPTOR_POS})"
            )
        out.update(
            {
                "donor_window": [ds, de],
                "acceptor_window": [aas, ae],
                "donor_pos": donor_pos,
                "acceptor_pos": acceptor_pos,
                "bridge_acceptor_pos": bridge_acceptor_pos,
                "donor_anchor_covariates": _finite_numeric_mapping(
                    candidate.get("donor_anchor_covariates"),
                    field="donor_anchor_covariates",
                ),
                "acceptor_anchor_covariates": _finite_numeric_mapping(
                    candidate.get("acceptor_anchor_covariates"),
                    field="acceptor_anchor_covariates",
                ),
            }
        )
        if not out["donor_anchor_covariates"] or not out["acceptor_anchor_covariates"]:
            raise ValueError(
                f"{candidate_id}: both scaffold-centered role anchor vectors are required"
            )
    else:
        for field in ("phase", "boundary_index", "coding_offset"):
            if field not in candidate:
                raise ValueError(f"{candidate_id}: module missing {field}")
        phase = int(candidate["phase"])
        boundary = int(candidate["boundary_index"])
        coding_offset = int(candidate["coding_offset"])
        if phase not in {0, 1, 2}:
            raise ValueError(f"{candidate_id}: phase must be 0, 1, or 2")
        if not 0 <= boundary <= len(sequence):
            raise ValueError(f"{candidate_id}: boundary_index out of range")
        if coding_offset not in {0, 1, 2}:
            raise ValueError(f"{candidate_id}: coding_offset must be 0, 1, or 2")
        for window_name, (relative_start, relative_end) in ROLE_WINDOWS[role].items():
            start, end = boundary + relative_start, boundary + relative_end
            if start < 0 or end > len(sequence):
                raise ValueError(
                    f"{candidate_id}: lacks complete {role} {window_name} support "
                    f"[{relative_start},{relative_end})"
                )
        annotation = candidate.get("annotation")
        if not isinstance(annotation, Mapping):
            raise ValueError(f"{candidate_id}: annotation provenance is required")
        required_annotation = {
            "phase_convention",
            "raw_gtf_frame",
            "upstream_cds_end",
            "downstream_cds_start",
            "basic_phase",
            "comprehensive_phase",
            "basic_support_ids",
            "comprehensive_support_ids",
            "checks",
        }
        if required_annotation - set(annotation):
            raise ValueError(
                f"{candidate_id}: annotation missing {sorted(required_annotation - set(annotation))}"
            )
        annotation_checks = {str(key): bool(value) for key, value in annotation["checks"].items()}
        required_checks = {
            "upstream_downstream_phase_equal",
            "basic_transcript_coordinate_agreement",
            "comprehensive_no_coding_conflict",
            "comprehensive_no_strand_conflict",
            "coding_offset_consistent_with_gtf_frame",
        }
        if (
            annotation["phase_convention"] != "intron_phase"
            or int(annotation["basic_phase"]) != phase
            or int(annotation["comprehensive_phase"]) != phase
            or required_checks - set(annotation_checks)
            or not all(annotation_checks[key] for key in required_checks)
            or not list(annotation["basic_support_ids"])
            or not list(annotation["comprehensive_support_ids"])
        ):
            raise ValueError(f"{candidate_id}: annotation phase/provenance checks failed")
        out["annotation"] = {
            "phase_convention": "intron_phase",
            "raw_gtf_frame": str(annotation["raw_gtf_frame"]),
            "upstream_cds_end": int(annotation["upstream_cds_end"]),
            "downstream_cds_start": int(annotation["downstream_cds_start"]),
            "basic_phase": int(annotation["basic_phase"]),
            "comprehensive_phase": int(annotation["comprehensive_phase"]),
            "basic_support_ids": sorted(str(value) for value in annotation["basic_support_ids"]),
            "comprehensive_support_ids": sorted(
                str(value) for value in annotation["comprehensive_support_ids"]
            ),
            "checks": dict(sorted(annotation_checks.items())),
        }
        motif = sequence[boundary : boundary + 2] if role == "donor" else sequence[boundary - 2 : boundary]
        expected = "GT" if role == "donor" else "AG"
        if motif != expected:
            raise ValueError(f"{candidate_id}: expected canonical {expected} at boundary")
        out.update(
            {
                "phase": phase,
                "boundary_index": boundary,
                "coding_offset": coding_offset,
            }
        )
        native_sequence = _validate_sequence(
            candidate.get("native_sequence", ""), candidate_id=candidate_id
        )
        native_donor_pos = int(candidate.get("native_donor_pos", -1))
        native_acceptor_pos = int(candidate.get("native_acceptor_pos", -1))
        if (
            len(native_sequence) != DEFAULT_CONSTRUCT_LENGTH
            or native_acceptor_pos != DEFAULT_ACCEPTOR_POS
            or not 0 <= native_donor_pos < native_acceptor_pos - 2
            or native_sequence[native_donor_pos : native_donor_pos + 2] != "GT"
            or native_sequence[
                native_acceptor_pos - 2 : native_acceptor_pos
            ]
            != "AG"
        ):
            raise ValueError(
                f"{candidate_id}: native source-site window/geometry is invalid"
            )
        out.update(
            {
                "native_sequence": native_sequence,
                "native_sequence_sha256": sequence_sha256(native_sequence),
                "native_donor_pos": native_donor_pos,
                "native_acceptor_pos": native_acceptor_pos,
            }
        )
        if "maxent" not in out["scores"]:
            raise ValueError(f"{candidate_id}: scores.maxent is required and must be precomputed")
    return out


def validate_candidate_package(candidate_package: Mapping[str, Any]) -> list[dict[str, Any]]:
    assert_outcome_blind(candidate_package)
    if candidate_package.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError(f"candidate schema must be {CANDIDATE_SCHEMA}")
    candidate_mode = candidate_package.get("candidate_mode")
    if candidate_mode not in {"production", "synthetic_fixture"}:
        raise ValueError("candidate_mode must explicitly be production or synthetic_fixture")
    if candidate_mode == "production" and candidate_package.get("stage") != "construction":
        raise ValueError("production candidate census must declare stage=construction")
    raw = candidate_package.get("candidates")
    if not isinstance(raw, list) or not raw:
        raise ValueError("candidate package needs a non-empty candidates list")
    candidates = [validate_candidate(candidate) for candidate in raw]
    ids = [candidate["candidate_id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique")
    loci = [
        (
            candidate["role"],
            candidate["chrom"],
            candidate["strand"],
            candidate["source_start"],
            candidate["source_end"],
        )
        for candidate in candidates
    ]
    if len(loci) != len(set(loci)):
        raise ValueError("duplicate source loci are forbidden")
    for role in ("donor", "acceptor"):
        score_schemas = {
            tuple(candidate["scores"])
            for candidate in candidates
            if candidate["role"] == role
        }
        geometry_schemas = {
            tuple(candidate["geometry"])
            for candidate in candidates
            if candidate["role"] == role
        }
        if len(score_schemas) > 1 or len(geometry_schemas) > 1:
            raise ValueError(f"{role} modules must share score and geometry schemas")
        feature_names = {
            tuple(extract_module_covariates(candidate))
            for candidate in candidates
            if candidate["role"] == role
        }
        if len(feature_names) != 1:
            raise ValueError(f"{role} modules must share the complete covariate schema")
        expected = next(iter(feature_names), ())
        anchor_field = f"{role}_anchor_covariates"
        for scaffold in (row for row in candidates if row["role"] == "scaffold"):
            if tuple(scaffold[anchor_field]) != expected:
                raise ValueError(
                    f"{scaffold['candidate_id']}: {anchor_field} must use the complete {role} schema"
                )
    return candidates


def validate_candidate_audit_binding(
    candidate_package: Mapping[str, Any], candidate_audit: Mapping[str, Any] | None
) -> None:
    """Require a biology-reconstructing census audit for every production package."""

    if candidate_package.get("candidate_mode") == "synthetic_fixture":
        if candidate_audit is not None:
            raise ValueError("synthetic fixtures must not masquerade as production-audited inputs")
        return
    if not isinstance(candidate_audit, Mapping):
        raise ValueError("production candidate census requires its independent audit")
    if (
        candidate_audit.get("schema") != CANDIDATE_AUDIT_SCHEMA
        or candidate_audit.get("stage") != "construction"
        or candidate_audit.get("status") != "AUDIT_PASS"
        or candidate_audit.get("passed") is not True
        or candidate_audit.get("candidate_census_sha256") != sha256_payload(candidate_package)
    ):
        raise ValueError("candidate census audit is failed or bound to another package")


def kmer_frequencies(sequence: str, k: int) -> dict[str, float]:
    kmers = tuple("".join(chars) for chars in itertools.product(BASES, repeat=k))
    denominator = max(1, len(sequence) - k + 1)
    counts = Counter(sequence[index : index + k] for index in range(len(sequence) - k + 1))
    return {kmer: counts.get(kmer, 0) / denominator for kmer in kmers}


def _coding_sequence(candidate: Mapping[str, Any]) -> tuple[str, int]:
    boundary = int(candidate["boundary_index"])
    relative_start, relative_end = ROLE_WINDOWS[str(candidate["role"])]["exonic"]
    start, end = boundary + relative_start, boundary + relative_end
    offset = (int(candidate["coding_offset"]) - start) % 3
    return str(candidate["sequence"])[start:end], offset


def stop_load(sequence: str, coding_offset: int) -> dict[str, float | int]:
    starts = range(int(coding_offset), max(int(coding_offset), len(sequence) - 2), 3)
    codons = [sequence[start : start + 3] for start in starts if start + 3 <= len(sequence)]
    count = sum(codon in STOPS for codon in codons)
    return {
        "stop_count": int(count),
        "codon_count": int(len(codons)),
        "stop_rate": float(count / len(codons)) if codons else 0.0,
    }


def extract_module_covariates(candidate: Mapping[str, Any]) -> dict[str, float]:
    """Extract model-free sequence and supplied scalar covariates for one module."""

    if candidate["role"] not in {"donor", "acceptor"}:
        raise ValueError("module covariates require a donor or acceptor")
    sequence = str(candidate["sequence"])
    boundary = int(candidate["boundary_index"])
    role = str(candidate["role"])
    out: dict[str, float] = {
        "sequence.length": float(len(sequence)),
        "source.exon_length": float(candidate["source_exon_length"]),
        "sequence.gc": float(sum(base in "GC" for base in sequence) / len(sequence)),
        "sequence.cpg_count": float(
            sum(sequence[index : index + 2] == "CG" for index in range(len(sequence) - 1))
        ),
        "sequence.cpg": float(
            sum(sequence[index : index + 2] == "CG" for index in range(len(sequence) - 1))
            / max(1, len(sequence) - 1)
        ),
        "strand.plus": float(candidate["strand"] == "+"),
        "strand.minus": float(candidate["strand"] == "-"),
    }
    for chrom in sorted(DEFAULT_ALLOWED_CHROMS):
        out[f"chromosome.{chrom}"] = float(candidate["chrom"] == chrom)
    core_start, core_end = ROLE_WINDOWS[role]["core"]
    for offset in range(core_start, core_end):
        position = boundary + offset
        base_at_position = sequence[position]
        label = f"p{offset}" if offset >= 0 else f"m{abs(offset)}"
        for base in BASES:
            out[f"core.{label}.{base}"] = float(base_at_position == base)
    for window_name in ("exonic", "proximal_intronic"):
        relative_start, relative_end = ROLE_WINDOWS[role][window_name]
        window = sequence[boundary + relative_start : boundary + relative_end]
        out[f"{window_name}.length"] = float(len(window))
        out[f"{window_name}.gc"] = float(sum(base in "GC" for base in window) / len(window))
        cpg_count = sum(window[index : index + 2] == "CG" for index in range(len(window) - 1))
        out[f"{window_name}.cpg_count"] = float(cpg_count)
        out[f"{window_name}.cpg_rate"] = float(cpg_count / max(1, len(window) - 1))
        for k, family in ((1, "mono"), (2, "di"), (3, "tri")):
            for kmer, value in kmer_frequencies(window, k).items():
                out[f"{window_name}.{family}.{kmer}"] = float(value)
    coding, coding_offset = _coding_sequence(candidate)
    out["coding.length"] = float(len(coding))
    slot_counts = {(slot, base): 0 for slot in range(3) for base in BASES}
    slot_totals = Counter()
    for index, base in enumerate(coding):
        slot = (index - coding_offset) % 3
        slot_counts[(slot, base)] += 1
        slot_totals[slot] += 1
    for slot in range(3):
        for base in BASES:
            out[f"coding.pos{slot}.{base}"] = float(
                slot_counts[(slot, base)] / max(1, slot_totals[slot])
            )
    tri_by_slot = {
        (slot, kmer): 0
        for slot in range(3)
        for kmer in ("".join(chars) for chars in itertools.product(BASES, repeat=3))
    }
    tri_totals = Counter()
    for index in range(max(0, len(coding) - 2)):
        slot = (index - coding_offset) % 3
        tri_by_slot[(slot, coding[index : index + 3])] += 1
        tri_totals[slot] += 1
    for (slot, kmer), count in sorted(tri_by_slot.items()):
        out[f"coding.tri_pos{slot}.{kmer}"] = float(count / max(1, tri_totals[slot]))
    load = stop_load(coding, coding_offset)
    out.update({f"coding.{key}": float(value) for key, value in load.items()})
    out.update({f"geometry.{key}": float(value) for key, value in candidate["geometry"].items()})
    out.update({f"score.{key}": float(value) for key, value in candidate["scores"].items()})
    return dict(sorted(out.items()))


def _joined_coding(donor: Mapping[str, Any], acceptor: Mapping[str, Any]) -> tuple[str, int]:
    donor_coding, donor_offset = _coding_sequence(donor)
    acceptor_coding, _acceptor_offset = _coding_sequence(acceptor)
    return donor_coding + acceptor_coding, donor_offset


def extract_cell_covariates(
    donor: Mapping[str, Any],
    acceptor: Mapping[str, Any],
    scaffold: Mapping[str, Any],
) -> dict[str, float]:
    """Full composition vector, including explicit donor/acceptor interactions."""

    donor_features = extract_module_covariates(donor)
    acceptor_features = extract_module_covariates(acceptor)
    out = {f"donor.{key}": value for key, value in donor_features.items()}
    out.update({f"acceptor.{key}": value for key, value in acceptor_features.items()})
    out.update(
        {
            "geometry.span_nt": float(scaffold["acceptor_pos"] - scaffold["donor_pos"]),
            "geometry.construct_length": float(len(scaffold["sequence"])),
            "geometry.donor_position": float(scaffold["donor_pos"]),
            "geometry.acceptor_position": float(scaffold["acceptor_pos"]),
            "geometry.donor_window_length": float(
                scaffold["donor_window"][1] - scaffold["donor_window"][0]
            ),
            "geometry.acceptor_window_length": float(
                scaffold["acceptor_window"][1] - scaffold["acceptor_window"][0]
            ),
        }
    )
    out.update(
        {f"scaffold.geometry.{key}": float(value) for key, value in scaffold["geometry"].items()}
    )
    shared = sorted(set(donor_features) & set(acceptor_features))
    for key in shared:
        left, right = donor_features[key], acceptor_features[key]
        out[f"pair.absdiff.{key}"] = abs(left - right)
        out[f"pair.product.{key}"] = left * right
    joined, offset = _joined_coding(donor, acceptor)
    for register in (0, 1, 2):
        joined_load = stop_load(joined, (offset + register) % 3)
        out.update(
            {
                f"pair.joined_register{register}_{key}": float(value)
                for key, value in joined_load.items()
            }
        )
    if not all(math.isfinite(value) for value in out.values()):
        raise ValueError("cell covariates must be finite")
    return dict(sorted(out.items()))


def plan_sample_size(
    frozen_variance: Mapping[str, Any],
    *,
    minimum_blocks: int = DEFAULT_MINIMUM_BLOCKS,
) -> SampleSizePlan:
    """Plan independent blocks from a frozen variance scalar, never live outcomes."""

    variance = float(frozen_variance["variance"])
    sesoi = float(frozen_variance.get("sesoi", 0.05))
    alpha = float(frozen_variance.get("alpha", 0.05))
    power = float(frozen_variance.get("power", 0.80))
    design_effect = float(frozen_variance.get("design_effect", 1.0))
    if variance < 0 or sesoi <= 0 or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("invalid frozen sample-size inputs")
    if design_effect < 1 or int(minimum_blocks) <= 0:
        raise ValueError("design_effect must be >=1 and minimum_blocks positive")
    standard_deviation = math.sqrt(variance)
    calculated = int(
        math.ceil(
            design_effect
            * ((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * standard_deviation / sesoi) ** 2
        )
    )
    calculated = max(1, calculated)
    return SampleSizePlan(
        variance=variance,
        standard_deviation=standard_deviation,
        sesoi=sesoi,
        alpha=alpha,
        power=power,
        design_effect=design_effect,
        calculated_blocks=calculated,
        minimum_blocks=int(minimum_blocks),
        planned_blocks=max(int(minimum_blocks), calculated),
    )


def _module_features(
    candidate: Mapping[str, Any], cache: dict[str, dict[str, float]] | None
) -> dict[str, float]:
    if cache is None:
        return extract_module_covariates(candidate)
    candidate_id = str(candidate["candidate_id"])
    if candidate_id not in cache:
        cache[candidate_id] = extract_module_covariates(candidate)
    return cache[candidate_id]


def _feature_scale(
    candidates: Sequence[Mapping[str, Any]],
    cache: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    feature_maps = [_module_features(candidate, cache) for candidate in candidates]
    names = sorted(set().union(*(features.keys() for features in feature_maps)))
    matrix = np.asarray([[features.get(name, 0.0) for name in names] for features in feature_maps])
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    return dict(zip(names, means.tolist())), dict(zip(names, scales.tolist()))


def _module_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    means: Mapping[str, float],
    scales: Mapping[str, float],
    feature_cache: dict[str, dict[str, float]] | None = None,
    distance_cache: dict[tuple[str, str], float] | None = None,
    vector_cache: dict[str, np.ndarray] | None = None,
) -> float:
    pair = tuple(sorted((str(left["candidate_id"]), str(right["candidate_id"]))))
    if distance_cache is not None and pair in distance_cache:
        return distance_cache[pair]
    if vector_cache is not None:
        difference = vector_cache[str(left["candidate_id"])] - vector_cache[str(right["candidate_id"])]
        distance = float(math.sqrt(float(np.mean(np.square(difference)))))
    else:
        left_features = _module_features(left, feature_cache)
        right_features = _module_features(right, feature_cache)
        values = [
            (left_features.get(name, means[name]) - right_features.get(name, means[name]))
            / scales[name]
            for name in means
        ]
        distance = float(math.sqrt(np.mean(np.square(values)))) if values else 0.0
    if distance_cache is not None:
        distance_cache[pair] = distance
    return distance


def _select_phase_triplet(
    pools: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    used_genes: set[str],
    means: Mapping[str, float],
    scales: Mapping[str, float],
    search_width: int,
    neighbor_width: int,
    namespace: str,
    anchor_features: Mapping[str, float],
    feature_cache: dict[str, dict[str, float]] | None = None,
    distance_cache: dict[tuple[str, str], float] | None = None,
    vector_cache: dict[str, np.ndarray] | None = None,
) -> tuple[dict[int, Mapping[str, Any]], float] | None:
    names = tuple(means)
    if tuple(sorted(anchor_features)) != names:
        raise CohortConstructionError("scaffold anchor does not match the frozen role feature schema")

    def anchor_distance(candidate: Mapping[str, Any]) -> float:
        features = _module_features(candidate, feature_cache)
        values = [
            (float(features[name]) - float(anchor_features[name])) / float(scales[name])
            for name in names
        ]
        return float(math.sqrt(float(np.mean(np.square(values))))) if values else 0.0

    eligible = {}
    for phase in (0, 1, 2):
        ranked = sorted(
            [candidate for candidate in pools[phase] if candidate["gene_id"] not in used_genes],
            key=lambda candidate: (
                anchor_distance(candidate),
                stable_digest(candidate["candidate_id"], namespace=namespace),
                candidate["candidate_id"],
            ),
        )[:search_width]
        eligible[phase] = ranked[: min(neighbor_width, len(ranked))]
    if any(not eligible[phase] for phase in (0, 1, 2)):
        return None
    best: tuple[tuple[float, str], dict[int, Mapping[str, Any]], float] | None = None
    for phase0 in eligible[0]:
        for phase1 in eligible[1]:
            for phase2 in eligible[2]:
                genes = {phase0["gene_id"], phase1["gene_id"], phase2["gene_id"]}
                if len(genes) != 3:
                    continue
                distances = tuple(anchor_distance(candidate) for candidate in (phase0, phase1, phase2))
                score = max(distances)
                ids = "|".join(
                    str(candidate["candidate_id"]) for candidate in (phase0, phase1, phase2)
                )
                key = (score, stable_digest(ids, namespace=namespace))
                if best is None or key < best[0]:
                    best = (key, {0: phase0, 1: phase1, 2: phase2}, score)
    return None if best is None else (best[1], best[2])


def _replace_windows(
    scaffold: Mapping[str, Any], donor: Mapping[str, Any], acceptor: Mapping[str, Any]
) -> str:
    ds, de = scaffold["donor_window"]
    aas, ae = scaffold["acceptor_window"]
    if len(donor["sequence"]) != de - ds or len(acceptor["sequence"]) != ae - aas:
        raise CohortConstructionError("module length does not match scaffold replacement window")
    if int(donor["boundary_index"]) != int(scaffold["donor_pos"]) - ds:
        raise CohortConstructionError("donor boundary does not align to scaffold donor_pos")
    if int(acceptor["boundary_index"]) != int(scaffold["acceptor_pos"]) - aas:
        raise CohortConstructionError("acceptor boundary does not align to scaffold acceptor_pos")
    sequence = (
        scaffold["sequence"][:ds]
        + donor["sequence"]
        + scaffold["sequence"][de:aas]
        + acceptor["sequence"]
        + scaffold["sequence"][ae:]
    )
    acceptor_pos = int(scaffold["acceptor_pos"])
    donor_pos = int(scaffold["donor_pos"])
    if sequence[acceptor_pos - 2 : acceptor_pos] != "AG":
        raise CohortConstructionError("constructed cell lost canonical acceptor AG")
    if sequence[donor_pos : donor_pos + 2] != "GT":
        raise CohortConstructionError("constructed cell lost canonical donor GT")
    bridge_pos = int(scaffold["bridge_acceptor_pos"])
    if sequence[bridge_pos - 2 : bridge_pos] != "AG":
        raise CohortConstructionError("constructed cell lost the frozen bridge AG")
    return sequence


def _acceptor_alt(sequence: str, acceptor_pos: int, alt: str) -> str:
    if alt not in {"AC", "AT"}:
        raise ValueError("this cohort freezes acceptor alternatives to AC and AT")
    if sequence[acceptor_pos - 2 : acceptor_pos] != "AG":
        raise CohortConstructionError("acceptor alternative requires an intact AG")
    return sequence[: acceptor_pos - 2] + alt + sequence[acceptor_pos:]


def _cell_record(
    *,
    block_id: str,
    split_role: str,
    scaffold: Mapping[str, Any],
    donor: Mapping[str, Any],
    acceptor: Mapping[str, Any],
) -> dict[str, Any]:
    intact = _replace_windows(scaffold, donor, acceptor)
    acceptor_pos = int(scaffold["acceptor_pos"])
    ac = _acceptor_alt(intact, acceptor_pos, "AC")
    at = _acceptor_alt(intact, acceptor_pos, "AT")
    donor_phase, acceptor_phase = int(donor["phase"]), int(acceptor["phase"])
    phase_match = donor_phase == acceptor_phase
    delta = (acceptor_phase - donor_phase) % 3
    delta_class = {0: "M0", 1: "Mplus", 2: "Mminus"}[delta]
    return {
        "cell_id": f"{block_id}:d{donor_phase}:a{acceptor_phase}",
        "block_id": block_id,
        "split_role": split_role,
        "chrom": scaffold["chrom"],
        "strand": scaffold["strand"],
        "donor_phase": donor_phase,
        "acceptor_phase": acceptor_phase,
        "phase_match": phase_match,
        "annotated_delta": delta,
        "delta_class": delta_class,
        "analysis_weight": 1.0 / (3.0 if phase_match else 6.0),
        "scaffold_source_id": scaffold["candidate_id"],
        "scaffold_gene_id": scaffold["gene_id"],
        "donor_source_id": donor["candidate_id"],
        "donor_gene_id": donor["gene_id"],
        "acceptor_source_id": acceptor["candidate_id"],
        "acceptor_gene_id": acceptor["gene_id"],
        "donor_pos": int(scaffold["donor_pos"]),
        "acceptor_pos": acceptor_pos,
        "span_nt": int(acceptor_pos - int(scaffold["donor_pos"])),
        "bridge_acceptor_pos": int(scaffold["bridge_acceptor_pos"]),
        "sequence_intact": intact,
        "sequence_acceptor_ac": ac,
        "sequence_acceptor_at": at,
        "sequence_sha256": {
            "intact": sequence_sha256(intact),
            "acceptor_ac": sequence_sha256(ac),
            "acceptor_at": sequence_sha256(at),
        },
        "covariates": extract_cell_covariates(donor, acceptor, scaffold),
    }


def _weighted_smd(values: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    class_values = []
    for label in (0, 1):
        keep = labels == label
        w = weights[keep]
        x = values[keep]
        mean = float(np.average(x, weights=w))
        variance = float(np.average(np.square(x - mean), weights=w))
        class_values.append((mean, variance))
    difference = class_values[1][0] - class_values[0][0]
    pooled = math.sqrt((class_values[0][1] + class_values[1][1]) / 2)
    if pooled <= 1e-15:
        return (
            0.0
            if abs(difference) <= 1e-15
            else math.copysign(SMD_ZERO_VARIANCE_SENTINEL, difference)
        )
    return difference / pooled


def standardized_mean_differences(
    feature_rows: Sequence[Mapping[str, float]],
    labels: Sequence[bool | int],
    weights: Sequence[float],
) -> dict[str, Any]:
    if not feature_rows or len(feature_rows) != len(labels) or len(labels) != len(weights):
        raise ValueError("feature rows, labels, and weights must be non-empty and aligned")
    names = sorted(set().union(*(row.keys() for row in feature_rows)))
    label_array = np.asarray(labels, dtype=np.int8)
    weight_array = np.asarray(weights, dtype=np.float64)
    if set(label_array.tolist()) != {0, 1} or np.any(weight_array <= 0):
        raise ValueError("SMD needs two classes and positive weights")
    smds: dict[str, float] = {}
    for name in names:
        values = np.asarray([float(row.get(name, 0.0)) for row in feature_rows])
        smds[name] = float(_weighted_smd(values, label_array, weight_array))
    ordered = sorted(smds.items(), key=lambda item: (-abs(item[1]), item[0]))
    maximum = abs(ordered[0][1]) if ordered else 0.0
    return {
        "max_abs_smd": float(maximum),
        "max_feature": ordered[0][0] if ordered else None,
        "n_features": len(names),
        "top": [{"feature": name, "smd": value} for name, value in ordered[:25]],
        "all": dict(sorted(smds.items())),
    }


def three_delta_smd_audit(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit M0-M+, M0-M-, and M+-M- for every frozen feature."""

    if not cells:
        raise ValueError("three-delta SMD audit needs cells")
    feature_rows = [dict(cell["covariates"]) for cell in cells]
    names = sorted(set().union(*(row.keys() for row in feature_rows)))
    classes = np.asarray([str(cell["delta_class"]) for cell in cells], dtype=object)
    expected = {"M0", "Mplus", "Mminus"}
    if set(classes.tolist()) != expected:
        raise ValueError("all three annotated-delta classes are required")
    contrasts = (
        ("M0_minus_Mplus", "M0", "Mplus"),
        ("M0_minus_Mminus", "M0", "Mminus"),
        ("Mplus_minus_Mminus", "Mplus", "Mminus"),
    )
    rows: list[dict[str, Any]] = []
    by_role: dict[str, float] = {}
    for name in names:
        values = np.asarray([float(row.get(name, 0.0)) for row in feature_rows])
        feature_role = name.split(".", 1)[0]
        for contrast_name, left_name, right_name in contrasts:
            left = values[classes == left_name]
            right = values[classes == right_name]
            left_mean, right_mean = float(left.mean()), float(right.mean())
            pooled = math.sqrt((float(left.var()) + float(right.var())) / 2)
            difference = left_mean - right_mean
            smd = (
                0.0
                if pooled <= 1e-15 and abs(difference) <= 1e-15
                else math.copysign(SMD_ZERO_VARIANCE_SENTINEL, difference)
                if pooled <= 1e-15
                else difference / pooled
            )
            rows.append(
                {
                    "feature": name,
                    "feature_role": feature_role,
                    "contrast": contrast_name,
                    "smd": float(smd),
                    "left_mean": left_mean,
                    "right_mean": right_mean,
                }
            )
            by_role[feature_role] = max(by_role.get(feature_role, 0.0), abs(float(smd)))
    ordered = sorted(rows, key=lambda row: (-abs(row["smd"]), row["feature"], row["contrast"]))
    return {
        "contrasts": [name for name, _left, _right in contrasts],
        "n_features": len(names),
        "n_family": len(rows),
        "max_abs_smd": abs(float(ordered[0]["smd"])) if ordered else 0.0,
        "max_feature": ordered[0]["feature"] if ordered else None,
        "max_contrast": ordered[0]["contrast"] if ordered else None,
        "max_abs_smd_by_role": dict(sorted(by_role.items())),
        "top": ordered[:25],
        "all": ordered,
    }


def _balanced_outer_fold_map(
    groups: Sequence[str],
    strata: Sequence[str],
    *,
    n_folds: int,
    seed: int,
) -> tuple[dict[str, int], list[dict[str, int]]]:
    """Deterministically distribute whole blocks within chromosome/strand strata."""

    if len(groups) != len(strata):
        raise ValueError("groups and strata must align")
    group_stratum: dict[str, str] = {}
    for group, stratum in zip(groups, strata):
        previous = group_stratum.setdefault(str(group), str(stratum))
        if previous != str(stratum):
            raise ValueError("all cells of a block must share one chromosome/strand stratum")
    by_stratum: dict[str, list[str]] = {}
    for group, stratum in group_stratum.items():
        by_stratum.setdefault(stratum, []).append(group)
    fold_map: dict[str, int] = {}
    fold_counts = [Counter() for _ in range(n_folds)]
    for stratum in sorted(by_stratum):
        ordered = sorted(
            by_stratum[stratum],
            key=lambda group: stable_digest(seed, stratum, group, namespace="pbc-fold-v1"),
        )
        offset = int(stable_digest(seed, stratum, namespace="pbc-fold-offset-v1"), 16) % n_folds
        for index, group in enumerate(ordered):
            fold = (offset + index) % n_folds
            fold_map[group] = fold
            fold_counts[fold][stratum] += 1
    return fold_map, [dict(sorted(count.items())) for count in fold_counts]


@dataclass(frozen=True)
class _NestedAUCDesign:
    """Label-independent arrays reused by the 99 pseudo-label fits."""

    names: tuple[str, ...]
    matrix: np.ndarray
    group_array: np.ndarray
    stratum_array: np.ndarray
    unique_groups: np.ndarray
    row_group_codes: np.ndarray
    groups_by_stratum: tuple[tuple[str, np.ndarray, np.ndarray], ...]


def _prepare_nested_auc_design(
    feature_rows: Sequence[Mapping[str, float]],
    groups: Sequence[str],
    strata: Sequence[str] | None,
) -> _NestedAUCDesign:
    """Materialize the frozen feature matrix and block indexes exactly once."""

    if not feature_rows or len(feature_rows) != len(groups):
        raise ValueError("AUC inputs must be non-empty and aligned")
    names = tuple(sorted(set().union(*(row.keys() for row in feature_rows))))
    matrix = np.asarray(
        [[float(row.get(name, 0.0)) for name in names] for row in feature_rows],
        dtype=np.float64,
    )
    group_array = np.asarray(groups, dtype=object)
    stratum_array = np.asarray(
        list(strata) if strata is not None else ["all"] * len(groups), dtype=object
    )
    if len(stratum_array) != len(group_array):
        raise ValueError("strata must align to groups")
    unique_groups = np.unique(group_array)
    group_code = {str(group): index for index, group in enumerate(unique_groups)}
    row_group_codes = np.asarray(
        [group_code[str(group)] for group in group_array], dtype=np.int32
    )
    groups_by_stratum_raw: dict[str, list[str]] = {}
    for group in unique_groups:
        positions = np.flatnonzero(group_array == group)
        group_strata = np.unique(stratum_array[positions])
        if len(group_strata) != 1:
            raise ValueError("all cells of a block must share one chromosome/strand stratum")
        groups_by_stratum_raw.setdefault(str(group_strata[0]), []).append(str(group))
    groups_by_stratum = tuple(
        (
            stratum,
            np.asarray(groups_by_stratum_raw[stratum], dtype=object),
            np.asarray(
                [group_code[group] for group in groups_by_stratum_raw[stratum]],
                dtype=np.int32,
            ),
        )
        for stratum in sorted(groups_by_stratum_raw)
    )
    return _NestedAUCDesign(
        names=names,
        matrix=matrix,
        group_array=group_array,
        stratum_array=stratum_array,
        unique_groups=unique_groups,
        row_group_codes=row_group_codes,
        groups_by_stratum=groups_by_stratum,
    )


def _draw_group_bootstrap_counts(
    design: _NestedAUCDesign,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Replay the legacy draw order while retaining only group multiplicities."""

    rng = np.random.default_rng(int(seed))
    counts = np.zeros((int(n_boot), len(design.unique_groups)), dtype=np.int32)
    for draw_index in range(int(n_boot)):
        for _stratum, stratum_groups, stratum_group_codes in design.groups_by_stratum:
            # `choice` receives the same sorted object array, size, and replacement
            # flag as the original implementation, in the same draw-major order.
            sampled = rng.choice(
                stratum_groups, size=len(stratum_groups), replace=True
            )
            sampled_codes = stratum_group_codes[
                np.searchsorted(stratum_groups, sampled)
            ]
            np.add.at(counts[draw_index], sampled_codes, 1)
    return counts


def _auc_from_group_counts(
    labels: np.ndarray,
    predictions: np.ndarray,
    row_group_codes: np.ndarray,
    counts: np.ndarray,
    *,
    chunk_size: int = _AUC_BOOTSTRAP_CHUNK_SIZE,
) -> np.ndarray:
    """Vectorized weighted Mann-Whitney AUC for whole-block resamples.

    Repeating a block `k` times is exactly equivalent to giving each of its
    rows integer weight `k`.  Tied prediction values receive the same one-half
    credit as ``sklearn.metrics.roc_auc_score``.
    """

    order = np.argsort(predictions, kind="mergesort")
    ordered_predictions = predictions[order]
    ordered_labels = labels[order].astype(np.float64, copy=False)
    ordered_group_codes = row_group_codes[order]
    score_starts = np.r_[0, np.flatnonzero(np.diff(ordered_predictions)) + 1]
    output = np.empty(len(counts), dtype=np.float64)
    for start in range(0, len(counts), int(chunk_size)):
        stop = min(start + int(chunk_size), len(counts))
        row_weights = counts[start:stop, ordered_group_codes].astype(
            np.float64, copy=False
        )
        positive_by_score = np.add.reduceat(
            row_weights * ordered_labels, score_starts, axis=1
        )
        negative_by_score = np.add.reduceat(
            row_weights * (1.0 - ordered_labels), score_starts, axis=1
        )
        negative_below = np.cumsum(negative_by_score, axis=1) - negative_by_score
        numerator = np.sum(
            positive_by_score * (negative_below + 0.5 * negative_by_score), axis=1
        )
        denominator = positive_by_score.sum(axis=1) * negative_by_score.sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError("bootstrap AUC resample contains only one label class")
        output[start:stop] = numerator / denominator
    return output


def _reference_group_bootstrap_auc(
    design: _NestedAUCDesign,
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Pre-optimization bootstrap retained only for threshold-edge fallback/tests."""

    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(n_boot), dtype=np.float64)
    by_group = {
        str(group): np.flatnonzero(design.group_array == group)
        for group in design.unique_groups
    }
    for draw_index in range(int(n_boot)):
        sampled: list[str] = []
        for _stratum, stratum_groups, _codes in design.groups_by_stratum:
            sampled.extend(
                rng.choice(
                    stratum_groups, size=len(stratum_groups), replace=True
                ).tolist()
            )
        positions = np.concatenate([by_group[str(group)] for group in sampled])
        bootstrap[draw_index] = float(
            roc_auc_score(labels[positions], predictions[positions])
        )
    return bootstrap


def _bootstrap_auc(
    design: _NestedAUCDesign,
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    counts = _draw_group_bootstrap_counts(design, n_boot=n_boot, seed=seed)
    bootstrap = _auc_from_group_counts(
        labels, predictions, design.row_group_codes, counts
    )
    bounds = np.quantile(bootstrap, (0.025, 0.975))
    # The vectorized weighted-rank calculation and sklearn's repeated-row ROC
    # calculation agree to floating tolerance.  At a frozen decision boundary,
    # replay the prior implementation so even a last-bit difference cannot flip
    # a 0.50-containment, 0.55-upper, or 0.70-lower gate.
    if any(
        abs(float(bound) - threshold) <= _AUC_BOOTSTRAP_EQUIVALENCE_ATOL
        for bound in bounds
        for threshold in _AUC_FROZEN_GATE_THRESHOLDS
    ):
        return _reference_group_bootstrap_auc(
            design, labels, predictions, n_boot=n_boot, seed=seed
        )
    return bootstrap


def _nested_crossfit_auc_prepared(
    design: _NestedAUCDesign,
    labels: Sequence[bool | int],
    *,
    seed: int,
    outer_folds: int,
    inner_folds: int,
    n_boot: int,
    c_grid: Sequence[float],
    executor: ThreadPoolExecutor | None = None,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    label_array = np.asarray(labels, dtype=np.int8)
    if len(label_array) != len(design.group_array):
        raise ValueError("AUC inputs must be non-empty and aligned")
    outer_count = int(outer_folds)
    if outer_count != 5:
        raise ValueError("the frozen classifier requires exactly five outer folds")
    if len(design.unique_groups) < outer_count:
        return {
            "evaluable": False,
            "reason": "at least five independent blocks are required",
            "n_groups": int(len(design.unique_groups)),
        }
    fold_map, fold_stratum_counts = _balanced_outer_fold_map(
        design.group_array.tolist(),
        design.stratum_array.tolist(),
        n_folds=outer_count,
        seed=seed,
    )
    row_folds = np.asarray(
        [fold_map[str(group)] for group in design.group_array], dtype=np.int8
    )

    def fit_outer_fold(fold: int) -> tuple[int, np.ndarray, np.ndarray, dict[str, Any]]:
        test = np.flatnonzero(row_folds == fold)
        train = np.flatnonzero(row_folds != fold)
        if test.size == 0:
            raise ValueError(f"outer fold {fold} is empty")
        train_groups = design.group_array[train]
        inner_count = min(int(inner_folds), len(np.unique(train_groups)))
        if inner_count < 2:
            raise ValueError("nested training split has fewer than two blocks")
        scores: dict[float, list[float]] = {float(c): [] for c in c_grid}
        inner = GroupKFold(n_splits=inner_count)
        for inner_train_rel, inner_val_rel in inner.split(
            design.matrix[train], label_array[train], train_groups
        ):
            inner_train, inner_val = train[inner_train_rel], train[inner_val_rel]
            scaler = StandardScaler().fit(design.matrix[inner_train])
            train_x = scaler.transform(design.matrix[inner_train])
            val_x = scaler.transform(design.matrix[inner_val])
            for c_value in c_grid:
                classifier = LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=int(seed),
                    solver="liblinear",
                ).fit(train_x, label_array[inner_train])
                probability = classifier.predict_proba(val_x)[:, 1]
                scores[float(c_value)].append(
                    float(roc_auc_score(label_array[inner_val], probability))
                )
        selected_c = max(
            scores,
            key=lambda c_value: (float(np.mean(scores[c_value])), -float(c_value)),
        )
        scaler = StandardScaler().fit(design.matrix[train])
        classifier = LogisticRegression(
            C=selected_c,
            class_weight="balanced",
            max_iter=2000,
            random_state=int(seed + fold),
            solver="liblinear",
        ).fit(scaler.transform(design.matrix[train]), label_array[train])
        fold_predictions = classifier.predict_proba(
            scaler.transform(design.matrix[test])
        )[:, 1]
        record = {
            "fold": fold,
            "selected_c": selected_c,
            "inner_auc_by_c": {
                str(c): float(np.mean(values)) for c, values in sorted(scores.items())
            },
            "n_train_blocks": int(len(np.unique(design.group_array[train]))),
            "n_test_blocks": int(len(np.unique(design.group_array[test]))),
            "test_stratum_blocks": fold_stratum_counts[fold],
        }
        return fold, test, fold_predictions, record

    owns_executor = executor is None and _AUC_OUTER_WORKERS > 1
    local_executor = executor
    if owns_executor:
        local_executor = ThreadPoolExecutor(
            max_workers=min(_AUC_OUTER_WORKERS, os.cpu_count() or 1)
        )
    try:
        if local_executor is None:
            fold_outputs = [fit_outer_fold(fold) for fold in range(outer_count)]
        else:
            fold_outputs = list(local_executor.map(fit_outer_fold, range(outer_count)))
    except ValueError as exc:
        if str(exc).startswith("outer fold ") or str(exc).startswith("nested training split"):
            return {"evaluable": False, "reason": str(exc)}
        raise
    finally:
        if owns_executor and local_executor is not None:
            local_executor.shutdown()
    predictions = np.full(len(label_array), np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for _fold, test, fold_predictions, record in fold_outputs:
        predictions[test] = fold_predictions
        fold_records.append(record)
    if not np.all(np.isfinite(predictions)):
        raise AssertionError("cross-fit predictions were not complete")
    estimate = float(roc_auc_score(label_array, predictions))
    bootstrap = _bootstrap_auc(
        design, label_array, predictions, n_boot=int(n_boot), seed=int(seed)
    )
    return {
        "evaluable": True,
        "orientation_free": False,
        "estimate": float(estimate),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "n_rows": len(label_array),
        "n_groups": len(design.unique_groups),
        "n_features": len(design.names),
        "outer_folds": outer_count,
        "inner_folds": int(inner_folds),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "folds": fold_records,
        "outer_fold_stratum_counts": fold_stratum_counts,
        "outer_folds_stratum_balanced": all(
            max(counts) - min(counts) <= 1
            for stratum in sorted(
                {key for fold_count in fold_stratum_counts for key in fold_count}
            )
            for counts in [
                [fold_count.get(stratum, 0) for fold_count in fold_stratum_counts]
            ]
        ),
        "prediction_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
    }


def nested_crossfit_auc(
    feature_rows: Sequence[Mapping[str, float]],
    labels: Sequence[bool | int],
    groups: Sequence[str],
    strata: Sequence[str] | None = None,
    *,
    seed: int,
    outer_folds: int = 5,
    inner_folds: int = 4,
    n_boot: int = 2000,
    c_grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
) -> dict[str, Any]:
    """Nested group-cross-fit distinguishability with a block bootstrap bound."""

    if not feature_rows or len(feature_rows) != len(labels) or len(labels) != len(groups):
        raise ValueError("AUC inputs must be non-empty and aligned")
    design = _prepare_nested_auc_design(feature_rows, groups, strata)
    return _nested_crossfit_auc_prepared(
        design,
        labels,
        seed=seed,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        n_boot=n_boot,
        c_grid=c_grid,
    )


def balance_audit(
    cells: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[bool | int] | None = None,
    seed: int,
    smd_threshold: float = 0.10,
    auc_upper_threshold: float = 0.55,
    outer_folds: int = 5,
    inner_folds: int = 4,
    auc_n_boot: int = 2000,
    extra_feature: Sequence[float] | None = None,
    _auc_design: _NestedAUCDesign | None = None,
    _auc_executor: ThreadPoolExecutor | None = None,
) -> dict[str, Any]:
    feature_rows = [dict(cell["covariates"]) for cell in cells]
    if extra_feature is not None:
        if len(extra_feature) != len(feature_rows):
            raise ValueError("extra feature must align to cells")
        for row, value in zip(feature_rows, extra_feature):
            row["calibration.planted_label"] = float(value)
    chosen_labels = list(labels) if labels is not None else [bool(cell["phase_match"]) for cell in cells]
    weights = [1.0 / (3.0 if bool(label) else 6.0) for label in chosen_labels]
    groups = [str(cell["block_id"]) for cell in cells]
    strata = [f"{cell['chrom']}|{cell['strand']}" for cell in cells]
    smd = (
        three_delta_smd_audit(cells)
        if labels is None and extra_feature is None
        else standardized_mean_differences(feature_rows, chosen_labels, weights)
    )
    design = _auc_design or _prepare_nested_auc_design(feature_rows, groups, strata)
    auc = _nested_crossfit_auc_prepared(
        design,
        chosen_labels,
        seed=seed,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        n_boot=auc_n_boot,
        c_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        executor=_auc_executor,
    )
    return {
        "smd": smd,
        "auc": auc,
        "thresholds": {
            "max_abs_smd": float(smd_threshold),
            "auc_upper_95": float(auc_upper_threshold),
        },
        "gates": {
            "max_smd": bool(smd["max_abs_smd"] <= smd_threshold),
            "nested_crossfit_auc": bool(
                auc.get("evaluable")
                and auc.get("outer_folds_stratum_balanced")
                and auc.get("ci_high", math.inf) <= auc_upper_threshold
            ),
        },
        "weight_audit": {
            "matched_total": float(
                sum(weight for weight, label in zip(weights, chosen_labels) if bool(label))
            ),
            "mismatched_total": float(
                sum(weight for weight, label in zip(weights, chosen_labels) if not bool(label))
            ),
            "minimum": float(min(weights)),
            "maximum": float(max(weights)),
        },
    }


def _pseudo_labels(
    cells: Sequence[Mapping[str, Any]], *, seed: int, assignment: int = 0
) -> list[bool]:
    by_block: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        by_block.setdefault(str(cell["block_id"]), []).append(index)
    labels = [False] * len(cells)
    permutations = list(itertools.permutations((0, 1, 2)))
    for block_index, block_id in enumerate(sorted(by_block)):
        # Cycle all six balanced permutations, with a stable seed-dependent offset.
        offset = int(
            stable_digest(seed, assignment, block_id, namespace="pbc-pseudo-label-v1"), 16
        ) % 6
        permutation = permutations[(block_index + offset + assignment) % 6]
        for index in by_block[block_id]:
            cell = cells[index]
            labels[index] = int(cell["donor_phase"]) == permutation[int(cell["acceptor_phase"])]
    return labels


def calibration_hooks(
    cells: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    outer_folds: int,
    inner_folds: int,
    auc_n_boot: int,
    auc_upper_threshold: float,
    pseudo_family_size: int = 99,
    _auc_design: _NestedAUCDesign | None = None,
) -> dict[str, Any]:
    observed_labels = [bool(cell["phase_match"]) for cell in cells]
    feature_rows = [dict(cell["covariates"]) for cell in cells]
    groups = [str(cell["block_id"]) for cell in cells]
    strata = [f"{cell['chrom']}|{cell['strand']}" for cell in cells]
    base_design = _auc_design or _prepare_nested_auc_design(
        feature_rows, groups, strata
    )
    max_workers = min(_AUC_OUTER_WORKERS, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        planted = balance_audit(
            cells,
            labels=observed_labels,
            seed=seed + 101,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            auc_n_boot=auc_n_boot,
            extra_feature=[float(label) for label in observed_labels],
            _auc_executor=executor,
        )
        primary_pseudo_labels = _pseudo_labels(cells, seed=seed, assignment=0)
        pseudo = balance_audit(
            cells,
            labels=primary_pseudo_labels,
            seed=seed + 202,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            auc_n_boot=auc_n_boot,
            auc_upper_threshold=auc_upper_threshold,
            _auc_design=base_design,
            _auc_executor=executor,
        )
        pseudo_results = []
        for assignment in range(int(pseudo_family_size)):
            labels = _pseudo_labels(cells, seed=seed, assignment=assignment)
            result = _nested_crossfit_auc_prepared(
                base_design,
                labels,
                seed=seed + 1000 + assignment,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                n_boot=auc_n_boot,
                c_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
                executor=executor,
            )
            pseudo_results.append(
                {
                    "assignment": int(assignment),
                    "estimate": float(result.get("estimate", math.inf)),
                    "ci_low": float(result.get("ci_low", -math.inf)),
                    "ci_high": float(result.get("ci_high", math.inf)),
                    "same_ci_gate_passed": bool(
                        result.get("evaluable")
                        and result.get("outer_folds_stratum_balanced")
                        and float(result.get("ci_high", math.inf))
                        <= auc_upper_threshold
                    ),
                }
            )
    false_positive_count = sum(
        not row["same_ci_gate_passed"] for row in pseudo_results
    )
    family_fpr = false_positive_count / max(1, int(pseudo_family_size))
    planted_detected = bool(
        planted["smd"]["max_abs_smd"] > 0.10
        and planted["auc"].get("evaluable")
        and planted["auc"].get("ci_low", 0.0) >= 0.70
    )
    pseudo_rejected = bool(
        pseudo["auc"].get("evaluable")
        and pseudo["auc"].get("ci_low", math.inf) <= 0.50
        and pseudo["auc"].get("ci_high", -math.inf) >= 0.50
        and family_fpr <= 0.05
    )
    return {
        "planted_imbalance": planted,
        "balanced_pseudo_label": pseudo,
        "pseudo_label_family": {
            "n": int(pseudo_family_size),
            "auc_false_positive_threshold": float(auc_upper_threshold),
            "false_positive_count": int(false_positive_count),
            "family_fpr": float(family_fpr),
            "decision": "same_nested_crossfit_and_bootstrap_ci_gate_as_primary",
            "bootstrap_draws_per_assignment": int(auc_n_boot),
            "assignments": pseudo_results,
        },
        "gates": {
            "planted_imbalance_detected": planted_detected,
            "permuted_labels_rejected": pseudo_rejected,
        },
    }


def _source_strata(
    blocks: Sequence[Mapping[str, Any]],
    *,
    allowed_chromosomes: Iterable[str] | None = None,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if allowed_chromosomes is not None:
        for role in ("donor", "acceptor"):
            for phase in (0, 1, 2):
                for chrom in sorted({normalize_chromosome(value) for value in allowed_chromosomes}):
                    for strand in ("+", "-"):
                        counts[f"{role}.phase{phase}.chrom.{chrom}.strand.{strand}"] = 0
    for block in blocks:
        for role in ("donor", "acceptor"):
            for phase in (0, 1, 2):
                counts[f"{role}.phase{phase}"] += 1
                source = block[f"{role}_modules"][str(phase)]
                counts[
                    f"{role}.phase{phase}.chrom.{source['chrom']}.strand.{source['strand']}"
                ] += 1
    return dict(sorted(counts.items()))


def _source_strata_complete(
    counts: Mapping[str, int],
    *,
    expected_blocks: int,
    allowed_chromosomes: Iterable[str] | None = None,
    minimum_per_source_stratum: int = 0,
) -> bool:
    base = {
        f"{role}.phase{phase}"
        for role in ("donor", "acceptor")
        for phase in (0, 1, 2)
    }
    expected_detail = (
        {
            f"{role}.phase{phase}.chrom.{normalize_chromosome(chrom)}.strand.{strand}"
            for role in ("donor", "acceptor")
            for phase in (0, 1, 2)
            for chrom in allowed_chromosomes
            for strand in ("+", "-")
        }
        if allowed_chromosomes is not None
        else {label for label in counts if ".chrom." in label and ".strand." in label}
    )
    return bool(
        all(int(counts.get(label, -1)) == int(expected_blocks) for label in base)
        and bool(expected_detail)
        and all(label in counts for label in expected_detail)
        and all(
            int(counts.get(label, 0)) >= int(minimum_per_source_stratum)
            for label in expected_detail
        )
        and sum(
            int(value) for label, value in counts.items() if ".chrom." in label
        )
        == 6 * int(expected_blocks)
    )


def _phase_cell_counts(cells: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        f"d{int(cell['donor_phase'])}.a{int(cell['acceptor_phase'])}" for cell in cells
    )
    return dict(sorted(counts.items()))


def _candidate_reference(candidate: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_id",
        "role",
        "gene_id",
        "gene_id_versioned",
        "transcript_id",
        "chrom",
        "strand",
        "source_start",
        "source_end",
        "source_exon_length",
        "sequence_sha256",
        "eligibility_checks",
    )
    reference = {key: candidate.get(key) for key in keys}
    if candidate["role"] != "scaffold":
        reference.update(
            {
                "phase": int(candidate["phase"]),
                "boundary_index": int(candidate["boundary_index"]),
                "coding_offset": int(candidate["coding_offset"]),
                "annotation": candidate["annotation"],
                "scores": candidate["scores"],
                "geometry": candidate["geometry"],
                "native_sequence_sha256": candidate["native_sequence_sha256"],
                "native_donor_pos": int(candidate["native_donor_pos"]),
                "native_acceptor_pos": int(candidate["native_acceptor_pos"]),
            }
        )
    else:
        reference.update(
            {
                "donor_window": list(candidate["donor_window"]),
                "acceptor_window": list(candidate["acceptor_window"]),
                "donor_pos": int(candidate["donor_pos"]),
                "acceptor_pos": int(candidate["acceptor_pos"]),
                "bridge_acceptor_pos": int(candidate["bridge_acceptor_pos"]),
                "donor_anchor_covariates": candidate["donor_anchor_covariates"],
                "acceptor_anchor_covariates": candidate["acceptor_anchor_covariates"],
            }
        )
    return reference


def _add_block_stop_excess(cells: list[dict[str, Any]]) -> None:
    match_counts = [
        float(cell["covariates"]["pair.joined_register0_stop_count"])
        for cell in cells
        if cell["delta_class"] == "M0"
    ]
    baseline = float(np.mean(match_counts))
    for cell in cells:
        count = float(cell["covariates"]["pair.joined_register0_stop_count"])
        cell["covariates"]["pair.stop_excess_vs_block_match"] = count - baseline
        cell["covariates"] = dict(sorted(cell["covariates"].items()))


def _proposal_imbalance_objective(cells: Sequence[Mapping[str, Any]]) -> float:
    """Complete-feature, three-contrast objective used before global selection."""

    names = sorted(set().union(*(cell["covariates"].keys() for cell in cells)))
    classes = np.asarray([cell["delta_class"] for cell in cells], dtype=object)
    maximum = 0.0
    for name in names:
        values = np.asarray([float(cell["covariates"].get(name, 0.0)) for cell in cells])
        scale = max(float(values.std()), 1e-8)
        means = {label: float(values[classes == label].mean()) for label in ("M0", "Mplus", "Mminus")}
        maximum = max(
            maximum,
            abs(means["M0"] - means["Mplus"]) / scale,
            abs(means["M0"] - means["Mminus"]) / scale,
            abs(means["Mplus"] - means["Mminus"]) / scale,
        )
    return float(maximum)


_CAPACITY_SLOT_ORDER = (
    "scaffold",
    "donor_phase_0",
    "donor_phase_1",
    "donor_phase_2",
    "acceptor_phase_0",
    "acceptor_phase_1",
    "acceptor_phase_2",
)


def _capacity_entry(gene_ids: Iterable[str], *, required: int) -> dict[str, Any]:
    """Return a compact exact certificate for one necessary gene-capacity constraint."""

    normalized = sorted({gene_root(gene_id) for gene_id in gene_ids})
    available = len(normalized)
    deficit = max(0, int(required) - available)
    return {
        "required_unique_genes": int(required),
        "available_unique_genes": int(available),
        "deficit_unique_genes": int(deficit),
        "passed": deficit == 0,
        "gene_ids_sha256": sha256_payload(normalized),
    }


def _necessary_capacity_report(
    scaffolds: Sequence[Mapping[str, Any]],
    module_pools: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    *,
    quotas: Mapping[str, int],
) -> dict[str, Any]:
    """Certify cheap necessary conditions for globally disjoint seven-gene blocks.

    Passing this certificate is deliberately not sufficient for construction:
    matching, block-role hashing, stratum supply, and balance may still fail.  A
    failed slot or union constraint, however, proves that the requested number
    of complete blocks cannot exist and should stop construction before the
    expensive matching and calibration path.
    """

    requested_blocks = sum(int(value) for value in quotas.values())
    slot_genes: dict[str, set[str]] = {
        "scaffold": {gene_root(str(row["gene_id"])) for row in scaffolds},
    }
    for role in ("donor", "acceptor"):
        for phase in (0, 1, 2):
            slot_genes[f"{role}_phase_{phase}"] = {
                gene_root(str(row["gene_id"])) for row in module_pools[role][phase]
            }
    slots = {
        name: _capacity_entry(slot_genes[name], required=requested_blocks)
        for name in _CAPACITY_SLOT_ORDER
    }
    family_specs = {
        "donor_all_phases": tuple(f"donor_phase_{phase}" for phase in (0, 1, 2)),
        "acceptor_all_phases": tuple(f"acceptor_phase_{phase}" for phase in (0, 1, 2)),
        "all_seven_source_roles": _CAPACITY_SLOT_ORDER,
    }
    families: dict[str, dict[str, Any]] = {}
    for name, members in family_specs.items():
        genes = set().union(*(slot_genes[member] for member in members))
        entry = _capacity_entry(genes, required=requested_blocks * len(members))
        entry["source_slots"] = list(members)
        families[name] = entry
    failed_constraints = [
        *(f"slot:{name}" for name in _CAPACITY_SLOT_ORDER if not slots[name]["passed"]),
        *(f"family:{name}" for name in family_specs if not families[name]["passed"]),
    ]
    return {
        "criterion": "necessary_unique_normalized_gene_capacity",
        "necessary_not_sufficient": True,
        "requested_complete_blocks": int(requested_blocks),
        "required_unique_source_genes": int(7 * requested_blocks),
        "slots": slots,
        "families": families,
        "failed_constraints": failed_constraints,
        "passed": not failed_constraints,
    }


def _construction_config(
    *,
    quotas: Mapping[str, int],
    minimum_confirmation_blocks: int,
    span_nt: int,
    construct_length: int,
    donor_pos: int,
    acceptor_pos: int,
    namespace: str,
    split_weights: Sequence[tuple[str, int]],
    allowed_chroms: set[str],
    forbidden_chroms: set[str],
    search_width: int,
    neighbor_width: int,
    global_search_width: int,
    seed: int,
    outer_folds: int,
    inner_folds: int,
    auc_n_boot: int,
    pseudo_family_size: int,
    minimum_stratum_blocks: int,
    smd_threshold: float,
    auc_upper_threshold: float,
) -> dict[str, Any]:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "namespace": namespace,
        "split_weights": [[name, int(weight)] for name, weight in split_weights],
        "requested_blocks_by_split": dict(quotas),
        "minimum_confirmation_blocks": int(minimum_confirmation_blocks),
        "span_nt": int(span_nt),
        "construct_length": int(construct_length),
        "donor_pos": int(donor_pos),
        "acceptor_pos": int(acceptor_pos),
        "allowed_chromosomes": sorted(allowed_chroms),
        "forbidden_chromosomes": sorted(forbidden_chroms),
        "search_width": int(search_width),
        "neighbor_width": int(neighbor_width),
        "global_search_width": int(global_search_width),
        "seed": int(seed),
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "auc_n_boot": int(auc_n_boot),
        "pseudo_family_size": int(pseudo_family_size),
        "minimum_stratum_blocks": int(minimum_stratum_blocks),
        "smd_threshold": float(smd_threshold),
        "auc_upper_threshold": float(auc_upper_threshold),
    }


def _capacity_stop_package(
    *,
    candidate_package: Mapping[str, Any],
    candidate_audit: Mapping[str, Any] | None,
    blacklist: set[str],
    frozen_variance: Mapping[str, Any],
    config: Mapping[str, Any],
    plan: SampleSizePlan,
    capacity: Mapping[str, Any],
    candidate_census_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a compact non-outcome artifact for a proved Phase-I supply stop."""

    candidate_package_sha256 = (
        str(candidate_audit["candidate_census_sha256"])
        if candidate_audit is not None
        else sha256_payload(candidate_package)
    )
    if candidate_audit is not None and candidate_census_receipt is None:
        raise ValueError("production capacity stop requires the candidate census receipt")
    quotas = {
        str(role): int(value)
        for role, value in config["requested_blocks_by_split"].items()
    }
    core: dict[str, Any] = {
        "schema": CAPACITY_STOP_SCHEMA,
        "stage": "construction",
        "status": "PHASE1_INSTRUMENT_UNRESOLVED",
        "disposition": "NECESSARY_CAPACITY_FAILED",
        "model_outcomes_read": False,
        "construction_gates_pass": False,
        "independent_audit_required": True,
        "target_execution_authorized": False,
        "selection_attempted": False,
        "balance_or_calibration_evaluated": False,
        "partial_blocks_recorded": False,
        "clean_checkout_reexecutable": False,
        "candidate_census_receipt": (
            None if candidate_census_receipt is None else dict(candidate_census_receipt)
        ),
        "config": dict(config),
        "sample_size_plan": plan.to_dict(),
        "blacklist": {
            "n_gene_ids": len(blacklist),
            "gene_ids_sha256": sha256_payload(sorted(blacklist)),
        },
        "counts": {
            "blocks": 0,
            "cells": 0,
            "genes": 0,
            "blocks_by_split": {role: 0 for role in quotas},
        },
        "capacity": dict(capacity),
        "gates": {"necessary_unique_gene_capacity": False},
        "unevaluated_construction_gates": [
            "block_assembly",
            "role_hash_supply",
            "chromosome_strand_strata",
            "composition_balance",
            "classifier_calibration",
        ],
        "model_execution": {
            "checkpoint_loaded": False,
            "forward_passes": 0,
            "outcome_rows": 0,
            "all_model_gates": "NOT_EVALUATED",
        },
        "unevaluated_model_stages": [
            "phase2_carrier_factorial_calibration",
            "phase2_model_preflight",
            "phase2_target",
            "phase3_register_preflight",
            "phase3_candidate_screen",
            "phase3_reserve",
            "phase3_mediation",
        ],
        "blocks": [],
        "cells": [],
        "evidence_boundary": (
            "This artifact proves only that the frozen Phase-I block request cannot be met "
            "with globally unique eligible genes. No SpliceBERT checkpoint or model outcome "
            "was loaded, and no biological or model-mechanism result is licensed."
        ),
    }
    core["seal"] = {
        "candidate_package_sha256": candidate_package_sha256,
        "candidate_audit_sha256": (
            None if candidate_audit is None else sha256_payload(candidate_audit)
        ),
        "frozen_variance_sha256": sha256_payload(frozen_variance),
        "blacklist_sha256": sha256_payload(sorted(blacklist)),
        "capacity_sha256": sha256_payload(capacity),
        "candidate_census_receipt_sha256": (
            None
            if candidate_census_receipt is None
            else sha256_payload(candidate_census_receipt)
        ),
    }
    core["seal"]["content_sha256"] = sha256_payload(
        {key: value for key, value in core.items() if key != "seal"}
    )
    return core


def build_composition_balanced_cohort(
    candidate_package: Mapping[str, Any],
    *,
    candidate_audit: Mapping[str, Any] | None = None,
    candidate_census_receipt: Mapping[str, Any] | None = None,
    blacklisted_gene_ids: Iterable[str],
    frozen_variance: Mapping[str, Any],
    requested_blocks_by_split: Mapping[str, int] | None = None,
    minimum_confirmation_blocks: int = 600,
    span_nt: int = DEFAULT_SPAN_NT,
    construct_length: int = DEFAULT_CONSTRUCT_LENGTH,
    donor_pos: int = DEFAULT_DONOR_POS,
    acceptor_pos: int = DEFAULT_ACCEPTOR_POS,
    namespace: str = NAMESPACE,
    split_weights: Sequence[tuple[str, int]] = DEFAULT_SPLIT_WEIGHTS,
    allowed_chromosomes: Iterable[str] = DEFAULT_ALLOWED_CHROMS,
    forbidden_chromosomes: Iterable[str] = DEFAULT_FORBIDDEN_CHROMS,
    search_width: int = 64,
    neighbor_width: int = 8,
    global_search_width: int = 16,
    seed: int = 73031,
    outer_folds: int = 5,
    inner_folds: int = 4,
    auc_n_boot: int = 2000,
    pseudo_family_size: int = 99,
    minimum_stratum_blocks: int = 5,
    smd_threshold: float = 0.10,
    auc_upper_threshold: float = 0.55,
) -> dict[str, Any]:
    """Jointly construct all role-hashed cohorts with global gene disjointness."""

    role_names = {name for name, _weight in split_weights}
    quotas = dict(DEFAULT_BLOCK_COUNTS if requested_blocks_by_split is None else requested_blocks_by_split)
    if set(quotas) - role_names or not quotas or any(int(value) < 0 for value in quotas.values()):
        raise ValueError("requested_blocks_by_split must use non-negative frozen role names")
    quotas = {str(name): int(quotas.get(name, 0)) for name, _weight in split_weights}
    if span_nt <= 0 or search_width <= 0 or neighbor_width <= 0 or global_search_width <= 0:
        raise ValueError("span and search widths must be positive")
    if acceptor_pos - donor_pos != span_nt:
        raise ValueError("acceptor_pos-donor_pos must equal span_nt")
    candidates = validate_candidate_package(candidate_package)
    validate_candidate_audit_binding(candidate_package, candidate_audit)
    blacklist = {gene_root(gene) for gene in blacklisted_gene_ids}
    allowed_chroms = {normalize_chromosome(chrom) for chrom in allowed_chromosomes}
    forbidden_chroms = {normalize_chromosome(chrom) for chrom in forbidden_chromosomes}
    if allowed_chroms & forbidden_chroms:
        raise ValueError("allowed and forbidden chromosome sets overlap")
    plan = plan_sample_size(frozen_variance, minimum_blocks=minimum_confirmation_blocks)
    filtered = [
        candidate
        for candidate in candidates
        if candidate["gene_id"] not in blacklist
        and candidate["chrom"] in allowed_chroms
        and candidate["chrom"] not in forbidden_chroms
    ]
    scaffolds = [
        candidate
        for candidate in filtered
        if candidate["role"] == "scaffold"
        and len(candidate["sequence"]) == int(construct_length)
        and int(candidate["donor_pos"]) == int(donor_pos)
        and int(candidate["acceptor_pos"]) == int(acceptor_pos)
        and tuple(candidate["donor_window"]) == DEFAULT_DONOR_WINDOW
        and tuple(candidate["acceptor_window"]) == DEFAULT_ACCEPTOR_WINDOW
    ]
    module_pools = {
        role: {
            phase: [
                candidate
                for candidate in filtered
                if candidate["role"] == role and int(candidate["phase"]) == phase
            ]
            for phase in (0, 1, 2)
        }
        for role in ("donor", "acceptor")
    }
    config = _construction_config(
        quotas=quotas,
        minimum_confirmation_blocks=minimum_confirmation_blocks,
        span_nt=span_nt,
        construct_length=construct_length,
        donor_pos=donor_pos,
        acceptor_pos=acceptor_pos,
        namespace=namespace,
        split_weights=split_weights,
        allowed_chroms=allowed_chroms,
        forbidden_chroms=forbidden_chroms,
        search_width=search_width,
        neighbor_width=neighbor_width,
        global_search_width=global_search_width,
        seed=seed,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        auc_n_boot=auc_n_boot,
        pseudo_family_size=pseudo_family_size,
        minimum_stratum_blocks=minimum_stratum_blocks,
        smd_threshold=smd_threshold,
        auc_upper_threshold=auc_upper_threshold,
    )
    capacity = _necessary_capacity_report(scaffolds, module_pools, quotas=quotas)
    if not capacity["passed"]:
        return _capacity_stop_package(
            candidate_package=candidate_package,
            candidate_audit=candidate_audit,
            blacklist=blacklist,
            frozen_variance=frozen_variance,
            config=config,
            plan=plan,
            capacity=capacity,
            candidate_census_receipt=candidate_census_receipt,
        )
    donor_all = [candidate for phase in (0, 1, 2) for candidate in module_pools["donor"][phase]]
    acceptor_all = [candidate for phase in (0, 1, 2) for candidate in module_pools["acceptor"][phase]]
    feature_cache: dict[str, dict[str, float]] = {}
    distance_cache: dict[tuple[str, str], float] = {}
    donor_means, donor_scales = _feature_scale(donor_all, feature_cache)
    acceptor_means, acceptor_scales = _feature_scale(acceptor_all, feature_cache)
    vector_cache: dict[str, np.ndarray] = {}
    for role_candidates, means, scales in (
        (donor_all, donor_means, donor_scales),
        (acceptor_all, acceptor_means, acceptor_scales),
    ):
        names = tuple(means)
        for candidate in role_candidates:
            features = feature_cache[str(candidate["candidate_id"])]
            vector_cache[str(candidate["candidate_id"])] = np.asarray(
                [
                    (features.get(name, means[name]) - means[name]) / scales[name]
                    for name in names
                ],
                dtype=np.float64,
            )
    remaining_scaffolds = sorted(
        scaffolds,
        key=lambda candidate: (
            stable_digest(candidate["candidate_id"], namespace=namespace),
            candidate["candidate_id"],
        ),
    )
    used_genes: set[str] = set()
    blocks: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    while any(role_counts[role] < quota for role, quota in quotas.items()):
        proposals: list[tuple[tuple[float, float, str], dict[str, Any]]] = []
        discard_ids: set[str] = set()
        for scaffold in remaining_scaffolds:
            if len(proposals) >= global_search_width:
                break
            if scaffold["gene_id"] in used_genes:
                discard_ids.add(scaffold["candidate_id"])
                continue
            donor_selection = _select_phase_triplet(
                module_pools["donor"],
                used_genes=used_genes | {scaffold["gene_id"]},
                means=donor_means,
                scales=donor_scales,
                search_width=search_width,
                neighbor_width=neighbor_width,
                namespace=namespace,
                anchor_features=scaffold["donor_anchor_covariates"],
                feature_cache=feature_cache,
                distance_cache=distance_cache,
                vector_cache=vector_cache,
            )
            if donor_selection is None:
                discard_ids.add(scaffold["candidate_id"])
                continue
            donors, donor_score = donor_selection
            donor_genes = {candidate["gene_id"] for candidate in donors.values()}
            acceptor_selection = _select_phase_triplet(
                module_pools["acceptor"],
                used_genes=used_genes | {scaffold["gene_id"]} | donor_genes,
                means=acceptor_means,
                scales=acceptor_scales,
                search_width=search_width,
                neighbor_width=neighbor_width,
                namespace=namespace,
                anchor_features=scaffold["acceptor_anchor_covariates"],
                feature_cache=feature_cache,
                distance_cache=distance_cache,
                vector_cache=vector_cache,
            )
            if acceptor_selection is None:
                discard_ids.add(scaffold["candidate_id"])
                continue
            acceptors, acceptor_score = acceptor_selection
            source_genes = {
                scaffold["gene_id"],
                *donor_genes,
                *(candidate["gene_id"] for candidate in acceptors.values()),
            }
            if len(source_genes) != 7 or source_genes & used_genes:
                discard_ids.add(scaffold["candidate_id"])
                continue
            split_role = stable_block_split(source_genes, namespace=namespace, weights=split_weights)
            if role_counts[split_role] >= quotas.get(split_role, 0):
                continue
            source_ids = [scaffold["candidate_id"]] + [
                donors[phase]["candidate_id"] for phase in (0, 1, 2)
            ] + [acceptors[phase]["candidate_id"] for phase in (0, 1, 2)]
            digest = stable_digest(*source_ids, namespace=namespace)
            block_id = f"pbc-{split_role}-{role_counts[split_role]:04d}-{digest[:12]}"
            block_cells = [
                _cell_record(
                    block_id=block_id,
                    split_role=split_role,
                    scaffold=scaffold,
                    donor=donors[donor_phase],
                    acceptor=acceptors[acceptor_phase],
                )
                for donor_phase in (0, 1, 2)
                for acceptor_phase in (0, 1, 2)
            ]
            _add_block_stop_excess(block_cells)
            accumulated_role_cells = [
                cell for cell in cells if cell["split_role"] == split_role
            ]
            objective = _proposal_imbalance_objective(
                [*accumulated_role_cells, *block_cells]
            )
            proposals.append(
                (
                    (objective, max(float(donor_score), float(acceptor_score)), digest),
                    {
                        "scaffold": scaffold,
                        "donors": donors,
                        "acceptors": acceptors,
                        "donor_score": donor_score,
                        "acceptor_score": acceptor_score,
                        "source_genes": source_genes,
                        "source_ids": source_ids,
                        "digest": digest,
                        "split_role": split_role,
                        "block_id": block_id,
                        "cells": block_cells,
                        "objective": objective,
                    },
                )
            )
        remaining_scaffolds = [
            scaffold
            for scaffold in remaining_scaffolds
            if scaffold["candidate_id"] not in discard_ids
        ]
        if not proposals:
            break
        _key, selected = min(proposals, key=lambda item: item[0])
        scaffold = selected["scaffold"]
        donors = selected["donors"]
        acceptors = selected["acceptors"]
        split_role = selected["split_role"]
        source_genes = selected["source_genes"]
        block = {
            "block_id": selected["block_id"],
            "block_identity_sha256": stable_digest(
                *sorted(source_genes), namespace=f"{namespace}:block-identity"
            ),
            "split_role": split_role,
            "scaffold_id": scaffold["candidate_id"],
            "scaffold_gene_id": scaffold["gene_id"],
            "scaffold": _candidate_reference(scaffold),
            "chrom": scaffold["chrom"],
            "strand": scaffold["strand"],
            "donor_pos": int(scaffold["donor_pos"]),
            "acceptor_pos": int(scaffold["acceptor_pos"]),
            "span_nt": int(span_nt),
            "construct_length": int(construct_length),
            "seams": {
                "donor_window": list(scaffold["donor_window"]),
                "acceptor_window": list(scaffold["acceptor_window"]),
            },
            "donor_modules": {
                str(phase): _candidate_reference(donors[phase]) for phase in (0, 1, 2)
            },
            "acceptor_modules": {
                str(phase): _candidate_reference(acceptors[phase]) for phase in (0, 1, 2)
            },
            "source_gene_ids": sorted(source_genes),
            "matching": {
                "donor_scaffold_anchor_max_standardized_distance": float(selected["donor_score"]),
                "acceptor_scaffold_anchor_max_standardized_distance": float(selected["acceptor_score"]),
                "complete_feature_three_contrast_objective": float(selected["objective"]),
            },
        }
        blocks.append(block)
        cells.extend(selected["cells"])
        used_genes.update(source_genes)
        role_counts[split_role] += 1
        remaining_scaffolds = [
            candidate
            for candidate in remaining_scaffolds
            if candidate["candidate_id"] != scaffold["candidate_id"]
        ]
    if any(role_counts[role] != quota for role, quota in quotas.items()):
        raise CohortConstructionError(
            f"constructed role counts {dict(role_counts)}; required exact counts {quotas}"
        )
    phase_counts = _phase_cell_counts(cells)
    source_counts = _source_strata(blocks, allowed_chromosomes=allowed_chroms)
    phase_counts_by_split = {
        role: _phase_cell_counts([cell for cell in cells if cell["split_role"] == role])
        for role in quotas
        if quotas[role]
    }
    source_counts_by_split = {
        role: _source_strata(
            [block for block in blocks if block["split_role"] == role],
            allowed_chromosomes=allowed_chroms,
        )
        for role in quotas
        if quotas[role]
    }
    balance_by_split: dict[str, Any] = {}
    calibration_by_split: dict[str, Any] = {}
    for role, quota in quotas.items():
        if quota == 0:
            continue
        role_cells = [cell for cell in cells if cell["split_role"] == role]
        role_seed = seed + [name for name, _weight in split_weights].index(role)
        role_feature_rows = [dict(cell["covariates"]) for cell in role_cells]
        role_groups = [str(cell["block_id"]) for cell in role_cells]
        role_strata = [f"{cell['chrom']}|{cell['strand']}" for cell in role_cells]
        role_auc_design = _prepare_nested_auc_design(
            role_feature_rows, role_groups, role_strata
        )
        balance_by_split[role] = balance_audit(
            role_cells,
            seed=role_seed,
            smd_threshold=smd_threshold,
            auc_upper_threshold=auc_upper_threshold,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            auc_n_boot=auc_n_boot,
            _auc_design=role_auc_design,
        )
        if role in {"confirmation", "phase3_reserve"}:
            calibration_by_split[role] = calibration_hooks(
                role_cells,
                seed=role_seed,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                auc_n_boot=auc_n_boot,
                auc_upper_threshold=auc_upper_threshold,
                pseudo_family_size=pseudo_family_size,
                _auc_design=role_auc_design,
            )
    stratum_counts_by_split: dict[str, dict[str, int]] = {}
    expected_strata = [
        f"{chrom}|{strand}" for chrom in sorted(allowed_chroms) for strand in ("+", "-")
    ]
    for role in quotas:
        role_blocks = [block for block in blocks if block["split_role"] == role]
        observed = Counter(f"{block['chrom']}|{block['strand']}" for block in role_blocks)
        stratum_counts_by_split[role] = {
            stratum: int(observed.get(stratum, 0)) for stratum in expected_strata
        }
    exact_geometry = all(
        int(block["construct_length"]) == construct_length
        and int(block["donor_pos"]) == donor_pos
        and int(block["acceptor_pos"]) == acceptor_pos
        and int(block["span_nt"]) == span_nt
        for block in blocks
    )
    gates = {
        "exact_role_counts": dict(role_counts) == {role: count for role, count in quotas.items() if count},
        "confirmation_variance_plan": quotas.get("confirmation", 0) == plan.planned_blocks,
        "complete_3x3": len(cells) == 9 * len(blocks)
        and set(phase_counts.values()) == {len(blocks)}
        and all(set(counts.values()) == {quotas[role]} for role, counts in phase_counts_by_split.items()),
        "source_strata": _source_strata_complete(
            source_counts,
            expected_blocks=len(blocks),
            allowed_chromosomes=allowed_chroms,
        )
        and all(
            _source_strata_complete(
                counts,
                expected_blocks=quotas[role],
                allowed_chromosomes=allowed_chroms,
                minimum_per_source_stratum=(
                    minimum_stratum_blocks
                    if role in {"confirmation", "phase3_reserve"}
                    else 0
                ),
            )
            for role, counts in source_counts_by_split.items()
        ),
        "gene_disjoint": len(used_genes) == 7 * len(blocks),
        "hard_blacklist": not bool(used_genes & blacklist),
        "forbidden_chromosomes": not bool({block["chrom"] for block in blocks} & forbidden_chroms),
        "allowed_chromosomes": all(block["chrom"] in allowed_chroms for block in blocks),
        "exact_geometry": exact_geometry,
        "max_smd_all_roles": all(audit["gates"]["max_smd"] for audit in balance_by_split.values()),
        "classifier_confirmation_reserve": all(
            balance_by_split[role]["gates"]["nested_crossfit_auc"]
            for role in ("confirmation", "phase3_reserve")
            if quotas.get(role, 0)
        ),
        "stratum_capacity_confirmation_reserve": all(
            counts and min(counts.values()) >= minimum_stratum_blocks
            for role, counts in stratum_counts_by_split.items()
            if role in {"confirmation", "phase3_reserve"} and quotas.get(role, 0)
        ),
        "planted_imbalance_detected": all(
            audit["gates"]["planted_imbalance_detected"]
            for audit in calibration_by_split.values()
        ),
        "pseudo_family_complete": pseudo_family_size == 99,
        "permuted_labels_rejected": all(
            audit["gates"]["permuted_labels_rejected"] for audit in calibration_by_split.values()
        ),
    }
    construction_passed = bool(all(gates.values()))
    development_block_ids = sorted(
        block["block_id"] for block in blocks if block["split_role"] == "development"
    )
    smoke_ids = sorted(
        development_block_ids,
        key=lambda block_id: (
            stable_digest(block_id, namespace=SMOKE_NAMESPACE),
            block_id,
        ),
    )[: min(12, len(development_block_ids))]
    core = {
        "schema": COHORT_SCHEMA,
        "stage": "construction",
        "status": (
            "PHASE1_CONSTRUCTION_PASS_AUDIT_PENDING"
            if construction_passed
            else "PHASE1_INSTRUMENT_UNRESOLVED"
        ),
        "model_outcomes_read": False,
        "construction_gates_pass": construction_passed,
        "independent_audit_required": True,
        "target_execution_authorized": False,
        "config": config,
        "sample_size_plan": plan.to_dict(),
        "blacklist": {
            "n_gene_ids": len(blacklist),
            "gene_ids_sha256": sha256_payload(sorted(blacklist)),
        },
        "counts": {
            "blocks": len(blocks),
            "cells": len(cells),
            "genes": len(used_genes),
            "blocks_by_split": dict(sorted(role_counts.items())),
            "phase_cells_all_roles": phase_counts,
            "phase_cells_by_split": phase_counts_by_split,
            "source_role_phase": source_counts,
            "source_role_phase_by_split": source_counts_by_split,
            "chromosome_strand_by_split": stratum_counts_by_split,
        },
        "development_smoke": {
            "method": "sha256_rank_whole_block_identity",
            "namespace": SMOKE_NAMESPACE,
            "n_blocks": len(smoke_ids),
            "block_ids": smoke_ids,
            "block_ids_sha256": sha256_payload(smoke_ids),
            "not_an_independent_split": True,
        },
        "gates": gates,
        "balance_audit_by_split": balance_by_split,
        "calibrations_by_split": calibration_by_split,
        "blocks": blocks,
        "cells": cells,
    }
    feature_rows = [cell["covariates"] for cell in cells]
    core["seal"] = {
        "candidate_package_sha256": sha256_payload(candidate_package),
        "candidate_audit_sha256": (
            None if candidate_audit is None else sha256_payload(candidate_audit)
        ),
        "frozen_variance_sha256": sha256_payload(frozen_variance),
        "selection_sha256": sha256_payload(blocks),
        "cell_identity_sha256": sha256_payload(
            [
                {
                    "cell_id": cell["cell_id"],
                    "sequence_sha256": cell["sequence_sha256"],
                }
                for cell in cells
            ]
        ),
        "feature_matrix_sha256": sha256_payload(feature_rows),
    }
    core["seal"]["content_sha256"] = sha256_payload(
        {key: value for key, value in core.items() if key != "seal"}
    )
    return core


def _structural_audit(
    package: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    blacklist: set[str],
) -> dict[str, bool]:
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    blocks = list(package.get("blocks", []))
    cells = list(package.get("cells", []))
    checks: dict[str, bool] = {
        "schema": package.get("schema") == COHORT_SCHEMA,
        "stage": package.get("stage") == "construction",
        "outcome_firewall": package.get("model_outcomes_read") is False,
        "block_count": len(cells) == 9 * len(blocks),
    }
    seen_genes: set[str] = set()
    complete = True
    sequences = True
    source_integrity = True
    blacklist_clear = True
    split_clear = True
    config = package.get("config", {})
    split_weights = tuple((str(name), int(weight)) for name, weight in config.get("split_weights", []))
    for block in blocks:
        block_cells = [cell for cell in cells if cell.get("block_id") == block.get("block_id")]
        complete &= {
            (int(cell["donor_phase"]), int(cell["acceptor_phase"])) for cell in block_cells
        } == set(itertools.product((0, 1, 2), repeat=2))
        source_genes = {gene_root(gene) for gene in block.get("source_gene_ids", [])}
        source_integrity &= len(source_genes) == 7 and not bool(source_genes & seen_genes)
        seen_genes.update(source_genes)
        blacklist_clear &= not bool(source_genes & blacklist)
        split_clear &= (
            stable_block_split(
                source_genes,
                namespace=str(config.get("namespace")),
                weights=split_weights,
            )
            == block.get("split_role")
        )
        try:
            scaffold = by_id[str(block["scaffold_id"])]
            donors = {
                phase: by_id[str(block["donor_modules"][str(phase)]["candidate_id"])]
                for phase in (0, 1, 2)
            }
            acceptors = {
                phase: by_id[str(block["acceptor_modules"][str(phase)]["candidate_id"])]
                for phase in (0, 1, 2)
            }
        except (KeyError, TypeError):
            sequences = False
            continue
        source_integrity &= (
            block.get("scaffold") == _candidate_reference(scaffold)
            and all(
                block["donor_modules"][str(phase)] == _candidate_reference(donors[phase])
                and block["acceptor_modules"][str(phase)] == _candidate_reference(acceptors[phase])
                for phase in (0, 1, 2)
            )
        )
        reconstructed = [
            _cell_record(
                block_id=str(block["block_id"]),
                split_role=str(block["split_role"]),
                scaffold=scaffold,
                donor=donors[donor_phase],
                acceptor=acceptors[acceptor_phase],
            )
            for donor_phase in (0, 1, 2)
            for acceptor_phase in (0, 1, 2)
        ]
        _add_block_stop_excess(reconstructed)
        expected_by_id = {cell["cell_id"]: cell for cell in reconstructed}
        for cell in block_cells:
            donor, acceptor = donors[int(cell["donor_phase"])], acceptors[int(cell["acceptor_phase"])]
            intact = _replace_windows(scaffold, donor, acceptor)
            ac = _acceptor_alt(intact, int(scaffold["acceptor_pos"]), "AC")
            at = _acceptor_alt(intact, int(scaffold["acceptor_pos"]), "AT")
            sequences &= (
                cell.get("sequence_intact") == intact
                and cell.get("sequence_acceptor_ac") == ac
                and cell.get("sequence_acceptor_at") == at
                and cell.get("sequence_sha256")
                == {
                    "intact": sequence_sha256(intact),
                    "acceptor_ac": sequence_sha256(ac),
                    "acceptor_at": sequence_sha256(at),
                }
                and cell == expected_by_id.get(cell.get("cell_id"))
            )
    checks.update(
        {
            "complete_3x3": bool(complete),
            "seven_sources_and_global_gene_disjointness": bool(source_integrity),
            "hard_blacklist": bool(blacklist_clear),
            "stable_split": bool(split_clear),
            "sequence_and_covariate_reconstruction": bool(sequences),
            "development_smoke_subset": package.get("development_smoke")
            == {
                "method": "sha256_rank_whole_block_identity",
                "namespace": SMOKE_NAMESPACE,
                "n_blocks": min(
                    12,
                    sum(block.get("split_role") == "development" for block in blocks),
                ),
                "block_ids": sorted(
                    [
                        str(block["block_id"])
                        for block in blocks
                        if block.get("split_role") == "development"
                    ],
                    key=lambda block_id: (
                        stable_digest(block_id, namespace=SMOKE_NAMESPACE),
                        block_id,
                    ),
                )[:12],
                "block_ids_sha256": sha256_payload(
                    sorted(
                        [
                            str(block["block_id"])
                            for block in blocks
                            if block.get("split_role") == "development"
                        ],
                        key=lambda block_id: (
                            stable_digest(block_id, namespace=SMOKE_NAMESPACE),
                            block_id,
                        ),
                    )[:12]
                ),
                "not_an_independent_split": True,
            },
        }
    )
    return checks


def _audit_feature_scale(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    """Auditor-local standardization; intentionally not the builder helper."""

    features = {
        str(candidate["candidate_id"]): extract_module_covariates(candidate)
        for candidate in candidates
    }
    names = sorted(set().union(*(row.keys() for row in features.values())))
    matrix = np.asarray(
        [[features[str(candidate["candidate_id"])][name] for name in names] for candidate in candidates],
        dtype=np.float64,
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    return (
        features,
        dict(zip(names, means.tolist())),
        dict(zip(names, scales.tolist())),
    )


def _audit_pick_scaffold_triplet(
    pools: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    excluded_genes: set[str],
    anchor: Mapping[str, float],
    feature_rows: Mapping[str, Mapping[str, float]],
    means: Mapping[str, float],
    scales: Mapping[str, float],
    search_width: int,
    neighbor_width: int,
    namespace: str,
) -> tuple[dict[int, Mapping[str, Any]], float] | None:
    """Independent scaffold-centred nearest-triplet implementation."""

    names = tuple(means)
    if tuple(sorted(anchor)) != names:
        raise CohortConstructionError("auditor found an incomplete scaffold anchor")

    def distance(candidate: Mapping[str, Any]) -> float:
        row = feature_rows[str(candidate["candidate_id"])]
        standardized = np.asarray(
            [(float(row[name]) - float(anchor[name])) / float(scales[name]) for name in names],
            dtype=np.float64,
        )
        return float(np.sqrt(np.mean(np.square(standardized)))) if standardized.size else 0.0

    nearest: dict[int, list[Mapping[str, Any]]] = {}
    for phase in (0, 1, 2):
        ranked = sorted(
            (candidate for candidate in pools[phase] if candidate["gene_id"] not in excluded_genes),
            key=lambda candidate: (
                distance(candidate),
                stable_digest(candidate["candidate_id"], namespace=namespace),
                candidate["candidate_id"],
            ),
        )[:search_width]
        nearest[phase] = ranked[: min(neighbor_width, len(ranked))]
    if any(not nearest[phase] for phase in (0, 1, 2)):
        return None
    proposals: list[tuple[tuple[float, str], dict[int, Mapping[str, Any]]]] = []
    for candidates in itertools.product(nearest[0], nearest[1], nearest[2]):
        if len({candidate["gene_id"] for candidate in candidates}) != 3:
            continue
        score = max(distance(candidate) for candidate in candidates)
        identity = "|".join(str(candidate["candidate_id"]) for candidate in candidates)
        proposals.append(
            (
                (score, stable_digest(identity, namespace=namespace)),
                {phase: candidates[phase] for phase in (0, 1, 2)},
            )
        )
    if not proposals:
        return None
    (score, _tie), selected = min(proposals, key=lambda item: item[0])
    return selected, float(score)


def _audit_replay_selection(
    candidates: Sequence[Mapping[str, Any]], config: Mapping[str, Any], blacklist: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay block assembly without calling the production builder/selector."""

    quotas = {str(key): int(value) for key, value in config["requested_blocks_by_split"].items()}
    namespace = str(config["namespace"])
    split_weights = tuple((str(name), int(weight)) for name, weight in config["split_weights"])
    allowed = {normalize_chromosome(value) for value in config["allowed_chromosomes"]}
    forbidden = {normalize_chromosome(value) for value in config["forbidden_chromosomes"]}
    filtered = [
        candidate
        for candidate in candidates
        if candidate["gene_id"] not in blacklist
        and candidate["chrom"] in allowed
        and candidate["chrom"] not in forbidden
    ]
    scaffolds = [
        candidate
        for candidate in filtered
        if candidate["role"] == "scaffold"
        and len(candidate["sequence"]) == int(config["construct_length"])
        and int(candidate["donor_pos"]) == int(config["donor_pos"])
        and int(candidate["acceptor_pos"]) == int(config["acceptor_pos"])
        and tuple(candidate["donor_window"]) == DEFAULT_DONOR_WINDOW
        and tuple(candidate["acceptor_window"]) == DEFAULT_ACCEPTOR_WINDOW
    ]
    pools = {
        role: {
            phase: [
                candidate
                for candidate in filtered
                if candidate["role"] == role and int(candidate["phase"]) == phase
            ]
            for phase in (0, 1, 2)
        }
        for role in ("donor", "acceptor")
    }
    donor_all = [candidate for phase in (0, 1, 2) for candidate in pools["donor"][phase]]
    acceptor_all = [candidate for phase in (0, 1, 2) for candidate in pools["acceptor"][phase]]
    donor_rows, donor_means, donor_scales = _audit_feature_scale(donor_all)
    acceptor_rows, acceptor_means, acceptor_scales = _audit_feature_scale(acceptor_all)
    remaining = sorted(
        scaffolds,
        key=lambda candidate: (
            stable_digest(candidate["candidate_id"], namespace=namespace),
            candidate["candidate_id"],
        ),
    )
    used: set[str] = set()
    counts: Counter[str] = Counter()
    blocks: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    while any(counts[role] < quota for role, quota in quotas.items()):
        proposals: list[tuple[tuple[float, float, str], dict[str, Any]]] = []
        discard: set[str] = set()
        for scaffold in remaining:
            if len(proposals) >= int(config["global_search_width"]):
                break
            if scaffold["gene_id"] in used:
                discard.add(str(scaffold["candidate_id"]))
                continue
            donor_selected = _audit_pick_scaffold_triplet(
                pools["donor"],
                excluded_genes=used | {str(scaffold["gene_id"])},
                anchor=scaffold["donor_anchor_covariates"],
                feature_rows=donor_rows,
                means=donor_means,
                scales=donor_scales,
                search_width=int(config["search_width"]),
                neighbor_width=int(config["neighbor_width"]),
                namespace=namespace,
            )
            if donor_selected is None:
                discard.add(str(scaffold["candidate_id"]))
                continue
            donors, donor_score = donor_selected
            donor_genes = {str(candidate["gene_id"]) for candidate in donors.values()}
            acceptor_selected = _audit_pick_scaffold_triplet(
                pools["acceptor"],
                excluded_genes=used | {str(scaffold["gene_id"])} | donor_genes,
                anchor=scaffold["acceptor_anchor_covariates"],
                feature_rows=acceptor_rows,
                means=acceptor_means,
                scales=acceptor_scales,
                search_width=int(config["search_width"]),
                neighbor_width=int(config["neighbor_width"]),
                namespace=namespace,
            )
            if acceptor_selected is None:
                discard.add(str(scaffold["candidate_id"]))
                continue
            acceptors, acceptor_score = acceptor_selected
            genes = {
                str(scaffold["gene_id"]),
                *donor_genes,
                *(str(candidate["gene_id"]) for candidate in acceptors.values()),
            }
            if len(genes) != 7 or genes & used:
                discard.add(str(scaffold["candidate_id"]))
                continue
            role = stable_block_split(genes, namespace=namespace, weights=split_weights)
            if counts[role] >= quotas.get(role, 0):
                continue
            source_ids = [str(scaffold["candidate_id"])] + [
                str(donors[phase]["candidate_id"]) for phase in (0, 1, 2)
            ] + [str(acceptors[phase]["candidate_id"]) for phase in (0, 1, 2)]
            digest = stable_digest(*source_ids, namespace=namespace)
            block_id = f"pbc-{role}-{counts[role]:04d}-{digest[:12]}"
            proposed_cells = [
                _cell_record(
                    block_id=block_id,
                    split_role=role,
                    scaffold=scaffold,
                    donor=donors[donor_phase],
                    acceptor=acceptors[acceptor_phase],
                )
                for donor_phase in (0, 1, 2)
                for acceptor_phase in (0, 1, 2)
            ]
            _add_block_stop_excess(proposed_cells)
            objective = _proposal_imbalance_objective(
                [*(cell for cell in cells if cell["split_role"] == role), *proposed_cells]
            )
            proposals.append(
                (
                    (objective, max(donor_score, acceptor_score), digest),
                    {
                        "scaffold": scaffold,
                        "donors": donors,
                        "acceptors": acceptors,
                        "donor_score": donor_score,
                        "acceptor_score": acceptor_score,
                        "genes": genes,
                        "digest": digest,
                        "role": role,
                        "block_id": block_id,
                        "cells": proposed_cells,
                        "objective": objective,
                    },
                )
            )
        remaining = [
            scaffold for scaffold in remaining if str(scaffold["candidate_id"]) not in discard
        ]
        if not proposals:
            break
        _rank, selected = min(proposals, key=lambda item: item[0])
        scaffold = selected["scaffold"]
        donors, acceptors = selected["donors"], selected["acceptors"]
        role, genes = str(selected["role"]), set(selected["genes"])
        blocks.append(
            {
                "block_id": selected["block_id"],
                "block_identity_sha256": stable_digest(
                    *sorted(genes), namespace=f"{namespace}:block-identity"
                ),
                "split_role": role,
                "scaffold_id": scaffold["candidate_id"],
                "scaffold_gene_id": scaffold["gene_id"],
                "scaffold": _candidate_reference(scaffold),
                "chrom": scaffold["chrom"],
                "strand": scaffold["strand"],
                "donor_pos": int(scaffold["donor_pos"]),
                "acceptor_pos": int(scaffold["acceptor_pos"]),
                "span_nt": int(config["span_nt"]),
                "construct_length": int(config["construct_length"]),
                "seams": {
                    "donor_window": list(scaffold["donor_window"]),
                    "acceptor_window": list(scaffold["acceptor_window"]),
                },
                "donor_modules": {
                    str(phase): _candidate_reference(donors[phase]) for phase in (0, 1, 2)
                },
                "acceptor_modules": {
                    str(phase): _candidate_reference(acceptors[phase]) for phase in (0, 1, 2)
                },
                "source_gene_ids": sorted(genes),
                "matching": {
                    "donor_scaffold_anchor_max_standardized_distance": float(
                        selected["donor_score"]
                    ),
                    "acceptor_scaffold_anchor_max_standardized_distance": float(
                        selected["acceptor_score"]
                    ),
                    "complete_feature_three_contrast_objective": float(selected["objective"]),
                },
            }
        )
        cells.extend(selected["cells"])
        used.update(genes)
        counts[role] += 1
        remaining = [
            candidate
            for candidate in remaining
            if candidate["candidate_id"] != scaffold["candidate_id"]
        ]
    if any(counts[role] != quota for role, quota in quotas.items()):
        raise CohortConstructionError(
            f"independent replay obtained {dict(counts)} rather than {quotas}"
        )
    return blocks, cells


def _audit_rederive_capacity(
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    blacklist: set[str],
) -> dict[str, Any]:
    """Auditor-local reconstruction of the necessary unique-gene certificate."""

    quotas = {
        str(role): int(value)
        for role, value in config["requested_blocks_by_split"].items()
    }
    requested_blocks = sum(quotas.values())
    allowed = {normalize_chromosome(value) for value in config["allowed_chromosomes"]}
    forbidden = {normalize_chromosome(value) for value in config["forbidden_chromosomes"]}
    slot_names = (
        "scaffold",
        "donor_phase_0",
        "donor_phase_1",
        "donor_phase_2",
        "acceptor_phase_0",
        "acceptor_phase_1",
        "acceptor_phase_2",
    )
    slot_genes: dict[str, set[str]] = {name: set() for name in slot_names}
    for candidate in candidates:
        gene_id = gene_root(str(candidate["gene_id"]))
        chrom = normalize_chromosome(str(candidate["chrom"]))
        if gene_id in blacklist or chrom not in allowed or chrom in forbidden:
            continue
        role = str(candidate["role"])
        if role == "scaffold":
            if (
                len(str(candidate["sequence"])) == int(config["construct_length"])
                and int(candidate["donor_pos"]) == int(config["donor_pos"])
                and int(candidate["acceptor_pos"]) == int(config["acceptor_pos"])
                and tuple(candidate["donor_window"]) == DEFAULT_DONOR_WINDOW
                and tuple(candidate["acceptor_window"]) == DEFAULT_ACCEPTOR_WINDOW
            ):
                slot_genes["scaffold"].add(gene_id)
        elif role in {"donor", "acceptor"}:
            phase = int(candidate["phase"])
            if phase in {0, 1, 2}:
                slot_genes[f"{role}_phase_{phase}"].add(gene_id)

    def entry(genes: set[str], required: int) -> dict[str, Any]:
        ordered = sorted(genes)
        available = len(ordered)
        deficit = max(0, int(required) - available)
        return {
            "required_unique_genes": int(required),
            "available_unique_genes": int(available),
            "deficit_unique_genes": int(deficit),
            "passed": deficit == 0,
            "gene_ids_sha256": sha256_payload(ordered),
        }

    slots = {
        name: entry(slot_genes[name], requested_blocks) for name in slot_names
    }
    family_members = {
        "donor_all_phases": (
            "donor_phase_0",
            "donor_phase_1",
            "donor_phase_2",
        ),
        "acceptor_all_phases": (
            "acceptor_phase_0",
            "acceptor_phase_1",
            "acceptor_phase_2",
        ),
        "all_seven_source_roles": slot_names,
    }
    families: dict[str, dict[str, Any]] = {}
    for name, members in family_members.items():
        genes: set[str] = set()
        for member in members:
            genes.update(slot_genes[member])
        family = entry(genes, requested_blocks * len(members))
        family["source_slots"] = list(members)
        families[name] = family
    failed_constraints = [
        *(f"slot:{name}" for name in slot_names if not slots[name]["passed"]),
        *(f"family:{name}" for name in family_members if not families[name]["passed"]),
    ]
    return {
        "criterion": "necessary_unique_normalized_gene_capacity",
        "necessary_not_sufficient": True,
        "requested_complete_blocks": int(requested_blocks),
        "required_unique_source_genes": int(7 * requested_blocks),
        "slots": slots,
        "families": families,
        "failed_constraints": failed_constraints,
        "passed": not failed_constraints,
    }


def _audit_config_contract(
    candidate_package: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, bool]:
    """Reject production artifacts that weaken any frozen construction setting."""

    if candidate_package.get("candidate_mode") == "synthetic_fixture":
        return {"explicit_synthetic_fixture_config_override": True}
    expected: dict[str, Any] = {
        "algorithm_version": ALGORITHM_VERSION,
        "namespace": NAMESPACE,
        "split_weights": [[name, int(weight)] for name, weight in DEFAULT_SPLIT_WEIGHTS],
        "requested_blocks_by_split": dict(DEFAULT_BLOCK_COUNTS),
        "minimum_confirmation_blocks": 600,
        "span_nt": DEFAULT_SPAN_NT,
        "construct_length": DEFAULT_CONSTRUCT_LENGTH,
        "donor_pos": DEFAULT_DONOR_POS,
        "acceptor_pos": DEFAULT_ACCEPTOR_POS,
        "allowed_chromosomes": sorted(
            {normalize_chromosome(chrom) for chrom in DEFAULT_ALLOWED_CHROMS}
        ),
        "forbidden_chromosomes": sorted(
            {normalize_chromosome(chrom) for chrom in DEFAULT_FORBIDDEN_CHROMS}
        ),
        "search_width": 64,
        "neighbor_width": 8,
        "global_search_width": 16,
        "seed": 73031,
        "outer_folds": 5,
        "inner_folds": 4,
        "auc_n_boot": 2000,
        "pseudo_family_size": 99,
        "minimum_stratum_blocks": 5,
        "smd_threshold": 0.10,
        "auc_upper_threshold": 0.55,
    }
    groups = {
        "production_config_exact_keyset": set(config) == set(expected),
        "production_role_counts": config.get("requested_blocks_by_split")
        == expected["requested_blocks_by_split"],
        "production_minimum_confirmation": config.get("minimum_confirmation_blocks") == 600,
        "production_geometry": all(
            config.get(key) == expected[key]
            for key in ("span_nt", "construct_length", "donor_pos", "acceptor_pos")
        ),
        "production_chromosome_firewall": all(
            config.get(key) == expected[key]
            for key in ("allowed_chromosomes", "forbidden_chromosomes")
        ),
        "production_split_contract": all(
            config.get(key) == expected[key]
            for key in ("namespace", "split_weights", "seed")
        ),
        "production_calibration_contract": all(
            config.get(key) == expected[key]
            for key in (
                "outer_folds",
                "inner_folds",
                "auc_n_boot",
                "pseudo_family_size",
                "minimum_stratum_blocks",
                "smd_threshold",
                "auc_upper_threshold",
            )
        ),
        "production_selector_contract": all(
            config.get(key) == expected[key]
            for key in (
                "algorithm_version",
                "search_width",
                "neighbor_width",
                "global_search_width",
            )
        ),
        "production_config_exact_values": dict(config) == expected,
    }
    return groups


def _audit_capacity_stop(
    package: Mapping[str, Any],
    candidate_package: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    candidate_audit: Mapping[str, Any] | None,
    candidate_census_receipt: Mapping[str, Any] | None,
    blacklist: set[str],
    frozen_variance: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit a fail-fast supply stop without replaying matching or calibration."""

    candidate_package_sha256 = (
        str(candidate_audit["candidate_census_sha256"])
        if candidate_audit is not None
        else sha256_payload(candidate_package)
    )
    config_error: str | None = None
    rederived_capacity: dict[str, Any] = {}
    rederived_plan: dict[str, Any] = {}
    expected_counts: dict[str, Any] = {}
    try:
        config = dict(package["config"])
        quotas = {
            str(role): int(value)
            for role, value in config["requested_blocks_by_split"].items()
        }
        rederived_capacity = _audit_rederive_capacity(candidates, config, blacklist)
        rederived_plan = plan_sample_size(
            frozen_variance,
            minimum_blocks=int(config["minimum_confirmation_blocks"]),
        ).to_dict()
        expected_counts = {
            "blocks": 0,
            "cells": 0,
            "genes": 0,
            "blocks_by_split": {role: 0 for role in quotas},
        }
    except Exception as exc:
        config_error = f"{type(exc).__name__}:{exc}"
        config = {}
    config_contract = _audit_config_contract(candidate_package, config)
    seal = package.get("seal", {})
    recorded_capacity = package.get("capacity", {})
    recorded_receipt = package.get("candidate_census_receipt")
    if candidate_audit is None:
        receipt_contract = {
            "synthetic_receipt_absent": recorded_receipt is None
            and candidate_census_receipt is None,
        }
    else:
        candidate_file_receipt = (
            candidate_census_receipt.get("candidate_file", {})
            if isinstance(candidate_census_receipt, Mapping)
            else {}
        )
        producer_manifest_receipt = (
            candidate_census_receipt.get("producer_manifest", {})
            if isinstance(candidate_census_receipt, Mapping)
            else {}
        )
        semantic_hashes = (
            candidate_census_receipt.get("semantic_hashes", {})
            if isinstance(candidate_census_receipt, Mapping)
            else {}
        )
        receipt_contract = {
            "candidate_receipt_supplied": isinstance(candidate_census_receipt, Mapping),
            "candidate_receipt_schema": isinstance(candidate_census_receipt, Mapping)
            and candidate_census_receipt.get("schema") == CANDIDATE_CENSUS_RECEIPT_SCHEMA,
            "candidate_receipt_byte_identity": candidate_file_receipt.get("name")
            == "candidate_census_v1.json"
            and isinstance(candidate_file_receipt.get("size_bytes"), int)
            and candidate_file_receipt.get("size_bytes", 0) > 0
            and _is_lower_hex(candidate_file_receipt.get("sha256"), length=64),
            "candidate_receipt_producer_manifest": producer_manifest_receipt.get("name")
            == "candidate_census_v1.manifest.json"
            and isinstance(producer_manifest_receipt.get("size_bytes"), int)
            and producer_manifest_receipt.get("size_bytes", 0) > 0
            and _is_lower_hex(producer_manifest_receipt.get("sha256"), length=64)
            and producer_manifest_receipt.get("experiment_id")
            == "splicebert-positional-boundary-coherence:candidate-census-v1"
            and producer_manifest_receipt.get("status") == "CENSUS_BUILT_AUDIT_PENDING"
            and producer_manifest_receipt.get("config")
            == {
                "acceptor_pos": DEFAULT_ACCEPTOR_POS,
                "algorithm_version": "pbc-real-census-v1",
                "allowed_chromosomes": sorted(DEFAULT_ALLOWED_CHROMS),
                "construct_length": DEFAULT_CONSTRUCT_LENGTH,
                "donor_pos": DEFAULT_DONOR_POS,
                "stage": "construction",
            }
            and producer_manifest_receipt.get("git_commit")
            == CANONICAL_CANDIDATE_PRODUCER_REVISION
            and _is_lower_hex(producer_manifest_receipt.get("git_commit"), length=40)
            and producer_manifest_receipt.get("git_dirty") is True
            and producer_manifest_receipt.get("output_record_verified") is True,
            "candidate_receipt_semantic_hashes": semantic_hashes
            == {
                "candidate_census_sha256": candidate_audit.get("candidate_census_sha256"),
                "candidate_content_sha256": candidate_audit.get("candidate_content_sha256"),
                "reconstructed_candidates_sha256": candidate_audit.get(
                    "reconstructed_candidates_sha256"
                ),
            }
            and all(_is_lower_hex(value, length=64) for value in semantic_hashes.values()),
            "candidate_receipt_regeneration_boundary": isinstance(
                candidate_census_receipt, Mapping
            )
            and candidate_census_receipt.get("clean_checkout_reexecutable") is False
            and candidate_census_receipt.get("regeneration_required") is True
            and candidate_census_receipt.get("regeneration_receipt")
            == "candidate_census_v1.manifest.json",
        }
    expected_construction_gates = [
        "block_assembly",
        "role_hash_supply",
        "chromosome_strand_strata",
        "composition_balance",
        "classifier_calibration",
    ]
    expected_model_stages = [
        "phase2_carrier_factorial_calibration",
        "phase2_model_preflight",
        "phase2_target",
        "phase3_register_preflight",
        "phase3_candidate_screen",
        "phase3_reserve",
        "phase3_mediation",
    ]
    expected_model_execution = {
        "checkpoint_loaded": False,
        "forward_passes": 0,
        "outcome_rows": 0,
        "all_model_gates": "NOT_EVALUATED",
    }
    expected_evidence_boundary = (
        "This artifact proves only that the frozen Phase-I block request cannot be met "
        "with globally unique eligible genes. No SpliceBERT checkpoint or model outcome "
        "was loaded, and no biological or model-mechanism result is licensed."
    )
    expected_core = {
        "schema": CAPACITY_STOP_SCHEMA,
        "stage": "construction",
        "status": "PHASE1_INSTRUMENT_UNRESOLVED",
        "disposition": "NECESSARY_CAPACITY_FAILED",
        "model_outcomes_read": False,
        "construction_gates_pass": False,
        "independent_audit_required": True,
        "target_execution_authorized": False,
        "selection_attempted": False,
        "balance_or_calibration_evaluated": False,
        "partial_blocks_recorded": False,
        "clean_checkout_reexecutable": False,
        "candidate_census_receipt": candidate_census_receipt,
        "config": config,
        "sample_size_plan": rederived_plan,
        "blacklist": {
            "n_gene_ids": len(blacklist),
            "gene_ids_sha256": sha256_payload(sorted(blacklist)),
        },
        "counts": expected_counts,
        "capacity": rederived_capacity,
        "gates": {"necessary_unique_gene_capacity": False},
        "unevaluated_construction_gates": expected_construction_gates,
        "model_execution": expected_model_execution,
        "unevaluated_model_stages": expected_model_stages,
        "blocks": [],
        "cells": [],
        "evidence_boundary": expected_evidence_boundary,
    }
    expected_seal = {
        "candidate_package_sha256": candidate_package_sha256,
        "candidate_audit_sha256": (
            None if candidate_audit is None else sha256_payload(candidate_audit)
        ),
        "frozen_variance_sha256": sha256_payload(frozen_variance),
        "blacklist_sha256": sha256_payload(sorted(blacklist)),
        "capacity_sha256": sha256_payload(rederived_capacity),
        "candidate_census_receipt_sha256": (
            None
            if candidate_census_receipt is None
            else sha256_payload(candidate_census_receipt)
        ),
        "content_sha256": sha256_payload(expected_core),
    }
    checks = {
        "schema": package.get("schema") == CAPACITY_STOP_SCHEMA,
        "stage": package.get("stage") == "construction",
        "status": package.get("status") == "PHASE1_INSTRUMENT_UNRESOLVED",
        "disposition": package.get("disposition") == "NECESSARY_CAPACITY_FAILED",
        "outcome_firewall": package.get("model_outcomes_read") is False
        and candidate_package.get("model_outcomes_read") is False,
        "model_execution_absent": package.get("model_execution")
        == expected_model_execution,
        "all_model_stages_unevaluated": package.get("unevaluated_model_stages")
        == expected_model_stages,
        "all_construction_gates_unevaluated": package.get(
            "unevaluated_construction_gates"
        )
        == expected_construction_gates,
        "evidence_boundary_exact": package.get("evidence_boundary")
        == expected_evidence_boundary,
        "top_level_keyset_exact": set(package) == {*expected_core, "seal"},
        "stop_payload_exact": {
            key: value for key, value in package.items() if key != "seal"
        }
        == expected_core,
        "seal_keyset_exact": isinstance(seal, Mapping)
        and set(seal) == set(expected_seal),
        "seal_exact": seal == expected_seal,
        "clean_checkout_boundary": package.get("clean_checkout_reexecutable") is False,
        **receipt_contract,
        "candidate_receipt_rederived": recorded_receipt == candidate_census_receipt,
        "config_reconstructable": config_error is None,
        **config_contract,
        "capacity_failure_rederived": bool(rederived_capacity)
        and rederived_capacity.get("passed") is False,
        "capacity_record_rederived": recorded_capacity == rederived_capacity,
        "no_selection_attempted": package.get("selection_attempted") is False,
        "no_balance_or_calibration": package.get("balance_or_calibration_evaluated") is False,
        "no_partial_blocks": package.get("partial_blocks_recorded") is False
        and package.get("blocks") == []
        and package.get("cells") == [],
        "counts_rederived": package.get("counts") == expected_counts,
        "sample_size_plan": package.get("sample_size_plan") == rederived_plan,
        "construction_failed": package.get("construction_gates_pass") is False
        and package.get("gates") == {"necessary_unique_gene_capacity": False},
        "authorization_withheld": package.get("target_execution_authorized") is False
        and package.get("independent_audit_required") is True,
        "blacklist_record": package.get("blacklist")
        == {
            "n_gene_ids": len(blacklist),
            "gene_ids_sha256": sha256_payload(sorted(blacklist)),
        },
        "candidate_seal": seal.get("candidate_package_sha256")
        == candidate_package_sha256,
        "candidate_audit_seal": seal.get("candidate_audit_sha256")
        == (None if candidate_audit is None else sha256_payload(candidate_audit)),
        "variance_seal": seal.get("frozen_variance_sha256")
        == sha256_payload(frozen_variance),
        "blacklist_seal": seal.get("blacklist_sha256")
        == sha256_payload(sorted(blacklist)),
        "capacity_seal": seal.get("capacity_sha256")
        == sha256_payload(recorded_capacity),
        "candidate_receipt_seal": seal.get("candidate_census_receipt_sha256")
        == (
            None
            if recorded_receipt is None
            else sha256_payload(recorded_receipt)
        ),
        "content_seal": seal.get("content_sha256")
        == sha256_payload({key: value for key, value in package.items() if key != "seal"}),
    }
    passed = bool(all(checks.values()))
    return {
        "schema": AUDIT_SCHEMA,
        "stage": "construction",
        "status": "AUDIT_PASS_INSTRUMENT_UNRESOLVED" if passed else "AUDIT_FAIL",
        "passed": passed,
        "model_outcomes_read": False,
        "clean_checkout_reexecutable": False,
        "model_execution": {
            "checkpoint_loaded": False,
            "forward_passes": 0,
            "outcome_rows": 0,
            "all_model_gates": "NOT_EVALUATED",
        },
        "unevaluated_model_stages": list(
            package.get("unevaluated_model_stages", [])
        ),
        "target_execution_authorized": False,
        "checks": checks,
        "rederived": {
            "capacity": rederived_capacity,
            "candidate_census_receipt": candidate_census_receipt,
            "gates": {"necessary_unique_gene_capacity": False},
            "status": (
                "PHASE1_INSTRUMENT_UNRESOLVED"
                if passed
                else "CAPACITY_STOP_AUDIT_FAIL"
            ),
            "config_error": config_error,
            "target_execution_authorized": False,
        },
        "input_seals": {
            "cohort_content_sha256": seal.get("content_sha256"),
            "candidate_package_sha256": candidate_package_sha256,
            "candidate_audit_sha256": (
                None if candidate_audit is None else sha256_payload(candidate_audit)
            ),
            "frozen_variance_sha256": sha256_payload(frozen_variance),
            "blacklist_sha256": sha256_payload(sorted(blacklist)),
            "candidate_census_receipt_sha256": (
                None
                if candidate_census_receipt is None
                else sha256_payload(candidate_census_receipt)
            ),
        },
        "evidence_boundary": (
            "The audit verifies an outcome-blind Phase-I supply failure only. It does not "
            "measure or authorize a model, behavioral, mechanism, or biological result."
        ),
    }


def audit_composition_balanced_cohort(
    package: Mapping[str, Any],
    candidate_package: Mapping[str, Any],
    *,
    candidate_audit: Mapping[str, Any] | None = None,
    candidate_census_receipt: Mapping[str, Any] | None = None,
    blacklisted_gene_ids: Iterable[str],
    frozen_variance: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently replay selection, features, calibrations, and every gate.

    This path deliberately does not call :func:`build_composition_balanced_cohort`
    or reuse its selector.  Shared helpers are restricted to deterministic
    feature/statistical primitives.
    """

    candidates = validate_candidate_package(candidate_package)
    validate_candidate_audit_binding(candidate_package, candidate_audit)
    blacklist = {gene_root(gene) for gene in blacklisted_gene_ids}
    if package.get("schema") == CAPACITY_STOP_SCHEMA:
        return _audit_capacity_stop(
            package,
            candidate_package,
            candidates,
            candidate_audit=candidate_audit,
            candidate_census_receipt=candidate_census_receipt,
            blacklist=blacklist,
            frozen_variance=frozen_variance,
        )
    structural = _structural_audit(package, candidates, blacklist)
    if not all(structural.values()):
        return {
            "schema": AUDIT_SCHEMA,
            "stage": "construction",
            "status": "AUDIT_FAIL",
            "passed": False,
            "target_execution_authorized": False,
            "checks": structural,
            "rederived": {
                "balance_audit": {},
                "calibrations": {},
                "gates": {},
                "status": "STRUCTURAL_AUDIT_FAIL",
                "replay_error": "structural checks failed before statistical replay",
                "target_execution_authorized": False,
            },
            "input_seals": {
                "cohort_content_sha256": package.get("seal", {}).get("content_sha256"),
                "candidate_package_sha256": sha256_payload(candidate_package),
                "candidate_audit_sha256": (
                    None if candidate_audit is None else sha256_payload(candidate_audit)
                ),
                "frozen_variance_sha256": sha256_payload(frozen_variance),
                "blacklist_sha256": sha256_payload(sorted(blacklist)),
            },
        }
    config = dict(package.get("config", {}))
    config_contract = _audit_config_contract(candidate_package, config)
    replay_error: str | None = None
    try:
        replay_blocks, replay_cells = _audit_replay_selection(candidates, config, blacklist)
    except Exception as exc:
        replay_blocks, replay_cells = [], []
        replay_error = f"{type(exc).__name__}:{exc}"
    quotas = {str(key): int(value) for key, value in config["requested_blocks_by_split"].items()}
    allowed = {normalize_chromosome(value) for value in config["allowed_chromosomes"]}
    forbidden = {normalize_chromosome(value) for value in config["forbidden_chromosomes"]}
    split_weights = tuple((str(name), int(weight)) for name, weight in config["split_weights"])
    rederived_balance: dict[str, Any] = {}
    rederived_calibrations: dict[str, Any] = {}
    for role, quota in quotas.items():
        if int(quota) == 0:
            continue
        role_cells = [cell for cell in replay_cells if cell.get("split_role") == role]
        role_seed = int(config["seed"]) + [name for name, _weight in split_weights].index(role)
        role_feature_rows = [dict(cell["covariates"]) for cell in role_cells]
        role_groups = [str(cell["block_id"]) for cell in role_cells]
        role_strata = [f"{cell['chrom']}|{cell['strand']}" for cell in role_cells]
        role_auc_design = _prepare_nested_auc_design(
            role_feature_rows, role_groups, role_strata
        )
        rederived_balance[role] = balance_audit(
            role_cells,
            seed=role_seed,
            smd_threshold=float(config["smd_threshold"]),
            auc_upper_threshold=float(config["auc_upper_threshold"]),
            outer_folds=int(config["outer_folds"]),
            inner_folds=int(config["inner_folds"]),
            auc_n_boot=int(config["auc_n_boot"]),
            _auc_design=role_auc_design,
        )
        if role in {"confirmation", "phase3_reserve"}:
            rederived_calibrations[role] = calibration_hooks(
                role_cells,
                seed=role_seed,
                outer_folds=int(config["outer_folds"]),
                inner_folds=int(config["inner_folds"]),
                auc_n_boot=int(config["auc_n_boot"]),
                auc_upper_threshold=float(config["auc_upper_threshold"]),
                pseudo_family_size=int(config["pseudo_family_size"]),
                _auc_design=role_auc_design,
            )
    replay_role_counts = Counter(str(block["split_role"]) for block in replay_blocks)
    replay_genes = {
        gene_root(gene)
        for block in replay_blocks
        for gene in block.get("source_gene_ids", [])
    }
    phase_counts = _phase_cell_counts(replay_cells) if replay_cells else {}
    phase_by_role = {
        role: _phase_cell_counts(
            [cell for cell in replay_cells if cell.get("split_role") == role]
        )
        for role, quota in quotas.items()
        if quota
    }
    source_counts = _source_strata(replay_blocks, allowed_chromosomes=allowed)
    source_by_role = {
        role: _source_strata(
            [block for block in replay_blocks if block.get("split_role") == role],
            allowed_chromosomes=allowed,
        )
        for role, quota in quotas.items()
        if quota
    }
    expected_scaffold_strata = [
        f"{chrom}|{strand}" for chrom in sorted(allowed) for strand in ("+", "-")
    ]
    scaffold_strata: dict[str, dict[str, int]] = {}
    for role in quotas:
        observed = Counter(
            f"{block['chrom']}|{block['strand']}"
            for block in replay_blocks
            if block.get("split_role") == role
        )
        scaffold_strata[role] = {
            stratum: int(observed.get(stratum, 0)) for stratum in expected_scaffold_strata
        }
    plan = plan_sample_size(
        frozen_variance, minimum_blocks=int(config["minimum_confirmation_blocks"])
    )
    exact_geometry = all(
        int(block["construct_length"]) == int(config["construct_length"])
        and int(block["donor_pos"]) == int(config["donor_pos"])
        and int(block["acceptor_pos"]) == int(config["acceptor_pos"])
        and int(block["span_nt"]) == int(config["span_nt"])
        for block in replay_blocks
    )
    rederived_gates = {
        "exact_role_counts": dict(replay_role_counts)
        == {role: count for role, count in quotas.items() if count},
        "confirmation_variance_plan": quotas.get("confirmation", 0) == plan.planned_blocks,
        "complete_3x3": len(replay_cells) == 9 * len(replay_blocks)
        and bool(phase_counts)
        and set(phase_counts.values()) == {len(replay_blocks)}
        and all(set(counts.values()) == {quotas[role]} for role, counts in phase_by_role.items()),
        "source_strata": _source_strata_complete(
            source_counts,
            expected_blocks=len(replay_blocks),
            allowed_chromosomes=allowed,
        )
        and all(
            _source_strata_complete(
                counts,
                expected_blocks=quotas[role],
                allowed_chromosomes=allowed,
                minimum_per_source_stratum=(
                    int(config["minimum_stratum_blocks"])
                    if role in {"confirmation", "phase3_reserve"}
                    else 0
                ),
            )
            for role, counts in source_by_role.items()
        ),
        "gene_disjoint": len(replay_genes) == 7 * len(replay_blocks),
        "hard_blacklist": not bool(replay_genes & blacklist),
        "forbidden_chromosomes": not bool(
            {str(block["chrom"]) for block in replay_blocks} & forbidden
        ),
        "allowed_chromosomes": all(str(block["chrom"]) in allowed for block in replay_blocks),
        "exact_geometry": exact_geometry,
        "max_smd_all_roles": all(
            audit["gates"]["max_smd"] for audit in rederived_balance.values()
        ),
        "classifier_confirmation_reserve": all(
            rederived_balance[role]["gates"]["nested_crossfit_auc"]
            for role in ("confirmation", "phase3_reserve")
            if quotas.get(role, 0)
        ),
        "stratum_capacity_confirmation_reserve": all(
            counts and min(counts.values()) >= int(config["minimum_stratum_blocks"])
            for role, counts in scaffold_strata.items()
            if role in {"confirmation", "phase3_reserve"} and quotas.get(role, 0)
        ),
        "planted_imbalance_detected": all(
            audit["gates"]["planted_imbalance_detected"]
            for audit in rederived_calibrations.values()
        ),
        "pseudo_family_complete": int(config["pseudo_family_size"]) == 99,
        "permuted_labels_rejected": all(
            audit["gates"]["permuted_labels_rejected"]
            for audit in rederived_calibrations.values()
        ),
    }
    replay_counts = {
        "blocks": len(replay_blocks),
        "cells": len(replay_cells),
        "genes": len(replay_genes),
        "blocks_by_split": dict(sorted(replay_role_counts.items())),
        "phase_cells_all_roles": phase_counts,
        "phase_cells_by_split": phase_by_role,
        "source_role_phase": source_counts,
        "source_role_phase_by_split": source_by_role,
        "chromosome_strand_by_split": scaffold_strata,
    }
    development_ids = [
        str(block["block_id"])
        for block in replay_blocks
        if block.get("split_role") == "development"
    ]
    smoke_ids = sorted(
        development_ids,
        key=lambda block_id: (
            stable_digest(block_id, namespace=SMOKE_NAMESPACE),
            block_id,
        ),
    )[: min(12, len(development_ids))]
    expected_smoke = {
        "method": "sha256_rank_whole_block_identity",
        "namespace": SMOKE_NAMESPACE,
        "n_blocks": len(smoke_ids),
        "block_ids": smoke_ids,
        "block_ids_sha256": sha256_payload(smoke_ids),
        "not_an_independent_split": True,
    }
    recorded_cells = list(package.get("cells", []))
    checks = {
        **structural,
        **config_contract,
        "candidate_seal": package.get("seal", {}).get("candidate_package_sha256")
        == sha256_payload(candidate_package),
        "candidate_audit_seal": package.get("seal", {}).get("candidate_audit_sha256")
        == (None if candidate_audit is None else sha256_payload(candidate_audit)),
        "variance_seal": package.get("seal", {}).get("frozen_variance_sha256")
        == sha256_payload(frozen_variance),
        "selection_seal": package.get("seal", {}).get("selection_sha256")
        == sha256_payload(package.get("blocks", [])),
        "feature_seal": package.get("seal", {}).get("feature_matrix_sha256")
        == sha256_payload([cell.get("covariates", {}) for cell in recorded_cells]),
        "content_seal": package.get("seal", {}).get("content_sha256")
        == sha256_payload({key: value for key, value in package.items() if key != "seal"}),
        "independent_replay_completed": replay_error is None,
        "deterministic_selection": package.get("blocks") == replay_blocks,
        "deterministic_cells": recorded_cells == replay_cells,
        "cell_identity_seal": package.get("seal", {}).get("cell_identity_sha256")
        == sha256_payload(
            [
                {"cell_id": cell["cell_id"], "sequence_sha256": cell["sequence_sha256"]}
                for cell in replay_cells
            ]
        ),
        "sample_size_plan": package.get("sample_size_plan") == plan.to_dict(),
        "balance_rederived": package.get("balance_audit_by_split") == rederived_balance,
        "calibrations_rederived": package.get("calibrations_by_split") == rederived_calibrations,
        "counts_rederived": package.get("counts") == replay_counts,
        "smoke_rederived": package.get("development_smoke") == expected_smoke,
        "gates_rederived": package.get("gates") == rederived_gates,
        "status_rederived": package.get("status")
        == (
            "PHASE1_CONSTRUCTION_PASS_AUDIT_PENDING"
            if all(rederived_gates.values())
            else "PHASE1_INSTRUMENT_UNRESOLVED"
        ),
        "authorization_withheld_pending_audit": package.get("target_execution_authorized") is False
        and package.get("independent_audit_required") is True,
    }
    passed = all(checks.values())
    instrument_passed = bool(all(rederived_gates.values()))
    authorized = bool(passed and instrument_passed)
    return {
        "schema": AUDIT_SCHEMA,
        "stage": "construction",
        "status": (
            "AUDIT_PASS_TARGET_AUTHORIZED"
            if authorized
            else "AUDIT_PASS_INSTRUMENT_UNRESOLVED"
            if passed
            else "AUDIT_FAIL"
        ),
        "passed": bool(passed),
        "target_execution_authorized": authorized,
        "checks": checks,
        "rederived": {
            "balance_audit": rederived_balance,
            "calibrations": rederived_calibrations,
            "counts": replay_counts,
            "gates": rederived_gates,
            "status": (
                "PHASE1_CONSTRUCTION_PASS_AUDIT_PENDING"
                if instrument_passed
                else "PHASE1_INSTRUMENT_UNRESOLVED"
            ),
            "replay_error": replay_error,
            "target_execution_authorized": authorized,
        },
        "input_seals": {
            "cohort_content_sha256": package.get("seal", {}).get("content_sha256"),
            "candidate_package_sha256": sha256_payload(candidate_package),
            "candidate_audit_sha256": (
                None if candidate_audit is None else sha256_payload(candidate_audit)
            ),
            "frozen_variance_sha256": sha256_payload(frozen_variance),
            "blacklist_sha256": sha256_payload(sorted(blacklist)),
        },
    }


__all__ = [
    "ALGORITHM_VERSION",
    "AUDIT_SCHEMA",
    "CAPACITY_STOP_SCHEMA",
    "CANDIDATE_CENSUS_RECEIPT_SCHEMA",
    "CANONICAL_CANDIDATE_PRODUCER_REVISION",
    "CANDIDATE_SCHEMA",
    "CANDIDATE_AUDIT_SCHEMA",
    "COHORT_SCHEMA",
    "CohortConstructionError",
    "DEFAULT_ACCEPTOR_POS",
    "DEFAULT_ACCEPTOR_WINDOW",
    "DEFAULT_ALLOWED_CHROMS",
    "DEFAULT_BLOCK_COUNTS",
    "DEFAULT_BRIDGE_ACCEPTOR_POS",
    "DEFAULT_CONSTRUCT_LENGTH",
    "DEFAULT_DONOR_POS",
    "DEFAULT_DONOR_WINDOW",
    "DEFAULT_FORBIDDEN_CHROMS",
    "DEFAULT_MINIMUM_BLOCKS",
    "DEFAULT_SPAN_NT",
    "DEFAULT_SPLIT_WEIGHTS",
    "NAMESPACE",
    "SampleSizePlan",
    "assert_outcome_blind",
    "audit_composition_balanced_cohort",
    "balance_audit",
    "build_composition_balanced_cohort",
    "calibration_hooks",
    "canonical_json",
    "extract_cell_covariates",
    "extract_module_covariates",
    "gene_root",
    "kmer_frequencies",
    "load_blacklist",
    "nested_crossfit_auc",
    "plan_sample_size",
    "sequence_sha256",
    "sha256_payload",
    "stable_split",
    "stable_block_split",
    "standardized_mean_differences",
    "stop_load",
    "three_delta_smd_audit",
    "validate_candidate",
    "validate_candidate_package",
]
