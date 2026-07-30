#!/usr/bin/env python
"""Freeze the outcome-blind Stage-3 biological-feature confirmation package.

This runner never loads checkpoint weights or invokes model inference.  It
reconstructs the 200 confirmation genes frozen by discovery, replays the four
frozen PSAM candidates and their three matched controls, and writes exact raw
low/high sequence states for later five-checkpoint confirmation.

Controls are not merely checked for dose matching after construction: their
low and high searches are each constrained to the corresponding candidate's
delivered low and high nucleotide Hamming dose.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import (  # noqa: E402
    ExperimentManifest,
    file_record,
    sha256_text,
)
from gfmcd.splicing.biological_cohort import (  # noqa: E402
    ACCEPTOR_WINDOW,
    DONOR_WINDOW,
)
from gfmcd.splicing.biological_features import (  # noqa: E402
    beam_synonymous_optimize,
    codon_amino_acid,
    codon_log_preference,
    complete_codon_starts,
    kmer_count_vector,
    load_rbns_psam_directory,
    psam_region_score,
    psam_score_vector,
    vienna_mfe,
    vienna_unpaired_mean,
)
from gfmcd.splicing.circuit import _dataset, eligible  # noqa: E402
from gfmcd.splicing.phase import load_cds_intervals  # noqa: E402
from gfmcd.splicing.splice_landscape import (  # noqa: E402
    compare_landscapes,
    maxent_scorers,
)


SCHEMA = "gfmcd.gtex_native_biological_confirmation_construction.v1"
STATE_SCHEMA = "gfmcd.gtex_native_biological_confirmation_state.v1"
EXPERIMENT = "gtex_native_biological_confirmation_construction_v1"
PROTOCOL = (
    REPO
    / "studies/splicebert_gtex_finetune_shift/"
    "PREREGISTRATION_NATIVE_BIOLOGICAL_FEATURES_V1.md"
)

N_CONFIRMATION = 200
MIN_CONFIRMATION_EXAMPLES = 60
MIN_CONFIRMATION_GENES = 45
BEAM_SIZE = 256
MIN_CHANGED_BASES = 1
MAX_CHANGED_BASES = 4
MAX_ABS_MAXENT_DELTA = 1.0
NUISANCE_CALIPER_QUANTILE = 0.95

EXPECTED_SOURCE_SHA256 = {
    "protocol": "24322a133a9b48f64bf37ba44481ea3ac6f6f29446448d4e0b47f34c1c38c429",
    "discovery": "11b0106aabff1e921c6f009aaec83157a8da8d7f5121d746df7c65f77e74c3c3",
    "construction_audit": "1cb6ca1f9b7e1f6db29eaaf276f9bef360a53e983fc968f7683bf0f574284724",
    "basic_gtf": "5b9ad9aa71f6ab5d55123c51f74723644a895bc0cbaa9cee8fbc11bb9dec0bf7",
    "comprehensive_gtf": "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453",
    "fasta": "5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51",
}
EXPECTED_PSAM_AGGREGATE_SHA256 = (
    "cc515bdd38d2fe777d17b468fece07943f0d2ef942b0493348838194aae4d35e"
)

EXPECTED_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "donor": {
        "DAZAP1": ("HNRNPA2B1", "RBM23", "TAF15"),
        "SFPQ": ("SRSF8", "SNRPA", "DAZ3"),
    },
    "acceptor": {
        "UNK": ("FUBP1", "RBMS3", "NUPL2"),
        "DAZAP1": ("HNRNPA2B1", "RBM23", "TAF15"),
    },
}

NUISANCE_NAMES = (
    "off_target_psam_l1",
    "codon_preference_high_minus_low",
    "exon_identity_high_minus_low",
    "rna_accessibility_high_minus_low",
    "mfe_high_minus_low",
    "gc3_high_minus_low",
    "codon_pair_high_minus_low",
    "canonical_ag_gt_high_minus_low",
)

FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "label",
        "labels",
        "logit",
        "logits",
        "classifier_weight",
        "classifier_bias",
        "native_margin",
        "model_margin",
        "model_output",
        "model_outputs",
        "feature_activation",
        "mlp_write",
        "outcome",
        "outcomes",
        "psi",
        "delta_psi",
    }
)

TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "experiment",
        "status",
        "evidence_tier",
        "protocol",
        "model_loaded",
        "confirmation_model_outputs_read",
        "outcome_blind",
        "confirmation_authorized",
        "source_records",
        "frozen_thresholds",
        "nuisance_calipers",
        "cohort",
        "candidate_summaries",
        "states",
        "stop_reasons",
        "evidence_boundary",
    }
)

STATE_KEYS = frozenset(
    {
        "schema",
        "state_id",
        "candidate_id",
        "endpoint",
        "candidate_name",
        "family_name",
        "family_role",
        "direction",
        "expected_high_minus_low_margin_sign",
        "example_id",
        "gene_id",
        "chrom",
        "strand",
        "target_position",
        "anchor_position",
        "relative_window",
        "region_start",
        "region_end",
        "reference_sequence_sha256",
        "sequence",
        "sequence_sha256",
        "arm_success",
        "pair_success",
        "failure_reason",
        "requested_changed_bases",
        "delivered_changed_bases",
        "changed_positions",
        "codon_changes",
        "objective_clean",
        "objective_score",
        "pair_score_high_minus_low",
        "invariants",
        "maxent",
        "pair_nuisance",
        "nuisance_difference_from_candidate",
        "nuisance_caliper_pass",
        "usable_matched_set",
    }
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_write_json(path: Path, value: object) -> None:
    """Atomically create a JSON artifact without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    if output.suffix != ".json":
        raise ValueError("confirmation construction output must be a .json file")
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    if output == manifest:
        raise ValueError("result and manifest paths must be distinct")
    return output, manifest


def require_frozen_source(path: Path, source_name: str) -> dict[str, Any]:
    record = file_record(path)
    expected = EXPECTED_SOURCE_SHA256[source_name]
    if record.get("sha256") != expected:
        raise RuntimeError(
            f"{source_name} source hash mismatch: "
            f"expected {expected}, found {record.get('sha256')}"
        )
    return record


def psam_source_record(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("*.txt"))
    records = [file_record(path.resolve()) for path in files]
    if len(records) != 79:
        raise RuntimeError(f"expected 79 frozen PSAM files, found {len(records)}")
    try:
        relative_root = root.resolve().relative_to(REPO)
    except ValueError as exc:
        raise RuntimeError(
            "the frozen PSAM directory must be inside the active worktree"
        ) from exc
    # Reproduce the preregistered sorted-file checksum exactly:
    # `find ... -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256`.
    aggregate_payload = "".join(
        f"{row['sha256']}  {relative_root / Path(row['path']).name}\n"
        for row in records
    )
    aggregate_sha256 = sha256_text(aggregate_payload)
    if aggregate_sha256 != EXPECTED_PSAM_AGGREGATE_SHA256:
        raise RuntimeError(
            "RBNS PSAM aggregate hash differs from the frozen 79-file source"
        )
    return {
        "root": str(root),
        "file_count": len(records),
        "aggregate_sha256": aggregate_sha256,
        "expected_aggregate_sha256": EXPECTED_PSAM_AGGREGATE_SHA256,
        "files": records,
    }


def _candidate_rows(discovery: Mapping[str, Any], endpoint: str) -> dict[str, Mapping]:
    rows = discovery.get("candidate_freeze", {}).get(endpoint)
    if not isinstance(rows, list):
        raise RuntimeError(f"discovery lacks frozen {endpoint} candidates")
    return {str(row["rbp"]): row for row in rows}


def validate_frozen_family_contract(
    discovery: Mapping[str, Any],
    construction_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Fail closed unless all four candidate/control families are unchanged."""

    if (
        discovery.get("experiment")
        != "gtex_native_biological_features_discovery_v1"
        or discovery.get("status") != "COMPLETE"
    ):
        raise RuntimeError("wrong or incomplete discovery artifact")
    if (
        construction_audit.get("experiment")
        != "gtex_native_biological_construction_audit_v2"
        or construction_audit.get("status") != "COMPLETE"
        or construction_audit.get("model_loaded") is not False
    ):
        raise RuntimeError("wrong or incomplete corrected construction audit")

    observed_endpoints = set(discovery.get("candidate_freeze", {}))
    if observed_endpoints != set(EXPECTED_FAMILIES):
        raise RuntimeError("frozen candidate endpoints changed")
    result: list[dict[str, Any]] = []
    for endpoint, expected_candidates in EXPECTED_FAMILIES.items():
        candidates = _candidate_rows(discovery, endpoint)
        if set(candidates) != set(expected_candidates):
            raise RuntimeError(f"frozen {endpoint} candidate set changed")
        discovery_construction = discovery.get("construction_freeze", {}).get(
            endpoint, {}
        )
        audit_construction = construction_audit.get("audit", {}).get(endpoint, {})
        if set(discovery_construction) != set(expected_candidates) or set(
            audit_construction
        ) != set(expected_candidates):
            raise RuntimeError(f"frozen {endpoint} construction family changed")
        for candidate_name, expected_controls in expected_candidates.items():
            candidate = candidates[candidate_name]
            discovery_row = discovery_construction[candidate_name]
            audit_row = audit_construction[candidate_name]
            expected_sign = int(candidate["coefficient_sign"])
            if expected_sign not in {-1, 1}:
                raise RuntimeError("frozen candidate sign must be -1 or +1")
            if (
                tuple(discovery_row["matched_controls"]) != expected_controls
                or tuple(audit_row["matched_controls"]) != expected_controls
                or int(discovery_row["expected_high_minus_low_margin_sign"])
                != expected_sign
                or int(audit_row["expected_high_minus_low_margin_sign"])
                != expected_sign
                or audit_row.get("candidate_construction_pass") is not True
            ):
                raise RuntimeError(
                    f"candidate/control/sign freeze changed for {endpoint}:{candidate_name}"
                )
            result.append(
                {
                    "candidate_id": f"{endpoint}:{candidate_name}",
                    "endpoint": endpoint,
                    "candidate_name": candidate_name,
                    "matched_controls": list(expected_controls),
                    "expected_high_minus_low_margin_sign": expected_sign,
                }
            )
    if len(result) != 4:
        raise RuntimeError("exactly four frozen endpoint/candidate assignments required")
    return result


def endpoint_spec(example: Any, endpoint: str) -> dict[str, Any]:
    if endpoint == "donor":
        anchor = int(example.donor_pos)
        return {
            "window": DONOR_WINDOW,
            "anchor": anchor,
            "target_position": anchor - 1,
        }
    if endpoint == "acceptor":
        if example.next_acceptor_pos is None:
            raise RuntimeError("confirmation example lacks its paired acceptor")
        anchor = int(example.next_acceptor_pos)
        return {
            "window": ACCEPTOR_WINDOW,
            "anchor": anchor,
            "target_position": anchor,
        }
    raise ValueError(f"unknown endpoint {endpoint!r}")


def reconstruct_frozen_examples(
    discovery: Mapping[str, Any],
    *,
    basic_gtf: Path,
    fasta: Path,
    cds_intervals: Mapping[str, Sequence[tuple]],
) -> tuple[list[Any], dict[str, Any]]:
    """Reconstruct both frozen partitions and verify their recorded identities."""

    confirmation_rows = discovery.get("cohort", {}).get(
        "confirmation_freeze", {}
    ).get("examples")
    discovery_rows = discovery.get("cohort", {}).get("discovery", {}).get("examples")
    if (
        not isinstance(confirmation_rows, list)
        or len(confirmation_rows) != N_CONFIRMATION
        or not isinstance(discovery_rows, list)
    ):
        raise RuntimeError("discovery artifact lacks the frozen cohort identities")
    if len({row["gene_id"] for row in confirmation_rows}) != N_CONFIRMATION:
        raise RuntimeError("confirmation freeze must contain 200 unique genes")
    needed = {
        str(row["example_id"])
        for row in [*confirmation_rows, *discovery_rows]
    }
    by_id: dict[str, Any] = {}
    for example in eligible(_dataset(str(basic_gtf), str(fasta)), "train"):
        if example.example_id not in needed:
            continue
        if example.example_id in by_id:
            raise RuntimeError(f"duplicate reconstructed example {example.example_id}")
        by_id[example.example_id] = example
    missing = sorted(needed - set(by_id))
    if missing:
        raise RuntimeError(
            f"{len(missing)} frozen examples could not be reconstructed; first={missing[:3]}"
        )

    def verify(row: Mapping[str, Any]) -> Any:
        example = by_id[str(row["example_id"])]
        if (
            str(row["gene_id"]) != example.gene_id
            or str(row["chrom"]) != example.chrom
        ):
            raise RuntimeError(f"frozen identity mismatch for {example.example_id}")
        for endpoint in ("donor", "acceptor"):
            spec = endpoint_spec(example, endpoint)
            start, end = spec["window"].absolute(
                spec["anchor"], len(example.sequence)
            )
            annotation = row["annotation"][endpoint]
            starts = complete_codon_starts(
                example,
                spec["window"],
                cds_intervals,
                anchor=spec["anchor"],
            )
            if (
                start != int(annotation["absolute_start"])
                or end != int(annotation["absolute_end"])
                or tuple(int(value) for value in annotation["complete_codon_starts"])
                != starts
                or len(starts) != 10
            ):
                raise RuntimeError(
                    f"frozen coding-window reconstruction mismatch for "
                    f"{example.example_id}:{endpoint}"
                )
        return example

    confirmation = [verify(row) for row in confirmation_rows]
    for row in discovery_rows:
        verify(row)
    return confirmation, by_id


def exon_scorer(record: Mapping[str, Any]) -> Callable[[str], float]:
    names = tuple(record["kmer_names"])
    mean = np.asarray(record["scaler_mean"], dtype=np.float64)
    scale = np.asarray(record["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(record["coefficient"], dtype=np.float64)
    intercept = float(record["intercept"])
    if (
        mean.shape != scale.shape
        or mean.shape != coefficient.shape
        or np.any(scale <= 0.0)
    ):
        raise RuntimeError("invalid frozen exon-identity baseline")

    def score(sequence: str) -> float:
        vector, observed = kmer_count_vector(sequence)
        if observed != names:
            raise RuntimeError("frozen exon-identity k-mer basis mismatch")
        return float(((vector - mean) / scale) @ coefficient + intercept)

    return score


def codon_pair_log_preference(
    sequence: str,
    codon_starts: Sequence[int],
    pair_counts: Counter[tuple[str, str]],
) -> float:
    codons = [sequence[int(start) : int(start) + 3] for start in codon_starts]
    denominator = float(sum(pair_counts.values()) + 61 * 61)
    return float(
        sum(
            np.log((pair_counts[(left, right)] + 1.0) / denominator)
            for left, right in zip(codons, codons[1:])
        )
    )


def descriptor_values(
    sequence: str,
    *,
    example: Any,
    window: Any,
    anchor: int,
    codon_starts: tuple[int, ...],
    codon_counts: Counter[str],
    psams: Mapping[str, Any],
    exon_score: Callable[[str], float],
) -> dict[str, float]:
    start, end = window.absolute(anchor, len(sequence))
    values = {
        f"psam:{name}": float(value)
        for name, value in psam_score_vector(
            sequence, psams, start=start, end=end
        ).items()
    }
    context_start, context_end = start - 84, end + 85
    if context_start < 0 or context_end > len(sequence):
        raise RuntimeError("frozen RNA folding context falls outside the sequence")
    context = sequence[context_start:context_end]
    if len(context) != 201:
        raise RuntimeError("frozen RNA folding context must be exactly 201 nt")
    values.update(
        {
            "codon_preference": codon_log_preference(
                sequence, codon_starts, codon_counts
            ),
            "exon_identity": float(exon_score(sequence[start:end])),
            "rna_accessibility": vienna_unpaired_mean(
                context,
                start=start - context_start,
                end=end - context_start,
                unpaired_length=7,
                window_size=150,
                max_bp_span=100,
            ),
            "mfe": vienna_mfe(context),
        }
    )
    return values


def overlapping_splice_dinucleotide_positions(
    sequence: str, start: int, end: int
) -> set[int]:
    """Return canonical AG/GT starts whose dinucleotide overlaps the edit region."""

    return {
        position
        for position in range(max(0, start - 1), min(end, len(sequence) - 1))
        if sequence[position : position + 2] in {"AG", "GT"}
    }


def constrained_validator(
    clean: str,
    start: int,
    end: int,
    score3: Callable[[str], float],
    score5: Callable[[str], float],
) -> Callable[[str], bool]:
    clean_motifs = overlapping_splice_dinucleotide_positions(clean, start, end)

    def validate(candidate: str) -> bool:
        candidate_motifs = overlapping_splice_dinucleotide_positions(
            candidate, start, end
        )
        if not candidate_motifs.issubset(clean_motifs):
            return False
        changes = (
            compare_landscapes(
                clean,
                candidate,
                start,
                end,
                window_length=23,
                scorer=score3,
            ),
            compare_landscapes(
                clean,
                candidate,
                start,
                end,
                window_length=9,
                scorer=score5,
            ),
        )
        return all(
            abs(change.delta_max) <= MAX_ABS_MAXENT_DELTA
            and not change.new_competitive_site
            for change in changes
        )

    return validate


def replay_codon_changes(clean: str, record: Mapping[str, Any]) -> str:
    sequence = clean
    seen: set[int] = set()
    for change in record.get("codon_changes", []):
        start = int(change["codon_start"])
        original = str(change["original"])
        alternate = str(change["alternate"])
        if start in seen or clean[start : start + 3] != original:
            raise RuntimeError("recorded codon edit cannot be replayed")
        if codon_amino_acid(original) != codon_amino_acid(alternate):
            raise RuntimeError("recorded codon edit is not synonymous")
        seen.add(start)
        sequence = sequence[:start] + alternate + sequence[start + 3 :]
    return sequence


def gc3_high_minus_low(
    high: str, low: str, codon_starts: Sequence[int]
) -> int:
    """Actual GC3 contrast; the legacy audit accidentally indexed codon position 1."""

    return int(
        sum(high[int(start) + 2] in "GC" for start in codon_starts)
        - sum(low[int(start) + 2] in "GC" for start in codon_starts)
    )


def nuisance_from_audit_row(
    row: Mapping[str, Any],
    *,
    clean: str,
    codon_starts: Sequence[int],
) -> dict[str, float]:
    nuisance = row.get("nuisance")
    if not isinstance(nuisance, Mapping) or nuisance.get("construction_failed"):
        raise RuntimeError("successful discovery construction lacks nuisance values")
    low = replay_codon_changes(clean, row["low"])
    high = replay_codon_changes(clean, row["high"])
    if (
        sha256_text(low) != row["low_sha256"]
        or sha256_text(high) != row["high_sha256"]
    ):
        raise RuntimeError("corrected discovery construction replay hash mismatch")
    values = {
        name: float(nuisance[name])
        for name in NUISANCE_NAMES
        if name != "gc3_high_minus_low"
    }
    values["gc3_high_minus_low"] = float(
        gc3_high_minus_low(high, low, codon_starts)
    )
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("non-finite discovery nuisance value")
    return values


def higher_quantile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("non-empty values and a valid quantile are required")
    ordered = sorted(float(value) for value in values)
    index = int(math.ceil(quantile * (len(ordered) - 1)))
    return ordered[index]


def freeze_nuisance_calipers(
    construction_audit: Mapping[str, Any],
    *,
    discovery_examples: Mapping[str, Any],
    cds_intervals: Mapping[str, Sequence[tuple]],
) -> dict[str, Any]:
    """Freeze candidate-level bands from corrected discovery rows only."""

    result: dict[str, Any] = {}
    for endpoint, candidates in EXPECTED_FAMILIES.items():
        result[endpoint] = {}
        for candidate_name, controls in candidates.items():
            audit = construction_audit["audit"][endpoint][candidate_name]
            candidate_by_id = {
                row["example_id"]: row
                for row in audit["families"][candidate_name]["rows"]
                if row["success"]
            }
            differences = {name: [] for name in NUISANCE_NAMES}
            per_control_counts: dict[str, int] = {}
            for control_name in controls:
                matched = 0
                for control_row in audit["families"][control_name]["rows"]:
                    candidate_row = candidate_by_id.get(control_row["example_id"])
                    if candidate_row is None or not control_row["success"]:
                        continue
                    candidate_doses = (
                        int(candidate_row["low"]["changed_bases"]),
                        int(candidate_row["high"]["changed_bases"]),
                    )
                    control_doses = (
                        int(control_row["low"]["changed_bases"]),
                        int(control_row["high"]["changed_bases"]),
                    )
                    if candidate_doses != control_doses:
                        continue
                    example = discovery_examples[str(control_row["example_id"])]
                    spec = endpoint_spec(example, endpoint)
                    starts = complete_codon_starts(
                        example,
                        spec["window"],
                        cds_intervals,
                        anchor=spec["anchor"],
                    )
                    candidate_nuisance = nuisance_from_audit_row(
                        candidate_row,
                        clean=example.sequence,
                        codon_starts=starts,
                    )
                    control_nuisance = nuisance_from_audit_row(
                        control_row,
                        clean=example.sequence,
                        codon_starts=starts,
                    )
                    for name in NUISANCE_NAMES:
                        differences[name].append(
                            abs(candidate_nuisance[name] - control_nuisance[name])
                        )
                    matched += 1
                per_control_counts[control_name] = matched
            if any(count == 0 for count in per_control_counts.values()):
                raise RuntimeError(
                    f"no dose-matched discovery calibration rows for "
                    f"{endpoint}:{candidate_name}"
                )
            calipers = {
                name: higher_quantile(values, NUISANCE_CALIPER_QUANTILE)
                for name, values in differences.items()
            }
            result[endpoint][candidate_name] = {
                "source_partition": "discovery",
                "source_audit_experiment": construction_audit["experiment"],
                "quantile": NUISANCE_CALIPER_QUANTILE,
                "quantile_method": "higher",
                "pooled_across_three_frozen_controls": True,
                "gc3_definition": "third nucleotide of each complete codon",
                "dose_matched_pairs_by_control": per_control_counts,
                "pooled_pair_count": sum(per_control_counts.values()),
                "calipers": calipers,
                "calibration_summaries": {
                    name: {
                        "minimum": float(min(values)),
                        "median": float(np.median(values)),
                        "maximum": float(max(values)),
                        "n": len(values),
                    }
                    for name, values in differences.items()
                },
                "calibration_values_sha256": sha256_text(
                    canonical_json(differences)
                ),
            }
    return result


def landscape_audit(
    clean: str,
    changed: str,
    *,
    start: int,
    end: int,
    score3: Callable[[str], float],
    score5: Callable[[str], float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, window_length, scorer in (
        ("acceptor_23nt", 23, score3),
        ("donor_9nt", 9, score5),
    ):
        change = compare_landscapes(
            clean,
            changed,
            start,
            end,
            window_length=window_length,
            scorer=scorer,
        )
        result[name] = {
            "clean_max": float(change.before.max_score),
            "changed_max": float(change.after.max_score),
            "delta_max": float(change.delta_max),
            "delta_n_strong": int(change.delta_n_strong),
            "new_competitive_site": bool(change.new_competitive_site),
            "pass": bool(
                abs(change.delta_max) <= MAX_ABS_MAXENT_DELTA
                and not change.new_competitive_site
            ),
        }
    return result


def arm_invariant_audit(
    *,
    example: Any,
    endpoint: str,
    sequence: str,
    beam_record: Mapping[str, Any],
    expected_dose: int,
    codon_starts: Sequence[int],
    start: int,
    end: int,
    score3: Callable[[str], float],
    score5: Callable[[str], float],
) -> tuple[dict[str, bool], dict[str, Any], tuple[int, ...]]:
    clean = str(example.sequence)
    changed_positions = tuple(
        index
        for index, (left, right) in enumerate(zip(clean, sequence))
        if left != right
    )
    replayed = replay_codon_changes(clean, beam_record)
    clean_motifs = overlapping_splice_dinucleotide_positions(clean, start, end)
    changed_motifs = overlapping_splice_dinucleotide_positions(
        sequence, start, end
    )
    maxent = landscape_audit(
        clean,
        sequence,
        start=start,
        end=end,
        score3=score3,
        score5=score5,
    )
    translation_preserved = all(
        codon_amino_acid(clean[int(codon_start) : int(codon_start) + 3])
        == codon_amino_acid(sequence[int(codon_start) : int(codon_start) + 3])
        for codon_start in codon_starts
    )
    target_sites = {
        "donor": (
            clean[int(example.donor_pos) : int(example.donor_pos) + 2],
            sequence[int(example.donor_pos) : int(example.donor_pos) + 2],
            "GT",
        ),
        "acceptor": (
            clean[
                int(example.next_acceptor_pos) - 2 : int(example.next_acceptor_pos)
            ],
            sequence[
                int(example.next_acceptor_pos) - 2 : int(example.next_acceptor_pos)
            ],
            "AG",
        ),
    }
    invariants = {
        "equal_length": len(clean) == len(sequence),
        "raw_sequence_replays_codon_edits": replayed == sequence,
        "changed_positions_match_delivered_dose": (
            len(changed_positions) == expected_dose
            and len(changed_positions)
            == int(beam_record.get("changed_bases", -1))
        ),
        "all_edits_inside_target_region": all(
            start <= position < end for position in changed_positions
        ),
        "outside_target_region_unchanged": (
            clean[:start] == sequence[:start] and clean[end:] == sequence[end:]
        ),
        "amino_acid_sequence_preserved": translation_preserved,
        "no_new_ag_or_gt_overlapping_target_region": (
            changed_motifs.issubset(clean_motifs)
        ),
        "canonical_target_donor_preserved": (
            target_sites["donor"][0]
            == target_sites["donor"][1]
            == target_sites["donor"][2]
        ),
        "canonical_target_acceptor_preserved": (
            target_sites["acceptor"][0]
            == target_sites["acceptor"][1]
            == target_sites["acceptor"][2]
        ),
        "maxent_acceptor_band_pass": bool(maxent["acceptor_23nt"]["pass"]),
        "maxent_donor_band_pass": bool(maxent["donor_9nt"]["pass"]),
    }
    return invariants, maxent, changed_positions


def failed_arm(clean: str, requested_dose: int | None, reason: str) -> dict[str, Any]:
    return {
        "sequence": clean,
        "sequence_sha256": sha256_text(clean),
        "success": False,
        "failure_reason": reason,
        "requested_changed_bases": requested_dose,
        "delivered_changed_bases": 0,
        "changed_positions": [],
        "codon_changes": [],
        "objective_clean": None,
        "objective_score": None,
        "invariants": {},
        "maxent": {},
    }


def construct_family_pair(
    *,
    example: Any,
    endpoint: str,
    model: Any,
    cds_intervals: Mapping[str, Sequence[tuple]],
    codon_counts: Counter[str],
    codon_pair_counts: Counter[tuple[str, str]],
    psams: Mapping[str, Any],
    exon_score: Callable[[str], float],
    score3: Callable[[str], float],
    score5: Callable[[str], float],
    exact_doses: tuple[int, int] | None,
    descriptor_function: Callable[..., dict[str, float]] = descriptor_values,
) -> dict[str, Any]:
    """Construct one candidate/control low-high pair, optionally at exact doses."""

    spec = endpoint_spec(example, endpoint)
    window, anchor = spec["window"], spec["anchor"]
    start, end = window.absolute(anchor, len(example.sequence))
    starts = complete_codon_starts(
        example, window, cds_intervals, anchor=anchor
    )
    objective = lambda sequence: psam_region_score(
        sequence, model, start=start, end=end
    )
    validator = constrained_validator(
        example.sequence, start, end, score3, score5
    )
    requested = {
        "low": None if exact_doses is None else int(exact_doses[0]),
        "high": None if exact_doses is None else int(exact_doses[1]),
    }
    arms: dict[str, Any] = {}
    for direction, maximize in (("low", False), ("high", True)):
        minimum = (
            MIN_CHANGED_BASES
            if requested[direction] is None
            else requested[direction]
        )
        maximum = (
            MAX_CHANGED_BASES
            if requested[direction] is None
            else requested[direction]
        )
        try:
            sequence, beam_record = beam_synonymous_optimize(
                example,
                window,
                cds_intervals,
                anchor=anchor,
                objective=objective,
                maximize=maximize,
                min_changed_bases=minimum,
                max_changed_bases=maximum,
                beam_size=BEAM_SIZE,
                validator=validator,
            )
            delivered = int(beam_record["changed_bases"])
            expected_dose = delivered if requested[direction] is None else maximum
            invariants, maxent, changed_positions = arm_invariant_audit(
                example=example,
                endpoint=endpoint,
                sequence=sequence,
                beam_record=beam_record,
                expected_dose=expected_dose,
                codon_starts=starts,
                start=start,
                end=end,
                score3=score3,
                score5=score5,
            )
            success = all(invariants.values())
            arms[direction] = {
                "sequence": sequence,
                "sequence_sha256": sha256_text(sequence),
                "success": bool(success),
                "failure_reason": None if success else "invariant_failure",
                "requested_changed_bases": requested[direction],
                "delivered_changed_bases": delivered,
                "changed_positions": list(changed_positions),
                "codon_changes": list(beam_record["codon_changes"]),
                "objective_clean": float(beam_record["objective_clean"]),
                "objective_score": float(beam_record["objective_changed"]),
                "invariants": invariants,
                "maxent": maxent,
            }
        except RuntimeError as exc:
            arms[direction] = failed_arm(
                example.sequence,
                requested[direction],
                f"beam_search:{type(exc).__name__}",
            )

    pair_success = bool(
        arms["low"]["success"]
        and arms["high"]["success"]
        and arms["low"]["sequence"] != arms["high"]["sequence"]
        and arms["high"]["objective_score"] > arms["low"]["objective_score"]
    )
    failure_reason = None
    if not pair_success:
        failure_reason = "low_high_contrast_unavailable"
    nuisance = None
    if pair_success:
        low_values = descriptor_function(
            arms["low"]["sequence"],
            example=example,
            window=window,
            anchor=anchor,
            codon_starts=starts,
            codon_counts=codon_counts,
            psams=psams,
            exon_score=exon_score,
        )
        high_values = descriptor_function(
            arms["high"]["sequence"],
            example=example,
            window=window,
            anchor=anchor,
            codon_starts=starts,
            codon_counts=codon_counts,
            psams=psams,
            exon_score=exon_score,
        )
        family_name = str(model.name)
        nuisance = {
            "off_target_psam_l1": float(
                sum(
                    abs(
                        high_values[f"psam:{other}"]
                        - low_values[f"psam:{other}"]
                    )
                    for other in psams
                    if other != family_name
                )
            ),
            "codon_preference_high_minus_low": float(
                high_values["codon_preference"]
                - low_values["codon_preference"]
            ),
            "exon_identity_high_minus_low": float(
                high_values["exon_identity"] - low_values["exon_identity"]
            ),
            "rna_accessibility_high_minus_low": float(
                high_values["rna_accessibility"]
                - low_values["rna_accessibility"]
            ),
            "mfe_high_minus_low": float(
                high_values["mfe"] - low_values["mfe"]
            ),
            "gc3_high_minus_low": float(
                gc3_high_minus_low(
                    arms["high"]["sequence"], arms["low"]["sequence"], starts
                )
            ),
            "codon_pair_high_minus_low": float(
                codon_pair_log_preference(
                    arms["high"]["sequence"], starts, codon_pair_counts
                )
                - codon_pair_log_preference(
                    arms["low"]["sequence"], starts, codon_pair_counts
                )
            ),
            "canonical_ag_gt_high_minus_low": float(
                len(
                    overlapping_splice_dinucleotide_positions(
                        arms["high"]["sequence"], start, end
                    )
                )
                - len(
                    overlapping_splice_dinucleotide_positions(
                        arms["low"]["sequence"], start, end
                    )
                )
            ),
        }
        if not all(math.isfinite(value) for value in nuisance.values()):
            raise RuntimeError("non-finite confirmation nuisance descriptor")
    return {
        "pair_success": pair_success,
        "failure_reason": failure_reason,
        "arms": arms,
        "nuisance": nuisance,
        "pair_score_high_minus_low": (
            float(
                arms["high"]["objective_score"] - arms["low"]["objective_score"]
            )
            if pair_success
            else None
        ),
    }


def unavailable_control_pair(
    clean: str, exact_doses: tuple[int, int] | None
) -> dict[str, Any]:
    low_dose = None if exact_doses is None else int(exact_doses[0])
    high_dose = None if exact_doses is None else int(exact_doses[1])
    return {
        "pair_success": False,
        "failure_reason": "candidate_dose_unavailable",
        "arms": {
            "low": failed_arm(clean, low_dose, "candidate_dose_unavailable"),
            "high": failed_arm(clean, high_dose, "candidate_dose_unavailable"),
        },
        "nuisance": None,
        "pair_score_high_minus_low": None,
    }


def nuisance_match(
    candidate: Mapping[str, float],
    control: Mapping[str, float],
    calipers: Mapping[str, float],
) -> tuple[bool, dict[str, float], dict[str, bool]]:
    if set(candidate) != set(NUISANCE_NAMES) or set(control) != set(
        NUISANCE_NAMES
    ):
        raise RuntimeError("nuisance descriptor family changed")
    differences = {
        name: abs(float(candidate[name]) - float(control[name]))
        for name in NUISANCE_NAMES
    }
    passes = {
        name: differences[name] <= float(calipers[name])
        for name in NUISANCE_NAMES
    }
    return all(passes.values()), differences, passes


def state_rows_for_pair(
    *,
    example: Any,
    endpoint: str,
    candidate_name: str,
    family_name: str,
    family_role: str,
    expected_sign: int,
    pair: Mapping[str, Any],
    nuisance_difference: Mapping[str, float] | None,
    nuisance_caliper_pass: bool | None,
    usable_matched_set: bool,
) -> list[dict[str, Any]]:
    spec = endpoint_spec(example, endpoint)
    start, end = spec["window"].absolute(
        spec["anchor"], len(example.sequence)
    )
    candidate_id = f"{endpoint}:{candidate_name}"
    rows = []
    for direction in ("low", "high"):
        arm = pair["arms"][direction]
        rows.append(
            {
                "schema": STATE_SCHEMA,
                "state_id": (
                    f"{candidate_id}:{family_name}:{example.example_id}:{direction}"
                ),
                "candidate_id": candidate_id,
                "endpoint": endpoint,
                "candidate_name": candidate_name,
                "family_name": family_name,
                "family_role": family_role,
                "direction": direction,
                "expected_high_minus_low_margin_sign": int(expected_sign),
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "chrom": example.chrom,
                "strand": example.strand,
                "target_position": int(spec["target_position"]),
                "anchor_position": int(spec["anchor"]),
                "relative_window": spec["window"].name,
                "region_start": int(start),
                "region_end": int(end),
                "reference_sequence_sha256": sha256_text(example.sequence),
                "sequence": arm["sequence"],
                "sequence_sha256": arm["sequence_sha256"],
                "arm_success": bool(arm["success"]),
                "pair_success": bool(pair["pair_success"]),
                "failure_reason": (
                    arm["failure_reason"]
                    if arm["failure_reason"] is not None
                    else pair["failure_reason"]
                ),
                "requested_changed_bases": arm["requested_changed_bases"],
                "delivered_changed_bases": int(arm["delivered_changed_bases"]),
                "changed_positions": list(arm["changed_positions"]),
                "codon_changes": list(arm["codon_changes"]),
                "objective_clean": arm["objective_clean"],
                "objective_score": arm["objective_score"],
                "pair_score_high_minus_low": pair[
                    "pair_score_high_minus_low"
                ],
                "invariants": dict(arm["invariants"]),
                "maxent": dict(arm["maxent"]),
                "pair_nuisance": (
                    None if pair["nuisance"] is None else dict(pair["nuisance"])
                ),
                "nuisance_difference_from_candidate": (
                    None
                    if nuisance_difference is None
                    else dict(nuisance_difference)
                ),
                "nuisance_caliper_pass": nuisance_caliper_pass,
                "usable_matched_set": bool(usable_matched_set),
            }
        )
    return rows


def construct_candidate_package(
    *,
    examples: Sequence[Any],
    endpoint: str,
    candidate_name: str,
    controls: Sequence[str],
    expected_sign: int,
    caliper_record: Mapping[str, Any],
    cds_intervals: Mapping[str, Sequence[tuple]],
    codon_counts: Counter[str],
    codon_pair_counts: Counter[tuple[str, str]],
    psams: Mapping[str, Any],
    exon_score: Callable[[str], float],
    score3: Callable[[str], float],
    score5: Callable[[str], float],
    descriptor_function: Callable[..., dict[str, float]] = descriptor_values,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    states: list[dict[str, Any]] = []
    candidate_success_ids: list[str] = []
    usable_ids: list[str] = []
    family_success: Counter[str] = Counter()
    nuisance_pass: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    calipers = caliper_record["calipers"]
    for example in examples:
        candidate_pair = construct_family_pair(
            example=example,
            endpoint=endpoint,
            model=psams[candidate_name],
            cds_intervals=cds_intervals,
            codon_counts=codon_counts,
            codon_pair_counts=codon_pair_counts,
            psams=psams,
            exon_score=exon_score,
            score3=score3,
            score5=score5,
            exact_doses=None,
            descriptor_function=descriptor_function,
        )
        exact_doses = None
        if candidate_pair["pair_success"]:
            exact_doses = (
                int(candidate_pair["arms"]["low"]["delivered_changed_bases"]),
                int(candidate_pair["arms"]["high"]["delivered_changed_bases"]),
            )
            candidate_success_ids.append(example.example_id)
            family_success[candidate_name] += 1
        else:
            failure_reasons[str(candidate_pair["failure_reason"])] += 1

        control_pairs: dict[str, dict[str, Any]] = {}
        control_differences: dict[str, dict[str, float] | None] = {}
        control_passes: dict[str, bool | None] = {}
        for control_name in controls:
            if exact_doses is None:
                control_pair = unavailable_control_pair(
                    example.sequence, exact_doses
                )
            else:
                control_pair = construct_family_pair(
                    example=example,
                    endpoint=endpoint,
                    model=psams[control_name],
                    cds_intervals=cds_intervals,
                    codon_counts=codon_counts,
                    codon_pair_counts=codon_pair_counts,
                    psams=psams,
                    exon_score=exon_score,
                    score3=score3,
                    score5=score5,
                    exact_doses=exact_doses,
                    descriptor_function=descriptor_function,
                )
            control_pairs[control_name] = control_pair
            if control_pair["pair_success"]:
                family_success[control_name] += 1
            else:
                failure_reasons[
                    f"{control_name}:{control_pair['failure_reason']}"
                ] += 1
            if (
                candidate_pair["pair_success"]
                and control_pair["pair_success"]
            ):
                matched, differences, _passes = nuisance_match(
                    candidate_pair["nuisance"],
                    control_pair["nuisance"],
                    calipers,
                )
                control_differences[control_name] = differences
                control_passes[control_name] = matched
                if matched:
                    nuisance_pass[control_name] += 1
            else:
                control_differences[control_name] = None
                control_passes[control_name] = None

        usable = bool(
            candidate_pair["pair_success"]
            and all(
                control_pairs[name]["pair_success"]
                and control_passes[name] is True
                for name in controls
            )
        )
        if usable:
            usable_ids.append(example.example_id)
        states.extend(
            state_rows_for_pair(
                example=example,
                endpoint=endpoint,
                candidate_name=candidate_name,
                family_name=candidate_name,
                family_role="candidate",
                expected_sign=expected_sign,
                pair=candidate_pair,
                nuisance_difference=None,
                nuisance_caliper_pass=None,
                usable_matched_set=usable,
            )
        )
        for control_name in controls:
            states.extend(
                state_rows_for_pair(
                    example=example,
                    endpoint=endpoint,
                    candidate_name=candidate_name,
                    family_name=control_name,
                    family_role="matched_control",
                    expected_sign=expected_sign,
                    pair=control_pairs[control_name],
                    nuisance_difference=control_differences[control_name],
                    nuisance_caliper_pass=control_passes[control_name],
                    usable_matched_set=usable,
                )
            )

    genes_by_id = {example.example_id: example.gene_id for example in examples}
    candidate_gene_count = len(
        {genes_by_id[example_id] for example_id in candidate_success_ids}
    )
    usable_gene_count = len({genes_by_id[example_id] for example_id in usable_ids})
    gate = (
        len(usable_ids) >= MIN_CONFIRMATION_EXAMPLES
        and usable_gene_count >= MIN_CONFIRMATION_GENES
    )
    summary = {
        "candidate_id": f"{endpoint}:{candidate_name}",
        "endpoint": endpoint,
        "candidate_name": candidate_name,
        "matched_controls": list(controls),
        "expected_high_minus_low_margin_sign": int(expected_sign),
        "candidate_construction_success_count": len(candidate_success_ids),
        "candidate_construction_success_gene_count": candidate_gene_count,
        "family_construction_success_count": {
            name: int(family_success[name])
            for name in (candidate_name, *controls)
        },
        "control_nuisance_match_count": {
            name: int(nuisance_pass[name]) for name in controls
        },
        "complete_dose_and_nuisance_matched_set_count": len(usable_ids),
        "complete_dose_and_nuisance_matched_gene_count": usable_gene_count,
        "eligible_example_ids": usable_ids,
        "minimum_confirmation_examples": MIN_CONFIRMATION_EXAMPLES,
        "minimum_confirmation_genes": MIN_CONFIRMATION_GENES,
        "confirmation_construction_gate": bool(gate),
        "failure_reasons": dict(sorted(failure_reasons.items())),
    }
    return summary, states


def _forbid_outcome_payload(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in FORBIDDEN_OUTCOME_KEYS:
                raise RuntimeError(f"outcome-bearing field escaped at {path}.{key}")
            _forbid_outcome_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _forbid_outcome_payload(item, path=f"{path}[{index}]")


def validate_confirmation_package(package: Mapping[str, Any]) -> None:
    """Strict final schema and outcome-blinding audit before serialization."""

    if set(package) != TOP_LEVEL_KEYS:
        raise RuntimeError(
            f"confirmation package keys changed: "
            f"missing={sorted(TOP_LEVEL_KEYS - set(package))}, "
            f"extra={sorted(set(package) - TOP_LEVEL_KEYS)}"
        )
    if (
        package.get("schema") != SCHEMA
        or package.get("model_loaded") is not False
        or package.get("confirmation_model_outputs_read") is not False
        or package.get("outcome_blind") is not True
    ):
        raise RuntimeError("confirmation package outcome-blinding contract failed")
    summaries = package.get("candidate_summaries")
    states = package.get("states")
    if not isinstance(summaries, list) or len(summaries) != 4:
        raise RuntimeError("exactly four candidate summaries required")
    expected_state_count = (
        len(summaries) * 4 * N_CONFIRMATION * 2
    )
    if not isinstance(states, list) or len(states) != expected_state_count:
        raise RuntimeError(
            f"expected {expected_state_count} exact low/high state rows"
        )
    state_ids: set[str] = set()
    for row in states:
        if not isinstance(row, Mapping) or set(row) != STATE_KEYS:
            raise RuntimeError("confirmation state schema changed")
        state_id = str(row["state_id"])
        if state_id in state_ids:
            raise RuntimeError(f"duplicate confirmation state {state_id}")
        state_ids.add(state_id)
        sequence = str(row["sequence"])
        if (
            not sequence
            or set(sequence) - set("ACGTN")
            or sha256_text(sequence) != row["sequence_sha256"]
            or row["direction"] not in {"low", "high"}
            or row["family_role"] not in {"candidate", "matched_control"}
        ):
            raise RuntimeError(f"invalid confirmation state {state_id}")
        if row["pair_success"]:
            if (
                not row["arm_success"]
                or not row["invariants"]
                or not all(row["invariants"].values())
                or not MIN_CHANGED_BASES
                <= int(row["delivered_changed_bases"])
                <= MAX_CHANGED_BASES
            ):
                raise RuntimeError(f"successful state failed audit: {state_id}")
        if row["family_role"] == "matched_control" and row["arm_success"]:
            if (
                row["requested_changed_bases"]
                != row["delivered_changed_bases"]
            ):
                raise RuntimeError(f"control dose mismatch: {state_id}")
    _forbid_outcome_payload(package)


def cohort_record(examples: Sequence[Any]) -> dict[str, Any]:
    return {
        "namespace": "gtex-native-biological-features-confirmation-v1",
        "n_examples": len(examples),
        "n_genes": len({example.gene_id for example in examples}),
        "examples": [
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "chrom": example.chrom,
                "strand": example.strand,
                "donor_pos": int(example.donor_pos),
                "next_acceptor_pos": int(example.next_acceptor_pos),
                "sequence": example.sequence,
                "sequence_sha256": sha256_text(example.sequence),
            }
            for example in examples
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--construction-audit", required=True)
    parser.add_argument("--basic-gtf", required=True)
    parser.add_argument("--comprehensive-gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--rbns-psams", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]

    output, manifest_path = output_paths(args.out)
    discovery_path = Path(args.discovery).expanduser().resolve()
    audit_path = Path(args.construction_audit).expanduser().resolve()
    basic_gtf = Path(args.basic_gtf).expanduser().resolve()
    comprehensive_gtf = Path(args.comprehensive_gtf).expanduser().resolve()
    fasta = Path(args.fasta).expanduser().resolve()
    rbns_root = Path(args.rbns_psams).expanduser().resolve()
    source_records = {
        "discovery": require_frozen_source(discovery_path, "discovery"),
        "construction_audit": require_frozen_source(
            audit_path, "construction_audit"
        ),
        "basic_gtf": require_frozen_source(basic_gtf, "basic_gtf"),
        "comprehensive_gtf": require_frozen_source(
            comprehensive_gtf, "comprehensive_gtf"
        ),
        "fasta": require_frozen_source(fasta, "fasta"),
        "protocol": require_frozen_source(PROTOCOL, "protocol"),
        "rbns_psams": psam_source_record(rbns_root),
    }
    discovery = json.loads(discovery_path.read_text())
    construction_audit = json.loads(audit_path.read_text())
    frozen_candidates = validate_frozen_family_contract(
        discovery, construction_audit
    )
    cds_intervals = load_cds_intervals(comprehensive_gtf)
    examples, all_frozen_examples = reconstruct_frozen_examples(
        discovery,
        basic_gtf=basic_gtf,
        fasta=fasta,
        cds_intervals=cds_intervals,
    )
    psams = load_rbns_psam_directory(rbns_root)
    required_psams = {
        family
        for candidates in EXPECTED_FAMILIES.values()
        for candidate, controls in candidates.items()
        for family in (candidate, *controls)
    }
    if len(psams) != 79 or not required_psams.issubset(psams):
        raise RuntimeError("frozen PSAM family is incomplete")

    nuisance_calipers = freeze_nuisance_calipers(
        construction_audit,
        discovery_examples=all_frozen_examples,
        cds_intervals=cds_intervals,
    )
    baselines = discovery["sequence_baselines"]
    codon_counts: Counter[str] = Counter(
        {str(name): int(count) for name, count in baselines["codon_counts"].items()}
    )
    codon_pair_counts: Counter[tuple[str, str]] = Counter(
        {
            tuple(name.split(":")): int(count)
            for name, count in baselines["codon_pair_counts"].items()
        }
    )
    exon_score = exon_scorer(baselines["exon_identity"])
    score3, score5 = maxent_scorers()

    candidate_summaries: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for frozen in frozen_candidates:
        endpoint = frozen["endpoint"]
        candidate_name = frozen["candidate_name"]
        summary, candidate_states = construct_candidate_package(
            examples=examples,
            endpoint=endpoint,
            candidate_name=candidate_name,
            controls=frozen["matched_controls"],
            expected_sign=frozen[
                "expected_high_minus_low_margin_sign"
            ],
            caliper_record=nuisance_calipers[endpoint][candidate_name],
            cds_intervals=cds_intervals,
            codon_counts=codon_counts,
            codon_pair_counts=codon_pair_counts,
            psams=psams,
            exon_score=exon_score,
            score3=score3,
            score5=score5,
        )
        candidate_summaries.append(summary)
        states.extend(candidate_states)
        print(
            f"{summary['candidate_id']}: "
            f"{summary['complete_dose_and_nuisance_matched_set_count']} examples, "
            f"{summary['complete_dose_and_nuisance_matched_gene_count']} genes",
            flush=True,
        )

    stop_reasons = [
        {
            "candidate_id": row["candidate_id"],
            "reason": "minimum_complete_matched_confirmation_cohort_not_met",
            "examples": row[
                "complete_dose_and_nuisance_matched_set_count"
            ],
            "genes": row["complete_dose_and_nuisance_matched_gene_count"],
        }
        for row in candidate_summaries
        if not row["confirmation_construction_gate"]
    ]
    authorized = not stop_reasons
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "status": "COMPLETE" if authorized else "STOPPED_MINIMUM_COHORT",
        "evidence_tier": "outcome-blind confirmation construction",
        "protocol": str(PROTOCOL.relative_to(REPO)),
        "model_loaded": False,
        "confirmation_model_outputs_read": False,
        "outcome_blind": True,
        "confirmation_authorized": authorized,
        "source_records": source_records,
        "frozen_thresholds": {
            "confirmation_examples": N_CONFIRMATION,
            "candidate_assignments": 4,
            "matched_controls_per_candidate": 3,
            "beam_size": BEAM_SIZE,
            "candidate_min_changed_bases": MIN_CHANGED_BASES,
            "candidate_max_changed_bases": MAX_CHANGED_BASES,
            "control_low_dose": "exact candidate delivered low dose",
            "control_high_dose": "exact candidate delivered high dose",
            "max_abs_maxent_delta": MAX_ABS_MAXENT_DELTA,
            "no_new_competitive_site": True,
            "no_new_ag_or_gt_overlapping_target_region": True,
            "minimum_complete_matched_examples": MIN_CONFIRMATION_EXAMPLES,
            "minimum_complete_matched_genes": MIN_CONFIRMATION_GENES,
            "nuisance_caliper_quantile": NUISANCE_CALIPER_QUANTILE,
            "nuisance_caliper_method": "higher",
        },
        "nuisance_calipers": nuisance_calipers,
        "cohort": cohort_record(examples),
        "candidate_summaries": candidate_summaries,
        "states": states,
        "stop_reasons": stop_reasons,
        "evidence_boundary": (
            "This package freezes sequence-only candidate and matched-control "
            "counterfactuals. It contains no confirmation model output and licenses "
            "no classifier, sparse-feature, RBP-binding, or biological-mechanism claim."
        ),
    }
    validate_confirmation_package(result)
    atomic_write_json(output, result)
    manifest = ExperimentManifest.create(
        EXPERIMENT,
        command,
        repo=REPO,
        status=result["status"],
        config=result["frozen_thresholds"],
        inputs=[
            source_records[name]
            for name in (
                "protocol",
                "discovery",
                "construction_audit",
                "basic_gtf",
                "comprehensive_gtf",
                "fasta",
            )
        ]
        + source_records["rbns_psams"]["files"],
        outputs=[file_record(output)],
        notes=[
            (
                "No checkpoint or neural-model weights were loaded and no model "
                "inference was invoked."
            ),
            (
                "Nuisance calipers were frozen from corrected discovery "
                "constructions before confirmation sequence construction."
            ),
            "Control low/high searches used exact candidate-delivered low/high edit counts.",
            (
                "A failed minimum-cohort gate is serialized as a preserved "
                "non-authorizing STOPPED artifact."
            ),
        ],
    )
    atomic_write_json(manifest_path, manifest.to_dict())
    print(
        json.dumps(
            {
                "status": result["status"],
                "confirmation_authorized": authorized,
                "out": str(output),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
