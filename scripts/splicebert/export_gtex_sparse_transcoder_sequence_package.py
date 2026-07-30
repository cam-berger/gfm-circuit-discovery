#!/usr/bin/env python
"""Reconstruct and freeze the label-free discovery states for sparse training."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.circuits.multiview_transcoder import (  # noqa: E402
    LABEL_FREE_SEQUENCE_PACKAGE_SCHEMA,
    label_free_package_binding,
    validate_label_free_state_rows,
)
from gfmcd.provenance import (  # noqa: E402
    ExperimentManifest,
    file_record,
)
from gfmcd.splicing.biological_cohort import (  # noqa: E402
    ACCEPTOR_WINDOW,
    DONOR_WINDOW,
    NAMESPACE,
    audit_coding_window,
)
from gfmcd.splicing.biological_features import (  # noqa: E402
    maximal_synonymous_recode,
    synonymous_saturation_variants,
)
from gfmcd.splicing.circuit import _dataset, eligible  # noqa: E402
from gfmcd.splicing.context_localization import (  # noqa: E402
    RelativeWindow,
    shuffle_relative_window,
)
from gfmcd.splicing.p3_annotation import (  # noqa: E402
    load_annotation_safety_index,
)
from gfmcd.splicing.phase import load_cds_intervals  # noqa: E402


DONOR_SHAM = RelativeWindow(16, 48)
ACCEPTOR_SHAM = RelativeWindow(-80, -48)
EXPECTED_DISCOVERY_EXAMPLES = 128
PRODUCER = "scripts/splicebert/export_gtex_sparse_transcoder_sequence_package.py"


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def changed_positions(clean: str, changed: str) -> list[int]:
    if len(clean) != len(changed):
        raise ValueError("counterfactual changed sequence length")
    return [
        index
        for index, (left, right) in enumerate(zip(clean, changed))
        if left != right
    ]


def source_record(path: Path, *, role: str) -> dict[str, Any]:
    record = file_record(path)
    if record.get("missing"):
        raise FileNotFoundError(path)
    return {
        "path": str(Path(record["path"]).resolve()),
        "sha256": str(record["sha256"]),
        "size_bytes": int(record["size_bytes"]),
        "role": role,
    }


def state_row(
    *,
    example: Any,
    endpoint: str,
    target_position: int,
    variant_id: str,
    variant_kind: str,
    sequence: str,
    clean_state_id: str | None,
    clean_sequence: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_id = f"{example.example_id}|{endpoint}|{variant_id}"
    differences = (
        [] if clean_state_id is None else changed_positions(clean_sequence, sequence)
    )
    row = {
        "state_id": state_id,
        "example_id": str(example.example_id),
        "gene_id": str(example.gene_id),
        "endpoint": endpoint,
        "variant_id": variant_id,
        "variant_kind": variant_kind,
        "sequence": sequence,
        "sequence_sha256": sequence_sha256(sequence),
        "target_position": int(target_position),
        "reference_state_id": clean_state_id,
        "changed_positions": differences,
        "hamming": len(differences),
    }
    if extra:
        row.update(extra)
    return row


def build_states(
    examples: Iterable[Any],
    cds_intervals: dict[str, list[tuple]],
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for example in examples:
        specifications = {
            "donor": {
                "window": DONOR_WINDOW,
                "sham": DONOR_SHAM,
                "anchor": int(example.donor_pos),
                "target": int(example.donor_pos) - 1,
            },
            "acceptor": {
                "window": ACCEPTOR_WINDOW,
                "sham": ACCEPTOR_SHAM,
                "anchor": int(example.next_acceptor_pos),
                "target": int(example.next_acceptor_pos),
            },
        }
        for endpoint, specification in specifications.items():
            clean = str(example.sequence)
            clean_id = f"{example.example_id}|{endpoint}|clean"
            states.append(
                state_row(
                    example=example,
                    endpoint=endpoint,
                    target_position=specification["target"],
                    variant_id="clean",
                    variant_kind="clean",
                    sequence=clean,
                    clean_state_id=None,
                    clean_sequence=clean,
                )
            )
            target_shuffle, _ = shuffle_relative_window(
                clean,
                anchor=specification["anchor"],
                window=specification["window"],
                key=(
                    example.example_id,
                    endpoint,
                    "biological-discovery-target",
                ),
                draw=0,
            )
            sham_shuffle, _ = shuffle_relative_window(
                clean,
                anchor=specification["anchor"],
                window=specification["sham"],
                key=(
                    example.example_id,
                    endpoint,
                    "biological-discovery-sham",
                ),
                draw=0,
            )
            for variant_id, variant_kind, sequence in (
                ("target_shuffle", "target_shuffle", target_shuffle),
                ("sham_shuffle", "sham_shuffle", sham_shuffle),
            ):
                states.append(
                    state_row(
                        example=example,
                        endpoint=endpoint,
                        target_position=specification["target"],
                        variant_id=variant_id,
                        variant_kind=variant_kind,
                        sequence=sequence,
                        clean_state_id=clean_id,
                        clean_sequence=clean,
                    )
                )
            for draw in (0, 1):
                recoded, _ = maximal_synonymous_recode(
                    example,
                    specification["window"],
                    cds_intervals,
                    anchor=specification["anchor"],
                    draw=draw,
                    forbid_new_splice_dinucleotides=True,
                )
                states.append(
                    state_row(
                        example=example,
                        endpoint=endpoint,
                        target_position=specification["target"],
                        variant_id=f"max_recode_draw{draw}",
                        variant_kind=f"max_recode_draw{draw}",
                        sequence=recoded,
                        clean_state_id=clean_id,
                        clean_sequence=clean,
                    )
                )
            for variant in synonymous_saturation_variants(
                example,
                specification["window"],
                cds_intervals,
                anchor=specification["anchor"],
            ):
                variant_id = (
                    f"synonymous:{variant.codon_start}:"
                    f"{variant.original_codon}>{variant.alternate_codon}"
                )
                states.append(
                    state_row(
                        example=example,
                        endpoint=endpoint,
                        target_position=specification["target"],
                        variant_id=variant_id,
                        variant_kind="synonymous_single",
                        sequence=variant.sequence,
                        clean_state_id=clean_id,
                        clean_sequence=clean,
                        extra={
                            "codon_start": int(variant.codon_start),
                            "original_codon": variant.original_codon,
                            "alternate_codon": variant.alternate_codon,
                        },
                    )
                )
    return states


def reconstruct_examples(
    *,
    discovery: dict[str, Any],
    basic_gtf: Path,
    fasta: Path,
    annotation_index: Any,
    cds_intervals: dict[str, list[tuple]],
) -> list[Any]:
    cohort = discovery.get("cohort", {}).get("discovery", {})
    recorded = cohort.get("examples")
    if (
        discovery.get("status") != "COMPLETE"
        or not isinstance(recorded, list)
        or len(recorded) != EXPECTED_DISCOVERY_EXAMPLES
    ):
        raise RuntimeError("discovery artifact lacks the frozen 128-example cohort")
    ids = [str(row["example_id"]) for row in recorded]
    if len(set(ids)) != len(ids):
        raise RuntimeError("recorded discovery example IDs are not unique")
    wanted = set(ids)
    found: dict[str, Any] = {}
    for example in eligible(_dataset(str(basic_gtf), str(fasta)), "train"):
        if example.example_id not in wanted:
            continue
        if example.example_id in found:
            raise RuntimeError(f"duplicate dataset example ID {example.example_id}")
        found[example.example_id] = example
        if len(found) == len(wanted):
            break
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"could not reconstruct discovery examples: {missing[:5]}")

    ordered = []
    for row in recorded:
        example = found[str(row["example_id"])]
        for field, observed in (
            ("gene_id", example.gene_id),
            ("chrom", example.chrom),
            ("donor_pos", int(example.donor_pos)),
            ("next_acceptor_pos", int(example.next_acceptor_pos)),
        ):
            if row[field] != observed:
                raise RuntimeError(
                    f"discovery identity mismatch for {example.example_id}:{field}"
                )
        replay = {
            "donor": audit_coding_window(
                example,
                endpoint="donor",
                window=DONOR_WINDOW,
                anchor=int(example.donor_pos),
                annotation_index=annotation_index,
                cds_intervals=cds_intervals,
            ),
            "acceptor": audit_coding_window(
                example,
                endpoint="acceptor",
                window=ACCEPTOR_WINDOW,
                anchor=int(example.next_acceptor_pos),
                annotation_index=annotation_index,
                cds_intervals=cds_intervals,
            ),
        }
        replay_json = {
            endpoint: json.loads(json.dumps(asdict(audit), sort_keys=True))
            for endpoint, audit in replay.items()
        }
        if replay_json != row["annotation"] or not all(
            audit.passed for audit in replay.values()
        ):
            raise RuntimeError(
                f"CDS audit does not replay for {example.example_id}"
            )
        ordered.append(example)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--basic-gtf", required=True)
    parser.add_argument("--comprehensive-gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preflight_path = Path(args.preflight).expanduser().resolve()
    discovery_path = Path(args.discovery).expanduser().resolve()
    basic_gtf = Path(args.basic_gtf).expanduser().resolve()
    comprehensive_gtf = Path(args.comprehensive_gtf).expanduser().resolve()
    fasta = Path(args.fasta).expanduser().resolve()
    output = guard_output_path(Path(args.out).expanduser().resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    if output == manifest_path or output.suffix != ".json":
        raise ValueError("sequence package requires a distinct .json output")
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_BIOLOGICAL_FEATURES_V1.md"
    )
    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_SPARSE_TRANSCODER_V1.md"
    )
    producer_path = REPO / PRODUCER

    preflight = json.loads(preflight_path.read_text())
    discovery = json.loads(discovery_path.read_text())
    if preflight.get("status") != "PASS":
        raise RuntimeError("biological-feature preflight did not pass")
    if discovery.get("status") != "COMPLETE":
        raise RuntimeError("biological-feature discovery is incomplete")
    if str(discovery.get("preflight")) != str(preflight_path):
        raise RuntimeError("discovery artifact is not bound to the requested preflight")

    annotation_index = load_annotation_safety_index(comprehensive_gtf)
    cds_intervals = load_cds_intervals(comprehensive_gtf)
    examples = reconstruct_examples(
        discovery=discovery,
        basic_gtf=basic_gtf,
        fasta=fasta,
        annotation_index=annotation_index,
        cds_intervals=cds_intervals,
    )
    states = build_states(examples, cds_intervals)
    counts = Counter(str(row["variant_kind"]) for row in states)
    endpoint_counts = Counter(str(row["endpoint"]) for row in states)
    sources = {
        role: source_record(path, role=role)
        for role, path in (
            ("preflight", preflight_path),
            ("discovery", discovery_path),
            ("protocol", protocol),
            ("addendum", addendum),
            ("basic_gtf", basic_gtf),
            ("comprehensive_gtf", comprehensive_gtf),
            ("fasta", fasta),
            ("producer", producer_path),
        )
    }
    authorization = {
        "status": "PRECONDITIONS_VERIFIED",
        "preflight_status": "PASS",
        "discovery_status": "COMPLETE",
        "producer": PRODUCER,
    }
    authorization["binding_sha256"] = label_free_package_binding(
        authorization,
        sources,
    )
    package = {
        "schema": LABEL_FREE_SEQUENCE_PACKAGE_SCHEMA,
        "cohort_namespace": NAMESPACE,
        "partition": "discovery",
        "label_free": True,
        "sparse_training_authorized": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol.relative_to(REPO)),
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "authorization": authorization,
        "sources": sources,
        "counts": {
            "states": len(states),
            "examples": len(examples),
            "genes": len({example.gene_id for example in examples}),
            "by_variant_kind": dict(sorted(counts.items())),
            "by_endpoint": dict(sorted(endpoint_counts.items())),
        },
        "states": states,
    }
    validated = validate_label_free_state_rows(
        package,
        namespace=NAMESPACE,
        partition="discovery",
    )
    if len(validated) != len(states):
        raise RuntimeError("sequence-package validation changed the state count")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:sparse_sequence_package_v1",
        sys.argv,
        repo=REPO,
        status="INSTRUMENT",
        config={
            "cohort_namespace": NAMESPACE,
            "partition": "discovery",
            "n_examples": len(examples),
            "n_states": len(states),
        },
        notes=[
            "The discovery JSON container was parsed, but only the frozen cohort identity and annotation subtree affected output.",
            "No classifier outcome, candidate rank, descriptor, or external assay value is present in the package.",
            "Activation capture independently re-hashes all bound sources.",
        ],
    )
    for path in (
        protocol,
        addendum,
        preflight_path,
        discovery_path,
        basic_gtf,
        comprehensive_gtf,
        fasta,
        producer_path,
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": authorization["status"],
                "counts": package["counts"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
