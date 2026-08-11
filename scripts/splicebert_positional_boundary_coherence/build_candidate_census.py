#!/usr/bin/env python
"""Build the real, model-free Phase-I source-candidate census."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.pbc_biology import (  # noqa: E402
    CANDIDATE_ALGORITHM,
    CANDIDATE_SCHEMA,
    DEFAULT_ALLOWED_CHROMS,
    FORBIDDEN_CHROMS,
    IndexedFasta,
    MaxEntTables,
    enumerate_edge_evidence,
    exclusion_record,
    file_record,
    module_record,
    scaffold_record,
)
from gfmcd.splicing.pbc_freshness import (  # noqa: E402
    AUDIT_SCHEMA as FRESHNESS_AUDIT_SCHEMA,
    REGISTRY_SCHEMA as FRESHNESS_REGISTRY_SCHEMA,
    sha256_payload,
)
from gfmcd.splicing.positional_boundary_coherence import (  # noqa: E402
    validate_candidate_package,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve(strict=True).open("rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _validate_freshness(
    registry: Mapping[str, Any], audit: Mapping[str, Any]
) -> set[str]:
    if registry.get("schema") != FRESHNESS_REGISTRY_SCHEMA:
        raise ValueError("candidate census requires the frozen freshness registry schema")
    if registry.get("stage") != "construction":
        raise ValueError("freshness registry must be a construction-stage artifact")
    if (
        audit.get("schema") != FRESHNESS_AUDIT_SCHEMA
        or audit.get("stage") != "construction"
        or audit.get("status") != "AUDIT_PASS"
        or audit.get("passed") is not True
    ):
        raise ValueError("candidate census requires a passing independent freshness audit")
    if audit.get("registry_content_sha256") != registry.get("content_sha256"):
        raise ValueError("freshness audit is bound to another registry")
    genes = registry.get("gene_ids")
    if not isinstance(genes, list) or not genes or genes != sorted(set(genes)):
        raise ValueError("freshness registry gene_ids must be non-empty, sorted, and unique")
    if registry.get("gene_ids_sha256") != sha256_payload(genes):
        raise ValueError("freshness registry gene hash mismatch")
    return {str(value) for value in genes}


def construct_candidate_package(
    *,
    fasta_path: str | Path,
    fai_path: str | Path,
    basic_gtf_path: str | Path,
    comprehensive_gtf_path: str | Path,
    donor_table_path: str | Path,
    acceptor_table_path: str | Path,
    freshness_registry: Mapping[str, Any],
    freshness_audit: Mapping[str, Any],
    allowed_chromosomes: set[str] | frozenset[str] = DEFAULT_ALLOWED_CHROMS,
) -> dict[str, Any]:
    """Enumerate authentic candidates without loading SpliceBERT."""

    allowed = frozenset(str(value) for value in allowed_chromosomes)
    if not allowed or allowed & FORBIDDEN_CHROMS:
        raise ValueError("allowed chromosomes are empty or include a globally sealed chromosome")
    blacklist = _validate_freshness(freshness_registry, freshness_audit)
    genome = IndexedFasta(fasta_path, fai_path)
    scorer = MaxEntTables(donor_table_path, acceptor_table_path)
    edges, annotation_exclusions = enumerate_edge_evidence(
        basic_gtf_path,
        comprehensive_gtf_path,
        allowed_chromosomes=allowed,
    )
    candidates: list[dict[str, object]] = []
    exclusions = list(annotation_exclusions)
    for edge in edges:
        root = edge.representative.gene_id.split(".", 1)[0]
        if root in blacklist:
            exclusions.append(exclusion_record(edge, "freshness_registry_gene"))
            continue
        try:
            donor = module_record(edge, genome, scorer, role="donor")
            acceptor = module_record(edge, genome, scorer, role="acceptor")
            candidates.extend((donor, acceptor))
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            exclusions.append(exclusion_record(edge, f"module:{type(exc).__name__}:{exc}"))
        try:
            candidates.append(scaffold_record(edge, genome, scorer))
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            exclusions.append(exclusion_record(edge, f"scaffold:{type(exc).__name__}:{exc}"))

    candidates.sort(key=lambda row: str(row["candidate_id"]))
    exclusions.sort(key=lambda row: (str(row["edge_id"]), tuple(row["reasons"])))
    if not candidates:
        raise RuntimeError("real census produced no eligible candidates")
    counts = Counter(str(row["role"]) for row in candidates)
    for row in candidates:
        role = str(row["role"])
        if role in {"donor", "acceptor"}:
            counts[f"{role}.phase{int(row['phase'])}"] += 1
        counts[f"{role}.chrom.{row['chrom']}.strand.{row['strand']}"] += 1
    exclusion_counts = Counter(
        reason for row in exclusions for reason in row.get("reasons", [])
    )
    pins = {
        "fasta": file_record(fasta_path),
        "fai": file_record(fai_path),
        "basic_gtf": {
            **file_record(basic_gtf_path),
            "release": "GENCODE_v50",
            "coordinate_convention": "GTF_1_based_closed_parsed_as_0_based_half_open",
            "phase_convention": "GTF_frame_to_conventional_intron_phase",
        },
        "comprehensive_gtf": {
            **file_record(comprehensive_gtf_path),
            "release": "GENCODE_v50",
            "role": "local_coding_strand_and_phase_safety_audit",
        },
        "maxent": scorer.pin(),
        "freshness_registry_sha256": sha256_payload(freshness_registry),
        "freshness_registry_content_sha256": freshness_registry["content_sha256"],
        "freshness_audit_sha256": sha256_payload(freshness_audit),
    }
    package: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "stage": "construction",
        "candidate_mode": "production",
        "algorithm_version": CANDIDATE_ALGORITHM,
        "model_outcomes_read": False,
        "allowed_chromosomes": sorted(allowed),
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMS),
        "input_pins": pins,
        "freshness": {
            "normalized_count": len(blacklist),
            "gene_ids_sha256": freshness_registry["gene_ids_sha256"],
            "audit_status": freshness_audit["status"],
        },
        "counts": dict(sorted(counts.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "exclusions": exclusions,
        "candidates": candidates,
    }
    # This validation is contract-only.  The independent audit later re-reads
    # FASTA/GTF/tables and does not accept this validation as evidence.
    validate_candidate_package(package)
    package["content_sha256"] = sha256_payload(package)
    return package


def build_artifact(
    *,
    fasta_path: str | Path,
    fai_path: str | Path,
    basic_gtf_path: str | Path,
    comprehensive_gtf_path: str | Path,
    donor_table_path: str | Path,
    acceptor_table_path: str | Path,
    freshness_registry_path: str | Path,
    freshness_audit_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    registry_file = Path(freshness_registry_path).expanduser().resolve(strict=True)
    audit_file = Path(freshness_audit_path).expanduser().resolve(strict=True)
    output = guard_output_path(Path(output_path).expanduser().resolve())
    if output.suffix != ".json":
        raise ValueError("candidate census output must be JSON")
    manifest_path = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    package = construct_candidate_package(
        fasta_path=fasta_path,
        fai_path=fai_path,
        basic_gtf_path=basic_gtf_path,
        comprehensive_gtf_path=comprehensive_gtf_path,
        donor_table_path=donor_table_path,
        acceptor_table_path=acceptor_table_path,
        freshness_registry=_read_json(registry_file),
        freshness_audit=_read_json(audit_file),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert-positional-boundary-coherence:candidate-census-v1",
        sys.argv,
        repo=REPO,
        status="CENSUS_BUILT_AUDIT_PENDING",
        config={
            "stage": "construction",
            "algorithm_version": CANDIDATE_ALGORITHM,
            "allowed_chromosomes": sorted(DEFAULT_ALLOWED_CHROMS),
            "construct_length": 1022,
            "donor_pos": 392,
            "acceptor_pos": 512,
        },
        notes=[
            "Real hg38/GENCODE-v50 model-free census; no SpliceBERT checkpoint was loaded.",
            "Candidate rows do not authorize cohort selection until the independent census audit passes.",
        ],
    )
    for path in (
        fasta_path,
        fai_path,
        basic_gtf_path,
        comprehensive_gtf_path,
        donor_table_path,
        acceptor_table_path,
        registry_file,
        audit_file,
        Path(__file__).resolve(),
        REPO / "src/gfmcd/splicing/pbc_biology.py",
        REPO / "src/gfmcd/splicing/positional_boundary_coherence.py",
        REPO / "studies/splicebert_positional_boundary_coherence/PREREGISTRATION.md",
    ):
        manifest.add_input(Path(path).expanduser().resolve(strict=True))
    manifest.add_output(output)
    manifest.write(manifest_path)
    return output, manifest_path, package


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--fai", required=True)
    parser.add_argument("--basic-gtf", required=True)
    parser.add_argument("--comprehensive-gtf", required=True)
    parser.add_argument("--maxent-donor-table", required=True)
    parser.add_argument("--maxent-acceptor-table", required=True)
    parser.add_argument("--freshness-registry", required=True)
    parser.add_argument("--freshness-audit", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output, manifest, package = build_artifact(
        fasta_path=args.fasta,
        fai_path=args.fai,
        basic_gtf_path=args.basic_gtf,
        comprehensive_gtf_path=args.comprehensive_gtf,
        donor_table_path=args.maxent_donor_table,
        acceptor_table_path=args.maxent_acceptor_table,
        freshness_registry_path=args.freshness_registry,
        freshness_audit_path=args.freshness_audit,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "status": "CENSUS_BUILT_AUDIT_PENDING",
                "counts": package["counts"],
                "output": str(output),
                "manifest": str(manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
