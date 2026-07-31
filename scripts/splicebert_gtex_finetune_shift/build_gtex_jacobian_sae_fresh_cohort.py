#!/usr/bin/env python
"""Build an outcome-blind gene-fresh cohort for J-SAE confirmation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.biological_cohort import (  # noqa: E402
    ACCEPTOR_WINDOW,
    DONOR_WINDOW,
    FORBIDDEN_CHROMOSOMES,
    collect_prior_gene_exclusion,
)
from gfmcd.splicing.circuit import _dataset, eligible  # noqa: E402
from gfmcd.splicing.context_localization import (  # noqa: E402
    RelativeWindow,
    shuffle_relative_window,
)


NAMESPACE = "gtex-jacobian-sae-fresh-delta-confirmation-v1"
DONOR_SHAM = RelativeWindow(16, 48)
ACCEPTOR_SHAM = RelativeWindow(-80, -48)
DEFAULT_GENES = 128


def gene_root(gene_id: str) -> str:
    return str(gene_id).split(".", 1)[0]


def stable_order_key(gene_id: str, example_id: str) -> str:
    return hashlib.sha256(
        f"{NAMESPACE}\0{gene_root(gene_id)}\0{example_id}".encode("utf-8")
    ).hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def changed_positions(clean: str, changed: str) -> list[int]:
    if len(clean) != len(changed):
        raise ValueError("counterfactual changed sequence length")
    return [
        position
        for position, (left, right) in enumerate(zip(clean, changed))
        if left != right
    ]


def state_row(
    example: Any,
    *,
    endpoint: str,
    target_position: int,
    variant_kind: str,
    sequence: str,
    clean_state_id: str | None,
) -> dict[str, Any]:
    state_id = f"{example.example_id}|{endpoint}|{variant_kind}"
    clean = str(example.sequence)
    differences = (
        [] if clean_state_id is None else changed_positions(clean, sequence)
    )
    return {
        "state_id": state_id,
        "example_id": str(example.example_id),
        "gene_id": str(example.gene_id),
        "endpoint": endpoint,
        "variant_id": variant_kind,
        "variant_kind": variant_kind,
        "sequence": sequence,
        "sequence_sha256": sequence_sha256(sequence),
        "target_position": int(target_position),
        "reference_state_id": clean_state_id,
        "changed_positions": differences,
        "hamming": len(differences),
    }


def build_states(examples: list[Any]) -> list[dict[str, Any]]:
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
                    example,
                    endpoint=endpoint,
                    target_position=specification["target"],
                    variant_kind="clean",
                    sequence=clean,
                    clean_state_id=None,
                )
            )
            target, _ = shuffle_relative_window(
                clean,
                anchor=specification["anchor"],
                window=specification["window"],
                key=(NAMESPACE, example.example_id, endpoint, "target"),
                draw=0,
            )
            sham, _ = shuffle_relative_window(
                clean,
                anchor=specification["anchor"],
                window=specification["sham"],
                key=(NAMESPACE, example.example_id, endpoint, "sham"),
                draw=0,
            )
            for variant_kind, sequence in (
                ("target_shuffle", target),
                ("sham_shuffle", sham),
            ):
                row = state_row(
                    example,
                    endpoint=endpoint,
                    target_position=specification["target"],
                    variant_kind=variant_kind,
                    sequence=sequence,
                    clean_state_id=clean_id,
                )
                if row["hamming"] < 1:
                    raise RuntimeError("shuffle produced an unchanged sequence")
                states.append(row)
    return states


def select_examples(
    basic_gtf: Path,
    fasta: Path,
    excluded_gene_ids: set[str],
    *,
    count: int,
) -> list[Any]:
    excluded_roots = {gene_root(gene) for gene in excluded_gene_ids}
    per_gene: dict[str, Any] = {}
    for example in eligible(_dataset(str(basic_gtf), str(fasta)), "train"):
        root = gene_root(example.gene_id)
        if (
            root in excluded_roots
            or str(example.chrom) in FORBIDDEN_CHROMOSOMES
        ):
            continue
        current = per_gene.get(root)
        if current is None or str(example.example_id) < str(current.example_id):
            per_gene[root] = example
    ordered = sorted(
        per_gene.values(),
        key=lambda example: (
            stable_order_key(example.gene_id, example.example_id),
            str(example.example_id),
        ),
    )
    if len(ordered) < count:
        raise RuntimeError(
            f"fresh cohort needs {count} genes but only {len(ordered)} are eligible"
        )
    return ordered[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-results", required=True)
    parser.add_argument("--basic-gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--genes", type=int, default=DEFAULT_GENES)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.genes != DEFAULT_GENES:
        raise ValueError("the frozen confirmation requires exactly 128 genes")

    prior_results = Path(args.prior_results).expanduser().resolve()
    basic_gtf = Path(args.basic_gtf).expanduser().resolve()
    fasta = Path(args.fasta).expanduser().resolve()
    output = guard_output_path(Path(args.out).expanduser().resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    if output == manifest_path or output.suffix != ".json":
        raise ValueError("fresh cohort requires a distinct .json output")
    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_FRESH_DELTA_CONFIRMATION_V1.md"
    )

    exclusion = collect_prior_gene_exclusion(prior_results)
    examples = select_examples(
        basic_gtf,
        fasta,
        set(exclusion.gene_ids),
        count=args.genes,
    )
    states = build_states(examples)
    counts = Counter(str(row["variant_kind"]) for row in states)
    if (
        len(states) != 6 * args.genes
        or len({gene_root(example.gene_id) for example in examples}) != args.genes
        or counts
        != Counter(
            {
                "clean": 2 * args.genes,
                "target_shuffle": 2 * args.genes,
                "sham_shuffle": 2 * args.genes,
            }
        )
    ):
        raise RuntimeError("fresh-cohort cardinality invariant failed")
    result = {
        "study": "SpliceBERT GTEx layer-5 Jacobian SAE",
        "stage": "outcome-blind fresh-cohort construction",
        "status": "FROZEN",
        "evidence_tier": "instrument",
        "namespace": NAMESPACE,
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_free": True,
        "classifier_output_observed": False,
        "prior_gene_exclusion": {
            "root": str(prior_results),
            "n_genes": len(exclusion.gene_ids),
            "gene_ids": list(exclusion.gene_ids),
            "source_files": list(exclusion.source_files),
            "aggregate_sha256": exclusion.aggregate_sha256,
        },
        "selection": {
            "partition": "train",
            "one_example_per_gene": True,
            "forbidden_chromosomes": sorted(FORBIDDEN_CHROMOSOMES),
            "ordering": "sha256(namespace, gene root, example ID)",
            "n_genes": args.genes,
        },
        "examples": [
            {
                "example_id": str(example.example_id),
                "gene_id": str(example.gene_id),
                "chrom": str(example.chrom),
                "strand": str(example.strand),
                "donor_pos": int(example.donor_pos),
                "next_acceptor_pos": int(example.next_acceptor_pos),
                "sequence_sha256": sequence_sha256(str(example.sequence)),
            }
            for example in examples
        ],
        "counts": {
            "states": len(states),
            "genes": args.genes,
            "by_variant_kind": dict(sorted(counts.items())),
        },
        "states": states,
        "capture_authorized": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:jacobian_sae_fresh_cohort_v1",
        sys.argv,
        repo=REPO,
        status="FROZEN",
        config={
            "namespace": NAMESPACE,
            "n_genes": args.genes,
            "partition": "train",
            "prior_exclusion_sha256": exclusion.aggregate_sha256,
        },
        notes=[
            "Selection used gene identity, coordinates, and sequence only.",
            "No model, classifier, SAE activation, or margin was loaded.",
        ],
    )
    for path in (
        addendum,
        basic_gtf,
        fasta,
        *(
            Path(record["path"])
            for record in exclusion.source_files
        ),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
