#!/usr/bin/env python
"""Build train/val/test exon-pair candidate datasets from GENCODE + GRCh38."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from gfmcd.provenance import ExperimentManifest
from gfmcd.splicing.candidates import CandidateConfig
from gfmcd.splicing.config import load_config, resolve_repo_path
from gfmcd.splicing.dataset import (
    DatasetBuildConfig,
    PyfaidxGenome,
    build_exon_pair_examples,
    split_examples,
)
from gfmcd.splicing.io import save_jsonl, write_dataset_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/splicebert_partner_selection/base.json")
    p.add_argument("--max-examples", type=int, default=None, help="smoke-test cap")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    cfg_all = load_config(resolve_repo_path(repo, args.config))
    d = cfg_all["data"]
    c = d["candidate"]
    candidate = CandidateConfig(
        donor_motifs=tuple(c["donor_motifs"]),
        acceptor_motifs=tuple(c["acceptor_motifs"]),
        min_exon_length=int(c["min_exon_length"]),
        max_exon_length=int(c["max_exon_length"]),
        edge_buffer=int(c["edge_buffer"]),
        max_pairs=c.get("max_pairs"),
    )
    build_cfg = DatasetBuildConfig(
        left_flank=int(d["left_flank"]),
        right_flank=int(d["right_flank"]),
        max_sequence_length=int(d["max_sequence_length"]),
        min_exon_length=int(d["min_exon_length"]),
        max_exon_length=int(d["max_exon_length"]),
        canonical_only=bool(d["canonical_only"]),
        minimum_candidate_pairs=int(d["minimum_candidate_pairs"]),
        candidate=candidate,
        split_method=d["split_method"],
        val_chromosomes=tuple(d["val_chromosomes"]),
        test_chromosomes=tuple(d["test_chromosomes"]),
        exclude_chromosomes=tuple(d["exclude_chromosomes"]),
    )
    gtf = resolve_repo_path(repo, d["gtf"])
    fasta = resolve_repo_path(repo, d["fasta"])
    for required in (gtf, fasta):
        if not required.exists():
            raise FileNotFoundError(
                f"missing {required}; run scripts/splicebert/fetch_resources.sh or update the config"
            )
    out_dir = resolve_repo_path(repo, args.out_dir or d["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    examples, skips = build_exon_pair_examples(
        gtf,
        PyfaidxGenome(fasta),
        build_cfg,
        max_examples=args.max_examples,
    )
    if not examples:
        raise RuntimeError("dataset builder produced no examples; inspect skip_counts.json")
    split = split_examples(examples)
    split_paths = {
        name: save_jsonl(items, out_dir / f"{name}.jsonl")
        for name, items in split.items()
    }
    (out_dir / "skip_counts.json").write_text(json.dumps(skips, indent=2, sort_keys=True) + "\n")
    manifest_path = write_dataset_manifest(
        split_paths,
        out_dir / "dataset_manifest.json",
        config=asdict(build_cfg),
        source_files={"gtf": gtf, "fasta": fasta},
        notes=[
            "Sequences are transcript-oriented and use 0-based nucleotide coordinates.",
            "The test split must remain unopened until preregistration is frozen.",
        ],
    )
    run_manifest = ExperimentManifest.create(
        f"{cfg_all['study_id']}:build_dataset",
        sys.argv,
        repo=repo,
        config=cfg_all,
        notes=[f"max_examples={args.max_examples}"],
    )
    run_manifest.add_input(gtf)
    run_manifest.add_input(fasta)
    for path in split_paths.values():
        run_manifest.add_output(path)
    run_manifest.add_output(manifest_path)
    run_manifest.write(out_dir / "build_run_manifest.json")

    print(json.dumps({"n_examples": len(examples), "splits": {k: len(v) for k, v in split.items()}, "skips": skips}, indent=2))


if __name__ == "__main__":
    main()
