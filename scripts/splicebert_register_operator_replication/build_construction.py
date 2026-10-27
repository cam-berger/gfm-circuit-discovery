#!/usr/bin/env python3
"""Prepare the immutable coordinate package before any checkpoint is loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.register_operator_construction import build_construction
from gfmcd.splicing.register_operator_io import guard_paths, write_artifact
from gfmcd.splicing.register_operator_statistics import balanced_assignments, PSEUDO_SEEDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    guard_paths(args.out)
    package = build_construction(args.cohort, args.selection, args.fasta)
    ids = package["selected_block_ids"]
    hashes = {}
    for seed in PSEUDO_SEEDS:
        assignments = balanced_assignments(ids, seed)
        rows = [{"block_id": bid, "acceptor_label_permutation": permutation.tolist()}
                for bid, permutation in zip(ids, assignments)]
        hashes[str(seed)] = hashlib.sha256(json.dumps(rows, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    package["pseudo_phase_family"] = {"seeds": list(PSEUDO_SEEDS), "assignment_sha256_by_seed": hashes}
    sources = [REPO / "src/gfmcd/splicing" / ("register_operator_" + name + ".py")
               for name in ("construction", "statistics", "io")]
    receipt = write_artifact(args.out, package,
        inputs=[args.cohort, args.selection, args.fasta, Path(str(args.fasta) + ".fai"), Path(__file__), *sources],
        experiment="register-operator:execution-construction", config=package["config"],
        seeds={"bootstrap": 20260803})
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

