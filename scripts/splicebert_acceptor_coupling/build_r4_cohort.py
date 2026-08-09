#!/usr/bin/env python
"""Serialise R4's cohort so the Colab run needs no GTF and no FASTA.

Writes the val cohort AFTER eligibility filtering and AFTER the tag-seeded shuffle, so a
consumer taking ``[:n]`` reproduces ``load_examples(..., tag=DRAW_TAG, min_effect=None)``
exactly.  Refuses to emit a sealed chromosome.

    uv run python scripts/splicebert_acceptor_coupling/build_r4_cohort.py --n 1200 \
        --out results/splicebert_acceptor_coupling/r4_cohort_v1.sequences.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, sha256_file                  # noqa: E402
from gfmcd.splicing.circuit import (                                          # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, CouplingConfig, _dataset, eligible,
)
from gfmcd.splicing.cohort_io import write_cohort                             # noqa: E402
from gfmcd.splicing.coupling import stable_rng                                # noqa: E402

DRAW_TAG = "r1b-d0"
SEALED = ("chr8", "chr9")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="prefix length to serialise")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tag", default=DRAW_TAG)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = CouplingConfig()
    X = eligible(_dataset(a.gtf, a.fasta), a.split)
    stable_rng(a.tag, cfg.seed).shuffle(X)          # identical to load_examples
    keep = X[:a.n]
    print(f"eligible {a.split}: {len(X)}  ->  serialising {len(keep)}")

    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    if out.exists():
        raise SystemExit(f"refuse to overwrite an existing cohort: {out}")
    summary = write_cohort(out, keep, refuse_chroms=SEALED)
    print(json.dumps(summary, indent=1))

    ExperimentManifest.create(
        "r4_cohort_v1", sys.argv, repo=REPO, status="EXPLORATORY",
        config={"split": a.split, "tag": a.tag, "n": len(keep),
                "gtf": a.gtf, "fasta": a.fasta, "sealed_refused": list(SEALED)},
        seeds={"seed": cfg.seed},
        inputs=[{"path": a.gtf, "role": "annotation"}, {"path": a.fasta, "role": "genome"}],
        outputs=[{"path": str(out), "role": "cohort", "sha256": summary["sha256"]}],
        notes=["Written after eligibility and the tag-seeded shuffle; a consumer taking [:n] "
               "reproduces load_examples(tag=..., min_effect=None). min_effect is deliberately "
               "not applied - it conditions on the outcome."],
    ).write(out.with_suffix(".manifest.json"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
