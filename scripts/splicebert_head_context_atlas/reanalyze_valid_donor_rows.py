#!/usr/bin/env python
"""Reanalyse only coordinate-valid donor rows from historical ``head_screen_v1``.

The raw matrix does not store ``split``/``chrom`` as separate fields, but every
``example_id`` begins with the chromosome and the split contract is chromosome-based.
This derived artifact excludes chr11/16/19 (the touched validation chromosomes), keeps
only ``alt5ss_donor`` targets, and never reads chr8/9 (absent from the source screen).

The output intentionally mirrors the keys consumed by ``run_group_mlp_closure.py`` so
the corrected donor-only ranking is pinned as an immutable input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

VAL_CHROMS = {"chr11", "chr16", "chr19"}
SEALED_CHROMS = {"chr8", "chr9"}
HEADS = tuple((layer, head) for layer in range(6) for head in range(16))


def label(head: tuple[int, int]) -> str:
    return f"L{head[0]}H{head[1]}"


def chrom_of(example_id: str) -> str:
    return example_id.split(":", 1)[0]


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_by_gene(by_gene: dict[str, list[float]]) -> dict:
    genes = sorted(by_gene)
    means = np.asarray([np.mean(by_gene[g]) for g in genes], dtype=float)
    point = float(means.mean())
    se = float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else None
    return {
        "n_genes": len(genes),
        "mean": point,
        "se_gene": se,
        "ci95": [point - 1.96 * se, point + 1.96 * se] if se is not None else [None, None],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    matrix = Path(args.matrix)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refuse to overwrite immutable result {out}")

    # (condition, head, estimand) -> gene -> values
    accum: dict[tuple[str, tuple[int, int], str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    c_abs: dict[tuple[str, tuple[int, int]], float] = {}
    examples: set[str] = set()
    genes: set[str] = set()
    n_rows = 0
    with matrix.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["site_type"] != "alt5ss_donor":
                continue
            chrom = chrom_of(row["example_id"])
            if chrom in VAL_CHROMS:
                continue
            if chrom in SEALED_CHROMS:
                raise RuntimeError("sealed chromosome unexpectedly present in source matrix")
            condition = row["condition"]
            if condition not in {"B", "C", "D"}:
                continue
            head = (int(row["layer"]), int(row["head"]))
            effect = float(row["effect_margin"])
            gene = row["gene_id"]
            eid = row["example_id"]
            accum[(condition, head, "signed")][gene].append(effect)
            accum[(condition, head, "magnitude")][gene].append(abs(effect))
            if condition == "C":
                c_abs[(eid, head)] = abs(effect)
            elif condition == "D":
                key = (eid, head)
                if key not in c_abs:
                    raise RuntimeError(f"D row appeared without its paired C row: {key}")
                accum[("C_minus_D", head, "magnitude")][gene].append(c_abs.pop(key) - abs(effect))
            examples.add(eid)
            genes.add(gene)
            n_rows += 1
    if c_abs:
        raise RuntimeError(f"{len(c_abs)} C rows lacked paired D rows")

    per_head = {}
    directional_b = {}
    cd = {}
    for head in HEADS:
        h = label(head)
        for condition in ("B", "C", "D"):
            signed = summarize_by_gene(
                accum[(condition, head, "signed")]
            )
            magnitude = summarize_by_gene(
                accum[(condition, head, "magnitude")]
            )
            per_head[f"{condition}|{h}"] = {
                "signed_mean": signed["mean"],
                "signed_ci95": signed["ci95"],
                "magnitude_mean": magnitude["mean"],
                "magnitude_ci95": magnitude["ci95"],
                "n_genes": signed["n_genes"],
            }
        directional_b[h] = {
            "donor_mean": per_head[f"B|{h}"]["signed_mean"],
            "donor_ci95": per_head[f"B|{h}"]["signed_ci95"],
        }
        cd[h] = summarize_by_gene(
            accum[("C_minus_D", head, "magnitude")]
        )

    def profile(condition: str, estimand: str) -> np.ndarray:
        field = "signed_mean" if estimand == "signed" else "magnitude_mean"
        return np.asarray([per_head[f"{condition}|{label(h)}"][field] for h in HEADS])

    report = {
        "schema": "head_context_atlas.valid_donor_reanalysis/1",
        "status": "EXPLORATORY_DERIVED_COORDINATE_VALID_SUBSET",
        "source_matrix": str(matrix.resolve()),
        "source_matrix_sha256": file_sha256(matrix),
        "scope": "alt5ss_donor; train chromosomes only; chr11/16/19 excluded; chr8/9 absent",
        "n_examples": len(examples),
        "n_genes": len(genes),
        "n_rows_BCD": n_rows,
        "per_head_train_only": per_head,
        "directional_B": directional_b,
        "C_minus_D_magnitude": cd,
        "correlations_valid_donor_train": {
            "signed_B_C": float(np.corrcoef(profile("B", "signed"), profile("C", "signed"))[0, 1]),
            "signed_B_D": float(np.corrcoef(profile("B", "signed"), profile("D", "signed"))[0, 1]),
            "signed_C_D": float(np.corrcoef(profile("C", "signed"), profile("D", "signed"))[0, 1]),
            "magnitude_C_D": float(np.corrcoef(profile("C", "magnitude"), profile("D", "magnitude"))[0, 1]),
        },
        "estimands": {
            "B": "gene-clustered signed mean on coordinate-valid donor targets",
            "C_D": "gene-clustered magnitude means on coordinate-valid donor targets",
            "C_minus_D": "paired gene-clustered E[|C_h|-|D_h|]",
            "CI": "95% normal interval over gene-level means (diagnostic ranking artifact)",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "out": str(out), "n_examples": len(examples), "n_genes": len(genes),
        "n_rows_BCD": n_rows, "correlations": report["correlations_valid_donor_train"],
    }, indent=2))


if __name__ == "__main__":
    main()
