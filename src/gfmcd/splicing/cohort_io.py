"""Serialise a coupling cohort so a run needs no GTF and no FASTA.

WHY THIS EXISTS
---------------
Every runner in this tree builds its cohort by parsing GENCODE and indexing hg38 through
pyfaidx.  That is correct locally and impossible on a remote runtime: the FASTA alone is
3.2 GB, and uploading it defeats the point of a self-contained bundle.  The R4 Colab package
claimed "no GTF or FASTA needed" before this module existed, which was false -- the claim is
what this module makes true.

WHAT IS PRESERVED
-----------------
The cohort is written **after** eligibility filtering and after the tag-seeded shuffle, in that
order, so a consumer taking ``[:n]`` reproduces ``load_examples(..., tag=..., min_effect=None)``
exactly.  Matched decoys are NOT stored: ``matched_decoy_donor`` / ``matched_decoy_acceptor``
are pure functions of an example, so recomputing them at load keeps one definition rather than
two that can drift.

``min_effect`` is deliberately unsupported.  That filter conditions on the outcome, so a cohort
file carrying it would silently bake an outcome-dependent selection into every downstream run.
Callers that need it must apply it themselves, on the model, and say so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Iterable, Sequence

from gfmcd.splicing.coupling import CouplingExample

SCHEMA = "gfmcd.coupling_cohort/1"

#: Fields stored as tuples on the dataclass; JSON round-trips them as lists.
_TUPLE_FIELDS = ("other_ag", "other_gt", "real_acc", "real_don")


def example_to_dict(example: CouplingExample) -> dict:
    payload = {}
    for field in fields(CouplingExample):
        value = getattr(example, field.name)
        payload[field.name] = list(value) if field.name in _TUPLE_FIELDS else value
    return payload


def example_from_dict(payload: dict) -> CouplingExample:
    kwargs = dict(payload)
    for name in _TUPLE_FIELDS:
        if name in kwargs and kwargs[name] is not None:
            kwargs[name] = tuple(kwargs[name])
    return CouplingExample(**kwargs)


def write_cohort(
    path: str | Path, examples: Sequence[CouplingExample], *, refuse_chroms: Iterable[str] = ()
) -> dict:
    """Write one example per line and return a summary for the manifest.

    ``refuse_chroms`` is a hard stop, not a filter: a sealed chromosome reaching a cohort file
    is a protocol failure and must fail loudly rather than be silently dropped.
    """
    refused = set(refuse_chroms)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    present = sorted({x.chrom for x in examples})
    illegal = sorted(refused & set(present))
    if illegal:
        raise ValueError(
            f"cohort contains sealed chromosomes {illegal}; refusing to write {out}"
        )
    with out.open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example_to_dict(example), sort_keys=True) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    return {
        "schema": SCHEMA, "path": str(out), "n": len(examples),
        "n_genes": len({x.gene_id for x in examples}), "chroms": present,
        "strands": sorted({x.strand for x in examples}),
        "splits": sorted({x.split for x in examples}),
        "sha256": digest, "bytes": out.stat().st_size,
    }


def read_cohort(path: str | Path, *, limit: int | None = None) -> list[CouplingExample]:
    """Read a cohort file.  Order is the file's order -- do not re-shuffle."""
    out: list[CouplingExample] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            out.append(example_from_dict(json.loads(line)))
            if limit is not None and len(out) >= limit:
                break
    if not out:
        raise ValueError(f"cohort file is empty: {path}")
    return out


def cohort_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
