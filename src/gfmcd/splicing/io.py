"""Serialization and manifests for splice-pair datasets."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from gfmcd.provenance import sha256_file
from gfmcd.splicing.schema import SplicePairExample, validate_examples


def save_jsonl(examples: Iterable[SplicePairExample], path: str | Path) -> Path:
    items = list(examples)
    validate_examples(items)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for example in items:
            f.write(example.to_json() + "\n")
    return p


def load_jsonl(path: str | Path, *, validate: bool = True) -> list[SplicePairExample]:
    p = Path(path)
    items: list[SplicePairExample] = []
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                items.append(SplicePairExample.from_json(line))
            except Exception as exc:
                raise ValueError(f"{p}:{line_no}: {exc}") from exc
    if validate:
        validate_examples(items)
    return items


def dataset_summary(examples: Iterable[SplicePairExample]) -> dict:
    items = list(examples)
    pair_counts = [len(x.pairs) for x in items]
    exon_lengths = [x.true_pair.exon_length for x in items]
    return {
        "n_examples": len(items),
        "splits": dict(Counter(x.split for x in items)),
        "chromosomes": dict(Counter(x.chrom for x in items)),
        "strands": dict(Counter(x.strand for x in items)),
        "n_genes": len({x.gene_id for x in items}),
        "n_transcripts": len({x.transcript_id for x in items}),
        "candidate_pairs": {
            "min": min(pair_counts) if pair_counts else None,
            "max": max(pair_counts) if pair_counts else None,
            "mean": sum(pair_counts) / len(pair_counts) if pair_counts else None,
        },
        "true_exon_length": {
            "min": min(exon_lengths) if exon_lengths else None,
            "max": max(exon_lengths) if exon_lengths else None,
            "mean": sum(exon_lengths) / len(exon_lengths) if exon_lengths else None,
        },
    }


def write_dataset_manifest(
    split_paths: dict[str, str | Path],
    path: str | Path,
    *,
    config: dict,
    source_files: dict[str, str | Path],
    notes: list[str] | None = None,
) -> Path:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "sources": {
            name: {
                "path": str(src),
                "sha256": sha256_file(src) if Path(src).exists() else None,
            }
            for name, src in source_files.items()
        },
        "splits": {},
        "notes": notes or [],
    }
    for split, split_path in split_paths.items():
        examples = load_jsonl(split_path)
        manifest["splits"][split] = {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
            "summary": dataset_summary(examples),
        }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out
