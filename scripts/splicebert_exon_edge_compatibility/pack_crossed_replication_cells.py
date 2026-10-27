#!/usr/bin/env python
"""Archive C43's raw replication rows as a byte-verified gzip beside the audit.

    uv run python scripts/splicebert_exon_edge_compatibility/pack_crossed_replication_cells.py \
        --source <crossed_direct_edge_replication_cells.json> \
        --out results/splicebert_exon_edge_compatibility/crossed_direct_edge_replication_cells.json.gz

    uv run python scripts/splicebert_exon_edge_compatibility/pack_crossed_replication_cells.py \
        --binds native_result_sha256 \
        --source <crossed_direct_edge_replication_native.json> \
        --out results/splicebert_exon_edge_compatibility/crossed_direct_edge_replication_native.json.gz

The source branch gitignored the 54 MB cell rows the result audit recomputed from (R45), and
R45 carried neither them nor the 1.09 MB native-source rows. This refuses any source whose
sha256 is not the audit's `--binds` field (`cells_result_sha256` by default), writes a gzip
with no embedded name or timestamp, and checks the archive decompresses to the same bytes.
The load-bearing hash is the decompressed one; the archive's own hash is recorded for
transport. See CORRECTIONS.md R49.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402


AUDIT = (
    REPO / "results/splicebert_exon_edge_compatibility/"
    "crossed_direct_edge_replication_audit.json"
)
COMPRESS_LEVEL = 9
# Audit field the source bytes must hash to -> (archive kind, why the rows were not carried).
BINDINGS = {
    "cells_result_sha256": (
        "cells",
        "Raw rows were gitignored on the source branch; the audit recomputed from them.",
    ),
    "native_result_sha256": (
        "native",
        "Native-source rows were tracked at fd26cc1b but not carried by R45; the audit "
        "recomputed gate 2 and the native floors from them.",
    ),
    "cohort_sha256": (
        "cohort",
        "The replication cohort existed only in a session-notes folder, where both CC "
        "protocol freezes pinned it; the audit and the CC runs read it (R56).",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binds", choices=sorted(BINDINGS), default="cells_result_sha256",
                        help="audit field the source sha256 must equal")
    parser.add_argument("--retrieval", default="",
                        help="how the source bytes were obtained; recorded in the manifest")
    args = parser.parse_args(argv)
    kind, provenance_note = BINDINGS[args.binds]
    out = guard_output_path(args.out.resolve())
    if not out.name.endswith(".json.gz"):
        raise ValueError("archive name must end in .json.gz")
    manifest_path = guard_output_path(
        out.with_name(out.name.removesuffix(".json.gz") + ".archive.manifest.json")
    )

    expected = json.loads(args.audit.read_text())[args.binds]
    raw = args.source.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != expected:
        raise ValueError(f"source sha256 {raw_sha256} is not the audited {expected}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=COMPRESS_LEVEL, fileobj=handle, mtime=0
        ) as stream:
            stream.write(raw)
    if hashlib.sha256(gzip.decompress(out.read_bytes())).hexdigest() != expected:
        raise RuntimeError("archive does not decompress to the audited bytes")

    config = {
        "format": "gzip",
        "compresslevel": COMPRESS_LEVEL,
        "embedded_name": "",
        "embedded_mtime": 0,
        "decompressed_sha256": raw_sha256,
        "decompressed_size_bytes": len(raw),
        "archive_sha256": sha256_file(out),
        "archive_size_bytes": out.stat().st_size,
        "binds": f"crossed_direct_edge_replication_audit.json {args.binds}",
        "extraction": f"gzip -dk {out.name}",
    }
    if args.retrieval:
        config["source_retrieval"] = args.retrieval
    manifest = ExperimentManifest.create(
        f"splicebert-exon-edge-compatibility:fresh-crossed-replication-{kind}-archive",
        sys.argv if argv is None else [__file__, *argv],
        repo=REPO,
        status="ARCHIVED_RAW_ROWS",
        config=config,
        notes=[
            provenance_note,
            "The decompressed sha256, not the archive sha256, is what the audit binds.",
        ],
    )
    manifest.add_input(args.source.resolve())
    manifest.add_input(args.audit.resolve())
    manifest.add_input(Path(__file__).resolve())
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({"out": str(out), "manifest": str(manifest_path),
                      **manifest.config}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
