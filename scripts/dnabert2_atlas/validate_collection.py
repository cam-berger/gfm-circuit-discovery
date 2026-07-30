#!/usr/bin/env python
"""Validate an atlas collection and print a compact census."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from gfmcd.reconstruction_atlas.schema import load_collection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    args = parser.parse_args()
    records = load_collection(args.collection)
    elements = [element for record in records for element in record.elements]
    summary = {
        "examples": len(records),
        "elements": len(elements),
        "site_classes": dict(sorted(Counter(x.site_class for x in elements).items())),
        "splits": dict(sorted(Counter(x.split for x in records).items())),
        "assemblies": dict(sorted(Counter(x.assembly for x in records).items())),
        "strands": dict(sorted(Counter(x.strand for x in records).items())),
        "homology_clusters": len({x.homology_cluster for x in records}),
        "sequence_length": {
            "min": min(len(x.sequence) for x in records),
            "max": max(len(x.sequence) for x in records),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
