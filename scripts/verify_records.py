#!/usr/bin/env python3
"""Verify result manifests by content, from a plain clone (standard library only).

    python3 scripts/verify_records.py                      # census of every tracked manifest
    python3 scripts/verify_records.py <manifest> [...]     # one line per record
    python3 scripts/verify_records.py --data-root ~/data   # also check external inputs by name

Records are classified ok / moved / drift / missing / external-ok / external-unchecked. By default
only git-tracked files count, so the census is the same on every machine; --include-untracked
also accepts local files at their recorded paths (raw payloads kept out of Git). The legacy
drift/missing set is pinned in tests/data/record_census.json; --write-census regenerates it.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from gfmcd import record  # noqa: E402

CENSUS = Path(__file__).resolve().parents[1] / "tests/data/record_census.json"


def census_document(reports: list[record.Report]) -> dict:
    failures = sorted(
        ({"manifest": r.manifest, "path": c.path, "sha256": c.sha256, "status": c.status}
         for r in reports for c in r.failures),
        key=lambda f: (f["manifest"], f["path"], f["sha256"], f["status"]),
    )
    ok = sum(r.counts["ok"] + r.counts["moved"] for r in reports)
    return {
        "_README": ("Legacy manifest records that do not verify from a clean checkout (tracked files "
                    "only). tests/test_record.py asserts the observed set is exactly this list: a new "
                    "failure fails, and so does a listed record that verifies again. Regenerate only "
                    "deliberately: python3 scripts/verify_records.py --write-census"),
        "floor_verified": ok,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifests", nargs="*", type=Path)
    parser.add_argument("--include-untracked", action="store_true")
    parser.add_argument("--data-root", action="append", type=Path, default=None)
    parser.add_argument("--write-census", action="store_true")
    args = parser.parse_args()

    reports = record.verify_all(args.manifests or None, tracked_only=not args.include_untracked,
                                data_roots=args.data_root)
    if args.manifests:
        for report in reports:
            for check in report.checks:
                where = f" -> {check.resolved}" if check.resolved and check.status != "ok" else ""
                print(f"{check.status:18} {check.path}{where}")
    totals = Counter()
    for report in reports:
        totals.update(report.counts)
    print(f"{len(reports)} manifests: " + ", ".join(f"{k} {totals[k]}" for k in sorted(totals)))
    if args.write_census:
        if args.manifests or args.include_untracked or args.data_root:
            parser.error("--write-census runs over every tracked manifest, tracked files only")
        document = census_document(reports)
        rows = ",\n  ".join(json.dumps(f, sort_keys=True) for f in document.pop("failures"))
        head = json.dumps(document, indent=1, sort_keys=True)[:-2]
        CENSUS.write_text(f'{head},\n "failures": [\n  {rows}\n ]\n}}\n')
        print(f"wrote {CENSUS}")
    return 1 if args.manifests and any(r.failures for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
