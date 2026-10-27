#!/usr/bin/env python3
"""Verify result manifests by content, from a plain clone (standard library only).

    python3 scripts/verify_records.py                      # census of every tracked manifest
    python3 scripts/verify_records.py <manifest> [...]     # one line per record
    python3 scripts/verify_records.py --data-root ~/data   # also check external inputs by name
    python3 scripts/verify_records.py --pins               # pins outside manifests (gfmcd.pins)

Records are classified ok / moved / drift / missing / external-ok / external-unchecked. By default
only git-tracked files count, so the census is the same on every machine; --include-untracked
also accepts local files at their recorded paths (raw payloads kept out of Git). The legacy
drift/missing set is pinned in tests/data/record_census.json; --write-census regenerates it.
Pins held by freezes, packages, sidecars and attestations are pinned in tests/data/pin_census.json
(--pins --write-census).
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

from gfmcd import pins, record  # noqa: E402

CENSUS = Path(__file__).resolve().parents[1] / "tests/data/record_census.json"
PIN_CENSUS = Path(__file__).resolve().parents[1] / "tests/data/pin_census.json"


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


def _write_ledger(path: Path, readme: str, floor: int, failures: list[dict]) -> None:
    rows = ",\n  ".join(json.dumps(f, sort_keys=True) for f in failures)
    head = json.dumps({"_README": readme, "floor_verified": floor}, indent=1, sort_keys=True)[:-2]
    path.write_text(f'{head},\n "failures": [\n  {rows}\n ]\n}}\n')
    print(f"wrote {path}")


def pin_census(write: bool) -> int:
    checks = pins.check()
    totals = Counter(c.status for c in checks)
    print(f"{len(checks)} pins in {len({c.source for c in checks})} files: "
          + ", ".join(f"{k} {totals[k]}" for k in sorted(totals)))
    if write:
        failures = [{"source": c.source, "path": c.path, "sha256": c.sha256, "status": c.status}
                    for c in checks if c.status in record.FAILURES]
        _write_ledger(PIN_CENSUS, "Pins outside result manifests (gfmcd.pins) that do not verify from "
                      "a clean checkout. tests/test_pins.py asserts the observed set is exactly this "
                      "list. Regenerate only deliberately: python3 scripts/verify_records.py --pins "
                      "--write-census", totals["ok"] + totals["moved"], failures)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifests", nargs="*", type=Path)
    parser.add_argument("--include-untracked", action="store_true")
    parser.add_argument("--data-root", action="append", type=Path, default=None)
    parser.add_argument("--write-census", action="store_true")
    parser.add_argument("--pins", action="store_true", help="check pins outside result manifests")
    args = parser.parse_args()
    if args.pins:
        if args.manifests or args.include_untracked or args.data_root:
            parser.error("--pins runs over every pin source, tracked files only")
        return pin_census(args.write_census)

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
        _write_ledger(CENSUS, document["_README"], document["floor_verified"], document["failures"])
    return 1 if args.manifests and any(r.failures for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
