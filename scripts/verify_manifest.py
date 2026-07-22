"""Prove that `results/MANIFEST.{csv,json}` and the results tree still describe each other.

The manifest is the machine-readable index required by review fix F2
(`docs/REVIEW_2026-07-22.md` §4): every file under `results/` must resolve to a claim, a
status, a controlling script, a sibling provenance manifest, and a consuming figure. That
index is only worth anything if it cannot silently drift from the tree it indexes.

Six checks, each of which can fail on a real repository state:

  1. coverage    every tracked file under `results/` has exactly one manifest row, and the
                 manifest invents no rows for files that are not there
  2. artifacts   every `artifact` path exists on disk
  3. references  every non-empty `script` / `manifest` / `figure` path exists on disk
                 (these columns may hold a `;`-separated list)
  4. status      `status` is one of canonical / development / superseded / infrastructure
  5. polarity    `polarity` is present on every row and is one of
                 positive / null / refutation / bounded / na
  6. parity      the CSV and the JSON carry the same eight columns for the same rows, and the
                 JSON's `row_count` / `status_counts` / `polarity_counts` summaries match its
                 own rows

Nothing here reads a scientific value; this is an index integrity check only.

`polarity` exists because an evidence level alone cannot tell a powered null or a refutation
apart from positive evidence at the same rung — C13 is a powered attention null, C14 is a
powered refutation of the nucleosome bridge, and C15 is a sensitivity bound, yet all three sat
in the index looking exactly like supporting evidence. The values are assigned from the study
ledgers' own status wording (`studies/*/CLAIMS.md`), which this script does not read: it only
enforces that the column exists and stays inside the vocabulary.

    uv run python scripts/verify_manifest.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import subprocess
from pathlib import Path

COLUMNS = [
    "claim_id",
    "status",
    "evidence_level",
    "polarity",
    "artifact",
    "script",
    "manifest",
    "figure",
]
LEGAL_STATUS = ("canonical", "development", "superseded", "infrastructure")
# What the artifact says about the hypothesis at its evidence level. `positive` is not the
# default: a powered null and a refutation are first-class results here, and flattening them
# into `positive` misreports the record.
LEGAL_POLARITY = ("positive", "null", "refutation", "bounded", "na")
PATH_COLUMNS = ("script", "manifest", "figure")

# The index files the manifest describes are written by this same review program, so they may
# not be tracked yet on the branch that first introduces them. Count them as expected once
# they exist, otherwise the manifest could never legally list itself.
SELF_DESCRIBING = ("results/MANIFEST.csv", "results/MANIFEST.json")

MAX_REPORTED = 20


def tracked_result_files(repo: Path) -> set[str]:
    """Paths git tracks under `results/`, plus the not-yet-committed index files."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", "results/"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git checkout (e.g. a scratch tree under test): fall back to what is present.
        return {
            str(p.relative_to(repo)) for p in (repo / "results").rglob("*") if p.is_file()
        }
    paths = {line for line in out.split("\n") if line}
    paths |= {p for p in SELF_DESCRIBING if (repo / p).exists()}
    paths |= {
        str(p.relative_to(repo))
        for p in (repo / "results").rglob("RESULTS_INDEX.md")
        if p.is_file()
    }
    return paths


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(
                f"FAIL {path}: header is {reader.fieldnames}, expected exactly {COLUMNS}"
            )
        return list(reader)


def read_json_doc(path: Path) -> dict:
    doc = json.loads(path.read_text())
    for key in ("generated_by", "row_count", "status_counts", "polarity_counts", "rows"):
        if key not in doc:
            raise SystemExit(f"FAIL {path}: missing top-level key {key!r}")
    return doc


def check_manifest(
    repo: Path,
    csv_rows: list[dict[str, str]],
    json_doc: dict,
    expected: set[str],
) -> list[str]:
    """Return a list of human-readable problems; empty means the manifest is consistent."""
    problems: list[str] = []

    # 1. coverage
    counts = collections.Counter(r["artifact"] for r in csv_rows)
    for path in sorted(expected - set(counts)):
        problems.append(f"coverage: tracked result file missing from the manifest: {path}")
    for path in sorted(set(counts) - expected):
        problems.append(f"coverage: manifest row for an untracked/absent file: {path}")
    for path, n in sorted(counts.items()):
        if n > 1:
            problems.append(f"coverage: {path} appears {n} times; every file gets exactly one row")

    for row in csv_rows:
        artifact = row["artifact"]

        # 2. artifacts
        if not artifact:
            problems.append("artifact: empty 'artifact' cell")
            continue
        if not (repo / artifact).exists():
            problems.append(f"artifact: does not exist on disk: {artifact}")

        # 3. references
        for col in PATH_COLUMNS:
            for ref in filter(None, row[col].split(";")):
                if not (repo / ref).exists():
                    problems.append(f"{col}: {artifact} points at a missing path: {ref}")

        # 4. status
        if row["status"] not in LEGAL_STATUS:
            problems.append(
                f"status: {artifact} has status {row['status']!r}; legal values are {LEGAL_STATUS}"
            )

        # 5. polarity — present on every row, and inside the vocabulary
        if "polarity" not in row or row["polarity"] is None:
            problems.append(f"polarity: {artifact} has no 'polarity' cell")
        elif row["polarity"] not in LEGAL_POLARITY:
            problems.append(
                f"polarity: {artifact} has polarity {row['polarity']!r}; "
                f"legal values are {LEGAL_POLARITY}"
            )

    # 6. parity between the two serialisations, and the JSON's own summaries
    json_rows = json_doc["rows"]
    csv_key = [tuple(r[c] for c in COLUMNS) for r in csv_rows]
    try:
        json_key = [tuple(r[c] for c in COLUMNS) for r in json_rows]
    except (KeyError, TypeError):
        problems.append("parity: MANIFEST.json rows do not all carry the eight manifest columns")
        json_key = None
    if json_key is not None and json_key != csv_key:
        artifact_at = COLUMNS.index("artifact")
        only_csv = sorted({k[artifact_at] for k in csv_key} - {k[artifact_at] for k in json_key})
        only_json = sorted({k[artifact_at] for k in json_key} - {k[artifact_at] for k in csv_key})
        detail = f" (csv-only: {only_csv[:3]}, json-only: {only_json[:3]})" if (only_csv or only_json) else ""
        problems.append(f"parity: MANIFEST.csv and MANIFEST.json disagree{detail}")
    if json_doc["row_count"] != len(json_rows):
        problems.append(
            f"parity: row_count {json_doc['row_count']} != {len(json_rows)} rows in MANIFEST.json"
        )
    for column, summary in (("status", "status_counts"), ("polarity", "polarity_counts")):
        recount = dict(sorted(collections.Counter(r.get(column) for r in json_rows).items()))
        if dict(sorted(json_doc[summary].items())) != recount:
            problems.append(f"parity: {summary} {json_doc[summary]} != recomputed {recount}")

    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    repo = args.repo.resolve()

    csv_path = repo / "results/MANIFEST.csv"
    json_path = repo / "results/MANIFEST.json"
    for path in (csv_path, json_path):
        if not path.exists():
            raise SystemExit(f"FAIL: {path} is missing; the results tree has no master index")

    csv_rows = read_csv_rows(csv_path)
    json_doc = read_json_doc(json_path)
    expected = tracked_result_files(repo)
    problems = check_manifest(repo, csv_rows, json_doc, expected)

    if problems:
        for line in problems[:MAX_REPORTED]:
            print(f"  {line}")
        if len(problems) > MAX_REPORTED:
            print(f"  ... and {len(problems) - MAX_REPORTED} more")
        raise SystemExit(
            f"FAIL: results/MANIFEST is inconsistent with the results tree "
            f"({len(problems)} problem(s), {len(csv_rows)} rows, {len(expected)} tracked files)"
        )

    status_tally = collections.Counter(r["status"] for r in csv_rows)
    polarity_tally = collections.Counter(r["polarity"] for r in csv_rows)
    print(f"results/MANIFEST.csv  {len(csv_rows)} rows")
    print("  status    " + "  ".join(f"{k} {status_tally[k]}" for k in LEGAL_STATUS))
    print("  polarity  " + "  ".join(f"{k} {polarity_tally[k]}" for k in LEGAL_POLARITY))
    print(f"OK: every tracked file under results/ resolves to exactly one manifest row,")
    print(f"    every artifact/script/manifest/figure path on those rows exists, and every")
    print(f"    row declares a legal evidence polarity.")


if __name__ == "__main__":
    main()
