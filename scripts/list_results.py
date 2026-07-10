"""List result provenance across the repo (TO_FIX.md §14).

Walks results/registry/*/manifest.json (run manifests) and results/**/*.json (stamped result
dumps), printing run_id/study/status/git for each so docs/RESULTS.md can be checked against the
machine-readable trail rather than trusting hand-copied numbers.

    uv run python scripts/list_results.py
"""

from __future__ import annotations

import json

from gfmcd.paths import project_paths


def main() -> None:
    results = project_paths().results
    if not results.exists():
        print(f"no results dir at {results}")
        return

    print("== run manifests (results/registry/*/manifest.json) ==")
    reg = results / "registry"
    manifests = sorted(reg.glob("*/manifest.json")) if reg.exists() else []
    if not manifests:
        print("  (none yet)")
    for m in manifests:
        b = json.loads(m.read_text())
        print(f"  {b.get('run_id','?'):24} {b.get('study','?'):16} {b.get('status','?'):12} {str(b.get('git_commit'))[:8]}")

    print("== stamped result dumps (results/**/*.json with _provenance) ==")
    found = False
    for j in sorted(results.rglob("*.json")):
        if j.name == "manifest.json":
            continue
        try:
            blob = json.loads(j.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        prov = blob.get("_provenance") if isinstance(blob, dict) else None
        if prov:
            found = True
            rel = j.relative_to(results)
            print(f"  {str(rel):48} {prov.get('status','?'):12} seed={prov.get('seed')} {str(prov.get('git_sha'))[:8]}")
    if not found:
        print("  (none carry _provenance yet — backfill as scripts are re-run)")


if __name__ == "__main__":
    main()
