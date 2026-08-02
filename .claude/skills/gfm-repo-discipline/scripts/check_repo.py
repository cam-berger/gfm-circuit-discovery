#!/usr/bin/env python3
"""Checks for the defect classes the pytest suite structurally cannot catch.

The suite here verifies that paths resolve, IDs are defined, links point somewhere, and tier
banners exist. It never verifies that a runner can *execute*, that a memorial can *fire*, that a
seed is *stable*, or that a claim's artifact is *obtainable*. Every check below exists because a
real defect in this repository survived a green suite. The R-number names the correction entry in
studies/splicebert_partner_selection/CORRECTIONS.md.

    uv run python .claude/skills/gfm-repo-discipline/scripts/check_repo.py
    ... --only dead_gates,unstable_seeds        # one or more check names
    ... --json                                  # machine-readable

Stdlib only, so it runs without the project environment. Exit 1 if any FAIL fires; REVIEW findings
are advisory and never fail the run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

# Retired labels that must not appear in live prose. Provenance files may quote them.
RETIRED = [
    (r"\b10\.67\b", "10.67 was a DFT-bin label, not a wavelength; the carrier is ~10.63"),
    (r"\b10\.58\b", "10.58 +/- 0.35 was never a measured constant"),
    (r"±\s*0\.005|\+/-\s*0\.005", "the '+/-0.005' carrier fit precision is withdrawn (R37)"),
    (r"splice[- ]site strength", "withdrawn (R22): say 'acceptor-local payload, semantics unresolved'"),
]
PROVENANCE_HINTS = ("CORRECTIONS", "REDTEAM", "DECISIONS", "archive/", "PREREGISTRATION",
                    "NOMENCLATURE", "PREREG_CONFORMANCE", "REVIEW_", "retired", "Retired")
DETRITUS = re.compile(
    r"(HANDOFF|_HANDOFF|NIGHT_RUN|TRIAGE|EXECUTION_REPORT|return_to_planner|"
    r"SESSION|SUMMARY_20\d\d|_20\d\d-\d\d-\d\d\.md$)", re.I)
BINARY_SUFFIXES = (".pt", ".npz", ".npy", ".safetensors", ".pkl")
STDLIB_OK = {"os", "sys", "re", "json", "math", "time", "typing", "pathlib", "argparse", "csv",
             "collections", "hashlib", "subprocess", "itertools", "functools", "random", "warnings",
             "dataclasses", "importlib", "statistics", "copy", "gzip", "shutil", "textwrap", "ast",
             "traceback", "contextlib", "abc", "enum", "glob", "io", "tempfile", "datetime", "string",
             "pickle", "inspect", "numpy", "torch", "pandas", "scipy", "sklearn", "matplotlib",
             "pytest", "gfmcd", "transformers", "pyfaidx", "maxentpy", "tqdm", "multimolecule",
             "seaborn", "networkx", "graphviz", "yaml", "platform", "logging", "operator", "unicodedata",
             "__future__", "annotations", "uuid", "zipfile", "tarfile", "base64", "struct", "array"}


def tracked(repo: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True)
    return out.stdout.split()


def parse(path: Path) -> ast.AST | None:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(path.read_text(errors="ignore"))
    except (SyntaxError, ValueError, OSError):
        return None


# --------------------------------------------------------------------------- checks

def dead_gates(repo: Path, files: list[str]) -> list[dict]:
    """R41: a conjunctive module-level skipif over a deleted artifact silently kills a whole file.

    Twelve memorials defending live claim C14 skipped for months because ONE of three ANDed paths
    was deleted. They still counted toward the suite total while asserting nothing.
    """
    out = []
    for f in (x for x in files if x.startswith("tests/") and x.endswith(".py")):
        src = (repo / f).read_text(errors="ignore")
        if "pytestmark" not in src:
            continue
        consts = dict(re.findall(r"^([A-Z_][A-Z_0-9]*)\s*=\s*REPO\s*/\s*[\"']([^\"']+)[\"']", src, re.M))
        gate = re.search(r"pytestmark\s*=\s*pytest\.mark\.skipif\((.*?)\n\n", src, re.S)
        if not gate:
            continue
        names = set(re.findall(r"\b([A-Z_][A-Z_0-9]*)\.exists\(\)", gate.group(1)))
        missing = sorted(n for n in names if n in consts and not (repo / consts[n]).exists())
        if missing and len(names) > 1:
            out.append({"check": "dead_gates", "severity": "FAIL", "file": f,
                        "detail": f"module-level skipif ANDs {sorted(names)}; "
                                  f"{missing} do not exist, so EVERY test in this file skips",
                        "fix": "gate per-artifact so tests not needing the missing file still fire; "
                               "mark the rest individually with the reason (R41)"})
    return out


def broken_sibling_imports(repo: Path, files: list[str]) -> list[dict]:
    """R41: a runner importing a deleted neighbour dies at import, so its artifact cannot be rebuilt.

    make_fig_phase_data.py -- canonical producer for C8's figure data -- imported make_fig_wave_fit,
    deleted two cleanups earlier. Nothing noticed: no test runs the figure producers.
    """
    names = {Path(f).stem for f in files if f.endswith(".py")}
    out = []
    for f in (x for x in files if x.endswith(".py")):
        tree = parse(repo / f)
        if tree is None:
            continue
        sib = Path(f).parent
        # Module level only. A lazy `import X` inside a function, or one guarded by
        # try/except ImportError, is the correct way to carry an optional dependency.
        for node in getattr(tree, "body", []):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if m in STDLIB_OK or (repo / sib / f"{m}.py").exists():
                    continue
                if m in names:  # exists elsewhere -- resolved by a sys.path.insert, fine
                    continue
                try:
                    __import__(m)
                except Exception:
                    out.append({"check": "broken_sibling_imports", "severity": "FAIL",
                                "file": f"{f}:{node.lineno}",
                                "detail": f"imports `{m}`, which is not beside it and not installed",
                                "fix": "restore the module or remove the import; a canonical producer "
                                       "that dies at import cannot regenerate its artifact (R41)"})
    return out


def _canonical_scripts(repo: Path) -> set[str]:
    csv_path = repo / "results/MANIFEST.csv"
    if not csv_path.exists():
        return set()
    out = set()
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") == "canonical":
                for s in (r.get("script") or "").split(";"):
                    if s.strip():
                        out.add(s.strip())
    return out


def unstable_seeds(repo: Path, files: list[str]) -> list[dict]:
    """R43: hash() of a str is randomized per process unless PYTHONHASHSEED is set AND recorded.

    C40's published n_sig was drawn from hash(("L3H1", T)) while its manifest declared seed: 0.
    """
    canonical = _canonical_scripts(repo)
    out = []
    for f in (x for x in files if x.endswith(".py") and not x.startswith("tests/")):
        for i, line in enumerate((repo / f).read_text(errors="ignore").splitlines(), 1):
            if re.search(r"(default_rng|RandomState|seed)\s*\(\s*[^)]*\bhash\s*\(", line):
                # FAIL only when the scientific record depends on the draw. A development-status
                # pilot has the same bug but nothing cites its numbers.
                sev = "FAIL" if f in canonical else "REVIEW"
                out.append({"check": "unstable_seeds", "severity": sev, "file": f"{f}:{i}",
                            "detail": "RNG seeded from builtin hash(); randomized per process for str keys",
                            "fix": "use int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) -- the "
                                   "`shash` idiom already in this repo (R43)"})
    return out


def claim_artifacts(repo: Path, files: list[str]) -> list[dict]:
    """HCA20/HCA22, HCA5/HCA8: a claim row naming an artifact that is not in the tree.

    verify_manifest.py is one-directional -- it checks that MANIFEST paths resolve, never that a
    CLAIMS row's artifact has a MANIFEST row. This is the reverse check.
    """
    by_name: dict[str, list[str]] = {}
    for t in files:
        by_name.setdefault(Path(t).name, []).append(t)
    out = []
    for f in (x for x in files if x.endswith("CLAIMS.md")):
        for line in (repo / f).read_text(errors="ignore").splitlines():
            m = re.match(r"^\|\s*([A-Z]+\d+)\s*\|", line)
            if not m:
                continue
            cells = line.split("|")
            if len(cells) < 6:
                continue
            for tok in re.findall(r"`([^`]+\.(?:json|jsonl|csv|npz|pt))`", cells[4]):
                tok = tok.strip()
                if tok in files or Path(tok).name in by_name:
                    continue
                if any((repo / base / tok).exists() for base in (".", "results", Path(f).parent)):
                    continue
                out.append({"check": "claim_artifacts", "severity": "FAIL",
                            "file": f, "detail": f"{m.group(1)} names `{tok}`, which is not in the tree",
                            "fix": "commit the compact artifact + manifest, or give a retrievable URL "
                                   "and sha256 in the row, or retire the claim with a correction"})
    return out


def ci_notation(repo: Path, files: list[str]) -> list[dict]:
    """R40: [a, b] must mean a confidence interval. Across-draw spread was rendered the same way.

    Advisory -- a row mentioning draws AND bracket notation needs a human to confirm which it is.
    """
    out = []
    for f in (x for x in files if x.endswith("CLAIMS.md")):
        for line in (repo / f).read_text(errors="ignore").splitlines():
            m = re.match(r"^\|\s*([A-Z]+\d+)\s*\|", line)
            if m and re.search(r"\bdraws?\b", line, re.I) and re.search(r"\[[-+−0-9.]+\s*,", line):
                out.append({"check": "ci_notation", "severity": "REVIEW", "file": f,
                            "detail": f"{m.group(1)} mixes draw language with [a, b] notation",
                            "fix": "confirm each bracket is a CI; label across-draw spread as "
                                   "'range' or 'SD', never [a, b] (R40)"})
    return out


def session_detritus(repo: Path, files: list[str]) -> list[dict]:
    """Agent-session exhaust: handoffs, triage notes, night-run reports, dated reviews."""
    out = []
    for f in files:
        if f.endswith(".md") and DETRITUS.search(Path(f).name):
            out.append({"check": "session_detritus", "severity": "REVIEW", "file": f,
                        "detail": "looks like an agent-session or dated review document",
                        "fix": "fold the durable content into CLAIMS/CORRECTIONS/DECISIONS or a test. "
                               "Keep only if executable code or a frozen protocol cites it -- and say so"})
    return out


ALLOWED_BINARIES = {
    # Small, and its own manifest names it as a hash-pinned input rather than a fitted model.
    "results/splicebert_gtex_finetune_shift/native_context_final_mlp_unit_evaluation_v1.npz",
}


def staged_binaries(repo: Path, files: list[str]) -> list[dict]:
    """A fitted dictionary or activation capture is not evidence. 847 MB were force-added once."""
    return [{"check": "staged_binaries", "severity": "FAIL", "file": f,
             "detail": f"{Path(f).suffix} is a training/runtime output, not evidence",
             "fix": "commit the compact result JSON + manifest instead; .gitignore already excludes it"}
            for f in files if f.endswith(BINARY_SUFFIXES) and f not in ALLOWED_BINARIES]


# A line that names a retired term in order to FORBID it is guidance, not a leak. Nomenclature
# docs, figure READMEs and results indexes are mostly made of such lines, so without this the check
# is pure noise and gets ignored -- which is the real failure mode for an advisory check.
PROHIBITION = re.compile(
    r"\bnot\b|\bnever\b|retired|withdraw|instead|rather than|\bwas the label\b|"
    r"\bdo not\b|\bstop\b|10\.63|use\s+\*\*", re.I)


def _prohibits(line: str) -> bool:
    return bool(PROHIBITION.search(line))


def retired_terms(repo: Path, files: list[str]) -> list[dict]:
    """Retired nomenclature leaking into live prose (docs/NOMENCLATURE.md)."""
    out = []
    for f in (x for x in files if x.endswith(".md")):
        if any(h in f for h in PROVENANCE_HINTS):
            continue
        for i, line in enumerate((repo / f).read_text(errors="ignore").splitlines(), 1):
            for pat, why in RETIRED:
                if not re.search(pat, line, re.I) or _prohibits(line):
                    continue
                if True:
                    out.append({"check": "retired_terms", "severity": "REVIEW", "file": f"{f}:{i}",
                                "detail": why, "fix": "see docs/NOMENCLATURE.md"})
    return out


def manifest_gaps(repo: Path, files: list[str]) -> list[dict]:
    """A canonical result with no sibling manifest has no recoverable execution environment."""
    csv_path = repo / "results/MANIFEST.csv"
    if not csv_path.exists():
        return []
    out = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            a = (r.get("artifact") or "").strip()
            name = Path(a).name
            if r.get("status") != "canonical" or not a.endswith(".json") or "manifest" in name:
                continue
            if not (r.get("manifest") or "").strip():
                out.append({"check": "manifest_gaps", "severity": "REVIEW", "file": a,
                            "detail": f"canonical artifact for {r.get('claim_id')} has no sibling manifest",
                            "fix": "a result without a manifest has no recoverable execution "
                                   "environment; new runs must emit one in the same step"})
    return out


CHECKS = {f.__name__: f for f in (dead_gates, broken_sibling_imports, unstable_seeds, claim_artifacts,
                                  ci_notation, session_detritus, staged_binaries, retired_terms,
                                  manifest_gaps)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="repository root (default: cwd)")
    ap.add_argument("--only", help="comma-separated check names")
    ap.add_argument("--json", action="store_true", dest="as_json")
    ap.add_argument("--list", action="store_true", help="list check names and exit")
    args = ap.parse_args()

    if args.list:
        for name, fn in CHECKS.items():
            first = ((fn.__doc__ or "no description").strip().splitlines() or ["no description"])[0]
            print(f"{name:26} {first}")
        return 0

    repo = Path(args.repo).resolve()
    files = tracked(repo)
    if not files:
        print(f"no tracked files under {repo} -- is it a git repo?", file=sys.stderr)
        return 2

    selected = [n.strip() for n in args.only.split(",")] if args.only else list(CHECKS)
    unknown = [n for n in selected if n not in CHECKS]
    if unknown:
        print(f"unknown check(s): {unknown}. --list to see names.", file=sys.stderr)
        return 2

    findings: list[dict] = []
    for name in selected:
        findings.extend(CHECKS[name](repo, files))

    if args.as_json:
        print(json.dumps(findings, indent=2))
    else:
        fails = [f for f in findings if f["severity"] == "FAIL"]
        review = [f for f in findings if f["severity"] == "REVIEW"]
        for group, label in ((fails, "FAIL"), (review, "REVIEW")):
            if not group:
                continue
            print(f"\n=== {label} ({len(group)}) ===")
            for f in group:
                print(f"  [{f['check']}] {f['file']}\n      {f['detail']}\n      -> {f['fix']}")
        if not findings:
            print("clean: no findings.")
        else:
            print(f"\n{len(fails)} FAIL, {len(review)} REVIEW "
                  f"(REVIEW is advisory and does not fail the run)")
    return 1 if any(f["severity"] == "FAIL" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
