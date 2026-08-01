"""Guard: a rerun must not silently overwrite the scientific record.

Why this exists
---------------
`README.md` states "no new run may silently overwrite the scientific record" and
`REPRODUCE.md` §4 says "Write a new immutable artifact; do not overwrite the controlling
result." Until `gfmcd.artifact_guard.guard_output_path` existed, that rule was documentation
only: **56 runners default `--out` to a path that `results/MANIFEST.csv` lists as an
artifact** (41 of them `canonical`), and `grep -rc 'out).exists()' scripts/` returned zero
files. Reproducing a claim by its documented command destroyed the artifact it was meant to
be compared against.

What this module asserts, in two layers
---------------------------------------
1. **The library guard works and cannot be defanged** (sections A-B). `guard_output_path`
   refuses by default, honours an explicit opt-in, has no environment escape hatch, and
   leaves the existing bytes untouched when it refuses.
2. **The residual hazard cannot grow** (section D). The 56 runners write their *result*
   JSON by their own means and their producing commits are pinned by manifests, so they are
   **inventoried, not edited** (see `CORRECTIONS.md` R35) — editing them would desynchronise
   a pinned provenance chain to fix a documentation defect. `tests/data/unguarded_out_defaults.json`
   records them; this test recomputes the set by AST and asserts it has not grown. A new
   runner that defaults `--out` at the record therefore fails CI on the day it lands.

Section D2 is the R7/R21 insurance: it builds its own fake repo so the *checker itself* is
proven able to report a violation, even on a machine where D1 could not run.

Mutation checks (each makes this module fail):
  * `if p.exists() and not overwrite` -> `and overwrite`            -> A2, A3
  * `overwrite: bool = False` -> `True`                             -> A2, A4, A5
  * add `and not os.environ.get(...)` to the refusal                -> B1
  * `raise` -> `warnings.warn` and fall through                     -> A2 (both halves)
  * add a runner with `--out` defaulting at a MANIFEST artifact     -> D1
  * break the AST checker so it silently finds nothing              -> D2
"""

from __future__ import annotations

import ast
import csv
import json
import os
import warnings
from pathlib import Path

import pytest

from gfmcd.artifact_guard import ArtifactExistsError, guard_output_path

REPO = Path(__file__).resolve().parents[1]
MANIFEST_CSV = REPO / "results/MANIFEST.csv"
INVENTORY = Path(__file__).resolve().parent / "data" / "unguarded_out_defaults.json"


# --------------------------------------------------------------------------- A: the guard
def test_a1_returns_path_when_target_is_absent(tmp_path: Path) -> None:
    target = tmp_path / "new_result.json"
    assert guard_output_path(target) == target
    assert not target.exists(), "guard must be a pure predicate — it creates nothing"


def test_a2_refuses_and_preserves_bytes_when_target_exists(tmp_path: Path) -> None:
    target = tmp_path / "a12_faithfulness_size.json"
    target.write_text('{"the": "record"}')
    before = target.read_bytes()
    with pytest.raises(ArtifactExistsError):
        guard_output_path(target)
    assert target.read_bytes() == before, "refusal must not touch the existing artifact"


def test_a3_explicit_opt_in_is_honoured(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text("{}")
    assert guard_output_path(target, overwrite=True) == target


def test_a4_default_is_refuse_and_opt_in_is_keyword_only() -> None:
    import inspect

    for fn in (guard_output_path,):
        param = inspect.signature(fn).parameters["overwrite"]
        assert param.default is False, f"{fn.__name__}: default must refuse"
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{fn.__name__}: overwrite must be keyword-only so no positional call can trip it"
        )


def test_a5_error_is_catchable_as_fileexisterror(tmp_path: Path) -> None:
    """Callers that already handle FileExistsError should keep working."""
    target = tmp_path / "x.json"
    target.write_text("{}")
    with pytest.raises(FileExistsError):
        guard_output_path(target)


# ------------------------------------------------------------- B: no environment escape
def test_b1_no_environment_variable_can_disable_the_guard(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "result.json"
    target.write_text("{}")
    for name in (
        "GFMCD_ALLOW_OVERWRITE",
        "GFMCD_OVERWRITE",
        "GFMCD_FORCE",
        "FORCE",
        "OVERWRITE",
    ):
        monkeypatch.setenv(name, "1")
    with pytest.raises(ArtifactExistsError):
        guard_output_path(target)
    assert "GFMCD_ALLOW_OVERWRITE" in os.environ  # the env really was set


# ------------------------------------------------- D: the residual hazard must not grow
def _out_default(script: Path) -> str | None:
    """Resolve a script's ``--out`` argparse default to its literal string, or None."""
    try:
        with warnings.catch_warnings():
            # Some runners contain regex literals with unescaped backslashes; that is a
            # pre-existing lint issue in those files, not a signal about --out defaults.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(script.read_text(errors="ignore"))
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if "--out" not in flags:
            continue
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            resolved = _resolve_default(tree, kw.value)
            if resolved is not None:
                return resolved
    return None


def _resolve_default(tree: ast.AST, node: ast.AST) -> str | None:
    """Resolve an `--out` default to a repo-relative path string.

    Originally this handled only a bare string literal, which made the checker blind to the
    `str(RESULTS_DIR / "name.json")` idiom that dominates the two newest studies — and therefore
    blind to `analyze_b5_context_signal_power.py`, whose default IS the canonical artifact
    controlling HCA23 and which had no exists-check at all (CORRECTIONS.md R43). Widening the
    resolver is why the inventory below is larger than the 56 it once held: the checker got
    sharper, not the tree worse.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # str(<expr>) -> resolve the inner expression
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "str" and len(node.args) == 1):
        return _resolve_default(tree, node.args[0])
    # <NAME> / "literal" [/ "literal" ...] -> join the module-level constant with the literals
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _resolve_default(tree, node.left)
        right = _resolve_default(tree, node.right)
        if left is not None and right is not None:
            # `left` is "" when the anchor is the checkout root, so lstrip keeps the result
            # repo-relative rather than absolute — otherwise it would never match a MANIFEST row.
            return f"{left.rstrip('/')}/{right.lstrip('/')}".lstrip("/")
        return None
    if isinstance(node, ast.Name):
        return _module_constant(tree, node.id)
    return None


def _module_constant(tree: ast.AST, name: str) -> str | None:
    """Best-effort value of a module-level `NAME = ...` path constant, repo-relative."""
    for stmt in getattr(tree, "body", []):
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
            continue
        # REPO / "results/..." collapses to the literal part; a bare REPO root resolves to "".
        value = _resolve_default(tree, stmt.value)
        if value is not None:
            return value
        # A repo-root anchor: `Path(__file__).resolve().parents[N]` (a Subscript) or a bare
        # `Path(...)` call. Both denote the checkout root, which is the empty repo-relative prefix.
        if isinstance(stmt.value, (ast.Call, ast.Subscript)):
            return ""
    return None


def _manifest_artifacts(csv_path: Path) -> set[str]:
    out: set[str] = set()
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            artifact = (row.get("artifact") or "").strip()
            if artifact and artifact != "na":
                out.add(artifact)
    return out


def _scripts_writing_at_the_record(repo: Path, scripts_dir: Path, csv_path: Path) -> set[str]:
    artifacts = _manifest_artifacts(csv_path)
    found: set[str] = set()
    for script in sorted(scripts_dir.rglob("*.py")):
        default = _out_default(script)
        if default and default in artifacts:
            found.add(str(script.relative_to(repo)))
    return found


@pytest.mark.skipif(not MANIFEST_CSV.exists(), reason="results/MANIFEST.csv absent")
def test_d1_inventory_of_runners_that_write_at_the_record_has_not_grown() -> None:
    recorded = {e["script"] for e in json.loads(INVENTORY.read_text())["entries"]}
    observed = _scripts_writing_at_the_record(REPO, REPO / "scripts", MANIFEST_CSV)
    new = observed - recorded
    assert not new, (
        "New runner(s) default --out to a path listed in results/MANIFEST.csv, so a plain "
        "re-run would overwrite the record. Give them a new-file default, or call "
        "gfmcd.artifact_guard.guard_output_path() before writing:\n  " + "\n  ".join(sorted(new))
    )
    fixed = recorded - observed
    assert not fixed, (
        "These runners no longer write at the record; prune them from "
        f"tests/data/unguarded_out_defaults.json:\n  " + "\n  ".join(sorted(fixed))
    )


def test_d2_the_checker_can_actually_detect_a_violation(tmp_path: Path) -> None:
    """Negative control: proves D1's checker is not vacuous, with no repo dependency."""
    (tmp_path / "results").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "results/MANIFEST.csv").write_text(
        "claim_id,status,evidence_level,polarity,artifact,script,manifest,figure\n"
        "C1,canonical,Behaviour,positive,results/x/precious.json,scripts/run_bad.py,na,na\n"
    )
    (tmp_path / "scripts/run_bad.py").write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--out', default='results/x/precious.json')\n"
    )
    (tmp_path / "scripts/run_ok.py").write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--out', default='results/x/fresh_run.json')\n"
    )
    found = _scripts_writing_at_the_record(
        tmp_path, tmp_path / "scripts", tmp_path / "results/MANIFEST.csv"
    )
    assert found == {"scripts/run_bad.py"}, (
        "the D1 checker failed to flag a runner writing at a MANIFEST artifact — "
        "D1 would pass vacuously"
    )


def test_d3_inventory_is_well_formed() -> None:
    inv = json.loads(INVENTORY.read_text())
    assert inv["count"] == len(inv["entries"])
    assert inv["entries"], "inventory must not be empty while the hazard exists"
    for e in inv["entries"]:
        assert e["manifest_status"] in {"canonical", "development", "superseded"}
