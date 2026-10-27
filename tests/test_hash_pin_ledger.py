"""Guard: content-addressed integrity seals must not drift undetected.

Why this exists
---------------
Commit `c38e1ba` inserted a one-line tier banner into documents whose SHA-256 is pinned by
other artifacts. It was framed as an additive, non-destructive edit. It was not: it broke
13 bound-file records of C36's frozen confirmation package and all four Head-Context-Atlas
protocol pins (29 stored occurrences), leaving `build_atlas_mechanism_package.py:295` unable
to run at all -- while the committed audits still record `"status": "PASS"` and the test
suite stayed green, because nothing exercised either gate.

The owner's reconciliation decision (2026-07-25) was to **log an amendment rather than revert
the banners** -- see `CORRECTIONS.md` R34. That decision creates a design problem this test
has to solve: a naive "assert every pin matches" test fails on the first run and gets
disabled, which is worse than no test.

So the assertion is inverted. `tests/data/hash_pin_ledger.json` records the known-broken pins
with a reason for each, and this module asserts the broken set is **exactly** that ledger:

  * a NEW drift -> a path appears broken that is not in the ledger  -> FAIL
  * a pin REPAIRED (e.g. banners reverted) -> a ledger path now verifies -> FAIL, so the
    ledger cannot silently rot into a list of things that were fixed years ago
  * silently widening the allowlist is a visible diff to a checked-in file, not a code change

Mutation checks (each of these makes this module fail, by construction):
  * append a byte to any pinned protocol doc          -> new drift, not in ledger
  * revert a tier banner                              -> ledger entry unexpectedly verifies
  * delete an entry from hash_pin_ledger.json         -> that path is broken but unlisted
  * move a relocated artifact back to its pinned path -> relocation entry resolves
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
LEDGER_PATH = Path(__file__).resolve().parent / "data" / "hash_pin_ledger.json"
PACKAGE = REPO / "results/phase-lattice/p3_phase_surgery_confirmation_package_v1.json"
AUDIT_SCRIPT = REPO / "scripts/splicebert/audit_p3_phase_surgery_confirmation_package.py"

# Uncommitted external inputs: model weights (gitignored) and the genome/annotation files, which the
# package binds by ABSOLUTE path. Neither is in a fresh checkout, so their ABSENCE is environment, not
# drift -- CI caught this the hard way. If such a file IS present it is still hash-verified below, so
# the guard is not weakened where it can actually check.
MODEL_PATH_MARKERS = ("data/models", "/models/")

# A fresh checkout must still be able to verify this many records (21 committed in-repo matches +
# the 13 ledger entries). Pinned so the env-dependent escape can never quietly erode the guard to
# nothing on a machine that happens to be missing everything.
MIN_VERIFIABLE_RECORDS = 34


def _is_uncommitted_external(path: str) -> bool:
    """True for inputs that are legitimately absent from a clean checkout."""
    return Path(path).is_absolute() or any(m in path for m in MODEL_PATH_MARKERS)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text())


def _ledger_paths(kind_prefix: str | None = None) -> set[str]:
    return {
        e["path"]
        for e in _ledger()["entries"]
        if kind_prefix is None or e["kind"].startswith(kind_prefix)
    }


def _flatten_records() -> list[dict]:
    spec = importlib.util.spec_from_file_location("audit_pkg", AUDIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.flatten_file_records(json.loads(PACKAGE.read_text()))


def _classify_package_records() -> tuple[set[str], int, int]:
    """Return (broken paths, verified count, env-dependent-absent count)."""
    broken: set[str] = set()
    verified = 0
    env_absent = 0
    for rec in _flatten_records():
        path, recorded = rec.get("path"), rec.get("sha256")
        if not path or not recorded:
            continue
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = REPO / path
        if not resolved.exists():
            # Absent AND uncommitted-external -> environment. Absent but expected in-repo -> drift.
            if _is_uncommitted_external(path):
                env_absent += 1
            else:
                broken.add(path)
            continue
        # Present: verify it, whatever kind of input it is.
        if _sha256(resolved) == recorded:
            verified += 1
        else:
            broken.add(path)
    return broken, verified, env_absent


def _observed_broken_package_paths() -> set[str]:
    return _classify_package_records()[0]


def test_ledger_is_well_formed() -> None:
    led = _ledger()
    assert led["entries"], "ledger must not be empty while pins are known broken"
    for e in led["entries"]:
        assert e["reason"].strip(), f"{e['path']} needs a reason"
        assert len(e["recorded_sha256"]) == 64


@pytest.mark.skipif(not PACKAGE.exists(), reason="C36 confirmation package not present")
def test_c36_package_broken_set_is_exactly_the_ledger() -> None:
    """No NEW drift, and no ledger entry that has quietly been repaired."""
    observed = _observed_broken_package_paths()
    expected = _ledger_paths("relocated") | _ledger_paths("content-drift") | _ledger_paths("missing")
    new_drift = observed - expected
    assert not new_drift, (
        "NEW hash-pin drift in the C36 frozen package -- something edited a pinned file:\n  "
        + "\n  ".join(sorted(new_drift))
    )
    repaired = expected - observed
    assert not repaired, (
        "These paths verify again; remove them from tests/data/hash_pin_ledger.json:\n  "
        + "\n  ".join(sorted(repaired))
    )


@pytest.mark.skipif(not PACKAGE.exists(), reason="C36 confirmation package not present")
def test_guard_still_verifies_enough_records_to_be_meaningful() -> None:
    """The env-dependent escape must never erode this guard to a vacuous pass.

    Absent uncommitted external inputs (model weights, the absolutely-pathed genome/annotation
    files) are skipped as environment. That escape is necessary for CI, and it is exactly the
    kind of concession that turns a guard into decoration. So: pin how much must still be
    checkable anywhere -- the 21 committed in-repo matches plus the 13 ledgered entries.
    """
    broken, verified, env_absent = _classify_package_records()
    checkable = verified + len(broken)
    assert checkable >= MIN_VERIFIABLE_RECORDS, (
        f"only {checkable} of the package's bound records were checkable "
        f"({verified} verified, {len(broken)} broken, {env_absent} skipped as environment). "
        f"Expected at least {MIN_VERIFIABLE_RECORDS}; the guard has eroded."
    )


def test_env_dependent_classification_is_correct() -> None:
    """Absent-external is environment; absent-in-repo is drift. No repo state needed."""
    assert _is_uncommitted_external("/Users/anyone/data/genomes/hg38.fa")
    assert _is_uncommitted_external("data/models/splicebert/pytorch_model.bin")
    # A committed, repo-relative artifact must NEVER be excusable as environment.
    assert not _is_uncommitted_external("results/phase-lattice/p3_phase_surgery_controls_n30.json")
    assert not _is_uncommitted_external("scripts/splicebert/freeze_p3_phase_surgery_confirmation.py")
    assert not _is_uncommitted_external(
        "studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_P3_CLOSURE.md"
    )


@pytest.mark.skipif(not PACKAGE.exists(), reason="C36 confirmation package not present")
def test_relocated_artifacts_are_byte_identical_at_their_new_path() -> None:
    """The relocation entries claim 'moved, not modified'. Prove it every run."""
    for e in _ledger()["entries"]:
        if e["kind"] != "relocated":
            continue
        new = REPO / e["new_path"]
        assert new.exists(), f"{e['path']} recorded as relocated to {e['new_path']}, which is missing"
        assert _sha256(new) == e["recorded_sha256"], (
            f"{e['new_path']} no longer matches the digest pinned for {e['path']} -- "
            "the relocation is no longer content-preserving"
        )


def test_repaired_pins_still_verify() -> None:
    """The pins c38e1ba broke were repaired on 2026-08-14; they must stay repaired.

    Retiring the document tier system (R47) removed the banner c38e1ba had inserted, restoring
    six documents -- the four Atlas protocol pins that gate `build_atlas_mechanism_package.py`,
    plus the P3-closure preregistration and its red-team synthesis -- to the exact bytes their
    pins record. The assertion is therefore the ordinary one now: they match. A failure here
    means a pinned protocol document has been edited again.
    """
    entries = [e for e in _ledger()["entries"] if e["kind"] == "repaired"]
    assert entries, "repaired pin entries missing from the ledger"
    for e in entries:
        doc = REPO / e["path"]
        if not doc.exists():
            pytest.fail(f"pinned protocol {e['path']} is missing")
        current = _sha256(doc)
        assert current == e["recorded_sha256"], (
            f"{e['path']} no longer matches its recorded pin "
            f"(pin {e['recorded_sha256'][:12]}..., now {current[:12]}...). "
            "This document was repaired on 2026-08-14 and has since been edited again."
        )


REGISTER_OPERATOR_FREEZES = {
    "studies/splicebert_register_operator_replication/protocol_freeze_v1.json": 45,
    "studies/splicebert_register_operator_replication/protocol_freeze_v2.json": 88,
}
# In-repo pins of those freezes known not to match this tree, each with its reason. Same contract
# as the C36 ledger: a new drift fails, and so does an entry that verifies again.
REGISTER_OPERATOR_KNOWN_DRIFT = {
    "pyproject.toml": "frozen with the DNABERT-2 `einops` extra, which this record drops (R56)",
    "uv.lock": "frozen with the DNABERT-2 `einops` extra, which this record drops (R56)",
}
# Where committed copies of the freezes' out-of-checkout inputs live (plain or gzip).
REGISTER_OPERATOR_INPUT_ARCHIVE = "results/splicebert_exon_edge_compatibility"
GENOME_MARKER = "/data/genomes/"


def _git_ignores(relative: str) -> bool:
    return subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", relative], cwd=REPO
    ).returncode == 0


@pytest.mark.parametrize("freeze", sorted(REGISTER_OPERATOR_FREEZES))
def test_register_operator_freeze_code_pins_match_the_tree(freeze: str) -> None:
    """`verify_freeze` refuses to run a frozen CC runner if any pinned file changed, but only at
    execution time. Check the real freezes here so an edit to pinned library code fails pytest."""
    pins = json.loads((REPO / freeze).read_text())["files"]
    drift: set[str] = set()
    verified = 0
    for name, recorded in pins.items():
        if Path(name).is_absolute() or name.startswith(".claude/"):
            continue  # outside this checkout: genome, session inputs, another worktree
        path = REPO / name
        if not path.is_file():
            if not (_git_ignores(name) or any(m in name for m in MODEL_PATH_MARKERS)):
                drift.add(name)  # a tracked input that disappeared, not environment
            continue
        if _sha256(path) == recorded:
            verified += 1
        else:
            drift.add(name)
    assert drift == set(REGISTER_OPERATOR_KNOWN_DRIFT), (
        f"{freeze}: unexpected drift {sorted(drift - set(REGISTER_OPERATOR_KNOWN_DRIFT))}; "
        f"verifies again {sorted(set(REGISTER_OPERATOR_KNOWN_DRIFT) - drift)}"
    )
    assert verified >= REGISTER_OPERATOR_FREEZES[freeze], f"{freeze}: only {verified} pins checkable"


@pytest.mark.parametrize("freeze", sorted(REGISTER_OPERATOR_FREEZES))
def test_register_operator_freeze_external_inputs_are_committed(freeze: str) -> None:
    """Every input a freeze pinned outside this checkout, other than the reference genome and
    annotation (pinned in REPRODUCE.md), must have its bytes committed: never only on one disk."""
    committed: set[str] = set()
    for path in (REPO / REGISTER_OPERATOR_INPUT_ARCHIVE).iterdir():
        if path.is_file():
            committed.add(_sha256(path))
            if path.suffix == ".gz":
                committed.add(hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest())
    pins = json.loads((REPO / freeze).read_text())["files"]
    external = {
        name: recorded for name, recorded in pins.items()
        if (Path(name).is_absolute() or name.startswith(".claude/")) and GENOME_MARKER not in name
    }
    assert external, f"{freeze}: expected out-of-checkout inputs to check"
    missing = sorted(name for name, recorded in external.items() if recorded not in committed)
    assert not missing, f"{freeze}: inputs with no committed bytes: {missing}"


def test_no_unpinned_atlas_protocol_doc_silently_joined_the_set() -> None:
    """If a new sha256 pin field appears in the atlas results, the ledger must cover it."""
    known = set(_ledger()["atlas_pin_fields"])
    found: set[str] = set()

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.endswith("_sha256") and isinstance(v, str) and len(v) == 64 and (
                    "plan" in k or "amendment" in k or "protocol" in k
                ):
                    found.add(k)
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    root = REPO / "results/splicebert_head_context_atlas"
    if not root.exists():
        pytest.skip("atlas results not present")
    for path in root.rglob("*.json"):
        try:
            walk(json.loads(path.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    unknown = found - known
    assert not unknown, (
        "New protocol-pin field(s) in the atlas results that the hash-pin ledger does not "
        f"track: {sorted(unknown)}"
    )
