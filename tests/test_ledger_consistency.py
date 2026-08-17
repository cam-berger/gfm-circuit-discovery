"""Guard: every claim and correction ID cited anywhere must resolve to a defining row.

Why this exists
---------------
This repository's dominant defect mode is **propagation failure**, not error: the same fact
lives in four to eight hand-maintained surfaces and a fix lands in one. The calibration
anchor is C37 -- a prior review found it had no defining ledger row, logged it, marked it
"resolved by branch reconciliation", and the reconciliation merged. `grep -c C37
CORRECTIONS.md` still returned 0 afterwards. Separately, five documents asserted the
correction record ended at "R1-R24" while `CORRECTIONS.md` ran to R28.

Neither class is catchable by reading. Both are trivially catchable mechanically, which is
what this module does. It is deliberately the cheapest possible check on the most expensive
failure mode.

Design notes that matter for anyone editing it
----------------------------------------------
* `CORRECTIONS.md` uses **two** formats: R1-R24 are table rows (some as *ranges*, e.g.
  `| R2-R3 |`), R25+ are `##` sections. The parser handles both; forgetting the ranges makes
  R2-R6/R12-R13 look undefined.
* `R100` is **git's rename-similarity score**, not a correction ID: `git diff -M --name-status`
  reports a rename as `R<percent>`, so a pure move with zero content change reads `R100`. Any
  doc quoting rename status therefore matches the `R<number>` scan and would otherwise look
  like an undeclared correction. It is excluded by name, not by a numeric cutoff, so a real
  R100 would still have to be declared.
* Exceptions are a small explicit map with a reason each. Adding one is a visible diff.

Mutation checks: cite a nonexistent `C99` in any tracked .md -> fail; write "R1-R24" in a
doc while R34 exists -> fail; set a null-polarity MANIFEST row to `positive` -> fail.
"""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

CLAIMS_FILES = sorted(
    path
    for path in (REPO / "studies").glob("*/CLAIMS.md")
    if not path.parent.name.startswith("TEMPLATE_")
)
CORRECTIONS = REPO / "studies/splicebert_partner_selection/CORRECTIONS.md"
MANIFEST_CSV = REPO / "results/MANIFEST.csv"
CLAIM_ROW_PATTERN = r"^\|\s*([A-Z]+\d+)\s*\|"
CLAIM_ID_PREFIXES = {"C", "HCA", "GFS", "PBC", "AC", "DBP", "PPM"}

# Experiment/run labels that collide with another established namespace and therefore must not be
# stored in MANIFEST.csv's claim_id field. R4 is both an acceptor experiment label and a global
# correction ID; treating the former as a claim silently resolves it to the unrelated latter.
NONCLAIM_EXPERIMENT_IDS = {"R4"}
SEMANTIC_EVIDENCE_ID_PATTERN = re.compile(
    r"[a-z0-9]+(?:-[a-z0-9]+)*/"
    r"[a-z0-9]+(?:-[a-z0-9]+)*/"
    r"[a-z0-9]+(?:-[a-z0-9]+)*"
)

# IDs cited in prose that legitimately have no standalone defining row.
CLAIM_ID_EXCEPTIONS = {
    "C37": (
        "Folded into C12 -- the B1 carrier-OV/behaviour arms are reported inside C12's "
        "closed-bounded row rather than as a standalone claim. 8 MANIFEST rows carry C37 as "
        "their claim_id. See CLAIMS.md C12 and CORRECTIONS.md R31."
    ),
    "HCA5": (
        "Removed from the Atlas ledger 2026-08-01 (CORRECTIONS.md R42): its controlling artifacts "
        "head_effect_summary.json / head_effect_matrix.jsonl were never committed on this branch, so "
        "the claim was not checkable. It was already terminal (withdrawn / invalid estimand) and is "
        "superseded by HCA10. The only remaining citation is the R42 entry that removed it, which "
        "quotes the row verbatim -- that is the intended end state, not a dangling reference."
    ),
    "HCA8": (
        "Removed from the Atlas ledger 2026-08-01 (CORRECTIONS.md R42): it cited "
        "head_effect_matrix.jsonl, which was never committed. Its checkable content -- competitor- and "
        "matched-decoy-destroy recruit the same head family, licensing no competition module -- is "
        "carried by HCA10 on exact joint patches with all artifacts present. HCA9 remains the live "
        "consumer of head_effect_donor_valid_reanalysis_v1.json and is unaffected. The only remaining "
        "citation is R42, which quotes the row verbatim."
    ),
}

# Tokens matching \bR\d+\b that are not correction IDs.
NON_CORRECTION_R_TOKENS = {
    "R100": (
        "git rename-similarity score, not a correction ID -- `git diff -M --name-status` reports "
        "a rename as `R<percent>`, so a pure move with zero content change reads `R100`. Excluded "
        "by name, not by a numeric cutoff: a real R100 correction would still need a defining row."
    ),
}

DASHES = "–—-"  # en dash, em dash, hyphen

# Documents whose own header freezes them to a past state, so a stale-looking range is correct.
FROZEN_SCOPE_DOCS = {
    "RESULTS_VERIFICATION.md": (
        'Header reads "this audit froze the C1-C26 record on 2026-07-16"; its "R1-R20" is an '
        "accurate snapshot of that date, not drift."
    ),
}


def _tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    return [candidate for p in out if (candidate := REPO / p).is_file()]


def _defined_claim_ids() -> set[str]:
    ids: set[str] = set()
    for f in CLAIMS_FILES:
        if f.exists():
            ids |= set(re.findall(CLAIM_ROW_PATTERN, f.read_text(), re.M))
    return ids


def _claim_id_pattern() -> str:
    """Match governed claim families, never run/correction IDs."""
    alternatives = "|".join(
        sorted(map(re.escape, CLAIM_ID_PREFIXES), key=len, reverse=True)
    )
    return rf"(?:{alternatives})\d+"


def _defined_correction_ids() -> set[str]:
    text = CORRECTIONS.read_text()
    ids = set(re.findall(r"^##\s*(R\d+)\s", text, re.M))          # R25+ sections
    ids |= set(re.findall(r"^\|\s*(R\d+)\s*\|", text, re.M))       # single-row form
    for lo, hi in re.findall(rf"^\|\s*R(\d+)\s*[{DASHES}]\s*R?(\d+)\s*\|", text, re.M):
        ids |= {f"R{n}" for n in range(int(lo), int(hi) + 1)}      # range rows
    return ids


def _cited(pattern: str) -> dict[str, set[str]]:
    cited: dict[str, set[str]] = {}
    for path in _tracked_markdown():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for tok in re.findall(pattern, text):
            cited.setdefault(tok, set()).add(str(path.relative_to(REPO)))
    return cited


def test_every_cited_claim_id_resolves() -> None:
    defined = _defined_claim_ids()
    assert defined, "no claim rows parsed -- the CLAIMS.md row format changed"
    unresolved = {
        cid: files
        for cid, files in _cited(rf"\b({_claim_id_pattern()})\b").items()
        if cid not in defined and cid not in CLAIM_ID_EXCEPTIONS
    }
    assert not unresolved, "Claim IDs cited but never defined:\n" + "\n".join(
        f"  {cid} -- cited in {len(files)} file(s): {sorted(files)[:3]}"
        for cid, files in sorted(unresolved.items())
    )


def test_claim_id_exceptions_are_still_needed() -> None:
    """Stops the exceptions map rotting into a list of things that were fixed."""
    defined = _defined_claim_ids()
    stale = sorted(set(CLAIM_ID_EXCEPTIONS) & defined)
    assert not stale, (
        f"{stale} now have defining rows; remove them from CLAIM_ID_EXCEPTIONS"
    )


def _acceptor_evidence_catalog() -> dict[str, str]:
    """Return descriptive acceptor evidence ID -> legacy claim alias."""
    claims = REPO / "studies/splicebert_acceptor_coupling/CLAIMS.md"
    text = claims.read_text()
    catalog: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("| `acceptor-coupling/"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert len(cells) == 4, f"malformed acceptor evidence-catalog row: {line}"
        semantic_id = cells[0].strip("`")
        legacy_alias = cells[2].strip("`")
        assert SEMANTIC_EVIDENCE_ID_PATTERN.fullmatch(semantic_id), (
            f"invalid reader-facing evidence ID: {semantic_id}"
        )
        assert semantic_id not in catalog, f"duplicate evidence ID: {semantic_id}"
        catalog[semantic_id] = legacy_alias
    return catalog


def test_acceptor_catalog_covers_legacy_claim_aliases() -> None:
    catalog = _acceptor_evidence_catalog()

    aliases = list(catalog.values())
    assert len(aliases) == len(set(aliases)), "legacy aliases map to multiple evidence IDs"
    defined_acceptor = {claim_id for claim_id in _defined_claim_ids() if claim_id.startswith("AC")}
    assert set(aliases) == defined_acceptor, (
        "acceptor evidence catalog and AC claim rows differ: "
        f"catalog={sorted(aliases)}, claims={sorted(defined_acceptor)}"
    )


@pytest.mark.skipif(not MANIFEST_CSV.exists(), reason="MANIFEST.csv not present")
def test_manifest_semantic_acceptor_ids_resolve() -> None:
    """Descriptive IDs in the fixed claim_id column must resolve to the study catalog.

    Multiple IDs are semicolon-separated. Their slash-separated internal structure must not
    be split, flattened back to AC shorthand, or allowed to drift independently of CLAIMS.md.
    """
    catalog = _acceptor_evidence_catalog()
    seen: set[str] = set()
    malformed: list[tuple[str, str]] = []
    with open(MANIFEST_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("claim_id") or "").strip()
            if "acceptor-coupling/" not in raw:
                continue
            for semantic_id in (part.strip() for part in raw.split(";")):
                if not SEMANTIC_EVIDENCE_ID_PATTERN.fullmatch(semantic_id):
                    malformed.append((row.get("artifact") or "", semantic_id))
                else:
                    seen.add(semantic_id)

    assert not malformed, f"malformed semantic acceptor IDs in MANIFEST.csv: {malformed}"
    unresolved = sorted(seen - set(catalog))
    assert not unresolved, (
        "MANIFEST.csv semantic acceptor IDs absent from the CLAIMS.md catalog: "
        f"{unresolved}"
    )


def test_every_cited_correction_id_resolves() -> None:
    defined = _defined_correction_ids()
    assert defined, "no correction rows parsed -- the CORRECTIONS.md format changed"
    unresolved = {
        rid: files
        for rid, files in _cited(r"\b(R\d+)\b").items()
        if rid not in defined and rid not in NON_CORRECTION_R_TOKENS
    }
    assert not unresolved, "Correction IDs cited but never defined:\n" + "\n".join(
        f"  {rid} -- cited in {sorted(files)[:3]}" for rid, files in sorted(unresolved.items())
    )


def test_no_document_understates_the_correction_range() -> None:
    """The R1-R24 defect: a doc asserting a range whose upper bound is already stale."""
    highest = max(int(r[1:]) for r in _defined_correction_ids())
    offenders: list[str] = []
    for path in _tracked_markdown():
        rel = str(path.relative_to(REPO))
        # Point-in-time records: their range assertion is a correct snapshot of the day they
        # were frozen, and updating it would falsify the record rather than repair it.
        if rel in FROZEN_SCOPE_DOCS:
            continue
        if "ULTRAREVIEW" in rel or "REVIEW_2026" in rel or "FORMALIZATION_REVIEW" in rel:
            continue
        text = path.read_text(errors="ignore")
        for m in re.finditer(rf"\bR(\d+)\s*[{DASHES}]\s*R(\d+)\b", text):
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo != 1 or hi >= highest:
                continue
            # "R1-R24 are table rows" describes the file's LAYOUT, not the record's extent.
            trailer = text[m.end() : m.end() + 40].lower()
            if any(w in trailer for w in ("table row", "section", "are `##`", "format")):
                continue
            offenders.append(f"{rel}: claims R{lo}-R{hi}, but R{highest} exists")
    assert not offenders, "Stale correction-range assertions:\n  " + "\n  ".join(offenders)


@pytest.mark.skipif(not MANIFEST_CSV.exists(), reason="MANIFEST.csv not present")
def test_manifest_claim_ids_resolve() -> None:
    defined = _defined_claim_ids()
    seen: set[str] = set()
    with open(MANIFEST_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            seen |= set(re.findall(rf"\b({_claim_id_pattern()})\b", row.get("claim_id") or ""))
    unresolved = sorted(seen - defined - set(CLAIM_ID_EXCEPTIONS))
    assert not unresolved, f"MANIFEST.csv claim_ids with no defining row: {unresolved}"


@pytest.mark.skipif(not MANIFEST_CSV.exists(), reason="MANIFEST.csv not present")
def test_manifest_claim_ids_exclude_known_experiment_codes() -> None:
    offenders: list[str] = []
    with open(MANIFEST_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            claim_ids = set(re.findall(r"\bR\d+\b", row.get("claim_id") or ""))
            collisions = sorted(claim_ids & NONCLAIM_EXPERIMENT_IDS)
            if collisions:
                offenders.append(f"{row.get('artifact')}: {', '.join(collisions)}")
    assert not offenders, (
        "experiment/run IDs used as MANIFEST claim_ids:\n  " + "\n  ".join(offenders)
    )


def test_gtex_controlling_artifact_references_exist() -> None:
    """Every artifact named by a GFS row must resolve without relying on prose inference."""
    claims = REPO / "studies/splicebert_gtex_finetune_shift/CLAIMS.md"
    if not claims.exists():
        pytest.skip("GTEx fine-tuning-shift claim ledger is absent")

    result_root = REPO / "results/splicebert_gtex_finetune_shift"
    missing: list[str] = []
    for line in claims.read_text().splitlines():
        if not re.match(r"^\|\s*GFS\d+\s*\|", line):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 5, f"malformed GFS claim row: {line[:100]}"
        claim_id, controlling = cells[0], cells[3]
        for raw in re.findall(r"`([^`]+\.(?:json|npz|pt|csv))`", controlling):
            path = REPO / raw if "/" in raw else result_root / raw
            if not path.exists():
                missing.append(f"{claim_id}: {raw}")

    assert not missing, "GFS controlling artifacts are missing:\n  " + "\n  ".join(missing)


@pytest.mark.skipif(not MANIFEST_CSV.exists(), reason="MANIFEST.csv not present")
def test_manifest_polarity_vocabulary_is_closed() -> None:
    """A typo'd polarity must not silently become an unguarded value."""
    allowed = {"positive", "null", "bounded", "refutation", "na", ""}
    bad: set[str] = set()
    with open(MANIFEST_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            value = (row.get("polarity") or "").strip()
            if value not in allowed:
                bad.add(value)
    assert not bad, f"unrecognised polarity value(s) in MANIFEST.csv: {sorted(bad)}"
