"""Repository-scrub guards for documentation and evidence navigation.

These checks encode the cheap layers of ``docs/REPO_SCRUB_PROCEDURE.md``. They deliberately
do not rewrite frozen records to make the tree look tidier: the documents listed below are
content-addressed protocol inputs whose bytes are bound by a result manifest, so they are
verified rather than edited.

The document **tier** system (``> **Tier: T1 — canonical.**`` banners) was retired on
2026-08-14 and its two enforcing tests were removed with it; see ``CORRECTIONS.md`` R47. The
hash-pin guard below outlived it and is kept, because it never had anything to do with tiers:
it checks that documents a manifest binds by sha256 still match those bytes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[1]

_PINNED_PROTOCOL_REASON = (
    "content-addressed protocol input; its bytes are bound by a result manifest"
)

# Documents whose bytes are hash-pinned by a result manifest. Editing one silently invalidates
# the manifest that binds it, so they are verified here rather than reformatted.
HASH_PINNED_DOCS = {
    "studies/splicebert_partner_selection/preregistrations/AMENDMENT_BG1_BG2_PREEXECUTION_2026-08-11.md":
        ("results/splicebert_coupling/bg1_boundary_preflight_v1.manifest.json",
         _PINNED_PROTOCOL_REASON),
    "studies/splicebert_gtex_finetune_shift/IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_BEHAVIOR_V1.md":
        ("results/splicebert_gtex_finetune_shift/jacobian_sae_behavior_preflight_v1.manifest.json",
         _PINNED_PROTOCOL_REASON),
    "studies/splicebert_gtex_finetune_shift/IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_DELTA_BEHAVIOR_V2.md":
        ("results/splicebert_gtex_finetune_shift/jacobian_sae_delta_preflight_v1.manifest.json",
         _PINNED_PROTOCOL_REASON),
    "studies/splicebert_gtex_finetune_shift/IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_FRESH_DELTA_CONFIRMATION_V1.md":
        ("results/splicebert_gtex_finetune_shift/jacobian_sae_fresh_delta_confirmation_v1.manifest.json",
         _PINNED_PROTOCOL_REASON),
    "studies/splicebert_gtex_finetune_shift/IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_LARGE_V1.md":
        ("results/splicebert_gtex_finetune_shift/jacobian_sae_large_v1.manifest.json",
         _PINNED_PROTOCOL_REASON),
    "studies/splicebert_gtex_finetune_shift/IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_REPLICATION_V1.md":
        ("results/splicebert_gtex_finetune_shift/jacobian_sae_replication_v1.manifest.json",
         _PINNED_PROTOCOL_REASON),
    "studies/splicebert_gtex_finetune_shift/IMPLEMENTATION_ADDENDUM_MODEL_DIFFERENCE_JSAE_V1.md":
        ("results/splicebert_gtex_finetune_shift/model_difference_jsae_pilot_v1.manifest.json",
         _PINNED_PROTOCOL_REASON),
    "studies/splicebert_gtex_finetune_shift/PREREGISTRATION_JACOBIAN_SAE_V1.md":
        ("results/splicebert_gtex_finetune_shift/jacobian_sae_instrument_v1.manifest.json",
         _PINNED_PROTOCOL_REASON),
    "studies/splicebert_gtex_finetune_shift/PREREGISTRATION_JACOBIAN_SAE_V2.md":
        ("results/splicebert_gtex_finetune_shift/jacobian_sae_instrument_v2.manifest.json",
         _PINNED_PROTOCOL_REASON),
    "studies/splicebert_gtex_finetune_shift/PREREGISTRATION_JACOBIAN_SAE_V3.md":
        ("results/splicebert_gtex_finetune_shift/jacobian_sae_instrument_v3.manifest.json",
         _PINNED_PROTOCOL_REASON),
}

INLINE_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
REFERENCE_DEFINITION = re.compile(r"^\s*\[[^\]]+]:\s*(?:<([^>]+)>|(\S+))", re.M)
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.M)
HTML_ANCHOR = re.compile(r"""<(?:a\s+(?:name|id)|[^>]+\sid)=["']([^"']+)["']""", re.I)
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "doi:", "ftp:", "data:", "javascript:")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _tracked_markdown() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    # `git ls-files` includes paths deleted in the working tree until the index is updated.
    # Repository hygiene should validate the tree that will remain, not crash on an intentional
    # deletion during a cleanup change.
    return [candidate for path in output if (candidate := REPO / path).is_file()]


def _github_slug(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text.lower())
    text = re.sub(r"[`*_~]", "", text)
    text = re.sub(r"[^\w\- ]", "", text)
    return re.sub(r"\s+", "-", text.strip())


def _anchors(path: Path) -> set[str]:
    text = path.read_text(errors="ignore")
    anchors = set(HTML_ANCHOR.findall(text))
    seen: dict[str, int] = {}
    for heading in HEADING.findall(text):
        base = _github_slug(heading)
        duplicate_index = seen.get(base, 0)
        seen[base] = duplicate_index + 1
        anchors.add(base if duplicate_index == 0 else f"{base}-{duplicate_index}")
    return anchors


def _local_links(path: Path) -> list[str]:
    text = path.read_text(errors="ignore")
    links = [match.group(1).strip() for match in INLINE_LINK.finditer(text)]
    links.extend((left or right).strip() for left, right in REFERENCE_DEFINITION.findall(text))
    return links


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _sha256_bound_paths() -> set[str]:
    """Every path a tracked ``*.manifest.json`` binds to a sha256, as recorded (often absolute)."""
    manifests = subprocess.run(
        ["git", "ls-files", "*.manifest.json"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    bound: set[str] = set()
    for manifest in manifests:
        if not (REPO / manifest).is_file():
            continue
        for record in _walk_dicts(json.loads((REPO / manifest).read_text())):
            if isinstance(record.get("path"), str) and SHA256_HEX.fullmatch(str(record.get("sha256", ""))):
                bound.add(record["path"])
    return bound


def _is_sealed_local_payload(target: Path, bound: set[str]) -> bool:
    """A missing link target is legitimate only if git ignores it by design and a committed
    manifest binds its bytes by sha256, so a reader who obtains the file can authenticate it."""
    try:
        relative = target.relative_to(REPO).as_posix()
    except ValueError:
        return False
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", relative], cwd=REPO
    ).returncode == 0
    return ignored and any(path == relative or path.endswith("/" + relative) for path in bound)


def test_internal_markdown_links_and_anchors_resolve() -> None:
    failures: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}
    bound_paths: set[str] | None = None
    for source in _tracked_markdown():
        for raw in _local_links(source):
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1]
            raw = raw.split(' "', 1)[0].split(" '", 1)[0]
            if not raw or raw.startswith(EXTERNAL_PREFIXES) or raw.startswith("/"):
                continue

            target_text, _, anchor = raw.partition("#")
            target_text = unquote(target_text).split("?", 1)[0]
            target = (source if not target_text else source.parent / target_text).resolve()
            if not target.exists():
                if bound_paths is None:
                    bound_paths = _sha256_bound_paths()
                if not _is_sealed_local_payload(target, bound_paths):
                    failures.append(f"{source.relative_to(REPO)} -> {raw}: missing target")
                continue
            if anchor and target.suffix.lower() == ".md":
                known = anchor_cache.setdefault(target, _anchors(target))
                if unquote(anchor).lower() not in known:
                    failures.append(f"{source.relative_to(REPO)} -> {raw}: missing anchor")

    assert not failures, "Broken local Markdown links:\n  " + "\n  ".join(failures)


def test_sealed_local_payload_rule_requires_both_ignore_and_sha256_binding() -> None:
    bound = _sha256_bound_paths()
    cc_results = REPO / "results/splicebert_register_operator_replication"
    # Ignored by design and bound by sha256 in cells_v2.manifest.json: a legitimate dead link.
    assert _is_sealed_local_payload(cc_results / "cells_v2.json.gz", bound)
    # Ignored (`*.records.jsonl.gz`) but bound by no manifest: still a broken link.
    assert not _is_sealed_local_payload(cc_results / "unbound.records.jsonl.gz", bound)
    # Not ignored: a typo or a deleted file, never excused.
    assert not _is_sealed_local_payload(cc_results / "cells_v2.json", bound)


def test_hash_pinned_protocol_documents_still_match_their_manifests() -> None:
    """A document a manifest binds by sha256 must still be byte-identical to that record."""
    for relative, (manifest_relative, reason) in HASH_PINNED_DOCS.items():
        assert reason.strip(), f"{relative} needs an explicit reason"
        document = REPO / relative
        assert document.is_file(), f"{relative} is hash-pinned but no longer exists"
        manifest = json.loads((REPO / manifest_relative).read_text())
        current_hash = _sha256(document)
        matches = [
            record
            for record in _walk_dicts(manifest)
            if str(record.get("path", "")).endswith(relative)
            and record.get("sha256") == current_hash
        ]
        assert matches, (
            f"{relative} is hash-pinned, but {manifest_relative} does not bind its current "
            "bytes -- the document was edited without re-freezing the manifest"
        )


def test_every_canonical_result_row_has_a_claim_id() -> None:
    with (REPO / "results/MANIFEST.csv").open(newline="") as handle:
        offenders = [
            row["artifact"]
            for row in csv.DictReader(handle)
            if row["status"] == "canonical" and not (row.get("claim_id") or "").strip()
        ]
    assert not offenders, "Canonical result rows without claim IDs:\n  " + "\n  ".join(offenders)


def test_legacy_gtex_smoke_provenance_gap_is_disclosed() -> None:
    artifact = "results/splicebert_gtex_finetune_shift/seed-2020/smoke_sealed_val_v1.failed.json"
    with (REPO / "results/MANIFEST.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["artifact"] == artifact]
    assert len(rows) == 1
    if not (rows[0].get("manifest") or "").strip():
        decisions = (REPO / "studies/splicebert_gtex_finetune_shift/DECISIONS.md").read_text()
        assert "No result JSON was written by the legacy runner" in decisions
        assert "console outcome is preserved" in decisions


def test_live_gtex_summaries_use_native_margin_units() -> None:
    """Native classifier margins are raw logit differences, never cross-entropy nats."""
    live_summaries = [
        REPO / "studies/splicebert_gtex_finetune_shift/CLAIMS.md",
        REPO / "studies/splicebert_gtex_finetune_shift/FINDINGS.md",
        REPO / "studies/splicebert_gtex_finetune_shift/README.md",
    ]
    offenders = [
        str(path.relative_to(REPO))
        for path in live_summaries
        if re.search(r"\bnats?\b", path.read_text(), re.I)
    ]
    assert not offenders, (
        "Live GTEx summaries label raw classifier-logit differences as nats:\n  "
        + "\n  ".join(offenders)
    )
