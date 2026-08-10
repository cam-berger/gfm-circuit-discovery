"""Fail-closed freshness registry for positional-boundary coherence.

The registry is derived only from an explicit source specification.  Every
required historical category must name one or more immutable source blobs,
their expected hashes, and exact extraction paths/columns.  Missing artifacts
are errors; they are never interpreted as empty cohorts.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence


SOURCE_SPEC_SCHEMA = "gfmcd.positional_boundary_coherence.freshness_sources.v1"
REGISTRY_SCHEMA = "gfmcd.positional_boundary_coherence.freshness_registry.v1"
AUDIT_SCHEMA = "gfmcd.positional_boundary_coherence.freshness_registry_audit.v1"
REQUIRED_CATEGORIES = (
    "c43_predecessor",
    "c43_replication",
    "c44_smoke",
    "c44_development",
    "c44_reserve",
    "gfs25_v1",
    "gfs25_v2",
    "source_protocol_registry",
)
GENE_PATTERN = re.compile(r"^ENSG[0-9]+$")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_payload(payload: Any) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def normalized_gene_id(value: object) -> str:
    root = str(value).strip().split(".", 1)[0]
    if not GENE_PATTERN.fullmatch(root):
        raise ValueError(f"not a version-normalizable Ensembl gene ID: {value!r}")
    return root


def _resolve_commit(repo: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_source_blob(
    source: Mapping[str, Any], *, repo: str | Path
) -> tuple[bytes, dict[str, str]]:
    """Read one explicitly pinned worktree or Git-object source."""

    root = Path(repo).resolve(strict=True)
    revision = str(source.get("revision", ""))
    path_text = str(source.get("path", ""))
    expected = str(source.get("expected_sha256", ""))
    if not revision or not path_text or len(expected) != 64:
        raise ValueError("every source needs revision, path, and expected_sha256")
    if revision == "WORKTREE":
        path = Path(path_text)
        if not path.is_absolute():
            path = root / path
        payload = path.resolve(strict=True).read_bytes()
        resolved_revision = "WORKTREE"
        command = f"read-bytes {path.resolve()}"
    else:
        resolved_revision = _resolve_commit(root, revision)
        object_name = f"{resolved_revision}:{path_text}"
        completed = subprocess.run(
            ["git", "show", object_name], cwd=root, check=True, capture_output=True
        )
        payload = completed.stdout
        command = f"git show {object_name}"
    observed = sha256_bytes(payload)
    if observed != expected:
        raise ValueError(
            f"source hash mismatch for {revision}:{path_text}: expected {expected}, observed {observed}"
        )
    return payload, {
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "path": path_text,
        "source_sha256": observed,
        "extraction_command": command,
    }


def _walk_explicit(value: Any, segments: Sequence[str], *, pointer: str) -> list[Any]:
    if not segments:
        return [value]
    segment, remainder = segments[0], segments[1:]
    if segment == "*":
        if isinstance(value, list):
            children = value
        elif isinstance(value, Mapping):
            children = [value[key] for key in sorted(value)]
        else:
            raise ValueError(f"pointer {pointer!r}: '*' reached a scalar")
        return [item for child in children for item in _walk_explicit(child, remainder, pointer=pointer)]
    if isinstance(value, Mapping):
        if segment not in value:
            raise ValueError(f"pointer {pointer!r}: missing key {segment!r}")
        return _walk_explicit(value[segment], remainder, pointer=pointer)
    if isinstance(value, list) and segment.isdigit():
        index = int(segment)
        if not 0 <= index < len(value):
            raise ValueError(f"pointer {pointer!r}: index {index} is absent")
        return _walk_explicit(value[index], remainder, pointer=pointer)
    raise ValueError(f"pointer {pointer!r}: cannot descend through a scalar")


def _flatten_terminal(value: Any, *, location: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{location}: explicitly extracted list is empty")
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"{location}: terminal list must contain only strings")
        return list(value)
    raise ValueError(f"{location}: terminal must be a gene string or list of gene strings")


def extract_gene_ids(payload: bytes, extraction: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Execute an explicit extraction recipe; no recursive key heuristic exists."""

    format_name = str(extraction.get("format", ""))
    raw_values: list[tuple[str, str]] = []
    records: list[dict[str, Any]] = []
    if format_name == "json":
        document = json.loads(payload)
        pointers = extraction.get("pointers")
        if not isinstance(pointers, list) or not pointers:
            raise ValueError("JSON extraction needs non-empty explicit pointers")
        for pointer_value in pointers:
            pointer = str(pointer_value)
            if not pointer:
                raise ValueError("JSON extraction pointer cannot be empty")
            terminals = _walk_explicit(document, pointer.split("."), pointer=pointer)
            extracted: list[str] = []
            for terminal in terminals:
                extracted.extend(_flatten_terminal(terminal, location=pointer))
            if not extracted:
                raise ValueError(f"pointer {pointer!r} extracted no genes")
            raw_values.extend((pointer, value) for value in extracted)
            records.append({"selector": pointer, "raw_count": len(extracted)})
    elif format_name == "jsonl":
        pointers = extraction.get("pointers")
        if not isinstance(pointers, list) or not pointers:
            raise ValueError("JSONL extraction needs non-empty explicit pointers")
        documents = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        if not documents:
            raise ValueError("JSONL source is empty")
        for pointer_value in pointers:
            pointer = str(pointer_value)
            extracted: list[str] = []
            for line_no, document in enumerate(documents, 1):
                terminals = _walk_explicit(document, pointer.split("."), pointer=pointer)
                for terminal in terminals:
                    extracted.extend(_flatten_terminal(terminal, location=f"line {line_no}:{pointer}"))
            if not extracted:
                raise ValueError(f"pointer {pointer!r} extracted no genes")
            raw_values.extend((pointer, value) for value in extracted)
            records.append({"selector": pointer, "raw_count": len(extracted)})
    elif format_name in {"tsv", "csv"}:
        columns = extraction.get("columns")
        if not isinstance(columns, list) or not columns:
            raise ValueError("tabular extraction needs non-empty explicit columns")
        delimiter = "\t" if format_name == "tsv" else ","
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8")), delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("tabular source has no header")
        rows = list(reader)
        if not rows:
            raise ValueError("tabular source contains no rows")
        for column_value in columns:
            column = str(column_value)
            if column not in reader.fieldnames:
                raise ValueError(f"tabular extraction column {column!r} is absent")
            values = [row[column] for row in rows if row[column].strip()]
            if not values:
                raise ValueError(f"column {column!r} extracted no genes")
            raw_values.extend((column, value) for value in values)
            records.append({"selector": column, "raw_count": len(values)})
    else:
        raise ValueError("extraction format must be json, jsonl, tsv, or csv")

    normalized = [normalized_gene_id(value) for _selector, value in raw_values]
    for record in records:
        selector = str(record["selector"])
        selector_ids = sorted(
            {normalized_gene_id(value) for source_selector, value in raw_values if source_selector == selector}
        )
        record["normalized_count"] = len(selector_ids)
        record["normalized_sha256"] = sha256_payload(selector_ids)
    return sorted(set(normalized)), records


def validate_source_spec(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    if spec.get("schema") != SOURCE_SPEC_SCHEMA:
        raise ValueError(f"source spec schema must be {SOURCE_SPEC_SCHEMA}")
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source spec needs a non-empty sources list")
    categories = {str(source.get("category", "")) for source in sources}
    missing = set(REQUIRED_CATEGORIES) - categories
    if missing:
        raise ValueError(f"freshness source spec is missing required categories: {sorted(missing)}")
    source_ids = [str(source.get("source_id", "")) for source in sources]
    if any(not value for value in source_ids) or len(source_ids) != len(set(source_ids)):
        raise ValueError("freshness source_id values must be non-empty and unique")
    for source in sources:
        if not isinstance(source.get("extraction"), Mapping):
            raise ValueError(f"{source.get('source_id')}: explicit extraction recipe is required")
        if str(source.get("category", "")) not in REQUIRED_CATEGORIES:
            raise ValueError(f"{source.get('source_id')}: unknown freshness category")
        expected_count = source.get("expected_normalized_count")
        if expected_count is None or int(expected_count) <= 0:
            raise ValueError(
                f"{source.get('source_id')}: expected_normalized_count must be pinned and positive"
            )
    return sorted((dict(source) for source in sources), key=lambda row: str(row["source_id"]))


def derive_registry(spec: Mapping[str, Any], *, repo: str | Path) -> dict[str, Any]:
    """Production registry derivation from all required, hash-pinned sources."""

    sources = validate_source_spec(spec)
    union: set[str] = set()
    category_ids: dict[str, set[str]] = {category: set() for category in REQUIRED_CATEGORIES}
    source_records: list[dict[str, Any]] = []
    for source in sources:
        payload, provenance = read_source_blob(source, repo=repo)
        genes, extraction_records = extract_gene_ids(payload, source["extraction"])
        expected_count = int(source["expected_normalized_count"])
        if len(genes) != expected_count:
            raise ValueError(
                f"{source['source_id']}: expected {expected_count} normalized genes, found {len(genes)}"
            )
        category = str(source["category"])
        union.update(genes)
        category_ids[category].update(genes)
        source_records.append(
            {
                "source_id": str(source["source_id"]),
                "category": category,
                **provenance,
                "source_size": len(payload),
                "extraction": dict(source["extraction"]),
                "extraction_records": extraction_records,
                "normalized_count": len(genes),
                "normalized_gene_ids_sha256": sha256_payload(genes),
                "normalized_gene_ids": genes,
            }
        )
    genes = sorted(union)
    categories = {
        category: {
            "n_sources": sum(record["category"] == category for record in source_records),
            "normalized_count": len(category_ids[category]),
            "normalized_gene_ids_sha256": sha256_payload(sorted(category_ids[category])),
        }
        for category in REQUIRED_CATEGORIES
    }
    core = {
        "schema": REGISTRY_SCHEMA,
        "stage": "construction",
        "model_outcomes_read": False,
        "required_categories": list(REQUIRED_CATEGORIES),
        "source_spec_sha256": sha256_payload(spec),
        "sources": source_records,
        "categories": categories,
        "normalized_count": len(genes),
        "gene_ids": genes,
        "gene_ids_sha256": sha256_payload(genes),
    }
    core["content_sha256"] = sha256_payload(core)
    return core


def _audit_normalize_gene(value: object) -> str:
    """Auditor-local normalization, intentionally separate from the producer."""

    text = str(value).strip()
    before_dot = text.partition(".")[0]
    if re.fullmatch(r"ENSG\d+", before_dot) is None:
        raise ValueError(f"auditor rejected non-Ensembl gene ID {value!r}")
    return before_dot


def _audit_select(document: Any, pointer: str) -> list[Any]:
    """Independent exact-selector traversal used only by the audit."""

    frontier = [document]
    for token in pointer.split("."):
        next_frontier: list[Any] = []
        for node in frontier:
            if token == "*":
                if isinstance(node, list):
                    next_frontier.extend(node)
                elif isinstance(node, Mapping):
                    next_frontier.extend(node[key] for key in sorted(node))
                else:
                    raise ValueError(f"audit selector {pointer!r} expanded a scalar")
            elif isinstance(node, Mapping):
                if token not in node:
                    raise ValueError(f"audit selector {pointer!r} lacks {token!r}")
                next_frontier.append(node[token])
            elif isinstance(node, list) and token.isdecimal():
                index = int(token)
                if index >= len(node):
                    raise ValueError(f"audit selector {pointer!r} index is absent")
                next_frontier.append(node[index])
            else:
                raise ValueError(f"audit selector {pointer!r} cannot descend")
        frontier = next_frontier
    return frontier


def _audit_terminal_strings(values: Iterable[Any], *, selector: str) -> list[str]:
    output: list[str] = []
    for value in values:
        if isinstance(value, str):
            output.append(value)
        elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            output.extend(value)
        else:
            raise ValueError(f"audit selector {selector!r} has a non-gene terminal")
    if not output:
        raise ValueError(f"audit selector {selector!r} produced no genes")
    return output


def _audit_extract_gene_ids(
    blob: bytes, recipe: Mapping[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Auditor-owned JSON/JSONL/table parser; does not call producer extraction."""

    kind = str(recipe.get("format", ""))
    values_by_selector: dict[str, list[str]] = {}
    if kind in {"json", "jsonl"}:
        selectors = recipe.get("pointers")
        if not isinstance(selectors, list) or not selectors:
            raise ValueError("audit requires explicit non-empty pointers")
        if kind == "json":
            documents = [json.loads(blob.decode("utf-8"))]
        else:
            documents = [
                json.loads(line)
                for line in blob.decode("utf-8").splitlines()
                if line.strip()
            ]
            if not documents:
                raise ValueError("audit found an empty JSONL source")
        for selector_value in selectors:
            selector = str(selector_value)
            selected: list[Any] = []
            for document in documents:
                selected.extend(_audit_select(document, selector))
            values_by_selector[selector] = _audit_terminal_strings(
                selected, selector=selector
            )
    elif kind in {"csv", "tsv"}:
        columns = recipe.get("columns")
        if not isinstance(columns, list) or not columns:
            raise ValueError("audit requires explicit non-empty table columns")
        dialect_delimiter = "," if kind == "csv" else "\t"
        table = csv.DictReader(
            io.StringIO(blob.decode("utf-8")), delimiter=dialect_delimiter
        )
        rows = list(table)
        if table.fieldnames is None or not rows:
            raise ValueError("audit found an empty headerless table")
        for column_value in columns:
            column = str(column_value)
            if column not in table.fieldnames:
                raise ValueError(f"audit table lacks column {column!r}")
            values = [row[column] for row in rows if row.get(column, "").strip()]
            if not values:
                raise ValueError(f"audit table column {column!r} is empty")
            values_by_selector[column] = values
    else:
        raise ValueError("audit extraction format is unsupported")
    records: list[dict[str, Any]] = []
    all_genes: set[str] = set()
    for selector, raw in values_by_selector.items():
        normalized = sorted({_audit_normalize_gene(value) for value in raw})
        all_genes.update(normalized)
        records.append(
            {
                "selector": selector,
                "raw_count": len(raw),
                "normalized_count": len(normalized),
                "normalized_sha256": sha256_payload(normalized),
            }
        )
    return sorted(all_genes), records


def audit_registry(
    registry: Mapping[str, Any], spec: Mapping[str, Any], *, repo: str | Path
) -> dict[str, Any]:
    """Independent second derivation; never calls :func:`derive_registry`."""

    failures: list[str] = []
    try:
        sources = validate_source_spec(spec)
    except Exception as exc:
        return {
            "schema": AUDIT_SCHEMA,
            "stage": "construction",
            "status": "AUDIT_FAIL",
            "passed": False,
            "failures": [f"source_spec:{type(exc).__name__}:{exc}"],
        }
    union: set[str] = set()
    category_union: dict[str, set[str]] = {category: set() for category in REQUIRED_CATEGORIES}
    independently_read: list[dict[str, Any]] = []
    for source in sources:
        try:
            blob, provenance = read_source_blob(source, repo=repo)
            genes, extraction_rows = _audit_extract_gene_ids(blob, source["extraction"])
            if len(genes) != int(source["expected_normalized_count"]):
                raise ValueError("normalized source count differs from its pin")
        except Exception as exc:
            failures.append(f"source:{source['source_id']}:{type(exc).__name__}:{exc}")
            continue
        union.update(genes)
        category_union[str(source["category"])].update(genes)
        independently_read.append(
            {
                "source_id": str(source["source_id"]),
                "category": str(source["category"]),
                **provenance,
                "source_size": len(blob),
                "extraction": dict(source["extraction"]),
                "extraction_records": extraction_rows,
                "normalized_count": len(genes),
                "normalized_gene_ids_sha256": sha256_payload(genes),
                "normalized_gene_ids": genes,
            }
        )
    independent_genes = sorted(union)
    expected_categories = {
        category: {
            "n_sources": sum(row["category"] == category for row in independently_read),
            "normalized_count": len(category_union[category]),
            "normalized_gene_ids_sha256": sha256_payload(sorted(category_union[category])),
        }
        for category in REQUIRED_CATEGORIES
    }
    checks = {
        "schema": registry.get("schema") == REGISTRY_SCHEMA,
        "stage": registry.get("stage") == "construction",
        "outcome_firewall": registry.get("model_outcomes_read") is False,
        "required_categories": registry.get("required_categories") == list(REQUIRED_CATEGORIES),
        "source_spec_pin": registry.get("source_spec_sha256") == sha256_payload(spec),
        "all_sources_reread": len(independently_read) == len(sources),
        "source_records": registry.get("sources") == independently_read,
        "category_records": registry.get("categories") == expected_categories,
        "sorted_unique_gene_ids": registry.get("gene_ids") == independent_genes,
        "gene_count": registry.get("normalized_count") == len(independent_genes),
        "gene_hash": registry.get("gene_ids_sha256") == sha256_payload(independent_genes),
        "content_hash": registry.get("content_sha256")
        == sha256_payload({key: value for key, value in registry.items() if key != "content_sha256"}),
    }
    failures.extend(name for name, value in checks.items() if not value)
    passed = not failures
    return {
        "schema": AUDIT_SCHEMA,
        "stage": "construction",
        "status": "AUDIT_PASS" if passed else "AUDIT_FAIL",
        "passed": passed,
        "checks": checks,
        "failures": failures,
        "registry_content_sha256": registry.get("content_sha256"),
        "source_spec_sha256": sha256_payload(spec),
        "independent_gene_ids_sha256": sha256_payload(independent_genes),
        "independent_normalized_count": len(independent_genes),
    }


__all__ = [
    "AUDIT_SCHEMA",
    "REGISTRY_SCHEMA",
    "REQUIRED_CATEGORIES",
    "SOURCE_SPEC_SCHEMA",
    "audit_registry",
    "canonical_json",
    "derive_registry",
    "extract_gene_ids",
    "normalized_gene_id",
    "read_source_blob",
    "sha256_bytes",
    "sha256_payload",
    "validate_source_spec",
]
