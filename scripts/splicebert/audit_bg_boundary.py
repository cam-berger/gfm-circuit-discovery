#!/usr/bin/env python
"""Independently audit BG1 behavior or BG2 payload-recovery result artifacts.

The audit is model-free: it loads no checkpoint and imports no decision code from
either runner.  It verifies the complete manifest hash inventory and reconstructs
the claim-bearing statistics and public control assignments from the saved rows.
It is an artifact/statistical-consistency audit, not a replay of model forwards.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402


AUDIT_SCHEMA = "gfmcd.splicebert.boundary_grammar.audit.v1"
BG1_SCHEMA = "gfmcd.splicebert.boundary_grammar.behavior.v1"
BG2_SCHEMA = "gfmcd.splicebert.boundary_payload_recovery.v1"
PREFLIGHT_SCHEMA = "gfmcd.splicebert.boundary_grammar.preflight.v1"
SESOI = 0.05
ANALYSIS_SEED = 20260809
BG1_BOOTSTRAP_DRAWS = 4999
BG1_PERMUTATION_DRAWS = 9999
BG2_BOOTSTRAP_DRAWS = 2000
MIN_BG1_TARGET_GENES = 128
MIN_BG1_TARGET_GENES_PER_STRAND = 32
MIN_OUTER_TARGET_GENES = 80
MIN_OUTER_TARGET_GENES_PER_STRAND = 20
MIN_BG2_TARGET_GENES = 96
MIN_BG2_TARGET_GENES_PER_STRAND = 24
SIGNED_SHAM_SEEDS = tuple(2026080900 + index for index in range(19))
ATOL = 1e-8
RTOL = 1e-8

REQUIRED_BG1_CODE_INPUTS = (
    REPO / "studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md",
    REPO / "src/gfmcd/splicing/boundary_grammar.py",
    REPO / "scripts/splicebert/run_bg1_boundary_grammar.py",
    REPO / "src/gfmcd/splicing/coupling.py",
    REPO / "src/gfmcd/splicing/circuit.py",
    REPO / "src/gfmcd/splicing/p3_stats.py",
    REPO / "src/gfmcd/circuits/hf_bert.py",
)
REQUIRED_BG2_CODE_INPUTS = (
    REPO / "studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md",
    REPO / "src/gfmcd/splicing/boundary_grammar.py",
    REPO / "scripts/splicebert/run_bg2_boundary_payload_recovery.py",
    REPO / "scripts/splicebert/audit_bg_boundary.py",
    REPO / "src/gfmcd/splicing/coupling.py",
    REPO / "src/gfmcd/splicing/circuit.py",
    REPO / "src/gfmcd/splicing/payload.py",
    REPO / "src/gfmcd/circuits/hf_bert.py",
)

TARGET_PANEL = "bg1_target_m3_by_p1"
MINUS3_PANEL = "bg1_inner_m3"
MINUS4_PANEL = "bg1_diagnostic_m4"
PLUS1_PANEL = "bg1_inner_p1"
PLUS2_PANEL = "bg1_diagnostic_p2"

PANEL_CELLS = {
    TARGET_PANEL: tuple(
        f"m3_{minus3}__p1_{plus1}"
        for minus3 in ("A", "C", "G", "T")
        for plus1 in ("A", "G")
    ),
    MINUS3_PANEL: ("m3_A", "m3_T"),
    MINUS4_PANEL: ("m4_A", "m4_T"),
    PLUS1_PANEL: ("p1_A", "p1_G"),
    PLUS2_PANEL: ("p2_A", "p2_G"),
}
PANEL_ALLELES = {
    TARGET_PANEL: {
        cell: {
            -3: cell.removeprefix("m3_").split("__p1_")[0],
            1: cell.split("__p1_")[1],
        }
        for cell in PANEL_CELLS[TARGET_PANEL]
    },
    MINUS3_PANEL: {"m3_A": {-3: "A"}, "m3_T": {-3: "T"}},
    MINUS4_PANEL: {"m4_A": {-4: "A"}, "m4_T": {-4: "T"}},
    PLUS1_PANEL: {"p1_A": {1: "A"}, "p1_G": {1: "G"}},
    PLUS2_PANEL: {"p2_A": {2: "A"}, "p2_G": {2: "G"}},
}


def _target_weights(kind: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for cell in PANEL_CELLS[TARGET_PANEL]:
        minus3, plus1 = cell.removeprefix("m3_").split("__p1_")
        if kind == "minus3":
            weights[cell] = 0.5 if minus3 == "T" else -0.5 if minus3 == "A" else 0.0
        elif kind == "plus1":
            weights[cell] = 0.25 if plus1 == "G" else -0.25
        elif kind == "interaction":
            weights[cell] = {
                ("T", "G"): 1.0,
                ("A", "G"): -1.0,
                ("T", "A"): -1.0,
                ("A", "A"): 1.0,
            }.get((minus3, plus1), 0.0)
        else:  # pragma: no cover - declaration-time guard
            raise AssertionError(kind)
    return weights


TARGET_CONTRASTS = (
    ("m3_T_minus_A", _target_weights("minus3")),
    ("p1_G_minus_A", _target_weights("plus1")),
    ("m3_by_p1_interaction", _target_weights("interaction")),
)
LOCALIZATION = {
    "minus3_vs_minus4": (
        MINUS3_PANEL,
        {"m3_A": -1.0, "m3_T": 1.0},
        MINUS4_PANEL,
        {"m4_A": -1.0, "m4_T": 1.0},
        "m3_T_minus_A_single_minus_m4_T_minus_A",
        ANALYSIS_SEED + 20,
    ),
    "plus1_vs_plus2": (
        PLUS1_PANEL,
        {"p1_A": -1.0, "p1_G": 1.0},
        PLUS2_PANEL,
        {"p2_A": -1.0, "p2_G": 1.0},
        "p1_G_minus_A_single_minus_p2_G_minus_A",
        ANALYSIS_SEED + 30,
    ),
}

DIRECTED_PAIRS = (
    ("minus3_T_to_A__p1_A", "minus3_T_to_A", "p1_A", "m3_T__p1_A", "m3_A__p1_A"),
    ("minus3_T_to_A__p1_G", "minus3_T_to_A", "p1_G", "m3_T__p1_G", "m3_A__p1_G"),
    ("plus1_G_to_A__m3_A", "plus1_G_to_A", "m3_A", "m3_A__p1_G", "m3_A__p1_A"),
    ("plus1_G_to_A__m3_C", "plus1_G_to_A", "m3_C", "m3_C__p1_G", "m3_C__p1_A"),
    ("plus1_G_to_A__m3_G", "plus1_G_to_A", "m3_G", "m3_G__p1_G", "m3_G__p1_A"),
    ("plus1_G_to_A__m3_T", "plus1_G_to_A", "m3_T", "m3_T__p1_G", "m3_T__p1_A"),
)
TRANSITION_BACKGROUNDS = {
    "minus3_T_to_A": ("p1_A", "p1_G"),
    "plus1_G_to_A": ("m3_A", "m3_C", "m3_G", "m3_T"),
}


class AuditError(RuntimeError):
    """The supplied record lacks or contradicts required audit evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be a list")
    return value


def _finite(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric")
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} must be numeric") from exc
    _require(math.isfinite(output), f"{label} must be finite")
    return output


def _close(actual: Any, expected: Any, label: str) -> None:
    a, e = _finite(actual, label), _finite(expected, f"expected {label}")
    _require(math.isclose(a, e, rel_tol=RTOL, abs_tol=ATOL), f"{label} mismatch: {a} != {e}")


def _compare_subset(actual: Any, expected: Any, label: str) -> None:
    """Compare all expected claim-bearing fields, allowing unrelated diagnostics."""

    if isinstance(expected, Mapping):
        source = _mapping(actual, label)
        for key, value in expected.items():
            _require(key in source, f"{label}.{key} is missing")
            _compare_subset(source[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        source = _list(actual, label)
        _require(len(source) == len(expected), f"{label} length mismatch")
        for index, value in enumerate(expected):
            _compare_subset(source[index], value, f"{label}[{index}]")
    elif isinstance(expected, float):
        _close(actual, expected, label)
    else:
        _require(actual == expected, f"{label} mismatch: {actual!r} != {expected!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {label}: {path}") from exc
    return _mapping(value, label)


def _record_path(value: Any) -> Path:
    _require(isinstance(value, str) and value, "manifest record path must be non-empty")
    path = Path(value)
    return (path if path.is_absolute() else REPO / path).resolve()


def _verify_manifest(
    manifest: Mapping[str, Any], manifest_path: Path, result_path: Path
) -> dict[str, dict[Path, str]]:
    records: dict[str, dict[Path, str]] = {"inputs": {}, "outputs": {}}
    for family in ("inputs", "outputs"):
        items = _list(manifest.get(family), f"manifest.{family}")
        _require(items, f"manifest.{family} cannot be empty")
        for index, raw in enumerate(items):
            item = _mapping(raw, f"manifest.{family}[{index}]")
            path = _record_path(item.get("path"))
            _require(path not in records[family], f"duplicate manifest {family} path: {path}")
            _require(item.get("missing") is not True, f"manifest declares missing {family[:-1]}: {path}")
            _require(path.is_file(), f"manifest {family[:-1]} does not exist: {path}")
            expected_hash = item.get("sha256")
            _require(
                isinstance(expected_hash, str)
                and len(expected_hash) == 64
                and all(char in "0123456789abcdef" for char in expected_hash),
                f"manifest {family[:-1]} hash is malformed: {path}",
            )
            actual_hash = _sha256(path)
            _require(actual_hash == expected_hash, f"manifest {family[:-1]} hash mismatch: {path}")
            size = item.get("size_bytes")
            _require(
                isinstance(size, int) and not isinstance(size, bool) and size == path.stat().st_size,
                f"manifest {family[:-1]} size mismatch: {path}",
            )
            records[family][path] = actual_hash
    resolved_result = result_path.resolve()
    _require(resolved_result in records["outputs"], "result is not a hash-verified manifest output")
    _require(manifest_path.resolve() != resolved_result, "manifest and result paths must differ")
    return records


def _bound_input(
    records: Mapping[str, Mapping[Path, str]], raw_path: Any, expected_hash: Any, label: str
) -> Path:
    path = _record_path(raw_path)
    _require(path in records["inputs"], f"{label} is not a manifest input: {path}")
    _require(records["inputs"][path] == expected_hash, f"{label} embedded hash mismatch")
    return path


def _require_manifest_inventory(
    records: Mapping[str, Mapping[Path, str]],
    *,
    code_paths: Sequence[Path],
    expected_hashes: Mapping[str, Any],
) -> None:
    inputs = records["inputs"]
    for path in code_paths:
        _require(path.resolve() in inputs, f"required code/protocol input is absent: {path}")
    available_hashes = set(inputs.values())
    for label, raw_hash in expected_hashes.items():
        _require(
            isinstance(raw_hash, str)
            and len(raw_hash) == 64
            and raw_hash in available_hashes,
            f"required manifest input hash is absent: {label}",
        )


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _biological_index(acceptor_pos: int, position: int) -> int:
    _require(position != 0, "biological position zero does not exist")
    return acceptor_pos + (position if position < 0 else position - 1)


def _motif_inventory(sequence: str) -> tuple[tuple[int, str], ...]:
    motifs = {"AG", "GT", "GC"}
    return tuple(
        (index, sequence[index:index + 2])
        for index in range(len(sequence) - 1)
        if sequence[index:index + 2] in motifs
    )


def _audit_constructs(
    raw_constructs: Any,
    groups: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Rebuild every saved cell from its native sequence and verify motif invariance."""

    constructs: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, raw in enumerate(_list(raw_constructs, "result.constructs")):
        item = _mapping(raw, f"result.constructs[{index}]")
        panel = item.get("panel_id")
        example = item.get("example_id")
        _require(panel in PANEL_CELLS, f"unknown construct panel: {panel!r}")
        _require(isinstance(example, str) and example, "construct example_id is missing")
        key = (str(panel), example)
        _require(key not in constructs, f"duplicate construct declaration: {panel}/{example}")
        _require(example in groups[str(panel)], f"construct has no matching cells: {panel}/{example}")
        sequence = item.get("native_sequence")
        _require(
            isinstance(sequence, str)
            and len(sequence) == 1024
            and not (set(sequence) - set("ACGT")),
            f"native sequence is not canonical 1,024-nt DNA: {panel}/{example}",
        )
        acceptor = int(item.get("next_acceptor_pos"))
        donor = int(item.get("donor_pos"))
        _require(4 <= acceptor < len(sequence) - 4, f"acceptor coordinate is invalid: {panel}/{example}")
        _require(0 <= donor < len(sequence) - 1, f"donor coordinate is invalid: {panel}/{example}")
        _require(sequence[acceptor - 2:acceptor] == "AG", f"native core AG is absent: {panel}/{example}")
        _require(sequence[donor:donor + 2] == "GT", f"native donor GT is absent: {panel}/{example}")
        _require(60 <= acceptor - donor <= 150, f"partner span is outside support: {panel}/{example}")
        _require(item.get("partner_span") == acceptor - donor, f"construct partner span mismatch: {panel}/{example}")
        _require(item.get("strand") in ("+", "-"), f"construct strand is invalid: {panel}/{example}")
        _require(
            item.get("native_sequence_sha256") == _sequence_hash(sequence),
            f"native sequence hash mismatch: {panel}/{example}",
        )

        cell_hashes = _mapping(
            item.get("cell_sequence_sha256"), f"{panel}/{example}.cell_sequence_sha256"
        )
        _require(set(cell_hashes) == set(PANEL_CELLS[str(panel)]), f"construct cell hashes are incomplete: {panel}/{example}")
        native_inventory = _motif_inventory(sequence)
        reconstructed_hashes: dict[str, str] = {}
        for cell in PANEL_CELLS[str(panel)]:
            edited = list(sequence)
            for position, base in PANEL_ALLELES[str(panel)][cell].items():
                coordinate = _biological_index(acceptor, position)
                _require(0 <= coordinate < len(sequence), f"edited coordinate is invalid: {panel}/{example}/{cell}")
                edited[coordinate] = base
            edited_sequence = "".join(edited)
            _require(edited_sequence[acceptor - 2:acceptor] == "AG", f"cell core AG changed: {panel}/{example}/{cell}")
            _require(
                _motif_inventory(edited_sequence) == native_inventory,
                f"positioned AG/GT/GC inventory changed: {panel}/{example}/{cell}",
            )
            reconstructed_hashes[cell] = _sequence_hash(edited_sequence)
            _require(
                cell_hashes.get(cell) == reconstructed_hashes[cell],
                f"cell sequence hash mismatch: {panel}/{example}/{cell}",
            )
        expected_construct = _canonical_hash({
            "panel_id": panel,
            "cells": [
                {"cell_id": cell, "sequence_sha256": reconstructed_hashes[cell]}
                for cell in sorted(reconstructed_hashes)
            ],
        })
        _require(item.get("construct_sha256") == expected_construct, f"construct hash mismatch: {panel}/{example}")

        rows = groups[str(panel)][example]
        for cell, row in rows.items():
            _require(row.get("sequence_sha256") == reconstructed_hashes[cell], f"row sequence hash differs from reconstruction: {panel}/{example}/{cell}")
            _require(row.get("construct_sha256") == expected_construct, f"row construct hash differs: {panel}/{example}/{cell}")
            for field, expected in (
                ("gene_id", item.get("gene_id")),
                ("chrom", item.get("chrom")),
                ("strand", item.get("strand")),
                ("donor_pos", donor),
                ("next_acceptor_pos", acceptor),
                ("partner_span", acceptor - donor),
            ):
                _require(row.get(field) == expected, f"row {field} differs from construct: {panel}/{example}/{cell}")
        constructs[key] = item
    expected_keys = {
        (panel, example) for panel, by_example in groups.items() for example in by_example
    }
    _require(set(constructs) == expected_keys, "construct declarations do not match result panels")
    return constructs


def _cohort_hash_from_constructs(
    membership: Mapping[str, Any],
    panel: str,
    constructs: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    rows = []
    for example, gene, strand in zip(
        membership["ids"], membership["genes"], membership["strands"]
    ):
        item = constructs[(panel, str(example))]
        _require(item.get("gene_id") == gene and item.get("strand") == strand, "construct membership identity mismatch")
        rows.append({
            "example_id": str(example),
            "gene_id": str(gene),
            "chrom": str(item.get("chrom")),
            "strand": str(strand),
            "donor_pos": item.get("donor_pos"),
            "next_acceptor_pos": item.get("next_acceptor_pos"),
            "sequence_sha256": item.get("native_sequence_sha256"),
        })
    return _canonical_hash(rows)


def _base_margins(values: Any, label: str, *, mapping: bool) -> tuple[float, float]:
    if mapping:
        logits = _mapping(values, label)
        _require(set(logits) == {"A", "C", "G", "T"}, f"{label} must contain exactly A,C,G,T")
        array = [_finite(logits[base], f"{label}.{base}") for base in ("A", "C", "G", "T")]
    else:
        raw = _list(values, label)
        _require(len(raw) == 4, f"{label} must have A,C,G,T length four")
        array = [_finite(value, f"{label}[{index}]") for index, value in enumerate(raw)]
    return array[2] - sum((array[0], array[1], array[3])) / 3.0, array[2] - max(
        array[0], array[1], array[3]
    )


def _max_family_p(observed: np.ndarray, null: np.ndarray, *, two_sided: bool) -> np.ndarray:
    null_max = np.max(np.abs(null) if two_sided else null, axis=1)
    targets = np.abs(observed) if two_sided else observed
    return np.asarray([
        (1 + np.count_nonzero(null_max >= value)) / (len(null_max) + 1)
        for value in targets
    ])


def _simultaneous(
    matrix: np.ndarray,
    names: Sequence[str],
    *,
    seed: int,
    n_bootstrap: int,
    n_permutation: int,
    two_sided: bool,
) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=float)
    _require(
        values.ndim == 2
        and values.shape[1] == len(names)
        and values.shape[0] >= 3
        and np.isfinite(values).all(),
        "BG1 inference matrix must contain at least three finite gene rows",
    )
    n_genes = values.shape[0]
    means = values.mean(axis=0)
    standard_errors = values.std(axis=0, ddof=1) / math.sqrt(n_genes)
    _require(np.all(standard_errors > 0), "BG1 endpoints require nonzero between-gene variance")
    rng = np.random.default_rng(seed)
    max_t = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = values[rng.integers(0, n_genes, size=n_genes)]
        sample_se = sample.std(axis=0, ddof=1) / math.sqrt(n_genes)
        max_t[index] = (
            np.max(np.abs((sample.mean(axis=0) - means) / sample_se))
            if np.all(sample_se > 0)
            else np.nan
        )
    max_t = max_t[np.isfinite(max_t)]
    _require(len(max_t) >= int(0.95 * n_bootstrap), "too many degenerate BG1 bootstrap draws")
    critical = float(np.quantile(max_t, 0.95, method="higher"))
    rng = np.random.default_rng(seed + 1)
    signs = rng.choice((-1.0, 1.0), size=(n_permutation, n_genes, 1))
    null = (signs * values[None, :, :]).mean(axis=1)
    pvalues = _max_family_p(means, null, two_sided=two_sided)
    return {
        "names": list(names),
        "means": means.tolist(),
        "standard_errors": standard_errors.tolist(),
        "simultaneous_95_ci": [
            [float(mean - critical * se), float(mean + critical * se)]
            for mean, se in zip(means, standard_errors)
        ],
        "max_family_add_one_p": pvalues.tolist(),
        "critical_value": critical,
        "n_genes": n_genes,
        "n_bootstrap": n_bootstrap,
        "n_permutation": n_permutation,
        "p_alternative": "two-sided" if two_sided else "greater",
    }


def _descriptive(matrix: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=float)
    _require(
        values.ndim == 2
        and values.shape[1] == len(names)
        and values.shape[0] >= 1
        and np.isfinite(values).all(),
        "descriptive matrix must be finite [genes,endpoints]",
    )
    return {
        "names": list(names),
        "means": values.mean(axis=0).tolist(),
        "standard_deviations": values.std(axis=0, ddof=1).tolist(),
        "standard_errors": (values.std(axis=0, ddof=1) / math.sqrt(len(values))).tolist(),
        "n_genes": len(values),
        "claim_bearing": False,
    }


def _holm(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def _bg1_groups(rows: Any) -> dict[str, dict[str, dict[str, Mapping[str, Any]]]]:
    grouped: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for index, raw in enumerate(_list(rows, "result.cells")):
        row = _mapping(raw, f"result.cells[{index}]")
        panel = row.get("panel_id")
        example = row.get("example_id")
        cell = row.get("cell_id")
        gene = row.get("gene_id")
        _require(panel in PANEL_CELLS, f"unknown BG1 panel: {panel!r}")
        _require(isinstance(example, str) and example, "BG1 row example_id is missing")
        _require(isinstance(gene, str) and gene, "BG1 row gene_id is missing")
        _require(cell in PANEL_CELLS[panel], f"unknown {panel} cell: {cell!r}")
        _require(cell not in grouped[panel][example], f"duplicate BG1 row: {panel}/{example}/{cell}")
        linear, maximum = _base_margins(row.get("base_logits"), f"{panel}/{example}/{cell}.base_logits", mapping=True)
        _close(row.get("linear_margin"), linear, f"{panel}/{example}/{cell}.linear_margin")
        _close(row.get("max_margin"), maximum, f"{panel}/{example}/{cell}.max_margin")
        grouped[panel][example][cell] = row

    _require(set(grouped) == set(PANEL_CELLS), "BG1 rows do not contain all and only frozen panels")
    for panel, by_example in grouped.items():
        _require(by_example, f"BG1 panel {panel} has no examples")
        genes: set[str] = set()
        for example, by_cell in by_example.items():
            _require(set(by_cell) == set(PANEL_CELLS[panel]), f"incomplete BG1 panel: {panel}/{example}")
            row_genes = {str(row["gene_id"]) for row in by_cell.values()}
            _require(len(row_genes) == 1, f"BG1 example crosses genes: {panel}/{example}")
            gene = next(iter(row_genes))
            _require(gene not in genes, f"BG1 panel has more than one example for gene {gene}")
            genes.add(gene)
            constructs = {row.get("construct_sha256") for row in by_cell.values()}
            _require(len(constructs) == 1, f"BG1 construct hash differs within {panel}/{example}")
            cell_hashes = []
            for cell, row in by_cell.items():
                sequence_hash = row.get("sequence_sha256")
                _require(
                    isinstance(sequence_hash, str) and len(sequence_hash) == 64,
                    f"BG1 sequence hash missing: {panel}/{example}/{cell}",
                )
                cell_hashes.append({"cell_id": cell, "sequence_sha256": sequence_hash})
            expected_construct = _canonical_hash({
                "panel_id": panel,
                "cells": sorted(cell_hashes, key=lambda item: item["cell_id"]),
            })
            _require(next(iter(constructs)) == expected_construct, f"BG1 construct hash mismatch: {panel}/{example}")
    _require(
        set(grouped[MINUS3_PANEL]) == set(grouped[MINUS4_PANEL]),
        "BG1 minus3/minus4 diagnostic examples are not paired",
    )
    _require(
        set(grouped[PLUS1_PANEL]) == set(grouped[PLUS2_PANEL]),
        "BG1 plus1/plus2 diagnostic examples are not paired",
    )
    return grouped


def _verify_bg1_membership(
    result: Mapping[str, Any],
    preflight: Mapping[str, Any],
    groups: Mapping[str, Any],
    constructs: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    cohort = _mapping(result.get("cohort"), "result.cohort")
    memberships = _mapping(cohort.get("memberships"), "result.cohort.memberships")
    split = cohort.get("split")
    _require(split in ("val", "test"), "BG1 split must be val or test")
    role_panels = {
        "target_behavior": TARGET_PANEL,
        "minus3_minus4_diagnostic": MINUS3_PANEL,
        "plus1_plus2_diagnostic": PLUS1_PANEL,
    }
    frozen = _mapping(preflight.get("target_cohorts"), "preflight.target_cohorts")
    frozen_split = _mapping(frozen.get(split), f"preflight.target_cohorts.{split}")
    counts = _mapping(cohort.get("counts"), "result.cohort.counts")
    for role, panel in role_panels.items():
        membership = _mapping(memberships.get(role), f"result membership {role}")
        _require(membership == frozen_split.get(role), f"BG1 {role} membership differs from preflight")
        ids = _list(membership.get("ids"), f"{role}.ids")
        genes = _list(membership.get("genes"), f"{role}.genes")
        strands = _list(membership.get("strands"), f"{role}.strands")
        _require(len(ids) == len(genes) == len(strands), f"BG1 {role} membership lengths differ")
        _require(len(ids) == len(set(genes)), f"BG1 {role} is not one-example-per-gene")
        _require(set(strands).issubset({"+", "-"}), f"BG1 {role} has an invalid strand")
        expected = {str(example): (str(gene), str(strand)) for example, gene, strand in zip(ids, genes, strands)}
        _require(len(expected) == len(ids), f"BG1 {role} has duplicate example IDs")
        observed = {
            example: (
                str(next(iter(by_cell.values()))["gene_id"]),
                str(next(iter(by_cell.values()))["strand"]),
            )
            for example, by_cell in groups[panel].items()
        }
        _require(observed == expected, f"BG1 {role} rows differ from declared membership")
        _require(counts.get(role) == len(ids), f"BG1 {role} count mismatch")
        _require(
            membership.get("sha256") == _cohort_hash_from_constructs(
                membership, panel, constructs
            ),
            f"BG1 {role} sequence-aware cohort hash mismatch",
        )
        if role == "target_behavior":
            _require(len(ids) >= MIN_BG1_TARGET_GENES, "BG1 target gene supply is below 128")
            strand_counts = {strand: strands.count(strand) for strand in ("+", "-")}
            _require(
                min(strand_counts.values()) >= MIN_BG1_TARGET_GENES_PER_STRAND,
                "BG1 target per-strand supply is below 32",
            )


def _gene_matrix(
    by_example: Mapping[str, Mapping[str, Mapping[str, Any]]],
    specs: Sequence[tuple[str, Mapping[str, float]]],
    metric: str,
) -> tuple[list[str], np.ndarray]:
    by_gene: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    for cells in by_example.values():
        gene = str(next(iter(cells.values()))["gene_id"])
        _require(gene not in by_gene, f"duplicate gene in BG1 panel: {gene}")
        by_gene[gene] = cells
    genes = sorted(by_gene)
    matrix = np.asarray([
        [sum(float(cells[cell][metric]) * weight for cell, weight in weights.items()) for _name, weights in specs]
        for gene in genes
        for cells in (by_gene[gene],)
    ], dtype=float)
    return genes, matrix


def _audit_bg1(
    result: Mapping[str, Any], records: Mapping[str, Mapping[Path, str]]
) -> dict[str, Any]:
    preflight_path = _bound_input(
        records, result.get("preflight_path"), result.get("preflight_sha256"), "BG1 preflight"
    )
    preflight = _json(preflight_path, "BG1 preflight")
    _require(preflight.get("schema") == PREFLIGHT_SCHEMA, "wrong BG1 preflight schema")
    _require(preflight.get("PASS") is True and preflight.get("smoke") is False, "BG1 preflight did not pass non-smoke")
    checks = _mapping(preflight.get("checks"), "preflight.checks")
    _require(
        checks and all(
            isinstance(check, Mapping) and check.get("PASS") is True
            for check in checks.values()
        ),
        "BG1 preflight contains a failed or malformed check",
    )
    config = _mapping(result.get("config"), "result.config")
    _require(config == preflight.get("config"), "BG1 result config differs from preflight")
    model_hashes = _mapping(config.get("model_file_sha256"), "BG1 model hashes")
    _require_manifest_inventory(
        records,
        code_paths=REQUIRED_BG1_CODE_INPUTS,
        expected_hashes={
            "GTF": config.get("gtf_sha256"),
            "FASTA": config.get("fasta_sha256"),
            **{f"model:{name}": digest for name, digest in model_hashes.items()},
        },
    )
    cohort = _mapping(result.get("cohort"), "result.cohort")
    split = cohort.get("split")
    expected_status = (
        "VERIFICATION" if split == "val" else "HYPOTHESIS_SPECIFIC_REPLICATION"
    )
    _require(result.get("status") == expected_status, "BG1 split/status label mismatch")
    if split == "test":
        _require(
            preflight.get("test_replication_enabled") is True,
            "BG1 test result was not prospectively enabled",
        )
    analysis = _mapping(config.get("analysis"), "result.config.analysis")
    _compare_subset(
        analysis,
        {
            "seed": ANALYSIS_SEED,
            "bg1_bootstrap_draws": BG1_BOOTSTRAP_DRAWS,
            "bg1_permutation_draws": BG1_PERMUTATION_DRAWS,
            "localization_threshold": SESOI,
            "localization_multiplicity": "Holm across enabled arms",
        },
        "result.config.analysis",
    )
    _close(config.get("sesoi"), SESOI, "result.config.sesoi")
    groups = _bg1_groups(result.get("cells"))
    constructs = _audit_constructs(result.get("constructs"), groups)
    _verify_bg1_membership(result, preflight, groups, constructs)

    primary_specs = TARGET_CONTRASTS[:2]
    _genes, primary_linear = _gene_matrix(groups[TARGET_PANEL], primary_specs, "linear_margin")
    _genes, primary_max = _gene_matrix(groups[TARGET_PANEL], primary_specs, "max_margin")
    _genes, interaction_linear = _gene_matrix(groups[TARGET_PANEL], TARGET_CONTRASTS[2:], "linear_margin")
    _genes, interaction_max = _gene_matrix(groups[TARGET_PANEL], TARGET_CONTRASTS[2:], "max_margin")
    target_expected = {
        "inference_linear": _simultaneous(
            primary_linear, [item[0] for item in primary_specs], seed=ANALYSIS_SEED,
            n_bootstrap=BG1_BOOTSTRAP_DRAWS, n_permutation=BG1_PERMUTATION_DRAWS,
            two_sided=False,
        ),
        "descriptive_max_bridge": _descriptive(
            primary_max, [item[0] for item in primary_specs]
        ),
        "interaction_linear_secondary": _simultaneous(
            interaction_linear, [TARGET_CONTRASTS[2][0]], seed=ANALYSIS_SEED + 11,
            n_bootstrap=BG1_BOOTSTRAP_DRAWS, n_permutation=BG1_PERMUTATION_DRAWS,
            two_sided=True,
        ),
        "descriptive_interaction_max": _descriptive(
            interaction_max, [TARGET_CONTRASTS[2][0]]
        ),
    }
    target = _mapping(result.get("target"), "result.target")
    _compare_subset(
        target,
        {
            "sesoi": SESOI,
            "primary_ids": [item[0] for item in primary_specs],
            "interaction_id": TARGET_CONTRASTS[2][0],
            **target_expected,
        },
        "result.target",
    )
    primary = target_expected["inference_linear"]
    intervals, pvalues = primary["simultaneous_95_ci"], primary["max_family_add_one_p"]
    if all(intervals[index][0] > SESOI and pvalues[index] <= 0.05 for index in range(2)):
        target_verdict = "SUPPORTED_VERIFICATION"
    elif any(intervals[index][1] <= SESOI for index in range(2)):
        target_verdict = "NOT_SUPPORTED"
    else:
        target_verdict = "INCONCLUSIVE"
    _require(target.get("verdict") == target_verdict, "BG1 target verdict mismatch")

    localization = _mapping(result.get("localization"), "result.localization")
    outer_power = _mapping(
        _mapping(preflight.get("diagnostics"), "preflight.diagnostics").get("outer_power"),
        "preflight.diagnostics.outer_power",
    )
    expected_localization: dict[str, dict[str, Any]] = {}
    enabled: list[str] = []
    for name, (inner_panel, inner_weights, outer_panel, outer_weights, contrast_name, seed) in LOCALIZATION.items():
        genes_i, inner_linear = _gene_matrix(groups[inner_panel], (("inner", inner_weights),), "linear_margin")
        genes_o, outer_linear = _gene_matrix(groups[outer_panel], (("outer", outer_weights),), "linear_margin")
        _require(genes_i == genes_o, f"BG1 {name} genes are not paired")
        _unused, inner_max = _gene_matrix(groups[inner_panel], (("inner", inner_weights),), "max_margin")
        _unused, outer_max = _gene_matrix(groups[outer_panel], (("outer", outer_weights),), "max_margin")
        inference = _simultaneous(
            inner_linear - outer_linear, [contrast_name], seed=seed,
            n_bootstrap=BG1_BOOTSTRAP_DRAWS, n_permutation=BG1_PERMUTATION_DRAWS,
            two_sided=False,
        )
        power = _mapping(outer_power.get(name), f"preflight outer power {name}")
        claim_bearing = power.get("claim_bearing")
        _require(isinstance(claim_bearing, bool), f"preflight {name} claim-bearing flag is missing")
        target_n = int(power.get("target_n_genes", -1))
        target_strands = _mapping(
            power.get("target_strands"), f"preflight {name}.target_strands"
        )
        projected_mde = power.get("projected_target_mde80")
        recomputed_claim_bearing = bool(
            target_n >= MIN_OUTER_TARGET_GENES
            and min(int(target_strands.get(strand, -1)) for strand in ("+", "-"))
            >= MIN_OUTER_TARGET_GENES_PER_STRAND
            and projected_mde is not None
            and _finite(projected_mde, f"preflight {name}.projected_target_mde80") <= SESOI
        )
        _require(
            claim_bearing == recomputed_claim_bearing,
            f"preflight {name} claim-bearing supply/MDE flag is inconsistent",
        )
        _require(target_n == len(genes_i), f"preflight {name} target count differs from rows")
        observed_strands = {
            strand: sum(
                str(next(iter(rows.values()))["strand"]) == strand
                for rows in groups[inner_panel].values()
            )
            for strand in ("+", "-")
        }
        _require(
            {strand: int(target_strands.get(strand, -1)) for strand in ("+", "-")}
            == observed_strands,
            f"preflight {name} target strand counts differ from rows",
        )
        if claim_bearing:
            enabled.append(name)
        expected_localization[name] = {
            "inference_linear": inference,
            "mean_max_margin_attenuation": float((inner_max - outer_max).mean()),
            "n_genes": len(genes_i),
            "claim_bearing_from_preflight": claim_bearing,
        }
    adjusted = _holm([
        expected_localization[name]["inference_linear"]["max_family_add_one_p"][0]
        for name in enabled
    ]) if enabled else []
    adjusted_by_name = dict(zip(enabled, adjusted))
    for name, expected in expected_localization.items():
        expected["holm_p_across_enabled_arms"] = adjusted_by_name.get(name)
        lower = expected["inference_linear"]["simultaneous_95_ci"][0][0]
        if name not in adjusted_by_name:
            expected["verdict"] = "PRESPECIFIED_DIAGNOSTIC_ONLY"
        elif lower > SESOI and adjusted_by_name[name] <= 0.05:
            expected["verdict"] = "MATERIALLY_ATTENUATED_OUTSIDE_BOUNDARY"
        else:
            expected["verdict"] = "INCONCLUSIVE"
        _compare_subset(localization.get(name), expected, f"result.localization.{name}")
    if len(enabled) == 2 and all(
        expected_localization[name]["verdict"] == "MATERIALLY_ATTENUATED_OUTSIDE_BOUNDARY"
        for name in enabled
    ):
        joint = "BOTH_FLANKS_MATERIALLY_ATTENUATED"
    elif not enabled:
        joint = "PRESPECIFIED_DIAGNOSTIC_ONLY"
    else:
        joint = "JOINT_LOCALIZATION_INCONCLUSIVE"
    _require(localization.get("joint_verdict") == joint, "BG1 joint localization verdict mismatch")
    return {
        "checks": {
            "panel_completeness": True,
            "native_sequence_edits_and_motif_inventory": True,
            "base_logit_margins": True,
            "target_contrasts_and_inference": True,
            "target_verdict": True,
            "localization_holm_and_verdict": True,
        },
        "recomputed": {
            "target_verdict": target_verdict,
            "target_linear_means": primary["means"],
            "localization_joint_verdict": joint,
            "localization_holm_p": {
                name: expected_localization[name]["holm_p_across_enabled_arms"]
                for name in expected_localization
            },
            "n_cells": len(result["cells"]),
        },
    }


def _transition_values(
    cells: Sequence[Mapping[str, Any]], value_fn: Callable[[Mapping[str, Any]], float]
) -> dict[str, tuple[np.ndarray, list[str]]]:
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for cell in cells:
        grouped[(str(cell["transition"]), str(cell["gene_id"]))].append(
            (str(cell["background"]), _finite(value_fn(cell), "BG2 aggregate value"))
        )
    output: dict[str, tuple[np.ndarray, list[str]]] = {}
    for transition, backgrounds in TRANSITION_BACKGROUNDS.items():
        values: list[float] = []
        genes: list[str] = []
        for (item_transition, gene), rows in grouped.items():
            if item_transition != transition:
                continue
            _require(
                len(rows) == len(backgrounds) and {background for background, _ in rows} == set(backgrounds),
                f"BG2 gene {gene} lacks a unique complete {transition} background average",
            )
            values.append(float(np.mean([value for _background, value in rows])))
            genes.append(gene)
        _require(values, f"BG2 transition has no rows: {transition}")
        output[transition] = (np.asarray(values), genes)
    return output


def _mean_ci(values: np.ndarray, genes: Sequence[str], *, seed: int) -> dict[str, Any]:
    by_gene: dict[str, list[float]] = defaultdict(list)
    for value, gene in zip(values, genes):
        if np.isfinite(value):
            by_gene[gene].append(float(value))
    ordered = list(by_gene)
    _require(ordered, "BG2 gene bootstrap has no finite values")
    rng = np.random.default_rng(seed)
    draws = np.empty(BG2_BOOTSTRAP_DRAWS)
    for index in range(BG2_BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(ordered), size=len(ordered))
        draws[index] = np.mean([
            value for gene_index in sampled for value in by_gene[ordered[gene_index]]
        ])
    return {
        "mean": float(np.mean(values)),
        "lower_95_one_sided": float(np.quantile(draws, 0.05)),
        "gene_bootstrap_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "n_genes": len(set(genes)),
        "n_bootstrap": BG2_BOOTSTRAP_DRAWS,
        "seed": seed,
    }


def _ratio_ci(numerator: np.ndarray, denominator: np.ndarray, genes: Sequence[str], seed: int) -> list[float]:
    by_gene: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for a, b, gene in zip(numerator, denominator, genes):
        by_gene[gene].append((float(a), float(b)))
    ordered = list(by_gene)
    rng = np.random.default_rng(seed)
    draws = np.empty(BG2_BOOTSTRAP_DRAWS)
    for index in range(BG2_BOOTSTRAP_DRAWS):
        sampled = rng.choice(len(ordered), len(ordered), replace=True)
        pairs = [pair for gene_index in sampled for pair in by_gene[ordered[gene_index]]]
        draws[index] = sum(pair[0] for pair in pairs) / max(sum(pair[1] for pair in pairs), 1e-9)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _audit_bg2_cells(result: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], dict[str, float]]:
    cells = [_mapping(row, f"result.cells[{index}]") for index, row in enumerate(_list(result.get("cells"), "result.cells"))]
    _require(cells, "BG2 cells cannot be empty")
    expected_pairs = {row[0]: row for row in DIRECTED_PAIRS}
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    by_example_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    cross_errors: list[float] = []
    signed_errors: list[float] = []
    for index, cell in enumerate(cells):
        example = cell.get("example_id")
        gene = cell.get("gene_id")
        pair_id = cell.get("pair_id")
        _require(isinstance(example, str) and example and isinstance(gene, str) and gene, f"BG2 cell {index} identity missing")
        _require(pair_id in expected_pairs, f"unknown BG2 directed pair: {pair_id!r}")
        _require(pair_id not in grouped[example], f"duplicate BG2 cell {example}/{pair_id}")
        pair = expected_pairs[pair_id]
        _compare_subset(
            cell,
            {
                "pair_id": pair[0], "transition": pair[1], "background": pair[2],
                "reference_cell": pair[3], "edit_cell": pair[4],
                "base_logit_order": ["A", "C", "G", "T"],
            },
            f"result.cells[{index}]",
        )
        span = int(cell.get("partner_span"))
        expected_span_bin = "060_104" if 60 <= span <= 104 else "105_150" if 105 <= span <= 150 else None
        _require(expected_span_bin is not None, f"BG2 partner span is outside support: {example}/{pair_id}")
        _require(cell.get("span_bin") == expected_span_bin, f"BG2 span bin mismatch: {example}/{pair_id}")
        for prefix, owner in (
            ("reference", cell), ("edit", cell), ("true", _mapping(cell.get("true"), "BG2 true")),
            ("cross_gene", _mapping(cell.get("cross_gene"), "BG2 cross_gene")),
        ):
            logits_key = f"{prefix}_base_logits_ACGT" if prefix in ("reference", "edit") else "base_logits_ACGT"
            margin_prefix = f"{prefix}_" if prefix in ("reference", "edit") else ""
            linear, maximum = _base_margins(owner.get(logits_key), f"{example}/{pair_id}/{prefix}.base_logits", mapping=False)
            _close(owner.get(f"{margin_prefix}linear_margin"), linear, f"{example}/{pair_id}/{prefix}.linear_margin")
            _close(owner.get(f"{margin_prefix}max_margin"), maximum, f"{example}/{pair_id}/{prefix}.max_margin")
        true = _mapping(cell["true"], "BG2 true")
        cross = _mapping(cell["cross_gene"], "BG2 cross_gene")
        _close(true.get("linear_lift"), float(true["linear_margin"]) - float(cell["edit_linear_margin"]), f"{example}/{pair_id}/true.linear_lift")
        _close(true.get("max_lift"), float(true["max_margin"]) - float(cell["edit_max_margin"]), f"{example}/{pair_id}/true.max_lift")
        _close(cross.get("linear_lift"), float(cross["linear_margin"]) - float(cell["edit_linear_margin"]), f"{example}/{pair_id}/cross.linear_lift")
        _close(cross.get("max_lift"), float(cross["max_margin"]) - float(cell["edit_max_margin"]), f"{example}/{pair_id}/cross.max_lift")
        true_norm = _finite(cell.get("true_head_delta_norm"), f"{example}/{pair_id}/true norm")
        _require(true_norm > 1e-12, f"BG2 true delta is zero: {example}/{pair_id}")
        for owner, label, errors in ((cross, "cross", cross_errors),):
            scale = _finite(owner.get("norm_scale"), f"{example}/{pair_id}/{label}.norm_scale")
            norm = _finite(owner.get("matched_head_delta_norm"), f"{example}/{pair_id}/{label}.norm")
            _require(scale > 0 and norm > 0, f"BG2 {label} norm scale/dose must be positive")
            error = abs(norm - true_norm) / true_norm
            _close(owner.get("norm_relative_error"), error, f"{example}/{pair_id}/{label}.norm_relative_error")
            _require(error <= 1e-3, f"BG2 {label} norm error exceeds 1e-3")
            errors.append(error)
        signed = _list(cell.get("signed_coordinate_shams"), f"{example}/{pair_id}.signed shams")
        _require([item.get("seed") for item in signed if isinstance(item, Mapping)] == list(SIGNED_SHAM_SEEDS), f"BG2 signed seed family mismatch: {example}/{pair_id}")
        for sham_index, raw_sham in enumerate(signed):
            sham = _mapping(raw_sham, f"{example}/{pair_id}.signed[{sham_index}]")
            linear, maximum = _base_margins(sham.get("base_logits_ACGT"), f"{example}/{pair_id}/signed[{sham_index}].base_logits", mapping=False)
            _close(sham.get("linear_margin"), linear, f"{example}/{pair_id}/signed[{sham_index}].linear_margin")
            _close(sham.get("max_margin"), maximum, f"{example}/{pair_id}/signed[{sham_index}].max_margin")
            _close(sham.get("linear_lift"), linear - float(cell["edit_linear_margin"]), f"{example}/{pair_id}/signed[{sham_index}].linear_lift")
            _close(sham.get("max_lift"), maximum - float(cell["edit_max_margin"]), f"{example}/{pair_id}/signed[{sham_index}].max_lift")
            scale = _finite(sham.get("norm_scale"), f"{example}/{pair_id}/signed[{sham_index}].norm_scale")
            norm = _finite(sham.get("matched_head_delta_norm"), f"{example}/{pair_id}/signed[{sham_index}].norm")
            _require(scale > 0 and norm > 0, "BG2 signed norm scale/dose must be positive")
            error = abs(norm - true_norm) / true_norm
            _close(sham.get("norm_relative_error"), error, f"{example}/{pair_id}/signed[{sham_index}].norm_relative_error")
            _require(error <= 1e-3, "BG2 signed norm error exceeds 1e-3")
            transformed_hash = sham.get("transformed_residual_delta_sha256")
            _require(
                isinstance(transformed_hash, str)
                and len(transformed_hash) == 64
                and all(char in "0123456789abcdef" for char in transformed_hash),
                "BG2 signed transformed-residual hash is missing or malformed",
            )
            signed_errors.append(error)
        grouped[example][str(pair_id)] = cell
        by_example_pair[(example, str(pair_id))] = cell

    genes: set[str] = set()
    for example, rows in grouped.items():
        _require(set(rows) == set(expected_pairs), f"BG2 example lacks six directed pairs: {example}")
        row_genes = {str(row["gene_id"]) for row in rows.values()}
        _require(len(row_genes) == 1, f"BG2 example crosses genes: {example}")
        gene = next(iter(row_genes))
        _require(gene not in genes, f"BG2 has more than one example for gene {gene}")
        genes.add(gene)
    strata: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        strata[(str(cell["pair_id"]), str(cell["strand"]), str(cell["span_bin"]))].append(cell)
    expected_sources: dict[tuple[str, str], str] = {}
    for key, rows in strata.items():
        _require(len(rows) >= 2, f"BG2 cross-gene stratum has fewer than two rows: {key}")
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"bg2-crossgene|{row['gene_id']}|{row['example_id']}".encode()
            ).hexdigest(),
        )
        shift = max(1, len(ordered) // 2)
        for position, target in enumerate(ordered):
            expected_sources[(str(target["example_id"]), str(target["pair_id"]))] = str(
                ordered[(position + shift) % len(ordered)]["example_id"]
            )

    observed_sources: list[tuple[str, str]] = []
    for cell in cells:
        cross = cell["cross_gene"]
        source_key = (str(cross.get("source_example_id")), str(cell["pair_id"]))
        _require(source_key in by_example_pair, f"BG2 cross-gene source row is missing: {source_key}")
        source = by_example_pair[source_key]
        _require(str(cross.get("source_gene_id")) == str(source["gene_id"]), "BG2 cross source gene mismatch")
        _require(str(source["gene_id"]) != str(cell["gene_id"]), "BG2 cross control reused target gene")
        _require(cross.get("source_strand") == source.get("strand") == cell.get("strand"), "BG2 cross control strand mismatch")
        _require(cross.get("source_span_bin") == source.get("span_bin") == cell.get("span_bin"), "BG2 cross control span stratum mismatch")
        _require(cross.get("source_residual_delta_sha256") == source.get("residual_delta_sha256"), "BG2 cross source residual hash mismatch")
        target_key = (str(cell["example_id"]), str(cell["pair_id"]))
        _require(
            str(cross.get("source_example_id")) == expected_sources[target_key],
            f"BG2 cross source differs from frozen deterministic half-roll: {target_key}",
        )
        observed_sources.append(source_key)
    _require(
        len(observed_sources) == len(set(observed_sources)),
        "BG2 cross-gene assignment is not one-to-one within directed pairs",
    )
    return cells, {
        "cross_max": max(cross_errors),
        "signed_max": max(signed_errors),
        "n_examples": len(grouped),
    }


def _audit_bg2_constructs(
    result: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str], Mapping[str, Any]]:
    raw = _list(result.get("constructs"), "result.constructs")
    fake_groups: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {
        TARGET_PANEL: {}
    }
    for index, value in enumerate(raw):
        item = _mapping(value, f"result.constructs[{index}]")
        _require(item.get("panel_id") == TARGET_PANEL, "BG2 construct must use the target panel")
        example = item.get("example_id")
        _require(isinstance(example, str) and example, "BG2 construct example_id is missing")
        hashes = _mapping(item.get("cell_sequence_sha256"), "BG2 construct cell hashes")
        fake_groups[TARGET_PANEL][example] = {
            cell: {
                "gene_id": item.get("gene_id"),
                "chrom": item.get("chrom"),
                "strand": item.get("strand"),
                "donor_pos": item.get("donor_pos"),
                "next_acceptor_pos": item.get("next_acceptor_pos"),
                "partner_span": int(item.get("next_acceptor_pos")) - int(item.get("donor_pos")),
                "sequence_sha256": hashes.get(cell),
                "construct_sha256": item.get("construct_sha256"),
            }
            for cell in PANEL_CELLS[TARGET_PANEL]
        }
    constructs = _audit_constructs(raw, fake_groups)
    for cell in cells:
        example = str(cell["example_id"])
        item = constructs[(TARGET_PANEL, example)]
        hashes = item["cell_sequence_sha256"]
        _require(
            cell.get("reference_sequence_sha256") == hashes.get(cell.get("reference_cell"))
            and cell.get("edit_sequence_sha256") == hashes.get(cell.get("edit_cell")),
            f"BG2 pair sequence hashes differ from reconstructed panel: {example}/{cell.get('pair_id')}",
        )
        _require(
            cell.get("panel_construct_sha256") == item.get("construct_sha256"),
            f"BG2 panel construct hash mismatch: {example}/{cell.get('pair_id')}",
        )
        for field, expected in (
            ("gene_id", item.get("gene_id")),
            ("strand", item.get("strand")),
            ("donor_pos", item.get("donor_pos")),
            ("acceptor_pos", item.get("next_acceptor_pos")),
        ):
            _require(cell.get(field) == expected, f"BG2 {field} differs from construct")
        acceptor = int(item["next_acceptor_pos"])
        _require(
            cell.get("acceptor_tokens") == [acceptor - 2, acceptor - 1, acceptor, acceptor + 1],
            f"BG2 ACC4 token coordinates are wrong: {example}/{cell.get('pair_id')}",
        )
        attention = _list(cell.get("edited_attention_ACC4"), "BG2 edited ACC4 attention")
        _require(
            len(attention) == 4 and all(math.isfinite(float(value)) for value in attention),
            f"BG2 edited ACC4 attention is malformed: {example}/{cell.get('pair_id')}",
        )
    return constructs


def _verify_bg2_membership(
    result: Mapping[str, Any],
    preflight: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    constructs: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    cohort = _mapping(result.get("cohort"), "result.cohort")
    ids = _list(_mapping(cohort.get("ids"), "result.cohort.ids").get("route"), "route ids")
    genes = _list(_mapping(cohort.get("genes"), "result.cohort.genes").get("route"), "route genes")
    _require(len(ids) == len(genes) and len(set(ids)) == len(ids) and len(set(genes)) == len(genes), "BG2 route membership malformed")
    strands = _list(cohort.get("strands"), "result.cohort.strands")
    _require(len(strands) == len(ids) and set(strands).issubset({"+", "-"}), "BG2 route strands are malformed")
    observed = {str(cell["example_id"]): str(cell["gene_id"]) for cell in cells}
    _require(observed == dict(zip(map(str, ids), map(str, genes))), "BG2 cells differ from route membership")
    frozen = _mapping(cohort.get("preflight_membership"), "result.cohort.preflight_membership")
    preflight_frozen = _mapping(
        _mapping(_mapping(preflight.get("target_cohorts"), "preflight.target_cohorts").get("val"), "preflight val targets").get("target_route"),
        "preflight target_route",
    )
    _require(frozen == preflight_frozen, "BG2 frozen target membership differs from preflight")
    _require(
        ids == frozen.get("ids")
        and genes == frozen.get("genes")
        and strands == frozen.get("strands"),
        "BG2 route ids/genes/strands differ from frozen membership",
    )
    _require(len(ids) >= MIN_BG2_TARGET_GENES, "BG2 route gene supply is below 96")
    strand_counts = {strand: strands.count(strand) for strand in ("+", "-")}
    _require(
        min(strand_counts.values()) >= MIN_BG2_TARGET_GENES_PER_STRAND,
        "BG2 route per-strand supply is below 24",
    )
    counts = _mapping(cohort.get("counts"), "result.cohort.counts")
    _require(counts.get("route") == len(ids), "BG2 route count mismatch")
    _require(
        counts.get("route_plus_strand") == strand_counts["+"]
        and counts.get("route_minus_strand") == strand_counts["-"],
        "BG2 route strand counts mismatch",
    )
    _require(
        frozen.get("sha256") == _cohort_hash_from_constructs(
            frozen, TARGET_PANEL, constructs
        ),
        "BG2 sequence-aware cohort hash differs from preflight",
    )
    _require(
        cohort.get("data_sha256") == frozen.get("sha256"),
        "BG2 cohort data hash differs from frozen membership",
    )


def _bg2_expected_summaries(cells: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    behavior_linear = _transition_values(cells, lambda cell: float(cell["reference_linear_margin"]) - float(cell["edit_linear_margin"]))
    behavior_max = _transition_values(cells, lambda cell: float(cell["reference_max_margin"]) - float(cell["edit_max_margin"]))
    true_linear = _transition_values(cells, lambda cell: float(cell["true"]["linear_lift"]))
    true_max = _transition_values(cells, lambda cell: float(cell["true"]["max_lift"]))
    cross_linear = _transition_values(cells, lambda cell: float(cell["cross_gene"]["linear_lift"]))
    true_minus_cross = _transition_values(cells, lambda cell: float(cell["true"]["linear_lift"]) - float(cell["cross_gene"]["linear_lift"]))
    transitions = list(TRANSITION_BACKGROUNDS)
    observed = np.asarray([np.mean(true_linear[name][0]) for name in transitions])
    null = np.empty((len(SIGNED_SHAM_SEEDS), len(transitions)))
    signed_controls: dict[str, list[dict[str, Any]]] = {name: [] for name in transitions}
    for seed_index, seed in enumerate(SIGNED_SHAM_SEEDS):
        aggregate = _transition_values(cells, lambda cell, i=seed_index: float(cell["signed_coordinate_shams"][i]["linear_lift"]))
        for transition_index, transition in enumerate(transitions):
            statistic = float(np.mean(aggregate[transition][0]))
            null[seed_index, transition_index] = statistic
            signed_controls[transition].append({"seed": seed, "mean": statistic})
    pvalues = _max_family_p(observed, null, two_sided=False)
    output: dict[str, dict[str, Any]] = {}
    for index, transition in enumerate(transitions):
        linear, genes = true_linear[transition]
        maximum, max_genes = true_max[transition]
        denominator, denominator_genes = behavior_max[transition]
        cross, cross_genes = cross_linear[transition]
        difference, difference_genes = true_minus_cross[transition]
        _require(genes == max_genes == denominator_genes == cross_genes == difference_genes, f"BG2 {transition} gene ordering drifted")
        denominator_sum = float(np.sum(denominator))
        primary = _mean_ci(linear, genes, seed=ANALYSIS_SEED + index)
        cross_difference = _mean_ci(difference, genes, seed=ANALYSIS_SEED + 10 + index)
        expected = {
            "behavior_linear_reference_minus_edit": _mean_ci(
                behavior_linear[transition][0], behavior_linear[transition][1],
                seed=ANALYSIS_SEED + 30 + index,
            ),
            "primary_linear_raw_lift": primary,
            "secondary_max_raw_lift": _mean_ci(maximum, genes, seed=ANALYSIS_SEED + 40 + index),
            "secondary_max_denominator_reference_minus_edit": {
                **_mean_ci(denominator, genes, seed=ANALYSIS_SEED + 60 + index),
                "sum": denominator_sum,
                "positive": denominator_sum > 0.0,
                "near_zero": abs(denominator_sum) <= 1e-9,
            },
            "secondary_max_recovery_ratio_of_means": (
                None if denominator_sum <= 1e-9 else float(np.sum(maximum) / max(denominator_sum, 1e-9))
            ),
            "secondary_max_recovery_ratio_ci95": (
                None if denominator_sum <= 1e-9 else _ratio_ci(
                    maximum, denominator, genes, ANALYSIS_SEED + 20 + index
                )
            ),
            "cross_gene_same_transition": {
                **_mean_ci(cross, genes, seed=ANALYSIS_SEED + 50 + index),
                "true_minus_cross_gene": cross_difference,
            },
            "signed_coordinate_permutation": {
                "controls": signed_controls[transition],
                "p_add_one_one_sided_fwer_max_family": float(pvalues[index]),
                "family_size": len(SIGNED_SHAM_SEEDS),
                "scanned_family": transitions,
            },
            "route_supported": bool(
                primary["lower_95_one_sided"] > 0.0
                and cross_difference["lower_95_one_sided"] > 0.0
                and pvalues[index] <= 0.05
            ),
        }
        output[transition] = expected
    return output


def _audit_bg2(
    result: Mapping[str, Any], records: Mapping[str, Mapping[Path, str]]
) -> dict[str, Any]:
    preflight_record = _mapping(result.get("preflight"), "result.preflight")
    preflight_path = _bound_input(
        records, preflight_record.get("path"), preflight_record.get("sha256"), "BG2 preflight"
    )
    preflight = _json(preflight_path, "BG2 preflight")
    _require(preflight.get("schema") == PREFLIGHT_SCHEMA, "wrong BG2 preflight schema")
    _require(preflight.get("PASS") is True and preflight.get("smoke") is False, "BG2 preflight did not pass non-smoke")
    preflight_checks = _mapping(preflight.get("checks"), "BG2 preflight checks")
    _require(
        preflight_checks and all(
            isinstance(check, Mapping) and check.get("PASS") is True
            for check in preflight_checks.values()
        ),
        "BG2 preflight contains a failed or malformed check",
    )
    authorization = _mapping(result.get("behavior_authorization"), "result.behavior_authorization")
    behavior_path = _bound_input(
        records, authorization.get("path"), authorization.get("sha256"), "BG2 behavior authorization"
    )
    behavior = _json(behavior_path, "BG2 behavior authorization")
    behavior_manifest_path = _bound_input(
        records,
        authorization.get("manifest_path"),
        authorization.get("manifest_sha256"),
        "BG2 behavior manifest",
    )
    behavior_audit_path = _bound_input(
        records,
        authorization.get("audit_path"),
        authorization.get("audit_sha256"),
        "BG2 behavior audit",
    )
    fresh_behavior_audit = build_audit(behavior_path, behavior_manifest_path)
    saved_behavior_audit = _json(behavior_audit_path, "BG2 behavior audit")
    _require(
        fresh_behavior_audit.get("PASS") is True
        and saved_behavior_audit == fresh_behavior_audit,
        "BG2 upstream BG1 independent audit is absent, stale, or failed",
    )
    _compare_subset(
        behavior,
        {
            "schema": BG1_SCHEMA,
            "status": "VERIFICATION",
            "preflight_sha256": _sha256(preflight_path),
            "cohort": {"split": "val"},
            "target": {
                "verdict": "SUPPORTED_VERIFICATION",
                "primary_ids": ["m3_T_minus_A", "p1_G_minus_A"],
            },
        },
        "BG2 behavior authorization source",
    )
    _compare_subset(
        authorization,
        {
            "schema": behavior["schema"],
            "status": behavior["status"],
            "verdict": behavior["target"]["verdict"],
            "primary_ids": behavior["target"]["primary_ids"],
            "preflight_sha256": behavior["preflight_sha256"],
            "manifest_sha256": _sha256(behavior_manifest_path),
            "audit_sha256": _sha256(behavior_audit_path),
            "audit_schema": AUDIT_SCHEMA,
            "audit_PASS": True,
        },
        "result.behavior_authorization",
    )
    config = _mapping(result.get("config"), "result.config")
    _require(result.get("status") == "VERIFICATION", "BG2 status must be VERIFICATION")
    _require(config.get("split") == "val", "BG2 split must be val")
    _require(config.get("design") == preflight.get("config"), "BG2 design differs from preflight")
    design = _mapping(config.get("design"), "BG2 design")
    model_hashes = _mapping(design.get("model_file_sha256"), "BG2 model hashes")
    _require_manifest_inventory(
        records,
        code_paths=REQUIRED_BG2_CODE_INPUTS,
        expected_hashes={
            "GTF": design.get("gtf_sha256"),
            "FASTA": design.get("fasta_sha256"),
            **{f"model:{name}": digest for name, digest in model_hashes.items()},
        },
    )
    _compare_subset(
        config,
        {
            "n_boot": BG2_BOOTSTRAP_DRAWS,
            "signed_sham_seeds": list(SIGNED_SHAM_SEEDS),
            "directed_pairs": [
                {
                    "pair_id": row[0], "transition": row[1], "background": row[2],
                    "reference_cell": row[3], "edit_cell": row[4],
                }
                for row in DIRECTED_PAIRS
            ],
        },
        "result.config",
    )
    cells, norm = _audit_bg2_cells(result)
    constructs = _audit_bg2_constructs(result, cells)
    _verify_bg2_membership(result, preflight, cells, constructs)
    route_genes = set(_mapping(result["cohort"]["genes"], "BG2 route genes")["route"])
    behavior_genes = set(
        _mapping(
            _mapping(_mapping(behavior.get("cohort"), "BG1 cohort").get("memberships"), "BG1 memberships").get("target_behavior"),
            "BG1 target behavior membership",
        ).get("genes", ())
    )
    _require(route_genes.isdisjoint(behavior_genes), "BG2 route genes overlap BG1 behavior genes")
    diagnostics = _mapping(result.get("diagnostics"), "result.diagnostics")
    value_max = max(_finite(cell.get("value_linearity_max_abs"), "BG2 value-linearity error") for cell in cells)
    reconstruction_max = max(_finite(cell.get("edited_head_reconstruction_max_abs"), "BG2 reconstruction error") for cell in cells)
    attention_max = max(_finite(cell.get("edited_attention_row_sum_abs_error"), "BG2 attention error") for cell in cells)
    capture_pass = value_max <= 1e-4 and reconstruction_max <= 1e-4 and attention_max <= 1e-4
    _compare_subset(
        diagnostics,
        {
            "value_linearity_max_abs": value_max,
            "edited_head_reconstruction_max_abs": reconstruction_max,
            "edited_attention_row_sum_max_abs": attention_max,
            "n_records": len(cells),
            "n_genes": int(norm["n_examples"]),
            "all_true_head_deltas_nonzero": True,
            "capture_PASS": capture_pass,
            "cross_gene_norm_match_max_relative": norm["cross_max"],
            "signed_coordinate_norm_match_max_relative": norm["signed_max"],
            "norm_match_PASS": norm["cross_max"] <= 1e-3 and norm["signed_max"] <= 1e-3,
        },
        "result.diagnostics",
    )
    _require(capture_pass and diagnostics.get("capture_PASS") is True, "BG2 capture diagnostics failed")
    _require(
        _finite(diagnostics.get("edited_output_replay_linear_max_abs"), "BG2 replay linear error") <= 1e-4
        and _finite(diagnostics.get("edited_output_replay_max_margin_max_abs"), "BG2 replay max error") <= 1e-4
        and diagnostics.get("edited_output_replay_PASS") is True,
        "BG2 edited-output replay diagnostics failed",
    )
    before = diagnostics.get("checkpoint_parameter_sha256_before")
    after = diagnostics.get("checkpoint_parameter_sha256_after")
    _require(
        isinstance(before, str)
        and len(before) == 64
        and isinstance(after, str)
        and len(after) == 64
        and before == after
        and diagnostics.get("checkpoint_unmutated_PASS") is True,
        "BG2 checkpoint mutation diagnostic failed",
    )
    expected = _bg2_expected_summaries(cells)
    summaries = _mapping(result.get("summaries"), "result.summaries")
    _require(set(summaries) == set(expected), "BG2 summaries must contain exactly both transitions")
    for transition, summary in expected.items():
        _compare_subset(summaries[transition], summary, f"result.summaries.{transition}")
    verdict = "BG2_ROUTE_SUPPORTED" if all(item["route_supported"] for item in expected.values()) else "BG2_INCONCLUSIVE"
    _require(result.get("verdict") == verdict, "BG2 joint verdict mismatch")
    return {
        "checks": {
            "panel_completeness_and_background_averaging": True,
            "native_sequence_edits_and_motif_inventory": True,
            "upstream_bg1_independent_audit": True,
            "base_logit_margins_and_lifts": True,
            "control_norm_tolerances": True,
            "max_family_add_one_pvalues": True,
            "route_verdict": True,
        },
        "recomputed": {
            "verdict": verdict,
            "transition_means": {
                name: expected[name]["primary_linear_raw_lift"]["mean"] for name in expected
            },
            "max_family_p": {
                name: expected[name]["signed_coordinate_permutation"]["p_add_one_one_sided_fwer_max_family"]
                for name in expected
            },
            "max_norm_relative_error": {
                "cross_gene": norm["cross_max"], "signed_coordinate": norm["signed_max"]
            },
            "n_cells": len(cells),
        },
    }


def build_audit(result_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Return a PASS/FAIL audit report without writing or loading a model."""

    result_file = Path(result_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    report: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "PASS": False,
        "model_forward_passes": 0,
        "sources": {
            "result": {
                "path": str(result_file),
                "sha256": _sha256(result_file) if result_file.is_file() else None,
            },
            "manifest": {
                "path": str(manifest_file),
                "sha256": _sha256(manifest_file) if manifest_file.is_file() else None,
            },
        },
        "failures": [],
        "claim_boundary": (
            "Independent artifact/statistical-consistency audit with panel and public-control "
            "reconstruction; no checkpoint-forward replay or biological/protein-identity claim."
        ),
    }
    try:
        result = _json(result_file, "result")
        manifest = _json(manifest_file, "manifest")
        records = _verify_manifest(manifest, manifest_file, result_file)
        schema = result.get("schema")
        report["audited_schema"] = schema
        dispatch = {BG1_SCHEMA: _audit_bg1, BG2_SCHEMA: _audit_bg2}
        _require(schema in dispatch, f"unsupported result schema: {schema!r}")
        details = dispatch[schema](result, records)
        report.update(details)
        report["checks"] = {"manifest_input_output_hashes": True, **details["checks"]}
        report["manifest_inventory"] = {
            "n_inputs": len(records["inputs"]), "n_outputs": len(records["outputs"])
        }
        report["PASS"] = True
    except (AuditError, KeyError, IndexError, TypeError, ValueError) as exc:
        report["failures"].append(str(exc) or type(exc).__name__)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result_path = Path(args.result).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else result_path.with_suffix(".audit.json")
    _require(out not in (result_path, manifest_path), "audit output must differ from its sources")
    out = guard_output_path(out)
    report = build_audit(result_path, manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"PASS": report["PASS"], "audit": str(out), "failures": report["failures"]}, indent=2))
    if not report["PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
