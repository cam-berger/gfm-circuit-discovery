#!/usr/bin/env python
"""Re-execute C43's crossed-register target family from its raw cell rows.

    uv run --extra dev python \
        scripts/splicebert_exon_edge_compatibility/reproduce_crossed_direct_edge_replication.py \
        --out <new path>.json

Model-free and self-contained. Reads the gzip-archived raw rows (3,456 cells and 2,304
native-source rows, 384 blocks), checks that they decompress to the bytes the controlling
audit recomputed from (`cells_result_sha256`, `native_result_sha256`), and then recomputes
from the rows:

- the primary `Psi` (mean, percentile 95% interval, bootstrap SD, both MDE80s) and
  `M0`/`M+`/`M-`;
- the five simultaneous directional bounds, the two mismatch-equivalence intervals and the
  paired AC-versus-AT sensitivity;
- all 99 pseudo-register controls, regenerated from the frozen hash rule, the exact add-one
  one-sided p, and the paired target-minus-mean-control lower bound;
- leave-one-block/chromosome/strand sensitivity and the 384-row block endpoint table;
- the 384 native block rows, the native floor calibration, and the native-source coupling
  behind gate 2;
- the capability endpoints and pass flags of gates 2-8, gate 8 against the recomputed native
  floors, and the decision-table class.

Every recomputed object is compared against `crossed_direct_edge_replication_audit.json`. The
native block rows and the full floor calibration are not stored in the audit; they are compared
against the native result's own copy and the cells header's copy, both of which the audit
binds by sha256 and checked against its own recomputation. The run passes only if each field
is within the audit's own comparison tolerance (`_close`: rel/abs 1e-10, discrete values
exact); bit-identity is reported separately for every field.

The archived arithmetic is transcribed here, not restored: the result auditor's reductions and
the inference primitives of `phase_crossed_inference.py`, both identified by the sha256 the audit
manifest records for them, and the floor calibration of `phase_crossed_runner.py`, identified by
the sha256 the audit-bound freeze package records for it. Bit-identity with the audit is what
shows the transcription faithful.

Not recomputed here, and listed as such in the output: the stop-load diagnostics (they need the
48 MB cohort), gates 1, 9 and 10, and the per-row source-ID checks against the cohort. Those
remain attestations of the audit. See CORRECTIONS.md R49 and its addendum.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402


RESULTS = REPO / "results/splicebert_exon_edge_compatibility"
CELLS_ARCHIVE = RESULTS / "crossed_direct_edge_replication_cells.json.gz"
NATIVE_ARCHIVE = RESULTS / "crossed_direct_edge_replication_native.json.gz"
AUDIT = RESULTS / "crossed_direct_edge_replication_audit.json"
AUDIT_MANIFEST = RESULTS / "crossed_direct_edge_replication_audit.manifest.json"
CELLS_RELATIVE = "results/splicebert_exon_edge_compatibility/crossed_direct_edge_replication_cells.json"
NATIVE_RELATIVE = (
    "results/splicebert_exon_edge_compatibility/crossed_direct_edge_replication_native.json"
)
# Code the audit manifest pins by sha256 and that this script executes as-is.
PINNED_CODE = ("src/gfmcd/provenance.py",)
# Archived code transcribed below (it stays archived: AGENTS.md, and the positional-boundary-
# coherence preregistration forbids restoring an archived module to the active tree). The
# hashes are the audit manifest's own input records, so each source is identified exactly.
TRANSCRIBED_FROM = {
    "src/gfmcd/splicing/phase_crossed_inference.py": (
        "02b93bdd7883da49437e99842a3a9c2307cfc64d46d192075b0dae79fbbd0cbc"
    ),
    "scripts/splicebert_exon_edge_compatibility/"
    "audit_crossed_direct_edge_replication_results.py": (
        "7d0ca77343e4f51ba6249cc9c36d7327c288abf7b13488a7c637aa6111393492"
    ),
}
# The audit manifest does not list the runner whose floor calibration is transcribed; the freeze
# package the audit binds (`freeze_package_sha256`) pins it by this hash. That package is not
# carried, so this record identifies the source and cannot be checked from a clean checkout.
TRANSCRIBED_FROM_FREEZE_PACKAGE = {
    "src/gfmcd/splicing/phase_crossed_runner.py": (
        "f9edfd52ef32230aff84d4e69037cb68f863db074d5b80eb55a4c62488cf3174"
    ),
}

# Frozen constants: preregistration sections 4-5, and `phase_crossed_replication.py` at fd26cc1b.
ROLE = "replication"
EXPECTED_BLOCKS = 384
EXPECTED_NATIVE_ROWS = 6 * EXPECTED_BLOCKS
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260803
PSEUDO_SEEDS = tuple(range(2026080401, 2026080500))
CELLS_SCHEMA = (
    "gfmcd.splicebert_exon_edge_compatibility."
    "crossed_direct_edge_replication_cells.v1"
)
NATIVE_SCHEMA = (
    "gfmcd.splicebert_exon_edge_compatibility."
    "crossed_direct_edge_replication_native.v1"
)
# The native floor calibration stratifies on `chrom NUL strand`; every other bootstrap here uses
# `chrom:strand`. The two sort differently (chr1 vs chr10), so they draw different indices.
NATIVE_FLOOR_STRATUM_SEPARATOR = "\0"
AUDIT_SCHEMA = (
    "gfmcd.splicebert_exon_edge_compatibility."
    "crossed_direct_edge_replication_result_audit.v1"
)
PROTECTED_POSITIONS = {
    "donor": (64, 96, 128, 160, 192, 224, 256, 288, 320, 344, 360, 376),
    "acceptor": (528, 544, 560, 576, 608, 640, 672, 704, 736, 768, 800, 832),
}
DIRECTION_NAMES = (
    "m0_minus_plus",
    "m0_minus_minus",
    "donor_phase_0_excess",
    "donor_phase_1_excess",
    "donor_phase_2_excess",
)
MISMATCH_NAMES = ("m_plus", "m_minus")
# Gates whose inputs are only the cell and native rows; the rest stay audit attestations.
RECOMPUTED_GATES = (
    "source_edge_positive_control",
    "crossed_lattice_capability",
    "negative_control",
    "resolution",
    "corruption_robustness",
    "protected_mlm_health",
    "phase_blind_site_health",
)
ATTESTED_GATES = (
    "construction",
    "numerics",
    "independent_audit",
)
# The audit's own raw-row checks on the native rows, reproduced here as booleans.
NATIVE_AUDIT_CHECKS = ("raw_native_rows_and_block_arithmetic", "native_floor_calibration")
NOT_RECOMPUTED = {
    "target.stop_sensitivity": "needs per-block stop diagnostics from the 48 MB cohort",
    "target.stop_count_mean_3x3": "needs per-block stop matrices from the 48 MB cohort",
    "instrument_endpoints.numerics": "model-run attestations copied from the run headers (gate 9)",
    "gates.construction": "construction and freeze audits of the cohort (gate 1)",
    "gates.independent_audit": "hash chain, manifests and one-shot chronology (gate 10)",
    "cell_rows.source_ids": "per-row donor/acceptor source IDs are checked only against the cohort",
    "native_rows.source_ids": (
        "checked here against the source IDs the cell rows carry, and against the cohort "
        "only in the audit"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, label: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _close(left: Any, right: Any, *, atol: float = 1e-10) -> bool:
    """The audit's own comparison: recursive, float serialization noise only."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _close(left[key], right[key], atol=atol) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b, atol=atol) for a, b in zip(left, right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=atol)
        )
    return left == right


def _max_abs_deviation(left: Any, right: Any) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return max((_max_abs_deviation(left[k], right[k]) for k in left if k in right),
                   default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        return max((_max_abs_deviation(a, b) for a, b in zip(left, right)), default=0.0)
    if (
        isinstance(left, (int, float)) and isinstance(right, (int, float))
        and not isinstance(left, bool) and not isinstance(right, bool)
    ):
        return abs(float(left) - float(right))
    return 0.0


def _json_value(value: Any) -> Any:
    """Normalize tuples and numpy scalars exactly as a JSON write/read round trip would."""

    return json.loads(json.dumps(value, allow_nan=False))


def _record_for(manifest: Mapping[str, Any], key: str, relative: str) -> Mapping[str, Any]:
    matches = [
        record for record in manifest.get(key, [])
        if str(record.get("path", "")).endswith("/" + relative)
    ]
    _require(len(matches) == 1, f"audit manifest has no unique {key} record for {relative}")
    return matches[0]


def _matches_record(path: Path, record: Mapping[str, Any]) -> bool:
    return (
        path.is_file()
        and record.get("size_bytes") == path.stat().st_size
        and record.get("sha256") == sha256_file(path)
    )


def load_archive(path: Path) -> tuple[dict[str, Any], str, int]:
    """Read gzip-archived (or plain) raw rows; return them with the raw-byte sha256 and size."""

    payload = path.read_bytes()
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    rows = json.loads(payload)
    _require(isinstance(rows, dict), f"{path.name} must hold a JSON object")
    return rows, hashlib.sha256(payload).hexdigest(), len(payload)


PREDECESSOR_FIELDS = (
    "cohort_sha256",
    "construction_audit_sha256",
    "construction_audit_manifest_sha256",
    "freeze_package_sha256",
    "freeze_manifest_sha256",
    "freeze_audit_sha256",
    "freeze_audit_manifest_sha256",
    "runner_sha256",
)


def verify_bindings(
    *,
    audit: Mapping[str, Any],
    audit_path: Path,
    audit_manifest: Mapping[str, Any],
    cells: Mapping[str, Any],
    cells_sha256: str,
    cells_size: int,
    native: Mapping[str, Any],
    native_sha256: str,
    native_size: int,
) -> dict[str, bool]:
    """Bind the audit, the raw rows and the executing code to each other before any arithmetic."""

    cells_record = _record_for(audit_manifest, "inputs", CELLS_RELATIVE)
    native_record = _record_for(audit_manifest, "inputs", NATIVE_RELATIVE)
    checks = {
        "audit_schema_and_status": (
            audit.get("schema") == AUDIT_SCHEMA
            and audit.get("status") == "PASS"
            and audit.get("role") == ROLE
            and audit.get("independent_audit_passed") is True
        ),
        "audit_bytes_equal_audit_manifest_output": _matches_record(
            audit_path, _record_for(audit_manifest, "outputs", audit_path.name)
        ),
        "raw_rows_equal_audited_cells_result": (
            cells_sha256 == audit.get("cells_result_sha256")
            == cells_record.get("sha256")
            and cells_size == cells_record.get("size_bytes")
        ),
        "raw_rows_bind_the_audited_predecessors": all(
            cells.get(field) == audit.get(field)
            for field in (*PREDECESSOR_FIELDS, "native_result_sha256", "native_manifest_sha256")
        ) and cells.get("attempt_sha256") == audit.get("cells_attempt_sha256"),
        "raw_rows_schema_and_stage": (
            cells.get("schema") == CELLS_SCHEMA
            and cells.get("status") == "COMPLETE"
            and cells.get("stage") == "cells"
            and cells.get("role") == ROLE
        ),
        "native_rows_equal_audited_native_result": (
            native_sha256 == audit.get("native_result_sha256")
            == cells.get("native_result_sha256") == native_record.get("sha256")
            and native_size == native_record.get("size_bytes")
        ),
        "native_rows_bind_the_audited_predecessors": all(
            native.get(field) == audit.get(field) for field in PREDECESSOR_FIELDS
        ) and native.get("attempt_sha256") == audit.get("native_attempt_sha256"),
        "native_rows_schema_and_stage": (
            native.get("schema") == NATIVE_SCHEMA
            and native.get("status") == "COMPLETE"
            and native.get("stage") == "native"
            and native.get("role") == ROLE
            and native.get("native_stage_complete") is True
            and native.get("crossed_cell_outcome_rows") == 0
            and native.get("replication_native_outcome_rows") == EXPECTED_NATIVE_ROWS
        ),
    }
    for relative in PINNED_CODE:
        checks[f"executing_code_is_audit_pinned:{relative}"] = _matches_record(
            REPO / relative, _record_for(audit_manifest, "inputs", relative)
        )
    for relative, sha256 in TRANSCRIBED_FROM.items():
        checks[f"transcription_source_is_audit_pinned:{relative}"] = (
            _record_for(audit_manifest, "inputs", relative).get("sha256") == sha256
        )
    return checks


def reduce_cells(cells: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rederive each block's 3x3 AC/AT lattices and phase-blind summaries from raw margins.

    Transcribes the target-relevant part of `recompute_cells` in the archived result auditor
    (`audit_crossed_direct_edge_replication_results.py` at fd26cc1b); only the source-ID check
    against the cohort is absent.
    """

    rows = cells.get("cell_rows")
    _require(
        isinstance(rows, list) and len(rows) == 9 * EXPECTED_BLOCKS,
        "crossed raw-row count is incomplete",
    )
    grouped: dict[str, dict[tuple[int, int], Mapping[str, Any]]] = defaultdict(dict)
    sites: dict[str, tuple[str, str]] = {}
    for row in rows:
        block_id = str(row["block_id"])
        site = (str(row["chrom"]), str(row["strand"]))
        _require(sites.setdefault(block_id, site) == site, "block changes chromosome/strand")
        p, q = int(row["donor_phase"]), int(row["acceptor_phase"])
        _require(
            p in range(3) and q in range(3) and (p, q) not in grouped[block_id],
            "duplicate or invalid crossed phase cell",
        )
        _require(
            int(row["p"]) == p
            and int(row["q"]) == q
            and int(row["phase_gap"]) == (q - p) % 3
            and bool(row["compatible"]) == (p == q),
            "crossed row phase labels are inconsistent",
        )
        margins = row["donor_margins"]
        intact = _finite(margins["intact"], "cell intact margin")
        expected_effects = {
            "c_ac": intact - _finite(margins["true_acceptor_ac"], "cell AC margin"),
            "c_at": intact - _finite(margins["true_acceptor_at"], "cell AT margin"),
            "n_bridge": intact - _finite(margins["bridge_ac"], "cell bridge margin"),
        }
        _finite(row["acceptor_margin_intact"], "cell acceptor margin")
        _require(
            _close(row["effects"], expected_effects, atol=1e-12),
            "crossed effects do not equal raw-margin differences",
        )
        probes = row["protected_probes"]
        _require(
            isinstance(probes, list) and len(probes) == 24,
            "cell does not contain 24 raw protected probes",
        )
        seen: set[tuple[str, int]] = set()
        deltas: list[float] = []
        construct_correct = 0
        source_correct = 0
        for probe in probes:
            module, position = str(probe["module"]), int(probe["position"])
            _require(
                module in PROTECTED_POSITIONS
                and position in PROTECTED_POSITIONS[module]
                and (module, position) not in seen,
                "protected probe position family differs from freeze",
            )
            seen.add((module, position))
            construct = _finite(probe["construct_true_base_log_probability"], "construct logp")
            source = _finite(probe["source_true_base_log_probability"], "source logp")
            delta = _finite(probe["true_base_log_probability_delta"], "protected delta")
            _require(
                math.isclose(delta, construct - source, rel_tol=0, abs_tol=1e-12),
                "protected delta does not recompute",
            )
            deltas.append(delta)
            construct_correct += int(bool(probe["construct_correct"]))
            source_correct += int(bool(probe["source_correct"]))
        expected_health = {
            "mean_true_base_log_probability_delta": float(np.mean(deltas)),
            "construct_correct_count": construct_correct,
            "source_correct_count": source_correct,
            "total_probe_count": 24,
        }
        _require(
            _close(row["protected_health"], expected_health, atol=1e-12),
            "cell protected-health aggregate does not recompute",
        )
        grouped[block_id][(p, q)] = row
    lattice = {(p, q) for p in range(3) for q in range(3)}
    _require(
        len(grouped) == EXPECTED_BLOCKS
        and all(set(by_phase) == lattice for by_phase in grouped.values()),
        f"raw rows do not form {EXPECTED_BLOCKS} complete 3x3 lattices",
    )

    blocks = []
    for block_id in sorted(grouped):
        by_phase = grouped[block_id]
        rows9 = [by_phase[(p, q)] for p in range(3) for q in range(3)]
        total = sum(int(row["protected_health"]["total_probe_count"]) for row in rows9)
        logp = sum(
            float(row["protected_health"]["mean_true_base_log_probability_delta"])
            * int(row["protected_health"]["total_probe_count"])
            for row in rows9
        ) / total

        def phase_blind(field: str, parent: str | None = None) -> float:
            return float(np.mean([
                float((row[parent] if parent else row)[field]) for row in rows9
            ]))

        blocks.append({
            "block_id": block_id,
            "stratum": f"{sites[block_id][0]}:{sites[block_id][1]}",
            "ac": [[float(by_phase[(p, q)]["effects"]["c_ac"]) for q in range(3)]
                   for p in range(3)],
            "at": [[float(by_phase[(p, q)]["effects"]["c_at"]) for q in range(3)]
                   for p in range(3)],
            "phase_blind": {
                "mean_intact_donor_margin": phase_blind("intact", "donor_margins"),
                "mean_intact_acceptor_margin": phase_blind("acceptor_margin_intact"),
                "mean_c_ac": phase_blind("c_ac", "effects"),
                "mean_c_at": phase_blind("c_at", "effects"),
                "mean_n_bridge": phase_blind("n_bridge", "effects"),
            },
            "protected_health": {
                "mean_true_base_log_probability_delta": float(logp),
                "construct_correct_count": sum(
                    int(row["protected_health"]["construct_correct_count"]) for row in rows9
                ),
                "source_correct_count": sum(
                    int(row["protected_health"]["source_correct_count"]) for row in rows9
                ),
                "total_probe_count": total,
            },
        })
    return blocks


def _cell_source_identities(
    cells: Mapping[str, Any],
) -> tuple[dict[str, set[tuple[str, int, str]]], dict[str, tuple[str, str]]]:
    """Each block's six (role, phase, source ID) triples and its site, as the cell rows record them.

    Stands in for the cohort, from which the audit read the same triples: a block's donor at
    phase p is shared by its three cells with donor phase p, and likewise for acceptors.
    """

    sources: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    sites: dict[str, tuple[str, str]] = {}
    for row in cells["cell_rows"]:
        block_id = str(row["block_id"])
        sites[block_id] = (str(row["chrom"]), str(row["strand"]))
        sources[(block_id, "donor", int(row["donor_phase"]))].add(str(row["donor_source_id"]))
        sources[(block_id, "acceptor", int(row["acceptor_phase"]))].add(
            str(row["acceptor_source_id"])
        )
    _require(
        all(len(ids) == 1 for ids in sources.values()),
        "a cell-row source phase carries more than one source ID",
    )
    expected: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
    for (block_id, role, phase), ids in sources.items():
        expected[block_id].add((role, phase, next(iter(ids))))
    return dict(expected), sites


def reduce_native(native: Mapping[str, Any], cells: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rederive the 384 native block rows from the raw native-source margins.

    Transcribes the block reduction of `recompute_native` in the archived result auditor; the
    source identities and sites it read from the cohort are read from the cell rows instead.
    """

    expected, block_meta = _cell_source_identities(cells)
    rows = native.get("native_rows")
    _require(
        isinstance(rows, list) and len(rows) == 6 * len(expected),
        "native raw-row count is incomplete",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    observed: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
    for row in rows:
        block_id = str(row["block_id"])
        role = str(row["source_role"])
        phase = int(row["phase"])
        source_id = str(row["source_id"])
        _require(
            block_id in expected
            and (str(row["chrom"]), str(row["strand"])) == block_meta[block_id],
            "native row leaves the crossed replication blocks",
        )
        intact = _finite(row["donor_margin_intact"], "native intact margin")
        corrupt = _finite(row["donor_margin_true_acceptor_ac"], "native corrupt margin")
        effect = _finite(row["c_ac"], "native coupling")
        _finite(row["acceptor_margin_intact"], "native acceptor margin")
        _require(
            math.isclose(effect, intact - corrupt, rel_tol=0, abs_tol=1e-12),
            "native C_AC does not equal its raw-margin difference",
        )
        key = (role, phase, source_id)
        _require(key not in observed[block_id], "duplicate native source row")
        observed[block_id].add(key)
        grouped[block_id].append(row)
    _require(
        dict(observed) == expected,
        "native source identities do not match the cell rows",
    )
    block_rows = []
    for block_id in sorted(grouped):
        block = grouped[block_id]

        def mean(field: str) -> float:
            return float(np.mean([_finite(row[field], field) for row in block]))

        block_rows.append({
            "block_id": block_id,
            "chrom": block_meta[block_id][0],
            "strand": block_meta[block_id][1],
            "mean_c_ac": mean("c_ac"),
            "mean_donor_margin_intact": mean("donor_margin_intact"),
            "mean_acceptor_margin_intact": mean("acceptor_margin_intact"),
        })
    return block_rows


def calibrate_native_floors(
    native_block_rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """The gate-8 floors and native summaries, frozen before any crossed-cell forward.

    Transcribes `calibrate_native_floors` in `phase_crossed_runner.py`
    (TRANSCRIBED_FROM_FREEZE_PACKAGE); the result auditor's `recompute_native` repeats the same
    arithmetic statement for statement. A floor is min(0, native lower 95% bound).
    """

    rows = list(native_block_rows)
    _require(
        len({str(row["block_id"]) for row in rows}) == len(rows),
        "native calibration block IDs must be unique",
    )
    strata = [f"{row['chrom']}{NATIVE_FLOOR_STRATUM_SEPARATOR}{row['strand']}" for row in rows]
    indices = stratified_resample_indices(strata, draws=draws, seed=seed)

    def endpoint(field: str, *, with_floor: bool) -> dict[str, Any]:
        summary = summarize([float(row[field]) for row in rows], indices)
        result: dict[str, Any] = {
            "mean": summary.mean,
            "ci95": [summary.ci95[0], summary.ci95[1]],
            "bootstrap_sd": summary.bootstrap_sd,
            "mde80": summary.mde80,
        }
        if with_floor:
            result["floor"] = float(min(0.0, summary.ci95[0]))
        return result

    return {
        "method": f"{draws} whole-block bootstrap draws within chromosome x strand; "
        "floor=min(0,native two-sided percentile-95% lower bound)",
        "draws": int(draws),
        "seed": int(seed),
        "native_source_c_ac": endpoint("mean_c_ac", with_floor=False),
        "donor_margin_intact": endpoint("mean_donor_margin_intact", with_floor=True),
        "acceptor_margin_intact": endpoint(
            "mean_acceptor_margin_intact", with_floor=True
        ),
    }


def native_audit_checks(
    native: Mapping[str, Any],
    cells: Mapping[str, Any],
    native_blocks: Sequence[Mapping[str, Any]],
    floors: Mapping[str, Any],
) -> dict[str, bool]:
    """The audit's two raw-row checks on the native stage, at its own 1e-12 tolerance."""

    return {
        "raw_native_rows_and_block_arithmetic": _close(
            list(native_blocks), native.get("native_block_rows"), atol=1e-12
        ),
        "native_floor_calibration": (
            _close(floors, native.get("floor_calibration"), atol=1e-12)
            and _close(floors, cells.get("native_floor_calibration"), atol=1e-12)
        ),
    }


# Inference primitives transcribed from `phase_crossed_inference.py` at fd26cc1b (TRANSCRIBED_FROM).
# The arithmetic is unchanged statement for statement; only argument validation that the frozen
# inputs cannot trigger was dropped. Bit-identity with the audit on every compared field is the
# check that the transcription is faithful.

MDE80_MULTIPLIER = 1.96 + 0.8416212335729143


@dataclass(frozen=True)
class IntervalSummary:
    mean: float
    ci95: tuple[float, float]
    bootstrap_sd: float
    mde80: float


@dataclass(frozen=True)
class SimultaneousIntervalSummary:
    means: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    critical_value: float


def matching_endpoints(values: Sequence[Sequence[float]]) -> dict[str, float]:
    """The three cyclic matchings and every frozen phase-coherence contrast of one lattice."""

    matrix = np.asarray(values, dtype=float)
    _require(
        matrix.shape == (3, 3) and bool(np.isfinite(matrix).all()),
        "lattice values must be a finite 3-by-3 matrix",
    )
    m0 = float(np.mean([matrix[p, p] for p in range(3)]))
    m_plus = float(np.mean([matrix[p, (p + 1) % 3] for p in range(3)]))
    m_minus = float(np.mean([matrix[p, (p - 1) % 3] for p in range(3)]))
    result = {
        "m0": m0,
        "m_plus": m_plus,
        "m_minus": m_minus,
        "psi_match": m0 - 0.5 * (m_plus + m_minus),
        "m0_minus_plus": m0 - m_plus,
        "m0_minus_minus": m0 - m_minus,
        "pooled": float(matrix.mean()),
    }
    for p in range(3):
        result[f"donor_phase_{p}_excess"] = float(
            matrix[p, p] - 0.5 * (matrix[p, (p + 1) % 3] + matrix[p, (p - 1) % 3])
        )
        result[f"acceptor_phase_{p}_excess"] = float(
            matrix[p, p] - 0.5 * (matrix[(p + 1) % 3, p] + matrix[(p - 1) % 3, p])
        )
    return result


def stratified_resample_indices(strata: Sequence[str], *, draws: int, seed: int) -> np.ndarray:
    """Whole-block bootstrap indices that preserve every stratum's size."""

    labels = np.asarray([str(value) for value in strata], dtype=object)
    groups = [np.flatnonzero(labels == label) for label in sorted(set(labels.tolist()))]
    _require(
        draws > 0 and all(len(group) >= 2 for group in groups),
        "each bootstrap stratum needs at least two independent blocks",
    )
    rng = np.random.default_rng(seed)
    sampled = [rng.choice(group, size=(draws, len(group)), replace=True) for group in groups]
    return np.concatenate(sampled, axis=1)


def summarize(values: Sequence[float], resample_indices: np.ndarray) -> IntervalSummary:
    array = np.asarray(values, dtype=float)
    draws = array[np.asarray(resample_indices, dtype=int)].mean(axis=1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    sd = float(draws.std(ddof=1))
    return IntervalSummary(
        mean=float(array.mean()),
        ci95=(float(lower), float(upper)),
        bootstrap_sd=sd,
        mde80=float(MDE80_MULTIPLIER * sd),
    )


def one_sided_lower(values: Sequence[float], resample_indices: np.ndarray) -> float:
    """95% one-sided lower bound from the shared draws (the frozen family size is one)."""

    draws = np.asarray(values, dtype=float)[np.asarray(resample_indices, dtype=int)].mean(axis=1)
    return float(np.quantile(draws, 0.05))


def simultaneous_studentized_intervals(
    values: np.ndarray, resample_indices: np.ndarray, *, confidence: float = 0.95
) -> SimultaneousIntervalSummary:
    """Symmetric max-|t| intervals over a block-by-endpoint matrix from shared resamples."""

    array = np.asarray(values, dtype=float)
    indices = np.asarray(resample_indices, dtype=int)
    _require(
        array.ndim == 2 and bool(np.isfinite(array).all()),
        "simultaneous intervals require a finite block-by-endpoint matrix",
    )
    means = array.mean(axis=0)
    draws = array[indices].mean(axis=1)
    standard_errors = draws.std(axis=0, ddof=1)
    active = standard_errors > 0
    if np.any(active):
        studentized = np.abs((draws[:, active] - means[active]) / standard_errors[active])
        critical = float(np.quantile(studentized.max(axis=1), confidence))
    else:
        critical = 0.0
    radii = critical * standard_errors
    return SimultaneousIntervalSummary(
        means=tuple(float(value) for value in means),
        lower=tuple(float(value) for value in means - radii),
        upper=tuple(float(value) for value in means + radii),
        critical_value=critical,
    )


def centered_stratified_mde80(
    values: Sequence[float], strata: Sequence[str], resample_indices: np.ndarray
) -> float:
    """Frozen resolution endpoint from within-stratum-centered block values."""

    centered = np.asarray(values, dtype=float).copy()
    labels = np.asarray([str(value) for value in strata], dtype=object)
    for label in sorted(set(labels.tolist())):
        selected = labels == label
        centered[selected] -= centered[selected].mean()
    draws = centered[np.asarray(resample_indices, dtype=int)].mean(axis=1)
    return float(MDE80_MULTIPLIER * draws.std(ddof=1))


def balanced_pseudo_phase_assignments(
    block_ids: Sequence[str], *, seed: int
) -> tuple[tuple[int, int, int], ...]:
    """Every S3 permutation equally often, after a seeded sha256 ordering of block IDs."""

    identifiers = tuple(str(block_id) for block_id in block_ids)
    choices = tuple(permutations(range(3)))
    _require(
        bool(identifiers) and len(identifiers) % len(choices) == 0
        and len(set(identifiers)) == len(identifiers),
        "balanced S3 controls need unique block IDs in a multiple of six",
    )
    order = sorted(
        range(len(identifiers)),
        key=lambda index: hashlib.sha256(
            f"crossed-phase-null-v1\0{seed}\0{identifiers[index]}".encode()
        ).hexdigest(),
    )
    assignments: list[tuple[int, int, int]] = [choices[0]] * len(identifiers)
    per_choice = len(identifiers) // len(choices)
    for rank, index in enumerate(order):
        assignments[index] = choices[rank // per_choice]
    return tuple(assignments)


def relabeled_phase_contrast(
    matrix: Sequence[Sequence[float]], acceptor_label_permutation: Sequence[int]
) -> float:
    """Diagonal-minus-off-diagonal contrast after relabeling the acceptor phase."""

    values = np.asarray(matrix, dtype=float)
    permutation = tuple(int(value) for value in acceptor_label_permutation)
    _require(sorted(permutation) == [0, 1, 2], "acceptor labels must permute 0,1,2")
    diagonal = [values[p, q] for p in range(3) for q in range(3) if permutation[q] == p]
    off_diagonal = [values[p, q] for p in range(3) for q in range(3) if permutation[q] != p]
    return float(np.mean(diagonal) - np.mean(off_diagonal))


def pseudo_phase_controls_from_assignments(
    matrices: Sequence[Sequence[Sequence[float]]],
    assignments_by_seed: Sequence[Sequence[Sequence[int]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Control means per seed, and each block's mean control contrast across seeds."""

    by_seed = np.empty((len(assignments_by_seed), len(matrices)), dtype=float)
    for i, seed_rows in enumerate(assignments_by_seed):
        _require(len(seed_rows) == len(matrices), "one assignment per lattice block")
        for j, (matrix, permutation) in enumerate(zip(matrices, seed_rows)):
            by_seed[i, j] = relabeled_phase_contrast(matrix, permutation)
    return by_seed.mean(axis=1), by_seed.mean(axis=0)


def _summary(values: Sequence[float], indices: np.ndarray, *, mde_key: str = "mde80") -> dict:
    row = summarize(values, indices)
    return {
        "mean": row.mean,
        "ci95": [row.ci95[0], row.ci95[1]],
        "bootstrap_sd": row.bootstrap_sd,
        mde_key: row.mde80,
    }


def _simultaneous(names: Sequence[str], values: np.ndarray, indices: np.ndarray) -> dict:
    row = simultaneous_studentized_intervals(values, indices)
    return {
        "method": "max-absolute-studentized whole-block bootstrap",
        "critical_value": row.critical_value,
        "endpoints": {
            name: {
                "mean": row.means[index],
                "simultaneous_ci95": [row.lower[index], row.upper[index]],
                "simultaneous_lower95": row.lower[index],
            }
            for index, name in enumerate(names)
        },
    }


def pseudo_register_family(block_ids: Sequence[str]) -> list[list[tuple[int, int, int]]]:
    """Regenerate the frozen 99-seed family by its recorded hash rule and check its balance.

    The cohort at fd26cc1b records the rule as `balanced_pseudo_phase_assignments` in this
    module over the preimage "crossed-phase-null-v1 NUL seed NUL block_id"; its construction
    audit required the stored family to regenerate from it exactly.
    """

    family = []
    for seed in PSEUDO_SEEDS:
        assignments = list(balanced_pseudo_phase_assignments(block_ids, seed=seed))
        counts = Counter(assignments)
        _require(
            len(counts) == 6 and set(counts.values()) == {EXPECTED_BLOCKS // 6},
            f"pseudo seed {seed} is not exactly balanced over S3",
        )
        family.append(assignments)
    return family


def target_family(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the frozen target family; transcribes `recompute_target` minus stop load."""

    block_ids = [str(row["block_id"]) for row in blocks]
    strata = [str(row["stratum"]) for row in blocks]
    indices = stratified_resample_indices(strata, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    ac = [matching_endpoints(row["ac"]) for row in blocks]
    at = [matching_endpoints(row["at"]) for row in blocks]
    psi = np.asarray([row["psi_match"] for row in ac], dtype=float)
    primary = _summary(psi, indices, mde_key="mde80_unadjusted")
    primary["mde80"] = centered_stratified_mde80(psi, strata, indices)
    direction_values = np.asarray([[row[name] for name in DIRECTION_NAMES] for row in ac])
    mismatch_values = np.asarray([[row[name] for name in MISMATCH_NAMES] for row in ac])
    operator_values = np.asarray(
        [left["psi_match"] - right["psi_match"] for left, right in zip(ac, at)]
    )
    controls, block_mean_control = pseudo_phase_controls_from_assignments(
        [row["ac"] for row in blocks], pseudo_register_family(block_ids)
    )
    paired = psi - block_mean_control
    leave_one = [float((psi.sum() - psi[i]) / (len(psi) - 1)) for i in range(len(psi))]
    chroms = np.asarray([value.split(":", 1)[0] for value in strata])
    strands = np.asarray([value.rsplit(":", 1)[-1] for value in strata])
    return {
        "n_blocks": len(block_ids),
        "primary_psi": primary,
        "directions": _simultaneous(DIRECTION_NAMES, direction_values, indices),
        "mismatch_equivalence": _simultaneous(MISMATCH_NAMES, mismatch_values, indices),
        "operator_sensitivity": _simultaneous(
            ("psi_ac_minus_at",), operator_values[:, None], indices
        ),
        "matching_means": {
            name: float(np.mean([row[name] for row in ac]))
            for name in ("m0", "m_plus", "m_minus", "pooled")
        },
        "pseudo_phase": {
            "seeds": list(PSEUDO_SEEDS),
            "control_means": controls.tolist(),
            "maximum_control": float(controls.max()),
            "target_exceeds_all": bool(float(psi.mean()) > float(controls.max())),
            "exact_add_one_one_sided_p": float(
                (1 + int(np.sum(controls >= psi.mean()))) / (len(controls) + 1)
            ),
            "target_minus_mean_control": {
                "mean": float(paired.mean()),
                "lower95": one_sided_lower(paired, indices),
            },
        },
        "sensitivity": {
            "leave_one_block": {"minimum": min(leave_one), "maximum": max(leave_one)},
            "leave_one_chromosome": {
                chrom: float(psi[chroms != chrom].mean()) for chrom in sorted(set(chroms))
            },
            "leave_one_strand": {
                strand: float(psi[strands != strand].mean()) for strand in sorted(set(strands))
            },
        },
        "block_endpoints": [
            {
                "block_id": block_id,
                "stratum": stratum,
                "ac": ac_row,
                "at": at_row,
                "psi_ac_minus_at": float(ac_row["psi_match"] - at_row["psi_match"]),
            }
            for block_id, stratum, ac_row, at_row in zip(block_ids, strata, ac, at)
        ],
    }


def instrument(
    blocks: Sequence[Mapping[str, Any]],
    native_blocks: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    native_floors: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Recompute the capability endpoints and gates 2-8 (`recompute_instrument`).

    `native_floors` is the recomputed calibration, where the audit read the native result's
    recorded copy; the audit separately required the two to agree.
    """

    _require(
        [str(row["block_id"]) for row in native_blocks]
        == [str(row["block_id"]) for row in blocks],
        "native and cell block orders differ",
    )
    indices = stratified_resample_indices(
        [str(row["stratum"]) for row in blocks], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    # Gate 2 resamples native block means with the cell blocks' `chrom:strand` indices.
    native_positive = _summary([float(row["mean_c_ac"]) for row in native_blocks], indices)
    phase_blind = [row["phase_blind"] for row in blocks]
    capability = _summary([float(row["mean_c_ac"]) for row in phase_blind], indices)
    bridge = _summary([float(row["mean_n_bridge"]) for row in phase_blind], indices)
    operator_row = simultaneous_studentized_intervals(
        np.asarray([
            float(row["mean_c_ac"]) - float(row["mean_c_at"]) for row in phase_blind
        ])[:, None],
        indices,
    )
    operator = {
        "mean": operator_row.means[0],
        "simultaneous_ci95": [operator_row.lower[0], operator_row.upper[0]],
        "critical_value": operator_row.critical_value,
    }
    health = [row["protected_health"] for row in blocks]
    protected = _summary(
        [float(row["mean_true_base_log_probability_delta"]) for row in health], indices
    )
    construct_correct = sum(int(row["construct_correct_count"]) for row in health)
    source_correct = sum(int(row["source_correct_count"]) for row in health)
    total = sum(int(row["total_probe_count"]) for row in health)
    retention = None if source_correct == 0 else construct_correct / source_correct
    donor = _summary([float(row["mean_intact_donor_margin"]) for row in phase_blind], indices)
    acceptor = _summary(
        [float(row["mean_intact_acceptor_margin"]) for row in phase_blind], indices
    )
    donor_floor = float(native_floors["donor_margin_intact"]["floor"])
    acceptor_floor = float(native_floors["acceptor_margin_intact"]["floor"])
    endpoints = {
        "native_source_c_ac": native_positive,
        "crossed_phase_blind_c_ac": capability,
        "bridge_effect": bridge,
        "phase_blind_ac_minus_at": operator,
        "protected_log_probability_delta": protected,
        "protected_accuracy": {
            "construct_correct": construct_correct,
            "source_correct": source_correct,
            "probe_pairs": total,
            "construct_accuracy": construct_correct / total,
            "source_accuracy": source_correct / total,
            "retention_ratio": retention,
        },
        "phase_blind_site_health": {
            "donor": {**donor, "native_floor": donor_floor},
            "acceptor": {**acceptor, "native_floor": acceptor_floor},
        },
        "target_resolution_mde80": float(target["primary_psi"]["mde80"]),
    }
    gates = {
        "source_edge_positive_control": native_positive["ci95"][0] > 0.05,
        "crossed_lattice_capability": capability["ci95"][0] > 0.02,
        "negative_control": bridge["ci95"][0] >= -0.05 and bridge["ci95"][1] <= 0.05,
        "resolution": float(target["primary_psi"]["mde80"]) <= 0.05,
        "corruption_robustness": (
            operator["simultaneous_ci95"][0] >= -0.05
            and operator["simultaneous_ci95"][1] <= 0.05
        ),
        "protected_mlm_health": (
            protected["ci95"][0] > -0.10 and retention is not None and retention >= 0.80
        ),
        "phase_blind_site_health": (
            donor["ci95"][0] > donor_floor and acceptor["ci95"][0] > acceptor_floor
        ),
    }
    return endpoints, gates


def classify(target: Mapping[str, Any], all_gates: bool) -> str:
    """The preregistered ordered decision table (section 7), as the audit transcribed it."""

    if not all_gates:
        return "INSTRUMENT_UNRESOLVED"
    lower, upper = map(float, target["primary_psi"]["ci95"])
    directional = all(
        float(row["simultaneous_lower95"]) > 0
        for row in target["directions"]["endpoints"].values()
    )
    pseudo = target["pseudo_phase"]
    pseudo_pass = (
        pseudo["target_exceeds_all"] is True
        and float(pseudo["exact_add_one_one_sided_p"]) <= 0.01
        and float(pseudo["target_minus_mean_control"]["lower95"]) > 0
    )
    operator = target["operator_sensitivity"]["endpoints"]["psi_ac_minus_at"]
    operator_pass = (
        float(operator["simultaneous_ci95"][0]) >= -0.05
        and float(operator["simultaneous_ci95"][1]) <= 0.05
    )
    controls = directional and pseudo_pass and operator_pass
    only = all(
        float(row["simultaneous_ci95"][0]) >= -0.05
        and float(row["simultaneous_ci95"][1]) <= 0.05
        for row in target["mismatch_equivalence"]["endpoints"].values()
    )
    if lower > 0.05 and controls and only:
        return "ONLY_WHEN_COHERENT_IN_FROZEN_DOMAIN"
    if lower > 0.05 and controls:
        return "MATERIAL_PREFERENTIAL_POSITIVE"
    if lower > 0 and controls:
        return "DETECTABLE_MATERIALITY_UNRESOLVED"
    if lower >= -0.05 and upper <= 0.05:
        return "BOUNDED_NULL"
    if upper < -0.05:
        return "MATERIAL_INVERTED"
    return "INCONCLUSIVE"


def compare_with_audit(
    *,
    audit: Mapping[str, Any],
    native: Mapping[str, Any],
    cells: Mapping[str, Any],
    target: Mapping[str, Any],
    endpoints: Mapping[str, Any],
    gates: Mapping[str, bool],
    classification: str,
    native_blocks: Sequence[Mapping[str, Any]],
    floors: Mapping[str, Any],
    native_checks: Mapping[str, bool],
) -> list[dict[str, Any]]:
    """Pair every recomputed object with its recorded value and say where that value lives.

    `audit` rows compare against the audit JSON. The native block rows and the full floor
    calibration are not in it, so they compare against the native result and the cells header,
    whose sha256s the audit binds and whose agreement with its own recomputation it checked.
    """

    recorded = audit["recomputed"]
    pairs: list[tuple[str, str, Any, Any]] = [
        ("audit", f"target.{key}", value, recorded["target"][key])
        for key, value in target.items()
    ]
    pairs += [
        ("audit", f"instrument_endpoints.{key}", value, recorded["instrument_endpoints"][key])
        for key, value in endpoints.items()
    ]
    pairs += [
        ("audit", f"gates.{name}.passed", gates[name], audit["gates"][name]["passed"])
        for name in RECOMPUTED_GATES
    ]
    pairs += [
        ("audit", f"independent_checks.{name}", native_checks[name],
         audit["independent_checks"][name])
        for name in NATIVE_AUDIT_CHECKS
    ]
    pairs += [
        ("audit", "final_replication_classification", classification,
         audit["final_replication_classification"]),
        ("audit", "counts.blocks", target["n_blocks"], audit["counts"]["blocks"]),
        ("audit", "counts.cell_rows", len(cells["cell_rows"]), audit["counts"]["cell_rows"]),
        ("audit", "counts.native_rows", len(native["native_rows"]),
         audit["counts"]["native_rows"]),
        ("native_result", "native_result.native_block_rows", native_blocks,
         native["native_block_rows"]),
        ("native_result", "native_result.floor_calibration", floors,
         native["floor_calibration"]),
        ("cells_header", "cells_header.native_floor_calibration", floors,
         cells["native_floor_calibration"]),
    ]
    rows = []
    for against, name, value, expected in pairs:
        value = _json_value(value)
        rows.append({
            "field": name,
            "against": against,
            "bitwise_identical": value == expected,
            "within_audit_tolerance": _close(value, expected),
            "max_abs_deviation": _max_abs_deviation(value, expected),
        })
    return rows


def recompute(
    audit: Mapping[str, Any], cells: Mapping[str, Any], native: Mapping[str, Any]
) -> dict[str, Any]:
    """Every recomputed object, and its comparison rows, from the raw cell and native rows."""

    blocks = reduce_cells(cells)
    native_blocks = reduce_native(native, cells)
    floors = calibrate_native_floors(native_blocks)
    native_checks = native_audit_checks(native, cells, native_blocks, floors)
    target = target_family(blocks)
    endpoints, gates = instrument(blocks, native_blocks, target, floors)
    all_gates = all(gates.values()) and all(
        audit["gates"][name]["passed"] is True for name in ATTESTED_GATES
    )
    classification = classify(target, all_gates)
    comparisons = compare_with_audit(
        audit=audit,
        native=native,
        cells=cells,
        target=target,
        endpoints=endpoints,
        gates=gates,
        classification=classification,
        native_blocks=native_blocks,
        floors=floors,
        native_checks=native_checks,
    )
    return {
        "target": target,
        "endpoints": endpoints,
        "gates": gates,
        "classification": classification,
        "floors": floors,
        "native_checks": native_checks,
        "comparisons": comparisons,
    }


def reproduce(
    *, cells_path: Path, native_path: Path, audit_path: Path, audit_manifest_path: Path
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text())
    audit_manifest = json.loads(audit_manifest_path.read_text())
    cells, cells_sha256, cells_size = load_archive(cells_path)
    native, native_sha256, native_size = load_archive(native_path)
    bindings = verify_bindings(
        audit=audit,
        audit_path=audit_path,
        audit_manifest=audit_manifest,
        cells=cells,
        cells_sha256=cells_sha256,
        cells_size=cells_size,
        native=native,
        native_sha256=native_sha256,
        native_size=native_size,
    )
    failed = [name for name, passed in bindings.items() if not passed]
    _require(not failed, f"binding checks failed before any arithmetic: {failed}")

    result = recompute(audit, cells, native)
    target, endpoints, gates = result["target"], result["endpoints"], result["gates"]
    classification, floors = result["classification"], result["floors"]
    comparisons = result["comparisons"]
    if not all(row["within_audit_tolerance"] for row in comparisons):
        status = "MISMATCH"
    elif all(row["bitwise_identical"] for row in comparisons):
        status = "REPRODUCED_BITWISE"
    else:
        status = "REPRODUCED_WITHIN_AUDIT_TOLERANCE"
    directions = target["directions"]["endpoints"]
    pseudo = target["pseudo_phase"]
    return {
        "schema": (
            "gfmcd.splicebert_exon_edge_compatibility."
            "crossed_direct_edge_replication_reproduction.v1"
        ),
        "status": status,
        "claim_id": "C43",
        "model_free": True,
        "cells_raw_sha256": cells_sha256,
        "cells_raw_size_bytes": cells_size,
        "native_raw_sha256": native_sha256,
        "native_raw_size_bytes": native_size,
        "audit_sha256": sha256_file(audit_path),
        "binding_checks": bindings,
        "transcribed_from": {
            "audit_manifest_pinned": TRANSCRIBED_FROM,
            "freeze_package_pinned_not_checked_here": TRANSCRIBED_FROM_FREEZE_PACKAGE,
        },
        "headline": {
            "native_source_c_ac": {
                "mean": endpoints["native_source_c_ac"]["mean"],
                "ci95": endpoints["native_source_c_ac"]["ci95"],
            },
            "native_floors": {
                "donor": floors["donor_margin_intact"]["floor"],
                "acceptor": floors["acceptor_margin_intact"]["floor"],
            },
            "psi": {
                "mean": target["primary_psi"]["mean"],
                "ci95": target["primary_psi"]["ci95"],
                "mde80": target["primary_psi"]["mde80"],
            },
            "m0": target["matching_means"]["m0"],
            "m_plus": target["matching_means"]["m_plus"],
            "m_minus": target["matching_means"]["m_minus"],
            "directional_simultaneous_lower95": {
                name: directions[name]["simultaneous_lower95"] for name in DIRECTION_NAMES
            },
            "pseudo_register_controls": len(pseudo["control_means"]),
            "maximum_control": pseudo["maximum_control"],
            "exact_add_one_one_sided_p": pseudo["exact_add_one_one_sided_p"],
            "classification": classification,
        },
        "recomputed_gates": gates,
        "attested_gates_taken_from_audit": {
            name: audit["gates"][name]["passed"] for name in ATTESTED_GATES
        },
        "recomputed_native_audit_checks": result["native_checks"],
        "not_recomputed": NOT_RECOMPUTED,
        "comparison": {
            "against": {
                "audit": (
                    "crossed_direct_edge_replication_audit.json "
                    "recomputed/gates/independent_checks/counts"
                ),
                "native_result": (
                    "the native result's own block rows and floor calibration "
                    "(audit native_result_sha256)"
                ),
                "cells_header": (
                    "the cells header's copy of the floor calibration "
                    "(audit cells_result_sha256)"
                ),
            },
            "pass_criterion": "every field within the audit's _close tolerance (rel/abs 1e-10)",
            "fields": len(comparisons),
            "bitwise_identical_fields": sum(row["bitwise_identical"] for row in comparisons),
            "max_abs_deviation": max(row["max_abs_deviation"] for row in comparisons),
            "rows": comparisons,
        },
        "target": target,
        "instrument_endpoints": endpoints,
        "native_floor_calibration": floors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cells", type=Path, default=CELLS_ARCHIVE,
                        help="raw cell rows, gzip-archived or plain JSON")
    parser.add_argument("--native", type=Path, default=NATIVE_ARCHIVE,
                        help="raw native-source rows, gzip-archived or plain JSON")
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--audit-manifest", type=Path, default=AUDIT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True,
                        help="new reproduction JSON; a sibling .manifest.json is written")
    args = parser.parse_args(argv)
    out = guard_output_path(args.out.resolve())
    manifest_path = guard_output_path(out.with_suffix(".manifest.json"))

    result = reproduce(
        cells_path=args.cells.resolve(),
        native_path=args.native.resolve(),
        audit_path=args.audit.resolve(),
        audit_manifest_path=args.audit_manifest.resolve(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    manifest = ExperimentManifest.create(
        "splicebert-exon-edge-compatibility:fresh-crossed-replication-reproduction",
        sys.argv if argv is None else [__file__, *argv],
        repo=REPO,
        status=result["status"],
        seeds={
            "bootstrap": BOOTSTRAP_SEED,
            "pseudo_first": PSEUDO_SEEDS[0],
            "pseudo_last": PSEUDO_SEEDS[-1],
        },
        config={
            "role": ROLE,
            "blocks": EXPECTED_BLOCKS,
            "draws": BOOTSTRAP_DRAWS,
            "pseudo_controls": len(PSEUDO_SEEDS),
            "model_free": True,
            "numpy": np.__version__,
        },
        notes=[
            "Model-free re-execution of C43's target family, gates 2-8 and native floors "
            "from the raw cell and native rows (R49 and its addendum).",
            "Stop-load diagnostics and gates 1/9/10 are not recomputed.",
        ],
    )
    for path in (
        args.cells.resolve(),
        args.native.resolve(),
        args.audit.resolve(),
        args.audit_manifest.resolve(),
        *(REPO / relative for relative in PINNED_CODE),
        Path(__file__).resolve(),
    ):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    comparison = result["comparison"]
    print(json.dumps({
        "status": result["status"],
        "headline": result["headline"],
        "fields_compared": comparison["fields"],
        "bitwise_identical_fields": comparison["bitwise_identical_fields"],
        "max_abs_deviation": comparison["max_abs_deviation"],
        "out": str(out),
    }, indent=2, sort_keys=True))
    return 0 if result["status"] != "MISMATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
