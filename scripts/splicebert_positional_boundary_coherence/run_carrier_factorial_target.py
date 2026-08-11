#!/usr/bin/env python
"""Run the sealed base-checkpoint P3 or P3×P10 boundary-coherence target.

The runner consumes only the new PBC cohort schema.  It never imports the
archived exon-edge implementation.  Model access is dependency-injectable for
tests and alternative compatible SpliceBERT checkpoints.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from hashlib import sha256
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import DEFAULT_MODEL, pick_device  # noqa: E402
from gfmcd.splicing.pbc_carrier_factorial import (  # noqa: E402
    MIN_STRUCTURED_CONTROLS,
    build_phase_control_family,
    build_world_tables,
    block_psi,
    c43_margin,
    coupling_from_scores,
    mechanical_audit,
    verify_target_seal,
)


COHORT_SCHEMA = "gfmcd.positional_boundary_coherence.cohort.v1"
TARGET_SCHEMA = "gfmcd.pbc.carrier_factorial_target.v1"
TARGET_SHARD_SCHEMA = "gfmcd.pbc.carrier_factorial_target_shard.v1"
COMPACT_EVIDENCE_SCHEMA = "gfmcd.pbc.carrier_factorial_compact_blocks.v1"
SEQUENCE_NT = 1022
TOKENIZED_LENGTH = 1024
DONOR_POSITION_ZERO_BASED = 392
ACCEPTOR_POSITION_ZERO_BASED = 512
BRIDGE_ACCEPTOR_POSITION_ZERO_BASED = 437
CONFIRMATION_BLOCKS = 600


class CellScorer(Protocol):
    """Minimum runtime interface; fakes can implement this without torch."""

    def position_table(self) -> np.ndarray: ...

    def score_conditions(
        self,
        cells: Sequence[Mapping[str, Any]],
        tables: Mapping[str, np.ndarray],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]: ...

    def metadata(self) -> Mapping[str, Any]: ...


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    return output, manifest


def _sequence_field(cell: Mapping[str, Any], state: str) -> str:
    field = {
        "intact": "sequence_intact",
        "ac": "sequence_acceptor_ac",
        "at": "sequence_acceptor_at",
    }[state]
    sequence = str(cell[field]).upper().replace("U", "T")
    if not sequence or set(sequence) - set("ACGTN"):
        raise ValueError(f"invalid nucleotide sequence in {field}")
    return sequence


def _split_cells(
    cohort: Mapping[str, Any], split_role: str
) -> tuple[Mapping[str, Any], ...]:
    if cohort.get("schema") != COHORT_SCHEMA:
        raise RuntimeError("wrong PBC cohort schema")
    cells = tuple(
        row
        for row in cohort.get("cells", [])
        if str(row.get("split_role")) == split_role
    )
    if not cells:
        raise RuntimeError(f"cohort has no {split_role!r} cells")
    by_block: dict[str, list[Mapping[str, Any]]] = {}
    for row in cells:
        by_block.setdefault(str(row["block_id"]), []).append(row)
    expected_pairs = {(donor, acceptor) for donor in range(3) for acceptor in range(3)}
    for block_id, rows in by_block.items():
        pairs = {
            (int(row["donor_phase"]), int(row["acceptor_phase"])) for row in rows
        }
        if len(rows) != 9 or pairs != expected_pairs:
            raise RuntimeError(f"block {block_id} is not a complete 3x3 lattice")
        if len({str(row["chrom"]) for row in rows}) != 1 or len(
            {str(row["strand"]) for row in rows}
        ) != 1:
            raise RuntimeError(f"block {block_id} changes sampling stratum")
    for row in cells:
        intact = _sequence_field(row, "intact")
        ac = _sequence_field(row, "ac")
        at = _sequence_field(row, "at")
        if not (len(intact) == len(ac) == len(at) == SEQUENCE_NT):
            raise RuntimeError(
                f"target sequences must contain {SEQUENCE_NT} nucleotides / "
                f"{TOKENIZED_LENGTH} tokens with specials"
            )
        donor = int(row["donor_pos"])
        acceptor = int(row["acceptor_pos"])
        if (
            donor != DONOR_POSITION_ZERO_BASED
            or acceptor != ACCEPTOR_POSITION_ZERO_BASED
            or int(row.get("bridge_acceptor_pos", -1))
            != BRIDGE_ACCEPTOR_POSITION_ZERO_BASED
            or acceptor - donor != 120
            or int(row.get("span_nt", -1)) != 120
        ):
            raise RuntimeError(
                "frozen Python geometry requires donor_pos=392 and "
                "acceptor_pos=512 (donor token 393; AG at nts [510,512))"
            )
        if intact[donor : donor + 2] != "GT":
            raise RuntimeError("target donor position does not point to canonical GT")
        if intact[
            BRIDGE_ACCEPTOR_POSITION_ZERO_BASED - 2 : BRIDGE_ACCEPTOR_POSITION_ZERO_BASED
        ] != "AG":
            raise RuntimeError("target fixed bridge position does not point to AG")
        if (
            intact[acceptor - 2 : acceptor] != "AG"
            or ac[acceptor - 2 : acceptor] != "AC"
            or at[acceptor - 2 : acceptor] != "AT"
        ):
            raise RuntimeError("true-acceptor AG/AC/AT operators are misplaced")
        if sum(left != right for left, right in zip(intact, ac, strict=True)) != 1:
            raise RuntimeError("AG->AC state must change exactly one nucleotide")
        if sum(left != right for left, right in zip(intact, at, strict=True)) != 1:
            raise RuntimeError("AG->AT state must change exactly one nucleotide")
    return cells


def _file_pin_failures(pinned: Mapping[str, Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for label, record in pinned.items():
        path = Path(str(record.get("path", ""))).expanduser()
        expected = str(record.get("sha256", ""))
        if not path.is_file():
            failures.append(f"{label}:missing")
        elif not expected or sha256_file(path) != expected:
            failures.append(f"{label}:sha256")
    return failures


def _provided_file_is_pinned(
    path: str | Path, pinned: Mapping[str, Mapping[str, Any]]
) -> bool:
    resolved = Path(path).expanduser().resolve()
    digest = sha256_file(resolved)
    return any(
        Path(str(record.get("path", ""))).expanduser().resolve() == resolved
        and str(record.get("sha256", "")) == digest
        for record in pinned.values()
    )


def _require_sealed_output_path(
    seal: Mapping[str, Any],
    output: Path,
    *,
    shard_index: int,
    shard_count: int,
    assembling: bool,
) -> None:
    plan = seal.get("target_outputs", {})
    sealed_count = int(plan.get("shard_count", -1))
    if assembling:
        expected = Path(str(plan.get("assembled", ""))).expanduser().resolve()
        if output.resolve() != expected:
            raise RuntimeError("assembled target output path differs from the audited seal")
        return
    if shard_count != sealed_count:
        raise RuntimeError("target shard count differs from the audited seal")
    shards = plan.get("shards")
    if (
        not isinstance(shards, list)
        or shard_index not in range(sealed_count)
        or output.resolve()
        != Path(str(shards[shard_index])).expanduser().resolve()
    ):
        raise RuntimeError("target shard output path differs from the audited seal")


def _require_runtime_revision(seal: Mapping[str, Any], repo: Path = REPO) -> None:
    """Refuse target execution outside the clean revision frozen in the seal."""

    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    )
    if status.strip():
        raise RuntimeError("sealed target requires a clean tracked worktree")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != str(seal.get("frozen_revision", "")):
        raise RuntimeError("runtime Git revision differs from the audited target seal")


def _raw_endpoint_stream_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = sha256()
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["condition_id"]),
            str(item["block_id"]),
            int(item["donor_phase"]),
            int(item["acceptor_phase"]),
        ),
    ):
        record = (
            str(row["condition_id"]),
            str(row["block_id"]),
            str(row["cell_id"]),
            int(row["donor_phase"]),
            int(row["acceptor_phase"]),
            float(row["coupling_ac"]),
            float(row["coupling_at"]),
            float(row["f_intact"]),
            float(row["f_ac"]),
            float(row["f_at"]),
        )
        digest.update(
            json.dumps(record, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def compact_target_result(
    raw: Mapping[str, Any],
    *,
    raw_shard_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Reduce raw condition/cell rows to the §11.4 sufficient endpoint table."""

    rows = raw.get("rows")
    specs = raw.get("condition_specs")
    if not isinstance(rows, list) or not rows or not isinstance(specs, Mapping):
        raise RuntimeError("compact assembly requires complete raw condition rows")
    by_condition_block: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        str(condition): defaultdict(list) for condition in specs
    }
    for row in rows:
        condition = str(row.get("condition_id"))
        if condition not in by_condition_block:
            raise RuntimeError("raw row references an unknown condition")
        by_condition_block[condition][str(row["block_id"])].append(row)
    arithmetic_error = 0.0
    for row in rows:
        numeric = np.asarray(
            [
                row["f_intact"], row["f_ac"], row["f_at"],
                row["coupling_ac"], row["coupling_at"],
            ],
            dtype=float,
        )
        if not np.isfinite(numeric).all():
            raise RuntimeError("raw target endpoint stream is non-finite")
        arithmetic_error = max(
            arithmetic_error,
            abs(float(numeric[3] - (numeric[0] - numeric[1]))),
            abs(float(numeric[4] - (numeric[0] - numeric[2]))),
        )
    if arithmetic_error > 1e-10:
        raise RuntimeError("raw coupling arithmetic is inconsistent before compaction")
    block_ids = tuple(sorted(by_condition_block.get("Y00", {})))
    if len(block_ids) != int(raw.get("n_blocks", -1)):
        raise RuntimeError("raw target block count is incomplete")
    if any(tuple(sorted(blocks)) != block_ids for blocks in by_condition_block.values()):
        raise RuntimeError("raw target conditions change block support")

    compact_blocks: list[dict[str, Any]] = []
    identity_by_block: dict[str, tuple[tuple[Any, ...], ...]] = {}
    for block_id in block_ids:
        y00_rows = sorted(
            by_condition_block["Y00"][block_id],
            key=lambda row: (int(row["donor_phase"]), int(row["acceptor_phase"])),
        )
        if len(y00_rows) != 9:
            raise RuntimeError("Y00 compact evidence requires nine phase cells per block")
        chroms = {str(row["chrom"]) for row in y00_rows}
        strands = {str(row["strand"]) for row in y00_rows}
        if len(chroms) != 1 or len(strands) != 1:
            raise RuntimeError("Y00 compact block changes sampling stratum")
        identity_by_block[block_id] = tuple(
            (
                str(row["cell_id"]),
                int(row["donor_phase"]),
                int(row["acceptor_phase"]),
                str(row["chrom"]),
                str(row["strand"]),
            )
            for row in y00_rows
        )
        compact_blocks.append(
            {
                "block_id": block_id,
                "chrom": next(iter(chroms)),
                "strand": next(iter(strands)),
                "y00_cells": [
                    {
                        "cell_id": str(row["cell_id"]),
                        "donor_phase": int(row["donor_phase"]),
                        "acceptor_phase": int(row["acceptor_phase"]),
                        "coupling_ac": float(row["coupling_ac"]),
                        "coupling_at": float(row["coupling_at"]),
                    }
                    for row in y00_rows
                ],
            }
        )

    condition_endpoints: dict[str, Any] = {}
    for condition, spec in specs.items():
        if condition == "Y00":
            continue
        psi_ac: list[float] = []
        psi_at: list[float] = []
        keep_at = str(spec["family"]) == "target"
        for block_id in block_ids:
            block_rows = sorted(
                by_condition_block[str(condition)][block_id],
                key=lambda row: (
                    int(row["donor_phase"]), int(row["acceptor_phase"])
                ),
            )
            if len(block_rows) != 9:
                raise RuntimeError(
                    f"condition {condition} lacks a complete phase lattice"
                )
            identity = tuple(
                (
                    str(row["cell_id"]),
                    int(row["donor_phase"]),
                    int(row["acceptor_phase"]),
                    str(row["chrom"]),
                    str(row["strand"]),
                )
                for row in block_rows
            )
            if identity != identity_by_block[block_id]:
                raise RuntimeError(
                    f"condition {condition} changes compact cell identity support"
                )
            psi_ac.append(float(block_psi(block_rows, value_key="coupling_ac")["Psi"]))
            if keep_at:
                psi_at.append(
                    float(block_psi(block_rows, value_key="coupling_at")["Psi"])
                )
        condition_endpoints[str(condition)] = {
            "psi_ac": psi_ac,
            **({"psi_at": psi_at} if keep_at else {}),
        }

    compact = {
        "schema": COMPACT_EVIDENCE_SCHEMA,
        "block_order": list(block_ids),
        "blocks": compact_blocks,
        "condition_endpoints": condition_endpoints,
        "raw_row_count": len(rows),
        "raw_endpoint_stream_sha256": _raw_endpoint_stream_sha256(rows),
        "raw_coupling_arithmetic_max_abs": float(arithmetic_error),
        "raw_identity_support_verified": True,
        "raw_shards": [dict(record) for record in raw_shard_records],
        "normalization": {
            "Y00": "nine cell couplings retained for pseudo-phase and directions",
            "target_worlds": "per-block Psi for AG->AC and AG->AT",
            "structured_nulls": "per-block AG->AC Psi only",
        },
    }
    result = {key: value for key, value in raw.items() if key not in {"rows", "shard"}}
    result["schema"] = TARGET_SCHEMA
    result["status"] = "TARGET_SCORED_UNANALYZED"
    result["compact_evidence"] = compact
    return result


class SpliceBertCellScorer:
    """Default official-checkpoint scorer for C43-compatible cell states."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 4,
        eager: bool = False,
    ) -> None:
        import torch

        from gfmcd.splicing.circuit import load_adapter

        self.torch = torch
        self.adapter = load_adapter(
            str(Path(model_path).expanduser().resolve()),
            pick_device(device),
            eager=eager,
        )
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        self.embedding = self.adapter.model.bert.embeddings.position_embeddings
        self._original_tensor = self.embedding.weight.detach().clone()
        self._original = (
            self._original_tensor.detach().cpu().to(dtype=torch.float64).numpy()
        )
        if self._original.shape != (1026, 512):
            raise RuntimeError(
                "expected SpliceBERT.1024nt position table (1026,512), "
                f"found {self._original.shape}"
            )
        self.mask_token_id = int(self.adapter.tokenizer.mask_token_id)
        self.base_ids = {
            base: int(self.adapter.tokenizer.convert_tokens_to_ids(base))
            for base in "ACGT"
        }
        if len(set(self.base_ids.values())) != 4:
            raise RuntimeError("tokenizer does not expose distinct A/C/G/T tokens")

    def position_table(self) -> np.ndarray:
        return np.array(self._original, copy=True)

    def metadata(self) -> Mapping[str, Any]:
        return self.adapter.metadata().__dict__

    @contextmanager
    def _using_table(self, table: np.ndarray) -> Iterator[None]:
        torch = self.torch
        values = np.asarray(table, dtype=np.float64)
        if values.shape != self._original.shape or not np.isfinite(values).all():
            raise ValueError("invalid intervention table")
        tensor = torch.as_tensor(
            values,
            dtype=self.embedding.weight.dtype,
            device=self.embedding.weight.device,
        )
        with torch.no_grad():
            self.embedding.weight.copy_(tensor)
        try:
            yield
        finally:
            with torch.no_grad():
                self.embedding.weight.copy_(self._original_tensor)
            if not torch.equal(self.embedding.weight, self._original_tensor):
                raise RuntimeError("checkpoint position table was not restored")

    def _score_state(
        self, cells: Sequence[Mapping[str, Any]], state: str
    ) -> list[float]:
        torch = self.torch
        values: list[float] = []
        for start in range(0, len(cells), self.batch_size):
            chunk = cells[start : start + self.batch_size]
            sequences = [_sequence_field(cell, state) for cell in chunk]
            batch = self.adapter.tokenize(sequences)
            if batch.input_ids.shape[1] != TOKENIZED_LENGTH or not bool(
                self.torch.all(batch.attention_mask.sum(dim=1) == TOKENIZED_LENGTH)
            ):
                raise RuntimeError(
                    f"tokenizer must emit exactly {TOKENIZED_LENGTH} unpadded tokens"
                )
            positions = [int(cell["donor_pos"]) for cell in chunk]
            for index, position in enumerate(positions):
                batch.input_ids[
                    index, self.adapter.nucleotide_to_token(position)
                ] = self.mask_token_id
            with torch.no_grad():
                raw = self.adapter.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    output_attentions=False,
                    return_dict=True,
                )
            for index, position in enumerate(positions):
                logits = raw.logits[
                    index, self.adapter.nucleotide_to_token(position)
                ].float()
                values.append(
                    c43_margin(
                        {
                            base: float(logits[token_id].detach().cpu())
                            for base, token_id in self.base_ids.items()
                        }
                    )
                )
        return values

    def score_conditions(
        self,
        cells: Sequence[Mapping[str, Any]],
        tables: Mapping[str, np.ndarray],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        output: dict[str, list[dict[str, Any]]] = {}
        for condition, table in tables.items():
            with self._using_table(table):
                by_state = {
                    state: self._score_state(cells, state)
                    for state in ("intact", "ac", "at")
                }
            rows: list[dict[str, Any]] = []
            for index, cell in enumerate(cells):
                f_intact = by_state["intact"][index]
                f_ac = by_state["ac"][index]
                f_at = by_state["at"][index]
                rows.append(
                    {
                        "cell_id": str(cell["cell_id"]),
                        "block_id": str(cell["block_id"]),
                        "chrom": str(cell["chrom"]),
                        "strand": str(cell["strand"]),
                        "donor_phase": int(cell["donor_phase"]),
                        "acceptor_phase": int(cell["acceptor_phase"]),
                        "f_intact": f_intact,
                        "f_ac": f_ac,
                        "f_at": f_at,
                        "coupling_ac": coupling_from_scores(f_intact, f_ac),
                        "coupling_at": coupling_from_scores(f_intact, f_at),
                    }
                )
            output[condition] = rows
        return output


def load_scorer(
    specification: str | None,
    *,
    model_path: str | Path,
    device: str | None,
    batch_size: int,
    eager: bool = False,
) -> CellScorer:
    """Load ``module:factory`` or the default official-checkpoint scorer."""

    if not specification:
        return SpliceBertCellScorer(
            model_path, device=device, batch_size=batch_size, eager=eager
        )
    try:
        module_name, attribute = specification.split(":", 1)
    except ValueError as exc:
        raise ValueError("scorer must be MODULE:FACTORY") from exc
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        eager=eager,
    )


def execute_target(
    *,
    seal: Mapping[str, Any],
    cohort: Mapping[str, Any],
    scorer: CellScorer,
    split_role: str,
    n_controls: int,
    control_seed: int,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Execute the sealed target with no inferential interpretation."""

    verify_target_seal(seal)
    if split_role != seal["authorized_split_role"] or split_role != "confirmation":
        raise RuntimeError("Phase-II target may read only the confirmation role")
    if n_controls != MIN_STRUCTURED_CONTROLS:
        raise ValueError(
            f"exactly {MIN_STRUCTURED_CONTROLS} controls per family are required"
        )
    if n_controls != int(seal["minimum_structured_controls_per_family"]):
        raise RuntimeError("target control count differs from the sealed value")
    if int(control_seed) != int(seal["structured_control_seed"]):
        raise RuntimeError("target structured-control seed differs from the seal")
    pin_failures = _file_pin_failures(seal.get("pinned_files", {}))
    if pin_failures:
        raise RuntimeError(f"target pin verification failed: {pin_failures}")
    all_cells = _split_cells(cohort, split_role)
    all_block_ids = tuple(sorted({str(cell["block_id"]) for cell in all_cells}))
    if len(all_block_ids) != CONFIRMATION_BLOCKS:
        raise RuntimeError(
            f"confirmation target requires exactly {CONFIRMATION_BLOCKS} complete blocks"
        )
    if shard_count < 1 or shard_index not in range(shard_count):
        raise ValueError("shard index must lie in [0, shard_count)")
    quotient, remainder = divmod(len(all_block_ids), shard_count)
    start = shard_index * quotient + min(shard_index, remainder)
    stop = start + quotient + (1 if shard_index < remainder else 0)
    shard_block_ids = all_block_ids[start:stop]
    if not shard_block_ids:
        raise RuntimeError("requested target shard contains no complete block")
    shard_support = set(shard_block_ids)
    cells = tuple(
        cell for cell in all_cells if str(cell["block_id"]) in shard_support
    )
    n_blocks = len(shard_block_ids)
    original = scorer.position_table()
    doses = seal["selected_doses"]
    p3_dose = float(doses["p3"])
    p10_dose = (
        float(doses["p10"])
        if seal["target_mode"] == "FULL_FACTORIAL"
        else None
    )
    worlds = build_world_tables(
        original, p3_dose=p3_dose, p10_dose=p10_dose
    )
    controls, control_specs = build_phase_control_family(
        original,
        p3_dose=p3_dose,
        p10_dose=p10_dose,
        n_controls=n_controls,
        seed=control_seed,
    )
    frozen_hashes = seal["structured_control_hashes"]
    if set(frozen_hashes) != set(controls):
        raise RuntimeError("target structured-control labels differ from the seal")
    for label, table in controls.items():
        expected = frozen_hashes[label]
        observed_hash = sha256(np.ascontiguousarray(table).tobytes()).hexdigest()
        if (
            observed_hash != str(expected.get("sha256"))
            or str(control_specs[label]["family"]) != str(expected.get("family"))
            or int(control_specs[label]["control_id"])
            != int(expected.get("control_id"))
            or str(control_specs[label]["world"]) != str(expected.get("world"))
        ):
            raise RuntimeError(f"structured-control freeze mismatch for {label}")
    mechanics = mechanical_audit(
        original,
        p3_dose=p3_dose,
        p10_dose=p10_dose,
        controls=controls,
        control_specs=control_specs,
    )
    if not mechanics["passed"]:
        raise RuntimeError("sealed target mechanical audit failed")
    tables = {**worlds, **controls}
    scored = scorer.score_conditions(cells, tables)
    if set(scored) != set(tables):
        raise RuntimeError("scorer did not return every requested condition")
    condition_specs: dict[str, dict[str, Any]] = {
        world: {"family": "target", "control_id": None, "world": world}
        for world in worlds
    }
    condition_specs.update(control_specs)
    rows: list[dict[str, Any]] = []
    expected_cells = len(cells)
    for condition in tables:
        condition_rows = list(scored[condition])
        if len(condition_rows) != expected_cells:
            raise RuntimeError(f"condition {condition} returned wrong row count")
        spec = condition_specs[condition]
        for row in condition_rows:
            rows.append(
                {
                    "condition_id": condition,
                    "family": spec["family"],
                    "control_id": spec.get("control_id"),
                    "world": spec["world"],
                    **dict(row),
                }
            )
    result = {
        "schema": TARGET_SCHEMA if shard_count == 1 else TARGET_SHARD_SCHEMA,
        "stage": "target",
        "status": (
            "TARGET_SCORED_UNANALYZED"
            if shard_count == 1
            else "TARGET_SHARD_SCORED_UNANALYZED"
        ),
        "target_mode": seal["target_mode"],
        "target_seal_sha256": seal["seal_sha256"],
        "split_role": split_role,
        "n_blocks": n_blocks,
        "total_confirmation_blocks": len(all_block_ids),
        "n_cells": len(cells),
        "n_controls_per_family": n_controls,
        "control_seed": control_seed,
        "selected_doses": dict(doses),
        "bands": dict(seal["bands"]),
        "scorer": dict(scorer.metadata()),
        "mechanical_audit": mechanics,
        "condition_specs": condition_specs,
        "rows": rows,
        "shard": {
            "index": int(shard_index),
            "count": int(shard_count),
            "block_start": int(start),
            "block_stop": int(stop),
            "block_ids": list(shard_block_ids),
            "partition_algorithm": "sorted_block_ids_contiguous_balanced_v1",
        },
    }
    return compact_target_result(result) if shard_count == 1 else result


def assemble_target_shards(
    shards: Sequence[Mapping[str, Any]],
    *,
    shard_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble immutable shards only after exact complete-support validation."""

    if not shards:
        raise ValueError("at least one target shard is required")
    ordered = sorted(shards, key=lambda row: int(row.get("shard", {}).get("index", -1)))
    shard_count = int(ordered[0].get("shard", {}).get("count", -1))
    if shard_count != len(ordered) or [int(row["shard"]["index"]) for row in ordered] != list(
        range(shard_count)
    ):
        raise RuntimeError("target shards must contain every index exactly once")
    invariant_keys = (
        "stage",
        "target_mode",
        "target_seal_sha256",
        "split_role",
        "n_controls_per_family",
        "control_seed",
        "selected_doses",
        "bands",
        "scorer",
        "mechanical_audit",
        "condition_specs",
        "total_confirmation_blocks",
        "source_cohort_sha256",
        "source_seal_file_sha256",
    )
    reference = ordered[0]
    for shard in ordered:
        if (
            shard.get("schema") != TARGET_SHARD_SCHEMA
            or shard.get("status") != "TARGET_SHARD_SCORED_UNANALYZED"
            or any(shard.get(key) != reference.get(key) for key in invariant_keys)
            or shard.get("shard", {}).get("count") != shard_count
            or shard.get("shard", {}).get("partition_algorithm")
            != "sorted_block_ids_contiguous_balanced_v1"
        ):
            raise RuntimeError("target shard invariant mismatch")
    block_ids = [
        str(block_id)
        for shard in ordered
        for block_id in shard["shard"]["block_ids"]
    ]
    if (
        len(block_ids) != CONFIRMATION_BLOCKS
        or len(set(block_ids)) != CONFIRMATION_BLOCKS
        or block_ids != sorted(block_ids)
        or int(reference["total_confirmation_blocks"]) != CONFIRMATION_BLOCKS
    ):
        raise RuntimeError("target shards are overlapping, incomplete, or out of order")
    rows = [row for shard in ordered for row in shard.get("rows", [])]
    n_cells = sum(int(shard.get("n_cells", -1)) for shard in ordered)
    if n_cells != CONFIRMATION_BLOCKS * 9:
        raise RuntimeError("assembled target does not contain 9 cells per block")
    expected_per_condition = n_cells
    counts = Counter(str(row.get("condition_id")) for row in rows)
    if set(counts) != set(reference["condition_specs"]) or any(
        count != expected_per_condition for count in counts.values()
    ):
        raise RuntimeError("assembled conditions do not have complete cell support")
    by_condition: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        condition = str(row["condition_id"])
        cell_id = str(row["cell_id"])
        if cell_id in by_condition[condition]:
            raise RuntimeError("duplicate condition/cell row across target shards")
        by_condition[condition].add(cell_id)
    support = next(iter(by_condition.values()))
    if any(values != support for values in by_condition.values()):
        raise RuntimeError("assembled condition cell supports differ")
    result = {
        key: reference[key]
        for key in invariant_keys
        if key not in {"total_confirmation_blocks"}
    }
    result.update(
        {
            "schema": TARGET_SCHEMA,
            "status": "TARGET_SCORED_UNANALYZED",
            "n_blocks": CONFIRMATION_BLOCKS,
            "total_confirmation_blocks": CONFIRMATION_BLOCKS,
            "n_cells": n_cells,
            "rows": rows,
            "assembly": {
                "status": "COMPLETE_SUPPORT_VERIFIED",
                "n_shards": shard_count,
                "shard_indices": list(range(shard_count)),
                "block_ids": block_ids,
                "partition_algorithm": "sorted_block_ids_contiguous_balanced_v1",
                "source_shards": list(shard_records or []),
            },
        }
    )
    return compact_target_result(
        result, raw_shard_records=list(shard_records or [])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", required=True)
    parser.add_argument("--cohort", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--assemble-shards", nargs="+", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--split-role", default="confirmation")
    parser.add_argument("--scorer", default=None, help="optional MODULE:FACTORY")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--controls", type=int, default=MIN_STRUCTURED_CONTROLS)
    parser.add_argument("--control-seed", type=int, default=20260810)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output, manifest_path = output_paths(args.out)
    seal_path = Path(args.seal).expanduser().resolve()
    seal = json.loads(seal_path.read_text())
    verify_target_seal(seal)
    _require_runtime_revision(seal)
    _require_sealed_output_path(
        seal,
        output,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        assembling=bool(args.assemble_shards),
    )
    input_paths: list[Path] = [seal_path]
    if args.assemble_shards:
        if args.cohort is not None or args.model is not None:
            parser.error("--assemble-shards cannot be combined with --cohort or --model")
        shard_paths = [Path(value).expanduser().resolve() for value in args.assemble_shards]
        shards = [json.loads(path.read_text()) for path in shard_paths]
        result = assemble_target_shards(
            shards,
            shard_records=[file_record(path) for path in shard_paths],
        )
        if result["target_seal_sha256"] != seal["seal_sha256"]:
            raise RuntimeError("assembled target shards do not match the supplied seal")
        input_paths.extend(shard_paths)
    else:
        if args.cohort is None or args.model is None:
            parser.error("scoring requires --cohort and --model")
        cohort_path = Path(args.cohort).expanduser().resolve()
        cohort = json.loads(cohort_path.read_text())
        pinned = seal.get("pinned_files", {})
        if not _provided_file_is_pinned(cohort_path, pinned):
            raise RuntimeError("provided cohort is not the hash-pinned preflight cohort")
        model_path = Path(args.model).expanduser().resolve()
        if args.scorer is not None:
            parser.error("custom scorers are fixture-only and cannot execute a target")
        if model_path != (REPO / DEFAULT_MODEL).resolve():
            raise RuntimeError("target requires the frozen official SpliceBERT.1024nt path")
        model_names = (
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        )
        provided_model_files = tuple(
            model_path / name for name in model_names if (model_path / name).is_file()
        )
        if not provided_model_files or not all(
            _provided_file_is_pinned(path, pinned) for path in provided_model_files
        ):
            raise RuntimeError("provided model files are not the hash-pinned checkpoint")
        scorer = load_scorer(
            args.scorer,
            model_path=model_path,
            device=args.device,
            batch_size=args.batch,
        )
        result = execute_target(
            seal=seal,
            cohort=cohort,
            scorer=scorer,
            split_role=args.split_role,
            n_controls=args.controls,
            control_seed=args.control_seed,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        result["source_cohort_sha256"] = sha256_file(cohort_path)
        result["source_seal_file_sha256"] = sha256_file(seal_path)
        input_paths.append(cohort_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-carrier-factorial-target-v1",
        sys.argv,
        repo=REPO,
        status=result["status"],
        config={
            "stage": "target",
            "target_mode": result["target_mode"],
            "selected_doses": result["selected_doses"],
            "bands": result["bands"],
            "n_controls_per_family": result["n_controls_per_family"],
            "control_seed": result["control_seed"],
            "split_role": result["split_role"],
            "shard": result.get("shard"),
            "assembly": result.get("assembly", {}).get("status"),
            "compact_evidence_schema": result["compact_evidence"]["schema"],
            "raw_row_count": result["compact_evidence"]["raw_row_count"],
            "raw_endpoint_stream_sha256": result["compact_evidence"][
                "raw_endpoint_stream_sha256"
            ],
        },
        seeds={"structured_controls": int(result["control_seed"])},
    )
    for path in input_paths:
        manifest.add_input(path)
    if not args.assemble_shards:
        for record in seal.get("pinned_files", {}).values():
            path = Path(str(record.get("path", "")))
            if path.is_file() and path not in input_paths:
                manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "target_mode": result["target_mode"],
                "n_blocks": result["n_blocks"],
                "n_conditions": len(result["condition_specs"]),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
