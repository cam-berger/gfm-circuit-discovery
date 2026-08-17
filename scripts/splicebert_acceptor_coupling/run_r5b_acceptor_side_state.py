#!/usr/bin/env python
"""R5b v2: develop and freeze the upstream masked-acceptor local-writer instrument.

The primary state is the output-projected contribution of L3H1 at the masked acceptor
query.  It is captured and intervened on immediately after L3 attention, before every
L4/L5 delivery candidate.  Dense necessity/sufficiency is primary.  A low-rank ``U_A``
is optional nested compression evidence and can never gate or redefine the dense path.

R5b uses only the two residual-validation reservations.  Its outputs are instrument
development, never confirmatory mechanism evidence.  Bucket 0 derives a candidate rank;
bucket 1 validates that frozen candidate without refitting or rescanning; only then does
a mechanical union refit emit the authenticated runtime basis consumed by R5c.  No R5c
preflight or chr8/9 target may fit, select, or alter that basis.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.circuits.hf_bert import (  # noqa: E402
    add_attention_output_delta,
    capture_head_outputs,
    project_head_context_to_residual,
)
from gfmcd.splicing.acceptor_coupling_mechanism import (  # noqa: E402
    PRIMARY_LOCAL_WRITER_SITE_ID,
    deterministic_nested_random_bases,
    energy_match_rows,
    mean_squared_principal_cosine,
    local_writer_dense_estimands,
    local_writer_subspace_estimands,
    orthogonal_projector,
    projector_diagnostics,
    replay_local_writer_dense_record,
    replay_local_writer_subspace_record,
)
from gfmcd.splicing.acceptor_mechanism_cohort import (  # noqa: E402
    R5_EXPECTED_CENSUS_IDENTITY,
    R5_EXPECTED_SOURCE_IDENTITY,
    R5_ROWS_PER_BUCKET,
    R5_VALIDATION_CHROMOSOMES,
    R5CohortRow,
    build_r5_reservations,
    validate_expected_r5_census_identity,
    validate_expected_r5_source_identity,
    validate_r5_reservation_audit,
)
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    CouplingTask,
    load_adapter,
    load_examples_unscored,
    matched_decoy_donor,
    pick_device,
)
from gfmcd.splicing.r5_upstream_provenance import (  # noqa: E402
    PINNED_CHECKPOINT_AGGREGATE_SHA256,
    R5_ENTRY_SCHEMA,
    derive_r5_exclusion_registry,
    load_r3c_exclusion_package,
    load_r3d_entry_package,
    validate_r5_exclusion_registry,
)
from gfmcd.splicing.r5_stack_binding import (  # noqa: E402
    build_r5_stack_binding,
    r5_stack_fingerprint,
    validate_r5_runtime_policy,
)
RUNNER = Path(__file__).resolve()
PROTOCOL = (
    REPO
    / "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R5B_ACCEPTOR_SIDE_STATE.md"
)

R5B_SCHEMA = "gfmcd.r5b.local_writer_development/2"
R5B_INSTRUMENT_SCHEMA = "gfmcd.r5b.local_writer_instrument/2"
R5B_BASIS_CONTRACT_SCHEMA = "gfmcd.r5b.frozen_local_writer_basis/2"
R5B_BASIS_BUNDLE_SCHEMA = "gfmcd.r5b.frozen_local_writer_basis_bundle/2"
R5B_DISPOSITION_SCHEMA = "gfmcd.r5b.local_writer_basis_disposition/2"
R5B_DERIVATION_SCHEMA = "gfmcd.r5b.local_writer_basis_derivation/2"
R5B_VALIDATION_SCHEMA = "gfmcd.r5b.local_writer_basis_validation/2"
R5B_SCORED_PROVIDER_SCHEMA = "gfmcd.r5b.output_projected_writer_provider/2"
R5B_DISPOSITION_STATUSES = (
    "VALIDATED_BASIS",
    "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS",
    "BASIS_INSTRUMENT_UNRESOLVED",
)
R5B_PATH_LOCAL_C2_BY_STATUS = {
    "VALIDATED_BASIS": (
        "EXECUTE_OPTIONAL_C2_WITH_AUTHENTICATED_FROZEN_LOCAL_WRITER_BASIS"
    ),
    "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS": (
        "SKIP_OPTIONAL_C2_NO_VALIDATED_BASIS_DENSE_PRIMARY_UNAFFECTED"
    ),
    "BASIS_INSTRUMENT_UNRESOLVED": (
        "SKIP_OPTIONAL_C2_BASIS_INSTRUMENT_UNRESOLVED_DENSE_PRIMARY_UNAFFECTED"
    ),
}
R5B_RESIDUAL_IDENTITY_SCHEMA = "gfmcd.r5b.residual_attempt_identity/1"
R5B_RESIDUAL_ATTEMPT_SCHEMA = "gfmcd.r5b.residual_attempt/1"
R5B_RESIDUAL_COMPLETION_SCHEMA = "gfmcd.r5b.residual_completion/1"
R5B_POST_WO_NUMERIC_PROVENANCE_SCHEMA = (
    "gfmcd.r5b.post_wo_numeric_provenance/1"
)
R5B_OPTIONAL_BASIS_FAILURE_SCHEMA = (
    "gfmcd.r5b.optional_basis_instrument_failure/1"
)

WRITER_LAYER = 3
WRITER_HEAD = 1
WRITER_SITE_ID = PRIMARY_LOCAL_WRITER_SITE_ID
DENSE_PATH_ROLE = "primary"
SUBSPACE_PATH_ROLE = "nested_candidate_only"

DERIVATION_BUCKET = 0
VALIDATION_BUCKET = 1
RANK_GRID = (1, 2, 4, 8, 16, 32)
N_RANDOM_CONTROLS = 32
ADJACENT_TOKEN_OFFSETS = (-2, -1, 1, 2)
TRIM_FRACTION = 0.10
SESOI = 0.05
HEALTH_NONINFERIORITY_MARGIN = 0.01
FAMILY_ALPHA = 0.05
DEVELOPMENT_INFERENCE_DRAWS = 10_000
MIN_PRODUCTION_INFERENCE_DRAWS = 10_000
MIN_GENES = 30
SEED = 0
PRODUCTION_N = R5_ROWS_PER_BUCKET
PRODUCTION_BATCH = 8
PRODUCTION_INFERENCE_DRAWS = MIN_PRODUCTION_INFERENCE_DRAWS
PRODUCTION_OVERLAP_DRAWS = 10_000
POST_WO_OPERATOR_RELATIVE_TOLERANCE = 0.0
POST_WO_OPERATOR_ABSOLUTE_TOLERANCE = 1e-5

# Every upstream/result pin is paired with its manifest pin.  None authorizes nothing.
FROZEN_PROTOCOL_SHA256: str | None = None
FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = None
FROZEN_COHORT_INPUT_SHA256: str | None = None
FROZEN_COHORT_AUDIT_SHA256: str | None = None
FROZEN_R3D_ARTIFACT_SHA256: str | None = None
FROZEN_R3D_MANIFEST_SHA256: str | None = None
FROZEN_R5A_ARTIFACT_SHA256: str | None = None
FROZEN_R5A_MANIFEST_SHA256: str | None = None
FROZEN_DERIVATION_ARTIFACT_SHA256: str | None = None
FROZEN_DERIVATION_MANIFEST_SHA256: str | None = None
FROZEN_DERIVATION_COMPLETION_SHA256: str | None = None
FROZEN_VALIDATION_ARTIFACT_SHA256: str | None = None
FROZEN_VALIDATION_MANIFEST_SHA256: str | None = None
FROZEN_VALIDATION_COMPLETION_SHA256: str | None = None
FROZEN_R5B_DISPOSITION_ARTIFACT_SHA256: str | None = None
FROZEN_R5B_DISPOSITION_MANIFEST_SHA256: str | None = None
FROZEN_CHECKPOINT_AGGREGATE_SHA256: str | None = None


class TargetSealError(RuntimeError):
    """Raised before a model, genome, reservation, or target path may be opened."""


class InstrumentError(RuntimeError):
    """Raised when a local-writer algebra or authenticated handoff is invalid."""


class OptionalBasisInstrumentError(InstrumentError):
    """A prespecified low-rank-basis instrument failure, never a scorer failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


def _optional_basis_failure_record(
    phase: str, error: OptionalBasisInstrumentError
) -> dict[str, Any]:
    if phase not in {"derivation", "validation", "freeze"}:
        raise ValueError("optional-basis failure phase differs")
    return {
        "schema": R5B_OPTIONAL_BASIS_FAILURE_SCHEMA,
        "phase": phase,
        "code": error.code,
        "message": str(error),
        "optional_basis_only": True,
        "dense_primary_unaffected": True,
        "union_refit_performed": False,
    }


def canonical_sha256(value: Any) -> str:
    def encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"unsupported canonical JSON value: {type(obj)!r}")

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=encode,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: Any, *, dtype: str | None = "<f8") -> str:
    array = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.ndim}|".encode("ascii"))
    digest.update(",".join(str(int(size)) for size in array.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def output_projected_head_contribution(
    context: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    head_index: int,
    num_heads: int,
) -> torch.Tensor:
    """Return one head's exact additive contribution through ``attention.output.dense``.

    Hugging Face BERT concatenates head contexts before the shared output projection.
    Linearity of that projection lets us isolate one head by multiplying only its column
    block.  The dense bias is deliberately excluded because it is shared by the attention
    block and cannot be assigned to L3H1.
    """

    if not isinstance(context, torch.Tensor) or not isinstance(
        output_weight, torch.Tensor
    ):
        raise TypeError("context and output_weight must be torch tensors")
    if context.ndim < 1 or output_weight.ndim != 2:
        raise ValueError("context and output_weight dimensions differ")
    output_width, input_width = map(int, output_weight.shape)
    if output_width != input_width or input_width % int(num_heads):
        raise ValueError("output projection must be square and divisible by num_heads")
    head_width = input_width // int(num_heads)
    if not 0 <= int(head_index) < int(num_heads) or context.shape[-1] != input_width:
        raise ValueError("context or head index differs from the output projection")
    start = int(head_index) * head_width
    isolated = context[..., start : start + head_width]
    block = output_weight[:, start : start + head_width]
    return torch.nn.functional.linear(isolated, block, bias=None)


def _post_wo_numeric_provenance(checkpoint_dtype: str) -> dict[str, Any]:
    """Freeze the numeric operator used by the checkpoint and later JSON replay."""

    if checkpoint_dtype not in {"float32", "float64"}:
        raise InstrumentError(
            "post-W_O checkpoint operator dtype is outside the frozen registry"
        )
    return {
        "schema": R5B_POST_WO_NUMERIC_PROVENANCE_SCHEMA,
        "checkpoint_operator_dtype": checkpoint_dtype,
        "head_context_dtype": checkpoint_dtype,
        "w_o_column_block_dtype": checkpoint_dtype,
        "writer_contribution_dtype": checkpoint_dtype,
        "replay_operator": "torch.nn.functional.linear",
        "relative_tolerance": POST_WO_OPERATOR_RELATIVE_TOLERANCE,
        "absolute_tolerance": POST_WO_OPERATOR_ABSOLUTE_TOLERANCE,
    }


def _validated_post_wo_numeric_provenance(record: Mapping[str, Any]) -> tuple[
    dict[str, Any], torch.dtype
]:
    provenance = record.get("numeric_provenance")
    if not isinstance(provenance, Mapping):
        raise InstrumentError("post-W_O provider lacks numeric provenance")
    dtype_name = provenance.get("checkpoint_operator_dtype")
    expected = _post_wo_numeric_provenance(str(dtype_name))
    if set(provenance) != set(expected) or canonical_sha256(provenance) != canonical_sha256(
        expected
    ):
        raise InstrumentError("post-W_O numeric provenance differs")
    operator_dtype = {"float32": torch.float32, "float64": torch.float64}[
        str(dtype_name)
    ]
    return expected, operator_dtype


def replay_output_projected_writer_provider(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate post-``W_O`` L3H1 contributions in residual width.

    A pre-projection head context is never accepted as the writer variable.  The scored
    provider must carry the exact L3H1 column block of ``W_O`` (shared bias excluded),
    both head-context rows, both output-projected contribution rows, and their difference.
    """

    if record.get("writer_site_id") != WRITER_SITE_ID:
        raise InstrumentError("scored provider names a different writer site")
    if record.get("bias_included") is not False:
        raise InstrumentError("the shared attention output bias cannot belong to L3H1")
    numeric_provenance, operator_dtype = _validated_post_wo_numeric_provenance(
        record
    )
    clean_context = np.asarray(record.get("clean_head_context"), dtype=np.float64)
    corrupt_context = np.asarray(record.get("corrupt_head_context"), dtype=np.float64)
    block = np.asarray(record.get("w_o_column_block"), dtype=np.float64)
    clean = np.asarray(record.get("clean_writer_contribution"), dtype=np.float64)
    corrupt = np.asarray(record.get("corrupt_writer_contribution"), dtype=np.float64)
    delta = np.asarray(record.get("writer_delta"), dtype=np.float64)
    pseudo_fields = (
        "pseudo_head_context",
        "pseudo_writer_contribution",
        "pseudo_delta",
    )
    pseudo_present = [field in record for field in pseudo_fields]
    if any(pseudo_present) and not all(pseudo_present):
        raise InstrumentError("post-W_O pseudo-control provider is incomplete")
    pseudo_context = (
        np.asarray(record["pseudo_head_context"], dtype=np.float64)
        if all(pseudo_present)
        else None
    )
    pseudo = (
        np.asarray(record["pseudo_writer_contribution"], dtype=np.float64)
        if all(pseudo_present)
        else None
    )
    pseudo_delta = (
        np.asarray(record["pseudo_delta"], dtype=np.float64)
        if all(pseudo_present)
        else None
    )
    if (
        clean_context.ndim != 2
        or corrupt_context.shape != clean_context.shape
        or block.ndim != 2
        or block.shape[1] != clean_context.shape[1]
        or clean.shape != (clean_context.shape[0], block.shape[0])
        or corrupt.shape != clean.shape
        or delta.shape != clean.shape
        or not all(
            np.isfinite(value).all()
            for value in (clean_context, corrupt_context, block, clean, corrupt, delta)
        )
    ):
        raise InstrumentError("post-W_O scored-provider array shapes differ")
    residual_width = int(block.shape[0])
    head_width = int(block.shape[1])
    if residual_width % head_width or WRITER_HEAD >= residual_width // head_width:
        raise InstrumentError("W_O block cannot occupy the declared L3H1 columns")
    num_heads = residual_width // head_width
    block_tensor = torch.as_tensor(block, dtype=operator_dtype)

    def project(head_context: np.ndarray) -> np.ndarray:
        projected = torch.nn.functional.linear(
            torch.as_tensor(head_context, dtype=operator_dtype),
            block_tensor,
            bias=None,
        )
        return np.asarray(projected.cpu(), dtype=np.float64)

    replayed_clean = project(clean_context)
    replayed_corrupt = project(corrupt_context)
    replayed_delta = replayed_clean - replayed_corrupt
    if (
        not np.allclose(
            clean,
            replayed_clean,
            rtol=POST_WO_OPERATOR_RELATIVE_TOLERANCE,
            atol=POST_WO_OPERATOR_ABSOLUTE_TOLERANCE,
        )
        or not np.allclose(
            corrupt,
            replayed_corrupt,
            rtol=POST_WO_OPERATOR_RELATIVE_TOLERANCE,
            atol=POST_WO_OPERATOR_ABSOLUTE_TOLERANCE,
        )
        or not np.array_equal(
            np.asarray(delta, dtype=str(numeric_provenance["checkpoint_operator_dtype"])),
            np.asarray(clean - corrupt, dtype=str(numeric_provenance["checkpoint_operator_dtype"])),
        )
    ):
        raise InstrumentError(
            "stored post-W_O writer delta or contribution does not replay"
        )
    if pseudo_context is not None and pseudo is not None and pseudo_delta is not None:
        if (
            pseudo_context.shape != clean_context.shape
            or pseudo.shape != clean.shape
            or pseudo_delta.shape != clean.shape
            or not all(
                np.isfinite(value).all()
                for value in (pseudo_context, pseudo, pseudo_delta)
            )
        ):
            raise InstrumentError("post-W_O pseudo-control provider arrays differ")
        replayed_pseudo = project(pseudo_context)
        replayed_pseudo_delta = replayed_clean - replayed_pseudo
        if (
            not np.allclose(
                pseudo,
                replayed_pseudo,
                rtol=POST_WO_OPERATOR_RELATIVE_TOLERANCE,
                atol=POST_WO_OPERATOR_ABSOLUTE_TOLERANCE,
            )
            or not np.array_equal(
                np.asarray(
                    pseudo_delta,
                    dtype=str(numeric_provenance["checkpoint_operator_dtype"]),
                ),
                np.asarray(
                    clean - pseudo,
                    dtype=str(numeric_provenance["checkpoint_operator_dtype"]),
                ),
            )
        ):
            raise InstrumentError("stored post-W_O pseudo-control write does not replay")
    provider_record = {
        "writer_site_id": WRITER_SITE_ID,
        "clean_head_context": clean_context.tolist(),
        "corrupt_head_context": corrupt_context.tolist(),
        "w_o_column_block": block.tolist(),
        "clean_writer_contribution": clean.tolist(),
        "corrupt_writer_contribution": corrupt.tolist(),
        "writer_delta": delta.tolist(),
        "numeric_provenance": numeric_provenance,
        "bias_included": False,
    }
    if pseudo_context is not None and pseudo is not None and pseudo_delta is not None:
        provider_record.update(
            {
                "pseudo_head_context": pseudo_context.tolist(),
                "pseudo_writer_contribution": pseudo.tolist(),
                "pseudo_delta": pseudo_delta.tolist(),
            }
        )
    replay = {
        "schema": R5B_SCORED_PROVIDER_SCHEMA,
        "writer_site_id": WRITER_SITE_ID,
        "representation": "post_W_O_L3H1_additive_contribution_at_acceptor_query",
        "head_width": int(clean_context.shape[1]),
        "residual_width": int(block.shape[0]),
        "n_rows": int(clean_context.shape[0]),
        "bias_included": False,
        "numeric_provenance": numeric_provenance,
        "w_o_column_block_sha256": array_sha256(block),
        "clean_head_context_sha256": array_sha256(clean_context),
        "corrupt_head_context_sha256": array_sha256(corrupt_context),
        "clean_writer_contribution_sha256": array_sha256(clean),
        "corrupt_writer_contribution_sha256": array_sha256(corrupt),
        "writer_delta_sha256": array_sha256(delta),
        "writer_delta": np.ascontiguousarray(delta, dtype=np.float64),
        "provider_record": provider_record,
    }
    if pseudo_context is not None and pseudo is not None and pseudo_delta is not None:
        replay.update(
            {
                "pseudo_head_context_sha256": array_sha256(pseudo_context),
                "pseudo_writer_contribution_sha256": array_sha256(pseudo),
                "pseudo_delta_sha256": array_sha256(pseudo_delta),
                "pseudo_delta": np.ascontiguousarray(pseudo_delta, dtype=np.float64),
            }
        )
    return replay


def replace_local_writer_at_queries(
    dense_output: torch.Tensor,
    *,
    current_writer: torch.Tensor,
    replacement_writer: torch.Tensor,
    query_positions: torch.Tensor,
) -> torch.Tensor:
    """Replace only the output-projected writer contribution at one query per row."""

    if dense_output.ndim != 3:
        raise ValueError("attention dense output must be [B,S,D]")
    batch, sequence, width = map(int, dense_output.shape)
    if tuple(current_writer.shape) != (batch, width) or tuple(
        replacement_writer.shape
    ) != (batch, width):
        raise ValueError("writer tensors must be [B,D] and aligned to dense output")
    if query_positions.ndim != 1 or int(query_positions.numel()) != batch:
        raise ValueError("there must be exactly one query position per row")
    positions = query_positions.to(device=dense_output.device, dtype=torch.long)
    if bool(((positions < 0) | (positions >= sequence)).any()):
        raise ValueError("query position is outside the model sequence")
    rows = torch.arange(batch, device=dense_output.device)
    changed = dense_output.clone()
    changed[rows, positions] += (
        replacement_writer.to(dense_output.device, dense_output.dtype)
        - current_writer.to(dense_output.device, dense_output.dtype)
    )
    return changed


@dataclass
class LocalWriterHookAudit:
    writer_site_id: str = WRITER_SITE_ID
    intervention_position: str = "before_L4_L5"
    calls: int = 0
    current_writer_sha256: str | None = None
    replacement_writer_sha256: str | None = None


@contextmanager
def local_writer_dense_hook(
    dense_module: torch.nn.Module,
    *,
    query_positions: torch.Tensor,
    replacement_writer: torch.Tensor,
    head_index: int = WRITER_HEAD,
    num_heads: int,
) -> Iterator[LocalWriterHookAudit]:
    """Patch the named writer at the attention output projection's actual hook site."""

    weight = getattr(dense_module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("local-writer hook requires a linear module with a weight")
    audit = LocalWriterHookAudit()

    def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> torch.Tensor:
        if (
            len(inputs) != 1
            or not isinstance(inputs[0], torch.Tensor)
            or not isinstance(output, torch.Tensor)
        ):
            raise InstrumentError("unexpected attention output-projection hook signature")
        context = inputs[0]
        all_writer = output_projected_head_contribution(
            context,
            weight,
            head_index=head_index,
            num_heads=num_heads,
        )
        if query_positions.ndim != 1 or query_positions.numel() != context.shape[0]:
            raise InstrumentError("writer hook requires one aligned query per row")
        positions = query_positions.to(context.device, torch.long)
        rows = torch.arange(context.shape[0], device=context.device)
        current = all_writer[rows, positions]
        audit.calls += 1
        audit.current_writer_sha256 = array_sha256(
            current.detach().cpu().numpy()
        )
        audit.replacement_writer_sha256 = array_sha256(
            replacement_writer.detach().cpu().numpy()
        )
        return replace_local_writer_at_queries(
            output,
            current_writer=current,
            replacement_writer=replacement_writer,
            query_positions=positions,
        )

    handle = dense_module.register_forward_hook(hook)
    try:
        yield audit
    finally:
        handle.remove()


def _finite_vector(name: str, value: Any, *, n: int | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    if n is not None and int(vector.size) != int(n):
        raise ValueError(f"{name} must contain exactly {n} rows")
    return vector


def _replayed_dense_estimands(
    name: str,
    record: Mapping[str, Any],
    *,
    n: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    replayed = replay_local_writer_dense_record(record)
    if replayed["writer_site_id"] != WRITER_SITE_ID:
        raise ValueError(f"{name} uses a non-primary writer site")
    sufficiency = _finite_vector(
        f"{name} sufficiency", replayed["estimands"]["S_writer"], n=n
    )
    necessity = _finite_vector(
        f"{name} necessity", replayed["estimands"]["N_writer"], n=sufficiency.size
    )
    return sufficiency, necessity


def build_dense_local_writer_family(
    *,
    partner: Mapping[str, Any],
    pseudo: Mapping[str, Any],
    adjacent_by_offset: Mapping[int, Mapping[str, Any]],
    gene_generic: Mapping[str, Any],
    readout_axis: Mapping[str, Any],
    health: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay and derive the complete basis-independent dense writer family."""

    partner_s, partner_n = _replayed_dense_estimands("partner", partner)
    n = int(partner_s.size)
    pseudo_s, pseudo_n = _replayed_dense_estimands("pseudo", pseudo, n=n)
    if set(adjacent_by_offset) != set(ADJACENT_TOKEN_OFFSETS):
        raise ValueError("adjacent controls must use the exact frozen offset registry")
    endpoints: dict[str, np.ndarray] = {
        "partner|S": partner_s,
        "partner|N": partner_n,
        "pseudo|S": pseudo_s,
        "pseudo|N": pseudo_n,
        "specific|S": partner_s - pseudo_s,
        "specific|N": partner_n - pseudo_n,
    }
    for offset in ADJACENT_TOKEN_OFFSETS:
        adjacent_s, adjacent_n = _replayed_dense_estimands(
            f"adjacent offset {offset:+d}", adjacent_by_offset[offset], n=n
        )
        endpoints[f"adjacent|offset={offset:+d}|S"] = adjacent_s
        endpoints[f"adjacent|offset={offset:+d}|N"] = adjacent_n
    for arm, record in (
        ("gene_generic", gene_generic),
        ("readout_axis", readout_axis),
    ):
        sufficiency, necessity = _replayed_dense_estimands(arm, record, n=n)
        endpoints[f"{arm}|S"] = sufficiency
        endpoints[f"{arm}|N"] = necessity
    if set(health) != {"clean_noop", "corrupt_noop"}:
        raise ValueError("health must contain exactly the paired clean/corrupt no-op rows")
    endpoints["health|clean_noop"] = _finite_vector(
        "clean no-op", health["clean_noop"], n=n
    )
    endpoints["health|corrupt_noop"] = _finite_vector(
        "corrupt no-op", health["corrupt_noop"], n=n
    )
    return {
        "schema": "gfmcd.r5b.dense_local_writer_family/2",
        "writer_site_id": WRITER_SITE_ID,
        "path_role": DENSE_PATH_ROLE,
        "subspace_required": False,
        "endpoint_names": list(endpoints),
        "endpoints": endpoints,
        "primitive_records": {
            "partner": dict(partner),
            "pseudo": dict(pseudo),
            "adjacent_by_offset": {
                str(offset): dict(adjacent_by_offset[offset])
                for offset in ADJACENT_TOKEN_OFFSETS
            },
            "gene_generic": dict(gene_generic),
            "readout_axis": dict(readout_axis),
            "health": {
                "clean_noop": np.asarray(health["clean_noop"], dtype=np.float64),
                "corrupt_noop": np.asarray(
                    health["corrupt_noop"], dtype=np.float64
                ),
            },
        },
        "primitive_record_sha256": {
            "partner": canonical_sha256(partner),
            "pseudo": canonical_sha256(pseudo),
            "adjacent_by_offset": canonical_sha256(adjacent_by_offset),
            "gene_generic": canonical_sha256(gene_generic),
            "readout_axis": canonical_sha256(readout_axis),
            "health": canonical_sha256(health),
        },
    }


ESTIMATOR_NAMES = ("mean", "trimmed_mean_10pct")
REQUIREMENT_KINDS = (
    "material_positive",
    "positive",
    "equivalent",
    "noninferior",
)


def _statistics_and_standard_errors(
    samples: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Vectorized mean/trimmed-mean statistics and analytic bootstrap-t scales."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < MIN_GENES:
        raise ValueError(f"studentized inference requires at least {MIN_GENES} genes")
    n = int(values.shape[1])
    trim = int(math.floor(TRIM_FRACTION * n))
    if trim < 1 or 2 * trim >= n:
        raise ValueError("sample is too small for the frozen ten-percent trim")
    ordered = np.sort(values, axis=1)
    mean = np.mean(values, axis=1)
    trimmed = np.mean(ordered[:, trim : n - trim], axis=1)
    mean_se = np.std(values, axis=1, ddof=1) / math.sqrt(n)
    winsorized = ordered.copy()
    winsorized[:, :trim] = ordered[:, trim : trim + 1]
    winsorized[:, n - trim :] = ordered[:, n - trim - 1 : n - trim]
    retained_fraction = (n - 2 * trim) / n
    trimmed_se = np.std(winsorized, axis=1, ddof=1) / (
        retained_fraction * math.sqrt(n)
    )
    return (
        {"mean": mean, "trimmed_mean_10pct": trimmed},
        {"mean": mean_se, "trimmed_mean_10pct": trimmed_se},
    )


def _requirement_edges(requirement: str) -> tuple[tuple[str, float, int], ...]:
    if requirement == "material_positive":
        return (("lower_material", SESOI, 1),)
    if requirement == "positive":
        return (("lower_zero", 0.0, 1),)
    if requirement == "equivalent":
        return (
            ("lower_equivalence", -SESOI, 1),
            ("upper_equivalence", SESOI, -1),
        )
    if requirement == "noninferior":
        return (("lower_noninferiority", -HEALTH_NONINFERIORITY_MARGIN, 1),)
    raise ValueError(f"unknown requirement: {requirement!r}")


def robust_studentized_family_inference(
    endpoints: Mapping[str, Any],
    requirements: Mapping[str, str],
    genes: Sequence[str],
    *,
    n_draws: int,
    alpha: float,
    seed: int,
    tier: str,
) -> dict[str, Any]:
    """Joint cluster-bootstrap-t inversion for mean and 10%-trimmed mean.

    One shared whole-gene resample is used across the full decision family.  The maximum
    signed null statistic adjusts every materiality, equivalence, and noninferiority edge;
    the maximum absolute bootstrap-t statistic supplies simultaneous intervals.  Add-one
    Monte Carlo p-values are reported, but this is explicitly not an exact randomisation
    test because the bootstrap distribution is estimated rather than enumerated.
    """

    if tier not in {"development", "production"}:
        raise ValueError("tier must be development or production")
    if tier == "production" and int(n_draws) < MIN_PRODUCTION_INFERENCE_DRAWS:
        raise ValueError("production inference requires at least 10000 bootstrap draws")
    if int(n_draws) < 199:
        raise ValueError("bootstrap inversion requires at least 199 draws")
    if not 0.0 < float(alpha) < 0.5:
        raise ValueError("alpha must lie strictly between zero and one half")
    gene_ids = tuple(str(gene) for gene in genes)
    if len(gene_ids) < MIN_GENES or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("R5b inference requires one unique row per gene")
    if set(endpoints) != set(requirements) or not endpoints:
        raise ValueError("endpoint and requirement registries must match exactly")
    if any(requirement not in REQUIREMENT_KINDS for requirement in requirements.values()):
        raise ValueError("requirement registry contains an unknown decision kind")
    n = len(gene_ids)
    arrays = {
        name: _finite_vector(name, value, n=n) for name, value in endpoints.items()
    }
    rng = np.random.default_rng(int(seed))
    draw_indices = rng.integers(0, n, size=(int(n_draws), n), endpoint=False)

    observed: dict[str, dict[str, float]] = {}
    observed_se: dict[str, dict[str, float]] = {}
    bootstrap_stats: dict[str, dict[str, np.ndarray]] = {}
    bootstrap_se: dict[str, dict[str, np.ndarray]] = {}
    deviation_columns: list[np.ndarray] = []
    signed_null_columns: list[np.ndarray] = []
    edge_specs: list[tuple[str, str, str, float, int]] = []
    numerical_floor = np.finfo(np.float64).eps * 64.0
    for name, values in arrays.items():
        point_stats, point_se = _statistics_and_standard_errors(values[None, :])
        draw_stats, draw_se = _statistics_and_standard_errors(values[draw_indices])
        observed[name] = {
            estimator: float(point_stats[estimator][0])
            for estimator in ESTIMATOR_NAMES
        }
        observed_se[name] = {
            estimator: float(point_se[estimator][0])
            for estimator in ESTIMATOR_NAMES
        }
        bootstrap_stats[name] = draw_stats
        bootstrap_se[name] = draw_se
        for estimator in ESTIMATOR_NAMES:
            numerator = draw_stats[estimator] - observed[name][estimator]
            denominator = draw_se[estimator]
            stable = denominator > numerical_floor
            t_deviation = np.zeros(int(n_draws), dtype=np.float64)
            t_deviation[stable] = numerator[stable] / denominator[stable]
            deviation_columns.append(np.abs(t_deviation))
            for edge_name, boundary, direction in _requirement_edges(
                requirements[name]
            ):
                signed_null_columns.append(float(direction) * t_deviation)
                edge_specs.append(
                    (name, estimator, edge_name, float(boundary), int(direction))
                )
    max_abs = np.max(np.stack(deviation_columns, axis=0), axis=0)
    critical = float(
        np.quantile(max_abs, 1.0 - float(alpha), method="higher")
    )
    max_signed_null = np.max(np.stack(signed_null_columns, axis=0), axis=0)

    by_endpoint: dict[str, dict[str, Any]] = {
        name: {} for name in arrays
    }
    for name in arrays:
        for estimator in ESTIMATOR_NAMES:
            point = observed[name][estimator]
            se = observed_se[name][estimator]
            interval = (
                [point, point]
                if se <= numerical_floor
                else [point - critical * se, point + critical * se]
            )
            by_endpoint[name][estimator] = {
                "observed": point,
                "standard_error": se,
                "simultaneous_interval": [float(interval[0]), float(interval[1])],
                "edges": {},
            }
    for name, estimator, edge_name, boundary, direction in edge_specs:
        row = by_endpoint[name][estimator]
        point = float(row["observed"])
        se = float(row["standard_error"])
        oriented_difference = float(direction) * (point - boundary)
        if se <= numerical_floor:
            observed_t = math.inf if oriented_difference > 0.0 else -math.inf
        else:
            observed_t = oriented_difference / se
        comparison_tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0, abs(observed_t) if math.isfinite(observed_t) else 1.0
        )
        exceedances = int(
            np.count_nonzero(max_signed_null >= observed_t - comparison_tolerance)
        )
        p_value = (1.0 + exceedances) / (int(n_draws) + 1.0)
        lower, upper = map(float, row["simultaneous_interval"])
        interval_pass = lower > boundary if direction > 0 else upper < boundary
        row["edges"][edge_name] = {
            "boundary": boundary,
            "direction": "greater" if direction > 0 else "less",
            "observed_studentized_statistic": observed_t,
            "add_one_monte_carlo_p_value": float(p_value),
            "simultaneous_interval_pass": bool(interval_pass),
            "pass": bool(p_value <= alpha and interval_pass),
        }
    endpoint_pass: dict[str, bool] = {}
    for name in arrays:
        endpoint_pass[name] = bool(
            all(
                edge["pass"]
                for estimator in ESTIMATOR_NAMES
                for edge in by_endpoint[name][estimator]["edges"].values()
            )
        )
        by_endpoint[name]["requirement"] = requirements[name]
        by_endpoint[name]["both_estimators_pass"] = endpoint_pass[name]
    return {
        "schema": "gfmcd.r5b.robust_studentized_family/2",
        "inference": "studentized cluster bootstrap inversion",
        "exact_test_claimed": False,
        "p_value_rule": "add-one Monte Carlo from joint max signed bootstrap-t null",
        "one_row_per_gene": True,
        "cluster_unit": "gene",
        "estimators": list(ESTIMATOR_NAMES),
        "trim_fraction": TRIM_FRACTION,
        "n_genes": n,
        "n_draws": int(n_draws),
        "alpha": float(alpha),
        "seed": int(seed),
        "tier": tier,
        "simultaneous_critical_value": critical,
        "draw_indices_sha256": array_sha256(draw_indices, dtype="<i8"),
        "by_endpoint": by_endpoint,
        "endpoint_pass": endpoint_pass,
        "all_requirements_pass": bool(all(endpoint_pass.values())),
    }


def dense_requirement_registry() -> dict[str, str]:
    requirements = {
        "health|clean_noop": "noninferior",
        "health|corrupt_noop": "noninferior",
        "partner|S": "material_positive",
        "partner|N": "material_positive",
        "pseudo|S": "equivalent",
        "pseudo|N": "equivalent",
        "specific|S": "material_positive",
        "specific|N": "material_positive",
    }
    for offset in ADJACENT_TOKEN_OFFSETS:
        requirements[f"adjacent|offset={offset:+d}|S"] = "equivalent"
        requirements[f"adjacent|offset={offset:+d}|N"] = "equivalent"
    return requirements


def classify_dense_local_writer(
    family: Mapping[str, Any],
    genes: Sequence[str],
    *,
    n_draws: int = DEVELOPMENT_INFERENCE_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Apply the frozen health -> dense effect -> placement sequence."""

    if family.get("schema") != "gfmcd.r5b.dense_local_writer_family/2":
        raise ValueError("dense local-writer family schema differs")
    if family.get("writer_site_id") != WRITER_SITE_ID:
        raise ValueError("dense family does not name the primary local writer")
    all_endpoints = family.get("endpoints")
    if not isinstance(all_endpoints, Mapping):
        raise ValueError("dense family endpoints are absent")
    requirements = dense_requirement_registry()
    inference = robust_studentized_family_inference(
        {name: all_endpoints[name] for name in requirements},
        requirements,
        genes,
        n_draws=n_draws,
        alpha=FAMILY_ALPHA,
        seed=seed,
        tier="development",
    )
    endpoint_pass = inference["endpoint_pass"]
    stages = (
        (
            "paired_noop_health",
            ("health|clean_noop", "health|corrupt_noop"),
        ),
        (
            "dense_partner_pseudo_specific",
            (
                "partner|S",
                "partner|N",
                "pseudo|S",
                "pseudo|N",
                "specific|S",
                "specific|N",
            ),
        ),
        (
            "adjacent_token_local_placement",
            tuple(
                f"adjacent|offset={offset:+d}|{estimand}"
                for offset in ADJACENT_TOKEN_OFFSETS
                for estimand in ("S", "N")
            ),
        ),
    )
    sequence: list[dict[str, Any]] = []
    still_open = True
    for stage, names in stages:
        passed = bool(still_open and all(endpoint_pass[name] for name in names))
        sequence.append(
            {
                "stage": stage,
                "endpoint_names": list(names),
                "evaluated": bool(still_open),
                "pass": passed,
            }
        )
        still_open = passed
    if not sequence[0]["pass"]:
        status = "INSTRUMENT_UNRESOLVED_NOOP_HEALTH"
    elif not sequence[1]["pass"]:
        status = "DENSE_LOCAL_WRITER_DEVELOPMENT_NOT_SUPPORTED"
    elif not sequence[2]["pass"]:
        status = "DENSE_LOCAL_WRITER_PLACEMENT_UNRESOLVED"
    else:
        status = "DENSE_LOCAL_WRITER_DEVELOPMENT_SUPPORTED"
    return {
        "schema": "gfmcd.r5b.dense_local_writer_decision/2",
        "writer_site_id": WRITER_SITE_ID,
        "path_role": DENSE_PATH_ROLE,
        "status": status,
        "fixed_sequence": sequence,
        "inference": inference,
        "diagnostic_controls": {
            name: {
                estimator: float(
                    np.mean(np.asarray(all_endpoints[name], dtype=np.float64))
                    if estimator == "mean"
                    else _statistics_and_standard_errors(
                        np.asarray(all_endpoints[name], dtype=np.float64)[None, :]
                    )[0][estimator][0]
                )
                for estimator in ESTIMATOR_NAMES
            }
            for name in (
                "gene_generic|S",
                "gene_generic|N",
                "readout_axis|S",
                "readout_axis|N",
            )
        },
        "claim_eligible": False,
        "scope": "residual-validation instrument development only",
    }


DERIVATION_FOLD_SALT = "r5b-v2-local-writer-crossfit"


def exact_derivation_fold_ids(genes: Sequence[str]) -> np.ndarray:
    gene_ids = tuple(str(gene) for gene in genes)
    if len(gene_ids) != 128 or len(set(gene_ids)) != 128:
        raise ValueError("R5b derivation requires exactly 128 unique genes")
    order = sorted(
        gene_ids,
        key=lambda gene: (
            hashlib.sha256(f"{DERIVATION_FOLD_SALT}|{gene}".encode()).hexdigest(),
            gene,
        ),
    )
    by_gene = {gene: index % 2 for index, gene in enumerate(order)}
    folds = np.asarray([by_gene[gene] for gene in gene_ids], dtype=np.int64)
    if tuple(np.bincount(folds, minlength=2).tolist()) != (64, 64):
        raise RuntimeError("R5b derivation folds must contain exactly 64 genes each")
    return folds


def fit_local_writer_basis(
    writer_delta: Any,
    *,
    maximum_rank: int,
    stage: str,
    source_bucket: int | None,
) -> dict[str, Any]:
    """Fit an uncentered local-writer SVD only at an allowed development stage."""

    if stage == "target":
        raise TargetSealError("target may not fit or refit a local-writer basis")
    if stage not in {"derivation", "validation_diagnostic", "mechanical_union_refit"}:
        raise ValueError("unknown local-writer basis-fit stage")
    if stage == "derivation" and source_bucket != DERIVATION_BUCKET:
        raise ValueError("candidate derivation is restricted to residual-val bucket 0")
    if stage == "validation_diagnostic" and source_bucket != VALIDATION_BUCKET:
        raise ValueError("validation diagnostic is restricted to residual-val bucket 1")
    if stage == "mechanical_union_refit" and source_bucket is not None:
        raise ValueError("union refit must name both buckets through its caller")
    rows = np.asarray(writer_delta, dtype=np.float64)
    if (
        rows.ndim != 2
        or min(rows.shape) < 1
        or not np.isfinite(rows).all()
        or int(maximum_rank) < 1
        or int(maximum_rank) > min(rows.shape)
    ):
        raise ValueError("writer delta cannot support the requested finite SVD")
    _left, singular_values, vh = np.linalg.svd(rows, full_matrices=False)
    leading = float(singular_values[0])
    if leading <= 0.0:
        raise OptionalBasisInstrumentError(
            "INSUFFICIENT_NUMERICAL_RANK",
            "local-writer derivation matrix is identically zero",
        )
    required_ratio = float(singular_values[int(maximum_rank) - 1] / leading)
    if required_ratio <= 1e-10:
        raise OptionalBasisInstrumentError(
            "INSUFFICIENT_NUMERICAL_RANK",
            "local-writer SVD is numerically rank deficient at relative tolerance 1e-10",
        )
    basis = np.ascontiguousarray(vh[: int(maximum_rank)], dtype=np.float64)
    projector = orthogonal_projector(basis)
    diagnostics = projector_diagnostics(projector)
    if diagnostics["rank"] != int(maximum_rank):
        raise OptionalBasisInstrumentError(
            "PROJECTOR_RANK_MISMATCH", "local-writer projector rank differs"
        )
    return {
        "schema": "gfmcd.r5b.local_writer_basis_fit/2",
        "writer_site_id": WRITER_SITE_ID,
        "stage": stage,
        "source_bucket": source_bucket,
        "n_rows": int(rows.shape[0]),
        "width": int(rows.shape[1]),
        "maximum_rank": int(maximum_rank),
        "basis": basis,
        "basis_sha256": array_sha256(basis),
        "writer_delta_sha256": array_sha256(rows),
        "singular_values_sha256": array_sha256(singular_values),
        "minimum_required_singular_value_ratio": required_ratio,
        "numerical_rank_relative_tolerance": 1e-10,
        "projector_diagnostics": diagnostics,
    }


def fit_crossfit_derivation_bases(
    writer_delta: Any,
    genes: Sequence[str],
    *,
    maximum_rank: int,
) -> dict[str, Any]:
    """Fit every bucket-0 applied basis on the opposite 64-gene fold."""

    rows = np.asarray(writer_delta, dtype=np.float64)
    folds = exact_derivation_fold_ids(genes)
    if rows.ndim != 2 or rows.shape[0] != len(folds):
        raise ValueError("crossfit writer deltas are not aligned to the 128 genes")
    gene_ids = tuple(str(gene) for gene in genes)
    evaluation_folds = []
    for evaluation_fold in (0, 1):
        fit_mask = folds != evaluation_fold
        evaluation_mask = folds == evaluation_fold
        fit = fit_local_writer_basis(
            rows[fit_mask],
            maximum_rank=maximum_rank,
            stage="derivation",
            source_bucket=DERIVATION_BUCKET,
        )
        evaluation_folds.append(
            {
                "evaluation_fold": evaluation_fold,
                "fit_fold": 1 - evaluation_fold,
                "n_fit_rows": int(np.count_nonzero(fit_mask)),
                "n_evaluation_rows": int(np.count_nonzero(evaluation_mask)),
                "fit_genes": [
                    gene for gene, include in zip(gene_ids, fit_mask) if include
                ],
                "evaluation_genes": [
                    gene
                    for gene, include in zip(gene_ids, evaluation_mask)
                    if include
                ],
                "fit": fit,
            }
        )
    whole = fit_local_writer_basis(
        rows,
        maximum_rank=maximum_rank,
        stage="derivation",
        source_bucket=DERIVATION_BUCKET,
    )
    return {
        "schema": "gfmcd.r5b.crossfit_derivation_bases/2",
        "writer_site_id": WRITER_SITE_ID,
        "source_bucket": DERIVATION_BUCKET,
        "fold_salt": DERIVATION_FOLD_SALT,
        "fold_ids": folds,
        "fold_ids_sha256": array_sha256(folds, dtype="<i8"),
        "evaluation_folds": evaluation_folds,
        "whole_bucket_fit": whole,
        "whole_bucket_basis_use": (
            "candidate handoff to bucket1 only; never bucket0 outcome evaluation"
        ),
    }


def select_derivation_candidate_rank(
    derivation_fit: Mapping[str, Any],
    outcomes_by_rank: Mapping[int, Mapping[str, Any]],
    *,
    dense_development_status: str,
) -> dict[str, Any]:
    """Select one optional nested rank on bucket 0 without affecting dense evidence."""

    if derivation_fit.get("stage") != "derivation" or derivation_fit.get(
        "source_bucket"
    ) != DERIVATION_BUCKET:
        raise ValueError("candidate selection requires the bucket-0 derivation fit")
    if dense_development_status != "DENSE_LOCAL_WRITER_DEVELOPMENT_SUPPORTED":
        raise ValueError("nested candidate selection cannot precede dense path support")
    if set(outcomes_by_rank) != set(RANK_GRID):
        raise ValueError("candidate selector requires the complete frozen rank grid")
    for rank in RANK_GRID:
        if rank > int(derivation_fit.get("maximum_rank", 0)):
            raise ValueError("derivation fit does not cover the complete rank grid")
        required = {
            "mean_partner_s",
            "trimmed_partner_s",
            "nested_path_pass",
            "crossfold_stability_pass",
            "all_rank_energy_random_controls_pass",
        }
        if set(outcomes_by_rank[rank]) != required:
            raise ValueError("candidate outcome summary schema differs")
        if any(
            not isinstance(outcomes_by_rank[rank][name], (bool, np.bool_))
            for name in (
                "nested_path_pass",
                "crossfold_stability_pass",
                "all_rank_energy_random_controls_pass",
            )
        ):
            raise ValueError("candidate gates must be literal Booleans")
    max_mean = max(float(outcomes_by_rank[rank]["mean_partner_s"]) for rank in RANK_GRID)
    max_trimmed = max(
        float(outcomes_by_rank[rank]["trimmed_partner_s"]) for rank in RANK_GRID
    )
    selected: int | None = None
    if max_mean > 0.0 and max_trimmed > 0.0:
        for rank in RANK_GRID:
            row = outcomes_by_rank[rank]
            if (
                float(row["mean_partner_s"]) >= 0.90 * max_mean
                and float(row["trimmed_partner_s"]) >= 0.90 * max_trimmed
                and bool(row["nested_path_pass"])
                and bool(row["crossfold_stability_pass"])
                and bool(row["all_rank_energy_random_controls_pass"])
            ):
                selected = int(rank)
                break
    return {
        "schema": "gfmcd.r5b.local_writer_candidate/2",
        "writer_site_id": WRITER_SITE_ID,
        "path_role": SUBSPACE_PATH_ROLE,
        "source_bucket": DERIVATION_BUCKET,
        "rank_grid": list(RANK_GRID),
        "selected_rank": selected,
        "selection_rule": (
            "smallest rank with both S estimators >=90% of their maxima and all "
            "nested outcome/stability/rank-energy-null gates"
        ),
        "outcomes_by_rank_sha256": canonical_sha256(outcomes_by_rank),
        "derivation_basis_sha256": derivation_fit["basis_sha256"],
        "dense_development_status": dense_development_status,
        "dense_path_depends_on_selection": False,
        "no_candidate_does_not_veto_dense": True,
    }


def _validation_overlap_bootstrap(
    reference_basis: np.ndarray,
    validation_delta: np.ndarray,
    *,
    rank: int,
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    if int(n_draws) < 199:
        raise ValueError("validation stability requires at least 199 bootstrap draws")
    point = fit_local_writer_basis(
        validation_delta,
        maximum_rank=rank,
        stage="validation_diagnostic",
        source_bucket=VALIDATION_BUCKET,
    )
    point_basis = np.asarray(point["basis"], dtype=np.float64)
    point_overlap = mean_squared_principal_cosine(reference_basis, point_basis)
    rng = np.random.default_rng(int(seed))
    overlaps = np.empty(int(n_draws), dtype=np.float64)
    for draw in range(int(n_draws)):
        indices = rng.integers(
            0, validation_delta.shape[0], size=validation_delta.shape[0]
        )
        refit = fit_local_writer_basis(
            validation_delta[indices],
            maximum_rank=rank,
            stage="validation_diagnostic",
            source_bucket=VALIDATION_BUCKET,
        )
        overlaps[draw] = mean_squared_principal_cosine(
            reference_basis, np.asarray(refit["basis"], dtype=np.float64)
        )
    lower = float(np.quantile(overlaps, FAMILY_ALPHA, method="lower"))
    return {
        "point_overlap": point_overlap,
        "bootstrap_lower_95pct": lower,
        "absolute_overlap_sesoi": 0.20,
        "pass": bool(lower > 0.20),
        "n_draws": int(n_draws),
        "seed": int(seed),
        "overlaps_sha256": array_sha256(overlaps),
        "validation_diagnostic_basis_sha256": point["basis_sha256"],
        "validation_basis_used_for_intervention": False,
    }


def validate_frozen_candidate_on_bucket1(
    candidate: Mapping[str, Any],
    derivation_fit: Mapping[str, Any],
    validation_delta: Any,
    *,
    evaluated_ranks: Sequence[int],
    nested_outcome_pass: bool,
    random_controls_pass: bool,
    n_overlap_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Validate exactly the bucket-0 candidate; never refit the applied basis."""

    rank = candidate.get("selected_rank")
    if not isinstance(rank, int) or rank not in RANK_GRID:
        raise ValueError("validation requires one selected bucket-0 candidate rank")
    if list(evaluated_ranks) != [rank]:
        raise ValueError("bucket-1 validation must evaluate the frozen rank exactly once")
    if not isinstance(nested_outcome_pass, bool) or not isinstance(
        random_controls_pass, bool
    ):
        raise ValueError("validation gates must be literal Booleans")
    if derivation_fit.get("stage") != "derivation":
        raise ValueError("validation requires the bucket-0 derivation basis")
    if candidate.get("derivation_basis_sha256") not in {
        None,
        derivation_fit.get("basis_sha256"),
    }:
        raise ValueError("candidate and derivation basis identities differ")
    rows = np.asarray(validation_delta, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != derivation_fit.get("width"):
        raise ValueError("validation writer delta shape differs from derivation")
    applied_basis = np.asarray(derivation_fit["basis"], dtype=np.float64)[:rank]
    stability = _validation_overlap_bootstrap(
        applied_basis,
        rows,
        rank=rank,
        n_draws=n_overlap_draws,
        seed=seed,
    )
    passed = bool(stability["pass"] and nested_outcome_pass and random_controls_pass)
    return {
        "schema": "gfmcd.r5b.local_writer_candidate_validation/2",
        "writer_site_id": WRITER_SITE_ID,
        "path_role": SUBSPACE_PATH_ROLE,
        "source_bucket": VALIDATION_BUCKET,
        "selected_rank": rank,
        "evaluated_ranks": [rank],
        "applied_derivation_basis_sha256": array_sha256(applied_basis),
        "rank_refit_during_validation": False,
        "rank_rescanned_during_validation": False,
        "validation_diagnostic_basis_used_for_intervention": False,
        "stability": stability,
        "nested_outcome_pass": nested_outcome_pass,
        "random_controls_pass": random_controls_pass,
        "status": (
            "NESTED_CANDIDATE_VALIDATED_FOR_REFIT"
            if passed
            else "NESTED_CANDIDATE_NOT_VALIDATED"
        ),
        "claim_eligible": False,
    }


def mechanical_union_refit(
    bucket0_delta: Any,
    bucket1_delta: Any,
    *,
    row_ids_bucket0: Sequence[str],
    row_ids_bucket1: Sequence[str],
    candidate: Mapping[str, Any],
    validation: Mapping[str, Any],
    readout_axis: Any,
    seed: int,
) -> dict[str, Any]:
    """Mechanically refit the validated rank once on the locked bucket union."""

    if validation.get("status") != "NESTED_CANDIDATE_VALIDATED_FOR_REFIT":
        raise TargetSealError("union refit requires a locked successful bucket-1 validation")
    rank = candidate.get("selected_rank")
    if rank != validation.get("selected_rank") or rank not in RANK_GRID:
        raise ValueError("candidate/validation selected ranks differ")
    left = np.asarray(bucket0_delta, dtype=np.float64)
    right = np.asarray(bucket1_delta, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("union refit requires width-matched bucket matrices")
    ids0 = tuple(str(value) for value in row_ids_bucket0)
    ids1 = tuple(str(value) for value in row_ids_bucket1)
    if (
        len(ids0) != left.shape[0]
        or len(ids1) != right.shape[0]
        or len(set(ids0)) != len(ids0)
        or len(set(ids1)) != len(ids1)
        or set(ids0) & set(ids1)
    ):
        raise ValueError("union refit row identities must be exact and bucket-disjoint")
    combined = np.concatenate((left, right), axis=0)
    generic = np.mean(combined, axis=0)
    generic_norm = float(np.linalg.norm(generic))
    readout = np.asarray(readout_axis, dtype=np.float64)
    readout_norm = float(np.linalg.norm(readout))
    if (
        not np.isfinite(generic).all()
        or not np.isfinite(readout).all()
        or generic_norm <= 0.0
        or readout.shape != (combined.shape[1],)
        or readout_norm <= 0.0
    ):
        raise InstrumentError("gene-generic/readout control axes must be finite nonzero vectors")
    generic = np.ascontiguousarray(generic / generic_norm, dtype=np.float64)
    readout = np.ascontiguousarray(readout / readout_norm, dtype=np.float64)
    fit = fit_local_writer_basis(
        combined,
        maximum_rank=int(rank),
        stage="mechanical_union_refit",
        source_bucket=None,
    )
    basis = np.asarray(fit["basis"], dtype=np.float64)
    random_bases = np.stack(
        [
            deterministic_nested_random_bases(
                combined.shape[1],
                (int(rank),),
                seed=int(seed),
                namespace="r5b-v2-frozen-rank-energy-null",
                draw=draw,
            )[int(rank)]
            for draw in range(N_RANDOM_CONTROLS)
        ],
        axis=0,
    )
    return {
        "schema": R5B_BASIS_BUNDLE_SCHEMA,
        "writer_site_id": WRITER_SITE_ID,
        "path_role": SUBSPACE_PATH_ROLE,
        "selected_rank": int(rank),
        "basis": basis,
        "random_bases": random_bases,
        "gene_generic_axis": generic,
        "readout_axis": readout,
        "basis_sha256": array_sha256(basis),
        "basis_id": "sha256:" + array_sha256(basis),
        "random_bases_sha256": array_sha256(random_bases),
        "random_basis_ids": [
            "sha256:" + array_sha256(random_bases[draw])
            for draw in range(N_RANDOM_CONTROLS)
        ],
        "gene_generic_axis_sha256": array_sha256(generic),
        "readout_axis_sha256": array_sha256(readout),
        "ordered_development_row_ids_sha256": canonical_sha256([*ids0, *ids1]),
        "bucket0_row_ids_sha256": canonical_sha256(ids0),
        "bucket1_row_ids_sha256": canonical_sha256(ids1),
        "source_buckets": [DERIVATION_BUCKET, VALIDATION_BUCKET],
        "fit_timing": "after_bucket1_validation_locked",
        "refit_is_mechanical_not_selection": True,
        "target_refit_forbidden": True,
        "target_rank_selection_forbidden": True,
        "target_basis_mutation_forbidden": True,
        "candidate_sha256": canonical_sha256(candidate),
        "validation_sha256": canonical_sha256(validation),
        "fit_record": {key: value for key, value in fit.items() if key != "basis"},
    }


def energy_matched_rank_random_projections(
    writer_delta: Any,
    *,
    basis: Any,
    random_bases: Any,
) -> dict[str, Any]:
    """Project writer deltas onto the candidate and all rank/energy-matched nulls."""

    rows = np.asarray(writer_delta, dtype=np.float64)
    target_basis = np.asarray(basis, dtype=np.float64)
    controls = np.asarray(random_bases, dtype=np.float64)
    if rows.ndim != 2 or target_basis.ndim != 2 or controls.ndim != 3:
        raise ValueError("writer delta/basis/random bases have invalid dimensions")
    rank, width = map(int, target_basis.shape)
    if (
        rows.shape[1] != width
        or controls.shape != (N_RANDOM_CONTROLS, rank, width)
    ):
        raise ValueError("rank/width/random-control family shape differs")
    target_projector = orthogonal_projector(target_basis)
    target_projection = rows @ target_projector
    random_projections = np.empty(
        (N_RANDOM_CONTROLS, rows.shape[0], width), dtype=np.float64
    )
    random_projector_diagnostics = []
    for draw in range(N_RANDOM_CONTROLS):
        projector = orthogonal_projector(controls[draw])
        diagnostics = projector_diagnostics(projector)
        if diagnostics["rank"] != rank:
            raise OptionalBasisInstrumentError(
                "RANK_MATCHED_RANDOM_CONTROL_RANK_MISMATCH",
                "random control rank differs from target rank",
            )
        raw = rows @ projector
        random_projections[draw] = energy_match_rows(target_projection, raw)
        random_projector_diagnostics.append(diagnostics)
    target_norm = np.linalg.norm(target_projection, axis=1)
    random_norm = np.linalg.norm(random_projections, axis=2)
    max_abs_energy_error = float(
        np.max(np.abs(random_norm - target_norm[None, :]))
    )
    tolerance = 1e-10 * max(1.0, float(np.max(target_norm)))
    if max_abs_energy_error > tolerance:
        raise OptionalBasisInstrumentError(
            "RANK_ENERGY_MATCH_FAILURE",
            "rank-matched random projections are not energy matched",
        )
    audit = {
        "schema": "gfmcd.r5b.rank_energy_projection_audit/2",
        "rank": rank,
        "width": width,
        "n_rows": int(rows.shape[0]),
        "n_random_controls": N_RANDOM_CONTROLS,
        "target_basis_sha256": array_sha256(target_basis),
        "random_bases_sha256": array_sha256(controls),
        "target_basis_id": "sha256:" + array_sha256(target_basis),
        "random_basis_ids": [
            "sha256:" + array_sha256(controls[draw])
            for draw in range(N_RANDOM_CONTROLS)
        ],
        "target_projection_sha256": array_sha256(target_projection),
        "random_projections_sha256": array_sha256(random_projections),
        "max_abs_row_energy_error": max_abs_energy_error,
        "energy_match_tolerance": tolerance,
        "all_projector_ranks_match": True,
        "all_rows_energy_matched": True,
        "random_projector_diagnostics_sha256": canonical_sha256(
            random_projector_diagnostics
        ),
    }
    return {
        "rank": rank,
        "n_random_controls": N_RANDOM_CONTROLS,
        "target_projection": target_projection,
        "random_projections": random_projections,
        "audit": audit,
    }


def _replayed_subspace_estimands(
    name: str,
    record: Mapping[str, Any],
    *,
    n: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    replayed = replay_local_writer_subspace_record(record)
    if replayed["writer_site_id"] != WRITER_SITE_ID:
        raise ValueError(f"{name} uses a non-primary writer site")
    sufficiency = _finite_vector(
        f"{name} S_U", replayed["estimands"]["S_U"], n=n
    )
    necessity = _finite_vector(
        f"{name} N_U", replayed["estimands"]["N_U"], n=sufficiency.size
    )
    return sufficiency, necessity, str(replayed["basis_id"])


def build_nested_local_writer_family(
    *,
    partner: Mapping[str, Any],
    pseudo: Mapping[str, Any],
    adjacent_by_offset: Mapping[int, Mapping[str, Any]],
    gene_generic: Mapping[str, Any],
    readout_axis: Mapping[str, Any],
    random_by_draw: Mapping[int, Mapping[str, Any]],
    rank: int,
    projection_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the complete optional U_A family without changing the dense verdict."""

    if rank not in RANK_GRID:
        raise ValueError("nested family rank is outside the frozen grid")
    if (
        projection_audit.get("rank") != rank
        or projection_audit.get("n_random_controls") != N_RANDOM_CONTROLS
        or projection_audit.get("all_projector_ranks_match") is not True
        or projection_audit.get("all_rows_energy_matched") is not True
    ):
        raise ValueError("nested family lacks its complete rank/energy projection audit")
    partner_s, partner_n, basis_id = _replayed_subspace_estimands(
        "partner", partner
    )
    if basis_id != projection_audit.get("target_basis_id"):
        raise ValueError("nested target record does not name the audited target basis")
    n = int(partner_s.size)
    pseudo_s, pseudo_n, pseudo_basis_id = _replayed_subspace_estimands(
        "pseudo", pseudo, n=n
    )
    if pseudo_basis_id != basis_id:
        raise ValueError("partner/pseudo nested basis identities differ")
    endpoints: dict[str, np.ndarray] = {
        "partner|S": partner_s,
        "partner|N": partner_n,
        "pseudo|S": pseudo_s,
        "pseudo|N": pseudo_n,
        "specific|S": partner_s - pseudo_s,
        "specific|N": partner_n - pseudo_n,
    }
    if set(adjacent_by_offset) != set(ADJACENT_TOKEN_OFFSETS):
        raise ValueError("nested adjacent controls use a different offset registry")
    for offset in ADJACENT_TOKEN_OFFSETS:
        values = _replayed_subspace_estimands(
            f"adjacent {offset:+d}", adjacent_by_offset[offset], n=n
        )
        if values[2] != basis_id:
            raise ValueError("adjacent nested basis identity differs")
        endpoints[f"adjacent|offset={offset:+d}|S"] = values[0]
        endpoints[f"adjacent|offset={offset:+d}|N"] = values[1]
    for arm, record in (
        ("gene_generic", gene_generic),
        ("readout_axis", readout_axis),
    ):
        values = _replayed_subspace_estimands(arm, record, n=n)
        if values[2] != basis_id:
            raise ValueError(f"{arm} nested basis identity differs")
        endpoints[f"{arm}|S"] = values[0]
        endpoints[f"{arm}|N"] = values[1]
    if set(random_by_draw) != set(range(N_RANDOM_CONTROLS)):
        raise ValueError("nested family requires every frozen random-control draw")
    for draw in range(N_RANDOM_CONTROLS):
        random_s, random_n, random_basis_id = _replayed_subspace_estimands(
            f"random {draw:02d}", random_by_draw[draw], n=n
        )
        random_ids = projection_audit.get("random_basis_ids")
        if (
            not isinstance(random_ids, list)
            or len(random_ids) != N_RANDOM_CONTROLS
            or random_basis_id != random_ids[draw]
        ):
            raise ValueError("nested random record does not name its audited basis")
        endpoints[f"random|draw={draw:02d}|S"] = random_s
        endpoints[f"random|draw={draw:02d}|N"] = random_n
        endpoints[f"target_minus_random|draw={draw:02d}|S"] = (
            partner_s - random_s
        )
        endpoints[f"target_minus_random|draw={draw:02d}|N"] = (
            partner_n - random_n
        )
    return {
        "schema": "gfmcd.r5b.nested_local_writer_family/2",
        "writer_site_id": WRITER_SITE_ID,
        "basis_id": basis_id,
        "rank": rank,
        "path_role": SUBSPACE_PATH_ROLE,
        "dense_path_depends_on_this_family": False,
        "rank_energy_random_control_count": N_RANDOM_CONTROLS,
        "projection_audit": dict(projection_audit),
        "endpoint_names": list(endpoints),
        "endpoints": endpoints,
        "primitive_records": {
            "partner": dict(partner),
            "pseudo": dict(pseudo),
            "adjacent_by_offset": {
                str(offset): dict(adjacent_by_offset[offset])
                for offset in ADJACENT_TOKEN_OFFSETS
            },
            "gene_generic": dict(gene_generic),
            "readout_axis": dict(readout_axis),
            "random_by_draw": {
                str(draw): dict(random_by_draw[draw])
                for draw in range(N_RANDOM_CONTROLS)
            },
        },
    }


def nested_requirement_registry(rank: int) -> dict[str, str]:
    if rank not in RANK_GRID:
        raise ValueError("nested requirement rank is outside the frozen grid")
    prefix = f"nested|rank={rank}|"
    requirements = {
        f"{prefix}partner|S": "material_positive",
        f"{prefix}partner|N": "material_positive",
        f"{prefix}pseudo|S": "equivalent",
        f"{prefix}pseudo|N": "equivalent",
        f"{prefix}specific|S": "material_positive",
        f"{prefix}specific|N": "material_positive",
    }
    for offset in ADJACENT_TOKEN_OFFSETS:
        requirements[f"{prefix}adjacent|offset={offset:+d}|S"] = "equivalent"
        requirements[f"{prefix}adjacent|offset={offset:+d}|N"] = "equivalent"
    for draw in range(N_RANDOM_CONTROLS):
        requirements[f"{prefix}target_minus_random|draw={draw:02d}|S"] = "positive"
        requirements[f"{prefix}target_minus_random|draw={draw:02d}|N"] = "positive"
    return requirements


def classify_nested_local_writer_family(
    family: Mapping[str, Any],
    genes: Sequence[str],
    *,
    n_draws: int = DEVELOPMENT_INFERENCE_DRAWS,
    seed: int = SEED,
    phase: str = "derivation",
) -> dict[str, Any]:
    """Classify one prespecified nested rank while leaving the dense path untouched."""

    if family.get("schema") != "gfmcd.r5b.nested_local_writer_family/2":
        raise ValueError("nested local-writer family schema differs")
    if phase not in {"derivation", "validation"}:
        raise ValueError("nested family phase must be derivation or validation")
    rank = family.get("rank")
    if rank not in RANK_GRID:
        raise ValueError("nested family rank differs from the frozen grid")
    endpoints = family.get("endpoints")
    if not isinstance(endpoints, Mapping):
        raise ValueError("nested family endpoints are absent")
    prefixed = nested_requirement_registry(int(rank))
    prefix = f"nested|rank={rank}|"
    unprefixed_requirements = {
        name.removeprefix(prefix): requirement
        for name, requirement in prefixed.items()
    }
    inference = robust_studentized_family_inference(
        {name: endpoints[name] for name in unprefixed_requirements},
        unprefixed_requirements,
        genes,
        n_draws=n_draws,
        alpha=(
            FAMILY_ALPHA / len(RANK_GRID)
            if phase == "derivation"
            else FAMILY_ALPHA
        ),
        seed=seed,
        tier="development",
    )
    target_names = (
        "partner|S",
        "partner|N",
        "pseudo|S",
        "pseudo|N",
        "specific|S",
        "specific|N",
        *tuple(
            f"adjacent|offset={offset:+d}|{estimand}"
            for offset in ADJACENT_TOKEN_OFFSETS
            for estimand in ("S", "N")
        ),
    )
    random_names = tuple(
        f"target_minus_random|draw={draw:02d}|{estimand}"
        for draw in range(N_RANDOM_CONTROLS)
        for estimand in ("S", "N")
    )
    target_pass = bool(all(inference["endpoint_pass"][name] for name in target_names))
    random_pass = bool(all(inference["endpoint_pass"][name] for name in random_names))
    diagnostics = {}
    for name in (
        "gene_generic|S",
        "gene_generic|N",
        "readout_axis|S",
        "readout_axis|N",
    ):
        point, _se = _statistics_and_standard_errors(
            np.asarray(endpoints[name], dtype=np.float64)[None, :]
        )
        diagnostics[name] = {
            estimator: float(point[estimator][0]) for estimator in ESTIMATOR_NAMES
        }
    return {
        "schema": "gfmcd.r5b.nested_local_writer_decision/2",
        "writer_site_id": WRITER_SITE_ID,
        "rank": int(rank),
        "phase": phase,
        "path_role": SUBSPACE_PATH_ROLE,
        "nested_path_pass": target_pass,
        "all_rank_energy_random_controls_pass": random_pass,
        "status": (
            "NESTED_CANDIDATE_RANK_PASSES"
            if target_pass and random_pass
            else "NESTED_CANDIDATE_RANK_DOES_NOT_PASS"
        ),
        "inference": inference,
        "gene_generic_and_readout_axis_diagnostics": diagnostics,
        "dense_path_status_changed": False,
        "claim_eligible": False,
    }


def causal_requirement_registry() -> dict[str, str]:
    """Complete decision registry bound into the prospective joint stack."""

    requirements = {
        f"dense|{name}": requirement
        for name, requirement in dense_requirement_registry().items()
    }
    for rank in RANK_GRID:
        requirements.update(nested_requirement_registry(rank))
    return requirements


def causal_primitive_names() -> tuple[str, ...]:
    """Exact margin-cell registry for dense and all scanned nested candidates."""

    dense_arms = ["partner", "pseudo", "gene_generic", "readout_axis"] + [
        f"adjacent|offset={offset:+d}" for offset in ADJACENT_TOKEN_OFFSETS
    ]
    names: list[str] = []
    for arm in dense_arms:
        for cell in (
            "m_clean",
            "m_corrupt",
            "m_corrupt_with_clean_writer",
            "m_clean_with_corrupt_writer",
        ):
            names.append(f"dense|{arm}|{cell}")
    names.extend(("dense|health|clean_noop", "dense|health|corrupt_noop"))
    for rank in RANK_GRID:
        prefix = f"nested|rank={rank}|"
        arms = ["partner", "pseudo", "gene_generic", "readout_axis"] + [
            f"adjacent|offset={offset:+d}" for offset in ADJACENT_TOKEN_OFFSETS
        ] + [f"random|draw={draw:02d}" for draw in range(N_RANDOM_CONTROLS)]
        for arm in arms:
            for cell in (
                "m_clean",
                "m_corrupt",
                "m_corrupt_with_projected_writer",
                "m_clean_without_projected_writer",
            ):
                names.append(f"{prefix}{arm}|{cell}")
    if len(names) != len(set(names)):
        raise RuntimeError("R5b v2 primitive registry contains duplicate cells")
    return tuple(names)


def causal_family_names() -> tuple[str, ...]:
    """Exact derived endpoint registry, including diagnostics and all random gaps."""

    dense = [
        "partner|S",
        "partner|N",
        "pseudo|S",
        "pseudo|N",
        "specific|S",
        "specific|N",
        "gene_generic|S",
        "gene_generic|N",
        "readout_axis|S",
        "readout_axis|N",
        "health|clean_noop",
        "health|corrupt_noop",
    ] + [
        f"adjacent|offset={offset:+d}|{estimand}"
        for offset in ADJACENT_TOKEN_OFFSETS
        for estimand in ("S", "N")
    ]
    names = [f"dense|{name}" for name in dense]
    for rank in RANK_GRID:
        prefix = f"nested|rank={rank}|"
        rank_names = [
            "partner|S",
            "partner|N",
            "pseudo|S",
            "pseudo|N",
            "specific|S",
            "specific|N",
            "gene_generic|S",
            "gene_generic|N",
            "readout_axis|S",
            "readout_axis|N",
        ] + [
            f"adjacent|offset={offset:+d}|{estimand}"
            for offset in ADJACENT_TOKEN_OFFSETS
            for estimand in ("S", "N")
        ]
        for draw in range(N_RANDOM_CONTROLS):
            rank_names.extend(
                (
                    f"random|draw={draw:02d}|S",
                    f"random|draw={draw:02d}|N",
                    f"target_minus_random|draw={draw:02d}|S",
                    f"target_minus_random|draw={draw:02d}|N",
                )
            )
        names.extend(f"{prefix}{name}" for name in rank_names)
    if len(names) != len(set(names)):
        raise RuntimeError("R5b v2 family registry contains duplicate endpoints")
    return tuple(names)


def causal_boundary_edge_registry() -> tuple[dict[str, Any], ...]:
    """Expand both robust estimators and every strict boundary edge."""

    rows: list[dict[str, Any]] = []
    for endpoint, requirement in causal_requirement_registry().items():
        for estimator in ESTIMATOR_NAMES:
            for edge_name, boundary, direction in _requirement_edges(requirement):
                edge_id = f"{endpoint}|estimator={estimator}|edge={edge_name}"
                rows.append(
                    {
                        "edge_id": edge_id,
                        "endpoint": endpoint,
                        "estimator": estimator,
                        "requirement": requirement,
                        "edge": edge_name,
                        "boundary": float(boundary),
                        "direction": "greater" if direction > 0 else "less",
                    }
                )
    if len(rows) != len({row["edge_id"] for row in rows}):
        raise RuntimeError("R5b v2 boundary registry contains duplicate edges")
    return tuple(rows)


def prospective_calibration_trial_count() -> dict[str, Any]:
    """Bind a deterministic two-pass engineering cycle over every decision edge."""

    edges = causal_boundary_edge_registry()
    return {
        "schema": "gfmcd.r5b.classifier_calibration_plan/2",
        "decision_edge_count": len(edges),
        "cycles_per_edge": 2,
        "trial_count": 2 * len(edges),
        "studentized_production_classifier_exercised": True,
        "mean_and_trimmed_mean_exercised": True,
        "scientific_error_control_claimed": False,
        "purpose": "engineering regression audit of strict decision surfaces",
        "edge_registry_sha256": canonical_sha256(edges),
    }


RUNTIME_ARRAY_NAMES = (
    "frozen_basis",
    "random_bases",
    "gene_generic_axis",
    "readout_axis",
    "selected_rank",
)


def frozen_basis_runtime_arrays(frozen: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract the only arrays R5c may use for optional nested interventions."""

    if frozen.get("schema") != R5B_BASIS_BUNDLE_SCHEMA:
        raise ValueError("frozen local-writer bundle schema differs")
    if frozen.get("writer_site_id") != WRITER_SITE_ID:
        raise ValueError("frozen bundle names a different writer site")
    rank = frozen.get("selected_rank")
    if rank not in RANK_GRID:
        raise ValueError("frozen bundle selected rank differs")
    basis = np.asarray(frozen.get("basis"), dtype=np.float64)
    random_bases = np.asarray(frozen.get("random_bases"), dtype=np.float64)
    generic = np.asarray(frozen.get("gene_generic_axis"), dtype=np.float64)
    readout = np.asarray(frozen.get("readout_axis"), dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != rank:
        raise ValueError("frozen basis shape differs from selected rank")
    width = int(basis.shape[1])
    if random_bases.shape != (N_RANDOM_CONTROLS, rank, width):
        raise ValueError("frozen random basis family shape differs")
    if generic.shape != (width,) or readout.shape != (width,):
        raise ValueError("frozen control axes do not match basis width")
    orthogonal_projector(basis)
    for draw in range(N_RANDOM_CONTROLS):
        orthogonal_projector(random_bases[draw])
    if not np.isclose(np.linalg.norm(generic), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("gene-generic axis is not unit norm")
    if not np.isclose(np.linalg.norm(readout), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("readout axis is not unit norm")
    if not (
        frozen.get("target_refit_forbidden") is True
        and frozen.get("target_rank_selection_forbidden") is True
        and frozen.get("target_basis_mutation_forbidden") is True
    ):
        raise ValueError("frozen bundle does not prohibit target refit/selection/mutation")
    return {
        "frozen_basis": np.ascontiguousarray(basis, dtype="<f8"),
        "random_bases": np.ascontiguousarray(random_bases, dtype="<f8"),
        "gene_generic_axis": np.ascontiguousarray(generic, dtype="<f8"),
        "readout_axis": np.ascontiguousarray(readout, dtype="<f8"),
        "selected_rank": np.asarray(rank, dtype="<i8"),
    }


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if set(arrays) != set(RUNTIME_ARRAY_NAMES):
        raise ValueError("runtime NPZ array registry differs")
    stream = io.BytesIO()
    np.savez(stream, **{name: arrays[name] for name in RUNTIME_ARRAY_NAMES})
    return stream.getvalue()


def _validate_sha256_literal(name: str, value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be one literal SHA-256")
    return value.lower()


def verify_r5a_entry_package(
    artifact_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_stack_fingerprint: str,
) -> dict[str, Any]:
    """Authenticate the frozen pretarget R5a method/preflight hook contract.

    R5b precedes every sealed biological target.  It may consume R3d Path A and the
    outcome-free R1c validation handoff only through this R5a preflight package; an R5a
    target result or any chr8/9-derived contract is rejected.
    """

    if FROZEN_R5A_ARTIFACT_SHA256 is None or FROZEN_R5A_MANIFEST_SHA256 is None:
        raise TargetSealError("R5a artifact and manifest pins are both required")
    artifact = Path(artifact_path).expanduser().resolve(strict=True)
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    artifact_hash = sha256_file(artifact)
    manifest_hash = sha256_file(manifest_file)
    if artifact_hash != FROZEN_R5A_ARTIFACT_SHA256:
        raise TargetSealError("R5a entry artifact differs from its literal pin")
    if manifest_hash != FROZEN_R5A_MANIFEST_SHA256:
        raise TargetSealError("R5a entry manifest differs from its literal pin")
    try:
        record = json.loads(artifact.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetSealError("R5a entry package is not valid JSON") from exc
    matches = [
        output
        for output in manifest.get("outputs", [])
        if isinstance(output, Mapping)
        and Path(str(output.get("path", ""))).name == artifact.name
        and output.get("sha256") == artifact_hash
        and output.get("size_bytes") == artifact.stat().st_size
    ]
    if len(matches) != 1 or manifest.get("git_dirty") is not False:
        raise TargetSealError("R5a entry is not bound by one clean sibling manifest")
    contract = record.get("r5_entry_contract")
    if not isinstance(contract, Mapping):
        raise TargetSealError("R5a /2 entry contract is absent")
    if (
        record.get("schema") != "gfmcd.r5a.acceptor_address_binding/2"
        or record.get("stage") != "preflight"
        or record.get("status") != "PREFLIGHT_PASS"
        or record.get("all_instrument_gates_pass") is not True
        or record.get("behavioral_endpoint") != "masked_acceptor_G_margin_only"
        or record.get("donor_readout_constructed") is not False
        or record.get("masked_donor_endpoint_constructed") is not False
        or record.get("r5a_opened_chr8_chr9") is not False
        or record.get("scientific_entry_authorized") is not False
        or record.get("observed_scientific_inference") is not None
        or contract.get("schema") != R5_ENTRY_SCHEMA
        or contract.get("instrument_gates_pass") is not True
        or contract.get("claim_eligible") is not False
        or contract.get("verdict") != "PREFLIGHT_PASS"
        or contract.get("masked_donor_endpoint_constructed") is not False
    ):
        raise TargetSealError("R5a pretarget method/preflight contract differs")
    expected_stack = _validate_sha256_literal(
        "expected R5 stack", expected_stack_fingerprint
    )
    if (
        record.get("r5_stack_fingerprint_sha256") != expected_stack
        or contract.get("r5_stack_fingerprint_sha256") != expected_stack
    ):
        raise TargetSealError("R5a entry joint-stack fingerprint differs")
    order = contract.get("scientific_entry_order")
    if order != [
        "authenticated_R3d_Path_A_prior",
        "authenticated_R1c_validation_preflight_nonentry",
        "R5a_instrument_preflight",
    ]:
        raise TargetSealError("R5a preflight scientific-entry order differs")
    if contract.get("r3d_path_b_used_to_gate") is not False:
        raise TargetSealError("R5a entry improperly uses R3d Path B as a gate")
    if contract.get("r3c_used_as_scientific_entry") is not False:
        raise TargetSealError("R5a entry improperly reopens R3c as scientific support")
    if (
        contract.get("acceptor_G_only") is not True
        or contract.get("masked_donor_endpoint_constructed") is not False
    ):
        raise TargetSealError("R5a entry is not masked-acceptor-G-only")
    contract_keys = {
        "schema",
        "instrument_gates_pass",
        "claim_eligible",
        "verdict",
        "scientific_entry_order",
        "r3d_path_b_used_to_gate",
        "r3c_used_as_scientific_entry",
        "acceptor_G_only",
        "masked_donor_endpoint_constructed",
        "checkpoint_sha256",
        "target_gene_ids_sha256",
        "cohort_sha256",
        "ordered_row_ids_sha256",
        "ordered_gene_ids_sha256",
        "construction_sha256",
        "r5_stack_fingerprint_sha256",
        "upstream_package_chain_sha256",
    }
    if set(contract) != contract_keys:
        raise TargetSealError("R5a preflight entry-contract fields differ")
    required_hashes = (
        "checkpoint_sha256",
        "target_gene_ids_sha256",
        "cohort_sha256",
        "ordered_row_ids_sha256",
        "ordered_gene_ids_sha256",
        "construction_sha256",
        "r5_stack_fingerprint_sha256",
        "upstream_package_chain_sha256",
    )
    for field in required_hashes:
        _validate_sha256_literal(f"R5a entry {field}", contract.get(field))
    return {
        "entry_chain_pass": True,
        "artifact": {
            "logical_name": artifact.name,
            "size_bytes": artifact.stat().st_size,
            "sha256": artifact_hash,
        },
        "manifest": {
            "logical_name": manifest_file.name,
            "size_bytes": manifest_file.stat().st_size,
            "sha256": manifest_hash,
        },
        "entry_contract": dict(contract),
        "r3d_path_a_prior_consumed_via_r5a": True,
        "r1c_validation_preflight_consumed_via_r5a": True,
        "sealed_target_result_consumed": False,
        "r3d_path_b_reopened": False,
        "r3c_reopened_as_scientific_entry": False,
    }


def frozen_basis_runtime_contract(
    frozen: Mapping[str, Any],
    *,
    bundle_path: str | Path,
    bundle_bytes: bytes,
    checkpoint_sha256: str,
    cohort_audit_sha256: str,
    r5_stack_fingerprint_sha256: str,
    artifact_bindings: Mapping[str, Mapping[str, str]],
    post_wo_writer_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the exact union-refit basis and every predecessor manifest."""

    arrays = frozen_basis_runtime_arrays(frozen)
    expected_bindings = {
        "r5a_entry",
        "r5b_derivation",
        "r5b_validation",
    }
    if set(artifact_bindings) != expected_bindings:
        raise ValueError("runtime contract requires every exact upstream artifact binding")
    normalized_bindings: dict[str, dict[str, str]] = {}
    for name in sorted(expected_bindings):
        binding = artifact_bindings[name]
        if set(binding) != {"artifact_sha256", "manifest_sha256"}:
            raise ValueError("each upstream binding needs artifact and manifest SHA-256")
        normalized_bindings[name] = {
            "artifact_sha256": _validate_sha256_literal(
                f"{name} artifact", binding["artifact_sha256"]
            ),
            "manifest_sha256": _validate_sha256_literal(
                f"{name} manifest", binding["manifest_sha256"]
            ),
        }
    path = Path(bundle_path).expanduser().resolve()
    if path.exists() and path.read_bytes() != bundle_bytes:
        raise ValueError("supplied runtime bytes differ from the named bundle")
    if (
        post_wo_writer_contract.get("writer_site_id") != WRITER_SITE_ID
        or post_wo_writer_contract.get("representation")
        != "post_W_O_L3H1_additive_contribution_at_acceptor_query"
        or post_wo_writer_contract.get("bias_included") is not False
        or post_wo_writer_contract.get("residual_width")
        != int(arrays["frozen_basis"].shape[1])
    ):
        raise ValueError("runtime contract requires the exact residual-width writer")
    payload = {
        "schema": R5B_BASIS_CONTRACT_SCHEMA,
        "writer_site_id": WRITER_SITE_ID,
        "path_role": SUBSPACE_PATH_ROLE,
        "selected_rank": int(frozen["selected_rank"]),
        "basis_id": "sha256:" + array_sha256(arrays["frozen_basis"], dtype=None),
        "random_basis_ids": [
            "sha256:" + array_sha256(arrays["random_bases"][draw], dtype=None)
            for draw in range(N_RANDOM_CONTROLS)
        ],
        "source_buckets": [DERIVATION_BUCKET, VALIDATION_BUCKET],
        "fit_timing": "after_bucket1_validation_locked",
        "development_only_source": True,
        "representation": "post_W_O_L3H1_additive_contribution_at_acceptor_query",
        "residual_width": int(arrays["frozen_basis"].shape[1]),
        "post_wo_writer_contract_sha256": canonical_sha256(
            post_wo_writer_contract
        ),
        "bundle": {
            "file_name": path.name,
            "size_bytes": len(bundle_bytes),
            "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        },
        "runtime_arrays": {
            name: {
                "shape": list(np.asarray(value).shape),
                "dtype": np.asarray(value).dtype.str,
                "sha256": array_sha256(value, dtype=None),
            }
            for name, value in arrays.items()
        },
        "checkpoint_sha256": _validate_sha256_literal(
            "checkpoint", checkpoint_sha256
        ),
        "cohort_audit_sha256": _validate_sha256_literal(
            "cohort audit", cohort_audit_sha256
        ),
        "r5_stack_fingerprint_sha256": _validate_sha256_literal(
            "R5 stack", r5_stack_fingerprint_sha256
        ),
        "upstream_artifact_bindings": normalized_bindings,
        "target_policy": {
            "refit": "forbidden",
            "rank_selection": "forbidden",
            "basis_mutation": "forbidden",
            "basis_application": "exact authenticated arrays only",
        },
        "r5c_target_attempt_hooks": r5c_target_attempt_hook_locations(),
    }
    return payload | {"contract_sha256": canonical_sha256(payload)}


def load_frozen_basis_runtime_bundle(
    path: str | Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Relocation-safe byte/shape/dtype validation of the R5c runtime bundle."""

    if contract.get("schema") != R5B_BASIS_CONTRACT_SCHEMA:
        raise InstrumentError("frozen runtime contract schema differs")
    stored_digest = contract.get("contract_sha256")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if stored_digest != canonical_sha256(payload):
        raise InstrumentError("frozen runtime contract digest differs")
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    bundle = contract.get("bundle")
    if not isinstance(bundle, Mapping) or (
        bundle.get("file_name") != source.name
        or int(bundle.get("size_bytes", -1)) != len(raw)
        or bundle.get("sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise InstrumentError("frozen runtime bundle bytes differ")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != set(RUNTIME_ARRAY_NAMES):
            raise InstrumentError("frozen runtime array registry differs")
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    records = contract.get("runtime_arrays")
    if not isinstance(records, Mapping) or set(records) != set(RUNTIME_ARRAY_NAMES):
        raise InstrumentError("frozen runtime array contract differs")
    for name, value in arrays.items():
        expected = records[name]
        if (
            list(value.shape) != expected.get("shape")
            or value.dtype.str != expected.get("dtype")
            or array_sha256(value, dtype=None) != expected.get("sha256")
        ):
            raise InstrumentError(f"frozen runtime array differs: {name}")
    if contract.get("basis_id") != "sha256:" + array_sha256(
        arrays["frozen_basis"], dtype=None
    ) or contract.get("random_basis_ids") != [
        "sha256:" + array_sha256(arrays["random_bases"][draw], dtype=None)
        for draw in range(N_RANDOM_CONTROLS)
    ]:
        raise InstrumentError("frozen runtime basis IDs differ from exact arrays")
    selected = arrays["selected_rank"]
    if selected.shape != () or selected.dtype.kind not in {"i", "u"}:
        raise InstrumentError("frozen runtime selected rank scalar differs")
    rank = int(selected)
    basis = np.asarray(arrays["frozen_basis"], dtype=np.float64)
    random_bases = np.asarray(arrays["random_bases"], dtype=np.float64)
    generic = np.asarray(arrays["gene_generic_axis"], dtype=np.float64)
    readout = np.asarray(arrays["readout_axis"], dtype=np.float64)
    if (
        contract.get("selected_rank") != rank
        or rank not in RANK_GRID
        or basis.ndim != 2
        or basis.shape[0] != rank
        or random_bases.shape != (N_RANDOM_CONTROLS, rank, basis.shape[1])
        or generic.shape != (basis.shape[1],)
        or readout.shape != (basis.shape[1],)
        or not np.isclose(np.linalg.norm(generic), 1.0, rtol=0.0, atol=1e-8)
        or not np.isclose(np.linalg.norm(readout), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise InstrumentError("frozen runtime selected rank or array geometry differs")
    orthogonal_projector(basis)
    for draw in range(N_RANDOM_CONTROLS):
        orthogonal_projector(random_bases[draw])
    return arrays


def _publication_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise InstrumentError(f"publication is not a regular file: {source}")
    return {
        "logical_name": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _load_json_publication(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InstrumentError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise InstrumentError(f"{label} must contain one JSON object")
    return value


def _require_manifest_output(
    manifest: Mapping[str, Any], identity: Mapping[str, Any]
) -> None:
    matches = [
        output
        for output in manifest.get("outputs", [])
        if isinstance(output, Mapping)
        and Path(str(output.get("path", ""))).name == identity.get("logical_name")
        and output.get("size_bytes") == identity.get("size_bytes")
        and output.get("sha256") == identity.get("sha256")
    ]
    if len(matches) != 1:
        raise InstrumentError("manifest does not bind exactly one publication output")


def _validate_linked_package(
    expected: Mapping[str, Any], artifact_path: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    if set(expected) != {"artifact", "manifest"}:
        raise InstrumentError("linked package identity fields differ")
    artifact_identity = _publication_identity(artifact_path)
    manifest_identity = _publication_identity(manifest_path)
    if expected.get("artifact") != artifact_identity or expected.get(
        "manifest"
    ) != manifest_identity:
        raise InstrumentError("linked package bytes differ from disposition")
    linked_manifest = _load_json_publication(
        manifest_path, label="linked package manifest"
    )
    _require_manifest_output(linked_manifest, artifact_identity)
    if linked_manifest.get("git_dirty") is not False:
        raise InstrumentError("linked package manifest is not clean")
    return _load_json_publication(artifact_path, label="linked package artifact")


def validate_local_writer_basis_disposition(
    artifact_path: str | Path,
    manifest_path: str | Path,
    *,
    bundle_path: str | Path | None = None,
    derivation_artifact_path: str | Path | None = None,
    derivation_manifest_path: str | Path | None = None,
    validation_artifact_path: str | Path | None = None,
    validation_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Offline authentication for R5c's optional path-local C2 decision."""

    artifact_identity = _publication_identity(artifact_path)
    manifest_identity = _publication_identity(manifest_path)
    record = _load_json_publication(artifact_path, label="R5b disposition")
    manifest = _load_json_publication(manifest_path, label="R5b disposition manifest")
    _require_manifest_output(manifest, artifact_identity)
    if manifest.get("git_dirty") is not False:
        raise InstrumentError("R5b disposition manifest is not clean")
    status = record.get("status")
    if (
        record.get("schema") != R5B_DISPOSITION_SCHEMA
        or status not in R5B_DISPOSITION_STATUSES
        or record.get("basis_disposition") != status
        or record.get("claim_eligible") is not False
        or record.get("residual_validation_only") is not True
        or record.get("sealed_chr8_chr9_opened") is not False
        or record.get("masked_donor_endpoint_constructed") is not False
        or record.get("dense_primary_unaffected") is not True
        or record.get("optional_c2_only") is not True
    ):
        raise InstrumentError("R5b disposition header differs")
    post_wo = record.get("post_wo_writer_contract")
    if (
        not isinstance(post_wo, Mapping)
        or post_wo.get("writer_site_id") != WRITER_SITE_ID
        or post_wo.get("representation")
        != "post_W_O_L3H1_additive_contribution_at_acceptor_query"
        or post_wo.get("bias_included") is not False
        or not isinstance(post_wo.get("residual_width"), int)
        or post_wo.get("residual_width", 0) < 1
    ):
        raise InstrumentError("R5b post-W_O writer contract differs")
    _validated_post_wo_numeric_provenance(post_wo)
    for field in (
        "w_o_column_block_sha256",
        "bucket0_clean_writer_contribution_sha256",
        "bucket0_corrupt_writer_contribution_sha256",
        "bucket0_writer_delta_sha256",
        "bucket0_pseudo_writer_contribution_sha256",
        "bucket0_pseudo_delta_sha256",
        "bucket1_clean_writer_contribution_sha256",
        "bucket1_corrupt_writer_contribution_sha256",
        "bucket1_writer_delta_sha256",
        "bucket1_pseudo_writer_contribution_sha256",
        "bucket1_pseudo_delta_sha256",
    ):
        _validate_sha256_literal(field, post_wo.get(field))

    replay = record.get("sufficient_replay")
    if not isinstance(replay, Mapping):
        raise InstrumentError("R5b disposition lacks sufficient replay")
    expected_status = (
        "BASIS_INSTRUMENT_UNRESOLVED"
        if replay.get("instrument_failure") is True
        else (
            "VALIDATED_BASIS"
            if replay.get("validation_status")
            == "NESTED_CANDIDATE_VALIDATED_FOR_REFIT"
            else "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS"
        )
    )
    if status != expected_status:
        raise InstrumentError("R5b terminal disposition does not replay")
    checkpoint_sha256 = _validate_sha256_literal(
        "R5b disposition checkpoint", record.get("checkpoint_sha256")
    )
    cohort_audit_sha256 = _validate_sha256_literal(
        "R5b disposition cohort audit", record.get("cohort_audit_sha256")
    )
    stack_sha256 = _validate_sha256_literal(
        "R5b disposition stack", record.get("r5_stack_fingerprint_sha256")
    )

    linked = record.get("development_packages")
    if not isinstance(linked, Mapping) or set(linked) != {
        "derivation",
        "validation",
    }:
        raise InstrumentError("R5b derivation/validation package links are incomplete")
    supplied_paths = (
        derivation_artifact_path,
        derivation_manifest_path,
        validation_artifact_path,
        validation_manifest_path,
    )
    if any(path is None for path in supplied_paths):
        raise InstrumentError(
            "offline validation requires exact derivation and validation packages"
        )
    assert derivation_artifact_path is not None
    assert derivation_manifest_path is not None
    assert validation_artifact_path is not None
    assert validation_manifest_path is not None
    derivation_record = _validate_linked_package(
        linked["derivation"], derivation_artifact_path, derivation_manifest_path
    )
    validation_record = _validate_linked_package(
        linked["validation"], validation_artifact_path, validation_manifest_path
    )
    replayed_derivation = _assert_development_record_replays(
        derivation_record, require_production=True
    )
    replayed_validation = _assert_development_record_replays(
        validation_record, derivation=derivation_record, require_production=True
    )
    if (
        status != _disposition_for_replays(derivation_record, validation_record)
        or replay.get("instrument_failure")
        is not (status == "BASIS_INSTRUMENT_UNRESOLVED")
        or replay.get("nested_basis_instrument_failures")
        != {
            "derivation": replayed_derivation.get(
                "nested_basis_instrument_failure"
            ),
            "validation": replayed_validation.get(
                "nested_basis_instrument_failure"
            ),
        }
        or replay.get("derivation_status") != replayed_derivation.get("status")
        or replay.get("validation_status")
        != (replayed_validation.get("candidate_validation") or {}).get("status")
        or replay.get("bucket0_dense_status")
        != replayed_derivation["dense_decision"]["status"]
        or replay.get("bucket1_dense_status")
        != replayed_validation["dense_decision"]["status"]
        or replay.get("derivation_record_sha256")
        != canonical_sha256(derivation_record)
        or replay.get("validation_record_sha256")
        != canonical_sha256(validation_record)
    ):
        raise InstrumentError("R5b disposition sufficient replay package hashes differ")
    scientific_prior = record.get("scientific_prior")
    r5a_preflight = record.get("r5a_method_preflight")
    if (
        not isinstance(scientific_prior, Mapping)
        or scientific_prior.get("role") != "R3d_Path_A_L3H1_reuse_only"
        or scientific_prior.get("path_b_used_to_gate") is not False
        or not isinstance(scientific_prior.get("package"), Mapping)
        or set(scientific_prior["package"])
        != {"artifact_sha256", "manifest_sha256"}
        or not isinstance(r5a_preflight, Mapping)
        or r5a_preflight.get("target_result_consumed") is not False
        or not isinstance(r5a_preflight.get("package"), Mapping)
        or set(r5a_preflight["package"])
        != {"artifact_sha256", "manifest_sha256"}
    ):
        raise InstrumentError("R5b upstream pretarget package bindings differ")
    for label, package in (
        ("R3d Path A", scientific_prior["package"]),
        ("R5a preflight", r5a_preflight["package"]),
    ):
        _validate_sha256_literal(f"{label} artifact", package["artifact_sha256"])
        _validate_sha256_literal(f"{label} manifest", package["manifest_sha256"])

    contract = record.get("frozen_basis_runtime_contract")
    bundle = record.get("basis_bundle")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise InstrumentError("R5b disposition outputs are missing")
    if status == "VALIDATED_BASIS":
        if not isinstance(contract, Mapping) or not isinstance(bundle, Mapping):
            raise InstrumentError("validated disposition lacks runtime basis contract")
        if bundle_path is None:
            raise InstrumentError("validated disposition requires the external NPZ")
        bundle_identity = _publication_identity(bundle_path)
        if bundle != bundle_identity:
            raise InstrumentError("R5b external basis bundle identity differs")
        _require_manifest_output(manifest, bundle_identity)
        arrays = load_frozen_basis_runtime_bundle(bundle_path, contract=contract)
        if int(arrays["frozen_basis"].shape[1]) != int(post_wo["residual_width"]):
            raise InstrumentError("frozen basis is not in residual writer width")
        expected_bindings = {
            "r5a_entry": dict(r5a_preflight["package"]),
            "r5b_derivation": {
                "artifact_sha256": linked["derivation"]["artifact"]["sha256"],
                "manifest_sha256": linked["derivation"]["manifest"]["sha256"],
            },
            "r5b_validation": {
                "artifact_sha256": linked["validation"]["artifact"]["sha256"],
                "manifest_sha256": linked["validation"]["manifest"]["sha256"],
            },
        }
        if (
            contract.get("writer_site_id") != WRITER_SITE_ID
            or contract.get("path_role") != SUBSPACE_PATH_ROLE
            or contract.get("source_buckets")
            != [DERIVATION_BUCKET, VALIDATION_BUCKET]
            or contract.get("fit_timing") != "after_bucket1_validation_locked"
            or contract.get("development_only_source") is not True
            or contract.get("representation")
            != "post_W_O_L3H1_additive_contribution_at_acceptor_query"
            or contract.get("residual_width") != post_wo["residual_width"]
            or contract.get("post_wo_writer_contract_sha256")
            != canonical_sha256(post_wo)
            or contract.get("checkpoint_sha256") != checkpoint_sha256
            or contract.get("cohort_audit_sha256") != cohort_audit_sha256
            or contract.get("r5_stack_fingerprint_sha256") != stack_sha256
            or contract.get("upstream_artifact_bindings") != expected_bindings
            or contract.get("target_policy")
            != {
                "refit": "forbidden",
                "rank_selection": "forbidden",
                "basis_mutation": "forbidden",
                "basis_application": "exact authenticated arrays only",
            }
            or contract.get("r5c_target_attempt_hooks")
            != r5c_target_attempt_hook_locations()
        ):
            raise InstrumentError("R5b runtime contract semantics differ")
        frozen_replay = record.get("frozen_basis_sufficient_replay")
        if not isinstance(frozen_replay, Mapping):
            raise InstrumentError("validated disposition lacks union-refit replay")
        writer0 = replay_output_projected_writer_provider(
            derivation_record["scored_provider"]["output_projected_writer"]
        )
        writer1 = replay_output_projected_writer_provider(
            validation_record["scored_provider"]["output_projected_writer"]
        )
        expected_frozen = mechanical_union_refit(
            writer0["writer_delta"],
            writer1["writer_delta"],
            row_ids_bucket0=derivation_record["row_ids"],
            row_ids_bucket1=validation_record["row_ids"],
            candidate=derivation_record["candidate"],
            validation=validation_record["candidate_validation"],
            readout_axis=derivation_record["scored_provider"]["readout_axis"],
            seed=SEED,
        )
        if canonical_sha256(frozen_replay) != canonical_sha256(
            _json_native(expected_frozen)
        ):
            raise InstrumentError("R5b mechanical union refit does not replay")
        replay_arrays = frozen_basis_runtime_arrays(frozen_replay)
        if any(
            not np.array_equal(arrays[name], replay_arrays[name])
            for name in RUNTIME_ARRAY_NAMES
        ):
            raise InstrumentError("R5b runtime arrays differ from union-refit replay")
        if record.get("path_local_c2_disposition") != R5B_PATH_LOCAL_C2_BY_STATUS[
            status
        ]:
            raise InstrumentError("validated path-local C2 disposition differs")
        if contract.get("basis_id") != "sha256:" + array_sha256(
            arrays["frozen_basis"], dtype=None
        ) or contract.get("random_basis_ids") != [
            "sha256:" + array_sha256(arrays["random_bases"][draw], dtype=None)
            for draw in range(N_RANDOM_CONTROLS)
        ]:
            raise InstrumentError("R5b basis/random-basis IDs do not authenticate arrays")
    else:
        if contract is not None or bundle is not None or bundle_path is not None:
            raise InstrumentError("nonvalidated disposition must not carry a basis bundle")
        if len(outputs) != 1:
            raise InstrumentError("nonvalidated disposition may publish only its artifact")
        if record.get("frozen_basis_sufficient_replay") is not None:
            raise InstrumentError("nonvalidated disposition cannot carry a union refit")
        expected_c2 = R5B_PATH_LOCAL_C2_BY_STATUS[status]
        if record.get("path_local_c2_disposition") != expected_c2:
            raise InstrumentError("nonvalidated path-local C2 disposition differs")
    return {
        "authenticated": True,
        "artifact": artifact_identity,
        "manifest": manifest_identity,
        "status": status,
        "path_local_c2_disposition": record["path_local_c2_disposition"],
        "frozen_basis_runtime_contract": contract,
        "dense_primary_unaffected": True,
        "record": record,
    }


def r5c_target_attempt_hook_locations() -> dict[str, Any]:
    """Declare where R5c must use the shared irreversible target ledger."""

    return {
        "owner_stage": "R5c biological target",
        "identity_builder": (
            "gfmcd.splicing.r5_target_ledger.build_r5_biological_target_identity"
        ),
        "reserve_hook": "gfmcd.splicing.r5_target_ledger.reserve_r5_target_attempt",
        "complete_hook": "gfmcd.splicing.r5_target_ledger.complete_r5_target_attempt",
        "reserve_after": "all predecessor and upstream packages authenticated",
        "reserve_before": "cohort/path/model/output inspection",
        "complete_after": "immutable result and manifest publication",
        "r5b_calls_target_ledger": False,
    }


def instrument_binding_payload() -> dict[str, Any]:
    """Bind the v2 writer definition, decisions, lifecycle, and complete joint stack."""

    stack = build_r5_stack_binding(repo=REPO)
    stack_fingerprint = r5_stack_fingerprint(stack)
    primitive_names = causal_primitive_names()
    family_names = causal_family_names()
    edges = causal_boundary_edge_registry()
    return {
        "schema": R5B_INSTRUMENT_SCHEMA,
        "r5_stack_binding": stack,
        "r5_stack_fingerprint_sha256": stack_fingerprint,
        "behavioral_endpoint": "single masked acceptor G hard margin only",
        "masked_donor_endpoint_constructed": False,
        "writer": {
            "site_id": WRITER_SITE_ID,
            "layer": WRITER_LAYER,
            "head": WRITER_HEAD,
            "tensor": "attention.output.dense additive head contribution",
            "representation": (
                "post_W_O_L3H1_additive_contribution_at_acceptor_query"
            ),
            "projection_equation": "context_L3H1_q @ W_O[:,L3H1_columns].T",
            "provider_schema": R5B_SCORED_PROVIDER_SCHEMA,
            "w_o_column_block_authenticated": True,
            "clean_corrupt_contribution_hashes_authenticated": True,
            "numeric_provenance_schema": R5B_POST_WO_NUMERIC_PROVENANCE_SCHEMA,
            "checkpoint_dtype_replayed": True,
            "operator_relative_tolerance": POST_WO_OPERATOR_RELATIVE_TOLERANCE,
            "operator_absolute_tolerance": POST_WO_OPERATOR_ABSOLUTE_TOLERANCE,
            "stored_delta_algebra_exact_in_checkpoint_dtype": True,
            "bias_assigned_to_head": False,
            "temporal_position": "after L3 attention and before L4/L5 delivery",
        },
        "evidence_roles": {
            "dense_local_writer": DENSE_PATH_ROLE,
            "low_rank_U_A": SUBSPACE_PATH_ROLE,
            "dense_depends_on_low_rank": False,
        },
        "development_lifecycle": {
            "derive_candidate_on_bucket": DERIVATION_BUCKET,
            "validate_frozen_candidate_on_bucket": VALIDATION_BUCKET,
            "mechanical_union_refit_after_validation": True,
            "target_refit_forbidden": True,
            "target_rank_selection_forbidden": True,
            "target_basis_mutation_forbidden": True,
            "residual_validation_claim_eligible": False,
            "optional_basis_failure_schema": R5B_OPTIONAL_BASIS_FAILURE_SCHEMA,
            "optional_basis_failure_preserves_dense_primary": True,
            "unclassified_scorer_or_core_failure_aborts": True,
        },
        "statistics": {
            "estimators": list(ESTIMATOR_NAMES),
            "trim_fraction": TRIM_FRACTION,
            "inference": "studentized cluster bootstrap inversion",
            "production_minimum_draws": MIN_PRODUCTION_INFERENCE_DRAWS,
            "exact_test_claimed": False,
            "family_alpha": FAMILY_ALPHA,
            "sesoi": SESOI,
            "paired_noop_noninferiority_margin": HEALTH_NONINFERIORITY_MARGIN,
        },
        "controls": {
            "paired_pseudo_specific": True,
            "adjacent_token_offsets": list(ADJACENT_TOKEN_OFFSETS),
            "gene_generic_axis": True,
            "readout_axis": True,
            "rank_energy_random_controls": N_RANDOM_CONTROLS,
        },
        "registry": {
            "rank_grid": list(RANK_GRID),
            "primitive_count": len(primitive_names),
            "primitive_sha256": canonical_sha256(primitive_names),
            "family_count": len(family_names),
            "family_sha256": canonical_sha256(family_names),
            "decision_edge_count": len(edges),
            "decision_edge_sha256": canonical_sha256(edges),
            "calibration_plan": prospective_calibration_trial_count(),
        },
        "upstream_entry": {
            "schema": R5_ENTRY_SCHEMA,
            "consume_only_authenticated_R5a_method_preflight_contract": True,
            "R3d_Path_A_consumed_via_R5a": True,
            "R1c_validation_preflight_consumed_via_R5a": True,
            "sealed_biological_target_result_consumed": False,
            "R3d_Path_B_gate": False,
            "R3c_scientific_entry": False,
        },
        "terminal_disposition": {
            "schema": R5B_DISPOSITION_SCHEMA,
            "statuses": list(R5B_DISPOSITION_STATUSES),
            "always_publish_artifact_and_manifest": True,
            "validated_only_runtime_contract_and_npz": True,
            "no_validated_basis_suppresses_only_optional_C2": True,
            "dense_primary_unaffected": True,
        },
        "freeze_pin_names": sorted(
            name for name in globals() if name.startswith("FROZEN_")
        ),
        "attempt_hooks": {
            "R5b": {
                "owner": "pretarget residual instrument development",
                "derivation": (
                    "create-only bucket0 receipt before cohort/model/output inspection; "
                    "completion after immutable artifact+manifest"
                ),
                "validation": (
                    "authenticate completed derivation, then create-only bucket1 receipt "
                    "before cohort/model/output inspection; completion after publication"
                ),
                "biological_target_ledger_used": False,
                "failure_remains_spent": True,
            },
            "R5c": r5c_target_attempt_hook_locations(),
        },
    }


def instrument_fingerprint(payload: Mapping[str, Any] | None = None) -> str:
    return canonical_sha256(
        instrument_binding_payload() if payload is None else dict(payload)
    )


class ResidualAttemptSpentError(RuntimeError):
    """A frozen residual bucket already has a create-only attempt receipt."""


def _git_common_dir(repo: str | Path) -> Path:
    root = Path(repo).expanduser().resolve(strict=True)
    process = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = process.stdout.strip()
    if not rendered:
        raise RuntimeError("git returned an empty common directory")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=True)


def build_r5b_residual_attempt_identity(
    *,
    stage: str,
    checkpoint_sha256: str,
    cohort_input_sha256: str,
    cohort_audit_sha256: str,
) -> dict[str, Any]:
    """Build a method-insensitive spend key before either bucket is inspected."""

    if stage not in {"derivation", "validation"}:
        raise ValueError("residual attempt stage must be derivation or validation")
    bucket = DERIVATION_BUCKET if stage == "derivation" else VALIDATION_BUCKET
    stable = {
        "schema": R5B_RESIDUAL_IDENTITY_SCHEMA,
        "program": "r5b-pretarget-output-projected-writer-v2",
        "stage": stage,
        "residual_validation_bucket": bucket,
        "chromosomes": ["chr11", "chr16", "chr19"],
        "declared_n": 128,
        "checkpoint_sha256": _validate_sha256_literal(
            "checkpoint", checkpoint_sha256
        ),
        "cohort_input_sha256": _validate_sha256_literal(
            "cohort input", cohort_input_sha256
        ),
        "cohort_audit_sha256": _validate_sha256_literal(
            "cohort audit", cohort_audit_sha256
        ),
    }
    return stable | {"residual_attempt_key": canonical_sha256(stable)}


def _residual_attempt_path(
    repo: str | Path, identity: Mapping[str, Any]
) -> Path:
    key = identity.get("residual_attempt_key")
    stage = identity.get("stage")
    if stage not in {"derivation", "validation"} or not isinstance(key, str):
        raise ValueError("residual attempt identity differs")
    return (
        _git_common_dir(repo)
        / "gfmcd_method_attempts"
        / "r5b-residual"
        / stage
        / f"{key}.json"
    )


def _write_create_only(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ResidualAttemptSpentError(
            f"{label} already exists; the residual bucket remains spent"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def reserve_r5b_residual_attempt(
    *,
    repo: str | Path,
    identity: Mapping[str, Any],
    method_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a bucket receipt before cohort/model/output path inspection."""

    expected_identity = build_r5b_residual_attempt_identity(
        stage=str(identity.get("stage")),
        checkpoint_sha256=str(identity.get("checkpoint_sha256")),
        cohort_input_sha256=str(identity.get("cohort_input_sha256")),
        cohort_audit_sha256=str(identity.get("cohort_audit_sha256")),
    )
    if dict(identity) != expected_identity:
        raise ValueError("residual attempt identity does not replay")
    required_context = {
        "protocol_sha256",
        "instrument_fingerprint_sha256",
        "r5_stack_fingerprint_sha256",
        "r3d_artifact_sha256",
        "r3d_manifest_sha256",
        "r5a_preflight_artifact_sha256",
        "r5a_preflight_manifest_sha256",
        "predecessor_artifact_sha256",
        "predecessor_manifest_sha256",
        "predecessor_completion_sha256",
        "requested_artifact_logical_name",
        "requested_manifest_logical_name",
        "declared_completion_logical_name",
        "scoring_origin",
    }
    if set(method_context) != required_context:
        raise ValueError("residual attempt method context fields differ")
    context = dict(method_context)
    hash_fields = {
        "protocol_sha256",
        "instrument_fingerprint_sha256",
        "r5_stack_fingerprint_sha256",
        "r3d_artifact_sha256",
        "r3d_manifest_sha256",
        "r5a_preflight_artifact_sha256",
        "r5a_preflight_manifest_sha256",
    }
    for name in hash_fields:
        context[name] = _validate_sha256_literal(name, context[name])
    predecessor = (
        context["predecessor_artifact_sha256"],
        context["predecessor_manifest_sha256"],
        context["predecessor_completion_sha256"],
    )
    if identity["stage"] == "derivation":
        if predecessor != (None, None, None):
            raise ValueError("derivation cannot name a predecessor result")
    else:
        context["predecessor_artifact_sha256"] = _validate_sha256_literal(
            "predecessor artifact", predecessor[0]
        )
        context["predecessor_manifest_sha256"] = _validate_sha256_literal(
            "predecessor manifest", predecessor[1]
        )
        context["predecessor_completion_sha256"] = _validate_sha256_literal(
            "predecessor completion", predecessor[2]
        )
    for name in (
        "requested_artifact_logical_name",
        "requested_manifest_logical_name",
        "declared_completion_logical_name",
    ):
        value = context[name]
        if not isinstance(value, str) or not value or Path(value).name != value:
            raise ValueError(f"{name} must be one bare relocation-safe name")
    if context["scoring_origin"] != "direct_checkpoint_scoring_in_r5b_process":
        raise ValueError("R5b residual stage must score the checkpoint directly")
    path = _residual_attempt_path(repo, identity)
    if context["declared_completion_logical_name"] != path.with_suffix(
        ".completion.json"
    ).name:
        raise ValueError("declared residual completion name differs")
    receipt_without_id = {
        "schema": R5B_RESIDUAL_ATTEMPT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUTHORIZED_RESIDUAL_BUCKET_MAY_BE_SPENT",
        "residual_attempt_key": identity["residual_attempt_key"],
        "identity": dict(identity),
        "method_context": context,
        "policy": (
            "create-only before residual cohort/model/output inspection; failures "
            "remain spent; method or code changes cannot reset the bucket"
        ),
        "cohort_inspected_before_record": False,
        "model_loaded_before_record": False,
        "output_inspected_before_record": False,
        "sealed_chr8_chr9_opened": False,
    }
    receipt = receipt_without_id | {
        "attempt_id": canonical_sha256(receipt_without_id)
    }
    _write_create_only(path, receipt, label="R5b residual attempt")
    return receipt | {"receipt_path": str(path)}


def residual_attempt_publication_identity(
    receipt: Mapping[str, Any]
) -> dict[str, Any]:
    path_raw = receipt.get("receipt_path")
    if not isinstance(path_raw, str):
        raise ValueError("residual receipt lacks its canonical path")
    identity = _publication_identity(path_raw)
    return identity | {
        "attempt_id": receipt.get("attempt_id"),
        "residual_attempt_key": receipt.get("residual_attempt_key"),
        "status": receipt.get("status"),
    }


def complete_r5b_residual_attempt(
    *,
    receipt: Mapping[str, Any],
    artifact_path: str | Path,
    manifest_path: str | Path,
    official_status: str,
) -> dict[str, Any]:
    """Complete one residual stage only after artifact and manifest publication."""

    path_raw = receipt.get("receipt_path")
    if not isinstance(path_raw, str):
        raise ValueError("residual receipt lacks its canonical path")
    receipt_path = Path(path_raw).expanduser().resolve(strict=True)
    persisted = _load_json_publication(receipt_path, label="residual attempt")
    supplied = {key: value for key, value in receipt.items() if key != "receipt_path"}
    if (
        persisted != supplied
        or persisted.get("schema") != R5B_RESIDUAL_ATTEMPT_SCHEMA
        or persisted.get("attempt_id")
        != canonical_sha256(
            {key: value for key, value in persisted.items() if key != "attempt_id"}
        )
    ):
        raise ValueError("residual attempt bytes do not replay")
    artifact = _publication_identity(artifact_path)
    manifest = _publication_identity(manifest_path)
    artifact_record = _load_json_publication(artifact_path, label="residual artifact")
    manifest_record = _load_json_publication(manifest_path, label="residual manifest")
    attempt = residual_attempt_publication_identity(receipt)
    context = persisted["method_context"]
    if (
        artifact["logical_name"] != context["requested_artifact_logical_name"]
        or manifest["logical_name"] != context["requested_manifest_logical_name"]
        or artifact_record.get("residual_attempt") != attempt
        or artifact_record.get("residual_completion")
        != {"logical_name": context["declared_completion_logical_name"]}
    ):
        raise ValueError("residual artifact does not bind its exact attempt")
    _require_manifest_output(manifest_record, artifact)
    completion_path = receipt_path.with_suffix(".completion.json")
    completion_without_id = {
        "schema": R5B_RESIDUAL_COMPLETION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RESIDUAL_STAGE_PUBLICATION_COMPLETE",
        "attempt_id": persisted["attempt_id"],
        "residual_attempt_key": persisted["residual_attempt_key"],
        "artifact": artifact,
        "manifest": manifest,
        "attempt": _publication_identity(receipt_path),
        "official_status": str(official_status),
    }
    completion = completion_without_id | {
        "completion_id": canonical_sha256(completion_without_id)
    }
    _write_create_only(
        completion_path, completion, label="R5b residual attempt completion"
    )
    return completion | {"completion_path": str(completion_path)}


def verify_r5b_residual_completion(
    artifact_path: str | Path,
    manifest_path: str | Path,
    completion_path: str | Path,
    *,
    expected_stage: str,
) -> dict[str, Any]:
    """Authenticate a completed create-only bucket publication."""

    if expected_stage not in {"derivation", "validation"}:
        raise ValueError("expected residual stage differs")
    artifact = _publication_identity(artifact_path)
    manifest = _publication_identity(manifest_path)
    record = _load_json_publication(artifact_path, label="R5b residual artifact")
    manifest_record = _load_json_publication(
        manifest_path, label="R5b residual manifest"
    )
    completion = _load_json_publication(
        completion_path, label="R5b residual completion"
    )
    _require_manifest_output(manifest_record, artifact)
    if manifest_record.get("git_dirty") is not False:
        raise InstrumentError("R5b residual manifest is not clean")
    expected_schema = (
        R5B_DERIVATION_SCHEMA
        if expected_stage == "derivation"
        else R5B_VALIDATION_SCHEMA
    )
    if (
        record.get("schema") != expected_schema
        or record.get("stage") != expected_stage
        or record.get("direct_model_scoring") is not True
        or record.get("external_provider_artifact_accepted") is not False
        or completion.get("schema") != R5B_RESIDUAL_COMPLETION_SCHEMA
        or completion.get("status") != "RESIDUAL_STAGE_PUBLICATION_COMPLETE"
        or completion.get("artifact") != artifact
        or completion.get("manifest") != manifest
        or completion.get("official_status") != record.get("status")
        or completion.get("completion_id")
        != canonical_sha256(
            {key: value for key, value in completion.items() if key != "completion_id"}
        )
    ):
        raise InstrumentError("R5b residual completion does not bind the publication")
    attempt = record.get("residual_attempt")
    completion_link = record.get("residual_completion")
    expected_attempt_publication = (
        None
        if not isinstance(attempt, Mapping)
        else {
            key: attempt.get(key)
            for key in ("logical_name", "size_bytes", "sha256")
        }
    )
    if (
        not isinstance(attempt, Mapping)
        or not isinstance(completion_link, Mapping)
        or attempt.get("status") != "AUTHORIZED_RESIDUAL_BUCKET_MAY_BE_SPENT"
        or completion.get("attempt") != expected_attempt_publication
        or completion_link.get("logical_name")
        != Path(completion_path).expanduser().name
        or completion.get("attempt_id") != attempt.get("attempt_id")
        or completion.get("residual_attempt_key")
        != attempt.get("residual_attempt_key")
    ):
        raise InstrumentError("R5b residual attempt/completion chain differs")
    return {
        "artifact": artifact,
        "manifest": manifest,
        "completion": _publication_identity(completion_path),
        "record": record,
        "completion_record": completion,
        "authenticated": True,
    }


def authorize_r5b_stage(stage: str) -> None:
    if stage not in {"audit", "derivation", "validation", "freeze"}:
        raise ValueError("unknown R5b development stage")


def _required_literal_pins(stage: str) -> dict[str, Any]:
    authorize_r5b_stage(stage)
    common = {
        "FROZEN_PROTOCOL_SHA256": FROZEN_PROTOCOL_SHA256,
        "FROZEN_INSTRUMENT_FINGERPRINT_SHA256": (
            FROZEN_INSTRUMENT_FINGERPRINT_SHA256
        ),
        "FROZEN_COHORT_INPUT_SHA256": FROZEN_COHORT_INPUT_SHA256,
        "FROZEN_COHORT_AUDIT_SHA256": FROZEN_COHORT_AUDIT_SHA256,
        "FROZEN_R3D_ARTIFACT_SHA256": FROZEN_R3D_ARTIFACT_SHA256,
        "FROZEN_R3D_MANIFEST_SHA256": FROZEN_R3D_MANIFEST_SHA256,
        "FROZEN_R5A_ARTIFACT_SHA256": FROZEN_R5A_ARTIFACT_SHA256,
        "FROZEN_R5A_MANIFEST_SHA256": FROZEN_R5A_MANIFEST_SHA256,
        "FROZEN_CHECKPOINT_AGGREGATE_SHA256": (
            FROZEN_CHECKPOINT_AGGREGATE_SHA256
        ),
    }
    if stage in {"validation", "freeze"}:
        common.update(
            {
                "FROZEN_DERIVATION_ARTIFACT_SHA256": (
                    FROZEN_DERIVATION_ARTIFACT_SHA256
                ),
                "FROZEN_DERIVATION_MANIFEST_SHA256": (
                    FROZEN_DERIVATION_MANIFEST_SHA256
                ),
                "FROZEN_DERIVATION_COMPLETION_SHA256": (
                    FROZEN_DERIVATION_COMPLETION_SHA256
                ),
            }
        )
    if stage == "freeze":
        common.update(
            {
                "FROZEN_VALIDATION_ARTIFACT_SHA256": (
                    FROZEN_VALIDATION_ARTIFACT_SHA256
                ),
                "FROZEN_VALIDATION_MANIFEST_SHA256": (
                    FROZEN_VALIDATION_MANIFEST_SHA256
                ),
                "FROZEN_VALIDATION_COMPLETION_SHA256": (
                    FROZEN_VALIDATION_COMPLETION_SHA256
                ),
            }
        )
    return common


def authorize_scored_stage(stage: str, *, device: str) -> dict[str, Any]:
    """Fail closed on literal seals before any caller-supplied path is resolved."""

    if stage not in {"derivation", "validation", "freeze"}:
        raise ValueError("scored authorization is only defined for R5b scored stages")
    pins = _required_literal_pins(stage)
    missing = sorted(name for name, value in pins.items() if value is None)
    if missing:
        raise TargetSealError(
            "R5b literal freeze pins are incomplete: " + ", ".join(missing)
        )
    for name, value in pins.items():
        _validate_sha256_literal(name, value)
    assert FROZEN_PROTOCOL_SHA256 is not None
    if sha256_file(PROTOCOL) != FROZEN_PROTOCOL_SHA256:
        raise TargetSealError("R5b protocol bytes differ from their literal freeze pin")
    protocol_text = PROTOCOL.read_text(encoding="utf-8")
    if "Status: FROZEN" not in protocol_text:
        raise TargetSealError("R5b protocol is not marked FROZEN")
    binding = instrument_binding_payload()
    fingerprint = instrument_fingerprint(binding)
    if fingerprint != FROZEN_INSTRUMENT_FINGERPRINT_SHA256:
        raise TargetSealError("R5b instrument fingerprint differs from its literal pin")
    runtime_policy = None
    if stage in {"derivation", "validation"}:
        try:
            runtime_policy = validate_r5_runtime_policy(
                device=device, environ=os.environ
            )
        except Exception as exc:
            raise TargetSealError("R5b strict MPS runtime policy differs") from exc
    return {
        "stage": stage,
        "pins": pins,
        "instrument_binding_payload": binding,
        "instrument_fingerprint_sha256": fingerprint,
        "r5_stack_binding": binding["r5_stack_binding"],
        "r5_stack_fingerprint_sha256": binding[
            "r5_stack_fingerprint_sha256"
        ],
        "runtime_policy": runtime_policy,
    }


def checkpoint_identity(model_path: str | Path) -> dict[str, Any]:
    """Hash every checkpoint file in stable relative-path order."""

    root = Path(model_path).expanduser()
    if not root.is_absolute():
        root = REPO / root
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise InstrumentError("R5b checkpoint root is not a directory")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise InstrumentError("R5b checkpoint bundle contains no regular files")
    ordered = [
        {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "root": str(root),
        "files": ordered,
        "aggregate_schema": "ordered relative_path/size_bytes/sha256 list",
        "aggregate_sha256": canonical_sha256(ordered),
    }


def load_r5b_upstream_chain(
    *,
    r3d_artifact: str | Path,
    r3d_manifest: str | Path,
    r3d_registry: str | Path,
    r3d_registry_manifest: str | Path,
    r3d_attempt: str | Path,
    r3d_completion: str | Path,
    r3c_artifact: str | Path,
    r3c_manifest: str | Path,
    r5a_artifact: str | Path,
    r5a_manifest: str | Path,
    expected_stack_fingerprint: str,
) -> dict[str, Any]:
    """Authenticate R3d Path A, exclusion provenance, and R5a method preflight."""

    r3d = load_r3d_entry_package(
        r3d_artifact,
        r3d_manifest,
        registry_path=r3d_registry,
        registry_manifest_path=r3d_registry_manifest,
        attempt_path=r3d_attempt,
        completion_path=r3d_completion,
    )
    if (
        r3d.get("entry", {}).get("classification") != "SUPPORTED"
        or r3d.get("entry", {}).get("path_b_used_to_gate_entry") is not False
        or r3d.get("artifact", {}).get("sha256") != FROZEN_R3D_ARTIFACT_SHA256
        or r3d.get("manifest", {}).get("sha256") != FROZEN_R3D_MANIFEST_SHA256
    ):
        raise TargetSealError("R5b R3d Path A package differs from literal entry pins")
    r3c = load_r3c_exclusion_package(r3c_artifact, r3c_manifest)
    registry = derive_r5_exclusion_registry(r3c, r3d)
    unpacked = validate_r5_exclusion_registry(registry)
    r5a = verify_r5a_entry_package(
        r5a_artifact,
        r5a_manifest,
        expected_stack_fingerprint=expected_stack_fingerprint,
    )
    r5a_checkpoint = r5a["entry_contract"].get("checkpoint_sha256")
    r3d_checkpoint = r3d["checkpoint_and_data"].get(
        "checkpoint_aggregate_sha256"
    )
    if r5a_checkpoint != r3d_checkpoint:
        raise TargetSealError("R5a preflight and R3d checkpoint identities differ")
    return {
        "r3d": r3d,
        "r3c": r3c,
        "exclusion_registry": registry,
        "exclusion_registry_unpacked": unpacked,
        "r5a": r5a,
        "chain_sha256": canonical_sha256(
            {
                "r3d": r3d["package_contract_sha256"],
                "r3c": r3c["package_contract_sha256"],
                "registry": registry["record_sha256"],
                "r5a_artifact": r5a["artifact"]["sha256"],
                "r5a_manifest": r5a["manifest"]["sha256"],
            }
        ),
    }


def build_r5b_residual_reservations(
    gtf: str | Path,
    fasta: str | Path,
    *,
    exclusion_registry: Mapping[str, Any],
) -> tuple[dict[int, tuple[R5CohortRow, ...]], dict[str, Any], dict[str, Any]]:
    """Construct only the two R5b residual-validation reservations."""

    source = validate_expected_r5_source_identity(gtf, fasta)
    unpacked = validate_r5_exclusion_registry(exclusion_registry)
    examples = load_examples_unscored(
        str(gtf),
        str(fasta),
        "val",
        None,
        tag="r5-acceptor-mechanism-v1",
        allowed_chromosomes=R5_VALIDATION_CHROMOSOMES,
    )
    pseudo = {
        str(example.example_id): matched_decoy_donor(example) for example in examples
    }
    reservations, audit = build_r5_reservations(
        examples,
        pseudo,
        historical_r3_gene_ids=unpacked["historical_r3_gene_ids"],
        r3c_gene_ids=unpacked["r3c_gene_ids"],
        r3d_registry_gene_ids=unpacked["r3d_registry_gene_ids"],
        r3d_target_gene_ids=unpacked["r3d_target_gene_ids"],
        rows_per_bucket=PRODUCTION_N,
        require_capacity=True,
    )
    validate_r5_reservation_audit(audit, rows_per_bucket=PRODUCTION_N)
    validate_expected_r5_census_identity(audit)
    cohort_input = {
        "source_identity": source,
        "exclusion_registry_record_sha256": exclusion_registry["record_sha256"],
        "construction": "build_r5_reservations@r5-acceptor-mechanism-v1",
    }
    if canonical_sha256(cohort_input) != FROZEN_COHORT_INPUT_SHA256:
        raise TargetSealError("R5b cohort input differs from its literal pin")
    if canonical_sha256(audit) != FROZEN_COHORT_AUDIT_SHA256:
        raise TargetSealError("R5b cohort audit differs from its literal pin")
    return reservations, audit, source


def _verify_exact_residual_predecessor(
    artifact_path: str | Path,
    manifest_path: str | Path,
    completion_path: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    expected = {
        "derivation": (
            FROZEN_DERIVATION_ARTIFACT_SHA256,
            FROZEN_DERIVATION_MANIFEST_SHA256,
            FROZEN_DERIVATION_COMPLETION_SHA256,
        ),
        "validation": (
            FROZEN_VALIDATION_ARTIFACT_SHA256,
            FROZEN_VALIDATION_MANIFEST_SHA256,
            FROZEN_VALIDATION_COMPLETION_SHA256,
        ),
    }
    if stage not in expected:
        raise ValueError("residual predecessor stage differs")
    package = verify_r5b_residual_completion(
        artifact_path,
        manifest_path,
        completion_path,
        expected_stage=stage,
    )
    observed = (
        package["artifact"]["sha256"],
        package["manifest"]["sha256"],
        package["completion"]["sha256"],
    )
    if observed != expected[stage]:
        raise TargetSealError(f"R5b {stage} package differs from literal pins")
    if stage == "derivation":
        _assert_development_record_replays(
            package["record"], require_production=True
        )
    return package


def run_model_free_audit(*, n_draws: int = 199) -> dict[str, Any]:
    """Exercise the v2 algebra and registries without opening any scientific input."""

    n = 64
    genes = [f"audit-gene-{index:03d}" for index in range(n)]
    strong = np.linspace(0.20, 0.24, n)
    null = np.linspace(-0.002, 0.002, n)
    classifier = robust_studentized_family_inference(
        {"material": strong, "equivalent": null, "health": null},
        {
            "material": "material_positive",
            "equivalent": "equivalent",
            "health": "noninferior",
        },
        genes,
        n_draws=n_draws,
        alpha=FAMILY_ALPHA,
        seed=0,
        tier="development",
    )
    context = torch.arange(2 * 3 * 12, dtype=torch.float64).reshape(2, 3, 12)
    weight = torch.eye(12, dtype=torch.float64)
    contributions = [
        output_projected_head_contribution(
            context, weight, head_index=head, num_heads=3
        )
        for head in range(3)
    ]
    decomposition_error = float(
        torch.max(torch.abs(sum(contributions) - context)).item()
    )
    calibration = prospective_calibration_trial_count()
    primitive = causal_primitive_names()
    family = causal_family_names()
    edges = causal_boundary_edge_registry()
    checks = {
        "writer_projection_decomposes_exactly": decomposition_error == 0.0,
        "robust_classifier_plant_passes": classifier["all_requirements_pass"] is True,
        "primitive_registry_unique": len(primitive) == len(set(primitive)),
        "family_registry_unique": len(family) == len(set(family)),
        "boundary_registry_unique": len(edges)
        == len({row["edge_id"] for row in edges}),
        "calibration_covers_every_edge_twice": calibration["trial_count"]
        == 2 * len(edges),
        "production_bootstrap_floor_at_least_10000": (
            MIN_PRODUCTION_INFERENCE_DRAWS >= 10_000
        ),
        "all_artifact_pins_have_manifest_pairs": all(
            (artifact is None) == (manifest is None)
            for artifact, manifest in (
                (FROZEN_R3D_ARTIFACT_SHA256, FROZEN_R3D_MANIFEST_SHA256),
                (FROZEN_R5A_ARTIFACT_SHA256, FROZEN_R5A_MANIFEST_SHA256),
                (
                    FROZEN_DERIVATION_ARTIFACT_SHA256,
                    FROZEN_DERIVATION_MANIFEST_SHA256,
                ),
                (
                    FROZEN_VALIDATION_ARTIFACT_SHA256,
                    FROZEN_VALIDATION_MANIFEST_SHA256,
                ),
                (
                    FROZEN_R5B_DISPOSITION_ARTIFACT_SHA256,
                    FROZEN_R5B_DISPOSITION_MANIFEST_SHA256,
                ),
            )
        ),
    }
    return {
        "schema": "gfmcd.r5b.model_free_audit/2",
        "stage": "audit",
        "status": "DEVELOPMENT",
        "primary_state": {
            "writer_site_id": WRITER_SITE_ID,
            "representation": (
                "post_W_O_L3H1_additive_contribution_at_acceptor_query"
            ),
            "temporal_position": "before L4/L5",
            "dense_path_role": DENSE_PATH_ROLE,
            "nested_subspace_role": SUBSPACE_PATH_ROLE,
            "dense_depends_on_low_rank": False,
        },
        "candidate_lifecycle": {
            "derive_bucket": DERIVATION_BUCKET,
            "validate_bucket": VALIDATION_BUCKET,
            "union_refit_after_validation": True,
            "target_refit_or_selection": False,
        },
        "classifier": classifier,
        "calibration": calibration,
        "registry_counts": {
            "primitive": len(primitive),
            "family": len(family),
            "decision_edges": len(edges),
        },
        "projection_decomposition_max_abs": decomposition_error,
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "model_loaded": False,
        "genome_opened": False,
        "sealed_chr8_chr9_opened": False,
        "mechanism_claim_licensed": False,
    }


def _attention_output_dense_module(adapter: Any) -> torch.nn.Module:
    """Resolve the exact L3 attention output projection used by the writer."""

    try:
        module = adapter.encoder_layers[WRITER_LAYER].attention.output.dense
    except (AttributeError, IndexError, TypeError) as error:
        raise InstrumentError(
            "checkpoint does not expose the frozen L3 attention output projection"
        ) from error
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise InstrumentError("L3 attention output projection lacks a matrix weight")
    return module


def checkpoint_readout_axis(adapter: Any, task: Any) -> np.ndarray:
    """Return the fixed G-minus-mean(A,C,T) decoder axis in hidden width."""

    current = getattr(adapter, "model", None)
    for path in (
        "cls.predictions.decoder",
        "cls.decoder",
        "lm_head.decoder",
        "lm_head",
    ):
        candidate = current
        for part in path.split("."):
            if candidate is None or not hasattr(candidate, part):
                candidate = None
                break
            candidate = getattr(candidate, part)
        weight = getattr(candidate, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            break
    else:
        raise InstrumentError("checkpoint does not expose its MLM decoder weight")
    try:
        ids = {base: int(task.bid[base]) for base in "ACGT"}
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise InstrumentError("masked-acceptor task lacks nucleotide token IDs") from error
    if any(index < 0 or index >= int(weight.shape[0]) for index in ids.values()):
        raise InstrumentError("nucleotide token ID lies outside the decoder")
    axis = (
        weight[ids["G"]]
        - torch.stack([weight[ids[base]] for base in "ACT"], dim=0).mean(dim=0)
    )
    value = np.asarray(axis.detach().cpu(), dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.ndim != 1 or not np.isfinite(value).all() or norm <= 0.0:
        raise InstrumentError("checkpoint readout axis is not finite and nonzero")
    return np.ascontiguousarray(value / norm, dtype=np.float64)


def _capture_writer_batch(
    adapter: Any,
    task: Any,
    encoded: Mapping[str, torch.Tensor],
    readout_positions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Capture L3H1 context/write and native full-MLM margins in one forward."""

    holder: dict[str, Any] = {}

    def run_forward() -> Any:
        result = adapter.model(**encoded)
        holder["result"] = result
        return result

    cache = capture_head_outputs(adapter, run_forward)
    if WRITER_LAYER not in cache or "result" not in holder:
        raise InstrumentError("L3H1 writer capture did not execute")
    heads = cache[WRITER_LAYER]
    if (
        not isinstance(heads, torch.Tensor)
        or heads.ndim != 4
        or WRITER_HEAD >= int(heads.shape[1])
        or int(heads.shape[0]) != len(readout_positions)
    ):
        raise InstrumentError("captured L3 head tensor has the wrong shape")
    query = torch.as_tensor(
        [int(position) + int(task.off) for position in readout_positions],
        device=heads.device,
        dtype=torch.long,
    )
    if bool(((query < 0) | (query >= int(heads.shape[2]))).any()):
        raise InstrumentError("masked acceptor query lies outside the captured context")
    rows = torch.arange(int(heads.shape[0]), device=heads.device)
    context = heads[rows, WRITER_HEAD, query]
    contribution = project_head_context_to_residual(
        adapter, (WRITER_LAYER, WRITER_HEAD), context
    )
    logits = getattr(holder["result"], "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise InstrumentError("checkpoint forward lacks MLM logits")
    margins = task.margin(logits.float(), readout_positions)
    arrays = tuple(
        np.ascontiguousarray(value.detach().cpu(), dtype=np.float64)
        for value in (context, contribution, margins)
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise InstrumentError("writer capture produced a non-finite tensor")
    return arrays  # type: ignore[return-value]


def capture_residual_writer_state(
    adapter: Any,
    task: Any,
    rows: Sequence[R5CohortRow],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Capture clean, paired-corrupt, and pseudo-corrupt L3H1 writes for one bucket."""

    if len(rows) != PRODUCTION_N or len(
        {str(row.example.gene_id) for row in rows}
    ) != PRODUCTION_N:
        raise InstrumentError("R5b scoring requires 128 one-row-per-gene records")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    captured: dict[str, dict[str, list[np.ndarray]]] = {
        leg: {"context": [], "writer": [], "margin": []}
        for leg in ("clean", "partner", "pseudo")
    }
    for start in range(0, len(rows), batch_size):
        chunk = list(rows[start : start + batch_size])
        readout = [int(row.acceptor_g) for row in chunk]
        sequences = {
            "clean": [row.clean_sequence for row in chunk],
            "partner": [row.paired_corrupt_sequence for row in chunk],
            "pseudo": [row.pseudo_corrupt_sequence for row in chunk],
        }
        for leg, values in sequences.items():
            encoded = task.encode(values, readout)
            context, writer, margin = _capture_writer_batch(
                adapter, task, encoded, readout
            )
            captured[leg]["context"].append(context)
            captured[leg]["writer"].append(writer)
            captured[leg]["margin"].append(margin)
    merged = {
        leg: {
            name: np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float64)
            for name, parts in values.items()
        }
        for leg, values in captured.items()
    }
    dense = _attention_output_dense_module(adapter)
    weight = dense.weight.detach().cpu()
    hidden_in = int(weight.shape[1])
    num_heads = int(adapter.num_heads)
    if hidden_in % num_heads:
        raise InstrumentError("L3 output projection width is not head-divisible")
    head_width = hidden_in // num_heads
    lo = WRITER_HEAD * head_width
    block = np.ascontiguousarray(weight[:, lo : lo + head_width], dtype=np.float64)
    clean_context = merged["clean"]["context"]
    partner_context = merged["partner"]["context"]
    pseudo_context = merged["pseudo"]["context"]
    checkpoint_dtype = str(weight.dtype).removeprefix("torch.")
    provider_record = {
        "writer_site_id": WRITER_SITE_ID,
        "clean_head_context": clean_context,
        "corrupt_head_context": partner_context,
        "w_o_column_block": block,
        "clean_writer_contribution": merged["clean"]["writer"],
        "corrupt_writer_contribution": merged["partner"]["writer"],
        "writer_delta": merged["clean"]["writer"] - merged["partner"]["writer"],
        "pseudo_head_context": pseudo_context,
        "pseudo_writer_contribution": merged["pseudo"]["writer"],
        "pseudo_delta": merged["clean"]["writer"] - merged["pseudo"]["writer"],
        "numeric_provenance": _post_wo_numeric_provenance(checkpoint_dtype),
        "bias_included": False,
    }
    replay_output_projected_writer_provider(provider_record)
    return {
        "provider_record": provider_record,
        "clean_writer": merged["clean"]["writer"],
        "partner_writer": merged["partner"]["writer"],
        "pseudo_writer": merged["pseudo"]["writer"],
        "partner_delta": merged["clean"]["writer"] - merged["partner"]["writer"],
        "pseudo_delta": merged["clean"]["writer"] - merged["pseudo"]["writer"],
        "margins": {leg: merged[leg]["margin"] for leg in merged},
        "w_o_column_block_sha256": array_sha256(block),
        "readout_axis": checkpoint_readout_axis(adapter, task),
    }


def _margin_with_output_delta(
    adapter: Any,
    task: Any,
    rows: Sequence[R5CohortRow],
    *,
    sequence_leg: str,
    delta: Any,
    query_offset: int,
    batch_size: int,
) -> np.ndarray:
    """Run the complete checkpoint with one post-W_O delta at one query per row."""

    if sequence_leg not in {"clean", "partner", "pseudo"}:
        raise ValueError("sequence leg must be clean, partner, or pseudo")
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(rows) or not np.isfinite(values).all():
        raise ValueError("local-writer delta must be finite [row,residual_width]")
    output: list[np.ndarray] = []
    dense = _attention_output_dense_module(adapter)
    for start in range(0, len(rows), batch_size):
        chunk = list(rows[start : start + batch_size])
        readout = [int(row.acceptor_g) for row in chunk]
        sequence = {
            "clean": [row.clean_sequence for row in chunk],
            "partner": [row.paired_corrupt_sequence for row in chunk],
            "pseudo": [row.pseudo_corrupt_sequence for row in chunk],
        }[sequence_leg]
        for row, position in zip(chunk, readout):
            target = position + int(query_offset)
            if target < 0 or target >= len(row.clean_sequence):
                raise InstrumentError("local-writer query-offset control is out of bounds")
        encoded = task.encode(sequence, readout)
        mask = torch.zeros_like(encoded["input_ids"], dtype=torch.bool)
        token_positions = torch.as_tensor(
            [position + int(task.off) + int(query_offset) for position in readout],
            device=mask.device,
            dtype=torch.long,
        )
        if bool(((token_positions < 0) | (token_positions >= mask.shape[1])).any()):
            raise InstrumentError("local-writer token mask includes padding/special bounds")
        mask[torch.arange(len(chunk), device=mask.device), token_positions] = True
        calls = 0

        def count_hook(_module: Any, _inputs: Any, _result: Any) -> None:
            nonlocal calls
            calls += 1

        counter = dense.register_forward_hook(count_hook)
        try:
            with add_attention_output_delta(
                adapter,
                WRITER_LAYER,
                torch.as_tensor(values[start : start + len(chunk)]),
                token_mask=mask,
            ):
                result = adapter.model(**encoded)
        finally:
            counter.remove()
        if calls != 1:
            raise InstrumentError("post-W_O local-writer hook did not execute exactly once")
        logits = getattr(result, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise InstrumentError("intervened checkpoint forward lacks MLM logits")
        output.append(
            np.ascontiguousarray(
                task.margin(logits.float(), readout).detach().cpu(), dtype=np.float64
            )
        )
    merged = np.ascontiguousarray(np.concatenate(output), dtype=np.float64)
    if not np.isfinite(merged).all():
        raise InstrumentError("local-writer intervention produced non-finite margins")
    return merged


def _axis_projection(rows: Any, axis: Any) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    direction = np.asarray(axis, dtype=np.float64)
    if values.ndim != 2 or direction.shape != (values.shape[1],):
        raise ValueError("axis projection dimensions differ")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction).all() or norm <= 0.0:
        raise InstrumentError("control axis is not finite and nonzero")
    unit = direction / norm
    return np.ascontiguousarray((values @ unit)[:, None] * unit[None, :])


def _score_delta_record(
    adapter: Any,
    task: Any,
    rows: Sequence[R5CohortRow],
    captured: Mapping[str, Any],
    *,
    delta: Any,
    sequence_leg: str,
    query_offset: int,
    leg_id: str,
    batch_size: int,
    basis_id: str | None = None,
) -> dict[str, Any]:
    """Score one dense or projected writer sufficiency/necessity four-cell arm."""

    values = np.asarray(delta, dtype=np.float64)
    corrupt_margin = np.asarray(captured["margins"][sequence_leg], dtype=np.float64)
    clean_margin = np.asarray(captured["margins"]["clean"], dtype=np.float64)
    restored = _margin_with_output_delta(
        adapter,
        task,
        rows,
        sequence_leg=sequence_leg,
        delta=values,
        query_offset=query_offset,
        batch_size=batch_size,
    )
    removed = _margin_with_output_delta(
        adapter,
        task,
        rows,
        sequence_leg="clean",
        delta=-values,
        query_offset=query_offset,
        batch_size=batch_size,
    )
    if basis_id is None:
        return local_writer_dense_estimands(
            m_clean=clean_margin,
            m_corrupt=corrupt_margin,
            m_corrupt_with_clean_writer=restored,
            m_clean_with_corrupt_writer=removed,
            leg_id=leg_id,
            writer_site_id=WRITER_SITE_ID,
        )
    return local_writer_subspace_estimands(
        m_clean=clean_margin,
        m_corrupt=corrupt_margin,
        m_corrupt_with_projected_writer=restored,
        m_clean_without_projected_writer=removed,
        leg_id=leg_id,
        writer_site_id=WRITER_SITE_ID,
        basis_id=basis_id,
    )


def score_dense_local_writer_family(
    adapter: Any,
    task: Any,
    rows: Sequence[R5CohortRow],
    captured: Mapping[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Score the basis-independent primary writer family through the full MLM."""

    partner_delta = np.asarray(captured["partner_delta"], dtype=np.float64)
    pseudo_delta = np.asarray(captured["pseudo_delta"], dtype=np.float64)
    generic_axis = np.mean(partner_delta, axis=0)
    if float(np.linalg.norm(generic_axis)) <= 0.0:
        raise InstrumentError("bucket-wide mean writer difference is zero")
    readout_axis = np.asarray(captured["readout_axis"], dtype=np.float64)
    partner = _score_delta_record(
        adapter,
        task,
        rows,
        captured,
        delta=partner_delta,
        sequence_leg="partner",
        query_offset=0,
        leg_id="partner",
        batch_size=batch_size,
    )
    pseudo = _score_delta_record(
        adapter,
        task,
        rows,
        captured,
        delta=pseudo_delta,
        sequence_leg="pseudo",
        query_offset=0,
        leg_id="pseudo",
        batch_size=batch_size,
    )
    adjacent = {
        offset: _score_delta_record(
            adapter,
            task,
            rows,
            captured,
            delta=partner_delta,
            sequence_leg="partner",
            query_offset=offset,
            leg_id=f"adjacent:{offset:+d}",
            batch_size=batch_size,
        )
        for offset in ADJACENT_TOKEN_OFFSETS
    }
    generic = _score_delta_record(
        adapter,
        task,
        rows,
        captured,
        delta=_axis_projection(partner_delta, generic_axis),
        sequence_leg="partner",
        query_offset=0,
        leg_id="gene_generic",
        batch_size=batch_size,
    )
    readout = _score_delta_record(
        adapter,
        task,
        rows,
        captured,
        delta=_axis_projection(partner_delta, readout_axis),
        sequence_leg="partner",
        query_offset=0,
        leg_id="readout_axis",
        batch_size=batch_size,
    )
    zero = np.zeros_like(partner_delta)
    clean_noop = _margin_with_output_delta(
        adapter,
        task,
        rows,
        sequence_leg="clean",
        delta=zero,
        query_offset=0,
        batch_size=batch_size,
    ) - np.asarray(captured["margins"]["clean"], dtype=np.float64)
    corrupt_noop = _margin_with_output_delta(
        adapter,
        task,
        rows,
        sequence_leg="partner",
        delta=zero,
        query_offset=0,
        batch_size=batch_size,
    ) - np.asarray(captured["margins"]["partner"], dtype=np.float64)
    return build_dense_local_writer_family(
        partner=partner,
        pseudo=pseudo,
        adjacent_by_offset=adjacent,
        gene_generic=generic,
        readout_axis=readout,
        health={"clean_noop": clean_noop, "corrupt_noop": corrupt_noop},
    )


def _project_rows_with_bases(rows: np.ndarray, row_bases: np.ndarray) -> np.ndarray:
    if rows.ndim != 2 or row_bases.ndim != 3 or (
        row_bases.shape[0] != rows.shape[0]
        or row_bases.shape[2] != rows.shape[1]
    ):
        raise ValueError("row-specific basis dimensions differ")
    coefficients = np.einsum("nd,nkd->nk", rows, row_bases)
    return np.ascontiguousarray(
        np.einsum("nk,nkd->nd", coefficients, row_bases), dtype=np.float64
    )


def crossfit_projection_package(
    writer_delta: Any,
    crossfit: Mapping[str, Any],
    *,
    rank: int,
    seed: int,
) -> dict[str, Any]:
    """Build outcome-free row-specific cross-fit projections and 32 exact nulls."""

    rows = np.asarray(writer_delta, dtype=np.float64)
    folds = np.asarray(crossfit.get("fold_ids"), dtype=np.int64)
    if rows.ndim != 2 or folds.shape != (rows.shape[0],) or rank not in RANK_GRID:
        raise ValueError("cross-fit projection inputs differ")
    if rows.shape[0] != PRODUCTION_N or set(folds.tolist()) != {0, 1}:
        raise ValueError("cross-fit projections require both exact 64-row folds")
    width = int(rows.shape[1])
    row_bases = np.empty((rows.shape[0], rank, width), dtype=np.float64)
    random_row_bases = np.empty(
        (N_RANDOM_CONTROLS, rows.shape[0], rank, width), dtype=np.float64
    )
    fold_basis_ids: dict[str, str] = {}
    random_fold_basis_ids: dict[str, list[str]] = {}
    records = crossfit.get("evaluation_folds")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("cross-fit basis package lacks both evaluation folds")
    for record in records:
        fold = int(record.get("evaluation_fold", -1))
        fit = record.get("fit")
        if fold not in {0, 1} or not isinstance(fit, Mapping):
            raise ValueError("cross-fit evaluation-fold record differs")
        mask = folds == fold
        basis = np.asarray(fit.get("basis"), dtype=np.float64)[:rank]
        orthogonal_projector(basis)
        row_bases[mask] = basis
        fold_basis_ids[str(fold)] = "sha256:" + array_sha256(basis)
        ids: list[str] = []
        for draw in range(N_RANDOM_CONTROLS):
            random_basis = deterministic_nested_random_bases(
                width,
                (rank,),
                seed=int(seed),
                namespace=f"r5b-v2-crossfit-rank-{rank}-fold-{fold}",
                draw=draw,
            )[rank]
            random_row_bases[draw, mask] = random_basis
            ids.append("sha256:" + array_sha256(random_basis))
        random_fold_basis_ids[str(fold)] = ids
    target = _project_rows_with_bases(rows, row_bases)
    random = np.empty((N_RANDOM_CONTROLS, rows.shape[0], width), dtype=np.float64)
    for draw in range(N_RANDOM_CONTROLS):
        raw = _project_rows_with_bases(rows, random_row_bases[draw])
        random[draw] = energy_match_rows(target, raw)
    energy_error = float(
        np.max(
            np.abs(
                np.linalg.norm(random, axis=2)
                - np.linalg.norm(target, axis=1)[None, :]
            )
        )
    )
    tolerance = 1e-10 * max(1.0, float(np.max(np.linalg.norm(target, axis=1))))
    if energy_error > tolerance:
        raise OptionalBasisInstrumentError(
            "RANK_ENERGY_MATCH_FAILURE",
            "cross-fit random controls are not row-energy matched",
        )
    basis_id = "sha256:" + canonical_sha256(fold_basis_ids)
    random_ids = [
        "sha256:"
        + canonical_sha256(
            {str(fold): random_fold_basis_ids[str(fold)][draw] for fold in (0, 1)}
        )
        for draw in range(N_RANDOM_CONTROLS)
    ]
    overlap = mean_squared_principal_cosine(
        row_bases[folds == 0][0], row_bases[folds == 1][0]
    )
    return {
        "target_projection": target,
        "random_projections": random,
        "row_bases": row_bases,
        "random_row_bases": random_row_bases,
        "crossfold_overlap": float(overlap),
        "crossfold_stability_pass": bool(overlap > 0.20),
        "audit": {
            "schema": "gfmcd.r5b.crossfit_rank_energy_projection_audit/2",
            "rank": int(rank),
            "width": width,
            "n_rows": int(rows.shape[0]),
            "n_random_controls": N_RANDOM_CONTROLS,
            "target_basis_id": basis_id,
            "random_basis_ids": random_ids,
            "fold_basis_ids": fold_basis_ids,
            "random_fold_basis_ids_sha256": canonical_sha256(random_fold_basis_ids),
            "target_projection_sha256": array_sha256(target),
            "random_projections_sha256": array_sha256(random),
            "max_abs_row_energy_error": energy_error,
            "energy_match_tolerance": tolerance,
            "all_projector_ranks_match": True,
            "all_rows_energy_matched": True,
            "applied_basis_fit_excludes_evaluation_fold": True,
        },
    }


def score_nested_local_writer_family(
    adapter: Any,
    task: Any,
    rows: Sequence[R5CohortRow],
    captured: Mapping[str, Any],
    *,
    target_projection: Any,
    pseudo_projection: Any,
    random_projections: Any,
    projection_audit: Mapping[str, Any],
    rank: int,
    batch_size: int,
) -> dict[str, Any]:
    """Score one prespecified nested basis family through the complete checkpoint."""

    target = np.asarray(target_projection, dtype=np.float64)
    pseudo_delta = np.asarray(pseudo_projection, dtype=np.float64)
    random = np.asarray(random_projections, dtype=np.float64)
    basis_id = str(projection_audit.get("target_basis_id"))
    random_ids = projection_audit.get("random_basis_ids")
    if (
        target.shape != np.asarray(captured["partner_delta"]).shape
        or pseudo_delta.shape != target.shape
        or random.shape != (N_RANDOM_CONTROLS, *target.shape)
        or not isinstance(random_ids, list)
        or len(random_ids) != N_RANDOM_CONTROLS
    ):
        raise ValueError("nested projected writer arrays differ")
    score = lambda **kwargs: _score_delta_record(  # noqa: E731
        adapter, task, rows, captured, batch_size=batch_size, **kwargs
    )
    partner = score(
        delta=target,
        sequence_leg="partner",
        query_offset=0,
        leg_id="partner",
        basis_id=basis_id,
    )
    pseudo = score(
        delta=pseudo_delta,
        sequence_leg="pseudo",
        query_offset=0,
        leg_id="pseudo",
        basis_id=basis_id,
    )
    adjacent = {
        offset: score(
            delta=target,
            sequence_leg="partner",
            query_offset=offset,
            leg_id=f"adjacent:{offset:+d}",
            basis_id=basis_id,
        )
        for offset in ADJACENT_TOKEN_OFFSETS
    }
    partner_delta = np.asarray(captured["partner_delta"], dtype=np.float64)
    generic = score(
        delta=_axis_projection(partner_delta, np.mean(partner_delta, axis=0)),
        sequence_leg="partner",
        query_offset=0,
        leg_id="gene_generic",
        basis_id=basis_id,
    )
    readout = score(
        delta=_axis_projection(partner_delta, captured["readout_axis"]),
        sequence_leg="partner",
        query_offset=0,
        leg_id="readout_axis",
        basis_id=basis_id,
    )
    random_records = {
        draw: score(
            delta=random[draw],
            sequence_leg="partner",
            query_offset=0,
            leg_id=f"random:{draw:02d}",
            basis_id=str(random_ids[draw]),
        )
        for draw in range(N_RANDOM_CONTROLS)
    }
    return build_nested_local_writer_family(
        partner=partner,
        pseudo=pseudo,
        adjacent_by_offset=adjacent,
        gene_generic=generic,
        readout_axis=readout,
        random_by_draw=random_records,
        rank=rank,
        projection_audit=projection_audit,
    )


def _internal_scored_provider(
    rows: Sequence[R5CohortRow],
    captured: Mapping[str, Any],
    *,
    phase: str,
    dense_family: Mapping[str, Any],
    nested_families: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Serialize sufficient statistics produced only by this process's model scorer."""

    bucket = DERIVATION_BUCKET if phase == "derivation" else VALIDATION_BUCKET
    return {
        "schema": R5B_SCORED_PROVIDER_SCHEMA,
        "origin": "direct_checkpoint_scoring_in_this_r5b_process",
        "external_provider_artifact_accepted": False,
        "phase": phase,
        "residual_validation_bucket": bucket,
        "chromosomes": list(R5_VALIDATION_CHROMOSOMES),
        "sealed_chr8_chr9_opened": False,
        "masked_donor_endpoint_constructed": False,
        "writer_site_id": WRITER_SITE_ID,
        "genes": [str(row.example.gene_id) for row in rows],
        "row_ids": [str(row.example.example_id) for row in rows],
        "output_projected_writer": captured["provider_record"],
        "readout_axis": captured["readout_axis"],
        "dense_family": dense_family,
        "nested_rank_families": {
            str(int(rank)): family for rank, family in sorted(nested_families.items())
        },
    }


def score_r5b_derivation(
    adapter: Any,
    task: Any,
    rows: Sequence[R5CohortRow],
    *,
    batch_size: int = PRODUCTION_BATCH,
    n_draws: int = PRODUCTION_INFERENCE_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run bucket 0 from checkpoint forwards through frozen rank selection."""

    genes = [str(row.example.gene_id) for row in rows]
    row_ids = [str(row.example.example_id) for row in rows]
    captured = capture_residual_writer_state(
        adapter, task, rows, batch_size=batch_size
    )
    dense_family = score_dense_local_writer_family(
        adapter, task, rows, captured, batch_size=batch_size
    )
    dense_decision = classify_dense_local_writer(
        dense_family, genes, n_draws=n_draws, seed=seed
    )
    instrument_failure = dense_decision["status"] == (
        "INSTRUMENT_UNRESOLVED_NOOP_HEALTH"
    )
    crossfit: dict[str, Any] | None = None
    nested_families: dict[int, dict[str, Any]] = {}
    rank_decisions: dict[int, dict[str, Any]] = {}
    outcomes: dict[int, dict[str, Any]] | None = None
    candidate: dict[str, Any] | None = None
    nested_basis_instrument_failure: dict[str, Any] | None = None
    if (
        not instrument_failure
        and dense_decision["status"] == "DENSE_LOCAL_WRITER_DEVELOPMENT_SUPPORTED"
    ):
        try:
            crossfit = fit_crossfit_derivation_bases(
                captured["partner_delta"], genes, maximum_rank=max(RANK_GRID)
            )
            outcomes = {}
            for rank in RANK_GRID:
                projections = crossfit_projection_package(
                    captured["partner_delta"], crossfit, rank=rank, seed=seed
                )
                pseudo_projection = _project_rows_with_bases(
                    np.asarray(captured["pseudo_delta"], dtype=np.float64),
                    np.asarray(projections["row_bases"], dtype=np.float64),
                )
                projection_audit = dict(projections["audit"]) | {
                    "pseudo_projection_sha256": array_sha256(pseudo_projection)
                }
                family = score_nested_local_writer_family(
                    adapter,
                    task,
                    rows,
                    captured,
                    target_projection=projections["target_projection"],
                    pseudo_projection=pseudo_projection,
                    random_projections=projections["random_projections"],
                    projection_audit=projection_audit,
                    rank=rank,
                    batch_size=batch_size,
                )
                decision = classify_nested_local_writer_family(
                    family,
                    genes,
                    n_draws=n_draws,
                    seed=seed,
                    phase="derivation",
                )
                nested_families[rank] = family
                rank_decisions[rank] = decision
                partner_s = np.asarray(
                    family["endpoints"]["partner|S"], dtype=np.float64
                )
                point, _se = _statistics_and_standard_errors(partner_s[None, :])
                outcomes[rank] = {
                    "mean_partner_s": float(point["mean"][0]),
                    "trimmed_partner_s": float(point["trimmed_mean_10pct"][0]),
                    "nested_path_pass": bool(decision["nested_path_pass"]),
                    "crossfold_stability_pass": bool(
                        projections["crossfold_stability_pass"]
                    ),
                    "all_rank_energy_random_controls_pass": bool(
                        decision["all_rank_energy_random_controls_pass"]
                    ),
                }
            candidate = select_derivation_candidate_rank(
                crossfit["whole_bucket_fit"],
                outcomes,
                dense_development_status=dense_decision["status"],
            )
        except OptionalBasisInstrumentError as error:
            instrument_failure = True
            nested_basis_instrument_failure = _optional_basis_failure_record(
                "derivation", error
            )
    provider = _internal_scored_provider(
        rows,
        captured,
        phase="derivation",
        dense_family=dense_family,
        nested_families=nested_families,
    )
    status = (
        "BASIS_INSTRUMENT_UNRESOLVED"
        if instrument_failure
        else "DERIVATION_CANDIDATE_SELECTED"
        if candidate is not None and candidate.get("selected_rank") is not None
        else "NO_DERIVATION_CANDIDATE_AT_FROZEN_THRESHOLDS"
    )
    return _json_native(
        {
            "schema": R5B_DERIVATION_SCHEMA,
            "stage": "derivation",
            "status": status,
            "writer_site_id": WRITER_SITE_ID,
            "residual_validation_bucket": DERIVATION_BUCKET,
            "genes": genes,
            "row_ids": row_ids,
            "direct_model_scoring": True,
            "external_provider_artifact_accepted": False,
            "post_wo_writer_audit": {
                key: value
                for key, value in replay_output_projected_writer_provider(
                    captured["provider_record"]
                ).items()
                if key not in {"writer_delta", "provider_record"}
            },
            "readout_axis_sha256": array_sha256(captured["readout_axis"]),
            "provider_record_sha256": canonical_sha256(
                replay_output_projected_writer_provider(
                    captured["provider_record"]
                )["provider_record"]
            ),
            "scored_provider": provider,
            "inference_draws": int(n_draws),
            "dense_decision": dense_decision,
            "crossfit_derivation": crossfit,
            "nested_rank_decisions": rank_decisions,
            "rank_outcomes": outcomes,
            "candidate": candidate,
            "instrument_failure": instrument_failure,
            "nested_basis_instrument_failure": nested_basis_instrument_failure,
            "claim_eligible": False,
            "residual_validation_only": True,
            "sealed_chr8_chr9_opened": False,
            "masked_donor_endpoint_constructed": False,
        }
    )


def score_r5b_validation(
    adapter: Any,
    task: Any,
    rows: Sequence[R5CohortRow],
    derivation: Mapping[str, Any],
    *,
    batch_size: int = PRODUCTION_BATCH,
    n_draws: int = PRODUCTION_INFERENCE_DRAWS,
    n_overlap_draws: int = PRODUCTION_OVERLAP_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run bucket 1 with exactly the already selected bucket-0 candidate."""

    if derivation.get("schema") != R5B_DERIVATION_SCHEMA:
        raise InstrumentError("bucket-1 scoring lacks the exact derivation record")
    _assert_development_record_replays(derivation)
    genes = [str(row.example.gene_id) for row in rows]
    row_ids = [str(row.example.example_id) for row in rows]
    if set(row_ids) & set(str(value) for value in derivation.get("row_ids", ())):
        raise InstrumentError("bucket-0 and bucket-1 row identities overlap")
    captured = capture_residual_writer_state(
        adapter, task, rows, batch_size=batch_size
    )
    dense_family = score_dense_local_writer_family(
        adapter, task, rows, captured, batch_size=batch_size
    )
    dense_decision = classify_dense_local_writer(
        dense_family, genes, n_draws=n_draws, seed=seed
    )
    instrument_failure = bool(
        derivation.get("instrument_failure") is True
        or dense_decision["status"] == "INSTRUMENT_UNRESOLVED_NOOP_HEALTH"
    )
    candidate = derivation.get("candidate")
    nested_families: dict[int, dict[str, Any]] = {}
    nested_decision: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    nested_basis_instrument_failure: dict[str, Any] | None = None
    if (
        not instrument_failure
        and isinstance(candidate, Mapping)
        and candidate.get("selected_rank") in RANK_GRID
    ):
        try:
            rank = int(candidate["selected_rank"])
            crossfit = derivation.get("crossfit_derivation")
            whole = (
                crossfit.get("whole_bucket_fit")
                if isinstance(crossfit, Mapping)
                else None
            )
            if not isinstance(whole, Mapping):
                raise InstrumentError(
                    "bucket-1 scoring lacks the frozen whole-bucket-0 fit"
                )
            basis = np.asarray(whole.get("basis"), dtype=np.float64)[:rank]
            random_bases = np.stack(
                [
                    deterministic_nested_random_bases(
                        basis.shape[1],
                        (rank,),
                        seed=int(seed),
                        namespace="r5b-v2-validation-rank-energy-null",
                        draw=draw,
                    )[rank]
                    for draw in range(N_RANDOM_CONTROLS)
                ],
                axis=0,
            )
            projections = energy_matched_rank_random_projections(
                captured["partner_delta"], basis=basis, random_bases=random_bases
            )
            pseudo_projection = np.asarray(
                captured["pseudo_delta"], dtype=np.float64
            ) @ (basis.T @ basis)
            projection_audit = dict(projections["audit"]) | {
                "pseudo_projection_sha256": array_sha256(pseudo_projection)
            }
            family = score_nested_local_writer_family(
                adapter,
                task,
                rows,
                captured,
                target_projection=projections["target_projection"],
                pseudo_projection=pseudo_projection,
                random_projections=projections["random_projections"],
                projection_audit=projection_audit,
                rank=rank,
                batch_size=batch_size,
            )
            nested_decision = classify_nested_local_writer_family(
                family,
                genes,
                n_draws=n_draws,
                seed=seed,
                phase="validation",
            )
            nested_families[rank] = family
            validation = validate_frozen_candidate_on_bucket1(
                candidate,
                whole,
                captured["partner_delta"],
                evaluated_ranks=[rank],
                nested_outcome_pass=bool(nested_decision["nested_path_pass"]),
                random_controls_pass=bool(
                    nested_decision["all_rank_energy_random_controls_pass"]
                ),
                n_overlap_draws=n_overlap_draws,
                seed=seed,
            )
        except OptionalBasisInstrumentError as error:
            instrument_failure = True
            nested_basis_instrument_failure = _optional_basis_failure_record(
                "validation", error
            )
    provider = _internal_scored_provider(
        rows,
        captured,
        phase="validation",
        dense_family=dense_family,
        nested_families=nested_families,
    )
    status = (
        "BASIS_INSTRUMENT_UNRESOLVED"
        if instrument_failure
        else validation["status"]
        if validation is not None
        else "NESTED_CANDIDATE_NOT_VALIDATED"
    )
    return _json_native(
        {
            "schema": R5B_VALIDATION_SCHEMA,
            "stage": "validation",
            "status": status,
            "writer_site_id": WRITER_SITE_ID,
            "residual_validation_bucket": VALIDATION_BUCKET,
            "genes": genes,
            "row_ids": row_ids,
            "direct_model_scoring": True,
            "external_provider_artifact_accepted": False,
            "post_wo_writer_audit": {
                key: value
                for key, value in replay_output_projected_writer_provider(
                    captured["provider_record"]
                ).items()
                if key not in {"writer_delta", "provider_record"}
            },
            "readout_axis_sha256": array_sha256(captured["readout_axis"]),
            "provider_record_sha256": canonical_sha256(
                replay_output_projected_writer_provider(
                    captured["provider_record"]
                )["provider_record"]
            ),
            "scored_provider": provider,
            "inference_draws": int(n_draws),
            "overlap_bootstrap_draws": int(n_overlap_draws),
            "dense_decision": dense_decision,
            "nested_validation_decision": nested_decision,
            "candidate_validation": validation,
            "instrument_failure": instrument_failure,
            "nested_basis_instrument_failure": nested_basis_instrument_failure,
            "derivation_sha256": canonical_sha256(derivation),
            "claim_eligible": False,
            "residual_validation_only": True,
            "sealed_chr8_chr9_opened": False,
            "masked_donor_endpoint_constructed": False,
        }
    )


def _json_native(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            default=lambda item: (
                item.tolist()
                if isinstance(item, np.ndarray)
                else item.item()
                if isinstance(item, np.generic)
                else str(item)
                if isinstance(item, Path)
                else (_ for _ in ()).throw(
                    TypeError(f"unsupported JSON value: {type(item)!r}")
                )
            ),
        )
    )


def _disposition_for_replays(
    derivation: Mapping[str, Any], validation: Mapping[str, Any]
) -> str:
    if derivation.get("instrument_failure") is True or validation.get(
        "instrument_failure"
    ) is True:
        return "BASIS_INSTRUMENT_UNRESOLVED"
    candidate_validation = validation.get("candidate_validation")
    if isinstance(candidate_validation, Mapping) and candidate_validation.get(
        "status"
    ) == "NESTED_CANDIDATE_VALIDATED_FOR_REFIT":
        return "VALIDATED_BASIS"
    return "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS"


def _replay_dense_family_from_primitives(
    family: Mapping[str, Any],
) -> dict[str, Any]:
    primitive = family.get("primitive_records")
    if not isinstance(primitive, Mapping) or set(primitive) != {
        "partner",
        "pseudo",
        "adjacent_by_offset",
        "gene_generic",
        "readout_axis",
        "health",
    }:
        raise InstrumentError("dense family lacks its complete primitive records")
    adjacent = primitive["adjacent_by_offset"]
    if not isinstance(adjacent, Mapping) or set(adjacent) != {
        str(offset) for offset in ADJACENT_TOKEN_OFFSETS
    }:
        raise InstrumentError("dense adjacent primitive registry differs")
    return build_dense_local_writer_family(
        partner=primitive["partner"],
        pseudo=primitive["pseudo"],
        adjacent_by_offset={
            offset: adjacent[str(offset)] for offset in ADJACENT_TOKEN_OFFSETS
        },
        gene_generic=primitive["gene_generic"],
        readout_axis=primitive["readout_axis"],
        health=primitive["health"],
    )


def _replay_nested_family_from_primitives(
    family: Mapping[str, Any],
) -> dict[str, Any]:
    primitive = family.get("primitive_records")
    projection_audit = family.get("projection_audit")
    rank = family.get("rank")
    if not isinstance(primitive, Mapping) or set(primitive) != {
        "partner",
        "pseudo",
        "adjacent_by_offset",
        "gene_generic",
        "readout_axis",
        "random_by_draw",
    }:
        raise InstrumentError("nested family lacks its complete primitive records")
    adjacent = primitive["adjacent_by_offset"]
    random = primitive["random_by_draw"]
    if (
        not isinstance(adjacent, Mapping)
        or set(adjacent) != {str(offset) for offset in ADJACENT_TOKEN_OFFSETS}
        or not isinstance(random, Mapping)
        or set(random) != {str(draw) for draw in range(N_RANDOM_CONTROLS)}
        or not isinstance(projection_audit, Mapping)
        or rank not in RANK_GRID
    ):
        raise InstrumentError("nested primitive registry differs")
    return build_nested_local_writer_family(
        partner=primitive["partner"],
        pseudo=primitive["pseudo"],
        adjacent_by_offset={
            offset: adjacent[str(offset)] for offset in ADJACENT_TOKEN_OFFSETS
        },
        gene_generic=primitive["gene_generic"],
        readout_axis=primitive["readout_axis"],
        random_by_draw={
            draw: random[str(draw)] for draw in range(N_RANDOM_CONTROLS)
        },
        rank=int(rank),
        projection_audit=projection_audit,
    )


def _validate_direct_scored_provider_header(
    record: Mapping[str, Any], provider: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    """Bind a serialized development result to the in-process checkpoint scorer."""

    schema = record.get("schema")
    if schema == R5B_DERIVATION_SCHEMA:
        phase = "derivation"
        bucket = DERIVATION_BUCKET
    elif schema == R5B_VALIDATION_SCHEMA:
        phase = "validation"
        bucket = VALIDATION_BUCKET
    else:
        raise InstrumentError("development record schema differs")
    genes = tuple(str(value) for value in record.get("genes", ()))
    row_ids = tuple(str(value) for value in record.get("row_ids", ()))
    if (
        record.get("stage") != phase
        or record.get("residual_validation_bucket") != bucket
        or record.get("writer_site_id") != WRITER_SITE_ID
        or record.get("direct_model_scoring") is not True
        or record.get("external_provider_artifact_accepted") is not False
        or record.get("claim_eligible") is not False
        or record.get("residual_validation_only") is not True
        or record.get("sealed_chr8_chr9_opened") is not False
        or record.get("masked_donor_endpoint_constructed") is not False
        or len(genes) != PRODUCTION_N
        or len(set(genes)) != PRODUCTION_N
        or len(row_ids) != PRODUCTION_N
        or len(set(row_ids)) != PRODUCTION_N
    ):
        raise InstrumentError("development scored-provider header differs")
    if (
        provider.get("schema") != R5B_SCORED_PROVIDER_SCHEMA
        or provider.get("origin")
        != "direct_checkpoint_scoring_in_this_r5b_process"
        or provider.get("external_provider_artifact_accepted") is not False
        or provider.get("phase") != phase
        or provider.get("residual_validation_bucket") != bucket
        or provider.get("chromosomes") != list(R5_VALIDATION_CHROMOSOMES)
        or provider.get("sealed_chr8_chr9_opened") is not False
        or provider.get("masked_donor_endpoint_constructed") is not False
        or provider.get("writer_site_id") != WRITER_SITE_ID
        or tuple(str(value) for value in provider.get("genes", ())) != genes
        or tuple(str(value) for value in provider.get("row_ids", ())) != row_ids
    ):
        raise InstrumentError("development scored-provider header differs")
    writer_record = provider.get("output_projected_writer")
    if not isinstance(writer_record, Mapping):
        raise InstrumentError("development scored-provider lacks post-W_O arrays")
    writer = replay_output_projected_writer_provider(writer_record)
    readout = np.asarray(provider.get("readout_axis"), dtype=np.float64)
    expected_audit = {
        key: value
        for key, value in writer.items()
        if key not in {"writer_delta", "provider_record"}
    }
    if (
        writer["n_rows"] != PRODUCTION_N
        or readout.shape != (writer["residual_width"],)
        or not np.isfinite(readout).all()
        or not np.isclose(np.linalg.norm(readout), 1.0, rtol=0.0, atol=1e-8)
        or record.get("readout_axis_sha256") != array_sha256(readout)
        or record.get("provider_record_sha256")
        != canonical_sha256(writer["provider_record"])
        or canonical_sha256(record.get("post_wo_writer_audit"))
        != canonical_sha256(expected_audit)
    ):
        raise InstrumentError("development scored-provider authentication differs")
    return genes, row_ids, writer


def _assert_development_record_replays(
    record: Mapping[str, Any],
    *,
    derivation: Mapping[str, Any] | None = None,
    require_production: bool = False,
) -> dict[str, Any]:
    """Recompute every decision from raw model-scored four-cell sufficient statistics."""

    provider = record.get("scored_provider")
    draws = record.get("inference_draws")
    if not isinstance(provider, Mapping):
        raise InstrumentError("development record is not from direct in-process scoring")
    if not isinstance(draws, int) or draws < 199:
        raise InstrumentError("development inference draw count differs")
    if require_production and draws != PRODUCTION_INFERENCE_DRAWS:
        raise InstrumentError("development production inference draw count differs")
    genes, row_ids, writer = _validate_direct_scored_provider_header(record, provider)
    raw_dense = provider.get("dense_family")
    if not isinstance(raw_dense, Mapping):
        raise InstrumentError("development record lacks dense primitive cells")
    dense = _replay_dense_family_from_primitives(raw_dense)
    if canonical_sha256(dense) != canonical_sha256(raw_dense):
        raise InstrumentError("stored dense family does not replay from primitive cells")
    dense_decision = classify_dense_local_writer(
        dense, genes, n_draws=draws, seed=SEED
    )
    instrument_failure = dense_decision["status"] == (
        "INSTRUMENT_UNRESOLVED_NOOP_HEALTH"
    )
    if record.get("schema") == R5B_DERIVATION_SCHEMA:
        expected_crossfit = None
        expected_rank_decisions: dict[int, dict[str, Any]] = {}
        expected_outcomes = None
        expected_candidate = None
        expected_nested_basis_failure = None
        if (
            not instrument_failure
            and dense_decision["status"]
            == "DENSE_LOCAL_WRITER_DEVELOPMENT_SUPPORTED"
        ):
            raw_nested = provider.get("nested_rank_families")
            if not isinstance(raw_nested, Mapping):
                raise InstrumentError("derivation nested rank registry is malformed")
            try:
                expected_crossfit = fit_crossfit_derivation_bases(
                    writer["writer_delta"], genes, maximum_rank=max(RANK_GRID)
                )
                expected_outcomes = {}
                for rank in RANK_GRID:
                    projection = crossfit_projection_package(
                        writer["writer_delta"], expected_crossfit, rank=rank, seed=SEED
                    )
                    stored_family = raw_nested.get(str(rank))
                    if not isinstance(stored_family, Mapping):
                        raise InstrumentError("nested rank family is malformed")
                    family = _replay_nested_family_from_primitives(stored_family)
                    if canonical_sha256(family) != canonical_sha256(stored_family):
                        raise InstrumentError(
                            f"stored nested rank {rank} does not replay from primitive cells"
                        )
                    if "pseudo_delta" not in writer:
                        raise InstrumentError(
                            "derivation lacks post-W_O pseudo-control rows"
                        )
                    pseudo_projection = _project_rows_with_bases(
                        np.asarray(writer["pseudo_delta"], dtype=np.float64),
                        np.asarray(projection["row_bases"], dtype=np.float64),
                    )
                    expected_projection_audit = dict(projection["audit"]) | {
                        "pseudo_projection_sha256": array_sha256(pseudo_projection)
                    }
                    if canonical_sha256(
                        expected_projection_audit
                    ) != canonical_sha256(family["projection_audit"]):
                        raise InstrumentError(
                            "cross-fit projection audit does not replay"
                        )
                    decision = classify_nested_local_writer_family(
                        family,
                        genes,
                        n_draws=draws,
                        seed=SEED,
                        phase="derivation",
                    )
                    expected_rank_decisions[rank] = decision
                    points, _se = _statistics_and_standard_errors(
                        np.asarray(
                            family["endpoints"]["partner|S"], dtype=np.float64
                        )[None, :]
                    )
                    expected_outcomes[rank] = {
                        "mean_partner_s": float(points["mean"][0]),
                        "trimmed_partner_s": float(
                            points["trimmed_mean_10pct"][0]
                        ),
                        "nested_path_pass": bool(decision["nested_path_pass"]),
                        "crossfold_stability_pass": bool(
                            projection["crossfold_stability_pass"]
                        ),
                        "all_rank_energy_random_controls_pass": bool(
                            decision["all_rank_energy_random_controls_pass"]
                        ),
                    }
                expected_candidate = select_derivation_candidate_rank(
                    expected_crossfit["whole_bucket_fit"],
                    expected_outcomes,
                    dense_development_status=dense_decision["status"],
                )
            except OptionalBasisInstrumentError as error:
                instrument_failure = True
                expected_nested_basis_failure = _optional_basis_failure_record(
                    "derivation", error
                )
            expected_rank_decisions = {
                str(rank): decision
                for rank, decision in expected_rank_decisions.items()
            }
            if expected_outcomes is not None:
                expected_outcomes = {
                    str(rank): outcome for rank, outcome in expected_outcomes.items()
                }
            if set(raw_nested) != set(expected_rank_decisions):
                raise InstrumentError(
                    "derivation nested families differ from the reached rank registry"
                )
        else:
            raw_nested = provider.get("nested_rank_families")
            if not isinstance(raw_nested, Mapping) or raw_nested:
                raise InstrumentError(
                    "unreached derivation optional path must carry no nested families"
                )
        expected_status = (
            "BASIS_INSTRUMENT_UNRESOLVED"
            if instrument_failure
            else "DERIVATION_CANDIDATE_SELECTED"
            if expected_candidate is not None
            and expected_candidate.get("selected_rank") is not None
            else "NO_DERIVATION_CANDIDATE_AT_FROZEN_THRESHOLDS"
        )
        expected = {
            "status": expected_status,
            "dense_decision": dense_decision,
            "crossfit_derivation": expected_crossfit,
            "nested_rank_decisions": expected_rank_decisions,
            "rank_outcomes": expected_outcomes,
            "candidate": expected_candidate,
            "instrument_failure": instrument_failure,
            "nested_basis_instrument_failure": expected_nested_basis_failure,
        }
    elif record.get("schema") == R5B_VALIDATION_SCHEMA:
        if not isinstance(derivation, Mapping):
            raise InstrumentError("validation replay requires its exact derivation")
        _assert_development_record_replays(
            derivation, require_production=require_production
        )
        if set(row_ids) & set(str(value) for value in derivation.get("row_ids", ())):
            raise InstrumentError("residual derivation and validation rows overlap")
        overlap_draws = record.get("overlap_bootstrap_draws")
        if not isinstance(overlap_draws, int) or overlap_draws < 199:
            raise InstrumentError("validation overlap draw count differs")
        if require_production and overlap_draws != PRODUCTION_OVERLAP_DRAWS:
            raise InstrumentError("validation production overlap draw count differs")
        instrument_failure = bool(
            derivation.get("instrument_failure") is True or instrument_failure
        )
        candidate = derivation.get("candidate")
        nested_decision = None
        validation = None
        expected_nested_basis_failure = None
        if (
            not instrument_failure
            and isinstance(candidate, Mapping)
            and candidate.get("selected_rank") in RANK_GRID
        ):
            rank = int(candidate["selected_rank"])
            raw_nested = provider.get("nested_rank_families")
            if not isinstance(raw_nested, Mapping):
                raise InstrumentError("validation nested rank registry is malformed")
            try:
                crossfit = derivation.get("crossfit_derivation")
                whole = (
                    crossfit.get("whole_bucket_fit")
                    if isinstance(crossfit, Mapping)
                    else None
                )
                if not isinstance(whole, Mapping):
                    raise InstrumentError("validation replay lacks bucket-0 basis")
                basis = np.asarray(whole.get("basis"), dtype=np.float64)[:rank]
                random_bases = np.stack(
                    [
                        deterministic_nested_random_bases(
                            basis.shape[1],
                            (rank,),
                            seed=SEED,
                            namespace="r5b-v2-validation-rank-energy-null",
                            draw=draw,
                        )[rank]
                        for draw in range(N_RANDOM_CONTROLS)
                    ],
                    axis=0,
                )
                projection = energy_matched_rank_random_projections(
                    writer["writer_delta"], basis=basis, random_bases=random_bases
                )
                stored_family = raw_nested.get(str(rank))
                if not isinstance(stored_family, Mapping):
                    raise InstrumentError("validation nested rank family is malformed")
                family = _replay_nested_family_from_primitives(stored_family)
                if canonical_sha256(family) != canonical_sha256(stored_family):
                    raise InstrumentError("stored validation family does not replay")
                nested_decision = classify_nested_local_writer_family(
                    family,
                    genes,
                    n_draws=draws,
                    seed=SEED,
                    phase="validation",
                )
                if "pseudo_delta" not in writer:
                    raise InstrumentError(
                        "validation lacks post-W_O pseudo-control rows"
                    )
                pseudo_projection = np.asarray(
                    writer["pseudo_delta"], dtype=np.float64
                ) @ (basis.T @ basis)
                expected_projection_audit = dict(projection["audit"]) | {
                    "pseudo_projection_sha256": array_sha256(pseudo_projection)
                }
                if canonical_sha256(
                    expected_projection_audit
                ) != canonical_sha256(family["projection_audit"]):
                    raise InstrumentError(
                        "bucket-1 projection audit does not replay"
                    )
                validation = validate_frozen_candidate_on_bucket1(
                    candidate,
                    whole,
                    writer["writer_delta"],
                    evaluated_ranks=[rank],
                    nested_outcome_pass=bool(nested_decision["nested_path_pass"]),
                    random_controls_pass=bool(
                        nested_decision["all_rank_energy_random_controls_pass"]
                    ),
                    n_overlap_draws=overlap_draws,
                    seed=SEED,
                )
            except OptionalBasisInstrumentError as error:
                instrument_failure = True
                expected_nested_basis_failure = _optional_basis_failure_record(
                    "validation", error
                )
            expected_nested_keys = {str(rank)} if nested_decision is not None else set()
            if set(raw_nested) != expected_nested_keys:
                raise InstrumentError(
                    "validation nested families differ from the reached rank registry"
                )
        else:
            raw_nested = provider.get("nested_rank_families")
            if not isinstance(raw_nested, Mapping) or raw_nested:
                raise InstrumentError(
                    "unreached validation optional path must carry no nested families"
                )
        expected_status = (
            "BASIS_INSTRUMENT_UNRESOLVED"
            if instrument_failure
            else validation["status"]
            if validation is not None
            else "NESTED_CANDIDATE_NOT_VALIDATED"
        )
        expected = {
            "status": expected_status,
            "dense_decision": dense_decision,
            "nested_validation_decision": nested_decision,
            "candidate_validation": validation,
            "instrument_failure": instrument_failure,
            "nested_basis_instrument_failure": expected_nested_basis_failure,
            "derivation_sha256": canonical_sha256(derivation),
        }
    else:
        raise InstrumentError("development record schema differs")
    mismatched = [
        key
        for key, expected_value in expected.items()
        if canonical_sha256(record.get(key)) != canonical_sha256(expected_value)
    ]
    if mismatched:
        raise InstrumentError(
            "development record does not replay: " + ", ".join(sorted(mismatched))
        )
    return expected


def _normalized_publication_pair(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"artifact", "manifest"}:
        raise ValueError("development package must bind artifact and manifest")
    normalized: dict[str, Any] = {}
    for kind in ("artifact", "manifest"):
        identity = value[kind]
        if not isinstance(identity, Mapping) or set(identity) != {
            "logical_name",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("publication identity fields differ")
        logical_name = identity.get("logical_name")
        size_bytes = identity.get("size_bytes")
        if (
            not isinstance(logical_name, str)
            or not logical_name
            or Path(logical_name).name != logical_name
            or not isinstance(size_bytes, int)
            or size_bytes < 1
        ):
            raise ValueError("publication identity is not relocation-safe")
        normalized[kind] = {
            "logical_name": logical_name,
            "size_bytes": size_bytes,
            "sha256": _validate_sha256_literal(
                f"{kind} publication", identity.get("sha256")
            ),
        }
    return normalized


def _post_wo_contract_from_replays(
    bucket0: Mapping[str, Any], bucket1: Mapping[str, Any]
) -> dict[str, Any]:
    provider0 = bucket0.get("scored_provider", {}).get("output_projected_writer")
    provider1 = bucket1.get("scored_provider", {}).get("output_projected_writer")
    if not isinstance(provider0, Mapping) or not isinstance(provider1, Mapping):
        raise InstrumentError("development packages lack post-W_O writer providers")
    writer0 = replay_output_projected_writer_provider(provider0)
    writer1 = replay_output_projected_writer_provider(provider1)
    if (
        writer0["residual_width"] != writer1["residual_width"]
        or writer0["head_width"] != writer1["head_width"]
        or writer0["w_o_column_block_sha256"]
        != writer1["w_o_column_block_sha256"]
        or writer0["numeric_provenance"] != writer1["numeric_provenance"]
    ):
        raise InstrumentError("residual buckets do not share one frozen W_O block")
    return {
        "writer_site_id": WRITER_SITE_ID,
        "representation": "post_W_O_L3H1_additive_contribution_at_acceptor_query",
        "projection_equation": "context_L3H1_q @ W_O[:,L3H1_columns].T",
        "projection_helper": "torch.nn.functional.linear@checkpoint_operator_dtype",
        "capture_helper": "gfmcd.circuits.hf_bert.project_head_context_to_residual",
        "bias_included": False,
        "numeric_provenance": writer0["numeric_provenance"],
        "head_width": writer0["head_width"],
        "residual_width": writer0["residual_width"],
        "w_o_column_block_sha256": writer0["w_o_column_block_sha256"],
        "bucket0_clean_head_context_sha256": writer0[
            "clean_head_context_sha256"
        ],
        "bucket0_corrupt_head_context_sha256": writer0[
            "corrupt_head_context_sha256"
        ],
        "bucket0_clean_writer_contribution_sha256": writer0[
            "clean_writer_contribution_sha256"
        ],
        "bucket0_corrupt_writer_contribution_sha256": writer0[
            "corrupt_writer_contribution_sha256"
        ],
        "bucket0_writer_delta_sha256": writer0["writer_delta_sha256"],
        "bucket0_pseudo_writer_contribution_sha256": writer0[
            "pseudo_writer_contribution_sha256"
        ],
        "bucket0_pseudo_delta_sha256": writer0["pseudo_delta_sha256"],
        "bucket1_clean_head_context_sha256": writer1[
            "clean_head_context_sha256"
        ],
        "bucket1_corrupt_head_context_sha256": writer1[
            "corrupt_head_context_sha256"
        ],
        "bucket1_clean_writer_contribution_sha256": writer1[
            "clean_writer_contribution_sha256"
        ],
        "bucket1_corrupt_writer_contribution_sha256": writer1[
            "corrupt_writer_contribution_sha256"
        ],
        "bucket1_writer_delta_sha256": writer1["writer_delta_sha256"],
        "bucket1_pseudo_writer_contribution_sha256": writer1[
            "pseudo_writer_contribution_sha256"
        ],
        "bucket1_pseudo_delta_sha256": writer1["pseudo_delta_sha256"],
    }


def build_local_writer_basis_disposition(
    derivation: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    development_packages: Mapping[str, Mapping[str, Any]],
    r3d_path_a_binding: Mapping[str, str],
    r5a_method_preflight_binding: Mapping[str, str],
    checkpoint_sha256: str,
    cohort_audit_sha256: str,
    r5_stack_fingerprint_sha256: str,
    bundle_path: str | Path | None,
    seed: int = SEED,
) -> tuple[dict[str, Any], bytes | None]:
    """Deterministically emit one of the three frozen R5b terminal dispositions."""

    replayed_derivation = _assert_development_record_replays(
        derivation, require_production=True
    )
    replayed_validation = _assert_development_record_replays(
        validation, derivation=derivation, require_production=True
    )
    if validation.get("derivation_sha256") != canonical_sha256(derivation):
        raise InstrumentError("validation is not bound to this exact derivation")
    if set(development_packages) != {"derivation", "validation"}:
        raise ValueError("both development package identities are required")
    packages = {
        name: _normalized_publication_pair(development_packages[name])
        for name in ("derivation", "validation")
    }
    for label, binding in (
        ("R3d Path A", r3d_path_a_binding),
        ("R5a method preflight", r5a_method_preflight_binding),
    ):
        if set(binding) != {"artifact_sha256", "manifest_sha256"}:
            raise ValueError(f"{label} binding fields differ")
        for kind, digest in binding.items():
            _validate_sha256_literal(f"{label} {kind}", digest)
    post_wo = _post_wo_contract_from_replays(derivation, validation)
    provider0 = derivation["scored_provider"]
    provider1 = validation["scored_provider"]
    if array_sha256(provider0["readout_axis"]) != array_sha256(
        provider1["readout_axis"]
    ):
        raise InstrumentError("residual buckets do not share one frozen readout axis")
    status = _disposition_for_replays(derivation, validation)
    sufficient_replay = {
        "instrument_failure": status == "BASIS_INSTRUMENT_UNRESOLVED",
        "derivation_status": replayed_derivation["status"],
        "validation_status": (
            replayed_validation.get("candidate_validation") or {}
        ).get("status"),
        "bucket0_dense_status": replayed_derivation["dense_decision"]["status"],
        "bucket1_dense_status": replayed_validation["dense_decision"]["status"],
        "derivation_record_sha256": canonical_sha256(derivation),
        "validation_record_sha256": canonical_sha256(validation),
        "bucket0_ordered_rows_sha256": canonical_sha256(derivation["row_ids"]),
        "bucket1_ordered_rows_sha256": canonical_sha256(validation["row_ids"]),
        "bucket0_provider_sha256": canonical_sha256(provider0),
        "bucket1_provider_sha256": canonical_sha256(provider1),
        "nested_basis_instrument_failures": {
            "derivation": replayed_derivation.get(
                "nested_basis_instrument_failure"
            ),
            "validation": replayed_validation.get(
                "nested_basis_instrument_failure"
            ),
        },
    }
    contract: dict[str, Any] | None = None
    bundle_identity: dict[str, Any] | None = None
    bundle_bytes: bytes | None = None
    frozen_record: dict[str, Any] | None = None
    if status == "VALIDATED_BASIS":
        if bundle_path is None:
            raise ValueError("validated disposition requires a prospective NPZ path")
        writer0 = replay_output_projected_writer_provider(
            provider0["output_projected_writer"]
        )
        writer1 = replay_output_projected_writer_provider(
            provider1["output_projected_writer"]
        )
        candidate = derivation.get("candidate")
        candidate_validation = validation.get("candidate_validation")
        if not isinstance(candidate, Mapping) or not isinstance(
            candidate_validation, Mapping
        ):
            raise InstrumentError("validated disposition lacks candidate replay")
        frozen = mechanical_union_refit(
            writer0["writer_delta"],
            writer1["writer_delta"],
            row_ids_bucket0=derivation["row_ids"],
            row_ids_bucket1=validation["row_ids"],
            candidate=candidate,
            validation=candidate_validation,
            readout_axis=provider0["readout_axis"],
            seed=seed,
        )
        arrays = frozen_basis_runtime_arrays(frozen)
        bundle_bytes = npz_bytes(arrays)
        bundle_name = Path(bundle_path).expanduser().name
        if not bundle_name:
            raise ValueError("basis bundle needs one relocation-safe logical name")
        bundle_identity = {
            "logical_name": bundle_name,
            "size_bytes": len(bundle_bytes),
            "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        }
        artifact_bindings = {
            "r5a_entry": dict(r5a_method_preflight_binding),
            "r5b_derivation": {
                "artifact_sha256": packages["derivation"]["artifact"]["sha256"],
                "manifest_sha256": packages["derivation"]["manifest"]["sha256"],
            },
            "r5b_validation": {
                "artifact_sha256": packages["validation"]["artifact"]["sha256"],
                "manifest_sha256": packages["validation"]["manifest"]["sha256"],
            },
        }
        contract = frozen_basis_runtime_contract(
            frozen,
            bundle_path=bundle_name,
            bundle_bytes=bundle_bytes,
            checkpoint_sha256=checkpoint_sha256,
            cohort_audit_sha256=cohort_audit_sha256,
            r5_stack_fingerprint_sha256=r5_stack_fingerprint_sha256,
            artifact_bindings=artifact_bindings,
            post_wo_writer_contract=post_wo,
        )
        frozen_record = _json_native(frozen)
        path_local_c2 = R5B_PATH_LOCAL_C2_BY_STATUS[status]
    else:
        if bundle_path is not None:
            raise ValueError("nonvalidated disposition cannot name a basis bundle")
        path_local_c2 = R5B_PATH_LOCAL_C2_BY_STATUS[status]
    record = {
        "schema": R5B_DISPOSITION_SCHEMA,
        "status": status,
        "basis_disposition": status,
        "writer_site_id": WRITER_SITE_ID,
        "claim_eligible": False,
        "residual_validation_only": True,
        "sealed_chr8_chr9_opened": False,
        "masked_donor_endpoint_constructed": False,
        "dense_primary_unaffected": True,
        "optional_c2_only": True,
        "path_local_c2_disposition": path_local_c2,
        "post_wo_writer_contract": post_wo,
        "sufficient_replay": sufficient_replay,
        "development_packages": packages,
        "scientific_prior": {
            "role": "R3d_Path_A_L3H1_reuse_only",
            "path_b_used_to_gate": False,
            "package": dict(r3d_path_a_binding),
        },
        "r5a_method_preflight": {
            "target_result_consumed": False,
            "package": dict(r5a_method_preflight_binding),
        },
        "checkpoint_sha256": _validate_sha256_literal(
            "checkpoint", checkpoint_sha256
        ),
        "cohort_audit_sha256": _validate_sha256_literal(
            "cohort audit", cohort_audit_sha256
        ),
        "r5_stack_fingerprint_sha256": _validate_sha256_literal(
            "R5 stack", r5_stack_fingerprint_sha256
        ),
        "frozen_basis_runtime_contract": contract,
        "basis_bundle": bundle_identity,
        "frozen_basis_sufficient_replay": frozen_record,
    }
    return _json_native(record), bundle_bytes


def atomic_write_new(path: str | Path, data: bytes) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing output: {destination}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(_json_native(value), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _manifest_input(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def publish_residual_development_stage(
    record: Mapping[str, Any],
    *,
    derivation: Mapping[str, Any] | None = None,
    receipt: Mapping[str, Any],
    output_path: str | Path,
    command: Sequence[str],
    input_paths: Sequence[str | Path],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Create artifact+manifest, then irreversibly complete the bucket receipt."""

    _assert_development_record_replays(
        record,
        derivation=derivation,
        require_production=True,
    )
    output = Path(output_path).expanduser().resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("R5b residual outputs are immutable; choose a fresh path")
    attempt = residual_attempt_publication_identity(receipt)
    context = receipt.get("method_context")
    if not isinstance(context, Mapping):
        raise InstrumentError("R5b receipt lacks its method context")
    if (
        output.name != context.get("requested_artifact_logical_name")
        or manifest_path.name != context.get("requested_manifest_logical_name")
    ):
        raise InstrumentError("R5b output names differ from the pre-inspection receipt")
    enriched = dict(record) | {
        "residual_attempt": attempt,
        "residual_completion": {
            "logical_name": context["declared_completion_logical_name"]
        },
    }
    result_bytes = _json_bytes(enriched)
    result_identity = {
        "path": str(output),
        "size_bytes": len(result_bytes),
        "sha256": hashlib.sha256(result_bytes).hexdigest(),
    }
    manifest = ExperimentManifest.create(
        f"splicebert_acceptor_coupling:r5b_{record['stage']}",
        command,
        repo=REPO,
        status=str(record["status"]),
        config={
            "stage": record["stage"],
            "residual_validation_bucket": record["residual_validation_bucket"],
            "n": PRODUCTION_N,
            "batch": PRODUCTION_BATCH,
            "inference_draws": record["inference_draws"],
            "overlap_draws": record.get("overlap_bootstrap_draws"),
            "writer_site_id": WRITER_SITE_ID,
            "instrument_fingerprint_sha256": authorization[
                "instrument_fingerprint_sha256"
            ],
            "r5_stack_fingerprint_sha256": authorization[
                "r5_stack_fingerprint_sha256"
            ],
            "direct_model_scoring": True,
            "external_provider_artifact_accepted": False,
        },
        seeds={"bootstrap": SEED, "basis": SEED},
        notes=[
            "Residual-validation instrument development only; no chr8/chr9 opened.",
            "All four-cell primitives were scored directly in this R5b process.",
            "Dense post-W_O L3H1 writer evidence is primary and basis-independent.",
        ],
    )
    if manifest.git_dirty is not False:
        raise InstrumentError("R5b scored publication requires a clean Git worktree")
    manifest.inputs = [_manifest_input(path) for path in input_paths]
    manifest.inputs.append(_manifest_input(receipt["receipt_path"]))
    manifest.outputs = [result_identity]
    manifest_bytes = _json_bytes(manifest.to_dict())
    atomic_write_new(output, result_bytes)
    atomic_write_new(manifest_path, manifest_bytes)
    completion = complete_r5b_residual_attempt(
        receipt=receipt,
        artifact_path=output,
        manifest_path=manifest_path,
        official_status=str(record["status"]),
    )
    return {
        "artifact": _publication_identity(output),
        "manifest": _publication_identity(manifest_path),
        "completion": _publication_identity(completion["completion_path"]),
        "completion_record": completion,
        "record": enriched,
    }


def publish_basis_disposition(
    record: Mapping[str, Any],
    bundle_bytes: bytes | None,
    *,
    output_path: str | Path,
    bundle_path: str | Path | None,
    command: Sequence[str],
    input_paths: Sequence[str | Path],
    derivation_paths: tuple[str | Path, str | Path],
    validation_paths: tuple[str | Path, str | Path],
) -> dict[str, Any]:
    """Create the terminal three-way disposition and optional runtime NPZ."""

    output = Path(output_path).expanduser().resolve()
    manifest_path = output.with_suffix(".manifest.json")
    bundle = None if bundle_path is None else Path(bundle_path).expanduser().resolve()
    targets = [output, manifest_path] + ([] if bundle is None else [bundle])
    if any(path.exists() for path in targets):
        raise FileExistsError("R5b disposition outputs are immutable; choose fresh paths")
    status = record.get("status")
    if (status == "VALIDATED_BASIS") != (bundle is not None and bundle_bytes is not None):
        raise InstrumentError("R5b disposition/bundle presence differs")
    result_bytes = _json_bytes(record)
    result_identity = {
        "path": str(output),
        "size_bytes": len(result_bytes),
        "sha256": hashlib.sha256(result_bytes).hexdigest(),
    }
    outputs = [result_identity]
    if bundle is not None and bundle_bytes is not None:
        outputs.append(
            {
                "path": str(bundle),
                "size_bytes": len(bundle_bytes),
                "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
            }
        )
    manifest = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r5b_basis_disposition",
        command,
        repo=REPO,
        status=str(status),
        config={
            "stage": "freeze",
            "writer_site_id": WRITER_SITE_ID,
            "basis_disposition": status,
            "claim_eligible": False,
            "sealed_chr8_chr9_opened": False,
        },
        seeds={"union_refit": SEED},
        notes=[
            "Terminal pre-unseal R5b basis disposition; never mechanism evidence.",
            "A nonvalidated basis suppresses only optional R5c C2.",
        ],
    )
    if manifest.git_dirty is not False:
        raise InstrumentError("R5b disposition publication requires a clean Git worktree")
    manifest.inputs = [_manifest_input(path) for path in input_paths]
    manifest.outputs = outputs
    manifest_bytes = _json_bytes(manifest.to_dict())
    if bundle is not None and bundle_bytes is not None:
        atomic_write_new(bundle, bundle_bytes)
    atomic_write_new(output, result_bytes)
    atomic_write_new(manifest_path, manifest_bytes)
    verified = validate_local_writer_basis_disposition(
        output,
        manifest_path,
        bundle_path=bundle,
        derivation_artifact_path=derivation_paths[0],
        derivation_manifest_path=derivation_paths[1],
        validation_artifact_path=validation_paths[0],
        validation_manifest_path=validation_paths[1],
    )
    return verified | {"manifest_record": manifest.to_dict()}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("audit", "derivation", "validation", "freeze"),
        default="audit",
    )
    parser.add_argument("--out")
    parser.add_argument("--basis-bundle-out")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=PRODUCTION_BATCH)
    parser.add_argument("--draws", type=int, default=PRODUCTION_INFERENCE_DRAWS)
    parser.add_argument(
        "--overlap-draws", type=int, default=PRODUCTION_OVERLAP_DRAWS
    )
    parser.add_argument("--audit-draws", type=int, default=199)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--r3d-artifact")
    parser.add_argument("--r3d-manifest")
    parser.add_argument("--r3d-registry")
    parser.add_argument("--r3d-registry-manifest")
    parser.add_argument("--r3d-attempt")
    parser.add_argument("--r3d-completion")
    parser.add_argument("--r3c-artifact")
    parser.add_argument("--r3c-manifest")
    parser.add_argument("--r5a-preflight-artifact")
    parser.add_argument("--r5a-preflight-manifest")
    parser.add_argument("--derivation-artifact")
    parser.add_argument("--derivation-manifest")
    parser.add_argument("--derivation-completion")
    parser.add_argument("--validation-artifact")
    parser.add_argument("--validation-manifest")
    parser.add_argument("--validation-completion")
    return parser.parse_args(argv)


def _required_argument(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is None:
        raise SystemExit(f"R5b {args.stage} requires --{name.replace('_', '-')}")
    return value


def _stack_source_paths(authorization: Mapping[str, Any]) -> list[Path]:
    stack = authorization.get("r5_stack_binding")
    if not isinstance(stack, Mapping):
        raise InstrumentError("R5b authorization lacks the joint stack")
    closure = stack.get("local_python_closure", {}).get("files")
    protocols = stack.get("protocols")
    if not isinstance(closure, Mapping) or not isinstance(protocols, Mapping):
        raise InstrumentError("R5b joint stack lacks its source registry")
    paths: list[Path] = []
    for relative in (*closure, *protocols):
        path = (REPO / str(relative)).resolve(strict=True)
        path.relative_to(REPO.resolve())
        if path not in paths:
            paths.append(path)
    return paths


def _upstream_args(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "r3d_artifact",
        "r3d_manifest",
        "r3d_registry",
        "r3d_registry_manifest",
        "r3d_attempt",
        "r3d_completion",
        "r3c_artifact",
        "r3c_manifest",
        "r5a_preflight_artifact",
        "r5a_preflight_manifest",
    )
    values = {name: _required_argument(args, name) for name in names}
    return {
        "r3d_artifact": values["r3d_artifact"],
        "r3d_manifest": values["r3d_manifest"],
        "r3d_registry": values["r3d_registry"],
        "r3d_registry_manifest": values["r3d_registry_manifest"],
        "r3d_attempt": values["r3d_attempt"],
        "r3d_completion": values["r3d_completion"],
        "r3c_artifact": values["r3c_artifact"],
        "r3c_manifest": values["r3c_manifest"],
        "r5a_artifact": values["r5a_preflight_artifact"],
        "r5a_manifest": values["r5a_preflight_manifest"],
    }


def _reserve_stage_attempt(
    args: argparse.Namespace,
    authorization: Mapping[str, Any],
    upstream: Mapping[str, Any],
    *,
    predecessor: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if args.stage not in {"derivation", "validation"}:
        raise ValueError("only residual scoring stages reserve attempts")
    output_name = Path(_required_argument(args, "out")).expanduser().name
    if not output_name:
        raise SystemExit("R5b output requires one file name")
    manifest_name = Path(output_name).with_suffix(".manifest.json").name
    identity = build_r5b_residual_attempt_identity(
        stage=args.stage,
        checkpoint_sha256=str(FROZEN_CHECKPOINT_AGGREGATE_SHA256),
        cohort_input_sha256=str(FROZEN_COHORT_INPUT_SHA256),
        cohort_audit_sha256=str(FROZEN_COHORT_AUDIT_SHA256),
    )
    predecessor_identity = (
        None
        if predecessor is None
        else {
            kind: predecessor[kind]["sha256"]
            for kind in ("artifact", "manifest", "completion")
        }
    )
    return reserve_r5b_residual_attempt(
        repo=REPO,
        identity=identity,
        method_context={
            "protocol_sha256": str(FROZEN_PROTOCOL_SHA256),
            "instrument_fingerprint_sha256": authorization[
                "instrument_fingerprint_sha256"
            ],
            "r5_stack_fingerprint_sha256": authorization[
                "r5_stack_fingerprint_sha256"
            ],
            "r3d_artifact_sha256": upstream["r3d"]["artifact"]["sha256"],
            "r3d_manifest_sha256": upstream["r3d"]["manifest"]["sha256"],
            "r5a_preflight_artifact_sha256": upstream["r5a"]["artifact"][
                "sha256"
            ],
            "r5a_preflight_manifest_sha256": upstream["r5a"]["manifest"][
                "sha256"
            ],
            "predecessor_artifact_sha256": (
                None if predecessor_identity is None else predecessor_identity["artifact"]
            ),
            "predecessor_manifest_sha256": (
                None if predecessor_identity is None else predecessor_identity["manifest"]
            ),
            "predecessor_completion_sha256": (
                None
                if predecessor_identity is None
                else predecessor_identity["completion"]
            ),
            "requested_artifact_logical_name": output_name,
            "requested_manifest_logical_name": manifest_name,
            "declared_completion_logical_name": (
                f"{identity['residual_attempt_key']}.completion.json"
            ),
            "scoring_origin": "direct_checkpoint_scoring_in_r5b_process",
        },
    )


def _run_scored_stage(
    args: argparse.Namespace,
    authorization: Mapping[str, Any],
    upstream: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    derivation_package: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reservations, cohort_audit, source_identity = build_r5b_residual_reservations(
        args.gtf,
        args.fasta,
        exclusion_registry=upstream["exclusion_registry"],
    )
    bucket = DERIVATION_BUCKET if args.stage == "derivation" else VALIDATION_BUCKET
    rows = list(reservations[bucket])
    checkpoint = checkpoint_identity(args.model)
    if (
        checkpoint["aggregate_sha256"] != FROZEN_CHECKPOINT_AGGREGATE_SHA256
        or checkpoint["aggregate_sha256"] != PINNED_CHECKPOINT_AGGREGATE_SHA256
        or checkpoint["aggregate_sha256"]
        != upstream["r3d"]["checkpoint_and_data"]["checkpoint_aggregate_sha256"]
    ):
        raise InstrumentError("R5b checkpoint differs from its frozen upstream identity")
    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=True)
    adapter.model.eval()
    task = CouplingTask(adapter, batch=args.batch)
    with torch.no_grad():
        if args.stage == "derivation":
            record = score_r5b_derivation(
                adapter,
                task,
                rows,
                batch_size=args.batch,
                n_draws=args.draws,
                seed=SEED,
            )
        else:
            if derivation_package is None:
                raise InstrumentError("R5b validation lacks its completed derivation")
            record = score_r5b_validation(
                adapter,
                task,
                rows,
                derivation_package["record"],
                batch_size=args.batch,
                n_draws=args.draws,
                n_overlap_draws=args.overlap_draws,
                seed=SEED,
            )
    # Every file capable of changing the result is rehashed after model execution.
    source_after = validate_expected_r5_source_identity(args.gtf, args.fasta)
    checkpoint_after = checkpoint_identity(args.model)
    upstream_after = load_r5b_upstream_chain(
        **_upstream_args(args),
        expected_stack_fingerprint=authorization[
            "r5_stack_fingerprint_sha256"
        ],
    )
    if (
        source_after != source_identity
        or checkpoint_after != checkpoint
        or upstream_after["chain_sha256"] != upstream["chain_sha256"]
        or sha256_file(PROTOCOL) != FROZEN_PROTOCOL_SHA256
        or instrument_fingerprint() != authorization[
            "instrument_fingerprint_sha256"
        ]
    ):
        raise InstrumentError("R5b model/data inputs changed during scoring")
    record.update(
        {
            "instrument_fingerprint_sha256": authorization[
                "instrument_fingerprint_sha256"
            ],
            "r5_stack_fingerprint_sha256": authorization[
                "r5_stack_fingerprint_sha256"
            ],
            "cohort_audit": cohort_audit,
            "cohort_audit_sha256": canonical_sha256(cohort_audit),
            "source_identity": source_identity,
            "checkpoint": checkpoint,
            "upstream_chain_sha256": upstream["chain_sha256"],
            "upstream_packages": {
                "r3d": {
                    "artifact": upstream["r3d"]["artifact"],
                    "manifest": upstream["r3d"]["manifest"],
                },
                "r5a_method_preflight": {
                    "artifact": upstream["r5a"]["artifact"],
                    "manifest": upstream["r5a"]["manifest"],
                    "target_result_consumed": False,
                },
            },
            "inputs_rehashed_after_scoring": True,
        }
    )
    input_paths: list[str | Path] = [
        args.gtf,
        args.fasta,
        f"{Path(args.fasta).expanduser()}.fai",
        *_upstream_args(args).values(),
        *_stack_source_paths(authorization),
    ]
    checkpoint_root = Path(checkpoint["root"])
    input_paths.extend(
        checkpoint_root / item["relative_path"] for item in checkpoint["files"]
    )
    if derivation_package is not None:
        input_paths.extend(
            [
                args.derivation_artifact,
                args.derivation_manifest,
                args.derivation_completion,
            ]
        )
    return publish_residual_development_stage(
        record,
        derivation=(
            None if derivation_package is None else derivation_package["record"]
        ),
        receipt=receipt,
        output_path=args.out,
        command=list(sys.argv),
        input_paths=input_paths,
        authorization=authorization,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.stage == "audit":
        record = run_model_free_audit(n_draws=args.audit_draws)
        payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        if args.out is None:
            sys.stdout.buffer.write(payload)
        else:
            atomic_write_new(args.out, payload)
            print(f"R5b model-free audit -> {Path(args.out).expanduser().resolve()}")
        return
    if args.batch != PRODUCTION_BATCH:
        raise SystemExit(f"R5b production batch is fixed at {PRODUCTION_BATCH}")
    if args.draws != PRODUCTION_INFERENCE_DRAWS:
        raise SystemExit(
            f"R5b production inference draws are fixed at {PRODUCTION_INFERENCE_DRAWS}"
        )
    if args.overlap_draws != PRODUCTION_OVERLAP_DRAWS:
        raise SystemExit(
            "R5b production overlap draws are fixed at "
            f"{PRODUCTION_OVERLAP_DRAWS}"
        )
    # This is deliberately first: no caller-supplied path is resolved before all literal
    # method pins and the strict runtime policy pass.
    authorization = authorize_scored_stage(args.stage, device=args.device)
    upstream_args = _upstream_args(args)
    upstream = load_r5b_upstream_chain(
        **upstream_args,
        expected_stack_fingerprint=authorization[
            "r5_stack_fingerprint_sha256"
        ],
    )
    if args.stage in {"derivation", "validation"}:
        derivation_package = None
        if args.stage == "validation":
            derivation_package = _verify_exact_residual_predecessor(
                _required_argument(args, "derivation_artifact"),
                _required_argument(args, "derivation_manifest"),
                _required_argument(args, "derivation_completion"),
                stage="derivation",
            )
        receipt = _reserve_stage_attempt(
            args,
            authorization,
            upstream,
            predecessor=derivation_package,
        )
        package = _run_scored_stage(
            args,
            authorization,
            upstream,
            receipt,
            derivation_package=derivation_package,
        )
        print(
            f"R5b {args.stage} status={package['record']['status']} -> "
            f"{package['artifact']['logical_name']} "
            f"completion={package['completion']['logical_name']}"
        )
        return

    derivation_package = _verify_exact_residual_predecessor(
        _required_argument(args, "derivation_artifact"),
        _required_argument(args, "derivation_manifest"),
        _required_argument(args, "derivation_completion"),
        stage="derivation",
    )
    validation_package = _verify_exact_residual_predecessor(
        _required_argument(args, "validation_artifact"),
        _required_argument(args, "validation_manifest"),
        _required_argument(args, "validation_completion"),
        stage="validation",
    )
    _assert_development_record_replays(
        validation_package["record"],
        derivation=derivation_package["record"],
        require_production=True,
    )
    if validation_package["record"].get("derivation_sha256") != canonical_sha256(
        derivation_package["record"]
    ):
        raise InstrumentError("R5b validation is not bound to this derivation")
    development_packages = {
        "derivation": {
            "artifact": derivation_package["artifact"],
            "manifest": derivation_package["manifest"],
        },
        "validation": {
            "artifact": validation_package["artifact"],
            "manifest": validation_package["manifest"],
        },
    }
    disposition, bundle_bytes = build_local_writer_basis_disposition(
        derivation_package["record"],
        validation_package["record"],
        development_packages=development_packages,
        r3d_path_a_binding={
            "artifact_sha256": upstream["r3d"]["artifact"]["sha256"],
            "manifest_sha256": upstream["r3d"]["manifest"]["sha256"],
        },
        r5a_method_preflight_binding={
            "artifact_sha256": upstream["r5a"]["artifact"]["sha256"],
            "manifest_sha256": upstream["r5a"]["manifest"]["sha256"],
        },
        checkpoint_sha256=str(FROZEN_CHECKPOINT_AGGREGATE_SHA256),
        cohort_audit_sha256=str(FROZEN_COHORT_AUDIT_SHA256),
        r5_stack_fingerprint_sha256=authorization[
            "r5_stack_fingerprint_sha256"
        ],
        bundle_path=args.basis_bundle_out,
        seed=SEED,
    )
    disposition["development_completions"] = {
        "derivation": derivation_package["completion"],
        "validation": validation_package["completion"],
    }
    output = _required_argument(args, "out")
    freeze_inputs: list[str | Path] = [
        *_upstream_args(args).values(),
        args.derivation_artifact,
        args.derivation_manifest,
        args.derivation_completion,
        args.validation_artifact,
        args.validation_manifest,
        args.validation_completion,
        *_stack_source_paths(authorization),
    ]
    published = publish_basis_disposition(
        disposition,
        bundle_bytes,
        output_path=output,
        bundle_path=args.basis_bundle_out,
        command=list(sys.argv if argv is None else [str(RUNNER), *argv]),
        input_paths=freeze_inputs,
        derivation_paths=(args.derivation_artifact, args.derivation_manifest),
        validation_paths=(args.validation_artifact, args.validation_manifest),
    )
    print(f"R5b basis disposition={published['status']} -> {Path(output).resolve()}")


if __name__ == "__main__":
    main()
