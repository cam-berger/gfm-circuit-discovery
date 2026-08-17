#!/usr/bin/env python
"""R1c corrected reverse-behavior successor.

Smoke and preflight use validation chromosomes only.  The sealed chr8/9 target is intentionally
unreachable while the protocol is draft: there is no draft override.  See
``DRAFT_R1C_SEALED_REVERSE_BEHAVIOR_CONFIRMATION.md``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    CouplingTask,
    load_adapter,
    load_examples_unscored,
    matched_decoy_donor,
    pick_device,
    substitute,
)
from gfmcd.splicing.coupling import (  # noqa: E402
    MOTIF_CLEAN_DONOR_ALT,
    planted_donor_motifs,
    stable_rng,
)
from gfmcd.splicing.paired_component import (  # noqa: E402
    BOUNDARY_COMPARISON_TOLERANCE_RULE,
    BOUNDARY_PVALUE_RULE,
    boundary_sign_flip_tests,
    gene_sign_flip,
    holm_adjust_named,
    intersection_union_pvalue,
    one_sided_binomial_bounds,
    simultaneous_family_decision,
    strict_interval_decision,
    studentized_gene_bootstrap_tests,
    studentized_gene_bootstrap_tost,
    tost_equivalence_test,
)
from gfmcd.splicing.reverse_coupling import (  # noqa: E402
    acceptor_readout,
    margins_at,
)
from gfmcd.splicing.r5_stack_binding import (  # noqa: E402
    R5StackBindingError,
    STACK_BINDING_FIELD,
    STACK_FINGERPRINT_FIELD,
    STACK_SCHEMA,
    build_r5_stack_binding,
    r5_stack_fingerprint,
    validate_serialized_r5_stack,
    validate_r5_runtime_policy,
)
from gfmcd.splicing.r5_target_ledger import (  # noqa: E402
    R1C_COMPLETION_SCHEMA,
    R1C_IDENTITY_SCHEMA,
    R1C_PROGRAM,
    R1C_RECEIPT_SCHEMA,
    R1CTargetSpentError,
    build_r1c_biological_target_identity,
    complete_r1c_target_attempt,
    reserve_r1c_target_attempt,
)

PROTOCOL = (
    REPO
    / "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R1C_SEALED_REVERSE_BEHAVIOR_CONFIRMATION.md"
)
TARGET_HOLDOUT_ID = "r1c-canonical-internal-short-intron-chr8-chr9-v1"
TARGET_ENDPOINT_POLICY = (
    "single masked acceptor G hard margin; paired versus "
    "distance-matched donor GT->AC 2x2"
)


def holdout_scope_disclosure() -> dict[str, Any]:
    """State exactly what remained blinded after prior sequence-only access.

    Earlier unrelated development tooling materialized chr8/9 sequence records before
    filtering them away, but did not score the masked-acceptor endpoint on those rows.
    The successor is therefore endpoint-outcome held out, not globally sequence unseen.
    This distinction is part of the frozen method rather than post-result prose.
    """

    return {
        "holdout_scope": "masked_acceptor_endpoint_outcomes_on_exact_R1c_target",
        "historical_chr8_chr9_sequence_materialization_disclosed": True,
        "globally_sequence_unseen": False,
        "known_prior_masked_acceptor_endpoint_scores_on_target_rows": 0,
        "licensed_term": "endpoint-outcome-held-out",
        "forbidden_terms": [
            "globally unseen",
            "sequence unseen",
            "virgin chromosome split",
        ],
    }
STATS = REPO / "src/gfmcd/splicing/paired_component.py"
CORE_DEPENDENCIES = (
    REPO / "src/gfmcd/splicing/circuit.py",
    REPO / "src/gfmcd/splicing/coupling.py",
    REPO / "src/gfmcd/splicing/reverse_coupling.py",
    REPO / "src/gfmcd/adapters/splicebert.py",
    REPO / "src/gfmcd/splicing/dataset.py",
    REPO / "src/gfmcd/splicing/gtf.py",
    REPO / "src/gfmcd/artifact_guard.py",
    REPO / "src/gfmcd/provenance.py",
    REPO / "pyproject.toml",
    REPO / "uv.lock",
)

# The old preflight is retained only as an engineering-history identity.  It was run
# before the successor added its FAI, downstream-stack, strict-runtime, robust-estimator,
# and common-ledger contracts, so it can never authorize this target.
HISTORICAL_ENGINEERING_PREFLIGHT_ARTIFACT_SHA256 = (
    "50cda24f0effcb8d9ce0ec9c766d3c6c20306ee8c07be27c182894e829fb6b6a"
)

# Every successor seal starts unset.  Freezing is an intentional source change after
# review, followed by a new clean strict-MPS preflight.  An acknowledgement alone can
# never make the current runner open chr8/9.
FROZEN_PROTOCOL_SHA256: str | None = None
FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = None
FROZEN_PREFLIGHT_ARTIFACT_SHA256: str | None = None
FROZEN_R5_STACK_FINGERPRINT_SHA256: str | None = None
UNSEAL_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_R1C_WILL_OPEN_SEALED_CHR8_CHR9"
SEALED_CHROMOSOMES = frozenset({"chr8", "chr9"})
VALIDATION_CHROMOSOMES = ("chr11", "chr16", "chr19")

SESOI = 0.05
FAMILY_ALPHA = 0.05
PREFLIGHT_POWER_FLOOR = 0.90
MIN_GENES = 30
HEALTH_FLOOR = 0.40
HEALTH_DAMAGE_LIMIT = 0.05
ACCEPTOR_MARGIN_FLOOR = 0.50
TARGET_N = 400
DECOY_TOL = 40
DYAD_DISTANCE_CALIPER = 20
PLANT_DELTA = 0.40
CONSTRUCTION_VERSION = "r1c-acceptor-readout-by-donor-edit-boundary-inference-v4"
COHORT_ORDER_VERSION = "r1c-single-g-primary-family-v2"
PRODUCTION_BATCH = 16
PRODUCTION_SEED = 0
PRODUCTION_RANDOMISATION = 4000
PRODUCTION_TRIALS = 1000
PRODUCTION_HEALTH_PROBES = 8
ROBUST_TRIM_FRACTION = 0.10
PRODUCTION_ROBUST_BOOTSTRAP_DRAWS = 10_000

PRIMARY_FAMILY_NAMES = (
    "Y_PD",
    "Y_PN",
    "S_P",
    "I",
)
CELL_NAMES = (
    "Y_PD",
    "Y_PN",
    "Y_XD",
    "Y_XN",
)
R1C_ROW_POSITION_FIELDS = (
    "prior_real_acceptor",
    "paired_donor",
    "partner_acceptor",
    "matched_decoy_donor",
    "matched_decoy_donor_distance_mismatch",
    "dyad_distance_mismatch",
)
R1C_PARTNER_MARGIN_LEVEL_FIELDS = ("P_clean", "P_under_D", "P_under_N")
DESCRIPTIVE_FAMILY_NAMES = (
    "Y_XD",
    "Y_XN",
    "S_X",
)
CORE_PATH_REQUIREMENTS = {
    0: "MATERIAL_POSITIVE",
    1: "EQUIVALENT_WITHIN_SESOI",
    2: "MATERIAL_POSITIVE",
}
PRIMARY_PLANT_REQUIREMENTS = {
    **CORE_PATH_REQUIREMENTS,
    3: "MATERIAL_POSITIVE",
}
PATH_NAMES = (
    "core_paired_donor",
    "stronger_downstream_preference",
)
POSITIVE_BOUNDARY_NAMES = ("Y_PD", "S_P", "I")
MAX_FALSE_POSITIVES = 38
MIN_POWER_SUCCESSES = 916
LFC_NUMERIC_GUARD = 1e-12
PREFLIGHT_GATE_NAMES = (
    "minimum_30_genes",
    "zero_GT_to_AC_planted_motifs",
    "downstream_acceptor_clean_margin_floor",
    "exon_mate_acceptor_clean_margin_floor",
    "health_clean_floor",
    "health_corrupt_floor",
    "health_damage_bounded",
    "production_path_core_boundary_fpr",
    "production_path_stronger_boundary_fpr",
    "production_path_core_plant_power",
    "production_path_stronger_plant_power",
    "observed_Y_PN_tost_and_strict_ci",
    "raw_trimmed_gene_robustness",
)
TARGET_GATE_NAMES = (
    "minimum_30_genes",
    "zero_GT_to_AC_planted_motifs",
    "downstream_acceptor_clean_margin_floor",
    "exon_mate_acceptor_clean_margin_floor",
    "health_clean_floor",
    "health_corrupt_floor",
    "health_damage_bounded",
    "passing_exact_bound_preflight",
    "raw_trimmed_gene_robustness",
)

PREFLIGHT_HASH_PIN_CONTRACT = (
    "exact immutable preflight artifact SHA-256 pinned in the seal-normalized runner"
)


class TargetSealError(RuntimeError):
    """Raised before any target model or cohort can be loaded."""


def authorize_production_runtime(
    device: str, *, environ: dict[str, str] | None = None
) -> dict:
    """Translate the shared strict-R5 runtime policy into an R1c seal failure."""

    try:
        return validate_r5_runtime_policy(device=device, environ=environ)
    except R5StackBindingError as error:
        raise TargetSealError(str(error)) from error


def downstream_r5_stack_identity(*, repo: Path = REPO) -> dict:
    """Build the one shared, normalized R1c->R5 stack identity."""

    binding = build_r5_stack_binding(repo=repo)
    return {
        STACK_BINDING_FIELD: binding,
        STACK_FINGERPRINT_FIELD: r5_stack_fingerprint(binding),
    }


def verify_frozen_r5_stack(
    stack: dict,
    *,
    frozen_fingerprint_sha256: str | None = FROZEN_R5_STACK_FINGERPRINT_SHA256,
) -> dict:
    """Require the reviewed downstream methods before production may run."""

    if frozen_fingerprint_sha256 is None:
        raise TargetSealError(
            "production requires a reviewed downstream R5 stack fingerprint pin"
        )
    if stack.get(STACK_FINGERPRINT_FIELD) != frozen_fingerprint_sha256:
        raise TargetSealError("current downstream R5 stack differs from its reviewed pin")
    try:
        return validate_serialized_r5_stack(stack)
    except R5StackBindingError as error:
        raise TargetSealError(str(error)) from error


def canonical_sha256(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(rendered.encode()).hexdigest()


def runtime_identity() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "torch", "transformers", "tokenizers", "pyfaidx"):
        try:
            packages[name] = package_version(name)
        except PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _normalise_chromosome(value: str) -> str:
    text = str(value)
    return text if text.startswith("chr") else f"chr{text}"


def resolve_input(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO / candidate
    return candidate.resolve()


def fasta_index_path(fasta: str | Path) -> Path:
    """Return the pre-existing samtools/pyfaidx index bound with the FASTA."""

    resolved = resolve_input(fasta)
    return Path(f"{resolved}.fai")


def logical_path_identity(path: str | Path) -> dict[str, str]:
    """Name repository sources without binding the checkout's absolute location."""

    resolved = resolve_input(path)
    try:
        logical_name = resolved.relative_to(REPO.resolve()).as_posix()
        location_scope = "repository"
    except ValueError:
        logical_name = resolved.name
        location_scope = "external"
    return {"logical_name": logical_name, "location_scope": location_scope}


def file_identity(path: str | Path) -> dict:
    resolved = resolve_input(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"required input is not a regular file: {resolved}")
    return {
        **logical_path_identity(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def snapshot_input_paths(paths: Sequence[str | Path]) -> dict[str, dict[str, object]]:
    """Content-address every bound input before semantic or model use."""
    snapshot: dict[str, dict[str, object]] = {}
    for raw_path in paths:
        path = resolve_input(raw_path)
        if not path.is_file():
            raise RuntimeError(f"bound input is missing or not a regular file: {path}")
        record = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        key = str(path)
        prior = snapshot.setdefault(key, record)
        if prior != record:
            raise RuntimeError(f"bound input changed while being snapshotted: {path}")
    return dict(sorted(snapshot.items()))


def extend_input_snapshot(
    snapshot: dict[str, dict[str, object]], paths: Sequence[str | Path]
) -> dict[str, dict[str, object]]:
    """Add newly created provenance files without tolerating earlier-input drift."""
    extension = snapshot_input_paths(paths)
    for path, record in extension.items():
        if path in snapshot and snapshot[path] != record:
            raise RuntimeError(f"bound input changed before provenance closure: {path}")
    return dict(sorted({**snapshot, **extension}.items()))


def snapshot_record(
    snapshot: dict[str, dict[str, object]], path: str | Path
) -> dict[str, object]:
    key = str(resolve_input(path))
    if key not in snapshot:
        raise RuntimeError(f"path is absent from the bound input snapshot: {key}")
    return snapshot[key]


def verify_input_snapshot(snapshot: dict[str, dict[str, object]]) -> None:
    """Fail closed when any snapshotted input changes during execution."""
    for raw_path, expected in snapshot.items():
        path = Path(raw_path)
        if (
            not path.is_file()
            or path.stat().st_size != expected.get("size_bytes")
            or sha256_file(path) != expected.get("sha256")
        ):
            raise RuntimeError(f"bound input changed during execution: {path}")


def logical_input_snapshot(
    snapshot: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Serialize operational absolute keys as relocation-safe content identities."""

    records = []
    for raw_path, identity in snapshot.items():
        logical = logical_path_identity(raw_path)
        records.append({**logical, **identity})
    return sorted(
        records,
        key=lambda item: (
            str(item["location_scope"]),
            str(item["logical_name"]),
            str(item["sha256"]),
        ),
    )


def atomic_write_new(path: str | Path, data: bytes) -> Path:
    """Durably publish complete bytes without a check-then-write overwrite race."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite immutable artifact {destination}"
            ) from None
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return destination


def normalized_runner_identity(
    path: str | Path = RUNNER,
    *,
    snapshot: dict[str, dict[str, object]] | None = None,
) -> dict:
    """Hash runner logic while normalizing only reviewed freeze-pin RHS values."""
    resolved = resolve_input(path)
    encoded_source = resolved.read_bytes()
    if snapshot is not None:
        expected = snapshot_record(snapshot, resolved)
        if (
            len(encoded_source) != expected["size_bytes"]
            or hashlib.sha256(encoded_source).hexdigest() != expected["sha256"]
        ):
            raise RuntimeError("runner bytes changed after the bound input snapshot")
    text = encoded_source.decode()
    normalized = re.sub(
        r"^(FROZEN_[A-Z0-9_]+_SHA256):[^=]*=.*$",
        r'\1: str | None = "<SEAL_LITERAL>"',
        text,
        flags=re.MULTILINE,
    )
    encoded = normalized.encode()
    return {
        **logical_path_identity(resolved),
        "normalization": "replace only top-level FROZEN_*_SHA256 assignment values",
        "normalized_size_bytes": len(encoded),
        "normalized_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def checkpoint_files(path: str | Path) -> tuple[Path, Path, list[Path]]:
    """Return a deterministic, symlink-free checkpoint inventory."""
    root = resolve_input(path)
    if root.is_file():
        if root.is_symlink():
            raise RuntimeError(f"checkpoint file may not be a symlink: {root}")
        files = [root]
        bundle_root = root.parent
    elif root.is_dir():
        entries = sorted(root.rglob("*"))
        links = [item for item in entries if item.is_symlink()]
        if links:
            raise RuntimeError(f"checkpoint bundle contains symlinks: {links[0]}")
        files = [item for item in entries if item.is_file()]
        bundle_root = root
    else:
        raise FileNotFoundError(f"checkpoint path does not exist: {root}")
    if not files:
        raise FileNotFoundError(f"checkpoint bundle contains no regular files: {root}")
    return root, bundle_root, files


def checkpoint_bundle_identity(
    path: str | Path,
    *,
    snapshot: dict[str, dict[str, object]] | None = None,
    expected_files: Sequence[str | Path] | None = None,
) -> dict:
    """Bind every checkpoint file and the complete relative-path inventory."""
    root, bundle_root, observed_files = checkpoint_files(path)
    files = (
        [resolve_input(item) for item in expected_files]
        if expected_files is not None
        else observed_files
    )
    if files != observed_files:
        raise RuntimeError("checkpoint directory inventory changed after it was frozen")
    records = [
        {
            "relative_path": str(item.relative_to(bundle_root)),
            **(
                snapshot_record(snapshot, item)
                if snapshot is not None
                else {
                    "size_bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            ),
        }
        for item in files
    ]
    return {
        "root": str(root),
        "n_files": len(records),
        "files": records,
        "bundle_sha256": canonical_sha256(records),
    }


def verify_checkpoint_inventory(
    model: str | Path,
    expected_files: Sequence[str | Path],
    snapshot: dict[str, dict[str, object]],
) -> None:
    """Detect checkpoint additions, removals, replacements, and byte drift."""
    _, _, observed = checkpoint_files(model)
    expected = [resolve_input(path) for path in expected_files]
    if observed != expected:
        raise RuntimeError("checkpoint directory inventory changed during execution")
    verify_input_snapshot(
        {str(path): snapshot_record(snapshot, path) for path in expected}
    )


def stage_checkpoint_bundle(
    model: str | Path,
    expected_files: Sequence[str | Path],
    snapshot: dict[str, dict[str, object]],
) -> tuple[tempfile.TemporaryDirectory, Path, dict]:
    """Copy only the frozen inventory into a private, byte-identical model bundle."""
    source_root, bundle_root, _ = checkpoint_files(model)
    verify_checkpoint_inventory(model, expected_files, snapshot)
    expected_identity = checkpoint_bundle_identity(
        model, snapshot=snapshot, expected_files=expected_files
    )
    staging = tempfile.TemporaryDirectory(prefix="gfmcd-r1c-checkpoint-")
    stage_container = Path(staging.name)
    stage_model = stage_container / (
        "model" if source_root.is_dir() else source_root.name
    )
    staged_files: list[Path] = []
    try:
        if source_root.is_dir():
            stage_model.mkdir(parents=True)
            for source in expected_files:
                source_path = resolve_input(source)
                destination = stage_model / source_path.relative_to(bundle_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, destination)
                staged_files.append(destination)
        else:
            shutil.copyfile(source_root, stage_model)
            staged_files.append(stage_model)
        verify_checkpoint_inventory(model, expected_files, snapshot)
        staged_identity = checkpoint_bundle_identity(stage_model)
        if (
            staged_identity["files"] != expected_identity["files"]
            or staged_identity["bundle_sha256"] != expected_identity["bundle_sha256"]
        ):
            raise RuntimeError(
                "staged checkpoint bytes differ from the frozen source snapshot"
            )
    except Exception:
        staging.cleanup()
        raise
    return staging, stage_model, expected_identity


def frozen_instrument_contract() -> dict:
    """The exact reviewed scientific and production contract bound into every fingerprint."""
    return {
        "construction_version": CONSTRUCTION_VERSION,
        "cohort_order_version": COHORT_ORDER_VERSION,
        "primary_family_names": list(PRIMARY_FAMILY_NAMES),
        "cell_names": list(CELL_NAMES),
        "descriptive_family_names": list(DESCRIPTIVE_FAMILY_NAMES),
        "descriptive_status": "REPORTED_NOT_DECISION_BEARING",
        "decision_paths": decision_path_requirements(),
        "path_constituents": {
            name: list(values) for name, values in path_constituent_names().items()
        },
        "boundary_test_contract": boundary_test_contract(),
        "sesoi": SESOI,
        "family_alpha": FAMILY_ALPHA,
        "minimum_genes": MIN_GENES,
        "studentized_simultaneous_intervals": True,
        "interval_scaling": "fixed_endpoint_bootstrap_sd_max_deviation",
        "preflight_power_floor": PREFLIGHT_POWER_FLOOR,
        "plant_delta": PLANT_DELTA,
        "cell_plant_vector": cell_plant_vector().tolist(),
        "primary_plant_vector": structured_plant_vector().tolist(),
        "primary_plant_requirements": {
            PRIMARY_FAMILY_NAMES[index]: label
            for index, label in PRIMARY_PLANT_REQUIREMENTS.items()
        },
        "least_favourable_boundary_cases": least_favourable_boundary_cases(),
        "nuisance_continuum_control": nuisance_continuum_control(),
        "lfc_numeric_guard_toward_composite_null": LFC_NUMERIC_GUARD,
        "maximum_false_positives": MAX_FALSE_POSITIVES,
        "minimum_power_successes": MIN_POWER_SUCCESSES,
        "preflight_gate_names": list(PREFLIGHT_GATE_NAMES),
        "preflight_artifact_hash_pin": PREFLIGHT_HASH_PIN_CONTRACT,
        "health_floor": HEALTH_FLOOR,
        "health_damage_limit": HEALTH_DAMAGE_LIMIT,
        "acceptor_margin_floor": ACCEPTOR_MARGIN_FLOOR,
        "target_n": TARGET_N,
        "target_holdout_id": TARGET_HOLDOUT_ID,
        "holdout_scope_disclosure": holdout_scope_disclosure(),
        "validation_chromosomes": list(VALIDATION_CHROMOSOMES),
        "target_chromosomes": sorted(SEALED_CHROMOSOMES),
        "chromosome_filter_stage": "before_GTF_materialization_and_FASTA_fetch",
        "fasta_index_policy": "pre-existing .fai snapshotted before any genome read",
        "matched_decoy_donor_caliper": DECOY_TOL,
        "dyad_distance_definition": "abs(exon_length-intron_length_down)",
        "dyad_distance_caliper": DYAD_DISTANCE_CALIPER,
        "cohort_scan": "exhaustive_allowed_chromosomes_before_first_n_selection",
        "outer_trial_null": (
            "endpoint-centered 2x2 cell residuals; one shared Rademacher sign per gene; "
            "derive the four-row family after flipping"
        ),
        "production_batch": PRODUCTION_BATCH,
        "production_seed": PRODUCTION_SEED,
        "production_randomisation": PRODUCTION_RANDOMISATION,
        "production_trials": PRODUCTION_TRIALS,
        "production_health_probes": PRODUCTION_HEALTH_PROBES,
        "robust_trim_fraction": ROBUST_TRIM_FRACTION,
        "production_robust_bootstrap_draws": PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
        "strict_production_runtime": {
            "device": "mps",
            "required_absent_environment_variables": [
                "PYTORCH_ENABLE_MPS_FALLBACK"
            ],
        },
    }


def instrument_fingerprint(
    model: str | Path,
    gtf: str | Path,
    fasta: str | Path,
    *,
    protocol: str | Path = PROTOCOL,
    runner: str | Path = RUNNER,
    stats: str | Path = STATS,
    dependencies: Sequence[str | Path] = CORE_DEPENDENCIES,
    input_snapshot: dict[str, dict[str, object]] | None = None,
    checkpoint_inventory: Sequence[str | Path] | None = None,
) -> dict:
    _, _, observed_checkpoint_files = checkpoint_files(model)
    checkpoint_inventory = (
        list(observed_checkpoint_files)
        if checkpoint_inventory is None
        else [resolve_input(path) for path in checkpoint_inventory]
    )
    bound_paths = [
        *checkpoint_inventory,
        resolve_input(gtf),
        resolve_input(fasta),
        fasta_index_path(fasta),
        resolve_input(protocol),
        resolve_input(runner),
        resolve_input(stats),
        *(resolve_input(path) for path in dependencies),
    ]
    snapshot = (
        snapshot_input_paths(bound_paths)
        if input_snapshot is None
        else input_snapshot
    )

    def frozen_file_identity(path: str | Path) -> dict:
        resolved = resolve_input(path)
        return {**logical_path_identity(resolved), **snapshot_record(snapshot, resolved)}

    record = {
        "schema": "gfmcd.r1c.instrument_fingerprint/4",
        "checkpoint": checkpoint_bundle_identity(
            model,
            snapshot=snapshot,
            expected_files=checkpoint_inventory,
        ),
        "gtf": frozen_file_identity(gtf),
        "fasta": frozen_file_identity(fasta),
        "fasta_fai": frozen_file_identity(fasta_index_path(fasta)),
        "protocol": frozen_file_identity(protocol),
        "runner": normalized_runner_identity(runner, snapshot=snapshot),
        "statistics": frozen_file_identity(stats),
        "measurement_dependencies": [
            frozen_file_identity(path) for path in dependencies
        ],
        "runtime": runtime_identity(),
        "contract": frozen_instrument_contract(),
    }
    record["fingerprint_sha256"] = canonical_sha256(record)
    verify_checkpoint_inventory(model, checkpoint_inventory, snapshot)
    verify_input_snapshot(snapshot)
    return record


def bind_runtime_contract(fingerprint: dict, runtime_contract: dict) -> dict:
    """Extend source identity with every runtime choice that can change inference."""
    bound = dict(fingerprint)
    bound["source_fingerprint_sha256"] = bound.pop("fingerprint_sha256")
    bound["runtime_contract"] = dict(runtime_contract)
    bound["fingerprint_sha256"] = canonical_sha256(bound)
    return bound


def authorize_target(
    protocol_sha256: str,
    acknowledgement: str | None,
    *,
    frozen_protocol_sha256: str | None = FROZEN_PROTOCOL_SHA256,
    protocol_text: str | None = None,
) -> None:
    """Require a pinned frozen protocol *and* the exact seal-spend acknowledgement."""
    if frozen_protocol_sha256 is None:
        raise TargetSealError(
            "R1c is DRAFT and target execution is hard-blocked. Review and freeze the protocol, "
            "pin its exact SHA-256 in the runner, and rerun preflight before opening chr8/9."
        )
    status_lines = [line.strip() for line in (protocol_text or "").splitlines()[:12]]
    if not any(line.startswith("> **Status: FROZEN") for line in status_lines):
        raise TargetSealError("the protocol does not declare FROZEN status in its header")
    if protocol_sha256 != frozen_protocol_sha256:
        raise TargetSealError("the protocol bytes do not match the reviewed frozen SHA-256")
    if acknowledgement != UNSEAL_ACKNOWLEDGEMENT:
        raise TargetSealError(
            "target requires the exact --unseal-acknowledgement "
            f"{UNSEAL_ACKNOWLEDGEMENT!r}"
        )


def authorize_production_stage(
    stage: str,
    *,
    protocol_sha256: str,
    protocol_text: str,
    acknowledgement: str | None = None,
    frozen_protocol_sha256: str | None = FROZEN_PROTOCOL_SHA256,
    frozen_instrument_fingerprint_sha256: str | None = (
        FROZEN_INSTRUMENT_FINGERPRINT_SHA256
    ),
    frozen_preflight_artifact_sha256: str | None = (
        FROZEN_PREFLIGHT_ARTIFACT_SHA256
    ),
    frozen_r5_stack_fingerprint_sha256: str | None = (
        FROZEN_R5_STACK_FINGERPRINT_SHA256
    ),
) -> dict:
    """Authorize only a completely frozen successor preflight or target."""

    if stage not in {"preflight", "target"}:
        raise ValueError("production stage must be preflight or target")
    core_pins = {
        "protocol": frozen_protocol_sha256,
        "instrument": frozen_instrument_fingerprint_sha256,
        "downstream R5 stack": frozen_r5_stack_fingerprint_sha256,
    }
    missing = [name for name, value in core_pins.items() if value is None]
    if missing:
        raise TargetSealError(
            "R1c successor is DRAFT; production requires reviewed pins for "
            + ", ".join(missing)
        )
    status_lines = [line.strip() for line in protocol_text.splitlines()[:12]]
    if not any(line.startswith("> **Status: FROZEN") for line in status_lines):
        raise TargetSealError("the successor protocol does not declare FROZEN status")
    if protocol_sha256 != frozen_protocol_sha256:
        raise TargetSealError("the protocol bytes do not match the reviewed frozen SHA-256")
    if stage == "target":
        if frozen_preflight_artifact_sha256 is None:
            raise TargetSealError(
                "target requires a reviewed successor preflight artifact pin"
            )
        if acknowledgement != UNSEAL_ACKNOWLEDGEMENT:
            raise TargetSealError(
                "target requires the exact --unseal-acknowledgement "
                f"{UNSEAL_ACKNOWLEDGEMENT!r}"
            )
    return {
        "stage": stage,
        "protocol_sha256": protocol_sha256,
        "instrument_fingerprint_pin_present": True,
        "r5_stack_fingerprint_sha256": frozen_r5_stack_fingerprint_sha256,
        "preflight_artifact_pin_required": stage == "target",
        "explicit_unseal_acknowledgement": (
            acknowledgement == UNSEAL_ACKNOWLEDGEMENT if stage == "target" else False
        ),
    }


def verify_frozen_instrument(
    fingerprint_sha256: str,
    *,
    frozen_fingerprint_sha256: str | None = FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
) -> None:
    if frozen_fingerprint_sha256 is None:
        raise TargetSealError(
            "target requires a reviewed instrument fingerprint pinned in the runner"
        )
    if fingerprint_sha256 != frozen_fingerprint_sha256:
        raise TargetSealError(
            "current checkpoint/data/source/runtime fingerprint differs from the reviewed pin"
        )


def verify_pinned_preflight_artifact(
    artifact_sha256: str,
    *,
    frozen_artifact_sha256: str | None = FROZEN_PREFLIGHT_ARTIFACT_SHA256,
) -> None:
    """Bind target authorization to one reviewed immutable preflight artifact.

    Replaying 1,000 outer trials, each containing several 4,000-draw production
    classifiers, on the target authorization path would be both costly and a second
    implementation of the calibrator.  The complete preflight JSON is therefore a
    third seal literal.  Its value is normalized out of the runner fingerprint, just
    like the protocol and instrument pins, so installing the reviewed artifact hash
    cannot create a fingerprint cycle.
    """
    if frozen_artifact_sha256 is None:
        raise TargetSealError(
            "target requires the reviewed preflight artifact SHA-256 pinned in the runner"
        )
    for label, value in (
        ("observed", artifact_sha256),
        ("frozen", frozen_artifact_sha256),
    ):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise TargetSealError(f"{label} preflight artifact SHA-256 is malformed")
    if artifact_sha256 != frozen_artifact_sha256:
        raise TargetSealError(
            "preflight artifact bytes differ from the reviewed literal SHA-256 pin"
        )


def require_target_seal_literals() -> None:
    """Stop a not-yet-authorised target before resolving any supplied input path."""

    if FROZEN_INSTRUMENT_FINGERPRINT_SHA256 is None:
        raise TargetSealError(
            "target requires a reviewed instrument fingerprint pinned in the runner"
        )
    if FROZEN_PREFLIGHT_ARTIFACT_SHA256 is None:
        raise TargetSealError(
            "target remains sealed until the reviewed preflight artifact SHA-256 is "
            "pinned in the runner"
        )


def worktree_identity(repo: str | Path = REPO) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(repo), text=True
    )
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def require_clean_worktree(repo: str | Path = REPO) -> str:
    identity = worktree_identity(repo)
    if identity["git_dirty"]:
        raise TargetSealError("target requires a clean Git worktree")
    return str(identity["git_commit"])


def require_clean_production_worktree(
    stage: str, *, repo: str | Path = REPO
) -> str:
    if stage not in {"preflight", "target"}:
        raise ValueError("clean-worktree authorization is production-only")
    return require_clean_worktree(repo)


def _preflight_semantic_error(message: str) -> TargetSealError:
    return TargetSealError(f"R1c preflight semantic contract mismatch: {message}")


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(value)
    )


def _preflight_config_contract() -> dict:
    return {
        "construction_version": CONSTRUCTION_VERSION,
        "cohort_order_version": COHORT_ORDER_VERSION,
        "donor_alt": MOTIF_CLEAN_DONOR_ALT,
        "single_G_acceptor_endpoint": True,
        "outcome_conditioning": None,
        "sesoi": SESOI,
        "family_alpha": FAMILY_ALPHA,
        "minimum_genes": MIN_GENES,
        "primary_family_names": list(PRIMARY_FAMILY_NAMES),
        "cell_names": list(CELL_NAMES),
        "descriptive_family_names": list(DESCRIPTIVE_FAMILY_NAMES),
        "descriptive_status": "REPORTED_NOT_DECISION_BEARING",
        "decision_paths": decision_path_requirements(),
        "path_constituents": {
            name: list(values) for name, values in path_constituent_names().items()
        },
        "boundary_test_contract": boundary_test_contract(),
        "studentized_simultaneous_intervals": True,
        "interval_scaling": "fixed_endpoint_bootstrap_sd_max_deviation",
        "least_favourable_boundary_cases": least_favourable_boundary_cases(),
        "nuisance_continuum_control": nuisance_continuum_control(),
        "lfc_numeric_guard_toward_composite_null": LFC_NUMERIC_GUARD,
        "maximum_false_positives": MAX_FALSE_POSITIVES,
        "minimum_power_successes": MIN_POWER_SUCCESSES,
        "matched_decoy_donor_caliper": DECOY_TOL,
        "dyad_distance_definition": "abs(exon_length-intron_length_down)",
        "dyad_distance_caliper": DYAD_DISTANCE_CALIPER,
        "cohort_scan": "exhaustive_allowed_chromosomes_before_first_n_selection",
        "outer_trial_null": (
            "endpoint-centered 2x2 cell residuals; one shared Rademacher sign per gene; "
            "derive the four-row family after flipping"
        ),
        "plant_delta": PLANT_DELTA,
        "cell_plant_vector": cell_plant_vector().tolist(),
        "primary_plant_vector": structured_plant_vector().tolist(),
        "target_holdout_id": TARGET_HOLDOUT_ID,
        "validation_chromosomes": list(VALIDATION_CHROMOSOMES),
        "target_chromosomes": sorted(SEALED_CHROMOSOMES),
        "chromosome_filter_stage": "before_GTF_materialization_and_FASTA_fetch",
        "health_floor": HEALTH_FLOOR,
        "health_damage_limit": HEALTH_DAMAGE_LIMIT,
        "acceptor_margin_floor": ACCEPTOR_MARGIN_FLOOR,
        "preflight_power_floor": PREFLIGHT_POWER_FLOOR,
        "preflight_artifact_hash_pin": PREFLIGHT_HASH_PIN_CONTRACT,
        "batch": PRODUCTION_BATCH,
        "health_probes": PRODUCTION_HEALTH_PROBES,
        "n_randomisation": PRODUCTION_RANDOMISATION,
        "n_trials": PRODUCTION_TRIALS,
        "robust_bootstrap_draws": PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
        "robust_trim_fraction": ROBUST_TRIM_FRACTION,
        "seed": PRODUCTION_SEED,
    }


def _component_plant_requirements() -> dict[str, str]:
    return {
        f"component_{index}": label
        for index, label in sorted(PRIMARY_PLANT_REQUIREMENTS.items())
    }


def _cohort_payload_from_rows(rows: Sequence[dict]) -> list[dict]:
    """Rebuild the exact selected-cohort payload used by ``cohort_sha256``."""
    payload: list[dict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise _preflight_semantic_error(f"row {index} is not a record")
        positions = row.get("positions")
        if not isinstance(positions, dict):
            raise _preflight_semantic_error(f"row {index} lacks position metadata")
        required = (
            "matched_decoy_donor",
            "matched_decoy_donor_distance_mismatch",
            "dyad_distance_mismatch",
        )
        if any(
            isinstance(positions.get(name), bool)
            or not isinstance(positions.get(name), int)
            for name in required
        ):
            raise _preflight_semantic_error(
                f"row {index} has malformed selected-cohort coordinates"
            )
        example_id = row.get("example_id")
        gene_id = row.get("gene_id")
        if not isinstance(example_id, str) or not example_id:
            raise _preflight_semantic_error(f"row {index} lacks an example_id")
        if not isinstance(gene_id, str) or not gene_id:
            raise _preflight_semantic_error(f"row {index} lacks a gene_id")
        payload.append(
            {
                "example_id": example_id,
                "gene_id": gene_id,
                "decoy_donor": positions["matched_decoy_donor"],
                "decoy_donor_distance_mismatch": positions[
                    "matched_decoy_donor_distance_mismatch"
                ],
                "dyad_distance_mismatch": positions["dyad_distance_mismatch"],
            }
        )
    return payload


def structural_target_cohort_identity(rows: Sequence[dict]) -> dict[str, str]:
    """Digest the outcome-blind sealed cohort before any target model score."""

    source_payload = _cohort_payload_from_rows(rows)
    ordered_six_position_rows: list[dict] = []
    for index, row in enumerate(rows):
        positions = row.get("positions")
        if not isinstance(positions, dict) or set(positions) != set(
            R1C_ROW_POSITION_FIELDS
        ):
            raise ValueError(f"row {index} does not carry the exact six position fields")
        identity = {
            "example_id": row["example_id"],
            "gene_id": row["gene_id"],
            "chrom": row.get("chrom"),
            "strand": row.get("strand"),
            "positions": {
                name: positions[name] for name in R1C_ROW_POSITION_FIELDS
            },
        }
        ordered_six_position_rows.append(identity)
    ordered_example_ids = [str(row["example_id"]) for row in rows]
    ordered_gene_ids = [str(row["gene_id"]) for row in rows]
    construction = {
        "construction_version": CONSTRUCTION_VERSION,
        "cohort_order_version": COHORT_ORDER_VERSION,
        "ordered_six_position_rows": ordered_six_position_rows,
    }
    return {
        "source_cohort_sha256": canonical_sha256(source_payload),
        "ordered_example_ids_sha256": canonical_sha256(ordered_example_ids),
        "target_gene_ids_sha256": canonical_sha256(sorted(set(ordered_gene_ids))),
        "ordered_gene_ids_sha256": canonical_sha256(ordered_gene_ids),
        "ordered_six_position_rows_sha256": canonical_sha256(
            ordered_six_position_rows
        ),
        "construction_sha256": canonical_sha256(construction),
    }


def target_cohort_identity(rows: Sequence[dict]) -> dict[str, str]:
    """Add same-row partner-margin identity to the structural target package."""

    structural = structural_target_cohort_identity(rows)
    ordered_partner_margin_levels = []
    for index, row in enumerate(rows):
        margin_levels = row.get("margin_levels")
        if not isinstance(margin_levels, dict) or any(
            name not in margin_levels for name in R1C_PARTNER_MARGIN_LEVEL_FIELDS
        ):
            raise ValueError(f"row {index} lacks partner margin levels")
        ordered_partner_margin_levels.append(
            {
                "example_id": row["example_id"],
                "gene_id": row["gene_id"],
                "margin_levels": {
                    name: margin_levels[name]
                    for name in R1C_PARTNER_MARGIN_LEVEL_FIELDS
                },
            }
        )
    return structural | {
        "ordered_partner_margin_levels_sha256": canonical_sha256(
            ordered_partner_margin_levels
        )
    }


def construction_contract_sha256() -> str:
    """Bind the pre-read deterministic construction without outcome-derived rows."""

    return canonical_sha256(
        {
            "construction_version": CONSTRUCTION_VERSION,
            "cohort_order_version": COHORT_ORDER_VERSION,
            "target_holdout_id": TARGET_HOLDOUT_ID,
            "target_chromosomes": sorted(SEALED_CHROMOSOMES),
            "target_n": TARGET_N,
            "row_position_fields": list(R1C_ROW_POSITION_FIELDS),
            "matched_decoy_donor_caliper": DECOY_TOL,
            "dyad_distance_caliper": DYAD_DISTANCE_CALIPER,
            "cohort_scan": "exhaustive_allowed_chromosomes_before_first_n_selection",
        }
    )


def replay_preflight_rows(preflight: dict, *, stage: str = "preflight") -> dict:
    """Independently reconstruct stored behavior from serialized primitive rows.

    This replay deliberately starts from margin levels, not from the artifact's derived
    family or its pass booleans.  It rebuilds all four primitive cells, the four-member
    decision family, health summaries, selected-cohort identity, and the exact observed
    boundary classifier under the frozen namespaces and seeds.
    """
    if stage not in {"preflight", "target"}:
        raise ValueError("row replay stage must be preflight or target")
    rows = preflight.get("rows")
    if not isinstance(rows, list) or len(rows) != TARGET_N:
        raise _preflight_semantic_error(
            f"exactly {TARGET_N} serialized primitive rows are required"
        )
    example_ids = [row.get("example_id") if isinstance(row, dict) else None for row in rows]
    if any(not isinstance(value, str) or not value for value in example_ids):
        raise _preflight_semantic_error("every primitive row needs an example_id")
    if len(set(example_ids)) != TARGET_N:
        raise _preflight_semantic_error("primitive-row example_ids must be unique")

    cohort_payload = _cohort_payload_from_rows(rows)
    groups = [str(item["gene_id"]) for item in cohort_payload]
    chromosomes = {_normalise_chromosome(str(row.get("chrom"))) for row in rows}
    expected_chromosomes = (
        set(VALIDATION_CHROMOSOMES)
        if stage == "preflight"
        else set(SEALED_CHROMOSOMES)
    )
    if not chromosomes or any(
        chrom not in expected_chromosomes
        for chrom in chromosomes
    ):
        raise _preflight_semantic_error(
            f"serialized {stage} rows use chromosomes outside the frozen allowlist"
        )

    level_names = (
        "P_clean",
        "P_under_D",
        "P_under_N",
        "X_clean",
        "X_under_D",
        "X_under_N",
    )
    level_rows: list[list[float]] = []
    recorded_cell_rows: list[list[float]] = []
    recorded_family_rows: list[list[float]] = []
    recorded_descriptive_rows: list[list[float]] = []
    health_rows: list[list[float]] = []
    for index, row in enumerate(rows):
        for section_name, names, destination in (
            ("margin_levels", level_names, level_rows),
            ("cell_effects", CELL_NAMES, recorded_cell_rows),
            ("family", PRIMARY_FAMILY_NAMES, recorded_family_rows),
            ("descriptive_2x2", DESCRIPTIVE_FAMILY_NAMES, recorded_descriptive_rows),
            (
                "health",
                ("clean_accuracy", "corrupt_accuracy", "damage"),
                health_rows,
            ),
        ):
            section = row.get(section_name)
            if not isinstance(section, dict) or set(section) != set(names):
                raise _preflight_semantic_error(
                    f"row {index} has an incomplete {section_name} record"
                )
            values = [section[name] for name in names]
            if not all(_is_finite_number(value) for value in values):
                raise _preflight_semantic_error(
                    f"row {index} has non-finite {section_name} values"
                )
            destination.append([float(value) for value in values])

    levels = np.asarray(level_rows, dtype=float).T
    recorded_cells = np.asarray(recorded_cell_rows, dtype=float).T
    recorded_family = np.asarray(recorded_family_rows, dtype=float).T
    recorded_descriptive = np.asarray(recorded_descriptive_rows, dtype=float).T
    health = np.asarray(health_rows, dtype=float).T
    reconstructed_cells = np.vstack(
        (
            levels[0] - levels[1],
            levels[0] - levels[2],
            levels[3] - levels[4],
            levels[3] - levels[5],
        )
    )
    if not np.allclose(
        recorded_cells, reconstructed_cells, rtol=0.0, atol=1e-12
    ):
        raise _preflight_semantic_error(
            "primitive cells do not reconstruct from the stored margin levels"
        )
    reconstructed_family = _family_from_cells(reconstructed_cells)
    if not np.allclose(
        recorded_family, reconstructed_family, rtol=0.0, atol=1e-12
    ):
        raise _preflight_semantic_error(
            "stored family does not reconstruct from the four primitive cells"
        )
    reconstructed_descriptive = np.vstack(
        (
            reconstructed_cells[2],
            reconstructed_cells[3],
            reconstructed_cells[2] - reconstructed_cells[3],
        )
    )
    if not np.allclose(
        recorded_descriptive, reconstructed_descriptive, rtol=0.0, atol=1e-12
    ):
        raise _preflight_semantic_error(
            "descriptive 2x2 row does not reconstruct from the primitive cells"
        )
    if (
        np.any(health[:2] < 0)
        or np.any(health[:2] > 1)
        or not np.allclose(health[2], health[0] - health[1], rtol=0.0, atol=1e-12)
    ):
        raise _preflight_semantic_error(
            "stored health damage does not reconstruct from bounded accuracies"
        )

    n_genes = len(set(groups))
    replayed_inference = r1c_boundary_classifier(
        reconstructed_family,
        groups,
        sesoi=SESOI,
        n_randomisation=PRODUCTION_RANDOMISATION,
        alpha=FAMILY_ALPHA,
        seed=PRODUCTION_SEED,
        namespace=f"r1c:{stage}:production-boundary",
        min_genes=MIN_GENES,
        studentized=True,
    )
    stored_inference = preflight.get("boundary_inference")
    if replayed_inference != stored_inference:
        raise _preflight_semantic_error(
            "observed estimates, statistics, intervals, p-values, or replay digests "
            "do not reproduce from primitive rows"
        )

    replayed_robustness = r1c_gene_robustness_gate(
        reconstructed_family,
        groups,
        n_draws=PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
        seed=PRODUCTION_SEED,
        namespace=f"r1c:{stage}:production-gene-robustness",
    )
    if preflight.get("gene_robustness") != replayed_robustness:
        raise _preflight_semantic_error(
            "raw/trimmed equal-gene robustness does not reproduce from primitive rows"
        )

    health_summary = {
        "n_probe_per_example_per_construct": PRODUCTION_HEALTH_PROBES,
        "clean_mean": float(health[0].mean()),
        "corrupt_mean": float(health[1].mean()),
        "damage_mean": float(health[2].mean()),
        "floor": HEALTH_FLOOR,
        "damage_limit": HEALTH_DAMAGE_LIMIT,
    }
    if preflight.get("health") != health_summary:
        raise _preflight_semantic_error(
            "health summary does not reproduce from primitive rows"
        )
    selected_sha256 = canonical_sha256(cohort_payload)
    if preflight.get("cohort_sha256") != selected_sha256:
        raise _preflight_semantic_error(
            "selected cohort SHA-256 does not reproduce from primitive rows"
        )
    expected_row_gates = {
        "minimum_30_genes": n_genes >= MIN_GENES,
        "downstream_acceptor_clean_margin_floor": float(levels[0].mean())
        >= ACCEPTOR_MARGIN_FLOOR,
        "exon_mate_acceptor_clean_margin_floor": float(levels[3].mean())
        >= ACCEPTOR_MARGIN_FLOOR,
        "health_clean_floor": health_summary["clean_mean"] >= HEALTH_FLOOR,
        "health_corrupt_floor": health_summary["corrupt_mean"] >= HEALTH_FLOOR,
        "health_damage_bounded": abs(health_summary["damage_mean"])
        <= HEALTH_DAMAGE_LIMIT,
        "raw_trimmed_gene_robustness": bool(
            replayed_robustness["paths"]["core_paired_donor"]["pass"]
        ),
    }
    if stage == "preflight":
        expected_row_gates["observed_Y_PN_tost_and_strict_ci"] = (
            replayed_inference["endpoint_p_values"]["Y_PN"] <= FAMILY_ALPHA
            and replayed_inference["family_decisions"]["Y_PN"]["classification"]
            == "EQUIVALENT_WITHIN_SESOI"
        )
    gates = preflight.get("gates")
    if not isinstance(gates, dict) or any(
        gates.get(name) is not expected
        for name, expected in expected_row_gates.items()
    ):
        raise _preflight_semantic_error(
            "health, margin, cohort-size, or observed-equivalence gates do not "
            "reproduce from primitive rows"
        )
    return {
        "cells": reconstructed_cells,
        "family": reconstructed_family,
        "groups": groups,
        "n_genes": n_genes,
        "cohort_sha256": selected_sha256,
        "selected_example_ids_sha256": canonical_sha256(example_ids),
        "inference": replayed_inference,
        "health": health_summary,
        "robustness": replayed_robustness,
        "target_cohort_identity": target_cohort_identity(rows),
        "row_gates": expected_row_gates,
    }


def verify_preflight_binding(preflight: dict, fingerprint_sha256: str) -> None:
    if preflight.get("schema") != "gfmcd.r1c.reverse_behavior/3":
        raise TargetSealError("target preflight input has the wrong R1c schema")
    if preflight.get("experiment") != "R1c_corrected_reverse_behavior":
        raise TargetSealError("target preflight input belongs to a different experiment")
    if preflight.get("stage") != "preflight":
        raise TargetSealError("target preflight input is not an R1c preflight artifact")
    if preflight.get("split") != "val" or preflight.get("n") != TARGET_N:
        raise TargetSealError(
            f"R1c preflight must be the fixed val cohort with n={TARGET_N}"
        )
    if preflight.get("sealed_split_opened") is not False:
        raise TargetSealError("preflight artifact does not attest that chr8/9 stayed sealed")
    if preflight.get("all_gates_pass") is not True:
        raise TargetSealError("R1c preflight did not pass every gate")
    n_genes = preflight.get("n_genes")
    if isinstance(n_genes, bool) or not isinstance(n_genes, int) or n_genes < MIN_GENES:
        raise _preflight_semantic_error(f"at least {MIN_GENES} independent genes are required")
    outer_contract = {
        "protocol_status": "FROZEN_HASH_PINNED",
        "status": "DEVELOPMENT",
        "evidence_rung": "BEHAVIOR_ONLY",
        "verdict": "PREFLIGHT_GO",
    }
    for key, expected in outer_contract.items():
        if preflight.get(key) != expected:
            raise _preflight_semantic_error(f"{key} must equal {expected!r}")
    if preflight.get("freeze_authorization") != {
        "protocol_declares_frozen": True,
        "protocol_hash_pinned": True,
        "instrument_fingerprint_pinned": True,
        "downstream_r5_stack_pinned": True,
        "strict_mps_runtime_verified": True,
        "authorization_ready": True,
    }:
        raise _preflight_semantic_error(
            "preflight must be generated after every reviewed successor pin is installed"
        )
    try:
        validate_serialized_r5_stack(preflight)
    except R5StackBindingError as error:
        raise _preflight_semantic_error(
            "serialized downstream R5 stack identity differs"
        ) from error

    config = preflight.get("config")
    if config != _preflight_config_contract():
        raise _preflight_semantic_error(
            "the complete production config differs from the frozen R1c contract"
        )

    gates = preflight.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(PREFLIGHT_GATE_NAMES):
        raise _preflight_semantic_error(
            "the preflight gate family is incomplete, extended, or reordered by a different design"
        )
    failed_gates = sorted(name for name, value in gates.items() if value is not True)
    if failed_gates:
        raise _preflight_semantic_error(f"required gates did not pass: {failed_gates}")

    inference = preflight.get("boundary_inference")
    if not isinstance(inference, dict):
        raise _preflight_semantic_error("the boundary-inference record is missing")
    expected_inference_header = {
        "schema": "gfmcd.r1c.boundary_classifier/1",
        "namespace": "r1c:preflight:production-boundary",
        "alpha": FAMILY_ALPHA,
        "sesoi": SESOI,
        "n_randomisation": PRODUCTION_RANDOMISATION,
        "minimum_genes_required": MIN_GENES,
        "boundary_test_contract": boundary_test_contract(),
    }
    for key, expected in expected_inference_header.items():
        if inference.get(key) != expected:
            raise _preflight_semantic_error(
                f"boundary inference {key} must equal {expected!r}"
            )

    decisions = inference.get("family_decisions")
    if not isinstance(decisions, dict) or set(decisions) != set(PRIMARY_FAMILY_NAMES):
        raise _preflight_semantic_error("the decision family is not the exact ordered four-row family")
    allowed_classes = {
        "MATERIAL_POSITIVE",
        "MATERIAL_NEGATIVE",
        "EQUIVALENT_WITHIN_SESOI",
        "UNRESOLVED",
    }
    if any(
        not isinstance(decision, dict)
        or decision.get("classification") not in allowed_classes
        or decision.get("classification_rule")
        != "strict simultaneous SESOI boundaries"
        for decision in decisions.values()
    ):
        raise _preflight_semantic_error("the decision family contains an invalid classification")
    for name, decision in decisions.items():
        interval = decision.get("simultaneous_ci")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(_is_finite_number(value) for value in interval)
        ):
            raise _preflight_semantic_error(f"{name} lacks a finite simultaneous interval")
        expected_class = strict_interval_decision(interval[0], interval[1], SESOI)
        if decision.get("classification") != expected_class:
            raise _preflight_semantic_error(
                f"{name} classification does not use strict SESOI boundaries"
            )
    if preflight.get("family_decisions") != decisions:
        raise _preflight_semantic_error(
            "top-level family decisions differ from boundary inference"
        )

    boundary_tests = inference.get("boundary_tests")
    expected_test_names = {
        "Y_PD_gt_SESOI",
        "S_P_gt_SESOI",
        "I_gt_SESOI",
        "Y_PN_gt_negative_SESOI",
        "Y_PN_lt_positive_SESOI",
    }
    if not isinstance(boundary_tests, dict) or set(boundary_tests) != expected_test_names:
        raise _preflight_semantic_error("the five constituent boundary tests are incomplete")
    expected_boundary_specs = {
        "Y_PD_gt_SESOI": (SESOI, "greater"),
        "S_P_gt_SESOI": (SESOI, "greater"),
        "I_gt_SESOI": (SESOI, "greater"),
        "Y_PN_gt_negative_SESOI": (-SESOI, "greater"),
        "Y_PN_lt_positive_SESOI": (SESOI, "less"),
    }
    expected_observed_boundary_namespace = (
        "r1c:preflight:production-boundary:shared-boundary-signs"
    )
    for test_name, test in boundary_tests.items():
        if not isinstance(test, dict):
            raise _preflight_semantic_error(f"boundary test {test_name} is malformed")
        exceedances = test.get("exceedances")
        B = test.get("B")
        if (
            isinstance(exceedances, bool)
            or not isinstance(exceedances, int)
            or isinstance(B, bool)
            or not isinstance(B, int)
            or B != PRODUCTION_RANDOMISATION
            or not 0 <= exceedances <= B
        ):
            raise _preflight_semantic_error(
                f"boundary test {test_name} has invalid exceedance accounting"
            )
        expected_p = (1 + exceedances) / (B + 1)
        if not _is_finite_number(test.get("p_value")) or not np.isclose(
            test["p_value"], expected_p, rtol=0.0, atol=1e-15
        ):
            raise _preflight_semantic_error(
                f"boundary test {test_name} violates the exact add-one rule"
            )
        expected_boundary, expected_alternative = expected_boundary_specs[test_name]
        if (
            test.get("seed") != PRODUCTION_SEED
            or test.get("namespace") != expected_observed_boundary_namespace
            or not _is_finite_number(test.get("boundary"))
            or not np.isclose(
                test["boundary"], expected_boundary, rtol=0.0, atol=1e-15
            )
            or test.get("alternative") != expected_alternative
            or test.get("n_examples") != TARGET_N
            or test.get("n_genes") != n_genes
            or test.get("randomisation_unit") != "gene"
            or test.get("pvalue_rule") != BOUNDARY_PVALUE_RULE
            or not _is_finite_number(test.get("comparison_tolerance"))
            or test["comparison_tolerance"] <= 0
            or not _is_finite_number(test.get("add_one_floor"))
            or not np.isclose(
                test["add_one_floor"], 1 / (B + 1), rtol=0.0, atol=1e-15
            )
        ):
            raise _preflight_semantic_error(
                f"boundary test {test_name} lacks its exact hypothesis or replay metadata"
            )
        for digest_key in (
            "null_draws_sha256",
            "shared_null_draws_sha256",
            "sign_matrix_sha256",
            "ordered_gene_labels_sha256",
        ):
            if not isinstance(test.get(digest_key), str) or not re.fullmatch(
                r"[0-9a-f]{64}", test[digest_key]
            ):
                raise _preflight_semantic_error(
                    f"boundary test {test_name} lacks {digest_key}"
                )
    sign_matrix_digests = {
        test["sign_matrix_sha256"] for test in boundary_tests.values()
    }
    gene_label_digests = {
        test["ordered_gene_labels_sha256"] for test in boundary_tests.values()
    }
    expected_shared_randomisation = {
        "namespace": expected_observed_boundary_namespace,
        "sign_matrix_sha256": next(iter(sign_matrix_digests)),
        "ordered_gene_labels_sha256": next(iter(gene_label_digests)),
        "all_five_boundary_tests_share_sign_matrix": True,
    }
    if (
        len(sign_matrix_digests) != 1
        or len(gene_label_digests) != 1
        or inference.get("shared_boundary_randomisation")
        != expected_shared_randomisation
    ):
        raise _preflight_semantic_error(
            "all five boundary tests must reuse one ordered-gene sign matrix"
        )

    endpoint_p_values = inference.get("endpoint_p_values")
    if not isinstance(endpoint_p_values, dict) or set(endpoint_p_values) != set(
        PRIMARY_FAMILY_NAMES
    ):
        raise _preflight_semantic_error("endpoint boundary p-values are incomplete")
    recomputed_endpoint_p = {
        "Y_PD": boundary_tests["Y_PD_gt_SESOI"]["p_value"],
        "Y_PN": max(
            boundary_tests["Y_PN_gt_negative_SESOI"]["p_value"],
            boundary_tests["Y_PN_lt_positive_SESOI"]["p_value"],
        ),
        "S_P": boundary_tests["S_P_gt_SESOI"]["p_value"],
        "I": boundary_tests["I_gt_SESOI"]["p_value"],
    }
    expected_tost = {
        "estimate": decisions["Y_PN"]["estimate"],
        "sesoi": SESOI,
        "p_value": recomputed_endpoint_p["Y_PN"],
        "iut_rule": "max(lower_edge_p, upper_edge_p)",
        "lower_edge_test": boundary_tests["Y_PN_gt_negative_SESOI"],
        "upper_edge_test": boundary_tests["Y_PN_lt_positive_SESOI"],
    }
    if inference.get("Y_PN_TOST") != expected_tost:
        raise _preflight_semantic_error("Y_PN TOST does not match its two edge tests")
    if any(
        not _is_finite_number(endpoint_p_values.get(name))
        or not np.isclose(
            endpoint_p_values[name], expected, rtol=0.0, atol=1e-15
        )
        for name, expected in recomputed_endpoint_p.items()
    ):
        raise _preflight_semantic_error("endpoint p-values do not match boundary tests")

    constituent_names = path_constituent_names()
    raw_path_p = {
        path: intersection_union_pvalue(
            [endpoint_p_values[name] for name in constituent_names[path]]
        )
        for path in PATH_NAMES
    }
    adjusted_path_p = holm_adjust_named(raw_path_p)
    path_tests = inference.get("path_tests")
    if not isinstance(path_tests, dict) or set(path_tests) != set(PATH_NAMES):
        raise _preflight_semantic_error("the exact two-path inference family is missing")
    requirements = decision_path_requirements()
    for path in PATH_NAMES:
        path_test = path_tests[path]
        ci_pass = all(
            decisions[name]["classification"] == label
            for name, label in requirements[path].items()
        )
        p_pass = adjusted_path_p[path] <= FAMILY_ALPHA
        disagreement = bool(p_pass) != bool(ci_pass)
        declared = bool(p_pass and ci_pass)
        path_decision = (
            "DECLARED"
            if declared
            else ("UNRESOLVED" if disagreement else "NOT_DECLARED")
        )
        expected_path_record = {
            "constituent_p_values": {
                name: endpoint_p_values[name] for name in constituent_names[path]
            },
            "iut_p_value": raw_path_p[path],
            "iut_rule": "maximum constituent p-value",
            "holm_adjusted_p_value": adjusted_path_p[path],
            "holm_family": list(PATH_NAMES),
            "alpha": FAMILY_ALPHA,
            "p_rejects": bool(p_pass),
            "strict_ci_requirements": requirements[path],
            "strict_ci_passes": bool(ci_pass),
            "p_ci_disagreement": disagreement,
            "decision": path_decision,
            "unresolved_reason": (
                "BOUNDARY_P_STRICT_CI_DISAGREEMENT"
                if disagreement
                else None
            ),
            "nondeclaration_reason": (
                "BOTH_NECESSARY_GATES_FAILED"
                if path_decision == "NOT_DECLARED"
                else None
            ),
        }
        if path_test != expected_path_record:
            raise _preflight_semantic_error(
                f"{path} does not reproduce IUT, Holm, and strict-CI inference"
            )
    if inference.get("holm_adjusted_path_p_values") != adjusted_path_p:
        raise _preflight_semantic_error("adjusted path p-values do not reproduce Holm")
    if preflight.get("observed_decision_paths") != observed_decision_paths(inference):
        raise _preflight_semantic_error("observed path declarations are inconsistent")
    if (
        endpoint_p_values["Y_PN"] > FAMILY_ALPHA
        or decisions["Y_PN"]["classification"] != "EQUIVALENT_WITHIN_SESOI"
    ):
        raise _preflight_semantic_error(
            "Y_PN must pass both TOST and the strict simultaneous equivalence interval"
        )

    row_replay = replay_preflight_rows(preflight)
    if row_replay["n_genes"] != n_genes:
        raise _preflight_semantic_error(
            "top-level gene count does not reproduce from primitive rows"
        )
    if preflight.get("target_cohort_identity") != row_replay[
        "target_cohort_identity"
    ]:
        raise _preflight_semantic_error(
            "complete cohort identity does not reproduce from primitive rows"
        )

    cohort_audit = preflight.get("cohort_audit")
    if not isinstance(cohort_audit, dict):
        raise _preflight_semantic_error("the exhaustive cohort audit is missing")
    expected_audit = {
        "selected": TARGET_N,
        "selected_genes": n_genes,
        "selected_example_ids_sha256": row_replay["selected_example_ids_sha256"],
        "selected_cohort_sha256": row_replay["cohort_sha256"],
        "dyad_distance_definition": "abs(exon_length-intron_length_down)",
        "dyad_distance_caliper": DYAD_DISTANCE_CALIPER,
        "selection_reads_model_outputs": False,
    }
    for key, expected in expected_audit.items():
        if cohort_audit.get(key) != expected:
            raise _preflight_semantic_error(
                f"cohort audit {key} must equal {expected!r}"
            )
    eligible = cohort_audit.get("distance_matched_2x2_eligible")
    if isinstance(eligible, bool) or not isinstance(eligible, int) or eligible < TARGET_N:
        raise _preflight_semantic_error(
            f"the fixed dyad caliper must supply at least {TARGET_N} eligible rows"
        )
    for key in (
        "screened_example_ids_sha256",
        "primary_comparator_complete_example_ids_sha256",
        "distance_matched_2x2_example_ids_sha256",
    ):
        value = cohort_audit.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise _preflight_semantic_error(f"cohort audit {key} must be a SHA-256")

    family_calibration = inference.get("family_calibration", {})
    expected_family_calibration = {
        "method": (
            "gene-cluster bootstrap endpoint-scale-normalized max-deviation interval"
        ),
        "interval_scaling": "fixed endpoint bootstrap SD; not nested bootstrap-t",
        "randomisation_unit": "gene",
        "n_components": len(PRIMARY_FAMILY_NAMES),
        "n_examples": TARGET_N,
        "n_genes": n_genes,
        "minimum_genes_required": MIN_GENES,
        "n_bootstrap_draws": PRODUCTION_RANDOMISATION,
        "n_sign_flip_draws": PRODUCTION_RANDOMISATION,
        "alpha": FAMILY_ALPHA,
        "sesoi": SESOI,
        "studentized": True,
        "pvalue": (
            "gene-level raw-cluster-sum sign flip; exact add-one Monte-Carlo "
            "full-family max-absolute-statistic"
        ),
    }
    for key, expected in expected_family_calibration.items():
        if family_calibration.get(key) != expected:
            raise _preflight_semantic_error(
                f"family calibration {key} must equal {expected!r}"
            )
    for key in ("simultaneous_half_widths", "bootstrap_scales"):
        values = family_calibration.get(key)
        if (
            not isinstance(values, dict)
            or set(values) != set(PRIMARY_FAMILY_NAMES)
            or any(not _is_finite_number(value) or value < 0 for value in values.values())
        ):
            raise _preflight_semantic_error(
                f"family calibration {key} must contain one finite nonnegative value per endpoint"
            )
    family_widths = family_calibration["simultaneous_half_widths"]
    family_max_width = max(family_widths.values())
    if (
        not _is_finite_number(family_calibration.get("simultaneous_critical_value"))
        or family_calibration["simultaneous_critical_value"] < 0
        or not _is_finite_number(family_calibration.get("simultaneous_half_width"))
        or not np.isclose(
            family_calibration["simultaneous_half_width"],
            family_max_width,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise _preflight_semantic_error(
            "family calibration must disclose a valid critical value and exact maximum width"
        )
    expected_family_resolution = family_max_width <= SESOI
    if family_calibration.get("resolution_ok") is not expected_family_resolution:
        raise _preflight_semantic_error(
            "family resolution_ok must equal the recorded maximum-width diagnostic"
        )
    if preflight.get("family_calibration") != family_calibration:
        raise _preflight_semantic_error(
            "top-level family calibration differs from boundary inference"
        )

    calibration = preflight.get("production_path_preflight", {})
    expected_path_calibration = {
        "randomisation_unit": "gene",
        "outer_trial_unit": "endpoint-centered 2x2 cell residual",
        "outer_trial_centering": "subtract each observed cell mean before sign flipping",
        "outer_trial_algebra": (
            "one shared sign per gene across Y_PD/Y_PN/Y_XD/Y_XN; derive S_P and I after flip"
        ),
        "translation_invariant": True,
        "classifier": (
            "exact one-sided boundary tests + Y_PN TOST + path IUT + Holm over exactly "
            "two named paths + strict simultaneous-CI gate on every LFC and plant trial"
        ),
        "classifier_schema": "gfmcd.r1c.boundary_classifier/1",
        "boundary_test_contract": boundary_test_contract(),
        "namespace": "r1c:production-path-preflight",
        "alpha": FAMILY_ALPHA,
        "sesoi": SESOI,
        "n_components": len(PRIMARY_FAMILY_NAMES),
        "n_examples": TARGET_N,
        "n_genes": n_genes,
        "minimum_genes_required": MIN_GENES,
        "n_calibration": PRODUCTION_RANDOMISATION,
        "n_trials": PRODUCTION_TRIALS,
        "reference_interval_calls": 1,
        "production_classifier_calls": PRODUCTION_TRIALS
        * (len(least_favourable_boundary_cases()) + 1),
        "studentized": True,
        "interval_scaling": "fixed endpoint bootstrap SD; not nested bootstrap-t",
        "planted_effect": None,
        "plant_delta": PLANT_DELTA,
        "cell_plant_vector": cell_plant_vector().tolist(),
        "plant_vector": structured_plant_vector().tolist(),
        "planted_requirements": _component_plant_requirements(),
        "decision_paths": decision_path_requirements(component_names=True),
        "path_holm_family": list(PATH_NAMES),
        "least_favourable_boundary_cases": least_favourable_boundary_cases(),
        "nuisance_continuum_control": nuisance_continuum_control(),
        "lfc_numeric_guard_toward_composite_null": LFC_NUMERIC_GUARD,
        "maximum_false_positives_allowed": MAX_FALSE_POSITIVES,
        "minimum_power_successes_required": MIN_POWER_SUCCESSES,
        "power_floor": PREFLIGHT_POWER_FLOOR,
        "binomial_bound": f"one-sided {100 * (1 - FAMILY_ALPHA):g}% Clopper-Pearson",
    }
    for key, expected in expected_path_calibration.items():
        if calibration.get(key) != expected:
            raise _preflight_semantic_error(
                f"production-path calibration {key} must equal {expected!r}"
            )
    widths = calibration.get("simultaneous_half_widths")
    if (
        not isinstance(widths, dict)
        or set(widths) != set(PRIMARY_FAMILY_NAMES)
        or any(not _is_finite_number(value) or value < 0 for value in widths.values())
    ):
        raise _preflight_semantic_error(
            "production-path widths must contain one finite nonnegative value per component"
        )
    path_max_width = max(widths.values())
    if (
        not _is_finite_number(calibration.get("simultaneous_half_width"))
        or not np.isclose(
            calibration["simultaneous_half_width"],
            path_max_width,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise _preflight_semantic_error(
            "production-path maximum width is inconsistent with its endpoint widths"
        )
    expected_path_resolution = path_max_width <= SESOI
    if calibration.get("resolution_ok") is not expected_path_resolution:
        raise _preflight_semantic_error(
            "production-path resolution_ok must equal the recorded maximum-width diagnostic"
        )
    expected_case_names = {
        path: {
            name
            for name, case in least_favourable_boundary_cases().items()
            if path in case["null_paths"]
        }
        for path in PATH_NAMES
    }
    false_positive_counts = calibration.get("boundary_false_positive_counts")
    worst_false_positives = calibration.get("worst_case_false_positives")
    plant_successes = calibration.get("plant_successes")
    if (
        not isinstance(false_positive_counts, dict)
        or set(false_positive_counts) != set(PATH_NAMES)
        or not isinstance(worst_false_positives, dict)
        or set(worst_false_positives) != set(PATH_NAMES)
        or not isinstance(plant_successes, dict)
        or set(plant_successes) != set(PATH_NAMES)
    ):
        raise _preflight_semantic_error("preflight path trial counts are incomplete")
    for path in PATH_NAMES:
        counts = false_positive_counts[path]
        if (
            not isinstance(counts, dict)
            or set(counts) != expected_case_names[path]
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= PRODUCTION_TRIALS
                for value in counts.values()
            )
        ):
            raise _preflight_semantic_error(
                f"{path} least-favourable false-positive counts are invalid"
            )
        expected_worst = max(counts.values())
        successes = plant_successes[path]
        if (
            worst_false_positives[path] != expected_worst
            or isinstance(successes, bool)
            or not isinstance(successes, int)
            or not 0 <= successes <= PRODUCTION_TRIALS
        ):
            raise _preflight_semantic_error(f"{path} trial counts are invalid")
        expected_fpr = expected_worst / PRODUCTION_TRIALS
        expected_fpr_upper = one_sided_binomial_bounds(
            expected_worst, PRODUCTION_TRIALS, alpha=FAMILY_ALPHA
        )[1]
        expected_power = successes / PRODUCTION_TRIALS
        expected_power_lower = one_sided_binomial_bounds(
            successes, PRODUCTION_TRIALS, alpha=FAMILY_ALPHA
        )[0]
        numeric = {
            "path_boundary_fpr": expected_fpr,
            "path_boundary_fpr_upper": expected_fpr_upper,
            "path_planted_power": expected_power,
            "path_planted_power_lower": expected_power_lower,
        }
        for field, expected in numeric.items():
            values = calibration.get(field)
            if (
                not isinstance(values, dict)
                or not _is_finite_number(values.get(path))
                or not np.isclose(values[path], expected, rtol=0.0, atol=1e-12)
            ):
                raise _preflight_semantic_error(
                    f"production-path calibration {field}.{path} is inconsistent"
                )
        if (
            expected_worst > MAX_FALSE_POSITIVES
            or expected_fpr_upper > FAMILY_ALPHA
            or calibration.get("path_boundary_fpr_ok", {}).get(path) is not True
        ):
            raise _preflight_semantic_error(
                f"{path} boundary false positives must be <= {MAX_FALSE_POSITIVES}"
            )
        if (
            successes < MIN_POWER_SUCCESSES
            or expected_power_lower < PREFLIGHT_POWER_FLOOR
            or calibration.get("path_planted_power_ok", {}).get(path) is not True
        ):
            raise _preflight_semantic_error(
                f"{path} planted successes must be >= {MIN_POWER_SUCCESSES}"
            )
    if plant_successes["stronger_downstream_preference"] > plant_successes[
        "core_paired_donor"
    ]:
        raise _preflight_semantic_error("nested stronger power cannot exceed core power")

    trial_records = calibration.get("trial_records")
    if not isinstance(trial_records, list) or len(trial_records) != PRODUCTION_TRIALS:
        raise _preflight_semantic_error("all production classifier trials must be serialized")
    if calibration.get("trial_records_sha256") != canonical_sha256(trial_records):
        raise _preflight_semantic_error("production classifier trial digest does not match")
    case_names = set(least_favourable_boundary_cases())
    seen_classifier_namespaces: set[str] = set()
    replayed_false_positive_counts = {
        path: {case_name: 0 for case_name in expected_case_names[path]}
        for path in PATH_NAMES
    }
    replayed_plant_successes = {path: 0 for path in PATH_NAMES}
    for trial_index, trial in enumerate(trial_records):
        if not isinstance(trial, dict) or trial.get("trial") != trial_index:
            raise _preflight_semantic_error("calibration trial ordering is not canonical")
        expected_outer_namespace = (
            f"r1c:production-path-preflight:centered-primitive-null:{trial_index}"
        )
        if trial.get("outer_sign_namespace") != expected_outer_namespace:
            raise _preflight_semantic_error(
                "calibration trial lacks its exact outer-sign namespace"
            )
        case_audits = trial.get("least_favourable_cases")
        if not isinstance(case_audits, dict) or set(case_audits) != case_names:
            raise _preflight_semantic_error(
                "calibration trial lacks every least-favourable classifier audit"
            )
        audits = [
            (
                case_audits[case_name],
                f"r1c:production-path-preflight:lfc:{case_name}:{trial_index}",
            )
            for case_name in sorted(case_names)
        ]
        audits.append(
            (
                trial.get("coherent_plant"),
                f"r1c:production-path-preflight:coherent-plant:{trial_index}",
            )
        )
        for audit, classifier_namespace in audits:
            if not isinstance(audit, dict) or audit.get("namespace") != classifier_namespace:
                raise _preflight_semantic_error(
                    "calibration classifier namespace is missing or non-canonical"
                )
            if classifier_namespace in seen_classifier_namespaces:
                raise _preflight_semantic_error(
                    "calibration classifier namespaces must be unique"
                )
            seen_classifier_namespaces.add(classifier_namespace)
            trial_boundary_tests = audit.get("boundary_tests")
            if (
                not isinstance(trial_boundary_tests, dict)
                or set(trial_boundary_tests) != expected_test_names
            ):
                raise _preflight_semantic_error(
                    "calibration classifier lacks all five boundary tests"
                )
            trial_sign_digests: set[str] = set()
            trial_gene_digests: set[str] = set()
            expected_boundary_namespace = (
                f"{classifier_namespace}:shared-boundary-signs"
            )
            for test_name, test in trial_boundary_tests.items():
                exceedances = test.get("exceedances")
                B = test.get("B")
                expected_boundary, expected_alternative = expected_boundary_specs[
                    test_name
                ]
                if (
                    isinstance(exceedances, bool)
                    or not isinstance(exceedances, int)
                    or B != PRODUCTION_RANDOMISATION
                    or not 0 <= exceedances <= B
                    or test.get("seed") != PRODUCTION_SEED
                    or test.get("namespace") != expected_boundary_namespace
                    or not _is_finite_number(test.get("boundary"))
                    or not np.isclose(
                        test["boundary"],
                        expected_boundary,
                        rtol=0.0,
                        atol=1e-15,
                    )
                    or test.get("alternative") != expected_alternative
                    or test.get("n_examples") != TARGET_N
                    or test.get("n_genes") != n_genes
                    or test.get("randomisation_unit") != "gene"
                    or test.get("pvalue_rule") != BOUNDARY_PVALUE_RULE
                    or not _is_finite_number(test.get("comparison_tolerance"))
                    or test["comparison_tolerance"] <= 0
                    or not _is_finite_number(test.get("p_value"))
                    or not np.isclose(
                        test["p_value"],
                        (1 + exceedances) / (B + 1),
                        rtol=0.0,
                        atol=1e-15,
                    )
                ):
                    raise _preflight_semantic_error(
                        "calibration boundary-test replay metadata is inconsistent"
                    )
                for digest_key in (
                    "null_draws_sha256",
                    "shared_null_draws_sha256",
                    "sign_matrix_sha256",
                    "ordered_gene_labels_sha256",
                ):
                    if not isinstance(test.get(digest_key), str) or not re.fullmatch(
                        r"[0-9a-f]{64}", test[digest_key]
                    ):
                        raise _preflight_semantic_error(
                            "calibration boundary-test digest is missing"
                        )
                trial_sign_digests.add(test["sign_matrix_sha256"])
                trial_gene_digests.add(test["ordered_gene_labels_sha256"])
            expected_shared = {
                "namespace": expected_boundary_namespace,
                "sign_matrix_sha256": next(iter(trial_sign_digests)),
                "ordered_gene_labels_sha256": next(iter(trial_gene_digests)),
                "all_five_boundary_tests_share_sign_matrix": True,
            }
            if (
                len(trial_sign_digests) != 1
                or len(trial_gene_digests) != 1
                or audit.get("shared_boundary_randomisation") != expected_shared
                or not isinstance(audit.get("holm_adjusted_path_p_values"), dict)
                or set(audit["holm_adjusted_path_p_values"]) != set(PATH_NAMES)
                or any(
                    not _is_finite_number(value) or not 0 <= value <= 1
                    for value in audit["holm_adjusted_path_p_values"].values()
                )
            ):
                raise _preflight_semantic_error(
                    "calibration classifier does not preserve shared signs and exact Holm family"
                )
        for case_name, audit in case_audits.items():
            declared = _rederive_trial_classifier_decisions(audit)
            for path in least_favourable_boundary_cases()[case_name]["null_paths"]:
                replayed_false_positive_counts[path][case_name] += int(
                    declared[path]
                )
        planted_declared = _rederive_trial_classifier_decisions(
            trial["coherent_plant"]
        )
        for path in PATH_NAMES:
            replayed_plant_successes[path] += int(planted_declared[path])

    if replayed_false_positive_counts != false_positive_counts:
        raise _preflight_semantic_error(
            "boundary false-positive counts do not reproduce from trial decisions"
        )
    if replayed_plant_successes != plant_successes:
        raise _preflight_semantic_error(
            "plant success counts do not reproduce from trial decisions"
        )

    structured = preflight.get("structured_plant")
    expected_structured = {
        "delta": PLANT_DELTA,
        "cell_vector": cell_plant_vector().tolist(),
        "vector": structured_plant_vector().tolist(),
        "requirements": {
            PRIMARY_FAMILY_NAMES[index]: label
            for index, label in PRIMARY_PLANT_REQUIREMENTS.items()
        },
    }
    if structured != expected_structured:
        raise _preflight_semantic_error("the named structured plant differs from the frozen design")

    instrument = preflight.get("instrument_fingerprint")
    if not isinstance(instrument, dict):
        raise _preflight_semantic_error("the full instrument fingerprint record is missing")
    if instrument.get("contract") != frozen_instrument_contract():
        raise _preflight_semantic_error("the embedded frozen instrument contract differs")
    runtime_contract = instrument.get("runtime_contract")
    if not isinstance(runtime_contract, dict) or set(runtime_contract) != {
        "n",
        "batch",
        "device",
        "seed",
        "n_randomisation",
        "n_trials",
        "robust_bootstrap_draws",
        "health_probes",
        "strict_runtime_policy",
        "r5_stack_fingerprint_sha256",
    }:
        raise _preflight_semantic_error("the runtime contract is incomplete or extended")
    expected_runtime = {
        "n": TARGET_N,
        "batch": PRODUCTION_BATCH,
        "seed": PRODUCTION_SEED,
        "n_randomisation": PRODUCTION_RANDOMISATION,
        "n_trials": PRODUCTION_TRIALS,
        "robust_bootstrap_draws": PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
        "health_probes": PRODUCTION_HEALTH_PROBES,
    }
    if any(runtime_contract.get(key) != value for key, value in expected_runtime.items()):
        raise _preflight_semantic_error("the runtime contract contains production-parameter drift")
    if (
        runtime_contract.get("device") != "mps"
        or runtime_contract.get("strict_runtime_policy")
        != authorize_production_runtime("mps", environ={})
        or runtime_contract.get("r5_stack_fingerprint_sha256")
        != preflight.get(STACK_FINGERPRINT_FIELD)
    ):
        raise _preflight_semantic_error(
            "the runtime contract does not bind strict MPS and the exact R5 stack"
        )

    observed = instrument.get("fingerprint_sha256")
    if observed != fingerprint_sha256:
        raise TargetSealError("preflight instrument fingerprint does not match this target runner")


def read_snapshotted_bytes(
    path: str | Path,
    snapshot: dict[str, dict[str, object]],
) -> bytes:
    resolved = resolve_input(path)
    expected = snapshot_record(snapshot, resolved)
    data = resolved.read_bytes()
    if (
        len(data) != expected["size_bytes"]
        or hashlib.sha256(data).hexdigest() != expected["sha256"]
    ):
        raise TargetSealError(f"snapshotted input changed before it was read: {resolved}")
    return data


def read_snapshotted_json(
    path: str | Path,
    snapshot: dict[str, dict[str, object]],
) -> dict:
    resolved = resolve_input(path)
    data = read_snapshotted_bytes(resolved, snapshot)
    try:
        return json.loads(data)
    except json.JSONDecodeError as error:
        raise TargetSealError(f"invalid snapshotted JSON: {resolved}: {error}") from error


def verify_preflight_manifest(
    preflight_path: str | Path,
    snapshot: dict[str, dict[str, object]] | None = None,
) -> dict:
    """Verify that the sibling manifest records the exact immutable preflight bytes."""
    artifact = resolve_input(preflight_path)
    manifest_path = artifact.with_suffix(".manifest.json")
    if not artifact.is_file() or not manifest_path.is_file():
        raise TargetSealError("target requires the preflight artifact and its sibling manifest")
    bound = snapshot or snapshot_input_paths([artifact, manifest_path])
    manifest = read_snapshotted_json(manifest_path, bound)
    if manifest.get("experiment_id") != "splicebert_acceptor_coupling:r1c_reverse_behavior":
        raise TargetSealError("preflight manifest belongs to a different experiment")
    artifact_record = snapshot_record(bound, artifact)
    artifact_identity = {
        "logical_name": artifact.name,
        "size_bytes": artifact_record["size_bytes"],
        "sha256": artifact_record["sha256"],
    }
    matching = [
        record
        for record in manifest.get("outputs", [])
        if Path(str(record.get("path", ""))).name == artifact.name
        and record.get("size_bytes") == artifact_identity["size_bytes"]
        and record.get("sha256") == artifact_identity["sha256"]
    ]
    if len(matching) != 1:
        raise TargetSealError("preflight artifact does not match its immutable manifest output hash")
    verify_input_snapshot(bound)
    return {
        "artifact": artifact_identity,
        "manifest": {
            "logical_name": manifest_path.name,
            **snapshot_record(bound, manifest_path),
        },
    }


def _target_package_error(message: str) -> TargetSealError:
    return TargetSealError(f"R1c target package semantic mismatch: {message}")


def _published_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise _target_package_error(f"publication is not a regular file: {resolved}")
    return {
        "logical_name": resolved.name,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _target_package_error(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise _target_package_error(f"{label} must contain one JSON object")
    return value


def _require_package_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise _target_package_error(f"{label} must be a lowercase SHA-256")
    return value


def _manifest_logical_inputs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_inputs = manifest.get("inputs")
    if not isinstance(raw_inputs, list):
        raise _target_package_error("manifest inputs are missing")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw_inputs):
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "location_scope",
            "size_bytes",
            "sha256",
        }:
            raise _target_package_error(
                f"manifest input {index} is not one relocation-safe identity"
            )
        logical_name = item.get("path")
        location_scope = item.get("location_scope")
        size = item.get("size_bytes")
        if (
            not isinstance(logical_name, str)
            or not logical_name
            or location_scope not in {"repository", "external"}
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
        ):
            raise _target_package_error(f"manifest input {index} is malformed")
        records.append(
            {
                "logical_name": logical_name,
                "location_scope": location_scope,
                "size_bytes": size,
                "sha256": _require_package_sha256(
                    f"manifest input {index}", item.get("sha256")
                ),
            }
        )
    return sorted(
        records,
        key=lambda item: (
            str(item["location_scope"]),
            str(item["logical_name"]),
            str(item["sha256"]),
        ),
    )


def _source_identity_in_manifest(
    source: Mapping[str, Any], manifest_inputs: Sequence[Mapping[str, Any]]
) -> bool:
    required = {
        "logical_name": source.get("logical_name"),
        "location_scope": source.get("location_scope"),
        "size_bytes": source.get("size_bytes"),
        "sha256": source.get("sha256"),
    }
    return any(dict(item) == required for item in manifest_inputs)


def verify_successor_preflight_package(
    artifact_path: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """Load and authenticate the outcome-blindly constructed 400-row val handoff."""

    artifact_resolved = Path(artifact_path).expanduser().resolve(strict=True)
    manifest_resolved = Path(manifest_path).expanduser().resolve(strict=True)
    if manifest_resolved != artifact_resolved.with_suffix(".manifest.json"):
        raise _target_package_error(
            "preflight manifest is not the artifact's canonical sibling"
        )
    package = verify_preflight_manifest(artifact_resolved)
    if package["manifest"] != _published_identity(manifest_resolved):
        raise _target_package_error("preflight manifest identity differs")
    record = _read_json_object(artifact_resolved, label="preflight artifact")
    instrument = record.get("instrument_fingerprint")
    if not isinstance(instrument, Mapping):
        raise _target_package_error("preflight instrument fingerprint is missing")
    fingerprint_sha256 = instrument.get("fingerprint_sha256")
    verify_preflight_binding(record, str(fingerprint_sha256))

    if (
        _require_package_sha256(
            "preflight instrument fingerprint", fingerprint_sha256
        )
        != canonical_sha256(
            {
                key: value
                for key, value in instrument.items()
                if key != "fingerprint_sha256"
            }
        )
    ):
        raise _target_package_error("preflight instrument fingerprint does not replay")
    source_fingerprint = instrument.get("source_fingerprint_sha256")
    source_record = {
        key: value
        for key, value in instrument.items()
        if key
        not in {
            "source_fingerprint_sha256",
            "runtime_contract",
            "fingerprint_sha256",
        }
    }
    if (
        _require_package_sha256(
            "preflight source fingerprint", source_fingerprint
        )
        != canonical_sha256(source_record)
    ):
        raise _target_package_error("preflight source fingerprint does not replay")

    manifest = _read_json_object(manifest_resolved, label="preflight manifest")
    manifest_inputs = _manifest_logical_inputs(manifest)
    if canonical_sha256(manifest_inputs) != record.get("input_snapshot_sha256"):
        raise _target_package_error(
            "preflight input snapshot does not replay from manifest"
        )
    for name in ("gtf", "fasta", "fasta_fai", "protocol", "statistics"):
        source = instrument.get(name)
        if (
            not isinstance(source, Mapping)
            or not _source_identity_in_manifest(source, manifest_inputs)
        ):
            raise _target_package_error(
                f"preflight manifest omits exact {name} source bytes"
            )
    dependencies = instrument.get("measurement_dependencies")
    if not isinstance(dependencies, list) or any(
        not isinstance(source, Mapping)
        or not _source_identity_in_manifest(source, manifest_inputs)
        for source in dependencies
    ):
        raise _target_package_error(
            "preflight manifest omits measurement dependency bytes"
        )
    checkpoint = instrument.get("checkpoint")
    checkpoint_files = (
        checkpoint.get("files") if isinstance(checkpoint, Mapping) else None
    )
    if not isinstance(checkpoint_files, list) or not checkpoint_files:
        raise _target_package_error("preflight checkpoint inventory is missing")
    for checkpoint_file in checkpoint_files:
        if not isinstance(checkpoint_file, Mapping) or not any(
            item["size_bytes"] == checkpoint_file.get("size_bytes")
            and item["sha256"] == checkpoint_file.get("sha256")
            and Path(str(item["logical_name"])).name
            == Path(str(checkpoint_file.get("relative_path", ""))).name
            for item in manifest_inputs
        ):
            raise _target_package_error(
                "preflight manifest omits a checkpoint inventory member"
            )
    authorization = record.get("target_authorization")
    production_authorization = (
        authorization.get("production_authorization")
        if isinstance(authorization, Mapping)
        else None
    )
    if (
        record.get("holdout_scope_disclosure") != holdout_scope_disclosure()
        or
        not isinstance(authorization, Mapping)
        or authorization.get("required") is not False
        or authorization.get("pre_run_clean_worktree") is not True
        or authorization.get("strict_runtime_policy")
        != authorize_production_runtime("mps", environ={})
        or authorization.get("model_loaded_from_private_staged_copy") is not True
        or not isinstance(production_authorization, Mapping)
        or production_authorization.get("stage") != "preflight"
        or production_authorization.get("protocol_sha256")
        != record.get("protocol_sha256")
        or production_authorization.get("r5_stack_fingerprint_sha256")
        != record.get(STACK_FINGERPRINT_FIELD)
    ):
        raise _target_package_error(
            "preflight clean strict-MPS authorization does not replay"
        )

    attestation = {
        "schema": "gfmcd.r1c.verified_preflight_handoff/1",
        "artifact": package["artifact"],
        "manifest": package["manifest"],
        "instrument_fingerprint_sha256": fingerprint_sha256,
        STACK_FINGERPRINT_FIELD: record[STACK_FINGERPRINT_FIELD],
        "n": record["n"],
        "n_genes": record["n_genes"],
        "target_cohort_identity": record["target_cohort_identity"],
        "verdict": record["verdict"],
        "all_gates_pass": record["all_gates_pass"],
        "gates": record["gates"],
    }
    return {
        "record": record,
        "attestation": attestation
        | {"attestation_sha256": canonical_sha256(attestation)},
    }


def verify_successor_target_package(
    artifact_path: str | Path,
    manifest_path: str | Path,
    attempt_path: str | Path,
    completion_path: str | Path,
    *,
    preflight_artifact_path: str | Path,
) -> dict[str, Any]:
    """Authenticate and semantically replay a published R1c successor target.

    This verifier is intentionally model- and genome-free.  It treats the serialized
    margin levels as sufficient statistics, reconstructs the four primitive cells,
    reruns the exact boundary classifier and both equal-gene robustness estimators, and
    then authenticates every publication and seal crosslink.  Downstream R5 entry may
    consume only the returned handoff (or perform the same replay independently).
    """

    paths = {
        "artifact": Path(artifact_path).expanduser().resolve(strict=True),
        "manifest": Path(manifest_path).expanduser().resolve(strict=True),
        "attempt": Path(attempt_path).expanduser().resolve(strict=True),
        "completion": Path(completion_path).expanduser().resolve(strict=True),
    }
    identities = {name: _published_identity(path) for name, path in paths.items()}
    artifact = _read_json_object(paths["artifact"], label="target artifact")
    manifest = _read_json_object(paths["manifest"], label="target manifest")
    attempt = _read_json_object(paths["attempt"], label="target attempt")
    completion = _read_json_object(paths["completion"], label="target completion")

    if manifest.get("experiment_id") != (
        "splicebert_acceptor_coupling:r1c_reverse_behavior"
    ):
        raise _target_package_error("manifest belongs to a different experiment")
    matching_outputs = [
        item
        for item in manifest.get("outputs", [])
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).name
        == identities["artifact"]["logical_name"]
        and item.get("size_bytes") == identities["artifact"]["size_bytes"]
        and item.get("sha256") == identities["artifact"]["sha256"]
    ]
    if len(matching_outputs) != 1:
        raise _target_package_error(
            "manifest does not bind exactly the supplied artifact bytes"
        )

    expected_attempt_fields = {
        "schema",
        "created_at_utc",
        "status",
        "biological_target_key",
        "target_identity",
        "method_context",
        "policy",
        "sealed_cohort_constructed_before_record",
        "model_loaded_before_record",
        "output_inspected_before_record",
        "attempt_id",
    }
    if (
        set(attempt) != expected_attempt_fields
        or attempt.get("schema") != R1C_RECEIPT_SCHEMA
        or attempt.get("status") != "AUTHORIZED_SEALED_OUTCOMES_MAY_BE_SPENT"
        or attempt.get("sealed_cohort_constructed_before_record") is not False
        or attempt.get("model_loaded_before_record") is not False
        or attempt.get("output_inspected_before_record") is not False
        or attempt.get("attempt_id")
        != canonical_sha256(
            {key: value for key, value in attempt.items() if key != "attempt_id"}
        )
    ):
        raise _target_package_error("attempt receipt does not replay exactly")

    target_identity = attempt.get("target_identity")
    if not isinstance(target_identity, Mapping):
        raise _target_package_error("attempt lacks its biological target identity")
    try:
        expected_target_identity = build_r1c_biological_target_identity(
            checkpoint_sha256=str(target_identity.get("checkpoint_sha256")),
            gtf_sha256=str(target_identity.get("gtf_sha256")),
            fasta_sha256=str(target_identity.get("fasta_sha256")),
            fasta_fai_sha256=str(target_identity.get("fasta_fai_sha256")),
        )
    except ValueError as error:
        raise _target_package_error("biological target identity is malformed") from error
    if (
        dict(target_identity) != expected_target_identity
        or target_identity.get("schema") != R1C_IDENTITY_SCHEMA
        or target_identity.get("program") != R1C_PROGRAM
        or target_identity.get("holdout_id") != TARGET_HOLDOUT_ID
        or target_identity.get("target_chromosomes")
        != sorted(SEALED_CHROMOSOMES)
        or target_identity.get("target_n") != TARGET_N
        or attempt.get("biological_target_key")
        != target_identity.get("biological_target_key")
    ):
        raise _target_package_error("biological target identity does not replay")

    attempt_package = identities["attempt"] | {
        "attempt_id": attempt["attempt_id"],
        "biological_target_key": attempt["biological_target_key"],
        "status": attempt["status"],
    }
    if artifact.get("target_attempt") != attempt_package:
        raise _target_package_error("artifact does not embed the exact attempt identity")

    expected_completion_fields = {
        "schema",
        "created_at_utc",
        "status",
        "attempt_id",
        "biological_target_key",
        "method_fingerprint_sha256",
        "r5_stack_fingerprint_sha256",
        "cohort_digests",
        "artifact",
        "manifest",
        "attempt",
        "official_verdict",
        "all_gates_pass",
        "gates",
        "completion_id",
    }
    if (
        set(completion) != expected_completion_fields
        or completion.get("schema") != R1C_COMPLETION_SCHEMA
        or completion.get("status") != "TARGET_PUBLICATION_COMPLETE"
        or completion.get("completion_id")
        != canonical_sha256(
            {
                key: value
                for key, value in completion.items()
                if key != "completion_id"
            }
        )
        or completion.get("attempt_id") != attempt.get("attempt_id")
        or completion.get("biological_target_key")
        != attempt.get("biological_target_key")
        or completion.get("artifact") != identities["artifact"]
        or completion.get("manifest") != identities["manifest"]
        or completion.get("attempt") != identities["attempt"]
    ):
        raise _target_package_error("completion does not authenticate the publication")
    completion_declaration = artifact.get("target_completion")
    if completion_declaration != {
        "logical_name": identities["completion"]["logical_name"]
    }:
        raise _target_package_error("artifact completion declaration differs")

    header = {
        "schema": "gfmcd.r1c.reverse_behavior/3",
        "experiment": "R1c_corrected_reverse_behavior",
        "protocol_status": "FROZEN_HASH_PINNED",
        "stage": "target",
        "status": "CONFIRMATORY",
        "evidence_rung": "BEHAVIOR_ONLY",
        "split": "test",
        "sealed_split_opened": True,
        "n": TARGET_N,
    }
    for name, expected in header.items():
        if artifact.get(name) != expected:
            raise _target_package_error(f"artifact {name} must equal {expected!r}")
    if artifact.get("holdout_scope_disclosure") != holdout_scope_disclosure():
        raise _target_package_error("target holdout scope disclosure differs")
    if artifact.get("config") != _preflight_config_contract():
        raise _target_package_error("target production config differs")
    if artifact.get("preconstruction_allowed_chromosomes") != sorted(
        SEALED_CHROMOSOMES
    ):
        raise _target_package_error("target chromosome construction allowlist differs")
    expected_freeze = {
        "protocol_declares_frozen": True,
        "protocol_hash_pinned": True,
        "instrument_fingerprint_pinned": True,
        "downstream_r5_stack_pinned": True,
        "strict_mps_runtime_verified": True,
        "authorization_ready": True,
    }
    if artifact.get("freeze_authorization") != expected_freeze:
        raise _target_package_error("target freeze authorization is incomplete")

    preflight_package = verify_preflight_manifest(preflight_artifact_path)
    if artifact.get("preflight_package") != preflight_package:
        raise _target_package_error("reviewed preflight link differs")

    try:
        validate_serialized_r5_stack(artifact)
    except R5StackBindingError as error:
        raise _target_package_error("serialized R5 stack differs from current code") from error
    stack_sha256 = artifact.get(STACK_FINGERPRINT_FIELD)
    _require_package_sha256("R5 stack fingerprint", stack_sha256)

    instrument = artifact.get("instrument_fingerprint")
    if not isinstance(instrument, Mapping):
        raise _target_package_error("instrument fingerprint is missing")
    instrument_sha256 = instrument.get("fingerprint_sha256")
    if (
        _require_package_sha256("instrument fingerprint", instrument_sha256)
        != canonical_sha256(
            {
                key: value
                for key, value in instrument.items()
                if key != "fingerprint_sha256"
            }
        )
    ):
        raise _target_package_error("instrument fingerprint does not replay")
    source_fingerprint_sha256 = instrument.get("source_fingerprint_sha256")
    source_record = {
        key: value
        for key, value in instrument.items()
        if key
        not in {
            "source_fingerprint_sha256",
            "runtime_contract",
            "fingerprint_sha256",
        }
    }
    if (
        _require_package_sha256(
            "source fingerprint", source_fingerprint_sha256
        )
        != canonical_sha256(source_record)
        or instrument.get("contract") != frozen_instrument_contract()
    ):
        raise _target_package_error("instrument source or method contract differs")
    runtime_contract = instrument.get("runtime_contract")
    expected_runtime = {
        "n": TARGET_N,
        "batch": PRODUCTION_BATCH,
        "device": "mps",
        "seed": PRODUCTION_SEED,
        "n_randomisation": PRODUCTION_RANDOMISATION,
        "n_trials": PRODUCTION_TRIALS,
        "robust_bootstrap_draws": PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
        "health_probes": PRODUCTION_HEALTH_PROBES,
        "strict_runtime_policy": authorize_production_runtime("mps", environ={}),
        "r5_stack_fingerprint_sha256": stack_sha256,
    }
    if runtime_contract != expected_runtime:
        raise _target_package_error("target runtime contract is not exact strict MPS")

    for name in ("gtf", "fasta", "fasta_fai", "protocol", "statistics"):
        source = instrument.get(name)
        if not isinstance(source, Mapping):
            raise _target_package_error(f"instrument lacks {name} source identity")
        _require_package_sha256(f"instrument {name}", source.get("sha256"))
    dependencies = instrument.get("measurement_dependencies")
    if not isinstance(dependencies, list) or any(
        not isinstance(item, Mapping) for item in dependencies
    ):
        raise _target_package_error("measurement dependency identities are missing")
    checkpoint = instrument.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise _target_package_error("checkpoint bundle identity is missing")
    checkpoint_sha256 = _require_package_sha256(
        "checkpoint bundle", checkpoint.get("bundle_sha256")
    )
    if (
        target_identity.get("checkpoint_sha256") != checkpoint_sha256
        or target_identity.get("gtf_sha256") != instrument["gtf"].get("sha256")
        or target_identity.get("fasta_sha256")
        != instrument["fasta"].get("sha256")
        or target_identity.get("fasta_fai_sha256")
        != instrument["fasta_fai"].get("sha256")
    ):
        raise _target_package_error("receipt source identities differ from instrument")

    context = attempt.get("method_context")
    expected_context_fields = {
        "protocol_sha256",
        "instrument_fingerprint_sha256",
        "r5_stack_fingerprint_sha256",
        "checkpoint_bundle_sha256",
        "construction_contract_sha256",
        "endpoint_policy",
        "preflight_artifact",
        "preflight_manifest",
        "input_snapshot_sha256",
        "requested_artifact_logical_name",
        "requested_manifest_logical_name",
        "declared_completion_logical_name",
    }
    if not isinstance(context, Mapping) or set(context) != expected_context_fields:
        raise _target_package_error("attempt method context fields differ")
    if (
        context.get("protocol_sha256") != artifact.get("protocol_sha256")
        or context.get("protocol_sha256") != instrument["protocol"].get("sha256")
        or context.get("instrument_fingerprint_sha256") != instrument_sha256
        or context.get("r5_stack_fingerprint_sha256") != stack_sha256
        or context.get("checkpoint_bundle_sha256") != checkpoint_sha256
        or context.get("construction_contract_sha256")
        != construction_contract_sha256()
        or context.get("endpoint_policy") != TARGET_ENDPOINT_POLICY
        or context.get("requested_artifact_logical_name")
        != identities["artifact"]["logical_name"]
        or context.get("requested_manifest_logical_name")
        != identities["manifest"]["logical_name"]
        or context.get("declared_completion_logical_name")
        != identities["completion"]["logical_name"]
    ):
        raise _target_package_error("attempt method context does not bind this target")

    if (
        context.get("preflight_artifact") != preflight_package["artifact"]
        or context.get("preflight_manifest") != preflight_package["manifest"]
        or Path(str(artifact.get("preflight_artifact", ""))).name
        != preflight_package["artifact"]["logical_name"]
    ):
        raise _target_package_error("reviewed preflight link differs")
    preflight = _read_json_object(
        preflight_artifact_path, label="preflight artifact"
    )
    verify_preflight_binding(preflight, str(instrument_sha256))
    if (
        preflight.get("instrument_fingerprint") != instrument
        or preflight.get(STACK_BINDING_FIELD) != artifact.get(STACK_BINDING_FIELD)
        or preflight.get(STACK_FINGERPRINT_FIELD) != stack_sha256
        or preflight.get("protocol_sha256") != artifact.get("protocol_sha256")
    ):
        raise _target_package_error("target differs from its exact preflight method")

    manifest_inputs = _manifest_logical_inputs(manifest)
    if canonical_sha256(manifest_inputs) != artifact.get("input_snapshot_sha256"):
        raise _target_package_error("target input snapshot does not replay from manifest")
    pre_attempt_inputs = [
        item
        for item in manifest_inputs
        if not (
            Path(str(item["logical_name"])).name
            == identities["attempt"]["logical_name"]
            and item["size_bytes"] == identities["attempt"]["size_bytes"]
            and item["sha256"] == identities["attempt"]["sha256"]
        )
    ]
    if (
        len(pre_attempt_inputs) != len(manifest_inputs) - 1
        or canonical_sha256(pre_attempt_inputs)
        != context.get("input_snapshot_sha256")
    ):
        raise _target_package_error("receipt was not bound after the pre-read snapshot")
    for name in ("gtf", "fasta", "fasta_fai", "protocol", "statistics"):
        if not _source_identity_in_manifest(instrument[name], manifest_inputs):
            raise _target_package_error(f"manifest omits exact {name} source bytes")
    for dependency in dependencies:
        if not _source_identity_in_manifest(dependency, manifest_inputs):
            raise _target_package_error(
                "manifest omits an exact measurement dependency"
            )
    checkpoint_files = checkpoint.get("files")
    if not isinstance(checkpoint_files, list) or not checkpoint_files:
        raise _target_package_error("checkpoint file inventory is missing")
    for checkpoint_file in checkpoint_files:
        if not isinstance(checkpoint_file, Mapping) or not any(
            item["size_bytes"] == checkpoint_file.get("size_bytes")
            and item["sha256"] == checkpoint_file.get("sha256")
            and Path(str(item["logical_name"])).name
            == Path(str(checkpoint_file.get("relative_path", ""))).name
            for item in manifest_inputs
        ):
            raise _target_package_error("manifest omits a checkpoint inventory member")

    replay = replay_preflight_rows(artifact, stage="target")
    if (
        artifact.get("n_genes") != replay["n_genes"]
        or artifact.get("cohort_sha256") != replay["cohort_sha256"]
        or artifact.get("target_cohort_identity")
        != replay["target_cohort_identity"]
        or artifact.get("family_decisions")
        != replay["inference"]["family_decisions"]
        or artifact.get("family_calibration")
        != replay["inference"]["family_calibration"]
        or artifact.get("observed_decision_paths")
        != observed_decision_paths(replay["inference"])
    ):
        raise _target_package_error("target top-level inference or cohort is stale")
    cohort_audit = artifact.get("cohort_audit")
    if (
        not isinstance(cohort_audit, Mapping)
        or cohort_audit.get("selected") != TARGET_N
        or cohort_audit.get("selected_genes") != replay["n_genes"]
        or cohort_audit.get("selected_example_ids_sha256")
        != replay["selected_example_ids_sha256"]
        or cohort_audit.get("selected_cohort_sha256") != replay["cohort_sha256"]
        or cohort_audit.get("selection_reads_model_outputs") is not False
    ):
        raise _target_package_error("target cohort audit does not replay")

    expected_gates = {
        **replay["row_gates"],
        "zero_GT_to_AC_planted_motifs": True,
        "passing_exact_bound_preflight": True,
    }
    gates = artifact.get("gates")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != set(TARGET_GATE_NAMES)
        or dict(gates) != expected_gates
    ):
        raise _target_package_error("target gates do not replay")
    all_gates_pass = all(expected_gates.values())
    instrument_gates = {
        name: value
        for name, value in expected_gates.items()
        if name != "raw_trimmed_gene_robustness"
    }
    verdict = production_verdict(
        replay["inference"], instrument_gates, "target", replay["robustness"]
    )
    if (
        artifact.get("all_gates_pass") is not all_gates_pass
        or artifact.get("verdict") != verdict
        or completion.get("cohort_digests")
        != replay["target_cohort_identity"]
        or completion.get("method_fingerprint_sha256") != instrument_sha256
        or completion.get("r5_stack_fingerprint_sha256") != stack_sha256
        or completion.get("official_verdict") != verdict
        or completion.get("all_gates_pass") is not all_gates_pass
        or completion.get("gates") != expected_gates
    ):
        raise _target_package_error("verdict, gates, or completion is stale")

    authorization = artifact.get("target_authorization")
    production_authorization = (
        authorization.get("production_authorization")
        if isinstance(authorization, Mapping)
        else None
    )
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("required") is not True
        or authorization.get("pre_run_clean_worktree") is not True
        or authorization.get("explicit_unseal_acknowledgement") is not True
        or authorization.get("strict_runtime_policy")
        != authorize_production_runtime("mps", environ={})
        or authorization.get("model_loaded_from_private_staged_copy") is not True
        or authorization.get("preflight_artifact_hash_pin_required") is not True
        or authorization.get("preflight_artifact_hash_pin_matched") is not True
        or not isinstance(production_authorization, Mapping)
        or production_authorization.get("stage") != "target"
        or production_authorization.get("protocol_sha256")
        != artifact.get("protocol_sha256")
        or production_authorization.get("r5_stack_fingerprint_sha256")
        != stack_sha256
        or production_authorization.get("explicit_unseal_acknowledgement") is not True
    ):
        raise _target_package_error("target authorization does not replay")

    handoff = {
        "schema": "gfmcd.r1c.verified_target_handoff/1",
        "attempt_id": attempt["attempt_id"],
        "completion_id": completion["completion_id"],
        "biological_target_key": attempt["biological_target_key"],
        "instrument_fingerprint_sha256": instrument_sha256,
        STACK_FINGERPRINT_FIELD: stack_sha256,
        "target_cohort_identity": replay["target_cohort_identity"],
        "observed_decision_paths": observed_decision_paths(replay["inference"]),
        "verdict": verdict,
        "all_gates_pass": all_gates_pass,
        "gates": expected_gates,
        "package_identities": identities,
        "preflight_package": preflight_package,
    }
    return handoff | {"attestation_sha256": canonical_sha256(handoff)}


def stage_split(stage: str) -> str:
    if stage in {"smoke", "preflight"}:
        return "val"
    if stage == "target":
        return "test"
    raise ValueError(f"unknown R1c stage: {stage!r}")


def assert_stage_chromosomes(examples: Sequence, stage: str) -> None:
    chromosomes = {_normalise_chromosome(example.chrom) for example in examples}
    if stage in {"smoke", "preflight"} and (
        not chromosomes or not chromosomes <= set(VALIDATION_CHROMOSOMES)
    ):
        raise TargetSealError(
            "R1c validation stages must contain only chr11/16/19; sealed chr8/9 "
            "or any undeclared chromosome is forbidden"
        )
    if stage == "target" and (not chromosomes or not chromosomes <= SEALED_CHROMOSOMES):
        raise TargetSealError("R1c target must contain only sealed chr8/9 examples")


@dataclass(frozen=True)
class CohortRecord:
    example: object
    matched_decoy_donor_pos: int
    matched_decoy_distance_mismatch: int
    dyad_distance_mismatch: int


def serialize_structural_rows(records: Sequence[CohortRecord]) -> list[dict]:
    return [
        {
            "example_id": record.example.example_id,
            "gene_id": record.example.gene_id,
            "chrom": _normalise_chromosome(record.example.chrom),
            "strand": record.example.strand,
            "positions": {
                "prior_real_acceptor": int(record.example.acceptor_pos),
                "paired_donor": int(record.example.donor_pos),
                "partner_acceptor": int(record.example.next_acceptor_pos),
                "matched_decoy_donor": record.matched_decoy_donor_pos,
                "matched_decoy_donor_distance_mismatch": (
                    record.matched_decoy_distance_mismatch
                ),
                "dyad_distance_mismatch": record.dyad_distance_mismatch,
            },
        }
        for record in records
    ]


def _control_record(example) -> tuple[CohortRecord | None, str | None]:
    """Build outcome-blind comparator coordinates, or return a disclosed exclusion reason."""
    partner = example.next_acceptor_pos
    if partner is None or example.sequence[partner - 2 : partner] != "AG":
        return None, "missing_or_noncanonical_partner_acceptor"
    if example.sequence[example.acceptor_pos - 2 : example.acceptor_pos] != "AG":
        return None, "missing_or_noncanonical_prior_real_acceptor"
    if example.sequence[example.donor_pos : example.donor_pos + 2] != "GT":
        return None, "noncanonical_paired_donor"
    if example.exon_length is None or example.intron_length_down is None:
        return None, "missing_exon_or_downstream_intron_length"

    decoy_donor = matched_decoy_donor(example)
    if decoy_donor is None:
        return None, "missing_matched_decoy_donor"
    distance_mismatch = abs(int(decoy_donor) - int(example.donor_pos))
    if distance_mismatch > DECOY_TOL:
        return None, "matched_decoy_donor_outside_distance_caliper"
    if example.sequence[decoy_donor : decoy_donor + 2] != "GT":
        raise RuntimeError(f"{example.example_id}: matched decoy donor is not GT")

    plants = {
        "paired_donor": planted_donor_motifs(
            example.sequence, example.donor_pos, MOTIF_CLEAN_DONOR_ALT
        ),
        "matched_decoy_donor": planted_donor_motifs(
            example.sequence, decoy_donor, MOTIF_CLEAN_DONOR_ALT
        ),
    }
    if any(plants.values()):
        raise RuntimeError(f"{example.example_id}: GT->AC planted a splice motif: {plants}")
    return CohortRecord(
        example,
        int(decoy_donor),
        distance_mismatch,
        abs(int(example.exon_length) - int(example.intron_length_down)),
    ), None


def prepare_cohort(examples: Sequence, n: int) -> tuple[list[CohortRecord], dict]:
    exclusions: Counter[str] = Counter()
    comparator_complete: list[CohortRecord] = []
    for example in examples:
        record, reason = _control_record(example)
        if record is None:
            exclusions[str(reason)] += 1
        else:
            comparator_complete.append(record)
    dyad_matched = [
        record
        for record in comparator_complete
        if record.dyad_distance_mismatch <= DYAD_DISTANCE_CALIPER
    ]
    exclusions["exon_intron_dyad_outside_distance_caliper"] += (
        len(comparator_complete) - len(dyad_matched)
    )
    selected = dyad_matched[:n]
    genes = {record.example.gene_id for record in selected}
    if len(selected) < n:
        raise RuntimeError(
            f"only {len(selected)} outcome-blind distance-matched 2x2 examples are "
            f"available under the fixed {DYAD_DISTANCE_CALIPER}-nt dyad caliper; "
            f"requested {n}"
        )
    if len(genes) < MIN_GENES:
        raise RuntimeError(
            f"R1c requires at least {MIN_GENES} independent genes; got {len(genes)}"
        )
    audit = {
        "screened": len(examples),
        "screened_example_ids_sha256": canonical_sha256(
            [example.example_id for example in examples]
        ),
        "primary_comparator_complete": len(comparator_complete),
        "primary_comparator_complete_example_ids_sha256": canonical_sha256(
            [record.example.example_id for record in comparator_complete]
        ),
        "distance_matched_2x2_eligible": len(dyad_matched),
        "distance_matched_2x2_genes": len({
            record.example.gene_id for record in dyad_matched
        }),
        "distance_matched_2x2_example_ids_sha256": canonical_sha256(
            [record.example.example_id for record in dyad_matched]
        ),
        "dyad_distance_definition": "abs(exon_length-intron_length_down)",
        "dyad_distance_caliper": DYAD_DISTANCE_CALIPER,
        "selected": len(selected),
        "selected_genes": len(genes),
        "selected_example_ids_sha256": canonical_sha256(
            [record.example.example_id for record in selected]
        ),
        "selected_cohort_sha256": canonical_sha256(
            [
                {
                    "example_id": record.example.example_id,
                    "gene_id": record.example.gene_id,
                    "decoy_donor": record.matched_decoy_donor_pos,
                    "decoy_donor_distance_mismatch": (
                        record.matched_decoy_distance_mismatch
                    ),
                    "dyad_distance_mismatch": record.dyad_distance_mismatch,
                }
                for record in selected
            ]
        ),
        "exclusions": dict(sorted(exclusions.items())),
        "selection_reads_model_outputs": False,
    }
    return selected, audit


def _primary_family_matrix(
    Y_PD: np.ndarray,
    Y_PN: np.ndarray,
    Y_XD: np.ndarray,
    Y_XN: np.ndarray,
) -> np.ndarray:
    """Derive the exact four-row family from the acceptor-readout x donor-edit cells."""
    S_P = Y_PD - Y_PN
    S_X = Y_XD - Y_XN
    return np.vstack(
        [
            Y_PD,
            Y_PN,
            S_P,
            S_P - S_X,
        ]
    )


def _cell_effect_matrix(
    Y_PD: np.ndarray,
    Y_PN: np.ndarray,
    Y_XD: np.ndarray,
    Y_XN: np.ndarray,
) -> np.ndarray:
    cells = np.vstack((Y_PD, Y_PN, Y_XD, Y_XN)).astype(float, copy=False)
    if cells.ndim != 2 or not np.isfinite(cells).all():
        raise ValueError("the 2x2 cell matrix must be finite with shape [4, examples]")
    return cells


def _family_from_cells(cells: np.ndarray) -> np.ndarray:
    values = np.asarray(cells, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(CELL_NAMES):
        raise ValueError("cells must have the ordered rows Y_PD, Y_PN, Y_XD, Y_XN")
    return _primary_family_matrix(*values)


def cell_plant_vector() -> np.ndarray:
    """Plant only the paired-donor effect at P; derive all contrasts afterward."""
    return np.asarray([PLANT_DELTA, 0.0, 0.0, 0.0], dtype=float)


def structured_plant_vector() -> np.ndarray:
    return _family_from_cells(cell_plant_vector()[:, None])[:, 0]


def decision_path_requirements(*, component_names: bool = False) -> dict[str, dict[str, str]]:
    names = (
        [f"component_{index}" for index in range(len(PRIMARY_FAMILY_NAMES))]
        if component_names
        else list(PRIMARY_FAMILY_NAMES)
    )
    return {
        "core_paired_donor": {
            names[index]: label for index, label in CORE_PATH_REQUIREMENTS.items()
        },
        "stronger_downstream_preference": {
            names[index]: label for index, label in PRIMARY_PLANT_REQUIREMENTS.items()
        },
    }


def path_constituent_names() -> dict[str, tuple[str, ...]]:
    return {
        "core_paired_donor": ("Y_PD", "Y_PN", "S_P"),
        "stronger_downstream_preference": ("Y_PD", "Y_PN", "S_P", "I"),
    }


def boundary_test_contract() -> dict:
    return {
        "positive_endpoints": {
            name: {
                "null": f"theta_{name} <= {SESOI}",
                "alternative": f"theta_{name} > {SESOI}",
                "boundary": SESOI,
                "test": "one-sided whole-gene boundary sign flip",
            }
            for name in POSITIVE_BOUNDARY_NAMES
        },
        "Y_PN": {
            "null": f"theta_Y_PN <= -{SESOI} or theta_Y_PN >= {SESOI}",
            "alternative": f"-{SESOI} < theta_Y_PN < {SESOI}",
            "test": "TOST intersection-union whole-gene boundary sign flip",
        },
        "closed_null_numeric_ties": {
            "pvalue_rule": BOUNDARY_PVALUE_RULE,
            "comparison_tolerance_rule": BOUNDARY_COMPARISON_TOLERANCE_RULE,
            "policy": "machine-scale near-ties are included conservatively in the null tail",
        },
        "path_iut": "maximum constituent p-value",
        "path_multiplicity": {
            "method": "Holm step-down",
            "family": list(PATH_NAMES),
            "alpha": FAMILY_ALPHA,
        },
        "second_necessary_gate": (
            "strict simultaneous CI: positive lower bound > +SESOI; "
            "equivalence interval strictly inside (-SESOI,+SESOI)"
        ),
        "disagreement_policy": "any boundary-p/strict-CI disagreement is unresolved",
        "agreed_failure_policy": (
            "both necessary gates fail => NOT_DECLARED valid nonconfirmation"
        ),
    }


def _serialise_tost(test) -> dict:
    return {
        "estimate": float(test.estimate),
        "sesoi": float(test.sesoi),
        "p_value": float(test.p_value),
        "iut_rule": "max(lower_edge_p, upper_edge_p)",
        "lower_edge_test": test.lower_edge.to_dict(),
        "upper_edge_test": test.upper_edge.to_dict(),
    }


def r1c_boundary_classifier(
    family: np.ndarray,
    groups: Sequence[str],
    *,
    sesoi: float,
    n_randomisation: int,
    alpha: float,
    seed: int,
    namespace: str,
    min_genes: int,
    studentized: bool,
) -> dict:
    """Run the exact boundary-test + IUT + Holm + strict-CI production classifier."""
    values = np.asarray(family, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(PRIMARY_FAMILY_NAMES):
        raise ValueError("family must have ordered rows Y_PD, Y_PN, S_P, I")
    if values.shape[1] != len(groups):
        raise ValueError("groups must have one entry per family example")
    if n_randomisation < 1:
        raise ValueError("n_randomisation must be positive")

    interval_raw, interval_calibration = simultaneous_family_decision(
        values,
        groups,
        PRIMARY_FAMILY_NAMES,
        sesoi=sesoi,
        n_draws=n_randomisation,
        alpha=alpha,
        seed=seed,
        namespace=f"{namespace}:simultaneous-ci",
        min_genes=min_genes,
        studentized=studentized,
    )
    interval_calibration = describe_endpoint_scale_normalization(
        interval_calibration
    )
    decisions: dict[str, dict] = {}
    for name, decision in interval_raw.items():
        lo, hi = (float(value) for value in decision.simultaneous_ci)
        strict_class = strict_interval_decision(lo, hi, sesoi)
        decisions[name] = {
            "estimate": float(decision.estimate),
            "simultaneous_ci": [lo, hi],
            "classification": strict_class,
            "classification_rule": "strict simultaneous SESOI boundaries",
            "zero_boundary_max_abs_p_diagnostic": float(decision.max_abs_p),
            "decision_bearing_p_value_location": "boundary_inference.endpoint_p_values",
        }

    positive_indices = [
        PRIMARY_FAMILY_NAMES.index(name) for name in POSITIVE_BOUNDARY_NAMES
    ]
    boundary_namespace = f"{namespace}:shared-boundary-signs"
    positive_tests = boundary_sign_flip_tests(
        values[positive_indices],
        groups,
        POSITIVE_BOUNDARY_NAMES,
        boundaries=float(sesoi),
        alternatives="greater",
        n_draws=n_randomisation,
        seed=seed,
        namespace=boundary_namespace,
        min_genes=min_genes,
    )
    ypn_tost = tost_equivalence_test(
        values[PRIMARY_FAMILY_NAMES.index("Y_PN")],
        groups,
        sesoi=sesoi,
        n_draws=n_randomisation,
        seed=seed,
        namespace=boundary_namespace,
        min_genes=min_genes,
    )
    boundary_tests = {
        "Y_PD_gt_SESOI": positive_tests["Y_PD"].to_dict(),
        "S_P_gt_SESOI": positive_tests["S_P"].to_dict(),
        "I_gt_SESOI": positive_tests["I"].to_dict(),
        "Y_PN_gt_negative_SESOI": ypn_tost.lower_edge.to_dict(),
        "Y_PN_lt_positive_SESOI": ypn_tost.upper_edge.to_dict(),
    }
    sign_matrix_digests = {
        test["sign_matrix_sha256"] for test in boundary_tests.values()
    }
    gene_label_digests = {
        test["ordered_gene_labels_sha256"] for test in boundary_tests.values()
    }
    if len(sign_matrix_digests) != 1 or len(gene_label_digests) != 1:
        raise RuntimeError(
            "R1c boundary tests must reuse one ordered-gene sign matrix"
        )
    shared_boundary_randomisation = {
        "namespace": boundary_namespace,
        "sign_matrix_sha256": next(iter(sign_matrix_digests)),
        "ordered_gene_labels_sha256": next(iter(gene_label_digests)),
        "all_five_boundary_tests_share_sign_matrix": True,
    }
    endpoint_p_values = {
        "Y_PD": float(positive_tests["Y_PD"].p_value),
        "Y_PN": float(ypn_tost.p_value),
        "S_P": float(positive_tests["S_P"].p_value),
        "I": float(positive_tests["I"].p_value),
    }
    constituents = path_constituent_names()
    raw_path_p = {
        path: intersection_union_pvalue(
            [endpoint_p_values[name] for name in constituents[path]]
        )
        for path in PATH_NAMES
    }
    adjusted_path_p = holm_adjust_named(raw_path_p)
    requirements = decision_path_requirements()
    path_tests: dict[str, dict] = {}
    for path in PATH_NAMES:
        ci_pass = all(
            decisions[name]["classification"] == expected
            for name, expected in requirements[path].items()
        )
        p_pass = adjusted_path_p[path] <= alpha
        disagreement = bool(p_pass) != bool(ci_pass)
        declared = bool(p_pass and ci_pass)
        path_decision = (
            "DECLARED"
            if declared
            else ("UNRESOLVED" if disagreement else "NOT_DECLARED")
        )
        path_tests[path] = {
            "constituent_p_values": {
                name: endpoint_p_values[name] for name in constituents[path]
            },
            "iut_p_value": float(raw_path_p[path]),
            "iut_rule": "maximum constituent p-value",
            "holm_adjusted_p_value": float(adjusted_path_p[path]),
            "holm_family": list(PATH_NAMES),
            "alpha": float(alpha),
            "p_rejects": bool(p_pass),
            "strict_ci_requirements": requirements[path],
            "strict_ci_passes": bool(ci_pass),
            "p_ci_disagreement": disagreement,
            "decision": path_decision,
            "unresolved_reason": (
                "BOUNDARY_P_STRICT_CI_DISAGREEMENT"
                if disagreement else None
            ),
            "nondeclaration_reason": (
                "BOTH_NECESSARY_GATES_FAILED"
                if path_decision == "NOT_DECLARED"
                else None
            ),
        }

    return {
        "schema": "gfmcd.r1c.boundary_classifier/1",
        "namespace": namespace,
        "alpha": float(alpha),
        "sesoi": float(sesoi),
        "n_randomisation": int(n_randomisation),
        "minimum_genes_required": int(min_genes),
        "boundary_test_contract": boundary_test_contract(),
        "family_decisions": decisions,
        "family_calibration": interval_calibration,
        "boundary_tests": boundary_tests,
        "shared_boundary_randomisation": shared_boundary_randomisation,
        "Y_PN_TOST": _serialise_tost(ypn_tost),
        "endpoint_p_values": endpoint_p_values,
        "path_tests": path_tests,
        "holm_adjusted_path_p_values": {
            name: float(adjusted_path_p[name]) for name in PATH_NAMES
        },
    }


def r1c_gene_robustness_gate(
    family: np.ndarray,
    groups: Sequence[str],
    *,
    n_draws: int = PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
    seed: int = PRODUCTION_SEED,
    namespace: str = "r1c:production-gene-robustness",
) -> dict:
    """Require raw and 10%-trimmed equal-gene support beside exact inference."""

    values = np.asarray(family, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(PRIMARY_FAMILY_NAMES):
        raise ValueError("robustness family must contain the ordered four R1c rows")
    if values.shape[1] != len(groups):
        raise ValueError("robustness family and gene labels differ in length")
    if n_draws < PRODUCTION_ROBUST_BOOTSTRAP_DRAWS:
        raise ValueError("production robustness requires at least 10,000 draws")

    positive_names = ("Y_PD", "S_P", "I")
    positive_indices = [PRIMARY_FAMILY_NAMES.index(name) for name in positive_names]
    positive = studentized_gene_bootstrap_tests(
        values[positive_indices],
        groups,
        positive_names,
        boundaries=SESOI,
        alternatives="greater",
        alpha=FAMILY_ALPHA,
        trim_fraction=ROBUST_TRIM_FRACTION,
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=MIN_GENES,
    )
    pseudo = studentized_gene_bootstrap_tost(
        values[[PRIMARY_FAMILY_NAMES.index("Y_PN")]],
        groups,
        ("Y_PN",),
        sesoi=SESOI,
        alpha=FAMILY_ALPHA,
        trim_fraction=ROBUST_TRIM_FRACTION,
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=MIN_GENES,
    )["Y_PN"]
    # The artifact is the contract: normalize tuple-valued quantiles before returning so
    # an immutable JSON round trip replays byte-for-byte rather than changing tuples to
    # lists only after publication.
    endpoints = json.loads(
        json.dumps(
            {
                **{name: positive[name].to_dict() for name in positive_names},
                "Y_PN": pseudo.to_dict(),
            }
        )
    )
    core_pass = bool(
        positive["Y_PD"].joint_boundary_pass
        and pseudo.equivalence_pass
        and positive["S_P"].joint_boundary_pass
    )
    stronger_pass = bool(core_pass and positive["I"].joint_boundary_pass)
    all_required = core_pass
    return {
        "schema": "gfmcd.r1c.equal_gene_robustness/1",
        "inference_role": (
            "necessary robustness conjunction alongside, not replacement for, "
            "exact sign-flip/Holm plus simultaneous-CI authority"
        ),
        "randomisation_unit": "unweighted gene",
        "trim_fraction": ROBUST_TRIM_FRACTION,
        "n_draws": int(n_draws),
        "minimum_draws": PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
        "alpha": FAMILY_ALPHA,
        "sesoi": SESOI,
        "endpoints": endpoints,
        "paths": {
            "core_paired_donor": {"pass": core_pass},
            "stronger_downstream_preference": {"pass": stronger_pass},
        },
        "all_required_robust": all_required,
        "failure_disposition": (
            None
            if all_required
            else "SCIENTIFIC_UNRESOLVED_TAIL_SENSITIVITY_NOT_INSTRUMENT_FAILURE"
        ),
    }


def observed_decision_paths(boundary_inference: dict) -> dict[str, bool]:
    path_tests = boundary_inference.get("path_tests", {})
    if set(path_tests) != set(PATH_NAMES):
        raise ValueError("boundary inference must contain exactly the two named paths")
    path_decisions = {
        name: path_tests[name].get("decision") for name in PATH_NAMES
    }
    if any(
        decision not in {"DECLARED", "UNRESOLVED", "NOT_DECLARED"}
        for decision in path_decisions.values()
    ):
        raise ValueError("each path must have a recognized three-state decision")
    declared = {
        name: path_decisions[name] == "DECLARED"
        for name in PATH_NAMES
    }
    if declared["stronger_downstream_preference"] and not declared[
        "core_paired_donor"
    ]:
        raise ValueError("the nested stronger path cannot be declared without core")
    return declared


def endpoint_centered_cell_residuals(cells: np.ndarray) -> np.ndarray:
    """Remove every cell's observed mean before whole-gene outer sign flips.

    Centering the four cells before deriving S_P and I makes the preflight invariant to an
    effect-bearing translation of any endpoint.  A single gene sign is then shared across
    all four cells, preserving the 2x2 algebra in every outer trial.
    """
    values = np.asarray(cells, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(CELL_NAMES):
        raise ValueError("cells must have the ordered rows Y_PD, Y_PN, Y_XD, Y_XN")
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("cells must contain finite example-level values")
    return values - values.mean(axis=1, keepdims=True)


def least_favourable_boundary_cases(
    *, sesoi: float = SESOI, alternative_level: float = PLANT_DELTA
) -> dict[str, dict]:
    """Return algebraically coherent primitive-cell shifts for every path boundary."""
    if not np.isfinite(sesoi) or sesoi <= 0:
        raise ValueError("sesoi must be positive and finite")
    if not np.isfinite(alternative_level) or alternative_level <= 3 * sesoi:
        raise ValueError("alternative_level must exceed 3 * sesoi")
    s = float(sesoi)
    a = float(alternative_level)
    g = float(LFC_NUMERIC_GUARD)
    positive_null_edge = s - g
    lower_equivalence_null_edge = -s - g
    upper_equivalence_null_edge = s + g
    ypd_specific = positive_null_edge + s / 2.0
    cases = {
        "Y_PD_at_positive_boundary": {
            "boundary_constituent": "Y_PD",
            "nominal_boundary": s,
            "numeric_guard_toward_composite_null": g,
            "cell_vector": [
                positive_null_edge,
                -s / 2.0,
                0.0,
                a - ypd_specific,
            ],
            "null_paths": list(PATH_NAMES),
        },
        "Y_PN_at_lower_equivalence_boundary": {
            "boundary_constituent": "Y_PN_lower_edge",
            "nominal_boundary": -s,
            "numeric_guard_toward_composite_null": g,
            "cell_vector": [
                a,
                lower_equivalence_null_edge,
                s + g,
                0.0,
            ],
            "null_paths": list(PATH_NAMES),
        },
        "Y_PN_at_upper_equivalence_boundary": {
            "boundary_constituent": "Y_PN_upper_edge",
            "nominal_boundary": s,
            "numeric_guard_toward_composite_null": g,
            "cell_vector": [
                a,
                upper_equivalence_null_edge,
                0.0,
                s + g,
            ],
            "null_paths": list(PATH_NAMES),
        },
        "S_P_at_positive_boundary": {
            "boundary_constituent": "S_P",
            "nominal_boundary": s,
            "numeric_guard_toward_composite_null": g,
            "cell_vector": [
                s / 2.0 + positive_null_edge,
                s / 2.0,
                0.0,
                a - positive_null_edge,
            ],
            "null_paths": list(PATH_NAMES),
        },
        "I_at_positive_boundary": {
            "boundary_constituent": "I",
            "nominal_boundary": s,
            "numeric_guard_toward_composite_null": g,
            "cell_vector": [a, 0.0, a - positive_null_edge, 0.0],
            "null_paths": ["stronger_downstream_preference"],
        },
    }
    for case in cases.values():
        cell_vector = np.asarray(case["cell_vector"], dtype=float)
        case["family_vector"] = _family_from_cells(cell_vector[:, None])[:, 0].tolist()
    return cases


def nuisance_surface_representatives(
    *, sesoi: float = SESOI, alternative_level: float = PLANT_DELTA
) -> dict[str, list[dict]]:
    """Prespecified coherent points illustrating the two coupled null faces.

    ``Y_PD`` and ``S_P`` cannot be moved independently of every nuisance endpoint:
    ``S_P = Y_PD - Y_PN``.  These points span the admissible negative/positive
    ``Y_PN`` interiors while keeping every other path constituent strictly in its
    alternative.  The middle point on each face is the classifier surface executed in
    every outer trial; the continuum guarantee itself is the event-inclusion proof in
    :func:`nuisance_continuum_control`, not interpolation between these points.
    """
    if not np.isfinite(sesoi) or sesoi <= 0:
        raise ValueError("sesoi must be positive and finite")
    if not np.isfinite(alternative_level) or alternative_level <= 3 * sesoi:
        raise ValueError("alternative_level must exceed 3 * sesoi")
    s = float(sesoi)
    a = float(alternative_level)
    g = float(LFC_NUMERIC_GUARD)
    edge = s - g
    specifications = {
        "Y_PD_positive_boundary_face": (
            "Y_PD",
            [(-0.75 * s, 2.0 * s), (-0.50 * s, a), (-0.25 * s, 0.50 * a)],
        ),
        "S_P_positive_boundary_face": (
            "S_P",
            [(0.25 * s, 2.0 * s), (0.50 * s, a), (0.75 * s, 0.50 * a)],
        ),
    }
    surfaces: dict[str, list[dict]] = {}
    for face, (boundary_name, nuisance_points) in specifications.items():
        rows: list[dict] = []
        for index, (ypn, interaction) in enumerate(nuisance_points):
            if boundary_name == "Y_PD":
                ypd = edge
                specificity = ypd - ypn
            else:
                specificity = edge
                ypd = specificity + ypn
            # Choose X_N=0 and solve X_D so I=S_P-(Y_XD-Y_XN).
            cells = np.asarray(
                [ypd, ypn, specificity - interaction, 0.0], dtype=float
            )
            family = _family_from_cells(cells[:, None])[:, 0]
            rows.append(
                {
                    "surface_index": index,
                    "boundary_constituent": boundary_name,
                    "nominal_boundary": s,
                    "numeric_guard_toward_composite_null": g,
                    "Y_PN_nuisance": float(ypn),
                    "I_nuisance": float(interaction),
                    "cell_vector": cells.tolist(),
                    "family_vector": family.tolist(),
                    "all_other_path_constituents_strictly_in_alternative": True,
                    "executed_in_outer_trials": index == 1,
                }
            )
        surfaces[face] = rows
    return surfaces


def nuisance_continuum_control() -> dict:
    """Machine-bound proof that nuisance values cannot inflate path type-I error.

    For a path ``h``, declaration is the conjunction of its strict-CI gate and
    ``Holm(p_h) <= alpha``.  Holm adjustment is never smaller than the raw path
    p-value, and the raw IUT p-value is the maximum constituent p-value.  Therefore,
    when constituent ``j`` is null, ``{declare h}`` is a subset of
    ``{p_j <= alpha}`` for *every* value of all nuisance constituents.  No monotonicity
    assumption about those nuisance values is needed.
    """
    surfaces = nuisance_surface_representatives()
    executed = {
        "Y_PD_positive_boundary_face": "Y_PD_at_positive_boundary",
        "S_P_positive_boundary_face": "S_P_at_positive_boundary",
    }
    cases = least_favourable_boundary_cases()
    for face, case_name in executed.items():
        if not np.allclose(
            surfaces[face][1]["family_vector"],
            cases[case_name]["family_vector"],
            rtol=0.0,
            atol=1e-14,
        ):
            raise RuntimeError(
                f"{face} middle representative drifted from its executed classifier surface"
            )
    return {
        "schema": "gfmcd.r1c.nuisance_continuum_control/1",
        "scope": ["Y_PD_positive_boundary_face", "S_P_positive_boundary_face"],
        "event_definition": (
            "declare_h = strict_CI_conjunction_h AND Holm_h(max_j p_hj) <= alpha"
        ),
        "ordered_inequality": (
            "for every constituent j: p_hj <= max_j(p_hj) <= Holm_h(max_j p_hj)"
        ),
        "null_event_inclusion": (
            "if constituent j is null, {declare_h} subseteq {p_hj <= alpha} "
            "for every nuisance-parameter value"
        ),
        "boundary_test_validity": (
            "the one-sided whole-gene sign-flip test centers endpoint j at its null "
            "boundary; outer trials impose the same whole-gene sign symmetry and the "
            "add-one p-value includes ties"
        ),
        "holm_property": "Holm adjusted p-values are never smaller than raw path p-values",
        "iut_property": "the raw path p-value is the maximum constituent p-value",
        "strict_ci_property": "the strict-CI conjunction can only remove declarations",
        "continuum_conclusion": (
            "pointwise nuisance maximization is unnecessary: the boundary-constituent "
            "level-alpha event is a uniform upper bound over the full nuisance continuum"
        ),
        "representative_classifier_surfaces": surfaces,
        "executed_surface_cases": executed,
    }


def _classifier_trial_audit(classifier: dict) -> dict:
    """Keep every boundary replay field while omitting bulky interval calibration diagnostics."""
    return {
        "namespace": classifier["namespace"],
        "family_decisions": classifier["family_decisions"],
        "boundary_tests": classifier["boundary_tests"],
        "shared_boundary_randomisation": classifier[
            "shared_boundary_randomisation"
        ],
        "endpoint_p_values": classifier["endpoint_p_values"],
        "path_tests": classifier["path_tests"],
        "holm_adjusted_path_p_values": classifier[
            "holm_adjusted_path_p_values"
        ],
    }


def _rederive_trial_classifier_decisions(audit: dict) -> dict[str, bool]:
    """Recompute a compact calibration audit's complete decision algebra."""
    decisions = audit.get("family_decisions")
    tests = audit.get("boundary_tests")
    if not isinstance(decisions, dict) or set(decisions) != set(PRIMARY_FAMILY_NAMES):
        raise _preflight_semantic_error(
            "calibration classifier decision family is incomplete"
        )
    expected_test_names = {
        "Y_PD_gt_SESOI",
        "S_P_gt_SESOI",
        "I_gt_SESOI",
        "Y_PN_gt_negative_SESOI",
        "Y_PN_lt_positive_SESOI",
    }
    if not isinstance(tests, dict) or set(tests) != expected_test_names:
        raise _preflight_semantic_error(
            "calibration classifier boundary family is incomplete"
        )
    estimate_tests = {
        "Y_PD": ("Y_PD_gt_SESOI",),
        "Y_PN": (
            "Y_PN_gt_negative_SESOI",
            "Y_PN_lt_positive_SESOI",
        ),
        "S_P": ("S_P_gt_SESOI",),
        "I": ("I_gt_SESOI",),
    }
    for name, test_names in estimate_tests.items():
        decision = decisions[name]
        interval = decision.get("simultaneous_ci") if isinstance(decision, dict) else None
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(_is_finite_number(value) for value in interval)
            or decision.get("classification_rule")
            != "strict simultaneous SESOI boundaries"
            or decision.get("classification")
            != strict_interval_decision(interval[0], interval[1], SESOI)
            or not _is_finite_number(decision.get("estimate"))
        ):
            raise _preflight_semantic_error(
                "calibration classifier strict-CI decision is inconsistent"
            )
        for test_name in test_names:
            test = tests[test_name]
            direction = 1.0 if test.get("alternative") == "greater" else -1.0
            if (
                not _is_finite_number(test.get("estimate"))
                or not np.isclose(
                    test["estimate"], decision["estimate"], rtol=0.0, atol=1e-12
                )
                or not _is_finite_number(test.get("oriented_statistic"))
                or not _is_finite_number(test.get("boundary"))
                or not np.isclose(
                    test["oriented_statistic"],
                    direction * (test["estimate"] - test["boundary"]),
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise _preflight_semantic_error(
                    "calibration classifier estimate/statistic does not reproduce"
                )

    endpoint_p = {
        "Y_PD": tests["Y_PD_gt_SESOI"]["p_value"],
        "Y_PN": max(
            tests["Y_PN_gt_negative_SESOI"]["p_value"],
            tests["Y_PN_lt_positive_SESOI"]["p_value"],
        ),
        "S_P": tests["S_P_gt_SESOI"]["p_value"],
        "I": tests["I_gt_SESOI"]["p_value"],
    }
    if audit.get("endpoint_p_values") != endpoint_p:
        raise _preflight_semantic_error(
            "calibration classifier endpoint p-values do not reproduce"
        )
    constituents = path_constituent_names()
    raw = {
        path: intersection_union_pvalue(
            [endpoint_p[name] for name in constituents[path]]
        )
        for path in PATH_NAMES
    }
    adjusted = holm_adjust_named(raw)
    requirements = decision_path_requirements()
    expected_paths: dict[str, dict] = {}
    for path in PATH_NAMES:
        ci_pass = all(
            decisions[name]["classification"] == label
            for name, label in requirements[path].items()
        )
        p_pass = adjusted[path] <= FAMILY_ALPHA
        disagreement = bool(p_pass) != bool(ci_pass)
        declared = bool(p_pass and ci_pass)
        path_decision = (
            "DECLARED"
            if declared
            else ("UNRESOLVED" if disagreement else "NOT_DECLARED")
        )
        expected_paths[path] = {
            "constituent_p_values": {
                name: endpoint_p[name] for name in constituents[path]
            },
            "iut_p_value": raw[path],
            "iut_rule": "maximum constituent p-value",
            "holm_adjusted_p_value": adjusted[path],
            "holm_family": list(PATH_NAMES),
            "alpha": FAMILY_ALPHA,
            "p_rejects": bool(p_pass),
            "strict_ci_requirements": requirements[path],
            "strict_ci_passes": bool(ci_pass),
            "p_ci_disagreement": disagreement,
            "decision": path_decision,
            "unresolved_reason": (
                "BOUNDARY_P_STRICT_CI_DISAGREEMENT" if disagreement else None
            ),
            "nondeclaration_reason": (
                "BOTH_NECESSARY_GATES_FAILED"
                if path_decision == "NOT_DECLARED"
                else None
            ),
        }
    if audit.get("path_tests") != expected_paths:
        raise _preflight_semantic_error(
            "calibration classifier path decision does not reproduce IUT/Holm/CI"
        )
    if audit.get("holm_adjusted_path_p_values") != adjusted:
        raise _preflight_semantic_error(
            "calibration classifier adjusted p-values do not reproduce"
        )
    return {
        path: expected_paths[path]["decision"] == "DECLARED" for path in PATH_NAMES
    }


def _binomial_count_thresholds(
    n_trials: int, *, alpha: float, power_floor: float
) -> tuple[int, int]:
    fp_allowed = [
        count
        for count in range(n_trials + 1)
        if one_sided_binomial_bounds(count, n_trials, alpha=alpha)[1] <= alpha
    ]
    power_required = [
        count
        for count in range(n_trials + 1)
        if one_sided_binomial_bounds(count, n_trials, alpha=alpha)[0]
        >= power_floor
    ]
    return (max(fp_allowed, default=-1), min(power_required, default=n_trials + 1))


def r1c_decision_path_preflight(
    cells: np.ndarray,
    groups: Sequence[str],
    *,
    sesoi: float,
    n_calibration: int,
    n_trials: int,
    alpha: float,
    seed: int,
    namespace: str,
    min_genes: int,
    power_floor: float,
    studentized: bool,
) -> dict:
    """Calibrate the exact boundary/IUT/Holm/strict-CI classifier on primitive cells."""
    cell_values = np.asarray(cells, dtype=float)
    if cell_values.ndim != 2 or cell_values.shape[0] != len(CELL_NAMES):
        raise ValueError("cells must have the ordered rows Y_PD, Y_PN, Y_XD, Y_XN")
    if cell_values.shape[1] != len(groups):
        raise ValueError("groups must have one entry per 2x2 example")
    if n_calibration < 1 or n_trials < 1:
        raise ValueError("n_calibration and n_trials must be positive")
    if not 0 < power_floor < 1:
        raise ValueError("power_floor must be in (0, 1)")

    family = _family_from_cells(cell_values)
    _, reference = simultaneous_family_decision(
        family,
        groups,
        PRIMARY_FAMILY_NAMES,
        sesoi=sesoi,
        n_draws=n_calibration,
        alpha=alpha,
        seed=seed,
        namespace=f"{namespace}:reference",
        min_genes=min_genes,
        studentized=studentized,
    )
    reference = describe_endpoint_scale_normalization(reference)
    residuals = endpoint_centered_cell_residuals(cell_values)
    cell_plant = cell_plant_vector()
    boundary_cases = least_favourable_boundary_cases(
        sesoi=sesoi, alternative_level=PLANT_DELTA
    )
    false_positive_counts = {
        path: {
            case_name: 0
            for case_name, case in boundary_cases.items()
            if path in case["null_paths"]
        }
        for path in PATH_NAMES
    }
    plant_successes = {path: 0 for path in PATH_NAMES}
    trial_records: list[dict] = []
    for trial in range(n_trials):
        outer_namespace = f"{namespace}:centered-primitive-null:{trial}"
        null_cells = gene_sign_flip(
            residuals,
            groups,
            seed=seed,
            namespace=outer_namespace,
        )
        case_audits: dict[str, dict] = {}
        for case_name, case in boundary_cases.items():
            shifted_cells = null_cells + np.asarray(case["cell_vector"])[:, None]
            classifier = r1c_boundary_classifier(
                _family_from_cells(shifted_cells),
                groups,
                sesoi=sesoi,
                n_randomisation=n_calibration,
                alpha=alpha,
                seed=seed,
                namespace=f"{namespace}:lfc:{case_name}:{trial}",
                min_genes=min_genes,
                studentized=studentized,
            )
            declared = observed_decision_paths(classifier)
            for path in case["null_paths"]:
                false_positive_counts[path][case_name] += int(declared[path])
            case_audits[case_name] = _classifier_trial_audit(classifier)

        planted_cells = null_cells + cell_plant[:, None]
        planted_classifier = r1c_boundary_classifier(
            _family_from_cells(planted_cells),
            groups,
            sesoi=sesoi,
            n_randomisation=n_calibration,
            alpha=alpha,
            seed=seed,
            namespace=f"{namespace}:coherent-plant:{trial}",
            min_genes=min_genes,
            studentized=studentized,
        )
        planted_paths = observed_decision_paths(planted_classifier)
        for path in PATH_NAMES:
            plant_successes[path] += int(planted_paths[path])
        trial_records.append(
            {
                "trial": trial,
                "outer_sign_namespace": outer_namespace,
                "least_favourable_cases": case_audits,
                "coherent_plant": _classifier_trial_audit(planted_classifier),
            }
        )

    worst_false_positives = {
        path: max(false_positive_counts[path].values()) for path in PATH_NAMES
    }
    fpr_upper = {
        path: one_sided_binomial_bounds(
            worst_false_positives[path], n_trials, alpha=alpha
        )[1]
        for path in PATH_NAMES
    }
    power_lower = {
        path: one_sided_binomial_bounds(
            plant_successes[path], n_trials, alpha=alpha
        )[0]
        for path in PATH_NAMES
    }
    max_fp_allowed, min_power_successes = _binomial_count_thresholds(
        n_trials, alpha=alpha, power_floor=power_floor
    )
    if (
        n_trials == PRODUCTION_TRIALS
        and np.isclose(alpha, FAMILY_ALPHA)
        and np.isclose(power_floor, PREFLIGHT_POWER_FLOOR)
    ):
        if (
            max_fp_allowed != MAX_FALSE_POSITIVES
            or min_power_successes != MIN_POWER_SUCCESSES
        ):
            raise RuntimeError("production Clopper-Pearson count boundaries drifted")
    return {
        "randomisation_unit": "gene",
        "outer_trial_unit": "endpoint-centered 2x2 cell residual",
        "outer_trial_centering": "subtract each observed cell mean before sign flipping",
        "outer_trial_algebra": (
            "one shared sign per gene across Y_PD/Y_PN/Y_XD/Y_XN; derive S_P and I after flip"
        ),
        "translation_invariant": True,
        "classifier": (
            "exact one-sided boundary tests + Y_PN TOST + path IUT + Holm over exactly "
            "two named paths + strict simultaneous-CI gate on every LFC and plant trial"
        ),
        "classifier_schema": "gfmcd.r1c.boundary_classifier/1",
        "boundary_test_contract": boundary_test_contract(),
        "namespace": namespace,
        "alpha": float(alpha),
        "sesoi": float(sesoi),
        "n_components": family.shape[0],
        "n_examples": family.shape[1],
        "n_genes": len(set(map(str, groups))),
        "minimum_genes_required": int(min_genes),
        "n_calibration": n_calibration,
        "n_trials": n_trials,
        "reference_interval_calls": 1,
        "production_classifier_calls": n_trials * (len(boundary_cases) + 1),
        "simultaneous_half_width": reference["simultaneous_half_width"],
        "simultaneous_half_widths": reference["simultaneous_half_widths"],
        "studentized": bool(studentized),
        "interval_scaling": reference["interval_scaling"],
        "resolution_ok": reference["resolution_ok"],
        "least_favourable_boundary_cases": boundary_cases,
        "nuisance_continuum_control": nuisance_continuum_control(),
        "lfc_numeric_guard_toward_composite_null": LFC_NUMERIC_GUARD,
        "boundary_false_positive_counts": false_positive_counts,
        "worst_case_false_positives": worst_false_positives,
        "maximum_false_positives_allowed": max_fp_allowed,
        "path_boundary_fpr": {
            path: worst_false_positives[path] / n_trials for path in PATH_NAMES
        },
        "path_boundary_fpr_upper": fpr_upper,
        "path_boundary_fpr_ok": {
            path: bool(
                worst_false_positives[path] <= max_fp_allowed
                and fpr_upper[path] <= alpha
            )
            for path in PATH_NAMES
        },
        "planted_effect": None,
        "plant_delta": PLANT_DELTA,
        "cell_plant_vector": cell_plant.tolist(),
        "plant_vector": structured_plant_vector().tolist(),
        "planted_requirements": _component_plant_requirements(),
        "decision_paths": decision_path_requirements(component_names=True),
        "path_holm_family": list(PATH_NAMES),
        "plant_successes": plant_successes,
        "minimum_power_successes_required": min_power_successes,
        "path_planted_power": {
            path: plant_successes[path] / n_trials for path in PATH_NAMES
        },
        "path_planted_power_lower": power_lower,
        "path_planted_power_ok": {
            path: bool(
                plant_successes[path] >= min_power_successes
                and power_lower[path] >= power_floor
            )
            for path in PATH_NAMES
        },
        "power_floor": power_floor,
        "binomial_bound": f"one-sided {100 * (1 - alpha):g}% Clopper-Pearson",
        "trial_records": trial_records,
        "trial_records_sha256": canonical_sha256(trial_records),
    }


def describe_endpoint_scale_normalization(calibration: dict) -> dict:
    """Replace the shared helper's ambiguous historical `studentized` label."""
    if calibration.get("studentized") is not True:
        raise RuntimeError("R1c requires endpoint-scale-normalized simultaneous inference")
    described = dict(calibration)
    if "method" in described:
        described["method"] = (
            "gene-cluster bootstrap endpoint-scale-normalized max-deviation interval"
        )
    described["interval_scaling"] = (
        "fixed endpoint bootstrap SD; not nested bootstrap-t"
    )
    return described


def measure_behavior(
    task: CouplingTask,
    records: Sequence[CohortRecord],
) -> tuple[np.ndarray, dict, dict]:
    examples = [record.example for record in records]
    clean_sequences = [example.sequence for example in examples]
    real_corrupt = [
        substitute(example.sequence, example.donor_pos, MOTIF_CLEAN_DONOR_ALT)
        for example in examples
    ]
    decoy_corrupt = [
        substitute(example.sequence, record.matched_decoy_donor_pos, MOTIF_CLEAN_DONOR_ALT)
        for example, record in zip(examples, records)
    ]
    partner_readouts = [acceptor_readout(example) for example in examples]
    exon_mate_readouts = [example.acceptor_pos - 1 for example in examples]

    partner_clean = margins_at(task, partner_readouts, clean_sequences)
    partner_real_corrupt = margins_at(task, partner_readouts, real_corrupt)
    partner_decoy_corrupt = margins_at(task, partner_readouts, decoy_corrupt)
    exon_mate_clean = margins_at(task, exon_mate_readouts, clean_sequences)
    exon_mate_real_corrupt = margins_at(task, exon_mate_readouts, real_corrupt)
    exon_mate_decoy_corrupt = margins_at(task, exon_mate_readouts, decoy_corrupt)

    cells = {
        "Y_PD": partner_clean - partner_real_corrupt,
        "Y_PN": partner_clean - partner_decoy_corrupt,
        "Y_XD": exon_mate_clean - exon_mate_real_corrupt,
        "Y_XN": exon_mate_clean - exon_mate_decoy_corrupt,
    }
    cell_matrix = _cell_effect_matrix(**cells)
    family = _family_from_cells(cell_matrix)
    descriptive = {
        "names": list(DESCRIPTIVE_FAMILY_NAMES),
        "status": "REPORTED_NOT_DECISION_BEARING",
        "values": np.vstack(
            (
                cells["Y_XD"],
                cells["Y_XN"],
                cells["Y_XD"] - cells["Y_XN"],
            )
        ),
        "cell_values": cell_matrix,
    }
    levels = {
        "P_clean": partner_clean,
        "P_under_D": partner_real_corrupt,
        "P_under_N": partner_decoy_corrupt,
        "X_clean": exon_mate_clean,
        "X_under_D": exon_mate_real_corrupt,
        "X_under_N": exon_mate_decoy_corrupt,
    }
    return family, levels, descriptive


def _health_positions(records: Sequence[CohortRecord], n_probe: int, seed: int) -> list[list[int]]:
    positions: list[list[int]] = []
    for record in records:
        example = record.example
        excluded = (
            int(example.donor_pos),
            int(example.acceptor_pos - 1),
            int(acceptor_readout(example)),
        )
        candidates = [
            position
            for position in range(10, len(example.sequence) - 10)
            if all(abs(position - site) > 3 for site in excluded)
        ]
        if len(candidates) < n_probe:
            raise RuntimeError(f"{example.example_id}: insufficient non-site health probes")
        rng = stable_rng(f"r1c-health|{example.example_id}", seed)
        positions.append(sorted(rng.sample(candidates, n_probe)))
    return positions


def per_example_reverse_health(
    task: CouplingTask,
    records: Sequence[CohortRecord],
    sequences: Sequence[str],
    *,
    n_probe: int,
    seed: int,
) -> np.ndarray:
    """Top-1 nucleotide accuracy with the acceptor and one non-site probe both masked."""
    if len(records) != len(sequences):
        raise ValueError("health sequences must have one entry per cohort record")
    probes = _health_positions(records, n_probe, seed)
    hits = np.zeros(len(records), dtype=float)
    trials = np.zeros(len(records), dtype=float)
    with torch.no_grad():
        for probe_index in range(n_probe):
            for start in range(0, len(records), task.batch):
                stop = min(start + task.batch, len(records))
                chunk_records = records[start:stop]
                chunk_sequences = list(sequences[start:stop])
                readouts = [acceptor_readout(record.example) for record in chunk_records]
                probe_chunk = [probes[index][probe_index] for index in range(start, stop)]
                encoded = task.encode(chunk_sequences, readouts)
                for row, position in enumerate(probe_chunk):
                    encoded["input_ids"][row, position + task.off] = task.mask_id
                logits = task.ad.model(**encoded).logits.float()
                for row, position in enumerate(probe_chunk):
                    nucleotide_logits = torch.stack(
                        [logits[row, position + task.off, task.bid[base]] for base in "ACGT"]
                    )
                    predicted = "ACGT"[int(nucleotide_logits.argmax().item())]
                    absolute_index = start + row
                    hits[absolute_index] += int(predicted == chunk_sequences[row][position])
                    trials[absolute_index] += 1
    if np.any(trials != n_probe):
        raise RuntimeError("health probe accounting mismatch")
    return hits / trials


def health_measurements(
    task: CouplingTask,
    records: Sequence[CohortRecord],
    *,
    n_probe: int,
    seed: int,
) -> dict[str, np.ndarray]:
    clean = [record.example.sequence for record in records]
    corrupt = [
        substitute(record.example.sequence, record.example.donor_pos, MOTIF_CLEAN_DONOR_ALT)
        for record in records
    ]
    clean_accuracy = per_example_reverse_health(
        task, records, clean, n_probe=n_probe, seed=seed
    )
    corrupt_accuracy = per_example_reverse_health(
        task, records, corrupt, n_probe=n_probe, seed=seed
    )
    return {
        "clean_accuracy": clean_accuracy,
        "corrupt_accuracy": corrupt_accuracy,
        "damage": clean_accuracy - corrupt_accuracy,
    }


def production_verdict(
    boundary_inference: dict,
    instrument_gates: dict,
    stage: str,
    robustness: dict | None = None,
) -> str:
    if not all(instrument_gates.values()):
        return "INCONCLUSIVE"
    robustness = robustness or {"paths": {name: {"pass": True} for name in PATH_NAMES}}
    robust_paths = robustness["paths"]
    if stage != "target":
        return (
            "PREFLIGHT_GO"
            if robust_paths["core_paired_donor"]["pass"]
            else "PREFLIGHT_SCIENTIFIC_UNRESOLVED_TAIL_SENSITIVITY"
        )
    paths = observed_decision_paths(boundary_inference)
    if (
        paths["stronger_downstream_preference"]
        and robust_paths["stronger_downstream_preference"]["pass"]
    ):
        return "CONFIRMED_DOWNSTREAM_ACCEPTOR_PREFERENCE"
    if paths["core_paired_donor"] and robust_paths["core_paired_donor"]["pass"]:
        return "CONFIRMED_PAIRED_DONOR_ACCEPTOR_BEHAVIOR"
    if paths["core_paired_donor"]:
        return "PAIRED_DONOR_PATH_UNRESOLVED_ROBUSTNESS"
    core_test = boundary_inference["path_tests"]["core_paired_donor"]
    if core_test["decision"] == "UNRESOLVED":
        return "PAIRED_DONOR_PATH_UNRESOLVED"
    if core_test["decision"] == "NOT_DECLARED":
        return "NO_CONFIRMED_PAIRED_DONOR_PATH"
    raise ValueError("unrecognized core paired-donor path decision")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "preflight", "target"), default="smoke")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-randomisation", type=int, default=PRODUCTION_RANDOMISATION
    )
    parser.add_argument("--n-trials", type=int, default=PRODUCTION_TRIALS)
    parser.add_argument(
        "--robust-bootstrap-draws",
        type=int,
        default=PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
    )
    parser.add_argument("--health-probes", type=int, default=8)
    parser.add_argument("--preflight-artifact", default=None)
    parser.add_argument("--unseal-acknowledgement", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.n is None:
        args.n = 96 if args.stage == "smoke" else TARGET_N
    if args.n < MIN_GENES:
        raise SystemExit(f"--n cannot be below the {MIN_GENES}-gene production minimum")
    if (
        args.batch < 1
        or args.n_randomisation < 1
        or args.n_trials < 1
        or args.robust_bootstrap_draws < 1
        or args.health_probes < 1
    ):
        raise SystemExit(
            "batch, randomisation, bootstrap, trial, and health-probe counts must be positive"
        )
    if args.stage != "smoke" and args.out is None:
        raise SystemExit("--out is required for preflight/target and must be a fresh path")
    if args.stage != "smoke":
        production_values = {
            "n": (args.n, TARGET_N),
            "batch": (args.batch, PRODUCTION_BATCH),
            "seed": (args.seed, PRODUCTION_SEED),
            "n-randomisation": (args.n_randomisation, PRODUCTION_RANDOMISATION),
            "n-trials": (args.n_trials, PRODUCTION_TRIALS),
            "robust-bootstrap-draws": (
                args.robust_bootstrap_draws,
                PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
            ),
            "health-probes": (args.health_probes, PRODUCTION_HEALTH_PROBES),
        }
        drift = [
            f"--{name}={observed} (fixed {expected})"
            for name, (observed, expected) in production_values.items()
            if observed != expected
        ]
        if drift:
            raise SystemExit(
                "R1c preflight/target production parameters are frozen: " + "; ".join(drift)
            )

    # Production authorization precedes every user-supplied path.  While the successor
    # pins are unset, both preflight and target exit here without touching model/genome.
    production_commit = None
    authorized_protocol_sha256 = None
    production_authorization = None
    r5_stack = None
    runtime_policy = None
    device = pick_device(args.device) if args.stage == "smoke" else None
    if args.stage in {"preflight", "target"}:
        try:
            authorized_protocol_sha256 = sha256_file(PROTOCOL)
            production_authorization = authorize_production_stage(
                args.stage,
                protocol_sha256=authorized_protocol_sha256,
                protocol_text=PROTOCOL.read_text(),
                acknowledgement=args.unseal_acknowledgement,
            )
            production_commit = require_clean_production_worktree(
                args.stage, repo=REPO
            )
            r5_stack = downstream_r5_stack_identity(repo=REPO)
            verify_frozen_r5_stack(r5_stack)
            device = pick_device(args.device)
            runtime_policy = authorize_production_runtime(device)
        except TargetSealError as error:
            raise SystemExit(str(error)) from error
        if args.stage == "target" and not args.preflight_artifact:
            raise SystemExit("target requires --preflight-artifact")

    # Preflight may reject a stale destination now.  Target does not inspect output
    # existence until its common-dir receipt has irreversibly spent the cohort.
    output = None
    manifest_path = None
    if args.stage == "preflight":
        output = guard_output_path(Path(args.out).resolve())
        manifest_path = guard_output_path(output.with_suffix(".manifest.json"))
    elif args.stage == "target":
        output = Path(args.out).expanduser().resolve()
        manifest_path = output.with_suffix(".manifest.json")

    pre_run_git = (
        {"git_commit": production_commit, "git_dirty": False}
        if args.stage in {"preflight", "target"}
        else worktree_identity(REPO)
    )
    gtf_path = resolve_input(args.gtf)
    fasta_path = resolve_input(args.fasta)
    fasta_fai_path = fasta_index_path(fasta_path)
    _, _, checkpoint_inventory = checkpoint_files(args.model)
    bound_input_paths: list[Path] = [
        *checkpoint_inventory,
        gtf_path,
        fasta_path,
        fasta_fai_path,
        PROTOCOL,
        RUNNER,
        STATS,
        *CORE_DEPENDENCIES,
    ]
    if args.stage == "target":
        preflight_path = resolve_input(args.preflight_artifact)
        bound_input_paths.extend(
            [preflight_path, preflight_path.with_suffix(".manifest.json")]
        )
    try:
        input_snapshot = snapshot_input_paths(bound_input_paths)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    assert device is not None
    fingerprint = bind_runtime_contract(
        instrument_fingerprint(
            args.model,
            gtf_path,
            fasta_path,
            input_snapshot=input_snapshot,
            checkpoint_inventory=checkpoint_inventory,
        ),
        {
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "seed": args.seed,
            "n_randomisation": args.n_randomisation,
            "n_trials": args.n_trials,
            "robust_bootstrap_draws": args.robust_bootstrap_draws,
            "health_probes": args.health_probes,
            "strict_runtime_policy": runtime_policy,
            "r5_stack_fingerprint_sha256": (
                r5_stack[STACK_FINGERPRINT_FIELD]
                if r5_stack is not None
                else None
            ),
        },
    )
    protocol_sha256 = str(snapshot_record(input_snapshot, PROTOCOL)["sha256"])
    if (
        authorized_protocol_sha256 is not None
        and protocol_sha256 != authorized_protocol_sha256
    ):
        raise SystemExit("protocol bytes changed after target authorization")
    protocol_text = read_snapshotted_bytes(PROTOCOL, input_snapshot).decode()
    protocol_declares_frozen = any(
        line.strip().startswith("> **Status: FROZEN")
        for line in protocol_text.splitlines()[:12]
    )
    freeze_authorization = {
        "protocol_declares_frozen": protocol_declares_frozen,
        "protocol_hash_pinned": FROZEN_PROTOCOL_SHA256 == protocol_sha256,
        "instrument_fingerprint_pinned": (
            FROZEN_INSTRUMENT_FINGERPRINT_SHA256
            == fingerprint["fingerprint_sha256"]
        ),
        "downstream_r5_stack_pinned": (
            r5_stack is not None
            and r5_stack[STACK_FINGERPRINT_FIELD]
            == FROZEN_R5_STACK_FINGERPRINT_SHA256
        ),
        "strict_mps_runtime_verified": (
            runtime_policy is not None and runtime_policy.get("verified") is True
        ),
    }
    freeze_authorization["authorization_ready"] = all(
        freeze_authorization.values()
    )
    preflight_record = None
    preflight_package = None
    if args.stage == "target":
        try:
            verify_frozen_instrument(fingerprint["fingerprint_sha256"])
            preflight_package = verify_preflight_manifest(
                args.preflight_artifact, input_snapshot
            )
            verify_pinned_preflight_artifact(
                str(preflight_package["artifact"]["sha256"])
            )
            preflight_record = read_snapshotted_json(
                args.preflight_artifact, input_snapshot
            )
            verify_preflight_binding(preflight_record, fingerprint["fingerprint_sha256"])
        except TargetSealError as error:
            raise SystemExit(str(error)) from error

    try:
        verify_checkpoint_inventory(args.model, checkpoint_inventory, input_snapshot)
        verify_input_snapshot(input_snapshot)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    target_receipt = None
    target_attempt = None
    target_completion_declaration = None
    target_biological_identity = None
    if args.stage == "target":
        assert preflight_package is not None
        assert r5_stack is not None
        assert output is not None and manifest_path is not None
        target_biological_identity = build_r1c_biological_target_identity(
            checkpoint_sha256=str(fingerprint["checkpoint"]["bundle_sha256"]),
            gtf_sha256=str(fingerprint["gtf"]["sha256"]),
            fasta_sha256=str(fingerprint["fasta"]["sha256"]),
            fasta_fai_sha256=str(fingerprint["fasta_fai"]["sha256"]),
        )
        try:
            completion_name = (
                f"{target_biological_identity['biological_target_key']}.completion.json"
            )
            target_receipt = reserve_r1c_target_attempt(
                repo=REPO,
                identity=target_biological_identity,
                method_context={
                    "protocol_sha256": protocol_sha256,
                    "instrument_fingerprint_sha256": fingerprint[
                        "fingerprint_sha256"
                    ],
                    "r5_stack_fingerprint_sha256": r5_stack[
                        STACK_FINGERPRINT_FIELD
                    ],
                    "checkpoint_bundle_sha256": fingerprint["checkpoint"][
                        "bundle_sha256"
                    ],
                    "construction_contract_sha256": (
                        construction_contract_sha256()
                    ),
                    "endpoint_policy": TARGET_ENDPOINT_POLICY,
                    "preflight_artifact": preflight_package["artifact"],
                    "preflight_manifest": preflight_package["manifest"],
                    "input_snapshot_sha256": canonical_sha256(
                        logical_input_snapshot(input_snapshot)
                    ),
                    "requested_artifact_logical_name": output.name,
                    "requested_manifest_logical_name": manifest_path.name,
                    "declared_completion_logical_name": completion_name,
                },
            )
            receipt_path = Path(target_receipt["receipt_path"])
            input_snapshot = extend_input_snapshot(
                input_snapshot, [receipt_path]
            )
            receipt_snapshot = snapshot_record(input_snapshot, receipt_path)
            target_attempt = {
                "logical_name": receipt_path.name,
                "size_bytes": receipt_snapshot["size_bytes"],
                "sha256": receipt_snapshot["sha256"],
                "attempt_id": target_receipt["attempt_id"],
                "biological_target_key": target_receipt[
                    "biological_target_key"
                ],
                "status": target_receipt["status"],
            }
            target_completion_declaration = {"logical_name": completion_name}
            # Output existence is target information from the ledger's perspective.
            # Inspect it only after the receipt exists and cannot be removed on failure.
            output = guard_output_path(output)
            manifest_path = guard_output_path(manifest_path)
        except (R1CTargetSpentError, RuntimeError, ValueError) as error:
            raise SystemExit(str(error)) from error

    checkpoint_stage = None
    checkpoint_model_path: str | Path = args.model
    if args.stage != "smoke":
        try:
            checkpoint_stage, checkpoint_model_path, staged_identity = (
                stage_checkpoint_bundle(
                    args.model, checkpoint_inventory, input_snapshot
                )
            )
            if staged_identity != fingerprint["checkpoint"]:
                raise RuntimeError(
                    "private staged checkpoint identity differs from the instrument"
                )
        except RuntimeError as error:
            raise SystemExit(str(error)) from error

    split = stage_split(args.stage)
    allowed_chromosomes = (
        tuple(sorted(SEALED_CHROMOSOMES))
        if args.stage == "target"
        else VALIDATION_CHROMOSOMES
    )
    examples = load_examples_unscored(
        str(gtf_path),
        str(fasta_path),
        split,
        None,
        tag=f"r1c-{args.stage}-{COHORT_ORDER_VERSION}",
        allowed_chromosomes=allowed_chromosomes,
    )
    assert_stage_chromosomes(examples, args.stage)
    records, cohort_audit = prepare_cohort(examples, args.n)
    groups = [record.example.gene_id for record in records]
    n_genes = len(set(groups))
    structural_rows = serialize_structural_rows(records)
    structural_cohort = structural_target_cohort_identity(structural_rows)
    if cohort_audit["selected_cohort_sha256"] != structural_cohort[
        "source_cohort_sha256"
    ]:
        raise RuntimeError("selected cohort audit digest does not match retained records")

    # Bind the in-memory cohort to the snapshotted GTF/FASTA before any model score.
    try:
        verify_checkpoint_inventory(args.model, checkpoint_inventory, input_snapshot)
        verify_input_snapshot(input_snapshot)
    except RuntimeError as error:
        if checkpoint_stage is not None:
            checkpoint_stage.cleanup()
        raise SystemExit(str(error)) from error

    adapter = load_adapter(str(checkpoint_model_path), device, eager=False)
    task = CouplingTask(adapter, batch=args.batch)

    family, levels, descriptive = measure_behavior(task, records)
    health = health_measurements(
        task, records, n_probe=args.health_probes, seed=args.seed
    )
    boundary_inference = r1c_boundary_classifier(
        family,
        groups,
        sesoi=SESOI,
        n_randomisation=args.n_randomisation,
        alpha=FAMILY_ALPHA,
        seed=args.seed,
        namespace=f"r1c:{args.stage}:production-boundary",
        min_genes=MIN_GENES,
        studentized=True,
    )
    robustness = r1c_gene_robustness_gate(
        family,
        groups,
        n_draws=args.robust_bootstrap_draws,
        seed=args.seed,
        namespace=f"r1c:{args.stage}:production-gene-robustness",
    )
    decisions = boundary_inference["family_decisions"]
    family_calibration = boundary_inference["family_calibration"]

    calibration = None
    if args.stage != "target":
        calibration = r1c_decision_path_preflight(
            descriptive["cell_values"],
            groups,
            sesoi=SESOI,
            n_calibration=args.n_randomisation,
            n_trials=args.n_trials,
            alpha=FAMILY_ALPHA,
            seed=args.seed,
            namespace="r1c:production-path-preflight",
            min_genes=MIN_GENES,
            power_floor=PREFLIGHT_POWER_FLOOR,
            studentized=True,
        )

    health_summary = {
        "n_probe_per_example_per_construct": args.health_probes,
        "clean_mean": float(health["clean_accuracy"].mean()),
        "corrupt_mean": float(health["corrupt_accuracy"].mean()),
        "damage_mean": float(health["damage"].mean()),
        "floor": HEALTH_FLOOR,
        "damage_limit": HEALTH_DAMAGE_LIMIT,
    }
    descriptive_values = descriptive["values"]
    descriptive_summary = {
        "status": descriptive["status"],
        "n": len(records),
        "n_genes": n_genes,
        "names": list(DESCRIPTIVE_FAMILY_NAMES),
        "means": {
            name: float(descriptive_values[index].mean())
            for index, name in enumerate(DESCRIPTIVE_FAMILY_NAMES)
        },
        "inference": None,
        "reason": (
            "Y_XD, Y_XN, and S_X describe the exon-mate row of the same 2x2; "
            "they are not required equivalent and cannot alter the core decision"
        ),
    }
    gates = {
        "minimum_30_genes": n_genes >= MIN_GENES,
        "zero_GT_to_AC_planted_motifs": True,
        "downstream_acceptor_clean_margin_floor": (
            float(levels["P_clean"].mean()) >= ACCEPTOR_MARGIN_FLOOR
        ),
        "exon_mate_acceptor_clean_margin_floor": (
            float(levels["X_clean"].mean()) >= ACCEPTOR_MARGIN_FLOOR
        ),
        "health_clean_floor": health_summary["clean_mean"] >= HEALTH_FLOOR,
        "health_corrupt_floor": health_summary["corrupt_mean"] >= HEALTH_FLOOR,
        "health_damage_bounded": abs(health_summary["damage_mean"]) <= HEALTH_DAMAGE_LIMIT,
    }
    if args.stage != "target":
        gates.update(
            {
                "production_path_core_boundary_fpr": bool(
                    calibration["path_boundary_fpr_ok"]["core_paired_donor"]
                ),
                "production_path_stronger_boundary_fpr": bool(
                    calibration["path_boundary_fpr_ok"][
                        "stronger_downstream_preference"
                    ]
                ),
                "production_path_core_plant_power": bool(
                    calibration["path_planted_power_ok"]["core_paired_donor"]
                ),
                "production_path_stronger_plant_power": bool(
                    calibration["path_planted_power_ok"][
                        "stronger_downstream_preference"
                    ]
                ),
                "observed_Y_PN_tost_and_strict_ci": (
                    boundary_inference["endpoint_p_values"]["Y_PN"]
                    <= FAMILY_ALPHA
                    and
                    decisions["Y_PN"]["classification"]
                    == "EQUIVALENT_WITHIN_SESOI"
                ),
            }
        )
    else:
        gates["passing_exact_bound_preflight"] = True

    instrument_gates = dict(gates)
    gates["raw_trimmed_gene_robustness"] = bool(
        robustness["paths"]["core_paired_donor"]["pass"]
    )
    all_gates_pass = all(gates.values())
    verdict = production_verdict(
        boundary_inference, instrument_gates, args.stage, robustness
    )
    cohort_sha256 = structural_cohort["source_cohort_sha256"]
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "example_id": record.example.example_id,
                "gene_id": record.example.gene_id,
                "chrom": _normalise_chromosome(record.example.chrom),
                "strand": record.example.strand,
                "positions": {
                    "prior_real_acceptor": int(record.example.acceptor_pos),
                    "paired_donor": int(record.example.donor_pos),
                    "partner_acceptor": int(record.example.next_acceptor_pos),
                    "matched_decoy_donor": record.matched_decoy_donor_pos,
                    "matched_decoy_donor_distance_mismatch": (
                        record.matched_decoy_distance_mismatch
                    ),
                    "dyad_distance_mismatch": record.dyad_distance_mismatch,
                },
                "family": {
                    name: float(family[family_index, index])
                    for family_index, name in enumerate(PRIMARY_FAMILY_NAMES)
                },
                "descriptive_2x2": {
                    name: float(descriptive_values[family_index, index])
                    for family_index, name in enumerate(DESCRIPTIVE_FAMILY_NAMES)
                },
                "cell_effects": {
                    name: float(descriptive["cell_values"][cell_index, index])
                    for cell_index, name in enumerate(CELL_NAMES)
                },
                "margin_levels": {
                    name: float(values[index]) for name, values in levels.items()
                },
                "health": {
                    "clean_accuracy": float(health["clean_accuracy"][index]),
                    "corrupt_accuracy": float(health["corrupt_accuracy"][index]),
                    "damage": float(health["damage"][index]),
                },
            }
        )
    target_cohort_package = target_cohort_identity(rows)
    if any(
        target_cohort_package[name] != value
        for name, value in structural_cohort.items()
    ):
        raise RuntimeError("outcome serialization changed the structural cohort identity")

    result_config = _preflight_config_contract()
    result_config.update(
        {
            "batch": args.batch,
            "health_probes": args.health_probes,
            "n_randomisation": args.n_randomisation,
            "n_trials": args.n_trials,
            "robust_bootstrap_draws": args.robust_bootstrap_draws,
            "robust_trim_fraction": ROBUST_TRIM_FRACTION,
            "seed": args.seed,
        }
    )
    result = {
        "schema": "gfmcd.r1c.reverse_behavior/3",
        "experiment": "R1c_corrected_reverse_behavior",
        "protocol_status": (
            "FROZEN_HASH_PINNED"
            if freeze_authorization["protocol_declares_frozen"]
            and freeze_authorization["protocol_hash_pinned"]
            else "DRAFT_NOT_FROZEN"
        ),
        "protocol_sha256": protocol_sha256,
        "freeze_authorization": freeze_authorization,
        "stage": args.stage,
        "status": "DEVELOPMENT" if args.stage != "target" else "CONFIRMATORY",
        "evidence_rung": "BEHAVIOR_ONLY",
        "split": split,
        "sealed_split_opened": args.stage == "target",
        "holdout_scope_disclosure": holdout_scope_disclosure(),
        "n": len(records),
        "n_genes": n_genes,
        "cohort_sha256": cohort_sha256,
        "cohort_audit": cohort_audit,
        "target_cohort_identity": target_cohort_package,
        "preconstruction_allowed_chromosomes": list(allowed_chromosomes),
        "instrument_fingerprint": fingerprint,
        STACK_BINDING_FIELD: (
            r5_stack[STACK_BINDING_FIELD] if r5_stack is not None else None
        ),
        STACK_FINGERPRINT_FIELD: (
            r5_stack[STACK_FINGERPRINT_FIELD] if r5_stack is not None else None
        ),
        "config": result_config,
        "family_decisions": decisions,
        "boundary_inference": boundary_inference,
        "gene_robustness": robustness,
        "observed_decision_paths": observed_decision_paths(boundary_inference),
        "family_calibration": family_calibration,
        "descriptive_2x2_summary": descriptive_summary,
        "production_path_preflight": calibration,
        "structured_plant": {
            "delta": PLANT_DELTA,
            "cell_vector": cell_plant_vector().tolist(),
            "vector": structured_plant_vector().tolist(),
            "requirements": {
                PRIMARY_FAMILY_NAMES[index]: label
                for index, label in PRIMARY_PLANT_REQUIREMENTS.items()
            },
        },
        "health": health_summary,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "verdict": verdict,
        "preflight_artifact": args.preflight_artifact,
        "preflight_package": preflight_package,
        "input_snapshot_sha256": canonical_sha256(
            logical_input_snapshot(input_snapshot)
        ),
        "target_attempt": target_attempt,
        "target_completion": target_completion_declaration,
        "target_attempt_disposition": (
            "SEALED_OUTCOMES_MAY_HAVE_BEEN_SPENT_EVEN_IF_THIS_RUN_CRASHES"
            if args.stage == "target"
            else None
        ),
        "target_authorization": {
            "required": args.stage == "target",
            "pre_run_clean_worktree": (
                not bool(pre_run_git["git_dirty"])
                if args.stage in {"preflight", "target"}
                else None
            ),
            "pre_run_git_commit": pre_run_git["git_commit"],
            "explicit_unseal_acknowledgement": args.stage == "target",
            "production_authorization": production_authorization,
            "strict_runtime_policy": runtime_policy,
            "model_loaded_from_private_staged_copy": args.stage != "smoke",
            "preflight_artifact_hash_pin_required": args.stage == "target",
            "preflight_artifact_hash_pin_matched": (
                preflight_package is not None
                and preflight_package["artifact"]["sha256"]
                == FROZEN_PREFLIGHT_ARTIFACT_SHA256
                if args.stage == "target"
                else None
            ),
        },
        "rows": rows,
        "claim_boundary": (
            "Behavior rung only. I is a downstream-versus-exon-mate preference contrast "
            "on outcome-blind distance-matched dyads; X cells are descriptive and need not "
            "be equivalent. No component, mediation, semantic, biological-mechanism, or "
            "circuit-completeness conclusion is licensed."
        ),
    }

    if args.stage == "smoke":
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "n": len(records),
                    "n_genes": n_genes,
                    "gates": gates,
                    "verdict": verdict,
                },
                indent=2,
            )
        )
        return

    assert output is not None and manifest_path is not None
    manifest = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r1c_reverse_behavior",
        sys.argv,
        repo=REPO,
        status=result["status"],
        config={
            "stage": args.stage,
            "n": len(records),
            "n_genes": n_genes,
            "construction_version": CONSTRUCTION_VERSION,
            "cohort_order_version": COHORT_ORDER_VERSION,
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "cohort_sha256": cohort_sha256,
            "target_holdout_id": TARGET_HOLDOUT_ID,
            "primary_family_names": list(PRIMARY_FAMILY_NAMES),
            "cell_names": list(CELL_NAMES),
            "descriptive_family_names": list(DESCRIPTIVE_FAMILY_NAMES),
            "descriptive_status": "REPORTED_NOT_DECISION_BEARING",
            "decision_paths": decision_path_requirements(),
            "boundary_test_contract": boundary_test_contract(),
            "dyad_distance_caliper": DYAD_DISTANCE_CALIPER,
            "plant_delta": PLANT_DELTA,
            "sesoi": SESOI,
            "interval_scaling": "fixed_endpoint_bootstrap_sd_max_deviation",
            "robust_bootstrap_draws": args.robust_bootstrap_draws,
            "robust_trim_fraction": ROBUST_TRIM_FRACTION,
            "target_cohort_identity": target_cohort_package,
            STACK_FINGERPRINT_FIELD: (
                r5_stack[STACK_FINGERPRINT_FIELD]
                if r5_stack is not None
                else None
            ),
            "input_snapshot_sha256": canonical_sha256(
                logical_input_snapshot(input_snapshot)
            ),
        },
        seeds={"randomisation": args.seed, "health": args.seed},
        notes=[
            "Behavior rung only; no internal or biological mechanism claim.",
            (
                "The target is masked-acceptor-endpoint-outcome held out, not globally "
                "sequence unseen; historical chr8/9 sequence materialization is disclosed."
            ),
            f"Stage {args.stage}; sealed chr8/9 opened={args.stage == 'target'}.",
            "Four derived endpoints entered the strict simultaneous-CI family.",
            "Decision-bearing p-values test the SESOI boundary; path IUTs are Holm-adjusted over exactly core and stronger.",
            "Y_XD, Y_XN, and S_X are descriptive cells/contrast from the same 2x2.",
            "The 2x2 cohort is exhaustively screened then distance-matched before first-n selection.",
            "GT->AC motif cleanliness checked at paired and matched-decoy donors.",
            "Production model load used a private byte-identical checkpoint stage.",
            "Manifest inputs are the pre-use snapshot; no live provenance rehash was used.",
            "Raw and 10%-trimmed unweighted-gene bootstrap support is a necessary robustness conjunction, not the exact inferential authority.",
        ],
    )
    manifest.git_commit = str(pre_run_git["git_commit"])
    manifest.git_dirty = bool(pre_run_git["git_dirty"])
    manifest.inputs = [
        {
            "path": record["logical_name"],
            "location_scope": record["location_scope"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in logical_input_snapshot(input_snapshot)
    ]
    result_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    manifest.outputs = [
        {
            "path": output.name,
            "size_bytes": len(result_bytes),
            "sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
    ]
    manifest_bytes = (
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode()
    try:
        verify_checkpoint_inventory(args.model, checkpoint_inventory, input_snapshot)
        verify_input_snapshot(input_snapshot)
    except RuntimeError as error:
        if checkpoint_stage is not None:
            checkpoint_stage.cleanup()
        raise SystemExit(str(error)) from error
    try:
        atomic_write_new(output, result_bytes)
        atomic_write_new(manifest_path, manifest_bytes)
    finally:
        if checkpoint_stage is not None:
            checkpoint_stage.cleanup()
    if args.stage == "target":
        assert target_receipt is not None
        assert r5_stack is not None
        try:
            complete_r1c_target_attempt(
                receipt=target_receipt,
                artifact_path=output,
                manifest_path=manifest_path,
                cohort_digests=target_cohort_package,
                method_fingerprint_sha256=fingerprint["fingerprint_sha256"],
                r5_stack_fingerprint_sha256=r5_stack[
                    STACK_FINGERPRINT_FIELD
                ],
                official_verdict=verdict,
                all_gates_pass=all_gates_pass,
                gates=gates,
            )
        except (R1CTargetSpentError, ValueError, OSError) as error:
            raise SystemExit(
                "R1c artifact and manifest published, but completion failed; "
                "the target remains spent: " + str(error)
            ) from error
    print(f"{verdict}: {output}")


if __name__ == "__main__":
    main()
