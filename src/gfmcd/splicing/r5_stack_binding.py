"""Canonical prospective binding for the complete R5a--R5c mechanism stack.

The three R5 runners deliberately keep their own experiment-specific instrument
payloads.  This module adds the stronger invariant that was previously missing: every
stage also carries *one identical* stack payload and fingerprint.  The payload binds all
three runners and protocols, the dynamically executed R3c cohort/entry runner, the full
repository-local Python import closure, the schema vocabulary, and the stable runtime.

Freeze-pin assignment values are the only normalized source text.  Consequently filling
reviewed literal pins does not create a self-referential hash, while any executable or
method change does.
"""

from __future__ import annotations

import ast
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping, Sequence


STACK_SCHEMA = "gfmcd.r5.stack_binding/1"
R3C_ENTRY_SCHEMA = "gfmcd.r3c.acceptor_head_confirmation/2"
R3C_INSTRUMENT_REVISION = "r3c-v2.1-monotone-health-early-preflight-authorization"
R3C_TARGET_HEALTH_SCHEMA = "gfmcd.r3c.target_path_health_authorization/1"
R3C_TARGET_HEALTH_RULE = (
    "effective target health = reviewed preflight health AND current health"
)
R3C_PATH_HEALTH_RULE = "preflight_health_ok AND current_health_ok"
R3C_PATH_NAMES = (
    "L3H1_reused_component",
    "L2H15_mean_component",
    "L1H12_mean_component",
    "L4H11_noising_component",
    "L4H9_noising_component",
    "L4H7_noising_component",
    "distinct_acceptor_route_candidate",
)
R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS = ("L2H15_mean_component",)
STACK_FINGERPRINT_FIELD = "r5_stack_fingerprint_sha256"
STACK_BINDING_FIELD = "r5_stack_binding"

R5_RUNNER_PATHS = (
    "scripts/splicebert_acceptor_coupling/run_r5a_acceptor_address_binding.py",
    "scripts/splicebert_acceptor_coupling/run_r5b_acceptor_side_state.py",
    "scripts/splicebert_acceptor_coupling/run_r5c_acceptor_write_delivery.py",
)
R5_PROTOCOL_PATHS = (
    "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R5A_ACCEPTOR_ADDRESS_BINDING.md",
    "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R5B_ACCEPTOR_SIDE_STATE.md",
    "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R5C_ACCEPTOR_WRITE_DELIVERY.md",
)
R3C_RUNNER_PATH = (
    "scripts/splicebert_acceptor_coupling/run_r3c_acceptor_head_confirmation.py"
)
RUNTIME_PACKAGES = (
    "numpy",
    "scipy",
    "torch",
    "transformers",
    "tokenizers",
    "pyfaidx",
)

# This is an explicit cross-stage contract, not a claim that every schema token is
# emitted in every run.  Literal schema strings discovered from the source closure are
# recorded separately so adding or removing an executable schema changes the binding.
STACK_CONTRACT = {
    "stage_order": ["R3c", "R5a", "R5b", "R5c"],
    "behavioral_endpoint": "single masked acceptor G hard margin only",
    "masked_donor_endpoint_constructed": False,
    "cohort_buckets": {"preflight": 0, "target": 1},
    "sealed_chr8_chr9_opened": False,
    "handoff_schemas": {
        "r3c_entry": R3C_ENTRY_SCHEMA,
        "entry": "gfmcd.r5.entry_contract/1",
        "exclusion_registry": "gfmcd.r5.exclusion_registry/1",
        "cohort": "gfmcd.r5.acceptor_mechanism_cohort/2",
        "r5b_runtime": "gfmcd.r5b.r5c_runtime_contract/1",
    },
}

# Public executable registries whose contents decide scanned families, paths, controls,
# calibration cases, or handoff structure.  Their defining AST nodes are recorded below;
# this makes the joint payload explicitly auditable without importing the runners (which
# would create circular imports and could accidentally execute module setup).
EXECUTABLE_REGISTRY_SYMBOLS = {
    R5_RUNNER_PATHS[0]: (
        "family_row_names",
        "named_path_specs",
        "classifier_calibration",
        "PATH_ORDER",
        "FAMILY_ORDER",
    ),
    R5_RUNNER_PATHS[1]: (
        "causal_primitive_names",
        "causal_family_names",
        "causal_boundary_edge_registry",
        "causal_requirement_registry",
        "prospective_calibration_trial_count",
        "RANK_GRID",
    ),
    R5_RUNNER_PATHS[2]: (
        "family_row_names",
        "path_registry",
        "random_ensemble_registry",
        "calibration_registry",
        "CAPABILITY_GATES",
        "FAMILY_NAMES",
    ),
}

_SCHEMA_TOKEN = re.compile(r"[\"'](gfmcd\.(?:r3c|r5)[A-Za-z0-9_.-]*/\d+)[\"']")


class R5StackBindingError(ValueError):
    """Raised when a serialized R5 stack identity is absent or differs."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _default_repo() -> Path:
    # src/gfmcd/splicing/r5_stack_binding.py -> repository root
    return Path(__file__).resolve().parents[3]


def stable_runtime_identity() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = package_version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "packages": packages,
    }


def normalize_freeze_pins(text: str) -> tuple[str, tuple[str, ...]]:
    """Normalize only RHS values of top-level freeze assignments.

    Replacing the value node instead of the whole assignment preserves annotations,
    whitespace, and comments byte-for-byte.  Multiline literal values are still replaced
    as one span.
    """

    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))
    replacements: list[tuple[int, int, str, str]] = []
    for node in tree.body:
        name: str | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
        if name is None or re.fullmatch(r"FROZEN_[A-Z0-9_]+", name) is None:
            continue
        value = node.value
        if value is None:
            raise R5StackBindingError(f"freeze pin {name} has no assignment value")
        start = line_offsets[value.lineno - 1] + value.col_offset
        end = line_offsets[value.end_lineno - 1] + value.end_col_offset
        replacements.append((start, end, "<FROZEN_PIN>", name))
    normalized = text
    for start, end, replacement, _name in reversed(replacements):
        normalized = normalized[:start] + replacement + normalized[end:]
    names = tuple(name for _start, _end, _replacement, name in replacements)
    return normalized, names


def normalized_source_record(path: Path, *, repo: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    normalized, pins = normalize_freeze_pins(text)
    relative = str(path.resolve().relative_to(repo.resolve()))
    if relative in (*R5_RUNNER_PATHS, R3C_RUNNER_PATH) and not pins:
        raise R5StackBindingError(f"expected freeze pins in bound runner: {relative}")
    return {
        "relative_path": relative,
        # The exact on-disk size cannot enter the prospective fingerprint: replacing
        # ``None`` with a reviewed literal pin generally changes it.  Exact source sizes
        # and hashes are separately recorded in each execution manifest.
        "normalized_size_bytes": len(normalized.encode("utf-8")),
        "normalization": (
            "all and only top-level FROZEN_* assignment values"
            if pins
            else "exact bytes"
        ),
        "normalized_pin_names": list(pins),
        "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }


def _module_path(module: str, *, repo: Path) -> Path | None:
    if not module.startswith("gfmcd"):
        return None
    base = repo / "src" / Path(*module.split("."))
    file_candidate = base.with_suffix(".py")
    package_candidate = base / "__init__.py"
    if file_candidate.is_file():
        return file_candidate.resolve()
    if package_candidate.is_file():
        return package_candidate.resolve()
    return None


def _parent_package_inits(path: Path, *, repo: Path) -> tuple[Path, ...]:
    src = (repo / "src").resolve()
    try:
        relative = path.resolve().relative_to(src)
    except ValueError:
        return ()
    parts = relative.parts[:-1]
    output: list[Path] = []
    for depth in range(1, len(parts) + 1):
        candidate = src.joinpath(*parts[:depth], "__init__.py")
        if candidate.is_file():
            output.append(candidate.resolve())
    return tuple(output)


def _local_imports(path: Path, *, repo: Path) -> tuple[Path, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise R5StackBindingError(f"could not parse bound source {path}") from exc
    found: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
            # ``from gfmcd.package import helper_module`` executes the submodule when
            # that name resolves to a local module.  Function/class aliases simply do
            # not resolve here and are harmless.
            modules.extend(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
        for module in modules:
            candidate = _module_path(module, repo=repo)
            if candidate is not None:
                found.add(candidate)
                found.update(_parent_package_inits(candidate, repo=repo))
    return tuple(sorted(found))


def executable_registry_records(*, repo: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for relative, requested in EXECUTABLE_REGISTRY_SYMBOLS.items():
        path = repo / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        nodes: dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nodes[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        nodes[target.id] = node
        missing = [name for name in requested if name not in nodes]
        if missing:
            raise R5StackBindingError(
                f"registered executable symbol is missing from {relative}: {missing[0]}"
            )
        output[relative] = {
            name: {
                "node_type": type(nodes[name]).__name__,
                "ast_sha256": hashlib.sha256(
                    ast.dump(nodes[name], annotate_fields=True, include_attributes=False).encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
            for name in requested
        }
    return output


def local_python_closure(
    roots: Sequence[Path], *, repo: Path | None = None
) -> tuple[Path, ...]:
    root = (repo or _default_repo()).resolve()
    pending = [path.resolve() for path in roots]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise R5StackBindingError(f"bound source is missing: {path}")
        seen.add(path)
        for imported in _local_imports(path, repo=root):
            if imported not in seen:
                pending.append(imported)
    return tuple(sorted(seen, key=lambda path: str(path.relative_to(root))))


def build_r5_stack_binding(*, repo: Path | None = None) -> dict[str, Any]:
    root = (repo or _default_repo()).resolve()
    runner_roots = tuple(root / path for path in (*R5_RUNNER_PATHS, R3C_RUNNER_PATH))
    protocols = tuple(root / path for path in R5_PROTOCOL_PATHS)
    missing = [str(path) for path in (*runner_roots, *protocols) if not path.is_file()]
    if missing:
        raise R5StackBindingError(f"R5 stack source is missing: {missing[0]}")

    closure = local_python_closure(runner_roots, repo=root)
    source_records = {
        str(path.relative_to(root)): normalized_source_record(path, repo=root)
        for path in closure
    }
    protocol_records = {
        str(path.relative_to(root)): {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in protocols
    }
    schema_tokens = sorted(
        {
            token
            for path in closure
            for token in _SCHEMA_TOKEN.findall(path.read_text(encoding="utf-8"))
        }
    )
    executable_registries = executable_registry_records(repo=root)
    return {
        "schema": STACK_SCHEMA,
        "contract": STACK_CONTRACT,
        "runner_roots": list(R5_RUNNER_PATHS),
        "dynamic_entry_runner": R3C_RUNNER_PATH,
        "protocols": protocol_records,
        "local_python_closure": {
            "discovery": (
                "recursive AST traversal of repository-local gfmcd imports plus package "
                "initializers; R3c dynamic runner is an explicit root"
            ),
            "n_files": len(source_records),
            "files": source_records,
            "ordered_paths_sha256": canonical_sha256(list(source_records)),
        },
        "schema_registry": schema_tokens,
        "schema_registry_sha256": canonical_sha256(schema_tokens),
        "executable_registries": executable_registries,
        "executable_registries_sha256": canonical_sha256(executable_registries),
        "runtime_identity": stable_runtime_identity(),
    }


def r5_stack_fingerprint(binding: Mapping[str, Any] | None = None) -> str:
    return canonical_sha256(build_r5_stack_binding() if binding is None else dict(binding))


def validate_r5_stack_binding(
    binding: Mapping[str, Any],
    fingerprint: str,
    *,
    current: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observed = dict(binding)
    expected = dict(current) if current is not None else build_r5_stack_binding()
    if observed.get("schema") != STACK_SCHEMA:
        raise R5StackBindingError("R5 joint stack binding schema differs")
    if canonical_sha256(observed) != fingerprint:
        raise R5StackBindingError("R5 joint stack fingerprint does not bind its payload")
    expected_fingerprint = canonical_sha256(expected)
    if observed != expected or fingerprint != expected_fingerprint:
        raise R5StackBindingError("R5 joint stack differs from the current frozen stack")
    return {
        "schema": STACK_SCHEMA,
        "verified": True,
        STACK_FINGERPRINT_FIELD: expected_fingerprint,
    }


def validate_serialized_r5_stack(
    record: Mapping[str, Any], *, current: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    binding = record.get(STACK_BINDING_FIELD)
    fingerprint = record.get(STACK_FINGERPRINT_FIELD)
    if not isinstance(binding, Mapping) or not isinstance(fingerprint, str):
        raise R5StackBindingError("serialized R5 joint stack identity is missing")
    return validate_r5_stack_binding(binding, fingerprint, current=current)


def validate_r3c_target_health_contract(
    record: Mapping[str, Any],
    *,
    required_supported_path: str = "L3H1_reused_component",
) -> dict[str, Any]:
    """Validate the corrected monotone R3c target-health authorization contract.

    This is shared by R5a and R5b so a handoff cannot silently interpret the v2 health
    eligibility fields differently.  It deliberately validates all seven registered
    paths even though R5 currently consumes only the supported L3H1 path.
    """

    if required_supported_path not in R3C_PATH_NAMES:
        raise R5StackBindingError("required R3c path is outside the frozen registry")
    if record.get("instrument_revision") != R3C_INSTRUMENT_REVISION:
        raise R5StackBindingError("R3c monotone-health instrument revision differs")
    paths = record.get("paths")
    authorization = record.get("path_health_authorization")
    if not isinstance(paths, Mapping) or tuple(paths) != R3C_PATH_NAMES:
        raise R5StackBindingError("R3c target lacks the ordered seven-path registry")
    if not isinstance(authorization, Mapping):
        raise R5StackBindingError("R3c target health authorization is missing")
    if (
        authorization.get("schema") != R3C_TARGET_HEALTH_SCHEMA
        or authorization.get("rule") != R3C_TARGET_HEALTH_RULE
        or authorization.get("can_only_shrink_from_preflight") is not True
    ):
        raise R5StackBindingError("R3c target health authorization policy differs")

    def ordered_subset(field: str) -> list[str]:
        value = authorization.get(field)
        if not isinstance(value, list):
            raise R5StackBindingError(f"R3c target health {field} is missing")
        if value != [name for name in R3C_PATH_NAMES if name in value]:
            raise R5StackBindingError(
                f"R3c target health {field} is not an ordered unique subset"
            )
        return value

    preflight_authorized = ordered_subset("preflight_authorized_paths")
    preflight_suppressed = ordered_subset("preflight_suppressed_paths")
    current_healthy = ordered_subset("current_healthy_paths")
    effective_authorized = ordered_subset("effective_authorized_paths")
    effective_suppressed = ordered_subset("effective_suppressed_paths")
    if (
        preflight_suppressed
        != [name for name in R3C_PATH_NAMES if name not in preflight_authorized]
        or effective_authorized
        != [
            name
            for name in R3C_PATH_NAMES
            if name in preflight_authorized and name in current_healthy
        ]
        or effective_suppressed
        != [name for name in R3C_PATH_NAMES if name not in effective_authorized]
    ):
        raise R5StackBindingError("R3c target health sets do not implement intersection")
    if any(
        name in preflight_authorized
        or name not in preflight_suppressed
        or name in effective_authorized
        or name not in effective_suppressed
        for name in R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS
    ):
        raise R5StackBindingError(
            "R3c target revived a path suppressed by the reviewed v1 health ceiling"
        )

    for name in R3C_PATH_NAMES:
        path = paths[name]
        if not isinstance(path, Mapping) or not isinstance(path.get("health"), Mapping):
            raise R5StackBindingError(f"R3c target path health is missing for {name}")
        health = path["health"]
        preflight_ok = name in preflight_authorized
        current_ok = name in current_healthy
        effective_ok = preflight_ok and current_ok
        if (
            health.get("preflight_health_ok") is not preflight_ok
            or health.get("current_health_ok") is not current_ok
            or health.get("ok") is not effective_ok
            or health.get("authorization_rule") != R3C_PATH_HEALTH_RULE
            or (
                not effective_ok
                and path.get("classification") != "HEALTH_FAILED"
            )
        ):
            raise R5StackBindingError(
                f"R3c target path decision bypasses health authorization for {name}"
            )

    required = paths[required_supported_path]
    if (
        required_supported_path not in effective_authorized
        or required.get("classification") != "SUPPORTED"
    ):
        raise R5StackBindingError("required R3c path is not health-authorized support")
    family = record.get("family")
    calibration = (
        family.get("production_path_preflight")
        if isinstance(family, Mapping)
        else None
    )
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("all_health_eligible_path_plants_powered") is not True
        or calibration.get("health_eligible_paths") != list(R3C_PATH_NAMES)
    ):
        raise R5StackBindingError(
            "R3c calibration does not power all seven health-enabled paths"
        )
    return {
        "schema": R3C_TARGET_HEALTH_SCHEMA,
        "verified": True,
        "required_supported_path": required_supported_path,
        "effective_authorized_paths": effective_authorized,
    }


def validate_r3c_target_manifest_contract(
    record: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate the semantic half of the R3c target sibling manifest."""

    config = manifest.get("config")
    construction = record.get("construction")
    git_identity = record.get("git_authorization_identity")
    if (
        manifest.get("experiment_id")
        != "splicebert_acceptor_coupling:r3c_acceptor_head_confirmation"
        or manifest.get("status") != "CONFIRMATORY"
        or manifest.get("git_dirty") is not False
        or not isinstance(config, Mapping)
        or not isinstance(construction, Mapping)
        or not isinstance(git_identity, Mapping)
        or config.get("stage") != "target"
        or config.get("gene_bucket") != 1
        or config.get("instrument_fingerprint_sha256")
        != record.get("instrument_fingerprint_sha256")
        or config.get("construction_sha256")
        != construction.get("ordered_rows_sha256")
        or config.get("input_snapshot_sha256")
        != record.get("input_snapshot_sha256")
        or manifest.get("git_commit") != git_identity.get("commit")
        or git_identity.get("dirty") is not False
    ):
        raise R5StackBindingError("R3c target sibling manifest context differs")
    return {
        "verified": True,
        "experiment_id": manifest["experiment_id"],
        "status": manifest["status"],
    }
